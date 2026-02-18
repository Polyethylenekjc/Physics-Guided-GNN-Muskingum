from typing import Tuple, Callable, Optional, List, Sequence

import numpy as np
from scipy.stats import spearmanr


def _build_lagged(series: np.ndarray, max_lag: int) -> np.ndarray:
    rows = []
    for lag in range(1, max_lag + 1):
        rows.append(series[max_lag - lag : -lag])
    return np.stack(rows, axis=1)


def _rss(y: np.ndarray, x: np.ndarray) -> float:
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ coef
    return float(np.sum(resid ** 2))


def compute_granger_scores(
    runoffs: np.ndarray,
    weather: np.ndarray,
    max_lag: int = 3,
    on_target: Optional[Callable[[int], None]] = None,
) -> np.ndarray:
    if runoffs.ndim != 2:
        raise ValueError("runoffs should have shape (nodes, time)")
    if weather.ndim != 3:
        raise ValueError("weather should have shape (nodes, time, features)")
    if runoffs.shape[0] != weather.shape[0] or runoffs.shape[1] != weather.shape[1]:
        raise ValueError("runoffs and weather must have matching nodes and time")

    nodes, length = runoffs.shape
    if length <= max_lag + 1:
        raise ValueError("Not enough samples for the given lag.")

    scores = np.zeros((nodes, nodes), dtype=np.float32)

    for target in range(nodes):
        if on_target is not None:
            on_target(target)
        y = runoffs[target, max_lag:]
        base_x = _build_lagged(runoffs[target], max_lag)
        weather_x = weather[target, max_lag:, :]
        base_x = np.concatenate(
            [np.ones((base_x.shape[0], 1)), base_x, weather_x], axis=1
        )
        rss_base = _rss(y, base_x)

        for source in range(nodes):
            if source == target:
                scores[target, source] = 1.0
                continue

            source_x = _build_lagged(runoffs[source], max_lag)
            full_x = np.concatenate([base_x, source_x], axis=1)
            rss_full = _rss(y, full_x)

            if rss_base <= 0.0:
                score = 0.0
            else:
                score = max(0.0, (rss_base - rss_full) / rss_base)

            scores[target, source] = score

    return scores


def compute_variable_granger_scores(
    runoffs: np.ndarray,
    weather: np.ndarray,
    features: np.ndarray,
    candidates: Sequence[Sequence[Tuple[int, int]]],
    max_lag: int = 3,
    on_target: Optional[Callable[[int], None]] = None,
) -> List[np.ndarray]:
    if runoffs.ndim != 2:
        raise ValueError("runoffs should have shape (nodes, time)")
    if weather.ndim != 3:
        raise ValueError("weather should have shape (nodes, time, features)")
    if features.ndim != 3:
        raise ValueError("features should have shape (nodes, time, features)")
    if runoffs.shape[0] != weather.shape[0] or runoffs.shape[1] != weather.shape[1]:
        raise ValueError("runoffs and weather must have matching nodes and time")
    if runoffs.shape[0] != features.shape[0] or runoffs.shape[1] != features.shape[1]:
        raise ValueError("runoffs and features must have matching nodes and time")

    nodes, length = runoffs.shape
    if length <= max_lag + 1:
        raise ValueError("Not enough samples for the given lag.")

    scores = []
    for target in range(nodes):
        if on_target is not None:
            on_target(target)
        y = runoffs[target, max_lag:]
        base_x = _build_lagged(runoffs[target], max_lag)
        weather_x = weather[target, max_lag:, :]
        base_x = np.concatenate(
            [np.ones((base_x.shape[0], 1)), base_x, weather_x], axis=1
        )
        rss_base = _rss(y, base_x)

        target_scores = []
        for src_idx, feat_idx in candidates[target]:
            source_series = features[src_idx, :, feat_idx]
            source_x = _build_lagged(source_series, max_lag)
            full_x = np.concatenate([base_x, source_x], axis=1)
            rss_full = _rss(y, full_x)

            if rss_base <= 0.0:
                score = 0.0
            else:
                score = max(0.0, (rss_base - rss_full) / rss_base)

            target_scores.append(score)

        scores.append(np.asarray(target_scores, dtype=np.float32))

    return scores


def _compute_topo_distance(
    edge_index: np.ndarray, num_nodes: int
) -> np.ndarray:
    """计算拓扑距离矩阵（BFS）"""
    dist = np.full((num_nodes, num_nodes), np.inf, dtype=np.float32)
    np.fill_diagonal(dist, 0)
    
    src, dst = edge_index
    for s, d in zip(src, dst):
        dist[d, s] = 1.0
    
    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                dist[i, j] = min(dist[i, j], dist[i, k] + dist[k, j])
    
    return dist


def compute_spearman_variable_scores(
    runoffs: np.ndarray,
    weather: np.ndarray,
    edge_index: np.ndarray,
    max_decay: float = 0.1,
    on_target: Optional[Callable[[int], None]] = None,
) -> List[np.ndarray]:
    """基于Spearman相关系数+距离衰减的变量选择
    
    Args:
        runoffs: shape (nodes, time)
        weather: shape (nodes, time, weather_features)
        edge_index: shape (2, edges) - (source, dest)
        max_decay: 最大衰减值（距离∞时权重为该值）
        on_target: 进度回调
    
    Returns:
        scores list，每个元素shape (num_candidates,)
    """
    if runoffs.ndim != 2:
        raise ValueError("runoffs should have shape (nodes, time)")
    if weather.ndim != 3:
        raise ValueError("weather should have shape (nodes, time, features)")
    
    nodes, length = runoffs.shape
    weather_features = weather.shape[2]
    
    topo_dist = _compute_topo_distance(edge_index.numpy() if hasattr(edge_index, 'numpy') else edge_index, nodes)
    
    all_scores = []
    for target in range(nodes):
        if on_target is not None:
            on_target(target)
        
        target_runoff = runoffs[target]
        
        target_scores = []
        for src in range(nodes):
            # 仅对气象等外部因子做变量选择，不再把径流本身作为候选变量
            # 对应的特征索引为 1..weather_features（0 是径流）
            for feat_idx in range(1, 1 + weather_features):
                candidate = weather[src, :, feat_idx - 1]
                
                corr, _ = spearmanr(target_runoff, candidate)
                if np.isnan(corr):
                    corr = 0.0
                
                distance = topo_dist[target, src]
                if np.isinf(distance):
                    weight = 0.0
                else:
                    weight = max(max_decay, 1.0 - 0.1 * distance)
                
                final_score = abs(corr) * weight
                target_scores.append(final_score)
        
        all_scores.append(np.asarray(target_scores, dtype=np.float32))
    
    return all_scores

