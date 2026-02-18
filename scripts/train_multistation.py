import argparse
import csv
import os
from typing import Dict, Any, Tuple

import yaml

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

import torch
from torch.utils.data import DataLoader

from pg_gnn.causal import compute_spearman_variable_scores
from pg_gnn.data import MultiStationDataset
from pg_gnn.graph import build_chain_edges, build_upstream_mask
from pg_gnn.model.loss import PhysicsGuidedLoss
from pg_gnn.model.model import PhysicsGuidedGNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/pg_gnn.yaml")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _save_causal_scores(
    path: str,
    station_order: list,
    scores: torch.Tensor,
    mask: torch.Tensor,
    topk_mask: torch.Tensor,
) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["target_station", "source_station", "score", "allowed", "topk"]
        )
        for i, target in enumerate(station_order):
            for j, source in enumerate(station_order):
                writer.writerow(
                    [
                        target,
                        source,
                        float(scores[i, j]),
                        int(mask[i, j].item()),
                        int(topk_mask[i, j].item()),
                    ]
                )


def _load_causal_scores(
    path: str, station_order: list
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    idx = {name: i for i, name in enumerate(station_order)}
    count = len(station_order)
    scores = torch.zeros((count, count), dtype=torch.float32)
    allowed = torch.zeros((count, count), dtype=torch.bool)
    topk_mask = torch.zeros((count, count), dtype=torch.bool)

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            target = row["target_station"]
            source = row["source_station"]
            if target not in idx or source not in idx:
                continue
            i = idx[target]
            j = idx[source]
            scores[i, j] = float(row["score"])
            allowed[i, j] = bool(int(row.get("allowed", "1")))
            topk_mask[i, j] = bool(int(row.get("topk", "1")))

    return scores, allowed, topk_mask


def _save_variable_scores(
    path: str,
    station_order: list,
    feature_names: list,
    candidates: list,
    scores: list,
    topk: list,
) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "target_station",
                "source_station",
                "feature_name",
                "feature_index",
                "score",
                "selected",
            ]
        )
        for target_idx, target in enumerate(station_order):
            target_scores = scores[target_idx]
            target_topk = set(topk[target_idx])
            for cand_idx, (src_idx, feat_idx) in enumerate(candidates[target_idx]):
                writer.writerow(
                    [
                        target,
                        station_order[src_idx],
                        feature_names[feat_idx],
                        feat_idx,
                        float(target_scores[cand_idx]),
                        int(cand_idx in target_topk),
                    ]
                )


def _load_variable_scores(
    path: str,
    station_order: list,
    feature_names: list,
    candidates: list,
) -> list:
    score_map = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            target = row["target_station"]
            source = row["source_station"]
            feat_idx = int(row["feature_index"])
            score = float(row["score"])
            score_map[(target, source, feat_idx)] = score

    scores = []
    for target_idx, target in enumerate(station_order):
        target_scores = []
        for src_idx, feat_idx in candidates[target_idx]:
            key = (target, station_order[src_idx], feat_idx)
            target_scores.append(score_map.get(key, 0.0))
        scores.append(torch.tensor(target_scores, dtype=torch.float32).numpy())

    return scores


def _build_all_variable_candidates(
    station_order: list, feature_names: list, upstream_mask: torch.Tensor
) -> list:
    """生成候选变量：本站所有因子 + 上游所有站点所有因子"""
    candidates = []
    for target_idx, _ in enumerate(station_order):
        target_candidates = []
        for src_idx, _ in enumerate(station_order):
            if not upstream_mask[target_idx, src_idx]:
                continue
            # 特征索引 0 为径流，本身已经被强制输入模型
            # 因此在变量选择阶段仅考虑气象等外部因子（索引 1..）
            for feat_idx in range(1, len(feature_names)):
                target_candidates.append((src_idx, feat_idx))
        candidates.append(target_candidates)
    return candidates


def _select_topk_indices(scores: list, top_k: int) -> list:
    topk = []
    for target_scores in scores:
        if len(target_scores) == 0:
            topk.append([])
            continue
        k = min(top_k, len(target_scores))
        values = torch.tensor(target_scores)
        _, idx = torch.topk(values, k)
        topk.append(idx.tolist())
    return topk


def _build_topk_mask(
    scores: torch.Tensor, allowed: torch.Tensor, top_k: int
) -> torch.Tensor:
    if top_k <= 0:
        return allowed.clone()

    masked_scores = scores.clone()
    masked_scores[~allowed] = float("-inf")

    topk_mask = torch.zeros_like(allowed)
    for i in range(masked_scores.size(0)):
        row = masked_scores[i]
        valid = torch.isfinite(row)
        if not valid.any():
            continue
        k = min(top_k, int(valid.sum().item()))
        _, idx = torch.topk(row, k)
        topk_mask[i, idx] = True

    return topk_mask


def _save_selection_weights(
    path: str, station_order: list, weights: torch.Tensor
) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["target_station"] + station_order
        writer.writerow(header)
        for i, target in enumerate(station_order):
            row = [target] + [float(v) for v in weights[i].tolist()]
            writer.writerow(row)


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)

    console = Console()

    station_order = cfg["station_order"]

    dataset = MultiStationDataset(
        root_dir=cfg["data_root"],
        station_order=station_order,
        history_len=cfg["history"],
        horizon=cfg["horizon"],
    )

    edge_index, edge_names = build_chain_edges(station_order)
    upstream_mask = build_upstream_mask(station_order)

    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    variable_causal_path = os.path.join(output_dir, "variable_causal_scores.csv")
    selection_path = os.path.join(output_dir, "variable_selection.csv")

    original_feature_names = list(dataset.feature_names)
    candidates = _build_all_variable_candidates(
        station_order, original_feature_names, upstream_mask
    )

    causal_progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}[/]"),
        BarColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    with causal_progress:
        if cfg["use_causal_cache"] and os.path.exists(variable_causal_path):
            cache_task = causal_progress.add_task("特征分析: 读取缓存", total=1)
            variable_scores = _load_variable_scores(
                variable_causal_path, station_order, original_feature_names, candidates
            )
            causal_progress.update(cache_task, completed=1)
        else:
            task_id = causal_progress.add_task(
                "特征分析: Spearman相关+距离衰减", total=len(station_order)
            )

            def _on_target(idx: int) -> None:
                name = station_order[idx]
                causal_progress.update(
                    task_id, completed=idx + 1, description=f"特征分析: {name}"
                )

            variable_scores = compute_spearman_variable_scores(
                dataset.runoffs,
                dataset.weather,
                edge_index,
                max_decay=cfg["spearman_max_decay"],
                on_target=_on_target,
            )
            causal_progress.update(task_id, completed=len(station_order))

    topk_indices = _select_topk_indices(variable_scores, cfg["variable_top_k"])
    
    flattened_candidates = _build_all_variable_candidates(
        station_order, original_feature_names, upstream_mask
    )
    _save_variable_scores(
        variable_causal_path,
        station_order,
        original_feature_names,
        flattened_candidates,
        variable_scores,
        topk_indices,
    )

    selection_map = []
    selection_names = ["runoff"] + [
        f"top_{i+1}" for i in range(cfg["variable_top_k"])
    ]
    for target_idx, _ in enumerate(station_order):
        mapping = [(target_idx, 0)]
        for cand_idx in topk_indices[target_idx]:
            mapping.append(flattened_candidates[target_idx][cand_idx])
        while len(mapping) < 1 + cfg["variable_top_k"]:
            mapping.append((target_idx, 0))
        selection_map.append(mapping)

    dataset.set_selection_map(selection_map, selection_names)

    with open(selection_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["target_station", "rank", "source_station", "feature_name", "score"]
        )
        for target_idx, target in enumerate(station_order):
            writer.writerow([target, 0, target, "runoff", "1.0"])
            for rank, cand_idx in enumerate(topk_indices[target_idx], start=1):
                src_idx, feat_idx = candidates[target_idx][cand_idx]
                score = float(variable_scores[target_idx][cand_idx])
                feature_name = (
                    original_feature_names[feat_idx]
                    if feat_idx < len(original_feature_names)
                    else f"feature_{feat_idx}"
                )
                writer.writerow(
                    [target, rank, station_order[src_idx], feature_name, score]
                )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PhysicsGuidedGNN(
        input_dim=len(dataset.feature_names),
        hidden_dim=cfg["hidden"],
        horizon=cfg["horizon"],
        edge_index=edge_index.to(device),
        dt=cfg["dt"],
        top_k=cfg["station_top_k"],
        causal_mask=upstream_mask.to(device),
        causal_matrix=None,
        causal_alpha=cfg["causal_alpha"],
    ).to(device)

    loss_fn = PhysicsGuidedLoss(model.routing, lambda_phy=cfg["lambda_phy"])
    data_loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    pretrain_epochs = cfg.get("pretrain_epochs", 0)

    checkpoint_path = os.path.join(output_dir, cfg["checkpoint_path"])
    start_epoch = 1
    if cfg["resume"] and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        console.print(
            f"[green]已从断点恢复[/]: {checkpoint_path} (epoch {start_epoch})"
        )

    loader = DataLoader(
        dataset,
        batch_size=cfg["batch"],
        shuffle=True,
        drop_last=True,
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
        persistent_workers=cfg["persistent_workers"] and cfg["num_workers"] > 0,
    )

    model.train()
    train_progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold green]{task.description}[/]"),
        BarColumn(),
        TextColumn("loss={task.fields[loss]:.4f}"),
        TextColumn("data={task.fields[data]:.4f}"),
        TextColumn("phy={task.fields[physics]:.4f}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    with train_progress:
        train_task = train_progress.add_task(
            "训练", total=cfg["epochs"], loss=0.0, data=0.0, physics=0.0
        )
        for epoch in range(start_epoch, cfg["epochs"] + 1):
            is_pretrain = epoch <= pretrain_epochs

            # 阶段一：仅预训练节点 LSTM（以及跨站注意力和输出头），冻结路由/图层参数
            if is_pretrain:
                model.routing.requires_grad_(False)
                for layer in model.graph_layers:
                    layer.requires_grad_(False)
            else:
                model.routing.requires_grad_(True)
                for layer in model.graph_layers:
                    layer.requires_grad_(True)

            epoch_loss = 0.0
            batch_task = train_progress.add_task(
                f"训练: epoch {epoch:03d}",
                total=len(loader),
                loss=0.0,
                data=0.0,
                physics=0.0,
            )
            last_data_loss = 0.0
            last_phy_loss = 0.0
            for batch_idx, (x, y, prev_y) in enumerate(loader, start=1):
                x = x.to(device)
                y = y.to(device)
                prev_y = prev_y.to(device)

                pred = model(x)
                if is_pretrain:
                    # 仅使用数据误差预训练时序编码器
                    loss = data_loss_fn(pred, y)
                    data_part = loss
                    phy_part = torch.zeros(1, device=device)
                else:
                    loss, parts = loss_fn(pred, y, prev_y, edge_index.to(device))
                    data_part = parts["data"]
                    phy_part = parts["physics"]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                last_data_loss = data_part.item()
                last_phy_loss = phy_part.item()
                train_progress.update(
                    batch_task,
                    completed=batch_idx,
                    loss=loss.item(),
                    data=last_data_loss,
                    physics=last_phy_loss,
                )

            avg_loss = epoch_loss / len(loader)
            console.print(
                f"Epoch {epoch:03d} | loss={avg_loss:.4f} | "
                f"data={last_data_loss:.4f} | physics={last_phy_loss:.4f}"
            )
            train_progress.update(
                train_task,
                completed=epoch,
                loss=avg_loss,
                data=last_data_loss,
                physics=last_phy_loss,
            )
            train_progress.remove_task(batch_task)

            if epoch % cfg["save_checkpoint_every"] == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    checkpoint_path,
                )

            if epoch % cfg["save_selection_every"] == 0:
                model.eval()
                weight_sum = None
                count = 0
                with torch.no_grad():
                    for x, _, _ in loader:
                        x = x.to(device)
                        _, weights = model(x, return_weights=True)
                        batch_mean = weights.mean(dim=0)
                        if weight_sum is None:
                            weight_sum = batch_mean
                        else:
                            weight_sum = weight_sum + batch_mean
                        count += 1
                if weight_sum is not None:
                    avg_weights = (weight_sum / count).cpu()
                    _save_selection_weights(
                        os.path.join(
                            output_dir,
                            f"selection_weights_epoch_{epoch:03d}.csv",
                        ),
                        station_order,
                        avg_weights,
                    )
                model.train()


if __name__ == "__main__":
    main()
