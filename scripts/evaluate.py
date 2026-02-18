import argparse
import csv
import os
from typing import Dict, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt
import yaml
import torch
from torch.utils.data import DataLoader, Subset

from pg_gnn.data import MultiStationDataset
from pg_gnn.graph import build_chain_edges, build_upstream_mask
from pg_gnn.model.model import PhysicsGuidedGNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/pg_gnn.yaml")
    parser.add_argument("--checkpoint", type=str, default="output/checkpoint.pt")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_train_val_test(total_len: int, train_ratio: float = 0.8) -> Tuple[int, int, int]:
    train_end = int(total_len * train_ratio)
    val_test_len = total_len - train_end
    test_end = train_end + val_test_len // 2
    return 0, train_end, test_end, total_len


def _compute_nse(obs: np.ndarray, pred: np.ndarray) -> float:
    numerator = np.sum((obs - pred) ** 2)
    denominator = np.sum((obs - obs.mean()) ** 2)
    if denominator < 1e-10:
        return np.nan
    return float(1.0 - numerator / denominator)


def _compute_kge(obs: np.ndarray, pred: np.ndarray) -> float:
    mean_obs = obs.mean()
    mean_pred = pred.mean()
    std_obs = obs.std()
    std_pred = pred.std()
    cov = np.mean((obs - mean_obs) * (pred - mean_pred))

    if std_obs < 1e-10 or std_pred < 1e-10:
        return np.nan

    r = cov / (std_obs * std_pred) if std_obs > 0 and std_pred > 0 else 0.0
    alpha = std_pred / std_obs
    beta = mean_pred / mean_obs if mean_obs != 0 else 0.0

    kge = 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return float(kge)


def _compute_mape(obs: np.ndarray, pred: np.ndarray) -> float:
    mask = obs != 0
    if not mask.any():
        return np.nan
    return float(100.0 * np.mean(np.abs((obs[mask] - pred[mask]) / obs[mask])))


def _compute_mse(obs: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean((obs - pred) ** 2))


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    station_order: list,
    device: torch.device,
    output_dir: str,
) -> None:
    model.eval()
    
    all_preds = {station: [] for station in station_order}
    all_targets = {station: [] for station in station_order}
    
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)
            
            pred = model(x)
            
            for i, station in enumerate(station_order):
                all_preds[station].append(pred[:, i, :].cpu().numpy())
                all_targets[station].append(y[:, i, :].cpu().numpy())
    
    metrics = []
    for station in station_order:
        preds = np.concatenate(all_preds[station], axis=0).flatten()
        targets = np.concatenate(all_targets[station], axis=0).flatten()
        
        nse = _compute_nse(targets, preds)
        kge = _compute_kge(targets, preds)
        mape = _compute_mape(targets, preds)
        mse = _compute_mse(targets, preds)
        
        metrics.append({
            "station": station,
            "nse": nse,
            "kge": kge,
            "mape": mape,
            "mse": mse,
        })
        
        print(f"{station:20s} | NSE={nse:7.4f} | KGE={kge:7.4f} | MAPE={mape:7.2f}% | MSE={mse:7.4f}")
    
    metrics_path = os.path.join(output_dir, "evaluation_metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["station", "nse", "kge", "mape", "mse"])
        writer.writeheader()
        writer.writerows(metrics)
    print(f"\n已保存指标到: {metrics_path}")


def visualize(
    model: torch.nn.Module,
    dataset: MultiStationDataset,
    loader: DataLoader,
    station_order: list,
    device: torch.device,
    output_dir: str,
) -> None:
    model.eval()
    
    all_preds = {station: [] for station in station_order}
    all_targets = {station: [] for station in station_order}
    
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)
            
            pred = model(x)
            
            for i, station in enumerate(station_order):
                all_preds[station].append(pred[:, i, :].cpu().numpy())
                all_targets[station].append(y[:, i, :].cpu().numpy())
    
    for i, station in enumerate(station_order):
        preds = np.concatenate(all_preds[station], axis=0).flatten()
        targets = np.concatenate(all_targets[station], axis=0).flatten()
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 5))
        
        time_steps = np.arange(len(targets))
        ax.plot(time_steps, targets, label="Observed", color="blue", linewidth=1.5, alpha=0.7)
        ax.plot(time_steps, preds, label="Predicted", color="red", linewidth=1.5, alpha=0.7)
        
        ax.set_xlabel("Time Steps", fontsize=12)
        ax.set_ylabel("Runoff (Z-score normalized)", fontsize=12)
        ax.set_title(f"Station: {station}", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plot_path = os.path.join(output_dir, f"plot_{station.replace(' ', '_')}.png")
        plt.savefig(plot_path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"已保存图表: {plot_path}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    
    station_order = cfg["station_order"]
    dataset = MultiStationDataset(
        root_dir=cfg["data_root"],
        station_order=station_order,
        history_len=cfg["history"],
        horizon=cfg["horizon"],
    )
    
    edge_index, _ = build_chain_edges(station_order)
    upstream_mask = build_upstream_mask(station_order)
    
    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    variable_causal_path = os.path.join(output_dir, "variable_causal_scores.csv")
    
    if os.path.exists(variable_causal_path):
        print(f"已找到变量选择缓存: {variable_causal_path}")
        with open(variable_causal_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            selection_map = [[] for _ in station_order]
            for row in reader:
                target = row["target_station"]
                if target not in station_order:
                    continue
                target_idx = station_order.index(target)
                source = row["source_station"]
                if source not in station_order:
                    continue
                source_idx = station_order.index(source)
                feat_idx = int(row["feature_index"])
                selected = int(row["selected"])
                if selected:
                    selection_map[target_idx].append((source_idx, feat_idx))
        
        non_empty_count = sum(1 for m in selection_map if len(m) > 0)
        if non_empty_count > 0:
            for i in range(len(selection_map)):
                if len(selection_map[i]) == 0:
                    selection_map[i] = [(i, 0)]
            selection_names = ["runoff"] + [f"top_{i+1}" for i in range(cfg["variable_top_k"])]
            dataset.set_selection_map(selection_map, selection_names)
            print(f"已应用变量选择映射 ({non_empty_count}/{len(station_order)} 站点)")
        else:
            print(f"警告：未找到任何选中的变量，将使用全部特征")
    else:
        print(f"未找到变量选择缓存，将使用全部特征")
    
    split_start, split_train_end, split_test_start, split_end = split_train_val_test(
        len(dataset), train_ratio=0.8
    )
    
    indices = list(range(split_test_start, split_end))
    test_dataset = torch.utils.data.Subset(dataset, indices)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
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
    
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])
        print(f"已加载检查点: {args.checkpoint}")
    else:
        print(f"未找到检查点: {args.checkpoint}")
        return
    
    print("\n===== 评估指标 =====")
    evaluate(model, test_loader, station_order, device, output_dir)
    
    print("\n===== 生成可视化图表 =====")
    visualize(model, dataset, test_loader, station_order, device, output_dir)
    
    print("\n===== 完成 =====")


if __name__ == "__main__":
    main()
