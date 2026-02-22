import argparse
import csv
import os
import sys
from typing import Dict, Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.metrics import calculate_all_metrics
from src.pg_gnn import MultiStationDataset, build_chain_edges
from src.pg_gnn.model import PhysicsGuidedGNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/pg_gnn.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def split_indices(total_len: int, train_ratio: float, val_ratio: float):
    train_end = int(total_len * train_ratio)
    val_end = train_end + int(total_len * val_ratio)
    test_idx = list(range(val_end, total_len))
    return test_idx


def inverse_by_station(values: np.ndarray, scalers) -> np.ndarray:
    restored = np.zeros_like(values, dtype=np.float32)
    for station_idx in range(values.shape[1]):
        restored[:, station_idx] = scalers[station_idx].inverse_transform(
            values[:, station_idx].reshape(-1, 1)
        ).reshape(-1)
    return restored


def save_metrics(metrics_by_station, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "evaluation_metrics.csv")

    fieldnames = ["station", "NSE", "MSE", "RMSE", "MAE", "MAPE", "R2"]
    with open(save_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for station_name, metrics in metrics_by_station.items():
            row = {"station": station_name}
            row.update(metrics)
            writer.writerow(row)
    return save_path


def save_predictions(y_true: np.ndarray, y_pred: np.ndarray, station_order, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "evaluation_predictions.csv")

    header = ["sample_index"]
    for station in station_order:
        header.extend([f"{station}_true", f"{station}_pred", f"{station}_error"])

    with open(save_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for sample_idx in range(y_true.shape[0]):
            row = [sample_idx]
            for station_idx in range(y_true.shape[1]):
                true_val = float(y_true[sample_idx, station_idx])
                pred_val = float(y_pred[sample_idx, station_idx])
                row.extend([true_val, pred_val, true_val - pred_val])
            writer.writerow(row)
    return save_path


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    checkpoint_path = args.checkpoint or config["checkpoint_path"]
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到checkpoint: {checkpoint_path}")

    station_order = config["station_order"]
    dataset = MultiStationDataset(
        weather_dir=config["data"]["weather_dir"],
        station_order=station_order,
        history_len=config["history"],
        horizon=config["horizon"],
        selection_dir=config["data"]["selection_dir"],
        train_ratio=config["train_ratio"],
    )

    test_indices = split_indices(len(dataset), config["train_ratio"], config["val_ratio"])
    test_loader = DataLoader(
        Subset(dataset, test_indices),
        batch_size=config["train"]["batch"],
        shuffle=False,
    )

    edge_index, _ = build_chain_edges(station_order)
    device = torch.device(config["train"]["device"] if torch.cuda.is_available() else "cpu")

    model = PhysicsGuidedGNN(
        station_order=station_order,
        station_feature_indices=dataset.station_feature_indices,
        input_size_map=dataset.input_size_map,
        model_dir=config["model"]["pretrained_model_dir"],
        hidden_dim=config["model"]["hidden"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        bidirectional=config["model"]["bidirectional"],
        edge_index=edge_index.to(device),
        dt=config["model"]["dt"],
        graph_layers=config["model"]["graph_layers"],
        freeze_lstm=config["model"]["freeze_lstm"],
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()

    all_true = []
    all_pred = []

    with torch.no_grad():
        for x, y, _ in test_loader:
            x = x.to(device)
            pred = model(x)
            all_true.append(y.cpu().numpy())
            all_pred.append(pred.cpu().numpy())

    y_true_scaled = np.concatenate(all_true, axis=0)
    y_pred_scaled = np.concatenate(all_pred, axis=0)

    y_true = inverse_by_station(y_true_scaled, dataset.runoff_scalers)
    y_pred = inverse_by_station(y_pred_scaled, dataset.runoff_scalers)

    metrics_by_station = {}
    print("\n===== Test Metrics (Original Scale) =====")
    for station_idx, station_name in enumerate(station_order):
        metrics = calculate_all_metrics(y_true[:, station_idx], y_pred[:, station_idx])
        metrics_by_station[station_name] = {
            key: (float(value) if not np.isnan(value) else np.nan)
            for key, value in metrics.items()
        }
        print(
            f"{station_name:25s} | NSE={metrics['NSE']:.4f} | RMSE={metrics['RMSE']:.4f} | MAE={metrics['MAE']:.4f}"
        )

    eval_output_dir = os.path.join(config["output_dir"], "evaluation")
    metrics_path = save_metrics(metrics_by_station, eval_output_dir)
    pred_path = save_predictions(y_true, y_pred, station_order, eval_output_dir)

    print("\n===== Saved =====")
    print(f"metrics: {metrics_path}")
    print(f"predictions: {pred_path}")


if __name__ == "__main__":
    main()
