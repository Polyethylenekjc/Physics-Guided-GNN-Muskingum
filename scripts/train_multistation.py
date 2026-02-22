import argparse
import os
from typing import Dict, Any
import sys

import torch
import yaml
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.pg_gnn import MultiStationDataset, build_chain_edges
from src.pg_gnn.model import PhysicsGuidedGNN, PhysicsGuidedLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/pg_gnn.yaml")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_indices(total_len: int, train_ratio: float, val_ratio: float):
    train_end = int(total_len * train_ratio)
    val_end = train_end + int(total_len * val_ratio)
    train_idx = list(range(0, train_end))
    val_idx = list(range(train_end, val_end))
    test_idx = list(range(val_end, total_len))
    return train_idx, val_idx, test_idx


def evaluate(model, loader, criterion, edge_index, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, y, prev_y in loader:
            x = x.to(device)
            y = y.to(device)
            prev_y = prev_y.to(device)
            pred = model(x)
            loss, _ = criterion(pred, y, prev_y, edge_index)
            total += loss.item()
    return total / max(1, len(loader))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    station_order = cfg["station_order"]
    dataset = MultiStationDataset(
        weather_dir=cfg["data"]["weather_dir"],
        station_order=station_order,
        history_len=cfg["history"],
        horizon=cfg["horizon"],
        selection_dir=cfg["data"]["selection_dir"],
        train_ratio=cfg["train_ratio"],
    )

    train_idx, val_idx, _ = split_indices(len(dataset), cfg["train_ratio"], cfg["val_ratio"])
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=cfg["train"]["batch"], shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=cfg["train"]["batch"], shuffle=False)

    edge_index, _ = build_chain_edges(station_order)
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")

    model = PhysicsGuidedGNN(
        station_order=station_order,
        station_feature_indices=dataset.station_feature_indices,
        input_size_map=dataset.input_size_map,
        model_dir=cfg["model"]["pretrained_model_dir"],
        hidden_dim=cfg["model"]["hidden"],
        num_layers=cfg["model"]["num_layers"],
        dropout=cfg["model"]["dropout"],
        bidirectional=cfg["model"]["bidirectional"],
        edge_index=edge_index.to(device),
        dt=cfg["model"]["dt"],
        graph_layers=cfg["model"]["graph_layers"],
        freeze_lstm=cfg["model"]["freeze_lstm"],
    ).to(device)

    criterion = PhysicsGuidedLoss(model.routing, lambda_phy=cfg["train"]["lambda_phy"])
    optim_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        optim_params,
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    print(f"可训练参数量: {sum(p.numel() for p in optim_params):,}")
    print(f"训练设备: {device}")

    best_val = float("inf")
    os.makedirs(cfg["output_dir"], exist_ok=True)

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        running = 0.0
        for x, y, prev_y in train_loader:
            x = x.to(device)
            y = y.to(device)
            prev_y = prev_y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss, parts = criterion(pred, y, prev_y, edge_index.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(optim_params, cfg["train"]["grad_clip"])
            optimizer.step()
            running += loss.item()

        train_loss = running / max(1, len(train_loader))
        val_loss = evaluate(model, val_loader, criterion, edge_index.to(device), device)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val": best_val,
                    "station_order": station_order,
                    "global_feature_names": dataset.global_feature_names,
                    "station_feature_indices": dataset.station_feature_indices,
                },
                cfg["checkpoint_path"],
            )

        print(f"Epoch {epoch:03d} | train={train_loss:.6f} | val={val_loss:.6f} | best={best_val:.6f}")


if __name__ == "__main__":
    main()
