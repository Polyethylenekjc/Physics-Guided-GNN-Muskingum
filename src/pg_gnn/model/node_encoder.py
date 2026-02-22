from typing import Dict, List

import torch
from torch import nn

from src.lstm_model import LSTMModel


class FrozenPretrainedLSTMEncoder(nn.Module):
    def __init__(
        self,
        station_order: List[str],
        station_feature_indices: Dict[str, List[int]],
        input_size_map: Dict[str, int],
        model_dir: str,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        freeze_lstm: bool = True,
    ) -> None:
        super().__init__()
        self.station_order = station_order
        self.station_feature_indices = station_feature_indices

        self.lstms = nn.ModuleDict()
        for station in station_order:
            input_size = input_size_map[station]
            backbone = LSTMModel(
                input_size=input_size,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                output_size=1,
                dropout=dropout,
                bidirectional=bidirectional,
            )

            checkpoint_path = f"{model_dir}/{station}.pt"
            try:
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                state_dict = checkpoint.get("model_state_dict", checkpoint)
                backbone.load_state_dict(state_dict, strict=False)
                print(f"已加载预训练LSTM: {checkpoint_path}")
            except Exception as exc:
                print(f"警告: 预训练LSTM加载失败({checkpoint_path})，将使用随机初始化: {exc}")

            lstm = backbone.lstm
            if freeze_lstm:
                for param in lstm.parameters():
                    param.requires_grad = False
            self.lstms[station] = lstm

        self.output_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for node_idx, station in enumerate(self.station_order):
            feature_indices = self.station_feature_indices[station]
            node_x = x[:, node_idx, :, :][:, :, feature_indices]
            lstm_out, _ = self.lstms[station](node_x)
            outputs.append(lstm_out[:, -1, :])
        return torch.stack(outputs, dim=1)
