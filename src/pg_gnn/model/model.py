from typing import Dict, List

import torch
from torch import nn

from .graph_layer import GraphRoutingLayer
from .node_encoder import FrozenPretrainedLSTMEncoder
from .routing import MuskingumRouting


class PhysicsGuidedGNN(nn.Module):
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
        edge_index: torch.Tensor,
        dt: float,
        graph_layers: int,
        freeze_lstm: bool = True,
    ) -> None:
        super().__init__()
        self.edge_index = edge_index
        self.encoder = FrozenPretrainedLSTMEncoder(
            station_order=station_order,
            station_feature_indices=station_feature_indices,
            input_size_map=input_size_map,
            model_dir=model_dir,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            freeze_lstm=freeze_lstm,
        )

        embed_dim = self.encoder.output_dim
        self.routing = MuskingumRouting(edge_count=edge_index.size(1), dt=dt)
        self.gnn_layers = nn.ModuleList([
            GraphRoutingLayer(dim=embed_dim, routing=self.routing, dropout=dropout)
            for _ in range(graph_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        for layer in self.gnn_layers:
            h = layer(h, self.edge_index)
        pred = self.head(h).squeeze(-1)
        return pred
