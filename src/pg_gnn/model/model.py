from typing import Optional, Tuple

import torch
from torch import nn

from .cross_station import CrossStationSelector
from .graph_layer import GraphRoutingLayer
from .node_encoder import LSTMNodeEncoder, PerNodeLSTMEncoder
from .routing import MuskingumRouting


class PhysicsGuidedGNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        horizon: int,
        edge_index: torch.Tensor,
        num_nodes: int = 7,
        dt: float = 1.0,
        num_graph_layers: int = 2,
        top_k: Optional[int] = None,
        causal_mask: Optional[torch.Tensor] = None,
        causal_matrix: Optional[torch.Tensor] = None,
        causal_alpha: float = 1.0,
        use_per_node_lstm: bool = True,
    ) -> None:
        super().__init__()
        self.edge_index = edge_index
        self.horizon = horizon
        self.causal_mask = causal_mask
        self.causal_matrix = causal_matrix
        self.causal_alpha = causal_alpha
        self.use_per_node_lstm = use_per_node_lstm
        self.num_nodes = num_nodes

        # Use per-node LSTM or shared LSTM
        if use_per_node_lstm:
            self.encoder = PerNodeLSTMEncoder(num_nodes, input_dim, hidden_dim)
        else:
            self.encoder = LSTMNodeEncoder(input_dim, hidden_dim)
        self.cross_selector = CrossStationSelector(hidden_dim, top_k=top_k)

        self.routing = MuskingumRouting(edge_index.size(1), dt=dt)
        self.graph_layers = nn.ModuleList(
            [
                GraphRoutingLayer(hidden_dim, self.routing, use_dual_channel=True)
                for _ in range(num_graph_layers)
            ]
        )

        # Output head: MLP + Dropout
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(
        self, x: torch.Tensor, return_weights: bool = False, pretrain_mode: bool = False
    ) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)

        outputs = []
        weights_out = []
        pretrain_preds = []

        for b in range(x.size(0)):
            # Use node encoder
            if pretrain_mode and self.use_per_node_lstm:
                # Pretrain mode: use per-station LSTM prediction heads
                node_emb, station_preds = self.encoder.forward_with_predictions(x[b])
                pretrain_preds.append(station_preds)
            else:
                node_emb = self.encoder(x[b])

            # For compatibility with return_weights interface
            if return_weights:
                num_nodes = node_emb.size(0)
                weights = torch.eye(num_nodes, device=node_emb.device)

            # Skip graph layers during pretraining
            if not pretrain_mode:
                for layer in self.graph_layers:
                    node_emb = layer(node_emb, self.edge_index)

            outputs.append(self.head(node_emb))
            if return_weights:
                weights_out.append(weights)

        preds = torch.stack(outputs, dim=0)

        # Pretrain mode returns per-station predictions
        if pretrain_mode and self.use_per_node_lstm:
            pretrain_preds = torch.stack(pretrain_preds, dim=0)  # (batch, num_nodes, 1)
            return pretrain_preds

        if return_weights:
            return preds, torch.stack(weights_out, dim=0)
        return preds
