from typing import Optional, Tuple

import torch
from torch import nn

from .cross_station import CrossStationSelector
from .graph_layer import GraphRoutingLayer
from .node_encoder import LSTMNodeEncoder
from .routing import MuskingumRouting


class PhysicsGuidedGNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        horizon: int,
        edge_index: torch.Tensor,
        dt: float = 1.0,
        num_graph_layers: int = 2,
        top_k: Optional[int] = None,
        causal_mask: Optional[torch.Tensor] = None,
        causal_matrix: Optional[torch.Tensor] = None,
        causal_alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.edge_index = edge_index
        self.horizon = horizon
        self.causal_mask = causal_mask
        self.causal_matrix = causal_matrix
        self.causal_alpha = causal_alpha

        self.encoder = LSTMNodeEncoder(input_dim, hidden_dim)
        self.cross_selector = CrossStationSelector(hidden_dim, top_k=top_k)

        self.routing = MuskingumRouting(edge_index.size(1), dt=dt)
        self.graph_layers = nn.ModuleList(
            [
                GraphRoutingLayer(hidden_dim, self.routing, use_dual_channel=True)
                for _ in range(num_graph_layers)
            ]
        )

        self.head = nn.Linear(hidden_dim, horizon)

    def forward(self, x: torch.Tensor, return_weights: bool = False) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)

        outputs = []
        weights_out = []
        for b in range(x.size(0)):
            # 仅使用节点自身的 LSTM 表征，取消跨站注意力融合
            node_emb = self.encoder(x[b])
            # 为兼容返回权重的接口，这里构造单位矩阵作为“自注意”权重
            if return_weights:
                num_nodes = node_emb.size(0)
                weights = torch.eye(num_nodes, device=node_emb.device)

            for layer in self.graph_layers:
                node_emb = layer(node_emb, self.edge_index)

            outputs.append(self.head(node_emb))
            if return_weights:
                weights_out.append(weights)

        preds = torch.stack(outputs, dim=0)
        if return_weights:
            return preds, torch.stack(weights_out, dim=0)
        return preds
