from typing import Tuple

import torch
from torch import nn

from .routing import MuskingumRouting


class GraphRoutingLayer(nn.Module):
    def __init__(
        self, dim: int, routing: MuskingumRouting, use_dual_channel: bool = True, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.use_dual_channel = use_dual_channel
        
        self.msg_physics = nn.Linear(dim, dim, bias=False)
        self.msg_neural = nn.Linear(dim, dim, bias=False)
        self.att = nn.Linear(dim * 2, 1, bias=False)
        self.routing = routing
        
        if use_dual_channel:
            self.channel_fusion = nn.Parameter(torch.tensor(0.5))
        
        # 增加 MLP 用于消息聚合后变换
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        x_src = x[src]
        x_dst = x[dst]

        att_scores = self.att(torch.cat([x_src, x_dst], dim=-1)).squeeze(-1)
        
        att_weights = torch.zeros_like(att_scores)
        for node in range(x.size(0)):
            mask = dst == node
            if mask.any():
                att_weights[mask] = torch.softmax(att_scores[mask], dim=0)

        if self.use_dual_channel:
            routing_factor = self.routing.routing_factor()
            
            physics_msg = routing_factor[:, None] * att_weights[:, None] * self.msg_physics(x_src)
            neural_msg = att_weights[:, None] * self.msg_neural(x_src)
            
            alpha = torch.sigmoid(self.channel_fusion)
            messages = alpha * physics_msg + (1.0 - alpha) * neural_msg
        else:
            routing_factor = self.routing.routing_factor()
            messages = routing_factor[:, None] * att_weights[:, None] * self.msg_physics(x_src)

        out = torch.zeros_like(x)
        out.index_add_(0, dst, messages)

        # Pre-LN 残差连接
        out = self.norm1(self.act(out) + x)
        out = self.norm2(self.mlp(out) + out)
        return out
