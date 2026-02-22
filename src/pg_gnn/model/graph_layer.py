import torch
from torch import nn

from .routing import MuskingumRouting


class GraphRoutingLayer(nn.Module):
    def __init__(self, dim: int, routing: MuskingumRouting, dropout: float = 0.1) -> None:
        super().__init__()
        self.msg = nn.Linear(dim, dim, bias=False)
        self.routing = routing
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        routing_factor = self.routing.routing_factor()

        messages = self.msg(x[:, src, :]) * routing_factor[None, :, None]
        out = torch.zeros_like(x)
        for edge_id in range(edge_index.size(1)):
            out[:, dst[edge_id], :] += messages[:, edge_id, :]

        out = self.norm1(x + torch.nn.functional.gelu(out))
        out = self.norm2(out + self.mlp(out))
        return out
