import math
from typing import Optional, Tuple

import torch
from torch import nn


class CrossStationSelector(nn.Module):
    def __init__(self, dim: int, top_k: Optional[int] = None) -> None:
        super().__init__()
        self.top_k = top_k
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        causal_matrix: Optional[torch.Tensor] = None,
        causal_alpha: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        scores = (q @ k.t()) / math.sqrt(x.size(-1))

        if causal_matrix is not None:
            causal_safe = torch.clamp(causal_matrix, min=1e-6)
            scores = scores + causal_alpha * torch.log(causal_safe)

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        if self.top_k is not None and self.top_k < x.size(0):
            topk_vals, topk_idx = torch.topk(scores, self.top_k, dim=-1)
            mask = torch.full_like(scores, float("-inf"))
            mask.scatter_(1, topk_idx, topk_vals)
            scores = mask

        weights = torch.softmax(scores, dim=-1)
        fused = weights @ v

        gate = self.gate(torch.cat([x, fused], dim=-1))
        out = gate * x + (1.0 - gate) * fused

        return out, weights
