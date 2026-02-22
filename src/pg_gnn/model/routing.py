import math
from typing import Tuple

import torch
from torch import nn


class MuskingumRouting(nn.Module):
    def __init__(
        self,
        edge_count: int,
        dt: float = 1.0,
        init_k: float = 1.0,
        init_x: float = 0.2,
    ) -> None:
        super().__init__()
        self.dt = dt
        self.k_raw = nn.Parameter(torch.full((edge_count,), math.log(math.exp(init_k) - 1.0)))
        init_x = min(max(init_x, 1e-3), 0.49)
        self.x_raw = nn.Parameter(torch.full((edge_count,), math.log(init_x / (0.5 - init_x))))

    def k(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.k_raw)

    def x(self) -> torch.Tensor:
        return 0.5 * torch.sigmoid(self.x_raw)

    def coefficients(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k = self.k()
        x = self.x()
        dt = self.dt

        denom = k - k * x + 0.5 * dt
        c0 = (-k * x + 0.5 * dt) / denom
        c1 = (k * x + 0.5 * dt) / denom
        c2 = (k - k * x - 0.5 * dt) / denom

        return c0, c1, c2

    def routing_factor(self) -> torch.Tensor:
        c0, _, _ = self.coefficients()
        return torch.clamp(c0, min=0.0, max=1.0)
