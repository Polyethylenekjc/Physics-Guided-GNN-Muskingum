from typing import Dict, Tuple

import torch
from torch import nn

from .routing import MuskingumRouting


class PhysicsGuidedLoss(nn.Module):
    def __init__(self, routing: MuskingumRouting, lambda_phy: float = 0.3) -> None:
        super().__init__()
        self.routing = routing
        self.lambda_phy = lambda_phy
        self.mse = nn.MSELoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        prev_target: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        data_loss = self.mse(pred, target)

        c0, c1, c2 = self.routing.coefficients()
        src, dst = edge_index

        expected = (
            c0[None, :] * pred[:, src]
            + c1[None, :] * prev_target[:, src]
            + c2[None, :] * prev_target[:, dst]
        )
        residual = pred[:, dst] - expected
        phy_loss = (residual ** 2).mean()

        total = data_loss + self.lambda_phy * phy_loss
        return total, {"data": data_loss, "physics": phy_loss}
