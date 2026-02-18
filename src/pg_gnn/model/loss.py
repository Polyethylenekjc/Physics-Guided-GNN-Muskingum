from typing import Dict, Tuple

import torch
from torch import nn

from .routing import MuskingumRouting


class PhysicsGuidedLoss(nn.Module):
    def __init__(self, routing: MuskingumRouting, lambda_phy: float = 1.0) -> None:
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
        if pred.dim() == 2:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
            prev_target = prev_target.unsqueeze(0)

        data_loss = self.mse(pred, target)

        c0, c1, c2 = self.routing.coefficients()
        src, dst = edge_index

        phy_losses = []
        for b in range(pred.size(0)):
            pred_t = pred[b, :, 0]
            prev = prev_target[b]

            upstream_t = pred_t[src]
            downstream_t = pred_t[dst]
            upstream_prev = prev[src]
            downstream_prev = prev[dst]

            expected = c0 * upstream_t + c1 * upstream_prev + c2 * downstream_prev
            residual = downstream_t - expected
            phy_losses.append((residual ** 2).mean())

        phy_loss = torch.stack(phy_losses).mean()
        total = data_loss + self.lambda_phy * phy_loss

        return total, {"data": data_loss, "physics": phy_loss}
