import torch
from torch import nn


class LSTMNodeEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        if self.bidirectional:
            h = h.view(self.num_layers, 2, x.size(0), self.hidden_dim)
            h_last = torch.cat([h[-1, 0], h[-1, 1]], dim=-1)
        else:
            h_last = h[-1]
        return h_last
