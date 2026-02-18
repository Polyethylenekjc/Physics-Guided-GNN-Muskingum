import torch
from torch import nn


class LSTMNodeEncoder(nn.Module):
    """共享 LSTM 编码器（所有站点共用一个 LSTM）"""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.2,
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
        # 单向时不需要投影，双向时输出维度翻倍需要映射回 hidden_dim
        out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.proj = nn.Linear(out_dim, hidden_dim) if bidirectional else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        if self.bidirectional:
            h = h.view(self.num_layers, 2, x.size(0), self.hidden_dim)
            h_last = torch.cat([h[-1, 0], h[-1, 1]], dim=-1)
        else:
            h_last = h[-1]
        out = self.proj(h_last)
        out = self.layer_norm(self.dropout(out))
        return out


class PerNodeLSTMEncoder(nn.Module):
    """每站独立 LSTM 编码器（每个站点有自己的 LSTM 参数）"""
    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 为每个站点创建独立的 LSTM
        self.lstms = nn.ModuleList([
            nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            for _ in range(num_nodes)
        ])
        
        # 每个站点独立的输出头（用于预训练时单独预测）
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_nodes)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (num_nodes, seq_len, input_dim)
        Returns:
            embeddings: (num_nodes, hidden_dim)
        """
        embeddings = []
        for node_idx in range(self.num_nodes):
            node_x = x[node_idx].unsqueeze(0)  # (1, seq_len, input_dim)
            _, (h, _) = self.lstms[node_idx](node_x)
            h_last = h[-1].squeeze(0)  # (hidden_dim,)
            embeddings.append(h_last)
        
        out = torch.stack(embeddings, dim=0)  # (num_nodes, hidden_dim)
        out = self.layer_norm(self.dropout(out))
        return out

    def forward_with_predictions(self, x: torch.Tensor) -> tuple:
        """
        用于分站预训练：同时返回嵌入和各站的预测值
        Args:
            x: (num_nodes, seq_len, input_dim)
        Returns:
            embeddings: (num_nodes, hidden_dim)
            predictions: (num_nodes, 1) - 每个站点的预测
        """
        embeddings = []
        predictions = []
        for node_idx in range(self.num_nodes):
            node_x = x[node_idx].unsqueeze(0)  # (1, seq_len, input_dim)
            _, (h, _) = self.lstms[node_idx](node_x)
            h_last = h[-1].squeeze(0)  # (hidden_dim,)
            embeddings.append(h_last)
            pred = self.heads[node_idx](h_last)  # (1,)
            predictions.append(pred)
        
        out = torch.stack(embeddings, dim=0)  # (num_nodes, hidden_dim)
        out = self.layer_norm(self.dropout(out))
        preds = torch.stack(predictions, dim=0)  # (num_nodes, 1)
        return out, preds
