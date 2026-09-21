"""Compact action-conditioned GATv2. Returns one binomial logit per graph."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.utils import scatter


def masked_mean(h: torch.Tensor, mask: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    weights = mask.to(h.dtype).unsqueeze(-1)
    total = scatter(h * weights, batch, dim=0, reduce="sum")
    count = scatter(weights, batch, dim=0, reduce="sum").clamp_min(1.0)
    return total / count


def _batch_index(data) -> torch.Tensor:
    batch = getattr(data, "batch", None)
    if batch is None:
        return torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)
    return batch


class ActionGATv2(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        global_dim: int,
        hidden_dim: int = 64,
        heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.15,
    ):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)
        self.global_dim = int(global_dim)
        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        self.n_layers = int(n_layers)
        self.dropout_p = float(dropout)
        # Two attachment indicators are concatenated onto chemistry node features.
        self.node_proj = nn.Linear(self.node_dim + 2, hidden_dim)
        self.layers = nn.ModuleList(
            [
                GATv2Conv(
                    hidden_dim,
                    hidden_dim // heads,
                    heads=heads,
                    concat=True,
                    edge_dim=edge_dim,
                    dropout=dropout,
                    add_self_loops=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        pooled_dim = hidden_dim * 3 + global_dim
        self.head = nn.Sequential(
            nn.Linear(pooled_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def model_config(self) -> dict[str, int | float]:
        return {
            "node_dim": self.node_dim,
            "edge_dim": self.edge_dim,
            "global_dim": self.global_dim,
            "hidden_dim": self.hidden_dim,
            "heads": self.heads,
            "n_layers": self.n_layers,
            "dropout": self.dropout_p,
        }

    def forward(self, data) -> torch.Tensor:
        batch = _batch_index(data)
        attach = torch.stack(
            [
                data.core_attachment_mask.to(dtype=data.x.dtype),
                data.fragment_attachment_mask.to(dtype=data.x.dtype),
            ],
            dim=-1,
        )
        h = self.node_proj(torch.cat([data.x, attach], dim=-1))
        for conv, norm in zip(self.layers, self.norms):
            update = conv(h, data.edge_index, data.edge_attr)
            h = norm(h + self.dropout(torch.relu(update)))
        whole = global_mean_pool(h, batch)
        core = masked_mean(h, data.core_mask, batch)
        fragment = masked_mean(h, data.fragment_mask, batch)
        global_features = data.global_features.view(whole.shape[0], -1)
        pooled = torch.cat([whole, core, fragment, global_features], dim=-1)
        return self.head(pooled).squeeze(-1)
