"""Two-hidden-layer MLP for predicted rollout value."""

from __future__ import annotations

import torch
from torch import nn


class ValueMLP(nn.Module):
    """Two hidden linear layers (default 64, 32) plus scalar output."""

    def __init__(self, n_features: int, hidden_sizes: tuple[int, ...] = (64, 32)):
        super().__init__()
        if len(hidden_sizes) != 2:
            raise ValueError("plan specifies a two-layer MLP (two hidden widths)")
        layers: list[nn.Module] = []
        in_dim = n_features
        for width in hidden_sizes:
            layers.extend([nn.Linear(in_dim, width), nn.ReLU()])
            in_dim = width
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def weighted_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    err = pred - target
    abs_err = err.abs()
    quadratic = torch.clamp(abs_err, max=delta)
    linear = abs_err - quadratic
    loss = 0.5 * quadratic**2 + delta * linear
    return (weight * loss).sum() / weight.sum().clamp_min(1e-8)
