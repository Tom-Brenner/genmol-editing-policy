"""Value models: locked two-layer MLP, and the V2 tabular logit MLP."""

from __future__ import annotations

import torch
import torch.nn.functional as F
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


def binomial_nll(
    logits: torch.Tensor,
    successes: torch.Tensor,
    trials: torch.Tensor,
) -> torch.Tensor:
    """Negative log likelihood, excluding the parameter-free combinatorial term.

    Equivalent to trial-weighted binary cross-entropy with target
    ``successes / trials``. Sampling variance is in the likelihood, so do not
    also apply inverse reward-variance weights.
    """
    if logits.shape != successes.shape or successes.shape != trials.shape:
        raise ValueError("logits, successes, and trials must share a shape")
    if torch.any(successes < 0) or torch.any(successes > trials):
        raise ValueError("binomial counts must satisfy 0 <= successes <= trials")
    per_action = trials * F.softplus(logits) - successes * logits
    return per_action.sum() / trials.sum().clamp_min(1.0)


class TabularMLP(nn.Module):
    """Shallow regularized MLP that returns one binomial logit per action."""

    def __init__(
        self,
        n_features: int,
        hidden_sizes: tuple[int, ...] = (64,),
        dropout: float = 0.15,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = n_features
        for width in hidden_sizes:
            layers += [
                nn.Linear(in_dim, width),
                nn.ReLU(),
                nn.LayerNorm(width),
                nn.Dropout(dropout),
            ]
            in_dim = width
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
