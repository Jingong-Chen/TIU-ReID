from __future__ import annotations

import torch
import torch.nn as nn


class LinearScrubber(nn.Module):
    """Lightweight, stable scrubber: z' = W z (optionally low-rank via bottleneck)."""

    def __init__(self, dim: int, bottleneck: int | None = None):
        super().__init__()
        if bottleneck is None or bottleneck >= dim:
            self.net = nn.Linear(dim, dim, bias=False)
        else:
            self.net = nn.Sequential(
                nn.Linear(dim, bottleneck, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(bottleneck, dim, bias=False),
            )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ResidualScrubber(nn.Module):
    """Residual scrubber: z' = z + alpha * f(z), with |alpha| <= alpha_max (alpha starts at 0)."""

    def __init__(self, dim: int, bottleneck: int | None = None, alpha_max: float = 0.1):
        super().__init__()
        # alpha starts at 0 but still receives gradients (STE)
        self.alpha_raw = nn.Parameter(torch.tensor(0.0))
        self.alpha_max = float(alpha_max)
        if bottleneck is None or bottleneck >= dim:
            self.net = nn.Linear(dim, dim, bias=False)
        else:
            self.net = nn.Sequential(
                nn.Linear(dim, bottleneck, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(bottleneck, dim, bias=False),
            )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        alpha = torch.clamp(self.alpha_raw, -self.alpha_max, self.alpha_max)
        # Straight-through estimator so alpha can keep receiving gradients even if clamped.
        alpha = alpha + (self.alpha_raw - self.alpha_raw.detach())
        return z + alpha * self.net(z)

    def alpha_value(self) -> float:
        alpha = torch.clamp(self.alpha_raw, -self.alpha_max, self.alpha_max)
        return float(alpha.detach().cpu())

    def delta(self, z: torch.Tensor) -> torch.Tensor:
        alpha = torch.clamp(self.alpha_raw, -self.alpha_max, self.alpha_max)
        alpha = alpha + (self.alpha_raw - self.alpha_raw.detach())
        return alpha * self.net(z)


class ForgetDiscriminator(nn.Module):
    def __init__(self, dim: int, n_forget_ids: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_forget_ids),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class MembershipDiscriminator(nn.Module):
    def __init__(self, dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


