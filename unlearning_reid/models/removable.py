from __future__ import annotations

import torch
import torch.nn as nn


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return _GradReverse.apply(x, lambd)


class TargetModule(nn.Module):
    def __init__(self, dim: int, hidden: int | None = None, dropout: float = 0.0):
        super().__init__()
        hidden = int(hidden) if hidden is not None else None
        if hidden is None or hidden <= 0 or hidden >= dim:
            self.net = nn.Linear(dim, dim, bias=False)
        else:
            self.net = nn.Sequential(
                nn.Linear(dim, hidden, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout(p=float(dropout)),
                nn.Linear(hidden, dim, bias=False),
            )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class TargetDiscriminator(nn.Module):
    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
