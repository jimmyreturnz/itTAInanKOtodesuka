"""
Lightweight long-range 1D block (Mug-Diffusion S4Layer interface).

Full Mug S4 (state-space) is large and dependency-heavy; this uses a deep
depthwise conv + pointwise conv residual path for long temporal context at
lower VRAM cost. Swap in a full S4 implementation later if needed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint


def _normalize(channels: int, num_groups: int = 8) -> nn.GroupNorm:
    while channels % num_groups != 0:
        num_groups //= 2
    return nn.GroupNorm(num_groups, channels, eps=1e-6, affine=True)


class S4StyleBlock(nn.Module):
    """
    Residual long-context block matching Mug's S4Layer placement.
    Input/output: [B, C, T]
    """

    def __init__(self, channels: int, kernel_size: int = 31, use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        pad = kernel_size // 2
        self.norm = _normalize(channels)
        self.dw = nn.Conv1d(channels, channels, kernel_size, padding=pad, groups=channels)
        self.pw = nn.Conv1d(channels, channels * 2, 1)
        self.out = nn.Conv1d(channels, channels, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.dw(h)
        h = self.pw(h)
        h, gate = h.chunk(2, dim=1)
        h = F.silu(h) * torch.sigmoid(gate)
        h = self.out(h)
        return x + h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return grad_checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)
