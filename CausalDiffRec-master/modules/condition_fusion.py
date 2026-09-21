"""E3: gated fusion of causal embedding z_c and semantic prior z_s."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionFusion(nn.Module):
    """Gate: z = g * z_c + (1-g) * Proj(z_s).

    Optional force_gate in [0,1]: if set, use a constant g (diagnostic).
    Smaller g => more semantic weight.
    """

    def __init__(
        self,
        causal_dim: int,
        semantic_dim: int,
        out_dim: int = None,
        force_gate: float = None,
    ):
        super().__init__()
        out_dim = out_dim or causal_dim
        self.proj_s = nn.Linear(semantic_dim, out_dim)
        self.proj_c = nn.Identity() if causal_dim == out_dim else nn.Linear(causal_dim, out_dim)
        self.gate = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, 1),
            nn.Sigmoid(),
        )
        self.out_dim = out_dim
        self.force_gate = None if force_gate is None else float(force_gate)

    def gate_values(self, z_c: torch.Tensor, z_s: torch.Tensor) -> torch.Tensor:
        """Return causal gate values g in [0, 1] for diagnostics."""
        c = self.proj_c(z_c)
        s = self.proj_s(z_s)
        if self.force_gate is None:
            return self.gate(torch.cat([c, s], dim=-1))
        return c.new_full((c.shape[0], 1), self.force_gate)

    def forward(self, z_c: torch.Tensor, z_s: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        z_c, z_s: [N, D]; mask: [N] bool — if False, keep z_c only.
        """
        c = self.proj_c(z_c)
        s = self.proj_s(z_s)
        g = self.gate_values(z_c, z_s)
        fused = g * c + (1.0 - g) * s
        if mask is not None:
            m = mask.view(-1, 1).to(dtype=fused.dtype)
            fused = m * fused + (1.0 - m) * c
        return fused
