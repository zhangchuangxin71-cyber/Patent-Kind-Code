"""E3: InfoNCE semantic alignment loss (item side)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticInfoNCE(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z: torch.Tensor, z_s: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        z, z_s: [B, D] (item embeddings vs semantic priors)
        mask: [B] optional — only positives with text
        """
        if mask is not None:
            idx = mask.nonzero(as_tuple=False).view(-1)
            if idx.numel() < 2:
                return z.new_zeros(())
            z = z[idx]
            z_s = z_s[idx]
        z = F.normalize(z, dim=-1)
        z_s = F.normalize(z_s, dim=-1)
        logits = z @ z_s.t() / self.temperature
        labels = torch.arange(z.size(0), device=z.device)
        return F.cross_entropy(logits, labels)
