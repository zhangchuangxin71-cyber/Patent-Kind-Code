"""Leakage-free score-level semantic fusion utilities."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F


def build_user_semantic_profiles(
    item_semantic: torch.Tensor,
    train_interactions: sp.csr_matrix,
) -> torch.Tensor:
    """Mean-pool each user's train-item semantics and L2-normalize."""
    if train_interactions.shape[1] != item_semantic.shape[0]:
        raise ValueError("interaction item count does not match semantic prior")
    profiles = item_semantic.new_zeros(
        (train_interactions.shape[0], item_semantic.shape[1]),
    )
    for user in range(train_interactions.shape[0]):
        start, end = train_interactions.indptr[user:user + 2]
        items = train_interactions.indices[start:end]
        if len(items):
            idx = torch.as_tensor(items, dtype=torch.long, device=item_semantic.device)
            profiles[user] = item_semantic[idx].mean(dim=0)
    return F.normalize(profiles, dim=1)


def row_standardize(scores: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Standardize each user's candidate scores to make score scales comparable."""
    return (scores - scores.mean(dim=1, keepdim=True)) / scores.std(
        dim=1, keepdim=True,
    ).clamp_min(eps)


def late_fusion_scores(
    collaborative_scores: torch.Tensor,
    semantic_scores: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """z(collaborative) + alpha * z(semantic), with alpha fixed on validation."""
    if collaborative_scores.shape != semantic_scores.shape:
        raise ValueError("collaborative and semantic score shapes must match")
    return row_standardize(collaborative_scores) + float(alpha) * row_standardize(
        semantic_scores,
    )


def adaptive_late_fusion_scores(
    collaborative_scores: torch.Tensor,
    semantic_scores: torch.Tensor,
    row_alphas: torch.Tensor,
) -> torch.Tensor:
    """Late fusion with a frozen semantic weight for each user row."""
    if collaborative_scores.shape != semantic_scores.shape:
        raise ValueError("collaborative and semantic score shapes must match")
    alpha = torch.as_tensor(
        row_alphas, dtype=collaborative_scores.dtype,
        device=collaborative_scores.device,
    ).reshape(-1, 1)
    if alpha.shape[0] != collaborative_scores.shape[0]:
        raise ValueError("row_alphas length must match the score row count")
    return row_standardize(collaborative_scores) + alpha * row_standardize(
        semantic_scores,
    )
