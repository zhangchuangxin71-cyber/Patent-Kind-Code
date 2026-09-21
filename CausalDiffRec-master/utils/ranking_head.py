"""Shared ranking-head lifecycle for controlled E0/LSCI comparisons."""
from __future__ import annotations

import torch

from modules.rec_model import LGCN_Encoder


def distribution_matched_random_embeddings(
    user_embeddings,
    item_embeddings,
    seed,
    eps=1e-8,
):
    """Return random embeddings with exactly matched per-dimension moments."""
    source = torch.cat([user_embeddings, item_embeddings], dim=0)
    generator = torch.Generator(device=source.device)
    generator.manual_seed(int(seed))
    random = torch.randn(
        source.shape, dtype=source.dtype, device=source.device,
        generator=generator,
    )
    random = random - random.mean(dim=0, keepdim=True)
    random = random / random.std(
        dim=0, keepdim=True, unbiased=False,
    ).clamp_min(eps)
    target = (
        random * source.std(dim=0, keepdim=True, unbiased=False)
        + source.mean(dim=0, keepdim=True)
    )
    return target[:len(user_embeddings)], target[len(user_embeddings):]


def initialize_or_refresh_ranking_head(
    num_user,
    norm_adj,
    user_embeddings,
    item_embeddings,
    device,
    rec_model=None,
    rec_optimizer=None,
    refresh=0.25,
    learning_rate=0.001,
    layers=3,
):
    """Create once, then refresh the same trainable state on later epochs."""
    refresh = float(refresh)
    if not 0.0 <= refresh <= 1.0:
        raise ValueError(f"refresh must be in [0, 1], got {refresh}")
    if rec_model is None:
        if rec_optimizer is not None:
            raise ValueError("rec_optimizer cannot exist before rec_model")
        rec_model = LGCN_Encoder(
            num_user, int(layers), norm_adj, user_embeddings, item_embeddings,
        ).to(device)
        rec_optimizer = torch.optim.Adam(
            rec_model.parameters(), lr=float(learning_rate),
        )
        return rec_model, rec_optimizer
    if rec_optimizer is None:
        raise ValueError("persistent rec_model requires rec_optimizer")
    with torch.no_grad():
        rec_model.embedding_dict["user_emb"].lerp_(user_embeddings, refresh)
        rec_model.embedding_dict["item_emb"].lerp_(item_embeddings, refresh)
    return rec_model, rec_optimizer
