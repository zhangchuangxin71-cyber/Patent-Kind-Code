"""Differentiable ranking objective for upstream-generated node embeddings."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def scipy_to_torch_sparse(matrix, device=None):
    coo = matrix.tocoo()
    indices = torch.from_numpy(
        np.vstack((coo.row, coo.col)).astype(np.int64),
    )
    values = torch.from_numpy(coo.data).float()
    tensor = torch.sparse_coo_tensor(indices, values, coo.shape).coalesce()
    return tensor.to(device) if device is not None else tensor


def functional_lightgcn(initial_embeddings, sparse_norm_adj, layers=3):
    """Propagate embeddings without cloning/detaching, preserving gradients."""
    propagated = initial_embeddings
    levels = [initial_embeddings]
    for _ in range(int(layers)):
        propagated = torch.sparse.mm(sparse_norm_adj, propagated)
        levels.append(propagated)
    return torch.stack(levels, dim=1).mean(dim=1)


def upstream_bpr_loss(
    initial_embeddings,
    sparse_norm_adj,
    user_ids,
    positive_item_ids,
    negative_item_ids,
    n_user,
    layers=3,
    l2_weight=1e-3,
):
    """BPR loss whose gradients flow into ``initial_embeddings`` and producers."""
    embeddings = functional_lightgcn(
        initial_embeddings, sparse_norm_adj, layers=layers,
    )
    users = embeddings[user_ids]
    positives = embeddings[positive_item_ids.long() + int(n_user)]
    negatives = embeddings[negative_item_ids.long() + int(n_user)]
    positive_scores = (users * positives).sum(dim=1)
    negative_scores = (users * negatives).sum(dim=1)
    ranking = F.softplus(negative_scores - positive_scores).mean()
    regularization = (
        users.pow(2).sum(dim=1)
        + positives.pow(2).sum(dim=1)
        + negatives.pow(2).sum(dim=1)
    ).mean()
    return ranking + float(l2_weight) * regularization
