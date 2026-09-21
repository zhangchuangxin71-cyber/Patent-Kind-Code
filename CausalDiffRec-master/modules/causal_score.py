import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalEdgeScorer(nn.Module):
    """Predict per-edge causal stability scores in (0, 1)."""

    def __init__(self, emb_dim: int, hidden: int = 64):
        super().__init__()
        # Stable node-pair features -> score.  Environment statistics are used
        # as a detached supervision target in InvariantLoss, not as zero-filled
        # inputs that disappear at inference time.
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        env_edge_predictions: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            node_embeddings: [N, D]
            edge_index: [2, E]
            env_edge_predictions: deprecated and ignored; retained for API compatibility
        Returns:
            edge_scores: [E] in (0, 1)
        """
        src, dst = edge_index[0], edge_index[1]
        hu = node_embeddings[src]
        hi = node_embeddings[dst]

        feat = torch.cat([hu, hi], dim=-1)
        scores = torch.sigmoid(self.mlp(feat)).squeeze(-1)
        return scores


def split_causal_variant_edges(edge_index: torch.Tensor, scores: torch.Tensor, keep_ratio: float = 0.7):
    """Keep top-ratio undirected pairs as causal, preserving both directions."""
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
    n_nodes = int(edge_index.max().item()) + 1
    lo = torch.minimum(edge_index[0], edge_index[1]).long()
    hi = torch.maximum(edge_index[0], edge_index[1]).long()
    pair_keys = lo * n_nodes + hi
    unique_keys, inverse = torch.unique(pair_keys, return_inverse=True)
    pair_scores = torch.zeros(unique_keys.numel(), device=scores.device, dtype=scores.dtype)
    pair_counts = torch.zeros_like(pair_scores)
    pair_scores.scatter_add_(0, inverse, scores)
    pair_counts.scatter_add_(0, inverse, torch.ones_like(scores))
    pair_scores = pair_scores / pair_counts.clamp_min(1.0)
    k = max(1, int(unique_keys.numel() * keep_ratio))
    topk = torch.topk(pair_scores, k=k, largest=True).indices
    causal_pairs = torch.zeros(unique_keys.numel(), dtype=torch.bool, device=scores.device)
    causal_pairs[topk] = True
    causal_mask = causal_pairs[inverse]
    causal_edge_index = edge_index[:, causal_mask]
    variant_edge_index = edge_index[:, ~causal_mask]
    return causal_edge_index, variant_edge_index, causal_mask


def _pair_index(edge_index: torch.Tensor):
    """Map directed entries of an undirected interaction graph to pair ids."""
    n_nodes = int(edge_index.max().item()) + 1
    lo = torch.minimum(edge_index[0], edge_index[1]).long()
    hi = torch.maximum(edge_index[0], edge_index[1]).long()
    keys = lo * n_nodes + hi
    return torch.unique(keys, return_inverse=True)


def symmetrize_edge_scores(edge_index: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """Average the two stored directions of every user-item interaction."""
    unique_keys, inverse = _pair_index(edge_index)
    pair_scores = torch.zeros(unique_keys.numel(), device=scores.device, dtype=scores.dtype)
    pair_counts = torch.zeros_like(pair_scores)
    pair_scores.scatter_add_(0, inverse, scores)
    pair_counts.scatter_add_(0, inverse, torch.ones_like(scores))
    return (pair_scores / pair_counts.clamp_min(1.0))[inverse]


def paired_soft_environment_weights(
    edge_index: torch.Tensor,
    scores: torch.Tensor,
    num_env: int,
    variant_keep_prob: float = 0.5,
) -> torch.Tensor:
    """Create differentiable, pair-symmetric edge weights for environments.

    A high score keeps an edge at weight 1 in every environment.  A low score
    receives an independently sampled keep/drop mask, so invariant-risk and
    ranking gradients directly teach the scorer which edges must stay stable.
    """
    if not 0.0 <= variant_keep_prob <= 1.0:
        raise ValueError("variant_keep_prob must be in [0, 1]")
    unique_keys, inverse = _pair_index(edge_index)
    pair_scores = torch.zeros(unique_keys.numel(), device=scores.device, dtype=scores.dtype)
    pair_counts = torch.zeros_like(pair_scores)
    pair_scores.scatter_add_(0, inverse, scores)
    pair_counts.scatter_add_(0, inverse, torch.ones_like(scores))
    pair_scores = pair_scores / pair_counts.clamp_min(1.0)
    keep = (
        torch.rand(num_env, unique_keys.numel(), device=scores.device)
        < float(variant_keep_prob)
    ).to(dtype=scores.dtype)
    pair_weights = pair_scores.unsqueeze(0) + (1.0 - pair_scores.unsqueeze(0)) * keep
    return pair_weights[:, inverse]


def paired_random_environment_weights(
    edge_index: torch.Tensor, num_env: int, keep_prob: float = 0.5,
) -> torch.Tensor:
    """Random pair-symmetric environment masks, independent of edge scores."""
    if not 0.0 <= keep_prob <= 1.0:
        raise ValueError("keep_prob must be in [0, 1]")
    _, inverse = _pair_index(edge_index)
    n_pairs = int(inverse.max().item()) + 1
    keep = (torch.rand(num_env, n_pairs, device=edge_index.device) < keep_prob).float()
    return keep[:, inverse]
