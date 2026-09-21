import torch
import torch.nn as nn


def invariant_risk_loss(
    risks: torch.Tensor, alpha: float = 1.0, mean_weight: float = 0.0,
) -> torch.Tensor:
    """
    L_inv_core = mean_weight * mean(R_k) + alpha * Var(R_k)
    risks: [K]
    """
    mean_r = risks.mean()
    var_r = risks.var(unbiased=False) if risks.numel() > 1 else torch.zeros((), device=risks.device)
    return float(mean_weight) * mean_r + float(alpha) * var_r


def score_budget_loss(scores: torch.Tensor, rho: float) -> torch.Tensor:
    """Penalize mean score drifting away from target keep ratio rho."""
    return (scores.mean() - rho) ** 2


class InvariantLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, beta: float = 0.1, rho: float = 0.7,
                 scorer_weight: float = 1.0, risk_mean_weight: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.scorer_weight = scorer_weight
        self.risk_mean_weight = risk_mean_weight

    def forward(self, risks: torch.Tensor, scores: torch.Tensor,
                env_predictions: torch.Tensor = None) -> torch.Tensor:
        loss = invariant_risk_loss(risks, self.alpha, self.risk_mean_weight)
        loss = loss + self.beta * score_budget_loss(scores, self.rho)
        if env_predictions is not None:
            if env_predictions.ndim != 2:
                raise ValueError("env_predictions must have shape [num_env, num_edges]")
            instability = env_predictions.var(dim=0, unbiased=False)
            scale = instability.max().clamp_min(1e-8)
            stability_target = (1.0 - instability / scale).detach()
            loss = loss + self.scorer_weight * torch.nn.functional.mse_loss(
                scores, stability_target,
            )
        return loss
