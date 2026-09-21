import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from modules.condition_fusion import ConditionFusion
from modules.causal_score import (
    paired_soft_environment_weights, split_causal_variant_edges,
)
from modules.diffusion import GaussianDiffusion, ModelMeanType
from modules.environment_inference import EVAE, evae_loss
from modules.generator_lsci import Graph_Editer_LSCI
from modules.invariant_loss import InvariantLoss, invariant_risk_loss
from modules.rec_model import LGCN_Encoder
from utils.evaulate import adjust_loss
from utils.load_strict import load_strict_datasets
from utils.late_fusion import (
    adaptive_late_fusion_scores,
    build_user_semantic_profiles,
    late_fusion_scores,
)
from utils.util_loss import compute_beta, fast_evaluation, get_rec_list, mask_seen_items
from utils.upstream_ranking import (
    functional_lightgcn, scipy_to_torch_sparse, upstream_bpr_loss,
)
from utils.ranking_head import (
    distribution_matched_random_embeddings,
    initialize_or_refresh_ranking_head,
)


class RankingProtocolTests(unittest.TestCase):
    def test_adaptive_late_fusion_applies_frozen_row_weights(self):
        collaborative = torch.tensor([[0.1, 0.5, 0.2], [0.4, 0.2, 0.1]])
        semantic = torch.tensor([[0.9, 0.1, 0.2], [0.1, 0.2, 0.8]])
        fused = adaptive_late_fusion_scores(
            collaborative, semantic, torch.tensor([0.0, 0.5]),
        )
        baseline = late_fusion_scores(collaborative, semantic, 0.0)
        half = late_fusion_scores(collaborative, semantic, 0.5)
        self.assertTrue(torch.allclose(fused[0], baseline[0]))
        self.assertTrue(torch.allclose(fused[1], half[1]))

    def test_late_fusion_uses_train_profiles_and_preserves_alpha_zero_ranking(self):
        train = sp.csr_matrix(
            (np.ones(3), ([0, 0, 1], [0, 1, 2])), shape=(2, 3),
        )
        items = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        profiles = build_user_semantic_profiles(items, train)
        self.assertTrue(torch.allclose(
            profiles[0], torch.tensor([2 ** -0.5, 2 ** -0.5]), atol=1e-6,
        ))
        collaborative = torch.tensor([[0.2, 0.8, 0.1], [0.9, 0.1, 0.2]])
        semantic = profiles @ torch.nn.functional.normalize(items, dim=1).t()
        fused = late_fusion_scores(collaborative, semantic, alpha=0.0)
        self.assertTrue(torch.equal(
            collaborative.argsort(dim=1, descending=True),
            fused.argsort(dim=1, descending=True),
        ))

    def test_seen_items_are_masked_and_ids_are_absolute(self):
        scores = torch.tensor([
            [0.99, 0.80, 0.70, 0.60],
            [0.50, 0.98, 0.40, 0.30],
        ])
        train = sp.csr_matrix(
            (np.ones(2), ([0, 1], [0, 1])), shape=(2, 4),
        )
        masked = mask_seen_items(scores, train, {0, 1})
        self.assertTrue(torch.isneginf(masked[0, 0]))
        self.assertTrue(torch.isneginf(masked[1, 1]))
        recs = get_rec_list({0, 1}, masked, max_user_id=2, topk=2)
        self.assertEqual([item for item, _ in recs[0]], ["3", "4"])
        self.assertEqual([item for item, _ in recs[1]], ["2", "4"])

    def test_epoch_selection_uses_only_requested_metric(self):
        best = []
        first = [
            "Top 20\n", "Hit Ratio:0.9\n", "Precision:0.2\n",
            "Recall:0.3\n", "NDCG:0.4\n",
        ]
        second = [
            "Top 20\n", "Hit Ratio:0.1\n", "Precision:0.1\n",
            "Recall:0.1\n", "NDCG:0.5\n",
        ]
        fast_evaluation(0, first, best, select_metric="NDCG")
        selected = fast_evaluation(1, second, best, select_metric="NDCG")
        self.assertEqual(selected, 2)
        self.assertEqual(best[1]["NDCG"], 0.5)


class ObjectiveTests(unittest.TestCase):
    def test_invariant_mean_weight_is_honoured(self):
        risks = torch.tensor([1.0, 3.0])
        variance_only = invariant_risk_loss(risks, alpha=1.0, mean_weight=0.0)
        with_mean = invariant_risk_loss(risks, alpha=1.0, mean_weight=2.0)
        self.assertAlmostEqual(variance_only.item(), 1.0)
        self.assertAlmostEqual(with_mean.item(), 5.0)

    def test_risk_weight_schedule_reaches_declared_endpoints(self):
        self.assertAlmostEqual(compute_beta(0, 5, start=0.2, end=1.0), 0.2)
        self.assertAlmostEqual(compute_beta(2, 5, start=0.2, end=1.0), 0.6)
        self.assertAlmostEqual(compute_beta(4, 5, start=0.2, end=1.0), 1.0)
        self.assertEqual([compute_beta(i, 3) for i in range(3)], [1.0] * 3)

    def test_vgae_deterministic_encoder_uses_posterior_mean(self):
        from modules.VGAE import Model

        torch.manual_seed(13)
        model = Model(2, 3, 2, torch.device("cpu"), 3)
        features = torch.randn(3, 2)
        edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        first = model.encoder(features, edges, sample=False)
        second = model.encoder(features, edges, sample=False)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first, model.mean))

    def test_bidirectional_interactions_are_split_together(self):
        # (0, 3) is one interaction stored in both directions; its two raw
        # scores disagree, but the split must still retain/remove both edges.
        edges = torch.tensor([[0, 3, 1, 4], [3, 0, 4, 1]])
        _, _, mask = split_causal_variant_edges(
            edges, torch.tensor([0.9, 0.1, 0.4, 0.4]), keep_ratio=0.5,
        )
        self.assertEqual(bool(mask[0]), bool(mask[1]))
        self.assertEqual(bool(mask[2]), bool(mask[3]))

    def test_soft_environment_gate_is_pair_symmetric_and_differentiable(self):
        torch.manual_seed(11)
        edges = torch.tensor([[0, 3, 1, 4], [3, 0, 4, 1]])
        scores = torch.tensor([0.2, 0.8, 0.4, 0.6], requires_grad=True)
        weights = paired_soft_environment_weights(edges, scores, num_env=3)
        self.assertTrue(torch.allclose(weights[:, 0], weights[:, 1]))
        self.assertTrue(torch.allclose(weights[:, 2], weights[:, 3]))
        weights.sum().backward()
        self.assertIsNotNone(scores.grad)
        self.assertGreater(scores.grad.abs().sum().item(), 0.0)

    def test_edge_gate_changes_encoder_forward(self):
        from modules.VGAE import Model

        torch.manual_seed(17)
        features = torch.randn(4, 2)
        edges = torch.tensor([[0, 2, 1, 3], [2, 0, 3, 1]])
        model = Model(2, 3, 2, torch.device("cpu"), 4)
        full = model.encoder(features, edges, sample=False)
        gated = model.encoder(
            features, edges, edge_weight=torch.tensor([1.0, 1.0, 0.0, 0.0]),
            sample=False,
        )
        self.assertFalse(torch.allclose(full, gated))

    def test_bipartite_editor_never_generates_same_type_edges(self):
        torch.manual_seed(3)
        editor = Graph_Editer_LSCI(4, 6, torch.device("cpu"), rank=2)
        causal = torch.tensor([[0, 3], [3, 0]])
        variant = torch.tensor([[1, 4, 2, 5], [4, 1, 5, 2]])
        generated, _ = editor(6, 5, 0, causal, variant, num_user=3)
        src, dst = generated
        self.assertTrue(torch.all((src < 3) != (dst < 3)))

    def test_reduced_sampling_steps_do_not_run_full_horizon(self):
        class TraceModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.timesteps = []

            def forward(self, x, t, e):
                self.timesteps.append(int(t[0]))
                return x

        diffusion = GaussianDiffusion(
            ModelMeanType.START_X, "linear", 0.1, 0.01, 0.09, 4,
            torch.device("cpu"),
        )
        model = TraceModel()
        diffusion.p_sample(model, torch.zeros(2, 2), torch.zeros(2, 2), steps=2)
        self.assertEqual(model.timesteps, [1, 0])

    def test_upstream_ranking_path_preserves_generator_gradient(self):
        initial = torch.randn(4, 3, requires_grad=True)
        adjacency = scipy_to_torch_sparse(sp.eye(4, dtype=np.float32, format="csr"))
        propagated = functional_lightgcn(initial, adjacency, layers=2)
        self.assertTrue(propagated.requires_grad)
        loss = upstream_bpr_loss(
            initial, adjacency,
            torch.tensor([0]), torch.tensor([0]), torch.tensor([1]),
            n_user=2, layers=2,
        )
        loss.backward()
        self.assertIsNotNone(initial.grad)
        self.assertGreater(initial.grad.abs().sum().item(), 0.0)

    def test_adjust_loss_argument_order(self):
        elbo = torch.tensor(2.0)
        vgae = torch.tensor(3.0)
        infer = torch.tensor(5.0)
        self.assertAlmostEqual(adjust_loss(elbo, vgae, infer, True).item(), 0.55)
        self.assertAlmostEqual(adjust_loss(elbo, vgae, infer, False).item(), 2.3)

    def test_invariant_scorer_receives_stability_gradient(self):
        scores = torch.tensor([0.2, 0.8, 0.5], requires_grad=True)
        risks = torch.tensor([1.0, 1.2, 0.9], requires_grad=True)
        env_predictions = torch.tensor([
            [0.5, 0.1, 0.2],
            [0.5, 0.9, 0.3],
            [0.5, 0.2, 0.4],
        ])
        loss = InvariantLoss()(risks, scores, env_predictions)
        loss.backward()
        self.assertIsNotNone(scores.grad)
        self.assertGreater(scores.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(risks.grad)

    def test_evae_objective_is_finite_and_graph_size_invariant(self):
        model = EVAE(2, 2)
        with torch.no_grad():
            model.fc1_log_std.weight.fill_(100.0)
            model.fc1_log_std.bias.fill_(100.0)
        x = torch.ones(4, 2)
        recon, mu, log_std = model(x)
        loss = evae_loss(recon, x, model.kl_divergence(mu, log_std))
        self.assertTrue(torch.isfinite(loss))
        self.assertLessEqual(log_std.max().item(), 5.0)
        base = evae_loss(torch.zeros(2, 3), torch.ones(2, 3), torch.tensor(0.5))
        doubled = evae_loss(torch.zeros(4, 3), torch.ones(4, 3), torch.tensor(0.5))
        self.assertTrue(torch.allclose(base, doubled))

    def test_semantic_mask_leaves_user_condition_unchanged(self):
        torch.manual_seed(7)
        fusion = ConditionFusion(2, 3, out_dim=2, force_gate=0.5)
        causal = torch.randn(4, 2)
        semantic = torch.randn(4, 3)
        mask = torch.tensor([False, False, True, True])
        fused = fusion(causal, semantic, mask)
        self.assertTrue(torch.allclose(fused[:2], causal[:2]))
        self.assertFalse(torch.allclose(fused[2:], causal[2:]))

    def test_forced_gate_reports_expected_semantic_weight(self):
        fusion = ConditionFusion(2, 3, out_dim=2, force_gate=0.7)
        gates = fusion.gate_values(torch.randn(5, 2), torch.randn(5, 3))
        self.assertTrue(torch.allclose(gates, torch.full((5, 1), 0.7)))
        self.assertAlmostEqual((1.0 - gates).mean().item(), 0.3, places=6)


class PersistenceAndDataTests(unittest.TestCase):
    def test_distribution_matched_random_control_preserves_moments(self):
        user = torch.arange(18, dtype=torch.float32).reshape(6, 3)
        item = torch.arange(18, 36, dtype=torch.float32).reshape(6, 3)
        random_user, random_item = distribution_matched_random_embeddings(
            user, item, seed=91,
        )
        source = torch.cat([user, item])
        control = torch.cat([random_user, random_item])
        self.assertTrue(torch.allclose(source.mean(0), control.mean(0), atol=1e-6))
        self.assertTrue(torch.allclose(
            source.std(0, unbiased=False),
            control.std(0, unbiased=False), atol=1e-6,
        ))
        again = distribution_matched_random_embeddings(user, item, seed=91)
        self.assertTrue(torch.equal(random_user, again[0]))

    def test_ranking_head_and_optimizer_persist_across_refresh(self):
        norm_adj = sp.eye(4, dtype=np.float32, format="csr")
        first_user, first_item = torch.zeros(2, 3), torch.zeros(2, 3)
        model, optimizer = initialize_or_refresh_ranking_head(
            2, norm_adj, first_user, first_item, torch.device("cpu"),
            refresh=0.25,
        )
        model_id, optimizer_id = id(model), id(optimizer)
        second_user, second_item = torch.ones(2, 3), torch.ones(2, 3)
        model, optimizer = initialize_or_refresh_ranking_head(
            2, norm_adj, second_user, second_item, torch.device("cpu"),
            rec_model=model, rec_optimizer=optimizer, refresh=0.25,
        )
        self.assertEqual(id(model), model_id)
        self.assertEqual(id(optimizer), optimizer_id)
        self.assertTrue(torch.allclose(
            model.embedding_dict["user_emb"], torch.full((2, 3), 0.25),
        ))

    def test_trained_recommender_state_restores_exactly(self):
        norm_adj = sp.eye(4, dtype=np.float32, format="csr")
        model = LGCN_Encoder(
            2, 1, norm_adj, torch.randn(2, 3), torch.randn(2, 3),
        )
        with torch.no_grad():
            model.embedding_dict["user_emb"].add_(1.0)
        expected = model()
        restored = LGCN_Encoder(
            2, 1, norm_adj, torch.zeros(2, 3), torch.zeros(2, 3),
        )
        restored.load_state_dict(model.state_dict())
        self.assertTrue(torch.allclose(expected, restored()))

    def test_strict_loader_exposes_distinct_validation_and_test_gt(self):
        root = Path("data_strict/processed/food/v1_strict")
        datasets, meta = load_strict_datasets(
            "food", data_root=str(root), eval_split="ood",
        )
        self.assertIn("val_origin_inter", datasets)
        self.assertIn("test_origin_inter", datasets)
        self.assertNotEqual(meta["val_gt_path"], meta["gt_path"])
        self.assertTrue(meta["val_gt_path"].endswith("val.parquet"))
        self.assertTrue(meta["gt_path"].endswith("ood_test.parquet"))
        self.assertIsNot(datasets["val_origin_inter"], datasets["test_origin_inter"])

    def test_validation_only_loader_does_not_expose_test_gt(self):
        root = Path("data_strict/processed/food/v1_strict")
        datasets, meta = load_strict_datasets(
            "food", data_root=str(root), eval_split="ood", load_test_gt=False,
        )
        self.assertIn("val_origin_inter", datasets)
        self.assertNotIn("test_origin_inter", datasets)
        self.assertNotIn("origin_inter", datasets)
        self.assertFalse(meta["test_gt_loaded"])
        self.assertIsNone(meta["gt_users"])


if __name__ == "__main__":
    unittest.main()
