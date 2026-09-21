import os
import random
import dgl
import time
import gc
import re
from pathlib import Path
from modules.DNN import DNN
from modules.VGAE import Model
from time import gmtime, strftime
from modules import diffusion as gd
from modules.causal_score import (
    CausalEdgeScorer, split_causal_variant_edges, symmetrize_edge_scores,
    paired_soft_environment_weights, paired_random_environment_weights,
)
from modules.invariant_loss import InvariantLoss

from parameters_lsci import args
from utils.evaulate import compute_vgae_loss, adjust_loss, compute_loss_para
from utils.input_data import UserItemDataset
from utils.preprocess import mask_test_edges_dgl
from torch.utils.data import DataLoader
from utils.util_loss import *
from modules.rec_model import LGCN_Encoder
from modules.environment_inference import *
from modules.condition_fusion import ConditionFusion
from modules.semantic_loss import SemanticInfoNCE
from utils.experiment_utils import (
    build_settings, parse_measure_block, parse_best_dict,
    count_params, save_run_record,
)
from utils.load_strict import load_strict_datasets, user_item_matrix_from_graph
from utils.semantic_prior import load_semantic_prior
from utils.upstream_ranking import scipy_to_torch_sparse, upstream_bpr_loss
from utils.ranking_head import (
    distribution_matched_random_embeddings,
    initialize_or_refresh_ranking_head,
)
from utils.late_fusion import build_user_semantic_profiles, late_fusion_scores

torch.autograd.set_detect_anomaly(True)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
PROTOCOL_VERSION = "corrected_v4_controlled"
RECORD_SUFFIX = "v4"


def setup_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_datasets(data_name, data_root="", eval_split="ood", load_test_gt=True):
    if data_root:
        datasets, meta = load_strict_datasets(
            data_name, data_root=data_root, eval_split=eval_split,
            load_test_gt=load_test_gt,
        )
        return datasets, meta

    dataset_paths = {
        "yelp2018": {
            "train": "./dataset/yelp2018/yelp_train_data.bin",
            "test": "./dataset/yelp2018/yelp_test_data.bin",
        },
        "douban": {
            "train": "./dataset/douban/douban_train_data.bin",
            "test": "./dataset/douban/douban_test_data.bin",
        },
        "food": {
            "train": "./dataset/food/food_train_data.bin",
            "test": "./dataset/food/food_test_data.bin",
        },
        "kuairec": {
            "train": "./dataset/kuairec/kuairec_train_data.bin",
            "test": "./dataset/kuairec/kuairec_test_data.bin",
        },
    }
    if data_name not in dataset_paths:
        raise ValueError(f"Unknown dataset: {data_name}")
    selected_paths = dataset_paths[data_name]
    graphs = {key: dgl.load_graphs(path)[0][0] for key, path in selected_paths.items()}
    graphs["strict"] = False
    return graphs, {"data_version": "legacy_bin", "data_status": "legacy_bin", "eval_split": "legacy_test"}


def prepare_data(graph, device):
    feats = graph.ndata['feat'].to(device)
    edge_index = torch.stack(graph.edges()).to(device)
    return feats, None, edge_index, graph.number_of_nodes()


def initialize_models(num_nodes, device, in_dim, mlp_in_dims, mlp_out_dims):
    vgae_model = Model(in_dim, args.hidden1, args.hidden2, device, num_nodes).to(device)
    diffusion_model = gd.GaussianDiffusion(
        gd.ModelMeanType.START_X, args.noise_schedule, args.noise_scale,
        args.noise_min, args.noise_max, args.steps, device
    ).to(device)
    mlp_model = DNN(
        mlp_in_dims, mlp_out_dims, args.emb_size, env_size=16,
        time_type="cat", norm=args.norm, act_func=args.mlp_act_func
    ).to(device)
    env_infer_model = EVAE(args.hidden2, args.hidden2).to(device)
    edge_scorer = CausalEdgeScorer(emb_dim=args.hidden2, hidden=64).to(device)
    inv_loss_fn = InvariantLoss(
        alpha=args.inv_alpha, beta=args.inv_beta, rho=args.causal_keep_ratio,
        risk_mean_weight=args.inv_mean_weight,
    )
    return vgae_model, diffusion_model, mlp_model, env_infer_model, edge_scorer, inv_loss_fn


def setup_optimizers(models):
    lr, lr2, wd2 = args.learning_rate, args.lr2, args.wd2
    opts = [
        torch.optim.Adam(models[0].parameters(), lr=lr),
        torch.optim.Adagrad(models[2].parameters(), lr=lr2, weight_decay=wd2),
        torch.optim.Adagrad(models[3].parameters(), lr=lr),
        torch.optim.Adam(models[4].parameters(), lr=float(args.score_lr)),
    ]
    # optional E3 fusion module at index 6
    if len(models) > 6 and models[6] is not None:
        opts.append(torch.optim.Adam(models[6].parameters(), lr=args.fusion_lr))
    return opts


def build_semantic_condition(env_embeddings, num_user, sem_ctx):
    """Build the diffusion condition; only item rows receive semantic context."""
    if sem_ctx is None:
        return env_embeddings
    fusion, _, z_s_items, mask_items = sem_ctx
    n_item = z_s_items.shape[0]
    if num_user + n_item > env_embeddings.shape[0]:
        raise ValueError("semantic prior contains more items than the graph")
    z_s = torch.zeros(
        env_embeddings.size(0), z_s_items.size(1),
        device=env_embeddings.device, dtype=env_embeddings.dtype,
    )
    z_s[num_user:num_user + n_item] = z_s_items
    mask = torch.zeros(env_embeddings.size(0), dtype=torch.bool, device=env_embeddings.device)
    mask[num_user:num_user + n_item] = mask_items
    return fusion(env_embeddings, z_s, mask)


def semantic_alignment_loss(pred_embeddings, num_user, sem_ctx, nce_sample: int = 512):
    """Align predicted item x0 with available item semantic priors."""
    if sem_ctx is None:
        return pred_embeddings.new_zeros(())
    fusion, sem_nce, z_s_items, mask_items = sem_ctx
    item_c = pred_embeddings[num_user:num_user + z_s_items.shape[0]]
    idx = mask_items.nonzero(as_tuple=False).view(-1)
    if idx.numel() < 2:
        return pred_embeddings.new_zeros(())
    if idx.numel() > nce_sample:
        idx = idx[torch.randperm(idx.numel(), device=idx.device)[:nce_sample]]
    return sem_nce(fusion.proj_c(item_c[idx]), fusion.proj_s(z_s_items[idx]), None)


def get_sem_ctx(models, datasets, device):
    if not args.use_semantic_prior or "semantic_prior" not in datasets:
        return None
    fusion = models[6] if len(models) > 6 else None
    sem_nce = models[7] if len(models) > 7 else None
    if fusion is None:
        return None
    bundle = datasets["semantic_prior"]
    z_s = bundle["item_emb"].to(device)
    mask = bundle["semantic_mask"].to(device)
    return fusion, sem_nce, z_s, mask


def generate_embeddings(vgae_model, diffusion_model, mlp_model, env_infer, features, edge_index,
                        num_samples, device, num_user, sem_ctx=None, edge_weight=None):
    features = features.to(device)
    edge_index = edge_index.to(device)
    embeddings_list = []
    active_modules = [vgae_model, mlp_model, env_infer]
    if sem_ctx is not None:
        active_modules.append(sem_ctx[0])
    training_states = [module.training for module in active_modules]
    for module in active_modules:
        module.eval()
    try:
        with torch.no_grad():
            for _ in range(num_samples):
                z = vgae_model.encoder(
                    features, edge_index, edge_weight=edge_weight,
                    sample=bool(args.sampling_noise),
                )
                env_embeddings = env_infer.decode(z)
                condition = build_semantic_condition(env_embeddings, num_user, sem_ctx)
                diffused_z = diffusion_model.p_sample(
                    mlp_model, z, condition, args.sampling_steps, args.sampling_noise
                )
                embeddings_list.append(diffused_z)
    finally:
        for module, was_training in zip(active_modules, training_states):
            module.train(was_training)
    return torch.mean(torch.stack(embeddings_list), dim=0)


def clip_optimizer_gradients(optimizer, max_norm):
    params = [
        p for group in optimizer.param_groups for p in group["params"]
        if p.grad is not None
    ]
    if params:
        torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm, error_if_nonfinite=True)


def require_finite(name, value, epoch):
    if not torch.isfinite(value).all():
        raise FloatingPointError(
            f"Non-finite {name} at epoch {epoch + 1}: {value.detach().cpu()}"
        )


def handle_gradient_step(optimizers, outer_loss, epoch):
    for opt in optimizers:
        opt.zero_grad()
    print(outer_loss)
    require_finite("outer_loss", outer_loss, epoch)
    outer_loss.backward()
    outer_indices = list(range(len(optimizers)))
    edge_scorer_grad_norm = 0.0
    edge_scorer_grads = [
        p.grad.detach().norm() for group in optimizers[3].param_groups
        for p in group["params"] if p.grad is not None
    ]
    if edge_scorer_grads:
        edge_scorer_grad_norm = float(torch.stack(edge_scorer_grads).norm().item())
    for idx in outer_indices:
        clip_optimizer_gradients(optimizers[idx], args.grad_clip)
        optimizers[idx].step()
    return edge_scorer_grad_norm


def safe_tag(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-") or "run"


def run_epoch(models, optimizers, feats, edge_index, adj, norm, weight_tensor, device, epoch,
              user_item_train_inter, num_user, num_item, train_graph,
              causal_edge_index, variant_edge_index, split_ready, datasets=None,
              rank_norm_adj=None, rec_model=None, rec_optimizer=None):
    vgae_model, diffusion_model, mlp_model, env_infer_model, edge_scorer, inv_loss_fn = models[:6]
    mlp_model.train()
    edge_scorer.train()
    sem_ctx = get_sem_ctx(models, datasets or {}, device)
    if sem_ctx is not None:
        sem_ctx[0].train()
    pretrain_loss, total_rec_loss = 0.0, 0.0
    upstream_rank_losses = []
    upstream_rank_grad_diagnostics = None
    num_nodes = feats.shape[0]

    rec_dataset = UserItemDataset(user_item_train_inter)
    rec_dataloader = DataLoader(
        rec_dataset, batch_size=int(args.rec_batch_size), shuffle=True,
    )
    rank_batches = []
    if float(args.lambda_rank_upstream) > 0:
        if rank_norm_adj is None:
            raise ValueError("rank_norm_adj is required when upstream ranking is enabled")
        rank_loader = DataLoader(
            rec_dataset, batch_size=int(args.rank_batch_size), shuffle=True,
        )
        rank_iter = iter(rank_loader)
        for _ in range(max(1, int(args.rank_batches_per_epoch))):
            try:
                batch = next(rank_iter)
            except StopIteration:
                break
            rank_batches.append([x.to(device) for x in batch])

    with torch.no_grad():
        base_latent = vgae_model.encoder(feats, edge_index, sample=False)

    raw_scores = edge_scorer(base_latent.detach(), edge_index)
    scores = symmetrize_edge_scores(edge_index, raw_scores)
    if args.edge_gate_mode == 'soft_pair_environment':
        if epoch < int(args.score_warmup_epochs):
            # Bootstrap a real stability target before trusting the scorer.
            env_weights = paired_random_environment_weights(
                edge_index, args.num_env, args.variant_keep_prob,
            )
        else:
            env_weights = paired_soft_environment_weights(
                edge_index, scores, args.num_env, args.variant_keep_prob,
            )
    elif args.edge_gate_mode == 'random_pair_environment':
        env_weights = paired_random_environment_weights(
            edge_index, args.num_env, args.variant_keep_prob,
        )
    else:
        env_weights = torch.ones(
            args.num_env, edge_index.shape[1], device=edge_index.device,
            dtype=feats.dtype,
        )
    # Learned and random environment arms must use the same checkpoint window;
    # otherwise random can select an early checkpoint that learned is forbidden
    # to select during scorer warm-up.
    split_ready = (
        args.edge_gate_mode == 'none'
        or epoch >= int(args.score_warmup_epochs)
    )

    for m in range(1):
        Loss, env_risks, env_preds = [], [], []
        for k in range(args.num_env):
            gc.collect()
            batch_latent = vgae_model.encoder(
                feats, edge_index, edge_weight=env_weights[k],
            )
            recon, mu, log_std = env_infer_model(batch_latent)
            kl_div = env_infer_model.kl_divergence(mu, log_std)
            infer_loss = evae_loss(recon, batch_latent, kl_div)
            env_embeddings = env_infer_model.decode(batch_latent)
            condition = build_semantic_condition(env_embeddings, num_user, sem_ctx)
            terms = diffusion_model.training_losses(
                mlp_model, batch_latent, condition, args.reweight,
            )
            elbo = terms["loss"].mean()
            logits = vgae_model.decoder(terms["pred_xstart"])
            vgae_loss = compute_vgae_loss(logits, adj, norm, vgae_model, weight_tensor)
            loss = adjust_loss(elbo, vgae_loss, infer_loss, args.reweight)
            if rank_batches:
                rank_loss = torch.stack([
                    upstream_bpr_loss(
                        terms["pred_xstart"], rank_norm_adj,
                        batch[0], batch[1], batch[2], num_user,
                        layers=int(args.rank_layers), l2_weight=1e-3,
                    )
                    for batch in rank_batches
                ]).mean()
                require_finite("upstream_rank_loss", rank_loss, epoch)
                loss = loss + float(args.lambda_rank_upstream) * rank_loss
                upstream_rank_losses.append(rank_loss.reshape(()))
                if epoch == 0 and k == 0:
                    diagnostic_params = {
                        "vgae": next(vgae_model.parameters()),
                        "diffusion_mlp": next(mlp_model.parameters()),
                        "env_infer": env_infer_model.fc2.weight,
                    }
                    diagnostic_grads = torch.autograd.grad(
                        rank_loss, list(diagnostic_params.values()),
                        retain_graph=True, allow_unused=True,
                    )
                    upstream_rank_grad_diagnostics = {
                        name: (
                            float(grad.detach().norm().item())
                            if grad is not None else 0.0
                        )
                        for name, grad in zip(diagnostic_params, diagnostic_grads)
                    }
            if sem_ctx is not None and float(args.lambda_sem) > 0:
                loss = loss + float(args.lambda_sem) * semantic_alignment_loss(
                    terms["pred_xstart"], num_user, sem_ctx,
                )
            for component_name, component in (
                ("elbo", elbo), ("vgae_loss", vgae_loss),
                ("infer_loss", infer_loss), ("environment_loss", loss),
            ):
                require_finite(component_name, component, epoch)
            Loss.append(loss.view(-1))
            env_risks.append(loss.reshape(()))
            edge_pred = (
                terms["pred_xstart"][edge_index[0]]
                * terms["pred_xstart"][edge_index[1]]
            ).sum(dim=-1)
            env_preds.append(torch.sigmoid(edge_pred))

        env_edge_predictions = torch.stack(env_preds, dim=0)
        inv_loss = inv_loss_fn(
            torch.stack(env_risks), scores, env_edge_predictions,
        )

        Var, Mean = torch.var_mean(torch.cat(Loss, dim=0))
        if Var is None:
            Var = 0
        effective_lambda_inv = 0.0 if args.edge_gate_mode == 'none' else float(args.lambda_inv)
        outer_loss = Var + Mean * compute_beta(
            epoch, args.epochs, args.risk_weight_start, args.risk_weight_end,
        ) + effective_lambda_inv * inv_loss
        pretrain_loss += float(outer_loss.item())
        soft_gate_grad_norm = handle_gradient_step(optimizers, outer_loss, epoch)

    # A hard partition is retained only as an interpretable diagnostic and is
    # never fed back into the encoder or used to cut the computation graph.
    causal_edge_index, variant_edge_index, _ = split_causal_variant_edges(
        edge_index, scores.detach(), keep_ratio=args.causal_keep_ratio,
    )
    if split_ready and args.edge_gate_mode != "none":
        print(
            f"[LSCI] {args.edge_gate_mode} diagnostic: "
            f"causal={causal_edge_index.shape[1]}, variant={variant_edge_index.shape[1]}"
        )

    all_embeddings = generate_embeddings(
        models[0], models[1], models[2], models[3], feats, edge_index, 1, device,
        num_user, sem_ctx,
        edge_weight=(scores.detach() if args.edge_gate_mode == 'soft_pair_environment' else None),
    )
    user_embeddings = all_embeddings[:num_user]
    item_embeddings = all_embeddings[num_user:]
    if args.rec_refresh_source == 'matched_random':
        user_embeddings, item_embeddings = distribution_matched_random_embeddings(
            user_embeddings, item_embeddings,
            seed=int(args.seed) * 1000003 + int(epoch),
        )
    ui_adj = generate_interaction_matrix_from_dgl(train_graph, num_user, num_item)
    norm_adj = normalize_graph_mat(ui_adj)
    # Keep the ranking head and its optimiser across outer epochs.  Recreating
    # them here made every epoch a one-pass BPR warm-up and discarded all
    # previous ranking learning.
    rec_model, rec_optimizer = initialize_or_refresh_ranking_head(
        num_user, norm_adj, user_embeddings, item_embeddings, device,
        rec_model=rec_model, rec_optimizer=rec_optimizer,
        refresh=args.rec_refresh, learning_rate=args.rec_lr, layers=3,
    )
    for _ in range(int(args.rec_epochs_per_outer)):
        for batch in rec_dataloader:
            user_id, pos_item_id, neg_item_id = [x.to(device) for x in batch]
            emb = rec_model().to(device)
            user_embedding = emb[user_id]
            pos_item_embedding = emb[(pos_item_id.long() + num_user)]
            neg_item_embedding = emb[(neg_item_id.long() + num_user)]
            rec_loss = bpr_loss(user_embedding, pos_item_embedding, neg_item_embedding) + \
                       l2_reg_loss(1e-3, user_embedding, pos_item_embedding, neg_item_embedding)
            rec_optimizer.zero_grad()
            rec_loss.backward()
            torch.nn.utils.clip_grad_norm_(rec_model.parameters(), 0.7)
            rec_optimizer.step()
            total_rec_loss += rec_loss.item()

    upstream_rank_loss_value = (
        float(torch.stack(upstream_rank_losses).mean().detach().item())
        if upstream_rank_losses else None
    )
    return (pretrain_loss, total_rec_loss, causal_edge_index, variant_edge_index,
            split_ready, rec_model, upstream_rank_loss_value,
            upstream_rank_grad_diagnostics, rec_optimizer, soft_gate_grad_norm)


def evaluate_epoch(datasets, split, trained_rec_model, device, num_user, num_item,
                   train_interactions, semantic_rank_ctx=None):
    """Evaluate a trained recommender state on validation or test context."""
    if split not in ("val", "test"):
        raise ValueError(f"Unknown evaluation split: {split}")
    graph = datasets[split]
    ui_adj = generate_interaction_matrix_from_dgl(graph, num_user, num_item)
    norm_adj = normalize_graph_mat(ui_adj)
    user_init = trained_rec_model.embedding_dict["user_emb"].detach()
    item_init = trained_rec_model.embedding_dict["item_emb"].detach()
    rec_model = LGCN_Encoder(num_user, 3, norm_adj, user_init, item_init).to(device)
    rec_model.load_state_dict(trained_rec_model.state_dict())
    rec_model.eval()
    with torch.no_grad():
        emb = rec_model().to(device)
        user_embeddings = emb[:num_user]
        item_embeddings = emb[num_user:]
        scores = torch.cat([
            torch.matmul(user_embeddings[i:i + 256], item_embeddings.t())
            for i in range(0, len(user_embeddings), 256)
        ], dim=0)
        if semantic_rank_ctx is not None:
            user_profiles, item_semantic, alpha = semantic_rank_ctx
            semantic_scores = torch.cat([
                torch.matmul(user_profiles[i:i + 256], item_semantic.t())
                for i in range(0, len(user_profiles), 256)
            ], dim=0)
            scores = late_fusion_scores(scores, semantic_scores, alpha)
    origin_inter = datasets[f"{split}_origin_inter"]
    user_set = datasets[f"{split}_user_set"]
    scores = mask_seen_items(scores, train_interactions, user_set)
    rec_dict = get_rec_list(user_set, scores, num_user, topk=20)
    measure = ranking_evaluation(origin_inter, rec_dict, [10, 20])
    return measure


def semantic_gate_diagnostics(models, datasets, feats, edge_index, num_user, device):
    """Measure how much semantic context the best checkpoint actually uses."""
    sem_ctx = get_sem_ctx(models, datasets, device)
    if sem_ctx is None:
        return None
    fusion, _, z_s_items, semantic_mask = sem_ctx
    fusion.eval()
    with torch.no_grad():
        # encoder() populates the posterior mean; use it rather than its random sample.
        models[0].encoder(feats, edge_index)
        env_embeddings = models[3].decode(models[0].mean)
        item_env = env_embeddings[num_user:num_user + z_s_items.shape[0]]
        valid = semantic_mask.bool()
        if not valid.any():
            return {"covered_items": 0}
        gate = fusion.gate_values(item_env[valid], z_s_items[valid]).view(-1)
        fused = fusion(item_env[valid], z_s_items[valid])
        causal_projected = fusion.proj_c(item_env[valid])
        shift_ratio = (
            (fused - causal_projected).norm(dim=1)
            / causal_projected.norm(dim=1).clamp_min(1e-8)
        )
    return {
        "covered_items": int(valid.sum().item()),
        "causal_gate_mean": float(gate.mean().item()),
        "causal_gate_std": float(gate.std(unbiased=False).item()),
        "causal_gate_min": float(gate.min().item()),
        "causal_gate_max": float(gate.max().item()),
        "semantic_weight_mean": float((1.0 - gate).mean().item()),
        "condition_shift_ratio_mean": float(shift_ratio.mean().item()),
        "condition_shift_ratio_std": float(shift_ratio.std(unbiased=False).item()),
    }


def train_model(models, optimizers, device, datasets, user_item_train_inter, num_user, num_item, run_meta=None):
    if not datasets.get("strict"):
        raise ValueError(
            f"{PROTOCOL_VERSION} training requires strict data with a separate val.parquet; "
            "legacy train/test bins cannot support leakage-free epoch selection"
        )
    graph = datasets['train']
    train_edge_idx = mask_test_edges_dgl(graph)
    train_graph = dgl.edge_subgraph(graph, train_edge_idx, relabel_nodes=False).to(device)
    adj = train_graph.adjacency_matrix().to_dense().to(device)
    weight_tensor, norm = compute_loss_para(adj, device)
    feats, _, edge_index, _ = prepare_data(graph, device)

    bestPerformance, val_measure_result, epoch_logs = [], {}, []
    causal_edge_index, variant_edge_index, split_ready = edge_index, edge_index[:, :0], False
    rec_model, rec_optimizer = None, None
    semantic_rank_ctx = None
    if args.use_semantic_prior and float(args.semantic_score_alpha) != 0.0:
        bundle = datasets["semantic_prior"]
        item_semantic = torch.nn.functional.normalize(
            bundle["item_emb"].float().to(device), dim=1,
        )
        user_profiles = build_user_semantic_profiles(
            item_semantic, user_item_train_inter,
        )
        semantic_rank_ctx = (
            user_profiles, item_semantic, float(args.semantic_score_alpha),
        )
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("experiments/records", exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    force_g = float(getattr(args, "force_sem_gate", -1.0))
    force_on = args.use_semantic_prior and 0.0 <= force_g <= 1.0
    edge_gate_mode = args.edge_gate_mode
    if args.use_semantic_prior:
        if force_on:
            method = f"lsci_e4_forcegate{force_g:g}_lamsem{float(args.lambda_sem):g}"
        elif float(getattr(args, "lambda_sem", 0.1)) == 0.0:
            method = "lsci_e4_lamsem0"
        else:
            method = "lsci_e3"
        if float(args.semantic_score_alpha) != 0.0:
            method += f"_scorefusion_a{float(args.semantic_score_alpha):g}"
    else:
        mode_tag = {
            "soft_pair_environment": "softgate",
            "random_pair_environment": "randomgate",
            "none": "baseline",
        }[args.edge_gate_mode]
        method = f"lsci_stage1_{mode_tag}"
        if float(args.lambda_rank_upstream) > 0:
            method += f"_end2endrank_lam{float(args.lambda_rank_upstream):g}"
    data_version = (run_meta or {}).get("data_version", "strict")
    eval_split = (run_meta or {}).get("eval_split", getattr(args, "eval_split", "ood"))
    run_tag = safe_tag(getattr(args, "run_id", "") or method)
    checkpoint_path = (
        f"checkpoints/{safe_tag(args.dataset)}_{safe_tag(method)}_"
        f"{safe_tag(data_version)}_{safe_tag(eval_split)}_{run_tag}_"
        f"seed{args.seed}_{RECORD_SUFFIX}.pt"
    )
    train_ui_adj = generate_interaction_matrix_from_dgl(
        train_graph, num_user, num_item,
    )
    rank_norm_adj = scipy_to_torch_sparse(
        normalize_graph_mat(train_ui_adj), device=device,
    )
    for epoch in range(args.epochs):
        (total_loss, rec_loss, causal_edge_index, variant_edge_index,
         split_ready, rec_model, upstream_rank_loss_value,
         upstream_rank_grad_diagnostics, rec_optimizer, soft_gate_grad_norm) = run_epoch(
            models, optimizers, feats, edge_index, adj, norm, weight_tensor, device, epoch,
            user_item_train_inter, num_user, num_item, train_graph,
            causal_edge_index, variant_edge_index, split_ready, datasets=datasets,
            rank_norm_adj=rank_norm_adj, rec_model=rec_model, rec_optimizer=rec_optimizer,
        )
        print(f"[LSCI] Epoch {epoch + 1}/{args.epochs}, Loss: {total_loss}, Rec_loss: {rec_loss}")
        measure = evaluate_epoch(
            datasets, "val", rec_model, device, num_user, num_item,
            user_item_train_inter, semantic_rank_ctx=semantic_rank_ctx,
        )
        measure_index = measure.index('Top 20\n')
        selection_eligible = split_ready or bool(args.allow_pre_split_selection)
        if selection_eligible:
            best_epoch = fast_evaluation(
                epoch, measure[measure_index:], bestPerformance, select_metric="NDCG",
            )
        else:
            best_epoch = None
            print("[LSCI] Checkpoint selection deferred until causal split is ready")
        val_measure_result[epoch] = measure
        epoch_logs.append({
            "epoch": epoch + 1,
            "pretrain_loss": float(total_loss),
            "rec_loss": float(rec_loss),
            "val_metrics": parse_measure_block(measure),
            "is_best": bool(selection_eligible and bestPerformance and bestPerformance[0] == epoch + 1),
            "selection_eligible": bool(selection_eligible),
            "causal_edges": int(causal_edge_index.shape[1]),
            "variant_edges": int(variant_edge_index.shape[1]),
            "upstream_rank_loss": upstream_rank_loss_value,
            "upstream_rank_grad_diagnostics": upstream_rank_grad_diagnostics,
            "soft_gate_grad_norm": soft_gate_grad_norm,
        })
        if selection_eligible and bestPerformance and bestPerformance[0] == epoch + 1:
            ckpt = {
                "epoch": epoch + 1, "dataset": args.dataset, "seed": args.seed,
                "method": method, "protocol_version": PROTOCOL_VERSION,
                "selection_split": "val", "selection_metric": "NDCG@20",
                "best_val": bestPerformance[1],
                "vgae": models[0].state_dict(), "mlp": models[2].state_dict(),
                "env_infer": models[3].state_dict(),
                "edge_scorer": models[4].state_dict(),
                "rec_model": rec_model.state_dict(),
            }
            if len(models) > 6 and models[6] is not None:
                ckpt["fusion"] = models[6].state_dict()
            torch.save(ckpt, checkpoint_path)
            print(f"Saved best validation checkpoint to {checkpoint_path}")

    if not bestPerformance:
        raise RuntimeError(
            "No eligible checkpoint; run beyond score warmup or use "
            "--allow_pre_split_selection for an ablation."
        )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    models[0].load_state_dict(checkpoint["vgae"])
    models[2].load_state_dict(checkpoint["mlp"])
    models[3].load_state_dict(checkpoint["env_infer"])
    models[4].load_state_dict(checkpoint["edge_scorer"])
    if len(models) > 6 and models[6] is not None and "fusion" in checkpoint:
        models[6].load_state_dict(checkpoint["fusion"])
    rec_model.load_state_dict(checkpoint["rec_model"])
    best_epoch = int(checkpoint["epoch"])
    best_val_metrics = parse_measure_block(val_measure_result[best_epoch - 1])
    gate_diagnostics = semantic_gate_diagnostics(
        models, datasets, feats, edge_index, num_user, device,
    )
    print('Best validation result of %s:\n%s' % (
        'lsci', ''.join(val_measure_result[best_epoch - 1]),
    ))
    if args.validation_only:
        test_measure = None
        best_metrics = None
        print('[protocol] validation_only=True: test GT was not evaluated')
    else:
        test_measure = evaluate_epoch(
            datasets, "test", rec_model, device, num_user, num_item,
            user_item_train_inter, semantic_rank_ctx=semantic_rank_ctx,
        )
        best_metrics = parse_measure_block(test_measure)
        print('One-shot test result of %s:\n%s' % ('lsci', ''.join(test_measure)))
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if torch.cuda.is_available() else None
    record = {
        "settings": build_settings(args, extra={**(run_meta or {}), "method": method,
                                                "protocol_version": PROTOCOL_VERSION,
                                                "stability": {
                                                    "evae_reduction": "mean",
                                                    "log_std_range": [-10.0, 5.0],
                                                    "grad_clip": float(args.grad_clip),
                                                    "fusion_lr": float(args.fusion_lr),
                                                },
                                                "selection_split": "val",
                                                "selection_metric": "NDCG@20",
                                                "validation_only": bool(args.validation_only),
                                                "causal_keep_ratio": args.causal_keep_ratio,
                                                "edge_gate_mode": edge_gate_mode,
                                                "variant_keep_prob": float(args.variant_keep_prob),
                                                "lambda_inv": args.lambda_inv,
                                                "effective_lambda_inv": (
                                                    0.0 if edge_gate_mode == "none" else float(args.lambda_inv)
                                                ),
                                                "inv_mean_weight": float(args.inv_mean_weight),
                                                "lambda_sem": getattr(args, "lambda_sem", 0.0),
                                                "lambda_rank_upstream": float(args.lambda_rank_upstream),
                                                "rank_batch_size": int(args.rank_batch_size),
                                                "rank_batches_per_epoch": int(args.rank_batches_per_epoch),
                                                "rank_layers": int(args.rank_layers),
                                                "rec_refresh": float(args.rec_refresh),
                                                "rec_refresh_source": str(args.rec_refresh_source),
                                                "rec_lr": float(args.rec_lr),
                                                "rec_batch_size": int(args.rec_batch_size),
                                                "rec_epochs_per_outer": int(args.rec_epochs_per_outer),
                                                "ranking_head_lifecycle": (
                                                    "persistent_initialization_only"
                                                    if float(args.rec_refresh) == 0.0 else
                                                    f"persistent_with_{args.rec_refresh_source}_refresh"
                                                ),
                                                "score_lr": float(args.score_lr),
                                                "use_semantic_prior": bool(args.use_semantic_prior),
                                                "semantic_score_alpha": float(args.semantic_score_alpha),
                                                "semantic_ranking_path": (
                                                    "train_profile_score_late_fusion"
                                                    if semantic_rank_ctx is not None else "disabled"
                                                ),
                                                "data_root": getattr(args, "data_root", "") or None,
                                                "eval_split": getattr(args, "eval_split", "ood")}),
        "best_epoch": int(best_epoch),
        "best_val_metrics": best_val_metrics,
        "best_metrics": best_metrics,
        "semantic_gate_diagnostics": gate_diagnostics,
        "epoch_logs": epoch_logs,
        "peak_vram_mb": peak_vram_mb,
        "checkpoint": checkpoint_path,
    }
    result_log = (
        f"logs/{safe_tag(args.dataset)}_{run_tag}_seed{args.seed}_"
        f"{RECORD_SUFFIX}_final_result.txt"
    )
    with open(result_log, "w") as f:
        f.write('Best validation result of %s:\n%s\n' % (
            'lsci', ''.join(val_measure_result[best_epoch - 1]),
        ))
        if test_measure is None:
            f.write('TEST_NOT_EVALUATED: validation_only=True\n')
        else:
            f.write('One-shot test result of %s:\n%s\n' % ('lsci', ''.join(test_measure)))
    return record


def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    print("Start LSCI-DiffRec Stage I training!")
    start_time = time.time()
    seed_it(args.seed)
    device = setup_device()
    data_root = (args.data_root or "").strip()
    print(f"Using device: {device}, seed={args.seed}, dataset={args.dataset}, data_root={data_root or 'legacy'}")

    datasets, data_meta = load_datasets(
        args.dataset, data_root=data_root, eval_split=args.eval_split,
        load_test_gt=not args.validation_only,
    )
    train_graph = datasets['train']
    feats, _, edge_index, num_nodes = prepare_data(train_graph, device)
    indim = feats.shape[-1]
    latent_size = args.emd_size
    mlp_out_dims = eval(args.mlp_dims) + [latent_size]
    mlp_in_dims = mlp_out_dims[::-1]
    models = initialize_models(num_nodes, device, indim, mlp_in_dims, mlp_out_dims)
    # E2 prior load + E3 fusion modules
    semantic_bundle = None
    fusion = None
    sem_nce = None
    if args.use_semantic_prior:
        prior_path = args.semantic_prior_path
        if not prior_path and data_root:
            prior_path = str(Path(data_root) / "features" / "semantic_prior.pt")
        if not prior_path:
            raise ValueError("--use_semantic_prior requires --semantic_prior_path or --data_root")
        semantic_bundle = load_semantic_prior(prior_path, map_location="cpu")
        print(
            f"[E2] loaded semantic_prior {prior_path} "
            f"shape={tuple(semantic_bundle['item_emb'].shape)} "
            f"coverage={float(semantic_bundle['semantic_mask'].float().mean()):.4f}"
        )
        # num_items known after matrix build below — validate later
        datasets["semantic_prior"] = semantic_bundle

    if datasets.get("strict"):
        n_user, n_item = int(data_meta["n_user"]), int(data_meta["n_item"])
        user_item_train_inter, num_users, num_items = user_item_matrix_from_graph(train_graph, n_user, n_item)
    else:
        user_item_train_inter, num_users, num_items = get_user_item_matrix(train_graph)

    if semantic_bundle is not None:
        if semantic_bundle["item_emb"].shape[0] != num_items:
            raise ValueError(
                f"semantic_prior n_item={semantic_bundle['item_emb'].shape[0]} != num_items={num_items}"
            )
        sem_dim = int(semantic_bundle["item_emb"].shape[1])
        causal_dim = int(args.hidden2)
        force_g = float(getattr(args, "force_sem_gate", -1.0))
        force_g = force_g if 0.0 <= force_g <= 1.0 else None
        fusion = ConditionFusion(
            causal_dim, sem_dim, out_dim=causal_dim, force_gate=force_g,
        ).to(device)
        sem_nce = SemanticInfoNCE(temperature=0.07).to(device)
        print(
            f"[E3] ConditionFusion causal_dim={causal_dim} sem_dim={sem_dim} "
            f"lambda_sem={args.lambda_sem} force_sem_gate={force_g}"
        )

    # pack models: 0..5 base, 6 fusion, 7 sem_nce
    models = tuple(list(models) + [fusion, sem_nce])
    optimizers = setup_optimizers(models)

    run_meta = {
        "num_nodes": int(num_nodes), "num_users": int(num_users), "num_items": int(num_items),
        "feat_dim": int(indim), "train_edges": int(train_graph.number_of_edges()),
        "test_edges": int(datasets['test'].number_of_edges()),
        "param_count": count_params(
            models[:5], names=["vgae", "diffusion", "mlp", "env_infer", "edge_scorer"],
        ), "device": str(device),
        "use_semantic_prior": bool(args.use_semantic_prior),
        "lambda_sem": float(args.lambda_sem) if args.use_semantic_prior else 0.0,
        "force_sem_gate": (
            float(args.force_sem_gate)
            if args.use_semantic_prior and 0.0 <= float(getattr(args, "force_sem_gate", -1.0)) <= 1.0
            else None
        ),
        **{k: data_meta[k] for k in (
            "data_version", "data_status", "eval_split", "data_root",
            "val_gt_users", "val_gt_interactions", "gt_users", "gt_interactions",
        )
           if k in data_meta},
    }
    if semantic_bundle is not None:
        run_meta["semantic_prior_meta"] = semantic_bundle.get("meta", {})
        run_meta["semantic_coverage"] = float(semantic_bundle["semantic_mask"].float().mean())

    # tag run_id / checkpoint names for strict vs legacy
    prior_path_l = (getattr(args, "semantic_prior_path", "") or "").lower()
    is_shuffle_diag = "shuffl" in prior_path_l
    force_g = float(getattr(args, "force_sem_gate", -1.0))
    force_on = args.use_semantic_prior and 0.0 <= force_g <= 1.0
    if data_root:
        args.run_id = f"{args.run_id}_strict_{args.eval_split}"
    if args.use_semantic_prior:
        if is_shuffle_diag:
            args.run_id = f"{args.run_id}_e4_shuffle"
        elif force_on:
            args.run_id = (
                f"{args.run_id}_e4_forcegate{force_g:g}_lamsem{float(args.lambda_sem):g}"
            )
        elif float(getattr(args, "lambda_sem", 0.1)) == 0.0:
            args.run_id = f"{args.run_id}_e4_lamsem0"
        else:
            args.run_id = f"{args.run_id}_e3"
    record = train_model(models, optimizers, device, datasets, user_item_train_inter, num_users, num_items, run_meta)
    wall = time.time() - start_time
    record["wall_time_sec"] = float(wall)
    record["wall_time_hms"] = strftime("%H:%M:%S", gmtime(wall))
    tag = f"{args.dataset}_lsci"
    if data_root:
        tag = f"{args.dataset}_lsci_strict_{args.eval_split}"
    if args.use_semantic_prior:
        # Diagnostics / ablations: never overwrite full E3 records
        if is_shuffle_diag:
            tag = f"{tag}_e4_shuffle"
        elif force_on:
            # sanitize tag: 0.3 -> 03
            gtag = f"{force_g:.2f}".replace(".", "")
            ltag = f"{float(args.lambda_sem):g}".replace(".", "p")
            tag = f"{tag}_e4_forcegate{gtag}_lamsem{ltag}"
        elif float(getattr(args, "lambda_sem", 0.1)) == 0.0:
            tag = f"{tag}_e4_lamsem0"
        else:
            tag = f"{tag}_e3"
    path = (
        f"experiments/records/{tag}_{safe_tag(args.run_id)}_"
        f"seed{args.seed}_{RECORD_SUFFIX}.json"
    )
    save_run_record(path, record)
    print(f"total time cost：{record['wall_time_hms']}")
    print(f"Saved experiment record to {path}")
