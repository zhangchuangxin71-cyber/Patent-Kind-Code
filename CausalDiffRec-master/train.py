import os
import random
import dgl
import time
import gc
import re
from modules.DNN import DNN
from modules.VGAE import Model
from time import gmtime, strftime
from modules import diffusion as gd
from modules.generator import Graph_Editer

from parameters import args
from utils.evaulate import compute_vgae_loss, adjust_loss, compute_loss_para
from utils.input_data import UserItemDataset
from utils.preprocess import mask_test_edges_dgl
from torch.utils.data import DataLoader
from utils.util_loss import *
from modules.rec_model import LGCN_Encoder
from modules.environment_inference import *
from utils.experiment_utils import (
    build_settings, count_params, parse_best_dict, parse_measure_block, save_run_record,
)
from utils.load_strict import load_strict_datasets, user_item_matrix_from_graph
from utils.ranking_head import initialize_or_refresh_ranking_head

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
    datasets = {key: dgl.load_graphs(path)[0][0] for key, path in selected_paths.items()}
    datasets["strict"] = False
    return datasets, {
        "data_version": "legacy_bin",
        "data_status": "legacy_bin",
        "eval_split": "legacy_test",
    }


def prepare_data(graph, device):
    num_nodes = graph.number_of_nodes()
    feats = graph.ndata['feat'].to(device)
    adj_orig = None
    edge_index = torch.stack(graph.edges()).to(device)
    return feats, adj_orig, edge_index, num_nodes


def initialize_models(num_nodes, device, in_dim, mlp_in_dims, mlp_out_dims):
    vgae_model = Model(in_dim, args.hidden1, args.hidden2, device, num_nodes).to(device)
    diffusion_model = gd.GaussianDiffusion(gd.ModelMeanType.START_X,
                                           args.noise_schedule, args.noise_scale, args.noise_min,
                                           args.noise_max, args.steps, device).to(device)
    mlp_model = DNN(mlp_in_dims, mlp_out_dims, args.emb_size, env_size=16, time_type="cat", norm=args.norm,
                    act_func=args.mlp_act_func).to(device)
    # 使用CPU优化版图编辑器，低秩分解减少内存占用
    generator = Graph_Editer(4, num_nodes, device, rank=128).to(device)
    env_infer_model = EVAE(args.hidden2, args.hidden2).to(device)

    # mlp_num = sum([param.neltopement() for param in mlp_model.parameters()])
    # diff_num = sum([param.nelement() for param in diffusion_model.parameters()])
    # vgae_num = sum([p.nelement() for p in vgae_model.parameters()])
    # env_infer_num = sum([p.nelement() for p in env_infer_model.parameters()])

    # params = mlp_num + diff_num + vgae_num + env_infer_num
    # print('Total Parameters:', params)
    return vgae_model, diffusion_model, mlp_model, generator, env_infer_model


def setup_optimizers(models):
    lr = args.learning_rate
    lr2 = args.lr2
    wd2 = args.wd2
    optimizers = [
        torch.optim.Adam(models[0].parameters(), lr=lr),
        torch.optim.Adagrad(models[2].parameters(), lr=lr2, weight_decay=wd2),
        torch.optim.Adagrad(models[3].parameters(), lr=lr),
        torch.optim.Adagrad(models[4].parameters(), lr=lr)
    ]
    return optimizers

def safe_tag(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-") or "run"


def train_model(model, optimizers, device, datasets, user_item_train_inter, num_user, num_item,
                run_meta=None):
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
    bestPerformance = []
    val_measure_result = {}
    epoch_logs = []
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("experiments/records", exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    data_version = (run_meta or {}).get("data_version", "strict")
    eval_split = (run_meta or {}).get("eval_split", getattr(args, "eval_split", "ood"))
    run_tag = safe_tag(getattr(args, "run_id", "") or "causaldiffrec_e0")
    checkpoint_path = (
        f"checkpoints/{safe_tag(args.dataset)}_causaldiffrec_e0_"
        f"{safe_tag(data_version)}_{safe_tag(eval_split)}_{run_tag}_"
        f"seed{args.seed}_{RECORD_SUFFIX}.pt"
    )
    rec_model, rec_optimizer = None, None

    for epoch in range(args.epochs):
        total_loss, rec_loss, rec_model, rec_optimizer = run_epoch(
            model, optimizers, feats, edge_index, adj, norm, weight_tensor,
            device, epoch, user_item_train_inter, num_user, num_item, train_graph,
            rec_model=rec_model, rec_optimizer=rec_optimizer,
        )
        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {total_loss}, Rec_loss: {rec_loss}")

        measure = evaluate_epoch(
            datasets, "val", rec_model, device, num_user, num_item,
            user_item_train_inter,
        )
        measure_index = measure.index('Top 20\n')
        best_epoch = fast_evaluation(
            epoch, measure[measure_index:], bestPerformance, select_metric="NDCG",
        )
        val_measure_result[epoch] = measure
        parsed = parse_measure_block(measure)
        epoch_logs.append({
            "epoch": epoch + 1,
            "pretrain_loss": float(total_loss),
            "rec_loss": float(rec_loss),
            "val_metrics": parsed,
            "is_best": bool(bestPerformance and bestPerformance[0] == epoch + 1),
        })
        if bestPerformance and bestPerformance[0] == epoch + 1:
            torch.save({
                "epoch": epoch + 1,
                "dataset": args.dataset,
                "seed": args.seed,
                "protocol_version": PROTOCOL_VERSION,
                "selection_split": "val",
                "selection_metric": "NDCG@20",
                "best_val": bestPerformance[1],
                "vgae": model[0].state_dict(),
                "mlp": model[2].state_dict(),
                "generator": model[3].state_dict(),
                "env_infer": model[4].state_dict(),
                "rec_model": rec_model.state_dict(),
            }, checkpoint_path)
            print(f"Saved best validation checkpoint to {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model[0].load_state_dict(checkpoint["vgae"])
    model[2].load_state_dict(checkpoint["mlp"])
    model[3].load_state_dict(checkpoint["generator"])
    model[4].load_state_dict(checkpoint["env_infer"])
    rec_model.load_state_dict(checkpoint["rec_model"])
    best_epoch = int(checkpoint["epoch"])
    best_val_metrics = parse_measure_block(val_measure_result[best_epoch - 1])
    print('Best validation result of %s:\n%s' % (
        'causal', ''.join(val_measure_result[best_epoch - 1]),
    ))
    if args.validation_only:
        test_measure = None
        best_metrics = None
        print('[protocol] validation_only=True: test GT was not evaluated')
    else:
        test_measure = evaluate_epoch(
            datasets, "test", rec_model, device, num_user, num_item,
            user_item_train_inter,
        )
        best_metrics = parse_measure_block(test_measure)
        print('One-shot test result of %s:\n%s' % ('causal', ''.join(test_measure)))

    peak_vram_mb = None
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    record = {
        "settings": build_settings(args, extra={
            **(run_meta or {}),
            "protocol_version": PROTOCOL_VERSION,
            "stability": {
                "evae_reduction": "mean",
                "log_std_range": [-10.0, 5.0],
                "grad_clip": float(args.grad_clip),
            },
            "selection_split": "val",
            "selection_metric": "NDCG@20",
            "validation_only": bool(args.validation_only),
            "ranking_head_lifecycle": "persistent_with_upstream_refresh",
            "rec_refresh": float(args.rec_refresh),
            "rec_lr": float(args.rec_lr),
            "rec_batch_size": int(args.rec_batch_size),
            "rec_epochs_per_outer": int(args.rec_epochs_per_outer),
        }),
        "best_epoch": int(best_epoch),
        "best_val_metrics": best_val_metrics,
        "best_metrics": best_metrics,
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
            'causal', ''.join(val_measure_result[best_epoch - 1]),
        ))
        if test_measure is None:
            f.write('TEST_NOT_EVALUATED: validation_only=True\n')
        else:
            f.write('One-shot test result of %s:\n%s\n' % (
                'causal', ''.join(test_measure),
            ))
        if peak_vram_mb is not None:
            f.write(f"peak_vram_mb:{peak_vram_mb:.2f}\n")
    return record


def run_epoch(models, optimizers, feats, edge_index, adj, norm, weight_tensor, device, epoch, user_item_train_inter,
              num_user, num_item, train_graph, rec_model=None, rec_optimizer=None):
    vgae_model, diffusion_model, mlp_model, generator, env_infer_model = models
    mlp_model.train()

    pretrain_loss, total_rec_loss = 0.0, 0.0

    for m in range(1):
        Loss, Log_p = [], 0
        for k in range(3):
            dge_index, log_p = generator(feats.shape[0], 5, k, edge_index, num_user=num_user)
            gc.collect()
            batch_latent = vgae_model.encoder(feats, dge_index)

            recon, mu, log_std = env_infer_model(batch_latent)
            kl_div = env_infer_model.kl_divergence(mu, log_std)
            infer_loss = evae_loss(recon, batch_latent, kl_div)
            env_embeddings = env_infer_model.decode(batch_latent)

            terms = diffusion_model.training_losses(mlp_model, batch_latent, env_embeddings, args.reweight)
            elbo = terms["loss"].mean()
            logits = vgae_model.decoder(terms["pred_xstart"])
            vgae_loss = compute_vgae_loss(logits, adj, norm, vgae_model, weight_tensor)
            loss = adjust_loss(elbo, vgae_loss, infer_loss, args.reweight)
            for component_name, component in (
                ("elbo", elbo), ("vgae_loss", vgae_loss),
                ("infer_loss", infer_loss), ("environment_loss", loss),
            ):
                require_finite(component_name, component, epoch)
            Loss.append(loss.view(-1))
            Log_p += log_p

        Var, Mean = torch.var_mean(torch.cat(Loss, dim=0))
        if Var is None:
            Var = 0
        outer_loss = Var + Mean * compute_beta(
            epoch, args.epochs, args.risk_weight_start, args.risk_weight_end,
        )

        pretrain_loss += outer_loss.item()

        handle_gradient_step(optimizers, outer_loss, Log_p, m, Var, epoch)

    all_embeddings = generate_embeddings(models[0], models[1], models[2], models[4], feats, edge_index, 1, device)
    user_embeddings = all_embeddings[:num_user]
    item_embeddings = all_embeddings[num_user:]

    ui_adj = generate_interaction_matrix_from_dgl(train_graph, num_user, num_item)
    norm_adj = normalize_graph_mat(ui_adj)

    rec_model, rec_optimizer = initialize_or_refresh_ranking_head(
        num_user, norm_adj, user_embeddings, item_embeddings, device,
        rec_model=rec_model, rec_optimizer=rec_optimizer,
        refresh=args.rec_refresh, learning_rate=args.rec_lr, layers=3,
    )

    dataset = UserItemDataset(user_item_train_inter)
    # CPU training typically requires smaller batch sizes for memory efficiency
    dataloader = DataLoader(dataset, batch_size=args.rec_batch_size, shuffle=True)
    for _ in range(int(args.rec_epochs_per_outer)):
        for batch in dataloader:
            user_id, pos_item_id, neg_item_id = [x.to(device) for x in batch]
            all_embeddings = rec_model().to(device)

            user_embedding = all_embeddings[user_id]
            pos_item_embedding = all_embeddings[pos_item_id.long() + num_user]
            neg_item_embedding = all_embeddings[neg_item_id.long() + num_user]

            rec_loss = bpr_loss(user_embedding, pos_item_embedding, neg_item_embedding) + \
                       l2_reg_loss(1e-3, user_embedding, pos_item_embedding, neg_item_embedding)
            rec_optimizer.zero_grad()
            rec_loss.backward()
            torch.nn.utils.clip_grad_norm_(rec_model.parameters(), 0.7)
            rec_optimizer.step()

            total_rec_loss += rec_loss.item()

    return pretrain_loss, total_rec_loss, rec_model, rec_optimizer


def evaluate_epoch(datasets, split, trained_rec_model, device, num_user, num_item,
                   train_interactions):
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
        all_embeddings = rec_model().to(device)
        user_embeddings = all_embeddings[:num_user]
        item_embeddings = all_embeddings[num_user:]

    # 批处理计算评分矩阵，避免一次性创建大矩阵（节省内存）
    # CPU训练使用更小的批次以避免内存溢出
    batch_size = 256
    scores_list = []
    for i in range(0, len(user_embeddings), batch_size):
        user_batch = user_embeddings[i:i+batch_size]
        batch_scores = torch.matmul(user_batch, item_embeddings.t())
        scores_list.append(batch_scores)

    scores = torch.cat(scores_list, dim=0)

    origin_inter = datasets[f"{split}_origin_inter"]
    user_set = datasets[f"{split}_user_set"]
    scores = mask_seen_items(scores, train_interactions, user_set)
    rec_dict = get_rec_list(user_set, scores, num_user, topk=20)

    measure = ranking_evaluation(origin_inter, rec_dict, [10, 20])
    return measure


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


def handle_gradient_step(optimizers, outer_loss, Log_p, m, Var, epoch):
    optimizer1, optimizer2, optimizer3, optimizer4 = optimizers
    optimizer1.zero_grad()
    optimizer2.zero_grad()
    optimizer3.zero_grad()
    optimizer4.zero_grad()
    print(outer_loss)
    require_finite("outer_loss", outer_loss, epoch)
    if m == 0:
        outer_loss.backward()
        for optimizer in (optimizer1, optimizer2, optimizer4):
            clip_optimizer_gradients(optimizer, args.grad_clip)
            optimizer.step()
    reward = Var.detach()
    inner_loss = - reward * Log_p
    inner_loss = inner_loss.mean()
    require_finite("generator_inner_loss", inner_loss, epoch)
    inner_loss.backward()
    clip_optimizer_gradients(optimizer3, args.grad_clip)
    optimizer3.step()


def generate_embeddings(vgae_model, diffusion_model, mlp_model, env_infer, features, edge_index, num_samples, device):
    features = features.to(device)
    edge_index = edge_index.to(device)
    vgae_model = vgae_model.to(device)
    mlp_model = mlp_model.to(device)
    env_infer = env_infer.to(device)

    embeddings_list = []
    modules = (vgae_model, mlp_model, env_infer)
    training_states = [module.training for module in modules]
    for module in modules:
        module.eval()
    try:
        with torch.no_grad():
            for _ in range(num_samples):
                z = vgae_model.encoder(
                    features, edge_index, sample=bool(args.sampling_noise),
                )
                env_embeddings = env_infer.decode(z)
                diffused_z = diffusion_model.p_sample(
                    mlp_model, z, env_embeddings, args.sampling_steps,
                    args.sampling_noise,
                )
                embeddings_list.append(diffused_z)
    finally:
        for module, was_training in zip(modules, training_states):
            module.train(was_training)

    final_embeddings = torch.mean(torch.stack(embeddings_list), dim=0)

    return final_embeddings


def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    print("Start model training!")
    start_time = time.time()
    seed_it(args.seed)
    device = setup_device()
    data_root = (args.data_root or "").strip()
    print(
        f"Using device: {device}, seed={args.seed}, dataset={args.dataset}, "
        f"data_root={data_root or 'legacy'}"
    )

    # loading data
    datasets, data_meta = load_datasets(
        args.dataset, data_root=data_root, eval_split=args.eval_split,
        load_test_gt=not args.validation_only,
    )
    train_graph = datasets['train']
    feats, adj_orig, edge_index, num_nodes = prepare_data(train_graph, device)
    indim = feats.shape[-1]
    latent_size = args.emd_size
    mlp_out_dims = eval(args.mlp_dims) + [latent_size]
    mlp_in_dims = mlp_out_dims[::-1]
    models = initialize_models(datasets['train'].number_of_nodes(), device, indim, mlp_in_dims, mlp_out_dims)
    optimizers = setup_optimizers(models)
    param_detail = count_params(models)

    if datasets.get("strict"):
        n_user, n_item = data_meta["n_user"], data_meta["n_item"]
        user_item_train_inter, num_users, num_items = user_item_matrix_from_graph(
            train_graph, n_user, n_item,
        )
    else:
        user_item_train_inter, num_users, num_items = get_user_item_matrix(train_graph)

    if data_root:
        args.run_id = f"{args.run_id}_strict_{args.eval_split}".lstrip("_")

    run_meta = {
        "num_nodes": int(num_nodes),
        "num_users": int(num_users),
        "num_items": int(num_items),
        "feat_dim": int(indim),
        "train_edges": int(train_graph.number_of_edges()),
        "test_edges": int(datasets['test'].number_of_edges()),
        "param_count": param_detail,
        "device": str(device),
        "method": "causaldiffrec_e0",
        **{
            k: data_meta[k]
            for k in (
                "data_version", "data_status", "eval_split", "data_root",
                "val_gt_users", "val_gt_interactions", "gt_users", "gt_interactions",
            )
            if k in data_meta
        },
    }
    record = train_model(models, optimizers, device, datasets, user_item_train_inter, num_users, num_items,
                         run_meta=run_meta)
    end_time = time.time()
    wall = end_time - start_time
    run_time = strftime("%H:%M:%S", gmtime(wall))
    print("total time cost：{0}".format(run_time))

    record["wall_time_sec"] = float(wall)
    record["wall_time_hms"] = run_time
    tag = args.dataset
    if data_root:
        tag = f"{args.dataset}_strict_{args.eval_split}"
    record_path = (
        f"experiments/records/{tag}_{safe_tag(args.run_id)}_"
        f"seed{args.seed}_{RECORD_SUFFIX}.json"
    )
    save_run_record(record_path, record)
    print(f"Saved experiment record to {record_path}")
