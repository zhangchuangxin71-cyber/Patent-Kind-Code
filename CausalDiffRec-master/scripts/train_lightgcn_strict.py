#!/usr/bin/env python
"""Validation-only pure LightGCN baseline under the corrected-v3 protocol."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.rec_model import LGCN_Encoder  # noqa: E402
from utils.experiment_utils import parse_measure_block, save_run_record  # noqa: E402
from utils.input_data import UserItemDataset  # noqa: E402
from utils.load_strict import load_strict_datasets, user_item_matrix_from_graph  # noqa: E402
from utils.util_loss import (  # noqa: E402
    bpr_loss,
    generate_interaction_matrix_from_dgl,
    get_rec_list,
    l2_reg_loss,
    mask_seen_items,
    normalize_graph_mat,
    ranking_evaluation,
)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate(model, datasets, train, n_user, device):
    model.eval()
    users = datasets["val_user_set"]
    with torch.no_grad():
        emb = model()
        scores = emb[:n_user] @ emb[n_user:].t()
        scores = mask_seen_items(scores, train, users)
        recs = get_rec_list(users, scores, n_user, topk=20)
    lines = ranking_evaluation(datasets["val_origin_inter"], recs, [10, 20])
    return lines, parse_measure_block(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="yelp2018")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--min_epochs", type=int, default=25)
    ap.add_argument(
        "--patience", type=int, default=0,
        help="Stop after this many validation epochs without improvement; 0 disables.",
    )
    ap.add_argument("--embedding_dim", type=int, default=8)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--l2", type=float, default=0.001)
    ap.add_argument(
        "--protocol_version", default="corrected_v3_lightgcn_validation_only",
        help="Recorded protocol identifier; does not alter optimization.",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--resume_checkpoint", default="")
    ap.add_argument("--resume_record", default="")
    ap.add_argument(
        "--init_checkpoint", default="",
        help="Initialize rec_model from a source checkpoint but use a fresh optimizer.",
    )
    ap.add_argument("--init_record", default="")
    args = ap.parse_args()

    start = time.time()
    seed_everything(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    datasets, meta = load_strict_datasets(
        args.dataset, args.data_root, eval_split="ood", load_test_gt=False,
    )
    n_user, n_item = int(meta["n_user"]), int(meta["n_item"])
    train, _, _ = user_item_matrix_from_graph(datasets["train"], n_user, n_item)
    norm_adj = normalize_graph_mat(generate_interaction_matrix_from_dgl(
        datasets["train"], n_user, n_item,
    ))
    user_init = torch.empty(n_user, args.embedding_dim, device=device)
    item_init = torch.empty(n_item, args.embedding_dim, device=device)
    torch.nn.init.xavier_uniform_(user_init)
    torch.nn.init.xavier_uniform_(item_init)
    model = LGCN_Encoder(
        n_user, args.layers, norm_adj, user_init, item_init,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_ndcg = -1.0
    best_epoch = None
    best_metrics = None
    epoch_logs = []
    first_epoch = 1
    epochs_without_improvement = 0
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    if bool(args.resume_checkpoint) != bool(args.resume_record):
        raise ValueError("--resume_checkpoint and --resume_record must be provided together")
    if bool(args.init_checkpoint) != bool(args.init_record):
        raise ValueError("--init_checkpoint and --init_record must be provided together")
    if args.resume_checkpoint and args.init_checkpoint:
        raise ValueError("Resume and initialization modes are mutually exclusive")
    if args.resume_checkpoint:
        resume = torch.load(args.resume_checkpoint, map_location=device)
        prior = json.load(open(args.resume_record, encoding="utf-8"))
        if int(prior["settings"]["seed"]) != args.seed:
            raise ValueError("Resume record seed does not match --seed")
        model.load_state_dict(resume["rec_model"])
        if "optimizer" not in resume:
            raise ValueError("Resume checkpoint has no optimizer state")
        optimizer.load_state_dict(resume["optimizer"])
        best_epoch = int(resume["epoch"])
        best_metrics = prior["best_val_metrics"]
        best_ndcg = float(best_metrics["Top20"]["NDCG"])
        epoch_logs = [
            entry for entry in prior.get("epoch_logs", [])
            if int(entry["epoch"]) <= best_epoch
        ]
        first_epoch = best_epoch + 1
        # Seed the new destination checkpoint with the prior best in case the
        # continuation never improves it.
        torch.save({
            **resume,
            "rec_model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }, args.checkpoint)
        print(
            f"resume seed={args.seed} best_epoch={best_epoch} "
            f"best_ndcg20={best_ndcg:.5f}",
            flush=True,
        )
    elif args.init_checkpoint:
        source_checkpoint = torch.load(args.init_checkpoint, map_location=device)
        source_record = json.load(open(args.init_record, encoding="utf-8"))
        if int(source_record["settings"]["seed"]) != args.seed:
            raise ValueError("Initialization record seed does not match --seed")
        model.load_state_dict(source_checkpoint["rec_model"])
        initial_lines, initial_metrics = evaluate(
            model, datasets, train, n_user, device,
        )
        best_ndcg = float(initial_metrics["Top20"]["NDCG"])
        best_epoch = 0
        best_metrics = initial_metrics
        epoch_logs = [{
            "epoch": 0,
            "train_loss_sum": None,
            "val_metrics": initial_metrics,
            "is_best": True,
            "source_best_epoch": int(source_record["best_epoch"]),
        }]
        torch.save({
            "epoch": 0,
            "dataset": args.dataset,
            "seed": args.seed,
            "protocol_version": args.protocol_version,
            "selection_split": "val",
            "selection_metric": "NDCG@20",
            "rec_model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "initialization_source_checkpoint": args.init_checkpoint,
        }, args.checkpoint)
        print(
            f"initialize_from_source seed={args.seed} "
            f"source_best_epoch={source_record['best_epoch']} "
            f"initial_ndcg20={best_ndcg:.5f}",
            flush=True,
        )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(first_epoch, args.epochs + 1):
        model.train()
        dataset = UserItemDataset(train)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        total_loss = 0.0
        for user, pos, neg in loader:
            user, pos, neg = user.to(device), pos.to(device), neg.to(device)
            emb = model()
            ue = emb[user]
            pe = emb[pos.long() + n_user]
            ne = emb[neg.long() + n_user]
            loss = bpr_loss(ue, pe, ne) + l2_reg_loss(args.l2, ue, pe, ne)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
            optimizer.step()
            total_loss += float(loss.detach())
        lines, metrics = evaluate(model, datasets, train, n_user, device)
        ndcg = float(metrics["Top20"]["NDCG"])
        is_best = ndcg > best_ndcg
        if is_best:
            best_ndcg, best_epoch, best_metrics = ndcg, epoch, metrics
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch,
                "dataset": args.dataset,
                "seed": args.seed,
                "protocol_version": args.protocol_version,
                "selection_split": "val",
                "selection_metric": "NDCG@20",
                "rec_model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, args.checkpoint)
        else:
            epochs_without_improvement += 1
        epoch_logs.append({
            "epoch": epoch,
            "train_loss_sum": total_loss,
            "val_metrics": metrics,
            "is_best": is_best,
        })
        print(
            f"epoch={epoch}/{args.epochs} loss={total_loss:.6f} "
            f"val_ndcg20={ndcg:.5f} best={best_ndcg:.5f}",
            flush=True,
        )
        if (
            args.patience > 0
            and epoch >= args.min_epochs
            and epochs_without_improvement >= args.patience
        ):
            print(
                f"early_stop epoch={epoch} best_epoch={best_epoch} "
                f"patience={args.patience}",
                flush=True,
            )
            break

    peak = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if torch.cuda.is_available() else None
    )
    record = {
        "settings": {
            "dataset": args.dataset,
            "seed": args.seed,
            "method": "pure_lightgcn_bpr",
            "protocol_version": args.protocol_version,
            "validation_only": True,
            "test_gt_loaded": False,
            "selection_split": "val",
            "selection_metric": "NDCG@20",
            "epochs": args.epochs,
            "trained_epochs": len(epoch_logs),
            "min_epochs": args.min_epochs,
            "early_stopping_patience": args.patience,
            "embedding_dim": args.embedding_dim,
            "lgcn_layers": args.layers,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "l2": args.l2,
            "initialization": (
                "source_rec_model" if args.init_checkpoint else "xavier_uniform"
            ),
            "negative_sampling": "one_random_unobserved_item_per_positive_resampled_each_epoch",
            "resume_checkpoint": args.resume_checkpoint or None,
            "resume_record": args.resume_record or None,
            "init_checkpoint": args.init_checkpoint or None,
            "init_record": args.init_record or None,
            "data_version": meta["data_version"],
            "data_root": meta["data_root"],
            "eval_split": "ood",
        },
        "best_epoch": best_epoch,
        "best_val_metrics": best_metrics,
        "best_metrics": None,
        "checkpoint": args.checkpoint,
        "epoch_logs": epoch_logs,
        "peak_vram_mb": peak,
        "wall_time_sec": time.time() - start,
    }
    save_run_record(args.out, record)
    print(json.dumps({
        "best_epoch": best_epoch,
        "best_val_metrics": best_metrics,
        "test_evaluated": False,
        "out": args.out,
    }, indent=2))


if __name__ == "__main__":
    main()
