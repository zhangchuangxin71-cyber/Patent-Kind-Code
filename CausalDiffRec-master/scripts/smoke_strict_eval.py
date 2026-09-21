#!/usr/bin/env python
"""Lightweight smoke: strict load + random-emb ranking (no dense VGAE train)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.load_strict import load_strict_datasets, user_item_matrix_from_graph  # noqa: E402
from utils.util_loss import (  # noqa: E402
    generate_interaction_matrix_from_dgl,
    get_rec_list,
    mask_seen_items,
    normalize_graph_mat,
    ranking_evaluation,
)
from modules.rec_model import LGCN_Encoder  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", default="food")
    ap.add_argument("--data_root", default="")
    ap.add_argument("--eval_split", default="ood", choices=["ood", "iid"])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    root = args.data_root or f"data_strict/processed/{args.dataset}/v1_strict"
    device = torch.device(args.device)

    datasets, meta = load_strict_datasets(args.dataset, data_root=root, eval_split=args.eval_split)
    n_user, n_item = meta["n_user"], meta["n_item"]
    mat, _, _ = user_item_matrix_from_graph(datasets["train"], n_user, n_item)
    assert mat.nnz > 0

    g = datasets["test"].to(device)
    feat = g.ndata["feat"].to(device)
    # use feat slices as init emb (no training)
    user_emb = feat[:n_user].detach()
    item_emb = feat[n_user:].detach()
    ui_adj = generate_interaction_matrix_from_dgl(g.cpu(), n_user, n_item)
    norm_adj = normalize_graph_mat(ui_adj)
    rec = LGCN_Encoder(n_user, 2, norm_adj, user_emb, item_emb).to(device)
    with torch.no_grad():
        emb = rec().to(device)
        u, i = emb[:n_user], emb[n_user:]
        scores = torch.matmul(u, i.t())
    origin, user_set = datasets["origin_inter"], datasets["user_set"]
    scores = mask_seen_items(scores.cpu(), mat, user_set)
    rec_dict = get_rec_list(user_set, scores, n_user, topk=20)
    measure = ranking_evaluation(origin, rec_dict, [10, 20])
    print("".join(measure))
    print("SMOKE_OK", meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
