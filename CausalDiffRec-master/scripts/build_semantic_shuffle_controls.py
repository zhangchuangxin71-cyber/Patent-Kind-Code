#!/usr/bin/env python
"""Build deterministic row-permutation semantic controls without changing vectors."""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.semantic_prior import load_semantic_prior, save_semantic_prior


def tensor_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seeds", default="1101,1102,1103,1104,1105")
    args = parser.parse_args()
    source = load_semantic_prior(args.source, "cpu")
    item = source["item_emb"].float().cpu()
    mask = source["semantic_mask"].bool().cpu()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_hash = tensor_hash(item)
    for seed in [int(value) for value in args.seeds.split(",")]:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        permutation = torch.randperm(item.shape[0], generator=generator)
        shuffled = item[permutation]
        shuffled_mask = mask[permutation]
        assert torch.equal(
            torch.sort(item.norm(dim=1)).values,
            torch.sort(shuffled.norm(dim=1)).values,
        )
        path = out_dir / f"semantic_prior_shuffled_seed{seed}.pt"
        save_semantic_prior(path, shuffled, shuffled_mask, meta={
            **source.get("meta", {}),
            "control": "item_row_permutation",
            "shuffle_seed": seed,
            "source_path": str(Path(args.source).resolve()),
            "source_item_tensor_sha256": source_hash,
            "shuffled_item_tensor_sha256": tensor_hash(shuffled),
            "permutation_sha256": tensor_hash(permutation),
            "vector_multiset_unchanged": True,
        })
        print(path)


if __name__ == "__main__":
    main()
