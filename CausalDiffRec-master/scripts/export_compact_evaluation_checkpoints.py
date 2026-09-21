#!/usr/bin/env python
"""Export small rec-model checkpoints sufficient for frozen OOD/alpha evaluation."""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def arm_from_record(record: dict) -> str:
    settings = record["settings"]
    if not settings.get("use_semantic_prior"):
        return "A0"
    return "A1" if float(settings.get("lambda_sem")) == 0.0 else "A2"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--datasets", default="yelp2018,movielens1m")
    parser.add_argument("--out_dir", default="artifacts/v6_evaluation")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    out_root = root / args.out_dir
    checkpoint_dir = out_root / "checkpoints"
    record_dir = out_root / "records"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    record_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for dataset in [name.strip() for name in args.datasets.split(",") if name.strip()]:
        patterns = [
            f"{dataset}_lsci_strict_ood_v6_module_ablation_a0_*_v4.json",
            f"{dataset}_lsci_strict_ood_e4_lamsem0_v6_module_ablation_a1_*_v4.json",
            f"{dataset}_lsci_strict_ood_e3_v6_module_ablation_a2_*_v4.json",
        ]
        paths = []
        for pattern in patterns:
            paths.extend(glob.glob(str(root / "experiments" / "records" / pattern)))
        for path_text in sorted(paths):
            source_record_path = Path(path_text)
            record = json.loads(source_record_path.read_text(encoding="utf-8"))
            arm = arm_from_record(record)
            seed = int(record["settings"]["seed"])
            source_checkpoint = root / record["checkpoint"]
            checkpoint = torch.load(source_checkpoint, map_location="cpu")
            compact_rel = Path(args.out_dir) / "checkpoints" / dataset / f"{arm.lower()}_seed{seed}.pt"
            compact_path = root / compact_rel
            compact_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "rec_model": checkpoint["rec_model"],
                "epoch": checkpoint["epoch"],
                "dataset": dataset,
                "seed": seed,
                "arm": arm,
                "selection_split": "val",
                "selection_metric": "NDCG@20",
                "source_training_checkpoint_sha256": sha256(source_checkpoint),
                "scope": "frozen ranking and semantic late-fusion evaluation only",
            }, compact_path)
            portable = dict(record)
            portable["checkpoint"] = str(compact_rel)
            portable["full_training_checkpoint"] = record["checkpoint"]
            portable["portable_checkpoint_scope"] = "rec_model only; sufficient for evaluate_late_fusion.py"
            portable_path = record_dir / f"{dataset}_{arm.lower()}_seed{seed}.json"
            portable_path.write_text(json.dumps(portable, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            entries.append({
                "dataset": dataset,
                "arm": arm,
                "seed": seed,
                "portable_record": str(portable_path.relative_to(root)),
                "compact_checkpoint": str(compact_rel),
                "compact_checkpoint_bytes": compact_path.stat().st_size,
                "compact_checkpoint_sha256": sha256(compact_path),
                "source_record_sha256": sha256(source_record_path),
                "source_training_checkpoint_sha256": sha256(source_checkpoint),
            })
    expected = len([name for name in args.datasets.split(",") if name.strip()]) * 15
    if len(entries) != expected:
        raise RuntimeError(f"expected {expected} A0/A1/A2 checkpoints, found {len(entries)}")
    manifest = {
        "protocol_version": "corrected_v6_compact_evaluation_checkpoints",
        "scope": "alpha search and frozen OOD evaluation; excludes optimizer/upstream training state",
        "entries": entries,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_root / "README.md").write_text(
        "# v6 精简评价检查点\n\n"
        "每个文件仅保留冻结的 `rec_model` 状态，足以配合对应 portable record、"
        "严格数据包和 `scripts/evaluate_late_fusion.py` 复核 A0/A1/A2/A3。"
        "它们不含优化器、VGAE、扩散或 ConditionFusion 完整训练状态，不能用于续训。\n",
        encoding="utf-8",
    )
    print(out_root / "manifest.json")


if __name__ == "__main__":
    main()
