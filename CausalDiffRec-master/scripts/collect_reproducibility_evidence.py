#!/usr/bin/env python
"""Collect environment, hashes, LLM cache and engineering-cost evidence."""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.semantic_prior import item_text_from_row


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def version(module: str):
    try:
        return getattr(importlib.import_module(module), "__version__", "unknown")
    except Exception as exc:
        return f"unavailable: {exc.__class__.__name__}"


def load_items(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return sorted(rows, key=lambda row: int(row["item_id"]))


def cache_audit(root: Path) -> dict:
    cache_candidates = sorted((root / "cache").glob("*.jsonl")) if (root / "cache").exists() else []
    if not cache_candidates:
        return {"available": False}
    # Prefer the canonical cache, then the largest versioned cache.
    canonical = [path for path in cache_candidates if path.name == "llm_rewrite.jsonl"]
    cache_path = canonical[0] if canonical else max(cache_candidates, key=lambda path: path.stat().st_size)
    cached = {}
    duplicate_lines = 0
    malformed_lines = 0
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
            index = int(entry["i"])
            duplicate_lines += int(index in cached)
            cached[index] = entry.get("text") or ""
        except Exception:
            malformed_lines += 1
    rows = load_items(root / "metadata" / "items.jsonl")
    raw = [item_text_from_row(row) for row in rows]
    fallback = [index for index, text in cached.items() if index < len(raw) and text.strip() == raw[index].strip()]
    missing = sorted(set(range(len(rows))) - set(cached))
    return {
        "available": True,
        "path": str(cache_path),
        "sha256": sha256(cache_path),
        "lines": sum(1 for line in cache_path.read_text(encoding="utf-8").splitlines() if line.strip()),
        "unique_item_indices": len(cached),
        "expected_items": len(rows),
        "duplicate_lines": duplicate_lines,
        "malformed_lines": malformed_lines,
        "missing_item_indices": missing,
        "raw_text_fallback_indices_inferred": fallback,
        "raw_text_fallback_count_inferred": len(fallback),
        "caveat": "Legacy cache schema has no explicit success/error field; fallback is inferred by exact raw-text equality.",
    }


def file_entry(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def training_records(root: Path, dataset: str) -> list[dict]:
    patterns = [
        f"{dataset}_lsci_strict_ood_v6_module_ablation_a0_*_v4.json",
        f"{dataset}_lsci_strict_ood_e4_lamsem0_v6_module_ablation_a1_*_v4.json",
        f"{dataset}_lsci_strict_ood_e3_v6_module_ablation_a2_*_v4.json",
    ]
    output = []
    for pattern in patterns:
        for path_text in sorted(glob.glob(str(root / "experiments" / "records" / pattern))):
            path = Path(path_text)
            record = json.loads(path.read_text(encoding="utf-8"))
            checkpoint = root / record["checkpoint"]
            output.append({
                "record": file_entry(path),
                "checkpoint": file_entry(checkpoint),
                "settings": record["settings"],
                "best_epoch": record["best_epoch"],
                "param_count": record["settings"].get("param_count"),
                "wall_time_sec": record.get("wall_time_sec"),
                "peak_vram_mb": record.get("peak_vram_mb"),
                "epoch_timing": [entry.get("timing") for entry in record.get("epoch_logs", []) if entry.get("timing")],
            })
    return output


def evaluation_records(root: Path, dataset: str) -> list[dict]:
    output = []
    pattern = root / "experiments" / "records" / f"{dataset}_v6_module_ablation_ood_a3_real_seed*.json"
    for path_text in sorted(glob.glob(str(pattern))):
        path = Path(path_text)
        record = json.loads(path.read_text(encoding="utf-8"))
        output.append({
            "record": file_entry(path),
            "seed": record["settings"]["seed"],
            "timing": record.get("timing"),
            "peak_vram_mb": record.get("peak_vram_mb"),
        })
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--datasets", default="yelp2018,movielens1m,food,amazon_beauty")
    parser.add_argument("--out", default="experiments/reports/reproducibility_evidence.json")
    parser.add_argument("--markdown", default="experiments/reports/reproducibility_evidence.md")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()
    except Exception:
        gpu = []
    evidence = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "pytorch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "dgl": version("dgl"),
            "torch_geometric": version("torch_geometric"),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "gpu": gpu,
        },
        "llm_semantic_pipeline": {
            "model": "deepseek-chat",
            "prompt": (
                "Summarize the following item for recommendation semantic matching. "
                "Keep key attributes, categories, and intent in one short English paragraph "
                "(<=80 words). No marketing fluff."
            ),
            "temperature": 0.2,
            "max_tokens": 160,
            "sbert_model": "sentence-transformers/all-MiniLM-L6-v2",
            "online_llm_required": False,
            "online_diffusion_sampling_required": False,
        },
        "datasets": {},
    }
    for dataset in datasets:
        ds_root = root / "data_strict" / "processed" / dataset / "v1_strict"
        semantic_files = sorted((ds_root / "features").glob("semantic_prior*.pt"))
        evidence["datasets"][dataset] = {
            "manifest": file_entry(ds_root / "manifest.json"),
            "semantic_prior_files": [file_entry(path) for path in semantic_files],
            "llm_cache": cache_audit(ds_root),
            "training_records": training_records(root, dataset),
            "evaluation_records": evaluation_records(root, dataset),
        }
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# 复现环境与工程证据", "",
        f"Git提交：`{evidence['git_commit']}`", "",
        "## 环境", "",
        f"- Python: `{evidence['environment']['python']}`",
        f"- PyTorch/CUDA: `{evidence['environment']['pytorch']}` / `{evidence['environment']['pytorch_cuda']}`",
        f"- DGL/PyG: `{evidence['environment']['dgl']}` / `{evidence['environment']['torch_geometric']}`",
        f"- GPU: `{'; '.join(gpu)}`", "",
        "## 离线/在线边界", "",
        "DeepSeek 与 SBERT 均为离线处理；在线评价只加载语义向量和 LightGCN 状态，"
        "不调用大语言模型，也不执行扩散采样。各数据集的逐文件哈希、训练耗时、显存和评价耗时见 JSON。",
    ]
    (root / args.markdown).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
