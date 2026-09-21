"""Experiment logging helpers: settings dump, metric parse, mean/std summary."""
import json
import os
import re
from collections import defaultdict


METRIC_KEYS = ("Hit Ratio", "Precision", "Recall", "NDCG")


def parse_measure_block(measure_lines):
    """Parse ranking_evaluation output list/str into {Top10/Top20: metrics}."""
    if isinstance(measure_lines, str):
        text = measure_lines
    else:
        text = "".join(measure_lines)

    result = {}
    current = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Top "):
            current = line.replace(" ", "")  # Top10 / Top20
            result[current] = {}
        elif current and ":" in line:
            k, v = line.split(":", 1)
            k = k.strip()
            if k in METRIC_KEYS:
                result[current][k] = float(v.strip())
    return result


def parse_best_dict(best_dict):
    """fast_evaluation stores Top-20 metrics only in bestPerformance[1]."""
    if not isinstance(best_dict, dict):
        return {}
    out = {}
    for k in METRIC_KEYS:
        if k in best_dict:
            out[k] = float(best_dict[k])
    return {"Top20": out}


def count_params(models, names=None):
    total = 0
    detail = {}
    if names is None:
        names = ["vgae", "diffusion", "mlp", "generator", "env_infer"]
    if len(names) != len(models):
        raise ValueError("names and models must have the same length")
    for name, m in zip(names, models):
        if m is None:
            detail[name] = 0
            continue
        n = sum(p.numel() for p in m.parameters())
        detail[name] = int(n)
        total += n
    detail["total"] = int(total)
    return detail


def build_settings(args, extra=None):
    settings = {
        "dataset": args.dataset,
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "lr2": args.lr2,
        "wd2": args.wd2,
        "hidden1": args.hidden1,
        "hidden2": args.hidden2,
        "emd_size": args.emd_size,
        "emb_size": args.emb_size,
        "mlp_dims": args.mlp_dims,
        "steps": args.steps,
        "sampling_steps": args.sampling_steps,
        "noise_schedule": args.noise_schedule,
        "noise_scale": args.noise_scale,
        "noise_min": args.noise_min,
        "noise_max": args.noise_max,
        "mean_type": args.mean_type,
        "reweight": args.reweight,
        "sampling_noise": args.sampling_noise,
        "gpu_id": args.gpu_id,
        "generator_rank": getattr(args, "generator_rank", 128),
        "num_environments_K": getattr(args, "num_env", 3),
        "bpr_batch_size": getattr(args, "rec_batch_size", 1024),
        "lgcn_layers": getattr(args, "rank_layers", 3),
        "rec_lr": getattr(args, "rec_lr", 0.001),
        "rec_epochs_per_outer": getattr(args, "rec_epochs_per_outer", 1),
    }
    if extra:
        settings.update(extra)
    return settings


def save_run_record(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


def load_run_records(records_dir, dataset=None):
    rows = []
    if not os.path.isdir(records_dir):
        return rows
    for name in sorted(os.listdir(records_dir)):
        if not name.endswith(".json"):
            continue
        if dataset and not name.startswith(dataset + "_"):
            continue
        with open(os.path.join(records_dir, name), "r", encoding="utf-8") as f:
            rows.append(json.load(f))
    return rows


def summarize_runs(records):
    """Compute mean and sample std (ddof=1) over repeated runs."""
    buckets = defaultdict(list)
    for rec in records:
        metrics = rec.get("best_metrics", {})
        for top, vals in metrics.items():
            for k, v in vals.items():
                buckets[f"{top}/{k}"].append(float(v))
        if "wall_time_sec" in rec:
            buckets["wall_time_sec"].append(float(rec["wall_time_sec"]))
        if "peak_vram_mb" in rec and rec["peak_vram_mb"] is not None:
            buckets["peak_vram_mb"].append(float(rec["peak_vram_mb"]))

    summary = {}
    for key, vals in buckets.items():
        n = len(vals)
        mean = sum(vals) / n
        if n >= 2:
            var = sum((x - mean) ** 2 for x in vals) / (n - 1)
            std = var ** 0.5
        else:
            var, std = 0.0, 0.0
        summary[key] = {
            "n": n,
            "mean": mean,
            "std": std,
            "var": var,
            "values": vals,
        }
    return summary


def format_mean_std(mean, std, digits=4):
    return f"{mean:.{digits}f} ± {std:.{digits}f}"
