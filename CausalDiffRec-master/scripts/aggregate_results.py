#!/usr/bin/env python
"""Aggregate multi-seed experiment records into mean ± std summaries."""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.experiment_utils import load_run_records, summarize_runs, format_mean_std

RECORDS = os.path.join(ROOT, "experiments", "records")
SUMMARIES = os.path.join(ROOT, "experiments", "summaries")
REPORT = os.path.join(ROOT, "experiments", "实验记录总表.md")


def write_dataset_summary(dataset, summary, records):
    os.makedirs(SUMMARIES, exist_ok=True)
    out = {
        "dataset": dataset,
        "n_runs": len(records),
        "seeds": [r.get("settings", {}).get("seed") for r in records],
        "summary": summary,
        "settings_template": records[0].get("settings") if records else {},
    }
    path = os.path.join(SUMMARIES, f"{dataset}_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return path


def metric_line(summary, key):
    if key not in summary:
        return "N/A"
    s = summary[key]
    return format_mean_std(s["mean"], s["std"])


def rebuild_master_report(datasets):
    lines = [
        "# CausalDiffRec 多随机种子实验记录总表",
        "",
        "## 实验协议",
        "",
        "- 重复次数：每个数据集 **5** 个独立随机种子 `{1024, 2048, 3072, 4096, 5120}`",
        "- 汇报形式：指标 = **均值 ± 标准差**（样本标准差，ddof=1）",
        "- 训练轮数：25；扩散步 T=100；环境数 K=3；低秩 generator rank=128",
        "- 详细单次记录：`experiments/records/<dataset>_seed<seed>.json`",
        "- 汇总：`experiments/summaries/<dataset>_summary.json`",
        "",
        "## 结果汇总（OOD 测试）",
        "",
        "| Dataset | n | R@10 | N@10 | R@20 | N@20 | 耗时(s) | 峰值显存(MB) |",
        "|---------|---|------|------|------|------|---------|--------------|",
    ]
    for ds in datasets:
        records = load_run_records(RECORDS, ds)
        if not records:
            lines.append(f"| {ds} | 0 | - | - | - | - | - | - |")
            continue
        summary = summarize_runs(records)
        lines.append(
            "| {ds} | {n} | {r10} | {n10} | {r20} | {n20} | {t} | {m} |".format(
                ds=ds,
                n=len(records),
                r10=metric_line(summary, "Top10/Recall"),
                n10=metric_line(summary, "Top10/NDCG"),
                r20=metric_line(summary, "Top20/Recall"),
                n20=metric_line(summary, "Top20/NDCG"),
                t=metric_line(summary, "wall_time_sec"),
                m=metric_line(summary, "peak_vram_mb"),
            )
        )

    lines += [
        "",
        "## 单次实验设置字段",
        "",
        "每条 JSON 记录包含：`settings`（超参）、`best_metrics`、`epoch_logs`、",
        "`wall_time_sec`、`peak_vram_mb`、`param_count`、数据规模（用户/物品/边数）。",
        "",
        "## 复杂度分析",
        "",
        "见 `experiments/复杂度分析.md`。",
        "",
    ]
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return REPORT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    datasets = []
    if args.dataset:
        datasets = [args.dataset]
    if args.all or not datasets:
        # Discover from records
        names = set()
        if os.path.isdir(RECORDS):
            for fn in os.listdir(RECORDS):
                if fn.endswith(".json") and "_seed" in fn:
                    names.add(fn.split("_seed")[0])
        datasets = sorted(names) if names else ["yelp2018", "douban", "food", "kuairec"]

    for ds in datasets:
        records = load_run_records(RECORDS, ds)
        if not records:
            print(f"[{ds}] no records yet")
            continue
        summary = summarize_runs(records)
        path = write_dataset_summary(ds, summary, records)
        print(f"[{ds}] n={len(records)} -> {path}")
        for key in ("Top10/Recall", "Top10/NDCG", "Top20/Recall", "Top20/NDCG"):
            if key in summary:
                s = summary[key]
                print(f"  {key}: {format_mean_std(s['mean'], s['std'])}  values={s['values']}")

    report = rebuild_master_report(datasets)
    print(f"Master report: {report}")


if __name__ == "__main__":
    main()
