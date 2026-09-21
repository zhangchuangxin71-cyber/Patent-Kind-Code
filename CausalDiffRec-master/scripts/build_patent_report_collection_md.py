#!/usr/bin/env python3
"""Build one Markdown collection from the final patent experiment materials."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


MARKDOWN_FILES = (
    "专利撰写AI实验交接稿.md",
    "专利实验结果与过程说明.md",
    "新增专利对比实验详细报告.md",
    "大模型语义信息生成与实验参与详细说明.md",
    "数据划分、文本时间边界与扩散接入LightGCN技术说明.md",
)
JSON_FILE = "专利实验机器可读摘要.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    source_dir = Path(args.source_dir)
    out = Path(args.out)

    blocks = [
        "# 专利实验报告合集",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 本次修订说明",
        "",
        "- Yelp2018 与 Food 扩展表的 12 个标准差已按五个 OOD seed 的样本标准差（`ddof=1`）更正；均值、配对差和方向不变。",
        "- corrected-v5 三方法审计现为 16 项可直接核验条件。Yelp2018、Food、Amazon Beauty 的 `alpha=0.75` 是固定组合参数，未提供外部语义权重确认文件，不表述为独立 validation confirmation。",
        "- 三方法表只支持组合方法在本次五个 seed、相应数据划分与 BPR25 预算下的整体方向；不拆分归因给因果干预、InfoNCE、语义条件或最终分数融合。",
        "",
    ]
    for filename in MARKDOWN_FILES:
        path = source_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        blocks.extend(["---", "", f"# 汇编文件：{filename}", "", path.read_text(encoding="utf-8").rstrip(), ""])

    json_path = source_dir / JSON_FILE
    if not json_path.exists():
        raise FileNotFoundError(json_path)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    blocks.extend([
        "---", "", f"# 汇编文件：{JSON_FILE}", "",
        "以下保留完整机器可读 JSON，便于另一套 AI 或审稿人复核。", "", "```json",
        json.dumps(data, ensure_ascii=False, indent=2), "```", "",
    ])
    out.write_text("\n".join(blocks), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
