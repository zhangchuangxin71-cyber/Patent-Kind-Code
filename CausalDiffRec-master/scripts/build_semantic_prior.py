#!/usr/bin/env python
"""E2: Build semantic_prior.pt from data_strict items.jsonl.

Modes:
  raw  — SBERT encode title/categories/description (no LLM; default if no key)
  llm  — DeepSeek rewrite then SBERT (requires DEEPSEEK_API_KEY)

Output:
  data_strict/processed/<ds>/v1_strict/features/semantic_prior.pt
  {
    item_emb: [n_item, D],
    semantic_mask: [n_item],
    meta: {...}
  }

User history prompts (train-only) are prepared separately for E3; this script
only builds **item** static priors (anti-leakage).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.semantic_prior import item_text_from_row, save_semantic_prior  # noqa: E402

DEFAULT_SBERT = "sentence-transformers/all-MiniLM-L6-v2"
PROXY = "http://127.0.0.1:7897"


def load_items_jsonl(path: Path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: int(r["item_id"]))
    return rows


def ensure_proxy_env():
    """Force HTTP(S) proxy; strip SOCKS env that httpx rejects."""
    for k in (
        "ALL_PROXY", "all_proxy", "SOCKS_PROXY", "socks_proxy",
        "SOCKS5_PROXY", "socks5_proxy",
    ):
        os.environ.pop(k, None)
    http_proxy = "http://127.0.0.1:7897"
    os.environ["HTTP_PROXY"] = http_proxy
    os.environ["HTTPS_PROXY"] = http_proxy
    os.environ["http_proxy"] = http_proxy
    os.environ["https_proxy"] = http_proxy


def deepseek_rewrite(
    texts,
    model: str = "deepseek-chat",
    batch_sleep: float = 0.05,
    cache_path: Path = None,
):
    """Rewrite item texts via DeepSeek chat. Returns list[str] same length.

    Optional JSONL cache: each line {"i": int, "text": str} for resume.
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY unset; use --mode raw or export the key")
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("pip install openai") from e

    ensure_proxy_env()
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    cached = {}
    if cache_path and cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                    cached[int(o["i"])] = o.get("text") or ""
                except Exception:
                    continue
        print(f"  LLM cache hit {len(cached)}/{len(texts)} from {cache_path}", flush=True)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_f = open(cache_path, "a", encoding="utf-8")
    else:
        cache_f = None

    out = [""] * len(texts)
    try:
        for i, t in enumerate(texts):
            if not t.strip():
                out[i] = ""
                continue
            if i in cached and cached[i].strip():
                out[i] = cached[i]
                continue
            prompt = (
                "Summarize the following item for recommendation semantic matching. "
                "Keep key attributes, categories, and intent in one short English paragraph "
                "(<=80 words). No marketing fluff.\n\n" + t
            )
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=160,
                )
                rewritten = (resp.choices[0].message.content or "").strip()
                out[i] = rewritten if rewritten else t
            except Exception as e:
                print(f"[warn] DeepSeek failed item {i}: {e}; fallback to raw", flush=True)
                out[i] = t
            if cache_f is not None:
                cache_f.write(json.dumps({"i": i, "text": out[i]}, ensure_ascii=False) + "\n")
                cache_f.flush()
            if batch_sleep > 0:
                time.sleep(batch_sleep)
            if (i + 1) % 50 == 0:
                print(f"  LLM rewritten {i + 1}/{len(texts)}", flush=True)
    finally:
        if cache_f is not None:
            cache_f.close()
    return out


def encode_hash(texts, dim: int = 384, seed: int = 1024):
    """Deterministic hash-bag embedding (no SBERT dependency; for offline smoke)."""
    import hashlib
    import numpy as np

    rng = np.random.RandomState(seed)
    # fixed projection table for token hashes
    table = rng.randn(10007, dim).astype(np.float32)
    table /= np.linalg.norm(table, axis=1, keepdims=True) + 1e-8
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        if not t.strip():
            continue
        toks = t.lower().replace("|", " ").split()
        acc = np.zeros(dim, dtype=np.float32)
        for tok in toks[:64]:
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16) % 10007
            acc += table[h]
        n = np.linalg.norm(acc) + 1e-8
        out[i] = acc / n
    return torch.from_numpy(out)


def encode_sbert(texts, model_name: str, batch_size: int = 64, device: str = "cpu"):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers not installed. "
            "pip install sentence-transformers  OR use --backend hash"
        ) from e
    model = SentenceTransformer(model_name, device=device)
    safe = [t if t.strip() else " " for t in texts]
    emb = model.encode(
        safe,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return torch.from_numpy(emb).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, choices=["food", "kuairec", "yelp2018", "douban", "movielens1m", "amazon_beauty"])
    ap.add_argument(
        "--data_root",
        default="",
        help="v1_strict root; default data_strict/processed/<ds>/v1_strict",
    )
    ap.add_argument("--mode", choices=["raw", "llm", "auto"], default="auto")
    ap.add_argument("--sbert_model", default=DEFAULT_SBERT)
    ap.add_argument("--backend", choices=["auto", "sbert", "hash"], default="auto")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_items", type=int, default=0, help="0=all; >0 for smoke")
    ap.add_argument("--out", default="", help="override output .pt path")
    ap.add_argument("--llm_cache", default="", help="JSONL cache for DeepSeek rewrites")
    args = ap.parse_args()

    root = Path(args.data_root) if args.data_root else Path("data_strict/processed") / args.dataset / "v1_strict"
    ready = root / "READY"
    if not ready.exists():
        raise FileNotFoundError(f"missing READY: {ready}")
    items_path = root / "metadata" / "items.jsonl"
    rows = load_items_jsonl(items_path)
    n_item = len(rows)

    mode = args.mode
    if mode == "auto":
        mode = "llm" if os.environ.get("DEEPSEEK_API_KEY", "").strip() else "raw"

    backend = args.backend
    if backend == "auto":
        try:
            import sentence_transformers  # noqa: F401
            backend = "sbert"
        except ImportError:
            backend = "hash"
            print("[warn] sentence-transformers missing; using hash backend")

    print(f"dataset={args.dataset} root={root} n_item={n_item} mode={mode} backend={backend}")

    texts = []
    mask_list = []
    for r in rows:
        t = item_text_from_row(r)
        ok = bool(r.get("text_available", False) and t.strip())
        texts.append(t if ok else "")
        mask_list.append(ok)

    cache_path = Path(args.llm_cache) if args.llm_cache else (root / "cache" / "llm_rewrite.jsonl")
    if mode == "llm":
        to_rewrite = [t if m else "" for t, m in zip(texts, mask_list)]
        if args.max_items > 0:
            head = deepseek_rewrite(to_rewrite[: args.max_items], cache_path=cache_path)
            texts = head + [""] * (n_item - len(head))
            mask_list = [bool(t.strip()) for t in texts]
        else:
            texts = deepseek_rewrite(to_rewrite, cache_path=cache_path)
            mask_list = [bool(t.strip()) for t in texts]


    def _encode(ts):
        if backend == "sbert":
            return encode_sbert(ts, args.sbert_model, args.batch_size, args.device)
        return encode_hash(ts, dim=384)

    if args.max_items > 0:
        encode_texts = texts[: args.max_items]
        encode_mask = mask_list[: args.max_items]
        for i in range(args.max_items, n_item):
            mask_list[i] = False
        emb_head = _encode(encode_texts)
        D = emb_head.shape[1]
        item_emb = torch.zeros(n_item, D)
        item_emb[: args.max_items] = emb_head
        for i, m in enumerate(encode_mask):
            if not m:
                item_emb[i].zero_()
        semantic_mask = torch.tensor(mask_list, dtype=torch.bool)
    else:
        item_emb = _encode(texts)
        for i, m in enumerate(mask_list):
            if not m:
                item_emb[i].zero_()
        semantic_mask = torch.tensor(mask_list, dtype=torch.bool)

    out = Path(args.out) if args.out else root / "features" / "semantic_prior.pt"
    meta = {
        "dataset": args.dataset,
        "mode": mode,
        "backend": backend,
        "sbert_model": args.sbert_model if backend == "sbert" else None,
        "n_item": n_item,
        "emb_dim": int(item_emb.shape[1]),
        "coverage": float(semantic_mask.float().mean().item()),
        "data_root": str(root),
    }
    save_semantic_prior(out, item_emb, semantic_mask, meta)
    print(f"Wrote {out} shape={tuple(item_emb.shape)} coverage={meta['coverage']:.4f}")


if __name__ == "__main__":
    main()
