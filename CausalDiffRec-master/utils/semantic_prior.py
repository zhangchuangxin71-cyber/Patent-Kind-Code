"""Semantic prior IO helpers for LSCI-DiffRec E2."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_semantic_prior(
    path: Path,
    item_emb: torch.Tensor,
    semantic_mask: torch.Tensor,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """
    item_emb: [n_item, D]
    semantic_mask: [n_item] bool — True if text_available / successfully encoded
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "item_emb": item_emb.cpu().float(),
        "semantic_mask": semantic_mask.cpu().bool(),
        "meta": meta or {},
    }
    torch.save(payload, path)


def load_semantic_prior(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    obj = torch.load(path, map_location=map_location)
    if "item_emb" not in obj or "semantic_mask" not in obj:
        raise KeyError(f"{path} missing item_emb/semantic_mask")
    return obj


def item_text_from_row(row: dict, max_chars: int = 512) -> str:
    """Compose offline text for SBERT / LLM prompt from items.jsonl row."""
    parts = []
    title = (row.get("title") or "").strip()
    if title:
        parts.append(title)
    cats = row.get("categories") or []
    if isinstance(cats, list) and cats:
        parts.append("categories: " + ", ".join(str(c) for c in cats[:20]))
    desc = (row.get("description") or "").strip()
    if desc:
        parts.append(desc)
    attrs = row.get("attributes") or {}
    if isinstance(attrs, dict) and attrs:
        kv = [f"{k}={v}" for k, v in list(attrs.items())[:10]]
        if kv:
            parts.append("attributes: " + "; ".join(kv))
    text = " | ".join(parts).strip()
    if not text:
        return ""
    return text[:max_chars]
