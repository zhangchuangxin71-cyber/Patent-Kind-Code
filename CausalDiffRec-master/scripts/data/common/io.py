"""Shared IO helpers for strict dataset rebuild."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY = Path(__file__).resolve().parents[1] / "dataset_registry.yaml"


def project_root() -> Path:
    return ROOT


def load_registry(path: Optional[Path] = None) -> Dict[str, Any]:
    p = path or DEFAULT_REGISTRY
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def data_strict_root(registry: Optional[Dict[str, Any]] = None) -> Path:
    reg = registry or load_registry()
    return project_root() / reg.get("root", "data_strict")
