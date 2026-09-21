"""Manifest helpers."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from .io import sha256_file, write_json


def build_manifest(**kwargs) -> Dict[str, Any]:
    m = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        **kwargs,
    }
    return m


def finalize_manifest(path: Path, manifest: Dict[str, Any], file_paths) -> Dict[str, Any]:
    hashes = {}
    for k, p in file_paths.items():
        pp = Path(p)
        if pp.exists() and pp.is_file():
            hashes[k] = sha256_file(pp)
    manifest["file_sha256"] = hashes
    write_json(path, manifest)
    return manifest
