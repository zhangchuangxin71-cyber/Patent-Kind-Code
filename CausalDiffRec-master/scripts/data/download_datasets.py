#!/usr/bin/env python
"""Download / stage raw datasets according to registry.

Food: Kaggle API (requires ~/.kaggle/kaggle.json)
KuaiRec: Zenodo (public)
Yelp / Douban: manual or blocked
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.io import data_strict_root, ensure_dir, load_registry, sha256_file, write_json  # noqa: E402

KAGGLE_FOOD = "shuyangli94/food-com-recipes-and-user-interactions"
FOOD_NEEDED = ("RAW_interactions.csv", "RAW_recipes.csv")
KAGGLE_YELP = "yelp-dataset/yelp-dataset"
YELP_NEEDED = (
    "yelp_academic_dataset_review.json",
    "yelp_academic_dataset_business.json",
)


def kaggle_key_path() -> Path:
    return Path.home() / ".kaggle" / "kaggle.json"


def require_kaggle_key() -> Path:
    key = kaggle_key_path()
    if not key.exists():
        raise FileNotFoundError(
            "缺少 Kaggle 密钥。请按下列步骤放置后重试：\n"
            "1) 打开 https://www.kaggle.com/settings → API → Create New Token\n"
            "2) 将下载的 kaggle.json 放到 ~/.kaggle/kaggle.json\n"
            "3) chmod 600 ~/.kaggle/kaggle.json\n"
            f"期望路径: {key}"
        )
    os.chmod(key, 0o600)
    return key


def run_cmd(cmd, **kwargs):
    print("RUN:", " ".join(map(str, cmd)))
    subprocess.check_call(cmd, **kwargs)


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    ensure_dir(out_dir)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)


def stage_food(reg, root: Path) -> dict:
    """Download Food.com RAW_* via Kaggle CLI/API."""
    require_kaggle_key()
    cache = ensure_dir(root / "cache" / "food")
    raw = ensure_dir(root / "raw" / "food" / "kaggle_food_com")
    status = {
        "dataset": "food",
        "source": KAGGLE_FOOD,
        "license": reg["datasets"]["food"]["license"],
        "files": {},
    }

    # Prefer kaggle CLI; fall back to python -m kaggle
    def kaggle_download():
        try:
            run_cmd(
                [
                    "kaggle", "datasets", "download",
                    "-d", KAGGLE_FOOD,
                    "-p", str(cache),
                    "--unzip",
                ]
            )
            return
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass
        run_cmd(
            [
                sys.executable, "-m", "kaggle", "datasets", "download",
                "-d", KAGGLE_FOOD,
                "-p", str(cache),
                "--unzip",
            ]
        )

    missing = [n for n in FOOD_NEEDED if not (cache / n).exists() and not (raw / n).exists()]
    if missing:
        print("Downloading Food from Kaggle (may take several minutes)...")
        kaggle_download()

    for name in FOOD_NEEDED:
        src = cache / name if (cache / name).exists() else None
        if src is None:
            # sometimes nested
            hits = list(cache.rglob(name))
            src = hits[0] if hits else None
        if src is None or not src.exists():
            status["files"][name] = {"missing": True}
            continue
        dst = raw / name
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)
        status["files"][name] = {
            "path": str(dst),
            "sha256": sha256_file(dst),
            "bytes": dst.stat().st_size,
        }

    if any(v.get("missing") for v in status["files"].values()):
        status["status"] = "INCOMPLETE"
    else:
        status["status"] = "OK"
    write_json(raw / "source_manifest.json", status)
    print("[food]", status["status"], raw)
    return status


def stage_kuairec(reg, root: Path) -> dict:
    cache = root / "cache"
    raw = ensure_dir(root / "raw" / "kuairec" / "zenodo_18164998")
    # Prefer clean zip; never resume-append onto a corrupt file
    zip_path = cache / "KuaiRec_clean.zip"
    if not zip_path.exists() or zip_path.stat().st_size != 431964858:
        alt = cache / "KuaiRec.zip"
        if alt.exists() and alt.stat().st_size == 431964858:
            zip_path = alt
        else:
            dest = cache / "KuaiRec_clean.zip"
            run_cmd(["wget", "-O", str(dest), "https://zenodo.org/records/18164998/files/KuaiRec.zip"])
            zip_path = dest

    marker = None
    for p in raw.rglob("big_matrix.csv"):
        marker = p
        break
    if marker is None:
        print("Extracting", zip_path)
        extract_zip(zip_path, raw)

    status = {"dataset": "kuairec", "files": {}, "license": reg["datasets"]["kuairec"]["license"],
              "source_url": "https://zenodo.org/records/18164998"}
    for name in ["big_matrix.csv", "small_matrix.csv", "item_categories.csv", "kuairec_caption_category.csv"]:
        hits = list(raw.rglob(name))
        if hits:
            p = hits[0]
            status["files"][name] = {"path": str(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}
        else:
            status["files"][name] = {"missing": True}
    write_json(raw / "source_manifest.json", status)
    return status


def is_zip_file(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(2) == b"PK"
    except OSError:
        return False


def ensure_unzipped_named(path: Path, expected_name: str, out_dir: Path) -> Path:
    """If path is a zip (even when named .json), extract expected_name into out_dir."""
    out = out_dir / expected_name
    if out.exists() and out.stat().st_size > 1_000_000 and not is_zip_file(out):
        return out
    src = path
    if not is_zip_file(src):
        if src.name == expected_name:
            if not out.exists() or out.stat().st_size != src.stat().st_size:
                shutil.copy2(src, out)
            return out
        raise ValueError(f"Not a zip and not {expected_name}: {src}")
    # Kaggle often saves file.zip content as the original filename
    print(f"Extracting zip-as-json: {src} -> {out_dir}")
    with zipfile.ZipFile(src) as z:
        # extract matching member
        members = z.namelist()
        target = None
        for m in members:
            if Path(m).name == expected_name:
                target = m
                break
        if target is None and len(members) == 1:
            target = members[0]
        if target is None:
            raise FileNotFoundError(f"{expected_name} not in {src}: {members[:5]}")
        # extract to temp then move
        z.extract(target, out_dir)
        extracted = out_dir / target
        if extracted.resolve() != out.resolve():
            ensure_dir(out.parent)
            if out.exists():
                out.unlink()
            shutil.move(str(extracted), str(out))
            # cleanup nested dirs if any
            parent = extracted.parent
            if parent != out_dir and parent.exists():
                try:
                    parent.rmdir()
                except OSError:
                    pass
    return out


def stage_yelp(reg, root: Path) -> dict:
    """Download review+business JSON via Kaggle (rule-equivalent Open Dataset).

    Requires accepting the dataset terms once in the Kaggle UI:
    https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset
    """
    require_kaggle_key()
    cache = ensure_dir(root / "cache" / "yelp")
    raw = ensure_dir(root / "raw" / "yelp2018" / "kaggle_yelp_dataset")
    status = {
        "dataset": "yelp2018",
        "source": KAGGLE_YELP,
        "license": reg["datasets"]["yelp2018"]["license"],
        "files": {},
        "note": "rule-equivalent; not guaranteed identical to CausalDiffRec 2018 snapshot",
    }

    def kaggle_file(name: str):
        # Prefer CLI; fall back to python -m kaggle
        for prefix in (["kaggle"], [sys.executable, "-m", "kaggle"]):
            try:
                run_cmd(
                    prefix
                    + [
                        "datasets",
                        "download",
                        "-d",
                        KAGGLE_YELP,
                        "-f",
                        name,
                        "-p",
                        str(cache),
                    ]
                )
                return
            except (FileNotFoundError, subprocess.CalledProcessError):
                continue
        raise RuntimeError(f"kaggle download failed for {name}")

    for name in YELP_NEEDED:
        dst = raw / name
        # already real JSON (not zip)
        if dst.exists() and dst.stat().st_size > 1_000_000 and not is_zip_file(dst):
            status["files"][name] = {
                "path": str(dst),
                "sha256": sha256_file(dst),
                "bytes": dst.stat().st_size,
                "cached": True,
            }
            continue

        zipped = cache / f"{name}.zip"
        plain = cache / name
        if not plain.exists() and not zipped.exists():
            print(f"Downloading {name} from Kaggle (large; may take a while)...")
            kaggle_file(name)

        candidate = zipped if zipped.exists() else plain
        if candidate is None or not candidate.exists():
            hits = list(cache.rglob(name)) + list(cache.rglob(f"{name}.zip"))
            candidate = hits[0] if hits else None
        if candidate is None or not candidate.exists():
            status["files"][name] = {"missing": True}
            continue

        try:
            # extract into cache first, then copy real JSON to raw
            extracted = ensure_unzipped_named(candidate, name, cache)
            if not dst.exists() or is_zip_file(dst) or dst.stat().st_size != extracted.stat().st_size:
                if dst.exists():
                    dst.unlink()
                shutil.copy2(extracted, dst)
            status["files"][name] = {
                "path": str(dst),
                "sha256": sha256_file(dst),
                "bytes": dst.stat().st_size,
            }
        except Exception as e:
            status["files"][name] = {"missing": True, "error": str(e)}

    if any(v.get("missing") for v in status["files"].values()):
        status["status"] = "INCOMPLETE"
        pending = ensure_dir(root / "raw" / "yelp2018" / "pending")
        write_json(
            pending / "NEED_KAGGLE_OR_MANUAL.json",
            {
                "hint": "Accept terms at https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset "
                "or drop review+business JSON into raw/yelp2018/",
                "needed": list(YELP_NEEDED),
            },
        )
    else:
        status["status"] = "OK"
    write_json(raw / "source_manifest.json", status)
    print("[yelp2018]", status["status"], raw)
    return status


def report_manual(name: str, reg, root: Path) -> dict:
    raw = ensure_dir(root / "raw" / name / "pending")
    info = {
        "dataset": name,
        "status": "NEED_MANUAL_DOWNLOAD",
        "license": reg["datasets"][name].get("license"),
        "sources": reg["datasets"][name].get("sources", []),
        "blocked_reason": reg["datasets"][name].get("blocked_reason"),
        "drop_files_into": str(raw),
    }
    write_json(raw / "NEED_MANUAL.json", info)
    print(f"[{name}] manual/blocked — see {raw / 'NEED_MANUAL.json'}")
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", choices=["all", "food", "kuairec", "yelp2018", "douban"])
    args = ap.parse_args()
    reg = load_registry()
    root = data_strict_root(reg)
    ensure_dir(root)
    targets = ["food", "kuairec", "yelp2018", "douban"] if args.dataset == "all" else [args.dataset]
    results = {}
    for ds in targets:
        if ds == "food":
            try:
                results[ds] = stage_food(reg, root)
            except FileNotFoundError as e:
                results[ds] = {"dataset": "food", "status": "NEED_KAGGLE_KEY", "error": str(e)}
                print(e)
                write_json(root / "raw" / "food" / "pending" / "NEED_KAGGLE_KEY.json", results[ds])
        elif ds == "kuairec":
            results[ds] = stage_kuairec(reg, root)
        elif ds == "yelp2018":
            try:
                results[ds] = stage_yelp(reg, root)
            except Exception as e:
                results[ds] = {"dataset": "yelp2018", "status": "FAILED", "error": str(e)}
                print(e)
                write_json(root / "raw" / "yelp2018" / "pending" / "NEED_MANUAL.json", results[ds])
        elif ds == "douban":
            results[ds] = report_manual(ds, reg, root)
            results[ds]["status"] = "BLOCKED"
        else:
            results[ds] = report_manual(ds, reg, root)
    write_json(root / "cache" / "download_status.json", results)
    print("Wrote", root / "cache" / "download_status.json")


if __name__ == "__main__":
    main()
