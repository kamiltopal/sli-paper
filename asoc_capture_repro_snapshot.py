"""Read-only reproducibility snapshot of an ASOC source checkout.

Usage from project root (Windows PowerShell):
  python .\asoc_capture_repro_snapshot.py --repo .

Does not inspect tokens, Git remotes, environment variables, user account names,
credentials, home directories, or pip URLs. It hashes the scientific data/run files
but does NOT modify any experiment result. Writes one new JSON manifest; fails if
its output path already exists.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata as meta
import json
import platform
import subprocess
import sys
from pathlib import Path

PACKAGES = ("numpy", "scipy", "pandas", "matplotlib", "torch", "einops", "tqdm", "scikit-learn")


def call_git(repo: Path, *args: str):
    try:
        result = subprocess.run(["git", "-C", str(repo), *args],
                                text=True, capture_output=True, timeout=10, check=True)
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def files_to_hash(repo: Path):
    paths = [repo / "data" / "real" / "epias" / "ptf.csv"]
    paths.extend(sorted((repo / "data" / "real").glob("*.npy")))
    paths.extend(sorted((repo / "results" / "asoc_revision").rglob("*.csv")))
    paths.extend(sorted((repo / "results" / "asoc_revision").rglob("*.json")))
    return [p for p in paths if p.is_file()]


def capture(repo: Path):
    repo = repo.resolve()
    snapshot = {
        "audit_version": "asoc_repro_snapshot_v1",
        "captured_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repository": "kamiltopal/sli-paper",
        "git": {
            "branch": call_git(repo, "branch", "--show-current"),
            "commit": call_git(repo, "rev-parse", "HEAD"),
            "dirty": None,
        },
        "environment": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "os": platform.system(),
            "os_release": platform.release(),
            "package_versions": {},
            "torch_cuda": None,
            "cudnn_version": None,
            "cuda_available": None,
            "gpus": [],
        },
        "sha256": {},
    }
    status = call_git(repo, "status", "--porcelain")
    if status is not None:
        snapshot["git"]["dirty"] = bool(status)
    for p in PACKAGES:
        try:
            snapshot["environment"]["package_versions"][p] = meta.version(p)
        except meta.PackageNotFoundError:
            snapshot["environment"]["package_versions"][p] = None
    try:
        import torch
        snapshot["environment"]["torch_cuda"] = torch.version.cuda
        snapshot["environment"]["cudnn_version"] = torch.backends.cudnn.version()
        snapshot["environment"]["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                snapshot["environment"]["gpus"].append({
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "compute_capability": list(torch.cuda.get_device_capability(i)),
                })
    except (ImportError, RuntimeError) as exc:
        snapshot["environment"]["torch_probe_warning"] = type(exc).__name__

    for path in files_to_hash(repo):
        rel = path.relative_to(repo).as_posix()
        snapshot["sha256"][rel] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return snapshot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path("."))
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    repo = args.repo.resolve()
    if not (repo / "src" / "data.py").is_file():
        p.error("--repo must point to the SLI project root, containing src/data.py")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.out or repo / "results" / "asoc_revision" / "reproducibility" / f"manifest_{stamp}.json"
    if not output.is_absolute():
        output = repo / output
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = capture(repo)
    # Exclusive creation; existing files never overwritten.
    with output.open("x", encoding="utf-8") as fh:
        json.dump(snapshot, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print("Snapshot:", output)
    print("Commit:", snapshot["git"]["commit"])
    print("Dirty:", snapshot["git"]["dirty"])
    print("Files hashed:", len(snapshot["sha256"]))
    print("CUDA:", snapshot["environment"]["torch_cuda"], snapshot["environment"]["gpus"])


if __name__ == "__main__":
    main()
