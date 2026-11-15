"""Prepare a *provisional* ASOC major-revision results archive without retraining.

Read-only with respect to source checkout. No raw datasets, model weights,
full local paths, environment variables, git remote URLs or private data.
Default public archive is only a candidate: review before publication.

Usage (PowerShell, from repo root):
  python asoc_prepare_revision_release.py --repo .
      --snapshot results/asoc_revision/reproducibility/manifest_20261111T112524Z.json
      --out ASOC_revision_results_candidate.zip (outside repo)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from fnmatch import fnmatch
from zipfile import ZipFile, ZIP_DEFLATED

ALLOWED = {".csv", ".json"}
EXCLUDED_DIRS = {"reproducibility", "pools", "checkpoints", "models", "__pycache__"}
REQUIRED = {"real_budget_controls.csv", "nonfourier/nonfourier_runs.csv"}
# Some reporting modules can emit a headerless, whitespace-only CSV for an
# empty diagnostics / missing / failure list. These are *not data tables*.
# Record their hashes and exclusion in the release manifest; never rewrite them.
EMPTY_REPORT_PATTERNS = ("*_incomplete.csv", "*_missing.csv", "*_errors.csv",
                         "*_failed.csv", "*_skipped.csv")


class EmptyReportPlaceholder(ValueError):
    """A structurally empty CSV with no field names and no observations."""


def is_known_empty_report(p: Path) -> bool:
    if not any(fnmatch(p.name.lower(), pattern) for pattern in EMPTY_REPORT_PATTERNS):
        return False
    # Only a file with no actual text or CSV tokens is safe to exclude.
    try:
        return not p.read_text(encoding="utf-8-sig").strip()
    except UnicodeError:
        return False


def sha256(p: Path) -> str:
    digest = hashlib.sha256()
    with p.open("rb") as source:
        for data in iter(lambda: source.read(1 << 20), b""):
            digest.update(data)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    try:
        run = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                             text=True, timeout=12, check=True)
        return run.stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"Cannot read git metadata: {type(exc).__name__}") from exc


def files_to_package(repo: Path, max_size: int):
    base = repo / "results" / "asoc_revision"
    if not base.is_dir():
        raise ValueError("Missing results/asoc_revision; no experiments will be run")
    for p in sorted(base.rglob("*")):
        if p.is_symlink():
            raise ValueError("Symlink in results is not permitted")
        if not p.is_file():
            continue
        rel = p.relative_to(base)
        if p.suffix.lower() not in ALLOWED or any(x.lower() in EXCLUDED_DIRS for x in rel.parts[:-1]):
            continue
        if p.stat().st_size > max_size:
            raise ValueError(f"File too large; review manually: {rel.as_posix()}")
        if not p.stat().st_size:
            raise ValueError(f"Empty result: {rel.as_posix()}")
        yield p, rel.as_posix()


def inspect_csv(p: Path) -> dict:
    """Validate a real table, reporting exact path and problem if malformed.

    Do not silently reinterpret a malformed CSV as a result or repair it.
    """
    try:
        with p.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            hdr = next(reader, None)
            if hdr is None or not hdr or all(not col.strip() for col in hdr):
                raise EmptyReportPlaceholder(
                    f"Headerless CSV at {p}: empty report or malformed file")
            blank = [i + 1 for i, col in enumerate(hdr) if not col.strip()]
            dupes = sorted({c for c in hdr if hdr.count(c) > 1})
            if blank or dupes:
                raise ValueError(
                    f"Invalid CSV header at {p}: "
                    f"blank column position(s)={blank}, duplicate names={dupes}. "
                    "No files were changed; inspect the report generator.")
            count = sum(1 for row in reader if row)
        return {"rows": count, "columns": hdr}
    except (UnicodeError, csv.Error) as exc:
        raise ValueError(f"Invalid UTF-8 CSV at {p}: {exc}") from exc


def build(repo: Path, snapshot_path: Path, output: Path, require_clean: bool = False,
          require_snapshot_coverage: bool = False, max_file_mb: int = 25) -> dict:
    repo, snapshot_path, output = repo.resolve(), snapshot_path.resolve(), output.resolve()
    if not (repo / "src" / "data.py").is_file():
        raise ValueError("--repo must be the SLI repository root")
    if not snapshot_path.is_file():
        raise ValueError("Snapshot JSON not found; run capture script first")
    if output.exists():
        raise FileExistsError("Output ZIP exists: refusing overwrite")
    if output.is_relative_to(repo):
        raise ValueError("Write release ZIP OUTSIDE the repository to avoid dirtying checkout")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if snapshot.get("audit_version") != "asoc_repro_snapshot_v1" or not isinstance(snapshot.get("sha256"), dict):
        raise ValueError("Unexpected snapshot format")
    commit = git(repo, "rev-parse", "HEAD")
    branch = git(repo, "branch", "--show-current")
    dirty = bool(git(repo, "status", "--porcelain"))
    if snapshot.get("git", {}).get("commit") != commit:
        raise ValueError("Commit changed since snapshot; recapture manifest")
    if require_clean and (dirty or snapshot.get("git", {}).get("dirty")):
        raise ValueError("Working tree is dirty. Resolve changes and recapture snapshot")
    found = list(files_to_package(repo, max_file_mb * (1 << 20)))
    names = {rel for _, rel in found}
    missing = sorted(REQUIRED - names)
    if missing:
        raise ValueError("Missing key result files: " + ", ".join(missing))
    entries = []
    not_in_snapshot = []
    skipped_empty_reports = []
    selected_files = []
    for p, rel in found:
        source_rel = (Path("results") / "asoc_revision" / rel).as_posix()
        file_sha = sha256(p)
        old = snapshot["sha256"].get(source_rel)
        if old is None:
            not_in_snapshot.append(source_rel)
        elif file_sha != old.get("sha256") or p.stat().st_size != old.get("bytes"):
            raise ValueError(f"Snapshot hash mismatch for {source_rel}; recapture snapshot")
        entry = {"file": source_rel, "bytes": p.stat().st_size, "sha256": file_sha,
                 "covered_by_snapshot": old is not None}
        if p.suffix.lower() == ".csv":
            try:
                entry.update(inspect_csv(p))
            except EmptyReportPlaceholder:
                if source_rel not in {str(Path("results") / "asoc_revision" / key)
                                      for key in REQUIRED} and is_known_empty_report(p):
                    skipped_empty_reports.append({
                        **entry, "reason": "Headerless zero-column empty report; no table to publish"})
                    continue
                raise ValueError(
                    f"Headerless CSV at {source_rel}: expected a table, not an empty "
                    "placeholder. Original file unchanged; inspect the producer.") from None
        entries.append(entry)
        selected_files.append((p, rel))
    if require_snapshot_coverage and not_in_snapshot:
        raise ValueError("Uncovered result files, recapture snapshot: " + ", ".join(not_in_snapshot[:5]))
    reasons = []
    if dirty or snapshot["git"].get("dirty"):
        reasons.append("dirty working tree: commit alone is not an exact execution state")
    if not_in_snapshot:
        reasons.append("some result files were not included in the source snapshot")
    if skipped_empty_reports:
        reasons.append("empty headerless report placeholders excluded with SHA256 evidence")
    if not snapshot.get("environment", {}).get("package_versions"):
        reasons.append("source snapshot does not contain package versions")
    manifest = {
        "schema": "asoc_revision_release_candidate_v1",
        "status": "provisional_not_publication_approved" if reasons else "candidate_for_manual_review",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"repository": "kamiltopal/sli-paper", "branch": branch,
                   "commit": commit, "dirty_now": dirty,
                   "dirty_at_snapshot": snapshot["git"].get("dirty")},
        "source_snapshot_captured_utc": snapshot.get("captured_utc"),
        "environment": snapshot["environment"],
        "safety": {"contains_raw_datasets": False,
                   "contains_models_or_synthetic_pools": False,
                   "contains_only_result_csv_json": True,
                   "manual_publication_review_required": True},
        "limitations": [
            "Original VAE training pools were not persisted; Phase 13 reconstructions are diagnostic, not original pools",
            "Seed/pool repetitions are not independent test periods or independent domains",
            "Optimizer-step equality does not imply equal generator+forecaster compute",
            "Source experiment scripts are referenced by commit but not copied into this archive",
        ] + reasons,
        "file_count": len(entries), "files": entries,
        "excluded_empty_report_placeholders": skipped_empty_reports,
    }
    readme = ("ASOC major revision: PROVISIONAL RESULTS-ONLY ARCHIVE\n\n"
              "Inspect and approve this archive before publication. No Git changes made.\n"
              "The archive contains only CSV/JSON results under results/asoc_revision.\n"
              f"Code: https://github.com/kamiltopal/sli-paper/tree/{commit}\n"
              "Review release_manifest.json for SHA256, row counts, versions, provenance, and caveats.\n"
              "Original raw datasets, checkpoints, source VAE pools, and raw personal\n"
              "environment metadata are not shipped. Interpret Phase 13 reconstructed\n"
              "fidelity as exploratory diagnostics.\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with ZipFile(output, "x", compression=ZIP_DEFLATED) as archive:
            archive.writestr("README.txt", readme)
            archive.writestr("release_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
            for p, rel in selected_files:
                archive.write(p, arcname=(Path("results") / "asoc_revision" / rel).as_posix())
        # Read back and verify checksums of archive contents.
        with ZipFile(output) as archive:
            for e in entries:
                if hashlib.sha256(archive.read(e["file"])).hexdigest() != e["sha256"]:
                    raise ValueError("Post-ZIP SHA256 verification failed")
    except Exception:
        output.unlink(missing_ok=True)
        raise
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path("."))
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--require-clean", action="store_true")
    p.add_argument("--require-snapshot-coverage", action="store_true")
    p.add_argument("--max-file-mb", type=int, default=25)
    args = p.parse_args()
    snapshot = args.snapshot if args.snapshot.is_absolute() else args.repo / args.snapshot
    try:
        if args.max_file_mb <= 0:
            raise ValueError("max-file-mb must be positive")
        result = build(args.repo, snapshot, args.out, args.require_clean,
                       args.require_snapshot_coverage, args.max_file_mb)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        p.error(str(exc))
    print("Created candidate release ZIP:", args.out)
    print("Files:", result["file_count"], "Status:", result["status"])
    for excluded in result["excluded_empty_report_placeholders"]:
        print("EXCLUDED EMPTY REPORT:", excluded["file"])
    for reason in result["limitations"]:
        print("NOTE:", reason)


if __name__ == "__main__":
    main()
