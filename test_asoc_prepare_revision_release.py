import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from asoc_prepare_revision_release import build, sha256


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / "repo"
        (self.repo / "src").mkdir(parents=True)
        (self.repo / "src" / "data.py").write_text("# data\n")
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        subprocess.run(["git", "-C", str(self.repo), "add", "src/data.py"], check=True)
        subprocess.run(["git", "-C", str(self.repo), "-c", "user.name=Test", "-c", "user.email=test@example.test", "commit", "-qm", "initial"], check=True)
        self.outbase = self.repo / "results" / "asoc_revision"
        (self.outbase / "nonfourier").mkdir(parents=True)
        (self.outbase / "real_budget_controls.csv").write_text("condition,test_mse\nvae_augmented,1.0\n")
        (self.outbase / "nonfourier" / "nonfourier_runs.csv").write_text("condition,test_mse\nvae_augmented,0.5\n")
        (self.outbase / "reproducibility").mkdir()
        (self.outbase / "reproducibility" / "manifest_old.json").write_text("{}")
        self.snap = self.root / "snapshot.json"
        commit = subprocess.check_output(["git", "-C", str(self.repo), "rev-parse", "HEAD"], text=True).strip()
        self.obj = {"audit_version": "asoc_repro_snapshot_v1", "git": {"commit": commit, "dirty": True},
                    "environment": {"package_versions": {"numpy": "1.26.1"}, "gpus": [{"name": "test"}]},
                    "sha256": {}, "captured_utc": "2026-11-11T01:00:00Z"}
        for p in self.outbase.rglob("*.csv"):
            rel = p.relative_to(self.repo).as_posix()
            self.obj["sha256"][rel] = {"sha256": sha256(p), "bytes": p.stat().st_size}
        self.snap.write_text(json.dumps(self.obj))
        self.out = self.root / "release.zip"

    def test_build_does_not_change_input_and_filters_snapshot_file(self):
        before = sha256(self.outbase / "real_budget_controls.csv")
        result = build(self.repo, self.snap, self.out)
        self.assertEqual(result["file_count"], 2)
        self.assertEqual(result["status"], "provisional_not_publication_approved")
        self.assertEqual(sha256(self.outbase / "real_budget_controls.csv"), before)
        with ZipFile(self.out) as z:
            self.assertIn("release_manifest.json", z.namelist())
            self.assertNotIn("results/asoc_revision/reproducibility/manifest_old.json", z.namelist())

    def test_mismatch_detected(self):
        (self.outbase / "real_budget_controls.csv").write_text("x\n1\n")
        with self.assertRaisesRegex(ValueError, "Snapshot hash mismatch"):
            build(self.repo, self.snap, self.out)

    def test_no_silent_overwrite(self):
        self.out.write_text("existing")
        with self.assertRaises(FileExistsError):
            build(self.repo, self.snap, self.out)

    def test_missing_major_results_fails(self):
        (self.outbase / "nonfourier" / "nonfourier_runs.csv").unlink()
        with self.assertRaisesRegex(ValueError, "Missing key result"):
            build(self.repo, self.snap, self.out)

    def test_publish_strict_refuses_dirty(self):
        with self.assertRaisesRegex(ValueError, "dirty"):
            build(self.repo, self.snap, self.out, require_clean=True)

    def test_no_output_inside_repo(self):
        with self.assertRaisesRegex(ValueError, "OUTSIDE"):
            build(self.repo, self.snap, self.outbase / "release.zip")

    def test_new_result_without_hash_flagged(self):
        (self.outbase / "new.csv").write_text("x\n2\n")
        with self.assertRaisesRegex(ValueError, "Uncovered result"):
            build(self.repo, self.snap, self.out, require_snapshot_coverage=True)

    def test_known_headerless_empty_incomplete_report_is_excluded_and_audited(self):
        p = self.outbase / "nonfourier" / "report" / "nonfourier_incomplete.csv"
        p.parent.mkdir(parents=True)
        p.write_text("\n", encoding="utf-8")  # pandas empty DataFrame with no columns
        self.obj["sha256"][p.relative_to(self.repo).as_posix()] = {
            "sha256": sha256(p), "bytes": p.stat().st_size}
        self.snap.write_text(json.dumps(self.obj))
        result = build(self.repo, self.snap, self.out, require_snapshot_coverage=True)
        self.assertEqual(result["file_count"], 2)
        self.assertEqual(len(result["excluded_empty_report_placeholders"]), 1)
        self.assertEqual(result["excluded_empty_report_placeholders"][0]["sha256"], sha256(p))
        with ZipFile(self.out) as z:
            self.assertNotIn(p.relative_to(self.repo).as_posix(), z.namelist())
            self.assertIn("excluded_empty_report_placeholders", json.loads(z.read("release_manifest.json")))

    def test_bad_header_in_actual_result_fails_with_filename(self):
        p = self.outbase / "real_budget_controls.csv"
        p.write_text("condition,,condition\nvae,1,vae\n")
        self.obj["sha256"][p.relative_to(self.repo).as_posix()] = {
            "sha256": sha256(p), "bytes": p.stat().st_size}
        self.snap.write_text(json.dumps(self.obj))
        with self.assertRaisesRegex(ValueError, "real_budget_controls.csv"):
            build(self.repo, self.snap, self.out, require_snapshot_coverage=True)
        self.assertFalse(self.out.exists())

    def test_unrecognized_blank_header_must_not_be_ignored(self):
        p = self.outbase / "some_important_results.csv"
        p.write_text("\n")
        self.obj["sha256"][p.relative_to(self.repo).as_posix()] = {
            "sha256": sha256(p), "bytes": p.stat().st_size}
        self.snap.write_text(json.dumps(self.obj))
        with self.assertRaisesRegex(ValueError, "some_important_results.csv"):
            build(self.repo, self.snap, self.out, require_snapshot_coverage=True)

    def test_empty_report_hash_mismatch_still_fails(self):
        p = self.outbase / "real_incomplete.csv"
        p.write_text("\n")
        self.obj["sha256"][p.relative_to(self.repo).as_posix()] = {
            "sha256": sha256(p), "bytes": p.stat().st_size}
        self.snap.write_text(json.dumps(self.obj))
        p.write_text("\n\n")
        with self.assertRaisesRegex(ValueError, "Snapshot hash mismatch"):
            build(self.repo, self.snap, self.out, require_snapshot_coverage=True)

if __name__ == "__main__":
    unittest.main()
