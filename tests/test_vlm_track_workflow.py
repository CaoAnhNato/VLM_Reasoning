# -*- coding: utf-8 -*-
"""
Comprehensive Unit & Integration Suite for VLM Track Research Workflow
(tests/test_vlm_track_workflow.py).
"""

import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.audit import run_auditor_checks
from scripts.benchmark import colab_drive_sync
from scripts.benchmark.evaluate_vitextvqa import _parse_ablation_bits, _resolve_nevfe_config
from scripts.benchmark.export_calibration_bank import (
    export_synthetic_calibration_bank_v2,
    main as cal_bank_main,
    run_calibration_eval_v2,
    validate_cal_bank_v2,
)
from scripts.benchmark.run_tier1_probe import (
    _write_ablation_summary,
    evaluate_loo_gate,
    load_baseline_probe_scores,
)
from scripts.workflows import track_ops
from src.nevfe.core.engine import AlgorithmConfig, NEVFEEngine


def _lf_sha_file(path: str) -> str:
    raw = open(path, "rb").read().replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest().lower()


def _build_workflow_test_repo(tmp: str, with_bare_remote: bool = False) -> dict:
    repo = os.path.join(tmp, "repo")
    for d in (
        "configs/tracks",
        "src/nevfe/core",
        "src/nevfe/adapter",
        "docs/specs/track_1",
        "docs/workflows",
        "scripts/benchmark",
        "tests",
        "reports/track_1",
    ):
        os.makedirs(os.path.join(repo, d), exist_ok=True)

    shutil.copy2(os.path.join(PROJECT_ROOT, "configs", "tracks", "workflow_defaults.json"), os.path.join(repo, "configs", "tracks", "workflow_defaults.json"))
    shutil.copy2(os.path.join(PROJECT_ROOT, "configs", "tracks", "track_1.json"), os.path.join(repo, "configs", "tracks", "track_1.json"))
    shutil.copy2(os.path.join(PROJECT_ROOT, "configs", "tracks", "track_2.json"), os.path.join(repo, "configs", "tracks", "track_2.json"))
    shutil.copy2(os.path.join(PROJECT_ROOT, "configs", "frsa011_baseline_summary.json"), os.path.join(repo, "configs", "frsa011_baseline_summary.json"))
    shutil.copy2(os.path.join(PROJECT_ROOT, "configs", "probe_100_manifest.json"), os.path.join(repo, "configs", "probe_100_manifest.json"))
    shutil.copy2(os.path.join(PROJECT_ROOT, "scripts", "benchmark", "metrics.py"), os.path.join(repo, "scripts", "benchmark", "metrics.py"))
    shutil.copy2(os.path.join(PROJECT_ROOT, "tests", "test_metrics_relaxation.py"), os.path.join(repo, "tests", "test_metrics_relaxation.py"))
    shutil.copy2(os.path.join(PROJECT_ROOT, "src", "nevfe", "core", "frozen_foveation.py"), os.path.join(repo, "src", "nevfe", "core", "frozen_foveation.py"))
    shutil.copy2(os.path.join(PROJECT_ROOT, "src", "nevfe", "core", "engine.py"), os.path.join(repo, "src", "nevfe", "core", "engine.py"))
    shutil.copy2(os.path.join(PROJECT_ROOT, "src", "nevfe", "adapter", "vintern_adapter.py"), os.path.join(repo, "src", "nevfe", "adapter", "vintern_adapter.py"))
    shutil.copy2(os.path.join(PROJECT_ROOT, "src", "nevfe", "pipeline.py"), os.path.join(repo, "src", "nevfe", "pipeline.py"))
    open(os.path.join(repo, ".gitignore"), "w", encoding="utf-8", newline="\n").write(".workflow_state/\nresults/track_runs/\ndata/*\n")

    bank_dir = os.path.join(repo, "data", "calibration_bank_v2")
    export_synthetic_calibration_bank_v2(
        os.path.join(repo, "configs", "probe_100_manifest.json"),
        bank_dir,
        max_steps=4,
        vocab_size=128,
        hidden_size=16,
        num_hidden_layers=4,
        base_commit_sha="base_sha",
    )

    def git(*args):
        subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True)

    git("init")
    git("config", "user.email", "tester@example.com")
    git("config", "user.name", "Tester")
    git("add", "-A")
    git("commit", "-m", "base")
    git("branch", "base/frsa-011-frozen")
    git("checkout", "-b", "research/track-1")
    base_sha = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()

    bare = None
    if with_bare_remote:
        bare = os.path.join(tmp, "remote.git")
        subprocess.run(["git", "init", "--bare", bare], check=True, capture_output=True)
        git("remote", "add", "origin", bare)

    return {
        "repo": repo,
        "bank_dir": bank_dir,
        "base_sha": base_sha,
        "bare": bare,
        "lock_path": os.path.join(tmp, "colab_l4.mutex"),
        "active_track": os.path.join(tmp, "colab_active_track.json"),
    }


class _FakeDriveFiles:
    def __init__(self):
        self.items = {}
        self.next_id = 1
        self.updates = []
        self.creates = []

    def _new_id(self):
        fid = f"fid_{self.next_id}"
        self.next_id += 1
        return fid

    def list(self, q="", fields="", pageSize=100):
        del fields, pageSize
        matched = []
        if "mimeType='application/vnd.google-apps.folder'" in q:
            name = q.split("name='", 1)[1].split("'", 1)[0]
            parent = q.split(" and '")[-1].split("' in parents")[0] if "' in parents" in q else None
            for it in self.items.values():
                if it["mimeType"] == "application/vnd.google-apps.folder" and it["name"] == name and it.get("parent") == parent:
                    matched.append(it)
        elif "' in parents" in q:
            parent = q.split("'", 2)[1]
            for it in self.items.values():
                if it.get("parent") == parent:
                    matched.append(it)
        return SimpleNamespace(execute=lambda num_retries=3: {"files": list(matched)})

    def create(self, body=None, media_body=None, fields=""):
        del fields
        fid = self._new_id()
        parent = (body.get("parents") or [None])[0]
        rec = {
            "id": fid,
            "name": body["name"],
            "mimeType": body.get("mimeType", "application/octet-stream"),
            "parent": parent,
            "size": os.path.getsize(media_body.filename()) if hasattr(media_body, "filename") else 10,
        }
        self.items[fid] = rec
        self.creates.append(body["name"])
        return SimpleNamespace(execute=lambda num_retries=3: rec)

    def update(self, fileId="", media_body=None, fields=""):
        del media_body, fields
        self.updates.append((fileId, self.items[fileId]["name"]))
        return SimpleNamespace(execute=lambda num_retries=3: self.items[fileId])


class TestVLMTrackWorkflow(unittest.TestCase):
    def test_01_track_ops_preflight_resume_conflict_and_post_boost(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_ctx = _build_workflow_test_repo(tmp)
            repo = env_ctx["repo"]
            lock_p = env_ctx["lock_path"]
            act_p = env_ctx["active_track"]
            bank_dir = env_ctx["bank_dir"]

            old_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
            old_enc = os.environ.get("PYTHONIOENCODING")
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.environ["PYTHONIOENCODING"] = "utf-8"
            try:
                # 1. Clean preflight succeeds (also test CLI main entrypoint)
                rc = track_ops.main([
                    "preflight", "--track", "1", "--mode", "preflight",
                    "--repo-root", repo, "--cal-bank-dir", bank_dir,
                    "--lock-path", lock_p, "--active-track-file", act_p,
                ])
                self.assertEqual(rc, 0)
                self.assertEqual(track_ops.main(["state", "init-or-load", "--track", "2", "--repo-root", repo, "--lock-path", lock_p]), 0)
                self.assertEqual(track_ops.main(["state", "status", "--track", "1", "--repo-root", repo, "--lock-path", lock_p]), 0)

                # Bad track config (>5 keys) triggers error remediation
                t1_cfg_p = os.path.join(repo, "configs", "tracks", "track_1.json")
                orig_t1 = open(t1_cfg_p, "r", encoding="utf-8").read()
                bad_t1 = json.loads(orig_t1)
                bad_t1["extra_key"] = True
                open(t1_cfg_p, "w", encoding="utf-8").write(json.dumps(bad_t1))
                bad_pre = track_ops.run_preflight(1, mode="run", repo_root=repo, cal_bank_dir=bank_dir, lock_path=lock_p, active_track_path=act_p)
                self.assertFalse(bad_pre["ready"])
                open(t1_cfg_p, "w", encoding="utf-8", newline="\n").write(orig_t1)

                # 2. Active wave crash auto-resume vs cross-track conflict vs expired TTL
                rc_sw = track_ops.main([
                    "state", "set-active-wave", "--track", "1", "--wave-id", "w_1",
                    "--run-folder-id", "fid_1", "--added-cell-ids-json", '["c1"]',
                    "--trace-id", "track_1_iter_1_attempt_1", "--repo-root", repo,
                    "--lock-path", lock_p, "--active-track-file", act_p,
                ])
                self.assertEqual(rc_sw, 0)
                res_poll = track_ops.run_preflight(1, mode="run", repo_root=repo, cal_bank_dir=bank_dir, lock_path=lock_p, active_track_path=act_p)
                self.assertTrue(res_poll["ready"], res_poll)
                self.assertEqual(res_poll["resume_action"], "COLAB_POLLING")

                res_conf = track_ops.run_preflight(2, mode="run", repo_root=repo, cal_bank_dir=bank_dir, lock_path=lock_p, active_track_path=act_p)
                self.assertFalse(res_conf["ready"])
                self.assertEqual(res_conf["resume_action"], "CROSS_TRACK_CONFLICT")

                # Expire wave ttl_s -> ORPHAN_EXPIRED
                track_ops.set_active_wave(
                    1, "w_old", "fid_1", ["c1"], "t1",
                    launched_at=time.time() - 7000, ttl_s=6000,
                    repo_root=repo, lock_path=lock_p, active_track_path=act_p,
                )
                res_exp = track_ops.run_preflight(1, mode="run", repo_root=repo, cal_bank_dir=bank_dir, lock_path=lock_p, active_track_path=act_p)
                self.assertFalse(res_exp["ready"])
                self.assertEqual(res_exp["resume_action"], "ORPHAN_EXPIRED")

                self.assertEqual(
                    track_ops.main(["state", "clear-active-wave", "--track", "1", "--repo-root", repo, "--lock-path", lock_p, "--active-track-file", act_p]),
                    0,
                )

                # 3. HALTED_FOR_BOOST refuses mode="run" (exit 2) and accepts valid mode="post_boost" once
                track_ops.transition_state(
                    1, "HALTED_FOR_BOOST", repo_root=repo, lock_path=lock_p,
                    updates={"iteration": 2, "consecutive_no_gain": 2, "halted_head_sha": env_ctx["base_sha"]},
                )
                rc_refuse = track_ops.main([
                    "preflight", "--track", "1", "--mode", "run",
                    "--repo-root", repo, "--cal-bank-dir", bank_dir,
                    "--lock-path", lock_p, "--active-track-file", act_p,
                ])
                self.assertEqual(rc_refuse, 2)

                notes_p = os.path.join(repo, "reports", "track_1", "iter_2", "BOOST_RESOLUTION_iter_2.md")
                os.makedirs(os.path.dirname(notes_p), exist_ok=True)
                open(notes_p, "w", encoding="utf-8", newline="\n").write("Missing required headers\n")
                res_bad_pb = track_ops.run_preflight(1, mode="post_boost", boost_notes_path=notes_p, repo_root=repo, cal_bank_dir=bank_dir, lock_path=lock_p, active_track_path=act_p)
                self.assertFalse(res_bad_pb["ready"])

                open(notes_p, "w", encoding="utf-8", newline="\n").write("## Diagnosis\nOver-pruned.\n## Next Direction\nUse soft threshold.\n")
                subprocess.run(["git", "-C", repo, "add", "-A"], check=True, capture_output=True)
                subprocess.run(["git", "-C", repo, "commit", "-m", "boost notes"], check=True, capture_output=True)
                rc_pb = track_ops.main([
                    "preflight", "--track", "1", "--mode", "post_boost", "--boost-notes-path", notes_p,
                    "--repo-root", repo, "--cal-bank-dir", bank_dir,
                    "--lock-path", lock_p, "--active-track-file", act_p,
                ])
                self.assertEqual(rc_pb, 0)
                st_after = track_ops.init_or_load_state(1, repo_root=repo, lock_path=lock_p)
                self.assertEqual(st_after["consecutive_no_gain"], 0)
                self.assertEqual(st_after["fsm_state"], "PREFLIGHT")

                # Replaying same boost notes after another halt fails
                track_ops.transition_state(1, "HALTED_FOR_BOOST", repo_root=repo, lock_path=lock_p, updates={"consecutive_no_gain": 2})
                with self.assertRaises(RuntimeError):
                    track_ops.verify_post_boost(1, boost_notes_path=notes_p, repo_root=repo, lock_path=lock_p)

                # Post-halt commit path works without boost_notes_path
                open(os.path.join(repo, "reports", "track_1", "manual_note.md"), "w", encoding="utf-8").write("note\n")
                subprocess.run(["git", "-C", repo, "add", "-A"], check=True, capture_output=True)
                subprocess.run(["git", "-C", repo, "commit", "-m", "post-halt guidance"], check=True, capture_output=True)
                self.assertEqual(track_ops.main(["state", "verify-post-boost", "--track", "1", "--repo-root", repo, "--lock-path", lock_p]), 0)

                # 4. CU accounting & budget exhaustion check
                track_ops.main([
                    "state", "record-cu", "--track", "1", "--trace-id", "t1", "--iter", "3",
                    "--connect-ts", "1000", "--unassign-ts", "41000", "--unassign-confirmed",
                    "--repo-root", repo, "--lock-path", lock_p,
                ])
                pre_budget = track_ops.run_preflight(1, mode="run", repo_root=repo, cal_bank_dir=bank_dir, lock_path=lock_p, active_track_path=act_p)
                self.assertFalse(pre_budget["ready"])
                self.assertEqual(pre_budget["resume_action"], "BUDGET_EXHAUSTED")

                # 5. Epistemic ledger compaction (< 3 KB) + boost handoff + final report
                for k in range(1, 14):
                    track_ops.main([
                        "state", "append-epistemic", "--track", "1", "--iter", str(k),
                        "--outcome", "NO_GAIN", "--hypothesis", f"Hyp {k} " + ("a" * 180),
                        "--dead-end-rule", f"Rule {k}: avoid pattern " + ("r" * 90),
                        "--detail", f"Detail {k}: " + ("d" * 240),
                        "--repo-root", repo, "--lock-path", lock_p,
                    ])
                pre_out_p = os.path.join(tmp, "preamble.md")
                self.assertEqual(track_ops.main(["state", "render-epistemic-preamble", "--track", "1", "--repo-root", repo, "--out-path", pre_out_p]), 0)
                self.assertLess(os.path.getsize(pre_out_p), 3072)

                self.assertEqual(track_ops.main(["report", "boost-handoff", "--track", "1", "--iter", "3", "--repo-root", repo]), 0)
                self.assertTrue(os.path.isfile(os.path.join(repo, "reports", "track_1", "iter_3", "BOOST_HANDOFF_iter_3.md")))
                self.assertEqual(track_ops.main(["report", "final-report", "--track", "1", "--repo-root", repo]), 0)
                self.assertTrue(os.path.isfile(os.path.join(repo, "reports", "track_1", "TRACK_1_FINAL_REPORT.md")))
            finally:
                if old_cuda is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda
                if old_enc is None:
                    os.environ.pop("PYTHONIOENCODING", None)
                else:
                    os.environ["PYTHONIOENCODING"] = old_enc

    def test_02_track_ops_git_scope_staging_rollback_and_papers_verify(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_ctx = _build_workflow_test_repo(tmp, with_bare_remote=True)
            repo = env_ctx["repo"]
            base_sha = env_ctx["base_sha"]

            # Commit spec
            spec_p = os.path.join(repo, "docs", "specs", "track_1", "spec_iter_1.md")
            open(spec_p, "w", encoding="utf-8", newline="\n").write("# Spec iter 1\n")
            self.assertEqual(track_ops.main(["git", "commit-spec", "--track", "1", "--iter", "1", "--spec-path", spec_p, "--repo-root", repo]), 0)
            spec_sha = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()

            # Out-of-scope file fails verify-scope
            bad_untracked = os.path.join(repo, "src", "nevfe", "core", "rogue.py")
            open(bad_untracked, "w", encoding="utf-8").write("x = 1\n")
            self.assertEqual(track_ops.main(["git", "verify-scope", "--track", "1", "--base-sha", spec_sha, "--repo-root", repo]), 1)
            os.remove(bad_untracked)

            # In-scope edit + candidate test commits cleanly and pushes/prunes staging ref on bare remote
            open(os.path.join(repo, "src", "nevfe", "core", "engine.py"), "a", encoding="utf-8", newline="\n").write("\n# track1 edit\n")
            open(os.path.join(repo, "tests", "test_track_candidate.py"), "w", encoding="utf-8", newline="\n").write("def test_c():\n    assert True\n")
            self.assertEqual(track_ops.main(["git", "verify-scope", "--track", "1", "--base-sha", spec_sha, "--repo-root", repo]), 0)
            self.assertEqual(track_ops.main(["git", "commit-impl", "--track", "1", "--iter", "1", "--spec-sha", spec_sha, "--repo-root", repo]), 0)
            self.assertEqual(track_ops.main(["git", "push-staging", "--track", "1", "--repo-root", repo]), 0)
            refs = subprocess.run(["git", "--git-dir", env_ctx["bare"], "show-ref"], capture_output=True, text=True, check=True).stdout
            self.assertIn("refs/heads/colab-staging/track-1", refs)
            self.assertEqual(track_ops.main(["git", "prune-staging", "--track", "1", "--repo-root", repo]), 0)

            # Symmetric rollback-clean removes tracked added & untracked files in src/nevfe & tests while preserving specs/reports/metrics test
            added_tracked = os.path.join(repo, "src", "nevfe", "extra_added.py")
            open(added_tracked, "w", encoding="utf-8").write("y = 2\n")
            subprocess.run(["git", "-C", repo, "add", added_tracked], check=True, capture_output=True)
            subprocess.run(["git", "-C", repo, "commit", "-m", "extra tracked"], check=True, capture_output=True)
            untracked_inside = os.path.join(repo, "src", "nevfe", "core", "untracked_tmp.py")
            open(untracked_inside, "w", encoding="utf-8").write("z = 3\n")
            rep_file = os.path.join(repo, "reports", "track_1", "epistemic_ledger.json")
            open(rep_file, "w", encoding="utf-8").write('{"entries": []}\n')

            self.assertEqual(track_ops.main(["git", "rollback-clean", "--track", "1", "--best-commit", base_sha, "--iter", "1", "--repo-root", repo]), 0)
            self.assertFalse(os.path.exists(added_tracked))
            self.assertFalse(os.path.exists(untracked_inside))
            self.assertFalse(os.path.exists(os.path.join(repo, "tests", "test_track_candidate.py")))
            self.assertTrue(os.path.isfile(spec_p))
            self.assertTrue(os.path.isfile(rep_file))
            self.assertTrue(os.path.isfile(os.path.join(repo, "tests", "test_metrics_relaxation.py")))
            diff_best = subprocess.run(["git", "-C", repo, "diff", base_sha, "--", "src/nevfe/", "tests/"], capture_output=True, text=True, check=True).stdout.strip()
            self.assertEqual(diff_best, "")

            # papers verify positive, missing anchor, and waiver gating
            cache_d = os.path.join(repo, ".workflow_state", "paper_cache")
            os.makedirs(cache_d, exist_ok=True)
            tex_p = os.path.join(cache_d, "2501_00001.txt")
            open(tex_p, "w", encoding="utf-8", newline="\n").write("3. Method\nalpha_t = tau * sqrt(1 - BC)\n")
            ver_p = os.path.join(tmp, "ver.json")
            open(ver_p, "w", encoding="utf-8").write(json.dumps({
                "papers": [{
                    "paper_id": "2501.00001",
                    "status": "VERIFIED_TEX",
                    "cached_path": tex_p,
                    "source_sha256": _lf_sha_file(tex_p),
                    "section": "3. Method",
                    "anchor": "alpha_t = tau * sqrt(1 - BC)",
                }]
            }))
            self.assertEqual(track_ops.main(["papers", "verify", "--verification-json", ver_p, "--cache-dir", cache_d, "--repo-root", repo]), 0)

            open(ver_p, "w", encoding="utf-8").write(json.dumps({"papers": [{"paper_id": "x", "status": "UNAVAILABLE_NEEDS_WAIVER", "waiver_path": None}]}))
            self.assertEqual(track_ops.main(["papers", "verify", "--verification-json", ver_p, "--cache-dir", cache_d, "--repo-root", repo]), 1)

    def test_03_colab_drive_sync_calbank_v2_and_probe_loo(self):
        with tempfile.TemporaryDirectory() as tmp:
            # 1. colab_drive_sync idempotent update, --verify-refresh-creds, --poll-markers
            old_tok = os.environ.get("RCLONE_DRIVE_TOKEN_JSON")
            try:
                os.environ.pop("RCLONE_DRIVE_TOKEN_JSON", None)
                self.assertEqual(colab_drive_sync.main(["--verify-refresh-creds"]), 1)
                os.environ["RCLONE_DRIVE_TOKEN_JSON"] = json.dumps({"access_token": "bare_only"})
                self.assertEqual(colab_drive_sync.main(["--verify-refresh-creds"]), 1)
                os.environ["RCLONE_DRIVE_TOKEN_JSON"] = json.dumps({
                    "client_id": "cid", "client_secret": "csec", "refresh_token": "rtok", "access_token": "atok"
                })
                self.assertEqual(colab_drive_sync.main(["--verify-refresh-creds"]), 0)
            finally:
                if old_tok is None:
                    os.environ.pop("RCLONE_DRIVE_TOKEN_JSON", None)
                else:
                    os.environ["RCLONE_DRIVE_TOKEN_JSON"] = old_tok

            fake_files = _FakeDriveFiles()
            fake_svc = SimpleNamespace(files=lambda: fake_files)
            up_dir = os.path.join(tmp, "upload")
            os.makedirs(os.path.join(up_dir, "sub"), exist_ok=True)
            open(os.path.join(up_dir, "a.json"), "w", encoding="utf-8").write("{}\n")
            open(os.path.join(up_dir, "sub", "b.pt"), "wb").write(b"123")

            class _FakeHttpError(Exception):
                resp = None

            class _FakeMedia:
                def __init__(self, p: str, resumable: bool = False) -> None:
                    self._p = p
                    self._resumable = resumable

                def filename(self) -> str:
                    return self._p

            stub_mods = {
                "googleapiclient": SimpleNamespace(
                    errors=SimpleNamespace(HttpError=_FakeHttpError),
                    http=SimpleNamespace(MediaFileUpload=_FakeMedia),
                ),
                "googleapiclient.errors": SimpleNamespace(HttpError=_FakeHttpError),
                "googleapiclient.http": SimpleNamespace(MediaFileUpload=_FakeMedia),
            }
            saved_mods = {k: sys.modules.get(k) for k in stub_mods}
            sys.modules.update(stub_mods)
            try:
                rec1 = colab_drive_sync.upload_dir_to_drive(up_dir, ["VLM_Track_Runs", "run_1"], service=fake_svc, write_sha256=True)
                self.assertIn("a.json", rec1["uploaded_files"])
                self.assertIn("sub/b.pt", rec1["uploaded_files"])
                self.assertEqual(fake_files.creates.count("drive_upload_receipt.json"), 1)

                # Second upload updates existing drive_upload_receipt.json in place rather than duplicating
                rec2 = colab_drive_sync.upload_dir_to_drive(up_dir, ["VLM_Track_Runs", "run_1"], service=fake_svc, write_sha256=True)
                self.assertEqual(fake_files.creates.count("drive_upload_receipt.json"), 1)
                self.assertTrue(any(name == "drive_upload_receipt.json" for _, name in fake_files.updates))

                poll1 = colab_drive_sync.poll_markers(folder_id=rec2["folder_id"], service=fake_svc)
                self.assertFalse(poll1["complete"])
                self.assertFalse(poll1["stalled"])
                open(os.path.join(up_dir, "WAVE_COMPLETE_MARKER"), "w").write("ok\n")
                colab_drive_sync.upload_dir_to_drive(up_dir, ["VLM_Track_Runs", "run_1"], service=fake_svc, write_sha256=False)
                poll2 = colab_drive_sync.poll_markers(folder_path="VLM_Track_Runs/run_1", service=fake_svc)
                self.assertTrue(poll2["complete"])
            finally:
                for k, orig in saved_mods.items():
                    if orig is None:
                        sys.modules.pop(k, None)
                    else:
                        sys.modules[k] = orig

            # 2. Calibration Bank v2 --mock + validate + Track 1/2 eval + tamper rejection
            bank_v2 = os.path.join(tmp, "cal_v2")
            self.assertEqual(cal_bank_main(["--bank-version", "2", "--mock", "--max-steps", "4", "--output-dir", bank_v2, "--track", "1"]), 0)
            v2_meta = validate_cal_bank_v2(bank_v2)
            self.assertEqual(v2_meta["num_samples"], 30)
            self.assertLess(v2_meta["total_bytes"], 200 * 1024 * 1024)
            t2_eval = run_calibration_eval_v2(NEVFEEngine(config=AlgorithmConfig()), bank_v2, track=2)
            self.assertEqual(t2_eval["finite_ratio"], 1.0)
            self.assertEqual(t2_eval["nan_count"], 0)
            self.assertEqual(t2_eval["max_kv_diff"], 0.0)

            open(os.path.join(bank_v2, "shared_readout.pt"), "ab").write(b"corrupt")
            with self.assertRaises(ValueError):
                validate_cal_bank_v2(bank_v2)

            # 3. Stage A LOO net-contribution gate (M=0, M=1, M=2 + tie-breaking) & shared config resolver
            self.assertEqual(evaluate_loo_gate({"base": {"comp_70": 41.5}}, 40.92)["loo_decision"], "FULL_PASS")
            self.assertEqual(evaluate_loo_gate({"base": {"comp_70": 40.5}}, 40.92)["loo_decision"], "FAIL")
            self.assertEqual(evaluate_loo_gate({"1": {"comp_70": 42.0}, "0": {"comp_70": 41.2}}, 40.92)["loo_decision"], "FULL_PASS")
            sub_hint = evaluate_loo_gate({"11": {"comp_70": 42.0}, "01": {"comp_70": 42.0}, "10": {"comp_70": 41.0}}, 40.92)
            self.assertEqual(sub_hint["loo_decision"], "SUBMASK_HINT")
            self.assertEqual(sub_hint["submask_hint"], "01")
            self.assertFalse(sub_hint["pass_stage_a"])
            self.assertEqual(evaluate_loo_gate({"11": {"comp_70": 40.0}, "01": {"comp_70": 40.5}}, 40.92)["loo_decision"], "FAIL")

            probe_dir = os.path.join(tmp, "probe_out")
            for mask_name, c70 in (("11", 43.0), ("01", 41.5), ("10", 41.2)):
                mdir = os.path.join(probe_dir, f"mask_{mask_name}")
                os.makedirs(mdir, exist_ok=True)
                open(os.path.join(mdir, "summary_metrics.json"), "w", encoding="utf-8").write(json.dumps({
                    "heldout_70": {"f1_70": c70, "sub_70": c70, "comp_70": c70},
                    "overall": {"candidate_token_f1": c70, "candidate_substring_match": c70},
                    "peak_vram_gb": 5.1,
                }))
            bs, _ = load_baseline_probe_scores(os.path.join(PROJECT_ROOT, "configs", "frsa011_baseline_summary.json"))
            ab_sum = _write_ablation_summary(probe_dir, ["11", "01", "10"], bs, best_probe_comp=40.92)
            self.assertEqual(ab_sum["loo_decision"], "FULL_PASS")
            self.assertEqual(ab_sum["selected_mask"], "11")

            self.assertEqual(_parse_ablation_bits("base"), ())
            ns = argparse.Namespace(config_json=None, lambda_param=None, beta_param=None, ablation_mask="base")
            self.assertEqual(_resolve_nevfe_config(ns), AlgorithmConfig())

    def test_04_main_py_top_level_await_ast_stdlib_and_workflow_json(self):
        wf_dir = os.path.join(PROJECT_ROOT, "scripts", "workflows", "vlm_track_research")
        main_p = os.path.join(wf_dir, "main.py")
        wf_json_p = os.path.join(wf_dir, "workflow.json")

        src = open(main_p, "r", encoding="utf-8").read()
        lines = src.splitlines()
        self.assertLessEqual(len(lines), 350)
        tree = compile(src, main_p, "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT | ast.PyCF_ONLY_AST)
        self.assertIsInstance(tree, ast.Module)

        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        non_stdlib = sorted(imported - sys.stdlib_module_names)
        self.assertEqual(non_stdlib, [], f"Non-stdlib imports in main.py: {non_stdlib}")

        wf_data = json.load(open(wf_json_p, "r", encoding="utf-8"))
        self.assertEqual(wf_data["entry"], "main.py")
        args_props = wf_data["args"]["properties"]
        self.assertEqual(args_props["track"]["enum"], [1, 2])
        self.assertEqual(args_props["mode"]["enum"], ["run", "post_boost", "preflight", "status"])
        self.assertEqual(args_props["max_burst_iters"]["default"], 2)

    def test_05_predicate_branch_coverage(self):
        # 1. evaluate_loo_gate remaining edge branches
        self.assertEqual(evaluate_loo_gate({}, 40.92)["loo_decision"], "FAIL")
        self.assertEqual(evaluate_loo_gate({"0": {"comp_70": 42.0}}, 40.92)["loo_decision"], "FAIL")
        only_full = evaluate_loo_gate({"1": {"comp_70": 42.0}}, 40.92)
        self.assertEqual(only_full["loo_decision"], "FULL_PASS")
        self.assertTrue(only_full["pass_stage_a"])

        with tempfile.TemporaryDirectory() as tmp:
            # 2. track_ops.verify_papers_provenance & auditor _check_1_paper_verification branches
            cache_dir = os.path.join(tmp, "paper_cache")
            os.makedirs(cache_dir, exist_ok=True)
            waiver_p = os.path.join(tmp, "waiver.md")
            open(waiver_p, "w", encoding="utf-8", newline="\n").write("Signed waiver reason\n")
            cached_tex = os.path.join(cache_dir, "in_cache.tex")
            open(cached_tex, "w", encoding="utf-8", newline="\n").write("Other text\n")

            self.assertFalse(track_ops.verify_papers_provenance(os.path.join(tmp, "missing.json"), cache_dir, repo_root=tmp)["passed"])
            empty_j = os.path.join(tmp, "empty_papers.json")
            open(empty_j, "w", encoding="utf-8").write(json.dumps({"papers": []}))
            self.assertFalse(track_ops.verify_papers_provenance(empty_j, cache_dir, repo_root=tmp)["passed"])

            waiver_j = os.path.join(tmp, "valid_waiver.json")
            open(waiver_j, "w", encoding="utf-8").write(json.dumps({
                "papers": [{"paper_id": "pw", "status": "UNAVAILABLE_NEEDS_WAIVER", "waiver_path": waiver_p}]
            }))
            self.assertTrue(track_ops.verify_papers_provenance(waiver_j, cache_dir, repo_root=tmp)["passed"])

            err1_j = os.path.join(tmp, "err1.json")
            open(err1_j, "w", encoding="utf-8").write(json.dumps({
                "papers": [
                    {"paper_id": "p_bad", "status": "UNSUPPORTED"},
                    {"paper_id": "p_nocp", "status": "VERIFIED_TEX", "cached_path": ""},
                    {"paper_id": "p_miss", "status": "VERIFIED_PDF", "cached_path": "nope/absent.pdf"},
                ]
            }))
            res_err1 = track_ops.verify_papers_provenance(err1_j, cache_dir, repo_root=tmp)
            self.assertFalse(res_err1["passed"])
            self.assertFalse(run_auditor_checks._check_1_paper_verification(err1_j, tmp)["passed"])

            err2_j = os.path.join(tmp, "err2.json")
            open(err2_j, "w", encoding="utf-8").write(json.dumps({
                "papers": [
                    {
                        "paper_id": "p_cache_fallback",
                        "status": "VERIFIED_TEX",
                        "cached_path": "missing_dir/in_cache.tex",
                        "source_sha256": "0" * 64,
                        "section": "Missing Section",
                        "anchor": "Eq. (99)",
                    }
                ]
            }))
            res_err2 = track_ops.verify_papers_provenance(err2_j, cache_dir, repo_root=tmp)
            self.assertFalse(res_err2["passed"])
            self.assertEqual(len(res_err2["errors"]), 3)

            # 3. run_preflight state/resume, immutable, ancestry, env, budget & mode=status branches
            env_ctx = _build_workflow_test_repo(tmp)
            repo = env_ctx["repo"]
            lock_p = env_ctx["lock_path"]
            act_p = env_ctx["active_track"]
            bank_dir = env_ctx["bank_dir"]

            old_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
            old_enc = os.environ.get("PYTHONIOENCODING")
            try:
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                os.environ["PYTHONIOENCODING"] = "ascii"
                dirty_pre = track_ops.run_preflight(
                    1, mode="run", repo_root=repo, cal_bank_dir=os.path.join(tmp, "no_bank"), lock_path=lock_p, active_track_path=act_p
                )
                self.assertFalse(dirty_pre["ready"])
                status_pre = track_ops.run_preflight(
                    1, mode="status", repo_root=repo, cal_bank_dir=os.path.join(tmp, "no_bank"), lock_path=lock_p, active_track_path=act_p
                )
                self.assertTrue(status_pre["ready"])

                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                os.environ["PYTHONIOENCODING"] = "utf-8"

                # Cross-track conflict while queried track ALSO holds unexpired & expired active_wave
                open(act_p, "w", encoding="utf-8").write(json.dumps({"track_id": 2, "wave_id": "other_w"}))
                track_ops.set_active_wave(1, "w_u", "fid", ["c1"], "tr", launched_at=time.time(), ttl_s=3600, repo_root=repo, lock_path=lock_p, active_track_path=os.path.join(tmp, "unused.json"))
                c_unexp = track_ops.run_preflight(1, mode="run", repo_root=repo, cal_bank_dir=bank_dir, lock_path=lock_p, active_track_path=act_p)
                self.assertEqual(c_unexp["resume_action"], "CROSS_TRACK_CONFLICT")
                track_ops.set_active_wave(1, "w_e", "fid", ["c1"], "tr", launched_at=time.time() - 7200, ttl_s=3600, repo_root=repo, lock_path=lock_p, active_track_path=os.path.join(tmp, "unused.json"))
                c_exp = track_ops.run_preflight(1, mode="run", repo_root=repo, cal_bank_dir=bank_dir, lock_path=lock_p, active_track_path=act_p)
                self.assertEqual(c_exp["resume_action"], "CROSS_TRACK_CONFLICT")
                os.remove(act_p)

                # Budget exhaustion both with active_wave (keeps COLAB_POLLING) and without (BUDGET_EXHAUSTED)
                track_ops.set_active_wave(1, "w_live", "fid", ["c1"], "tr", launched_at=time.time(), ttl_s=3600, repo_root=repo, lock_path=lock_p, active_track_path=act_p)
                track_ops.transition_state(1, "RESEARCH", repo_root=repo, lock_path=lock_p, updates={"iteration": 99})
                b_poll = track_ops.run_preflight(1, mode="run", repo_root=repo, cal_bank_dir=bank_dir, lock_path=lock_p, active_track_path=act_p)
                self.assertEqual(b_poll["resume_action"], "COLAB_POLLING")
                self.assertFalse(b_poll["checks"]["budget"]["passed"])
                track_ops.clear_active_wave(1, repo_root=repo, lock_path=lock_p, active_track_path=act_p)
                track_ops.transition_state(1, "HALTED_FOR_BOOST", repo_root=repo, lock_path=lock_p)
                b_halted_pf = track_ops.run_preflight(1, mode="preflight", repo_root=repo, cal_bank_dir=bank_dir, lock_path=lock_p, active_track_path=act_p)
                self.assertTrue(b_halted_pf["checks"]["cal_bank_v2"]["passed"])
                b_halted_run = track_ops.run_preflight(1, mode="run", repo_root=repo, cal_bank_dir=bank_dir, lock_path=lock_p, active_track_path=act_p)
                self.assertEqual(b_halted_run["resume_action"], "HALTED_FOR_BOOST_REFUSED")
                track_ops.transition_state(1, "RESEARCH", repo_root=repo, lock_path=lock_p)
                b_exh = track_ops.run_preflight(1, mode="run", repo_root=repo, cal_bank_dir=bank_dir, lock_path=lock_p, active_track_path=act_p)
                self.assertEqual(b_exh["resume_action"], "BUDGET_EXHAUSTED")

                # Ancestry failure + tampered/missing immutable file + auditor _check_4 & _check_7a failures
                subprocess.run(["git", "-C", repo, "checkout", "--orphan", "orphan_b"], check=True, capture_output=True)
                subprocess.run(["git", "-C", repo, "commit", "-m", "orphan"], check=True, capture_output=True)
                fov_p = os.path.join(repo, "src", "nevfe", "core", "frozen_foveation.py")
                orig_fov = open(fov_p, "rb").read()
                open(fov_p, "wb").write(orig_fov + b"# tamper\n")
                tamper_pre = track_ops.run_preflight(1, mode="preflight", repo_root=repo, cal_bank_dir=bank_dir, lock_path=lock_p, active_track_path=act_p)
                self.assertFalse(tamper_pre["checks"]["git_ancestry"]["passed"])
                self.assertFalse(tamper_pre["checks"]["immutable_files"]["passed"])

                os.remove(fov_p)
                miss_pre = track_ops.run_preflight(1, mode="preflight", repo_root=repo, cal_bank_dir=bank_dir, lock_path=lock_p, active_track_path=act_p)
                self.assertFalse(miss_pre["checks"]["immutable_files"]["passed"])

                met_p = os.path.join(repo, "scripts", "benchmark", "metrics.py")
                c4_bad = run_auditor_checks._check_4_metrics_lock(repo, base_ref="base/frsa-011-frozen", pinned_sha256={"scripts/benchmark/metrics.py": "0" * 64})
                self.assertFalse(c4_bad["passed"])
                os.remove(met_p)
                c4_miss = run_auditor_checks._check_4_metrics_lock(repo, base_ref="base/frsa-011-frozen", pinned_sha256={"scripts/benchmark/metrics.py": "0" * 64})
                self.assertFalse(c4_miss["passed"])

                c7a_miss = run_auditor_checks._check_7a_spec_binding(repo, repo, mode="pre-colab", spec_sha="deadbeef", impl_sha="HEAD", spec_path_override=os.path.join(tmp, "no_spec.md"))
                self.assertFalse(c7a_miss["passed"])
            finally:
                if old_cuda is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda
                if old_enc is None:
                    os.environ.pop("PYTHONIOENCODING", None)
                else:
                    os.environ["PYTHONIOENCODING"] = old_enc

            # 4. Auditor _check_8_post_colab edge cases
            self.assertFalse(run_auditor_checks._check_8_post_colab(os.path.join(tmp, "no_res_dir"), 1222)["passed"])
            bad_res = os.path.join(tmp, "bad_res")
            os.makedirs(bad_res, exist_ok=True)
            open(os.path.join(bad_res, "null.md"), "wb").write(b"a\x00b")
            open(os.path.join(bad_res, "badutf8.md"), "wb").write(b"\xff\xfe")
            open(os.path.join(bad_res, "bad.json"), "w", encoding="utf-8").write("{invalid")
            open(os.path.join(bad_res, "STAGE_A_VETO"), "w", encoding="utf-8").write("veto\n")
            open(os.path.join(bad_res, "drive_upload_receipt.json"), "w", encoding="utf-8").write(json.dumps({"uploaded_count": 0}))
            c8_bad1 = run_auditor_checks._check_8_post_colab(bad_res, 1222, track=2)
            self.assertFalse(c8_bad1["passed"])

            bad_tel_res = os.path.join(tmp, "bad_tel_res")
            os.makedirs(bad_tel_res, exist_ok=True)
            open(os.path.join(bad_tel_res, "colab_run_telemetry.json"), "w", encoding="utf-8").write(
                json.dumps({"decoder_forward_calls_per_step_max": 2.0, "max_kv_diff": 0.25})
            )
            c8_bad2 = run_auditor_checks._check_8_post_colab(bad_tel_res, 1222, track=2)
            self.assertFalse(c8_bad2["passed"])
            self.assertTrue(any("single-step" in e for e in c8_bad2["errors"]))
            self.assertTrue(any("zero-KV" in e for e in c8_bad2["errors"]))
            self.assertFalse(run_auditor_checks._check_8_post_colab(bad_tel_res, 1222, track=1)["passed"])

            veto_probe_res = os.path.join(tmp, "veto_probe_res")
            os.makedirs(veto_probe_res, exist_ok=True)
            open(os.path.join(veto_probe_res, "ok.md"), "w", encoding="utf-8").write("# ok\n")
            open(os.path.join(veto_probe_res, "probe_ablation_summary.json"), "w", encoding="utf-8").write(json.dumps({"pass_stage_a": False}))
            open(os.path.join(veto_probe_res, "sha256sum.txt"), "w", encoding="utf-8").write("\n")
            self.assertFalse(run_auditor_checks._check_8_post_colab(veto_probe_res, 1222, track=1)["passed"])
            self.assertFalse(run_auditor_checks._check_8_post_colab(veto_probe_res, 1222, track=1, stage_a_vetoed=False)["passed"])
            self.assertFalse(run_auditor_checks._check_4_metrics_lock(tmp, base_ref="HEAD", pinned_sha256=None)["passed"])


if __name__ == "__main__":
    unittest.main()
