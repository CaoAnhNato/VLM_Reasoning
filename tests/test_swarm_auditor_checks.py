# -*- coding: utf-8 -*-
"""
Unit & Adversarial Audit + Liveness Test Suite (tests/test_swarm_auditor_checks.py).
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest

from scripts.audit.check_swarm_liveness import evaluate_liveness, main as liveness_main
from scripts.audit.run_auditor_checks import (
    EXPECTED_ARTIFACTS,
    _check_1_paper_verification,
    _check_2_and_3_and_5,
    _check_4_metrics_lock,
    _check_6_guarded_pytest,
    _check_7a_spec_binding,
    _check_8_post_colab,
    main as auditor_main,
)
from scripts.benchmark.export_calibration_bank import export_synthetic_calibration_bank_v2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _lf_sha(text_or_bytes) -> str:
    raw = text_or_bytes.encode("utf-8") if isinstance(text_or_bytes, str) else text_or_bytes
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest().lower()


def _init_git_repo(repo_dir: str, engine_code: str, spec_bytes: bytes) -> dict:
    os.makedirs(os.path.join(repo_dir, "src", "nevfe", "core"), exist_ok=True)
    os.makedirs(os.path.join(repo_dir, "src", "nevfe", "adapter"), exist_ok=True)
    os.makedirs(os.path.join(repo_dir, "docs", "specs"), exist_ok=True)
    os.makedirs(os.path.join(repo_dir, "reports"), exist_ok=True)
    os.makedirs(os.path.join(repo_dir, "scripts", "benchmark"), exist_ok=True)
    os.makedirs(os.path.join(repo_dir, "scripts", "local"), exist_ok=True)
    os.makedirs(os.path.join(repo_dir, "tests"), exist_ok=True)

    metrics_txt = "# metrics\n"
    metrics_test_txt = "def test_ok():\n    assert True\n"
    frozen_txt = "def foveate_frozen_011(x, q=None):\n    return (0, 0, 1, 1), 0.0, x\n"
    adapter_txt = (
        "def resolve_relative_decoder_layer_indices(n, frac=(0.5, 1.0)):\n"
        "    return list(range(n // 2, n))\n"
        "def helper(q):\n"
        "    return q.strip()\n"
    )
    pipeline_txt = "def pipe_ok():\n    return True\n"

    open(os.path.join(repo_dir, "scripts", "benchmark", "metrics.py"), "w", encoding="utf-8", newline="\n").write(metrics_txt)
    open(os.path.join(repo_dir, "tests", "test_metrics_relaxation.py"), "w", encoding="utf-8", newline="\n").write(metrics_test_txt)
    open(os.path.join(repo_dir, "tests", "test_nevfe_engine.py"), "w", encoding="utf-8", newline="\n").write("def test_eng():\n    assert True\n")
    open(os.path.join(repo_dir, "tests", "test_frsa011_base_parity.py"), "w", encoding="utf-8", newline="\n").write("def test_par():\n    assert True\n")
    real_guard = open(os.path.join(PROJECT_ROOT, "scripts", "local", "run_guarded.py"), "r", encoding="utf-8").read()
    open(os.path.join(repo_dir, "scripts", "local", "run_guarded.py"), "w", encoding="utf-8", newline="\n").write(real_guard)
    real_exp = open(os.path.join(PROJECT_ROOT, "scripts", "benchmark", "export_calibration_bank.py"), "r", encoding="utf-8").read()
    open(os.path.join(repo_dir, "scripts", "benchmark", "export_calibration_bank.py"), "w", encoding="utf-8", newline="\n").write(real_exp)

    open(os.path.join(repo_dir, "src", "__init__.py"), "w", encoding="utf-8").write("")
    open(os.path.join(repo_dir, "src", "nevfe", "__init__.py"), "w", encoding="utf-8").write("")
    open(os.path.join(repo_dir, "src", "nevfe", "core", "__init__.py"), "w", encoding="utf-8").write("")
    open(os.path.join(repo_dir, "src", "nevfe", "core", "frozen_foveation.py"), "w", encoding="utf-8", newline="\n").write(frozen_txt)
    open(os.path.join(repo_dir, "src", "nevfe", "core", "engine.py"), "w", encoding="utf-8", newline="\n").write(engine_code)
    open(os.path.join(repo_dir, "src", "nevfe", "adapter", "__init__.py"), "w", encoding="utf-8").write("")
    open(os.path.join(repo_dir, "src", "nevfe", "adapter", "vintern_adapter.py"), "w", encoding="utf-8", newline="\n").write(adapter_txt)
    open(os.path.join(repo_dir, "src", "nevfe", "pipeline.py"), "w", encoding="utf-8", newline="\n").write(pipeline_txt)
    open(os.path.join(repo_dir, "docs", "specs", "spec_iter_1.md"), "wb").write(spec_bytes)

    pinned = {
        "scripts/benchmark/metrics.py": _lf_sha(metrics_txt),
        "tests/test_metrics_relaxation.py": _lf_sha(metrics_test_txt),
        "src/nevfe/core/frozen_foveation.py": _lf_sha(frozen_txt),
    }
    pinned_path = os.path.join(repo_dir, "pinned.json")
    open(pinned_path, "w", encoding="utf-8", newline="\n").write(json.dumps(pinned))

    def git(*args):
        subprocess.run(["git", "-C", repo_dir, *args], check=True, capture_output=True)

    git("init")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    git("add", "-A")
    git("commit", "-m", "init")
    git("branch", "base/frsa-011-frozen")
    head_sha = subprocess.run(["git", "-C", repo_dir, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    return {"head_sha": head_sha, "pinned": pinned, "pinned_path": pinned_path}


class TestSwarmAuditorChecks(unittest.TestCase):
    def test_01_core_paper_verification_and_waiver_gating(self):
        with tempfile.TemporaryDirectory() as tmp:
            tex_p = os.path.join(tmp, "paper_a.txt")
            pdf_p = os.path.join(tmp, "paper_b.txt")
            waiver_p = os.path.join(tmp, "waiver.md")
            open(tex_p, "w", encoding="utf-8", newline="\n").write("Section 3. Method\nWe steer logits via Eq. (4): z* = z + alpha * dz.\n")
            open(pdf_p, "w", encoding="utf-8", newline="\n").write("Page 4 Methodology\nUpper FFN drift delta_ffn.\n")
            open(waiver_p, "w", encoding="utf-8", newline="\n").write("AUTHOR_WAIVER: Verified manually by CaoAnhNato.\n")

            ver_ok = os.path.join(tmp, "core_paper_verification.json")
            open(ver_ok, "w", encoding="utf-8").write(
                json.dumps(
                    {
                        "papers": [
                            {
                                "paper_id": "2501.00001",
                                "status": "VERIFIED_TEX",
                                "cached_path": tex_p,
                                "source_sha256": hashlib.sha256(open(tex_p, "rb").read()).hexdigest(),
                                "section": "Section 3. Method",
                                "anchor": "z* = z + alpha * dz",
                            },
                            {
                                "paper_id": "2501.00002",
                                "status": "VERIFIED_PDF",
                                "cached_path": pdf_p,
                                "source_sha256": hashlib.sha256(open(pdf_p, "rb").read()).hexdigest(),
                                "section": "Page 4 Methodology",
                                "anchor": "delta_ffn",
                            },
                        ]
                    }
                )
            )
            res_ok = _check_1_paper_verification(ver_ok, tmp)
            self.assertTrue(res_ok["passed"], res_ok["errors"])
            self.assertEqual(res_ok["paper_count"], 2)

            # UNAVAILABLE_NEEDS_WAIVER passes ONLY when waiver_path exists and is non-empty
            ver_waived = os.path.join(tmp, "waived.json")
            open(ver_waived, "w", encoding="utf-8").write(
                json.dumps({"papers": [{"paper_id": "10.1/xyz", "status": "UNAVAILABLE_NEEDS_WAIVER", "waiver_path": waiver_p}]})
            )
            self.assertTrue(_check_1_paper_verification(ver_waived, tmp)["passed"])

            ver_unwaived = os.path.join(tmp, "unwaived.json")
            open(ver_unwaived, "w", encoding="utf-8").write(
                json.dumps({"papers": [{"paper_id": "10.1/xyz", "status": "UNAVAILABLE_NEEDS_WAIVER", "waiver_path": None}]})
            )
            self.assertFalse(_check_1_paper_verification(ver_unwaived, tmp)["passed"])

            # Tampered SHA, missing section/anchor, > 3 papers, missing JSON all fail
            ver_bad = os.path.join(tmp, "bad.json")
            open(ver_bad, "w", encoding="utf-8").write(
                json.dumps(
                    {
                        "papers": [
                            {
                                "paper_id": "2501.00001",
                                "status": "VERIFIED_TEX",
                                "cached_path": tex_p,
                                "source_sha256": "0" * 64,
                                "section": "Missing Section",
                                "anchor": "missing eq",
                            }
                        ]
                    }
                )
            )
            self.assertFalse(_check_1_paper_verification(ver_bad, tmp)["passed"])
            open(ver_bad, "w", encoding="utf-8").write(json.dumps({"papers": [{"paper_id": "x"}] * 4}))
            self.assertFalse(_check_1_paper_verification(ver_bad, tmp)["passed"])
            self.assertFalse(_check_1_paper_verification(os.path.join(tmp, "missing.json"), tmp)["passed"])

    def test_02_ast_checks_2_3_5_track_gates_and_diff_filtered_adapter(self):
        base_ok = (
            "# -*- coding: utf-8 -*-\n"
            "# SPEC_REF: docs/specs/spec_iter_1.md@v1.0 (sha256=" + ("a" * 64) + ")\n"
            '"""Engine docstring."""\n'
            "from dataclasses import dataclass\n"
            "import torch\n"
            "@dataclass(frozen=True)\n"
            "class AlgorithmConfig:\n"
            "    lambda_param: float = 0.50\n"
            "    beta_param: float = 0.05\n"
            "    new_tau: float = 0.25\n"
            "    ablation_mask: tuple = ()\n"
            "def op(x):\n"
            "    if x.dim() == 3:\n"
            "        y = x.mean(dim=(-2, 1), keepdim=True)\n"
            "        z = -y.mean(dim=-2, keepdim=True)\n"
            "        return z.permute((0, 2, -3)).unsqueeze(-3).view(-1, 2)\n"
            "    return x * -1.0\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            repo = os.path.join(tmp, "repo")
            meta = _init_git_repo(repo, base_ok, b"# Spec\n")
            core = os.path.join(repo, "src", "nevfe", "core")

            # Baseline passes with pre-existing .strip() in vintern_adapter.py
            res = _check_2_and_3_and_5(
                core,
                repo_root=repo,
                base_ref="base/frsa-011-frozen",
                target_sha="HEAD",
                snapshot_dir=repo,
                pinned_sha256=meta["pinned"],
                track=1,
            )
            self.assertTrue(res["check2"]["passed"], res["check2"]["errors"])
            self.assertTrue(res["check3"]["passed"], res["check3"]["errors"])
            self.assertTrue(res["check5"]["passed"], res["check5"]["errors"])

            # Added heuristic .strip() & hardcoded layer index 12 in vintern_adapter.py + Track 2 clone_kv_cache -> fail
            bad_adapter = (
                "def resolve_relative_decoder_layer_indices(n, frac=(0.5, 1.0)):\n"
                "    return list(range(n // 2, n))\n"
                "def helper(q):\n"
                "    return q.strip()\n"
                "def added_bad(layers, s):\n"
                "    x = layers[12]\n"
                "    for i in range(12, 24):\n"
                "        pass\n"
                "    clone_kv_cache(x)\n"
                "    return s.strip(), x\n"
            )
            open(os.path.join(repo, "src", "nevfe", "adapter", "vintern_adapter.py"), "w", encoding="utf-8", newline="\n").write(bad_adapter)
            subprocess.run(["git", "-C", repo, "add", "-A"], check=True, capture_output=True)
            subprocess.run(["git", "-C", repo, "commit", "-m", "bad adapter"], check=True, capture_output=True)
            res_adp = _check_2_and_3_and_5(
                core,
                repo_root=repo,
                base_ref="base/frsa-011-frozen",
                target_sha="HEAD",
                snapshot_dir=repo,
                pinned_sha256=meta["pinned"],
                track=2,
            )
            self.assertFalse(res_adp["check3"]["passed"])
            self.assertFalse(res_adp["check5"]["passed"])

            # > 4 scalars (i.e. > 2 new scalars) & unlocked lambda_param -> Check #2 fail
            bad_5f = base_ok.replace(
                "    ablation_mask: tuple = ()",
                "    c: float = 0.1\n    d: float = 0.2\n    e: float = 0.3\n    ablation_mask: tuple = ()",
            ).replace("lambda_param: float = 0.50", "lambda_param: float = 0.40")
            open(os.path.join(core, "engine.py"), "w", encoding="utf-8").write(bad_5f)
            self.assertFalse(_check_2_and_3_and_5(core)["check2"]["passed"])

            # Undeclared numeric literal -> Check #2 fail
            open(os.path.join(core, "engine.py"), "w", encoding="utf-8").write(
                base_ok + "def bad(x):\n    return (x.size(0) == 3) or x * 0.37 + (-0.37)\n"
            )
            self.assertFalse(_check_2_and_3_and_5(core)["check2"]["passed"])

            # Missing AlgorithmConfig -> Check #2 fail
            open(os.path.join(core, "engine.py"), "w", encoding="utf-8").write("def op(x):\n    return x\n")
            self.assertFalse(_check_2_and_3_and_5(core)["check2"]["passed"])

            # Tampered frozen_foveation.py, extra file, forbidden import, >250 lines, backbone int 151936 -> Check #3/#5 fail
            open(os.path.join(core, "frozen_foveation.py"), "w", encoding="utf-8").write("# tampered\n")
            open(os.path.join(core, "extra.py"), "w", encoding="utf-8").write("x = 1\n")
            many_lines = "\n".join(f"x_{i} = 1" for i in range(260))
            open(os.path.join(core, "engine.py"), "w", encoding="utf-8").write(
                "import re\n" + base_ok + "def h(d):\n    v = 151936\n    return d['answer'], v\n" + many_lines + "\n"
            )
            res_bad = _check_2_and_3_and_5(core, repo_root=repo, base_ref="base/frsa-011-frozen", pinned_sha256=meta["pinned"])
            self.assertFalse(res_bad["check3"]["passed"])
            self.assertFalse(res_bad["check5"]["passed"])

    def test_03_spec_binding_metrics_lock_and_guarded_calbank_v2(self):
        spec_lf = b"# Spec\nline2\n"
        spec_crlf = b"# Spec\r\nline2\r\n"
        digest = hashlib.sha256(spec_lf).hexdigest()
        engine_ok = (
            "# -*- coding: utf-8 -*-\n"
            f"# SPEC_REF: docs/specs/spec_iter_1.md@v1.0 (sha256={digest})\n"
            "from dataclasses import dataclass\n"
            "@dataclass(frozen=True)\n"
            "class AlgorithmConfig:\n"
            "    lambda_param: float = 0.50\n"
            "    beta_param: float = 0.05\n"
            "    ablation_mask: tuple = ()\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            repo = os.path.join(tmp, "repo")
            meta = _init_git_repo(repo, engine_ok, spec_crlf)
            spec_sha = meta["head_sha"]

            # Commit clean impl
            open(os.path.join(repo, "src", "nevfe", "core", "engine.py"), "w", encoding="utf-8", newline="\n").write(engine_ok + "\n")
            subprocess.run(["git", "-C", repo, "add", "-A"], check=True, capture_output=True)
            subprocess.run(["git", "-C", repo, "commit", "-m", "impl"], check=True, capture_output=True)
            impl_sha = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()

            self.assertTrue(_check_7a_spec_binding(repo, repo, "pre-colab", spec_sha, impl_sha, None)["passed"])
            self.assertTrue(_check_4_metrics_lock(repo, base_ref="base/frsa-011-frozen", pinned_sha256=meta["pinned"])["passed"])
            # Non-zero git diff exit code on missing base_ref fails Check #4
            self.assertFalse(_check_4_metrics_lock(repo, base_ref="missing_ref", pinned_sha256=meta["pinned"])["passed"])

            lock_p = os.path.join(tmp, ".test_exp.lock")
            rec_dir = os.path.join(tmp, "receipts")
            mini_manifest = os.path.join(tmp, "mini_manifest.json")
            open(mini_manifest, "w", encoding="utf-8").write(
                json.dumps([{"question_id": 10, "error_class": "CLASS_A", "is_calibration": True, "image_name": "a.jpg"}])
            )
            bank_dir = os.path.join(repo, "cal_v2_test")
            export_synthetic_calibration_bank_v2(mini_manifest, bank_dir, max_steps=4, vocab_size=64, hidden_size=16, num_hidden_layers=4)
            # Patch Engine in repo to real frozen signature for cal_bank_v2 eval
            real_engine_code = (
                "# -*- coding: utf-8 -*-\n"
                f"# SPEC_REF: docs/specs/spec_iter_1.md@v1.0 (sha256={digest})\n"
                "from dataclasses import dataclass\n"
                "import torch\n"
                "@dataclass(frozen=True)\n"
                "class AlgorithmConfig:\n"
                "    lambda_param: float = 0.50\n"
                "    beta_param: float = 0.05\n"
                "    ablation_mask: tuple = ()\n"
                "class NEVFEEngine:\n"
                "    def __init__(self, config=None):\n"
                "        self.config = config or AlgorithmConfig()\n"
                "    def decode_step(self, lp, lz, q_mask=None, step_idx=1, step_context=None):\n"
                "        return lp.argmax(dim=-1), lp.float(), torch.tensor([[0.5]]), torch.tensor([[0.1]])\n"
            )
            open(os.path.join(repo, "src", "nevfe", "core", "engine.py"), "w", encoding="utf-8", newline="\n").write(real_engine_code)
            c6_ok = _check_6_guarded_pytest(repo, repo_root=repo, track=2, lock_path=lock_p, receipt_dir=rec_dir, cal_bank_dir="cal_v2_test")
            self.assertTrue(c6_ok["passed"], c6_ok["errors"])
            self.assertIsNotNone(c6_ok["cal_bank_eval"])
            self.assertTrue(os.path.isfile(os.path.join(rec_dir, "check6_unit_track_2.json")))
            self.assertFalse(_check_6_guarded_pytest(repo, repo_root=repo, track=1, lock_path=lock_p, cal_bank_dir="missing_bank")["passed"])

            # Mutate docs/specs/ -> Check #7a fails
            open(os.path.join(repo, "docs", "specs", "spec_iter_1.md"), "wb").write(b"# Spec\nmutated\n")
            subprocess.run(["git", "-C", repo, "add", "-A"], check=True, capture_output=True)
            subprocess.run(["git", "-C", repo, "commit", "-m", "bad spec"], check=True, capture_output=True)
            bad_impl_sha = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
            self.assertFalse(_check_7a_spec_binding(repo, repo, "pre-colab", spec_sha, bad_impl_sha, None)["passed"])

            # Mutate metrics.py -> Check #4 fails; bad test -> Check #6 fails
            open(os.path.join(repo, "scripts", "benchmark", "metrics.py"), "w").write("# hacked\n")
            open(os.path.join(repo, "tests", "test_metrics_relaxation.py"), "w").write("def test_fail():\n    assert False\n")
            self.assertFalse(_check_4_metrics_lock(repo, base_ref="base/frsa-011-frozen", pinned_sha256=meta["pinned"])["passed"])

            open(os.path.join(repo, "tests", "test_nevfe_engine.py"), "w").write("def test_fail():\n    assert False\n")
            self.assertFalse(_check_6_guarded_pytest(repo, lock_path=lock_p)["passed"])

            open(os.path.join(repo, "src", "nevfe", "core", "engine.py"), "w").write("# no ref\n")
            self.assertFalse(_check_7a_spec_binding(repo, repo, "bootstrap", None, None, None)["passed"])

    def test_04_pre_colab_bootstrap_and_post_colab_check8_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            tex_p = os.path.join(tmp, "paper_1.txt")
            open(tex_p, "w", encoding="utf-8", newline="\n").write("Section 2. Theory\nEq. (1): alpha_t = 0.5\n")
            ver_json = os.path.join(tmp, "core_paper_verification.json")
            open(ver_json, "w", encoding="utf-8").write(
                json.dumps(
                    {
                        "papers": [
                            {
                                "paper_id": "2501.00010",
                                "status": "VERIFIED_TEX",
                                "cached_path": tex_p,
                                "source_sha256": hashlib.sha256(open(tex_p, "rb").read()).hexdigest(),
                                "section": "Section 2. Theory",
                                "anchor": "alpha_t = 0.5",
                            }
                        ]
                    }
                )
            )
            spec_bytes = b"# Spec\nbody\n"
            digest = hashlib.sha256(spec_bytes).hexdigest()
            engine_ok = (
                "# -*- coding: utf-8 -*-\n"
                f"# SPEC_REF: docs/specs/spec_iter_1.md@v1.0 (sha256={digest})\n"
                "from dataclasses import dataclass\n"
                "@dataclass(frozen=True)\n"
                "class AlgorithmConfig:\n"
                "    lambda_param: float = 0.50\n"
                "    beta_param: float = 0.05\n"
                "    ablation_mask: tuple = ()\n"
                "def step(x):\n"
                "    return x\n"
            )
            repo = os.path.join(tmp, "repo")
            meta = _init_git_repo(repo, engine_ok, spec_bytes)
            sha = meta["head_sha"]
            lock_p = os.path.join(tmp, ".test_exp.lock")
            rec_dir = os.path.join(tmp, "receipts")

            out_pre = os.path.join(tmp, "pre.json")
            rc = auditor_main(
                [
                    "--mode", "pre-colab", "--repo-root", repo, "--track", "2", "--iter", "1",
                    "--base-ref", "base/frsa-011-frozen", "--spec-sha", sha, "--impl-sha", sha,
                    "--paper-verification-json", ver_json, "--lock-path", lock_p,
                    "--receipt-dir", rec_dir, "--pinned-sha256-json", meta["pinned_path"],
                    "--output-json", out_pre,
                ]
            )
            self.assertEqual(rc, 0, open(out_pre).read())
            self.assertEqual(
                auditor_main(
                    [
                        "--mode", "bootstrap", "--repo-root", repo, "--track", "1",
                        "--base-ref", "base/frsa-011-frozen", "--lock-path", lock_p,
                        "--pinned-sha256-json", meta["pinned_path"],
                        "--spec-path", os.path.join(repo, "docs", "specs", "spec_iter_1.md"),
                    ]
                ),
                0,
            )

            # Post-colab Check #8: Stage B complete + telemetry + drive receipt + UTF-8 lint
            res_dir = os.path.join(tmp, "res")
            drv_dir = os.path.join(tmp, "drv")
            os.makedirs(res_dir)
            os.makedirs(drv_dir)
            sha_lines = []
            for art in EXPECTED_ARTIFACTS:
                if art == "predictions.jsonl":
                    raw = (json.dumps({"question_id": 1}) + "\n" + json.dumps({"question_id": 2}) + "\n").encode("utf-8")
                elif art.endswith(".json"):
                    raw = json.dumps({"artifact": art}).encode("utf-8")
                else:
                    raw = f"data-{art}\n".encode("utf-8")
                open(os.path.join(res_dir, art), "wb").write(raw)
                open(os.path.join(drv_dir, art), "wb").write(raw)
                sha_lines.append(f"{hashlib.sha256(raw).hexdigest()}  {art}\n")

            tel_bytes = json.dumps({"decoder_forward_calls_per_step_max": 1.0, "max_kv_diff": 0.0}).encode("utf-8")
            open(os.path.join(res_dir, "colab_run_telemetry.json"), "wb").write(tel_bytes)
            open(os.path.join(res_dir, "drive_upload_receipt.json"), "w", encoding="utf-8").write(
                json.dumps({"folder_id": "fid_123", "uploaded_count": len(EXPECTED_ARTIFACTS)})
            )
            open(os.path.join(res_dir, "sha256sum.txt"), "w", encoding="utf-8").writelines(sha_lines)

            c_init = os.path.join(tmp, "c_init.json")
            c_curr = os.path.join(tmp, "c_curr.json")
            open(c_init, "w").write(json.dumps([{"id": "cell1"}]))
            open(c_curr, "w").write(json.dumps([{"id": "cell1"}]))

            out_post = os.path.join(tmp, "post.json")
            rc_post = auditor_main(
                [
                    "--mode", "post-colab", "--track", "2", "--results-dir", res_dir, "--expected-qids", "2",
                    "--drive-dir", drv_dir, "--initial-cells-json", c_init, "--current-cells-json", c_curr,
                    "--output-json", out_post,
                ]
            )
            self.assertEqual(rc_post, 0, open(out_post).read())

            # Stage-A-vetoed mode passes without Stage B artifacts
            veto_dir = os.path.join(tmp, "veto")
            os.makedirs(veto_dir)
            p_bytes = json.dumps({"loo_decision": "FAIL", "pass_stage_a": False}).encode("utf-8")
            t_bytes = json.dumps({"decoder_forward_calls_per_step_max": 1.0, "max_kv_diff": 0.0}).encode("utf-8")
            open(os.path.join(veto_dir, "probe_ablation_summary.json"), "wb").write(p_bytes)
            open(os.path.join(veto_dir, "colab_run_telemetry.json"), "wb").write(t_bytes)
            open(os.path.join(veto_dir, "STAGE_A_VETO"), "w").write("stage_a_decision=FAIL\n")
            open(os.path.join(veto_dir, "sha256sum.txt"), "w").write(
                f"{hashlib.sha256(p_bytes).hexdigest()}  probe_ablation_summary.json\n"
                f"{hashlib.sha256(t_bytes).hexdigest()}  colab_run_telemetry.json\n"
            )
            self.assertTrue(_check_8_post_colab(veto_dir, 1222, track=2)["passed"])

            # Track 2 telemetry violation + null byte lint + missing files fail Check #8
            open(os.path.join(veto_dir, "colab_run_telemetry.json"), "wb").write(
                json.dumps({"decoder_forward_calls_per_step_max": 2.0, "max_kv_diff": 0.01}).encode("utf-8")
            )
            open(os.path.join(veto_dir, "bad.md"), "wb").write(b"utf16\x00null")
            bad_veto = _check_8_post_colab(veto_dir, 1222, track=2, stage_a_vetoed=True)
            self.assertFalse(bad_veto["passed"])
            self.assertGreaterEqual(len(bad_veto["errors"]), 3)

            open(c_curr, "w").write(json.dumps([{"id": "cell1"}, {"id": "extra"}]))
            open(os.path.join(drv_dir, "checkpoint.json"), "wb").write(b"corrupt")
            os.remove(os.path.join(res_dir, "config_used.json"))
            bad_post = _check_8_post_colab(res_dir, 5, drv_dir, c_init, c_curr, track=2, stage_a_vetoed=False)
            self.assertFalse(bad_post["passed"])

    def test_05_liveness_monitor_active_75m_level1_30m_level2_60m(self):
        with tempfile.TemporaryDirectory() as tmp:
            wt = os.path.join(tmp, "wt")
            os.makedirs(wt)
            open(os.path.join(wt, "work.py"), "w").write("x = 1\n")
            subprocess.run(["git", "-C", wt, "init"], check=True, capture_output=True)
            subprocess.run(["git", "-C", wt, "config", "user.email", "t@e.com"], check=True, capture_output=True)
            subprocess.run(["git", "-C", wt, "config", "user.name", "T"], check=True, capture_output=True)
            subprocess.run(["git", "-C", wt, "add", "work.py"], check=True, capture_output=True)
            subprocess.run(["git", "-C", wt, "commit", "-m", "c"], check=True, capture_output=True)
            branch_name = subprocess.run(["git", "-C", wt, "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()

            tr = os.path.join(tmp, "tr.jsonl")
            open(tr, "w", encoding="utf-8").write(
                json.dumps({"step_index": 2, "status": "DONE", "tool_calls": [{"name": "read_file"}]})
                + "\n\n"
                + json.dumps({"step_index": 1, "status": "DONE"})
                + "\n"
            )
            hb_dir = os.path.join(tmp, "hb")
            os.makedirs(hb_dir)
            open(os.path.join(hb_dir, "1.json"), "w").write(json.dumps({"worker_id": "w1", "bytes_emitted": 100, "child_pid": os.getpid()}))
            open(os.path.join(hb_dir, "2.json"), "w").write(json.dumps({"worker_id": "w1", "bytes_emitted": 50, "child_pid": None}))
            open(os.path.join(hb_dir, "other.txt"), "w").write("ignore")
            ckpt = os.path.join(tmp, "ckpt.json")
            open(ckpt, "w").write(json.dumps({"total_completed": 10}))

            ledger = {
                "workers": {
                    "w_idle": {"state": "waiting"},
                    "w1": {"state": "working", "transcript_path": tr, "worktree_path": wt, "branch": branch_name, "colab_checkpoint_path": ckpt},
                    "w2": {"state": "working", "transcript_path": os.path.join(tmp, "missing.jsonl"), "child_pid": 999999},
                }
            }
            snap_path = os.path.join(tmp, "snap.json")
            self.assertEqual(liveness_main(["--ledger", self._write_json(tmp, "l1.json", ledger), "--snapshot", snap_path, "--heartbeat-dir", os.path.join(tmp, "no_hb"), "--now-epoch", "1000"]), 0)
            self.assertEqual(liveness_main(["--ledger", self._write_json(tmp, "l2.json", ledger), "--snapshot", snap_path, "--heartbeat-dir", hb_dir, "--now-epoch", "1000", "--output-json", os.path.join(tmp, "out1.json")]), 0)
            snap = json.load(open(snap_path))

            open(tr, "a", encoding="utf-8").write(json.dumps({"step_index": 3, "status": "RUNNING", "tool_calls": [{"name": "run_command"}]}) + "\n")
            open(ckpt, "w").write(json.dumps({"total_completed": 25}))
            rep_75, snap2 = evaluate_liveness(ledger, snap, hb_dir, now_epoch=5500.0)
            self.assertEqual(rep_75["workers"]["w1"]["classification"], "ACTIVE_PROGRESS")
            self.assertEqual(rep_75["workers"]["w2"]["classification"], "LEVEL_2_SESSION_FREEZE")

            rep_mon, _ = evaluate_liveness(ledger, snap2, hb_dir, now_epoch=6500.0)
            self.assertEqual(rep_mon["workers"]["w1"]["classification"], "MONITORING")
            rep_l1, _ = evaluate_liveness(ledger, snap2, hb_dir, now_epoch=7300.0)
            self.assertEqual(rep_l1["workers"]["w1"]["classification"], "LEVEL_1_SUBPROCESS_HANG")

            open(os.path.join(hb_dir, "1.json"), "w").write(json.dumps({"worker_id": "w1", "bytes_emitted": 100, "child_pid": None}))
            rep_l2, _ = evaluate_liveness(ledger, snap2, hb_dir, now_epoch=9100.0)
            self.assertEqual(rep_l2["workers"]["w1"]["classification"], "LEVEL_2_SESSION_FREEZE")

    @staticmethod
    def _write_json(directory: str, name: str, data: dict) -> str:
        p = os.path.join(directory, name)
        open(p, "w", encoding="utf-8").write(json.dumps(data))
        return p


if __name__ == "__main__":
    unittest.main()
