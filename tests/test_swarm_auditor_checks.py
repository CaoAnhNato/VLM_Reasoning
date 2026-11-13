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
    _check_2_and_3_and_5,
    _check_4_metrics_lock,
    _check_6_guarded_pytest,
    _check_7a_spec_binding,
    _check_8_post_colab,
    main as auditor_main,
)
from scripts.audit.verify_transcript_pdf_reads import main as lit_main, verify_literature_audit

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _make_pdf(path: str, size_bytes: int = 55000, magic: bytes = b"%PDF-1.4\n", tag: bytes = b"") -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    header = magic + tag
    payload = header + (b"0" * max(0, size_bytes - len(header)))
    with open(path, "wb") as f:
        f.write(payload)
    return hashlib.sha256(payload).hexdigest()


def _write_reading_files(
    directory: str,
    cache_dir: str,
    buckets: list,
    venue_d: str = "IEEE TPAMI (ISSN 0162-8828, Q1)",
    drop_bib_last: bool = False,
    page_1_only: bool = False,
    with_manifest: bool = True,
    use_arxiv_latex: bool = False,
) -> tuple:
    os.makedirs(directory, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    md_lines = [
        "# Reading History",
        "- **Direction ID**: direction_1",
        "- **Problem Framing**: test",
        "- **Iterations Covered**: iter_1",
        "- **Verdict**: IN_PROGRESS",
        "",
        "| Iteration | BibTeX Key | Title | Venue & Q1/A* Tier | DOI/ArXiv | Cached PDF SHA256 | Bucket [A/B/C/D] | Pages Inspected | Verbatim Page>=2 Anchor & Principle | Repo URL |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    bib_lines = []
    manifest = []
    for idx, bkt in enumerate(buckets):
        key = f"paper_{idx}"
        pdf_path = os.path.join(cache_dir, f"{key}.pdf")
        sha = _make_pdf(pdf_path, 55000, tag=f"{key}\n".encode("utf-8"))
        venue = venue_d if bkt == "D" else "CVPR 2025"
        arxiv_id = f"2501.000{idx:02d}"
        pages = "1" if page_1_only else "2-7"
        md_lines.append(
            f"| iter_1 | {key} | Title {idx} | {venue} | {arxiv_id} | {sha} | {bkt} | {pages} | Page 2 quote | https://github.com/org/repo |"
        )
        if not (drop_bib_last and idx == len(buckets) - 1):
            bib_lines.append(f"@article{{{key},\n  title={{Title {idx}}},\n  year={{2025}}\n}}")
        if with_manifest:
            manifest.append(
                {
                    "team": 1,
                    "iter": 1,
                    "paper_id": key,
                    "pdf_path": pdf_path,
                    "pdf_sha256": sha,
                    "pages_read": [1] if page_1_only else [2, 3],
                    "body_chars": 800,
                }
            )
    md_path = os.path.join(directory, "reading_history.md")
    bib_path = os.path.join(directory, "reading_history.bib")
    open(md_path, "w", encoding="utf-8").write("\n".join(md_lines) + "\n")
    open(bib_path, "w", encoding="utf-8").write("\n".join(bib_lines) + "\n")
    if with_manifest:
        open(os.path.join(cache_dir, "read_manifest.json"), "w", encoding="utf-8").write(json.dumps(manifest))

    transcript_path = os.path.join(directory, "transcript.jsonl")
    full_path = os.path.join(directory, "transcript_full.jsonl")
    # Step 1 (old iteration before cursor=2)
    old_step = {"step_index": 1, "tool_calls": [{"name": "run_command", "arguments": {"CommandLine": "echo hi"}}]}
    # Step 3 (new iteration, truncated in compact transcript, full in transcript_full)
    tcs = []
    if not use_arxiv_latex:
        tcs.append({"name": "run_command", "arguments": {"CommandLine": "python tools/s2_folks_cli/fetch_papers.py --inspect-pages 2-7 --pdf x.pdf"}})
    else:
        for idx in range(len(buckets)):
            tcs.append(
                {
                    "name": "call_mcp_tool",
                    "arguments": {
                        "ServerName": "arxiv",
                        "ToolName": "get_paper_latex_section",
                        "Arguments": {"paper_id": f"2501.000{idx:02d}", "section": "3. Method"},
                    },
                }
            )
    full_step3 = {"step_index": 3, "tool_calls": tcs}
    compact_step3 = {"step_index": 3, "truncated_fields": ["tool_calls"], "tool_calls": []}
    open(transcript_path, "w", encoding="utf-8").write(json.dumps(old_step) + "\n" + json.dumps(compact_step3) + "\n")
    open(full_path, "w", encoding="utf-8").write(json.dumps(old_step) + "\n" + json.dumps(full_step3) + "\n")
    return md_path, bib_path, transcript_path


def _init_git_repo(repo_dir: str, engine_code: str, spec_bytes: bytes) -> tuple:
    os.makedirs(os.path.join(repo_dir, "src", "nevfe", "core"), exist_ok=True)
    os.makedirs(os.path.join(repo_dir, "docs", "specs"), exist_ok=True)
    os.makedirs(os.path.join(repo_dir, "reports"), exist_ok=True)
    os.makedirs(os.path.join(repo_dir, "scripts", "benchmark"), exist_ok=True)
    os.makedirs(os.path.join(repo_dir, "scripts", "local"), exist_ok=True)
    os.makedirs(os.path.join(repo_dir, "tests"), exist_ok=True)

    open(os.path.join(repo_dir, "scripts", "benchmark", "metrics.py"), "w", encoding="utf-8").write("# metrics\n")
    open(os.path.join(repo_dir, "tests", "test_metrics_relaxation.py"), "w", encoding="utf-8").write(
        "def test_ok():\n    assert True\n"
    )
    open(os.path.join(repo_dir, "tests", "test_nevfe_engine.py"), "w", encoding="utf-8").write(
        "def test_eng():\n    assert True\n"
    )
    real_guard = open(os.path.join(PROJECT_ROOT, "scripts", "local", "run_guarded.py"), "r", encoding="utf-8").read()
    open(os.path.join(repo_dir, "scripts", "local", "run_guarded.py"), "w", encoding="utf-8").write(real_guard)
    open(os.path.join(repo_dir, "src", "nevfe", "__init__.py"), "w", encoding="utf-8").write("")
    open(os.path.join(repo_dir, "src", "nevfe", "core", "__init__.py"), "w", encoding="utf-8").write("")
    open(os.path.join(repo_dir, "src", "nevfe", "core", "engine.py"), "w", encoding="utf-8").write(engine_code)
    open(os.path.join(repo_dir, "docs", "specs", "spec_iter_1.md"), "wb").write(spec_bytes)

    def git(*args):
        subprocess.run(["git", "-C", repo_dir, *args], check=True, capture_output=True)

    git("init")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    git("add", "-A")
    git("commit", "-m", "init")
    git("branch", "iter23_staging")
    spec_sha = subprocess.run(["git", "-C", repo_dir, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    return spec_sha, spec_sha


class TestSwarmAuditorChecks(unittest.TestCase):
    def test_01_literature_audit_positive_and_negative_fixtures(self):
        buckets_9 = ["A", "A", "B", "B", "C", "C", "D", "D", "D"]
        with tempfile.TemporaryDirectory() as tmp:
            # Valid 9 papers via read_manifest + transcript_full join
            md, bib, tr = _write_reading_files(os.path.join(tmp, "ok9"), os.path.join(tmp, "c9"), buckets_9)
            res = verify_literature_audit(tr, 2, os.path.join(tmp, "c9"), md, bib)
            self.assertTrue(res["passed"], res["errors"])

            # Valid 18 papers (>15 allowed)
            buckets_18 = ["A"] * 3 + ["B"] * 5 + ["C"] * 5 + ["D"] * 5
            md18, bib18, tr18 = _write_reading_files(os.path.join(tmp, "ok18"), os.path.join(tmp, "c18"), buckets_18)
            res18 = verify_literature_audit(tr18, 2, os.path.join(tmp, "c18"), md18, bib18)
            self.assertTrue(res18["passed"], res18["errors"])
            self.assertEqual(res18["paper_count"], 18)

            # Valid via arxiv latex section
            md_ax, bib_ax, tr_ax = _write_reading_files(
                os.path.join(tmp, "ax"), os.path.join(tmp, "c_ax"), buckets_9, with_manifest=False, use_arxiv_latex=True
            )
            self.assertTrue(verify_literature_audit(tr_ax, 2, os.path.join(tmp, "c_ax"), md_ax, bib_ax)["passed"])

            # < 7 PDFs fail
            md6, bib6, tr6 = _write_reading_files(os.path.join(tmp, "lt7"), os.path.join(tmp, "c6"), ["A", "B", "B", "C", "D", "D"])
            self.assertFalse(verify_literature_audit(tr6, 2, os.path.join(tmp, "c6"), md6, bib6)["passed"])

            # < 50 KB dummy PDF fail
            md_sm, bib_sm, tr_sm = _write_reading_files(os.path.join(tmp, "sm"), os.path.join(tmp, "c_sm"), buckets_9)
            _make_pdf(os.path.join(tmp, "c_sm", "paper_0.pdf"), 1024)
            self.assertFalse(verify_literature_audit(tr_sm, 2, os.path.join(tmp, "c_sm"), md_sm, bib_sm)["passed"])

            # Page-1-only without valid page>=2 manifest fail
            md_p1, bib_p1, tr_p1 = _write_reading_files(
                os.path.join(tmp, "p1"), os.path.join(tmp, "c_p1"), buckets_9, page_1_only=True, with_manifest=False
            )
            self.assertFalse(verify_literature_audit(tr_p1, 2, os.path.join(tmp, "c_p1"), md_p1, bib_p1)["passed"])

            # Missing Bucket D Q1 journal (conference only) fail
            md_nd, bib_nd, tr_nd = _write_reading_files(
                os.path.join(tmp, "nd"), os.path.join(tmp, "c_nd"), buckets_9, venue_d="CVPR 2025"
            )
            self.assertFalse(verify_literature_audit(tr_nd, 2, os.path.join(tmp, "c_nd"), md_nd, bib_nd)["passed"])

            # BibTeX 1:1 mismatch + invalid bucket fail
            md_bm, bib_bm, tr_bm = _write_reading_files(
                os.path.join(tmp, "bm"), os.path.join(tmp, "c_bm"), ["A", "B", "B", "C", "C", "D", "X"], drop_bib_last=True
            )
            self.assertFalse(verify_literature_audit(tr_bm, 2, os.path.join(tmp, "c_bm"), md_bm, bib_bm)["passed"])

            # CLI main() exit code 0 and output_json
            out_json = os.path.join(tmp, "lit_out.json")
            self.assertEqual(
                lit_main(["--transcript", tr, "--cursor-step", "2", "--cache-dir", os.path.join(tmp, "c9"), "--reading-history", md, "--bib", bib, "--output-json", out_json]),
                0,
            )
            self.assertTrue(os.path.isfile(out_json))

    def test_02_ast_checks_2_3_5_positive_and_negative(self):
        base_ok = (
            "# -*- coding: utf-8 -*-\n"
            "# SPEC_REF: docs/specs/spec_iter_1.md@v1.0 (sha256=" + ("a" * 64) + ")\n"
            '"""Engine docstring."""\n'
            "from dataclasses import dataclass\n"
            "import torch\n"
            "@dataclass(frozen=True)\n"
            "class AlgorithmConfig:\n"
            "    lambda_param: float = 0.5\n"
            "    beta_param: float = 0.05\n"
            "    ablation_mask: tuple = (1, 1, 1)\n"
            "def op(x):\n"
            "    if x.dim() == 3:\n"
            "        y = x.mean(dim=(-2, 1), keepdim=True)\n"
            "        z = -y.mean(dim=-2, keepdim=True)\n"
            "        return z.permute((0, 2, -3)).unsqueeze(-3).view(-1, 2)\n"
            "    return x * -1.0\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            core = os.path.join(tmp, "core")
            os.makedirs(core)
            open(os.path.join(core, "__init__.py"), "w").write("")
            open(os.path.join(core, "engine.py"), "w").write(base_ok)
            res = _check_2_and_3_and_5(core)
            self.assertTrue(res["check2"]["passed"], res["check2"]["errors"])
            self.assertTrue(res["check3"]["passed"], res["check3"]["errors"])
            self.assertTrue(res["check5"]["passed"], res["check5"]["errors"])

            # 4-field AlgorithmConfig -> Check #2 fail
            bad_4f = base_ok.replace("    ablation_mask: tuple = (1, 1, 1)", "    c: float = 0.1\n    d: float = 0.2\n    ablation_mask: tuple = (1, 1, 1)")
            open(os.path.join(core, "engine.py"), "w").write(bad_4f)
            self.assertFalse(_check_2_and_3_and_5(core)["check2"]["passed"])

            # Undeclared 0.37, -0.37, tuple in non-axis call, and integer 3 in non-dim compare -> Check #2 fail
            open(os.path.join(core, "engine.py"), "w").write(base_ok + "def bad(x):\n    return (x.size(0) == 3) or x * 0.37 + (-0.37) + sum((4, 5))\n")
            self.assertFalse(_check_2_and_3_and_5(core)["check2"]["passed"])

            # Missing AlgorithmConfig -> Check #2 fail
            open(os.path.join(core, "engine.py"), "w").write("def op(x):\n    return x\n")
            self.assertFalse(_check_2_and_3_and_5(core)["check2"]["passed"])

            # Extra file + forbidden import + .strip() + GT key + >250 lines + 151936 literal
            open(os.path.join(core, "extra.py"), "w").write("x = 1\n")
            many_lines = "\n".join(f"x_{i} = 1" for i in range(260))
            bad_all = (
                "import re\nfrom difflib import SequenceMatcher\n"
                + base_ok
                + "def h(s, d):\n    v = 151936\n    return s.strip(), d['answer'], v\n"
                + many_lines
                + "\n"
            )
            open(os.path.join(core, "engine.py"), "w").write(bad_all)
            res_bad = _check_2_and_3_and_5(core)
            self.assertFalse(res_bad["check3"]["passed"])
            self.assertFalse(res_bad["check5"]["passed"])

    def test_03_spec_binding_crlf_lf_and_git_diff(self):
        spec_lf = b"# Spec\nline2\n"
        spec_crlf = b"# Spec\r\nline2\r\n"
        digest = hashlib.sha256(spec_lf).hexdigest()
        engine_ok = (
            "# -*- coding: utf-8 -*-\n"
            f"# SPEC_REF: docs/specs/spec_iter_1.md@v1.0 (sha256={digest})\n"
            "from dataclasses import dataclass\n"
            "@dataclass(frozen=True)\n"
            "class AlgorithmConfig:\n"
            "    lambda_param: float = 0.5\n"
            "    ablation_mask: tuple = (1, 1, 1)\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            repo = os.path.join(tmp, "repo")
            spec_sha, _ = _init_git_repo(repo, engine_ok, spec_crlf)
            # Commit clean impl
            open(os.path.join(repo, "src", "nevfe", "core", "engine.py"), "w", encoding="utf-8").write(engine_ok + "\n")
            subprocess.run(["git", "-C", repo, "add", "-A"], check=True, capture_output=True)
            subprocess.run(["git", "-C", repo, "commit", "-m", "impl"], check=True, capture_output=True)
            impl_sha = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()

            self.assertTrue(_check_7a_spec_binding(repo, repo, "pre-colab", spec_sha, impl_sha, None)["passed"])
            self.assertTrue(_check_4_metrics_lock(repo)["passed"])
            self.assertTrue(_check_6_guarded_pytest(repo)["passed"])

            # Mutate docs/specs/ in a new commit -> Check #7a fails
            open(os.path.join(repo, "docs", "specs", "spec_iter_1.md"), "wb").write(b"# Spec\nmutated\n")
            subprocess.run(["git", "-C", repo, "add", "-A"], check=True, capture_output=True)
            subprocess.run(["git", "-C", repo, "commit", "-m", "bad spec edit"], check=True, capture_output=True)
            bad_impl_sha = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
            self.assertFalse(_check_7a_spec_binding(repo, repo, "pre-colab", spec_sha, bad_impl_sha, None)["passed"])

            # Mutate metrics.py -> Check #4 fails; break test -> Check #6 fails; bad SPEC_REF -> Check #7a fails
            open(os.path.join(repo, "scripts", "benchmark", "metrics.py"), "w").write("# hacked\n")
            open(os.path.join(repo, "tests", "test_metrics_relaxation.py"), "w").write("def test_fail():\n    assert False\n")
            self.assertFalse(_check_4_metrics_lock(repo)["passed"])

            open(os.path.join(repo, "tests", "test_nevfe_engine.py"), "w").write("def test_fail():\n    assert False\n")
            self.assertFalse(_check_6_guarded_pytest(repo)["passed"])

            open(os.path.join(repo, "src", "nevfe", "core", "engine.py"), "w").write("# no ref\n")
            self.assertFalse(_check_7a_spec_binding(repo, repo, "bootstrap", None, None, None)["passed"])
            # Missing spec file branch
            open(os.path.join(repo, "src", "nevfe", "core", "engine.py"), "w").write(
                "# -*- coding: utf-8 -*-\n# SPEC_REF: docs/specs/missing.md@v1.0 (sha256=" + ("0" * 64) + ")\n"
            )
            self.assertFalse(_check_7a_spec_binding(repo, repo, "bootstrap", None, None, None)["passed"])

    def test_04_pre_colab_and_post_colab_cli_end_to_end(self):
        buckets_9 = ["A", "A", "B", "B", "C", "C", "D", "D", "D"]
        with tempfile.TemporaryDirectory() as tmp:
            # Build clean repo + ledger + literature fixture for pre-colab CLI
            lit_dir = os.path.join(tmp, "lit")
            cache_dir = os.path.join(tmp, "cache")
            md, bib, tr = _write_reading_files(lit_dir, cache_dir, buckets_9)
            spec_bytes = b"# Spec\nbody\n"
            digest = hashlib.sha256(spec_bytes).hexdigest()
            engine_ok = (
                "# -*- coding: utf-8 -*-\n"
                f"# SPEC_REF: docs/specs/spec_iter_1.md@v1.0 (sha256={digest})\n"
                "from dataclasses import dataclass\n"
                "@dataclass(frozen=True)\n"
                "class AlgorithmConfig:\n"
                "    lambda_param: float = 0.5\n"
                "    ablation_mask: tuple = (1, 1, 1)\n"
                "def step(x):\n"
                "    return x\n"
            )
            repo = os.path.join(tmp, "repo")
            spec_sha, impl_sha = _init_git_repo(repo, engine_ok, spec_bytes)
            ledger_path = os.path.join(tmp, "ledger.json")
            open(ledger_path, "w", encoding="utf-8").write(
                json.dumps(
                    {
                        "teams": {
                            "team_1": {
                                "transcript_path": tr,
                                "cursor_step": 2,
                                "cache_dir": cache_dir,
                                "reading_history_path": md,
                                "bib_path": bib,
                            }
                        }
                    }
                )
            )
            out_pre = os.path.join(tmp, "pre.json")
            rc = auditor_main(
                [
                    "--mode", "pre-colab", "--repo-root", repo, "--team", "1", "--iter", "1",
                    "--spec-sha", spec_sha, "--impl-sha", impl_sha, "--ledger", ledger_path, "--output-json", out_pre,
                ]
            )
            self.assertEqual(rc, 0, open(out_pre).read())
            # Also exercise bootstrap mode CLI
            self.assertEqual(auditor_main(["--mode", "bootstrap", "--repo-root", repo, "--spec-path", os.path.join(repo, "docs", "specs", "spec_iter_1.md")]), 0)

            # Post-colab Check #8 pass & fail cases
            res_dir = os.path.join(tmp, "res")
            drv_dir = os.path.join(tmp, "drv")
            os.makedirs(res_dir)
            os.makedirs(drv_dir)
            sha_lines = ["\n", "malformed_line\n"]
            for art in EXPECTED_ARTIFACTS:
                content = (
                    json.dumps({"question_id": 1}) + "\n\n" + json.dumps({"question_id": 2}) + "\n"
                    if art == "predictions.jsonl"
                    else f"data-{art}\n"
                ).encode("utf-8")
                open(os.path.join(res_dir, art), "wb").write(content)
                open(os.path.join(drv_dir, art), "wb").write(content)
                sha_lines.append(f"{hashlib.sha256(content).hexdigest()}  {art}\n")
            open(os.path.join(res_dir, "sha256sum.txt"), "w", encoding="utf-8").writelines(sha_lines)

            c_init = os.path.join(tmp, "c_init.json")
            c_curr = os.path.join(tmp, "c_curr.json")
            open(c_init, "w").write(json.dumps([{"id": "cell1"}]))
            open(c_curr, "w").write(json.dumps([{"id": "cell1"}]))

            out_post = os.path.join(tmp, "post.json")
            rc_post = auditor_main(
                [
                    "--mode", "post-colab", "--results-dir", res_dir, "--expected-qids", "2",
                    "--drive-dir", drv_dir, "--initial-cells-json", c_init, "--current-cells-json", c_curr,
                    "--output-json", out_post,
                ]
            )
            self.assertEqual(rc_post, 0, open(out_post).read())
            self.assertTrue(_check_8_post_colab(res_dir, 2)["passed"])

            # Fail cases for Check #8: wrong qid count, bad checksum, missing drive file, altered cells, missing artifact, missing sha256sum/predictions
            open(c_curr, "w").write(json.dumps([{"id": "cell1"}, {"id": "extra"}]))
            open(os.path.join(drv_dir, "checkpoint.json"), "wb").write(b"corrupt")
            open(os.path.join(res_dir, "evaluation.log"), "wb").write(b"tampered")
            os.remove(os.path.join(res_dir, "config_used.json"))
            bad_post = _check_8_post_colab(res_dir, 5, drv_dir, c_init, c_curr)
            self.assertFalse(bad_post["passed"])
            self.assertGreaterEqual(len(bad_post["errors"]), 4)

            os.remove(os.path.join(res_dir, "predictions.jsonl"))
            os.remove(os.path.join(res_dir, "sha256sum.txt"))
            self.assertFalse(_check_8_post_colab(res_dir, 2)["passed"])

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
            open(os.path.join(hb_dir, "1.json"), "w").write(
                json.dumps({"worker_id": "w1", "bytes_emitted": 100, "child_pid": os.getpid()})
            )
            open(os.path.join(hb_dir, "2.json"), "w").write(
                json.dumps({"worker_id": "w1", "bytes_emitted": 50, "child_pid": None})
            )
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
            # Initial tick at t=0 without --output-json and with missing heartbeat_dir branch
            self.assertEqual(liveness_main(["--ledger", self._write_json(tmp, "l1.json", ledger), "--snapshot", snap_path, "--heartbeat-dir", os.path.join(tmp, "no_hb"), "--now-epoch", "1000"]), 0)
            # Second tick with real hb_dir and --output-json
            self.assertEqual(liveness_main(["--ledger", self._write_json(tmp, "l2.json", ledger), "--snapshot", snap_path, "--heartbeat-dir", hb_dir, "--now-epoch", "1000", "--output-json", os.path.join(tmp, "out1.json")]), 0)
            snap = json.load(open(snap_path))

            # Advance transcript + colab qids at t=1000 + 4500 (75 min) -> ACTIVE_PROGRESS
            open(tr, "a", encoding="utf-8").write(
                json.dumps({"step_index": 3, "status": "RUNNING", "tool_calls": [{"name": "run_command"}]}) + "\n"
            )
            open(ckpt, "w").write(json.dumps({"total_completed": 25}))
            rep_75, snap2 = evaluate_liveness(ledger, snap, hb_dir, now_epoch=5500.0)
            self.assertEqual(rep_75["workers"]["w1"]["classification"], "ACTIVE_PROGRESS")
            self.assertEqual(rep_75["workers"]["w2"]["classification"], "LEVEL_2_SESSION_FREEZE")

            # Now freeze w1 for 1000s -> MONITORING; 1800s with live child_pid -> LEVEL_1_SUBPROCESS_HANG
            rep_mon, _ = evaluate_liveness(ledger, snap2, hb_dir, now_epoch=6500.0)
            self.assertEqual(rep_mon["workers"]["w1"]["classification"], "MONITORING")
            rep_l1, _ = evaluate_liveness(ledger, snap2, hb_dir, now_epoch=7300.0)
            self.assertEqual(rep_l1["workers"]["w1"]["classification"], "LEVEL_1_SUBPROCESS_HANG")

            # Remove live child_pid -> at 3600s -> LEVEL_2_SESSION_FREEZE
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
