#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated Binary Gatekeeper Runner (scripts/audit/run_auditor_checks.py).

Modes:
- bootstrap: Runs Checks #2, #3, #4, #5, #6, #7a against target ref (skipping Check #1)
- pre-colab: Runs Checks #1, #2, #3, #4, #5, #6, #7a against --impl-sha unpacked via git archive
- post-colab: Runs Check #8 verifying telemetry, sha256sum.txt, UTF-8/null-byte lint,
  Drive upload receipt, notebook cell restoration, and Stage-A-veto vs Stage-B-complete artifacts.
"""

import argparse
import ast
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, Dict, List, Optional, Set, Tuple

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:  # pragma: no cover
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_BASE_REF = "base/frsa-011-frozen"
DEFAULT_LOCK_PATH = os.path.join(os.path.expanduser("~"), ".gemini", "local_exp.lock")

PINNED_LF_SHA256 = {
    "scripts/benchmark/metrics.py": "7908ca075862694dab1e959ad0b2e170ca70bab850243f450a651d36bb0b7cb1",
    "tests/test_metrics_relaxation.py": "71300eb40214a16a574369ce9e4a7063f16d7428e1f18d713593432a96ef7092",
    "src/nevfe/core/frozen_foveation.py": "fac7eb19fc6805bb1d4cfbafee73f22b76dd5e57deebcc78477998067a9feacf",
}

ALLOWED_NUMERIC_LITERALS = {0, 0.0, 0.5, 1, 1.0, -1, 2, 2.0, 1e-6, 1e-8, 1e-12, -1e9}
RANK_AXIS_INTS = {-4, -3, -2, -1, 0, 1, 2, 3, 4}
AXIS_CALL_NAMES = {"permute", "transpose", "unsqueeze", "view"}
FORBIDDEN_IMPORTS = {"re", "difflib", "unicodedata", "pytesseract", "paddleocr", "easyocr"}
FORBIDDEN_STR_METHODS = {"replace", "strip", "lstrip", "rstrip"}
FORBIDDEN_GT_KEYS = {"answer", "gt", "verified", "gt_answers", "verified_all_answers"}
FORBIDDEN_BACKBONE_INTS = {151936, 151643, 151667, 896, 256, 448, 64, 24, 14}
TRACK2_FORBIDDEN_CALLS = {"clone_kv_cache", "_clone_kv_cache", "rollout_fn", "beam_search"}
EXPECTED_ARTIFACTS = [
    "summary_metrics.json",
    "predictions.jsonl",
    "per_question_type.csv",
    "pareto_latency_f1.csv",
    "qualitative_gallery_examples.json",
    "bbox_diagnostics.jsonl",
    "checkpoint.json",
    "evaluation.log",
    "config_used.json",
]
STAGE_A_VETO_ARTIFACTS = [
    "probe_ablation_summary.json",
    "colab_run_telemetry.json",
]


def _lf_sha256(path: str) -> str:
    with open(path, "rb") as f:
        raw = f.read().replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest().lower()


def _norm_ws(text: str) -> str:
    return " ".join((text or "").lower().split())


def _load_pinned_hashes(
    repo_root: str,
    pinned_sha256_json: Optional[Any] = None,
) -> Dict[str, str]:
    merged = dict(PINNED_LF_SHA256)
    defaults_path = os.path.join(repo_root, "configs", "tracks", "workflow_defaults.json")
    if os.path.isfile(defaults_path):
        try:
            with open(defaults_path, "r", encoding="utf-8") as df:
                data = json.load(df)
            if isinstance(data.get("pinned_sha256"), dict):
                merged.update({str(k): str(v).lower() for k, v in data["pinned_sha256"].items()})
        except Exception:
            pass
    if isinstance(pinned_sha256_json, dict):
        merged.update({str(k): str(v).lower() for k, v in pinned_sha256_json.items()})
    elif isinstance(pinned_sha256_json, str) and os.path.isfile(pinned_sha256_json):
        with open(pinned_sha256_json, "r", encoding="utf-8") as pf:
            override = json.load(pf)
        if isinstance(override, dict):
            merged.update({str(k): str(v).lower() for k, v in override.items()})
    return merged


def _unpack_git_archive(repo_root: str, sha: str, dest_dir: str) -> None:
    proc = subprocess.run(
        ["git", "-C", repo_root, "archive", "--format=tar", sha],
        capture_output=True,
        check=True,
    )
    with tarfile.open(fileobj=io.BytesIO(proc.stdout), mode="r:") as tf:
        if sys.version_info >= (3, 12):  # pragma: no branch
            tf.extractall(dest_dir, filter="data")
        else:  # pragma: no cover
            tf.extractall(dest_dir)


def _is_rank_or_axis_node(node: ast.AST, parent_map: Dict[ast.AST, ast.AST]) -> bool:
    parent = parent_map.get(node)
    if isinstance(parent, ast.UnaryOp) and isinstance(parent.op, ast.USub):
        parent = parent_map.get(parent)
    if isinstance(parent, ast.keyword) and parent.arg == "dim":
        return True
    if isinstance(parent, ast.Tuple):
        grand = parent_map.get(parent)
        if isinstance(grand, ast.keyword) and grand.arg == "dim":
            return True
        if isinstance(grand, ast.Call) and isinstance(grand.func, ast.Attribute) and grand.func.attr in AXIS_CALL_NAMES:
            return True
    if isinstance(parent, ast.Call) and isinstance(parent.func, ast.Attribute) and parent.func.attr in AXIS_CALL_NAMES:
        return True
    if isinstance(parent, ast.Compare):
        left = parent.left
        if isinstance(left, ast.Call) and isinstance(left.func, ast.Attribute) and left.func.attr == "dim":
            return True
    return False


def _added_lines_by_file(
    repo_root: str,
    base_ref: str,
    target_sha: str,
    paths: List[str],
) -> Tuple[Optional[Dict[str, Set[int]]], Optional[str]]:
    """Return {rel_path: set(1-based added line numbers)} between base_ref and target_sha."""
    cmd = ["git", "-C", repo_root, "diff", "-U0", base_ref]
    if target_sha and target_sha != "HEAD":
        cmd.append(target_sha)
    else:
        cmd.append("HEAD")
    cmd.append("--")
    cmd.extend(paths)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return None, f"git diff failed (rc={proc.returncode}) against {base_ref}: {proc.stderr.strip()}"

    added: Dict[str, Set[int]] = {}
    cur_file: Optional[str] = None
    for line in proc.stdout.splitlines():
        if line.startswith("+++ "):
            target_p = line[4:].strip()
            if target_p == "/dev/null":
                cur_file = None
            elif target_p.startswith("b/"):
                cur_file = target_p[2:]
            else:
                cur_file = target_p
        elif line.startswith("@@ ") and cur_file is not None:
            m = re.search(r"\+(\d+)(?:,(\d+))?", line)
            if m:
                start = int(m.group(1))
                count = int(m.group(2)) if m.group(2) is not None else 1
                if count > 0:
                    bucket = added.setdefault(cur_file, set())
                    for ln in range(start, start + count):
                        bucket.add(ln)
    return added, None


def _scan_heuristics_on_ast(
    tree: ast.AST,
    line_filter: Optional[Set[int]] = None,
    rel_label: str = "",
) -> List[str]:
    errors: List[str] = []
    for node in ast.walk(tree):
        lineno = getattr(node, "lineno", None)
        if line_filter is not None and lineno not in line_filter:
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name.split(".")[0] for alias in node.names]
            if isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module.split(".")[0])
            for nm in names:
                if nm in FORBIDDEN_IMPORTS:
                    errors.append(f"Forbidden heuristic/OCR import {nm} in {rel_label}:{lineno}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in FORBIDDEN_STR_METHODS:
                errors.append(f"Forbidden string heuristic .{node.func.attr}() in {rel_label}:{lineno}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in FORBIDDEN_GT_KEYS:
                errors.append(f"Forbidden ground-truth key literal {node.value!r} in {rel_label}:{lineno}")
    return errors


def _extract_const_num(node: Optional[ast.AST]) -> Optional[float]:
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _extract_const_num(node.operand)
        return -inner if inner is not None else None
    return None


def _check_1_paper_verification(
    paper_verification_json: Optional[str],
    repo_root: str,
) -> Dict[str, Any]:
    errors: List[str] = []
    if not paper_verification_json or not os.path.isfile(paper_verification_json):
        return {"passed": False, "paper_count": 0, "errors": [f"Missing paper verification JSON: {paper_verification_json}"]}

    with open(paper_verification_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    entries = data.get("papers") if isinstance(data, dict) else data
    if not isinstance(entries, list) or not (1 <= len(entries) <= 3):
        return {
            "passed": False,
            "paper_count": len(entries) if isinstance(entries, list) else 0,
            "errors": [f"core_paper_verification.json must list 1..3 core papers, got {type(entries).__name__}"],
        }

    for idx, entry in enumerate(entries):
        pid = entry.get("paper_id") or entry.get("id") or f"paper_{idx}"
        status = str(entry.get("status", ""))
        if status not in {"VERIFIED_TEX", "VERIFIED_PDF", "UNAVAILABLE_NEEDS_WAIVER"}:
            errors.append(f"Invalid verification status {status!r} for {pid}")
            continue
        if status == "UNAVAILABLE_NEEDS_WAIVER":
            waiver = str(entry.get("waiver_path") or "").strip()
            w_abs = waiver if os.path.isabs(waiver) else os.path.join(repo_root, waiver) if waiver else ""
            if not waiver or not os.path.isfile(w_abs) or os.path.getsize(w_abs) == 0:
                errors.append(f"{pid} has UNAVAILABLE_NEEDS_WAIVER without valid non-empty waiver_path")
            continue

        cached_raw = str(entry.get("cached_path") or "").strip()
        cached_abs = cached_raw if os.path.isabs(cached_raw) else os.path.join(repo_root, cached_raw) if cached_raw else ""
        if not cached_raw or not os.path.isfile(cached_abs):
            errors.append(f"Missing cached paper source for {pid}: {cached_raw!r}")
            continue

        expected_sha = str(entry.get("source_sha256") or "").lower().strip()
        with open(cached_abs, "rb") as cf:
            raw_bytes = cf.read()
        actual_sha = hashlib.sha256(raw_bytes).hexdigest().lower()
        if not expected_sha or actual_sha != expected_sha:
            errors.append(f"Source SHA-256 mismatch for {pid}: expected={expected_sha} actual={actual_sha}")

        text_norm = _norm_ws(raw_bytes.decode("utf-8", errors="replace"))
        section = str(entry.get("section") or entry.get("target_section") or "").strip()
        anchor = str(entry.get("anchor") or entry.get("equation_anchor") or "").strip()
        if not section or _norm_ws(section) not in text_norm:
            errors.append(f"Section {section!r} not found in cached source for {pid}")
        if anchor and _norm_ws(anchor) not in text_norm:
            errors.append(f"Anchor {anchor!r} not found in cached source for {pid}")

    return {"passed": len(errors) == 0, "paper_count": len(entries), "errors": errors}


def _check_2_and_3_and_5(
    core_dir: str,
    repo_root: Optional[str] = None,
    base_ref: str = DEFAULT_BASE_REF,
    target_sha: str = "HEAD",
    snapshot_dir: Optional[str] = None,
    pinned_sha256: Optional[Dict[str, str]] = None,
    track: int = 1,
) -> Dict[str, Any]:
    entries = {e for e in os.listdir(core_dir) if e != "__pycache__"}
    check2_errors: List[str] = []
    check3_errors: List[str] = []
    check5_errors: List[str] = []

    allowed_core = {"__init__.py", "frozen_foveation.py", "engine.py"}
    if entries != allowed_core:
        check3_errors.append(
            f"src/nevfe/core/ must contain only {sorted(allowed_core)}, got {sorted(entries)}"
        )

    frozen_path = os.path.join(core_dir, "frozen_foveation.py")
    if not os.path.isfile(frozen_path):
        check3_errors.append("Missing immutable src/nevfe/core/frozen_foveation.py")
    else:
        if pinned_sha256 and "src/nevfe/core/frozen_foveation.py" in pinned_sha256:
            exp_frov = pinned_sha256["src/nevfe/core/frozen_foveation.py"]
            act_frov = _lf_sha256(frozen_path)
            if act_frov != exp_frov:
                check3_errors.append(
                    f"frozen_foveation.py SHA-256 mismatch: expected={exp_frov} actual={act_frov}"
                )
        if repo_root:
            fdiff = subprocess.run(
                ["git", "-C", repo_root, "diff", base_ref, "--", "src/nevfe/core/frozen_foveation.py"],
                capture_output=True,
                text=True,
                check=False,
            )
            if fdiff.returncode != 0:
                check3_errors.append(
                    f"git diff failed on frozen_foveation.py (rc={fdiff.returncode}): {fdiff.stderr.strip()}"
                )
            elif fdiff.stdout.strip():
                check3_errors.append("frozen_foveation.py differs from base_ref")

    engine_path = os.path.join(core_dir, "engine.py")
    src = open(engine_path, "r", encoding="utf-8").read()
    tree = ast.parse(src, filename=engine_path)

    parent_map: Dict[ast.AST, ast.AST] = {}
    for p in ast.walk(tree):
        for ch in ast.iter_child_nodes(p):
            parent_map[ch] = p

    # Check #2: AlgorithmConfig locks lambda_param=0.50, beta_param=0.05, and declares <= 2 new scalars (<= 4 total)
    algo_cls = next(
        (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AlgorithmConfig"),
        None,
    )
    if algo_cls is None:
        check2_errors.append("Missing AlgorithmConfig class in engine.py")
    else:
        field_defaults: Dict[str, Optional[float]] = {}
        tunable_fields: List[str] = []
        for stmt in algo_cls.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                fname = stmt.target.id
                if fname != "ablation_mask":
                    tunable_fields.append(fname)
                    field_defaults[fname] = _extract_const_num(stmt.value)
        if len(tunable_fields) > 4:
            check2_errors.append(
                f"AlgorithmConfig has {len(tunable_fields)} scalar fields (> 4, i.e. > 2 new scalars): {tunable_fields}"
            )
        if "lambda_param" not in field_defaults or abs((field_defaults["lambda_param"] or 0.0) - 0.50) > 1e-12:
            check2_errors.append(
                f"AlgorithmConfig.lambda_param must remain locked at 0.50, got {field_defaults.get('lambda_param')}"
            )
        if "beta_param" not in field_defaults or abs((field_defaults["beta_param"] or 0.0) - 0.05) > 1e-12:
            check2_errors.append(
                f"AlgorithmConfig.beta_param must remain locked at 0.05, got {field_defaults.get('beta_param')}"
            )

    # Count non-docstring AST statement lines in engine.py
    stmt_lines: Set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.stmt):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                continue
            stmt_lines.add(node.lineno)
    if len(stmt_lines) > 250:
        check3_errors.append(f"engine.py has {len(stmt_lines)} non-docstring AST statement lines (> 250)")

    # Scan Check #3 (heuristics/OCR/GT keys) & Check #5 (backbone ints) across src/nevfe/core/
    for core_file in sorted(entries):
        if not core_file.endswith(".py"):
            continue
        cf_path = os.path.join(core_dir, core_file)
        cf_src = open(cf_path, "r", encoding="utf-8").read()
        cf_tree = tree if core_file == "engine.py" else ast.parse(cf_src, filename=cf_path)
        check3_errors.extend(_scan_heuristics_on_ast(cf_tree, line_filter=None, rel_label=f"src/nevfe/core/{core_file}"))
        for node in ast.walk(cf_tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
                if node.value in FORBIDDEN_BACKBONE_INTS:
                    check5_errors.append(
                        f"Forbidden backbone dimension literal {node.value} at src/nevfe/core/{core_file}:{node.lineno}"
                    )

    # Numeric literal scan strictly outside AlgorithmConfig in engine.py
    for top in tree.body:
        if isinstance(top, ast.ClassDef) and top.name == "AlgorithmConfig":
            continue
        for node in ast.walk(top):
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                if isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, (int, float)):
                    val = -node.operand.value
                    if val not in ALLOWED_NUMERIC_LITERALS and not (
                        isinstance(val, int) and val in RANK_AXIS_INTS and _is_rank_or_axis_node(node.operand, parent_map)
                    ):
                        check2_errors.append(f"Disallowed numeric literal {val} at line {node.lineno}")
                continue
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                parent = parent_map.get(node)
                if isinstance(parent, ast.UnaryOp) and isinstance(parent.op, ast.USub):
                    continue
                val = node.value
                if val in ALLOWED_NUMERIC_LITERALS:
                    continue
                if isinstance(val, int) and val in RANK_AXIS_INTS and _is_rank_or_axis_node(node, parent_map):
                    continue
                check2_errors.append(f"Disallowed numeric literal {val} at line {node.lineno}")

    # AST-diff-filtered scan on adapter/ and pipeline.py (Check #3 heuristics, Check #5 layer indexing, Track 2 static bans)
    if repo_root and snapshot_dir:
        added_map, diff_err = _added_lines_by_file(
            repo_root,
            base_ref,
            target_sha,
            ["src/nevfe/adapter", "src/nevfe/pipeline.py", "src/nevfe/core/engine.py"],
        )
        if diff_err:
            check3_errors.append(diff_err)
        elif added_map:
            for rel_p, line_set in sorted(added_map.items()):
                if not rel_p.endswith(".py") or not line_set:
                    continue
                abs_p = os.path.join(snapshot_dir, rel_p)
                if not os.path.isfile(abs_p):
                    continue
                mod_src = open(abs_p, "r", encoding="utf-8").read()
                mod_tree = ast.parse(mod_src, filename=abs_p)
                if rel_p.startswith("src/nevfe/adapter/") or rel_p == "src/nevfe/pipeline.py":
                    check3_errors.extend(_scan_heuristics_on_ast(mod_tree, line_filter=line_set, rel_label=rel_p))

                if rel_p == "src/nevfe/adapter/vintern_adapter.py":
                    for node in ast.walk(mod_tree):
                        lineno = getattr(node, "lineno", None)
                        if lineno not in line_set:
                            continue
                        if isinstance(node, ast.Subscript):
                            sl = node.slice
                            if isinstance(sl, ast.Constant) and isinstance(sl.value, int) and 12 <= sl.value <= 23:
                                val_txt = ast.unparse(node.value) if hasattr(ast, "unparse") else ""
                                if "layer" in val_txt.lower():
                                    check5_errors.append(
                                        f"Hardcoded decoder layer index {sl.value} in {rel_p}:{lineno}; use resolve_relative_decoder_layer_indices"
                                    )
                        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range":
                            nums = [_extract_const_num(a) for a in node.args]
                            if any(n is not None and 12 <= int(n) <= 24 for n in nums):
                                check5_errors.append(
                                    f"Hardcoded decoder layer range {nums} in {rel_p}:{lineno}; use resolve_relative_decoder_layer_indices"
                                )

                if int(track) == 2:
                    for node in ast.walk(mod_tree):
                        lineno = getattr(node, "lineno", None)
                        if lineno not in line_set:
                            continue
                        if isinstance(node, ast.Call):
                            cname = (
                                node.func.id
                                if isinstance(node.func, ast.Name)
                                else (node.func.attr if isinstance(node.func, ast.Attribute) else "")
                            )
                            if cname in TRACK2_FORBIDDEN_CALLS:
                                check5_errors.append(
                                    f"Track 2 single-step lock forbids added {cname}() call at {rel_p}:{lineno}"
                                )

    return {
        "check2": {"passed": len(check2_errors) == 0, "errors": check2_errors},
        "check3": {"passed": len(check3_errors) == 0, "ast_stmt_lines": len(stmt_lines), "errors": check3_errors},
        "check5": {"passed": len(check5_errors) == 0, "errors": check5_errors},
    }


def _check_4_metrics_lock(
    repo_root: str,
    base_ref: str = DEFAULT_BASE_REF,
    pinned_sha256: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    diff_proc = subprocess.run(
        ["git", "-C", repo_root, "diff", base_ref, "--", "scripts/benchmark/metrics.py", "tests/test_metrics_relaxation.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    test_proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_metrics_relaxation.py", "-q"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    errors: List[str] = []
    if diff_proc.returncode != 0:
        errors.append(f"git diff failed on metrics lock (rc={diff_proc.returncode}): {diff_proc.stderr.strip()}")
    if diff_proc.stdout.strip():
        errors.append(f"metrics.py or test_metrics_relaxation.py differs from {base_ref}")
    if pinned_sha256:
        for rel_p in ("scripts/benchmark/metrics.py", "tests/test_metrics_relaxation.py"):
            if rel_p in pinned_sha256:
                full_p = os.path.join(repo_root, rel_p)
                if not os.path.isfile(full_p):
                    errors.append(f"Missing immutable file {rel_p}")
                else:
                    act_sha = _lf_sha256(full_p)
                    if act_sha != pinned_sha256[rel_p]:
                        errors.append(f"Pinned SHA-256 mismatch for {rel_p}: expected={pinned_sha256[rel_p]} actual={act_sha}")
    if test_proc.returncode != 0:
        errors.append(f"test_metrics_relaxation.py failed: {test_proc.stdout}\n{test_proc.stderr}")
    return {"passed": len(errors) == 0, "errors": errors}


def _check_6_guarded_pytest(
    snapshot_dir: str,
    repo_root: Optional[str] = None,
    track: int = 1,
    lock_path: Optional[str] = None,
    receipt_dir: Optional[str] = None,
    cal_bank_dir: Optional[str] = None,
) -> Dict[str, Any]:
    resolved_lock = lock_path or os.path.join(snapshot_dir, ".snapshot_exp.lock")
    hb_dir = os.path.join(receipt_dir, "heartbeats") if receipt_dir else os.path.join(snapshot_dir, ".snapshot_hb")
    unit_receipt = os.path.join(receipt_dir, f"check6_unit_track_{track}.json") if receipt_dir else os.path.join(snapshot_dir, ".check6_unit_receipt.json")

    pytest_targets = ["tests/test_nevfe_engine.py"]
    if os.path.isfile(os.path.join(snapshot_dir, "tests", "test_frsa011_base_parity.py")):
        pytest_targets.append("tests/test_frsa011_base_parity.py")
    if os.path.isfile(os.path.join(snapshot_dir, "tests", "test_track_candidate.py")):
        pytest_targets.append("tests/test_track_candidate.py")

    proc = subprocess.run(
        [
            sys.executable,
            os.path.join(snapshot_dir, "scripts", "local", "run_guarded.py"),
            "--lock-path",
            resolved_lock,
            "--heartbeat-dir",
            hb_dir,
            "--receipt",
            unit_receipt,
            "--",
            sys.executable,
            "-m",
            "pytest",
            *pytest_targets,
            "-q",
        ],
        cwd=snapshot_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    errors: List[str] = []
    if proc.returncode != 0:
        errors.append(f"Guarded pytest failed (rc={proc.returncode}): {proc.stdout}\n{proc.stderr}")

    cal_eval_metrics: Optional[Dict[str, Any]] = None
    if cal_bank_dir:
        base_root = repo_root or snapshot_dir
        abs_bank_dir = cal_bank_dir if os.path.isabs(cal_bank_dir) else os.path.join(base_root, cal_bank_dir)
        cal_receipt = os.path.join(receipt_dir, f"check6_calbank_track_{track}.json") if receipt_dir else os.path.join(snapshot_dir, ".check6_cal_receipt.json")
        cal_env = os.environ.copy()
        cal_env["PYTHONPATH"] = snapshot_dir
        cal_proc = subprocess.run(
            [
                sys.executable,
                os.path.join(snapshot_dir, "scripts", "local", "run_guarded.py"),
                "--lock-path",
                resolved_lock,
                "--heartbeat-dir",
                hb_dir,
                "--receipt",
                cal_receipt,
                "--",
                sys.executable,
                os.path.join(snapshot_dir, "scripts", "benchmark", "export_calibration_bank.py"),
                "--eval-only",
                "--bank-version",
                "2",
                "--output-dir",
                abs_bank_dir,
                "--track",
                str(track),
            ],
            cwd=snapshot_dir,
            env=cal_env,
            capture_output=True,
            text=True,
            check=False,
        )
        if cal_proc.returncode != 0:
            errors.append(f"Calibration Bank v2 eval failed (rc={cal_proc.returncode}): {cal_proc.stdout}\n{cal_proc.stderr}")
        else:
            for line in cal_proc.stdout.splitlines():
                if "CPU eval:" in line:
                    try:
                        cal_eval_metrics = json.loads(line.split("CPU eval:", 1)[1].strip())
                    except Exception:
                        pass
            if not cal_eval_metrics:
                errors.append("Calibration Bank v2 eval produced no parseable CPU eval metrics")
            else:
                if float(cal_eval_metrics.get("finite_ratio", 0.0)) < 1.0:
                    errors.append(f"Calibration Bank v2 finite_ratio < 1.0: {cal_eval_metrics.get('finite_ratio')}")
                if int(cal_eval_metrics.get("nan_count", 0)) > 0 or int(cal_eval_metrics.get("inf_count", 0)) > 0:
                    errors.append(f"Calibration Bank v2 non-finite outputs: {cal_eval_metrics}")
                if float(cal_eval_metrics.get("max_kv_diff", 1.0)) != 0.0:
                    errors.append(f"Calibration Bank v2 max_kv_diff != 0.0: {cal_eval_metrics.get('max_kv_diff')}")

    return {"passed": len(errors) == 0, "cal_bank_eval": cal_eval_metrics, "errors": errors}


def _check_7a_spec_binding(
    snapshot_dir: str,
    repo_root: str,
    mode: str,
    spec_sha: Optional[str],
    impl_sha: Optional[str],
    spec_path_override: Optional[str],
) -> Dict[str, Any]:
    errors: List[str] = []
    engine_path = os.path.join(snapshot_dir, "src", "nevfe", "core", "engine.py")
    lines = open(engine_path, "r", encoding="utf-8").read().splitlines()
    header_line = lines[1] if len(lines) > 1 else ""
    m = re.match(r"^#\s*SPEC_REF:\s*(\S+@v[\d.]+\s+\(sha256=([0-9a-f]{64})\))\s*$", header_line)
    if not m:
        errors.append(f"Line 2 missing valid '# SPEC_REF: <path>@v<ver> (sha256=<hex>)', got: {header_line!r}")
    else:
        full_ref = m.group(1)
        rel_spec = full_ref.split("@v", 1)[0]
        expected_digest = m.group(2)
        resolved_spec = spec_path_override or os.path.join(snapshot_dir, rel_spec)
        if not os.path.isfile(resolved_spec):
            errors.append(f"Referenced spec file not found: {resolved_spec}")
        else:
            actual_digest = _lf_sha256(resolved_spec)
            if actual_digest != expected_digest:
                errors.append(f"SPEC_REF sha256 mismatch: expected {expected_digest}, got {actual_digest}")

    if mode == "pre-colab" and spec_sha and impl_sha:
        diff_proc = subprocess.run(
            ["git", "-C", repo_root, "diff", f"{spec_sha}..{impl_sha}", "--", "docs/specs/"],
            capture_output=True,
            text=True,
            check=False,
        )
        if diff_proc.returncode != 0:
            errors.append(f"git diff on docs/specs/ failed (rc={diff_proc.returncode}): {diff_proc.stderr.strip()}")
        elif diff_proc.stdout.strip():
            errors.append("Implementer modified docs/specs/ between spec_sha and impl_sha")

    return {"passed": len(errors) == 0, "errors": errors}


def _check_8_post_colab(
    results_dir: str,
    expected_qids: int,
    drive_dir: Optional[str] = None,
    initial_cells_json: Optional[str] = None,
    current_cells_json: Optional[str] = None,
    track: int = 1,
    stage_a_vetoed: Optional[bool] = None,
) -> Dict[str, Any]:
    errors: List[str] = []
    if not results_dir or not os.path.isdir(results_dir):
        return {"passed": False, "errors": [f"Results directory not found: {results_dir}"]}

    # UTF-8 / zero-null-byte lint on all emitted .md and .json artifacts
    for root, _, files in os.walk(results_dir):
        for fname in sorted(files):
            if not (fname.endswith(".md") or fname.endswith(".json")):
                continue
            fpath = os.path.join(root, fname)
            try:
                raw = open(fpath, "rb").read()
            except OSError as exc:
                errors.append(f"Cannot read artifact {fname}: {exc}")
                continue
            if b"\x00" in raw:
                errors.append(f"Null byte detected in text artifact: {fname}")
                continue
            try:
                txt = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                errors.append(f"Invalid UTF-8 in artifact {fname}: {exc}")
                continue
            if fname.endswith(".json"):
                try:
                    json.loads(txt)
                except json.JSONDecodeError as exc:
                    errors.append(f"Malformed JSON in {fname}: {exc}")

    # Auto-detect Stage-A-veto when flag omitted
    resolved_vetoed = stage_a_vetoed
    probe_sum_path = os.path.join(results_dir, "probe_ablation_summary.json")
    if resolved_vetoed is None:
        if os.path.isfile(os.path.join(results_dir, "STAGE_A_VETO")):
            resolved_vetoed = True
        elif os.path.isfile(probe_sum_path) and not os.path.isfile(os.path.join(results_dir, "predictions.jsonl")):
            try:
                p_data = json.load(open(probe_sum_path, "r", encoding="utf-8"))
                resolved_vetoed = not bool(p_data.get("pass_stage_a", False))
            except Exception:
                resolved_vetoed = False
        else:
            resolved_vetoed = False

    # Validate colab_run_telemetry.json when present (authoritative for Track 2 invariants)
    telemetry_path = os.path.join(results_dir, "colab_run_telemetry.json")
    if os.path.isfile(telemetry_path):
        try:
            tel = json.load(open(telemetry_path, "r", encoding="utf-8"))
            if int(track) == 2:
                fwd_max = float(tel.get("decoder_forward_calls_per_step_max", 999.0))
                kv_diff = float(tel.get("max_kv_diff", -1.0))
                if fwd_max > 1.0 + 1e-9:
                    errors.append(f"Track 2 single-step violation in telemetry: decoder_forward_calls_per_step_max={fwd_max}")
                if kv_diff != 0.0:
                    errors.append(f"Track 2 zero-KV invariant violation in telemetry: max_kv_diff={kv_diff}")
        except Exception as exc:
            errors.append(f"Failed parsing colab_run_telemetry.json: {exc}")
    elif resolved_vetoed:
        errors.append("Missing colab_run_telemetry.json for Stage-A-vetoed wave")

    # Validate drive_upload_receipt.json when present
    receipt_path = os.path.join(results_dir, "drive_upload_receipt.json")
    if os.path.isfile(receipt_path):
        try:
            rec = json.load(open(receipt_path, "r", encoding="utf-8"))
            if not rec.get("folder_id") or int(rec.get("uploaded_count", 0)) <= 0:
                errors.append("Invalid drive_upload_receipt.json: missing folder_id or uploaded_count <= 0")
        except Exception as exc:
            errors.append(f"Malformed drive_upload_receipt.json: {exc}")

    sha_file = os.path.join(results_dir, "sha256sum.txt")
    if not os.path.isfile(sha_file):
        errors.append(f"Missing sha256sum.txt in {results_dir}")
        checksums: Dict[str, str] = {}
    else:
        checksums = {}
        for line in open(sha_file, "r", encoding="utf-8"):
            parts = line.strip().split()
            if len(parts) >= 2:
                checksums[os.path.basename(parts[1])] = parts[0].lower()

    if resolved_vetoed:
        required_artifacts = list(STAGE_A_VETO_ARTIFACTS)
    else:
        required_artifacts = list(EXPECTED_ARTIFACTS)
        pred_path = os.path.join(results_dir, "predictions.jsonl")
        if not os.path.isfile(pred_path):
            errors.append(f"Missing predictions.jsonl in {results_dir}")
        else:
            qids = set()
            for line in open(pred_path, "r", encoding="utf-8"):
                if line.strip():
                    qids.add(json.loads(line).get("question_id"))
            if len(qids) != expected_qids:
                errors.append(f"Expected {expected_qids} unique QIDs in predictions.jsonl, got {len(qids)}")

    for art in required_artifacts:
        art_path = os.path.join(results_dir, art)
        if not os.path.isfile(art_path):
            errors.append(f"Missing required artifact: {art}")
            continue
        digest = hashlib.sha256(open(art_path, "rb").read()).hexdigest().lower()
        if sha_file and os.path.isfile(sha_file) and checksums.get(art) != digest:
            errors.append(f"Checksum mismatch for {art}: manifest={checksums.get(art)} actual={digest}")
        if drive_dir:
            drv_path = os.path.join(drive_dir, art)
            if not os.path.isfile(drv_path) or hashlib.sha256(open(drv_path, "rb").read()).hexdigest().lower() != digest:
                errors.append(f"Drive sync mismatch or missing artifact: {drv_path}")

    if initial_cells_json and current_cells_json:
        init_ids = [c.get("id") for c in json.load(open(initial_cells_json, "r", encoding="utf-8"))]
        curr_ids = [c.get("id") for c in json.load(open(current_cells_json, "r", encoding="utf-8"))]
        if set(curr_ids) != set(init_ids):
            errors.append(f"Notebook cells altered: initial={init_ids} current={curr_ids}")

    return {"passed": len(errors) == 0, "stage_a_vetoed": bool(resolved_vetoed), "errors": errors}


def run_auditor(
    mode: str,
    repo_root: str = PROJECT_ROOT,
    track: int = 1,
    iter_k: int = 0,
    base_ref: str = DEFAULT_BASE_REF,
    spec_sha: Optional[str] = None,
    impl_sha: Optional[str] = None,
    paper_verification_json: Optional[str] = None,
    cal_bank_dir: Optional[str] = None,
    receipt_dir: Optional[str] = None,
    lock_path: Optional[str] = None,
    pinned_sha256_json: Optional[Any] = None,
    results_dir: Optional[str] = None,
    expected_qids: int = 1222,
    spec_path_override: Optional[str] = None,
    drive_dir: Optional[str] = None,
    initial_cells_json: Optional[str] = None,
    current_cells_json: Optional[str] = None,
    stage_a_vetoed: Optional[bool] = None,
    output_json: Optional[str] = None,
) -> Dict[str, Any]:
    pinned_hashes = _load_pinned_hashes(repo_root, pinned_sha256_json=pinned_sha256_json)
    checks: Dict[str, Any] = {}

    if mode in ("bootstrap", "pre-colab"):
        target_sha = impl_sha or "HEAD"
        with tempfile.TemporaryDirectory() as snap:
            _unpack_git_archive(repo_root, target_sha, snap)
            if mode == "pre-colab":
                checks["check1"] = _check_1_paper_verification(paper_verification_json, repo_root=repo_root)
            c235 = _check_2_and_3_and_5(
                os.path.join(snap, "src", "nevfe", "core"),
                repo_root=repo_root,
                base_ref=base_ref,
                target_sha=target_sha,
                snapshot_dir=snap,
                pinned_sha256=pinned_hashes,
                track=track,
            )
            checks.update(c235)
            checks["check4"] = _check_4_metrics_lock(
                repo_root,
                base_ref=base_ref,
                pinned_sha256=pinned_hashes,
            )
            checks["check6"] = _check_6_guarded_pytest(
                snap,
                repo_root=repo_root,
                track=track,
                lock_path=lock_path,
                receipt_dir=receipt_dir,
                cal_bank_dir=cal_bank_dir,
            )
            checks["check7a"] = _check_7a_spec_binding(
                snapshot_dir=snap,
                repo_root=repo_root,
                mode=mode,
                spec_sha=spec_sha,
                impl_sha=target_sha,
                spec_path_override=spec_path_override,
            )
    else:
        checks["check8"] = _check_8_post_colab(
            results_dir=results_dir or "",
            expected_qids=expected_qids,
            drive_dir=drive_dir,
            initial_cells_json=initial_cells_json,
            current_cells_json=current_cells_json,
            track=track,
            stage_a_vetoed=stage_a_vetoed,
        )

    passed = all(c.get("passed", False) for c in checks.values())
    report = {"mode": mode, "track": track, "iter": iter_k, "base_ref": base_ref, "passed": passed, "checks": checks}
    dest = output_json or os.path.join(repo_root, "reports", "audit", f"track_{track}_iter_{iter_k}.json")
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    with open(dest, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Automated 7-Binary + Post-Colab Auditor Gatekeeper.")
    parser.add_argument("--mode", required=True, choices=["bootstrap", "pre-colab", "post-colab"])
    parser.add_argument("--repo-root", default=PROJECT_ROOT)
    parser.add_argument("--track", type=int, choices=[1, 2], default=None)
    parser.add_argument("--team", type=int, default=None, help="Deprecated alias for --track")
    parser.add_argument("--iter", dest="iter_k", type=int, default=0)
    parser.add_argument("--base-ref", default=DEFAULT_BASE_REF)
    parser.add_argument("--spec-sha", default=None)
    parser.add_argument("--impl-sha", default=None)
    parser.add_argument("--paper-verification-json", default=None)
    parser.add_argument("--cal-bank-dir", default=None)
    parser.add_argument("--receipt-dir", default=None)
    parser.add_argument("--lock-path", default=DEFAULT_LOCK_PATH)
    parser.add_argument("--pinned-sha256-json", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--expected-qids", type=int, default=1222)
    parser.add_argument("--spec-path", default=None)
    parser.add_argument("--drive-dir", default=None)
    parser.add_argument("--initial-cells-json", default=None)
    parser.add_argument("--current-cells-json", default=None)
    parser.add_argument("--stage-a-vetoed", action="store_true", default=False)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args(argv)

    track_id = args.track if args.track is not None else (args.team if args.team in (1, 2) else 1)
    vetoed_arg: Optional[bool] = True if args.stage_a_vetoed else None

    report = run_auditor(
        mode=args.mode,
        repo_root=args.repo_root,
        track=track_id,
        iter_k=args.iter_k,
        base_ref=args.base_ref,
        spec_sha=args.spec_sha,
        impl_sha=args.impl_sha,
        paper_verification_json=args.paper_verification_json,
        cal_bank_dir=args.cal_bank_dir,
        receipt_dir=args.receipt_dir,
        lock_path=args.lock_path,
        pinned_sha256_json=args.pinned_sha256_json,
        results_dir=args.results_dir,
        expected_qids=args.expected_qids,
        spec_path_override=args.spec_path,
        drive_dir=args.drive_dir,
        initial_cells_json=args.initial_cells_json,
        current_cells_json=args.current_cells_json,
        stage_a_vetoed=vetoed_arg,
        output_json=args.output_json,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
