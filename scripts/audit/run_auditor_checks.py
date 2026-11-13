#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated Binary Gatekeeper Runner (scripts/audit/run_auditor_checks.py).

Modes:
- bootstrap: Runs Checks #2, #3, #4, #5, #6, #7a against HEAD / baseline_iter23.md
- pre-colab: Runs Checks #1, #2, #3, #4, #5, #6, #7a against --impl-sha unpacked via git archive
- post-colab: Runs Check #8 verifying QIDs, sha256sum.txt, Drive sync, and notebook cell cleanliness
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
from typing import Any, Dict, List, Optional, Set

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:  # pragma: no cover
    sys.path.insert(0, PROJECT_ROOT)

from scripts.audit.verify_transcript_pdf_reads import verify_literature_audit

ALLOWED_NUMERIC_LITERALS = {0, 0.0, 0.5, 1, 1.0, -1, 2, 2.0, 1e-6, 1e-8, 1e-12, -1e9}
RANK_AXIS_INTS = {-4, -3, -2, -1, 0, 1, 2, 3, 4}
AXIS_CALL_NAMES = {"permute", "transpose", "unsqueeze", "view"}
FORBIDDEN_IMPORTS = {"re", "difflib", "unicodedata", "pytesseract", "paddleocr", "easyocr"}
FORBIDDEN_STR_METHODS = {"replace", "strip", "lstrip", "rstrip"}
FORBIDDEN_GT_KEYS = {"answer", "gt", "verified", "gt_answers", "verified_all_answers"}
FORBIDDEN_BACKBONE_INTS = {151936, 151643, 151667, 896, 256, 448, 64, 24, 14}
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


def _check_2_and_3_and_5(core_dir: str) -> Dict[str, Any]:
    entries = {e for e in os.listdir(core_dir) if e != "__pycache__"}
    check3_errors: List[str] = []
    if entries != {"__init__.py", "engine.py"}:
        check3_errors.append(f"src/nevfe/core/ must contain only __init__.py and engine.py, got {sorted(entries)}")

    engine_path = os.path.join(core_dir, "engine.py")
    src = open(engine_path, "r", encoding="utf-8").read()
    tree = ast.parse(src, filename=engine_path)

    parent_map: Dict[ast.AST, ast.AST] = {}
    for p in ast.walk(tree):
        for ch in ast.iter_child_nodes(p):
            parent_map[ch] = p

    check2_errors: List[str] = []
    check5_errors: List[str] = []

    # Check AlgorithmConfig field count (<= 3 excluding ablation_mask)
    algo_cls = next(
        (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AlgorithmConfig"),
        None,
    )
    if algo_cls is None:
        check2_errors.append("Missing AlgorithmConfig class in engine.py")
    else:
        tunable_fields = [
            stmt.target.id
            for stmt in algo_cls.body
            if isinstance(stmt, ast.AnnAssign)
            and isinstance(stmt.target, ast.Name)
            and stmt.target.id != "ablation_mask"
        ]
        if len(tunable_fields) > 3:
            check2_errors.append(f"AlgorithmConfig has {len(tunable_fields)} tunable fields (> 3): {tunable_fields}")

    # Count non-docstring AST statement lines
    stmt_lines: Set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.stmt):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                continue
            stmt_lines.add(node.lineno)
    if len(stmt_lines) > 250:
        check3_errors.append(f"engine.py has {len(stmt_lines)} non-docstring AST statement lines (> 250)")

    # Scan Check #2 (literals outside AlgorithmConfig), Check #3 (heuristics/GT keys), Check #5 (backbone ints)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name.split(".")[0] for alias in node.names]
            if isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module.split(".")[0])
            for nm in names:
                if nm in FORBIDDEN_IMPORTS:
                    check3_errors.append(f"Forbidden heuristic/OCR import: {nm}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in FORBIDDEN_STR_METHODS:
                check3_errors.append(f"Forbidden string heuristic call .{node.func.attr}()")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in FORBIDDEN_GT_KEYS:
                check3_errors.append(f"Forbidden ground-truth key literal: {node.value!r}")

        # Backbone ints in entire engine.py
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            if node.value in FORBIDDEN_BACKBONE_INTS:
                check5_errors.append(f"Forbidden backbone dimension literal {node.value} at line {node.lineno}")

    # Numeric literal scan strictly on top-level operator functions and NEVFEEngine methods (excluding AlgorithmConfig)
    for top in tree.body:
        if isinstance(top, ast.ClassDef) and top.name == "AlgorithmConfig":
            continue
        for node in ast.walk(top):
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                if isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, (int, float)):
                    val = -node.operand.value
                    if val not in ALLOWED_NUMERIC_LITERALS and not (isinstance(val, int) and val in RANK_AXIS_INTS and _is_rank_or_axis_node(node.operand, parent_map)):
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

    return {
        "check2": {"passed": len(check2_errors) == 0, "errors": check2_errors},
        "check3": {"passed": len(check3_errors) == 0, "ast_stmt_lines": len(stmt_lines), "errors": check3_errors},
        "check5": {"passed": len(check5_errors) == 0, "errors": check5_errors},
    }


def _check_4_metrics_lock(repo_root: str) -> Dict[str, Any]:
    diff_proc = subprocess.run(
        ["git", "-C", repo_root, "diff", "iter23_staging", "--", "scripts/benchmark/metrics.py", "tests/test_metrics_relaxation.py"],
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
    if diff_proc.stdout.strip():
        errors.append("metrics.py or test_metrics_relaxation.py differs from iter23_staging")
    if test_proc.returncode != 0:
        errors.append(f"test_metrics_relaxation.py failed: {test_proc.stdout}\n{test_proc.stderr}")
    return {"passed": len(errors) == 0, "errors": errors}


def _check_6_guarded_pytest(snapshot_dir: str) -> Dict[str, Any]:
    lock_path = os.path.join(snapshot_dir, ".snapshot_exp.lock")
    hb_dir = os.path.join(snapshot_dir, ".snapshot_hb")
    proc = subprocess.run(
        [
            sys.executable,
            os.path.join(snapshot_dir, "scripts", "local", "run_guarded.py"),
            "--lock-path",
            lock_path,
            "--heartbeat-dir",
            hb_dir,
            "--",
            sys.executable,
            "-m",
            "pytest",
            "tests/test_nevfe_engine.py",
            "-q",
        ],
        cwd=snapshot_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    errors = [] if proc.returncode == 0 else [f"Guarded pytest failed (rc={proc.returncode}): {proc.stdout}\n{proc.stderr}"]
    return {"passed": proc.returncode == 0, "errors": errors}


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
    m = re.match(r"^#\s*SPEC_REF:\s*(\S+)@v[\d.]+\s+\(sha256=([0-9a-f]{64})\)\s*$", header_line)
    if not m:
        errors.append(f"Line 2 missing valid '# SPEC_REF: <path>@v<ver> (sha256=<hex>)', got: {header_line!r}")
    else:
        rel_spec, expected_digest = m.group(1), m.group(2)
        resolved_spec = spec_path_override or os.path.join(snapshot_dir, rel_spec)
        if not os.path.isfile(resolved_spec):
            errors.append(f"Referenced spec file not found: {resolved_spec}")
        else:
            norm_bytes = open(resolved_spec, "rb").read().replace(b"\r\n", b"\n")
            actual_digest = hashlib.sha256(norm_bytes).hexdigest()
            if actual_digest != expected_digest:
                errors.append(f"SPEC_REF sha256 mismatch: expected {expected_digest}, got {actual_digest}")

    if mode == "pre-colab" and spec_sha and impl_sha:
        diff_proc = subprocess.run(
            ["git", "-C", repo_root, "diff", f"{spec_sha}..{impl_sha}", "--", "docs/specs/", "reports/"],
            capture_output=True,
            text=True,
            check=False,
        )
        if diff_proc.stdout.strip():
            errors.append("Implementer modified docs/specs/ or reports/ between spec_sha and impl_sha")

    return {"passed": len(errors) == 0, "errors": errors}


def _check_8_post_colab(
    results_dir: str,
    expected_qids: int,
    drive_dir: Optional[str] = None,
    initial_cells_json: Optional[str] = None,
    current_cells_json: Optional[str] = None,
) -> Dict[str, Any]:
    errors: List[str] = []
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

    sha_file = os.path.join(results_dir, "sha256sum.txt")
    if not os.path.isfile(sha_file):
        errors.append(f"Missing sha256sum.txt in {results_dir}")
    else:
        checksums: Dict[str, str] = {}
        for line in open(sha_file, "r", encoding="utf-8"):
            parts = line.strip().split()
            if len(parts) >= 2:
                checksums[os.path.basename(parts[1])] = parts[0].lower()
        for art in EXPECTED_ARTIFACTS:
            art_path = os.path.join(results_dir, art)
            if not os.path.isfile(art_path):
                errors.append(f"Missing required artifact: {art}")
                continue
            digest = hashlib.sha256(open(art_path, "rb").read()).hexdigest().lower()
            if checksums.get(art) != digest:
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

    return {"passed": len(errors) == 0, "errors": errors}


def run_auditor(
    mode: str,
    repo_root: str = PROJECT_ROOT,
    team: int = 0,
    iter_k: int = 0,
    spec_sha: Optional[str] = None,
    impl_sha: Optional[str] = None,
    ledger_path: Optional[str] = None,
    results_dir: Optional[str] = None,
    expected_qids: int = 1222,
    spec_path_override: Optional[str] = None,
    drive_dir: Optional[str] = None,
    initial_cells_json: Optional[str] = None,
    current_cells_json: Optional[str] = None,
    output_json: Optional[str] = None,
) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    if mode in ("bootstrap", "pre-colab"):
        target_sha = impl_sha or "HEAD"
        with tempfile.TemporaryDirectory() as snap:
            _unpack_git_archive(repo_root, target_sha, snap)
            if mode == "pre-colab":
                ledger = json.load(open(ledger_path, "r", encoding="utf-8"))
                team_key = f"team_{team}"
                t_meta = ledger["teams"][team_key]
                rh_p = t_meta["reading_history_path"]
                bib_p = t_meta["bib_path"]
                lit_res = verify_literature_audit(
                    transcript_path=t_meta["transcript_path"],
                    cursor_step=int(t_meta.get("cursor_step", 0)),
                    cache_dir=t_meta["cache_dir"],
                    reading_history_path=rh_p if os.path.isabs(rh_p) else os.path.join(snap, rh_p),
                    bib_path=bib_p if os.path.isabs(bib_p) else os.path.join(snap, bib_p),
                )
                checks["check1"] = lit_res
            c235 = _check_2_and_3_and_5(os.path.join(snap, "src", "nevfe", "core"))
            checks.update(c235)
            checks["check4"] = _check_4_metrics_lock(repo_root)
            checks["check6"] = _check_6_guarded_pytest(snap)
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
        )

    passed = all(c.get("passed", False) for c in checks.values())
    report = {"mode": mode, "team": team, "iter": iter_k, "passed": passed, "checks": checks}
    dest = output_json or os.path.join(repo_root, "reports", "audit", f"team_{team}_iter_{iter_k}.json")
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Automated 7-Binary + Post-Colab Auditor Gatekeeper.")
    parser.add_argument("--mode", required=True, choices=["bootstrap", "pre-colab", "post-colab"])
    parser.add_argument("--repo-root", default=PROJECT_ROOT)
    parser.add_argument("--team", type=int, default=0)
    parser.add_argument("--iter", dest="iter_k", type=int, default=0)
    parser.add_argument("--spec-sha", default=None)
    parser.add_argument("--impl-sha", default=None)
    parser.add_argument("--ledger", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--expected-qids", type=int, default=1222)
    parser.add_argument("--spec-path", default=None)
    parser.add_argument("--drive-dir", default=None)
    parser.add_argument("--initial-cells-json", default=None)
    parser.add_argument("--current-cells-json", default=None)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args(argv)

    report = run_auditor(
        mode=args.mode,
        repo_root=args.repo_root,
        team=args.team,
        iter_k=args.iter_k,
        spec_sha=args.spec_sha,
        impl_sha=args.impl_sha,
        ledger_path=args.ledger,
        results_dir=args.results_dir,
        expected_qids=args.expected_qids,
        spec_path_override=args.spec_path,
        drive_dir=args.drive_dir,
        initial_cells_json=args.initial_cells_json,
        current_cells_json=args.current_cells_json,
        output_json=args.output_json,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
