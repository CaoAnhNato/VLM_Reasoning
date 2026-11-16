#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 21 — Colab T4 Conditional Gate B Pilot Runner (Phases B & C).

Enforces the exact Project Owner conditional execution protocol:
  1. Phase B: Execute resource and precision preflight FIRST in the active session
     before downloading weights or instantiating the model.
  2. Phase C.1: Run one controlled 1-sample real-model smoke check on `sg_vcd`.
     If it fails, preserve logs and stop immediately without switching precision.
  3. Phase C.2: Run the frozen 20-QA manifest (`data/vitextvqa/manifest_integration_20qa.json`)
     across the exact three approved arms (`ctrl_0`, `oracle_roi`, `sg_vcd`) using
     `scoring_mode=canonical_signed` and the Phase A selected precision.
  4. Phase C.3: Save consolidated runtime, VRAM/RAM, latency, failure, and comparison evidence.
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.benchmark.colab_gate_b_preflight import (
    audit_hardware_and_environment,
    format_human_readable_report,
)


FROZEN_MANIFEST_REL = os.path.join("data", "vitextvqa", "manifest_integration_20qa.json")
APPROVED_ARMS = ("ctrl_0", "oracle_roi", "sg_vcd")


def run_phase_b_preflight(
    output_dir: str,
    precision: str,
    allow_bf16_exception: bool,
    device_index: int = 0,
) -> Dict[str, Any]:
    """Run Phase B preflight before any model download or instantiation."""
    os.makedirs(output_dir, exist_ok=True)
    report = audit_hardware_and_environment(
        device_index=device_index,
        precision=precision,
        allow_bf16_exception=allow_bf16_exception,
    )
    preflight_path = os.path.join(output_dir, "preflight_report.json")
    with open(preflight_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(format_human_readable_report(report))
    print(f"[PREFLIGHT] Saved raw preflight JSON to: {preflight_path}")
    return report


def _is_preflight_admitted(report: Dict[str, Any]) -> bool:
    adm = report.get("admission", {})
    if adm.get("verdict") == "PASS":
        return True
    exc = adm.get("conditional_exception", {})
    return bool(exc.get("authorized") and exc.get("hard_resource_gates_passed") and exc.get("proceed_under_exception"))


def _run_evaluator_subprocess(
    *,
    python_exe: str,
    manifest_path: str,
    method: str,
    precision: str,
    output_dir: str,
    model_name: str,
    device: str,
    test_samples: int = 0,
    overwrite: bool = True,
    mock: bool = False,
) -> Dict[str, Any]:
    eval_script = os.path.join(THIS_DIR, "evaluate_vitextvqa.py")
    cmd = [
        python_exe,
        eval_script,
        "--manifest", manifest_path,
        "--method", method,
        "--scoring-mode", "canonical_signed",
        "--precision", precision,
        "--model-name", model_name,
        "--device", device,
        "--output-dir", output_dir,
    ]
    if test_samples > 0:
        cmd.extend(["--test-samples", str(test_samples)])
    if overwrite:
        cmd.append("--overwrite")
    if mock:
        cmd.append("--mock")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    t0 = time.time()
    proc = subprocess.run(cmd, env=env, check=False)
    wall_s = round(time.time() - t0, 3)

    return {
        "command": cmd,
        "returncode": int(proc.returncode),
        "wall_time_s": wall_s,
        "output_dir": output_dir,
    }


def _inspect_arm_outputs(arm_dir: str, expected_rows: int) -> Dict[str, Any]:
    summary_path = os.path.join(arm_dir, "summary_metrics.json")
    metadata_path = os.path.join(arm_dir, "run_metadata.json")
    jsonl_path = os.path.join(arm_dir, "predictions.jsonl")

    summary = None
    metadata = None
    records: List[Dict[str, Any]] = []

    if os.path.isfile(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    if os.path.isfile(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    if os.path.isfile(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    qids = [r.get("question_id") for r in records]
    run_statuses: Dict[str, int] = {}
    ans_statuses: Dict[str, int] = {}
    loc_statuses: Dict[str, int] = {}
    for r in records:
        rs = r.get("run_status", "unknown")
        run_statuses[rs] = run_statuses.get(rs, 0) + 1
        as_ = (r.get("answer") or {}).get("status", "unknown")
        ans_statuses[as_] = ans_statuses.get(as_, 0) + 1
        ls = (r.get("localization") or {}).get("status", "unknown")
        loc_statuses[ls] = loc_statuses.get(ls, 0) + 1

    return {
        "row_count": len(records),
        "expected_rows": expected_rows,
        "complete": len(records) == expected_rows,
        "qids": qids,
        "run_status_counts": run_statuses,
        "answer_status_counts": ans_statuses,
        "localization_status_counts": loc_statuses,
        "summary": summary,
        "runtime_metadata": metadata,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Task 21 Colab T4 Conditional Gate B Pilot Runner (20-QA / 3-Arm)."
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["bf16", "fp16"],
        help="Precision selected in Phase A (default: fp16; bf16 requires --allow-bf16-exception on T4).",
    )
    parser.add_argument(
        "--allow-bf16-exception",
        action="store_true",
        help="Authorize Owner's T4 BF16 hardware-compatibility exception when running bf16.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=os.path.join(PROJECT_ROOT, FROZEN_MANIFEST_REL),
        help="Path to frozen 20-QA manifest JSON.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="5CD-AI/Vintern-1B-v3_5",
        help="Hugging Face model identifier.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Target device ('cuda' for Colab T4; 'cpu' only with --mock).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "results", "task21_t4_gate_b_pilot"),
        help="Root output directory for preflight, smoke check, and 3-arm results.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run local synthetic mock orchestration test (skips CUDA hardware rejection for wiring test).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    consolidated: Dict[str, Any] = {
        "timestamp": datetime.datetime.now().isoformat(),
        "precision": args.precision,
        "allow_bf16_exception": bool(args.allow_bf16_exception),
        "manifest": args.manifest,
        "model_name": args.model_name,
        "mock": bool(args.mock),
        "status": "STARTED",
    }

    # 1. Phase B Preflight
    preflight = run_phase_b_preflight(
        output_dir=args.output_dir,
        precision=args.precision,
        allow_bf16_exception=args.allow_bf16_exception,
    )
    consolidated["preflight"] = preflight

    if not args.mock and not _is_preflight_admitted(preflight):
        consolidated["status"] = "ABORTED_PREFLIGHT_BLOCKER"
        with open(os.path.join(args.output_dir, "pilot_consolidated_summary.json"), "w", encoding="utf-8") as f:
            json.dump(consolidated, f, indent=2, ensure_ascii=False)
        print("\n[ABORT] Phase B preflight blocked execution before downloading weights.")
        sys.exit(1)

    # 2. Phase C.1 First-Sample Smoke Check
    smoke_dir = os.path.join(args.output_dir, "smoke_check_1qa_sg_vcd")
    print("\n" + "=" * 82)
    print(" PHASE C.1 — FIRST-SAMPLE REAL INFERENCE SMOKE CHECK (SG-VCD, 1 QA)")
    print("=" * 82)
    smoke_exec = _run_evaluator_subprocess(
        python_exe=sys.executable,
        manifest_path=args.manifest,
        method="sg_vcd",
        precision=args.precision,
        output_dir=smoke_dir,
        model_name=args.model_name,
        device=args.device,
        test_samples=1,
        overwrite=True,
        mock=args.mock,
    )
    smoke_inspect = _inspect_arm_outputs(smoke_dir, expected_rows=1)
    consolidated["smoke_check"] = {"exec": smoke_exec, "inspect": smoke_inspect}

    smoke_ok = (
        smoke_exec["returncode"] == 0
        and smoke_inspect["complete"]
        and smoke_inspect["run_status_counts"].get("completed", 0) == 1
        and smoke_inspect["answer_status_counts"].get("valid", 0) == 1
        and smoke_inspect["localization_status_counts"].get("valid", 0) == 1
    )
    if not smoke_ok:
        consolidated["status"] = "ABORTED_SMOKE_CHECK_FAILED"
        with open(os.path.join(args.output_dir, "pilot_consolidated_summary.json"), "w", encoding="utf-8") as f:
            json.dump(consolidated, f, indent=2, ensure_ascii=False)
        print("\n[ABORT] Phase C.1 1-sample smoke check failed! Preserved logs and stopping pilot.")
        sys.exit(1)

    # 3. Phase C.2 Approved 20-QA / 3-Arm Pilot
    arms_report: Dict[str, Any] = {}
    ref_qids = None
    for arm in APPROVED_ARMS:
        arm_dir = os.path.join(args.output_dir, f"arm_{arm}")
        print("\n" + "=" * 82)
        print(f" PHASE C.2 — EXECUTING ARM: {arm.upper()} (20 QA, precision={args.precision}, canonical_signed)")
        print("=" * 82)
        arm_exec = _run_evaluator_subprocess(
            python_exe=sys.executable,
            manifest_path=args.manifest,
            method=arm,
            precision=args.precision,
            output_dir=arm_dir,
            model_name=args.model_name,
            device=args.device,
            test_samples=0,
            overwrite=True,
            mock=args.mock,
        )
        arm_inspect = _inspect_arm_outputs(arm_dir, expected_rows=20)
        if ref_qids is None:
            ref_qids = arm_inspect["qids"]
        else:
            arm_inspect["qid_order_matches_ctrl0"] = (arm_inspect["qids"] == ref_qids)
        arms_report[arm] = {"exec": arm_exec, "inspect": arm_inspect}

        if arm_exec["returncode"] != 0:
            consolidated["arms"] = arms_report
            consolidated["status"] = f"ABORTED_ARM_{arm.upper()}_FAILED"
            with open(os.path.join(args.output_dir, "pilot_consolidated_summary.json"), "w", encoding="utf-8") as f:
                json.dump(consolidated, f, indent=2, ensure_ascii=False)
            print(f"\n[ABORT] Arm {arm} failed with exit code {arm_exec['returncode']}. Preserved partial outputs.")
            sys.exit(1)

    consolidated["arms"] = arms_report
    consolidated["status"] = "PILOT_COMPLETED"
    out_summary_path = os.path.join(args.output_dir, "pilot_consolidated_summary.json")
    with open(out_summary_path, "w", encoding="utf-8") as f:
        json.dump(consolidated, f, indent=2, ensure_ascii=False)
    print(f"\n[DONE] Consolidated Task 21 pilot summary saved to: {out_summary_path}")


if __name__ == "__main__":
    main()

