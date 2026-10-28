#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master E2E Test Suite Runner for SG-VCD ViTextVQA 400-Image Benchmark.
Executes Tiers 1-4, aggregates structured summaries, prints an executive report table,
and exits with code 0 on 100% pass.

Usage:
    .\\venv\\Scripts\\python.exe tests/run_e2e_tests.py
"""

import argparse
import io
import os
import sys
import time
import unittest
from typing import Any, Dict, List, Tuple

# Reconfigure stdout for UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tests.test_tier1_features import (
    TestTier1Feature1DataLoading,
    TestTier1Feature2SpatialMetrics,
    TestTier1Feature3VQAMetrics,
    TestTier1Feature4TokenizerConfig,
    TestTier1Feature5PipelineResult,
    TestTier1Round3AlgorithmicUpgrades,
)
from tests.test_tier2_boundaries import (
    TestTier2Boundary1AspectRatios,
    TestTier2Boundary2VietnameseQuestions,
    TestTier2Boundary3MultiBoxSubBboxes,
    TestTier2Boundary4MissingBoxAndCoordinates,
    TestTier2Boundary5WhitespaceAndPunctuation,
)
from tests.test_tier3_combinations import (
    TestTier3Combination1VisualCacheAndCheckpoint,
    TestTier3Combination2Stage1CroppingAndStage2Decoding,
    TestTier3Combination3DualMetricEvaluation,
    TestTier3Combination4ResumableBenchmarkPersistence,
)
from tests.test_tier4_workloads import TestTier4RealWorldWorkloads


TIER_DEFINITIONS = [
    {
        "id": "Tier 1",
        "name": "Feature Coverage",
        "classes": [
            TestTier1Feature1DataLoading,
            TestTier1Feature2SpatialMetrics,
            TestTier1Feature3VQAMetrics,
            TestTier1Feature4TokenizerConfig,
            TestTier1Feature5PipelineResult,
            TestTier1Round3AlgorithmicUpgrades,
        ],
    },
    {
        "id": "Tier 2",
        "name": "Boundary & Corner Cases",
        "classes": [
            TestTier2Boundary1AspectRatios,
            TestTier2Boundary2VietnameseQuestions,
            TestTier2Boundary3MultiBoxSubBboxes,
            TestTier2Boundary4MissingBoxAndCoordinates,
            TestTier2Boundary5WhitespaceAndPunctuation,
        ],
    },
    {
        "id": "Tier 3",
        "name": "Cross-Feature Combinations",
        "classes": [
            TestTier3Combination1VisualCacheAndCheckpoint,
            TestTier3Combination2Stage1CroppingAndStage2Decoding,
            TestTier3Combination3DualMetricEvaluation,
            TestTier3Combination4ResumableBenchmarkPersistence,
        ],
    },
    {
        "id": "Tier 4",
        "name": "Real-World Workloads",
        "classes": [
            TestTier4RealWorldWorkloads,
        ],
    },
]


def run_test_tier(tier_info: Dict[str, Any], verbosity: int = 1) -> Dict[str, Any]:
    """Execute all test cases in a given tier and return structured metrics."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for test_cls in tier_info["classes"]:
        suite.addTests(loader.loadTestsFromTestCase(test_cls))

    stream = io.StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=verbosity)

    start_time = time.perf_counter()
    result = runner.run(suite)
    elapsed = time.perf_counter() - start_time

    total_runs = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total_runs - failures - errors

    return {
        "tier_id": tier_info["id"],
        "tier_name": tier_info["name"],
        "total": total_runs,
        "passed": passed,
        "failed": failures,
        "errors": errors,
        "elapsed_sec": elapsed,
        "is_success": result.wasSuccessful(),
        "log": stream.getvalue(),
        "failure_details": result.failures + result.errors,
    }


def format_summary_table(tier_results: List[Dict[str, Any]]) -> str:
    """Format an ASCII executive summary table of test results."""
    lines = []
    lines.append("=" * 86)
    lines.append(f"{'SG-VCD ViTextVQA 400-IMAGE BENCHMARK: E2E TEST SUITE EXECUTIVE REPORT':^86}")
    lines.append("=" * 86)
    header = f"| {'Tier':<8} | {'Tier Name':<28} | {'Total':>6} | {'Passed':>7} | {'Failed':>7} | {'Time':>8} | {'Status':^8} |"
    lines.append(header)
    lines.append("|" + "-" * 10 + "|" + "-" * 30 + "|" + "-" * 8 + "|" + "-" * 9 + "|" + "-" * 9 + "|" + "-" * 10 + "|" + "-" * 10 + "|")

    total_tests = 0
    total_passed = 0
    total_failed = 0
    total_errors = 0
    total_time = 0.0

    for tr in tier_results:
        total_tests += tr["total"]
        total_passed += tr["passed"]
        total_failed += tr["failed"]
        total_errors += tr["errors"]
        total_time += tr["elapsed_sec"]

        status_str = "PASS" if tr["is_success"] else "FAIL"
        time_str = f"{tr['elapsed_sec']:.2f}s"
        row = (
            f"| {tr['tier_id']:<8} "
            f"| {tr['tier_name']:<28} "
            f"| {tr['total']:>6} "
            f"| {tr['passed']:>7} "
            f"| {tr['failed'] + tr['errors']:>7} "
            f"| {time_str:>8} "
            f"| {status_str:^8} |"
        )
        lines.append(row)

    lines.append("|" + "=" * 10 + "|" + "=" * 30 + "|" + "=" * 8 + "|" + "=" * 9 + "|" + "=" * 9 + "|" + "=" * 10 + "|" + "=" * 10 + "|")
    overall_status = "PASS" if (total_failed == 0 and total_errors == 0) else "FAIL"
    summary_row = (
        f"| {'TOTAL':<8} "
        f"| {'All 4 Tiers Verified':<28} "
        f"| {total_tests:>6} "
        f"| {total_passed:>7} "
        f"| {total_failed + total_errors:>7} "
        f"| {total_time:.2f}s{'':>3} "
        f"| {overall_status:^8} |"
    )
    lines.append(summary_row)
    lines.append("=" * 86)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Master E2E Test Runner for SG-VCD")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3, 4], help="Run a specific tier only (1, 2, 3, 4)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose test logs")
    args = parser.parse_args()

    selected_tiers = TIER_DEFINITIONS
    if args.tier:
        selected_tiers = [t for t in TIER_DEFINITIONS if t["id"] == f"Tier {args.tier}"]

    print(f"\n[INFO] Starting SG-VCD E2E Test Suite Execution across {len(selected_tiers)} tier(s)...")

    results = []
    for tier in selected_tiers:
        print(f"  -> Running {tier['id']}: {tier['name']}...")
        tr = run_test_tier(tier, verbosity=2 if args.verbose else 1)
        results.append(tr)
        if args.verbose and tr["log"]:
            print(tr["log"])

    report_table = format_summary_table(results)
    print("\n" + report_table + "\n")

    # If any failures, print details
    has_failure = any(not r["is_success"] for r in results)
    if has_failure:
        print("[FAILURES / ERRORS DETECTED]:")
        for r in results:
            if not r["is_success"]:
                print(f"\n--- {r['tier_id']}: {r['tier_name']} Failures ---")
                for test_case, err_trace in r["failure_details"]:
                    print(f"Test: {test_case}")
                    print(err_trace)
        return 1

    print("[SUCCESS] All test cases passed with 100% success rate! Exit Code 0.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
