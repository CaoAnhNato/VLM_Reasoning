# -*- coding: utf-8 -*-
"""
ADVERSARIAL STRESS TEST SUITE: SHARD MANAGEMENT AND AGGREGATION
Milestone R2-1 Challenger Review (challenger_r2_1)

Adversarial Verification Dimensions:
1. Empty & Missing Shards:
   - All shards missing (empty dir)
   - Sparse missing shards (e.g. 2 out of 6 present)
   - 0-byte and whitespace-only shard files
   - Corrupted & truncated JSONL lines (process kill simulation)
   - Checkpoint fallback when jsonl is missing or unreadable
2. Deterministic Equivalence & Ordering:
   - 1-worker vs 2-worker vs 4-worker equivalence on synthetic mock data
   - Prime number shard counts with uneven sample partitioning (19 samples / 7 workers)
3. Out-of-Order Completion:
   - Reverse-order worker completion simulation (worker 3 -> 2 -> 1 -> 0)
   - Random shuffled intra-shard record ordering
   - Cross-shard duplicate key deduplication via seen_keys
4. Robustness against missing index fields in records.
"""

import copy
import glob
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from typing import Any, Dict, List

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.benchmark.evaluate_vitextvqa import (
    CheckpointManager,
    aggregate_shards,
    generate_benchmark_report,
)
from scripts.benchmark.metrics import MetricsAggregator


def _create_synthetic_record(idx: int, qid: int = None, method: str = "sg_vcd") -> Dict[str, Any]:
    """Helper to generate valid synthetic benchmark record."""
    if qid is None:
        qid = 1000 + idx
    return {
        "sample_key": f"{qid}_image_{idx:03d}.jpg",
        "index": idx,
        "global_index": idx,
        "question_id": qid,
        "image_name": f"image_{idx:03d}.jpg",
        "image_size": [1920, 1080],
        "question": f"Biển báo này ghi chữ gì số {idx}?",
        "gt_answers": [f"Biển báo {idx}"],
        "verified_all_answers": [f"Biển báo {idx}", f"Bien bao {idx}"],
        "added_gt_answers": [],
        "gt_bbox": [100, 200, 300, 400],
        "pred_bbox": [105, 205, 295, 395],
        "sub_bboxes": [],
        "pred_answer": f"Biển báo {idx}",
        "stage1_metrics": {
            "iou": 0.82,
            "recall": 0.91,
            "precision": 0.89,
            "hit_iou_03": 1.0,
            "hit_iou_05": 1.0,
            "hit_recall_05": 1.0,
            "sub_coverage_recall": 1.0,
            "max_sub_iou": 0.82,
            "multi_box_hit": 1.0,
        },
        "stage2_metrics": {
            "em_gt": 1.0,
            "em_verified": 1.0,
            "vqa_acc_gt": 1.0,
            "vqa_acc_verified": 1.0,
            "f1_gt": 1.0,
            "f1_verified": 1.0,
            "substring_gt": 1.0,
            "substring_verified": 1.0,
            "anls_gt": 1.0,
            "anls_verified": 1.0,
        },
        "spatial_confidence": 0.95,
        "latency_s": 0.12,
    }


class TestAdversarialEmptyAndMissingShards(unittest.TestCase):
    """Stress test empty and missing shard scenarios."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="adv_test_empty_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_adv_01_all_shards_missing(self):
        """Adversarial: num_workers=4, but no shard files exist in directory."""
        summary = aggregate_shards(
            output_dir=self.test_dir,
            num_workers=4,
            method="sg_vcd",
            model_name="5CD-AI/Vintern-1B-v3_5",
        )
        self.assertIsNotNone(summary)
        self.assertEqual(summary["total_evaluated_samples"], 0)
        # Verify master files created safely without crashing
        self.assertTrue(os.path.isfile(os.path.join(self.test_dir, "predictions.jsonl")))
        self.assertTrue(os.path.isfile(os.path.join(self.test_dir, "predictions.csv")))
        self.assertTrue(os.path.isfile(os.path.join(self.test_dir, "summary_metrics.json")))
        self.assertTrue(os.path.isfile(os.path.join(self.test_dir, "checkpoint.json")))

    def test_adv_02_sparse_missing_shards(self):
        """Adversarial: 6 workers configured, but only shard_1 and shard_4 exist; 0, 2, 3, 5 are missing."""
        rec1 = _create_synthetic_record(1)
        rec4 = _create_synthetic_record(4)

        # Write shard 1 and shard 4
        with open(os.path.join(self.test_dir, "predictions_shard_1.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps(rec1) + "\n")
        with open(os.path.join(self.test_dir, "predictions_shard_4.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps(rec4) + "\n")

        summary = aggregate_shards(
            output_dir=self.test_dir,
            num_workers=6,
            method="sg_vcd",
        )
        self.assertEqual(summary["total_evaluated_samples"], 2)

        # Verify predictions.jsonl contains exactly indices 1 and 4 in sorted order
        with open(os.path.join(self.test_dir, "predictions.jsonl"), "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        self.assertEqual(len(lines), 2)
        self.assertEqual([r["global_index"] for r in lines], [1, 4])

    def test_adv_03_zero_byte_and_whitespace_shards(self):
        """Adversarial: Shard 0 is 0 bytes, Shard 1 has only blank whitespace lines, Shard 2 is valid."""
        # Shard 0: 0 bytes
        open(os.path.join(self.test_dir, "predictions_shard_0.jsonl"), "w").close()
        # Shard 1: spaces and newlines
        with open(os.path.join(self.test_dir, "predictions_shard_1.jsonl"), "w", encoding="utf-8") as f:
            f.write("   \n\n\t  \n")
        # Shard 2: valid record
        rec2 = _create_synthetic_record(2)
        with open(os.path.join(self.test_dir, "predictions_shard_2.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps(rec2) + "\n")

        summary = aggregate_shards(output_dir=self.test_dir, num_workers=3)
        self.assertEqual(summary["total_evaluated_samples"], 1)
        with open(os.path.join(self.test_dir, "predictions.jsonl"), "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["global_index"], 2)

    def test_adv_04_corrupted_and_truncated_jsonl(self):
        """Adversarial: Worker process crash creates partially written/corrupt JSON lines."""
        rec0 = _create_synthetic_record(0)
        rec1 = _create_synthetic_record(1)

        with open(os.path.join(self.test_dir, "predictions_shard_0.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps(rec0) + "\n")
            f.write('{"sample_key": "broken_json", "index": 999, "pred_answer": "trun\n')  # truncated
            f.write("NOT_A_JSON_STRING\n")
            f.write(json.dumps(rec1) + "\n")

        summary = aggregate_shards(output_dir=self.test_dir, num_workers=1)
        # Should gracefully salvage rec0 and rec1, skipping broken lines
        self.assertEqual(summary["total_evaluated_samples"], 2)
        with open(os.path.join(self.test_dir, "predictions.jsonl"), "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        self.assertEqual(len(lines), 2)
        self.assertEqual([r["global_index"] for r in lines], [0, 1])

    def test_adv_05_fallback_to_checkpoint_json_when_jsonl_missing(self):
        """Adversarial: predictions_shard_0.jsonl does not exist, but checkpoint_shard_0.json is present."""
        rec0 = _create_synthetic_record(0)
        rec1 = _create_synthetic_record(1)
        ckpt_data = {
            "checkpoint_version": "1.0",
            "total_completed": 2,
            "records": [rec0, rec1],
        }
        with open(os.path.join(self.test_dir, "checkpoint_shard_0.json"), "w", encoding="utf-8") as f:
            json.dump(ckpt_data, f)

        summary = aggregate_shards(output_dir=self.test_dir, num_workers=1)
        self.assertEqual(summary["total_evaluated_samples"], 2)


class TestAdversarialOutOfOrderAndDeduplication(unittest.TestCase):
    """Stress test out-of-order completion, shuffled inputs, and duplicate keys."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="adv_test_order_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_adv_06_reverse_order_completion(self):
        """Adversarial: Simulate 4 workers finishing in reverse order (worker 3, then 2, then 1, then 0)."""
        num_workers = 4
        total_samples = 12

        # Partition indices according to i % 4 == worker_id
        for w_id in reversed(range(num_workers)):
            worker_records = [
                _create_synthetic_record(i)
                for i in range(total_samples)
                if i % num_workers == w_id
            ]
            shard_file = os.path.join(self.test_dir, f"predictions_shard_{w_id}.jsonl")
            with open(shard_file, "w", encoding="utf-8") as f:
                for r in worker_records:
                    f.write(json.dumps(r) + "\n")

        summary = aggregate_shards(output_dir=self.test_dir, num_workers=num_workers)
        self.assertEqual(summary["total_evaluated_samples"], total_samples)

        # Master predictions must be strictly sorted by global_index ascending
        with open(os.path.join(self.test_dir, "predictions.jsonl"), "r", encoding="utf-8") as f:
            merged = [json.loads(l) for l in f if l.strip()]
        self.assertEqual(len(merged), total_samples)
        indices = [r["global_index"] for r in merged]
        self.assertEqual(indices, list(range(total_samples)))

    def test_adv_07_shuffled_intra_shard_ordering(self):
        """Adversarial: Records inside shards are jumbled randomly. Merged output must be sorted."""
        total_samples = 15
        all_recs = [_create_synthetic_record(i) for i in range(total_samples)]
        random.seed(42)
        random.shuffle(all_recs)

        # Split into 2 shards randomly
        shard0 = all_recs[:8]
        shard1 = all_recs[8:]

        with open(os.path.join(self.test_dir, "predictions_shard_0.jsonl"), "w", encoding="utf-8") as f:
            for r in shard0:
                f.write(json.dumps(r) + "\n")
        with open(os.path.join(self.test_dir, "predictions_shard_1.jsonl"), "w", encoding="utf-8") as f:
            for r in shard1:
                f.write(json.dumps(r) + "\n")

        summary = aggregate_shards(output_dir=self.test_dir, num_workers=2)
        self.assertEqual(summary["total_evaluated_samples"], total_samples)

        with open(os.path.join(self.test_dir, "predictions.jsonl"), "r", encoding="utf-8") as f:
            merged = [json.loads(l) for l in f if l.strip()]
        indices = [r["global_index"] for r in merged]
        self.assertEqual(indices, list(range(total_samples)))

    def test_adv_08_cross_shard_duplicate_keys(self):
        """Adversarial: Shard 0 and Shard 1 both contain the same sample (retry/race condition). Deduplication must drop duplicate."""
        rec_shared = _create_synthetic_record(5)
        rec0 = _create_synthetic_record(0)
        rec1 = _create_synthetic_record(1)

        with open(os.path.join(self.test_dir, "predictions_shard_0.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps(rec0) + "\n")
            f.write(json.dumps(rec_shared) + "\n")
        with open(os.path.join(self.test_dir, "predictions_shard_1.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps(rec_shared) + "\n")
            f.write(json.dumps(rec1) + "\n")

        summary = aggregate_shards(output_dir=self.test_dir, num_workers=2)
        self.assertEqual(summary["total_evaluated_samples"], 3)  # 0, 1, 5
        with open(os.path.join(self.test_dir, "predictions.jsonl"), "r", encoding="utf-8") as f:
            merged = [json.loads(l) for l in f if l.strip()]
        self.assertEqual(len(merged), 3)
        self.assertEqual([r["global_index"] for r in merged], [0, 1, 5])

    def test_adv_09_robustness_when_global_index_missing_or_none(self):
        """Adversarial: Records without 'global_index' or with None should not crash sorting with TypeError."""
        rec_a = _create_synthetic_record(0)
        del rec_a["global_index"]  # only 'index' remains
        rec_b = _create_synthetic_record(1)
        # rec_b has global_index

        with open(os.path.join(self.test_dir, "predictions_shard_0.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps(rec_a) + "\n")
            f.write(json.dumps(rec_b) + "\n")

        summary = aggregate_shards(output_dir=self.test_dir, num_workers=1)
        self.assertEqual(summary["total_evaluated_samples"], 2)


class TestAdversarialDeterministicEquivalenceCLI(unittest.TestCase):
    """Stress test 1 worker vs 2 workers vs 4 workers execution equivalence via CLI."""

    def setUp(self):
        self.base_dir = tempfile.mkdtemp(prefix="adv_test_equiv_")
        self.dir_1w = os.path.join(self.base_dir, "run_1w")
        self.dir_2w = os.path.join(self.base_dir, "run_2w")
        self.dir_4w = os.path.join(self.base_dir, "run_4w")

    def tearDown(self):
        shutil.rmtree(self.base_dir, ignore_errors=True)

    def test_adv_10_deterministic_equivalence_1w_vs_2w_vs_4w(self):
        """
        Adversarial: Execute evaluate_vitextvqa.py --mock on 20 samples with:
        - 1 worker (--num-workers 1)
        - 2 workers (--num-workers 2 --multi-proc 2)
        - 4 workers (--num-workers 4 --multi-proc 4)
        Assert identical sample counts, identical metric values, and identical sorted predictions.
        """
        python_exe = sys.executable
        script_path = os.path.join(PROJECT_ROOT, "scripts", "benchmark", "evaluate_vitextvqa.py")

        runs = [
            ("1w", self.dir_1w, ["--num-workers", "1"]),
            ("2w", self.dir_2w, ["--num-workers", "2", "--multi-proc", "2"]),
            ("4w", self.dir_4w, ["--num-workers", "4", "--multi-proc", "4"]),
        ]

        results = {}
        for label, out_dir, extra_args in runs:
            cmd = [
                python_exe,
                script_path,
                "--test-samples", "20",
                "--mock",
                "--device", "cpu",
                "--output-dir", out_dir,
                "--overwrite",
            ] + extra_args

            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                cwd=PROJECT_ROOT,
            )
            stdout_str = proc.stdout or ""
            stderr_str = proc.stderr or ""
            self.assertEqual(
                proc.returncode, 0,
                f"Execution failed for {label}: stdout={stdout_str[-500:]}\nstderr={stderr_str[-500:]}"
            )

            # Load summary metrics
            metrics_path = os.path.join(out_dir, "summary_metrics.json")
            self.assertTrue(os.path.isfile(metrics_path), f"Missing summary_metrics.json for {label}")
            with open(metrics_path, "r", encoding="utf-8") as f:
                summary = json.load(f)

            # Load predictions
            pred_path = os.path.join(out_dir, "predictions.jsonl")
            self.assertTrue(os.path.isfile(pred_path), f"Missing predictions.jsonl for {label}")
            with open(pred_path, "r", encoding="utf-8") as f:
                preds = [json.loads(l) for l in f if l.strip()]

            results[label] = {"summary": summary, "preds": preds}

        # 1. Assert sample counts
        self.assertEqual(results["1w"]["summary"]["total_evaluated_samples"], 20)
        self.assertEqual(results["2w"]["summary"]["total_evaluated_samples"], 20)
        self.assertEqual(results["4w"]["summary"]["total_evaluated_samples"], 20)

        self.assertEqual(len(results["1w"]["preds"]), 20)
        self.assertEqual(len(results["2w"]["preds"]), 20)
        self.assertEqual(len(results["4w"]["preds"]), 20)

        # 2. Assert exact index ordering (0 to 19)
        idx_1w = [r["global_index"] for r in results["1w"]["preds"]]
        idx_2w = [r["global_index"] for r in results["2w"]["preds"]]
        idx_4w = [r["global_index"] for r in results["4w"]["preds"]]
        self.assertEqual(idx_1w, list(range(20)))
        self.assertEqual(idx_2w, list(range(20)))
        self.assertEqual(idx_4w, list(range(20)))

        # 3. Assert question_ids and pred_answers match exactly between 1w, 2w, 4w
        for i in range(20):
            p1 = results["1w"]["preds"][i]
            p2 = results["2w"]["preds"][i]
            p4 = results["4w"]["preds"][i]

            self.assertEqual(p1["question_id"], p2["question_id"])
            self.assertEqual(p1["question_id"], p4["question_id"])
            self.assertEqual(p1["pred_answer"], p2["pred_answer"])
            self.assertEqual(p1["pred_answer"], p4["pred_answer"])
            self.assertEqual(p1["pred_bbox"], p2["pred_bbox"])
            self.assertEqual(p1["pred_bbox"], p4["pred_bbox"])

        # 4. Assert aggregated metrics match identically across 1w, 2w, 4w
        s1_metrics_keys = ["mean_iou_pct", "mean_recall_pct", "hit_rate_iou_03_pct", "hit_rate_iou_05_pct", "multi_box_hit_rate_pct"]
        s2_metrics_keys = ["exact_match_verified_pct", "f1_verified_pct", "substring_match_verified_pct", "anls_verified_pct"]

        for k in s1_metrics_keys:
            val_1w = results["1w"]["summary"]["stage1"][k]
            val_2w = results["2w"]["summary"]["stage1"][k]
            val_4w = results["4w"]["summary"]["stage1"][k]
            self.assertEqual(val_1w, val_2w, f"Stage 1 metric {k} mismatch 1w ({val_1w}) vs 2w ({val_2w})")
            self.assertEqual(val_1w, val_4w, f"Stage 1 metric {k} mismatch 1w ({val_1w}) vs 4w ({val_4w})")

        for k in s2_metrics_keys:
            val_1w = results["1w"]["summary"]["stage2"][k]
            val_2w = results["2w"]["summary"]["stage2"][k]
            val_4w = results["4w"]["summary"]["stage2"][k]
            self.assertEqual(val_1w, val_2w, f"Stage 2 metric {k} mismatch 1w ({val_1w}) vs 2w ({val_2w})")
            self.assertEqual(val_1w, val_4w, f"Stage 2 metric {k} mismatch 1w ({val_1w}) vs 4w ({val_4w})")

    def test_adv_11_prime_number_shards_uneven_partition(self):
        """Adversarial: 19 samples sharded across 7 workers (prime number, uneven partition)."""
        total_samples = 19
        num_workers = 7
        samples = [{"global_index": i, "question_id": 2000 + i} for i in range(total_samples)]

        shards = {}
        for w in range(num_workers):
            shards[w] = [s for s in samples if s["global_index"] % num_workers == w]

        # Verify union is complete and non-overlapping
        all_gathered = []
        for w in range(num_workers):
            all_gathered.extend(shards[w])

        self.assertEqual(len(all_gathered), total_samples)
        ids = sorted([s["global_index"] for s in all_gathered])
        self.assertEqual(ids, list(range(total_samples)))


if __name__ == "__main__":
    unittest.main()
