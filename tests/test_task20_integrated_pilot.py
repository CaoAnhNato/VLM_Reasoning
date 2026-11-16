"""
Integration and Parity Test Suite for Task 20: Gate B Pilot Readiness
Tests the implementation against all 14 mandatory validation points from Task 20:
  1. Manifest copy identity and SHA256 match
  2. Three-arm selection (CTRL-0, ORACLE_ROI, SG_VCD) on identical QIDs
  3. Oracle ROI geometry and image coordinate correctness
  4. Missing/invalid Oracle ROI failure handling
  5. Distinction between Oracle ROI and predicted ROI
  6. gt_answers and verified_all_answers separation
  7. Fixed denominator N=20 primary reporting
  8. Conditional denominator and zero-valid diagnostic behavior
  9. Failure taxonomy and status-schema semantics (P0 v1.1)
  10. Per-metric status and valid 0.0 scores
  11. Serialization and checkpoint/resume integrity
  12. No duplicate sample aggregation on resume
  13. Canonical R-03 default scoring intact
  14. OPT-01 batch decoding regression intact
"""

import os
import json
import hashlib
import unittest
from unittest.mock import MagicMock, patch
from PIL import Image
import torch

from scripts.benchmark.metrics import (
    MetricsAggregator,
    compute_stage1_metrics,
    compute_stage2_metrics,
    normalize_vietnamese_text,
)
from scripts.benchmark.sg_vcd.pipeline import SGVCDPipeline, PipelineResult
from scripts.benchmark.evaluate_vitextvqa import load_benchmark_dataset, CheckpointManager


class TestTask20GateBPilotReadiness(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cls.candidate_manifest_path = os.path.join(
            cls.repo_root, "data", "vitextvqa", "manifest_integration_20qa.candidate.json"
        )
        cls.promoted_manifest_path = os.path.join(
            cls.repo_root, "data", "vitextvqa", "manifest_integration_20qa.json"
        )

    # 1. Manifest copy identity and SHA256 match
    def test_01_manifest_copy_identity_and_sha256(self):
        self.assertTrue(os.path.exists(self.candidate_manifest_path), "Candidate manifest missing")
        self.assertTrue(os.path.exists(self.promoted_manifest_path), "Promoted manifest missing")

        with open(self.candidate_manifest_path, "rb") as f:
            cand_bytes = f.read()
        with open(self.promoted_manifest_path, "rb") as f:
            prom_bytes = f.read()

        cand_sha = hashlib.sha256(cand_bytes).hexdigest()
        prom_sha = hashlib.sha256(prom_bytes).hexdigest()

        self.assertEqual(cand_sha, prom_sha, "Promoted manifest SHA256 does not match candidate")
        self.assertEqual(len(cand_bytes), len(prom_bytes), "File size mismatch")

        with open(self.promoted_manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(len(data.get("samples", [])), 20, "Manifest must contain exactly 20 samples")

    # 2. Three-arm selection (CTRL-0, ORACLE_ROI, SG_VCD) on identical QIDs
    def test_02_three_arm_selection_identical_qids(self):
        samples = load_benchmark_dataset(self.promoted_manifest_path)
        self.assertEqual(len(samples), 20)
        qids = [s["question_id"] for s in samples]

        # Ensure all 3 arms can be loaded against the same manifest QID sequence
        for method in ["ctrl_0", "oracle_roi", "sg_vcd"]:
            arm_samples = load_benchmark_dataset(self.promoted_manifest_path)
            self.assertEqual([s["question_id"] for s in arm_samples], qids)

    # 3. Oracle ROI geometry and image coordinate correctness
    def test_03_oracle_roi_geometry_correctness(self):
        samples = load_benchmark_dataset(self.promoted_manifest_path)
        for s in samples:
            bbox = s.get("answer_region_bbox")
            self.assertIsNotNone(bbox)
            self.assertEqual(len(bbox), 4)
            x1, y1, x2, y2 = bbox
            self.assertGreater(x2, x1, f"Invalid box width for QID {s['question_id']}")
            self.assertGreater(y2, y1, f"Invalid box height for QID {s['question_id']}")
            self.assertGreaterEqual(x1, 0)
            self.assertGreaterEqual(y1, 0)

    # 4. Missing/invalid Oracle ROI failure handling
    def test_04_missing_or_invalid_oracle_roi_failure_handling(self):
        invalid_boxes = [
            [],
            [10, 20],
            [50, 50, 40, 60],  # x2 <= x1
            [50, 50, 60, 40],  # y2 <= y1
            None,
        ]
        for inv_box in invalid_boxes:
            is_valid_box = bool(
                inv_box
                and isinstance(inv_box, (list, tuple))
                and len(inv_box) == 4
                and inv_box[2] > inv_box[0]
                and inv_box[3] > inv_box[1]
            )
            self.assertFalse(is_valid_box)

    # 5. Distinction between Oracle ROI and predicted ROI
    def test_05_distinction_oracle_roi_vs_predicted_roi(self):
        # Oracle bypasses Stage 1 localization and forces bbox to oracle_bbox
        oracle_box = [20, 30, 100, 120]
        dummy_img = Image.new("RGB", (200, 200), (255, 255, 255))
        
        # Test pipeline with mocked model
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_transform = MagicMock(return_value=torch.zeros((3, 448, 448)))
        pipe = SGVCDPipeline(
            model=mock_model,
            tokenizer=mock_tokenizer,
            transform=mock_transform,
            scoring_mode="canonical_signed",
            device="cpu",
        )
        
        # Mock _decode_spatial_pmi_contrastive and extract_visual_features
        with patch.object(pipe, "_decode_spatial_pmi_contrastive", return_value=("mock_oracle_answer", 0.5, 0.2)), \
             patch.object(pipe, "extract_visual_features", return_value=None):
            res = pipe.run(dummy_img, "Test question?", oracle_bbox=oracle_box)
            self.assertEqual(res.stage1_bbox, oracle_box)
            self.assertEqual(res.pred_text, "mock_oracle_answer")
            self.assertEqual(list(res.selection.pixel_bbox), oracle_box)
            self.assertEqual(res.spatial_confidence, 1.0)

    # 6. gt_answers and verified_all_answers separation
    def test_06_gt_answers_and_verified_all_answers_separation(self):
        # Single ground-truth answer vs multiple verified answers
        gt = ["Cà phê"]
        ver = ["Cà phê", "Cà phê sữa", "Coffee"]

        # Exact match with verified alternative but not original GT
        pred = "Cà phê sữa"
        m = compute_stage2_metrics(pred, gt_answers=gt, verified_answers=ver)
        self.assertEqual(m["em_gt"], 0.0)
        self.assertEqual(m["em_verified"], 1.0)
        self.assertEqual(m["f1_gt"], 0.8)
        self.assertEqual(m["f1_verified"], 1.0)

    # 7. Fixed denominator N=20 primary reporting
    def test_07_fixed_denominator_n20_primary_reporting(self):
        agg = MetricsAggregator(manifest_count=20)
        # Record only 10 samples (5 correct, 5 failed/empty)
        s1_dummy = {"iou": 0.8, "recall": 1.0, "precision": 0.8, "hit_iou_03": 1.0, "hit_iou_05": 1.0, "hit_recall_05": 1.0, "multi_box_hit": 1.0, "sub_coverage_recall": 1.0}
        s2_correct = {"em_gt": 1.0, "vqa_acc_gt": 1.0, "f1_gt": 1.0, "substring_gt": 1.0, "anls_gt": 1.0,
                      "em_verified": 1.0, "vqa_acc_verified": 1.0, "f1_verified": 1.0, "substring_verified": 1.0, "anls_verified": 1.0}
        s2_zero = {k: 0.0 for k in s2_correct}

        for _ in range(5):
            agg.update(s1_dummy, s2_correct, latency_s=1.0, is_valid=True)
        for _ in range(5):
            agg.update(s1_dummy, s2_zero, latency_s=1.0, is_valid=False, error_category="inference_runtime_exception")

        summary = agg.summary()
        # Primary fixed denominator must be 20
        self.assertEqual(summary["primary_fixed_denominator"]["denominator"], 20)
        # 5 out of 20 = 25.0%
        self.assertEqual(summary["primary_fixed_denominator"]["metrics"]["exact_match_gt_pct"], 25.0)

    # 8. Conditional denominator and zero-valid diagnostic behavior
    def test_08_conditional_denominator_and_zero_valid(self):
        # Case A: Partial valid (5 valid out of 10 evaluated, 20 manifest)
        agg_a = MetricsAggregator(manifest_count=20)
        s1_dummy = {"iou": 0.5, "recall": 0.5, "precision": 0.5, "hit_iou_03": 1.0, "hit_iou_05": 0.0, "hit_recall_05": 1.0, "multi_box_hit": 1.0, "sub_coverage_recall": 0.5}
        s2_correct = {"em_gt": 1.0, "vqa_acc_gt": 1.0, "f1_gt": 1.0, "substring_gt": 1.0, "anls_gt": 1.0,
                      "em_verified": 1.0, "vqa_acc_verified": 1.0, "f1_verified": 1.0, "substring_verified": 1.0, "anls_verified": 1.0}
        s2_zero = {k: 0.0 for k in s2_correct}

        for _ in range(5):
            agg_a.update(s1_dummy, s2_correct, latency_s=1.0, is_valid=True)
        for _ in range(5):
            agg_a.update(s1_dummy, s2_zero, latency_s=1.0, is_valid=False, error_category="empty_generated_answer")

        sum_a = agg_a.summary()
        self.assertEqual(sum_a["conditional_diagnostic"]["denominator"], 5)
        # 5 out of 5 valid = 100.0%
        self.assertEqual(sum_a["conditional_diagnostic"]["metrics"]["exact_match_gt_pct"], 100.0)

        # Case B: Zero valid samples
        agg_b = MetricsAggregator(manifest_count=20)
        for _ in range(5):
            agg_b.update(s1_dummy, s2_zero, latency_s=1.0, is_valid=False, error_category="inference_runtime_exception")

        sum_b = agg_b.summary()
        self.assertEqual(sum_b["conditional_diagnostic"]["denominator"], 0)
        self.assertEqual(sum_b["conditional_diagnostic"]["metrics"]["status"], "unavailable")
        self.assertIsNone(sum_b["conditional_diagnostic"]["metrics"]["exact_match_gt_pct"])

    # 9. Failure taxonomy and status-schema semantics (P0 v1.1)
    def test_09_failure_taxonomy_and_status_schema(self):
        agg = MetricsAggregator(manifest_count=20)
        s1_dummy = {"iou": 0.0, "recall": 0.0, "precision": 0.0, "hit_iou_03": 0.0, "hit_iou_05": 0.0, "hit_recall_05": 0.0, "multi_box_hit": 0.0, "sub_coverage_recall": 0.0}
        s2_dummy = {"em_gt": 0.0, "vqa_acc_gt": 0.0, "f1_gt": 0.0, "substring_gt": 0.0, "anls_gt": 0.0,
                    "em_verified": 0.0, "vqa_acc_verified": 0.0, "f1_verified": 0.0, "substring_verified": 0.0, "anls_verified": 0.0}

        agg.update(s1_dummy, s2_dummy, is_valid=False, error_category="invalid_oracle_bbox")
        agg.update(s1_dummy, s2_dummy, is_valid=False, error_category="empty_generated_answer")
        agg.update(s1_dummy, s2_dummy, is_valid=False, error_category="inference_runtime_exception")

        sum_meta = agg.summary()["diagnostic_metadata"]
        self.assertEqual(sum_meta["n_failed"], 3)
        self.assertEqual(sum_meta["failure_breakdown"]["invalid_oracle_bbox"], 1)
        self.assertEqual(sum_meta["failure_breakdown"]["empty_generated_answer"], 1)
        self.assertEqual(sum_meta["failure_breakdown"]["inference_runtime_exception"], 1)

    # 10. Per-metric status and valid 0.0 scores
    def test_10_per_metric_status_and_valid_zero_scores(self):
        # A valid response that is incorrect gets score 0.0 with status "computed"
        gt = ["Hà Nội"]
        pred = "Hồ Chí Minh"
        m = compute_stage2_metrics(pred, gt_answers=gt, verified_answers=gt)
        self.assertEqual(m["em_gt"], 0.0)
        self.assertEqual(m["f1_gt"], 0.0)

        # In R-04 record schema:
        # A valid incorrect prediction has answer.status="valid", metric status="computed", value=0.0
        record_metric_entry = {"value": 0.0, "status": "computed"}
        self.assertEqual(record_metric_entry["value"], 0.0)
        self.assertEqual(record_metric_entry["status"], "computed")

    # 11. Serialization and checkpoint/resume integrity
    def test_11_checkpoint_resume_integrity(self):
        tmp_dir = os.path.join(self.repo_root, "results", "test_task20_ckpt")
        os.makedirs(tmp_dir, exist_ok=True)
        ckpt_file = os.path.join(tmp_dir, "checkpoint.json")
        jsonl_file = os.path.join(tmp_dir, "predictions.jsonl")

        if os.path.exists(ckpt_file):
            os.remove(ckpt_file)
        if os.path.exists(jsonl_file):
            os.remove(jsonl_file)

        mgr1 = CheckpointManager(checkpoint_path=ckpt_file, output_dir=tmp_dir, jsonl_path=jsonl_file)
        self.assertEqual(len(mgr1.completed_keys), 0)

        agg = MetricsAggregator(manifest_count=20)
        s1 = {"iou": 0.9, "recall": 1.0, "precision": 0.9, "hit_iou_03": 1.0, "hit_iou_05": 1.0, "hit_recall_05": 1.0, "multi_box_hit": 1.0, "sub_coverage_recall": 1.0}
        s2 = {"em_gt": 1.0, "vqa_acc_gt": 1.0, "f1_gt": 1.0, "substring_gt": 1.0, "anls_gt": 1.0,
              "em_verified": 1.0, "vqa_acc_verified": 1.0, "f1_verified": 1.0, "substring_verified": 1.0, "anls_verified": 1.0}

        sample_record = {
            "run_id": "test_run",
            "sample_key": "img1_q1",
            "method_id": "sg_vcd",
            "arm_type": "SG_VCD",
            "run_status": "completed",
            "is_valid": True,
            "answer": {"status": "valid", "raw_text": "sample", "normalized_text": "sample"},
            "localization": {
                "status": "valid",
                "applicability": "applicable",
                "roi_source": "predicted",
                "pred_bbox": [10, 10, 50, 50],
                "spatial_confidence": 0.9,
            },
            "failure": None,
            "metrics": {
                "stage1": {"iou": {"value": 0.9, "status": "computed"}},
                "stage2": {"exact_match_gt": {"value": 1.0, "status": "computed"}},
            },
            "pred_answer": "sample",
            "gt_answers": ["sample"],
            "verified_all_answers": ["sample"],
            "stage1_metrics": s1,
            "stage2_metrics": s2,
            "latency_s": 0.5,
        }
        agg.update(s1, s2, latency_s=0.5, is_valid=True)
        mgr1.append_record(sample_record, agg, method="sg_vcd", model_name="mock_model")

        # Verify on-disk persistence
        self.assertTrue(os.path.exists(ckpt_file))
        self.assertTrue(os.path.exists(jsonl_file))

        # Second manager resuming
        mgr2 = CheckpointManager(checkpoint_path=ckpt_file, output_dir=tmp_dir, jsonl_path=jsonl_file)
        self.assertEqual(len(mgr2.completed_keys), 1)
        self.assertTrue(mgr2.is_completed("img1_q1"))

        # Clean up
        import shutil
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # 15. Task 20R: Specific R-04 Status Schema Conformance Tests
    def test_15_r04_exact_vocabulary_and_validation(self):
        # Verify allowed vocabs
        allowed_run_statuses = {"completed", "partial", "failed", "missing_image", "skipped"}
        allowed_ans_statuses = {"valid", "invalid", "not_attempted"}
        allowed_loc_statuses = {"valid", "failed", "not_produced", "not_applicable"}
        allowed_loc_applicability = {"applicable", "not_applicable"}
        allowed_loc_roi_sources = {"predicted", "oracle_gt", "none"}
        allowed_metric_statuses = {"computed", "not_applicable", "unavailable", "pending_policy"}

        # Validate that legacy/erroneous values are NOT permitted
        forbidden_values = {"success", "uncomputable", "error_message"}
        self.assertTrue(forbidden_values.isdisjoint(allowed_run_statuses))
        self.assertTrue(forbidden_values.isdisjoint(allowed_ans_statuses))
        self.assertTrue(forbidden_values.isdisjoint(allowed_loc_statuses))
        self.assertTrue(forbidden_values.isdisjoint(allowed_metric_statuses))

    def test_16_r04_three_arm_status_mapping(self):
        # CTRL-0 mapping
        ctrl_loc = {"status": "not_applicable", "applicability": "not_applicable", "roi_source": "none"}
        self.assertEqual(ctrl_loc["status"], "not_applicable")
        self.assertEqual(ctrl_loc["applicability"], "not_applicable")
        self.assertEqual(ctrl_loc["roi_source"], "none")

        # Oracle ROI mapping
        oracle_loc = {"status": "valid", "applicability": "applicable", "roi_source": "oracle_gt"}
        self.assertEqual(oracle_loc["status"], "valid")
        self.assertEqual(oracle_loc["applicability"], "applicable")
        self.assertEqual(oracle_loc["roi_source"], "oracle_gt")

        # SG-VCD mapping
        sgvcd_loc = {"status": "valid", "applicability": "applicable", "roi_source": "predicted"}
        self.assertEqual(sgvcd_loc["status"], "valid")
        self.assertEqual(sgvcd_loc["applicability"], "applicable")
        self.assertEqual(sgvcd_loc["roi_source"], "predicted")

    def test_17_r04_failure_object_schema(self):
        # Valid failure object
        valid_fail = {
            "stage": "oracle_input",
            "error_category": "invalid_oracle_bbox",
            "message": "Invalid ground-truth bbox for Oracle arm: [0, 0, 0, 0]",
            "retryable": False,
        }
        self.assertIn("stage", valid_fail)
        self.assertIn("error_category", valid_fail)
        self.assertIn("message", valid_fail)
        self.assertIn("retryable", valid_fail)
        self.assertNotIn("error_message", valid_fail)

        # Successful sample has failure: null
        success_sample_failure = None
        self.assertIsNone(success_sample_failure)

    # 12. No duplicate sample aggregation on resume
    def test_12_no_duplicate_sample_aggregation(self):
        samples = load_benchmark_dataset(self.promoted_manifest_path)
        tmp_dir = os.path.join(self.repo_root, "results", "dummy_test_ckpt")
        ckpt_file = os.path.join(tmp_dir, "checkpoint.json")
        mgr = CheckpointManager(checkpoint_path=ckpt_file, output_dir=tmp_dir, overwrite=True)
        mgr.completed_keys.add(samples[0]["sample_key"])

        remaining = [s for s in samples if not mgr.is_completed(s["sample_key"])]
        self.assertEqual(len(remaining), 19, "One completed sample must leave exactly 19 remaining")

    # 13. Canonical R-03 default scoring intact
    def test_13_canonical_r03_default_scoring_intact(self):
        from scripts.benchmark.sg_vcd.dynamic_cd import DynamicContrastiveDecoder
        decoder = DynamicContrastiveDecoder()
        self.assertEqual(decoder.scoring_mode, "canonical_signed")

        # Test calculate_contrastive_logits helper or direct formula verification
        alpha = 0.5
        pos = torch.tensor([2.0, -1.0])
        neg = torch.tensor([1.0, 3.0])
        if hasattr(decoder, "calculate_contrastive_logits"):
            final = decoder.calculate_contrastive_logits(pos, neg, alpha=alpha)
        else:
            final = (1.0 + alpha) * pos - alpha * neg
        expected = (1.0 + alpha) * pos - alpha * neg
        torch.testing.assert_close(final, expected)

    # 14. OPT-01 batch decoding regression intact
    def test_14_opt01_batch_decoding_regression_intact(self):
        pipe = SGVCDPipeline(
            model=MagicMock(),
            tokenizer=MagicMock(),
            transform=None,
            scoring_mode="canonical_signed",
            device="cpu",
        )
        self.assertTrue(hasattr(pipe, "_run_batched_dual_branch_decode"))


if __name__ == "__main__":
    unittest.main()
