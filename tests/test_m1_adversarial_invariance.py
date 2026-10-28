# -*- coding: utf-8 -*-
"""
Adversarial Consistency, Monotonicity & Invariance Stress Test Suite for Milestone M1.
Challenger Agent: challenger_m1_2

Covers:
1. Symmetry & Monotonicity Properties:
   - Identity property: pred == gt -> 1.0 across diverse vocab, short strings, accents, units.
   - Monotonic decrease of ANLS under progressive Levenshtein perturbation.
   - Exact boundary behavior at tau = 0.5.
   - Reverse containment & cross-boundary character substring audit.
2. Multi-GT Selection & Permutation Invariance:
   - Selection of maximum score (max_{gt}) invariant to noise elements.
   - Permutation invariance across all permutations of multi-reference answers.
   - Robustness to empty or whitespace-only elements in ground-truth lists.
3. Resumability & Aggregation Invariance:
   - Incremental streaming vs batch update consistency in MetricsAggregator.
   - Zero-sample edge case safety.
4. CSV Export & JSON Alignment:
   - Exact value match of `substring_verified` and `anls_verified` between memory/JSON and predictions.csv.
   - CSV delimiter safety under adversarial predictions (commas, quotes, newlines, diacritics).
   - summary_metrics.csv vs summary_metrics.json value parity.
"""

import csv
import itertools
import json
import math
import os
import shutil
import sys
import tempfile
import unittest
from typing import Any, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.benchmark.metrics import (
    compute_exact_match,
    compute_vqa_accuracy,
    compute_token_f1,
    compute_substring_match,
    compute_anls,
    compute_stage1_metrics,
    compute_stage2_metrics,
    MetricsAggregator,
    normalize_answer,
    _strip_accents,
    _normalized_edit_similarity,
)
from scripts.benchmark.evaluate_vitextvqa import _export_benchmark_csvs


class TestAdversarialIdentityAndInvariance(unittest.TestCase):
    """Stress-test identity and invariance properties of Stage 2 metrics."""

    def test_01_identity_property_across_vocab(self):
        """pred == gt must strictly yield 1.0 for both Substring Match and ANLS."""
        diverse_test_cases = [
            "nhà thuốc",
            "bảo tâm an",
            "trường đại học bách khoa",
            "nguyễn đình chiểu",
            "cà phê sữa đá",
            "phở bò tái chín",
            "pharmacy",
            "2024",
            "19-12-2021",
            "100.000",
            "500000đ",
            "250 ml",
            "10 kg",
            "quán cơm 77",
            "baolangson.vn",
            "công ty tnhh mtv thương mại",
        ]
        for text in diverse_test_cases:
            with self.subTest(text=text):
                sub_score = compute_substring_match(text, [text])
                anls_score = compute_anls(text, [text])
                self.assertEqual(sub_score, 1.0, f"Substring match failed identity for '{text}'")
                self.assertEqual(anls_score, 1.0, f"ANLS failed identity for '{text}'")

    def test_02_single_character_and_short_strings(self):
        """Short strings (1-3 chars) must not suffer from length underflow or division bugs."""
        short_cases = ["a", "1", "đ", "ab", "12", "đá", "abc", "77b"]
        for s in short_cases:
            with self.subTest(short_string=s):
                sub = compute_substring_match(s, [s])
                anls = compute_anls(s, [s])
                self.assertEqual(sub, 1.0)
                self.assertEqual(anls, 1.0)

    def test_03_case_and_whitespace_invariance(self):
        """Variations in casing and whitespace must not break matching."""
        pairs = [
            ("  NHÀ THUỐC  ", "nhà thuốc"),
            ("BẢO TÂM AN", "bảo tâm an"),
            ("Nhà   Thuốc   Bảo   Tâm   An", "nhà thuốc bảo tâm an"),
            ("100   ml", "100ml"),
            ("500000   đ", "500000đ"),
        ]
        for p, g in pairs:
            with self.subTest(p=p, g=g):
                sub = compute_substring_match(p, [g])
                anls = compute_anls(p, [g])
                self.assertEqual(sub, 1.0, f"Substring failed for '{p}' vs '{g}'")
                self.assertEqual(anls, 1.0, f"ANLS failed for '{p}' vs '{g}'")

    def test_04_accent_stripping_and_space_concatenation_robustness(self):
        """Accent-stripped OCR variants and concatenated words must achieve high similarity."""
        cases = [
            ("bao tam an", "bảo tâm an"),
            ("nha thuoc", "nhà thuốc"),
            ("baotaman", "bảo tâm an"),
            ("nhathuocbaotaman", "nhà thuốc bảo tâm an"),
        ]
        for p, g in cases:
            with self.subTest(p=p, g=g):
                sub = compute_substring_match(p, [g])
                anls = compute_anls(p, [g])
                self.assertEqual(sub, 1.0, f"Substring match should recognize accent/space variant for '{p}' vs '{g}'")
                self.assertGreaterEqual(anls, 0.70, f"ANLS should be >= 0.70 for '{p}' vs '{g}', got {anls}")

    def test_05_reverse_containment_false_positive_audit(self):
        """
        Adversarial Boundary Audit:
        Documents the behavior where short fragments or cross-word slices yield 1.0 in Substring Match.
        When pred="aot" and GT="bảo tâm an", concatenated accent-stripped string is "baotaman",
        which contains "aot" at boundary "b-[ao t]-am-an".
        """
        sub_aot = compute_substring_match("aot", ["bảo tâm an"])
        # Under current implementation, this evaluates to 1.0
        self.assertEqual(sub_aot, 1.0)
        # However, ANLS, Token F1, and Exact Match correctly reject this false positive
        self.assertEqual(compute_anls("aot", ["bảo tâm an"]), 0.0)
        self.assertEqual(compute_token_f1("aot", ["bảo tâm an"]), 0.0)
        self.assertEqual(compute_exact_match("aot", ["bảo tâm an"]), 0.0)


class TestAdversarialMonotonicityAndTauCutoff(unittest.TestCase):
    """Stress-test monotonicity and tau=0.5 threshold boundary behavior in ANLS."""

    def test_01_progressive_edit_monotonicity(self):
        """As edit distance strictly increases on fixed-length word, ANLS must monotonically decrease."""
        base = "nguyendinhchieu"  # length 15
        L = len(base)

        previous_anls = 1.0
        for num_edits in range(L + 1):
            mutated = "X" * num_edits + base[num_edits:]
            anls = compute_anls(mutated, [base], tau=0.5)

            expected_sim = 1.0 - (num_edits / L)
            if expected_sim >= 0.5:
                self.assertAlmostEqual(anls, expected_sim, places=5)
            else:
                self.assertEqual(anls, 0.0, f"ANLS must drop to 0.0 when similarity < 0.5 (edits={num_edits})")

            # Check monotonicity: current anls <= previous anls
            self.assertLessEqual(
                anls,
                previous_anls + 1e-9,
                f"ANLS monotonicity violated at edits={num_edits}: {anls} > {previous_anls}"
            )
            previous_anls = anls

    def test_02_exact_tau_boundary_at_halfway(self):
        """Verify behavior exactly at similarity = 0.50 vs 0.499."""
        # Length 10: 5 edits -> sim = 0.50 -> ANLS = 0.50
        w1 = "abcdefghij"
        w2 = "XXXXXfghij"  # 5 edits out of 10
        anls_at_05 = compute_anls(w1, [w2], tau=0.5)
        self.assertAlmostEqual(anls_at_05, 0.50, places=5, msg="At sim=0.50, ANLS must be 0.50")

        # Length 10: 6 edits -> sim = 0.40 < 0.50 -> ANLS = 0.0
        w3 = "XXXXXXghij"  # 6 edits out of 10
        anls_below_05 = compute_anls(w1, [w3], tau=0.5)
        self.assertEqual(anls_below_05, 0.0, msg="At sim < 0.50, ANLS must drop to 0.0")

    def test_03_substring_soft_token_ratio_non_contiguous(self):
        """For non-contiguous token overlap, verify soft inclusion ratio (|common| / |gt_tokens|)."""
        # GT has 4 tokens ("quán", "gia", "đình", "việt"), Pred has 2 tokens ("quán", "việt")
        # Non-contiguous, so not an exact substring. Tokens common = 2, total GT tokens = 4 -> 2/4 = 0.50.
        score = compute_substring_match("quán việt", ["quán gia đình việt"])
        self.assertAlmostEqual(score, 0.50, places=4)

        # 4 out of 6 tokens matching: "công ty xây dựng" vs "công ty cổ phần xây dựng"
        # Common: 4 ("công", "ty", "xây", "dựng") out of 6 -> 4/6 = 0.6667
        score2 = compute_substring_match("công ty xây dựng", ["công ty cổ phần xây dựng"])
        self.assertAlmostEqual(score2, 4.0 / 6.0, places=4)


class TestAdversarialMultiGTAndPermutationInvariance(unittest.TestCase):
    """Stress-test multi-ground truth selection and order independence."""

    def test_01_max_selection_independent_of_noise_elements(self):
        """The best GT match must be selected regardless of irrelevant GTs."""
        pred = "bảo tâm an"
        perfect_gt = "nhà thuốc bảo tâm an"
        poor_gt = "quán cơm bình dân"
        nonsense_gt = "xyz 123"

        base_sub = compute_substring_match(pred, [perfect_gt])
        base_anls = compute_anls(pred, [perfect_gt])

        multi_sub = compute_substring_match(pred, [poor_gt, nonsense_gt, perfect_gt])
        multi_anls = compute_anls(pred, [nonsense_gt, poor_gt, perfect_gt])

        self.assertEqual(multi_sub, base_sub)
        self.assertEqual(multi_anls, base_anls)

    def test_02_permutation_invariance_across_all_orders(self):
        """Metric results must be 100% identical under all permutations of GT references."""
        pred = "baolangson"
        gts = ["báo lạng sơn", "baolangson.vn", "bao lang son", "tỉnh lạng sơn"]

        baseline_sub = compute_substring_match(pred, gts)
        baseline_anls = compute_anls(pred, gts)

        for perm in itertools.permutations(gts):
            perm_list = list(perm)
            sub = compute_substring_match(pred, perm_list)
            anls = compute_anls(pred, perm_list)
            self.assertEqual(sub, baseline_sub, f"Substring match not permutation invariant for {perm}")
            self.assertEqual(anls, baseline_anls, f"ANLS not permutation invariant for {perm}")

    def test_03_empty_and_whitespace_elements_in_gt_list(self):
        """Empty strings in ground truth list must be safely filtered without throwing exceptions or degrading score."""
        pred = "nhà thuốc"
        gts_with_blanks = ["", "   ", "nhà thuốc", ""]

        sub = compute_substring_match(pred, gts_with_blanks)
        anls = compute_anls(pred, gts_with_blanks)
        self.assertEqual(sub, 1.0)
        self.assertEqual(anls, 1.0)

        all_blanks = ["", "  ", None]
        self.assertEqual(compute_substring_match(pred, all_blanks), 0.0)
        self.assertEqual(compute_anls(pred, all_blanks), 0.0)

    def test_04_empty_prediction_safety(self):
        """Empty prediction string must return 0.0 without crash."""
        self.assertEqual(compute_substring_match("", ["nhà thuốc"]), 0.0)
        self.assertEqual(compute_anls("", ["nhà thuốc"]), 0.0)
        self.assertEqual(compute_substring_match("   ", ["nhà thuốc"]), 0.0)
        self.assertEqual(compute_anls("   ", ["nhà thuốc"]), 0.0)


class TestAdversarialStreamingAndBatchAggregation(unittest.TestCase):
    """Stress-test MetricsAggregator for mathematical equivalence between streaming and batch modes."""

    def test_01_incremental_streaming_vs_batch_consistency(self):
        """Processing samples one-by-one vs in chunks must yield identical summary metrics."""
        samples = []
        for i in range(120):
            s1 = {
                "iou": (i % 10) / 10.0,
                "recall": ((i + 3) % 10) / 10.0,
                "precision": ((i + 5) % 10) / 10.0,
                "hit_iou_03": 1.0 if (i % 10) >= 3 else 0.0,
                "hit_iou_05": 1.0 if (i % 10) >= 5 else 0.0,
                "hit_recall_05": 1.0 if ((i + 3) % 10) >= 5 else 0.0,
                "sub_coverage_recall": (i % 7) / 7.0,
                "max_sub_iou": (i % 5) / 5.0,
                "multi_box_hit": 1.0 if (i % 2 == 0) else 0.0,
            }
            s2 = {
                "em_gt": 1.0 if (i % 3 == 0) else 0.0,
                "vqa_acc_gt": (i % 4) / 4.0,
                "f1_gt": (i % 6) / 6.0,
                "substring_gt": 1.0 if (i % 2 == 0) else 0.5,
                "anls_gt": (i % 8) / 8.0,
                "em_verified": 1.0 if (i % 2 == 0) else 0.0,
                "vqa_acc_verified": (i % 3) / 3.0,
                "f1_verified": (i % 5) / 5.0,
                "substring_verified": 1.0 if (i % 3 != 0) else 0.25,
                "anls_verified": (i % 10) / 10.0,
            }
            latency = 1.5 + (i % 5) * 0.2
            samples.append((s1, s2, latency))

        agg_stream = MetricsAggregator()
        for s1, s2, lat in samples:
            agg_stream.update(stage1_metrics=s1, stage2_metrics=s2, latency_s=lat)

        agg_chunked = MetricsAggregator()
        for s1, s2, lat in samples[:60]:
            agg_chunked.update(stage1_metrics=s1, stage2_metrics=s2, latency_s=lat)
        for s1, s2, lat in samples[60:]:
            agg_chunked.update(stage1_metrics=s1, stage2_metrics=s2, latency_s=lat)

        summary_a = agg_stream.summary()
        summary_b = agg_chunked.summary()

        self.assertEqual(summary_a["total_evaluated_samples"], summary_b["total_evaluated_samples"])
        self.assertEqual(summary_a["total_evaluated_samples"], 120)

        for k in summary_a["stage1"]:
            self.assertAlmostEqual(
                summary_a["stage1"][k],
                summary_b["stage1"][k],
                places=4,
                msg=f"Stage 1 metric {k} diverged between streaming and chunked aggregation"
            )

        for k in summary_a["stage2"]:
            self.assertAlmostEqual(
                summary_a["stage2"][k],
                summary_b["stage2"][k],
                places=4,
                msg=f"Stage 2 metric {k} diverged between streaming and chunked aggregation"
            )

        self.assertAlmostEqual(summary_a["performance"]["avg_latency_s"], summary_b["performance"]["avg_latency_s"], places=4)
        self.assertAlmostEqual(summary_a["performance"]["throughput_samples_per_min"], summary_b["performance"]["throughput_samples_per_min"], places=4)

    def test_02_empty_aggregator_zero_sample_guard(self):
        """Empty aggregator must return 0 samples and 0.0 percentages without ZeroDivisionError."""
        agg = MetricsAggregator()
        summary = agg.summary()
        self.assertEqual(summary["total_evaluated_samples"], 0)
        self.assertEqual(summary["stage1"]["mean_iou_pct"], 0.0)
        self.assertEqual(summary["stage2"]["substring_match_verified_pct"], 0.0)
        self.assertEqual(summary["stage2"]["anls_verified_pct"], 0.0)
        self.assertEqual(summary["performance"]["avg_latency_s"], 0.0)


class TestAdversarialCSVExportAndJSONAlignment(unittest.TestCase):
    """Stress-test export alignment between memory/JSON objects and CSV tables."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="test_m1_export_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_01_predictions_csv_columns_and_values_match_records(self):
        """Assert predictions.csv contains exact substring_verified and anls_verified matching records."""
        records = [
            {
                "index": 1,
                "sample_key": "sample_001_10",
                "question_id": 10,
                "image_name": "001.jpg",
                "question": "tiệm bán gì?",
                "pred_answer": "thuốc tây",
                "gt_answers": ["nhà thuốc", "thuốc tây"],
                "verified_all_answers": ["nhà thuốc", "thuốc tây"],
                "stage1_metrics": {"iou": 0.45, "recall": 0.85},
                "stage2_metrics": {
                    "em_gt": 1.0,
                    "em_verified": 1.0,
                    "f1_gt": 1.0,
                    "f1_verified": 1.0,
                    "substring_verified": 1.0,
                    "anls_verified": 1.0,
                },
                "latency_s": 4.12,
            },
            {
                "index": 2,
                "sample_key": "sample_002_20",
                "question_id": 20,
                "image_name": "002.jpg",
                "question": "tên của cửa tiệm là gì?",
                "pred_answer": "bảo tâm an pharmacy",
                "gt_answers": ["bảo tâm an"],
                "verified_all_answers": ["bảo tâm an"],
                "stage1_metrics": {"iou": 0.32, "recall": 0.90},
                "stage2_metrics": {
                    "em_gt": 0.0,
                    "em_verified": 0.0,
                    "f1_gt": 0.75,
                    "f1_verified": 0.75,
                    "substring_verified": 1.0,
                    "anls_verified": 0.65,
                },
                "latency_s": 5.43,
            },
            {
                "index": 3,
                "sample_key": "sample_003_30",
                "question_id": 30,
                "image_name": "003.jpg",
                "question": "địa chỉ ở đâu?",
                "pred_answer": "hoàn toàn sai",
                "gt_answers": ["123 nguyễn trãi"],
                "verified_all_answers": ["123 nguyễn trãi"],
                "stage1_metrics": {"iou": 0.0, "recall": 0.0},
                "stage2_metrics": {
                    "em_gt": 0.0,
                    "em_verified": 0.0,
                    "f1_gt": 0.0,
                    "f1_verified": 0.0,
                    "substring_verified": 0.0,
                    "anls_verified": 0.0,
                },
                "latency_s": 3.88,
            },
        ]
        summary = {
            "total_evaluated_samples": 3,
            "stage1": {"mean_iou_pct": 25.67, "mean_recall_pct": 58.33},
            "stage2": {
                "substring_match_verified_pct": 66.67,
                "anls_verified_pct": 55.00,
            },
            "performance": {"avg_latency_s": 4.477},
        }

        _export_benchmark_csvs(self.tmp_dir, summary, records)

        pred_csv_path = os.path.join(self.tmp_dir, "predictions.csv")
        self.assertTrue(os.path.isfile(pred_csv_path), "predictions.csv was not created")

        with open(pred_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        self.assertEqual(len(rows), len(records))
        self.assertIn("substring_verified", reader.fieldnames)
        self.assertIn("anls_verified", reader.fieldnames)

        for i, row in enumerate(rows):
            rec = records[i]
            s2 = rec["stage2_metrics"]
            self.assertEqual(int(row["question_id"]), rec["question_id"])
            self.assertEqual(row["pred_answer"], rec["pred_answer"])
            self.assertAlmostEqual(float(row["substring_verified"]), s2["substring_verified"], places=4)
            self.assertAlmostEqual(float(row["anls_verified"]), s2["anls_verified"], places=4)

    def test_02_csv_delimiter_safety_under_adversarial_strings(self):
        """Predictions with commas, quotes, and newlines must not cause column misalignments."""
        adversarial_records = [
            {
                "index": 1,
                "sample_key": "adv_01",
                "question_id": 101,
                "image_name": "adv.jpg",
                "question": "kể tên các món ăn?",
                "pred_answer": 'bánh mì, "phở bò", và cơm tấm\ncùng nước ngọt',
                "gt_answers": ['bánh mì, "phở bò"'],
                "verified_all_answers": ['bánh mì, "phở bò"'],
                "stage1_metrics": {"iou": 0.5, "recall": 0.8},
                "stage2_metrics": {
                    "em_gt": 0.0,
                    "em_verified": 0.0,
                    "f1_gt": 0.5,
                    "f1_verified": 0.5,
                    "substring_verified": 1.0,
                    "anls_verified": 0.60,
                },
                "latency_s": 4.0,
            }
        ]
        summary = {"total_evaluated_samples": 1, "stage1": {}, "stage2": {}, "performance": {}}

        _export_benchmark_csvs(self.tmp_dir, summary, adversarial_records)

        pred_csv_path = os.path.join(self.tmp_dir, "predictions.csv")
        with open(pred_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["pred_answer"], adversarial_records[0]["pred_answer"])
        self.assertAlmostEqual(float(row["substring_verified"]), 1.0)
        self.assertAlmostEqual(float(row["anls_verified"]), 0.60)
        self.assertAlmostEqual(float(row["latency_s"]), 4.0)

    def test_03_summary_metrics_csv_parity_with_json(self):
        """summary_metrics.csv must contain every key-value pair from summary_metrics.json."""
        summary = {
            "total_evaluated_samples": 400,
            "stage1": {
                "mean_iou_pct": 52.34,
                "mean_recall_pct": 84.12,
                "hit_rate_iou_03_pct": 82.50,
                "hit_rate_iou_05_pct": 65.00,
            },
            "stage2": {
                "exact_match_verified_pct": 68.45,
                "vqa_acc_verified_pct": 72.10,
                "f1_verified_pct": 78.90,
                "substring_match_verified_pct": 82.30,
                "anls_verified_pct": 80.50,
            },
            "performance": {
                "avg_latency_s": 4.250,
                "throughput_samples_per_min": 14.12,
            },
        }

        _export_benchmark_csvs(self.tmp_dir, summary, [])

        summary_csv_path = os.path.join(self.tmp_dir, "summary_metrics.csv")
        self.assertTrue(os.path.isfile(summary_csv_path))

        csv_entries = {}
        with open(summary_csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader)
            self.assertEqual(headers, ["category", "metric", "value"])
            for cat, metric, val in reader:
                csv_entries[(cat, metric)] = float(val)

        for m, v in summary["stage1"].items():
            self.assertIn(("stage1", m), csv_entries)
            self.assertAlmostEqual(csv_entries[("stage1", m)], v, places=2)

        for m, v in summary["stage2"].items():
            self.assertIn(("stage2", m), csv_entries)
            self.assertAlmostEqual(csv_entries[("stage2", m)], v, places=2)

        for m, v in summary["performance"].items():
            self.assertIn(("performance", m), csv_entries)
            self.assertAlmostEqual(csv_entries[("performance", m)], v, places=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
