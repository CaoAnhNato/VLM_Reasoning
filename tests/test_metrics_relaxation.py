# -*- coding: utf-8 -*-
"""
Unit Test Suite for Strict A* Evaluation Metrics in ViTextVQA & CERV Benchmark.
Verifies:
1. Official SQuAD / ViTextVQA text normalization (NFC + lowercase + punctuation to space).
2. Strict Exact Match (EM) and VQA Consensus Accuracy without regex/unit rewriting.
3. Honest SQuAD bag-of-words Token F1 (no substring=1.0 override, no recall=1.0 override).
4. Honest ICDAR DocVQA ANLS with tau=0.5 (no substring=1.0 override, no sub-span sliding window).
5. Independent Substring Match diagnostic indicator.
6. Zero query-echo stripping in compute_stage2_metrics.
"""

import unittest

from scripts.benchmark.metrics import (
    compute_anls,
    compute_exact_match,
    compute_stage2_metrics,
    compute_substring_match,
    compute_token_f1,
    compute_vqa_accuracy,
    normalize_answer,
)


class TestStrictAStarEvaluationProtocol(unittest.TestCase):
    """Verify that metrics.py strictly follows the honest A* evaluation protocol."""

    def test_01_normalize_answer_standard(self):
        self.assertEqual(normalize_answer("  Nhà Thuốc Minh-An!  "), "nhà thuốc minh an")
        self.assertEqual(normalize_answer("19-12-2021"), "19 12 2021")
        self.assertEqual(normalize_answer(None), "")

    def test_02_exact_match_and_vqa_accuracy(self):
        self.assertEqual(compute_exact_match("Minh An", ["minh an"]), 1.0)
        self.assertEqual(compute_exact_match("Nhà thuốc Minh An", ["minh an"]), 0.0)
        self.assertEqual(
            compute_vqa_accuracy("minh an", ["minh an", "minh an", "bảo tâm", "minh an"]),
            1.0,
        )
        self.assertAlmostEqual(
            compute_vqa_accuracy("minh an", ["minh an", "bảo tâm", "an khang"]),
            1.0 / 3.0,
            places=5,
        )

    def test_03_token_f1_strict_no_substring_inflation(self):
        # pred has 5 tokens ("nhà thuốc bảo tâm an"), gt has 2 tokens ("nhà thuốc")
        # P = 2/5 = 0.4, R = 2/2 = 1.0 -> F1 = 2 * 0.4 * 1.0 / 1.4 = 4/7 (~0.5714), NOT 1.0!
        f1 = compute_token_f1("Nhà thuốc Bảo Tâm An", ["nhà thuốc"])
        self.assertAlmostEqual(f1, 4.0 / 7.0, places=5)

    def test_04_anls_strict_no_substring_inflation(self):
        # "nhà thuốc" (9 chars) vs "nhà thuốc bảo tâm an" (20 chars): edit distance = 11 -> sim = 9/20 = 0.45 < 0.5 -> ANLS = 0.0
        anls = compute_anls("Nhà thuốc Bảo Tâm An", ["nhà thuốc"], tau=0.5)
        self.assertEqual(anls, 0.0)

        # Minor 1-char OCR typo: "028 3925 1842" (13 chars) vs "028 3925 1843" (13 chars) -> sim = 12/13 (~0.9231)
        anls_close = compute_anls("028 3925 1843", ["028 3925 1842"], tau=0.5)
        self.assertAlmostEqual(anls_close, 12.0 / 13.0, places=5)

    def test_05_substring_match_independent_indicator(self):
        self.assertEqual(compute_substring_match("Số điện thoại là 028 3925 1842", ["028 3925 1842"]), 1.0)
        # Concise prediction inside verbose GT is NOT a substring match of GT in pred
        self.assertEqual(compute_substring_match("Thái", ["nồi lẩu thái siêu to"]), 0.0)

    def test_06_stage2_metrics_no_query_echo_stripping(self):
        # Even if question contains "Số điện thoại trên bảng là", a verbose answer repeating it
        # must NOT get EM = 1.0 or F1 = 1.0
        s2 = compute_stage2_metrics(
            pred_answer="Số điện thoại trên bảng là 028 3925 1842",
            gt_answers=["028 3925 1842"],
            verified_answers=["028 3925 1842"],
            question="Số điện thoại trên bảng là gì?",
        )
        self.assertEqual(s2["em_verified"], 0.0)
        self.assertLess(s2["f1_verified"], 0.60)
        self.assertEqual(s2["substring_verified"], 1.0)


if __name__ == "__main__":
    unittest.main()
