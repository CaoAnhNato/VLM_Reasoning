# -*- coding: utf-8 -*-
"""
Unit Test Suite for Relaxed Evaluation Metrics in ViTextVQA Benchmark.
Verifies:
1. Symmetric numeric punctuation spacing and trailing currency/measurement unit normalization.
2. Sub-span ANLS with bounded entity expansion (<= 4x length, <= 6 extra tokens) >= 0.85.
3. Relaxed precision on complete GT token recall (Recall == 1.0) >= 0.80 - 0.85.
4. Anti-gaming paragraph dump defense (> 8 extra tokens or > 4x length strictly not relaxed).
5. Monotonic non-regression guarantee: Score_relaxed >= Score_standard across all metrics.
6. Full compute_stage2_metrics pipeline integration.
"""

import unittest
from typing import List, Sequence

from scripts.benchmark.metrics import (
    _compute_anls_normalized,
    _compute_exact_match_normalized,
    _compute_token_f1_normalized,
    _compute_vqa_accuracy_normalized,
    _normalize_with_units,
    canonical_normalize_vietnamese_vqa,
    compute_anls,
    compute_exact_match,
    compute_stage2_metrics,
    compute_substring_match,
    compute_token_f1,
    compute_vqa_accuracy,
    normalize_numeric_units,
)


class TestNumericUnitNormalization(unittest.TestCase):
    """Test symmetric numeric punctuation spacing and currency/measurement unit handling."""

    def test_01_numeric_punctuation_spacing_variants(self):
        """Punctuation spacing inside numbers must normalize to identical core numbers."""
        variants = [
            "15 . 000",
            "15 , 000",
            "15.000",
            "15,000",
            "15 000",
            "15000",
        ]
        norm_results = [normalize_numeric_units(v) for v in variants]
        expected = "15000"
        for v, res in zip(variants, norm_results):
            self.assertEqual(res, expected, f"Failed for variant '{v}': got '{res}', expected '{expected}'")

    def test_02_trailing_currency_and_compound_units(self):
        """Trailing currency and per-unit combinations attached to numbers must normalize to core quantity."""
        unit_variants = [
            ("15.000đ/kg", "15000"),
            ("15 . 000 đ/lít", "15000"),
            ("15.000 vnđ", "15000"),
            ("15.000 vnd", "15000"),
            ("15.000 đồng", "15000"),
            ("15.000 đ", "15000"),
            ("15.000 đ / kg", "15000"),
            ("15.000đ / chai", "15000"),
            ("50 kg", "50"),
            ("50kg", "50"),
            ("12 lít", "12"),
            ("250 ml", "250"),
            ("30k", "30000"),
            ("30 k", "30000"),
        ]
        for inp, expected in unit_variants:
            res = normalize_numeric_units(inp)
            self.assertEqual(res, expected, f"Failed for unit '{inp}': got '{res}', expected '{expected}'")

    def test_03_symmetric_matching_unit_vs_unitless(self):
        """Unit-bearing model output must match unit-less GT annotations symmetrically across EM, VQA, F1, ANLS."""
        pairs = [
            ("15.000đ/kg", "15 . 000"),
            ("15 . 000", "15.000đ/kg"),
            ("15, 000 đ/lít", "15.000"),
            ("15.000 vnđ", "15000"),
            ("15.000 đồng", "15.000"),
            ("50 kg", "50"),
            ("50", "50 kg"),
        ]
        for pred, gt in pairs:
            gt_seq = [gt]
            self.assertEqual(compute_exact_match(pred, gt_seq), 1.0, f"EM failed for {pred} vs {gt}")
            self.assertEqual(compute_vqa_accuracy(pred, gt_seq), 1.0, f"VQA failed for {pred} vs {gt}")
            self.assertEqual(compute_token_f1(pred, gt_seq), 1.0, f"F1 failed for {pred} vs {gt}")
            self.assertEqual(compute_anls(pred, gt_seq), 1.0, f"ANLS failed for {pred} vs {gt}")
            self.assertEqual(compute_substring_match(pred, gt_seq), 1.0, f"Sub failed for {pred} vs {gt}")

    def test_04_mismatched_numbers_must_not_match(self):
        """Different numbers with or without units must NOT match."""
        pred = "15.000đ/kg"
        gt_different = ["987.654"]
        self.assertEqual(compute_exact_match(pred, gt_different), 0.0)
        self.assertEqual(compute_vqa_accuracy(pred, gt_different), 0.0)
        self.assertEqual(compute_token_f1(pred, gt_different), 0.0)
        self.assertLess(compute_anls(pred, gt_different), 0.5)

        gt_close = ["20.000"]
        self.assertEqual(compute_exact_match(pred, gt_close), 0.0)
        self.assertEqual(compute_vqa_accuracy(pred, gt_close), 0.0)
        self.assertLess(compute_token_f1(pred, gt_close), 0.5)




class TestSignboardEntityRelaxation(unittest.TestCase):
    """Test Type C signboard entity readings where GT is underspecified compared to detailed model prediction."""

    def test_01_type_c_signboard_detail_reading_anls(self):
        """Model reading full signboard entity ('Nhà thuốc Bảo Tâm An') against GT ('nhà thuốc') receives ANLS >= 0.85 (calibrated to 1.0)."""
        cases = [
            ("nhà thuốc", "Nhà thuốc Bảo Tâm An"),
            ("khách sạn", "Khách sạn Mường Thanh Grand"),
            ("cà phê", "Cà phê Trung Nguyên Legend"),
            ("tiệm bánh", "Tiệm bánh ABC Bakery"),
            ("bệnh viện", "Bệnh viện Chợ Rẫy"),
        ]
        for gt, pred in cases:
            anls = compute_anls(pred, [gt])
            self.assertEqual(
                anls, 1.0,
                f"ANLS relaxation failed for pred '{pred}' vs GT '{gt}': got {anls:.3f} (expected 1.0)"
            )

    def test_02_type_c_signboard_detail_reading_f1(self):
        """Model reading full signboard entity with complete GT recall receives Token F1 >= 0.80 - 0.85 (calibrated to 1.0)."""
        cases = [
            ("nhà thuốc", "Nhà thuốc Bảo Tâm An"),
            ("khách sạn", "Khách sạn Mường Thanh Grand"),
            ("cà phê", "Cà phê Trung Nguyên Legend"),
            ("tiệm bánh", "Tiệm bánh ABC Bakery"),
            ("bệnh viện", "Bệnh viện Chợ Rẫy"),
        ]
        for gt, pred in cases:
            f1 = compute_token_f1(pred, [gt])
            self.assertEqual(
                f1, 1.0,
                f"Token F1 relaxation failed for pred '{pred}' vs GT '{gt}': got {f1:.3f} (expected 1.0)"
            )

    def test_03_bounded_expansion_constraint(self):
        """Entity expansion must be bounded: <= 25 tokens or <= 4x length."""
        gt = "nhà thuốc"  # 2 tokens, 9 chars
        # 6 extra tokens (total 8 tokens <= 25 tokens): within boundary
        valid_pred = "nhà thuốc Bảo Tâm An chi nhánh 1"
        self.assertEqual(compute_anls(valid_pred, [gt]), 1.0)
        self.assertEqual(compute_token_f1(valid_pred, [gt]), 1.0)

    def test_04_symmetric_concise_pred_verbose_gt(self):
        """Model giving concise answer against verbose GT sentence receives ANLS=1.0 and F1=1.0."""
        cases = [
            ("nồi lẩu thái siêu to , siêu cay !", "Thái"),
            ("em chỉ làm sai 1 câu", "1 câu"),
            ("đang nói về tiền của người khác !", "của người khác"),
        ]
        for gt, pred in cases:
            anls = compute_anls(pred, [gt])
            f1 = compute_token_f1(pred, [gt])
            self.assertEqual(anls, 1.0, f"Symmetric ANLS failed for pred '{pred}' vs GT '{gt}': got {anls}")
            self.assertEqual(f1, 1.0, f"Symmetric F1 failed for pred '{pred}' vs GT '{gt}': got {f1}")


class TestParagraphDumpDefense(unittest.TestCase):
    """Test anti-gaming protection: paragraph dumps (> 25 tokens and > 4x length) strictly not relaxed."""

    def test_01_paragraph_dump_tokens_strict_no_relaxation(self):
        """Paragraph dump with > 25 tokens and > 4x length must NOT receive relaxation."""
        gt = "nhà thuốc"
        paragraph_dump = (
            "Nhà thuốc Bảo Tâm An chuyên bán các loại thuốc tây thực phẩm chức năng "
            "dụng cụ y tế và mỹ phẩm chính hãng giá rẻ"
        )
        # 26 tokens (> 25) and 104 chars (> 4x of 9 chars) -> must NOT be relaxed
        anls = compute_anls(paragraph_dump, [gt])
        f1 = compute_token_f1(paragraph_dump, [gt])

        self.assertEqual(anls, 0.0, f"Paragraph dump received unwarranted ANLS: {anls}")
        self.assertLess(f1, 0.35, f"Paragraph dump received unwarranted F1: {f1}")

    def test_02_length_ratio_exceeding_bounds_strict_no_relaxation(self):
        """Pred exceeding 25 tokens and 4x GT length must NOT be relaxed."""
        gt = "quán"
        long_paragraph_dump = (
            "quán cơm tấm bình dân sài gòn 123 phục vụ các món ăn sáng trưa chiều tối "
            "với đầy đủ các món sườn bì chả trứng thịt kho tàu canh chua cá lóc thơm ngon đậm đà giá bình dân"
        )
        anls = compute_anls(long_paragraph_dump, [gt])
        f1 = compute_token_f1(long_paragraph_dump, [gt])
        self.assertEqual(anls, 0.0)
        self.assertLess(f1, 0.20)

    def test_03_incomplete_gt_recall_strict_no_f1_relaxation(self):
        """If model missed any GT token (Recall < 1.0) and is not a valid sub-sequence, Token F1 relaxation is strictly NOT triggered."""
        gt = "nhà thuốc tây"  # 3 tokens
        pred = "nhà thuốc Bảo Tâm An"  # missing 'tây', Recall = 2/3
        f1 = compute_token_f1(pred, [gt])
        # Standard F1: 2 * (2/5) * (2/3) / (2/5 + 2/3) = 2 * 0.4 * 0.667 / 1.067 = 0.50
        self.assertLess(f1, 0.60, f"Incomplete recall received unwarranted relaxation: {f1}")


class TestMonotonicNonRegression(unittest.TestCase):
    """Test that all metrics strictly satisfy Score_relaxed >= Score_standard with zero violations."""

    def test_01_monotonic_guarantee_across_domains(self):
        test_pairs = [
            ("15.000đ/kg", ["15 . 000"]),
            ("nhà thuốc bảo tâm an", ["nhà thuốc"]),
            ("quận 1", ["q.1"]),
            ("100000", ["100.000 vnđ"]),
            ("phòng ngủ", ["phòng ngủ"]),
            ("hoàn toàn khác", ["không liên quan"]),
            ("cà phê sữa đá", ["cà phê"]),
            ("0901234567", ["090 123 4567"]),
        ]
        for pred, gt in test_pairs:
            # EM
            em_std = _compute_exact_match_normalized(pred, gt, _normalize_with_units)
            em_max = compute_exact_match(pred, gt)
            self.assertGreaterEqual(em_max, em_std - 1e-9)

            # VQA Acc
            vqa_std = _compute_vqa_accuracy_normalized(pred, gt, _normalize_with_units)
            vqa_max = compute_vqa_accuracy(pred, gt)
            self.assertGreaterEqual(vqa_max, vqa_std - 1e-9)

            # Token F1
            f1_std = _compute_token_f1_normalized(pred, gt, _normalize_with_units)
            f1_max = compute_token_f1(pred, gt)
            self.assertGreaterEqual(f1_max, f1_std - 1e-9)

            # ANLS
            anls_std = _compute_anls_normalized(pred, gt, 0.5, _normalize_with_units)
            anls_max = compute_anls(pred, gt, 0.5)
            self.assertGreaterEqual(anls_max, anls_std - 1e-9)


class TestStage2MetricsPipeline(unittest.TestCase):
    """Test integration of compute_stage2_metrics with relaxed evaluation."""

    def test_01_stage2_metrics_numeric_and_type_c(self):
        # 1. Numeric with units
        res_num = compute_stage2_metrics("15.000đ/kg", ["15 . 000"])
        self.assertEqual(res_num["em_gt"], 1.0)
        self.assertEqual(res_num["f1_gt"], 1.0)
        self.assertEqual(res_num["anls_gt"], 1.0)

        # 2. Type C Signboard
        res_type_c = compute_stage2_metrics("Nhà thuốc Bảo Tâm An", ["nhà thuốc"])
        self.assertGreaterEqual(res_type_c["f1_gt"], 0.85)
        self.assertGreaterEqual(res_type_c["anls_gt"], 0.85)
        self.assertEqual(res_type_c["substring_gt"], 1.0)


if __name__ == "__main__":
    unittest.main()
