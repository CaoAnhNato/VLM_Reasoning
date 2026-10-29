# -*- coding: utf-8 -*-
"""
Adversarial Empirical Stress Harness for Calibrated Metrics (Round 7).
Designed by challenger_r7_2 to rigorously probe:
1. Type C Signboard & Numeric Expansions
2. Anti-Gaming Defenses (Paragraph dumps, length ratio bounds, false positive containment)
3. Extreme Edge Cases (Empty strings, multi-dotted numbers, diacritics, Unicode forms)
"""

import sys
import os
import unittest
import unicodedata

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.benchmark.metrics import (
    compute_exact_match,
    compute_vqa_accuracy,
    compute_token_f1,
    compute_substring_match,
    compute_anls,
    compute_stage2_metrics,
    normalize_numeric_units,
    canonical_normalize_vietnamese_vqa,
    _normalize_with_units,
)


class TestAdversarialTypeCSignboards(unittest.TestCase):
    """Stress-test Type C Signboard Expansions & Concise Answers."""

    def test_01_mandated_signboard_reading(self):
        # GT ['nhà thuốc'], Pred: 'Nhà thuốc Bảo Tâm An' -> MUST be 1.0 (ANLS=1.0, F1=1.0, Substring=1.0)
        gt = ["nhà thuốc"]
        pred = "Nhà thuốc Bảo Tâm An"
        self.assertEqual(compute_anls(pred, gt), 1.0, "Mandated signboard ANLS must be 1.0")
        self.assertEqual(compute_token_f1(pred, gt), 1.0, "Mandated signboard F1 must be 1.0")
        self.assertEqual(compute_substring_match(pred, gt), 1.0, "Mandated signboard Substring must be 1.0")

    def test_02_mandated_numeric_reading(self):
        # GT ['15 . 000'], Pred: '15.000đ/kg' -> MUST be 1.0
        gt = ["15 . 000"]
        pred = "15.000đ/kg"
        self.assertEqual(compute_anls(pred, gt), 1.0, "Mandated numeric ANLS must be 1.0")
        self.assertEqual(compute_token_f1(pred, gt), 1.0, "Mandated numeric F1 must be 1.0")
        self.assertEqual(compute_substring_match(pred, gt), 1.0, "Mandated numeric Substring must be 1.0")
        self.assertEqual(compute_exact_match(pred, gt), 1.0, "Mandated numeric EM must be 1.0")

    def test_03_mandated_concise_to_verbose_gt(self):
        # GT ['nồi lẩu thái siêu to , siêu cay !'], Pred: 'Thái' -> MUST be 1.0
        gt = ["nồi lẩu thái siêu to , siêu cay !"]
        pred = "Thái"
        self.assertEqual(compute_anls(pred, gt), 1.0, "Mandated concise answer ANLS must be 1.0")
        self.assertEqual(compute_token_f1(pred, gt), 1.0, "Mandated concise answer F1 must be 1.0")
        self.assertEqual(compute_substring_match(pred, gt), 1.0, "Mandated concise answer Substring must be 1.0")

    def test_04_additional_signboard_expansions(self):
        test_cases = [
            ("tiệm vàng", "Tiệm vàng Kim Cương 9999"),
            ("quán ăn", "Quán ăn gia đình Cô Ba"),
            ("bệnh viện", "Bệnh viện Đa khoa Quốc tế"),
            ("trường thpt", "Trường THPT Chuyên Lê Hồng Phong"),
            ("ngân hàng", "Ngân hàng Thương mại Cổ phần Ngoại thương"),
        ]
        for gt_str, pred_str in test_cases:
            gt = [gt_str]
            self.assertEqual(compute_anls(pred_str, gt), 1.0, f"ANLS failed for {pred_str} vs {gt_str}")
            self.assertEqual(compute_token_f1(pred_str, gt), 1.0, f"F1 failed for {pred_str} vs {gt_str}")
            self.assertEqual(compute_substring_match(pred_str, gt), 1.0, f"Sub failed for {pred_str} vs {gt_str}")

    def test_05_additional_numeric_units(self):
        test_cases = [
            ("25 . 000 đ", "25.000"),
            ("50k", "50.000 vnđ"),
            ("30 nghìn", "30000"),
            ("10 ngàn", "10.000đ/kg"),
            ("500 k", "500.000"),
            ("1.500.000 đồng", "1500000"),
            ("100.000đ/lít", "100 . 000"),
        ]
        for gt_str, pred_str in test_cases:
            gt = [gt_str]
            self.assertEqual(compute_anls(pred_str, gt), 1.0, f"ANLS failed for {pred_str} vs {gt_str}")
            self.assertEqual(compute_token_f1(pred_str, gt), 1.0, f"F1 failed for {pred_str} vs {gt_str}")
            self.assertEqual(compute_exact_match(pred_str, gt), 1.0, f"EM failed for {pred_str} vs {gt_str}")


class TestAdversarialAntiGaming(unittest.TestCase):
    """Stress-test Anti-Gaming Defenses."""

    def test_01_paragraph_dump_exceeding_token_and_length_ratio(self):
        gt = ["nhà thuốc"]
        # Dump 50+ words containing 'nhà thuốc'
        dump = (
            "đây là một đoạn văn bản rất dài được sinh ra bởi mô hình ngôn ngữ lớn nhằm mục đích "
            "gian lận điểm số benchmark bằng cách nhồi nhét rất nhiều từ ngữ vô nghĩa và dài dòng "
            "xung quanh từ khóa nhà thuốc nhưng mô hình không thực sự biết câu trả lời chính xác mà chỉ "
            "cố gắng ăn gian điểm số mà thôi"
        )
        tokens = dump.split()
        self.assertGreater(len(tokens), 25, "Must exceed 25 tokens")
        self.assertGreater(len(dump), 4 * len(gt[0]), "Must exceed 4x length")

        anls = compute_anls(dump, gt)
        f1 = compute_token_f1(dump, gt)

        self.assertEqual(anls, 0.0, f"Paragraph dump received unwarranted ANLS: {anls}")
        self.assertLess(f1, 0.20, f"Paragraph dump received unwarranted F1: {f1}")

    def test_02_paragraph_dump_with_different_anchor_positions(self):
        # Test keyword at beginning, middle, and end of 40-word dump
        gt = ["cà phê"]
        prefix = "rất nhiều từ ngữ rác rưởi không liên quan kéo dài nhằm tăng độ dài câu văn và đánh lừa thuật toán đánh giá tự động "
        suffix = " tiếp tục thêm các từ ngữ rác rưởi khác để kiểm tra xem thuật toán có bị đánh lừa bởi vị trí của từ khóa hay không"

        # Beginning
        dump_start = "cà phê " + prefix + suffix
        self.assertEqual(compute_anls(dump_start, gt), 0.0)
        self.assertLess(compute_token_f1(dump_start, gt), 0.20)

        # End
        dump_end = prefix + suffix + " cà phê"
        self.assertEqual(compute_anls(dump_end, gt), 0.0)
        self.assertLess(compute_token_f1(dump_end, gt), 0.20)

    def test_03_false_positive_single_letters(self):
        # Single letters must NOT get credit on multi-letter entities
        cases = [
            ("a", ["nhà thuốc"]),
            ("n", ["nhà thuốc"]),
            ("t", ["nhà thuốc"]),
            ("x", ["xe máy"]),
            ("1", ["15.000"]),
            ("0", ["15.000"]),
            ("k", ["50k"]),
        ]
        for pred, gt in cases:
            self.assertEqual(compute_anls(pred, gt), 0.0, f"Single letter '{pred}' got ANLS credit for {gt}")
            self.assertEqual(compute_token_f1(pred, gt), 0.0, f"Single letter '{pred}' got F1 credit for {gt}")

    def test_04_short_substring_below_threshold(self):
        # 2-letter prediction vs long GT: symmetric containment requires len >= 3
        gt = ["nồi lẩu thái siêu to"]
        pred = "th"
        self.assertEqual(compute_anls(pred, gt), 0.0, "2-letter pred must not trigger symmetric ANLS")
        self.assertEqual(compute_token_f1(pred, gt), 0.0, "2-letter pred must not trigger symmetric F1")

    def test_05_unrelated_strings_zero_credit(self):
        cases = [
            ("quán cơm", ["bệnh viện"]),
            ("tiệm sửa xe máy", ["ngân hàng"]),
            ("123456", ["987654"]),
            ("không có", ["trường đại học"]),
        ]
        for pred, gt in cases:
            self.assertEqual(compute_anls(pred, gt), 0.0, f"Unrelated string got ANLS: {pred} vs {gt}")
            self.assertEqual(compute_token_f1(pred, gt), 0.0, f"Unrelated string got F1: {pred} vs {gt}")
            self.assertEqual(compute_substring_match(pred, gt), 0.0, f"Unrelated string got Substring: {pred} vs {gt}")
            self.assertEqual(compute_exact_match(pred, gt), 0.0, f"Unrelated string got EM: {pred} vs {gt}")


class TestAdversarialEdgeCases(unittest.TestCase):
    """Stress-test Extreme Edge Cases & Robustness."""

    def test_01_empty_predictions(self):
        cases = ["", "   ", "\t\n", None]
        gt = ["nhà thuốc"]
        for p in cases:
            self.assertEqual(compute_anls(p, gt), 0.0)
            self.assertEqual(compute_token_f1(p, gt), 0.0)
            self.assertEqual(compute_substring_match(p, gt), 0.0)
            self.assertEqual(compute_exact_match(p, gt), 0.0)

    def test_02_empty_ground_truths(self):
        pred = "nhà thuốc"
        gt_cases = [[], [""], ["   "], [None], ["", "   "]]
        for gt in gt_cases:
            self.assertEqual(compute_anls(pred, gt), 0.0)
            self.assertEqual(compute_token_f1(pred, gt), 0.0)
            self.assertEqual(compute_substring_match(pred, gt), 0.0)
            self.assertEqual(compute_exact_match(pred, gt), 0.0)

    def test_03_both_empty(self):
        self.assertEqual(compute_anls("", []), 0.0)
        self.assertEqual(compute_token_f1("", []), 0.0)
        self.assertEqual(compute_substring_match("", []), 0.0)
        self.assertEqual(compute_exact_match("", []), 0.0)

    def test_04_multiple_dots_and_commas(self):
        # IP addresses, versions, dates, phone numbers
        cases = [
            ("1.2.3.4", ["1.2.3.4"], 1.0),
            ("192.168.1.1", ["192.168.1.1"], 1.0),
            ("v1.0.0", ["v1.0.0"], 1.0),
            ("20/11/2024", ["20/11/2024"], 1.0),
            ("090.123.4567", ["090 123 4567"], 1.0),
            ("15.000.000", ["15000000"], 1.0),
            ("15,000,000,000", ["15000000000"], 1.0),
            ("1.000.000đ", ["1 000 000"], 1.0),
        ]
        for pred, gt, expected in cases:
            self.assertEqual(compute_anls(pred, gt), expected, f"ANLS failed for {pred} vs {gt}")
            self.assertEqual(compute_token_f1(pred, gt), expected, f"F1 failed for {pred} vs {gt}")

    def test_05_unicode_normalization_forms(self):
        text = "Nhà thuốc Bảo Tâm An"
        nfc = unicodedata.normalize("NFC", text)
        nfd = unicodedata.normalize("NFD", text)
        self.assertNotEqual(nfc, nfd, "NFC and NFD strings should differ in raw bytes")

        # NFD pred vs NFC GT
        self.assertEqual(compute_anls(nfd, [nfc]), 1.0)
        self.assertEqual(compute_token_f1(nfd, [nfc]), 1.0)
        self.assertEqual(compute_exact_match(nfd, [nfc]), 1.0)

        # NFC pred vs NFD GT
        self.assertEqual(compute_anls(nfc, [nfd]), 1.0)
        self.assertEqual(compute_token_f1(nfc, [nfd]), 1.0)
        self.assertEqual(compute_exact_match(nfc, [nfd]), 1.0)

    def test_06_casing_invariance(self):
        cases = [
            ("NHÀ THUỐC BẢO TÂM AN", ["nhà thuốc"]),
            ("nHa ThUoC bAo TaM aN", ["nhà thuốc"]),
            ("15.000Đ/KG", ["15 . 000"]),
            ("THÁI", ["nồi lẩu thái siêu to , siêu cay !"]),
        ]
        for pred, gt in cases:
            self.assertEqual(compute_anls(pred, gt), 1.0, f"Casing ANLS failed for {pred} vs {gt}")
            self.assertEqual(compute_token_f1(pred, gt), 1.0, f"Casing F1 failed for {pred} vs {gt}")

    def test_07_full_stage2_pipeline_with_question_context(self):
        # Verify compute_stage2_metrics handles query echo stripping and verified answers
        question = "đây là quán ăn gì ?"
        pred = "đây là quán ăn Cơm Tấm Ba Ghiền"
        gt = ["cơm tấm ba ghiền"]
        verified = ["cơm tấm ba ghiền", "quán cơm tấm ba ghiền"]

        res = compute_stage2_metrics(pred, gt, verified, question=question)
        self.assertEqual(res["anls_gt"], 1.0)
        self.assertEqual(res["f1_gt"], 1.0)
        self.assertEqual(res["substring_gt"], 1.0)
        self.assertEqual(res["anls_verified"], 1.0)
        self.assertEqual(res["f1_verified"], 1.0)
        self.assertEqual(res["substring_verified"], 1.0)


if __name__ == "__main__":
    unittest.main()
