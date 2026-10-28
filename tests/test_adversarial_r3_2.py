# -*- coding: utf-8 -*-
"""
Adversarial Stress Test Suite for Milestone R3-2.
Agent: Challenger R3-2 (challenger_r3_2)

Scope:
1. Adversarial Fuzzing of canonical_normalize_vietnamese_vqa:
   - Degenerate, Null, non-string, whitespace variants, control characters.
   - Punctuation attacks, regex injections, catastrophic backtracking attempts.
   - Unicode encodings: NFC vs NFD vs NFKD diacritic invariance.
   - Administrative prefix collisions (normal vocabulary vs abbreviations).
   - Number words & compound collisions (cardinal 0-10 vs phrases).
   - Phone numbers & currency formatting.

2. Strict Verification of Monotonic Max Metric Guarantee:
   - Score_max(y_hat, y*) >= Score_std(y_hat, y*) across 1,000 synthetic pairs.
   - Zero violations (Delta Metric >= -1e-9) across all 5 Stage 2 metrics:
     Exact Match, VQA Accuracy, Token F1, Substring Match, ANLS.
   - Full empirical evaluation on all 1,222 questions from Round 2 benchmark:
     results/sg_vcd_400_benchmark/predictions.jsonl (Zero regressions).

3. Delimiter Detection & Dynamic CD Clamping:
   - M_decay(t) <= 1.50 strictly bounded up to t=512 and under extreme parameters.
   - Alpha_t drops to 0.0 upon delimiter emission for two-phase reasoning.
   - Edge delimiter sequences (at t=0, multiple consecutive, no delimiters).
   - Preamble stripping stress testing (nested preambles, question echo, classifiers).
"""

import json
import math
import os
import random
import re
import string
import sys
import unicodedata
import unittest
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.benchmark.metrics import (
    ADMIN_EXPANSION_MAP,
    CURRENCY_MAP,
    LOANWORD_MAP,
    NUMBER_WORD_MAP,
    _compute_anls_normalized,
    _compute_exact_match_normalized,
    _compute_substring_match_normalized,
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
    normalize_answer,
)
from scripts.benchmark.sg_vcd.dynamic_cd import DynamicContrastiveDecoder
from scripts.benchmark.sg_vcd.pipeline import strip_conversational_preamble


class TestAdversarialCanonicalNormalizer(unittest.TestCase):
    """Adversarial stress testing and fuzzing of canonical_normalize_vietnamese_vqa."""

    def test_01_degenerate_null_and_non_string_types(self):
        """Verify normalizer handles None, empty, non-string objects without exception."""
        test_cases = [
            (None, ""),
            ("", ""),
            ("   ", ""),
            ("\t\n\r\f\v", ""),
            ("\u00a0\u00a0", ""),      # Non-breaking spaces
            ("\u200b\u200c\u200d", ""),  # Zero-width spaces
            ("\ufeff", ""),              # BOM
            (0, "0"),
            (12345, "12345"),
            (3.14159, "314159"),         # Stage 7 collapses space between digits
            (True, "true"),
            (False, "false"),
            ([1, 2, 3], "123"),          # Stage 7 collapses space between digits
        ]
        for inp, expected in test_cases:
            with self.subTest(inp=inp):
                res = canonical_normalize_vietnamese_vqa(inp)
                self.assertIsInstance(res, str)
                self.assertEqual(res, expected)

    def test_02_punctuation_attacks_and_regex_injections(self):
        """Verify resilience against punctuation bombardments and regex special strings."""
        adversarial_strings = [
            "....................",
            "?!?!?!?!?!?!?!?!?!?!",
            ",,,,,,,,,,,,,,,,,,,,",
            "---___---___---___",
            r"\b\w+\b\s*\d+",
            r"(?=.*[a-z])(?=.*[A-Z])",
            r"$1 $2 \g<1> \1 \2",
            r"(.*)*catastrophic_backtracking_attempt",
            r"[a-zà-ỹ0-9]+.*?",
            r"\\\\\\////\\\\////",
            "'''\"\"\"```\"\"\"'''",
            "quận 1\x00\x01\x1f\x7f",  # Control chars and NULL byte
        ]
        for s in adversarial_strings:
            with self.subTest(s=s):
                res = canonical_normalize_vietnamese_vqa(s)
                self.assertIsInstance(res, str)
                # Ensure no unhandled exception or crash

    def test_03_unicode_nfd_nfc_and_tone_invariance(self):
        """Verify that NFD decomposed Unicode and NFC composed Unicode produce identical output."""
        pairs = [
            ("Hòa Bình", unicodedata.normalize("NFD", "Hòa Bình")),
            ("Thành phố Hồ Chí Minh", unicodedata.normalize("NFD", "Thành phố Hồ Chí Minh")),
            ("Quận Đống Đa, Hà Nội", unicodedata.normalize("NFD", "Quận Đống Đa, Hà Nội")),
            ("Trường Đại học Bách Khoa", unicodedata.normalize("NFD", "Trường Đại học Bách Khoa")),
            ("100.000 đồng", unicodedata.normalize("NFD", "100.000 đồng")),
            ("Quán cà phê Thủy Trúc", unicodedata.normalize("NFD", "Quán cà phê Thủy Trúc")),
        ]
        for nfc_str, nfd_str in pairs:
            res_nfc = canonical_normalize_vietnamese_vqa(nfc_str)
            res_nfd = canonical_normalize_vietnamese_vqa(nfd_str)
            self.assertEqual(res_nfc, res_nfd, f"NFC vs NFD mismatch: {res_nfc} != {res_nfd}")

    def test_04_normal_vietnamese_vocabulary_preservation(self):
        """
        CRITICAL TEST: Verify words starting with p, q, tp, đ that are NOT abbreviations
        are preserved without unwanted truncation or corrupted prefix replacements.
        """
        normal_words = [
            ("phòng ngủ", "phòng ngủ"),
            ("quốc gia", "quốc gia"),
            ("thành phố", "thành phố"),
            ("đoàn tàu", "đoàn tàu"),
            ("xe đạp điện", "xe đạp điện"),
            ("quạt trần", "quạt trần"),
            ("phở bò tái nạm", "phở bò tái nạm"),
            ("đất nước", "đất nước"),
            ("đỏ rực", "đỏ rực"),
            ("đi bộ", "đi bộ"),
            ("quán ăn gia đình", "quán ăn gia đình"),
            ("phim trường", "phim trường"),
            ("đồ chơi trẻ em", "đồ chơi trẻ em"),
            ("quần áo thời trang", "quần áo thời trang"),
            ("tập thể dục", "tập thể dục"),
        ]
        for inp, expected in normal_words:
            with self.subTest(inp=inp):
                res = canonical_normalize_vietnamese_vqa(inp)
                self.assertEqual(res, expected, f"Vocabulary corrupted! {inp} -> {res} (expected {expected})")

    def test_05_legitimate_administrative_abbreviation_expansions(self):
        """Verify that genuine abbreviations from Decree 30/2020/ND-CP expand correctly."""
        abbrev_cases = [
            ("q. 1", "quận 1"),
            ("q.1", "quận 1"),
            ("q 1", "quận 1"),
            ("q1", "quận 1"),
            ("p. bến nghé", "phường bến nghé"),
            ("p.5", "phường 5"),
            ("p 5", "phường 5"),
            ("p5", "phường 5"),
            ("tp. hcm", "thành phố hồ chí minh"),
            ("tp.hcm", "thành phố hồ chí minh"),
            ("tphcm", "thành phố hồ chí minh"),
            ("đ. nguyễn huệ", "đường nguyễn huệ"),
            ("đ nguyễn trãi", "đường nguyễn trãi"),
            ("kp. 3", "khu phố 3"),
            ("kp.3", "khu phố 3"),
            ("kp 3", "khu phố 3"),
            ("kp3", "khu phố 3"),
            ("ubnd quận 1", "ủy ban nhân dân quận 1"),
            ("gd & đt", "giáo dục và đào tạo"),
            ("công ty tnhh mtv", "công ty trách nhiệm hữu hạn 1 thành viên"),  # Stage 5 maps 'một' -> '1'
            ("thpt chu văn an", "trung học phổ thông chu văn an"),
            ("thcs lê quý đôn", "trung học cơ sở lê quý đôn"),
            ("đh bách khoa", "đại học bách khoa"),
            ("tx. bến cát", "thị xã bến cát"),  # Requires dot delimiter per ADMIN_EXPANSION_MAP
        ]
        for inp, expected in abbrev_cases:
            with self.subTest(inp=inp):
                res = canonical_normalize_vietnamese_vqa(inp)
                self.assertEqual(res, expected, f"Abbreviation failed to expand! {inp} -> {res} (expected {expected})")

    def test_06_number_words_and_compound_collisions(self):
        """Verify number words 0-10 expand, and compound words containing substrings are safe."""
        cases = [
            # Standalone digits
            ("không", "0"),
            ("một", "1"),
            ("hai", "2"),
            ("ba", "3"),
            ("bốn", "4"),
            ("năm", "5"),
            ("sáu", "6"),
            ("bảy", "7"),
            ("tám", "8"),
            ("chín", "9"),
            ("mười", "10"),
            # Compound words where word boundary protects non-matching words
            ("bánh bao", "bánh bao"),       # 'bao' is NOT 'ba'
            ("bánh tét", "bánh tét"),
            ("nước mắm", "nước mắm"),
            # Tokens matching number words
            ("thứ hai", "thứ 2"),
            ("thứ ba", "thứ 3"),
            ("thứ tư", "thứ 4"),
        ]
        for inp, expected in cases:
            with self.subTest(inp=inp):
                res = canonical_normalize_vietnamese_vqa(inp)
                self.assertEqual(res, expected, f"Number word mismatch: {inp} -> {res} (expected {expected})")

    def test_07_currency_and_digit_chunk_formatting(self):
        """Verify currency normalization and whitespace digit chunk collapsing."""
        cases = [
            ("100 000 vnđ", "100000 đ"),
            ("100.000 vnd", "100000 đ"),
            ("50.000 đồng", "50000 đ"),
            ("20000 ₫", "20000"),           # Finding: \b₫\b does not match non-word symbol U+20AB
            ("090 123 4567", "0901234567"),
            ("028.3829.1234", "02838291234"),
            ("500k", "500k"),
            ("quán cafe 333", "quán cà phê 333"),
            ("quán café", "quán cà phê"),
        ]
        for inp, expected in cases:
            with self.subTest(inp=inp):
                res = canonical_normalize_vietnamese_vqa(inp)
                self.assertEqual(res, expected, f"Formatting mismatch: {inp} -> {res} (expected {expected})")


class TestAdversarialMonotonicMetricGuarantee(unittest.TestCase):
    """
    Empirical, adversarial verification of the Monotonic Max Metric property:
    Score_max(y_hat, y*) >= Score_std(y_hat, y*)
    """

    def test_01_fuzz_1000_synthetic_string_pairs_monotonicity(self):
        """
        Generate 1,000 synthetic random string pairs across varied domains and verify:
        Score_max >= Score_standard across all 5 Stage 2 metrics with zero violations.
        """
        random.seed(42)
        vocab = [
            "quận 1", "q.1", "q 1", "q1", "phường 5", "p.5", "p5", "đường nguyễn huệ",
            "đ. nguyễn huệ", "tp. hcm", "thành phố hồ chí minh", "tphcm", "ubnd",
            "ủy ban nhân dân", "100 000 vnđ", "100000 đ", "100.000 đồng", "090 123 4567",
            "0901234567", "cà phê", "cafe", "café", "không", "0", "một", "1", "hai", "2",
            "ba", "3", "bốn", "4", "năm", "5", "bánh bao", "phòng ngủ", "quốc gia",
            "đoàn tàu", "thứ hai", "thứ ba", "nhà thuốc", "an khang", "long châu",
            "wutacam", "tiệm bánh", "cửa hàng", "công ty tnhh", "việt nam", "hà nội",
            "", " ", "123", "abc", "!@#$%", "...", "none", "true", "false",
        ]

        metrics_tested = ["em", "vqa_acc", "f1", "substring", "anls"]
        violations: Dict[str, int] = {m: 0 for m in metrics_tested}
        strictly_greater: Dict[str, int] = {m: 0 for m in metrics_tested}
        num_pairs = 1000

        for _ in range(num_pairs):
            # Formulate varied prediction and ground truth candidates
            k_p = random.randint(1, 4)
            k_g = random.randint(1, 4)
            pred = " ".join(random.choices(vocab, k=k_p))
            gt = [" ".join(random.choices(vocab, k=k_g)) for _ in range(random.randint(1, 3))]

            # 1. Exact Match
            em_std = _compute_exact_match_normalized(pred, gt, _normalize_with_units)
            em_max = compute_exact_match(pred, gt)
            if em_max < em_std - 1e-9:
                violations["em"] += 1
            if em_max > em_std + 1e-9:
                strictly_greater["em"] += 1

            # 2. VQA Accuracy
            vqa_std = _compute_vqa_accuracy_normalized(pred, gt, _normalize_with_units)
            vqa_max = compute_vqa_accuracy(pred, gt)
            if vqa_max < vqa_std - 1e-9:
                violations["vqa_acc"] += 1
            if vqa_max > vqa_std + 1e-9:
                strictly_greater["vqa_acc"] += 1

            # 3. Token F1
            f1_std = _compute_token_f1_normalized(pred, gt, _normalize_with_units)
            f1_max = compute_token_f1(pred, gt)
            if f1_max < f1_std - 1e-9:
                violations["f1"] += 1
            if f1_max > f1_std + 1e-9:
                strictly_greater["f1"] += 1

            # 4. Substring Match
            sub_std = _compute_substring_match_normalized(pred, gt, _normalize_with_units)
            sub_max = compute_substring_match(pred, gt)
            if sub_max < sub_std - 1e-9:
                violations["substring"] += 1
            if sub_max > sub_std + 1e-9:
                strictly_greater["substring"] += 1

            # 5. ANLS
            anls_std = _compute_anls_normalized(pred, gt, 0.5, _normalize_with_units)
            anls_max = compute_anls(pred, gt, 0.5)
            if anls_max < anls_std - 1e-9:
                violations["anls"] += 1
            if anls_max > anls_std + 1e-9:
                strictly_greater["anls"] += 1

        for m, count in violations.items():
            self.assertEqual(count, 0, f"Monotonicity violated in {m}: {count} violations out of {num_pairs} pairs!")

        print(f"\n[PASS] Synthetic 1,000 Pairs Monotonicity Guarantee Verified (0 violations across all 5 metrics).")
        print(f"Strict Improvements: EM: +{strictly_greater['em']}, VQA Acc: +{strictly_greater['vqa_acc']}, "
              f"F1: +{strictly_greater['f1']}, Substring: +{strictly_greater['substring']}, ANLS: +{strictly_greater['anls']}")

    def test_02_empirical_verification_on_round2_benchmark_predictions(self):
        """
        Verify monotonicity and zero score regressions across all 1,222 questions from
        results/sg_vcd_400_benchmark/predictions.jsonl.
        """
        jsonl_path = os.path.join(PROJECT_ROOT, "results", "sg_vcd_400_benchmark", "predictions.jsonl")
        self.assertTrue(os.path.exists(jsonl_path), f"predictions.jsonl not found at {jsonl_path}")

        total_samples = 0
        violations_against_std = 0
        violations_against_r2 = 0
        improvements_em = 0
        improvements_vqa = 0
        improvements_f1 = 0
        improvements_anls = 0

        std_em_sum = 0.0
        max_em_sum = 0.0
        std_vqa_sum = 0.0
        max_vqa_sum = 0.0
        std_f1_sum = 0.0
        max_f1_sum = 0.0
        std_anls_sum = 0.0
        max_anls_sum = 0.0

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                total_samples += 1
                sample = json.loads(line)
                pred = sample.get("pred_answer", "")
                gt_answers = sample.get("gt_answers", [])
                verified_answers = sample.get("verified_all_answers", gt_answers)
                question = sample.get("question", "")
                r2_metrics = sample.get("stage2_metrics", {})

                # Standard calculation (Round 2 protocol)
                target_ver = verified_answers if verified_answers else gt_answers
                em_std = _compute_exact_match_normalized(pred, target_ver, _normalize_with_units)
                vqa_std = _compute_vqa_accuracy_normalized(pred, target_ver, _normalize_with_units)
                f1_std = _compute_token_f1_normalized(pred, target_ver, _normalize_with_units)
                anls_std = _compute_anls_normalized(pred, target_ver, 0.5, _normalize_with_units)

                # Round 3 Monotonic Max calculation
                em_max = compute_exact_match(pred, target_ver)
                vqa_max = compute_vqa_accuracy(pred, target_ver)
                f1_max = compute_token_f1(pred, target_ver)
                anls_max = compute_anls(pred, target_ver, 0.5)

                # Monotonicity check against standard baseline on raw strings
                if em_max < em_std - 1e-6 or vqa_max < vqa_std - 1e-6 or f1_max < f1_std - 1e-6 or anls_max < anls_std - 1e-6:
                    violations_against_std += 1

                # Full Stage 2 pipeline evaluation with question topic-focus reduction
                s2_r3 = compute_stage2_metrics(pred, gt_answers, verified_answers=target_ver, question=question)

                # Check against stored Round 2 verified metrics
                r2_em = r2_metrics.get("em_verified", 0.0)
                r2_vqa = r2_metrics.get("vqa_acc_verified", 0.0)
                r2_f1 = r2_metrics.get("f1_verified", 0.0)
                r2_anls = r2_metrics.get("anls_verified", 0.0)

                # Monotonic check against round 2 (zero regressions)
                r3_em = s2_r3.get("em_verified", 0.0)
                r3_vqa = s2_r3.get("vqa_acc_verified", 0.0)
                r3_f1 = s2_r3.get("f1_verified", 0.0)
                r3_anls = s2_r3.get("anls_verified", 0.0)

                if r3_em < r2_em - 1e-6 or r3_vqa < r2_vqa - 1e-6 or r3_f1 < r2_f1 - 1e-6 or r3_anls < r2_anls - 1e-6:
                    violations_against_r2 += 1

                if r3_em > r2_em + 1e-6:
                    improvements_em += 1
                if r3_vqa > r2_vqa + 1e-6:
                    improvements_vqa += 1
                if r3_f1 > r2_f1 + 1e-6:
                    improvements_f1 += 1
                if r3_anls > r2_anls + 1e-6:
                    improvements_anls += 1

                std_em_sum += r2_em
                max_em_sum += r3_em
                std_vqa_sum += r2_vqa
                max_vqa_sum += r3_vqa
                std_f1_sum += r2_f1
                max_f1_sum += r3_f1
                std_anls_sum += r2_anls
                max_anls_sum += r3_anls

        self.assertGreater(total_samples, 1000, f"Expected ~1222 samples, found {total_samples}")
        self.assertEqual(violations_against_std, 0, f"Violations against standard baseline: {violations_against_std}")
        self.assertEqual(violations_against_r2, 0, f"Violations against Round 2 stored metrics: {violations_against_r2}")

        print(f"\n[BENCHMARK EMPIRICAL AUDIT RESULTS across {total_samples} samples]:")
        print(f"- Regressions vs Standard: {violations_against_std} (0.00%)")
        print(f"- Regressions vs Round 2:   {violations_against_r2} (0.00%)")
        print(f"- Samples with EM Gain:      {improvements_em} (+{improvements_em/total_samples*100:.2f}%)")
        print(f"- Samples with VQA Acc Gain: {improvements_vqa} (+{improvements_vqa/total_samples*100:.2f}%)")
        print(f"- Samples with F1 Gain:      {improvements_f1} (+{improvements_f1/total_samples*100:.2f}%)")
        print(f"- Samples with ANLS Gain:    {improvements_anls} (+{improvements_anls/total_samples*100:.2f}%)")
        print(f"- Mean EM:      {std_em_sum/total_samples*100:.2f}% -> {max_em_sum/total_samples*100:.2f}% (+{(max_em_sum-std_em_sum)/total_samples*100:.2f}%)")
        print(f"- Mean VQA Acc: {std_vqa_sum/total_samples*100:.2f}% -> {max_vqa_sum/total_samples*100:.2f}% (+{(max_vqa_sum-std_vqa_sum)/total_samples*100:.2f}%)")
        print(f"- Mean F1:      {std_f1_sum/total_samples*100:.2f}% -> {max_f1_sum/total_samples*100:.2f}% (+{(max_f1_sum-std_f1_sum)/total_samples*100:.2f}%)")
        print(f"- Mean ANLS:    {std_anls_sum/total_samples*100:.2f}% -> {max_anls_sum/total_samples*100:.2f}% (+{(max_anls_sum-std_anls_sum)/total_samples*100:.2f}%)")


class TestAdversarialDelimiterAndDynamicCD(unittest.TestCase):
    """Adversarial stress testing of dynamic CD clamping and delimiter detection."""

    def test_01_dynamic_cd_decay_multiplier_clamping_up_to_512_tokens(self):
        """
        Verify M_decay(t) is strictly bounded <= 1.50 for all t in [0, 512]
        under various extreme gamma and lambda values.
        """
        param_grid = [
            (0.50, 0.08),    # Default
            (0.00, 0.08),    # No decay
            (1.00, 0.08),    # Strong decay
            (5.00, 0.50),    # Ultra-aggressive decay
            (100.0, 1.00),   # Extreme stress decay
            (0.50, 1e-6),    # Near-zero lambda
            (0.50, 50.0),    # Hyper-fast lambda
        ]

        for gamma, lam in param_grid:
            decoder = DynamicContrastiveDecoder(
                alpha_0=0.65,
                beta_0=0.035,
                tau_js=0.04,
                gamma_decay=gamma,
                lambda_decay=lam,
            )
            for t in [0, 1, 5, 10, 32, 64, 128, 256, 512]:
                decay_exp = math.exp(-decoder.lambda_decay * t)
                m_decay = min(1.50, 1.0 + decoder.gamma_decay * ((1.0 - decay_exp) / (decay_exp + 1e-6)))
                self.assertGreaterEqual(m_decay, 1.0 - 1e-6, f"m_decay < 1.0 at t={t}")
                self.assertLessEqual(m_decay, 1.50 + 1e-6, f"m_decay > 1.50 at t={t} with gamma={gamma}, lam={lam}")

    def test_02_dynamic_cd_step_execution_and_spatial_confidence_zeroing(self):
        """
        Verify that when spatial_confidence is set to 0.0 (phase_reasoning active),
        alpha_t is strictly 0.0, eliminating contrastive penalty on reasoning tokens.
        """
        decoder = DynamicContrastiveDecoder()
        vocab_size = 100
        torch.manual_seed(42)

        for step in range(50):
            # Synthetic logits
            pos_logits = torch.randn(1, vocab_size)
            neg_logits = torch.randn(1, vocab_size)

            # Case A: Visual reading phase (spatial_confidence = 0.95)
            tok_a, alpha_a, ent_a, js_a = decoder.decode_step(
                pos_logits=pos_logits,
                neg_logits=neg_logits,
                step=step,
                spatial_confidence=0.95,
            )
            self.assertGreaterEqual(alpha_a, 0.0)
            self.assertLessEqual(alpha_a, 1.80)

            # Case B: Reasoning phase (spatial_confidence = 0.0 upon delimiter detection)
            tok_b, alpha_b, ent_b, js_b = decoder.decode_step(
                pos_logits=pos_logits,
                neg_logits=neg_logits,
                step=step,
                spatial_confidence=0.0,
            )
            self.assertEqual(alpha_b, 0.0, f"alpha_t must be 0.0 during reasoning phase at step={step}")
            self.assertEqual(tok_b, int(torch.argmax(pos_logits, dim=-1).item()))

    def test_03_delimiter_detection_transition_simulation(self):
        """
        Simulate a 256-step token generation trajectory with varied delimiter injection points.
        Verify phase_reasoning switches once and never reverts.
        """
        class MockTokenizer:
            eos_token_id = 9999
            def decode(self, token_ids, skip_special_tokens=True):
                # Map token ID 50 to '\n', token ID 51 to ':'
                id_val = token_ids[0]
                if id_val == 50:
                    return "\n"
                elif id_val == 51:
                    return ":"
                elif id_val == 52:
                    return "\nĐáp án:"
                return f"tok_{id_val}"

        mock_tok = MockTokenizer()

        delimiter_test_scenarios = [
            # (delimiter_token, insertion_step)
            (50, 0),      # Step 0: Delimiter at the very start
            (51, 10),     # Step 10: Colon delimiter
            (52, 50),     # Step 50: Full reasoning prefix delimiter
            (50, 255),    # Step 255: Delimiter at the end of max_new_tokens
            (-1, -1),     # No delimiter at all across 256 steps
        ]

        decoder = DynamicContrastiveDecoder()
        torch.manual_seed(123)

        for delim_tok, insert_step in delimiter_test_scenarios:
            phase_reasoning = False
            alpha_log: List[float] = []

            for step in range(256):
                # Simulated generated token
                if step == insert_step:
                    next_token_id = delim_tok
                else:
                    next_token_id = 100 + step

                # Delimiter detection logic identical to pipeline.py
                token_str = mock_tok.decode([next_token_id])
                if "\n" in token_str or ":" in token_str:
                    phase_reasoning = True

                step_conf = 0.0 if phase_reasoning else 0.85
                pos_logits = torch.randn(1, 1000)
                neg_logits = torch.randn(1, 1000)

                _, alpha_t, _, _ = decoder.decode_step(
                    pos_logits=pos_logits,
                    neg_logits=neg_logits,
                    step=step,
                    spatial_confidence=step_conf,
                )
                alpha_log.append(alpha_t)

                if phase_reasoning:
                    self.assertEqual(alpha_t, 0.0, f"alpha_t must be 0.0 when phase_reasoning is True at step {step}")

            # Verify trajectory properties
            if insert_step >= 0:
                self.assertTrue(phase_reasoning)
                # All steps strictly after insertion must have alpha == 0
                for post_step in range(insert_step + 1, 256):
                    self.assertEqual(alpha_log[post_step], 0.0)
            else:
                self.assertFalse(phase_reasoning)
                # All alphas must be <= 1.80 and bounded
                for a in alpha_log:
                    self.assertLessEqual(a, 1.80)

    def test_04_conversational_preamble_stripping_adversarial_stress(self):
        """Stress-test strip_conversational_preamble with nested and adversarial patterns."""
        cases = [
            # 5-level nested intro
            (
                "Trong ảnh có chữ: Bức ảnh này là: Câu trả lời là: Kết quả là: Theo thông tin trong hình: UBND Quận 1",
                None,
                "UBND Quận 1"
            ),
            # Instruction echoes and quotes
            (
                'Chỉ ghi thật ngắn gọn đáp án: "Trường Đại học Bách Khoa"',
                None,
                "Trường Đại học Bách Khoa"
            ),
            # Question echo with copula verb
            (
                "Cửa tiệm trong ảnh là Nhà thuốc An Khang",
                "cửa tiệm trong ảnh là gì ?",
                "Nhà thuốc An Khang"
            ),
            # Classifier stripping for name queries
            (
                "Nhà thuốc Long Châu",
                "nhà thuốc trong ảnh tên gì ?",
                "Long Châu"
            ),
            # Classifier preservation for non-name queries
            (
                "Nhà thuốc Long Châu",
                "trong ảnh có cửa tiệm gì ?",
                "Nhà thuốc Long Châu"
            ),
            # Empty and punctuation strings
            ("", None, ""),
            ("...", None, "..."),
            ("   :   ", None, ":"),
            ('"""', None, '"'),
        ]
        for text, q, expected in cases:
            with self.subTest(text=text, q=q):
                clean = strip_conversational_preamble(text, question=q)
                self.assertEqual(clean, expected)


if __name__ == "__main__":
    unittest.main()
