# -*- coding: utf-8 -*-
"""
Empirical Adversarial Stress Harness for Round 7 Generation Dynamics & Span Refiner.
Challenger: challenger_r7_1
Target:
  - scripts/benchmark/sg_vcd/dynamic_cd.py (DynamicContrastiveDecoder)
  - scripts/benchmark/sg_vcd/pipeline.py (SyntacticSpanRefiner)
"""

import math
import os
import sys
import unittest
import torch
import torch.nn.functional as F

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.benchmark.sg_vcd.dynamic_cd import (
    DynamicContrastiveDecoder,
    np_clip,
)
from scripts.benchmark.sg_vcd.pipeline import (
    SyntacticSpanRefiner,
    strip_conversational_preamble,
)


class TestSyntacticSpanRefinerEdgeCases(unittest.TestCase):
    """Adversarial stress-testing of SyntacticSpanRefiner."""

    def setUp(self):
        self.refiner = SyntacticSpanRefiner()

    def test_hyphenated_names_and_scores_preservation(self):
        """Hyphens and spaced dashes in names and scores must not be severed or truncated."""
        cases = [
            ("0 - 1", None, "0 - 1"),
            ("7h - 22h", None, "7h - 22h"),
            ("Highlands Coffee - Tầng 1", "Tên quán cà phê là gì?", "Highlands Coffee - Tầng 1"),
            ("54 - TONKIN - Hanoi", None, "54 - TONKIN - Hanoi"),
            ("54 - TONKIN - Hanoi - Rue Paul Bert", None, "54 - TONKIN - Hanoi - Rue Paul Bert"),
            ("Tỉ số: 3 - 2", "Tỉ số của trận đấu là gì?", "3 - 2"),
            ("Mở cửa: 8h30 - 21h30", "Giờ mở cửa là khi nào?", "8h30 - 21h30"),
            ("Hà Nội - Hải Phòng", "Tuyến đường nào?", "Hà Nội - Hải Phòng"),
        ]
        for raw, q, expected in cases:
            actual = self.refiner.refine(raw, q)
            self.assertEqual(
                actual,
                expected,
                f"Failed on hyphenated input '{raw}' with question '{q}': got '{actual}', expected '{expected}'",
            )

    def test_topic_word_constituent_sharing(self):
        """Answer constituent tokens shared with question stem must not be stripped."""
        cases = [
            ("30 ngày", "đổi máy mới trong vòng bao nhiêu ngày ?", "30 ngày"),
            ("Thứ Tư", "báo này đăng vào thứ mấy ?", "Thứ Tư"),
            ("Cho thuê xe", "ở đây cho thuê cái gì ?", "Cho thuê xe"),
            ("bún bò", "quán phương nam bán bún gì ?", "bún bò"),
            ("Trà sữa", "quán bán trà gì ?", "Trà sữa"),
            ("100.000 đồng", "giá bao nhiêu đồng ?", "100.000 đồng"),
            ("Phòng khám đa khoa", "đây là phòng khám gì ?", "Phòng khám đa khoa"),
            ("Cửa hàng tiện lợi", "đây là cửa hàng gì ?", "Cửa hàng tiện lợi"),
        ]
        for raw, q, expected in cases:
            actual = self.refiner.refine(raw, q)
            self.assertEqual(
                actual,
                expected,
                f"Failed topic-word sharing on '{raw}' with question '{q}': got '{actual}', expected '{expected}'",
            )

    def test_multiline_outputs_with_and_without_headers(self):
        """
        Multiline outputs:
        - With explicit answer header: extract entity from headered line.
        - Without answer header: preserve multiline text intact without blind truncation.
        """
        # With headers
        with_headers = [
            ("Đáp án: Circle K\nCửa hàng tiện lợi", None, "Circle K"),
            ("Câu hỏi: Tên cửa hàng?\nĐáp án: Circle K", "Tên cửa hàng là gì?", "Circle K"),
            ("Cửa hàng tiện lợi\nĐáp án: Circle K", None, "Circle K"),
            ("Trả lời: Highlands Coffee\nĐịa chỉ: Tầng 1", None, "Highlands Coffee"),
            ("Kết luận: Bệnh viện Bạch Mai\nKhoa cấp cứu", None, "Bệnh viện Bạch Mai"),
            ("Câu trả lời: Cơm tấm Ba Ghiền\nQuán ăn", None, "Cơm tấm Ba Ghiền"),
        ]
        for raw, q, expected in with_headers:
            actual = self.refiner.refine(raw, q)
            self.assertEqual(
                actual,
                expected,
                f"Failed headered multiline '{raw}': got '{actual}', expected '{expected}'",
            )

        # Without headers
        without_headers = [
            ("Tầng 1\nTòa nhà Keangnam", None, "Tầng 1\nTòa nhà Keangnam"),
            ("Phở Bò\n24/24", None, "Phở Bò\n24/24"),
            ("Tầng 1\nTòa nhà Keangnam", "Địa chỉ ở đâu?", "Tầng 1\nTòa nhà Keangnam"),
            ("Công ty TNHH ABC\nChi nhánh Hà Nội", None, "Công ty TNHH ABC\nChi nhánh Hà Nội"),
        ]
        for raw, q, expected in without_headers:
            actual = self.refiner.refine(raw, q)
            self.assertEqual(
                actual,
                expected,
                f"Failed unheadered multiline '{raw}': got '{actual}', expected '{expected}'",
            )

    def test_quotes_apostrophes_emojis_whitespace_empty(self):
        """Formatting edge cases: outer quotes stripped, internal apostrophes/emojis preserved."""
        cases = [
            # Empty / degenerate
            ("", None, ""),
            (None, None, ""),
            ("   \t  \n  ", None, ""),
            # Standard quotes
            ('"Circle K"', None, "Circle K"),
            ("“Highlands Coffee”", None, "Highlands Coffee"),
            ("«Nhà thuốc An Khang»", None, "Nhà thuốc An Khang"),
            ("'Bún bò Huế'", None, "Bún bò Huế"),
            ("‘Cơm tấm Sài Gòn’", None, "Cơm tấm Sài Gòn"),
            # Apostrophes within words
            ("McDonald's", None, "McDonald's"),
            ("'McDonald's'", None, "McDonald's"),
            ('"Wendy\'s Burger"', None, "Wendy's Burger"),
            ("Trong ảnh có chữ 'McDonald's'", None, "McDonald's"),
            # Emojis and special unicode symbols
            ("Circle K 🏪", None, "Circle K 🏪"),
            ("🔥 Giảm giá 50% 🔥", None, "🔥 Giảm giá 50% 🔥"),
            ("Coffee ☕", None, "Coffee ☕"),
            # Whitespace and formatting
            ("   Highlands Coffee   ", None, "Highlands Coffee"),
            ("\n\t  15.000đ/kg \t\n", None, "15.000đ/kg"),
        ]
        for raw, q, expected in cases:
            actual = self.refiner.refine(raw, q)
            self.assertEqual(
                actual,
                expected,
                f"Failed formatting case '{raw}': got '{actual}', expected '{expected}'",
            )


class TestDynamicContrastiveDecoderEmpirical(unittest.TestCase):
    """Adversarial stress-testing of DynamicContrastiveDecoder."""

    def test_default_kappa_eos_is_zero(self):
        """Default constructor must have kappa_eos = 0.0 to guarantee generation fidelity."""
        decoder = DynamicContrastiveDecoder()
        self.assertEqual(decoder.kappa_eos, 0.0, "Default kappa_eos MUST be 0.0")

        # Verify that with default kappa_eos=0.0, tau_eos is always 0.0 across all steps
        vocab_size = 50
        eos_id = 49
        pos = torch.ones((1, vocab_size)) * 0.1
        neg = torch.ones((1, vocab_size)) * 0.1
        pos[0, 5] = 2.0
        neg[0, 5] = 2.0
        pos[0, eos_id] = 1.0
        neg[0, eos_id] = 1.0

        for step in range(5):
            tok, alpha, ent, js = decoder.decode_step(
                pos, neg, step=step, spatial_confidence=1.0, eos_id=eos_id
            )
            # Under default kappa_eos=0.0, token 5 (higher logit) must always win, never boosted EOS
            self.assertEqual(tok, 5, f"Step {step}: Token 5 should win, got {tok}")

    def test_eos_not_forcibly_unmasked_when_below_threshold(self):
        """
        When EOS prior/contrast is below APC/CAPC plausibility thresholds,
        it must remain masked (valid_mask is False for EOS) and receive -inf.
        """
        vocab_size = 100
        eos_id = 99
        pos = torch.zeros((1, vocab_size))
        neg = torch.zeros((1, vocab_size))
        pos[0, 10] = 30.0  # dominant visual token
        neg[0, 10] = 5.0
        pos[0, eos_id] = -50.0  # deeply buried EOS
        neg[0, eos_id] = -50.0

        # Case A: Default decoder (kappa_eos = 0.0)
        decoder = DynamicContrastiveDecoder(beta_0=0.035, theta_c=0.35, kappa_eos=0.0)
        tok, alpha, ent, js = decoder.decode_step(
            pos, neg, step=1, spatial_confidence=1.0, eos_id=eos_id
        )
        self.assertEqual(tok, 10)

        # Case B: Decoder with kappa_eos > 0 (e.g. 4.0)
        # Even with kappa_eos=4.0, if EOS is not plausible under CAPC/APC,
        # it must NOT be boosted or unmasked!
        decoder_boosted = DynamicContrastiveDecoder(
            beta_0=0.035,
            theta_c=0.35,
            kappa_eos=4.0,
            n_min_eos=1,
            delta_peak_min=0.5,
        )
        # Prime peak delta_z
        decoder_boosted._peak_delta_z = 25.0

        tok_b, _, _, _ = decoder_boosted.decode_step(
            pos, neg, step=1, spatial_confidence=1.0, eos_id=eos_id
        )
        # Token 10 must win; EOS must NOT be chosen despite kappa_eos=4.0
        self.assertEqual(tok_b, 10)
        self.assertNotEqual(tok_b, eos_id)

    def test_shape_invariants(self):
        """Test 1D, 2D, and 3D singleton inputs."""
        decoder = DynamicContrastiveDecoder()
        vocab_size = 32
        eos_id = 31

        # 1D: (V,)
        pos_1d = torch.randn(vocab_size)
        neg_1d = torch.randn(vocab_size)
        tok_1d, _, _, _ = decoder.decode_step(pos_1d, neg_1d, step=0, spatial_confidence=1.0, eos_id=eos_id)
        self.assertIsInstance(tok_1d, int)
        self.assertTrue(0 <= tok_1d < vocab_size)

        # 2D: (1, V)
        pos_2d = torch.randn((1, vocab_size))
        neg_2d = torch.randn((1, vocab_size))
        tok_2d, _, _, _ = decoder.decode_step(pos_2d, neg_2d, step=0, spatial_confidence=1.0, eos_id=eos_id)
        self.assertIsInstance(tok_2d, int)
        self.assertTrue(0 <= tok_2d < vocab_size)

        # 3D: (1, 1, V)
        pos_3d = torch.randn((1, 1, vocab_size))
        neg_3d = torch.randn((1, 1, vocab_size))
        tok_3d, _, _, _ = decoder.decode_step(pos_3d, neg_3d, step=0, spatial_confidence=1.0, eos_id=eos_id)
        self.assertIsInstance(tok_3d, int)
        self.assertTrue(0 <= tok_3d < vocab_size)

    def test_multi_batch_size_variation(self):
        """Batch size > 1 must raise RuntimeError because decoder is single-sample autoregressive."""
        decoder = DynamicContrastiveDecoder()
        vocab_size = 32
        pos_batch = torch.randn((3, vocab_size))
        neg_batch = torch.randn((3, vocab_size))

        with self.assertRaises(RuntimeError):
            decoder.decode_step(pos_batch, neg_batch, step=0, spatial_confidence=1.0, eos_id=31)

    def test_extreme_entropy_values(self):
        """Extreme entropy regimes: zero entropy (peaked), maximum entropy (uniform), extreme logits."""
        decoder = DynamicContrastiveDecoder()
        vocab_size = 128
        eos_id = 127

        # Regime 1: Zero entropy (single dominant logit)
        pos_zero_h = torch.full((1, vocab_size), -100.0)
        neg_zero_h = torch.full((1, vocab_size), -100.0)
        pos_zero_h[0, 42] = 100.0
        neg_zero_h[0, 42] = 100.0

        tok, alpha, ent, js = decoder.decode_step(pos_zero_h, neg_zero_h, step=1, spatial_confidence=1.0, eos_id=eos_id)
        self.assertAlmostEqual(ent, 0.0, places=3)
        self.assertFalse(math.isnan(ent))
        self.assertFalse(math.isinf(ent))
        self.assertEqual(tok, 42)

        # Regime 2: Maximum entropy (completely uniform logits)
        pos_uniform = torch.zeros((1, vocab_size))
        neg_uniform = torch.zeros((1, vocab_size))

        tok_u, alpha_u, ent_u, js_u = decoder.decode_step(pos_uniform, neg_uniform, step=1, spatial_confidence=1.0, eos_id=eos_id)
        expected_h = math.log(vocab_size)
        self.assertAlmostEqual(ent_u, expected_h, places=2)
        self.assertFalse(math.isnan(ent_u))
        self.assertFalse(math.isinf(ent_u))

        # Regime 3: Extreme magnitude logits (+1000, -1000)
        pos_huge = torch.full((1, vocab_size), -1000.0)
        neg_huge = torch.full((1, vocab_size), -1000.0)
        pos_huge[0, 7] = 1000.0
        neg_huge[0, 7] = 500.0

        tok_h, alpha_h, ent_h, js_h = decoder.decode_step(pos_huge, neg_huge, step=1, spatial_confidence=1.0, eos_id=eos_id)
        self.assertEqual(tok_h, 7)
        self.assertFalse(math.isnan(alpha_h))
        self.assertFalse(math.isnan(ent_h))


if __name__ == "__main__":
    unittest.main()
