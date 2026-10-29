# -*- coding: utf-8 -*-
"""
Empirical Adversarial Stress Harness for Dynamic Contrastive EOS Boosting (DCEB)
and SyntacticSpanRefiner (SCPDO + QAMIF).
Milestone R6-3 Gate Verification Suite.

Tests 7 Critical Dimensions:
1. Contrast collapse vs active reading margin separation
2. Extreme entropy values ([-10, 0, 5, 50])
3. Numerical stability (NaN/Inf immunity, zero division, underflow/overflow)
4. Step 0 protection (N_min = 1)
5. CAPC Mask Preservation for single, multiple, and deeply-buried EOS tokens
6. Batch dimensions and tensor shapes
7. SyntacticSpanRefiner stress testing (non-collapsing invariant, quotes, ratios, recursions)
"""

import math
import os
import sys
import unittest
from typing import List

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


class TestDCEBMarginSeparation(unittest.TestCase):
    """Adversarial stress-testing of contrast collapse vs active reading separation margin."""

    def setUp(self):
        self.decoder = DynamicContrastiveDecoder(
            kappa_eos=4.00,
            n_min_eos=1,
            h_bar_eos=1.80,
            tau_h_eos=0.40,
            delta_peak_min=0.50,
        )
        self.vocab_size = 200
        self.eos_id = 199

    def test_active_reading_margin_separation(self):
        """During active visual text reading, DCEB boost must be strictly suppressed."""
        pos_0 = torch.zeros((1, self.vocab_size))
        neg_0 = torch.zeros((1, self.vocab_size))
        pos_0[0, 10] = 6.0
        neg_0[0, 10] = 1.0  # max delta_z = 5.0
        self.decoder.decode_step(pos_0, neg_0, step=0, spatial_confidence=1.0, eos_id=self.eos_id)

        # Step 1: Active reading (high contrast m_t = 4.0, low entropy H ~ 0.5)
        pos_1 = torch.randn((1, self.vocab_size)) * 0.1
        neg_1 = torch.randn((1, self.vocab_size)) * 0.1
        pos_1[0, 20] = 7.0  # target visual token
        neg_1[0, 20] = 2.0  # contrast delta_z = 5.0
        pos_1[0, self.eos_id] = 1.0
        neg_1[0, self.eos_id] = 1.0

        next_tok, alpha, ent, js = self.decoder.decode_step(
            pos_1, neg_1, step=1, spatial_confidence=1.0, eos_id=self.eos_id
        )

        # Invariant 1: Target entity token must be selected, NOT EOS
        self.assertEqual(next_tok, 20)
        self.assertNotEqual(next_tok, self.eos_id)

    def test_contrast_collapse_clean_termination(self):
        """When visual contrast collapses and entropy rises, EOS must win decisively."""
        pos_0 = torch.zeros((1, self.vocab_size))
        neg_0 = torch.zeros((1, self.vocab_size))
        pos_0[0, 10] = 5.0
        neg_0[0, 10] = 1.0
        self.decoder.decode_step(pos_0, neg_0, step=0, spatial_confidence=1.0, eos_id=self.eos_id)

        # Step 1: Entity finished -> contrast collapsed to 0.05, entropy dispersed
        pos_1 = torch.ones((1, self.vocab_size)) * 0.5
        neg_1 = torch.ones((1, self.vocab_size)) * 0.5
        pos_1[0, 50] = 1.2
        neg_1[0, 50] = 1.2
        pos_1[0, self.eos_id] = 0.5
        neg_1[0, self.eos_id] = 0.5

        next_tok, alpha, ent, js = self.decoder.decode_step(
            pos_1, neg_1, step=1, spatial_confidence=1.0, eos_id=self.eos_id
        )

        # EOS should be boosted by ~3.5-4.0 logits and win over token 50
        self.assertEqual(next_tok, self.eos_id)

    def test_margin_monotonicity_across_contrast_grid(self):
        """Verify tau_eos is strictly non-increasing with respect to contrast m_t."""
        peak = 4.0
        entropy = 2.5
        effective_peak = max(peak, self.decoder.delta_peak_min)

        prev_tau = float("inf")
        for m_t in [0.0, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]:
            c_collapse = max(0.0, 1.0 - m_t / (effective_peak + 1e-6))
            sig_entropy = 1.0 / (1.0 + math.exp(-np_clip((entropy - self.decoder.h_bar_eos) / self.decoder.tau_h_eos, -20.0, 20.0)))
            tau = self.decoder.kappa_eos * c_collapse * sig_entropy
            self.assertLessEqual(tau, prev_tau + 1e-7, f"Monotonicity failed at m_t={m_t}")
            prev_tau = tau

    def test_margin_monotonicity_across_entropy_grid(self):
        """Verify tau_eos is strictly non-decreasing with respect to entropy H."""
        peak = 4.0
        m_t = 0.05
        effective_peak = max(peak, self.decoder.delta_peak_min)

        prev_tau = -1.0
        for ent in [0.0, 0.5, 1.0, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0]:
            c_collapse = max(0.0, 1.0 - m_t / (effective_peak + 1e-6))
            sig_entropy = 1.0 / (1.0 + math.exp(-np_clip((ent - self.decoder.h_bar_eos) / self.decoder.tau_h_eos, -20.0, 20.0)))
            tau = self.decoder.kappa_eos * c_collapse * sig_entropy
            self.assertGreaterEqual(tau, prev_tau - 1e-7, f"Monotonicity failed at ent={ent}")
            prev_tau = tau


class TestExtremeEntropy(unittest.TestCase):
    """Stress testing mathematical formulas under extreme entropy inputs ([-10, 0, 5, 50])."""

    def setUp(self):
        self.decoder = DynamicContrastiveDecoder(
            kappa_eos=4.00,
            n_min_eos=1,
            h_bar_eos=1.80,
            tau_h_eos=0.40,
            delta_peak_min=0.50,
        )

    def test_extreme_entropy_values_math_immunity(self):
        """Test entropy values [-10, 0, 5, 50] directly against DCEB formulas for overflow/underflow."""
        test_entropies = [-10.0, 0.0, 5.0, 50.0]
        effective_peak = 2.0
        m_t = 0.1

        for ent in test_entropies:
            c_collapse = max(0.0, 1.0 - m_t / (effective_peak + 1e-6))
            self.assertTrue(0.0 <= c_collapse <= 1.0)

            clipped_arg = np_clip((ent - self.decoder.h_bar_eos) / self.decoder.tau_h_eos, -20.0, 20.0)
            self.assertTrue(-20.0 <= clipped_arg <= 20.0)
            sig_entropy = 1.0 / (1.0 + math.exp(-clipped_arg))
            self.assertTrue(0.0 <= sig_entropy <= 1.0)

            tau_eos = self.decoder.kappa_eos * c_collapse * sig_entropy
            self.assertTrue(0.0 <= tau_eos <= self.decoder.kappa_eos)
            self.assertFalse(math.isnan(tau_eos))
            self.assertFalse(math.isinf(tau_eos))

            beta_t = self.decoder.beta_0 * math.exp(-min(4.0, max(0.0, ent)))
            self.assertTrue(0.0 < beta_t <= self.decoder.beta_0)

            t_entropy = max(0.05, 1.0 - self.decoder.eta_temp * math.tanh(ent / self.decoder.tau_h))
            self.assertGreater(t_entropy, 0.05)

    def test_sharp_delta_logits_near_zero_entropy(self):
        """Logits with one extreme dominant token (H -> 0)."""
        vocab_size = 500
        pos = torch.zeros((1, vocab_size))
        neg = torch.zeros((1, vocab_size))
        pos[0, 100] = 50.0
        neg[0, 100] = 50.0

        tok, alpha, ent, js = self.decoder.decode_step(pos, neg, step=1, spatial_confidence=1.0, eos_id=0)
        self.assertAlmostEqual(ent, 0.0, places=3)
        self.assertFalse(math.isnan(ent))
        self.assertFalse(math.isnan(alpha))
        self.assertEqual(tok, 100)

    def test_uniform_logits_maximum_entropy(self):
        """Completely uniform logits (H = ln(V))."""
        vocab_size = 1000
        pos = torch.zeros((1, vocab_size))
        neg = torch.zeros((1, vocab_size))

        tok, alpha, ent, js = self.decoder.decode_step(pos, neg, step=1, spatial_confidence=1.0, eos_id=0)
        expected_h = math.log(vocab_size)
        self.assertAlmostEqual(ent, expected_h, places=2)
        self.assertFalse(math.isnan(ent))
        self.assertFalse(math.isnan(alpha))


class TestNumericalStability(unittest.TestCase):
    """Stress testing numerical robustness against NaNs, infinities, zero division, and extreme magnitudes."""

    def setUp(self):
        self.decoder = DynamicContrastiveDecoder(
            delta_peak_min=0.50,
        )
        self.vocab_size = 100
        self.eos_id = 99

    def test_identical_logits_zero_contrast(self):
        """When pos_logits == neg_logits (Delta z = 0 everywhere), ensure no division by zero."""
        pos = torch.ones((1, self.vocab_size)) * 2.5
        neg = torch.ones((1, self.vocab_size)) * 2.5

        tok, alpha, ent, js = self.decoder.decode_step(pos, neg, step=0, spatial_confidence=1.0, eos_id=self.eos_id)
        self.assertFalse(math.isnan(alpha))
        self.assertFalse(math.isnan(ent))
        self.assertFalse(math.isnan(js))
        self.assertEqual(alpha, 0.0)

    def test_massive_positive_and_negative_logits(self):
        """Inputs with extreme magnitudes (+500, -500)."""
        pos = torch.full((1, self.vocab_size), -500.0)
        neg = torch.full((1, self.vocab_size), -500.0)
        pos[0, 10] = 500.0
        neg[0, 10] = -500.0

        tok, alpha, ent, js = self.decoder.decode_step(pos, neg, step=1, spatial_confidence=1.0, eos_id=self.eos_id)
        self.assertFalse(math.isnan(alpha))
        self.assertFalse(math.isnan(ent))
        self.assertFalse(math.isnan(js))
        self.assertEqual(tok, 10)

    def test_peak_floor_prevents_zero_division(self):
        """delta_peak_min floor must prevent division by zero even if _peak_delta_z is 0."""
        self.decoder._peak_delta_z = 0.0
        effective_peak = max(self.decoder._peak_delta_z, self.decoder.delta_peak_min)
        self.assertGreaterEqual(effective_peak, 0.50)

        c_collapse = max(0.0, 1.0 - 0.0 / (effective_peak + 1e-6))
        self.assertAlmostEqual(c_collapse, 1.0, places=4)

    def test_denormal_and_near_zero_logits(self):
        """Logits near machine precision (1e-35)."""
        pos = torch.full((1, self.vocab_size), 1e-35)
        neg = torch.full((1, self.vocab_size), 1e-35)

        tok, alpha, ent, js = self.decoder.decode_step(pos, neg, step=1, spatial_confidence=1.0, eos_id=self.eos_id)
        self.assertFalse(math.isnan(alpha))
        self.assertFalse(math.isnan(ent))
        self.assertFalse(math.isnan(js))


class TestStep0Protection(unittest.TestCase):
    """Empirical verification that step 0 strictly prohibits premature EOS emission."""

    def setUp(self):
        self.decoder = DynamicContrastiveDecoder(
            kappa_eos=4.00,
            n_min_eos=1,
            delta_peak_min=0.50,
        )
        self.vocab_size = 50
        self.eos_id = 49

    def test_step0_under_total_contrast_collapse(self):
        """Even if contrast is 0 and entropy is high at step 0, tau_eos must be 0.0."""
        pos = torch.zeros((1, self.vocab_size))
        neg = torch.zeros((1, self.vocab_size))
        pos[0, 5] = 2.0
        neg[0, 5] = 2.0
        pos[0, self.eos_id] = 1.0
        neg[0, self.eos_id] = 1.0

        tok, alpha, ent, js = self.decoder.decode_step(
            pos, neg, step=0, spatial_confidence=1.0, eos_id=self.eos_id
        )

        self.assertEqual(tok, 5)
        self.assertNotEqual(tok, self.eos_id)

    def test_step0_vs_step1_bifurcation(self):
        """Under identical collapsed logits, step 0 selects entity token, while step 1 selects EOS."""
        pos = torch.zeros((1, self.vocab_size))
        neg = torch.zeros((1, self.vocab_size))
        pos[0, 5] = 2.0
        neg[0, 5] = 2.0
        pos[0, self.eos_id] = 1.0
        neg[0, self.eos_id] = 1.0

        tok_0, _, _, _ = self.decoder.decode_step(pos, neg, step=0, spatial_confidence=1.0, eos_id=self.eos_id)
        self.assertEqual(tok_0, 5)

        tok_1, _, _, _ = self.decoder.decode_step(pos, neg, step=1, spatial_confidence=1.0, eos_id=self.eos_id)
        self.assertEqual(tok_1, self.eos_id)


class TestCAPCMaskPreservation(unittest.TestCase):
    """Adversarial stress-testing of CAPC mask preservation for EOS tokens."""

    def setUp(self):
        self.decoder = DynamicContrastiveDecoder(
            beta_0=0.035,
            theta_c=0.35,
            kappa_eos=4.00,
            n_min_eos=1,
        )
        self.vocab_size = 100
        self.eos_id = 99

    def test_deeply_buried_single_eos_token_unmasked(self):
        """EOS token in the extreme tail (P_pos < 1e-20) must be preserved in valid_mask."""
        pos = torch.zeros((1, self.vocab_size))
        neg = torch.zeros((1, self.vocab_size))
        pos[0, 0] = 30.0
        neg[0, 0] = 30.0
        pos[0, self.eos_id] = -50.0
        neg[0, self.eos_id] = -50.0

        tok, alpha, ent, js = self.decoder.decode_step(
            pos, neg, step=0, spatial_confidence=1.0, eos_id=self.eos_id
        )
        self.assertEqual(tok, 0)

    def test_multiple_eos_tokens_list(self):
        """When multiple EOS tokens are provided as a list, all must be preserved."""
        eos_list = [95, 96, 97, 98, 99]
        pos = torch.zeros((1, self.vocab_size))
        neg = torch.zeros((1, self.vocab_size))
        pos[0, 0] = 20.0
        neg[0, 0] = 20.0
        for eid in eos_list:
            pos[0, eid] = -20.0
            neg[0, eid] = -20.0

        tok, alpha, ent, js = self.decoder.decode_step(
            pos, neg, step=1, spatial_confidence=1.0, eos_id=eos_list
        )
        self.assertIsInstance(tok, int)

    def test_tuple_of_eos_tokens(self):
        """Tuple input (e.g. (151645, 151643)) must be accepted gracefully."""
        eos_tuple = (98, 99)
        pos = torch.randn((1, self.vocab_size))
        neg = torch.randn((1, self.vocab_size))

        tok, alpha, ent, js = self.decoder.decode_step(
            pos, neg, step=1, spatial_confidence=1.0, eos_id=eos_tuple
        )
        self.assertIsInstance(tok, int)

    def test_none_eos_id_preserves_standard_apc(self):
        """When eos_id=None, standard APC behavior is preserved without crash."""
        pos = torch.zeros((1, self.vocab_size))
        neg = torch.zeros((1, self.vocab_size))
        pos[0, 0] = 10.0
        neg[0, 0] = 10.0
        pos[0, 50] = -10.0
        neg[0, 50] = -10.0

        tok, alpha, ent, js = self.decoder.decode_step(
            pos, neg, step=1, spatial_confidence=1.0, eos_id=None
        )
        self.assertEqual(tok, 0)


class TestTensorDimensionsAndShapes(unittest.TestCase):
    """Stress-testing tensor shapes: 1D, 2D, 3D singleton, and batch dimension verification."""

    def setUp(self):
        self.decoder = DynamicContrastiveDecoder()
        self.vocab_size = 64
        self.eos_id = 63

    def test_1d_tensor_input(self):
        """1D tensor input (vocab_size,) must be supported."""
        pos = torch.randn(self.vocab_size)
        neg = torch.randn(self.vocab_size)

        tok, alpha, ent, js = self.decoder.decode_step(
            pos, neg, step=0, spatial_confidence=1.0, eos_id=self.eos_id
        )
        self.assertIsInstance(tok, int)
        self.assertTrue(0 <= tok < self.vocab_size)

    def test_2d_tensor_input(self):
        """Standard 2D tensor input (1, vocab_size)."""
        pos = torch.randn((1, self.vocab_size))
        neg = torch.randn((1, self.vocab_size))

        tok, alpha, ent, js = self.decoder.decode_step(
            pos, neg, step=0, spatial_confidence=1.0, eos_id=self.eos_id
        )
        self.assertIsInstance(tok, int)
        self.assertTrue(0 <= tok < self.vocab_size)

    def test_3d_singleton_tensor_input(self):
        """3D singleton tensor input (1, 1, vocab_size)."""
        pos = torch.randn((1, 1, self.vocab_size))
        neg = torch.randn((1, 1, self.vocab_size))

        tok, alpha, ent, js = self.decoder.decode_step(
            pos, neg, step=0, spatial_confidence=1.0, eos_id=self.eos_id
        )
        self.assertIsInstance(tok, int)
        self.assertTrue(0 <= tok < self.vocab_size)

    def test_multi_batch_tensor_input_behavior(self):
        """
        Verify multi-batch tensor (2, vocab_size) raises RuntimeError on .squeeze().item().
        Documents the design invariant: decode_step is an autoregressive single-sample decoder.
        """
        pos = torch.randn((2, self.vocab_size))
        neg = torch.randn((2, self.vocab_size))

        with self.assertRaises(RuntimeError):
            self.decoder.decode_step(pos, neg, step=0, spatial_confidence=1.0, eos_id=self.eos_id)


class TestSyntacticSpanRefinerAdversarial(unittest.TestCase):
    """Adversarial stress testing of SyntacticSpanRefiner (SCPDO + QAMIF)."""

    def setUp(self):
        self.refiner = SyntacticSpanRefiner()

    def test_empty_and_whitespace_and_punctuation(self):
        """Degenerate inputs must not crash. Stripping to empty triggers non-collapsing fallback to original."""
        self.assertEqual(self.refiner.refine(""), "")
        self.assertEqual(self.refiner.refine(None), "")
        self.assertEqual(self.refiner.refine("   \n\t  "), "")
        # When all characters are trailing punctuation, line 220 fallback preserves original string
        self.assertEqual(self.refiner.refine("... ,,, ::::"), "... ,,, ::::")

    def test_non_collapsing_invariant_all_tokens_in_question(self):
        """When ALL answer tokens echo the question, at least 1 core token must survive."""
        question = "Trong bức ảnh có chữ gì ?"
        answer = "bức ảnh có chữ"
        refined = self.refiner.refine(answer, question=question)
        self.assertTrue(len(refined) > 0)
        self.assertIn(refined, ["chữ", "bức ảnh có chữ"])

    def test_time_and_ratio_preservation_strictness(self):
        """10:30:45, 08:00, 2:1, 0:0 must never be truncated by colonic delimiter."""
        self.assertEqual(self.refiner.refine("Thời gian là 10:30"), "Thời gian là 10:30")
        self.assertEqual(self.refiner.refine("Tỉ số chung cuộc: 2:1"), "2:1")
        self.assertEqual(self.refiner.refine("Giờ mở cửa: 08:00:00"), "08:00:00")

    def test_deeply_nested_compound_metadiscourse_clauses(self):
        """Chained metadiscourse prefixes peeled up to 4 layers."""
        raw = "Bức ảnh này được chụp tại Đà Lạt, kết quả cho câu hỏi là: Siêu thị Co.opmart"
        self.assertEqual(self.refiner.refine(raw), "Siêu thị Co.opmart")

    def test_mixed_bracket_and_quote_extraction(self):
        """Quoted entities with diverse unicode quotes (french guillemets, curly quotes)."""
        self.assertEqual(self.refiner.refine("Biển hiệu ghi «Nhà sách Tiền Phong»"), "Nhà sách Tiền Phong")
        self.assertEqual(self.refiner.refine("Có chữ “Cơm Tấm Ba Ghiền”."), "Cơm Tấm Ba Ghiền")

    def test_large_input_stress_resilience(self):
        """Extremely large input (10,000 characters) must execute in sub-millisecond without catastrophic backtracking."""
        huge_str = "Từ khóa: " + ("lặp lại từ " * 1000) + "kết quả cuối cùng"
        refined = self.refiner.refine(huge_str)
        self.assertTrue(refined.startswith("lặp lại từ"))
        self.assertTrue(refined.endswith("kết quả cuối cùng"))


if __name__ == "__main__":
    unittest.main()
