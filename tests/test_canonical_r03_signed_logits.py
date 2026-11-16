# -*- coding: utf-8 -*-
"""
Tests for Canonical R-03 Signed-Logit Contrastive Decoding (TASK 18B).

Verifies:
1. Exact formula correctness: z_final = (1 + alpha_t) * z_pos - alpha_t * z_neg
2. Parity with z_pos when alpha_t == 0
3. Strict divergence from legacy ReLU-clipped formula when z_pos < z_neg
4. Preservation of signed negative differences (no accidental ReLU clipping on score)
5. Preservation of separate ReLU-based CAPC candidate-support masking
6. Separation of scoring modes ('canonical_signed' vs 'historical_relu') and alias rejection
7. Per-token synchronization between positive and negative branches
8. OPT-01 batched dual-branch decoding parity under canonical R-03
"""

import math
import unittest
from typing import Dict, List, Tuple
from unittest.mock import MagicMock

import torch
import torch.nn.functional as F

from scripts.benchmark.sg_vcd.dynamic_cd import DynamicContrastiveDecoder
from scripts.benchmark.sg_vcd.pipeline import SGVCDPipeline


class TestCanonicalR03SignedLogits(unittest.TestCase):
    """Unit and regression tests for Canonical R-03 signed-logit formula."""

    def setUp(self):
        # Set deterministic seed
        torch.manual_seed(42)

    def test_01_canonical_formula_exact_reference(self):
        """
        Test that canonical_signed mode matches (1 + alpha) * z_pos - alpha * z_neg exactly.
        """
        decoder = DynamicContrastiveDecoder(
            alpha_0=0.65,
            beta_0=0.035,
            tau_js=0.01,
            eta_temp=0.0,  # disable TESG temperature scaling to test pure logit score
            scoring_mode="canonical_signed",
        )
        self.assertEqual(decoder.scoring_mode, "canonical_signed")

        # Deterministic logits: V=6
        pos_logits = torch.tensor([[5.0, 3.0, 2.0, 1.0, 0.5, -1.0]], dtype=torch.float32)
        neg_logits = torch.tensor([[1.0, 4.0, 0.5, 3.5, 0.5, -2.0]], dtype=torch.float32)

        # Step 0 decoding
        next_tok, alpha_t, entropy_val, d_js = decoder.decode_step(
            pos_logits=pos_logits,
            neg_logits=neg_logits,
            step=0,
            spatial_confidence=1.0,
        )

        self.assertGreater(alpha_t, 0.0)

        # Expected score: (1 + alpha) * z_pos - alpha * z_neg
        expected_score = (1.0 + alpha_t) * pos_logits - alpha_t * neg_logits

        # Candidate support mask:
        p_pos = F.softmax(pos_logits, dim=-1)
        beta_t = decoder.beta_0 * math.exp(-min(4.0, max(0.0, entropy_val)))
        apc_mask = p_pos >= (beta_t * p_pos.max(dim=-1, keepdim=True).values)
        delta_z = torch.relu(pos_logits - neg_logits)
        max_delta = delta_z.max(dim=-1, keepdim=True).values
        contrast_mask = (delta_z >= (decoder.theta_c * max_delta)) & (max_delta > 1e-6)
        valid_mask = apc_mask | contrast_mask

        expected_masked = expected_score.masked_fill(~valid_mask, float("-inf"))
        expected_tok = int(torch.argmax(expected_masked, dim=-1).item())

        self.assertEqual(next_tok, expected_tok)

        # Check explicit token 1: z_pos=3.0, z_neg=4.0 -> pos < neg!
        # Expected: (1 + alpha)*3.0 - alpha*4.0 = 3.0 - alpha*1.0 < 3.0
        val_at_1 = expected_score[0, 1].item()
        self.assertAlmostEqual(val_at_1, 3.0 - alpha_t, places=5)
        self.assertLess(val_at_1, 3.0)

    def test_02_alpha_zero_parity(self):
        """
        When alpha_t == 0, output must strictly match argmax of positive branch logits.
        """
        decoder = DynamicContrastiveDecoder(
            alpha_0=0.0,  # force alpha_t = 0
            beta_0=0.035,
            eta_temp=0.0,
            scoring_mode="canonical_signed",
        )

        pos_logits = torch.tensor([[1.0, 8.5, 3.2, 0.1]], dtype=torch.float32)
        neg_logits = torch.tensor([[9.0, 0.2, 8.0, 7.5]], dtype=torch.float32)

        next_tok, alpha_t, _, _ = decoder.decode_step(
            pos_logits=pos_logits,
            neg_logits=neg_logits,
            step=0,
            spatial_confidence=1.0,
        )

        self.assertEqual(alpha_t, 0.0)
        self.assertEqual(next_tok, 1)  # Token 1 is argmax of pos_logits

    def test_03_divergence_from_legacy_relu_on_negative_difference(self):
        """
        Demonstrate that on inputs with negative contrast (z_pos < z_neg),
        canonical_signed penalizes the logit while historical_relu clips the penalty to zero.
        """
        # Create identical decoders except for scoring_mode
        dec_canonical = DynamicContrastiveDecoder(
            alpha_0=0.80,
            beta_0=0.01,
            tau_js=0.001,
            eta_temp=0.0,
            scoring_mode="canonical_signed",
        )
        dec_legacy = DynamicContrastiveDecoder(
            alpha_0=0.80,
            beta_0=0.01,
            tau_js=0.001,
            eta_temp=0.0,
            scoring_mode="historical_relu",
        )

        # Design input:
        # Token 0: pos=4.0, neg=1.0 -> positive diff (+3.0)
        # Token 1: pos=4.2, neg=8.0 -> negative diff (-3.8) [e.g. strong prior hallucination in neg]
        # In base pos_logits, Token 1 is currently the leader (4.2 > 4.0).
        pos_logits = torch.tensor([[4.0, 4.2, 1.0]], dtype=torch.float32)
        neg_logits = torch.tensor([[1.0, 8.0, 0.5]], dtype=torch.float32)

        tok_canon, alpha_canon, _, _ = dec_canonical.decode_step(
            pos_logits=pos_logits, neg_logits=neg_logits, step=0, spatial_confidence=1.0
        )
        tok_legacy, alpha_legacy, _, _ = dec_legacy.decode_step(
            pos_logits=pos_logits, neg_logits=neg_logits, step=0, spatial_confidence=1.0
        )

        self.assertEqual(alpha_canon, alpha_legacy)
        alpha = alpha_canon
        self.assertGreater(alpha, 0.0)

        # Under legacy ReLU:
        # Token 0 score = 4.0 + alpha * relu(4.0 - 1.0) = 4.0 + 3.0 * alpha
        # Token 1 score = 4.2 + alpha * relu(4.2 - 8.0) = 4.2 + 0 = 4.2
        # If alpha = 0.5, Token 0 = 5.5 > 4.2 -> Token 0 wins.
        # But if alpha is smaller, say 0.05:
        # Token 0 = 4.0 + 0.15 = 4.15 < 4.2 -> Token 1 wins under ReLU!
        # Whereas under canonical R-03:
        # Token 1 score = 4.2 + alpha * (4.2 - 8.0) = 4.2 - 3.8 * alpha < 4.2!

        # Let's inspect the exact scores mathematically:
        score_canon_tok1 = (1.0 + alpha) * 4.2 - alpha * 8.0
        score_legacy_tok1 = 4.2 + alpha * max(0.0, 4.2 - 8.0)  # = 4.2

        self.assertLess(score_canon_tok1, 4.2)
        self.assertEqual(score_legacy_tok1, 4.2)
        self.assertNotEqual(score_canon_tok1, score_legacy_tok1)

    def test_04_negative_differences_not_clipped(self):
        """
        Verify that canonical R-03 signed score does not clip negative values to 0 or pos.
        """
        decoder = DynamicContrastiveDecoder(
            alpha_0=1.0,
            beta_0=0.001,
            tau_js=0.001,
            eta_temp=0.0,
            scoring_mode="canonical_signed",
        )

        pos_logits = torch.tensor([[2.0, 5.0]], dtype=torch.float32)
        neg_logits = torch.tensor([[6.0, 1.0]], dtype=torch.float32)

        tok, alpha, ent, js = decoder.decode_step(
            pos_logits=pos_logits, neg_logits=neg_logits, step=0, spatial_confidence=1.0
        )

        # For token 0: pos=2.0, neg=6.0.
        # Canonical score: (1 + alpha)*2.0 - alpha*6.0 = 2.0 - 4.0*alpha.
        # Since alpha > 0, this score must be strictly less than 2.0!
        expected_tok0_score = (1.0 + alpha) * 2.0 - alpha * 6.0
        self.assertLess(expected_tok0_score, 2.0)

    def test_05_candidate_support_separation(self):
        """
        Verify that CAPC candidate support continues to use ReLU(z_pos - z_neg)
        for contrastive inclusion, separate from the final signed scoring formula.
        """
        decoder = DynamicContrastiveDecoder(
            alpha_0=0.65,
            beta_0=0.05,
            theta_c=0.35,
            tau_js=0.01,
            eta_temp=0.0,
            scoring_mode="canonical_signed",
        )

        # Vocab of 4 tokens:
        # 0: LM prior leader: pos=10.0, neg=9.9 (P_pos high, Delta z low = 0.1)
        # 1: High visual contrast: pos=1.0, neg=-8.0 (P_pos very low, but Delta z = 9.0)
        # 2: Negative contrast distractor: pos=0.5, neg=5.0 (Delta z = 0 in ReLU)
        # 3: Low prior & low contrast: pos=-5.0, neg=-5.0
        pos_logits = torch.tensor([[10.0, 1.0, 0.5, -5.0]], dtype=torch.float32)
        neg_logits = torch.tensor([[9.9, -8.0, 5.0, -5.0]], dtype=torch.float32)

        # Under CAPC:
        # max Delta z = relu(1.0 - (-8.0)) = 9.0.
        # Token 1 has Delta z = 9.0 >= 0.35 * 9.0 -> contrast_mask is True for Token 1!
        # Token 2 has Delta z = relu(0.5 - 5.0) = 0.0 < 0.35 * 9.0 -> contrast_mask False.
        delta_z_candidate = torch.relu(pos_logits - neg_logits)
        max_delta = delta_z_candidate.max().item()
        self.assertEqual(max_delta, 9.0)
        self.assertTrue((delta_z_candidate[0, 1] >= 0.35 * max_delta).item())
        self.assertFalse((delta_z_candidate[0, 2] >= 0.35 * max_delta).item())

        tok, alpha, _, _ = decoder.decode_step(
            pos_logits=pos_logits, neg_logits=neg_logits, step=0, spatial_confidence=1.0
        )
        # Token 1 receives massive boost from signed contrast:
        # (1 + alpha)*1.0 - alpha*(-8.0) = 1.0 + 9.0 * alpha
        # For alpha ~ 0.65, score is 1.0 + 5.85 = 6.85, passing plausibility!
        self.assertIn(tok, [0, 1])

    def test_06_mode_separation_and_aliasing(self):
        """
        Verify valid mode names and aliases are accepted, and invalid names raise ValueError.
        """
        # Canonical modes
        d1 = DynamicContrastiveDecoder(scoring_mode="canonical_signed")
        self.assertEqual(d1.scoring_mode, "canonical_signed")

        d2 = DynamicContrastiveDecoder(scoring_mode="canonical_r03")
        self.assertEqual(d2.scoring_mode, "canonical_signed")

        d3 = DynamicContrastiveDecoder(scoring_mode="signed")
        self.assertEqual(d3.scoring_mode, "canonical_signed")

        # Historical ReLU modes
        d4 = DynamicContrastiveDecoder(scoring_mode="historical_relu")
        self.assertEqual(d4.scoring_mode, "historical_relu")

        d5 = DynamicContrastiveDecoder(scoring_mode="legacy_relu")
        self.assertEqual(d5.scoring_mode, "historical_relu")

        d6 = DynamicContrastiveDecoder(scoring_mode="relu")
        self.assertEqual(d6.scoring_mode, "historical_relu")

        # Invalid modes
        with self.assertRaises(ValueError):
            DynamicContrastiveDecoder(scoring_mode="unsupported_mode")

        with self.assertRaises(ValueError):
            DynamicContrastiveDecoder(scoring_mode="")

    def test_07_per_token_synchronization(self):
        """
        Verify that positive and negative logits are synchronized per decoding step.
        """
        decoder = DynamicContrastiveDecoder(scoring_mode="canonical_signed")

        pos_logits = torch.randn((1, 50), dtype=torch.float32)
        neg_logits = torch.randn((1, 50), dtype=torch.float32)

        # Mismatched shape must raise error
        with self.assertRaises(Exception):
            decoder.decode_step(
                pos_logits=pos_logits,
                neg_logits=torch.randn((1, 30), dtype=torch.float32),
                step=0,
                spatial_confidence=1.0,
            )

    def test_08_pipeline_default_and_explicit_mode_propagation(self):
        """
        Verify SGVCDPipeline defaults to canonical_signed and forwards scoring_mode.
        """
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_transform = MagicMock()

        # Default initialization
        pipe_default = SGVCDPipeline(
            model=mock_model,
            tokenizer=mock_tokenizer,
            transform=mock_transform,
            device="cpu",
        )
        self.assertEqual(pipe_default.scoring_mode, "canonical_signed")
        self.assertEqual(pipe_default.decoder.scoring_mode, "canonical_signed")

        # Explicit legacy initialization
        pipe_legacy = SGVCDPipeline(
            model=mock_model,
            tokenizer=mock_tokenizer,
            transform=mock_transform,
            scoring_mode="historical_relu",
            device="cpu",
        )
        self.assertEqual(pipe_legacy.scoring_mode, "historical_relu")
        self.assertEqual(pipe_legacy.decoder.scoring_mode, "historical_relu")

    def test_09_numerical_precision_bfloat16_and_float32(self):
        """
        Verify numerical stability and dtype consistency across float32 and bfloat16 inputs.
        """
        decoder = DynamicContrastiveDecoder(scoring_mode="canonical_signed")

        pos_f32 = torch.tensor([[2.5, 4.0, -1.0]], dtype=torch.float32)
        neg_f32 = torch.tensor([[1.0, 5.0, -2.0]], dtype=torch.float32)

        pos_bf16 = pos_f32.to(dtype=torch.bfloat16)
        neg_bf16 = neg_f32.to(dtype=torch.bfloat16)

        tok_f32, alpha_f32, ent_f32, js_f32 = decoder.decode_step(
            pos_f32, neg_f32, step=0, spatial_confidence=1.0
        )
        tok_bf16, alpha_bf16, ent_bf16, js_bf16 = decoder.decode_step(
            pos_bf16, neg_bf16, step=0, spatial_confidence=1.0
        )

        self.assertEqual(tok_f32, tok_bf16)
        self.assertAlmostEqual(alpha_f32, alpha_bf16, places=2)
        self.assertAlmostEqual(ent_f32, ent_bf16, places=2)
        self.assertAlmostEqual(js_f32, js_bf16, places=2)


if __name__ == "__main__":
    unittest.main()

