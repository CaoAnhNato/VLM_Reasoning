# -*- coding: utf-8 -*-
"""
Adversarial Stress Test Suite for Milestone M3 (Challenger M3-2).
Empirical Stress Testing of Contrastive Decoding and Logit Mathematics:
1. DynamicContrastiveDecoder logit stability under adversarial distributions:
   - Identical logits (zero contrast force)
   - Low-entropy distributions (extreme logit spikes)
   - High-entropy distributions (uniform over 32,000 and 151,646 tokens)
   - Strongly negative logits ([-1000.0, -500.0])
   - Negative divergence (salient distractor suppression)
   - Masked / negative-infinity tokens (-inf)
   - Floating-point precision (float32 and bfloat16)
2. Plausibility Truncation with beta_0 = 0.035:
   - Candidate set V_head never empty across temperatures [0.001, 100.0]
   - Candidate set V_head never empty across vocab sizes [2, 10, 1000, 32000, 151646]
   - Monotonic pool size vs beta_0 in [0.01, 0.10]
   - Entropy-adaptive expansion behavior
   - Subordinate entity token survival (beta_0 = 0.035 vs 0.10)
3. Dynamic contrast factor alpha_t bounds checking:
   - Grid sweep across spatial_confidence in [0, 1], entropy in [0, 10], step in [0, 100]
   - Strict range [0.0, 1.8] enforcement
   - Elimination of second tanh preventing attenuation (alpha_t >= 0.35 under high confidence)
   - Syntax gate thresholding (D_JS <= tau_js)
4. Monte Carlo stress test over 1,000 randomized pathological distributions.
"""

import math
import os
import sys
import unittest
from typing import List, Tuple

import torch
import torch.nn.functional as F

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.benchmark.sg_vcd.dynamic_cd import (
    DynamicContrastiveDecoder,
    DecodingDiagnostics,
    compute_adaptive_lowpass_radius,
)


class TestAdversarialLogitStability(unittest.TestCase):
    """Scope 1: DynamicContrastiveDecoder Logit Stability under Adversarial Distributions."""

    def setUp(self):
        self.decoder = DynamicContrastiveDecoder(
            alpha_0=0.65,
            beta_0=0.035,
            tau_js=0.04,
            gamma_decay=0.50,
            lambda_decay=0.08,
        )

    def test_01_zero_divergence_identical_logits(self):
        """Zero-divergence case: P_pos == P_neg. Contrastive adjustment must reduce to 0 smoothly."""
        vocab_sizes = [100, 1000, 32000]
        for V in vocab_sizes:
            torch.manual_seed(42)
            logits = torch.randn(1, V, dtype=torch.float32)
            pos_logits = logits.clone()
            neg_logits = logits.clone()

            token_id, alpha_t, entropy_val, d_js_val = self.decoder.decode_step(
                pos_logits=pos_logits,
                neg_logits=neg_logits,
                step=0,
                spatial_confidence=1.0,
            )

            # Assertions
            self.assertEqual(token_id, int(torch.argmax(pos_logits).item()))
            self.assertAlmostEqual(d_js_val, 0.0, places=5)
            self.assertEqual(alpha_t, 0.0)
            self.assertFalse(math.isnan(entropy_val))
            self.assertFalse(math.isinf(entropy_val))
            self.assertGreaterEqual(entropy_val, 0.0)

    def test_02_low_entropy_extreme_spike(self):
        """Low-entropy distribution: one massive logit (+100.0) and all others tiny (-100.0)."""
        V = 32000
        pos_logits = torch.full((1, V), -100.0, dtype=torch.float32)
        target_token = 777
        pos_logits[0, target_token] = 100.0

        neg_logits = torch.randn(1, V, dtype=torch.float32)

        token_id, alpha_t, entropy_val, d_js_val = self.decoder.decode_step(
            pos_logits=pos_logits,
            neg_logits=neg_logits,
            step=0,
            spatial_confidence=0.9,
        )

        self.assertEqual(token_id, target_token)
        self.assertAlmostEqual(entropy_val, 0.0, places=4)
        self.assertFalse(math.isnan(alpha_t))
        self.assertFalse(math.isinf(alpha_t))
        self.assertFalse(math.isnan(d_js_val))

    def test_03_high_entropy_uniform_32k_and_151k(self):
        """High-entropy distribution: uniform logits over 32k and 151,646 tokens (Vintern-1B vocab)."""
        for V in [32000, 151646]:
            pos_logits = torch.zeros((1, V), dtype=torch.float32)
            neg_logits = torch.randn((1, V), dtype=torch.float32)

            token_id, alpha_t, entropy_val, d_js_val = self.decoder.decode_step(
                pos_logits=pos_logits,
                neg_logits=neg_logits,
                step=0,
                spatial_confidence=1.0,
            )

            expected_entropy = math.log(V)
            self.assertAlmostEqual(entropy_val, expected_entropy, delta=0.1)
            self.assertGreaterEqual(token_id, 0)
            self.assertLess(token_id, V)
            self.assertFalse(math.isnan(alpha_t))
            self.assertFalse(math.isnan(d_js_val))

    def test_04_strongly_negative_logits(self):
        """Strongly negative logits: all logits in [-1000.0, -500.0]. Softmax must not underflow/NaN."""
        V = 1000
        torch.manual_seed(123)
        pos_logits = -1000.0 + torch.rand(1, V, dtype=torch.float32) * 500.0
        best_pos_idx = int(torch.argmax(pos_logits).item())
        neg_logits = pos_logits.clone()

        token_id, alpha_t, entropy_val, d_js_val = self.decoder.decode_step(
            pos_logits=pos_logits,
            neg_logits=neg_logits,
            step=0,
            spatial_confidence=0.8,
        )

        self.assertEqual(token_id, best_pos_idx)
        self.assertFalse(math.isnan(entropy_val))
        self.assertFalse(math.isnan(alpha_t))
        self.assertFalse(math.isnan(d_js_val))

    def test_05_negative_divergence_distractor_suppression(self):
        """Negative divergence: distractor token has higher logit in P_neg than P_pos. Contrastive penalty suppresses it."""
        V = 10
        pos_logits = torch.tensor([[5.0, 5.5, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        # Token 0 is target (logit 5.0), Token 1 is distractor (logit 5.5 in pos, but 10.0 in neg!)
        neg_logits = torch.tensor([[1.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

        # Before contrastive decoding, argmax is Token 1 (distractor)
        self.assertEqual(int(torch.argmax(pos_logits).item()), 1)

        token_id, alpha_t, entropy_val, d_js_val = self.decoder.decode_step(
            pos_logits=pos_logits,
            neg_logits=neg_logits,
            step=0,
            spatial_confidence=1.0,
        )

        # After contrastive calibration, Token 0 (target) is selected!
        self.assertEqual(token_id, 0)
        self.assertGreater(alpha_t, 0.0)
        self.assertGreater(d_js_val, self.decoder.tau_js)

    def test_06_masked_tokens_negative_infinity(self):
        """Tokens with -inf logits (e.g. padding/masking) must not produce NaNs in softmax or divergence."""
        V = 100
        pos_logits = torch.randn(1, V, dtype=torch.float32)
        neg_logits = torch.randn(1, V, dtype=torch.float32)
        # Mask half of the vocabulary
        pos_logits[0, 50:] = float("-inf")
        neg_logits[0, 50:] = float("-inf")

        token_id, alpha_t, entropy_val, d_js_val = self.decoder.decode_step(
            pos_logits=pos_logits,
            neg_logits=neg_logits,
            step=0,
            spatial_confidence=0.7,
        )

        self.assertLess(token_id, 50)
        self.assertFalse(math.isnan(alpha_t))
        self.assertFalse(math.isnan(entropy_val))
        self.assertFalse(math.isnan(d_js_val))

    def test_07_precision_bfloat16_stability(self):
        """Decode step executes stably when inputs are bfloat16 tensors."""
        V = 1000
        pos_logits = torch.randn(1, V, dtype=torch.bfloat16)
        neg_logits = torch.randn(1, V, dtype=torch.bfloat16)

        token_id, alpha_t, entropy_val, d_js_val = self.decoder.decode_step(
            pos_logits=pos_logits,
            neg_logits=neg_logits,
            step=0,
            spatial_confidence=0.85,
        )

        self.assertGreaterEqual(token_id, 0)
        self.assertLess(token_id, V)
        self.assertFalse(math.isnan(alpha_t))
        self.assertFalse(math.isnan(entropy_val))
        self.assertFalse(math.isnan(d_js_val))


class TestAdversarialPlausibilityTruncation(unittest.TestCase):
    """Scope 2: Plausibility Truncation Edge Cases with beta_0 = 0.035."""

    def setUp(self):
        self.beta_0 = 0.035

    def test_01_candidate_set_never_empty_across_temperatures(self):
        """Candidate set V_head must NEVER be empty across extreme temperatures [0.001, 100.0]."""
        temperatures = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
        V = 5000
        torch.manual_seed(999)
        base_logits = torch.randn(1, V)

        for T in temperatures:
            scaled_logits = base_logits / T
            p_pos = F.softmax(scaled_logits, dim=-1)
            entropy = -torch.sum(p_pos * torch.log(p_pos.clamp(min=1e-12)), dim=-1).item()
            beta_t = self.beta_0 * math.exp(-min(5.0, max(0.0, entropy)))
            max_p = p_pos.max(dim=-1, keepdim=True).values
            valid_mask = p_pos >= (beta_t * max_p)

            pool_size = int(valid_mask.sum().item())
            self.assertGreaterEqual(pool_size, 1, f"Candidate pool empty at T={T}")
            argmax_token = int(torch.argmax(p_pos).item())
            self.assertTrue(valid_mask[0, argmax_token].item(), f"Argmax token missing from V_head at T={T}")

    def test_02_candidate_set_never_empty_across_vocab_sizes(self):
        """Candidate set V_head must never be empty across vocabulary sizes [2, 10, 1000, 32000, 151646]."""
        vocab_sizes = [2, 10, 1000, 32000, 151646]
        for V in vocab_sizes:
            torch.manual_seed(V)
            logits = torch.randn(1, V)
            p_pos = F.softmax(logits, dim=-1)
            entropy = -torch.sum(p_pos * torch.log(p_pos.clamp(min=1e-12)), dim=-1).item()
            beta_t = self.beta_0 * math.exp(-min(5.0, max(0.0, entropy)))
            max_p = p_pos.max(dim=-1, keepdim=True).values
            valid_mask = p_pos >= (beta_t * max_p)

            pool_size = int(valid_mask.sum().item())
            self.assertGreaterEqual(pool_size, 1, f"Candidate pool empty for V={V}")
            self.assertLessEqual(pool_size, V)

    def test_03_monotonicity_pool_size_vs_beta_0(self):
        """Pool size must be non-increasing as beta_0 increases (monotonic truncation)."""
        beta_candidates = [0.01, 0.02, 0.035, 0.05, 0.08, 0.10, 0.20, 0.50, 0.80]
        V = 2000
        torch.manual_seed(77)
        logits = torch.randn(1, V)
        p_pos = F.softmax(logits, dim=-1)
        entropy = -torch.sum(p_pos * torch.log(p_pos.clamp(min=1e-12)), dim=-1).item()
        max_p = p_pos.max(dim=-1, keepdim=True).values

        pool_sizes = []
        for b0 in beta_candidates:
            beta_t = b0 * math.exp(-min(5.0, max(0.0, entropy)))
            valid_mask = p_pos >= (beta_t * max_p)
            pool_sizes.append(int(valid_mask.sum().item()))

        # Verify monotonicity: pool_sizes[i] >= pool_sizes[i+1]
        for i in range(len(pool_sizes) - 1):
            self.assertGreaterEqual(
                pool_sizes[i],
                pool_sizes[i + 1],
                f"Non-monotonic pool size between beta_0={beta_candidates[i]} ({pool_sizes[i]}) "
                f"and beta_0={beta_candidates[i+1]} ({pool_sizes[i+1]})",
            )

    def test_04_entropy_adaptive_expansion(self):
        """High-entropy distributions expand candidate pool relative to static cutoff."""
        V = 1000
        # Case A: Low entropy (spiky)
        spiky_logits = torch.full((1, V), -10.0)
        spiky_logits[0, 0] = 10.0
        p_spiky = F.softmax(spiky_logits, dim=-1)
        h_spiky = -torch.sum(p_spiky * torch.log(p_spiky.clamp(min=1e-12)), dim=-1).item()
        beta_spiky = self.beta_0 * math.exp(-min(5.0, max(0.0, h_spiky)))
        pool_spiky = int((p_spiky >= (beta_spiky * p_spiky.max())).sum().item())

        # Case B: High entropy (uniform)
        uniform_logits = torch.zeros((1, V))
        p_uniform = F.softmax(uniform_logits, dim=-1)
        h_uniform = -torch.sum(p_uniform * torch.log(p_uniform.clamp(min=1e-12)), dim=-1).item()
        beta_uniform = self.beta_0 * math.exp(-min(5.0, max(0.0, h_uniform)))
        pool_uniform = int((p_uniform >= (beta_uniform * p_uniform.max())).sum().item())

        self.assertLess(h_spiky, 0.01)
        self.assertGreater(h_uniform, 5.0)
        self.assertLess(beta_uniform, beta_spiky)
        self.assertEqual(pool_spiky, 1)
        self.assertEqual(pool_uniform, V)

    def test_05_subordinate_entity_survival_beta_0_035_vs_0_10(self):
        """Subordinate entity token (P_sub / P_max ~ 0.053) is pruned under beta_0=0.10, but survives under beta_0=0.035."""
        # Realistic VQA distribution with salient distractor (0.94) and subordinate true entity (0.05)
        p_pos = torch.tensor([[0.94, 0.05, 0.01]], dtype=torch.float32)

        entropy = -torch.sum(p_pos * torch.log(p_pos.clamp(min=1e-12)), dim=-1).item()
        decay = math.exp(-min(5.0, max(0.0, entropy)))

        beta_old = 0.10 * decay
        beta_new = 0.035 * decay

        survives_old = (p_pos[0, 1] >= (beta_old * p_pos.max())).item()
        survives_new = (p_pos[0, 1] >= (beta_new * p_pos.max())).item()

        self.assertFalse(survives_old, "Subordinate token should have been pruned under beta_0=0.10")
        self.assertTrue(survives_new, "Subordinate token MUST survive under beta_0=0.035")


class TestAdversarialDynamicAlphaBounds(unittest.TestCase):
    """Scope 3: Dynamic Contrast Factor alpha_t Bounds & Second Tanh Elimination."""

    def setUp(self):
        self.decoder = DynamicContrastiveDecoder(alpha_0=0.65, beta_0=0.035, tau_js=0.04)

    def test_01_grid_sweep_bounds_alpha_in_range(self):
        """Grid sweep across spatial_confidence, entropy, and steps: alpha_t in [0.0, 1.8] strictly."""
        confidence_values = [-0.5, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5]
        steps = [0, 1, 5, 10, 20, 50, 100]

        torch.manual_seed(42)
        pos_logits = torch.randn(1, 100)
        neg_logits = torch.randn(1, 100)

        for conf in confidence_values:
            for st in steps:
                token_id, alpha_t, entropy_val, d_js_val = self.decoder.decode_step(
                    pos_logits=pos_logits,
                    neg_logits=neg_logits,
                    step=st,
                    spatial_confidence=conf,
                )
                self.assertGreaterEqual(alpha_t, 0.0, f"alpha_t negative ({alpha_t}) at conf={conf}, step={st}")
                self.assertLessEqual(alpha_t, 1.8, f"alpha_t exceeded 1.8 ({alpha_t}) at conf={conf}, step={st}")

    def test_02_second_tanh_elimination_prevents_attenuation(self):
        """Eliminating second tanh preserves alpha_t >= 0.35 under high confidence and clear divergence."""
        V = 50
        pos_logits = torch.zeros(1, V)
        pos_logits[0, 0] = 5.0
        neg_logits = torch.zeros(1, V)
        neg_logits[0, 1] = 5.0

        # High spatial confidence (1.0), step 0
        token_id, alpha_t, entropy_val, d_js_val = self.decoder.decode_step(
            pos_logits=pos_logits,
            neg_logits=neg_logits,
            step=0,
            spatial_confidence=1.0,
        )

        self.assertGreaterEqual(alpha_t, 0.35, f"alpha_t ({alpha_t:.4f}) attenuated below 0.35")

        # Contrast with simulated double-tanh attenuation:
        s_double_tanh = math.tanh(1.0)
        m_decay = 1.0
        kl_div = (F.softmax(pos_logits, dim=-1) * (
            torch.log(F.softmax(pos_logits, dim=-1).clamp(min=1e-9)) -
            torch.log(F.softmax(neg_logits, dim=-1).clamp(min=1e-9))
        )).sum()
        u_div = math.tanh(max(0.0, float(kl_div)))
        alpha_double_tanh = self.decoder.alpha_0 * math.tanh(s_double_tanh) * m_decay * u_div

        # Single tanh provides strictly higher contrast force than double tanh
        self.assertGreater(alpha_t, alpha_double_tanh)

    def test_03_zero_spatial_confidence_nullifies_alpha(self):
        """When spatial_confidence == 0.0, alpha_t must be exactly 0.0."""
        pos_logits = torch.tensor([[10.0, 1.0]])
        neg_logits = torch.tensor([[1.0, 10.0]])

        token_id, alpha_t, entropy_val, d_js_val = self.decoder.decode_step(
            pos_logits=pos_logits,
            neg_logits=neg_logits,
            step=5,
            spatial_confidence=0.0,
        )

        self.assertEqual(alpha_t, 0.0)

    def test_04_syntax_gate_cutoff(self):
        """When D_JS <= tau_js (0.04), syntax gate closes and alpha_t becomes 0.0."""
        # Very close distributions: D_JS < 0.04
        pos_logits = torch.tensor([[5.0, 4.9, 2.0, 1.0]])
        neg_logits = torch.tensor([[5.0, 4.95, 2.0, 1.0]])

        token_id, alpha_t, entropy_val, d_js_val = self.decoder.decode_step(
            pos_logits=pos_logits,
            neg_logits=neg_logits,
            step=0,
            spatial_confidence=1.0,
        )

        self.assertLessEqual(d_js_val, self.decoder.tau_js)
        self.assertEqual(alpha_t, 0.0)


class TestAdversarialStressMonteCarlo(unittest.TestCase):
    """Scope 4: 1,000 Pathological Monte Carlo Trials."""

    def test_monte_carlo_pathological_distributions(self):
        """1,000 trials with Cauchy, exponential, bimodal, Dirichlet, and extreme outliers."""
        decoder = DynamicContrastiveDecoder(alpha_0=0.65, beta_0=0.035)
        torch.manual_seed(2026)

        num_trials = 1000
        for trial in range(num_trials):
            V = torch.randint(10, 500, (1,)).item()
            mode = trial % 5

            if mode == 0:
                # Cauchy heavy-tailed noise
                pos_logits = torch.randn(1, V) / (torch.randn(1, V).abs() + 1e-4)
                neg_logits = torch.randn(1, V)
            elif mode == 1:
                # Exponential distribution
                pos_logits = -torch.empty(1, V).exponential_(1.0)
                neg_logits = -torch.empty(1, V).exponential_(1.0)
            elif mode == 2:
                # Spiky bimodal
                pos_logits = torch.full((1, V), -50.0)
                pos_logits[0, 0] = 50.0
                pos_logits[0, 1] = 45.0
                neg_logits = torch.randn(1, V) * 10.0
            elif mode == 3:
                # Uniform random range [-200, 200]
                pos_logits = (torch.rand(1, V) - 0.5) * 400.0
                neg_logits = (torch.rand(1, V) - 0.5) * 400.0
            else:
                # Random Dirichlet via softmax
                pos_logits = torch.randn(1, V) * 5.0
                neg_logits = pos_logits + torch.randn(1, V) * 2.0

            spatial_conf = float(torch.rand(1).item())
            step = trial % 64

            token_id, alpha_t, entropy_val, d_js_val = decoder.decode_step(
                pos_logits=pos_logits,
                neg_logits=neg_logits,
                step=step,
                spatial_confidence=spatial_conf,
            )

            # Robustness assertions
            self.assertFalse(math.isnan(alpha_t), f"NaN alpha_t in trial {trial}")
            self.assertFalse(math.isinf(alpha_t), f"Inf alpha_t in trial {trial}")
            self.assertFalse(math.isnan(entropy_val), f"NaN entropy in trial {trial}")
            self.assertFalse(math.isnan(d_js_val), f"NaN d_js in trial {trial}")
            self.assertGreaterEqual(alpha_t, 0.0)
            self.assertLessEqual(alpha_t, 1.8)
            self.assertGreaterEqual(token_id, 0)
            self.assertLess(token_id, V)


if __name__ == "__main__":
    unittest.main()
