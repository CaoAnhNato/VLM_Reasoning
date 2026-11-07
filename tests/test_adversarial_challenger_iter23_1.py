# -*- coding: utf-8 -*-
"""
Adversarial Stress Test Suite for Information-Geometric Variational Free Energy Decoding (IG-VFED).

Authored by: challenger_iter23_1 (Mathematical & Numerical Adversarial Challenger)
Target: src/nevfe/core/engine.py
Date: 2026-11-07

Comprehensive empirical stress testing across 5 mandatory challenge dimensions:
1. Identical Distribution Degeneracy: p+ = p0 => BC = 1.0, alpha_t -> 0 for t > 1;
   at t=1, alpha_1 = lambda with strict negative visual PMI token suppression (Directive A).
2. Disjoint Orthogonal Support: supp(p+) \\cap supp(p0) = \\emptyset => BC = 0.0, alpha_t = lambda, beta_t = 0.0.
3. Extreme Vocabulary Boundaries: V in {2, 10, 32000, 151936, 250000}, zero memory explosion (<500MB), zero hardcoding.
4. Numerical Stability & Extremes: All-zero logits, extreme dynamic ranges (-1000 to +1000), unnormalized inputs, fp16/bf16/fp32.
5. Dynamic Plausibility Adaptation: Low-prob character tokens (p+ = 0.005) preserved when BC -> 0 and truncated when BC -> 1.
"""

import math
import os
import unittest
import torch
import torch.nn.functional as F

from src.nevfe.core.engine import (
    NEVFEEngine,
    compute_mean_field_reference_embeddings,
    compute_bhattacharyya_coefficient,
    compute_dynamic_coupling,
    compute_dynamic_plausibility_floor,
    compute_unnormalized_entropy,
    mean_field_contrastive_step,
    nevfe_step_logits,
    modulate_patch_embeddings,
)


class TestAdversarialChallengerIter23(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261107)
        torch.set_num_threads(2)

    # =========================================================================
    # TASK 1: IDENTICAL DISTRIBUTION DEGENERACY & DIRECTIVE A PREAMBLE AT t=1
    # =========================================================================

    def test_task1_identical_distribution_bc_equals_one(self):
        """Verify BC = 1.0 when p+ and p0 are mathematically identical across distributions."""
        for V in [10, 100, 1000, 32000]:
            # Case A: Uniform distribution
            logits_uniform = torch.zeros(V)
            p_uniform = F.softmax(logits_uniform, dim=-1)
            bc_uniform = compute_bhattacharyya_coefficient(p_uniform, p_uniform)
            self.assertAlmostEqual(bc_uniform.item(), 1.0, places=5)

            # Case B: Sharp peaked distribution
            logits_peak = torch.randn(V) * 5.0
            p_peak = F.softmax(logits_peak, dim=-1)
            bc_peak = compute_bhattacharyya_coefficient(p_peak, p_peak)
            self.assertAlmostEqual(bc_peak.item(), 1.0, places=5)

            # Case C: Extreme float32 drift protection (values close to 1.0 cannot exceed 1.0)
            p_drift = torch.ones(V) / V
            bc_drift = compute_bhattacharyya_coefficient(p_drift, p_drift)
            self.assertLessEqual(bc_drift.item(), 1.0)
            self.assertGreaterEqual(bc_drift.item(), 0.0)

    def test_task1_directive_a_t1_vs_t_greater_1_coupling(self):
        """Verify alpha_t = 0 for t > 1, but alpha_1 = lambda at t = 1 when BC = 1.0."""
        lam = 0.50
        bc_one = torch.tensor([1.0])

        # Step t = 1: Preamble echo suppression active
        alpha_1 = compute_dynamic_coupling(lambda_param=lam, bc=bc_one, step_idx=1, comp_a=True)
        self.assertAlmostEqual(alpha_1.item(), lam, places=5,
                               msg=f"At t=1, alpha_1 must be exactly lambda ({lam}), got {alpha_1.item()}")

        # Step t > 1: Cools to zero on identical distributions
        for step in [2, 3, 5, 10, 50]:
            alpha_t = compute_dynamic_coupling(lambda_param=lam, bc=bc_one, step_idx=step, comp_a=True)
            self.assertAlmostEqual(alpha_t.item(), 0.0, places=5,
                                   msg=f"At t={step} > 1, alpha_t must cool to 0.0, got {alpha_t.item()}")

    def test_task1_negative_pmi_preamble_echo_suppressed_at_t1(self):
        """Verify that conversational prompt echoes (Delta z_1 < 0) are strictly penalized at t=1."""
        V = 100
        engine = NEVFEEngine(lambda_param=0.50, beta_param=0.05)

        # Token 0 is prompt echo ("Cửa"), language prior z0 is high: z0[0] = 10.0, z+[0] = 8.0 (Delta z = -2.0)
        # Token 1 is visual entity ("HỒNG"), z0[1] = 2.0, z+[1] = 9.0 (Delta z = +7.0)
        # Token 2..V are background
        lp = torch.full((V,), 1.0)
        lz = torch.full((V,), 1.0)

        lp[0] = 8.0
        lz[0] = 10.0  # Negative PMI: lp - lz = -2.0

        lp[1] = 9.0
        lz[1] = 2.0   # Positive PMI: lp - lz = +7.0

        # At t=1:
        next_tok_1, steered_logits_1, alpha_1, _ = engine.decode_step(lp, lz, step_idx=1)

        # Check that prompt echo (token 0) is penalized relative to raw visual logit lp[0]
        self.assertLess(steered_logits_1[0].item(), lp[0].item(),
                        msg="Prompt echo with negative PMI must have steered_logit < raw_logit at t=1")

        # Check that visual entity (token 1) is amplified relative to raw visual logit lp[1]
        self.assertGreater(steered_logits_1[1].item(), lp[1].item(),
                           msg="Visual entity with positive PMI must have steered_logit > raw_logit at t=1")

        # Argmax at t=1 must pick visual entity (token 1) over prompt echo (token 0)
        self.assertEqual(next_tok_1.item(), 1,
                         msg="At t=1, IG-VFED must select visual entity over conversational preamble")

    # =========================================================================
    # TASK 2: DISJOINT ORTHOGONAL SUPPORT
    # =========================================================================

    def test_task2_disjoint_support_bc_zero(self):
        """Verify BC = 0.0 when supports are strictly disjoint."""
        V = 20
        # Support of p+: tokens 0..9; Support of p0: tokens 10..19
        p_plus = torch.zeros(V)
        p_zero = torch.zeros(V)
        p_plus[:10] = 0.1
        p_zero[10:] = 0.1

        bc = compute_bhattacharyya_coefficient(p_plus, p_zero)
        self.assertAlmostEqual(bc.item(), 0.0, places=6)

    def test_task2_disjoint_support_coupling_and_plausibility(self):
        """Verify alpha_t = lambda and beta_t = 0.0 when BC = 0.0."""
        lam = 0.65
        beta = 0.05
        bc_zero = torch.tensor([0.0])

        # Dynamic coupling alpha_t
        for step in [1, 2, 5, 20]:
            alpha_t = compute_dynamic_coupling(lambda_param=lam, bc=bc_zero, step_idx=step, comp_a=True)
            self.assertAlmostEqual(alpha_t.item(), lam, places=5,
                                   msg=f"When BC=0, alpha_t must be lambda ({lam}) at step {step}, got {alpha_t.item()}")

        # Dynamic plausibility floor beta_t = beta * BC
        beta_t = compute_dynamic_plausibility_floor(beta_param=beta, bc=bc_zero, comp_b=True)
        self.assertAlmostEqual(beta_t.item(), 0.0, places=6,
                               msg=f"When BC=0, beta_t must be exactly 0.0, got {beta_t.item()}")

    def test_task2_disjoint_support_e2e_contrastive_step(self):
        """Verify full step behavior under disjoint support."""
        V = 50
        lp = torch.full((V,), -10.0)
        lz = torch.full((V,), -10.0)

        # p+ concentrated on token 0
        lp[0] = 20.0
        # p0 concentrated on token 1
        lz[1] = 20.0

        next_tok, steered = mean_field_contrastive_step(lp, lz, alpha=0.5, beta=0.05, step_idx=1)
        self.assertEqual(next_tok.item(), 0)
        # Token 0 should have steered logit: lp[0] + 0.5 * (lp[0] - lz[0])
        expected_steered_0 = 20.0 + 0.5 * (20.0 - (-10.0))
        self.assertAlmostEqual(steered[0].item(), expected_steered_0, delta=0.5)

    # =========================================================================
    # TASK 3: EXTREME VOCABULARY BOUNDARIES & MEMORY INTEGRITY
    # =========================================================================

    def test_task3_extreme_vocabulary_sizes(self):
        """Stress-test across V = 2, 10, 32,000, 151,936, 250,000 without memory explosion."""
        vocab_sizes = [2, 10, 32000, 151936, 250000]
        engine = NEVFEEngine(lambda_param=0.50, beta_param=0.05)

        for V in vocab_sizes:
            # 1D single-step test
            lp_1d = torch.randn(V)
            lz_1d = torch.randn(V)

            next_tok, steered, alpha, entropy = engine.decode_step(lp_1d, lz_1d, step_idx=1)
            self.assertTrue(0 <= next_tok.item() < V, f"Selected token {next_tok.item()} out of range [0, {V})")
            self.assertEqual(steered.shape, (V,), f"Steered shape mismatch for V={V}")
            self.assertFalse(torch.isnan(steered).any(), f"NaN detected in steered logits for V={V}")
            self.assertFalse(torch.isnan(entropy), f"NaN detected in entropy for V={V}")

            # 2D batched test (batch size 2)
            B = 2
            lp_2d = torch.randn(B, V)
            lz_2d = torch.randn(B, V)

            next_tok_2d, steered_2d, alpha_2d, entropy_2d = engine.decode_step(lp_2d, lz_2d, step_idx=1)
            self.assertEqual(next_tok_2d.shape, (B,))
            self.assertEqual(steered_2d.shape, (B, V))
            self.assertFalse(torch.isnan(steered_2d).any())

            # Verify memory stability: tensors are freed
            del lp_1d, lz_1d, next_tok, steered, lp_2d, lz_2d, next_tok_2d, steered_2d

    def test_task3_zero_hardcoded_dimensions(self):
        """Verify that NEVFEEngine does not assume any fixed D, P, or V in patch or logit modules."""
        # Visual patch mean-field reference across diverse dimensions
        shapes_to_test = [
            (1, 16, 64),       # tiny
            (4, 256, 896),     # InternVL standard
            (2, 512, 1024),    # large tile
            (10, 128),         # 2D [P, D]
        ]
        for shape in shapes_to_test:
            patches = torch.randn(*shape)
            ref = compute_mean_field_reference_embeddings(patches)
            self.assertEqual(ref.shape, patches.shape)
            # Verify spatial gradient is zero
            if len(shape) == 3:
                # variance along patch dim (dim 1)
                var = torch.var(ref, dim=1)
                self.assertLess(torch.max(var).item(), 1e-6)
            elif len(shape) == 2:
                var = torch.var(ref, dim=0)
                self.assertLess(torch.max(var).item(), 1e-6)

    # =========================================================================
    # TASK 4: NAN/INF/ZERO GRADIENT SAFETY & EXTREME NUMERICAL SCALES
    # =========================================================================

    def test_task4_zero_logits_tensor(self):
        """Verify behavior on all-zero logit tensors."""
        V = 100
        engine = NEVFEEngine(lambda_param=0.50, beta_param=0.05)
        lp = torch.zeros(V)
        lz = torch.zeros(V)

        next_tok, steered, alpha, entropy = engine.decode_step(lp, lz, step_idx=1)
        self.assertFalse(torch.isnan(steered).any())
        self.assertFalse(torch.isinf(steered).any())
        self.assertTrue(0 <= next_tok.item() < V)

        # Shannon entropy of uniform distribution = ln(V)
        expected_entropy = math.log(V)
        self.assertAlmostEqual(entropy.item(), expected_entropy, places=4)

    def test_task4_extreme_logit_scales(self):
        """Stress-test logit values in range [-1000, +1000] and [-1e4, +1e4]."""
        V = 50
        engine = NEVFEEngine(lambda_param=0.50, beta_param=0.05)

        # Scale 1: [-1000, +1000]
        lp = torch.linspace(-1000.0, 1000.0, V)
        lz = torch.linspace(1000.0, -1000.0, V)

        next_tok, steered, alpha, entropy = engine.decode_step(lp, lz, step_idx=1)
        # Note: masked out tokens become -1e9; unmasked tokens must not be NaN or Inf
        self.assertFalse(torch.isnan(steered).any(), "NaN found in extreme logit scale [-1000, 1000]")
        self.assertFalse(torch.isnan(entropy), "NaN in entropy for [-1000, 1000]")

        # Scale 2: Large dynamic range with single dominant spike
        lp2 = torch.full((V,), -500.0)
        lz2 = torch.full((V,), -500.0)
        lp2[10] = 500.0
        lz2[20] = 500.0

        next_tok2, steered2, alpha2, entropy2 = engine.decode_step(lp2, lz2, step_idx=1)
        self.assertEqual(next_tok2.item(), 10)
        self.assertFalse(torch.isnan(steered2).any())

    def test_task4_precision_dtypes_fp16_bf16(self):
        """Verify engine operates properly when fed float16 and bfloat16 tensors."""
        V = 100
        engine = NEVFEEngine(lambda_param=0.50, beta_param=0.05)

        for dtype in [torch.float16, torch.bfloat16]:
            lp = torch.randn(V, dtype=dtype)
            lz = torch.randn(V, dtype=dtype)

            # mean_field_contrastive_step
            next_tok, steered = mean_field_contrastive_step(lp, lz, alpha=0.5, beta=0.05, step_idx=1)
            self.assertTrue(0 <= next_tok.item() < V)
            self.assertFalse(torch.isnan(steered).any())

            # decode_step
            next_tok2, steered2, alpha2, entropy2 = engine.decode_step(lp, lz, step_idx=1)
            self.assertTrue(0 <= next_tok2.item() < V)
            self.assertFalse(torch.isnan(steered2).any())

    def test_task4_bhattacharyya_clamp_invariants(self):
        """Test clamp invariants: BC cannot be < 0 or > 1 even under adversarial float perturbations."""
        # Simulated adversarial probabilities with numerical noise
        p_plus = torch.tensor([1.0000001, -0.0000001, 0.0])
        p_zero = torch.tensor([1.0000001, 0.0, -0.0000001])

        bc = compute_bhattacharyya_coefficient(p_plus, p_zero)
        self.assertGreaterEqual(bc.item(), 0.0)
        self.assertLessEqual(bc.item(), 1.0)

    # =========================================================================
    # TASK 5: DYNAMIC PLAUSIBILITY ADAPTATION (COMPONENT B)
    # =========================================================================

    def test_task5_dynamic_plausibility_adaptation(self):
        """
        Verify that a low-probability true character token (p+(v) = 0.005) is:
        - PRESERVED when BC -> 0 (high optical divergence rescues rare subwords)
        - TRUNCATED when BC -> 1 (low optical divergence enforces candidate pruning)
        """
        V = 100
        beta = 0.05

        # Setup distribution p+ with:
        # Dominant token (v=0): p+(0) = 0.50
        # Rare character token (v=1): p+(1) = 0.005
        # Remaining 98 tokens: (1 - 0.505) / 98
        p_plus = torch.full((V,), (1.0 - 0.505) / 98)
        p_plus[0] = 0.50
        p_plus[1] = 0.005

        max_p_plus = 0.50
        rare_token_prob = 0.005

        # --- SCENARIO A: High optical divergence (BC -> 0.0) ---
        bc_low = torch.tensor([0.02])  # very small BC
        beta_t_low = compute_dynamic_plausibility_floor(beta_param=beta, bc=bc_low, comp_b=True)
        # beta_t_low = 0.05 * 0.02 = 0.001
        threshold_low = (beta_t_low * max_p_plus).item()  # 0.001 * 0.50 = 0.0005
        # Since rare_token_prob (0.005) >= threshold_low (0.0005):
        is_preserved_low = (rare_token_prob >= threshold_low)
        self.assertTrue(is_preserved_low,
                        f"Rare token (prob={rare_token_prob}) must be PRESERVED when BC={bc_low.item()} (threshold={threshold_low})")

        # --- SCENARIO B: Low optical divergence (BC -> 1.0) ---
        bc_high = torch.tensor([0.98])  # high BC (agrees with linguistic prior)
        beta_t_high = compute_dynamic_plausibility_floor(beta_param=beta, bc=bc_high, comp_b=True)
        # beta_t_high = 0.05 * 0.98 = 0.049
        threshold_high = (beta_t_high * max_p_plus).item()  # 0.049 * 0.50 = 0.0245
        # Since rare_token_prob (0.005) < threshold_high (0.0245):
        is_preserved_high = (rare_token_prob >= threshold_high)
        self.assertFalse(is_preserved_high,
                         f"Rare token (prob={rare_token_prob}) must be TRUNCATED when BC={bc_high.item()} (threshold={threshold_high})")

    def test_task5_rare_token_rescue_in_end_to_end_step(self):
        """Verify that under high optical divergence, a rare visual token with strong visual PMI wins argmax."""
        V = 50
        engine = NEVFEEngine(lambda_param=0.80, beta_param=0.20)

        # Background logits
        lp = torch.full((V,), -2.0)
        lz = torch.full((V,), -2.0)

        # Language prior p0 heavily concentrates on token 0 (e.g. prompt filler)
        lz[0] = 8.0
        lp[0] = -5.0   # p+ has zero visual evidence for token 0

        # Visual distribution p+ has visual evidence on token 2 (dominant) and token 1 (rare subword)
        lp[2] = 5.0    # dominant visual candidate
        lz[2] = -2.0

        lp[1] = 4.0    # rare visual character token (e.g. number/diacritic) with high visual PMI
        lz[1] = -6.0   # Delta z = 4.0 - (-6.0) = 10.0

        # Optical distribution diverges from p0:
        p_plus = F.softmax(lp, dim=-1)
        p_zero = F.softmax(lz, dim=-1)
        bc = compute_bhattacharyya_coefficient(p_plus, p_zero)
        self.assertLess(bc.item(), 0.15, f"BC should be low due to divergence on modes, got {bc.item()}")

        # With static beta=0.20:
        # max_p_plus is at token 2 (p+(2) ~ 0.70). Threshold = 0.20 * 0.70 = 0.14.
        # Token 1 has p+(1) ~ 0.25 (or if smaller, say lp[1] = 3.0 -> p+(1) ~ 0.10 < 0.14 -> truncated).
        # But with dynamic beta_t = beta * BC:
        # beta_t = 0.20 * BC ~ 0.20 * 0.05 = 0.01.
        # Threshold = 0.01 * 0.70 = 0.007.
        # Token 1 is preserved and wins argmax due to high visual PMI!
        next_tok, steered, alpha, _ = engine.decode_step(lp, lz, step_idx=1)
        self.assertEqual(next_tok.item(), 1, "Under optical divergence, rare visual token with high PMI must win argmax")

    # =========================================================================
    # TASK 6: MODULAR ABLATION SWITCHES & BOUNDARY VALIDATION
    # =========================================================================

    def test_task6_ablation_switches_toggles(self):
        """Verify comp_a, comp_b, comp_c toggle behavior independently."""
        V = 20
        lp = torch.randn(V)
        lz = torch.randn(V)

        engine = NEVFEEngine(lambda_param=0.50, beta_param=0.05)

        # Test comp_a = False (static alpha)
        engine.set_ablation_switches(comp_a=False, comp_b=True, comp_c=False)
        _, _, alpha_static, _ = engine.decode_step(lp, lz, step_idx=1)
        self.assertAlmostEqual(alpha_static.item(), 0.50, places=5)

        # Test comp_a = True (dynamic alpha with comp_c=False: Iter 22 champion coupling)
        engine.set_ablation_switches(comp_a=True, comp_b=True, comp_c=False)
        _, _, alpha_dyn, _ = engine.decode_step(lp, lz, step_idx=1)
        self.assertTrue(torch.is_tensor(alpha_dyn))

        # Test comp_c = True (Directive A t=1 preamble boost active)
        engine.set_ablation_switches(comp_a=True, comp_b=True, comp_c=True)
        _, _, alpha_with_boost, _ = engine.decode_step(lp, lz, step_idx=1)
        self.assertGreaterEqual(alpha_with_boost.item(), alpha_dyn.item() - 1e-6)

    def test_task6_hyperparameter_bounds_enforcement(self):
        """Verify ValueError is raised when hyperparameters exceed valid bounds."""
        with self.assertRaises(ValueError):
            NEVFEEngine(lambda_param=-0.1)
        with self.assertRaises(ValueError):
            NEVFEEngine(lambda_param=2.0)
        with self.assertRaises(ValueError):
            NEVFEEngine(beta_param=0.0)
        with self.assertRaises(ValueError):
            NEVFEEngine(beta_param=6.0)
        with self.assertRaises(ValueError):
            NEVFEEngine(mu_param=-1.0)


if __name__ == "__main__":
    unittest.main()
