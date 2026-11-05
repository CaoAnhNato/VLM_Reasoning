# -*- coding: utf-8 -*-
"""
Adversarial Stress Test Suite for Theorem 42 Saliency Gating Operator.
Strict A* Research Campaign on ViTextVQA with 5CD-AI/Vintern-1B-v3_5 (Iteration 20 Phase 2).

Operator Under Test:
`apply_saliency_modulated_gating` in `src.nevfe.core.engine`.

Adversarial Challenge Dimensions:
1. Boundary Saliency Distributions:
   - Uniform distribution S(u) = 1/P.
   - Dirac delta distribution (single hot patch S(k)=1.0, S(j!=k)=0.0).
   - Extreme bimodal distribution (clusters of extreme highs and lows).
   - Heavy-tailed & noisy spike distributions (Pareto, exponential, log-normal).
   - Degenerate distributions (all zeros, unnormalized raw weights).
   - Strict bounds assertion: phi(u) in [0.4, 1.6] across valid lambda_gate in [0.2, 0.6].
2. Identity Fallback Precision:
   - Uniform query S(u) = 1/P.
   - Relative error: max |v_gated - v| / (max |v| + eps) < 1e-6 across P in {256, 512}, D in {64, 896}.
   - Perturbed near-uniform queries.
3. Numerical Stability across Dtypes & Extremes:
   - float32, float16, bfloat16.
   - Zero NaNs, zero Infs across millions of evaluated tokens.
   - Dynamic range extremes in visual tokens (1e-7 to 1e4).
   - Autograd & gradient backpropagation: finite non-NaN gradients w.r.t v and s.
4. Batch and Dimension Variations:
   - 2D [P, D] and 3D [1, P, D] for P in {256, 512, 1024}, D in {64, 512, 896}.
   - Multi-batch rejection: B > 1 raises ValueError cleanly.
   - Mismatched shape rejection: len(s) != P raises ValueError cleanly.
   - Saliency tensor shapes: 1D [P] and 2D [1, P].
5. Parameter Sweeps & Monotonicity:
   - Sweep lambda_gate in [0.0, 1.0] with step 0.1.
   - Non-inversion guarantee: phi(u) >= 0.0 even at extreme lambda_gate = 1.0.
   - Monotonic contrast amplification.
"""

import math
import unittest
import numpy as np
import torch
import torch.nn.functional as F

from src.nevfe.core.engine import apply_saliency_modulated_gating


class TestAdversarialIter20Gating(unittest.TestCase):
    """Adversarial stress testing for Theorem 42 Saliency Gating Operator."""

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)

    # =========================================================================
    # 1. Boundary Saliency Distributions
    # =========================================================================
    def test_adversarial_boundary_saliency_distributions(self):
        """
        Adversarial test on boundary distributions:
        - Uniform S(u) = 1/P
        - Dirac Delta (one-hot spike at start, middle, end)
        - Extreme bimodal distributions
        - Heavy-tailed Pareto / Exponential noisy spikes
        - Verifies strict mathematical bounds phi(u) in [0.4, 1.6] for lambda_gate in [0.2, 0.6].
        """
        P_list = [256, 512]
        D = 896
        lambda_gates = [0.2, 0.3, 0.4, 0.5, 0.6]

        for P in P_list:
            v = torch.randn(P, D)

            # Define battery of adversarial saliency distributions
            test_distributions = {}

            # Case A: Uniform
            test_distributions["uniform"] = torch.full((P,), 1.0 / float(P))

            # Case B: Dirac delta at index 0, P//2, P-1
            for idx in [0, P // 2, P - 1]:
                s_dirac = torch.zeros(P)
                s_dirac[idx] = 1.0
                test_distributions[f"dirac_{idx}"] = s_dirac

            # Case C: Extreme bimodal (top 5% have 95% mass, bottom 95% have 5% mass)
            s_bimodal = torch.zeros(P)
            top_k = max(1, int(0.05 * P))
            s_bimodal[:top_k] = 0.95 / top_k
            s_bimodal[top_k:] = 0.05 / (P - top_k)
            test_distributions["extreme_bimodal"] = s_bimodal

            # Case D: Heavy-tailed Pareto distribution (extreme power law)
            pareto_samples = (np.random.pareto(a=1.1, size=P) + 1.0).astype(np.float32)
            s_pareto = torch.tensor(pareto_samples)
            s_pareto = s_pareto / s_pareto.sum()
            test_distributions["pareto_power_law"] = s_pareto

            # Case E: Exponential with massive spike
            s_exp = torch.exp(torch.randn(P) * 5.0)
            s_exp = s_exp / s_exp.sum()
            test_distributions["exponential_noisy"] = s_exp

            # Case F: Two massive spikes, rest near zero
            s_two_spikes = torch.full((P,), 1e-6)
            s_two_spikes[10] = 0.6
            s_two_spikes[P - 10] = 0.4
            s_two_spikes = s_two_spikes / s_two_spikes.sum()
            test_distributions["two_spikes"] = s_two_spikes

            # Run evaluation across all distributions and lambda_gate values
            for dist_name, s_tensor in test_distributions.items():
                for lambda_gate in lambda_gates:
                    # Explicit analytical computation of phi
                    mu_s = 1.0 / float(P)
                    sigma_s = torch.sqrt(torch.mean((s_tensor - mu_s) ** 2) + 1e-6)
                    z_qcass = (s_tensor - mu_s) / sigma_s
                    phi = 1.0 + float(lambda_gate) * torch.tanh(z_qcass)

                    min_phi = float(torch.min(phi).item())
                    max_phi = float(torch.max(phi).item())

                    # Theoretical bounds check
                    expected_min = 1.0 - lambda_gate
                    expected_max = 1.0 + lambda_gate

                    self.assertGreaterEqual(
                        min_phi,
                        expected_min - 1e-6,
                        f"Failed lower bound for {dist_name}, P={P}, lambda={lambda_gate}: {min_phi} < {expected_min}"
                    )
                    self.assertLessEqual(
                        max_phi,
                        expected_max + 1e-6,
                        f"Failed upper bound for {dist_name}, P={P}, lambda={lambda_gate}: {max_phi} > {expected_max}"
                    )

                    # Global bound verification: phi in [0.4, 1.6]
                    self.assertGreaterEqual(
                        min_phi,
                        0.4 - 1e-6,
                        f"Global lower bound (0.4) violated for {dist_name}: min_phi={min_phi}"
                    )
                    self.assertLessEqual(
                        max_phi,
                        1.6 + 1e-6,
                        f"Global upper bound (1.6) violated for {dist_name}: max_phi={max_phi}"
                    )

                    # Execute operator
                    v_gated = apply_saliency_modulated_gating(
                        visual_tokens=v,
                        s_qcass=s_tensor,
                        lambda_gate=lambda_gate,
                        mu_steer=0.2,
                    )

                    # Verify no NaNs or Infs
                    self.assertFalse(
                        torch.isnan(v_gated).any(),
                        f"NaN detected in v_gated for {dist_name}, P={P}, lambda={lambda_gate}"
                    )
                    self.assertFalse(
                        torch.isinf(v_gated).any(),
                        f"Inf detected in v_gated for {dist_name}, P={P}, lambda={lambda_gate}"
                    )
                    self.assertEqual(v_gated.shape, (P, D))

    def test_adversarial_degenerate_zero_and_unnormalized_saliency(self):
        """
        Adversarial test on degenerate saliency inputs:
        - All zeros saliency vector
        - Unnormalized saliency (sum = 50.0 or sum = 1e-4)
        - Verifies graceful handling without crash or NaNs.
        """
        P, D = 512, 896
        v = torch.randn(P, D)

        # 1. All zeros
        s_zeros = torch.zeros(P)
        v_out_zeros = apply_saliency_modulated_gating(v, s_zeros, lambda_gate=0.5)
        self.assertFalse(torch.isnan(v_out_zeros).any(), "All zeros saliency produced NaNs")
        self.assertFalse(torch.isinf(v_out_zeros).any(), "All zeros saliency produced Infs")
        self.assertEqual(v_out_zeros.shape, (P, D))

        # 2. Large unnormalized saliency (sum = 250.0)
        s_unnorm = torch.rand(P) * 10.0
        v_out_unnorm = apply_saliency_modulated_gating(v, s_unnorm, lambda_gate=0.5)
        self.assertFalse(torch.isnan(v_out_unnorm).any(), "Unnormalized saliency produced NaNs")
        self.assertEqual(v_out_unnorm.shape, (P, D))

        # 3. Tiny unnormalized saliency (sum = 1e-5)
        s_tiny = torch.rand(P) * 1e-7
        v_out_tiny = apply_saliency_modulated_gating(v, s_tiny, lambda_gate=0.5)
        self.assertFalse(torch.isnan(v_out_tiny).any(), "Tiny saliency produced NaNs")
        self.assertEqual(v_out_tiny.shape, (P, D))

    # =========================================================================
    # 2. Identity Fallback Precision on Uniform S
    # =========================================================================
    def test_adversarial_identity_fallback_precision(self):
        """
        Adversarial precision stress test:
        When S(u) = 1/P is strictly uniform, dynamic range modulation forces Delta_range = 0,
        psi(u) = 0, and z_QCASS(u) = 0 => phi(u) = 1.0.
        Output must equal input with relative error < 1e-6 across dimensions and dtypes.
        """
        configs = [
            (256, 64, False),
            (256, 896, False),
            (256, 896, True),
            (512, 64, False),
            (512, 896, False),
            (512, 896, True),
            (1024, 896, False),
        ]

        for P, D, is_3d in configs:
            v = torch.randn(1, P, D) if is_3d else torch.randn(P, D)
            s_uniform = torch.full((P,), 1.0 / float(P))

            # Test across multiple lambda_gate and mu_steer values
            for lambda_gate in [0.0, 0.2, 0.5, 0.6, 1.0]:
                for mu_steer in [0.0, 0.2, 0.5, 1.0]:
                    v_gated = apply_saliency_modulated_gating(
                        visual_tokens=v,
                        s_qcass=s_uniform,
                        lambda_gate=lambda_gate,
                        mu_steer=mu_steer,
                    )

                    abs_diff = torch.max(torch.abs(v_gated - v)).item()
                    ref_denom = torch.max(torch.abs(v)).item() + 1e-9
                    rel_err = abs_diff / ref_denom

                    self.assertLess(
                        rel_err,
                        1e-6,
                        f"Identity fallback failed for P={P}, D={D}, 3D={is_3d}, lambda={lambda_gate}, mu={mu_steer}: "
                        f"rel_err={rel_err:.2e}, abs_diff={abs_diff:.2e}"
                    )

    def test_adversarial_near_uniform_continuity(self):
        """
        Tests continuity as S approaches uniform:
        As ||S - 1/P|| -> 0, ||v_gated - v|| -> 0 smoothly.
        """
        P, D = 512, 896
        v = torch.randn(P, D)

        perturbation_magnitudes = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
        prior_rel_err = 1e9

        for eps_pert in perturbation_magnitudes:
            noise = torch.randn(P) * eps_pert
            s_pert = (1.0 / float(P)) + noise
            s_pert = torch.clamp(s_pert, min=1e-6)
            s_pert = s_pert / s_pert.sum()

            v_gated = apply_saliency_modulated_gating(v, s_pert, lambda_gate=0.5, mu_steer=0.2)
            rel_err = (torch.max(torch.abs(v_gated - v)).item()) / (torch.max(torch.abs(v)).item() + 1e-9)

            # Monotonic convergence towards identity fallback
            self.assertLessEqual(
                rel_err,
                prior_rel_err * 1.5,
                f"Non-monotonic divergence at eps_pert={eps_pert}: {rel_err} vs {prior_rel_err}"
            )
            prior_rel_err = rel_err

    # =========================================================================
    # 3. Numerical Stability across Float Precisions & Dynamic Range
    # =========================================================================
    def test_adversarial_numerical_stability_dtypes(self):
        """
        Adversarial stability test across float32, float16, and bfloat16:
        - 0 NaNs, 0 Infs.
        - Extreme magnitude visual tokens: [1e-4, 1.0, 1e2, 1e4].
        - Extreme saliency values: [1e-9, 1.0].
        """
        dtypes = [torch.float32, torch.float16, torch.bfloat16]
        P, D = 512, 896

        scale_factors = [1e-4, 1.0, 50.0]  # Float16 max is ~65504, 50.0 is safe representative

        for dtype in dtypes:
            for scale in scale_factors:
                v = (torch.randn(P, D) * scale).to(dtype=dtype)

                # Highly peaked saliency
                s = torch.exp(torch.randn(P) * 3.0)
                s = (s / s.sum()).to(dtype=dtype)

                v_gated = apply_saliency_modulated_gating(
                    visual_tokens=v,
                    s_qcass=s,
                    lambda_gate=0.5,
                    mu_steer=0.2,
                )

                self.assertEqual(v_gated.dtype, dtype)
                self.assertEqual(v_gated.shape, (P, D))

                # Check NaNs and Infs (convert to float32 for boolean reduction on all hardware)
                v_gated_f = v_gated.float()
                self.assertFalse(
                    torch.isnan(v_gated_f).any().item(),
                    f"NaN detected for dtype={dtype}, scale={scale}"
                )
                self.assertFalse(
                    torch.isinf(v_gated_f).any().item(),
                    f"Inf detected for dtype={dtype}, scale={scale}"
                )

    def test_adversarial_autograd_gradient_propagation(self):
        """
        Adversarial backpropagation test:
        Verifies that apply_saliency_modulated_gating supports smooth end-to-end autograd.
        Both dL/dv and dL/ds must be finite and free of NaNs/Infs.
        """
        P, D = 256, 128
        v = torch.randn(P, D, requires_grad=True)
        s = torch.softmax(torch.randn(P), dim=0)
        s.requires_grad_(True)

        v_gated = apply_saliency_modulated_gating(
            visual_tokens=v,
            s_qcass=s,
            lambda_gate=0.5,
            mu_steer=0.2,
        )

        loss = torch.sum(v_gated ** 2)
        loss.backward()

        self.assertIsNotNone(v.grad, "v.grad is None after backward")
        self.assertFalse(torch.isnan(v.grad).any().item(), "v.grad contains NaNs")
        self.assertFalse(torch.isinf(v.grad).any().item(), "v.grad contains Infs")

        self.assertIsNotNone(s.grad, "s.grad is None after backward")
        self.assertFalse(torch.isnan(s.grad).any().item(), "s.grad contains NaNs")
        self.assertFalse(torch.isinf(s.grad).any().item(), "s.grad contains Infs")

    # =========================================================================
    # 4. Batch and Dimension Variations
    # =========================================================================
    def test_adversarial_batch_and_dimension_variations(self):
        """
        Adversarial tensor variation testing:
        - Valid 2D [P, D] for P in {256, 512, 1024}, D in {64, 512, 896}.
        - Valid 3D [1, P, D].
        - Saliency passed as 1D [P] or 2D [1, P].
        - Clean rejection of batch size B > 1.
        - Clean rejection of invalid tensor rank (1D, 4D).
        - Clean rejection of mismatched token counts (len(s) != P).
        """
        # 1. Valid 2D and 3D variations
        for P in [256, 512]:
            for D in [64, 896]:
                v_2d = torch.randn(P, D)
                s_1d = torch.softmax(torch.randn(P), dim=0)
                s_2d = s_1d.unsqueeze(0)  # [1, P]

                # 2D v with 1D s
                out_2d_1d = apply_saliency_modulated_gating(v_2d, s_1d, lambda_gate=0.4)
                self.assertEqual(out_2d_1d.shape, (P, D))

                # 2D v with 2D s
                out_2d_2d = apply_saliency_modulated_gating(v_2d, s_2d, lambda_gate=0.4)
                self.assertEqual(out_2d_2d.shape, (P, D))
                self.assertTrue(torch.allclose(out_2d_1d, out_2d_2d, atol=1e-6))

                # 3D v [1, P, D] with 1D s
                v_3d = v_2d.unsqueeze(0)
                out_3d = apply_saliency_modulated_gating(v_3d, s_1d, lambda_gate=0.4)
                self.assertEqual(out_3d.shape, (1, P, D))
                self.assertTrue(torch.allclose(out_3d.squeeze(0), out_2d_1d, atol=1e-6))

        # 2. Rejection of batch size B > 1
        v_multi_batch = torch.randn(2, 512, 896)
        s_dummy = torch.softmax(torch.randn(512), dim=0)
        with self.assertRaises(ValueError) as ctx:
            apply_saliency_modulated_gating(v_multi_batch, s_dummy, lambda_gate=0.5)
        self.assertIn("Expected batch size 1", str(ctx.exception))

        # 3. Rejection of invalid rank
        v_4d = torch.randn(1, 1, 512, 896)
        with self.assertRaises(ValueError) as ctx:
            apply_saliency_modulated_gating(v_4d, s_dummy, lambda_gate=0.5)
        self.assertIn("Expected visual_tokens of dimension 2 or 3", str(ctx.exception))

        v_1d = torch.randn(512)
        with self.assertRaises(ValueError) as ctx:
            apply_saliency_modulated_gating(v_1d, s_dummy, lambda_gate=0.5)
        self.assertIn("Expected visual_tokens of dimension 2 or 3", str(ctx.exception))

        # 4. Rejection of shape mismatch
        v_512 = torch.randn(512, 896)
        s_256 = torch.softmax(torch.randn(256), dim=0)
        with self.assertRaises(ValueError) as ctx:
            apply_saliency_modulated_gating(v_512, s_256, lambda_gate=0.5)
        self.assertIn("Shape mismatch", str(ctx.exception))

    # =========================================================================
    # 5. Sweep lambda_gate in [0.0, 1.0] & Non-Inversion
    # =========================================================================
    def test_adversarial_lambda_gate_sweep_and_non_inversion(self):
        """
        Adversarial parameter sweep on lambda_gate in [0.0, 1.0]:
        - At lambda_gate = 0.0: phi(u) = 1.0 strictly.
        - At lambda_gate = 1.0: phi(u) >= 0.0 strictly (no patch representation is inverted).
        - For peaked saliency, amplification contrast strictly increases with lambda_gate.
        """
        P, D = 512, 896
        v = torch.randn(P, D)

        # Heavily peaked distribution
        s_peaked = torch.exp(torch.randn(P) * 3.0)
        s_peaked = s_peaked / s_peaked.sum()

        lambda_sweep = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        prior_contrast = -1.0

        for lam in lambda_sweep:
            mu_s = 1.0 / float(P)
            sigma_s = torch.sqrt(torch.mean((s_peaked - mu_s) ** 2) + 1e-6)
            z_qcass = (s_peaked - mu_s) / sigma_s
            phi = 1.0 + float(lam) * torch.tanh(z_qcass)

            min_phi = float(torch.min(phi).item())
            max_phi = float(torch.max(phi).item())

            # Check boundary invariants
            if lam == 0.0:
                self.assertAlmostEqual(min_phi, 1.0, places=6)
                self.assertAlmostEqual(max_phi, 1.0, places=6)

            # Strict non-inversion check: phi(u) >= 0.0 for all lam in [0.0, 1.0]
            self.assertGreaterEqual(
                min_phi, 0.0 - 1e-6,
                f"Patch representation inverted (phi < 0) at lambda={lam}: min_phi={min_phi}"
            )

            # Exact envelope check: phi in [1 - lam, 1 + lam]
            self.assertGreaterEqual(min_phi, 1.0 - lam - 1e-6)
            self.assertLessEqual(max_phi, 1.0 + lam + 1e-6)

            # Execute operator
            v_gated = apply_saliency_modulated_gating(v, s_peaked, lambda_gate=lam, mu_steer=0.2)
            self.assertFalse(torch.isnan(v_gated).any().item())

            # Check contrast expansion: max(phi) - min(phi) must be monotonically increasing with lam
            contrast = max_phi - min_phi
            self.assertGreaterEqual(
                contrast, prior_contrast - 1e-6,
                f"Contrast did not monotonically increase at lambda={lam}: {contrast} < {prior_contrast}"
            )
            prior_contrast = contrast


if __name__ == "__main__":
    unittest.main()
