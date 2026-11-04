# -*- coding: utf-8 -*-
"""
Adversarial Stress Test Suite for NE-VFE Implementation.
Author: challenger_r7 (Teamwork Empirical Challenger)

Target Modules:
- src/nevfe/core/engine.py
- src/nevfe/adapter/vintern_adapter.py

Stress Dimensions:
1. Degenerate Saliency & Spatial Variance Floor:
   - Uniform tokens, all-constant embeddings, zero variance.
   - Check for zero-division, NaNs, and evaluate whether Delta* reaches min_span (0.15).
2. Extreme Boundary Centroids:
   - Point-mass saliency concentrated at lattice corners: (0,0), (1,1), (0,1), (1,0).
   - Verify coordinate clamping to [0.0, 1.0] and valid non-inverted bounding boxes.
3. Dual-Tile Patch Modulation Across Tensor Shapes & Dtypes:
   - Shapes: [256, 896], [512, 896], [1, 512, 896], [4, 512, 896].
   - Dtypes: float32, bfloat16, float16.
4. Fisher-Rao Dynamic Coupling Limits:
   - Identical distributions (BC = 1.0 -> lambda = 0.0).
   - Orthogonal/disjoint distributions (BC = 0.0 -> lambda = lambda_0).
5. Free-Energy Support Truncation (FEST) Invariance:
   - Verify unconditional retention of argmax v_max across adversarial logit profiles.
"""

import math
import unittest
import torch
import torch.nn.functional as F

from src.nevfe.core.engine import (
    NEVFEEngine,
    compute_thermodynamic_spatial_window,
    modulate_patch_embeddings,
    compute_bhattacharyya_coefficient,
    compute_dynamic_coupling,
    compute_unnormalized_entropy,
    compute_free_energy_support_mask,
    nevfe_step_logits,
)


class TestAdversarialNEVFE(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1337)

    # --------------------------------------------------------------------------
    # Adversarial Test 1: Degenerate Saliency & Variance Floor
    # --------------------------------------------------------------------------
    def test_01_degenerate_saliency_constant_embeddings(self):
        """
        Adversarial Test 1: Uniform tokens, all-constant embeddings, zero feature variance.
        Verifies numerical stability (no NaN, no zero-division) and examines Delta*.
        """
        P, D = 256, 896

        # Test A: All-zero embeddings
        v_zeros = torch.zeros(P, D, dtype=torch.float32)
        box_zeros = compute_thermodynamic_spatial_window(v_zeros, grid_size=16, k_sigma=2.0, min_span=0.15)
        self.assertFalse(any(math.isnan(c) for c in box_zeros), "NaN detected in spatial window on zero embeddings!")
        self.assertFalse(any(math.isinf(c) for c in box_zeros), "Inf detected in spatial window on zero embeddings!")

        # Test B: All-constant non-zero embeddings
        v_const = torch.full((P, D), 3.14159, dtype=torch.float32)
        box_const = compute_thermodynamic_spatial_window(v_const, grid_size=16, k_sigma=2.0, min_span=0.15)
        self.assertFalse(any(math.isnan(c) for c in box_const), "NaN detected in spatial window on constant embeddings!")
        self.assertFalse(any(math.isinf(c) for c in box_const), "Inf detected in spatial window on constant embeddings!")

        # Coordinate clamping validation
        for box in [box_zeros, box_const]:
            x_min, y_min, x_max, y_max = box
            self.assertGreaterEqual(x_min, 0.0)
            self.assertLessEqual(x_max, 1.0)
            self.assertGreaterEqual(y_min, 0.0)
            self.assertLessEqual(y_max, 1.0)
            self.assertGreater(x_max, x_min)
            self.assertGreater(y_max, y_min)

        # Empirical measurement of Delta* (half-span)
        # Under uniform tokens, s_p has zero variance, triggering the graceful degenerate fallback
        # which sets center to (0.5, 0.5) and half_span to min_span (0.15).
        half_span_x = (box_const[2] - box_const[0]) / 2.0
        half_span_y = (box_const[3] - box_const[1]) / 2.0

        self.assertAlmostEqual(half_span_x, 0.15, places=5)
        self.assertAlmostEqual(half_span_y, 0.15, places=5)

    def test_01b_reachability_of_min_span_floor_and_localized_zoom(self):
        """
        Adversarial Test 1b: Empirical verification that Rectified Contrastive Saliency
        suppresses background patches to 0 weight, allowing radial sigma to drop and
        enabling localized optical zoom with Delta* reaching min_span (0.15).
        """
        P, D = 256, 896

        # Try to construct the most concentrated spatial pattern possible
        # 1. Single patch peak
        v_single = torch.zeros(P, D)
        v_single[136] = 1000.0  # Center patch massive spike
        box_single = compute_thermodynamic_spatial_window(v_single)
        span_single = (box_single[2] - box_single[0]) / 2.0

        # 2. Compact 2x2 cluster
        v_cluster = torch.zeros(P, D)
        for r in range(7, 9):
            for c in range(7, 9):
                v_cluster[r * 16 + c] = 1000.0
        box_cluster = compute_thermodynamic_spatial_window(v_cluster)
        span_cluster = (box_cluster[2] - box_cluster[0]) / 2.0

        # Both reach the minimum span floor (0.15) for true optical zoom (3.33x)
        self.assertAlmostEqual(span_single, 0.15, delta=0.01)
        self.assertAlmostEqual(span_cluster, 0.15, delta=0.01)
        # Ensure it is strictly localized and NOT degenerate full canvas (0.50)
        self.assertLess(span_single, 0.30)
        self.assertLess(span_cluster, 0.30)

    # --------------------------------------------------------------------------
    # Adversarial Test 2: Extreme Boundary Centroids
    # --------------------------------------------------------------------------
    def test_02_extreme_boundary_centroids(self):
        """
        Adversarial Test 2: Point mass saliency at corners (0,0), (1,1), (0,1), (1,0).
        Verify clamping to [0.0, 1.0] produces valid, non-inverted bounding boxes.
        """
        P, D = 256, 896
        corners = {
            "top_left (0,0)": 0,
            "top_right (1,0)": 15,
            "bottom_left (0,1)": 240,
            "bottom_right (1,1)": 255,
        }

        for name, patch_idx in corners.items():
            v = torch.zeros(P, D)
            v[patch_idx] = 500.0  # Dominant point mass

            x_min, y_min, x_max, y_max = compute_thermodynamic_spatial_window(v, grid_size=16)

            # Coordinates must be strictly in [0.0, 1.0]
            self.assertGreaterEqual(x_min, 0.0, f"x_min < 0 for {name}")
            self.assertLessEqual(x_max, 1.0, f"x_max > 1 for {name}")
            self.assertGreaterEqual(y_min, 0.0, f"y_min < 0 for {name}")
            self.assertLessEqual(y_max, 1.0, f"y_max > 1 for {name}")

            # Non-inverted boxes
            self.assertGreater(x_max, x_min, f"Inverted x coordinates for {name}")
            self.assertGreater(y_max, y_min, f"Inverted y coordinates for {name}")

            # Verify that corner patch coordinate is strictly enclosed inside the box
            patch_x = ((patch_idx % 16) + 0.5) / 16.0
            patch_y = ((patch_idx // 16) + 0.5) / 16.0
            self.assertLessEqual(x_min, patch_x + 1e-4, f"Corner x not enclosed for {name}")
            self.assertGreaterEqual(x_max, patch_x - 1e-4, f"Corner x not enclosed for {name}")
            self.assertLessEqual(y_min, patch_y + 1e-4, f"Corner y not enclosed for {name}")
            self.assertGreaterEqual(y_max, patch_y - 1e-4, f"Corner y not enclosed for {name}")

    # --------------------------------------------------------------------------
    # Adversarial Test 3: Dual-Tile Modulation Shapes & Dtypes
    # --------------------------------------------------------------------------
    def test_03_dual_tile_modulation_shapes_and_dtypes(self):
        """
        Adversarial Test 3: Test modulate_patches across shapes:
        [256, 896], [512, 896], [1, 512, 896], [4, 512, 896]
        across dtypes: float32, bfloat16, float16.
        """
        shapes = [
            (256, 896),
            (512, 896),
            (1, 512, 896),
            (4, 512, 896),
        ]
        dtypes = [
            torch.float32,
            torch.bfloat16,
            torch.float16,
        ]

        engine = NEVFEEngine()

        for s in shapes:
            for dt in dtypes:
                t = torch.randn(*s, dtype=dt)

                # Test via direct function
                mod_func = modulate_patch_embeddings(t)
                self.assertEqual(mod_func.shape, s, f"Shape mismatch in modulate_patch_embeddings for {s}, {dt}")
                self.assertEqual(mod_func.dtype, dt, f"Dtype mismatch in modulate_patch_embeddings for {s}, {dt}")
                self.assertFalse(torch.isnan(mod_func.float()).any(), f"NaN in modulate_patch_embeddings for {s}, {dt}")
                self.assertFalse(torch.isinf(mod_func.float()).any(), f"Inf in modulate_patch_embeddings for {s}, {dt}")

                # Test via engine method
                mod_eng = engine.modulate_patches(t)
                self.assertEqual(mod_eng.shape, s, f"Shape mismatch in engine.modulate_patches for {s}, {dt}")
                self.assertEqual(mod_eng.dtype, dt, f"Dtype mismatch in engine.modulate_patches for {s}, {dt}")
                self.assertFalse(torch.isnan(mod_eng.float()).any(), f"NaN in engine.modulate_patches for {s}, {dt}")

        # Invalid shape test (e.g. 1D or 4D)
        with self.assertRaises(ValueError):
            modulate_patch_embeddings(torch.randn(896))
        with self.assertRaises(ValueError):
            modulate_patch_embeddings(torch.randn(2, 2, 256, 896))

    # --------------------------------------------------------------------------
    # Adversarial Test 4: Fisher-Rao Dynamic Coupling Limits
    # --------------------------------------------------------------------------
    def test_04_fisher_rao_dynamic_coupling_limits(self):
        """
        Adversarial Test 4: Fisher-Rao Dynamic Coupling Limits:
        - When p_plus == p_zero (BC = 1.0): lambda_t = 0.0
        - When p_plus perp p_zero (BC = 0.0): lambda_t = lambda_0
        - Monotonic geodesic decay for intermediate BC values.
        """
        lambda_0 = 1.25

        # Limit 1: Total agreement (p_plus == p_zero)
        p1 = torch.tensor([[0.4, 0.3, 0.2, 0.1]])
        bc_full = compute_bhattacharyya_coefficient(p1, p1)
        self.assertAlmostEqual(bc_full.item(), 1.0, places=5)
        lam_full = compute_dynamic_coupling(lambda_0, bc_full)
        self.assertAlmostEqual(lam_full.item(), 0.0, places=5, msg="When BC=1.0, dynamic coupling must be identically 0.0!")

        # Limit 2: Orthogonal / disjoint distributions (support disjoint)
        p_a = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        p_b = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
        bc_zero = compute_bhattacharyya_coefficient(p_a, p_b)
        self.assertLess(bc_zero.item(), 1e-5)
        lam_zero = compute_dynamic_coupling(lambda_0, bc_zero)
        self.assertAlmostEqual(lam_zero.item(), lambda_0, places=4, msg="When BC=0.0, dynamic coupling must equal lambda_0!")

        # Monotonicity check across intermediate BC grid
        bc_grid = torch.linspace(0.0, 1.0, 50).unsqueeze(-1)
        lam_grid = compute_dynamic_coupling(lambda_0, bc_grid)
        for i in range(len(lam_grid) - 1):
            self.assertGreaterEqual(
                lam_grid[i].item(), lam_grid[i + 1].item(),
                "Dynamic coupling must be monotonically non-increasing in BC!"
            )

    # --------------------------------------------------------------------------
    # Adversarial Test 5: FEST Support Truncation Invariance
    # --------------------------------------------------------------------------
    def test_05_fest_support_truncation_invariance(self):
        """
        Adversarial Test 5: Verify that under extreme and adversarial logit distributions,
        the argmax candidate v_max is unconditionally retained in the FEST support set.
        """
        K = 1000

        adversarial_profiles = {
            "dirac_delta": torch.cat([torch.tensor([1e6]), torch.full((K - 1,), -1e6)]),
            "uniform_flat": torch.zeros(K),
            "massive_dynamic_range": torch.linspace(-1e5, 1e5, K),
            "microscopic_margin": torch.cat([torch.tensor([1.000001]), torch.ones(K - 1)]),
            "negative_infinity_tail": torch.cat([torch.randn(10), torch.full((K - 10,), float("-inf"))]),
            "extreme_sparse_bimodal": torch.cat([torch.tensor([50.0, 49.999]), torch.full((K - 2,), -100.0)]),
        }

        for name, logits in adversarial_profiles.items():
            log_q = F.log_softmax(logits, dim=-1)
            q = torch.exp(log_q)
            entropy = compute_unnormalized_entropy(q, log_q)

            # Shannon entropy in nats must be non-negative
            self.assertGreaterEqual(
                entropy.item(), -1e-6,
                f"Thermodynamic entropy became negative for profile {name}: {entropy.item()}"
            )
            self.assertFalse(torch.isnan(entropy).any(), f"Entropy NaN for {name}")

            mask = compute_free_energy_support_mask(log_q.unsqueeze(0), entropy.unsqueeze(0)).squeeze(0)

            # Find argmax index
            v_max = int(torch.argmax(log_q).item())

            # Unconditional retention invariant: v_max MUST be True in support mask
            self.assertTrue(
                mask[v_max].item(),
                f"FEST support truncation violated! v_max (id {v_max}) was pruned in profile {name}!"
            )

            # Test single-step steering operator with this adversarial distribution
            lp = logits.unsqueeze(0)
            lz = torch.zeros_like(lp)
            next_tok, steered_logits, _, ent = nevfe_step_logits(lp, lz, lambda_param=1.0, beta_param=1.0)
            self.assertFalse(torch.isnan(steered_logits).any(), f"NaN in steered logits for {name}")
            self.assertGreater(steered_logits[0, next_tok.item()].item(), -1e8, f"Selected token was pruned in {name}!")


if __name__ == "__main__":
    unittest.main()
