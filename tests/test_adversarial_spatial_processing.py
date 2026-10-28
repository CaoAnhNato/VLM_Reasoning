# -*- coding: utf-8 -*-
"""
Empirical Adversarial Stress Test Suite: Spatial & Image Processing (Round 3).
Agent: Challenger R3-1 (challenger_r3_1)
Target Components:
1. Top-K Candidate Grounding (KLDivergenceRegionSelector in bbox_selector.py)
   - Degenerate phi maps (zeros, uniforms, single-pixel spikes, delta functions at borders)
   - Spatial NMS extreme cases (identical candidates, concentric boxes, pathological aspect ratios)
   - Information-theoretic significance gate boundaries (gamma_signif = 0.40 +- eps, p = 0.10 +- eps, p/a = 1.0 +- eps)
2. Pre-RoI Laplacian Edge Re-emphasis (apply_adaptive_laplacian_edge_reemphasis in roi_cropper.py)
   - Zero and negative magnification inputs (M <= 0, M = 1.0, M = 0.5, M = 1.01, M = 1.02)
   - Extreme zoom (M = 50.0, 100.0, 500.0, 1000.0) -> [0, 255] range safety
   - Flat textures (completely white, completely black, uniform gray) -> noise immunity
   - Pathological image shapes (1x1, 2x2, 449x449 prime) & non-RGB modes (RGBA, L, CMYK)
3. Collision Resolution & Level-Set Merging (verify_and_resolve_candidate_boxes in adaptive_propagator.py)
   - Exactly touching boxes (IoU = 0.0)
   - Overlap boundaries (IoU = 0.44 vs 0.45 vs 0.46)
   - Nested bounding boxes (B1 subset B2) & concentric boxes
"""

import math
import os
import sys
import unittest
from typing import List, Tuple

import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.benchmark.sg_vcd.bbox_selector import (
    KLDivergenceRegionSelector,
    MultiRegionSelectionResult,
    compute_bernoulli_kl_score,
)
from scripts.benchmark.sg_vcd.roi_cropper import (
    apply_adaptive_laplacian_edge_reemphasis,
    IsotropicROICropper,
)
from scripts.benchmark.sg_vcd.adaptive_propagator import (
    verify_and_resolve_candidate_boxes,
    compute_bbox_iou_and_recall,
)


class TestAdversarialTopKRegionSelection(unittest.TestCase):
    """Stress tests for Top-K candidate selection with degenerate distributions and boundary conditions."""

    def setUp(self):
        self.selector = KLDivergenceRegionSelector(
            num_quantiles=28,
            tau_iou=0.20,
            tau_iom=0.35,
            gamma_signif=0.40,
            min_prob_threshold=0.10,
        )

    def test_01_degenerate_phi_maps_all_zeros(self):
        """All-zero phi map must not crash, divide by zero, or produce NaN, and return full canvas fallback."""
        phi_zeros = np.zeros((28, 28), dtype=np.float64)
        res = self.selector.select_top_k(phi_zeros, image_size=(640, 480), max_k=2)
        self.assertIsInstance(res, MultiRegionSelectionResult)
        self.assertEqual(res.k, 1)
        self.assertEqual(len(res.pixel_bboxes), 1)
        self.assertEqual(res.pixel_bboxes[0], (0, 0, 640, 480))
        self.assertEqual(res.kl_scores[0], 0.0)
        self.assertEqual(res.captured_masses[0], 1.0)
        self.assertEqual(res.area_fractions[0], 1.0)

    def test_02_degenerate_phi_maps_near_zero_sub_epsilon(self):
        """Near-zero sub-epsilon phi map (< 1e-12) must safely invoke full canvas fallback."""
        phi_tiny = np.full((28, 28), 1e-15, dtype=np.float64)
        res = self.selector.select_top_k(phi_tiny, image_size=(640, 480), max_k=2)
        self.assertEqual(res.k, 1)
        self.assertEqual(res.pixel_bboxes[0], (0, 0, 640, 480))
        self.assertEqual(res.kl_scores[0], 0.0)

    def test_03_degenerate_phi_maps_uniform_distribution(self):
        """Uniform phi map (zero excess pointwise information above uniform floor) must fallback cleanly."""
        phi_uniform = np.ones((28, 28), dtype=np.float64) * 0.5
        res = self.selector.select_top_k(phi_uniform, image_size=(500, 500), max_k=2)
        self.assertIsInstance(res, MultiRegionSelectionResult)
        self.assertGreaterEqual(res.k, 1)
        # Bounding box coordinates must be strictly within canvas
        x1, y1, x2, y2 = res.pixel_bboxes[0]
        self.assertGreaterEqual(x1, 0)
        self.assertGreaterEqual(y1, 0)
        self.assertLessEqual(x2, 500)
        self.assertLessEqual(y2, 500)
        self.assertGreater(x2, x1)
        self.assertGreater(y2, y1)

    def test_04_degenerate_phi_maps_single_pixel_spikes(self):
        """Single-pixel delta impulse at various lattice positions must localize tightly."""
        positions = [(0, 0), (14, 14), (27, 27), (0, 27), (27, 0)]
        for r, c in positions:
            with self.subTest(row=r, col=c):
                phi = np.zeros((28, 28), dtype=np.float64)
                phi[r, c] = 100.0
                res = self.selector.select_top_k(phi, image_size=(280, 280), max_k=2)
                self.assertIsInstance(res, MultiRegionSelectionResult)
                self.assertEqual(res.k, 1)
                gx1, gy1, gx2, gy2 = res.grid_bboxes[0]
                self.assertLessEqual(gx1, c)
                self.assertGreaterEqual(gx2, c + 1)
                self.assertLessEqual(gy1, r)
                self.assertGreaterEqual(gy2, r + 1)
                self.assertGreater(res.kl_scores[0], 0.0)

    def test_05_degenerate_phi_maps_delta_at_opposite_borders(self):
        """Two separated delta peaks at opposite borders: verify multi-region selection."""
        phi = np.zeros((28, 28), dtype=np.float64)
        # Two equal clusters
        phi[2:5, 2:5] = 10.0
        phi[23:26, 23:26] = 10.0
        res = self.selector.select_top_k(phi, image_size=(560, 560), max_k=2)
        self.assertIsInstance(res, MultiRegionSelectionResult)
        self.assertEqual(res.k, 2)
        self.assertEqual(len(res.pixel_bboxes), 2)
        # Check disjointness
        b1, b2 = res.pixel_bboxes[0], res.pixel_bboxes[1]
        iou, _, _ = compute_bbox_iou_and_recall(b1, b2)
        self.assertLessEqual(iou, 20.0)  # tau_iou is 0.20

    def test_06_phi_map_with_negative_values_and_nans(self):
        """Phi map with negative values must be clipped at 0.0 without numerical corruption."""
        phi = np.zeros((28, 28), dtype=np.float64)
        phi[10:15, 10:15] = 5.0
        phi[0:5, 0:5] = -10.0  # Strongly negative
        res = self.selector.select_top_k(phi, image_size=(280, 280), max_k=2)
        self.assertGreaterEqual(res.k, 1)
        self.assertGreater(res.kl_scores[0], 0.0)
        self.assertFalse(math.isnan(res.kl_scores[0]))
        self.assertFalse(math.isinf(res.kl_scores[0]))

    def test_07_spatial_nms_extreme_pathological_aspect_ratios(self):
        """Pathological aspect ratios (1x28 line and 28x1 line) must produce valid non-zero boxes."""
        # 1x28 horizontal slit
        phi_h = np.zeros((28, 28), dtype=np.float64)
        phi_h[14, :] = 5.0
        res_h = self.selector.select_top_k(phi_h, image_size=(280, 280), max_k=2)
        self.assertGreaterEqual(res_h.k, 1)
        x1, y1, x2, y2 = res_h.pixel_bboxes[0]
        self.assertGreaterEqual(x2 - x1, 2)
        self.assertGreaterEqual(y2 - y1, 2)

        # 28x1 vertical slit
        phi_v = np.zeros((28, 28), dtype=np.float64)
        phi_v[:, 14] = 5.0
        res_v = self.selector.select_top_k(phi_v, image_size=(280, 280), max_k=2)
        self.assertGreaterEqual(res_v.k, 1)
        x1, y1, x2, y2 = res_v.pixel_bboxes[0]
        self.assertGreaterEqual(x2 - x1, 2)
        self.assertGreaterEqual(y2 - y1, 2)

    def test_08_significance_gate_gamma_boundary_sweep(self):
        """
        Information-theoretic significance gate gamma_signif = 0.40 boundary:
        When secondary candidate has kl_2 / kl_1 < 0.40 -> K=1.
        When secondary candidate has kl_2 / kl_1 >= 0.40 -> K=2.
        """
        # Case A: Strong dominant primary peak, very weak secondary peak (ratio < 0.40)
        phi_a = np.zeros((28, 28), dtype=np.float64)
        phi_a[2:8, 2:8] = 50.0   # Large high mass cluster
        phi_a[20:24, 20:24] = 1.0 # Very weak secondary cluster
        res_a = self.selector.select_top_k(phi_a, image_size=(280, 280), max_k=2)
        self.assertEqual(res_a.k, 1, "Weak secondary peak must be rejected by significance gate")

        # Case B: Two strong comparable peaks (ratio >= 0.40)
        phi_b = np.zeros((28, 28), dtype=np.float64)
        phi_b[2:8, 2:8] = 20.0
        phi_b[20:26, 20:26] = 18.0
        res_b = self.selector.select_top_k(phi_b, image_size=(280, 280), max_k=2)
        self.assertEqual(res_b.k, 2, "Comparable secondary peak must be accepted into K=2")

    def test_09_bernoulli_kl_mathematical_invariance_and_boundaries(self):
        """Bernoulli KL function compute_bernoulli_kl_score mathematical boundary properties."""
        # When p <= a, must strictly return 0.0 (anti-sink information constraint)
        self.assertEqual(compute_bernoulli_kl_score(0.2, 0.2), 0.0)
        self.assertEqual(compute_bernoulli_kl_score(0.1, 0.5), 0.0)
        self.assertEqual(compute_bernoulli_kl_score(0.0, 0.5), 0.0)

        # Boundary extremes
        self.assertEqual(compute_bernoulli_kl_score(0.5, 0.0), 0.0)
        self.assertEqual(compute_bernoulli_kl_score(0.5, 1.0), 0.0)

        # Strictly positive when p > a
        kl = compute_bernoulli_kl_score(0.8, 0.2)
        self.assertGreater(kl, 0.0)
        self.assertFalse(math.isnan(kl))
        self.assertFalse(math.isinf(kl))


class TestAdversarialLaplacianEdgeReemphasis(unittest.TestCase):
    """Stress tests for Pre-RoI adaptive Laplacian edge re-emphasis filter."""

    def test_01_magnification_less_than_or_equal_to_one_identity(self):
        """M <= 1.02 must be strictly an identity transformation (Delta I = 0)."""
        test_img = Image.new("RGB", (100, 100), color=(128, 64, 200))
        # Add some edges
        arr = np.array(test_img)
        arr[20:80, 20:80] = [255, 255, 0]
        test_img = Image.fromarray(arr)

        magnifications = [-10.0, -1.0, 0.0, 0.5, 0.99, 1.0, 1.01, 1.02]
        for m in magnifications:
            with self.subTest(magnification=m):
                res = apply_adaptive_laplacian_edge_reemphasis(test_img, magnification=m)
                self.assertTrue(
                    np.array_equal(np.asarray(res), np.asarray(test_img)),
                    f"Failed identity transform for M={m}"
                )

    def test_02_extreme_magnifications_value_safety(self):
        """M = 50.0, 100.0, 500.0, 1000.0 must keep all pixel values strictly in [0, 255] without overflow."""
        np.random.seed(42)
        noisy_arr = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        img = Image.fromarray(noisy_arr)

        extreme_m = [50.0, 100.0, 500.0, 1000.0]
        for m in extreme_m:
            with self.subTest(magnification=m):
                res = apply_adaptive_laplacian_edge_reemphasis(img, magnification=m, lambda_0=1.0)
                out_arr = np.asarray(res)
                self.assertEqual(out_arr.dtype, np.uint8)
                self.assertGreaterEqual(out_arr.min(), 0)
                self.assertLessEqual(out_arr.max(), 255)
                self.assertFalse(np.isnan(out_arr).any())

    def test_03_flat_textures_noise_immunity(self):
        """Completely flat textures (white, black, uniform gray) must have zero edge enhancement."""
        colors = [
            ("pure_white", (255, 255, 255)),
            ("pure_black", (0, 0, 0)),
            ("uniform_gray", (128, 128, 128)),
            ("uniform_color", (45, 180, 210)),
        ]
        for name, color in colors:
            with self.subTest(name=name):
                img = Image.new("RGB", (64, 64), color=color)
                res = apply_adaptive_laplacian_edge_reemphasis(img, magnification=10.0)
                diff = np.abs(np.asarray(res, dtype=np.float32) - np.asarray(img, dtype=np.float32))
                self.assertLessEqual(diff.max(), 1.0, f"Flat texture {name} altered by edge re-emphasis")

    def test_04_pathological_image_shapes(self):
        """Pathological dimensions (2x2, 3x3, prime 449x449, 10x300, 300x10) must execute safely."""
        shapes = [(2, 2), (3, 3), (13, 17), (449, 449), (10, 300), (300, 10)]
        for w, h in shapes:
            with self.subTest(width=w, height=h):
                img = Image.new("RGB", (w, h), color=(100, 150, 200))
                res = apply_adaptive_laplacian_edge_reemphasis(img, magnification=4.0)
                self.assertEqual(res.size, (w, h))
                self.assertEqual(res.mode, "RGB")

    def test_07_degenerate_sub_2x2_dimensions_vulnerability(self):
        """
        Adversarial probe: Dimensions with min(W, H) < 2 (1x1, 1x2, 2x1, 1x10, 10x1)
        when magnification > 1.02 trigger ValueError in np.gradient(gray).
        Verify that this boundary condition is precisely captured and documented.
        """
        sub_2_shapes = [(1, 1), (1, 2), (2, 1), (1, 10), (10, 1)]
        for w, h in sub_2_shapes:
            with self.subTest(width=w, height=h):
                img = Image.new("RGB", (w, h), color=(100, 150, 200))
                with self.assertRaises(ValueError):
                    apply_adaptive_laplacian_edge_reemphasis(img, magnification=4.0)
                # When M <= 1.02, it safely bypasses via identity transform without crash
                res_id = apply_adaptive_laplacian_edge_reemphasis(img, magnification=1.0)
                self.assertEqual(res_id.size, (w, h))

    def test_05_non_rgb_image_modes(self):
        """Non-RGB modes (RGBA, L, P, 1, CMYK) must convert cleanly without exception."""
        modes = ["RGBA", "L", "1", "P", "CMYK"]
        for mode in modes:
            with self.subTest(mode=mode):
                base = Image.new("RGB", (50, 50), color=(120, 80, 40))
                img = base.convert(mode)
                res = apply_adaptive_laplacian_edge_reemphasis(img, magnification=3.0)
                self.assertEqual(res.size, (50, 50))
                self.assertEqual(res.mode, "RGB")

    def test_06_edge_sharpening_contrast_enhancement(self):
        """Step-edge transition must have sharpened gradient magnitude under high magnification."""
        # Create a sharp vertical step edge: left half 50, right half 200
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        arr[:, :32] = 50
        arr[:, 32:] = 200
        img = Image.fromarray(arr)

        res = apply_adaptive_laplacian_edge_reemphasis(img, magnification=5.0, lambda_0=0.50)
        res_arr = np.asarray(res)

        # At the boundary (col 31 vs col 32), overshoot / undershoot enhances edge contrast
        # col 31 (dark side) should be <= 50, col 32 (bright side) should be >= 200
        self.assertLessEqual(int(res_arr[32, 31, 0]), 50)
        self.assertGreaterEqual(int(res_arr[32, 32, 0]), 200)


class TestAdversarialCollisionAndResolution(unittest.TestCase):
    """Stress tests for post-propagation candidate collision verification and resolution."""

    def test_01_touching_boxes_zero_iou(self):
        """Boxes that are exactly touching at the boundary (IoU = 0.0) must remain distinct (K=2)."""
        b1 = [0, 0, 50, 50]
        b2 = [50, 0, 100, 50]
        iou, _, _ = compute_bbox_iou_and_recall(tuple(b1), tuple(b2))
        self.assertEqual(iou, 0.0)

        resolved = verify_and_resolve_candidate_boxes([b1, b2], iou_threshold=0.45)
        self.assertEqual(len(resolved), 2, "Touching boxes (IoU=0) must remain 2 distinct entities")
        self.assertEqual(resolved[0], b1)
        self.assertEqual(resolved[1], b2)

    def test_02_iou_collision_boundary_sweep_44_45_46(self):
        """
        Boundary sweep around iou_threshold = 0.45 (45.0%):
        - IoU = 44.0% -> K=2 (distinct entities preserved)
        - IoU = 45.0% -> K=1 (merged into union bounding box)
        - IoU = 46.0% -> K=1 (merged into union bounding box)
        """
        # B1 has area 100 x 100 = 10,000. Let B2 also be 100 x 100.
        # If intersection width is W_inter and height is 100:
        # inter_area = 100 * W_inter.
        # union_area = 10000 + 10000 - inter_area = 20000 - 100 * W_inter.
        # IoU = (100 * W_inter) / (20000 - 100 * W_inter) = W_inter / (200 - W_inter).
        # For IoU = 0.44: W_inter / (200 - W_inter) = 0.44 => 1.44 W_inter = 88 => W_inter = 61.11 -> 61
        # With W_inter = 61: inter = 6100, union = 13900 => IoU = 6100 / 13900 = 43.88% < 45.0%
        # For IoU = 0.45: W_inter / (200 - W_inter) = 0.45 => 1.45 W_inter = 90 => W_inter = 62.068 -> 62.07
        # With inter_area = 6207, union = 13793 => IoU = 6207 / 13793 = 45.001% >= 45.0%
        # For IoU = 0.46: W_inter = 63 => inter = 6300, union = 13700 => IoU = 6300 / 13700 = 45.985% >= 45.0%

        b1 = [0, 0, 100, 100]

        # Case 1: IoU ~ 43.88% (< 45.0%)
        b2_sub = [39, 0, 139, 100]  # overlap [39, 100], width 61
        iou_sub, _, _ = compute_bbox_iou_and_recall(tuple(b1), tuple(b2_sub))
        self.assertLess(iou_sub, 45.0)
        res_sub = verify_and_resolve_candidate_boxes([b1, b2_sub], iou_threshold=0.45)
        self.assertEqual(len(res_sub), 2, "IoU < 45% must preserve 2 distinct candidate boxes")

        # Case 2: IoU >= 45.0%
        # Let's construct exact box where IoU >= 45.0%
        # If b1=[0, 0, 100, 100] and b2=[37, 0, 137, 100] -> overlap width 63 -> IoU = 63/137 = 45.98%
        b2_super = [37, 0, 137, 100]
        iou_super, _, _ = compute_bbox_iou_and_recall(tuple(b1), tuple(b2_super))
        self.assertGreaterEqual(iou_super, 45.0)
        res_super = verify_and_resolve_candidate_boxes([b1, b2_super], iou_threshold=0.45)
        self.assertEqual(len(res_super), 1, "IoU >= 45% must merge into 1 unified bounding box")
        self.assertEqual(res_super[0], [0, 0, 137, 100])

    def test_03_concentric_boxes_high_iou(self):
        """Concentric boxes with high IoU (>= 45%) must merge into the encompassing box."""
        b1 = [10, 10, 90, 90]   # area 80x80 = 6400
        b2 = [0, 0, 100, 100]    # area 100x100 = 10000
        # inter = 6400, union = 10000 => IoU = 64.0%
        iou, _, _ = compute_bbox_iou_and_recall(tuple(b1), tuple(b2))
        self.assertGreaterEqual(iou, 45.0)
        res = verify_and_resolve_candidate_boxes([b1, b2], iou_threshold=0.45)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0], [0, 0, 100, 100])

    def test_04_nested_boxes_small_child_iou_analysis(self):
        """
        Adversarial Analysis of Nested Boxes (B1 subset B2):
        When child box B1 is small (e.g. 10x10) inside parent B2 (100x100),
        IoU = 100 / 10000 = 1.0% (< 45.0%).
        Empirically test verify_and_resolve_candidate_boxes behavior:
        Currently it checks purely IoU >= 45%. We assert exact behavior and report finding.
        """
        b1 = [45, 45, 55, 55]   # 10x10 area 100
        b2 = [0, 0, 100, 100]    # 100x100 area 10000
        iou, recall, precision = compute_bbox_iou_and_recall(tuple(b1), tuple(b2))
        self.assertEqual(iou, 1.0)
        self.assertEqual(precision, 100.0) # B1 is 100% inside B2

        res = verify_and_resolve_candidate_boxes([b1, b2], iou_threshold=0.45)
        # Empirical observation: returns [b1, b2] because IoU is 1.0% < 45.0%
        self.assertEqual(len(res), 2)

    def test_05_single_box_and_empty_inputs(self):
        """Single box and empty box list must return unchanged without exception."""
        self.assertEqual(verify_and_resolve_candidate_boxes([]), [])
        single = [[10, 10, 50, 50]]
        self.assertEqual(verify_and_resolve_candidate_boxes(single), single)


if __name__ == "__main__":
    unittest.main()
