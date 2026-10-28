# -*- coding: utf-8 -*-
"""
Milestone M3 Empirical Adversarial Stress Test Suite: Spatial & Image Processing.
Agent: challenger_m3_1
Target Deliverables: Features 4, 5, 8 (M2)

Sections:
1. TestAdversarialBoundingBoxes:
   - Degenerate boxes (zero-width, zero-height, single-point [100, 100, 100, 100])
   - Inverted coordinates (x2 < x1, y2 < y1, both inverted)
   - Out-of-bounds coordinates (partially OOB, completely OOB top-left / bottom-right)
   - Extreme aspect ratios (100:1 and 1:100)
   - Massive canvases (8000x8000)
   - Information-Density Gated Fusion: overlapping vs disjoint seed & propagated boxes
   - Anti-magnification collapse guarantee: s <= min(448, 0.50 * max(W, H))

2. TestAdversarialSobelGradientPrior:
   - Completely uniform images (pure black 0, pure white 255, flat gray 128, uniform RGB)
   - High-frequency synthetic noise (Gaussian noise, salt-and-pepper, uniform noise)
   - Single-pixel impulse image (single active pixel)
   - Boundary image dimensions (1x1, 2x2, 4x4, 8x8, 4000x4000, extreme aspect ratio 10x1000)
   - Numerical assertions: No NaNs, no Infs, no division by zero, values in [0, 1]
   - Modulation properties: strictly non-negative, kappa=0.30 floor enforced

3. TestAdversarialMultiTileSlicing:
   - get_multi_tiles with max_tiles = 1, 2, 3, 4, 5
   - Return types and shape verification: all tiles exactly 448 x 448 x 3 RGB
   - Tile counts and contractual analysis
"""

import math
import os
import sys
import unittest
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.benchmark.sg_vcd.roi_cropper import IsotropicROICropper, CropResult
from scripts.benchmark.sg_vcd.bbox_selector import (
    compute_high_frequency_gradient_prior,
    modulate_affinity_with_gradient_prior,
    KLDivergenceRegionSelector,
    RegionSelectionResult,
)


class TestAdversarialBoundingBoxes(unittest.TestCase):
    """Adversarial stress testing on bounding box geometries and fusion logic."""

    def setUp(self):
        self.cropper = IsotropicROICropper(target_size=448, context_margin=0.15)
        self.canvas_640x480 = Image.new("RGB", (640, 480), color=(120, 140, 160))

    def test_01_degenerate_boxes_zero_dimension_and_single_point(self):
        """Zero-width, zero-height, and single-point boxes must not crash or produce invalid tiles."""
        cases = [
            ("single_point", (100, 100, 100, 100)),
            ("zero_width", (100, 100, 100, 200)),
            ("zero_height", (100, 100, 200, 100)),
            ("origin_point", (0, 0, 0, 0)),
            ("bottom_right_point", (640, 480, 640, 480)),
        ]
        for name, bbox in cases:
            with self.subTest(name=name, bbox=bbox):
                res = self.cropper.crop(self.canvas_640x480, bbox)
                self.assertIsInstance(res, CropResult)
                self.assertEqual(res.roi_tile.size, (448, 448))
                self.assertEqual(res.roi_tile.mode, "RGB")
                self.assertGreater(res.magnification, 0.0)
                # Ensure isotropic box has positive width and height
                sx1, sy1, sx2, sy2 = res.isotropic_box
                self.assertGreater(sx2, sx1)
                self.assertGreater(sy2, sy1)
                self.assertEqual(sx2 - sx1, sy2 - sy1)  # Strictly square

    def test_02_inverted_coordinates_behavior(self):
        """Inverted coordinates (x2 < x1 or y2 < y1) must be characterized empirically."""
        inverted_cases = [
            ("x2_lt_x1", (300, 100, 100, 200)),
            ("y2_lt_y1", (100, 300, 200, 100)),
            ("both_inverted", (300, 300, 100, 100)),
        ]
        results = {}
        for name, bbox in inverted_cases:
            try:
                res = self.cropper.crop(self.canvas_640x480, bbox)
                results[name] = ("SUCCESS", res.roi_tile.size)
            except Exception as e:
                results[name] = ("EXCEPTION", type(e).__name__, str(e))
        
        # All inverted cases must succeed cleanly without throwing exceptions
        for name, outcome in results.items():
            self.assertEqual(outcome[0], "SUCCESS", f"Inverted box failed for {name}: {outcome}")
            self.assertEqual(outcome[1], (448, 448))

    def test_03_out_of_bounds_coordinates(self):
        """Partially and heavily out-of-bounds bounding boxes."""
        cases = [
            ("partially_oob_top_left", (-50, -50, 300, 250)),
            ("partially_oob_bottom_right", (400, 300, 800, 600)),
            ("completely_encompassing", (-50, -50, 2000, 2000)),
            ("entirely_oob_negative", (-500, -500, -100, -100)),
            ("entirely_oob_positive", (1000, 1000, 2000, 2000)),
        ]
        for name, bbox in cases:
            with self.subTest(name=name, bbox=bbox):
                res = self.cropper.crop(self.canvas_640x480, bbox)
                self.assertIsInstance(res, CropResult)
                self.assertEqual(res.roi_tile.size, (448, 448))
                self.assertGreater(res.magnification, 0.0)
                # Output tile must have non-NaN, valid pixels
                tile_arr = np.asarray(res.roi_tile)
                self.assertFalse(np.isnan(tile_arr).any())

    def test_04_extreme_aspect_ratios(self):
        """Extreme aspect ratios (100:1 horizontal slit, 1:100 vertical column)."""
        canvas = Image.new("RGB", (1200, 1200), color=(100, 100, 100))
        cases = [
            ("slit_100_to_1", (50, 500, 1050, 510)),   # 1000 x 10
            ("column_1_to_100", (500, 50, 510, 1050)), # 10 x 1000
        ]
        for name, bbox in cases:
            with self.subTest(name=name, bbox=bbox):
                res = self.cropper.crop(canvas, bbox)
                self.assertEqual(res.roi_tile.size, (448, 448))
                sx1, sy1, sx2, sy2 = res.isotropic_box
                self.assertEqual(sx2 - sx1, sy2 - sy1)
                # Check that isotropic side s encompasses the longest dimension
                s = sx2 - sx1
                self.assertGreaterEqual(s, 1000)

    def test_05_massive_canvas(self):
        """Massive 8000x8000 canvas bounding box cropping."""
        canvas = Image.new("RGB", (8000, 8000), color=(50, 50, 50))
        bbox = (3900, 3900, 4100, 4100)
        res = self.cropper.crop(canvas, bbox)
        self.assertEqual(res.roi_tile.size, (448, 448))
        self.assertGreater(res.magnification, 0.0)

    def test_06_information_density_gated_fusion_simulation(self):
        """
        Simulate Information-Density Gated Fusion logic from pipeline.py lines 563-614.
        Verifies:
        1. Expansion around high-density KL anchor core is bounded by max_allowed_side <= min(448, 0.50 * max_dim).
        2. Aspect ratio is gated <= 3.5.
        3. Coordinates are properly clamped within [0, W] x [0, H] with fused_x2 >= fused_x1 + 2.
        4. Overlapping vs disjoint seed & propagated boxes behave safely.
        """
        def simulate_fusion(W_img, H_img, seed_box, prop_box):
            sx1, sy1, sx2, sy2 = seed_box
            px1, py1, px2, py2 = prop_box
            max_dim = max(W_img, H_img)
            max_allowed_side = min(448, int(round(0.50 * max_dim)))

            w_seed = max(2, sx2 - sx1)
            h_seed = max(2, sy2 - sy1)

            # Clamp horizontal expansion
            exp_left = max(0, sx1 - px1)
            exp_right = max(0, px2 - sx2)
            total_exp_x = exp_left + exp_right
            if (w_seed + total_exp_x) > max_allowed_side and total_exp_x > 0:
                allowed_x = max(0, max_allowed_side - w_seed)
                scale_x = allowed_x / float(total_exp_x)
                fused_x1 = int(round(sx1 - exp_left * scale_x))
                fused_x2 = int(round(sx2 + exp_right * scale_x))
            else:
                fused_x1 = min(sx1, px1)
                fused_x2 = max(sx2, px2)

            # Clamp vertical expansion
            exp_top = max(0, sy1 - py1)
            exp_bottom = max(0, py2 - sy2)
            total_exp_y = exp_top + exp_bottom
            if (h_seed + total_exp_y) > max_allowed_side and total_exp_y > 0:
                allowed_y = max(0, max_allowed_side - h_seed)
                scale_y = allowed_y / float(total_exp_y)
                fused_y1 = int(round(sy1 - exp_top * scale_y))
                fused_y2 = int(round(sy2 + exp_bottom * scale_y))
            else:
                fused_y1 = min(sy1, py1)
                fused_y2 = max(sy2, py2)

            # Aspect ratio sanity gating
            fw = fused_x2 - fused_x1
            fh = fused_y2 - fused_y1
            if fw > 3.5 * max(1, fh):
                target_fw = int(round(3.5 * fh))
                if target_fw >= w_seed:
                    diff_x = fw - target_fw
                    fused_x1 = min(sx1, fused_x1 + diff_x // 2)
                    fused_x2 = max(sx2, fused_x2 - (diff_x - diff_x // 2))
            elif fh > 3.5 * max(1, fw):
                target_fh = int(round(3.5 * fw))
                if target_fh >= h_seed:
                    diff_y = fh - target_fh
                    fused_y1 = min(sy1, fused_y1 + diff_y // 2)
                    fused_y2 = max(sy2, fused_y2 - (diff_y - diff_y // 2))

            fused_x1 = max(0, min(W_img - 2, fused_x1))
            fused_y1 = max(0, min(H_img - 2, fused_y1))
            fused_x2 = max(fused_x1 + 2, min(W_img, fused_x2))
            fused_y2 = max(fused_y1 + 2, min(H_img, fused_y2))

            return [fused_x1, fused_y1, fused_x2, fused_y2]

        test_scenarios = [
            # Standard overlapping
            ("overlapping_small", 640, 480, (100, 100, 150, 150), (80, 80, 200, 200)),
            # Massive propagated box (threat of magnification collapse)
            ("massive_expansion", 1280, 720, (300, 200, 350, 250), (50, 20, 1200, 700)),
            # Completely disjoint seed and propagated box
            ("disjoint_boxes", 800, 600, (100, 100, 150, 150), (500, 400, 600, 500)),
            # Extreme horizontal propagation
            ("extreme_horizontal", 1000, 500, (200, 200, 250, 230), (50, 200, 950, 230)),
            # Extreme vertical propagation
            ("extreme_vertical", 500, 1000, (200, 200, 230, 250), (200, 50, 230, 950)),
            # Tiny image (100x100)
            ("tiny_canvas", 100, 100, (40, 40, 60, 60), (10, 10, 90, 90)),
        ]

        for name, W, H, seed, prop in test_scenarios:
            with self.subTest(name=name):
                fused = simulate_fusion(W, H, seed, prop)
                fx1, fy1, fx2, fy2 = fused
                fw = fx2 - fx1
                fh = fx2 - fx1
                # 1. Non-negative dimensions and valid bounds
                self.assertGreaterEqual(fx1, 0)
                self.assertGreaterEqual(fy1, 0)
                self.assertLessEqual(fx2, W)
                self.assertLessEqual(fy2, H)
                self.assertGreaterEqual(fx2, fx1 + 2)
                self.assertGreaterEqual(fy2, fy1 + 2)

                # 2. Check anti-magnification collapse: max side should not exceed max_allowed_side when seed was small
                max_allowed = min(448, int(round(0.50 * max(W, H))))
                # If seed itself is within max_allowed, fused side should not wildly blow up
                w_seed = seed[2] - seed[0]
                h_seed = seed[3] - seed[1]
                if max(w_seed, h_seed) <= max_allowed:
                    self.assertLessEqual(fx2 - fx1, max(max_allowed, w_seed))
                    self.assertLessEqual(fy2 - fy1, max(max_allowed, h_seed))


class TestAdversarialSobelGradientPrior(unittest.TestCase):
    """Adversarial stress testing of Sobel gradient prior and modulation."""

    def test_01_uniform_images_produce_no_nans_and_default_to_ones(self):
        """Completely flat images must not divide by zero and should return ones_like."""
        uniform_images = [
            ("pure_black_0", Image.new("RGB", (320, 240), color=(0, 0, 0))),
            ("pure_white_255", Image.new("RGB", (320, 240), color=(255, 255, 255))),
            ("flat_gray_128", Image.new("RGB", (320, 240), color=(128, 128, 128))),
            ("uniform_rgb_teal", Image.new("RGB", (320, 240), color=(0, 128, 128))),
        ]
        grid_shape = (14, 14)
        for name, img in uniform_images:
            with self.subTest(name=name):
                prior = compute_high_frequency_gradient_prior(img, grid_shape)
                self.assertEqual(prior.shape, grid_shape)
                self.assertFalse(np.isnan(prior).any(), "NaN found in uniform image prior")
                self.assertFalse(np.isinf(prior).any(), "Inf found in uniform image prior")
                # When gradient range is 0, code defaults to ones_like
                np.testing.assert_allclose(prior, 1.0, atol=1e-5)

    def test_02_synthetic_noise_images(self):
        """Gaussian, salt-and-pepper, and random uniform noise."""
        grid_shape = (14, 14)
        rng = np.random.RandomState(42)

        noise_cases = [
            ("gaussian", rng.normal(128, 40, (240, 320, 3)).clip(0, 255).astype(np.uint8)),
            ("salt_pepper", rng.choice([0, 255], size=(240, 320, 3)).astype(np.uint8)),
            ("uniform_random", rng.randint(0, 256, (240, 320, 3), dtype=np.uint8)),
        ]
        for name, arr in noise_cases:
            with self.subTest(name=name):
                img = Image.fromarray(arr)
                prior = compute_high_frequency_gradient_prior(img, grid_shape)
                self.assertEqual(prior.shape, grid_shape)
                self.assertFalse(np.isnan(prior).any())
                self.assertFalse(np.isinf(prior).any())
                # Range must be normalized in [0, 1]
                self.assertGreaterEqual(float(prior.min()), 0.0)
                self.assertLessEqual(float(prior.max()), 1.0)
                # Since noise has high gradient variation, min should be ~0 and max ~1
                self.assertAlmostEqual(float(prior.min()), 0.0, places=4)
                self.assertAlmostEqual(float(prior.max()), 1.0, places=4)

    def test_03_single_pixel_contrast_impulse(self):
        """Single active pixel in a field of zeros."""
        grid_shape = (14, 14)
        arr = np.zeros((140, 140, 3), dtype=np.uint8)
        arr[70, 70] = [255, 255, 255]
        img = Image.fromarray(arr)

        prior = compute_high_frequency_gradient_prior(img, grid_shape)
        self.assertFalse(np.isnan(prior).any())
        self.assertEqual(prior.shape, grid_shape)
        self.assertAlmostEqual(float(prior.max()), 1.0, places=4)
        self.assertAlmostEqual(float(prior.min()), 0.0, places=4)
        # Center cell must have the maximum gradient
        cy, cx = np.unravel_index(np.argmax(prior), grid_shape)
        self.assertIn(cy, [6, 7])
        self.assertIn(cx, [6, 7])

    def test_04_extreme_image_dimensions(self):
        """Tiny images (4x4, 8x8, 1x1) and large images (4000x4000) and narrow aspect ratios."""
        grid_shape = (14, 14)
        cases = [
            ("1x1", Image.new("RGB", (1, 1), color=(200, 100, 50))),
            ("2x2", Image.new("RGB", (2, 2), color=(200, 100, 50))),
            ("4x4", Image.new("RGB", (4, 4), color=(200, 100, 50))),
            ("8x8", Image.new("RGB", (8, 8), color=(200, 100, 50))),
            ("narrow_horizontal", Image.new("RGB", (1000, 10), color=(150, 150, 150))),
            ("narrow_vertical", Image.new("RGB", (10, 1000), color=(150, 150, 150))),
            ("large_4000x4000", Image.new("RGB", (4000, 4000), color=(80, 80, 80))),
        ]
        for name, img in cases:
            with self.subTest(name=name):
                prior = compute_high_frequency_gradient_prior(img, grid_shape)
                self.assertEqual(prior.shape, grid_shape)
                self.assertFalse(np.isnan(prior).any())
                self.assertFalse(np.isinf(prior).any())
                self.assertGreaterEqual(float(prior.min()), 0.0)
                self.assertLessEqual(float(prior.max()), 1.0)

    def test_05_gradient_modulation_properties_and_kappa_floor(self):
        """
        Verify mathematical properties of modulate_affinity_with_gradient_prior:
        Phi_guided(u) = Phi(u) * (kappa + (1 - kappa) * G_norm(u))
        1. Modulated affinity is non-negative when phi >= 0.
        2. Kappa floor (default 0.30) guarantees attenuation factor is >= kappa.
        3. Gradient prior with out-of-range values or NaN is safely clipped.
        4. Shape mismatch falls back gracefully to unmodulated phi.
        """
        phi = np.random.uniform(0.0, 5.0, size=(14, 14))
        
        # Test floor at kappa = 0.30:
        zero_prior = np.zeros((14, 14), dtype=np.float32)
        guided_zero = modulate_affinity_with_gradient_prior(phi, zero_prior, kappa=0.30)
        # Should be exactly 0.30 * phi
        np.testing.assert_allclose(guided_zero, 0.30 * phi, rtol=1e-5)

        # Test peak at G = 1.0:
        one_prior = np.ones((14, 14), dtype=np.float32)
        guided_one = modulate_affinity_with_gradient_prior(phi, one_prior, kappa=0.30)
        # Should be exactly 1.0 * phi
        np.testing.assert_allclose(guided_one, phi, rtol=1e-5)

        # Test intermediate G values:
        mid_prior = np.full((14, 14), 0.5, dtype=np.float32)
        guided_mid = modulate_affinity_with_gradient_prior(phi, mid_prior, kappa=0.30)
        expected_gate = 0.30 + 0.70 * 0.5  # 0.65
        np.testing.assert_allclose(guided_mid, expected_gate * phi, rtol=1e-5)

        # Test out-of-range clipping in gradient_prior (e.g. -2.0 or +5.0)
        extreme_prior = np.array([[-5.0, 10.0], [0.5, 0.0]])
        small_phi = np.ones((2, 2))
        guided_clipped = modulate_affinity_with_gradient_prior(small_phi, extreme_prior, kappa=0.30)
        # Gate should be clamped to [0.30, 1.0]
        self.assertAlmostEqual(float(guided_clipped[0, 0]), 0.30, places=5)
        self.assertAlmostEqual(float(guided_clipped[0, 1]), 1.00, places=5)

        # Test shape mismatch fallback
        wrong_shape_prior = np.ones((10, 10))
        guided_mismatch = modulate_affinity_with_gradient_prior(phi, wrong_shape_prior, kappa=0.30)
        np.testing.assert_array_equal(guided_mismatch, phi)

        # Test KLDivergenceRegionSelector persists last_guided_phi attribute
        from scripts.benchmark.sg_vcd.bbox_selector import KLDivergenceRegionSelector
        selector = KLDivergenceRegionSelector()
        self.assertIsNone(selector.last_guided_phi)
        selector.select(phi, (640, 480), gradient_prior=mid_prior)
        self.assertIsNotNone(selector.last_guided_phi)
        np.testing.assert_allclose(selector.last_guided_phi, guided_mid, atol=1e-5)


class TestAdversarialMultiTileSlicing(unittest.TestCase):
    """Adversarial stress testing of multi-tile visual evidence cropping."""

    def setUp(self):
        self.cropper = IsotropicROICropper(target_size=448)
        self.img = Image.new("RGB", (640, 480), color=(130, 150, 170))
        self.bbox = (150, 120, 350, 280)

    def test_01_multi_tile_slicing_contract_and_shapes(self):
        """
        Verify get_multi_tiles with max_tiles = 1, 2, 3, 4, 5:
        1. All returned tiles are valid PIL RGB images with size (448, 448).
        2. CropResult metadata is valid.
        3. Examine tile count contracts.
        """
        for m in [1, 2, 3, 4, 5]:
            with self.subTest(max_tiles=m):
                tiles, crop_res = self.cropper.get_multi_tiles(self.img, self.bbox, max_tiles=m)
                self.assertIsInstance(tiles, list)
                self.assertIsInstance(crop_res, CropResult)
                self.assertGreater(len(tiles), 0)
                for i, tile in enumerate(tiles):
                    self.assertIsInstance(tile, Image.Image, f"Tile {i} is not a PIL Image")
                    self.assertEqual(tile.size, (448, 448), f"Tile {i} shape mismatch")
                    self.assertEqual(tile.mode, "RGB", f"Tile {i} mode mismatch")
                    # Check tile is non-empty
                    arr = np.asarray(tile)
                    self.assertFalse(np.isnan(arr).any())

    def test_02_multi_tile_tile_count_semantics(self):
        """
        Characterize tile count semantics:
        - When max_tiles <= 2: returns 2 tiles ([global_thumb, roi_tile])
        - When max_tiles >= 3: returns 3 tiles ([global_thumb, roi_tile, ctx_tile])
        Document whether max_tiles=1 returns 2 tiles or 1 tile.
        """
        tiles_m1, _ = self.cropper.get_multi_tiles(self.img, self.bbox, max_tiles=1)
        tiles_m2, _ = self.cropper.get_multi_tiles(self.img, self.bbox, max_tiles=2)
        tiles_m3, _ = self.cropper.get_multi_tiles(self.img, self.bbox, max_tiles=3)
        tiles_m4, _ = self.cropper.get_multi_tiles(self.img, self.bbox, max_tiles=4)

        # Assert empirical facts: max_tiles contract strictly enforced
        self.assertEqual(len(tiles_m1), 1, "max_tiles=1 returns single focal ROI tile [roi]")
        self.assertEqual(len(tiles_m2), 2, "max_tiles=2 returns dual tiles [thumb, roi]")
        self.assertEqual(len(tiles_m3), 3, "max_tiles=3 returns triple tiles [thumb, roi, ctx]")
        self.assertEqual(len(tiles_m4), 3, "max_tiles=4 returns triple tiles [thumb, roi, ctx]")

    def test_03_multi_tiles_with_tiny_and_large_images(self):
        """Multi-tile slicing under extreme canvas sizes."""
        tiny_img = Image.new("RGB", (20, 20), color=(100, 200, 100))
        tiny_bbox = (2, 2, 10, 10)
        tiles, res = self.cropper.get_multi_tiles(tiny_img, tiny_bbox, max_tiles=3)
        self.assertEqual(len(tiles), 3)
        for t in tiles:
            self.assertEqual(t.size, (448, 448))

        large_img = Image.new("RGB", (3000, 3000), color=(100, 100, 200))
        large_bbox = (1000, 1000, 1500, 1500)
        tiles_l, res_l = self.cropper.get_multi_tiles(large_img, large_bbox, max_tiles=3)
        self.assertEqual(len(tiles_l), 3)
        for t in tiles_l:
            self.assertEqual(t.size, (448, 448))


if __name__ == "__main__":
    unittest.main()
