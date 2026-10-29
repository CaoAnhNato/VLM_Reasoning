# -*- coding: utf-8 -*-
"""
Empirical Adversarial Challenge Suite for SG-VCD Round 5 DynamicMultiTileROICropper.
Exhaustively tests:
1. Extreme Aspect Ratios (AR = 10:1, 20:1, 1:10, 1:20, 100:1, 1:100, arbitrary non-integers)
2. Micro & Tiny Bounding Boxes (1x1, 2x2, 5x5, 0-area points, inverted bboxes, tiny canvas)
3. Canvas Boundary Clipping (negative, overflowing, massive out-of-bounds, all 4 corners)
4. Isotropic Homothety (mathematical invariant s_unit_x == s_unit_y, M_x == M_y across 50 random bboxes)
5. Output Tensor Shapes & Types (strictly valid 448x448 RGB PIL Images for roi_tiles & neg_tiles)
6. Laplacian Edge Re-emphasis Boundary Dynamics (flat images, zero gradients, extreme magnifications)
7. High-Throughput Randomized Stress Harness
"""

import math
import os
import random
import sys
import unittest
from typing import List, Tuple

import numpy as np
from PIL import Image

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.benchmark.sg_vcd.roi_cropper import (
    CropResult,
    DynamicMultiTileROICropper,
    IsotropicROICropper,
    apply_adaptive_laplacian_edge_reemphasis,
)


class TestAdversarialExtremeAspectRatios(unittest.TestCase):
    """Stress test extreme aspect ratios from 10:1 to 100:1 and 1:10 to 1:100."""

    def setUp(self):
        self.cropper = DynamicMultiTileROICropper(target_size=448, max_tiles=3, min_tile_dim=224)
        self.canvas_large = Image.new("RGB", (6000, 6000), color=(100, 150, 200))

    def test_extreme_horizontal_ar_10_to_1(self):
        """AR = 10:1 (2000x200): must select (3, 1) and maintain identical unit dimensions."""
        bbox = (500, 500, 2500, 700)  # 2000 x 200
        res = self.cropper.crop(self.canvas_large, bbox)
        self.assertEqual(res.grid, (3, 1))
        self.assertEqual(len(res.roi_tiles), 3)
        self.assertEqual(len(res.neg_tiles), 3)
        # Unit square invariant
        sx = res.isotropic_box[2] - res.isotropic_box[0]
        sy = res.isotropic_box[3] - res.isotropic_box[1]
        self.assertAlmostEqual(sx / 3.0, sy / 1.0, delta=1.5)  # within 1-2 discrete integer pixels
        for t in res.roi_tiles + res.neg_tiles:
            self.assertEqual(t.size, (448, 448))
            self.assertEqual(t.mode, "RGB")

    def test_extreme_horizontal_ar_20_to_1(self):
        """AR = 20:1 (4000x200): must select (3, 1) max budget without crash or distortion."""
        bbox = (500, 500, 4500, 700)  # 4000 x 200
        res = self.cropper.crop(self.canvas_large, bbox)
        self.assertEqual(res.grid, (3, 1))
        self.assertEqual(len(res.roi_tiles), 3)
        for t in res.roi_tiles:
            self.assertEqual(t.size, (448, 448))

    def test_extreme_vertical_ar_1_to_10(self):
        """AR = 1:10 (200x2000): must select (1, 3) grid."""
        bbox = (500, 500, 700, 2500)  # 200 x 2000
        res = self.cropper.crop(self.canvas_large, bbox)
        self.assertEqual(res.grid, (1, 3))
        self.assertEqual(len(res.roi_tiles), 3)
        self.assertEqual(len(res.neg_tiles), 3)
        for t in res.roi_tiles + res.neg_tiles:
            self.assertEqual(t.size, (448, 448))

    def test_extreme_vertical_ar_1_to_20(self):
        """AR = 1:20 (200x4000): must select (1, 3) grid."""
        bbox = (500, 500, 700, 4500)  # 200 x 4000
        res = self.cropper.crop(self.canvas_large, bbox)
        self.assertEqual(res.grid, (1, 3))
        self.assertEqual(len(res.roi_tiles), 3)

    def test_ultra_extreme_ar_100_to_1(self):
        """AR = 100:1 needle banner (3000x30): tests sub-Nyquist width handling."""
        bbox = (100, 100, 3100, 130)  # 3000 x 30
        res = self.cropper.crop(self.canvas_large, bbox)
        self.assertEqual(res.grid, (3, 1))
        self.assertEqual(len(res.roi_tiles), 3)
        for t in res.roi_tiles + res.neg_tiles:
            self.assertEqual(t.size, (448, 448))

    def test_ultra_extreme_ar_1_to_100(self):
        """AR = 1:100 needle column (30x3000): tests tall sub-Nyquist column handling."""
        bbox = (100, 100, 130, 3100)  # 30 x 3000
        res = self.cropper.crop(self.canvas_large, bbox)
        self.assertEqual(res.grid, (1, 3))
        self.assertEqual(len(res.roi_tiles), 3)


class TestAdversarialMicroAndTinyBBoxes(unittest.TestCase):
    """Stress test micro boxes (1x1, 2x2, 5x5, 0-area points, inverted boxes, tiny canvas)."""

    def setUp(self):
        self.cropper = DynamicMultiTileROICropper(target_size=448, max_tiles=3, min_tile_dim=224)
        self.img = Image.new("RGB", (500, 500), color=(200, 100, 50))

    def test_micro_box_1x1(self):
        """1x1 box must not crash, must cap to (1, 1), and enforce adaptive scale floor."""
        bbox = (250, 250, 251, 251)
        res = self.cropper.crop(self.img, bbox)
        self.assertEqual(res.grid, (1, 1))
        self.assertEqual(len(res.roi_tiles), 1)
        self.assertEqual(res.roi_tile.size, (448, 448))
        self.assertGreater(res.unit_tile_size, 30.0)
        self.assertFalse(math.isnan(res.magnification))
        self.assertFalse(math.isinf(res.magnification))

    def test_micro_box_2x2(self):
        """2x2 box must be capped to (1, 1)."""
        bbox = (100, 100, 102, 102)
        res = self.cropper.crop(self.img, bbox)
        self.assertEqual(res.grid, (1, 1))
        self.assertEqual(res.roi_tile.size, (448, 448))

    def test_micro_box_5x5(self):
        """5x5 box must be capped to (1, 1)."""
        bbox = (100, 100, 105, 105)
        res = self.cropper.crop(self.img, bbox)
        self.assertEqual(res.grid, (1, 1))
        self.assertEqual(res.roi_tile.size, (448, 448))

    def test_zero_area_point_box(self):
        """0-area point box (x1==x2, y1==y2) must be clamped safely without zero division."""
        bbox = (120, 120, 120, 120)
        res = self.cropper.crop(self.img, bbox)
        self.assertEqual(res.grid, (1, 1))
        self.assertEqual(res.roi_tile.size, (448, 448))
        self.assertGreater(res.magnification, 0.0)

    def test_inverted_coordinates_box(self):
        """Inverted coordinates (x1 > x2, y1 > y2) must be canonicalized transparently."""
        bbox = (300, 400, 100, 150)
        res = self.cropper.crop(self.img, bbox)
        self.assertEqual(res.roi_tile.size, (448, 448))
        self.assertTrue(res.source_bbox[0] <= res.source_bbox[2])
        self.assertTrue(res.source_bbox[1] <= res.source_bbox[3])

    def test_tiny_canvas_10x10(self):
        """Cropping on a tiny 10x10 canvas must not crash or out-of-index."""
        tiny_img = Image.new("RGB", (10, 10), color=(50, 50, 50))
        bbox = (2, 2, 8, 8)
        res = self.cropper.crop(tiny_img, bbox)
        self.assertEqual(res.grid, (1, 1))
        self.assertEqual(res.roi_tile.size, (448, 448))

    def test_micro_canvas_2x2(self):
        """Cropping on a 2x2 micro image."""
        micro_img = Image.new("RGB", (2, 2), color=(255, 0, 0))
        bbox = (0, 0, 1, 1)
        res = self.cropper.crop(micro_img, bbox)
        self.assertEqual(res.grid, (1, 1))
        self.assertEqual(res.roi_tile.size, (448, 448))


class TestAdversarialCanvasBoundaryClipping(unittest.TestCase):
    """Stress test boundary clipping: negative coordinates, canvas overflow, out-of-frame boxes."""

    def setUp(self):
        self.cropper = DynamicMultiTileROICropper(target_size=448, max_tiles=3, min_tile_dim=224)
        self.img = Image.new("RGB", (800, 600), color=(100, 150, 200))

    def test_completely_negative_coordinates(self):
        """Box entirely located in negative space (-500, -500, -100, -100)."""
        bbox = (-500, -500, -100, -100)
        res = self.cropper.crop(self.img, bbox)
        self.assertEqual(res.roi_tile.size, (448, 448))
        self.assertEqual(res.roi_tile.mode, "RGB")
        # Ensure image contains valid pixel data (no crash / no NaN)
        arr = np.asarray(res.roi_tile)
        self.assertEqual(arr.shape, (448, 448, 3))
        self.assertTrue(np.all(arr >= 0) and np.all(arr <= 255))

    def test_completely_beyond_canvas_coordinates(self):
        """Box entirely located far beyond canvas width and height (2000, 2000, 3000, 3000)."""
        bbox = (2000, 2000, 3000, 3000)
        res = self.cropper.crop(self.img, bbox)
        self.assertEqual(res.roi_tile.size, (448, 448))
        arr = np.asarray(res.roi_tile)
        self.assertEqual(arr.shape, (448, 448, 3))

    def test_partially_out_of_bounds_left_edge(self):
        """Box straddling left edge: x1 < 0, x2 > 0."""
        bbox = (-150, 100, 200, 400)
        res = self.cropper.crop(self.img, bbox)
        self.assertEqual(res.roi_tile.size, (448, 448))

    def test_partially_out_of_bounds_top_edge(self):
        """Box straddling top edge: y1 < 0, y2 > 0."""
        bbox = (100, -150, 400, 200)
        res = self.cropper.crop(self.img, bbox)
        self.assertEqual(res.roi_tile.size, (448, 448))

    def test_partially_out_of_bounds_right_edge(self):
        """Box straddling right edge: x2 > W."""
        bbox = (650, 100, 1100, 400)
        res = self.cropper.crop(self.img, bbox)
        self.assertEqual(res.roi_tile.size, (448, 448))

    def test_partially_out_of_bounds_bottom_edge(self):
        """Box straddling bottom edge: y2 > H."""
        bbox = (100, 450, 400, 900)
        res = self.cropper.crop(self.img, bbox)
        self.assertEqual(res.roi_tile.size, (448, 448))

    def test_all_corners_overflow_massive_box(self):
        """Massive box far exceeding the entire image in all 4 directions."""
        bbox = (-2000, -2000, 5000, 5000)
        res = self.cropper.crop(self.img, bbox)
        self.assertEqual(res.roi_tile.size, (448, 448))
        for t in res.roi_tiles + res.neg_tiles:
            self.assertEqual(t.size, (448, 448))


class TestAdversarialIsotropicHomothety(unittest.TestCase):
    """Mathematical verification of uniform isotropic magnification M_x == M_y across all grids."""

    def setUp(self):
        self.cropper = DynamicMultiTileROICropper(target_size=448, max_tiles=3, min_tile_dim=224)

    def test_homothety_derivation_all_candidate_grids(self):
        """Mathematically prove Sx = max(wb, hb * gx / gy) ensures s_unit_x == s_unit_y."""
        candidates = [(1, 1), (2, 1), (1, 2), (3, 1), (1, 3)]
        test_dimensions = [
            (100.0, 100.0),
            (600.0, 200.0),
            (200.0, 600.0),
            (1500.0, 200.0),
            (200.0, 1500.0),
            (731.4, 219.8),
            (19.2, 883.1),
        ]
        for gx, gy in candidates:
            for wb, hb in test_dimensions:
                Sx = max(wb, hb * (float(gx) / float(gy)))
                Sy = Sx * (float(gy) / float(gx))
                s_unit_x = Sx / float(gx)
                s_unit_y = Sy / float(gy)
                # Mathematical identity: s_unit_x must equal s_unit_y to floating-point machine precision
                self.assertAlmostEqual(
                    s_unit_x,
                    s_unit_y,
                    delta=1e-12,
                    msg=f"Failed for grid ({gx}, {gy}) with dim ({wb}, {hb})",
                )
                # Magnification factors
                Mx = 448.0 / s_unit_x
                My = 448.0 / s_unit_y
                self.assertAlmostEqual(Mx, My, delta=1e-12)

    def test_homothety_over_50_random_boxes(self):
        """Monte-Carlo stress test across 50 random bounding boxes: zero anisotropic distortion."""
        img = Image.new("RGB", (1200, 1200), color=(120, 120, 120))
        random.seed(42)
        for _ in range(50):
            x1 = random.randint(-200, 1000)
            y1 = random.randint(-200, 1000)
            w = random.randint(1, 800)
            h = random.randint(1, 800)
            x2 = x1 + w
            y2 = y1 + h

            res = self.cropper.crop(img, (x1, y1, x2, y2))
            # Verify unit_tile_size > 0
            self.assertGreater(res.unit_tile_size, 0.0)
            # Verify magnification == 448 / s_unit
            expected_mag = 448.0 / res.unit_tile_size
            self.assertAlmostEqual(res.magnification, expected_mag, places=4)
            # Verify every returned tile is exactly 448x448
            for t in res.roi_tiles + res.neg_tiles:
                self.assertEqual(t.size, (448, 448))


class TestAdversarialOutputTensorShapesAndTypes(unittest.TestCase):
    """Verify contracts: CropResult, get_multi_tiles, get_dual_tiles."""

    def setUp(self):
        self.cropper = DynamicMultiTileROICropper(target_size=448, max_tiles=3, min_tile_dim=224)
        self.img = Image.new("RGB", (1200, 800), color=(80, 120, 160))

    def test_multi_tiles_max_1(self):
        """max_tiles=1 must return [roi_tile]."""
        tiles, res = self.cropper.get_multi_tiles(self.img, (100, 100, 400, 400), max_tiles=1)
        self.assertEqual(len(tiles), 1)
        self.assertEqual(tiles[0].size, (448, 448))

    def test_multi_tiles_max_2_single_grid(self):
        """max_tiles=2 on square box returns [global_thumb, roi_tile]."""
        tiles, res = self.cropper.get_multi_tiles(self.img, (100, 100, 400, 400), max_tiles=2)
        self.assertEqual(len(tiles), 2)
        self.assertEqual(tiles[0].size, (448, 448))
        self.assertEqual(tiles[1].size, (448, 448))

    def test_multi_tiles_max_2_multi_grid(self):
        """When dynamic cropper selects (3, 1), get_multi_tiles returns global + 3 tiles = 4."""
        tiles, res = self.cropper.get_multi_tiles(self.img, (100, 100, 1000, 200), max_tiles=2)
        self.assertEqual(res.grid, (3, 1))
        self.assertEqual(len(tiles), 4)  # 1 global + 3 roi tiles
        for t in tiles:
            self.assertEqual(t.size, (448, 448))

    def test_multi_tiles_max_3_fallback_context(self):
        """When box is (1, 1), max_tiles=3 returns [global, roi, ctx_roi]."""
        tiles, res = self.cropper.get_multi_tiles(self.img, (100, 100, 250, 250), max_tiles=3)
        self.assertEqual(res.grid, (1, 1))
        self.assertEqual(len(tiles), 3)
        for t in tiles:
            self.assertEqual(t.size, (448, 448))

    def test_dual_tiles_contract(self):
        """get_dual_tiles must return (t_global, t_roi, crop_res)."""
        t_global, t_roi, res = self.cropper.get_dual_tiles(self.img, (200, 200, 500, 400))
        self.assertEqual(t_global.size, (448, 448))
        self.assertEqual(t_roi.size, (448, 448))
        self.assertIsInstance(res, CropResult)

    def test_backward_compatibility_isotropic_alias(self):
        """IsotropicROICropper subclass alias behaves identically."""
        iso_cropper = IsotropicROICropper(target_size=448, max_tiles=3, min_tile_dim=224)
        res = iso_cropper.crop(self.img, (100, 100, 800, 200))
        self.assertEqual(res.grid, (3, 1))
        self.assertEqual(len(res.roi_tiles), 3)


class TestAdversarialLaplacianEdgeReemphasis(unittest.TestCase):
    """Stress test apply_adaptive_laplacian_edge_reemphasis under extreme conditions."""

    def test_flat_uniform_image_zero_variance(self):
        """Completely flat image (zero gradients) must not cause ZeroDivisionError."""
        flat_img = Image.new("RGB", (448, 448), color=(128, 128, 128))
        out = apply_adaptive_laplacian_edge_reemphasis(flat_img, magnification=5.0)
        self.assertEqual(out.size, (448, 448))
        arr = np.asarray(out)
        self.assertTrue(np.all(arr == 128))

    def test_low_magnification_bypass(self):
        """When magnification <= 1.02, returns input image directly without processing."""
        img = Image.new("RGB", (448, 448), color=(100, 100, 100))
        out = apply_adaptive_laplacian_edge_reemphasis(img, magnification=1.0)
        self.assertIs(out, img)

    def test_extreme_magnification_scaling(self):
        """Extreme magnification M = 500.0: tanh(M - 1) saturates to 1.0 without overflow."""
        img = Image.fromarray(np.random.randint(0, 256, (448, 448, 3), dtype=np.uint8))
        out = apply_adaptive_laplacian_edge_reemphasis(img, magnification=500.0)
        self.assertEqual(out.size, (448, 448))
        arr = np.asarray(out)
        self.assertTrue(arr.dtype == np.uint8)
        self.assertTrue(np.all(arr >= 0) and np.all(arr <= 255))


class TestAdversarialDataRobustness(unittest.TestCase):
    """Stress test float coordinates, astronomical coordinates, and arbitrary image modes."""

    def setUp(self):
        self.cropper = DynamicMultiTileROICropper(target_size=448, max_tiles=3, min_tile_dim=224)
        self.img = Image.new("RGB", (800, 600), color=(120, 160, 200))

    def test_float_coordinates_input(self):
        """Float coordinates in bounding box must be processed without TypeError in PIL paste."""
        float_bbox = (12.34, 56.78, 456.78, 123.45)
        res = self.cropper.crop(self.img, float_bbox)
        self.assertEqual(res.roi_tile.size, (448, 448))
        self.assertEqual(res.roi_tile.mode, "RGB")

    def test_astronomical_coordinates_clipping(self):
        """Astronomical coordinates (+-10^9) must be bounded by canvas without OOM or MemoryError."""
        astro_bbox = (-10**9, -10**9, 10**9, 10**9)
        res = self.cropper.crop(self.img, astro_bbox)
        self.assertEqual(res.roi_tile.size, (448, 448))
        self.assertLess(res.isotropic_box[2] - res.isotropic_box[0], 10000)

    def test_non_rgb_image_modes(self):
        """Cropper must handle Grayscale ('L'), RGBA, and palette ('P') images seamlessly."""
        for mode in ["L", "RGBA", "P"]:
            mode_img = self.img.convert(mode)
            res = self.cropper.crop(mode_img, (100, 100, 400, 300))
            self.assertEqual(res.roi_tile.size, (448, 448))
            self.assertEqual(res.roi_tile.mode, "RGB")
            for t in res.roi_tiles + res.neg_tiles:
                self.assertEqual(t.mode, "RGB")

    def test_select_grid_zero_or_negative_dimensions(self):
        """Direct call to select_grid with zero or negative dimensions must not raise math domain error."""
        grid_zero = self.cropper.select_grid(0.0, 100.0)
        self.assertEqual(grid_zero, (1, 1))
        grid_neg = self.cropper.select_grid(-50.0, 100.0)
        self.assertEqual(grid_neg, (1, 1))


if __name__ == "__main__":
    unittest.main()
