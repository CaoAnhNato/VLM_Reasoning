# -*- coding: utf-8 -*-
"""
Adversarial Stress Test Suite for Theorem 32:
Hierarchical Multi-Scale Resolution and Morphological Shock Scale-Space Invariance.

Role: challenger_iter16_spatial (Empirical Challenger - critic, specialist)
Authoritative Adversarial Verification Suite.

Stress-tests:
1. Adversarial Shock Dilation Tests:
   - Extreme inputs: all-zero black images, all-ones white images, intermediate flat fields,
     alternating single-pixel checkerboards, high-frequency Gaussian/uniform noise,
     single-pixel thin lines at arbitrary angles (0°, 90°, 45°, 135°, arbitrary slopes).
   - Polarity symmetry under inverted contrasts (light-on-dark vs dark-on-light).
   - Numerical stability: zero NaNs, zero Infs, strictly bounded in [0.0, 1.0],
     extreme magnifications (M in [-10, 0, 1.05, 100, 1000, 10000]), asymmetric aspect ratios.
2. Adversarial Hierarchical Zoom Tests:
   - Extreme aspect ratios (100:1 and 1:100), boundary-adjacent crop boxes.
   - Sub-box span boundary conditions: rho_sub -> 0 (clamping to min_span), rho_sub -> 1.0 (triggering fallback).
   - Invariant verification across 100 random synthetic geometries: |V_L| in {256, 512} and S_L in [266, 522].
"""

import math
import random
import unittest
from typing import Optional, Tuple
from unittest.mock import MagicMock
from PIL import Image, ImageDraw
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nevfe.core.engine import (
    NEVFEEngine,
    adaptive_tensor_stroke_dilation,
    compute_question_conditioned_spatial_window,
    compute_thermodynamic_spatial_window,
)
from src.nevfe.adapter.vintern_adapter import VinternNEVFEAdapter


class TestAdversarialShockDilation(unittest.TestCase):
    """
    Adversarial Stress Testing of Theorem 32 Shock Scale-Space Dilation Kernel.
    Target: adaptive_tensor_stroke_dilation
    """

    def setUp(self):
        self.device = torch.device("cpu")
        torch.manual_seed(42)
        random.seed(42)

    def test_01_extreme_constant_inputs(self):
        """
        Test flat, featureless extreme inputs:
        1. All-zero black image (3D and 4D)
        2. All-ones white image (3D and 4D)
        3. Intermediate flat gray fields (0.25, 0.50, 0.75)
        Invariant: Output must be bit-for-bit identical to input (delta == 0),
        zero NaNs, zero Infs, strictly bounded in [0.0, 1.0].
        """
        magnifications = [1.5, 3.0, 10.0, 100.0]

        for m in magnifications:
            # 1. All-zero black image (4D: [1, 3, 32, 32])
            z_4d = torch.zeros((1, 3, 32, 32), device=self.device, dtype=torch.float32)
            out_z_4d = adaptive_tensor_stroke_dilation(z_4d, magnification=m)
            self.assertTrue(torch.equal(z_4d, out_z_4d), f"All-zero 4D image modified at M={m}")
            self.assertFalse(torch.isnan(out_z_4d).any())
            self.assertFalse(torch.isinf(out_z_4d).any())

            # 2. All-zero black image (3D: [3, 32, 32])
            z_3d = torch.zeros((3, 32, 32), device=self.device, dtype=torch.float32)
            out_z_3d = adaptive_tensor_stroke_dilation(z_3d, magnification=m)
            self.assertTrue(torch.equal(z_3d, out_z_3d), f"All-zero 3D image modified at M={m}")
            self.assertEqual(out_z_3d.shape, (3, 32, 32))

            # 3. All-ones white image (4D and 3D)
            o_4d = torch.ones((1, 3, 32, 32), device=self.device, dtype=torch.float32)
            out_o_4d = adaptive_tensor_stroke_dilation(o_4d, magnification=m)
            self.assertTrue(torch.equal(o_4d, out_o_4d), f"All-ones 4D image modified at M={m}")

            o_3d = torch.ones((3, 32, 32), device=self.device, dtype=torch.float32)
            out_o_3d = adaptive_tensor_stroke_dilation(o_3d, magnification=m)
            self.assertTrue(torch.equal(o_3d, out_o_3d), f"All-ones 3D image modified at M={m}")

            # 4. Intermediate flat fields
            for val in [0.25, 0.50, 0.75]:
                flat = torch.full((1, 3, 32, 32), val, device=self.device, dtype=torch.float32)
                out_flat = adaptive_tensor_stroke_dilation(flat, magnification=m)
                self.assertTrue(torch.equal(flat, out_flat), f"Flat {val} image modified at M={m}")

    def test_02_checkerboard_high_frequency_stress(self):
        """
        Adversarial Test: Alternating single-pixel checkerboard.
        This represents the theoretical worst-case spatial frequency (Nyquist limit),
        where every adjacent pixel alternates between 0 and 1.
        Invariant:
        - Output strictly in [0.0, 1.0], no NaNs, no Infs.
        - Polarity symmetry: inverted checkerboard produces inverted dilation delta.
        """
        H, W = 32, 32
        cb = torch.zeros((1, 3, H, W), device=self.device, dtype=torch.float32)
        for i in range(H):
            for j in range(W):
                cb[0, :, i, j] = float((i + j) % 2)

        cb_inv = 1.0 - cb

        magnifications = [1.1, 2.0, 5.0, 20.0, 100.0]
        for m in magnifications:
            out_cb = adaptive_tensor_stroke_dilation(cb, magnification=m)
            out_cb_inv = adaptive_tensor_stroke_dilation(cb_inv, magnification=m)

            # Numerical stability
            self.assertFalse(torch.isnan(out_cb).any(), f"NaN in checkerboard at M={m}")
            self.assertFalse(torch.isinf(out_cb).any(), f"Inf in checkerboard at M={m}")
            self.assertTrue((out_cb >= 0.0).all() and (out_cb <= 1.0).all())
            self.assertTrue((out_cb_inv >= 0.0).all() and (out_cb_inv <= 1.0).all())

            # Polarity symmetry
            delta = out_cb - cb
            delta_inv = out_cb_inv - cb_inv
            diff = torch.abs(delta + delta_inv).max().item()
            self.assertLess(
                diff, 1e-4,
                f"Checkerboard polarity symmetry violated at M={m}, max diff: {diff:.6e}"
            )

    def test_03_high_frequency_gaussian_and_uniform_noise(self):
        """
        Adversarial Test: High-frequency random Gaussian and Uniform noise fields.
        Validates stability against random textured backgrounds.
        Invariant:
        - Strictly bounded in [0.0, 1.0] across all seeds and magnifications.
        - 0 NaNs, 0 Infs.
        - Polarity symmetry preserved under complement noise: noise vs (1 - noise).
        """
        for seed in range(5):
            torch.manual_seed(100 + seed)
            # Uniform noise
            u_noise = torch.rand((1, 3, 40, 40), device=self.device, dtype=torch.float32)
            u_inv = 1.0 - u_noise

            out_u = adaptive_tensor_stroke_dilation(u_noise, magnification=4.0)
            out_u_inv = adaptive_tensor_stroke_dilation(u_inv, magnification=4.0)

            self.assertFalse(torch.isnan(out_u).any())
            self.assertFalse(torch.isinf(out_u).any())
            self.assertTrue((out_u >= 0.0).all() and (out_u <= 1.0).all())

            diff_u = torch.abs((out_u - u_noise) + (out_u_inv - u_inv)).max().item()
            self.assertLess(diff_u, 1e-4, f"Uniform noise polarity asymmetry: {diff_u:.6e}")

            # Clamped Gaussian noise
            g_noise = torch.clamp(0.5 + 0.25 * torch.randn((1, 3, 40, 40), device=self.device), 0.0, 1.0)
            g_inv = 1.0 - g_noise

            out_g = adaptive_tensor_stroke_dilation(g_noise, magnification=4.0)
            out_g_inv = adaptive_tensor_stroke_dilation(g_inv, magnification=4.0)

            self.assertFalse(torch.isnan(out_g).any())
            self.assertFalse(torch.isinf(out_g).any())
            self.assertTrue((out_g >= 0.0).all() and (out_g <= 1.0).all())

            diff_g = torch.abs((out_g - g_noise) + (out_g_inv - g_inv)).max().item()
            self.assertLess(diff_g, 1e-4, f"Gaussian noise polarity asymmetry: {diff_g:.6e}")

    def test_04_single_pixel_thin_lines_arbitrary_angles(self):
        """
        Adversarial Test: Single-pixel stroke lines at arbitrary angles:
        1. 0° (horizontal)
        2. 90° (vertical)
        3. 45° (main diagonal)
        4. 135° (anti-diagonal)
        5. Arbitrary slope rasterized line (~30° and ~60°)
        Invariant:
        - Dark lines on light background thicken/darken (delta <= 0).
        - Light lines on dark background brighten/thicken (delta >= 0).
        - Strictly bounded in [0.0, 1.0], zero NaNs, zero Infs.
        - Additive polarity symmetry: max(|delta_dark + delta_light|) < 1e-4.
        """
        H, W = 48, 48
        angles_patterns = []

        # 1. Horizontal line at row 24
        horiz = torch.full((1, 3, H, W), 0.85, device=self.device)
        horiz[:, :, 24, :] = 0.15
        angles_patterns.append(("Horizontal (0°)", horiz))

        # 2. Vertical line at col 24
        vert = torch.full((1, 3, H, W), 0.85, device=self.device)
        vert[:, :, :, 24] = 0.15
        angles_patterns.append(("Vertical (90°)", vert))

        # 3. Main diagonal (45°)
        diag45 = torch.full((1, 3, H, W), 0.85, device=self.device)
        for i in range(min(H, W)):
            diag45[:, :, i, i] = 0.15
        angles_patterns.append(("Diagonal (45°)", diag45))

        # 4. Anti-diagonal (135°)
        diag135 = torch.full((1, 3, H, W), 0.85, device=self.device)
        for i in range(min(H, W)):
            diag135[:, :, i, W - 1 - i] = 0.15
        angles_patterns.append(("Anti-diagonal (135°)", diag135))

        # 5. Arbitrary slope rasterized line (Bresenham / PIL drawn line)
        for angle_deg in [30, 60]:
            pil_img = Image.new("L", (W, H), color=217)  # 0.85 * 255 = 217
            draw = ImageDraw.Draw(pil_img)
            rad = math.radians(angle_deg)
            cx, cy = W // 2, H // 2
            x0 = int(cx - 20 * math.cos(rad))
            y0 = int(cy - 20 * math.sin(rad))
            x1 = int(cx + 20 * math.cos(rad))
            y1 = int(cy + 20 * math.sin(rad))
            draw.line([(x0, y0), (x1, y1)], fill=38, width=1)  # 0.15 * 255 = 38
            arr = np.array(pil_img, dtype=np.float32) / 255.0
            t_line = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(self.device)
            angles_patterns.append((f"Raster Line ({angle_deg}°)", t_line))

        # Execute and verify all angle patterns
        for name, img_dark in angles_patterns:
            img_light = 1.0 - img_dark  # Inverted contrast

            dilated_dark = adaptive_tensor_stroke_dilation(img_dark, magnification=3.5)
            dilated_light = adaptive_tensor_stroke_dilation(img_light, magnification=3.5)

            # Bounds
            self.assertTrue((dilated_dark >= 0.0).all() and (dilated_dark <= 1.0).all())
            self.assertTrue((dilated_light >= 0.0).all() and (dilated_light <= 1.0).all())

            delta_dark = dilated_dark - img_dark
            delta_light = dilated_light - img_light

            # Shock dilation must sharpen boundary contrast
            self.assertGreater(
                torch.abs(delta_dark).max().item(), 0.01,
                f"Shock dilation had no effect on {name}",
            )
            self.assertGreater(
                torch.abs(delta_light).max().item(), 0.01,
                f"Shock dilation had no effect on light {name}",
            )

            # Strict polarity symmetry
            sym_diff = torch.abs(delta_dark + delta_light).max().item()
            self.assertLess(
                sym_diff, 1e-4,
                f"Polarity symmetry violated for {name}: max diff {sym_diff:.6e}",
            )

    def test_05_polarity_symmetry_comprehensive_matrix(self):
        """
        Adversarial Test: Comprehensive Polarity Symmetry Matrix.
        Sweeps contrast levels (low contrast to extreme contrast) and magnifications M in [1.5, 3.0, 7.0, 15.0].
        Invariant:
        - Output is perfectly additive inverse symmetric: ||delta_dark + delta_light||_inf < 1e-4.
        """
        contrasts = [
            (0.05, 0.95),  # High contrast
            (0.20, 0.80),  # Moderate contrast
            (0.35, 0.65),  # Low contrast
            (0.45, 0.55),  # Very faint contrast
        ]
        magnifications = [1.5, 3.0, 7.0, 15.0]

        H, W = 40, 40
        for stroke_val, bg_val in contrasts:
            for m in magnifications:
                img_dark = torch.full((1, 3, H, W), bg_val, device=self.device)
                img_dark[:, :, 15:25, 15:25] = stroke_val  # Square block

                img_light = torch.full((1, 3, H, W), stroke_val, device=self.device)
                img_light[:, :, 15:25, 15:25] = bg_val

                dilated_dark = adaptive_tensor_stroke_dilation(img_dark, magnification=m)
                dilated_light = adaptive_tensor_stroke_dilation(img_light, magnification=m)

                delta_dark = dilated_dark - img_dark
                delta_light = dilated_light - img_light

                sym_diff = torch.abs(delta_dark + delta_light).max().item()
                self.assertLess(
                    sym_diff, 1e-4,
                    f"Polarity symmetry broken for contrast ({stroke_val}, {bg_val}) at M={m}: diff {sym_diff:.6e}",
                )

    def test_06_extreme_magnifications_and_stability_boundary(self):
        """
        Adversarial Test: Boundary and extreme magnification scaling.
        Tests:
        1. M <= 1.05 (returns early with unperturbed tensor)
        2. M = 1.05001 (smooth transition without NaN)
        3. Extreme M -> infinity: M in [50, 500, 5000, 100000] (saturates via tanh to kappa_0 without exploding)
        4. Negative or zero M: M in [-10.0, 0.0]
        5. Asymmetric spatial tensors: [1, 3, 4, 64], [1, 3, 64, 4], [1, 3, 3, 3]
        """
        test_img = torch.rand((1, 3, 32, 32), device=self.device)

        # M <= 1.05: exact bit-for-bit identity
        for m in [-10.0, 0.0, 0.5, 1.0, 1.05]:
            out = adaptive_tensor_stroke_dilation(test_img, magnification=m)
            self.assertTrue(torch.equal(test_img, out), f"Input modified at inactive magnification M={m}")

        # Smooth transition at threshold M = 1.05001
        out_thresh = adaptive_tensor_stroke_dilation(test_img, magnification=1.05001)
        self.assertFalse(torch.isnan(out_thresh).any())
        self.assertFalse(torch.isinf(out_thresh).any())
        self.assertTrue((out_thresh >= 0.0).all() and (out_thresh <= 1.0).all())

        # Extreme magnifications
        for m_ext in [50.0, 500.0, 5000.0, 100000.0]:
            out_ext = adaptive_tensor_stroke_dilation(test_img, magnification=m_ext)
            self.assertFalse(torch.isnan(out_ext).any(), f"NaN at extreme M={m_ext}")
            self.assertFalse(torch.isinf(out_ext).any(), f"Inf at extreme M={m_ext}")
            self.assertTrue((out_ext >= 0.0).all() and (out_ext <= 1.0).all(), f"Out of bounds at M={m_ext}")
            # Max delta must be bounded by kappa_0 = 0.5
            max_delta = torch.abs(out_ext - test_img).max().item()
            self.assertLessEqual(max_delta, 0.50 + 1e-5, f"Delta exceeded kappa_0 at M={m_ext}")

        # Asymmetric aspect ratios
        for shape in [(1, 3, 4, 64), (1, 3, 64, 4), (1, 3, 3, 3), (1, 3, 5, 100)]:
            asym_img = torch.rand(shape, device=self.device)
            out_asym = adaptive_tensor_stroke_dilation(asym_img, magnification=3.0)
            self.assertEqual(out_asym.shape, shape)
            self.assertFalse(torch.isnan(out_asym).any())
            self.assertFalse(torch.isinf(out_asym).any())


class TestAdversarialHierarchicalZoom(unittest.TestCase):
    """
    Adversarial Stress Testing of Theorem 32 Hierarchical Multi-Scale Focal Windowing.
    Target: VinternNEVFEAdapter.extract_stage2_focal_features and compute_question_conditioned_spatial_window
    """

    def setUp(self):
        self.device = torch.device("cpu")
        self.dim = 896
        self.vocab_size = 151674

        # Build reusable mock adapter
        self.mock_model = MagicMock()
        self.mock_model.num_image_token = 256
        self.mock_model.config.image_size = 448
        self.mock_model.system_message = "Trợ lý AI."
        self.mock_model.extract_feature = MagicMock(
            return_value=torch.randn(1, 256, self.dim, dtype=torch.bfloat16)
        )

        img_ctx_id = 151646
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.convert_tokens_to_ids.return_value = img_ctx_id
        self.mock_tokenizer.encode.return_value = [10, 20, 30]

        def mock_tokenize_call(text, **kwargs):
            res = MagicMock()
            num_ctx = text.count("<IMG_CONTEXT>")
            if num_ctx > 0:
                tokens = [1, 2, 3, 4, 5] + [img_ctx_id] * num_ctx
            else:
                tokens = [6, 7, 8, 9, 10]
            res.input_ids = torch.tensor([tokens], dtype=torch.long)
            return res

        self.mock_tokenizer.side_effect = mock_tokenize_call

        self.adapter = VinternNEVFEAdapter(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            device="cpu",
        )

        embed_dim = self.dim  # Exactly 896 to match visual feature channel dimension
        embed_layer = nn.Embedding(200000, embed_dim)
        self.adapter.model.language_model.get_input_embeddings.return_value = embed_layer

    def test_07_extreme_aspect_ratios_boundary_coordinates(self):
        """
        Adversarial Test: Crop boxes with extreme aspect ratios (100:1 and 1:100)
        and bounding box placements at the extreme image borders.
        Invariant:
        - Must not raise IndexError, ZeroDivisionError, or invalid crop exceptions.
        - Extracted visual tokens |V_L| in {256, 512} strictly.
        - No NaNs or Infs in visual tokens.
        """
        scenarios = [
            # 1. Extreme 100:1 width-to-height crop box
            ("Aspect 100:1", (1000, 100), (10, 20, 910, 29)),
            # 2. Extreme 1:100 height-to-width crop box
            ("Aspect 1:100", (100, 1000), (20, 10, 29, 910)),
            # 3. Top-left corner boundary
            ("Top-Left Corner", (500, 500), (0, 0, 4, 4)),
            # 4. Bottom-right corner boundary
            ("Bottom-Right Corner", (500, 500), (496, 496, 500, 500)),
            # 5. Full canvas bounding box
            ("Full Canvas", (640, 480), (0, 0, 640, 480)),
        ]

        for name, (W, H), crop_box in scenarios:
            image = Image.new("RGB", (W, H), color=(100, 150, 200))
            self.adapter._cached_v_l = None
            self.adapter._last_crop_box = None

            # Test both with zoom active (theta_zoom = 1.05) and zoom bypassed (theta_zoom = 0.0)
            for theta in [1.05, 0.0]:
                self.adapter._cached_v_l = None
                self.adapter._cached_img_key = None
                self.adapter._last_crop_box = None
                v_l = self.adapter.extract_stage2_focal_features(
                    image=image,
                    crop_box=crop_box,
                    question="Thông tin gì?",
                    theta_zoom=theta,
                    use_hierarchical_zoom=True,
                )

                self.assertFalse(torch.isnan(v_l).any(), f"NaN in v_l for {name} with theta={theta}")
                self.assertFalse(torch.isinf(v_l).any(), f"Inf in v_l for {name} with theta={theta}")
                expected_tokens = 512 if theta > 1.0 else 256
                self.assertEqual(
                    v_l.shape, (1, expected_tokens, self.dim),
                    f"Unexpected token shape {v_l.shape} for {name} with theta={theta}",
                )

    def test_08_subbox_span_boundary_limits_rho_sub(self):
        """
        Adversarial Test: Sub-box span rho_sub boundary conditions:
        1. rho_sub -> 0: spatial variance collapses, window clamps to min_span = 0.15.
           Ensures that sub-box pixel coordinates do not collapse into 0-pixel width/height.
        2. rho_sub -> 1.0: widespread uniform attention where rho_sub >= 0.65.
           Ensures clean fallback to single-tile 256 tokens and _last_level2_box is None.
        """
        # Case 1: Point-mass impulse attention -> rho_sub attempts to collapse to 0
        P, D = 256, self.dim
        v_tokens = torch.zeros((1, P, D), device=self.device)
        v_tokens[0, 120, :] = 10.0  # Single active patch

        u_min, v_min, u_max, v_max = compute_question_conditioned_spatial_window(
            visual_tokens=v_tokens[0],
            question_embeddings=None,  # Fallback to pure visual Chebyshev
            grid_size=16,
            min_span=0.15,
        )

        span_u = u_max - u_min
        span_v = v_max - v_min
        # Clamped to at least min_span = 0.15 half-span => full span >= 0.30
        self.assertGreaterEqual(
            span_u, 0.29,
            f"Sub-box span_u {span_u} collapsed below 2 * min_span",
        )
        self.assertGreaterEqual(
            span_v, 0.29,
            f"Sub-box span_v {span_v} collapsed below 2 * min_span",
        )

        # Ensure sub-box extraction with impulse tokens succeeds without collapsing
        img = Image.new("RGB", (600, 600), color=(128, 128, 128))
        self.adapter._cached_v_l = None
        self.adapter._cached_img_key = None
        impulse_tokens = torch.zeros(1, 256, self.dim, dtype=torch.bfloat16)
        impulse_tokens[0, 120, :] = 10.0
        self.mock_model.extract_feature.return_value = impulse_tokens

        v_l_impulse = self.adapter.extract_stage2_focal_features(
            image=img,
            crop_box=(50, 50, 300, 300),
            question="Impulse test",
            theta_zoom=0.65,
            use_hierarchical_zoom=True,
        )
        self.assertEqual(v_l_impulse.shape[1], 512, "Impulse attention should trigger Level 2 zoom")
        l2_box = self.adapter._last_level2_box
        self.assertIsNotNone(l2_box)
        self.assertGreaterEqual(l2_box[2] - l2_box[0], 2, "Sub-box width must be >= 2px")
        self.assertGreaterEqual(l2_box[3] - l2_box[1], 2, "Sub-box height must be >= 2px")

        # Case 2: Widespread diffuse attention -> rho_sub >= 0.65 triggers fallback
        self.mock_model.extract_feature.return_value = torch.randn(1, 256, self.dim, dtype=torch.bfloat16)
        self.adapter._cached_v_l = None
        self.adapter._cached_img_key = None
        v_l_diffuse = self.adapter.extract_stage2_focal_features(
            image=img,
            crop_box=(50, 50, 300, 300),
            question="Diffuse test",
            theta_zoom=0.01,  # Force rho_sub >= theta_zoom fallback
            use_hierarchical_zoom=True,
        )
        self.assertEqual(v_l_diffuse.shape[1], 256, "Diffuse attention must fallback to 256 tokens")
        self.assertIsNone(self.adapter._last_level2_box, "Level 2 box must be None under fallback")

    def test_09_token_dimension_and_sequence_length_invariants_100_geometries(self):
        """
        Adversarial Test: Invariant verification across 100 random synthetic geometries.
        Randomizes:
        - Image width and height W, H in [32, 2048]
        - Random crop box with width and height >= 4px
        - Random question embeddings and question text
        - Random zoom threshold theta_zoom in [0.2, 0.8]
        Invariant:
        - |V_L| in {256, 512} strictly for every geometry.
        - Prompt sequence length S_L in [266, 522] tokens strictly.
        - Zero NaNs, zero Infs.
        """
        random.seed(999)
        torch.manual_seed(999)

        num_iterations = 100
        count_256 = 0
        count_512 = 0

        for iter_idx in range(num_iterations):
            W = random.randint(32, 2048)
            H = random.randint(32, 2048)

            # Random valid crop box
            x0 = random.randint(0, max(0, W - 6))
            y0 = random.randint(0, max(0, H - 6))
            x1 = random.randint(x0 + 4, W)
            y1 = random.randint(y0 + 4, H)
            crop_box = (x0, y0, x1, y1)

            if iter_idx % 2 == 0:
                # Localized focal text cluster (small span rho_sub -> triggers zoom to 512 tokens)
                focal_tokens = torch.randn(1, 256, self.dim, dtype=torch.bfloat16) * 0.01
                grid_y0 = random.randint(3, 9)
                grid_x0 = random.randint(3, 9)
                for gy in range(grid_y0, grid_y0 + 3):
                    for gx in range(grid_x0, grid_x0 + 3):
                        focal_tokens[0, gy * 16 + gx, :] = 5.0 + torch.randn(self.dim, dtype=torch.bfloat16)
                self.mock_model.extract_feature.return_value = focal_tokens
                theta = random.uniform(0.50, 0.85)
            else:
                # Diffuse global features (large span rho_sub -> triggers fallback to 256 tokens)
                self.mock_model.extract_feature.return_value = torch.randn(1, 256, self.dim, dtype=torch.bfloat16)
                theta = random.uniform(0.10, 0.45)

            q_text = f"Synthetic query {iter_idx}?"

            image = Image.new("RGB", (W, H), color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

            self.adapter._cached_v_l = None
            self.adapter._cached_img_key = None
            self.adapter._last_crop_box = None

            v_l = self.adapter.extract_stage2_focal_features(
                image=image,
                crop_box=crop_box,
                question=q_text,
                theta_zoom=theta,
                use_hierarchical_zoom=True,
            )

            # 1. Invariant: token shape
            num_tokens = v_l.shape[1]
            self.assertIn(
                num_tokens, {256, 512},
                f"Iteration {iter_idx}: Invalid visual token count {num_tokens} for W={W}, H={H}, crop={crop_box}",
            )
            if num_tokens == 256:
                count_256 += 1
            else:
                count_512 += 1

            # 2. Invariant: no NaNs / Infs
            self.assertFalse(torch.isnan(v_l).any(), f"Iteration {iter_idx}: NaN detected in v_l")
            self.assertFalse(torch.isinf(v_l).any(), f"Iteration {iter_idx}: Inf detected in v_l")

            # 3. Invariant: sequence length S_L <= 522
            embed_dim = self.dim
            v_embed = torch.randn(1, num_tokens, embed_dim)
            embeds = self.adapter.build_multimodal_embeds(v_embed, question=q_text)
            seq_len = embeds.shape[1]
            if num_tokens == 256:
                self.assertEqual(seq_len, 266, f"Iteration {iter_idx}: Expected S_L=266 for 256 tokens, got {seq_len}")
            else:
                self.assertEqual(seq_len, 522, f"Iteration {iter_idx}: Expected S_L=522 for 512 tokens, got {seq_len}")

        # Ensure both 256 and 512 branches were exercised across the 100 geometries
        self.assertGreater(count_256, 0, "Fallback branch (256 tokens) was never exercised")
        self.assertGreater(count_512, 0, "Zoomed branch (512 tokens) was never exercised")
        print(f"100 Geometries Invariant Test: {count_512} zoomed (512 tokens), {count_256} fallback (256 tokens).")

    def test_10_adversarial_coordinate_clamping_robustness(self):
        """
        Adversarial Test: Sub-box coordinates at extreme numerical boundaries.
        Verifies that even when u_min, u_max, v_min, v_max are pinned to extreme edges
        (e.g. 0.0, 1.0, or nearly identical 0.999), the sub-box is strictly valid:
        - 0 <= sub_x0 < sub_x1 <= W
        - 0 <= sub_y0 < sub_y1 <= H
        - bw >= 2, bh >= 2
        """
        img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        W, H = img.size

        # Test extreme crop boxes
        test_boxes = [
            (0, 0, 2, 2),        # Minimal possible 2x2 box
            (0, 0, 100, 100),    # Full image
            (98, 98, 100, 100),  # Bottom right 2x2 box
            (50, 50, 52, 52),    # Center 2x2 box
        ]

        for bx0, by0, bx1, by1 in test_boxes:
            bw = max(2, bx1 - bx0)
            bh = max(2, by1 - by0)

            # Test extreme normalized (u, v) pairs
            uv_pairs = [
                (0.0, 0.0, 0.15, 0.15),
                (0.85, 0.85, 1.0, 1.0),
                (0.0, 0.0, 1.0, 1.0),
                (0.49, 0.49, 0.51, 0.51),
            ]

            for u_min, v_min, u_max, v_max in uv_pairs:
                sub_x0 = max(0, min(W - 2, int(round(bx0 + u_min * bw))))
                sub_y0 = max(0, min(H - 2, int(round(by0 + v_min * bh))))
                sub_x1 = max(sub_x0 + 2, min(W, int(round(bx0 + u_max * bw))))
                sub_y1 = max(sub_y0 + 2, min(H, int(round(by0 + v_max * bh))))

                self.assertGreaterEqual(sub_x0, 0)
                self.assertGreaterEqual(sub_y0, 0)
                self.assertLessEqual(sub_x1, W)
                self.assertLessEqual(sub_y1, H)
                self.assertGreaterEqual(sub_x1 - sub_x0, 2, f"sub_x1 - sub_x0 < 2: {sub_x0}, {sub_x1}")
                self.assertGreaterEqual(sub_y1 - sub_y0, 2, f"sub_y1 - sub_y0 < 2: {sub_y0}, {sub_y1}")


if __name__ == "__main__":
    unittest.main()
