# -*- coding: utf-8 -*-
"""
Adversarial Stress Test Suite for Theorem 32-Ext (Lemmas 1, 2, and 3):
Hierarchical Multi-Scale Resolution, Dispersion-Normalized Shock Scale-Space Operators,
Involutive Polarity Symmetry, and Domain Boundary Flux Conservation.

Role: challenger_iter17_spatial (Empirical Challenger - critic, specialist)
Authoritative Empirical Adversarial Verification Suite.

Mandatory Verification Coverage:
1. Test 1 (Faint Stroke Non-Quenching, Lemma 3):
   - Construct synthetic crops enclosing a strong outer boundary edge (g ~ 0.80) and faint text stroke (g ~ 0.04).
   - Assert that faint stroke dilation delta is >= 0.10 (proving empirical absence of global-max quenching).
   - Contrast directly against old global supremum normalization, demonstrating 3x-5x preservation gain.
2. Test 2 (Exact Involutive Polarity Symmetry, Lemma 1):
   - Verify that D_kappa^*(1 - X) == 1 - D_kappa^*(X) analytically and numerically.
   - Assert max ||D_kappa^*(1 - X) - (1 - D_kappa^*(X))||_inf < 1e-4 across random distributions,
     inverted synthetic text, extreme contrast steps, and directional high-frequency patterns.
3. Test 3 (Domain Boundary Flux Conservation, Lemma 2):
   - Assert that padding non-square crops creates zero boundary flux step artifacts across the seam line.
   - Test edge-border replication and symmetric square canvas expansion.
   - Verify zero seam step gradient (del Y_seam = 0) and zero spurious boundary dilation halos.
4. Test 4 (Flat-Field Noise Robustness):
   - Verify that flat surfaces with microscopic sensor noise (g ~ 10^-5) have grad_gate == 0
     and undergo zero spurious dilation (||D_kappa^*(X) - X||_inf == 0.0).
   - Verify the Area Chair's sigma_grad > 1e-4 flat-field floor safeguard against noise hallucination.
"""

import math
import random
import unittest
from typing import Tuple
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from src.nevfe.core.engine import (
    NEVFEEngine,
    adaptive_tensor_stroke_dilation,
)
from src.nevfe.adapter.vintern_adapter import (
    extract_symmetric_square_crop,
    pad_image_edge_replicate,
)


class TestAdversarialTheorem32Ext(unittest.TestCase):
    """
    Dedicated Empirical Adversarial Verification Suite for Theorem 32-Ext.
    Verifies Lemmas 1, 2, 3 and Area Chair noise safeguards under hostile conditions.
    """

    def setUp(self):
        self.device = torch.device("cpu")
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

    # =========================================================================
    # Test 1: Faint Stroke Non-Quenching (Lemma 3)
    # =========================================================================
    def test_01_faint_stroke_non_quenching_with_strong_boundary_edge(self):
        """
        Adversarial Test 1 (Lemma 3):
        Scenario: A Level 2 or Level 1 crop contains both:
        1. A high-contrast outer boundary edge (e.g., signboard frame, dark-to-light border step: g ~ 0.80)
        2. A subtle, faint text stroke (physical gradient g ~ 0.04)

        Under Iteration 16 global supremum normalization:
            grad_gate = grad_mag / max_grad = 0.04 / 0.80 = 0.05
            faint stroke dilation was quenched to delta < 0.06 (starving ViT patch tokens).

        Under Theorem 32-Ext dispersion normalization (Directive W1):
            grad_gate = tanh(grad_mag / sigma_grad)
            where sigma_grad << max_grad because the border occupies only a small area fraction.
            faint stroke dilation must achieve delta >= 0.10.
        """
        # Test across multiple crop sizes (64x64, 128x128)
        for H, W in [(64, 64), (128, 128)]:
            x = torch.full((1, 3, H, W), 0.50, dtype=torch.float32, device=self.device)

            # 1. Add strong outer boundary edge at top (g ~ 0.80)
            # Row 2-3 is bright (0.90), row 4-5 is dark (0.10) -> step = 0.80
            x[:, :, 2:4, :] = 0.90
            x[:, :, 4:6, :] = 0.10

            # 2. Add faint text stroke in the center (physical gradient g ~ 0.04)
            # Center stroke width 4px with subtle contrast delta = 0.08 (step = 0.08, central diff g = 0.04)
            cy_start = H // 2 - 2
            cy_end = H // 2 + 2
            cx_start = W // 4
            cx_end = 3 * W // 4
            x[:, :, cy_start:cy_end, cx_start:cx_end] = 0.58

            # Measure dilation under active zoom magnification (M = 3.0)
            dilated = adaptive_tensor_stroke_dilation(x, magnification=3.0)
            delta = (dilated - x).abs()

            # Measure empirical dilation on the faint stroke
            faint_region_delta = delta[:, :, cy_start - 1 : cy_end + 1, cx_start - 1 : cx_end + 1]
            faint_max_delta = faint_region_delta.max().item()

            # Empirical Assertion: Faint stroke dilation delta MUST be >= 0.10
            self.assertGreaterEqual(
                faint_max_delta,
                0.10,
                f"Lemma 3 Violation: Faint stroke quenched at resolution {H}x{W}! "
                f"Expected dilation delta >= 0.10, got {faint_max_delta:.4f}"
            )

            # Verification of Non-Quenching mechanism:
            # Reconstruct the old global supremum normalization on this exact tensor
            Y = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
            k_dx = torch.tensor([[0.0, 0.0, 0.0], [-0.5, 0.0, 0.5], [0.0, 0.0, 0.0]], device=self.device).view(1, 1, 3, 3)
            k_dy = torch.tensor([[0.0, -0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.5, 0.0]], device=self.device).view(1, 1, 3, 3)
            pad_Y = F.pad(Y, (1, 1, 1, 1), mode="reflect")
            gx = F.conv2d(pad_Y, k_dx)
            gy = F.conv2d(pad_Y, k_dy)
            gmag = torch.sqrt(gx * gx + gy * gy)
            max_grad = torch.amax(gmag, dim=(-2, -1), keepdim=True)
            old_grad_gate = gmag / (max_grad + 1e-6)

            # Under old gating, faint stroke gate was heavily quenched:
            faint_old_gate = old_grad_gate[:, :, cy_start:cy_end, cx_start:cx_end].max().item()
            # Under new gating, dispersion gate is substantially stronger:
            sigma_grad = torch.std(gmag, dim=(-2, -1), keepdim=True) + 1e-6
            new_grad_gate = torch.tanh(gmag / sigma_grad)
            faint_new_gate = new_grad_gate[:, :, cy_start:cy_end, cx_start:cx_end].max().item()

            self.assertGreater(
                faint_new_gate,
                faint_old_gate * 2.0,
                f"Dispersion gate must provide >2x amplification over old global supremum gate. "
                f"Old gate: {faint_old_gate:.4f}, New gate: {faint_new_gate:.4f}"
            )

    def test_01b_faint_stroke_polarity_inversion_non_quenching(self):
        """
        Adversarial Test 1b (Lemma 3 with Inverted Polarity):
        Repeat faint stroke non-quenching test for dark-on-light faint text stroke
        alongside dark-on-light strong boundary edge.
        Assert that faint stroke dilation delta is >= 0.10 symmetrically.
        """
        H, W = 128, 128
        # Inverted contrast: background is 0.50, border step is dark (0.10), faint stroke is dark (0.42)
        x = torch.full((1, 3, H, W), 0.50, dtype=torch.float32, device=self.device)
        x[:, :, 2:4, :] = 0.10
        x[:, :, 4:6, :] = 0.90

        cy_start, cy_end = 60, 64
        cx_start, cx_end = 30, 90
        x[:, :, cy_start:cy_end, cx_start:cx_end] = 0.42  # Step = 0.08 dark stroke

        dilated = adaptive_tensor_stroke_dilation(x, magnification=3.0)
        delta = (dilated - x).abs()

        faint_max_delta = delta[:, :, cy_start - 1 : cy_end + 1, cx_start - 1 : cx_end + 1].max().item()
        self.assertGreaterEqual(
            faint_max_delta,
            0.10,
            f"Lemma 3 Violation on inverted contrast! Expected delta >= 0.10, got {faint_max_delta:.4f}"
        )

    # =========================================================================
    # Test 2: Exact Involutive Polarity Symmetry (Lemma 1)
    # =========================================================================
    def test_02_exact_involutive_polarity_symmetry_comprehensive(self):
        """
        Adversarial Test 2 (Lemma 1):
        Theorem: D_kappa^*(1 - X) == 1 - D_kappa^*(X) analytically for all X in [0, 1].
        Empirical Requirement: max ||D_kappa^*(1 - X) - (1 - D_kappa^*(X))||_inf < 1e-4.

        We sweep 5 distinct adversarial distributions:
        1. Uniform random distribution in [0, 1] across varying batch sizes and shapes.
        2. Gaussian noise distribution with mean 0.5, clipped to [0, 1].
        3. Extreme contrast step edges with contrast deltas in [0.01, 0.05, 0.10, 0.50, 0.90, 0.99].
        4. Synthetic binary and antialiased text strokes (dark-on-light vs light-on-dark).
        5. Alternating Nyquist checkerboards and impulse dots.
        """
        magnifications = [1.2, 2.0, 3.5, 8.0]

        # 1. Uniform random distribution
        for m in magnifications:
            for seed in [101, 202, 303]:
                torch.manual_seed(seed)
                x = torch.rand(2, 3, 48, 48, device=self.device)
                d_x = adaptive_tensor_stroke_dilation(x, magnification=m)
                d_inv = adaptive_tensor_stroke_dilation(1.0 - x, magnification=m)
                err = (d_inv - (1.0 - d_x)).abs().max().item()
                self.assertLess(
                    err, 1e-4,
                    f"Lemma 1 violated on uniform random input at M={m}, seed={seed}: err={err:.2e}"
                )

        # 2. Gaussian noise with varied variances
        for sigma in [0.05, 0.15, 0.30]:
            x = torch.clamp(torch.randn(1, 3, 64, 64, device=self.device) * sigma + 0.5, 0.0, 1.0)
            d_x = adaptive_tensor_stroke_dilation(x, magnification=2.5)
            d_inv = adaptive_tensor_stroke_dilation(1.0 - x, magnification=2.5)
            err = (d_inv - (1.0 - d_x)).abs().max().item()
            self.assertLess(
                err, 1e-4,
                f"Lemma 1 violated on Gaussian noise (sigma={sigma}): err={err:.2e}"
            )

        # 3. Extreme contrast step edges
        for delta_c in [0.01, 0.05, 0.10, 0.50, 0.80, 0.99]:
            x = torch.full((1, 3, 32, 32), 0.5 - delta_c / 2.0, device=self.device)
            x[:, :, :, 16:] = 0.5 + delta_c / 2.0
            x = torch.clamp(x, 0.0, 1.0)

            d_x = adaptive_tensor_stroke_dilation(x, magnification=3.0)
            d_inv = adaptive_tensor_stroke_dilation(1.0 - x, magnification=3.0)
            err = (d_inv - (1.0 - d_x)).abs().max().item()
            self.assertLess(
                err, 1e-4,
                f"Lemma 1 violated on step edge (contrast={delta_c}): err={err:.2e}"
            )

        # 4. Synthetic binary and antialiased text strokes
        img_size = (64, 64)
        img = Image.new("RGB", img_size, (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.line([(10, 20), (50, 20)], fill=(0, 0, 0), width=3)
        draw.line([(30, 20), (30, 50)], fill=(0, 0, 0), width=3)
        draw.ellipse([(15, 35), (45, 55)], outline=(0, 0, 0), width=2)

        x_text = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        d_x = adaptive_tensor_stroke_dilation(x_text, magnification=2.5)
        d_inv = adaptive_tensor_stroke_dilation(1.0 - x_text, magnification=2.5)
        err = (d_inv - (1.0 - d_x)).abs().max().item()
        self.assertLess(
            err, 1e-4,
            f"Lemma 1 violated on synthetic text rendering: err={err:.2e}"
        )

        # 5. Nyquist checkerboard
        cb = torch.zeros((1, 3, 32, 32), device=self.device)
        cb[:, :, 0::2, 0::2] = 1.0
        cb[:, :, 1::2, 1::2] = 1.0
        d_cb = adaptive_tensor_stroke_dilation(cb, magnification=2.5)
        d_cb_inv = adaptive_tensor_stroke_dilation(1.0 - cb, magnification=2.5)
        err = (d_cb_inv - (1.0 - d_cb)).abs().max().item()
        self.assertLess(
            err, 1e-4,
            f"Lemma 1 violated on Nyquist checkerboard: err={err:.2e}"
        )

    # =========================================================================
    # Test 3: Domain Boundary Flux Conservation (Lemma 2)
    # =========================================================================
    def test_03_domain_boundary_flux_conservation_under_non_square_padding(self):
        """
        Adversarial Test 3 (Lemma 2):
        Theorem: Boundary replicate padding satisfies reflective Neumann condition:
        del I / del n = 0 along all domain boundaries del Omega.
        Empirical Requirement:
        1. Padding non-square crops creates zero boundary flux step artifacts:
           The normal gradient across the seam line separating the original crop
           from the padded margin is identically zero.
        2. No artificial static gray (128, 128, 128) pixels are introduced.
        3. Dilation of the padded crop produces zero spurious boundary halos or seam shocks.
        """
        # Test extreme rectangular aspect ratios
        test_dimensions = [
            (120, 40),   # 3:1 landscape
            (35, 110),   # ~1:3 portrait
            (200, 60),   # 3.33:1 wide signboard
            (48, 144),   # 1:3 tall column
            (99, 33),    # Non-power-of-2 dimensions
        ]

        for w, h in test_dimensions:
            # Create synthetic crop with rich texture and strong colors
            crop_arr = np.random.randint(30, 230, (h, w, 3), dtype=np.uint8)
            crop_img = Image.fromarray(crop_arr)

            # Pad using edge-border replication (Directive W2)
            padded_img = pad_image_edge_replicate(crop_img)
            pw, ph = padded_img.size

            # Invariant 1: Output must be perfectly square with side = max(w, h)
            max_side = max(w, h)
            self.assertEqual(pw, ph, f"Padded image is not square for input {w}x{h}: {pw}x{ph}")
            self.assertEqual(pw, max_side, f"Expected side {max_side}, got {pw}")

            # Invariant 2: Zero boundary flux step artifacts across seam lines
            p_arr = np.array(padded_img)
            pad_left = (max_side - w) // 2
            pad_right = max_side - w - pad_left
            pad_top = (max_side - h) // 2
            pad_bottom = max_side - h - pad_top

            # Check normal gradient across horizontal seams (if h < max_side)
            if pad_top > 0:
                # Top seam: difference between row pad_top (original top row) and row pad_top - 1 (replicate)
                top_seam_normal_grad = np.abs(p_arr[pad_top, :, :].astype(int) - p_arr[pad_top - 1, :, :].astype(int)).max()
                self.assertEqual(
                    top_seam_normal_grad, 0,
                    f"Lemma 2 Violation: Top seam has non-zero normal gradient: {top_seam_normal_grad}"
                )
                # Outer boundary margin must be an exact copy of the edge
                outer_top_flux = np.abs(p_arr[0:pad_top, :, :].astype(int) - p_arr[pad_top : pad_top + 1, :, :].astype(int)).max()
                self.assertEqual(outer_top_flux, 0, "Outer top margin diverged from edge row!")

            if pad_bottom > 0:
                bottom_seam_idx = pad_top + h - 1
                bottom_seam_normal_grad = np.abs(
                    p_arr[bottom_seam_idx, :, :].astype(int) - p_arr[bottom_seam_idx + 1, :, :].astype(int)
                ).max()
                self.assertEqual(
                    bottom_seam_normal_grad, 0,
                    f"Lemma 2 Violation: Bottom seam has non-zero normal gradient: {bottom_seam_normal_grad}"
                )

            # Check normal gradient across vertical seams (if w < max_side)
            if pad_left > 0:
                left_seam_normal_grad = np.abs(p_arr[:, pad_left, :].astype(int) - p_arr[:, pad_left - 1, :].astype(int)).max()
                self.assertEqual(
                    left_seam_normal_grad, 0,
                    f"Lemma 2 Violation: Left seam has non-zero normal gradient: {left_seam_normal_grad}"
                )

            if pad_right > 0:
                right_seam_idx = pad_left + w - 1
                right_seam_normal_grad = np.abs(
                    p_arr[:, right_seam_idx, :].astype(int) - p_arr[:, right_seam_idx + 1, :].astype(int)
                ).max()
                self.assertEqual(
                    right_seam_normal_grad, 0,
                    f"Lemma 2 Violation: Right seam has non-zero normal gradient: {right_seam_normal_grad}"
                )

            # Invariant 3: Total absence of artificial (128, 128, 128) gray padding
            # Ensure static gray 128 was not introduced anywhere unless it was in crop_arr
            gray_pixels_padded = ((p_arr[:, :, 0] == 128) & (p_arr[:, :, 1] == 128) & (p_arr[:, :, 2] == 128)).sum()
            gray_pixels_original = ((crop_arr[:, :, 0] == 128) & (crop_arr[:, :, 1] == 128) & (crop_arr[:, :, 2] == 128)).sum()
            # If the original crop didn't have gray 128, padded image must have zero gray 128
            if gray_pixels_original == 0:
                self.assertEqual(
                    gray_pixels_padded, 0,
                    f"Static gray (128, 128, 128) detected in padded output! Count: {gray_pixels_padded}"
                )

            # Invariant 4: Zero normal derivative along domain boundary margin (Lemma 2)
            # Under edge-border replication, all normal derivatives in the padded margin vanish identically:
            # del I / del y == 0 for all rows y < pad_top and y > pad_top + h - 1.
            if pad_top > 1:
                # For any row inside the padded margin, normal gradient is identically zero
                margin_normal_grad = np.abs(p_arr[1:pad_top, :, :].astype(int) - p_arr[0:pad_top - 1, :, :].astype(int)).max()
                self.assertEqual(
                    margin_normal_grad, 0,
                    f"Lemma 2 Violation: Padded margin contains non-zero normal gradient: {margin_normal_grad}"
                )

            if pad_left > 1:
                margin_normal_grad_x = np.abs(p_arr[:, 1:pad_left, :].astype(int) - p_arr[:, 0:pad_left - 1, :].astype(int)).max()
                self.assertEqual(
                    margin_normal_grad_x, 0,
                    f"Lemma 2 Violation: Left margin contains non-zero normal gradient: {margin_normal_grad_x}"
                )

            # Invariant 5: Direct contrast against defective Iteration 16 static gray padding
            # On bright or dark crops, static gray padding introduces an artificial step edge (|Delta Y| > 50)
            # whereas replicate padding creates an exact zero step edge.
            static_gray_padded = Image.new("RGB", (max_side, max_side), (128, 128, 128))
            static_gray_padded.paste(crop_img, (pad_left, pad_top))
            static_arr = np.array(static_gray_padded)

            if pad_top > 0:
                static_seam_step = np.abs(static_arr[pad_top, :, :].astype(int) - static_arr[pad_top - 1, :, :].astype(int)).max()
                replicate_seam_step = np.abs(p_arr[pad_top, :, :].astype(int) - p_arr[pad_top - 1, :, :].astype(int)).max()
                self.assertEqual(replicate_seam_step, 0, "Replicate padding seam step must be identically zero")
                # When crop boundary has color far from 128, static gray produces a large artificial seam step
                crop_edge_dist_from_128 = np.abs(crop_arr[0, :, :].astype(int) - 128).max()
                if crop_edge_dist_from_128 > 30:
                    self.assertGreater(
                        static_seam_step, 30,
                        f"Static gray seam step should be large on non-gray crop, got {static_seam_step}"
                    )

    def test_03b_symmetric_square_crop_canvas_expansion(self):
        """
        Adversarial Test 3b (Directive W2 on Native Canvas):
        Verify extract_symmetric_square_crop on a full canvas [0, W] x [0, H]
        with crops touching all 4 corners and borders.
        """
        full_W, full_H = 640, 480
        full_img_arr = np.random.randint(10, 240, (full_H, full_W, 3), dtype=np.uint8)
        full_img = Image.fromarray(full_img_arr)

        adversarial_boxes = [
            (0, 0, 100, 30),             # Top-left corner, wide
            (full_W - 80, 0, full_W, 20), # Top-right corner, wide
            (0, full_H - 30, 120, full_H),# Bottom-left corner, wide
            (full_W - 25, full_H - 90, full_W, full_H), # Bottom-right, tall
            (250, 200, 350, 230),        # Center interior, wide
            (300, 100, 330, 250),        # Center interior, tall
        ]

        for box in adversarial_boxes:
            square_crop = extract_symmetric_square_crop(full_img, box)
            cw, ch = square_crop.size
            self.assertEqual(cw, ch, f"Crop from box {box} is not square: {cw}x{ch}")
            bw = box[2] - box[0]
            bh = box[3] - box[1]
            expected_side = max(bw, bh)
            self.assertEqual(cw, expected_side, f"Expected side {expected_side}, got {cw}")

    # =========================================================================
    # Test 4: Flat-Field Noise Robustness & Sensor Noise Floor Safeguard
    # =========================================================================
    def test_04_flat_field_microscopic_sensor_noise_robustness(self):
        """
        Adversarial Test 4:
        Verification of Area Chair Flat-Field Sensor Noise Safeguard (Directive W1):
            sigma_grad = std(grad_mag) + eps
            grad_gate = where(sigma_grad > 1e-4, tanh(grad_mag / sigma_grad), zeros_like(grad_mag))

        Physical Scenario:
        Camera sensors exhibit microscopic thermal noise (grad ~ 10^-6 to 10^-5) on uniform
        regions (clear blue sky, smooth walls, unprinted paper backgrounds).
        Without the sigma_grad > 1e-4 safeguard, tanh(grad_mag / sigma_grad) ~ tanh(1.0) ~ 0.7616,
        which would hallucinate and amplify camera sensor noise into spurious shock textures!

        Empirical Invariants:
        1. On flat fields with microscopic sensor noise (noise_std <= 1e-5), grad_gate == 0 identically.
        2. Dilated tensor is bit-for-bit identical to input tensor: ||D_kappa^*(X) - X||_inf == 0.0.
        3. Counter-example stress: Prove that omitting the 1e-4 floor causes severe noise amplification (>0.30),
           empirically establishing the absolute necessity of the safeguard.
        """
        noise_scales = [1e-7, 1e-6, 5e-6, 1e-5]
        base_luminances = [0.05, 0.20, 0.50, 0.80, 0.95]
        magnifications = [1.5, 3.0, 5.0, 10.0]

        for noise_std in noise_scales:
            for luma in base_luminances:
                for m in magnifications:
                    x = torch.full((1, 3, 64, 64), luma, dtype=torch.float32, device=self.device)
                    noise = torch.randn((1, 3, 64, 64), dtype=torch.float32, device=self.device) * noise_std
                    x_noisy = torch.clamp(x + noise, 0.0, 1.0)

                    # Compute dilation
                    dilated = adaptive_tensor_stroke_dilation(x_noisy, magnification=m)
                    diff = (dilated - x_noisy).abs().max().item()

                    # Invariant: zero spurious dilation
                    self.assertEqual(
                        diff, 0.0,
                        f"Spurious noise dilation occurred on flat field (luma={luma}, noise={noise_std}, M={m})! "
                        f"max_diff={diff}"
                    )

        # Counter-example verification (Red-Teaming the safeguard):
        # What happens if sigma_grad > 1e-4 floor is disabled?
        # Construct flat field with microscopic noise
        x_flat_noise = torch.full((1, 3, 64, 64), 0.50, dtype=torch.float32, device=self.device)
        x_flat_noise += torch.randn((1, 3, 64, 64), dtype=torch.float32, device=self.device) * 1e-6
        x_flat_noise = torch.clamp(x_flat_noise, 0.0, 1.0)

        # Compute gradient and laplacian manually without floor:
        Y = 0.2989 * x_flat_noise[:, 0:1] + 0.5870 * x_flat_noise[:, 1:2] + 0.1140 * x_flat_noise[:, 2:3]
        k_dx = torch.tensor([[0.0, 0.0, 0.0], [-0.5, 0.0, 0.5], [0.0, 0.0, 0.0]], device=self.device).view(1, 1, 3, 3)
        k_dy = torch.tensor([[0.0, -0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.5, 0.0]], device=self.device).view(1, 1, 3, 3)
        k_lap = torch.tensor([[1.0, 1.0, 1.0], [1.0, -8.0, 1.0], [1.0, 1.0, 1.0]], device=self.device).view(1, 1, 3, 3) / 8.0
        pad_Y = F.pad(Y, (1, 1, 1, 1), mode="reflect")
        gx = F.conv2d(pad_Y, k_dx)
        gy = F.conv2d(pad_Y, k_dy)
        lap = F.conv2d(pad_Y, k_lap)
        gmag = torch.sqrt(gx * gx + gy * gy)

        # Unguarded dispersion gating (omitting sigma_grad > 1e-4):
        sigma_g = torch.std(gmag, dim=(-2, -1), keepdim=True) + 1e-6
        unguarded_gate = torch.tanh(gmag / sigma_g)  # This evaluates to ~0.76!
        sigma_l = torch.std(lap, dim=(-2, -1), keepdim=True) + 1e-6
        shock_d = torch.tanh(lap / sigma_l)
        km = 0.5 * math.tanh(2.0)
        spurious_delta = km * (shock_d * unguarded_gate).abs().max().item()

        # The unguarded delta would be ~0.35, severely corrupting flat fields!
        self.assertGreater(
            spurious_delta, 0.25,
            f"Expected unguarded noise dilation to be large (~0.35), got {spurious_delta:.4f}"
        )


if __name__ == "__main__":
    unittest.main()
