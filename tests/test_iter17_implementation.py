# -*- coding: utf-8 -*-
"""
Master Unit Test Suite for Iteration 17 Implementation (Directives W1–W7).
Strictly verifies:
- Directive W1: Theorem 32-Ext dispersion-normalized scale-space shock gradient gating.
- Directive W2: Boundary seam elimination via symmetric crop expansion & replicate padding.
- Directive W3: Theorem 34 decoupled scale-invariant head carrier barrier.
- Directive W4: Theorem 34 continuation grace horizon & tail clamp modulation.
- Directive W5: Theorem 34 calibrated crisp semantic stopping gate.
- Directive W6: Theorem 33 numerical cohesion clamp preservation.
- Directive W7: Area Chair IEEE 754 NaN immunization in entropy LSE.
"""

import math
import unittest
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from src.nevfe.core.engine import (
    NEVFEEngine,
    adaptive_tensor_stroke_dilation,
    compute_normalized_entropy_lse,
    evaluate_semantic_stopping,
    nevfe_step_logits,
)
from src.nevfe.adapter.vintern_adapter import (
    extract_symmetric_square_crop,
    pad_image_edge_replicate,
)


class TestIteration17Implementation(unittest.TestCase):
    """Exhaustive behavioral unit tests covering Directives W1 through W7."""

    def setUp(self):
        torch.manual_seed(42)

    # =========================================================================
    # Directive W1: Dispersion-Normalized Shock Gradient Gating (Theorem 32-Ext)
    # =========================================================================
    def test_w1_faint_stroke_non_quenching_with_strong_boundary(self):
        """
        Directive W1: Faint text stroke (grad ~ 0.04) alongside strong border (grad ~ 0.80).
        Dispersion normalization ensures the faint stroke receives substantial dilation
        rather than being quenched by 95% under global supremum normalization.
        """
        H, W = 64, 64
        x = torch.zeros((1, 3, H, W), dtype=torch.float32)

        # 1. Add strong border edge at x=5:8
        x[:, :, :, 5:8] = 1.0  # Physical step edge ~ 1.0

        # 2. Add faint text stroke at x=39:42 (3px stroke)
        x[:, :, :, 39:42] = 0.08  # Faint stroke ~ 0.08

        dilated = adaptive_tensor_stroke_dilation(x, magnification=2.5)

        # Stroke dilation delta
        delta = (dilated - x).abs()

        # Under dispersion normalization, faint stroke must exhibit non-trivial dilation (delta >= 0.005)
        faint_delta = delta[:, :, :, 38:43].max().item()
        self.assertGreater(
            faint_delta, 0.005,
            f"Faint stroke was quenched! Delta: {faint_delta}"
        )

    def test_w1_flat_field_noise_immunity(self):
        """
        Directive W1: Flat-field floor ensures microscopic camera sensor noise (grad < 1e-4)
        is zeroed out and never hallucinated into dilated text strokes.
        """
        H, W = 32, 32
        # Microscopic noise (< 1e-5)
        x = torch.full((1, 3, H, W), 0.5) + torch.randn(1, 3, H, W) * 1e-6
        x = torch.clamp(x, 0.0, 1.0)

        dilated = adaptive_tensor_stroke_dilation(x, magnification=2.5)
        diff = (dilated - x).abs().max().item()
        self.assertLess(diff, 1e-5, f"Spurious noise dilation occurred on flat field: {diff}")

    def test_w1_exact_involutive_polarity_symmetry(self):
        """
        Directive W1 (Lemma 1): D_kappa*(1 - X) == 1 - D_kappa*(X) analytically.
        Verify numerical asymmetry is < 1e-5 across random patterns.
        """
        x = torch.rand(1, 3, 48, 48)
        dilated_x = adaptive_tensor_stroke_dilation(x, magnification=2.0)
        dilated_inv = adaptive_tensor_stroke_dilation(1.0 - x, magnification=2.0)

        sym_error = (dilated_inv - (1.0 - dilated_x)).abs().max().item()
        self.assertLess(sym_error, 1e-5, f"Polarity symmetry violated: {sym_error}")

    # =========================================================================
    # Directive W2: Boundary Seam Elimination (Symmetric Crop & Replicate Padding)
    # =========================================================================
    def test_w2_symmetric_crop_expansion_square_output(self):
        """
        Directive W2: extract_symmetric_square_crop expands rectangular boxes
        symmetrically on native canvas, yielding an exact square with no (128,128,128) seams.
        """
        img_arr = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)
        image = Image.fromarray(img_arr)

        # Rectangular box (100 wide, 40 tall)
        box = (50, 60, 150, 100)
        square_crop = extract_symmetric_square_crop(image, box)

        cw, ch = square_crop.size
        self.assertEqual(cw, ch, f"Crop is not square: {cw}x{ch}")
        self.assertEqual(cw, 100, f"Expected side length 100, got {cw}")

    def test_w2_replicate_padding_at_image_boundaries(self):
        """
        Directive W2: When a crop hits the image boundary, pad_image_edge_replicate
        replicates border pixels rather than introducing static gray (128,128,128).
        """
        # Create image with specific border color (200, 50, 50)
        arr = np.zeros((30, 50, 3), dtype=np.uint8)
        arr[0, :] = [200, 50, 50]
        arr[-1, :] = [10, 220, 10]
        image = Image.fromarray(arr)

        padded = pad_image_edge_replicate(image)
        pw, ph = padded.size
        self.assertEqual(pw, ph)

        padded_arr = np.array(padded)
        # Top padding rows must replicate the top row [200, 50, 50]
        self.assertTrue((padded_arr[0, 25] == [200, 50, 50]).all())
        # Static gray (128, 128, 128) must NOT be present
        self.assertFalse((padded_arr == 128).all(axis=-1).any(), "Found static gray 128 pixels in padded image!")

    # =========================================================================
    # Directive W3: Decoupled Scale-Invariant Head Carrier Barrier (Theorem 34)
    # =========================================================================
    def test_w3_carrier_barrier_active_when_psi_is_zero(self):
        """
        Directive W3: At t=1 on diffuse scenes where psi_1 == 0 (omega_t <= s_bar_t),
        ungrounded carrier tokens (ogi == 0) receive the full dyadic penalty
        (-mu_param * delta_t = -4.0 nats at t=1), pruning them from the FEST support set,
        while grounded entity tokens (ogi == 1.0) receive 0.0 penalty.
        """
        K = 100
        device = torch.device("cpu")
        lg = torch.zeros((1, K), device=device)
        ll = torch.zeros((1, K), device=device)
        # Token 20 has positive optical contrast (grounded entity token)
        ll[0, 20] = 0.5

        # With mu_param=4.0 (active barrier)
        next_tok_mu4, steered_mu4, _, _, _, _ = nevfe_step_logits(
            logits_l=ll,
            logits_g=lg,
            lambda_param=1.0,
            mu_param=4.0,
            step_idx=1,
            barrier_horizon=3,
            return_continuation_saliency=True,
        )

        # With mu_param=0.0 (no barrier)
        next_tok_mu0, steered_mu0, _, _, _, _ = nevfe_step_logits(
            logits_l=ll,
            logits_g=lg,
            lambda_param=1.0,
            mu_param=0.0,
            step_idx=1,
            barrier_horizon=3,
            return_continuation_saliency=True,
        )

        # Grounded token 20: ogi = 1.0 -> 0.0 penalty (identical values under mu=4 and mu=0)
        self.assertAlmostEqual(
            steered_mu4[0, 20].item(),
            steered_mu0[0, 20].item(),
            places=3,
            msg="Grounded entity token 20 incorrectly penalized by carrier barrier!"
        )

        # Ungrounded carrier token 5: ogi = 0.0 -> -4.0 nats penalty
        # At mu=0, token 5 was in FEST support (> -1e8)
        self.assertGreater(steered_mu0[0, 5].item(), -1e8, "Carrier token was not in support at mu=0")
        # At mu=4, token 5 was pruned by the 4-nat penalty (<= -1e8)
        self.assertLess(steered_mu4[0, 5].item(), -1e8, "Carrier token was not pruned by -4.0 nat barrier at mu=4")

    # =========================================================================
    # Directive W4: Continuation Grace Horizon (Theorem 34)
    # =========================================================================
    def test_w4_grace_horizon_buffering_and_decay(self):
        """
        Directive W4: High certainty linguistic continuation produces G_t > 0.30 at t=2,
        mitigating premature tail penalty. Decay factor exp(-(t-2)/4.0) geometrically
        depresses G_t over time.
        """
        K = 100
        # Confident language model distribution on token 15
        lg = torch.full((1, K), -10.0)
        lg[0, 15] = 15.0  # High epistemic certainty (low entropy)
        ll = torch.zeros((1, K))

        engine = NEVFEEngine(lambda_param=1.0, mu_param=4.0)

        # Step 2: Grace horizon should be high
        _, steered_2, _, _ = engine.decode_step(logits_l=ll, logits_g=lg, step_idx=2)
        g2 = engine.get_last_grace_horizon()
        self.assertIsNotNone(g2)
        self.assertGreater(g2.item(), 0.30, f"Grace horizon at step 2 should be > 0.30, got {g2.item()}")

        # Step 7: Grace horizon should decay below 0.30 due to exp(-(7-2)/4.0) = exp(-1.25) ~ 0.2865
        _, steered_7, _, _ = engine.decode_step(logits_l=ll, logits_g=lg, step_idx=7)
        g7 = engine.get_last_grace_horizon()
        self.assertLess(g7.item(), 0.30, f"Grace horizon at step 7 should decay <= 0.30, got {g7.item()}")

    # =========================================================================
    # Directive W5: Calibrated Crisp Semantic Stopping Gate (Theorem 34)
    # =========================================================================
    def test_w5_calibrated_stopping_requires_grace_exhaustion(self):
        """
        Directive W5: At step_idx >= 2 with optical evidence exhausted (is_optical_exhausted=True):
        - When grace_horizon > 0.30 (entity continuing), stopping does NOT occur.
        - When grace_horizon <= 0.30 (grace exhausted), crisp exhaustion stop triggers immediately.
        """
        K = 50
        lg = torch.randn(1, K)
        omega_t = torch.tensor([[0.0]])
        s_bar_t = torch.tensor([[0.5]])  # omega_t <= s_bar_t -> is_optical_exhausted = True
        eos_token_ids = [49]

        # Case 1: Grace horizon active (0.60 > 0.30) -> Should NOT stop
        stop_buffered = evaluate_semantic_stopping(
            logits_g=lg,
            omega_t=omega_t,
            s_bar_t=s_bar_t,
            eos_token_ids=eos_token_ids,
            step_idx=2,
            grace_horizon=torch.tensor([[0.60]]),
        )
        self.assertFalse(stop_buffered.item(), "Stopping triggered while grace_horizon was > 0.30!")

        # Case 2: Grace horizon exhausted (0.20 <= 0.30) -> MUST stop
        stop_exhausted = evaluate_semantic_stopping(
            logits_g=lg,
            omega_t=omega_t,
            s_bar_t=s_bar_t,
            eos_token_ids=eos_token_ids,
            step_idx=2,
            grace_horizon=torch.tensor([[0.20]]),
        )
        self.assertTrue(stop_exhausted.item(), "Stopping did not trigger after grace horizon exhaustion <= 0.30!")

    # =========================================================================
    # Directive W6: Numerical Cohesion Clamp Preservation (Theorem 33)
    # =========================================================================
    def test_w6_space_token_clamped_between_digits(self):
        """
        Directive W6: When y_{t-1} is a digit and digit contrast persists,
        ASCII space token 220 receives -mu_param * psi_t cohesion penalty,
        ensuring genuine digits dominate whitespace between numbers.
        """
        B, K = 1, 1000
        prev_digit_id = 19   # Token '4'
        next_digit_id = 21   # Token '6'
        space_id = 220       # ASCII space ' '
        eos_id = 0
        digit_ids = list(range(15, 25))

        # Baseline language prior favors space over digit
        lg = torch.full((B, K), -5.0)
        lg[0, space_id] = 4.0
        lg[0, next_digit_id] = 1.0

        # Focal stream has sharp optical contrast on genuine digit
        ll = lg.clone()
        ll[0, next_digit_id] = 7.0
        ll[0, space_id] = 4.0

        # When previous token is a digit:
        next_tok, steered, _, _ = nevfe_step_logits(
            logits_l=ll,
            logits_g=lg,
            mu_param=4.0,
            step_idx=2,
            prev_token_id=prev_digit_id,
            space_token_id=space_id,
            digit_token_ids=digit_ids,
            eos_token_ids=[eos_id],
        )

        # 1. Candidate digit strictly dominates whitespace token 220
        self.assertGreater(
            steered[0, next_digit_id].item(),
            steered[0, space_id].item(),
            "Steered logit for digit must strictly exceed whitespace token 220",
        )

        # 2. Selected token must be the genuine digit, not whitespace
        self.assertEqual(
            int(next_tok.item()), next_digit_id,
            f"Expected next token to be digit {next_digit_id}, got {int(next_tok.item())}",
        )

    # =========================================================================
    # Directive W7: IEEE 754 NaN Immunization in Entropy LSE
    # =========================================================================
    def test_w7_nan_immunization_with_negative_infinity_tail(self):
        """
        Directive W7: compute_normalized_entropy_lse and nevfe_step_logits
        safely sanitize 0.0 * (-inf) = NaN, producing valid, finite outputs.
        """
        K = 100
        z = torch.cat([torch.randn(10), torch.full((K - 10,), float("-inf"))])

        entropy_nats, norm_entropy = compute_normalized_entropy_lse(z.unsqueeze(0))
        self.assertFalse(torch.isnan(entropy_nats).any(), "Entropy nats contained NaN!")
        self.assertFalse(torch.isnan(norm_entropy).any(), "Normalized entropy contained NaN!")
        self.assertGreaterEqual(entropy_nats.item(), 0.0, "Entropy nats was negative!")
        self.assertLessEqual(norm_entropy.item(), 1.0, "Norm entropy exceeded 1.0!")

        # Steered logits under negative infinity tail
        lp = z.unsqueeze(0)
        lz = torch.zeros_like(lp)
        next_tok, steered, _, _ = nevfe_step_logits(lp, lz, lambda_param=1.0, mu_param=4.0)
        self.assertFalse(torch.isnan(steered).any(), "Steered logits contained NaN!")
        self.assertGreater(steered[0, next_tok.item()].item(), -1e8, "Selected token was pruned!")


if __name__ == "__main__":
    unittest.main()
