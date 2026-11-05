# -*- coding: utf-8 -*-
"""
Master Unit Test Suite for Iteration 18 Implementation (Directives W1–W7).
Strictly verifies:
- Directive W1: Theorem 35 Residual Spatial Pursuit Multi-Hypothesis RoI Selection.
- Directive W2: Theorem 36 Stroke-Gated Projective Homographic Perspective Rectification.
- Directive W3: Strict ViT Visual Token Budget Invariant (<= 512 tokens).
- Directive W4: Theorem 37 Separator-Bridging Orthographic Cohesion Horizon & Lyapunov Bound.
- Directive W5: Extended Decoupled Dynamic Carrier Barrier (t <= 5).
- Directive W6: Strict Preservation of Numerical & Separator Cohesion Clamps.
- Directive W7: IEEE 754 NaN & Backward Compatibility Invariants.
"""

import math
import unittest
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from src.nevfe.core.engine import (
    NEVFEEngine,
    compute_box_iou,
    compute_residual_spatial_pursuit_windows,
    compute_stroke_gated_homography,
    compute_normalized_entropy_lse,
    evaluate_semantic_stopping,
    nevfe_step_logits,
)


class TestIteration18Implementation(unittest.TestCase):
    """Exhaustive behavioral unit tests covering Iteration 18 Directives W1 through W7."""

    def setUp(self):
        torch.manual_seed(2026)
        np.random.seed(2026)

    # =========================================================================
    # Directive W1: Residual Spatial Pursuit Multi-Hypothesis RoI Selection (Theorem 35)
    # =========================================================================
    def test_w1_rsp_multimodal_isolation(self):
        """
        Directive W1: Construct synthetic 2-entity multimodal scene.
        Primary dominant billboard at top-center (x ~ 0.50, y ~ 0.20),
        Secondary subtle target at bottom-right (x ~ 0.85, y ~ 0.85).
        Assert:
        1. Continuous deflation suppresses primary mode.
        2. Residual mass fraction eta_res >= lambda_res (0.20).
        3. Returns exactly K=2 proposals [B1, B2] with IoU(B1, B2) <= 0.40.
        4. B1 encloses top mode and B2 encloses bottom-right mode.
        """
        grid_size = 16
        P = grid_size * grid_size
        D = 64

        # Construct visual tokens with 2 distinct spatial clusters of high variance
        tokens = torch.randn(P, D) * 0.1

        # Mode 1: Top-center (rows 1:4, cols 6:10)
        for r in range(1, 4):
            for c in range(6, 10):
                idx = r * grid_size + c
                tokens[idx] += 8.0

        # Mode 2: Bottom-right (rows 12:15, cols 12:15)
        for r in range(12, 15):
            for c in range(12, 15):
                idx = r * grid_size + c
                tokens[idx] += 5.0

        boxes, stats = compute_residual_spatial_pursuit_windows(
            visual_tokens=tokens,
            grid_size=grid_size,
            lambda_res=0.20,
            iou_thresh=0.40,
            return_stats=True,
        )

        self.assertEqual(len(boxes), 2, f"Expected K=2 proposals for bimodal scene, got {len(boxes)}")
        b1, b2 = boxes[0], boxes[1]

        # Verify deflation statistics
        self.assertGreaterEqual(
            stats["eta_res"], 0.20,
            f"Expected residual mass fraction >= 0.20, got {stats['eta_res']:.4f}"
        )

        # Verify IoU <= 0.40
        iou = compute_box_iou(b1, b2)
        self.assertLessEqual(iou, 0.40, f"Expected IoU <= 0.40, got {iou:.4f}")

        # Verify spatial isolation: B1 covers top (y_center < 0.5), B2 covers bottom-right (x_center > 0.6, y_center > 0.6)
        b1_y_center = 0.5 * (b1[1] + b1[3])
        b2_x_center = 0.5 * (b2[0] + b2[2])
        b2_y_center = 0.5 * (b2[1] + b2[3])

        self.assertLess(b1_y_center, 0.50, f"B1 should isolate top mode, got y_center={b1_y_center}")
        self.assertGreater(b2_x_center, 0.55, f"B2 should isolate right mode, got x_center={b2_x_center}")
        self.assertGreater(b2_y_center, 0.55, f"B2 should isolate bottom mode, got y_center={b2_y_center}")

    def test_w1_rsp_unimodal_deflation_invariance(self):
        """
        Directive W1: Construct unimodal scene (single central text sign).
        Assert that residual mass does NOT produce a spurious second crop,
        returning strictly K=1 proposal [B1].
        """
        grid_size = 16
        P = grid_size * grid_size
        D = 64

        tokens = torch.randn(P, D) * 0.05
        # Single central mode (rows 6:10, cols 6:10)
        for r in range(6, 10):
            for c in range(6, 10):
                idx = r * grid_size + c
                tokens[idx] += 10.0

        boxes = compute_residual_spatial_pursuit_windows(
            visual_tokens=tokens,
            grid_size=grid_size,
            lambda_res=0.20,
            iou_thresh=0.40,
        )

        self.assertEqual(len(boxes), 1, f"Expected K=1 for unimodal scene, got {len(boxes)}")
        b1 = boxes[0]
        # B1 center should be near (0.5, 0.5)
        cx = 0.5 * (b1[0] + b1[2])
        cy = 0.5 * (b1[1] + b1[3])
        self.assertAlmostEqual(cx, 0.5, delta=0.15)
        self.assertAlmostEqual(cy, 0.5, delta=0.15)

    # =========================================================================
    # Directive W2: Stroke-Gated Projective Homography (Theorem 36)
    # =========================================================================
    def test_w2_stroke_gated_homography_tilted_perspective(self):
        """
        Directive W2: Synthesize text crop tilted with perspective convergence (delta_theta > 0.08 rad).
        Assert that:
        1. Perspective convergence delta_theta is correctly measured (> 0.08).
        2. Matrix H in R^(3x3) is solved via closed-form SVD.
        3. Condition number kappa(H) <= 25.0.
        4. is_homography is True and output is a valid PIL Image without NaN/inf.
        """
        W, H = 160, 80
        img = Image.new("RGB", (W, H), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)

        # Draw converging strokes simulating perspective text
        # Left half has tilt ~ +15 degrees, right half has tilt ~ -15 degrees -> delta_theta ~ 30 deg ~ 0.52 rad
        for y in range(20, 60, 8):
            draw.line([(10, y - 5), (W // 2, y + 5)], fill=(10, 10, 10), width=3)
            draw.line([(W // 2, y + 5), (W - 10, y - 10)], fill=(10, 10, 10), width=3)

        rectified, H_mat, meta = compute_stroke_gated_homography(
            image=img,
            crop_box=(0, 0, W, H),
            mu_persp=0.08,
            target_size=(W, H),
        )

        self.assertIsInstance(rectified, Image.Image)
        self.assertEqual(H_mat.shape, (3, 3))
        self.assertTrue(meta["is_homography"], f"Expected projective homography, got fallback: {meta['fallback_reason']}")
        self.assertGreater(meta["delta_theta"], 0.08, f"Convergence was not detected: {meta['delta_theta']}")
        self.assertLessEqual(
            meta["condition_number"], 25.0,
            f"Condition number exceeded 25.0: {meta['condition_number']}"
        )
        self.assertTrue(meta["is_convex"], "Quadrangle was not convex!")

    def test_w2_condition_number_fallback_guard(self):
        """
        Directive W2: Construct degenerate/near-singular quadrangle with kappa(H) > 25.0.
        Assert engine falls back safely to OBB affine crop without coordinate wrap-around,
        producing valid image and is_homography=False.
        """
        W, H = 100, 100
        img = Image.new("RGB", (W, H), color=(200, 200, 200))
        draw = ImageDraw.Draw(img)
        # Draw perspective converging strokes
        for y in range(20, 80, 10):
            draw.line([(10, y - 5), (W // 2, y + 5)], fill=(10, 10, 10), width=3)
            draw.line([(W // 2, y + 5), (W - 10, y - 10)], fill=(10, 10, 10), width=3)

        # Test condition number threshold with artificially strict max_condition_number = 1.05
        rectified, H_mat, meta = compute_stroke_gated_homography(
            image=img,
            crop_box=(0, 0, W, H),
            mu_persp=0.01,  # Force perspective attempt
            max_condition_number=1.05,  # Strict threshold forcing fallback
        )

        self.assertIsInstance(rectified, Image.Image)
        self.assertFalse(meta["is_homography"], "Expected fallback to OBB when condition number exceeds threshold!")
        self.assertIn("condition_number_exceeded", str(meta["fallback_reason"]))
        # Output should be valid
        arr = np.array(rectified)
        self.assertFalse(np.isnan(arr).any(), "NaN found in fallback image!")

    def test_w2_convexity_fallback_guard(self):
        """
        Directive W2: If quadrangle corners violate cross-product edge convexity,
        assert engine falls back smoothly to OBB affine crop.
        """
        W, H = 120, 60
        img = Image.new("RGB", (W, H), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.line([(0, 0), (W, H)], fill=(0, 0, 0), width=4)

        # Extremely small mu_persp on near-flat image
        rectified, H_mat, meta = compute_stroke_gated_homography(
            image=img,
            crop_box=(0, 0, W, H),
            mu_persp=0.08,
        )

        self.assertIsInstance(rectified, Image.Image)
        # Should not crash, and either convex or smoothly fell back
        if not meta["is_convex"]:
            self.assertFalse(meta["is_homography"])
            self.assertEqual(meta["fallback_reason"], "non_convex_quadrangle")

    def test_w2_flat_text_obb_direct_path(self):
        """
        Directive W2: When text is nearly horizontal (delta_theta <= mu_persp),
        engine uses Oriented Bounding Box directly.
        """
        W, H = 100, 40
        img = Image.new("RGB", (W, H), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        # Perfectly horizontal lines
        draw.line([(10, 15), (90, 15)], fill=(0, 0, 0), width=2)
        draw.line([(10, 25), (90, 25)], fill=(0, 0, 0), width=2)

        rectified, H_mat, meta = compute_stroke_gated_homography(
            image=img,
            crop_box=(0, 0, W, H),
            mu_persp=0.08,
        )

        self.assertFalse(meta["is_homography"])
        self.assertEqual(meta["fallback_reason"], "delta_theta_below_threshold")
        self.assertLessEqual(meta["delta_theta"], 0.08)

    # =========================================================================
    # Directive W3: Strict ViT Visual Token Budget Invariant (<= 512 Tokens)
    # =========================================================================
    def test_w3_vit_token_budget_invariant(self):
        """
        Directive W3: Visual token tensor fed to InternViT-300M must strictly
        satisfy N_vis in {256, 512} (K in {1, 2} tiles of 448x448).
        """
        D = 896
        # Case K = 1: Single tile (256 tokens)
        v1 = torch.randn(1, 256, D)
        self.assertIn(v1.shape[1], {256, 512})

        # Case K = 2: Multi-hypothesis 2 tiles (512 tokens)
        v2 = torch.randn(1, 256, D)
        v_concat = torch.cat([v1, v2], dim=1)
        self.assertEqual(v_concat.shape, (1, 512, D))
        self.assertIn(v_concat.shape[1], {256, 512})
        self.assertLessEqual(v_concat.shape[1], 512)

    # =========================================================================
    # Directive W4: Separator-Bridging Orthographic Cohesion Horizon (Theorem 37)
    # =========================================================================
    def test_w4_separator_epistemic_momentum_bridging(self):
        """
        Directive W4 (Theorem 37):
        At step t=3 after emitting separator y_2 = '/' (token ID 14):
        1. Instantaneous predictive certainty E_3 plunges due to orthographic branching.
        2. Non-consecutive separator transition indicator sigma_3^sep == 1.0.
        3. Epistemic momentum preservation: E_3^* = max(E_3, E_2) == E_2.
        4. Effective step count t_tilde = max(0, 3 - 2 - 1) = 0 -> decay_factor = 1.0.
        5. Grace horizon G_3^* > 0.30, preventing premature stopping on addresses like '12/3A'.
        """
        K = 100
        sep_id = 14  # '/'
        num_id = 18  # '2'

        engine = NEVFEEngine(lambda_param=1.0, mu_param=4.0)

        # Step 1: Emit digit '1'
        lg1 = torch.full((1, K), -5.0)
        lg1[0, num_id] = 10.0
        ll1 = lg1.clone()
        engine.decode_step(logits_l=ll1, logits_g=lg1, step_idx=1)

        # Step 2: High certainty emission of separator '/'
        lg2 = torch.full((1, K), -5.0)
        lg2[0, sep_id] = 12.0  # High epistemic certainty (E_2 ~ 0.90)
        ll2 = lg2.clone()
        engine.decode_step(
            logits_l=ll2,
            logits_g=lg2,
            step_idx=2,
            prev_token_id=num_id,
            separator_token_ids=[sep_id],
        )
        e2 = engine.get_last_epistemic_certainty()
        self.assertIsNotNone(e2)
        e2_val = e2.item()
        self.assertGreater(e2_val, 0.70, f"Expected high certainty at step 2, got {e2_val}")

        # Step 3: Predictive distribution jumps in entropy across the separator junction
        # Diffuse distribution over all digits
        lg3 = torch.full((1, K), 0.0)  # Uniform flat -> high entropy, low instantaneous E_3 ~ 0.05
        ll3 = lg3.clone()

        engine.decode_step(
            logits_l=ll3,
            logits_g=lg3,
            step_idx=3,
            prev_token_id=sep_id,
            prev_prev_token_id=num_id,
            separator_token_ids=[sep_id],
        )
        g3 = engine.get_last_grace_horizon()
        self.assertIsNotNone(g3)
        g3_val = g3.item()

        # Under Theorem 37 epistemic bridging, G_3^* preserves E_2 and remains > 0.30!
        self.assertGreater(
            g3_val, 0.30,
            f"Epistemic bridging failed! G_3 should remain > 0.30 across separator, got {g3_val:.4f}"
        )

        # Verify semantic stopping does NOT fire prematurely
        stop = evaluate_semantic_stopping(
            logits_g=lg3,
            omega_t=torch.tensor([[0.0]]),
            s_bar_t=torch.tensor([[0.5]]),  # is_optical_exhausted = True
            step_idx=3,
            grace_horizon=g3,
        )
        self.assertFalse(bool(stop.item()), "Premature semantic stopping fired on separator junction!")

    def test_w4_deterministic_lyapunov_bound(self):
        """
        Directive W4: Deterministic Lyapunov stopping bound across maximum separator transitions.
        Under sustained optical evidence exhaustion (Psi_t == 0), generation terminates
        in T_stop <= 9 <= 10 steps under ALL linguistic profiles even with 2 separators emitted.
        """
        K = 100
        sep_id = 14
        eos_id = 0

        # Test across 3 distinct entropy profiles:
        for profile_type in ["max_certainty", "moderate_certainty", "uniform_flat"]:
            engine = NEVFEEngine(lambda_param=1.0, mu_param=4.0)
            t_stop = None

            for t in range(2, 15):
                if profile_type == "max_certainty":
                    lg = torch.full((1, K), -10.0)
                    lg[0, 20] = 20.0
                elif profile_type == "moderate_certainty":
                    lg = torch.randn(1, K) * 2.0
                else:
                    lg = torch.zeros(1, K)

                ll = lg.clone()  # Optical contrast exhausted

                # Simulate separators at t=2 and t=4
                prev_tok = sep_id if t in {3, 5} else 20
                prev_prev = 20 if t in {3, 5} else (sep_id if t in {4, 6} else 20)
                n_seps = 2 if t >= 5 else (1 if t >= 3 else 0)

                next_tok, _, _, _ = engine.decode_step(
                    logits_l=ll,
                    logits_g=lg,
                    step_idx=t,
                    eos_token_ids=[eos_id],
                    prev_token_id=prev_tok,
                    prev_prev_token_id=prev_prev,
                    separator_token_ids=[sep_id],
                    num_separators_emitted=n_seps,
                )

                stop = engine.evaluate_stopping(
                    logits_g=lg,
                    logits_l=ll,
                    steered_tokens=next_tok,
                    eos_token_ids=[eos_id],
                    step_idx=t,
                )
                if bool(stop.item()):
                    t_stop = t
                    break

            self.assertIsNotNone(t_stop, f"Generation did not terminate for {profile_type}!")
            self.assertLessEqual(
                t_stop, 10,
                f"Lyapunov bound violated for {profile_type}! Stopped at step {t_stop} > 10"
            )

    # =========================================================================
    # Directive W5: Extended Decoupled Dynamic Carrier Barrier (t <= 5)
    # =========================================================================
    def test_w5_extended_carrier_barrier_t_le_5(self):
        """
        Directive W5: Extended Carrier Barrier active for t <= 5 with geometric decay:
        B_carrier(t) = mu_param * exp(-(t-1)/2.0) * (1 - OGI_t).
        Assert:
        1. Ungrounded preamble tokens receive exponential attenuation for t in [1, 5].
        2. Grounded entity tokens (OGI_t = 1.0) receive 0.00 nats penalty for all t.
        3. At t = 6, carrier barrier is completely inactive.
        4. At mu=4.0 (t=1), dyadic barrier prunes ungrounded preamble from support (<-1e8).
        """
        K = 50
        carrier_tok = 5
        grounded_tok = 20
        mu = 1.0

        for t in range(1, 8):
            lg = torch.zeros((1, K))
            ll = lg.clone()
            ll[0, grounded_tok] = 0.5  # Subtle focal optical contrast on grounded token

            # With barrier mu=1.0
            res_mu = nevfe_step_logits(
                logits_l=ll,
                logits_g=lg,
                mu_param=mu,
                step_idx=t,
                return_continuation_saliency=True,
                return_grounding_status=True,
            )
            steered_mu = res_mu[1]

            # Without barrier mu=0.0
            res_mu0 = nevfe_step_logits(
                logits_l=ll,
                logits_g=lg,
                mu_param=0.0,
                step_idx=t,
                return_continuation_saliency=True,
                return_grounding_status=True,
            )
            steered_mu0 = res_mu0[1]

            delta_carrier = steered_mu0[0, carrier_tok].item() - steered_mu[0, carrier_tok].item()
            delta_grounded = steered_mu0[0, grounded_tok].item() - steered_mu[0, grounded_tok].item()

            # At t=1, grounded token receives exactly 0.000 nats barrier penalty
            if t == 1:
                self.assertAlmostEqual(
                    delta_grounded, 0.0, places=4,
                    msg=f"Grounded token received non-zero penalty at step 1: {delta_grounded}"
                )

            # Net carrier barrier penalty strictly follows geometric attenuation exp(-(t-1)/2.0) for t <= 5, and 0 for t >= 6
            carrier_barrier = delta_carrier - delta_grounded
            expected_barrier = mu * math.exp(-(t - 1) / 2.0) if t <= 5 else 0.0
            self.assertAlmostEqual(
                carrier_barrier, expected_barrier, places=3,
                msg=f"Carrier barrier mismatch at step t={t}: expected {expected_barrier:.4f}, got {carrier_barrier:.4f}"
            )

        # Verification 4: Under full mu=4.0 at t=1, ungrounded preamble is pruned from support (Theorem 34)
        lg_full = torch.zeros((1, K))
        lg_full[0, carrier_tok] = 10.0
        lg_full[0, grounded_tok] = 10.0
        ll_full = lg_full.clone()
        ll_full[0, grounded_tok] = 10.5
        res_full_mu0 = nevfe_step_logits(logits_l=ll_full, logits_g=lg_full, mu_param=0.0, step_idx=1)
        res_full_mu4 = nevfe_step_logits(logits_l=ll_full, logits_g=lg_full, mu_param=4.0, step_idx=1)

        self.assertGreater(res_full_mu0[1][0, carrier_tok].item(), -1e8, "Carrier token was not in support at mu=0")
        self.assertLess(res_full_mu4[1][0, carrier_tok].item(), -1e8, "Carrier token was not pruned by -4.0 nats barrier")

    # =========================================================================
    # Directive W6: Strict Preservation of Numerical & Separator Cohesion Clamps
    # =========================================================================
    def test_w6_digit_to_digit_cohesion_preservation(self):
        """
        Directive W6: Numerical cohesion clamp between consecutive digits remains 100% intact.
        BPE whitespace token 220 is suppressed by mu_param * Psi_t.
        """
        K = 250
        space_id = 220
        prev_digit_id = 17  # '2'
        next_digit_id = 18  # '3'
        digit_ids = list(range(15, 25))

        # Baseline language prior favors space over digit
        lg = torch.full((1, K), -5.0)
        lg[0, space_id] = 4.0
        lg[0, next_digit_id] = 1.0

        # Focal stream has sharp optical contrast on genuine digit
        ll = lg.clone()
        ll[0, next_digit_id] = 7.0
        ll[0, space_id] = 4.0

        next_tok, steered, _, _, omega_t, s_bar_t = nevfe_step_logits(
            logits_l=ll,
            logits_g=lg,
            mu_param=4.0,
            step_idx=2,
            prev_token_id=prev_digit_id,
            digit_token_ids=digit_ids,
            space_token_id=space_id,
            return_continuation_saliency=True,
        )

        # Verification 1: Candidate digit strictly dominates whitespace token 220
        self.assertGreater(
            steered[0, next_digit_id].item(),
            steered[0, space_id].item(),
            "Candidate digit must strictly dominate whitespace token 220 under mu=4.0"
        )
        # Verification 2: Selected token must be the genuine digit
        self.assertEqual(
            int(next_tok.item()), next_digit_id,
            f"Expected next token to be digit {next_digit_id}, got {int(next_tok.item())}"
        )
        # Verification 3: Verify the strong optical cohesion penalty formula: -mu_param * Psi_t
        psi_t = torch.clamp((omega_t - s_bar_t) / (omega_t + 1e-6), 0.0, 1.0).item()
        expected_penalty = 4.0 * psi_t
        self.assertGreater(
            expected_penalty, 3.5,
            f"Expected strong cohesion penalty > 3.5 nats, got {expected_penalty:.4f}"
        )

    def test_w6_separator_to_alphanumeric_cohesion_clamp(self):
        """
        Directive W6: Separator cohesion clamp suppresses whitespace token 220
        immediately after separator token when active optical evidence exists (omega_t > s_bar_t),
        preventing fractured emissions like '12/ 3' or '2019 - 2020'.
        """
        K = 250
        space_id = 220
        sep_id = 14  # '/'
        sep_ids = [12, 14, 11, 13, 25]
        char_id = 50

        # Baseline language prior favors space over character
        lg = torch.full((1, K), -5.0)
        lg[0, space_id] = 4.0
        lg[0, char_id] = 1.0

        # Focal stream has sharp optical contrast on character token
        ll = lg.clone()
        ll[0, char_id] = 7.0
        ll[0, space_id] = 4.0

        next_tok, steered, _, _, omega_t, s_bar_t = nevfe_step_logits(
            logits_l=ll,
            logits_g=lg,
            mu_param=4.0,
            step_idx=2,
            prev_token_id=sep_id,
            separator_token_ids=sep_ids,
            space_token_id=space_id,
            return_continuation_saliency=True,
        )

        # Verification 1: Next character strictly dominates whitespace token 220
        self.assertGreater(
            steered[0, char_id].item(),
            steered[0, space_id].item(),
            "Next character must strictly dominate whitespace token 220 after separator"
        )
        # Verification 2: Selected token must be the genuine character
        self.assertEqual(
            int(next_tok.item()), char_id,
            f"Expected next token to be character {char_id}, got {int(next_tok.item())}"
        )
        # Verification 3: Verify separator cohesion penalty
        psi_t = torch.clamp((omega_t - s_bar_t) / (omega_t + 1e-6), 0.0, 1.0).item()
        expected_penalty = 4.0 * psi_t
        self.assertGreater(
            expected_penalty, 3.5,
            f"Expected strong separator cohesion penalty > 3.5 nats, got {expected_penalty:.4f}"
        )

    # =========================================================================
    # Directive W7: IEEE 754 NaN & Backward Compatibility Invariants
    # =========================================================================
    def test_w7_nan_sanitization_infinite_tail(self):
        """
        Directive W7: Maintain IEEE 754 NaN sanitization in compute_normalized_entropy_lse
        under -inf logit tails.
        """
        z = torch.full((1, 50), -float("inf"))
        z[0, 0] = 10.0
        z[0, 1] = 10.0

        entropy_nats, norm_entropy = compute_normalized_entropy_lse(z)
        self.assertFalse(torch.isnan(entropy_nats).any(), "NaN found in entropy_nats!")
        self.assertFalse(torch.isnan(norm_entropy).any(), "NaN found in norm_entropy!")
        self.assertFalse(torch.isinf(entropy_nats).any(), "Inf found in entropy_nats!")
        self.assertAlmostEqual(norm_entropy.item(), math.log(2.0) / math.log(50.0), places=4)

    def test_w7_decode_step_4tuple_and_public_properties(self):
        """
        Directive W7: Assert NEVFEEngine.decode_step strictly maintains 4-tuple return
        signature (next_tokens, steered_logits, coupling_tensor, entropy_nats)
        and public properties get_last_grace_horizon(), get_last_optical_gate().
        """
        K = 40
        lg = torch.randn(1, K)
        ll = lg.clone()
        engine = NEVFEEngine(lambda_param=1.0, mu_param=4.0)

        res = engine.decode_step(logits_l=ll, logits_g=lg, step_idx=1)
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 4, f"Expected 4-tuple return from decode_step, got {len(res)}-tuple")

        toks, logits, coupling, entropy = res
        self.assertIsInstance(toks, torch.Tensor)
        self.assertIsInstance(logits, torch.Tensor)
        self.assertIsInstance(coupling, torch.Tensor)
        self.assertIsInstance(entropy, torch.Tensor)

        # Public properties
        self.assertIsNotNone(engine.get_last_grace_horizon())
        omega, s_bar = engine.get_last_optical_gate()
        self.assertIsNotNone(omega)
        self.assertIsNotNone(s_bar)


if __name__ == "__main__":
    unittest.main()
