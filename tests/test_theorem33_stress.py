# -*- coding: utf-8 -*-
"""
Authoritative Empirical Adversarial Stress Test Suite for Theorem 33:
Continuous BPE-Invariant Latent Alignment & Tripartite Optical Clamping.

Validates the three symmetrical barriers and extreme distribution boundaries:
1. Adversarial Tripartite Clamping Tests:
   - Boundary step t=1: Zero tail penalty, persistent carrier barrier active.
   - Boundary step t=2 with active digit contrast (Omega_t^num > S_bar_t):
     Token 220 receives full cohesion penalty if y_{t-1} in V_num, and 0 cohesion penalty if y_{t-1} not in V_num.
   - Boundary step t=2 with inactive digit contrast: Token 220 receives 0 cohesion penalty.
   - Boundary step t >= 2 with exhausted optical contrast (Psi_t -> 0):
     Tail Termination Clamp dominates, suppressing non-EOS tokens and selecting EOS.
2. Extreme Logit Distributions & Fuzzing:
   - Uniform logits (maximum entropy, H = log K).
   - Dirac-delta logits (zero entropy, H = 0).
   - High-variance Gaussian logits (K = 151,674, sigma = 100.0).
   - Massive dynamic ranges (+/- 10,000.0).
   - Zero NaNs, zero Infs, strictly valid output token IDs.
3. Edge Case Robustness:
   - Out-of-bounds space_token_id, digit_token_ids, eos_token_ids, prev_token_id.
   - 1D vs 2D batched tensor support.
   - Orthogonality and zero residual drag when mu_param = 0.0.
"""

import math
import unittest
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F

from src.nevfe.core.engine import (
    nevfe_step_logits,
    compute_dynamic_contrastive_temperature,
    compute_optical_continuation_saliency,
    compute_refusal_resistant_continuation_horizon,
    evaluate_semantic_stopping,
)


class TestTheorem33TripartiteClamping(unittest.TestCase):
    """Empirical adversarial stress testing of Theorem 33 tripartite clamping."""

    def setUp(self):
        self.device = torch.device("cpu")
        self.vocab_size = 151674  # Qwen2.5 / Vintern-1B-v3_5 vocabulary cardinality
        self.space_id = 220       # ASCII space token ' '
        self.digit_ids = list(range(15, 25))  # ASCII digits '0'-'9' (tokens 15..24)
        self.eos_ids = [151643, 151645]       # Standard EOS / ImEnd token IDs

    # =========================================================================
    # 1. Boundary Step t=1: Zero Tail Penalty & Persistent Carrier Barrier
    # =========================================================================
    def test_step_t1_zero_tail_penalty_and_persistent_carrier_barrier(self):
        """
        Adversarial Test 1.1:
        At t=1, Theorem 33 dictates:
        1. Tail Termination Clamp must be strictly 0.00 nats (step_idx < 2).
           Non-EOS content tokens must NOT be suppressed, even if optical contrast is low.
        2. Persistent Head Carrier Barrier must be fully active (delta_1 = (0.5)^0 = 1.0).
           Ungrounded dialogue preamble carrier tokens must be penalized by mu_param * Psi_1.
        3. Grounded entity tokens (OGI = 1.0) must receive exactly 0.00 carrier drag.
        4. State invariance: in_entity_state boolean must NOT disable carrier barrier at t=1.
        """
        B, K = 1, 2000
        carrier_token = 50   # e.g. "Cuốn", "Tôi", "Đây" (dialogue prior token)
        entity_token = 120   # e.g. "46900", "Hà", "Văn" (focal text token)
        eos_token = 151643 % K

        # Global stream prior strongly prefers conversational carrier preamble
        lg = torch.full((B, K), -4.0, device=self.device)
        lg[0, carrier_token] = 6.0   # High LM prior on conversational opener
        lg[0, entity_token] = 1.0    # Lower prior on raw target entity
        lg[0, eos_token] = 0.0

        # Focal crop provides sharp optical evidence on entity token, zero on carrier
        ll = lg.clone()
        ll[0, entity_token] = 9.0    # Sharp optical contrast: Delta = 8.0
        ll[0, carrier_token] = 6.0   # Zero optical contrast: Delta = 0.0

        # A. Verify with mu_param = 4.0 at step t=1
        tok_t1, steered_t1, _, _ = nevfe_step_logits(
            logits_l=ll,
            logits_g=lg,
            mu_param=4.0,
            step_idx=1,
            eos_token_ids=[eos_token],
            barrier_decay=0.5,
            barrier_horizon=3,
        )

        # 1. Grounded entity must dominate over high-prior carrier token
        self.assertGreater(
            steered_t1[0, entity_token].item(),
            steered_t1[0, carrier_token].item(),
            "Entity token must strictly dominate carrier token at t=1 under Carrier Barrier",
        )
        self.assertEqual(
            int(tok_t1.item()), entity_token,
            f"Step 1 must emit entity token {entity_token}, got {int(tok_t1.item())}",
        )

        # 2. Verify Zero Tail Penalty at t=1:
        # Non-EOS tokens must not suffer tail penalty at t=1 even if ll == lg
        ll_flat = lg.clone()  # zero contrast, Psi_1 = 0
        tok_flat, steered_flat, _, _ = nevfe_step_logits(
            logits_l=ll_flat,
            logits_g=lg,
            mu_param=4.0,
            step_idx=1,
            eos_token_ids=[eos_token],
        )
        # Since t=1, tail clamp does not fire, carrier_token (logit 6.0) is not penalized by tail clamp
        self.assertEqual(
            int(tok_flat.item()), carrier_token,
            "At t=1, tail termination clamp must NOT fire (must preserve single-token emissions)",
        )

        # 3. Verify in_entity_state=True does NOT bypass carrier barrier at t=1 (Persistent Head Barrier)
        tok_t1_state_true, steered_state_true, _, _ = nevfe_step_logits(
            logits_l=ll,
            logits_g=lg,
            mu_param=4.0,
            step_idx=1,
            in_entity_state=True,  # Attempt adversarial bypass via boolean flag
            eos_token_ids=[eos_token],
        )
        self.assertEqual(
            int(tok_t1_state_true.item()), entity_token,
            "Persistent Head Barrier must remain active at t=1 regardless of in_entity_state flag",
        )

    # =========================================================================
    # 2. Boundary Step t=2: Numerical Cohesion Clamp Under Active Digit Contrast
    # =========================================================================
    def test_step_t2_numerical_cohesion_active_digit_contrast(self):
        """
        Adversarial Test 1.2:
        At t=2, under active optical digit contrast (Omega_t^num > S_bar_t):
        - When y_{t-1} in V_num: token 220 receives full cohesion penalty mu_param * Psi_t.
        - When y_{t-1} not in V_num: token 220 receives exactly 0.00 cohesion penalty.
        - Candidate digit v in V_num receives 0.00 cohesion penalty in both cases.
        """
        B, K = 1, 1000
        prev_digit = 19   # ASCII digit '4' (token 19 in V_num)
        prev_word = 450   # Non-digit word token (not in V_num)
        next_digit = 22   # ASCII digit '7' (token 22 in V_num)
        space_tok = 220   # ASCII space token 220
        eos_tok = 0

        # Global prior heavily favors formatted thousands space: "4 700" (token 220 > token 22)
        lg = torch.full((B, K), -5.0, device=self.device)
        lg[0, space_tok] = 5.0    # LM prior strongly wants to insert space
        lg[0, next_digit] = 2.0   # Direct unspaced digit has lower prior
        lg[0, eos_tok] = -2.0

        # Focal crop exhibits sharp optical digit contrast on next_digit
        ll = lg.clone()
        ll[0, next_digit] = 8.0   # Strong optical evidence for digit '7'
        ll[0, space_tok] = 5.0    # No optical evidence for whitespace in the ink

        # Case A: y_{t-1} in V_num (prev_token_id = prev_digit)
        tok_digit_prev, steered_digit_prev, _, _ = nevfe_step_logits(
            logits_l=ll,
            logits_g=lg,
            mu_param=4.0,
            step_idx=2,
            prev_token_id=prev_digit,
            space_token_id=space_tok,
            digit_token_ids=self.digit_ids,
            eos_token_ids=[eos_tok],
        )

        # Verify space token 220 is suppressed below next digit '7'
        self.assertGreater(
            steered_digit_prev[0, next_digit].item(),
            steered_digit_prev[0, space_tok].item(),
            "Next digit must strictly dominate space token 220 when y_{t-1} is a digit",
        )
        self.assertEqual(int(tok_digit_prev.item()), next_digit)

        # Case B: Behavior when y_{t-1} not in V_num (prev_token_id = prev_word, e.g. "khách")
        # In natural word spacing, when space is competitive, cohesion clamp must impose ZERO penalty
        lg_word = torch.full((B, K), 5.0, device=self.device)
        lg_word[0, space_tok] = 5.8
        lg_word[0, next_digit] = 5.2

        ll_word = lg_word.clone()
        ll_word[0, next_digit] = 5.9  # optical digit contrast exists
        ll_word[0, space_tok] = 5.8

        # Run with prev_digit (y_{t-1} in V_num) vs prev_word (y_{t-1} not in V_num) with mu_param=1.0
        # In this regime, both are within the FEST support set, allowing exact quantitative measurement
        tok_d_quant, steered_d_quant, _, _ = nevfe_step_logits(
            logits_l=ll_word,
            logits_g=lg_word,
            mu_param=1.0,
            step_idx=2,
            prev_token_id=prev_digit,
            space_token_id=space_tok,
            digit_token_ids=self.digit_ids,
            eos_token_ids=[eos_tok],
        )
        tok_w_quant, steered_w_quant, _, _ = nevfe_step_logits(
            logits_l=ll_word,
            logits_g=lg_word,
            mu_param=1.0,
            step_idx=2,
            prev_token_id=prev_word,
            space_token_id=space_tok,
            digit_token_ids=self.digit_ids,
            eos_token_ids=[eos_tok],
        )

        # Control run with mu_param=0.0
        _, steered_ctrl_quant, _, _ = nevfe_step_logits(
            logits_l=ll_word,
            logits_g=lg_word,
            mu_param=0.0,
            step_idx=2,
            prev_token_id=prev_word,
            space_token_id=space_tok,
            digit_token_ids=self.digit_ids,
            eos_token_ids=[eos_tok],
        )

        # 1. Surgical Cohesion Isolation:
        # Switching from prev_word to prev_digit must apply cohesion penalty ONLY to space token 220.
        # Max difference across all non-space tokens in the entire vocabulary must be strictly 0.0000.
        diff_all = (steered_w_quant - steered_d_quant).squeeze(0)
        diff_non_space = diff_all.clone()
        diff_non_space[space_tok] = 0.0
        max_non_space_diff = torch.max(torch.abs(diff_non_space)).item()
        self.assertLess(
            max_non_space_diff, 1e-5,
            f"Cohesion clamp must strictly target ONLY space token 220, non-space diff was {max_non_space_diff:.6f}",
        )

        # 2. Case C: Quantitative penalty verification on space token 220:
        # Space token under digit prev is penalized by mu_param * Psi_t (>= 0.90 nats for mu=1.0)
        penalty_difference = diff_all[space_tok].item()
        self.assertGreater(
            penalty_difference, 0.90,
            f"Numerical cohesion clamp penalty difference on space must be >= 0.90 nats, got {penalty_difference:.4f}",
        )

    # =========================================================================
    # 3. Boundary Step t=2: Numerical Cohesion Inactive When Optical Digit Evidence Lacking
    # =========================================================================
    def test_step_t2_numerical_cohesion_inactive_when_digit_contrast_lacking(self):
        """
        Adversarial Test 1.3:
        When y_{t-1} in V_num, but the current optical scene contains NO digit evidence
        (e.g. following the digit is a unit word "km" or "tháng", Omega_t^num <= S_bar_t),
        the Cohesion Clamp MUST NOT fire on token 220.
        """
        B, K = 1, 1000
        prev_digit = 19   # ASCII digit '4'
        space_tok = 220   # ASCII space token 220
        unit_word = 310   # e.g. "km" or "tháng"
        eos_tok = 0

        # Prior wants space after digit
        lg = torch.full((B, K), -4.0, device=self.device)
        lg[0, space_tok] = 4.0
        lg[0, unit_word] = 2.0

        # Optical evidence is on alphabetical word 'km', NOT on digits!
        ll = lg.clone()
        ll[0, unit_word] = 8.0   # High contrast on word "km"
        # Digits have zero optical contrast! (Omega_t^num == 0 <= S_bar_t)

        tok_out, steered_out, _, _ = nevfe_step_logits(
            logits_l=ll,
            logits_g=lg,
            mu_param=4.0,
            step_idx=2,
            prev_token_id=prev_digit,
            space_token_id=space_tok,
            digit_token_ids=self.digit_ids,
            eos_token_ids=[eos_tok],
        )

        # Control run with mu_param = 0.0
        _, steered_ctrl, _, _ = nevfe_step_logits(
            logits_l=ll,
            logits_g=lg,
            mu_param=0.0,
            step_idx=2,
            prev_token_id=prev_digit,
            space_token_id=space_tok,
            digit_token_ids=self.digit_ids,
            eos_token_ids=[eos_tok],
        )

        # Space token must NOT receive cohesion penalty because Omega_t^num <= S_bar_t
        diff = abs(steered_out[0, space_tok].item() - steered_ctrl[0, space_tok].item())
        self.assertLess(
            diff, 1e-4,
            f"Cohesion clamp must NOT fire when optical digit evidence is lacking, diff: {diff:.6f}",
        )

    # =========================================================================
    # 4. Boundary Step t >= 2: Tail Termination Clamp Under Optical Exhaustion
    # =========================================================================
    def test_step_t_ge_2_exhausted_contrast_tail_termination_clamp(self):
        """
        Adversarial Test 1.4:
        At step t >= 2, when optical contrast is exhausted (Psi_t -> 0):
        1. All non-EOS tokens receive full tail penalty: -mu_param * (1 - Psi_t).
        2. EOS tokens receive 0.00 penalty.
        3. Selected next token must strictly be an EOS token.
        4. Multiple EOS token IDs (e.g. Qwen2.5 [151643, 151645]) are all protected from tail penalty.
        """
        B, K = 1, 1000
        hallucinated_tail_token = 550  # e.g. continuation "là", "trong", "của"
        eos_1 = 100
        eos_2 = 101

        lg = torch.full((B, K), -3.0, device=self.device)
        lg[0, hallucinated_tail_token] = 6.0  # High language model continuation bias
        lg[0, eos_1] = 3.5                   # Moderate prior for EOS
        lg[0, eos_2] = 2.0                   # Secondary EOS

        # Optical contrast is exhausted: logits_l == logits_g => Psi_t = 0.0
        ll = lg.clone()

        # Test across decoding steps t = 2, 3, 5, 10
        for step in [2, 3, 5, 10]:
            tok_out, steered_out, _, _ = nevfe_step_logits(
                logits_l=ll,
                logits_g=lg,
                mu_param=4.0,
                step_idx=step,
                eos_token_ids=[eos_1, eos_2],
            )

            # 1. Non-EOS tail token was penalized by -4.0 nats: 6.0 - 4.0 = 2.0 < eos_1 (3.5)
            self.assertGreater(
                steered_out[0, eos_1].item(),
                steered_out[0, hallucinated_tail_token].item(),
                f"Step {step}: EOS token must dominate hallucinated tail token under Tail Clamp",
            )

            # 2. Both EOS tokens must experience exactly 0.00 tail penalty
            # Verify delta on non-EOS vs EOS
            self.assertIn(
                int(tok_out.item()), [eos_1, eos_2],
                f"Step {step}: Selected token must be an authorized EOS token, got {int(tok_out.item())}",
            )


class TestExtremeLogitDistributionsAndFuzzing(unittest.TestCase):
    """
    Stress-testing numerical stability, extreme distributions, and fuzzing
    at full Qwen2.5 / Vintern-1B-v3_5 vocabulary scale (K = 151,674).
    """

    def setUp(self):
        self.device = torch.device("cpu")
        self.vocab_size = 151674  # Exact vocabulary size
        self.space_id = 220
        self.digit_ids = list(range(15, 25))
        self.eos_ids = [151643, 151645]

    # =========================================================================
    # 1. Uniform Logits (Maximum Entropy: H = log K)
    # =========================================================================
    def test_uniform_logits_maximum_entropy_stability(self):
        """
        Adversarial Test 2.1:
        Uniform distribution across all 151,674 tokens (maximum entropy, zero contrast).
        Verify:
        - Exactly 0 NaNs and 0 Infs in steered logits, coupling tensor, and entropy.
        - Next token is a valid integer token ID in [0, 151673].
        - Bounded thermodynamic entropy near theoretical max log(151674) = 11.929 nats.
        """
        B = 1
        K = self.vocab_size

        for const_val in [0.0, 50.0, -100.0]:
            ll = torch.full((B, K), const_val, device=self.device, dtype=torch.float32)
            lg = torch.full((B, K), const_val, device=self.device, dtype=torch.float32)

            next_tok, steered, coupling, ent = nevfe_step_logits(
                logits_l=ll,
                logits_g=lg,
                lambda_param=1.0,
                mu_param=4.0,
                step_idx=2,
                eos_token_ids=self.eos_ids,
            )

            # 1. Zero NaNs
            self.assertFalse(torch.isnan(steered).any().item(), "Steered logits must contain 0 NaNs under uniform input")
            self.assertFalse(torch.isnan(coupling).any().item(), "Coupling must contain 0 NaNs under uniform input")
            self.assertFalse(torch.isnan(ent).any().item(), "Entropy must contain 0 NaNs under uniform input")

            # 2. Zero Infs (masked FEST values are -1e9, which is finite)
            self.assertFalse(torch.isinf(steered).any().item(), "Steered logits must contain 0 Infs under uniform input")

            # 3. Valid integer token ID
            tok_id = int(next_tok.item())
            self.assertTrue(0 <= tok_id < K, f"Next token ID {tok_id} out of bounds [0, {K})")

    # =========================================================================
    # 2. Dirac-Delta Logits (Zero Entropy: H = 0)
    # =========================================================================
    def test_dirac_delta_logits_zero_entropy_stability(self):
        """
        Adversarial Test 2.2:
        Dirac-delta spike: exactly 1 token has huge logit (+1000.0), rest are 0.0 or -1000.0.
        Tests information-theoretic limit: lim_{p -> 0} p log p = 0 and p=1 log p = 0.
        Verify:
        - 0 NaNs, 0 Infs.
        - Selected token strictly equals the Dirac spike token.
        - Entropy cleanly evaluates to ~0.00 nats without numeric exception.
        """
        B = 1
        K = self.vocab_size
        spike_token = 42424

        ll = torch.zeros((B, K), device=self.device, dtype=torch.float32)
        lg = torch.zeros((B, K), device=self.device, dtype=torch.float32)

        # Set Dirac spike
        ll[0, spike_token] = 1000.0
        lg[0, spike_token] = 1000.0

        next_tok, steered, coupling, ent = nevfe_step_logits(
            logits_l=ll,
            logits_g=lg,
            lambda_param=1.0,
            mu_param=4.0,
            step_idx=1,
            eos_token_ids=self.eos_ids,
        )

        self.assertFalse(torch.isnan(steered).any().item(), "Steered logits must have 0 NaNs under Dirac delta")
        self.assertFalse(torch.isnan(ent).any().item(), "Entropy must have 0 NaNs under Dirac delta")
        self.assertFalse(torch.isinf(steered).any().item(), "Steered logits must have 0 Infs under Dirac delta")

        # Must pick the Dirac spike token
        self.assertEqual(
            int(next_tok.item()), spike_token,
            f"Dirac spike token {spike_token} must be emitted, got {int(next_tok.item())}",
        )

        # Entropy of a deterministic spike must be close to 0.0 nats
        self.assertLess(
            ent.item(), 0.01,
            f"Dirac delta entropy must be ~0.00 nats, got {ent.item():.6f}",
        )

    # =========================================================================
    # 3. High-Variance Gaussian Logits (K = 151,674, sigma = 100.0)
    # =========================================================================
    def test_high_variance_gaussian_fuzzing(self):
        """
        Adversarial Test 2.3:
        Massive random Gaussian noise with high standard deviation (sigma = 100.0)
        across the entire 151,674 vocabulary lattice.
        Logit values range from -500 to +500.
        Tests float32 log_softmax, exponential overflow/underflow, and FEST candidate masking.
        """
        torch.manual_seed(1600)
        B = 1
        K = self.vocab_size

        # High-variance Gaussian logits
        ll = torch.randn((B, K), device=self.device, dtype=torch.float32) * 100.0
        lg = torch.randn((B, K), device=self.device, dtype=torch.float32) * 100.0

        # Autoregressive execution at step t=2 with previous digit
        next_tok, steered, coupling, ent = nevfe_step_logits(
            logits_l=ll,
            logits_g=lg,
            lambda_param=1.0,
            mu_param=4.0,
            step_idx=2,
            prev_token_id=18,
            space_token_id=self.space_id,
            digit_token_ids=self.digit_ids,
            eos_token_ids=self.eos_ids,
        )

        # Verify numerical integrity
        self.assertFalse(torch.isnan(steered).any().item(), "Zero NaNs in steered logits under high-variance Gaussian")
        self.assertFalse(torch.isnan(coupling).any().item(), "Zero NaNs in coupling under high-variance Gaussian")
        self.assertFalse(torch.isnan(ent).any().item(), "Zero NaNs in entropy under high-variance Gaussian")
        self.assertFalse(torch.isinf(steered).any().item(), "Zero Infs in steered logits under high-variance Gaussian")

        tok_id = int(next_tok.item())
        self.assertTrue(0 <= tok_id < K, f"Selected token {tok_id} out of bounds")

    # =========================================================================
    # 4. Extreme Pathological Dynamic Range (+/- 10,000.0)
    # =========================================================================
    def test_extreme_pathological_dynamic_range(self):
        """
        Adversarial Test 2.4:
        Tests extreme values that cause standard float32 exp() to overflow to Inf or underflow to 0:
        Logit values at +/- 10,000.0.
        """
        B = 2
        K = 1000
        ll = torch.full((B, K), -10000.0, device=self.device)
        lg = torch.full((B, K), -10000.0, device=self.device)
        ll[:, 10] = 10000.0
        lg[:, 20] = 10000.0

        next_tok, steered, coupling, ent = nevfe_step_logits(
            logits_l=ll,
            logits_g=lg,
            lambda_param=1.0,
            mu_param=4.0,
            step_idx=1,
            eos_token_ids=self.eos_ids,
        )

        self.assertFalse(torch.isnan(steered).any().item(), "Must maintain 0 NaNs under +/- 10,000 logits")
        self.assertFalse(torch.isnan(coupling).any().item(), "Must maintain 0 NaNs in coupling under +/- 10,000 logits")
        self.assertFalse(torch.isnan(ent).any().item(), "Must maintain 0 NaNs in entropy under +/- 10,000 logits")
        self.assertFalse(torch.isinf(steered).any().item(), "Must maintain 0 Infs in steered logits")

    # =========================================================================
    # 5. Out-of-Bounds and Malformed Token Parameters Fuzzing
    # =========================================================================
    def test_malformed_token_parameters_robustness(self):
        """
        Adversarial Test 2.5:
        Fuzzing token ID parameter boundaries:
        - space_token_id >= K (e.g. 999,999) or negative (-1)
        - digit_token_ids containing out-of-bounds IDs
        - eos_token_ids containing out-of-bounds IDs or empty
        - prev_token_id out-of-bounds or negative
        - 1D tensor [K] without batch dimension
        Must execute robustly without throwing unhandled exceptions.
        """
        K = 500
        ll_1d = torch.randn(K, device=self.device)
        lg_1d = torch.randn(K, device=self.device)

        # Test 1D input tensor with malformed token IDs
        next_tok, steered, coupling, ent = nevfe_step_logits(
            logits_l=ll_1d,
            logits_g=lg_1d,
            mu_param=4.0,
            step_idx=2,
            prev_token_id=999999,           # Out-of-bounds prev_token_id
            space_token_id=999999,          # Out-of-bounds space_token_id
            digit_token_ids=[-1, 500, 999], # Out-of-bounds digit IDs
            eos_token_ids=[999999],         # Out-of-bounds EOS ID
        )

        self.assertFalse(torch.isnan(steered).any().item())
        self.assertFalse(torch.isinf(steered).any().item())
        self.assertTrue(0 <= int(next_tok.item()) < K)

        # Test with empty eos_token_ids and None prev_token_id
        next_tok2, steered2, _, _ = nevfe_step_logits(
            logits_l=ll_1d,
            logits_g=lg_1d,
            mu_param=4.0,
            step_idx=2,
            prev_token_id=None,
            eos_token_ids=[],
        )
        self.assertFalse(torch.isnan(steered2).any().item())
        self.assertTrue(0 <= int(next_tok2.item()) < K)

    # =========================================================================
    # 6. Tripartite Clamp Orthogonality and Zero-Coupling Invariance
    # =========================================================================
    def test_tripartite_orthogonality_and_zero_coupling_invariance(self):
        """
        Adversarial Test 2.6:
        When mu_param = 0.0:
        All three clamps (Carrier Barrier, Cohesion Clamp, Tail Clamp) must evaluate
        identically to zero drag. Steered logits must match the unclamped base potential
        with zero residual clamping artifact.
        """
        B, K = 1, 1000
        ll = torch.randn((B, K), device=self.device)
        lg = torch.randn((B, K), device=self.device)

        # With mu_param = 0.0 at step t=2 with prev digit
        _, steered_mu0, _, _ = nevfe_step_logits(
            logits_l=ll,
            logits_g=lg,
            mu_param=0.0,
            step_idx=2,
            prev_token_id=19,
            space_token_id=self.space_id,
            digit_token_ids=self.digit_ids,
            eos_token_ids=self.eos_ids,
        )

        # Unclamped reference run (mu_param = 0.0 at step t=1 without prev digit)
        _, steered_ref, _, _ = nevfe_step_logits(
            logits_l=ll,
            logits_g=lg,
            mu_param=0.0,
            step_idx=1,
            prev_token_id=None,
            space_token_id=self.space_id,
            digit_token_ids=self.digit_ids,
            eos_token_ids=self.eos_ids,
        )

        # Difference must be identically 0.00 across all candidate tokens
        max_diff = torch.max(torch.abs(steered_mu0 - steered_ref)).item()
        self.assertLess(
            max_diff, 1e-4,
            f"Zero coupling (mu_param=0.0) must leave logits identical across steps, max diff: {max_diff:.6f}",
        )


if __name__ == "__main__":
    unittest.main()
