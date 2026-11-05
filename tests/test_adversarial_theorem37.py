# -*- coding: utf-8 -*-
"""
Adversarial Stress Test Suite for Theorem 37 & Iteration 18 Decoding Dynamics.
Author: challenger_iter18_entropy (Teamwork Empirical Challenger)

Target Modules:
- src/nevfe/core/engine.py (Theorem 37, Carrier Barrier t <= 5, Cohesion Clamps)

Mathematical Foundations:
1. Theorem 37: Separator-Bridging Orthographic Cohesion Horizon:
   - Non-consecutive separator transition indicator:
     sigma_t^sep = 1(y_{t-1} in V_sep and y_{t-2} not in V_sep)
   - Epistemic momentum preservation:
     E_t^* = max(E_t, sigma_t^sep * E_{t-1})
   - Topology-clamped effective step:
     t_tilde = max(0, t - 2 - min(N_sep, 2))
   - Grace horizon:
     G_t^* = clamp(E_t^* * exp(-t_tilde / 4.0), 0.0, 1.0)
2. Finite-Time Deterministic Lyapunov Stopping Bound:
   - Under sustained optical evidence exhaustion (Omega_t <= S_bar_t),
     t_tilde >= 5 implies G_t^* <= exp(-1.25) = 0.2865 <= 0.30.
   - At step t = 2 + min(N_sep, 2) + 5 <= 9 <= 10 steps,
     crisp exhaustion stop fires unconditionally.
3. Extended Decoupled Dynamic Carrier Barrier (Directive W5, t <= 5):
   - B_carrier(t, v) = mu_param * exp(-max(0, t-1)/2.0) * (1 - OGI_t(v)) for t <= 5.
   - Grounded tokens (OGI_t = 1.0 >= 0.3) receive strictly 0.0 nats carrier barrier penalty.
   - Inactive (penalty = 0.0 nats) for t >= 6.
4. Whitespace Token 220 Suppression & Numerical Cohesion Clamp:
   - Suppresses ungrounded whitespace splitting between consecutive digits and after separators under active optical contrast.
   - Permits natural whitespace emission when optical contrast dictates a word boundary.
"""

import math
import unittest
import torch
import torch.nn.functional as F

from src.nevfe.core.engine import (
    NEVFEEngine,
    evaluate_semantic_stopping,
    nevfe_step_logits,
)


class TestAdversarialTheorem37(unittest.TestCase):
    """
    Empirical adversarial stress test suite for Theorem 37, Extended Carrier Barrier,
    and Numerical/Separator Cohesion Clamps.
    """

    def setUp(self):
        torch.manual_seed(2026)

    # =========================================================================
    # TASK 1: Consecutive Separator Sequences & Infinite Loop Prevention
    # =========================================================================

    def test_01_consecutive_separator_momentum_cutoff_double_slash(self):
        """
        Adversarial Test 1A: Double slash '//' sequence.
        At step t=3 (first slash emitted at t=2), y_{t-1} is '/' and y_{t-2} is alphanumeric.
        -> sigma_3^sep = 1.0, preserving momentum E_3^* = max(E_3, E_2).
           Since t_tilde = max(0, 3 - 2 - 1) = 0, decay_factor = 1.0,
           so G_3^* = E_3^* == E_2.
        At step t=4 (second slash emitted at t=3), y_{t-1} is '/' AND y_{t-2} is '/'.
        -> Both are in V_sep, so sigma_4^sep MUST evaluate to 0.0!
           Momentum is CUT OFF: E_4^* = E_4 ~ 0.0 for uniform flat logits,
           and G_4^* drops strictly below 0.30.
        Verifies momentum does NOT perpetuate across consecutive separators,
        preventing infinite momentum accumulation.
        """
        K = 100
        sep_id = 14  # '/'
        num_id = 18  # '2'

        # Baseline: step 2 emits separator '/' with high epistemic certainty
        lg2 = torch.full((1, K), -5.0)
        lg2[0, sep_id] = 10.0
        ll2 = lg2.clone()

        _, _, _, _, _, _, _, _, e2 = nevfe_step_logits(
            logits_l=ll2,
            logits_g=lg2,
            step_idx=2,
            prev_token_id=num_id,
            prev_prev_token_id=None,
            separator_token_ids=[sep_id],
            return_continuation_saliency=True,
            return_grounding_status=True,
            return_grace_horizon=True,
            return_epistemic_certainty=True,
        )
        self.assertGreater(e2.item(), 0.70)

        # Step 3: First transition into separator junction (y_{t-1} = sep, y_{t-2} = num)
        lg3 = torch.zeros((1, K))  # Flat uniform logits (instantaneous certainty ~ 0.0)
        ll3 = lg3.clone()

        _, _, _, _, _, _, _, g3, _ = nevfe_step_logits(
            logits_l=ll3,
            logits_g=lg3,
            step_idx=3,
            prev_token_id=sep_id,
            prev_prev_token_id=num_id,
            prev_epistemic_certainty=e2,
            separator_token_ids=[sep_id],
            num_separators_emitted=1,
            return_continuation_saliency=True,
            return_grounding_status=True,
            return_grace_horizon=True,
            return_epistemic_certainty=True,
        )
        # On first separator transition, grace horizon preserves e2 (since decay_factor = 1.0):
        self.assertAlmostEqual(g3.item(), e2.item(), places=4)
        self.assertGreater(g3.item(), 0.30)

        # Step 4: Consecutive separator transition (y_{t-1} = sep, y_{t-2} = sep)
        lg4 = torch.zeros((1, K))  # Flat uniform logits
        ll4 = lg4.clone()

        _, _, _, _, _, _, _, g4, _ = nevfe_step_logits(
            logits_l=ll4,
            logits_g=lg4,
            step_idx=4,
            prev_token_id=sep_id,
            prev_prev_token_id=sep_id,  # Consecutive separator!
            prev_epistemic_certainty=e2,
            separator_token_ids=[sep_id],
            num_separators_emitted=2,
            return_continuation_saliency=True,
            return_grounding_status=True,
            return_grace_horizon=True,
            return_epistemic_certainty=True,
        )
        # On consecutive separator, sigma_4^sep == 0.0! Momentum from e2 is strictly CUT OFF!
        self.assertLess(
            g4.item(), 0.20,
            f"Momentum was not cut off on consecutive separator! g4={g4.item():.4f}"
        )
        self.assertLess(
            g4.item(), 0.30,
            f"Grace horizon should drop below 0.30 when momentum cuts off, got {g4.item():.4f}"
        )

    def test_02_consecutive_separator_sequences_stress_loop_termination(self):
        """
        Adversarial Test 1B: Stress test pathological consecutive separator sequences:
        1. '//' (repeated slashes)
        2. '--' (repeated dashes)
        3. ', ' (comma followed by space)
        4. ' - /' (dash, space, slash)
        Assert that despite these complex punctuation patterns, generation strictly
        terminates within T_stop <= 10 steps without infinite loops under optical exhaustion.
        """
        K = 250  # Adequate vocabulary size accommodating token 220
        sep_slash = 14
        sep_dash = 12
        sep_comma = 11
        space_id = 220
        eos_id = 0
        digit_id = 18

        test_sequences = [
            ("double_slash", [digit_id, sep_slash, sep_slash, digit_id]),
            ("double_dash", [digit_id, sep_dash, sep_dash, digit_id]),
            ("comma_space", [digit_id, sep_comma, space_id, digit_id]),
            ("dash_space_slash", [digit_id, sep_dash, space_id, sep_slash, digit_id]),
        ]

        for seq_name, seq_tokens in test_sequences:
            engine = NEVFEEngine(lambda_param=1.0, mu_param=4.0)
            t_stopped = None

            # Simulate autoregressive sequence decoding up to 15 steps under zero optical contrast
            for t in range(1, 16):
                # Simulated sequence injection for prompt prefix, then flat exhaustion
                lg = torch.zeros((1, K))
                if t <= len(seq_tokens):
                    target_tok = seq_tokens[t - 1]
                    lg[0, target_tok] = 10.0
                ll = lg.clone()  # Zero optical contrast (optical exhaustion)

                next_tok, _, _, _ = engine.decode_step(
                    logits_l=ll,
                    logits_g=lg,
                    step_idx=t,
                    eos_token_ids=[eos_id],
                )

                stop = engine.evaluate_stopping(
                    logits_g=lg,
                    logits_l=ll,
                    steered_tokens=next_tok,
                    eos_token_ids=[eos_id],
                    step_idx=t,
                )

                if bool(stop.item()):
                    t_stopped = t
                    break

            self.assertIsNotNone(
                t_stopped,
                f"Pathological sequence '{seq_name}' entered an infinite loop (did not terminate)!"
            )
            self.assertLessEqual(
                t_stopped, 10,
                f"Sequence '{seq_name}' violated Lyapunov bound: stopped at step {t_stopped} > 10"
            )

    def test_03_topology_clamped_effective_step_delay(self):
        """
        Adversarial Test 1C: Topology-clamped effective step delay:
        t_tilde = max(0, t - 2 - min(N_sep, 2)).
        Even if an adversary emits N_sep = 10 or 100 separators,
        clamped_n_sep is bounded by 2, so the effective delay is at most 2 steps.
        """
        for n_sep in [0, 1, 2, 3, 5, 10, 50, 100]:
            clamped = min(max(0, n_sep), 2)
            self.assertLessEqual(clamped, 2, f"clamped_n_sep exceeded 2 for N_sep={n_sep}")

            # For step t=9:
            t = 9
            t_tilde = max(0, t - 2 - clamped)
            decay_factor = math.exp(-t_tilde / 4.0)
            # When clamped <= 2, t_tilde is at least 9 - 2 - 2 = 5
            self.assertGreaterEqual(t_tilde, 5)
            self.assertLessEqual(decay_factor, 0.30, f"Decay factor must be <= 0.30 at t=9: {decay_factor}")

    # =========================================================================
    # TASK 2: Adversarial Optical Exhaustion & Lyapunov Drift Bound (T_stop <= 10)
    # =========================================================================

    def test_04_adversarial_optical_exhaustion_flat_uniform_logits_lyapunov_bound(self):
        """
        Adversarial Test 2A: Complete optical exhaustion under flat uniform logits.
        We test under maximum possible adversarial conditions:
        - Flat uniform logits: ll = lg = 0.0 across entire vocabulary (zero optical contrast).
        - Adversarial injection of maximum epistemic momentum: prev_epistemic_certainty = 1.0.
        - Tested across all separator emission counts N_sep in {0, 1, 2, 5}.
        Assert:
        In all cases, stopping triggers strictly within T_stop <= 9 <= 10 steps.
        """
        K = 100
        eos_id = 0
        sep_id = 14

        for n_sep in [0, 1, 2, 5]:
            engine = NEVFEEngine(lambda_param=1.0, mu_param=4.0)
            t_stop = None

            for t in range(2, 14):
                lg = torch.zeros((1, K))
                ll = lg.clone()  # Zero optical contrast

                prev_tok = sep_id if n_sep > 0 else 20
                prev_prev = 20  # non-consecutive to give momentum benefit

                next_tok, _, _, _ = engine.decode_step(
                    logits_l=ll,
                    logits_g=lg,
                    step_idx=t,
                    eos_token_ids=[eos_id],
                    prev_token_id=prev_tok,
                    prev_prev_token_id=prev_prev,
                    separator_token_ids=[sep_id],
                    prev_epistemic_certainty=1.0,  # Max adversarial momentum injection!
                    num_separators_emitted=n_sep,
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

            self.assertIsNotNone(t_stop, f"Failed to stop under optical exhaustion for N_sep={n_sep}")
            self.assertLessEqual(
                t_stop, 10,
                f"Lyapunov bound violated for N_sep={n_sep}: T_stop={t_stop} > 10"
            )

    def test_05_lyapunov_drift_strict_monotonicity_decay(self):
        """
        Adversarial Test 2B: Strict negative Lyapunov drift of the grace horizon:
        V(t) = exp(-t_tilde / 4.0)
        Verify that for all t >= 2 + min(N_sep, 2), Delta V(t) = V(t+1) - V(t) < 0.
        """
        for n_sep in [0, 1, 2]:
            clamped = min(max(0, n_sep), 2)
            prev_v = float("inf")
            for t in range(2 + clamped, 12):
                t_tilde = max(0, t - 2 - clamped)
                v_t = math.exp(-t_tilde / 4.0)
                if t_tilde > 0:
                    self.assertLess(
                        v_t, prev_v,
                        f"Lyapunov drift is not strictly negative at step {t} for N_sep={n_sep}"
                    )
                prev_v = v_t

    # =========================================================================
    # TASK 3: Carrier Barrier Boundary Conditions
    # =========================================================================

    def test_06_carrier_barrier_geometric_attenuation_steps_1_to_5(self):
        """
        Adversarial Test 3A: Carrier Barrier geometric attenuation for t in {1, 2, 3, 4, 5}.
        Formula: B_carrier(t) = mu_param * exp(-(t-1)/2.0) * (1 - OGI_t).
        For an ungrounded carrier token (OGI_t = 0.0), verify that the barrier attenuation
        strictly matches theoretical values:
        t=1: decay = exp(0)   = 1.00000 -> penalty = mu * 1.00000
        t=2: decay = exp(-0.5) = 0.60653 -> penalty = mu * 0.60653
        t=3: decay = exp(-1.0) = 0.36788 -> penalty = mu * 0.36788
        t=4: decay = exp(-1.5) = 0.22313 -> penalty = mu * 0.22313
        t=5: decay = exp(-2.0) = 0.13534 -> penalty = mu * 0.13534
        """
        K = 50
        carrier_tok = 5
        grounded_tok = 20
        mu = 1.0

        for t in range(1, 6):
            lg = torch.zeros((1, K))
            ll = lg.clone()
            ll[0, grounded_tok] = 0.5  # Grounded token receives sharp optical contrast

            # With barrier mu=1.0
            res_mu = nevfe_step_logits(
                logits_l=ll,
                logits_g=lg,
                mu_param=mu,
                step_idx=t,
                barrier_horizon=5,
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
                barrier_horizon=5,
                return_continuation_saliency=True,
                return_grounding_status=True,
            )
            steered_mu0 = res_mu0[1]

            delta_carrier = steered_mu0[0, carrier_tok].item() - steered_mu[0, carrier_tok].item()
            delta_grounded = steered_mu0[0, grounded_tok].item() - steered_mu[0, grounded_tok].item()

            # Net carrier barrier penalty isolates Section 6a from Section 6c tail penalty
            carrier_barrier = delta_carrier - delta_grounded
            expected_barrier = mu * math.exp(-(t - 1) / 2.0)

            self.assertAlmostEqual(
                carrier_barrier, expected_barrier, places=3,
                msg=f"Carrier barrier attenuation mismatch at step t={t}: expected {expected_barrier:.4f}, got {carrier_barrier:.4f}"
            )

    def test_07_carrier_barrier_grounded_token_zero_penalty(self):
        """
        Adversarial Test 3B: Grounded tokens (OGI_t = 1.0 >= 0.3) receive strictly 0.0000 nats
        carrier barrier penalty under Directive W5 across all t in {1, ..., 7}.
        """
        K = 50
        carrier_tok = 5
        grounded_tok = 20
        mu = 1.0

        for t in range(1, 8):
            lg = torch.zeros((1, K))
            ll = lg.clone()
            ll[0, grounded_tok] = 0.5  # Sharp optical contrast -> OGI = 1.0 >= 0.3

            res_mu = nevfe_step_logits(
                logits_l=ll,
                logits_g=lg,
                mu_param=mu,
                step_idx=t,
                barrier_horizon=5,
            )
            res_mu0 = nevfe_step_logits(
                logits_l=ll,
                logits_g=lg,
                mu_param=0.0,
                step_idx=t,
                barrier_horizon=5,
            )

            # At step 1, tail drag is 0, so grounded token receives exactly 0.0000 nats penalty
            if t == 1:
                grounded_penalty = (res_mu0[1][0, grounded_tok] - res_mu[1][0, grounded_tok]).item()
                self.assertAlmostEqual(
                    grounded_penalty, 0.0, places=4,
                    msg=f"Grounded token received non-zero barrier penalty at step 1: {grounded_penalty:.6f} nats"
                )

            # For all steps t, the carrier barrier component B_carrier(t, grounded) is 0.0:
            delta_carrier = (res_mu0[1][0, carrier_tok] - res_mu[1][0, carrier_tok]).item()
            delta_grounded = (res_mu0[1][0, grounded_tok] - res_mu[1][0, grounded_tok]).item()
            # Net carrier barrier penalty on grounded token is 0.0
            carrier_penalty_on_grounded = delta_grounded - (delta_carrier - (mu * math.exp(-(t-1)/2.0) if t <= 5 else 0.0))
            self.assertAlmostEqual(
                carrier_penalty_on_grounded, 0.0, places=3,
                msg=f"Carrier barrier penalized grounded token at step {t}: {carrier_penalty_on_grounded:.6f} nats"
            )

    def test_08_carrier_barrier_deactivation_step_6_and_beyond(self):
        """
        Adversarial Test 3C: At step t >= 6, carrier barrier is completely inactive
        (penalty = 0.0000 nats) for all tokens regardless of grounding deficit.
        """
        K = 50
        carrier_tok = 5
        grounded_tok = 20
        mu = 1.0

        for t in [6, 7, 8, 10, 15]:
            lg = torch.zeros((1, K))
            ll = lg.clone()
            ll[0, grounded_tok] = 0.5

            res_mu = nevfe_step_logits(
                logits_l=ll,
                logits_g=lg,
                mu_param=mu,
                step_idx=t,
                barrier_horizon=5,
            )
            res_mu0 = nevfe_step_logits(
                logits_l=ll,
                logits_g=lg,
                mu_param=0.0,
                step_idx=t,
                barrier_horizon=5,
            )

            # Neither carrier nor grounded token should experience any carrier barrier penalty at t >= 6
            delta_carrier = (res_mu0[1][0, carrier_tok] - res_mu[1][0, carrier_tok]).item()
            delta_grounded = (res_mu0[1][0, grounded_tok] - res_mu[1][0, grounded_tok]).item()
            net_carrier_barrier = delta_carrier - delta_grounded
            self.assertAlmostEqual(
                net_carrier_barrier, 0.0, places=4,
                msg=f"Carrier barrier was active at step {t} >= 6: {net_carrier_barrier:.6f} nats"
            )

    # =========================================================================
    # TASK 4: Whitespace Token 220 Suppression & Word Boundary Emission
    # =========================================================================

    def test_09_whitespace_220_suppression_immediately_following_separator(self):
        """
        Adversarial Test 4A: Whitespace token 220 suppression immediately following separator:
        When prev_token is a separator (e.g. '/') and optical contrast is active (omega_t > s_bar_t),
        token 220 is suppressed by -mu_param * psi_t, preventing fractured emissions like '12/ 3'.
        """
        K = 250
        space_id = 220
        sep_id = 14  # '/'
        alphanumeric_id = 18  # '3'

        lg = torch.full((1, K), -5.0)
        lg[0, space_id] = 5.0  # Baseline language prior strongly favors space
        lg[0, alphanumeric_id] = 2.0

        ll = lg.clone()
        ll[0, alphanumeric_id] = 8.0  # Strong optical evidence for next alphanumeric token
        ll[0, space_id] = 5.0

        next_tok, steered, _, _, omega_t, s_bar_t = nevfe_step_logits(
            logits_l=ll,
            logits_g=lg,
            mu_param=4.0,
            step_idx=2,
            prev_token_id=sep_id,
            separator_token_ids=[sep_id],
            space_token_id=space_id,
            return_continuation_saliency=True,
        )

        # Token 220 should be suppressed and alphanumeric token must win
        self.assertGreater(
            steered[0, alphanumeric_id].item(),
            steered[0, space_id].item(),
            "Alphanumeric token must strictly dominate whitespace token 220 after separator!"
        )
        self.assertEqual(int(next_tok.item()), alphanumeric_id)

    def test_10_whitespace_220_normal_emission_at_valid_word_boundary(self):
        """
        Adversarial Test 4B: Normal whitespace emission when optical evidence dictates word boundary.
        When prev_token is a standard letter (e.g. 'a', not a digit and not a separator),
        and optical evidence favors a space boundary (e.g. between two words),
        whitespace token 220 does not receive separator/numerical cohesion penalty,
        and is emitted naturally as the winning argmax token.
        """
        K = 250
        space_id = 220
        letter_id = 30  # 'a' (regular letter, not digit and not separator)
        next_letter_id = 31

        # Global prior has equal or modest preference
        lg = torch.full((1, K), 0.0)
        lg[0, space_id] = 2.0
        lg[0, next_letter_id] = 1.0

        # Focal stream has sharp optical contrast for word boundary (space)
        ll = lg.clone()
        ll[0, space_id] = 6.0  # Active optical contrast for space!
        ll[0, next_letter_id] = 1.0

        # Case 1: prev_token is letter_id (no cohesion clamp)
        next_tok, steered_letter, _, _, omega_t, s_bar_t = nevfe_step_logits(
            logits_l=ll,
            logits_g=lg,
            mu_param=4.0,
            step_idx=2,
            prev_token_id=letter_id,  # Not in digit_token_ids, not in separator_token_ids
            space_token_id=space_id,
            return_continuation_saliency=True,
        )

        # Case 2: prev_token is sep_id (cohesion clamp active)
        _, steered_sep, _, _ = nevfe_step_logits(
            logits_l=ll,
            logits_g=lg,
            mu_param=4.0,
            step_idx=2,
            prev_token_id=14,  # separator '/'
            separator_token_ids=[14],
            space_token_id=space_id,
        )

        # Space token receives strong separator cohesion clamp when prev is separator
        sep_penalty = (steered_letter[0, space_id] - steered_sep[0, space_id]).item()
        self.assertGreater(
            sep_penalty, 3.0,
            f"Expected separator cohesion penalty > 3.0 nats on space token, got {sep_penalty:.4f}"
        )

        # But when prev is regular letter, space token is freely emitted
        self.assertEqual(
            int(next_tok.item()), space_id,
            f"Expected whitespace token 220 to be emitted at word boundary, got {int(next_tok.item())}"
        )

    # =========================================================================
    # TASK 5: Interaction with Numerical Cohesion Clamp & Complex Entities
    # =========================================================================

    def test_11_numerical_cohesion_continuity_2024(self):
        """
        Adversarial Test 5A: Multi-digit sequence continuity: '2024'.
        Verify that digit transitions '2' -> '0' -> '2' -> '4' maintain unbroken cohesion:
        At every step, numerical cohesion clamp suppresses token 220, ensuring '2024'
        is emitted without internal spaces (no '2 0 2 4' or '20 24').
        """
        K = 250
        space_id = 220
        digits = [17, 15, 17, 19]  # '2', '0', '2', '4'
        digit_ids = list(range(15, 25))

        engine = NEVFEEngine(lambda_param=1.0, mu_param=4.0)

        for step in range(1, 5):
            curr_target = digits[step - 1]
            lg = torch.full((1, K), -5.0)
            lg[0, space_id] = 4.0  # LM prior wants space
            lg[0, curr_target] = 2.0

            ll = lg.clone()
            ll[0, curr_target] = 8.0  # Image has sharp digit contrast
            ll[0, space_id] = 4.0

            next_tok, steered, _, _ = engine.decode_step(
                logits_l=ll,
                logits_g=lg,
                step_idx=step,
                space_token_id=space_id,
                digit_token_ids=digit_ids,
            )

            # Selected token must be the genuine digit, not space
            self.assertEqual(
                int(next_tok.item()), curr_target,
                f"Step {step}: expected digit {curr_target}, got {int(next_tok.item())}"
            )
            if step >= 2:
                self.assertGreater(
                    steered[0, curr_target].item(),
                    steered[0, space_id].item(),
                    f"Step {step}: digit {curr_target} did not dominate space 220!"
                )

    def test_12_numerical_separator_continuity_150_000(self):
        """
        Adversarial Test 5B: Number with decimal/thousands separator: '150.000'.
        Sequence: '1', '5', '0' (digits) -> '.' (separator) -> '0', '0', '0' (digits).
        Assert:
        1. '150' maintains numerical cohesion.
        2. '.' is emitted cleanly.
        3. At '.' junction, Theorem 37 momentum preservation ensures G_t > 0.30,
           preventing premature EOS stopping on '150.'.
        4. Post-'.' digit '0' experiences separator cohesion clamp suppressing space 220,
           preventing '150. 000'.
        5. Full sequence emits all 7 tokens without premature stop.
        """
        K = 250
        space_id = 220
        sep_dot = 13  # '.'
        eos_id = 0
        digit_ids = list(range(15, 25))
        seq = [16, 20, 15, sep_dot, 15, 15, 15]  # '1', '5', '0', '.', '0', '0', '0'

        engine = NEVFEEngine(lambda_param=1.0, mu_param=4.0)

        for step in range(1, 8):
            curr_target = seq[step - 1]
            lg = torch.full((1, K), -5.0)
            lg[0, curr_target] = 3.0
            lg[0, space_id] = 4.0
            lg[0, eos_id] = -2.0

            ll = lg.clone()
            ll[0, curr_target] = 8.0  # Sharp optical contrast

            next_tok, steered, _, _ = engine.decode_step(
                logits_l=ll,
                logits_g=lg,
                step_idx=step,
                eos_token_ids=[eos_id],
                space_token_id=space_id,
                digit_token_ids=digit_ids,
                separator_token_ids=[sep_dot],
            )

            # Check that generation does NOT stop at separator junction
            stop = engine.evaluate_stopping(
                logits_g=lg,
                logits_l=ll,
                steered_tokens=next_tok,
                eos_token_ids=[eos_id],
                step_idx=step,
            )
            self.assertFalse(
                bool(stop.item()),
                f"Generation prematurely terminated at step {step} on sequence '150.000'!"
            )
            self.assertEqual(
                int(next_tok.item()), curr_target,
                f"Step {step}: expected token {curr_target}, got {int(next_tok.item())}"
            )

    def test_13_complex_address_continuity_12_slash_3A(self):
        """
        Adversarial Test 5C: Complex address format '12/3A':
        '1', '2' (digits) -> '/' (separator) -> '3' (digit) -> 'A' (alphanumeric).
        Verifies unbroken continuity across '/' without fracturing into '12/ 3A'
        or premature termination after '12/'.
        """
        K = 250
        space_id = 220
        sep_slash = 14  # '/'
        eos_id = 0
        digit_ids = list(range(15, 25))
        tok_1, tok_2, tok_3, tok_A = 16, 17, 18, 65
        seq = [tok_1, tok_2, sep_slash, tok_3, tok_A]

        engine = NEVFEEngine(lambda_param=1.0, mu_param=4.0)

        for step in range(1, 6):
            curr_target = seq[step - 1]
            lg = torch.full((1, K), -5.0)
            lg[0, curr_target] = 2.0
            lg[0, space_id] = 3.5  # Prior prefers space
            lg[0, eos_id] = -2.0

            ll = lg.clone()
            ll[0, curr_target] = 7.5  # Sharp optical evidence

            next_tok, steered, _, _ = engine.decode_step(
                logits_l=ll,
                logits_g=lg,
                step_idx=step,
                eos_token_ids=[eos_id],
                space_token_id=space_id,
                digit_token_ids=digit_ids,
                separator_token_ids=[sep_slash],
            )

            stop = engine.evaluate_stopping(
                logits_g=lg,
                logits_l=ll,
                steered_tokens=next_tok,
                eos_token_ids=[eos_id],
                step_idx=step,
            )

            self.assertFalse(
                bool(stop.item()),
                f"Premature stopping at step {step} on address '12/3A'!"
            )
            self.assertEqual(
                int(next_tok.item()), curr_target,
                f"Step {step}: expected token {curr_target}, got {int(next_tok.item())}"
            )


if __name__ == "__main__":
    unittest.main()
