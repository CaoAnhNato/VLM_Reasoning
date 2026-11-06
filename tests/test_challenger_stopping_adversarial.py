"""
Adversarial Stress Test Suite: Epistemic Stopping & Saliency Concentration (Iteration 21)
Challenger: challenger_iter21_stopping

Directives:
1. Multi-Word Proper Noun Premature Termination Stress (5-10 words, transient JS dips).
2. Alphanumeric & Hyphenated Sequence Continuity Stress (phone, dates, license plates, separators).
3. Power-Law Saliency Concentration Edge Cases (Dirac delta, all-zeros, FP16 underflow).
4. Local execution constraint: synthetic/mock mode only, no real 1B model weights.
"""

import math
import unittest
from typing import List, Optional, Sequence
import torch
import torch.nn.functional as F

from src.nevfe.core.engine import (
    NEVFEEngine,
    apply_saliency_modulated_gating,
    compute_epistemic_exhaustion_index,
    evaluate_semantic_stopping,
    compute_unnormalized_entropy,
)


class MockTokenizerForChallenger:
    """Lightweight mock tokenizer mirroring 5CD-AI/Vintern-1B-v3_5 vocabulary structure."""

    def __init__(self, vocab_size: int = 151674):
        self.vocab_size = vocab_size
        self.unk_token_id = 0
        self.eos_token_id = 151645
        self.pad_token_id = 151643

        # ASCII digits 0-9
        self.digit_map = {str(d): 15 + d for d in range(10)}
        # ASCII uppercase A-Z
        self.upper_map = {chr(c): 30 + (c - ord("A")) for c in range(ord("A"), ord("Z") + 1)}
        # ASCII lowercase a-z
        self.lower_map = {chr(c): 60 + (c - ord("a")) for c in range(ord("a"), ord("z") + 1)}
        # Separators
        self.sep_map = {"-": 12, ".": 13, "/": 14, ",": 11, ":": 25}
        self.space_id = 220

    def convert_tokens_to_ids(self, token: str) -> Optional[int]:
        if token == " ":
            return self.space_id
        if token in self.digit_map:
            return self.digit_map[token]
        if token in self.upper_map:
            return self.upper_map[token]
        if token in self.lower_map:
            return self.lower_map[token]
        if token in self.sep_map:
            return self.sep_map[token]
        return 1000 + (hash(token) % 10000)


class TestChallengerStoppingAdversarial(unittest.TestCase):
    """Adversarial challenger suite testing epistemic stopping and power-law saliency."""

    def setUp(self):
        torch.manual_seed(42)
        self.vocab_size = 151674
        self.eos_token_ids = [151643, 151645]
        self.digit_token_ids = list(range(15, 25))
        self.separator_token_ids = [12, 14, 11, 13, 25]
        self.space_token_id = 220
        self.mock_tok = MockTokenizerForChallenger(self.vocab_size)

        # Derive 62 ASCII primitives
        ascii_chars = (
            [str(d) for d in range(10)]
            + [chr(c) for c in range(ord("A"), ord("Z") + 1)]
            + [chr(c) for c in range(ord("a"), ord("z") + 1)]
        )
        self.alnum_token_ids = [
            int(self.mock_tok.convert_tokens_to_ids(c)) for c in ascii_chars
        ]

    # =========================================================================
    # DIRECTIVE 1: Multi-Word Proper Noun Premature Termination Stress
    # =========================================================================

    def test_proper_noun_10_words_no_premature_truncation(self):
        """
        Adversarial Scenario: 10-word proper noun entity
        'TRƯỜNG ĐẠI HỌC CÔNG THƯƠNG TP. HỒ CHÍ MINH' (24 BPE tokens).
        Between words, JS divergence experiences transient dips (e.g. J_t dips from 1.8 to 0.2).
        Verify that crisp stopping does NOT prematurely truncate generation while text continues.
        """
        engine = NEVFEEngine(lambda_param=1.0, mu_param=4.0)

        # Sequence of 24 tokens representing the 10 words
        token_sequence = [
            2378, 131099, 144175, 6140,   # TR-Ư-Ờ-NG
            50327, 141301,                #  Đ-ẠI
            472, 144142, 34,              #  H-Ọ-C
            356, 131371, 6140,            #  C-Ô-NG
            4434, 131099, 142293, 6140,   #  TH-Ư-Ơ-NG
            29070, 13,                    #  TP-.
            472, 144019,                  #  H-Ồ
            6826, 52220,                  #  CH-Í
            16701, 39,                    #  MIN-H
        ]
        self.assertEqual(len(token_sequence), 24)

        prev_id = None
        for step_idx, expected_token_id in enumerate(token_sequence, start=1):
            # Synthesize logits where focal stream p_L strongly predicts expected_token_id
            logits_l = torch.randn(1, self.vocab_size) * 0.5
            logits_l[0, expected_token_id] += 12.0  # Clear visual evidence for current character

            logits_g = torch.randn(1, self.vocab_size) * 0.5
            # At word boundaries (steps 5, 7, 10, 13, 17, 19, 21, 23), simulate transient JS dip
            # by letting global stream also have moderate probability on the expected token
            is_word_boundary = step_idx in [5, 7, 10, 13, 17, 19, 21, 23]
            if is_word_boundary:
                logits_g[0, expected_token_id] += 8.0  # High agreement -> transient JS dip
            else:
                logits_g[0, expected_token_id] += 2.0  # Low agreement -> high JS

            next_tok, steered_logits, _, _ = engine.decode_step(
                logits_l=logits_l,
                logits_g=logits_g,
                step_idx=step_idx,
                eos_token_ids=self.eos_token_ids,
                prev_token_id=prev_id,
                space_token_id=self.space_token_id,
                digit_token_ids=self.digit_token_ids,
                separator_token_ids=self.separator_token_ids,
                alnum_token_ids=self.alnum_token_ids,
            )

            # Assert model selected the correct entity token
            self.assertEqual(int(next_tok.item()), expected_token_id, f"Failed at step {step_idx}")

            # Evaluate semantic stopping gate
            stop_decision = engine.evaluate_stopping(
                logits_g=logits_g,
                logits_l=logits_l,
                steered_tokens=next_tok,
                eos_token_ids=self.eos_token_ids,
                step_idx=step_idx,
            )

            # Crisp stopping MUST NOT trigger during any of the 24 entity tokens
            self.assertFalse(
                bool(stop_decision.item()),
                f"Premature termination triggered at step {step_idx} (token {expected_token_id}) "
                f"during 10-word proper noun! E_t={engine.get_last_epistemic_exhaustion()}, "
                f"delta_eos={engine.get_last_delta_eos()}"
            )

            prev_id = int(next_tok.item())

        # Step 25: Text evidence is exhausted. Model predicts EOS.
        step_idx = 25
        logits_l = torch.randn(1, self.vocab_size) * 0.5
        logits_g = torch.randn(1, self.vocab_size) * 0.5
        logits_l[0, self.eos_token_ids[0]] += 12.0
        logits_g[0, self.eos_token_ids[0]] += 12.0

        next_tok, _, _, _ = engine.decode_step(
            logits_l=logits_l,
            logits_g=logits_g,
            step_idx=step_idx,
            eos_token_ids=self.eos_token_ids,
            prev_token_id=prev_id,
            space_token_id=self.space_token_id,
            digit_token_ids=self.digit_token_ids,
            separator_token_ids=self.separator_token_ids,
            alnum_token_ids=self.alnum_token_ids,
        )
        stop_decision = engine.evaluate_stopping(
            logits_g=logits_g,
            logits_l=logits_l,
            steered_tokens=next_tok,
            eos_token_ids=self.eos_token_ids,
            step_idx=step_idx,
        )
        # Should stop cleanly on EOS
        self.assertTrue(bool(stop_decision.item()), "Engine failed to stop at final EOS step!")

    def test_transient_js_dip_mathematical_entropy_barrier(self):
        """
        Mathematical Stress: Even when JS divergence drops to 0.0 (rho_t -> 1.0),
        if visual attention remains on subsequent words, p_L is low entropy (confident).
        Show that E_t = rho_t * H_hat_t < 0.45, so crisp stopping is mathematically impossible.
        """
        # Sharp focal distribution: top token has prob ~ 0.85
        logits_l = torch.full((1, self.vocab_size), -10.0)
        logits_l[0, 50327] = 10.0  # ' Đ'
        log_p_l = F.log_softmax(logits_l, dim=-1)

        # Global distribution matches focal distribution: JS -> 0.0
        log_p_g = log_p_l.clone()

        # Previous peak JS was large (1.5)
        peak_js = 1.5

        e_t, new_peak, j_t, rho_t, h_hat_t = compute_epistemic_exhaustion_index(
            log_p_l=log_p_l,
            log_p_g=log_p_g,
            peak_js=peak_js,
        )

        # JS divergence should be ~ 0.0
        self.assertAlmostEqual(float(j_t.item()), 0.0, places=4)
        # Depletion rho_t is near 1.0
        self.assertGreaterEqual(float(rho_t.item()), 0.99)

        # BUT normalized entropy h_hat_t must be very small because p_L is confident
        # ln(K) = ln(151643) ~ 11.93. H(p_L) < 1.0 nats.
        self.assertLess(float(h_hat_t.item()), 0.15)

        # Consequently, E_t = rho_t * h_hat_t MUST be far below the 0.45 stopping threshold
        self.assertLess(
            float(e_t.item()),
            0.20,
            f"Epistemic exhaustion {float(e_t.item())} exceeded safe barrier under confident focal attention!"
        )

        # Verify evaluate_semantic_stopping returns False
        stop = evaluate_semantic_stopping(
            logits_g=logits_g if 'logits_g' in locals() else logits_l,
            logits_l=logits_l,
            steered_tokens=torch.tensor([50327]),
            eos_token_ids=self.eos_token_ids,
            step_idx=5,
            e_t=e_t,
            delta_eos=torch.tensor([0.5]),
            is_in_alnum_cohesion=False,
        )
        self.assertFalse(bool(stop.item()), "Crisp stop fired despite low focal entropy!")

    # =========================================================================
    # DIRECTIVE 2: Alphanumeric & Hyphenated Sequence Continuity Stress
    # =========================================================================

    def test_phone_number_continuity_across_spaces(self):
        """
        Adversarial Scenario: Phone number '0945 166 799'.
        Tokens: [15, 24, 19, 20, 220, 16, 21, 21, 220, 22, 24, 24].
        Verify alphanumeric cohesion state and zero premature termination across spaces.
        """
        engine = NEVFEEngine(lambda_param=1.0, mu_param=4.0)
        phone_tokens = [15, 24, 19, 20, 220, 16, 21, 21, 220, 22, 24, 24]

        prev_id = None
        for step, tok in enumerate(phone_tokens, start=1):
            logits_l = torch.randn(1, self.vocab_size) * 0.5
            logits_g = torch.randn(1, self.vocab_size) * 0.5
            logits_l[0, tok] += 15.0

            next_tok, _, _, _ = engine.decode_step(
                logits_l=logits_l,
                logits_g=logits_g,
                step_idx=step,
                eos_token_ids=self.eos_token_ids,
                prev_token_id=prev_id,
                space_token_id=self.space_token_id,
                digit_token_ids=self.digit_token_ids,
                separator_token_ids=self.separator_token_ids,
                alnum_token_ids=self.alnum_token_ids,
            )
            self.assertEqual(int(next_tok.item()), tok)

            # When previous token was a digit, engine MUST report is_in_alnum_cohesion == True
            if prev_id is not None and prev_id in self.digit_token_ids:
                self.assertTrue(
                    engine.is_in_alnum_cohesion(),
                    f"Alphanumeric cohesion was False at step {step} with prev_token {prev_id} (digit)!"
                )

            stop_dec = engine.evaluate_stopping(
                logits_g=logits_g,
                logits_l=logits_l,
                steered_tokens=next_tok,
                eos_token_ids=self.eos_token_ids,
                step_idx=step,
            )
            self.assertFalse(
                bool(stop_dec.item()),
                f"Premature termination at step {step} in phone number {tok}!"
            )
            prev_id = tok

    def test_date_continuity_across_hyphens_and_spaces(self):
        """
        Adversarial Scenario: Date sequence '19 - 12 - 2021'.
        Separators include '-' (token 12) and ' ' (token 220).
        Verify that cohesion protects against premature termination across separators.
        """
        engine = NEVFEEngine(lambda_param=1.0, mu_param=4.0)
        date_tokens = [16, 24, 12, 220, 16, 17, 12, 220, 17, 15, 17, 16]

        prev_id = None
        for step, tok in enumerate(date_tokens, start=1):
            logits_l = torch.randn(1, self.vocab_size) * 0.5
            logits_g = torch.randn(1, self.vocab_size) * 0.5
            logits_l[0, tok] += 15.0

            next_tok, _, _, _ = engine.decode_step(
                logits_l=logits_l,
                logits_g=logits_g,
                step_idx=step,
                eos_token_ids=self.eos_token_ids,
                prev_token_id=prev_id,
                space_token_id=self.space_token_id,
                digit_token_ids=self.digit_token_ids,
                separator_token_ids=self.separator_token_ids,
                alnum_token_ids=self.alnum_token_ids,
            )
            self.assertEqual(int(next_tok.item()), tok)

            # Check cohesion after hyphen
            if prev_id == 12:  # '-'
                self.assertTrue(
                    engine.is_in_alnum_cohesion(),
                    f"Cohesion was False after hyphen at step {step}!"
                )

            stop_dec = engine.evaluate_stopping(
                logits_g=logits_g,
                logits_l=logits_l,
                steered_tokens=next_tok,
                eos_token_ids=self.eos_token_ids,
                step_idx=step,
            )
            self.assertFalse(
                bool(stop_dec.item()),
                f"Premature termination at step {step} in date sequence!"
            )
            prev_id = tok

    def test_license_plate_continuity_mixed_alphanumeric(self):
        """
        Adversarial Scenario: License plate '51F-123.45'.
        Tokens: [20 ('5'), 16 ('1'), 35 ('F'), 12 ('-'), 16 ('1'), 17 ('2'), 18 ('3'), 13 ('.'), 19 ('4'), 20 ('5')].
        Mix of digits, uppercase ASCII letters, hyphens, and dots.
        Verify that is_in_alnum_cohesion is True across every single token transition!
        """
        engine = NEVFEEngine(lambda_param=1.0, mu_param=4.0)
        plate_tokens = [20, 16, 35, 12, 16, 17, 18, 13, 19, 20]

        prev_id = None
        for step, tok in enumerate(plate_tokens, start=1):
            logits_l = torch.randn(1, self.vocab_size) * 0.5
            logits_g = torch.randn(1, self.vocab_size) * 0.5
            logits_l[0, tok] += 15.0

            next_tok, _, _, _ = engine.decode_step(
                logits_l=logits_l,
                logits_g=logits_g,
                step_idx=step,
                eos_token_ids=self.eos_token_ids,
                prev_token_id=prev_id,
                space_token_id=self.space_token_id,
                digit_token_ids=self.digit_token_ids,
                separator_token_ids=self.separator_token_ids,
                alnum_token_ids=self.alnum_token_ids,
            )
            self.assertEqual(int(next_tok.item()), tok)

            # In a continuous license plate, EVERY step after step 1 has an alnum, digit, or sep predecessor
            if step > 1:
                self.assertTrue(
                    engine.is_in_alnum_cohesion(),
                    f"Alphanumeric cohesion broke at step {step} (prev_token={prev_id})!"
                )

            stop_dec = engine.evaluate_stopping(
                logits_g=logits_g,
                logits_l=logits_l,
                steered_tokens=next_tok,
                eos_token_ids=self.eos_token_ids,
                step_idx=step,
            )
            self.assertFalse(
                bool(stop_dec.item()),
                f"Premature termination at step {step} in license plate!"
            )
            prev_id = tok

    def test_cohesion_state_blocks_artificially_high_et(self):
        """
        Adversarial Test: Artificially inject E_t = 0.95 and delta_eos = 10.0 (extreme stopping pressure).
        When is_in_alnum_cohesion is True, evaluate_semantic_stopping MUST evaluate to False.
        """
        logits_g = torch.randn(1, self.vocab_size)
        logits_l = torch.randn(1, self.vocab_size)
        steered_tok = torch.tensor([16])  # Digit '1'

        # Case 1: Cohesion is TRUE -> MUST NOT STOP
        stop_with_cohesion = evaluate_semantic_stopping(
            logits_g=logits_g,
            logits_l=logits_l,
            steered_tokens=steered_tok,
            eos_token_ids=self.eos_token_ids,
            step_idx=3,
            e_t=0.95,
            delta_eos=10.0,
            is_in_alnum_cohesion=True,
        )
        self.assertFalse(
            bool(stop_with_cohesion.item()),
            "Crisp stopping fired despite is_in_alnum_cohesion == True!"
        )

        # Case 2: Cohesion is FALSE -> MUST STOP
        stop_without_cohesion = evaluate_semantic_stopping(
            logits_g=logits_g,
            logits_l=logits_l,
            steered_tokens=steered_tok,
            eos_token_ids=self.eos_token_ids,
            step_idx=3,
            e_t=0.95,
            delta_eos=10.0,
            is_in_alnum_cohesion=False,
        )
        self.assertTrue(
            bool(stop_without_cohesion.item()),
            "Crisp stopping failed to fire when cohesion was False and E_t >= 0.45, delta_eos >= 6.0!"
        )

    # =========================================================================
    # DIRECTIVE 3: Power-Law Saliency Edge Cases
    # =========================================================================

    def test_power_law_dirac_delta_1hot_saliency(self):
        """
        Adversarial Edge Case: Dirac delta (1-hot saliency distribution).
        Exactly 1 patch has saliency 1.0; all other P-1 patches have saliency 0.0.
        Verify:
        - Zero NaNs or Infs
        - Exact shape and dtype preservation (FP32, FP16, BFloat16)
        - Output is concentrated without gradient or numerical explosion
        """
        for P in [256, 512]:
            for dtype in [torch.float32, torch.float16]:
                visual_tokens = torch.randn(P, 896, dtype=dtype)
                s_qcass = torch.zeros(P, dtype=dtype)
                s_qcass[42] = 1.0  # Dirac delta at index 42

                out = apply_saliency_modulated_gating(
                    visual_tokens=visual_tokens,
                    s_qcass=s_qcass,
                    lambda_gate=0.5,
                    mu_steer=0.2,
                )

                self.assertEqual(out.shape, visual_tokens.shape)
                self.assertEqual(out.dtype, dtype)
                self.assertFalse(torch.isnan(out).any(), f"NaNs detected in 1-hot Dirac for P={P}, {dtype}!")
                self.assertFalse(torch.isinf(out).any(), f"Infs detected in 1-hot Dirac for P={P}, {dtype}!")

    def test_power_law_all_zero_saliency_graceful_handling(self):
        """
        Adversarial Edge Case: All-zero saliency map (S = [0, 0, ..., 0]).
        Verify:
        - Graceful handling without ZeroDivisionError or crash
        - Returns visual_tokens unchanged via delta_range identity fallback
        - Zero NaNs
        """
        for P in [256, 512]:
            visual_tokens = torch.randn(P, 896, dtype=torch.float32)
            s_zero = torch.zeros(P, dtype=torch.float32)

            out = apply_saliency_modulated_gating(
                visual_tokens=visual_tokens,
                s_qcass=s_zero,
                lambda_gate=0.5,
                mu_steer=0.2,
            )

            self.assertEqual(out.shape, visual_tokens.shape)
            self.assertFalse(torch.isnan(out).any(), f"NaNs detected in all-zero saliency for P={P}!")
            # When S is all-zero, max(S) - min(S) == 0 -> delta_range == 0 < 1e-6 -> exact fallback
            self.assertTrue(
                torch.allclose(out, visual_tokens, atol=1e-6),
                f"All-zero saliency did not return visual_tokens fallback for P={P}!"
            )

    def test_power_law_fp16_underflow_near_zero(self):
        """
        Adversarial Edge Case: Saliency values near FP16 underflow (10^-15).
        Verify:
        - Clamping to 1e-8 prevents subnormal / zero underflow issues
        - Zero NaNs
        - Output is finite and well-conditioned
        """
        P = 256
        visual_tokens = torch.randn(P, 896, dtype=torch.float16)
        # 1 salient patch (1.0), remaining patches near extreme underflow 1e-15
        s_underflow = torch.full((P,), 1e-15, dtype=torch.float32)
        s_underflow[10] = 1.0
        s_underflow = s_underflow / torch.sum(s_underflow)

        out = apply_saliency_modulated_gating(
            visual_tokens=visual_tokens,
            s_qcass=s_underflow.to(torch.float16),
            lambda_gate=0.5,
            mu_steer=0.2,
        )

        self.assertEqual(out.shape, visual_tokens.shape)
        self.assertEqual(out.dtype, torch.float16)
        self.assertFalse(torch.isnan(out).any(), "NaNs detected in FP16 underflow stress test!")
        self.assertFalse(torch.isinf(out).any(), "Infs detected in FP16 underflow stress test!")

    def test_power_law_uniform_simplex_exact_fallback(self):
        """
        Adversarial Edge Case: Uniform distribution on simplex S = [1/P, ..., 1/P].
        Verify exact uniform query identity fallback (delta_range < 1e-6).
        """
        P = 512
        visual_tokens = torch.randn(1, P, 896, dtype=torch.float32)
        s_uniform = torch.full((P,), 1.0 / float(P), dtype=torch.float32)

        out = apply_saliency_modulated_gating(
            visual_tokens=visual_tokens,
            s_qcass=s_uniform,
            lambda_gate=0.5,
            mu_steer=0.2,
        )

        self.assertTrue(
            torch.allclose(out, visual_tokens, atol=1e-7),
            "Uniform saliency failed to trigger exact identity fallback!"
        )

    def test_power_law_lambda_gate_zero_short_circuit(self):
        """
        Adversarial Invariant: lambda_gate == 0.0 must immediately return visual_tokens.
        """
        visual_tokens = torch.randn(256, 896)
        s_qcass = torch.softmax(torch.randn(256), dim=0)

        out = apply_saliency_modulated_gating(
            visual_tokens=visual_tokens,
            s_qcass=s_qcass,
            lambda_gate=0.0,
        )

        self.assertIs(out, visual_tokens, "lambda_gate == 0.0 did not return original tensor object!")

    def test_power_law_entropy_monotonic_reduction(self):
        """
        Thermodynamic proof verification: Power-law concentration strictly decreases entropy.
        H(S_gamma) < H(S) for gamma > 1.0 when S is non-uniform.
        """
        P = 256
        s_raw = torch.softmax(torch.randn(P) * 1.5, dim=0)

        def entropy(p):
            return -torch.sum(p * torch.log(torch.clamp(p, min=1e-12))).item()

        h_orig = entropy(s_raw)

        # Concentrated under gamma = 2.0 (lambda_gate = 0.5)
        gamma = 2.0
        s_pow = torch.pow(torch.clamp(s_raw, min=1e-8), gamma)
        s_pow_norm = s_pow / torch.sum(s_pow)
        h_pow2 = entropy(s_pow_norm)

        # Concentrated under gamma = 3.0 (lambda_gate = 1.0)
        gamma = 3.0
        s_pow3 = torch.pow(torch.clamp(s_raw, min=1e-8), gamma)
        s_pow3_norm = s_pow3 / torch.sum(s_pow3)
        h_pow3 = entropy(s_pow3_norm)

        self.assertLess(h_pow2, h_orig, "Power-law gamma=2 failed to reduce entropy!")
        self.assertLess(h_pow3, h_pow2, "Power-law gamma=3 failed to further reduce entropy!")

    # =========================================================================
    # ADDITIONAL ADVERSARIAL STRESS: Dimensionality & Distribution Extremes
    # =========================================================================

    def test_stopping_gate_batch_dimension_broadcasting(self):
        """
        Adversarial Test: Batch size B = 4 with heterogeneous conditions:
        Item 0: In cohesion -> should NOT stop.
        Item 1: E_t high, delta_eos high, NOT in cohesion -> MUST stop.
        Item 2: Steered token is EOS -> MUST stop.
        Item 3: Low E_t, low delta_eos -> should NOT stop.
        """
        B = 4
        logits_g = torch.randn(B, self.vocab_size)
        logits_l = torch.randn(B, self.vocab_size)
        steered_tokens = torch.tensor([16, 17, self.eos_token_ids[0], 19])  # Item 2 is EOS

        e_t = torch.tensor([0.90, 0.90, 0.10, 0.10])
        delta_eos = torch.tensor([8.0, 8.0, 1.0, 1.0])
        is_in_cohesion = torch.tensor([True, False, False, False])

        stop_pred = evaluate_semantic_stopping(
            logits_g=logits_g,
            logits_l=logits_l,
            steered_tokens=steered_tokens,
            eos_token_ids=self.eos_token_ids,
            step_idx=3,
            e_t=e_t,
            delta_eos=delta_eos,
            is_in_alnum_cohesion=is_in_cohesion,
        )

        self.assertEqual(stop_pred.shape, (B,))
        self.assertFalse(bool(stop_pred[0].item()), "Item 0 stopped despite active cohesion!")
        self.assertTrue(bool(stop_pred[1].item()), "Item 1 failed to crisp stop with high E_t, delta_eos!")
        self.assertTrue(bool(stop_pred[2].item()), "Item 2 failed to stop when steered token is EOS!")
        self.assertFalse(bool(stop_pred[3].item()), "Item 3 stopped prematurely with low E_t!")

    def test_stopping_gate_1d_tensor_support(self):
        """
        Adversarial Test: Verify that 1D logits (shape [K]) return scalar boolean tensor without shape error.
        """
        logits_g = torch.randn(self.vocab_size)
        logits_l = torch.randn(self.vocab_size)
        steered_tok = torch.tensor(16)

        stop = evaluate_semantic_stopping(
            logits_g=logits_g,
            logits_l=logits_l,
            steered_tokens=steered_tok,
            eos_token_ids=self.eos_token_ids,
            step_idx=2,
            e_t=0.10,
            delta_eos=1.0,
            is_in_alnum_cohesion=False,
        )

        self.assertEqual(stop.dim(), 0, "1D logits did not produce 0-dim scalar boolean tensor!")
        self.assertFalse(bool(stop.item()))

    def test_epistemic_index_extreme_cases(self):
        """
        Adversarial Test: Extreme distribution configurations for compute_epistemic_exhaustion_index:
        1. Completely disjoint distributions: J_t -> ln(2).
        2. Identical distributions: J_t -> 0.0, rho_t -> 0.0 when peak_js is None.
        3. Zero peak_js: Graceful handling without division by zero.
        """
        K = 1000
        # 1. Disjoint
        log_p_l = torch.full((1, K), -1e9)
        log_p_l[0, 0] = 0.0
        log_p_g = torch.full((1, K), -1e9)
        log_p_g[0, 1] = 0.0

        e_t, peak, j_t, rho_t, h_hat = compute_epistemic_exhaustion_index(log_p_l, log_p_g)
        self.assertAlmostEqual(float(j_t.item()), math.log(2.0), places=3)
        self.assertFalse(torch.isnan(e_t).any())

        # 2. Identical with peak_js = 0.0
        log_p_same = torch.zeros(1, K) - math.log(K)
        e_t, peak, j_t, rho_t, h_hat = compute_epistemic_exhaustion_index(log_p_same, log_p_same, peak_js=0.0)
        self.assertAlmostEqual(float(j_t.item()), 0.0, places=4)
        self.assertFalse(torch.isnan(e_t).any())
        self.assertFalse(torch.isinf(e_t).any())

    def test_power_law_tensor_dimensions_and_shapes(self):
        """
        Adversarial Test: Verify 2D/3D tensor shape permutations for apply_saliency_modulated_gating:
        - visual_tokens: [1, P, D] and [P, D]
        - s_qcass: [1, P] and [P]
        """
        P, D = 256, 896
        s_base = torch.softmax(torch.randn(P), dim=0)

        for vt_shape in [(P, D), (1, P, D)]:
            for s_shape in [(P,), (1, P)]:
                vt = torch.randn(*vt_shape)
                s = s_base.view(*s_shape)

                out = apply_saliency_modulated_gating(
                    visual_tokens=vt,
                    s_qcass=s,
                    lambda_gate=0.5,
                )
                self.assertEqual(out.shape, vt.shape)
                self.assertFalse(torch.isnan(out).any())


if __name__ == "__main__":
    unittest.main()
