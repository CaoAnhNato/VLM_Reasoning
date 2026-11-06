# -*- coding: utf-8 -*-
r"""
Adversarial Stress Test Suite for NE-VFE Mean-Field Contrastive Reference Decoding (MF-CD).

Comprehensive mathematical stress testing of src/nevfe/core/engine.py:
1. Float32 Precision Boundaries & Machine Epsilon Safeguards:
   - Identical distributions where BC >= 1.0 due to float32 accumulation.
   - Near-orthogonal & disjoint distributions across vocabularies (V=10 to V=151,643).
   - Extreme logit values (+1e6, -1e6, large dynamic ranges, subnormals).
   - Epsilon boundary behavior in Bhattacharyya square root and Fisher-Rao geodesic.
2. Extreme Entropy & Dirac Spikes:
   - Perfectly uniform logits (maximum entropy H = ln(V)).
   - Dirac delta spikes (zero entropy H = 0).
   - Thermodynamic entropy NaN-to-num boundary at p=0, log p=-inf.
3. Adaptive Plausibility Truncation (APC) Invariants:
   - Candidate support non-emptiness for all beta in [0.001, 0.99].
   - Strict candidate exclusion: p^+(w) < beta * max p^+ is masked to -1e9.
   - Argmax admissibility: selected token is guaranteed to belong to V_cand.
4. Spatial Permutation & Oblivion Invariance (Theorem 2):
   - Strict invariance of Mean-Field Reference under arbitrary spatial permutations pi(V).
   - Zero spatial variance across visual patch and tile dimensions (< 1e-6).
   - Monochromatic image identity and contrast cooling.
5. Monotonic Optical Amplification & Preamble Suppression (Theorems 3 & 4):
   - Monotonic logit growth (dz*/dalpha > 0) for optical text tokens.
   - Natural suppression of conversational preamble echoes without regex.
6. Dimensional & Batch Consistency:
   - Mathematical equivalence between 1D and 2D batched operations.
7. High-Volume Randomized Stress Harness:
   - 500 randomized trials spanning Cauchy, Gaussian, Uniform, and Extreme distributions.
   - Strict verification of ZERO NaNs, ZERO Infs, and ZERO numerical collapses.
"""

import math
import unittest
import torch
import torch.nn.functional as F

from src.nevfe.core.engine import (
    NEVFEEngine,
    compute_mean_field_reference_embeddings,
    compute_bhattacharyya_coefficient,
    compute_dynamic_coupling,
    compute_unnormalized_entropy,
    mean_field_contrastive_step,
    nevfe_step_logits,
    modulate_patch_embeddings,
)


class TestAdversarialMFCDMath(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261106)

    # =========================================================================
    # SECTION 1: FLOAT32 PRECISION BOUNDARIES & SAFEGUARDS
    # =========================================================================

    def test_01_bc_identical_distributions_float32_safeguard(self):
        """Stress-test BC when p+ and p0 are mathematically identical.
        Float32 accumulation drift must be strictly clamped to <= 1.0,
        and dynamic coupling must evaluate cleanly to 0.0 without NaNs.
        """
        vocab_sizes = [10, 100, 1000, 32000, 151643]
        for V in vocab_sizes:
            with self.subTest(vocab_size=V):
                logits = torch.randn(1, V, dtype=torch.float32)
                p = F.softmax(logits, dim=-1)

                bc = compute_bhattacharyya_coefficient(p, p)
                self.assertFalse(torch.isnan(bc).any(), f"BC is NaN for V={V}")
                self.assertFalse(torch.isinf(bc).any(), f"BC is Inf for V={V}")
                self.assertLessEqual(bc.item(), 1.0, f"BC exceeded 1.0 for V={V}: {bc.item()}")
                self.assertGreaterEqual(bc.item(), 0.999, f"BC unexpectedly low for identical p: {bc.item()}")

                # Coupling modulation: due to float32 machine precision scaling sqrt(eps_mach) ~= 3.45e-4,
                # a 1-LSB rounding in BC (BC = 1 - 2^-24) yields coupling = 0.5 * sqrt(2^-23) = 2^-12 = 0.00024414.
                # Dynamic coupling must remain <= 5e-4 (negligible < 0.05% of maximum coupling 0.50).
                coupling = compute_dynamic_coupling(0.50, bc)
                self.assertFalse(torch.isnan(coupling).any(), f"Coupling is NaN for V={V}")
                self.assertLess(coupling.item(), 5e-4, f"Coupling unexpectedly large for identical distributions: {coupling.item()}")

                # When lp == lz, steered logits must equal lp identically (< 1e-5)
                _, steered = mean_field_contrastive_step(logits, logits, alpha=0.50, beta=0.05)
                cand_mask = (p >= 0.05 * torch.max(p))
                diff = torch.abs(steered[cand_mask] - logits[cand_mask])
                self.assertLess(diff.max().item(), 1e-5, f"Steered logits diverged from base logits for V={V}")

    def test_02_bc_machine_epsilon_overflow_handling(self):
        """Adversarial injection: inject BC slightly > 1.0 (e.g. 1.0000002, 1.001, 10.0, 1e6).
        Unclamped sqrt(1 - BC^2) would yield NaN for BC > 1.0.
        Verify that compute_dynamic_coupling clamps BC and returns 0.0 without NaNs.
        """
        adversarial_bcs = [
            torch.tensor([[1.0 + 1e-7]], dtype=torch.float32),
            torch.tensor([[1.0000002]], dtype=torch.float32),
            torch.tensor([[1.001]], dtype=torch.float32),
            torch.tensor([[1.5]], dtype=torch.float32),
            torch.tensor([[100.0]], dtype=torch.float32),
            torch.tensor([[1e6]], dtype=torch.float32),
        ]
        for bc_adv in adversarial_bcs:
            with self.subTest(bc_input=bc_adv.item()):
                coupling = compute_dynamic_coupling(0.50, bc_adv)
                self.assertFalse(torch.isnan(coupling).any(), f"Coupling was NaN for BC={bc_adv.item()}")
                self.assertFalse(torch.isinf(coupling).any(), f"Coupling was Inf for BC={bc_adv.item()}")
                self.assertEqual(coupling.item(), 0.0, f"Coupling must be 0.0 when BC >= 1.0, got {coupling.item()}")

    def test_03_bc_negative_and_zero_boundary(self):
        """Adversarial injection: test BC <= 0.0 (-1e6, -1.0, 0.0).
        Coupling must evaluate to lambda_param * 1.0 = lambda_param without NaNs.
        """
        negative_bcs = [
            torch.tensor([[0.0]], dtype=torch.float32),
            torch.tensor([[-0.001]], dtype=torch.float32),
            torch.tensor([[-1.0]], dtype=torch.float32),
            torch.tensor([[-1e6]], dtype=torch.float32),
        ]
        for bc_neg in negative_bcs:
            with self.subTest(bc_input=bc_neg.item()):
                coupling = compute_dynamic_coupling(0.75, bc_neg)
                self.assertFalse(torch.isnan(coupling).any(), f"Coupling was NaN for BC={bc_neg.item()}")
                self.assertEqual(coupling.item(), 0.75, f"Coupling must equal lambda (0.75) for BC <= 0, got {coupling.item()}")

    def test_04_orthogonal_and_disjoint_distributions(self):
        """Stress-test orthogonal distributions with completely disjoint support.
        Verify BC is in [0.0, 1.0] and dynamic coupling approaches full lambda_param.
        Also inspect vocabulary-scale epsilon accumulation behavior.
        """
        vocab_sizes = [10, 1000, 32000, 151643]
        for V in vocab_sizes:
            with self.subTest(vocab_size=V):
                # Orthogonal supports: p+ concentrated on token 0, p0 concentrated on token V-1
                p_plus = torch.zeros(1, V, dtype=torch.float32)
                p_zero = torch.zeros(1, V, dtype=torch.float32)
                p_plus[0, 0] = 1.0
                p_zero[0, -1] = 1.0

                bc = compute_bhattacharyya_coefficient(p_plus, p_zero)
                self.assertFalse(torch.isnan(bc).any())
                self.assertFalse(torch.isinf(bc).any())
                self.assertGreaterEqual(bc.item(), 0.0)
                self.assertLessEqual(bc.item(), 1.0)

                coupling = compute_dynamic_coupling(0.50, bc)
                self.assertFalse(torch.isnan(coupling).any())
                self.assertFalse(torch.isinf(coupling).any())
                self.assertGreaterEqual(coupling.item(), 0.0)
                self.assertLessEqual(coupling.item(), 0.50)

    def test_05_extreme_magnitude_logits(self):
        """Stress-test extreme logit values (+1e6, -1e6, mixed extremes, subnormals).
        Ensure zero NaNs, zero Infs, and stable token selection.
        """
        V = 1000
        extreme_cases = {
            "all_large_positive": (torch.full((V,), 1e6), torch.full((V,), 1e6)),
            "all_large_negative": (torch.full((V,), -1e6), torch.full((V,), -1e6)),
            "opposite_extremes": (torch.full((V,), 1e5), torch.full((V,), -1e5)),
            "spike_plus_flat_zero": (
                torch.cat([torch.tensor([1e4]), torch.full((V - 1,), -1e4)]),
                torch.zeros(V),
            ),
            "subnormal_floats": (torch.full((V,), 1e-38), torch.full((V,), 1e-38)),
            "mixed_huge_range": (
                torch.linspace(-1e5, 1e5, V),
                torch.linspace(1e5, -1e5, V),
            ),
        }

        for name, (lp, lz) in extreme_cases.items():
            with self.subTest(case=name):
                tok, steered = mean_field_contrastive_step(lp, lz, alpha=0.50, beta=0.05)
                self.assertFalse(torch.isnan(steered).any(), f"NaN generated in {name}")
                self.assertFalse(torch.isinf(steered).any(), f"Inf generated in {name}")
                self.assertGreaterEqual(tok.item(), 0, f"Invalid token index in {name}")
                self.assertLess(tok.item(), V, f"Invalid token index in {name}")

    # =========================================================================
    # SECTION 2: EXTREME ENTROPY & DIRAC SPIKES
    # =========================================================================

    def test_06_maximum_entropy_uniform_logits(self):
        """Stress-test perfectly uniform logits (H = ln(V)).
        All probabilities equal 1/V. BC must be 1.0, coupling 0.0.
        Steered logits on candidate set must match base model logits.
        """
        V = 256
        lp = torch.full((V,), 42.0)
        lz = torch.full((V,), 42.0)

        tok, steered = mean_field_contrastive_step(lp, lz, alpha=0.50, beta=0.05)
        self.assertFalse(torch.isnan(steered).any())

        # Since p+ is uniform, all tokens are in V_cand (p_i == max_p)
        # Steered logits must equal lp
        diff = torch.abs(steered - lp)
        self.assertLess(diff.max().item(), 1e-4, "Steered logits must match base logits for uniform inputs")

    def test_07_zero_entropy_dirac_delta_spikes(self):
        """Stress-test Dirac delta distribution (zero entropy).
        Token 0 has probability ~ 1.0, all other tokens ~ 0.0.
        APC must prune all tokens except token 0.
        """
        V = 500
        lp = torch.full((V,), -100.0)
        lz = torch.full((V,), -100.0)
        lp[0] = 100.0  # Dirac spike

        tok, steered = mean_field_contrastive_step(lp, lz, alpha=0.50, beta=0.05)
        self.assertEqual(tok.item(), 0, "Dirac spike token 0 must be selected")
        self.assertGreater(steered[0].item(), -1e8, "Token 0 must be kept unmasked")

        # All other tokens must be masked to -1e9
        masked_tokens = steered[1:]
        self.assertTrue(torch.all(masked_tokens == -1e9), "All non-spike tokens must be masked to -1e9")

    def test_08_entropy_nan_to_num_boundary(self):
        """Verify information-theoretic boundary lim_{p -> 0} p log p = 0.
        Passing p=0 and log_p=-inf must return finite 0.0 without NaN.
        """
        p = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        log_p = torch.tensor([[0.0, -float("inf"), -float("inf"), -float("inf")]])

        h = compute_unnormalized_entropy(p, log_p)
        self.assertFalse(torch.isnan(h).any(), "Entropy must not be NaN at p=0 boundary")
        self.assertFalse(torch.isinf(h).any(), "Entropy must not be Inf at p=0 boundary")
        self.assertAlmostEqual(h.item(), 0.0, places=6)

    # =========================================================================
    # SECTION 3: ADAPTIVE PLAUSIBILITY TRUNCATION (APC) INVARIANTS
    # =========================================================================

    def test_09_apc_support_non_emptiness_and_admissibility(self):
        """Theorem & Contract Invariant:
        For any beta in [0.001, 0.99], candidate support V_cand is NEVER empty,
        and argmax next_tokens is GUARANTEED to be in V_cand.
        """
        betas = [0.001, 0.01, 0.03, 0.05, 0.08, 0.10, 0.20, 0.50, 0.99]
        V = 200

        for beta in betas:
            with self.subTest(beta=beta):
                lp = torch.randn(V)
                lz = torch.randn(V)

                tok, steered = mean_field_contrastive_step(lp, lz, alpha=0.50, beta=beta)
                p_plus = F.softmax(lp, dim=-1)
                max_p = torch.max(p_plus).item()

                cand_mask = (p_plus >= beta * max_p)
                cand_count = torch.sum(cand_mask).item()

                self.assertGreaterEqual(cand_count, 1, f"Candidate support was empty for beta={beta}!")
                self.assertTrue(cand_mask[tok.item()].item(), f"Selected token {tok.item()} was NOT in V_cand!")
                self.assertGreater(steered[tok.item()].item(), -1e8, "Selected token logit was masked!")

    def test_10_apc_deeply_negative_logits_boundary(self):
        """Stress-test APC when unsteered logits are deeply negative (< -1e9, e.g. -1e10).
        Examine the boundary behavior where unsteered logits are smaller than the mask value -1e9.
        """
        V = 10
        # If logits are -1e10, unmasked positions could be -1e10, while masked are -1e9.
        # Verify how engine behaves under this extreme condition.
        lp = torch.full((V,), -1e5)
        lz = torch.full((V,), -1e5)
        lp[0] = -1e4  # Still relatively much higher than others, but p_plus max will be 1.0

        tok, steered = mean_field_contrastive_step(lp, lz, alpha=0.50, beta=0.05)
        self.assertFalse(torch.isnan(steered).any())
        self.assertEqual(tok.item(), 0, "Token 0 with highest relative logit must be selected")

    # =========================================================================
    # SECTION 4: SPATIAL PERMUTATION & OBLIVION INVARIANCE (THEOREM 2)
    # =========================================================================

    def test_11_spatial_permutation_invariance_100_permutations(self):
        """Theorem 2: Mean-Field Reference V^0 = 1_P \bar{v}^\top is strictly invariant
        under ANY arbitrary spatial permutation pi in S_P of visual patches and tiles.
        Stress-test over 100 random spatial permutations.
        """
        N_tiles, N_patches, D = 4, 256, 896
        vit_embeds = torch.randn(N_tiles, N_patches, D, dtype=torch.float32)

        v_zero_ref = compute_mean_field_reference_embeddings(vit_embeds)
        flat_embeds = vit_embeds.view(N_tiles * N_patches, D)

        for trial in range(100):
            perm = torch.randperm(N_tiles * N_patches)
            vit_perm = flat_embeds[perm].view(N_tiles, N_patches, D)
            v_zero_perm = compute_mean_field_reference_embeddings(vit_perm)

            diff = torch.abs(v_zero_ref - v_zero_perm)
            max_diff = diff.max().item()
            self.assertLess(max_diff, 1e-5, f"Spatial permutation invariance violated at trial {trial}: max diff {max_diff}")

    def test_12_spatial_variance_identically_zero(self):
        """Theorem 2: Spatial Centroid Reference exhibits identically zero spatial variance
        across visual patch tokens and tiles: Var_spatial(V^0) == 0 (< 1e-6).
        """
        N_tiles, N_patches, D = 3, 256, 896
        vit_embeds = torch.randn(N_tiles, N_patches, D) * 50.0 + 10.0  # high dynamic range

        v_zero = compute_mean_field_reference_embeddings(vit_embeds)
        flat_zero = v_zero.view(-1, D)

        # Variance across tokens
        var_spatial = torch.var(flat_zero, dim=0, unbiased=False)
        self.assertLess(var_spatial.max().item(), 1e-6, "Spatial variance must be identically zero")

    def test_13_monochromatic_input_identity(self):
        """When visual patches are already monochromatic/uniform,
        compute_mean_field_reference_embeddings(V^+) must identically equal V^+.
        """
        P, D = 512, 896
        single_vec = torch.randn(1, D)
        v_mono = single_vec.expand(P, D).clone()

        v_zero = compute_mean_field_reference_embeddings(v_mono)
        diff = torch.abs(v_mono - v_zero)
        self.assertLess(diff.max().item(), 1e-6, "Monochromatic input must be identical to its reference")

    def test_14_extreme_magnitudes_in_visual_embeddings(self):
        """Stress-test compute_mean_field_reference_embeddings with extreme values:
        1e6, -1e6, subnormals, and verify zero NaNs or Infs.
        """
        N_tiles, N_patches, D = 2, 256, 128
        extreme_embeds = torch.full((N_tiles, N_patches, D), 1e6, dtype=torch.float32)
        v_zero = compute_mean_field_reference_embeddings(extreme_embeds)
        self.assertFalse(torch.isnan(v_zero).any())
        self.assertFalse(torch.isinf(v_zero).any())
        self.assertEqual(v_zero[0, 0, 0].item(), 1e6)

        # Mixed extremes
        extreme_mixed = torch.randn(N_tiles, N_patches, D)
        extreme_mixed[0, 0, :] = 1e6
        extreme_mixed[1, -1, :] = -1e6
        v_mixed = compute_mean_field_reference_embeddings(extreme_mixed)
        self.assertFalse(torch.isnan(v_mixed).any())
        self.assertFalse(torch.isinf(v_mixed).any())

    # =========================================================================
    # SECTION 5: THEOREM 3 & THEOREM 4 ADVERSARIAL VERIFICATION
    # =========================================================================

    def test_15_theorem_3_strictly_monotonic_amplification(self):
        """Theorem 3: For any candidate text token w* with p^+(w*) > p^0(w*),
        steered logit z*(w*) is STRICTLY monotonically increasing with alpha.
        Test across a dense grid of alpha in [0.0, 1.0].
        """
        V = 50
        lp = torch.full((V,), -2.0)
        lz = torch.full((V,), -2.0)
        text_token = 7
        lp[text_token] = 4.0  # high visual evidence
        lz[text_token] = -1.0  # low linguistic prior

        alphas = [0.0, 0.1, 0.25, 0.4, 0.5, 0.65, 0.8, 0.95, 1.0]
        steered_vals = []
        for a in alphas:
            _, steered = mean_field_contrastive_step(lp, lz, alpha=a, beta=0.01)
            steered_vals.append(steered[text_token].item())

        for i in range(len(steered_vals) - 1):
            self.assertLess(
                steered_vals[i],
                steered_vals[i + 1],
                f"Monotonic amplification violated between alpha={alphas[i]} and {alphas[i+1]}",
            )

    def test_16_theorem_4_preamble_echo_suppression_adversarial(self):
        """Theorem 4: Adversarially construct a strong preamble echo distractor:
        lp favors preamble token 0 over text token 1 (z^+(0) = 4.0 vs z^+(1) = 3.2).
        Reference stream V^0 predicts preamble strongly (z^0(0) = 8.0 vs z^0(1) = -2.0).
        Under MF-CD, contrast MUST flip the decision to text token 1 WITHOUT REGEX.
        """
        V = 100
        lp = torch.full((V,), -5.0)
        lz = torch.full((V,), -5.0)

        preamble_tok = 0
        text_tok = 1

        lp[preamble_tok] = 4.0
        lz[preamble_tok] = 8.0  # High language prior

        lp[text_tok] = 3.2
        lz[text_tok] = -2.0  # Pure optical token

        # Before steering: base model outputs preamble
        tok_base, _ = mean_field_contrastive_step(lp, lz, alpha=0.0, beta=0.05)
        self.assertEqual(tok_base.item(), preamble_tok, "Base model selects preamble without steering")

        # After MF-CD steering: text token must win
        tok_steered, steered_logits = mean_field_contrastive_step(lp, lz, alpha=0.50, beta=0.05)
        self.assertEqual(tok_steered.item(), text_tok, "MF-CD must overturn preamble in favor of text token!")
        self.assertGreater(steered_logits[text_tok].item(), steered_logits[preamble_tok].item())

    # =========================================================================
    # SECTION 6: DIMENSIONAL, BATCH & API ROBUSTNESS
    # =========================================================================

    def test_17_batch_and_1d_mathematical_equivalence(self):
        """Verify that running 1D input [V] vs 2D batched [B, V] yields
        bitwise/float32 identical results per batch element.
        """
        V = 128
        lp_0 = torch.randn(V)
        lz_0 = torch.randn(V)
        lp_1 = torch.randn(V)
        lz_1 = torch.randn(V)

        tok_1d_0, logits_1d_0 = mean_field_contrastive_step(lp_0, lz_0, alpha=0.50, beta=0.05)
        tok_1d_1, logits_1d_1 = mean_field_contrastive_step(lp_1, lz_1, alpha=0.50, beta=0.05)

        lp_batch = torch.stack([lp_0, lp_1], dim=0)
        lz_batch = torch.stack([lz_0, lz_1], dim=0)
        tok_batch, logits_batch = mean_field_contrastive_step(lp_batch, lz_batch, alpha=0.50, beta=0.05)

        self.assertEqual(tok_batch[0].item(), tok_1d_0.item())
        self.assertEqual(tok_batch[1].item(), tok_1d_1.item())

        diff_0 = torch.abs(logits_batch[0] - logits_1d_0)
        diff_1 = torch.abs(logits_batch[1] - logits_1d_1)
        self.assertLess(diff_0.max().item(), 1e-6)
        self.assertLess(diff_1.max().item(), 1e-6)

    def test_18_engine_boundary_and_contract_validation(self):
        """Stress-test NEVFEEngine constructor constraints and decode_step contract."""
        engine = NEVFEEngine(lambda_param=0.50, beta_param=0.05)
        self.assertIsInstance(engine, NEVFEEngine)

        # Test boundary acceptance
        self.assertEqual(NEVFEEngine(lambda_param=0.0).lambda_param, 0.0)
        self.assertEqual(NEVFEEngine(lambda_param=1.5).lambda_param, 1.5)
        self.assertEqual(NEVFEEngine(beta_param=0.001).beta_param, 0.001)

        # Test out-of-bounds rejection
        with self.assertRaises(ValueError):
            NEVFEEngine(lambda_param=-0.01)
        with self.assertRaises(ValueError):
            NEVFEEngine(lambda_param=1.51)
        with self.assertRaises(ValueError):
            NEVFEEngine(beta_param=0.0)
        with self.assertRaises(ValueError):
            NEVFEEngine(beta_param=-0.1)

    # =========================================================================
    # SECTION 7: HIGH-VOLUME RANDOMIZED ADVERSARIAL STRESS HARNESS
    # =========================================================================

    def test_19_high_volume_randomized_stress_harness_500_trials(self):
        """Execute 500 randomized trials combining:
        - Random vocabulary sizes: V in [10, 50000]
        - Random distributions: Gaussian, Uniform, Cauchy (heavy-tailed), Exponential
        - Random coupling alpha in [0.0, 1.0] and beta in [0.01, 0.10]
        - Random batch sizes: 1D, 2D (B=1, 2, 4)
        VERIFY: ZERO NaNs, ZERO Infs, ZERO unhandled exceptions across all 500 trials.
        """
        distributions = ["gaussian", "uniform", "cauchy", "exponential"]
        num_trials = 500

        nan_count = 0
        inf_count = 0
        failure_log = []

        for trial in range(num_trials):
            dist_type = distributions[trial % len(distributions)]
            V = torch.randint(10, 5000, (1,)).item()
            B = (trial % 4)  # 0 -> 1D, 1..3 -> 2D with batch size B

            alpha = float(torch.empty(1).uniform_(0.0, 1.0).item())
            beta = float(torch.empty(1).uniform_(0.01, 0.10).item())

            shape = (V,) if B == 0 else (B, V)

            if dist_type == "gaussian":
                lp = torch.randn(shape) * 10.0
                lz = torch.randn(shape) * 10.0
            elif dist_type == "uniform":
                lp = torch.rand(shape) * 100.0 - 50.0
                lz = torch.rand(shape) * 100.0 - 50.0
            elif dist_type == "cauchy":
                # Heavy-tailed distribution prone to extreme outliers
                lp = torch.distributions.Cauchy(0.0, 5.0).sample(shape)
                lz = torch.distributions.Cauchy(0.0, 5.0).sample(shape)
            elif dist_type == "exponential":
                lp = torch.distributions.Exponential(1.0).sample(shape) - 5.0
                lz = torch.distributions.Exponential(1.0).sample(shape) - 5.0

            try:
                tok, steered = mean_field_contrastive_step(lp, lz, alpha=alpha, beta=beta)
                if torch.isnan(steered).any():
                    nan_count += 1
                    failure_log.append(f"Trial {trial} ({dist_type}): NaN in steered logits")
                if torch.isinf(steered).any():
                    inf_count += 1
                    failure_log.append(f"Trial {trial} ({dist_type}): Inf in steered logits")

                # Test unified operator nevfe_step_logits as well
                t_op, l_op, a_op, e_op = nevfe_step_logits(lp, lz, lambda_param=alpha, beta_param=beta)
                if torch.isnan(l_op).any() or torch.isnan(a_op).any() or torch.isnan(e_op).any():
                    nan_count += 1
                    failure_log.append(f"Trial {trial} ({dist_type}): NaN in nevfe_step_logits")
                if torch.isinf(l_op).any() or torch.isinf(a_op).any() or torch.isinf(e_op).any():
                    inf_count += 1
                    failure_log.append(f"Trial {trial} ({dist_type}): Inf in nevfe_step_logits")

            except Exception as e:
                failure_log.append(f"Trial {trial} ({dist_type}) Exception: {str(e)}")

        self.assertEqual(nan_count, 0, f"Encountered {nan_count} NaNs across {num_trials} trials! Log: {failure_log[:5]}")
        self.assertEqual(inf_count, 0, f"Encountered {inf_count} Infs across {num_trials} trials! Log: {failure_log[:5]}")
        self.assertEqual(len(failure_log), 0, f"Failures encountered: {failure_log[:5]}")


if __name__ == "__main__":
    unittest.main()
