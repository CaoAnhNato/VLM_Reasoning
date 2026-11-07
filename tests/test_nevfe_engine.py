# -*- coding: utf-8 -*-
r"""
Unit tests for NE-VFE Mean-Field Contrastive Reference Decoding (MF-CD) Engine.

Verifies:
1. Fisher-Saliency Patch Energy Modulation on 2D and 3D tensors (utility primitive).
2. Theorem 2: Spatial information oblivion of Mean-Field Reference:
   - Identically zero spatial variance (\nabla_{spatial} V^0 == 0).
   - Strict spatial permutation invariance (\pi(V^0) == V^0).
3. Bhattacharyya Coefficient computation with strict float32 clamping to [0.0, 1.0].
4. Theorem 5: Fisher-Rao Geodesic Dynamic Coupling Modulation:
   - Geodesic cooling to 0.0 as BC -> 1.0 on syntactic connectors.
   - Float32 precision boundary safeguard against NaNs when BC exceeds 1.0 by machine epsilon.
5. Thermodynamic Shannon Entropy computation in nats.
6. Adaptive Plausibility Truncation (APC) candidate support set mask (beta in [0.01, 0.10]).
7. Theorem 3: Monotonic optical amplification under Mean-Field contrast.
8. Theorem 4: Natural conversational preamble suppression via prior likelihood ratio without regex.
9. Monochromatic / solid-color image invariance (V^+ == V^0 => alpha_t == 0 => base logits preserved).
10. Canonical single-step MF-CD API (mean_field_contrastive_step) across 1D and 2D inputs.
11. Unified nevfe_step_logits operator backwards compatibility (returning 4-tuple).
12. NEVFEEngine interface contracts and hyperparameter bounds.
13. Query token mask casing and BPE prefix invariance utility.
"""

import unittest
import math
import torch
import torch.nn.functional as F

from src.nevfe.core.engine import (
    NEVFEEngine,
    compute_mean_field_reference_embeddings,
    mean_field_contrastive_step,
    modulate_patch_embeddings,
    compute_bhattacharyya_coefficient,
    compute_dynamic_coupling,
    compute_unnormalized_entropy,
    compute_candidate_support_mask,
    compute_free_energy_support_mask,
    nevfe_step_logits,
)


class TestNEVFEEngineMFCD(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_01_modulate_patch_embeddings(self):
        # Test 2D tensor [P, D]
        p_2d = torch.randn(16, 64, dtype=torch.float32)
        mod_2d = modulate_patch_embeddings(p_2d)
        self.assertEqual(mod_2d.shape, p_2d.shape)
        self.assertEqual(mod_2d.dtype, p_2d.dtype)
        self.assertFalse(torch.isnan(mod_2d).any())
        self.assertFalse(torch.isinf(mod_2d).any())

        # Test 3D tensor [B, P, D]
        p_3d = torch.randn(2, 16, 64, dtype=torch.bfloat16)
        mod_3d = modulate_patch_embeddings(p_3d)
        self.assertEqual(mod_3d.shape, p_3d.shape)
        self.assertEqual(mod_3d.dtype, torch.bfloat16)
        self.assertFalse(torch.isnan(mod_3d.float()).any())

    def test_02_theorem_2_spatial_information_oblivion(self):
        # Theorem 2: Spatial Centroid Reference exhibits identically zero spatial variance
        # and strict spatial permutation invariance across visual patches
        N_tiles, N_tokens, D = 3, 256, 896
        vit_embeds = torch.randn(N_tiles, N_tokens, D, dtype=torch.float32)

        v_zero = compute_mean_field_reference_embeddings(vit_embeds)
        self.assertEqual(v_zero.shape, vit_embeds.shape)

        # 1. Spatial variance across all tokens must be identically zero
        flat_zero = v_zero.reshape(N_tiles * N_tokens, D)
        spatial_var = torch.var(flat_zero, dim=0, unbiased=False)
        self.assertLess(spatial_var.max().item(), 1e-6, "V^0 must have zero spatial variance!")

        # 2. Permutation Invariance: \pi(V^0) == V^0
        flat_embeds = vit_embeds.reshape(N_tiles * N_tokens, D)
        perm_idx = torch.randperm(N_tiles * N_tokens)
        vit_perm = flat_embeds[perm_idx].reshape(N_tiles, N_tokens, D)
        v_zero_perm = compute_mean_field_reference_embeddings(vit_perm)

        diff_perm = torch.abs(v_zero - v_zero_perm)
        self.assertLess(diff_perm.max().item(), 1e-5, "V^0 must be strictly invariant to spatial permutation!")

    def test_03_bhattacharyya_coefficient_and_clamp(self):
        # 1. Toy vocabulary test case (V=4)
        # Identical distributions should give BC = 1.0
        p = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        bc_ident = compute_bhattacharyya_coefficient(p, p)
        self.assertAlmostEqual(bc_ident.item(), 1.0, places=5)

        # Disjoint distributions should give BC ~ 0.0
        p1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        p2 = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        bc_disjoint = compute_bhattacharyya_coefficient(p1, p2)
        self.assertLess(bc_disjoint.item(), 1e-4)

        # Numerical clamping verification: BC must never exceed 1.0 even with accumulation error
        bc_overflow = compute_bhattacharyya_coefficient(p * 1.0001, p * 1.0001)
        self.assertLessEqual(bc_overflow.item(), 1.0)

        # 2. Realistic LLM vocabulary test case (V=151,936 for Vintern-1B)
        # Verifies elimination of vocabulary-scale epsilon accumulation bias (V * sqrt(eps))
        V = 151936
        # Case A: Two disjoint distributions over large vocabulary
        p_disjoint_1 = torch.zeros(1, V)
        p_disjoint_2 = torch.zeros(1, V)
        p_disjoint_1[0, 0] = 1.0
        p_disjoint_2[0, 1] = 1.0
        bc_large_disjoint = compute_bhattacharyya_coefficient(p_disjoint_1, p_disjoint_2)
        self.assertLessEqual(bc_large_disjoint.item(), 1e-6, "Disjoint distributions must have BC <= 1e-6 (strictly 0.0)!")
        self.assertEqual(bc_large_disjoint.item(), 0.0)

        # Case B: Two distributions with high overlap (true BC ~ 0.90)
        # Under V*sqrt(eps) accumulation, this would artificially clamp to 1.0 and zero out alpha_t
        delta = math.acos(0.90)
        p_overlap_1 = torch.zeros(1, V)
        p_overlap_2 = torch.zeros(1, V)
        p_overlap_1[0, 0] = math.cos(math.pi / 4.0 + delta) ** 2
        p_overlap_1[0, 1] = math.sin(math.pi / 4.0 + delta) ** 2
        p_overlap_2[0, 0] = 0.5
        p_overlap_2[0, 1] = 0.5
        bc_overlap = compute_bhattacharyya_coefficient(p_overlap_1, p_overlap_2)
        self.assertGreaterEqual(bc_overlap.item(), 0.89, "BC must be >= 0.89 for true BC=0.90")
        self.assertLessEqual(bc_overlap.item(), 0.91, "BC must be <= 0.91 for true BC=0.90")
        self.assertLess(bc_overlap.item(), 1.0, "High overlap BC must NOT clamp to 1.0!")

        # Dynamic coupling must remain active (alpha_t > 0)
        alpha_t = compute_dynamic_coupling(0.50, bc_overlap)
        self.assertGreater(alpha_t.item(), 0.15, "Dynamic coupling alpha_t must remain active when BC ~ 0.90!")

    def test_04_dynamic_coupling_and_float32_safeguard(self):
        # When BC = 1.0 (total agreement), visual guidance dynamic coupling should be 0.0 at t > 1
        bc_full = torch.tensor([[1.0]])
        lam_0 = compute_dynamic_coupling(0.50, bc_full, step_idx=2)
        self.assertAlmostEqual(lam_0.item(), 0.0, places=5)

        # When BC = 0.0 (total disagreement), dynamic coupling should equal alpha (0.50)
        bc_zero = torch.tensor([[0.0]])
        lam_max = compute_dynamic_coupling(0.50, bc_zero, step_idx=2)
        self.assertAlmostEqual(lam_max.item(), 0.50, places=5)

        # Intermediate BC: alpha * sqrt(1 - BC^2)
        # For BC = 0.6: sqrt(1 - 0.36) = 0.8 -> 0.50 * 0.8 = 0.40
        bc_mid = torch.tensor([[0.6]])
        lam_mid = compute_dynamic_coupling(0.50, bc_mid, step_idx=2)
        self.assertAlmostEqual(lam_mid.item(), 0.40, places=5)

        # Float32 precision boundary test: when BC slightly exceeds 1.0 due to float32 rounding
        bc_eps_above = torch.tensor([[1.0000002]], dtype=torch.float32)
        lam_safe = compute_dynamic_coupling(0.50, bc_eps_above, step_idx=2)
        self.assertFalse(torch.isnan(lam_safe).any(), "Coupling must not be NaN on float32 epsilon overflow!")
        self.assertEqual(lam_safe.item(), 0.0)

    def test_05_unnormalized_entropy(self):
        # One-hot distribution has H = 0.0
        p_onehot = torch.tensor([[1.0, 0.0, 0.0]])
        log_p_onehot = torch.tensor([[0.0, -100.0, -100.0]])
        h_onehot = compute_unnormalized_entropy(p_onehot, log_p_onehot)
        self.assertAlmostEqual(h_onehot.item(), 0.0, places=5)

        # Uniform distribution over K elements has H = log(K)
        k = 4
        p_unif = torch.full((1, k), 1.0 / k)
        log_p_unif = torch.full((1, k), math.log(1.0 / k))
        h_unif = compute_unnormalized_entropy(p_unif, log_p_unif)
        self.assertAlmostEqual(h_unif.item(), math.log(k), places=5)

    def test_06_adaptive_plausibility_truncation(self):
        # Adaptive Plausibility Constraint (APC):
        # V_cand = {w : p^+(w) >= beta * max p^+}
        # Tokens with p^+ >= beta * max_p are retained; others pruned to -1e9
        beta = 0.05
        logits_p = torch.tensor([[3.0, 1.0, -2.0, -5.0]])
        logits_z = torch.tensor([[0.0, 0.0, 0.0, 0.0]])

        next_tok, steered = mean_field_contrastive_step(
            logits_plus=logits_p,
            logits_zero=logits_z,
            alpha=0.50,
            beta=beta,
        )

        p_plus = F.softmax(logits_p, dim=-1)
        max_p = torch.max(p_plus).item()
        cutoff = beta * max_p

        # Token 0 is max (p ~ 0.84) -> must be kept
        self.assertGreater(steered[0, 0].item(), -1e8)
        # Token 3 is tiny (p ~ 0.00028 < 0.05 * 0.84 = 0.042) -> must be pruned
        self.assertEqual(steered[0, 3].item(), -1e9)
        self.assertEqual(next_tok.item(), 0)

    def test_07_theorem_3_monotonic_optical_amplification(self):
        # Theorem 3: For any genuine optical text token w^* with p^+(w^*) > p^0(w^*),
        # steered logit z_t^*(w^*) is strictly monotonically increasing with respect to alpha
        K = 20
        lp = torch.full((K,), 1.0)
        lz = torch.full((K,), 1.0)
        # Text token 2 has high visual evidence in p+ and low prior in p0
        lp[2] = 5.0
        lz[2] = 0.0

        _, steered_00 = mean_field_contrastive_step(lp, lz, alpha=0.0, beta=0.05)
        _, steered_30 = mean_field_contrastive_step(lp, lz, alpha=0.30, beta=0.05)
        _, steered_60 = mean_field_contrastive_step(lp, lz, alpha=0.60, beta=0.05)
        _, steered_90 = mean_field_contrastive_step(lp, lz, alpha=0.90, beta=0.05)

        val_00 = steered_00[2].item()
        val_30 = steered_30[2].item()
        val_60 = steered_60[2].item()
        val_90 = steered_90[2].item()

        self.assertLess(val_00, val_30, "Optical logit must strictly increase with alpha!")
        self.assertLess(val_30, val_60, "Optical logit must strictly increase with alpha!")
        self.assertLess(val_60, val_90, "Optical logit must strictly increase with alpha!")

    def test_08_theorem_4_natural_preamble_suppression(self):
        # Theorem 4: In the absence of spatial text in V^0, conversational preamble token w_pre
        # has p^0(w_pre) >= p^+(w_pre), producing non-positive contrast Delta <= 0.
        # Under MF-CD, optical scene text w^* overtakes preamble echo naturally without regex!
        K = 50
        lp = torch.full((K,), -8.0)
        lz = torch.full((K,), -8.0)

        # Token 0: preamble echo "Dựa" (high language prior, non-positive VMI)
        # lp[0] = 3.0, lz[0] = 5.0 -> Delta = 3.0 - 5.0 = -2.0 < 0
        lp[0] = 3.0
        lz[0] = 5.0

        # Token 1: genuine scene text "Bảo" (moderate multimodal evidence, near zero prior)
        # lp[1] = 2.4, lz[1] = -2.0 -> Delta = 2.4 - (-2.0) = +4.4 > 0
        lp[1] = 2.4
        lz[1] = -2.0

        # Before steering (alpha = 0.0), base model outputs preamble token 0 (3.0 > 2.4)
        tok_base, _ = mean_field_contrastive_step(lp, lz, alpha=0.0, beta=0.05)
        self.assertEqual(tok_base.item(), 0, "Base model selects preamble token 0 before steering")

        # Under MF-CD steering (alpha = 0.50, beta = 0.05), scene text token 1 defeats preamble!
        tok_steered, steered_logits = mean_field_contrastive_step(lp, lz, alpha=0.50, beta=0.05)
        self.assertEqual(tok_steered.item(), 1, "MF-CD must select scene text token 1 over preamble!")
        self.assertGreater(steered_logits[1].item(), steered_logits[0].item())

    def test_09_theorem_5_solid_color_monochromatic_invariance(self):
        # When an input image is a solid color, V^+ == V^0, so p^+ == p^0.
        # Under Theorem 5, BC == 1.0, so alpha_t == 0.0.
        # The steered distribution must match the base model logits identically!
        K = 30
        lp = torch.randn(K)
        lz = lp.clone()  # identical distributions

        tok_steered, steered_logits = mean_field_contrastive_step(lp, lz, alpha=0.50, beta=0.05)

        # Since BC = 1.0, alpha_t = 0.0, so steered_logits on admissible candidates == lp
        candidate_mask = (F.softmax(lp, dim=-1) >= 0.05 * torch.max(F.softmax(lp, dim=-1)))
        diff = torch.abs(steered_logits[candidate_mask] - lp[candidate_mask])
        self.assertLess(diff.max().item(), 1e-5, "On monochromatic image, steered logits must equal base logits!")

    def test_10_mean_field_contrastive_step_api(self):
        # 1D input tensor
        lp_1d = torch.randn(64)
        lz_1d = torch.randn(64)
        tok_1d, logits_1d = mean_field_contrastive_step(lp_1d, lz_1d, alpha=0.50, beta=0.05)
        self.assertEqual(tok_1d.dim(), 0)  # scalar tensor
        self.assertEqual(logits_1d.shape, torch.Size([64]))
        self.assertFalse(torch.isnan(logits_1d).any())

        # 2D batch tensor [B, V]
        lp_2d = torch.randn(3, 128)
        lz_2d = torch.randn(3, 128)
        tok_2d, logits_2d = mean_field_contrastive_step(lp_2d, lz_2d, alpha=0.50, beta=0.05)
        self.assertEqual(tok_2d.shape, torch.Size([3]))
        self.assertEqual(logits_2d.shape, torch.Size([3, 128]))
        self.assertFalse(torch.isnan(logits_2d).any())

    def test_11_nevfe_step_logits_compatibility(self):
        # Verify that nevfe_step_logits preserves the standard 4-tuple return signature:
        # (next_tokens, steered_logits, alpha_t, entropy_nats)
        lp = torch.randn(2, 50)
        lz = torch.randn(2, 50)
        toks, logits, alpha_t, entropy = nevfe_step_logits(
            logits_plus=lp,
            logits_zero=lz,
            lambda_param=0.50,
            beta_param=0.05,
        )
        self.assertEqual(toks.shape, torch.Size([2]))
        self.assertEqual(logits.shape, torch.Size([2, 50]))
        self.assertEqual(alpha_t.shape, torch.Size([2, 1]))
        self.assertEqual(entropy.shape, torch.Size([2, 1]))
        self.assertFalse(torch.isnan(logits).any())
        self.assertFalse(torch.isnan(alpha_t).any())
        self.assertFalse(torch.isnan(entropy).any())

    def test_12_engine_class_interface_and_bounds(self):
        engine = NEVFEEngine(lambda_param=0.50, beta_param=0.05)
        self.assertEqual(engine.lambda_param, 0.50)
        self.assertEqual(engine.beta_param, 0.05)

        # Valid boundary acceptance
        engine_edge = NEVFEEngine(lambda_param=0.0, beta_param=0.01)
        self.assertEqual(engine_edge.lambda_param, 0.0)

        # Invalid hyperparameter range tests
        with self.assertRaises(ValueError):
            NEVFEEngine(lambda_param=-0.1)  # < 0.0
        with self.assertRaises(ValueError):
            NEVFEEngine(lambda_param=1.6)   # > 1.5
        with self.assertRaises(ValueError):
            NEVFEEngine(beta_param=-0.01)   # <= 0.0

        # Batch 2D decode step
        b_plus = torch.randn(2, 32)
        b_zero = torch.randn(2, 32)
        tokens, logits, dyn_lam, ent = engine.decode_step(b_plus, b_zero)
        self.assertEqual(tokens.shape, torch.Size([2]))
        self.assertEqual(logits.shape, torch.Size([2, 32]))
        self.assertEqual(dyn_lam.shape, torch.Size([2, 1]))
        self.assertEqual(ent.shape, torch.Size([2, 1]))

    def test_13_query_token_mask_utility(self):
        # Verifies that build_query_token_mask functions correctly with Vintern tokenizer
        from src.nevfe.adapter.vintern_adapter import build_query_token_mask
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("5CD-AI/Vintern-1B-v3_5", trust_remote_code=True)
        question = "ở đây có bán điện thoại của hãng nào ?"
        vocab_size = len(tokenizer)
        device = torch.device("cpu")

        q_mask = build_query_token_mask(question, tokenizer, vocab_size, device)

        # Upper-case 'Ở' (token 132774) and lower-case 'ở' (token 78820)
        self.assertTrue(q_mask[132774].item(), "Capitalized 'Ở' (132774) must be in q_mask")
        self.assertTrue(q_mask[78820].item(), "Lower-case 'ở' (78820) must be in q_mask")
        # Ensure answer entity ' Apple' is NOT in q_mask
        apple_ids = tokenizer.encode(" Apple", add_special_tokens=False)
        for aid in apple_ids:
            self.assertFalse(q_mask[aid].item(), f"Entity token {aid} (' Apple') must NOT be masked")


if __name__ == "__main__":
    unittest.main()
