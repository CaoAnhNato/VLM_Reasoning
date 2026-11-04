# -*- coding: utf-8 -*-
"""
Unit tests for NE-VFE Core Engine with Free-Energy Support Truncation (FEST).

Verifies:
1. Fisher-Saliency Patch Energy Modulation on 2D and 3D tensors.
2. Bhattacharyya Coefficient and Fisher-Rao Geodesic Dynamic Coupling.
3. Thermodynamic Shannon Entropy computation in nats.
4. Free-Energy Support Truncation (FEST) support mask:
   - Proposition 3 Dual Equivalence between potential and probability truncation.
   - Unconditional retention of maximal potential token.
5. End-to-end Single-Step FEST Operator:
   - Theorem 1 & Theorem 2: Suppression of conversational preamble echo in favor of high-VMI visual entity.
   - 1D and 2D batch dimension preservation.
   - Numerical stability (no NaNs, no Infs).
6. NEVFEEngine interface contracts and boundary validation.
"""

import unittest
import math
import torch
import torch.nn.functional as F

from src.nevfe.core.engine import (
    NEVFEEngine,
    modulate_patch_embeddings,
    compute_bhattacharyya_coefficient,
    compute_dynamic_coupling,
    compute_unnormalized_entropy,
    compute_free_energy_support_mask,
    compute_candidate_support_mask,
    nevfe_step_logits,
)


class TestNEVFEEngineFEST(unittest.TestCase):
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

    def test_02_bhattacharyya_coefficient(self):
        # Identical distributions should give BC = 1.0
        p = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        bc_ident = compute_bhattacharyya_coefficient(p, p)
        self.assertAlmostEqual(bc_ident.item(), 1.0, places=5)

        # Disjoint distributions should give BC ~ 0.0
        p1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        p2 = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        bc_disjoint = compute_bhattacharyya_coefficient(p1, p2)
        self.assertLess(bc_disjoint.item(), 1e-4)

    def test_03_dynamic_coupling(self):
        # When BC = 1.0 (total agreement), visual guidance dynamic_lambda should be 0.0
        bc_full = torch.tensor([[1.0]])
        lam_0 = compute_dynamic_coupling(0.45, bc_full)
        self.assertAlmostEqual(lam_0.item(), 0.0, places=5)

        # When BC = 0.0 (total disagreement), dynamic_lambda should equal lambda_param
        bc_zero = torch.tensor([[0.0]])
        lam_max = compute_dynamic_coupling(0.45, bc_zero)
        self.assertAlmostEqual(lam_max.item(), 0.45, places=5)

        # Intermediate BC: lambda * sqrt(1 - BC^2)
        bc_mid = torch.tensor([[0.6]])
        lam_mid = compute_dynamic_coupling(0.45, bc_mid)
        # sqrt(1 - 0.36) = 0.8 -> 0.45 * 0.8 = 0.36
        self.assertAlmostEqual(lam_mid.item(), 0.36, places=5)

    def test_04_unnormalized_entropy(self):
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

    def test_05_proposition_3_dual_equivalence(self):
        # Proposition 3: log q^*(v) >= max log q^*(w) - H(q^*)
        # is identically equivalent to z^*(v) >= max z^*(w) - H(q^*) / beta
        beta = 1.5
        z_star = torch.tensor([[2.5, -0.5, 4.0, 1.2, -3.0]])
        scaled_z = beta * z_star
        log_q_star = F.log_softmax(scaled_z, dim=-1)
        q_star = torch.exp(log_q_star)
        entropy = compute_unnormalized_entropy(q_star, log_q_star)

        mask_from_log_q = compute_free_energy_support_mask(log_q_star, entropy)
        mask_from_potential = compute_candidate_support_mask(z_star, entropy, beta_param=beta)

        self.assertTrue(torch.equal(mask_from_log_q, mask_from_potential))
        # The argmax token (index 2) must be unconditionally included
        self.assertTrue(mask_from_log_q[0, 2].item())

    def test_06_preamble_suppression_and_entity_crossover(self):
        # Realistic simulation of Question 248 ("Apple" entity vs "Ở" preamble echo)
        # Token 0: preamble echo "Ở" (high language prior, non-positive VMI)
        # Token 1: visual entity "Apple" (lower language prior, high VMI from image)
        # Tokens 2..49: other visual scene text candidates in p+
        # Tokens 50..99: other text prior candidates in p0
        K = 100
        lp = torch.full((K,), -10.0)
        lz = torch.full((K,), -10.0)
        lp[0] = math.log(0.62); lz[0] = math.log(0.70)
        lp[1] = math.log(0.05); lz[1] = math.log(1e-5)
        lp[2:50] = math.log((1.0 - 0.62 - 0.05) / 48)
        lz[50:100] = math.log((1.0 - 0.70 - 1e-5) / 50)

        # 1. Verify that Old Pre-Steering Support Mask wrongly pruned the entity token
        log_pp = F.log_softmax(lp, dim=-1)
        pp = torch.exp(log_pp)
        h_pp = compute_unnormalized_entropy(pp, log_pp)
        old_support_mask = log_pp >= (torch.max(log_pp) - h_pp / 1.5)
        self.assertFalse(old_support_mask[1].item(), "Old pre-steering mask wrongly pruned visual entity token!")
        self.assertTrue(old_support_mask[0].item(), "Old pre-steering mask kept preamble token.")

        # 2. Verify that FEST retains the entity token and prunes the preamble
        next_tok, steered_logits, dynamic_lambda, entropy_nats = nevfe_step_logits(
            logits_plus=lp,
            logits_zero=lz,
            lambda_param=0.45,
            beta_param=1.5,
        )

        # Under FEST, visual entity token (index 1) must defeat preamble token (index 0)
        self.assertEqual(next_tok.item(), 1, "FEST should select entity token over preamble!")
        self.assertGreater(steered_logits[1].item(), steered_logits[0].item(), "Entity potential must exceed preamble!")
        self.assertGreater(steered_logits[1].item(), -1e9, "Entity token must be in admissible support set!")
        self.assertEqual(steered_logits[50].item(), -1e9, "Unadmissible candidate token correctly pruned to -1e9!")
        self.assertFalse(torch.isnan(steered_logits).any())

    def test_07_engine_class_interface(self):
        engine = NEVFEEngine(lambda_param=1.0, beta_param=1.5)
        self.assertEqual(engine.lambda_param, 1.0)
        self.assertEqual(engine.beta_param, 1.5)

        # Boundary acceptance test
        engine_upper = NEVFEEngine(lambda_param=1.5, beta_param=1.5)
        self.assertEqual(engine_upper.lambda_param, 1.5)

        # Invalid hyperparameter range tests
        with self.assertRaises(ValueError):
            NEVFEEngine(lambda_param=0.05)  # < 0.1
        with self.assertRaises(ValueError):
            NEVFEEngine(lambda_param=1.6)   # > 1.5
        with self.assertRaises(ValueError):
            NEVFEEngine(beta_param=0.1)     # < 0.5
        with self.assertRaises(ValueError):
            NEVFEEngine(beta_param=6.0)     # > 5.0

        # Batch 2D decode step
        b_plus = torch.randn(2, 32)
        b_zero = torch.randn(2, 32)
        tokens, logits, dyn_lam, ent = engine.decode_step(b_plus, b_zero)
        self.assertEqual(tokens.shape, torch.Size([2]))
        self.assertEqual(logits.shape, torch.Size([2, 32]))
        self.assertEqual(dyn_lam.shape, torch.Size([2, 1]))
        self.assertEqual(ent.shape, torch.Size([2, 1]))

    def test_08_mfvr_spatial_centroid_properties(self):
        # Proposition 2: Spatial Centroid Reference exhibits zero spatial variance
        N_tiles, N_tokens, D = 3, 256, 896
        vit_embeds = torch.randn(N_tiles, N_tokens, D, dtype=torch.bfloat16)

        v_mean = vit_embeds.mean(dim=(0, 1), keepdim=True)
        vit_embeds_zero = v_mean.expand_as(vit_embeds)

        self.assertEqual(vit_embeds_zero.shape, vit_embeds.shape)
        # Mean across spatial positions must match v_mean
        diff_mean = torch.abs(vit_embeds_zero.float().mean(dim=(0, 1), keepdim=True) - v_mean.float())
        self.assertLess(diff_mean.max().item(), 1e-5)

        # Spatial variance across all (tiles, tokens) must be identically zero
        flat_zero = vit_embeds_zero.float().reshape(N_tiles * N_tokens, D)
        spatial_var = torch.var(flat_zero, dim=0, unbiased=False)
        self.assertLess(spatial_var.max().item(), 1e-6, "MFVR reference field must have zero spatial variance!")

    def test_09_direct_visual_coupling_theorem_1(self):
        # Theorem 1-MFVR: Exact Preamble Extinction under Direct Visual Coupling
        # Case from empirical GPU logit measurements (Question 74):
        # Preamble 'C' (id 0): p+ = 0.6901, p0 = 0.4650 -> VMI = +0.3947 nats
        # Entity 'NH' (id 1): p+ = 0.0344, p0 = 0.0004 -> VMI = +4.4572 nats
        K = 100
        lp = torch.full((K,), -12.0)
        lz = torch.full((K,), -12.0)
        lp[0] = math.log(0.6901); lz[0] = math.log(0.4650)
        lp[1] = math.log(0.0344); lz[1] = math.log(0.0004)
        lp[2:] = math.log((1.0 - 0.6901 - 0.0344) / (K - 2))
        lz[2:] = math.log((1.0 - 0.4650 - 0.0004) / (K - 2))

        # At lambda = 1.0, beta = 1.5:
        next_tok, steered_logits, coupling, entropy = nevfe_step_logits(
            logits_plus=lp,
            logits_zero=lz,
            lambda_param=1.0,
            beta_param=1.5,
        )

        # Entity (index 1) must defeat preamble (index 0)
        self.assertEqual(next_tok.item(), 1, "Entity token must dominate argmax under lambda=1.0!")
        margin = (steered_logits[1] - steered_logits[0]).item()
        self.assertGreater(margin, 1.0, f"Entity margin must exceed 1.0 nats, got {margin:.3f}")
        self.assertAlmostEqual(coupling.item(), 1.0, places=5)

    def test_10_qo_vfe_barrier_suppression(self):
        # Case representing Question 248:
        # Token 0: query preamble token (e.g. "Ở", in q_mask)
        # Token 1: genuine visual entity token (e.g. "Apple", not in q_mask)
        K = 100
        lp = torch.full((K,), -10.0)
        lz = torch.full((K,), -10.0)
        # Token 0 (preamble): log_p_plus = -0.47, log_p_zero = -3.26 -> VMI = +2.79 nats
        lp[0] = -0.47
        lz[0] = -3.26
        # Token 1 (entity): log_p_plus = -3.22, log_p_zero = -6.91 -> VMI = +3.69 nats
        lp[1] = -3.22
        lz[1] = -6.91

        q_mask = torch.zeros(K, dtype=torch.bool)
        q_mask[0] = True  # Token 0 is in the question prompt

        # 1. Without barrier (q_mask=None or mu=0.0): preamble token 0 wins
        tok_no_barrier, logits_no_barrier, _, _ = nevfe_step_logits(
            logits_plus=lp,
            logits_zero=lz,
            lambda_param=1.0,
            beta_param=1.0,
            mu_param=0.0,
            q_mask=None,
            step_idx=1,
        )
        self.assertEqual(tok_no_barrier.item(), 0, "Without barrier, preamble token should dominate argmax")

        # 2. With QO-VFE barrier (q_mask active, mu=3.0, step=1):
        tok_barrier, logits_barrier, _, _ = nevfe_step_logits(
            logits_plus=lp,
            logits_zero=lz,
            lambda_param=1.0,
            beta_param=1.0,
            mu_param=3.0,
            q_mask=q_mask,
            step_idx=1,
        )
        self.assertEqual(tok_barrier.item(), 1, "With QO-VFE barrier at step 1, entity token must defeat preamble!")
        self.assertGreater(logits_barrier[1].item(), logits_barrier[0].item(), "Entity logit must exceed preamble logit")
        margin = (logits_barrier[1] - logits_barrier[0]).item()
        self.assertGreater(margin, 0.5, f"Entity margin should be substantial, got {margin:.3f}")

    def test_11_qo_vfe_barrier_decay(self):
        # Verifies the transient dyadic decay schedule of QO-VFE barrier:
        # step 1: barrier = 3.0 * (0.5)^0 = 3.0
        # step 2: barrier = 3.0 * (0.5)^1 = 1.5
        # step 3: barrier = 3.0 * (0.5)^2 = 0.75
        # step 4: step_idx > barrier_horizon (3) -> barrier = 0.0
        K = 50
        lp = torch.randn(K)
        lz = torch.randn(K)
        q_mask = torch.zeros(K, dtype=torch.bool)
        q_mask[5] = True
        q_mask[10] = True

        # At step t=4, barrier is inactive (barrier_horizon=3), output must match standard NE-VFE (q_mask=None)
        tok_t4, logits_t4, _, _ = nevfe_step_logits(
            logits_plus=lp,
            logits_zero=lz,
            lambda_param=1.0,
            beta_param=1.0,
            mu_param=3.0,
            q_mask=q_mask,
            step_idx=4,
            barrier_horizon=3,
        )
        tok_none, logits_none, _, _ = nevfe_step_logits(
            logits_plus=lp,
            logits_zero=lz,
            lambda_param=1.0,
            beta_param=1.0,
            mu_param=3.0,
            q_mask=None,
            step_idx=4,
            barrier_horizon=3,
        )
        self.assertEqual(tok_t4.item(), tok_none.item())
        self.assertTrue(torch.allclose(logits_t4, logits_none, atol=1e-6), "At step t=4, logits must match standard NE-VFE exactly!")

        # Verify step 1 has full barrier 3.0 on masked tokens
        lp_supp = lp.clone()
        lz_supp = lz.clone()
        lp_supp[5] = 10.0
        lz_supp[5] = 10.0
        _, logits_t1_mask, _, _ = nevfe_step_logits(lp_supp, lz_supp, lambda_param=1.0, beta_param=1.0, mu_param=3.0, q_mask=q_mask, step_idx=1)
        _, logits_t1_nomask, _, _ = nevfe_step_logits(lp_supp, lz_supp, lambda_param=1.0, beta_param=1.0, mu_param=3.0, q_mask=None, step_idx=1)
        diff_masked = (logits_t1_nomask[5] - logits_t1_mask[5]).item()
        self.assertAlmostEqual(diff_masked, 3.0, places=4, msg="Barrier at step 1 on masked token must equal exactly 3.0 nats")

    def test_12_query_token_mask_casing_coverage(self):
        # Directive 4: Verifies that capitalized 'Ở' (132774) and lower-case 'ở' (78820)
        # are both present in q_mask when the question begins with "ở".
        from src.nevfe.adapter.vintern_adapter import build_query_token_mask
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("5CD-AI/Vintern-1B-v3_5", trust_remote_code=True)
        question = "ở đây có bán điện thoại của hãng nào ?"
        vocab_size = len(tokenizer)
        device = torch.device("cpu")

        q_mask = build_query_token_mask(question, tokenizer, vocab_size, device)

        # Upper-case 'Ở' (token 132774)
        self.assertTrue(q_mask[132774].item(), "Capitalized 'Ở' (132774) must be in q_mask")
        # Lower-case 'ở' (token 78820)
        self.assertTrue(q_mask[78820].item(), "Lower-case 'ở' (78820) must be in q_mask")
        # Ensure legitimate answer entity ' Apple' is NOT in q_mask
        apple_ids = tokenizer.encode(" Apple", add_special_tokens=False)
        for aid in apple_ids:
            self.assertFalse(q_mask[aid].item(), f"Entity token {aid} (' Apple') must NOT be masked")


if __name__ == "__main__":
    unittest.main()
