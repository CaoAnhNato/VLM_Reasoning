# -*- coding: utf-8 -*-
r"""
Unit tests for NE-VFE Information-Geometric Variational Free Energy Decoding (IG-VFED) Engine.

Verifies:
1. AlgorithmConfig frozen dataclass defaults, bounds, and ablation_mask validation.
2. Theorem 2: Spatial information oblivion of Mean-Field Reference.
3. Bhattacharyya Coefficient computation with strict float32 clamping to [0.0, 1.0].
4. Theorem 5: Fisher-Rao Geodesic Dynamic Coupling Modulation.
5. Thermodynamic Shannon Entropy computation in nats.
6. Dynamic Geodesic Plausibility Boundary.
7. Theorem 3: Monotonic optical amplification under Mean-Field contrast.
8. Theorem 4: Natural conversational preamble suppression via prior likelihood ratio without regex.
9. Monochromatic / solid-color image invariance.
10. Unified nevfe_step_logits API across 1D and 2D inputs.
11. NEVFEEngine interface contracts and hyperparameter bounds.
12. Offline-safe query token mask casing and BPE prefix invariance utility.
13. Exact numerical parity (<= 1e-7) with 148f870 IG-VFED across all (comp_a, comp_b, comp_c) masks.
"""

import unittest
import math
import torch
import torch.nn.functional as F

from src.nevfe.core.engine import (
    AlgorithmConfig,
    NEVFEEngine,
    SPEC_REF,
    compute_bhattacharyya_coefficient,
    compute_dynamic_coupling,
    compute_dynamic_plausibility_floor,
    compute_mean_field_reference_embeddings,
    compute_optical_anchor_posterior,
    compute_unnormalized_entropy,
    nevfe_step_logits,
    plan_overlapped_strip_boxes,
    restore_wiener_optical_strokes,
    select_optical_foveal_loci,
)


class _OfflineMockTokenizer:
    """Deterministic offline tokenizer fallback for CPU unit tests."""

    def __init__(self):
        self._vocab = {
            "<|im_start|>": 1,
            "<|im_end|>": 2,
            "<IMG_CONTEXT>": 3,
            "Ở": 132774,
            "ở": 78820,
            " Apple": 9001,
            "đây": 1001,
            "có": 1002,
        }
        self.eos_token_id = 2

    def __len__(self):
        return 151936

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        ids = []
        for tok, tid in self._vocab.items():
            if tok in text:
                ids.append(tid)
        return ids or [abs(hash(text)) % 50000 + 10]

    def convert_tokens_to_ids(self, tok):
        return self._vocab.get(tok, 0)

    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(str(i) for i in ids)


def _reference_iter23_step(lp, lz, lam, bet, step_idx, comp_a, comp_b, comp_c):
    is_1d = lp.dim() == 1
    if is_1d:
        lp = lp.unsqueeze(0)
        lz = lz.unsqueeze(0)
    lp32 = lp.float()
    lz32 = lz.float()
    p_p = F.softmax(lp32, dim=-1)
    p_z = F.softmax(lz32, dim=-1)
    bc = torch.clamp(torch.sum(torch.sqrt(torch.clamp(p_p * p_z, min=0.0)), dim=-1, keepdim=True), 0.0, 1.0)
    if not comp_a:
        alpha_t = torch.full_like(bc, float(lam))
    else:
        alpha_t = float(lam) * torch.sqrt(torch.clamp(1.0 - bc * bc, 0.0, 1.0))
        if step_idx == 1 and comp_c:
            alpha_t = alpha_t + float(lam) * bc
    steered_pot = lp32 + alpha_t * (lp32 - lz32)
    beta_t = torch.full_like(bc, float(bet)) if not comp_b else torch.clamp(float(bet) * bc, 0.0, 1.0)
    max_p = torch.max(p_p, dim=-1, keepdim=True).values
    steered = torch.where(p_p >= beta_t * max_p, steered_pot, torch.full_like(steered_pot, -1e9))
    toks = torch.argmax(steered, dim=-1)
    ent = -torch.sum(torch.nan_to_num(p_p * F.log_softmax(lp32, dim=-1), nan=0.0, neginf=0.0, posinf=0.0), dim=-1, keepdim=True)
    if is_1d:
        return toks.squeeze(0), steered.squeeze(0), alpha_t.squeeze(0), ent.squeeze(0)
    return toks, steered, alpha_t, ent


class TestNEVFEEngineMFCD(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_01_algorithm_config_and_ablation_mask(self):
        cfg = AlgorithmConfig()
        self.assertEqual(cfg.lambda_param, 0.50)
        self.assertEqual(cfg.beta_param, 0.05)
        self.assertEqual(cfg.ablation_mask, (1, 1, 1))
        with self.assertRaises(ValueError):
            AlgorithmConfig(ablation_mask=(1, 1))
        with self.assertRaises(ValueError):
            AlgorithmConfig(lambda_param=-0.1)
        with self.assertRaises(ValueError):
            AlgorithmConfig(beta_param=0.0)

    def test_02_theorem_2_spatial_information_oblivion(self):
        N_tiles, N_tokens, D = 3, 256, 896
        vit_embeds = torch.randn(N_tiles, N_tokens, D, dtype=torch.float32)

        v_zero = compute_mean_field_reference_embeddings(vit_embeds)
        self.assertEqual(v_zero.shape, vit_embeds.shape)

        flat_zero = v_zero.reshape(N_tiles * N_tokens, D)
        spatial_var = torch.var(flat_zero, dim=0, unbiased=False)
        self.assertLess(spatial_var.max().item(), 1e-6, "V^0 must have zero spatial variance!")

        flat_embeds = vit_embeds.reshape(N_tiles * N_tokens, D)
        perm_idx = torch.randperm(N_tiles * N_tokens)
        vit_perm = flat_embeds[perm_idx].reshape(N_tiles, N_tokens, D)
        v_zero_perm = compute_mean_field_reference_embeddings(vit_perm)
        self.assertLess(torch.abs(v_zero - v_zero_perm).max().item(), 1e-5)

        v_2d = torch.randn(16, 64)
        v_2d_zero = compute_mean_field_reference_embeddings(v_2d)
        self.assertEqual(v_2d_zero.shape, v_2d.shape)
        with self.assertRaises(ValueError):
            compute_mean_field_reference_embeddings(torch.randn(1, 2, 3, 4))

    def test_03_bhattacharyya_coefficient_and_clamp(self):
        p = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        bc_ident = compute_bhattacharyya_coefficient(p, p)
        self.assertAlmostEqual(bc_ident.item(), 1.0, places=5)

        p1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        p2 = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        bc_disjoint = compute_bhattacharyya_coefficient(p1, p2)
        self.assertLess(bc_disjoint.item(), 1e-4)

        bc_overflow = compute_bhattacharyya_coefficient(p * 1.0001, p * 1.0001)
        self.assertLessEqual(bc_overflow.item(), 1.0)

        V = 151936
        p_disjoint_1 = torch.zeros(1, V)
        p_disjoint_2 = torch.zeros(1, V)
        p_disjoint_1[0, 0] = 1.0
        p_disjoint_2[0, 1] = 1.0
        bc_large_disjoint = compute_bhattacharyya_coefficient(p_disjoint_1, p_disjoint_2)
        self.assertEqual(bc_large_disjoint.item(), 0.0)

        delta = math.acos(0.90)
        p_overlap_1 = torch.zeros(1, V)
        p_overlap_2 = torch.zeros(1, V)
        p_overlap_1[0, 0] = math.cos(math.pi / 4.0 + delta) ** 2
        p_overlap_1[0, 1] = math.sin(math.pi / 4.0 + delta) ** 2
        p_overlap_2[0, 0] = 0.5
        p_overlap_2[0, 1] = 0.5
        bc_overlap = compute_bhattacharyya_coefficient(p_overlap_1, p_overlap_2)
        self.assertGreaterEqual(bc_overlap.item(), 0.89)
        self.assertLessEqual(bc_overlap.item(), 0.91)
        self.assertLess(bc_overlap.item(), 1.0)

        alpha_t = compute_dynamic_coupling(0.50, bc_overlap)
        self.assertGreater(alpha_t.item(), 0.15)

    def test_04_dynamic_coupling_and_float32_safeguard(self):
        bc_full = torch.tensor([[1.0]])
        lam_0 = compute_dynamic_coupling(0.50, bc_full, step_idx=2)
        self.assertAlmostEqual(lam_0.item(), 0.0, places=5)

        bc_zero = torch.tensor([[0.0]])
        lam_max = compute_dynamic_coupling(0.50, bc_zero, step_idx=2)
        self.assertAlmostEqual(lam_max.item(), 0.50, places=5)

        bc_mid = torch.tensor([[0.6]])
        lam_mid = compute_dynamic_coupling(0.50, bc_mid, step_idx=2)
        self.assertAlmostEqual(lam_mid.item(), 0.40, places=5)

        bc_eps_above = torch.tensor([[1.0000002]], dtype=torch.float32)
        lam_safe = compute_dynamic_coupling(0.50, bc_eps_above, step_idx=2)
        self.assertFalse(torch.isnan(lam_safe).any())
        self.assertEqual(lam_safe.item(), 0.0)

    def test_05_unnormalized_entropy(self):
        p_onehot = torch.tensor([[1.0, 0.0, 0.0]])
        log_p_onehot = torch.tensor([[0.0, -100.0, -100.0]])
        h_onehot = compute_unnormalized_entropy(p_onehot, log_p_onehot)
        self.assertAlmostEqual(h_onehot.item(), 0.0, places=5)

        k = 4
        p_unif = torch.full((1, k), 1.0 / k)
        log_p_unif = torch.full((1, k), math.log(1.0 / k))
        h_unif = compute_unnormalized_entropy(p_unif, log_p_unif)
        self.assertAlmostEqual(h_unif.item(), math.log(k), places=5)

    def test_06_adaptive_plausibility_truncation(self):
        beta = 0.05
        logits_p = torch.tensor([[3.0, 1.0, -2.0, -5.0]])
        logits_z = torch.tensor([[0.0, 0.0, 0.0, 0.0]])

        next_tok, steered, _, _ = nevfe_step_logits(
            logits_plus=logits_p,
            logits_zero=logits_z,
            lambda_param=0.50,
            beta_param=beta,
        )
        self.assertGreater(steered[0, 0].item(), -1e8)
        self.assertEqual(steered[0, 3].item(), -1e9)
        self.assertEqual(next_tok.item(), 0)

    def test_07_theorem_3_monotonic_optical_amplification(self):
        K = 20
        lp = torch.full((K,), 1.0)
        lz = torch.full((K,), 1.0)
        lp[2] = 5.0
        lz[2] = 0.0

        _, steered_00, _, _ = nevfe_step_logits(lp, lz, lambda_param=0.0, beta_param=0.05)
        _, steered_30, _, _ = nevfe_step_logits(lp, lz, lambda_param=0.30, beta_param=0.05)
        _, steered_60, _, _ = nevfe_step_logits(lp, lz, lambda_param=0.60, beta_param=0.05)
        _, steered_90, _, _ = nevfe_step_logits(lp, lz, lambda_param=0.90, beta_param=0.05)

        self.assertLess(steered_00[2].item(), steered_30[2].item())
        self.assertLess(steered_30[2].item(), steered_60[2].item())
        self.assertLess(steered_60[2].item(), steered_90[2].item())

    def test_08_theorem_4_natural_preamble_suppression(self):
        K = 50
        lp = torch.full((K,), -8.0)
        lz = torch.full((K,), -8.0)
        lp[0] = 3.0
        lz[0] = 5.0
        lp[1] = 2.4
        lz[1] = -2.0

        tok_base, _, _, _ = nevfe_step_logits(lp, lz, lambda_param=0.0, beta_param=0.05)
        self.assertEqual(tok_base.item(), 0)

        tok_steered, steered_logits, _, _ = nevfe_step_logits(lp, lz, lambda_param=0.50, beta_param=0.05)
        self.assertEqual(tok_steered.item(), 1)
        self.assertGreater(steered_logits[1].item(), steered_logits[0].item())

    def test_09_theorem_5_solid_color_monochromatic_invariance(self):
        K = 30
        lp = torch.randn(K)
        lz = lp.clone()

        _, steered_logits, _, _ = nevfe_step_logits(lp, lz, lambda_param=0.50, beta_param=0.05, step_idx=2)
        candidate_mask = F.softmax(lp, dim=-1) >= 0.05 * torch.max(F.softmax(lp, dim=-1))
        diff = torch.abs(steered_logits[candidate_mask] - lp[candidate_mask])
        self.assertLess(diff.max().item(), 1e-5)

    def test_10_nevfe_step_logits_1d_and_2d_api(self):
        lp_1d = torch.randn(64)
        lz_1d = torch.randn(64)
        tok_1d, logits_1d, _, _ = nevfe_step_logits(lp_1d, lz_1d)
        self.assertEqual(tok_1d.dim(), 0)
        self.assertEqual(logits_1d.shape, torch.Size([64]))
        self.assertFalse(torch.isnan(logits_1d).any())

        lp_2d = torch.randn(3, 128)
        lz_2d = torch.randn(3, 128)
        tok_2d, logits_2d, alpha_2d, ent_2d = nevfe_step_logits(lp_2d, lz_2d)
        self.assertEqual(tok_2d.shape, torch.Size([3]))
        self.assertEqual(logits_2d.shape, torch.Size([3, 128]))
        self.assertEqual(alpha_2d.shape, torch.Size([3, 1]))
        self.assertEqual(ent_2d.shape, torch.Size([3, 1]))

    def test_11_engine_class_interface_and_bounds(self):
        engine = NEVFEEngine(config=AlgorithmConfig())
        self.assertEqual(engine.lambda_param, 0.50)
        self.assertEqual(engine.beta_param, 0.05)

        engine_edge = NEVFEEngine(lambda_param=0.0, beta_param=0.01)
        self.assertEqual(engine_edge.lambda_param, 0.0)

        with self.assertRaises(ValueError):
            NEVFEEngine(lambda_param=-0.1)
        with self.assertRaises(ValueError):
            NEVFEEngine(lambda_param=1.6)
        with self.assertRaises(ValueError):
            NEVFEEngine(beta_param=-0.01)

        b_plus = torch.randn(2, 32)
        b_zero = torch.randn(2, 32)
        tokens, logits, dyn_lam, ent = engine.decode_step(b_plus, b_zero)
        self.assertEqual(tokens.shape, torch.Size([2]))
        self.assertEqual(logits.shape, torch.Size([2, 32]))
        self.assertEqual(dyn_lam.shape, torch.Size([2, 1]))
        self.assertEqual(ent.shape, torch.Size([2, 1]))

    def test_12_adapter_hook_cleanup_and_dynamic_image_size(self):
        from src.nevfe.adapter.vintern_adapter import VinternNEVFEAdapter, managed_forward_hooks

        class _DummyModule:
            def __init__(self):
                self.removed = False

            def register_forward_hook(self, fn):
                del fn

                class _Handle:
                    def __init__(self, outer):
                        self.outer = outer

                    def remove(self):
                        self.outer.removed = True

                return _Handle(self)

        mod = _DummyModule()
        with self.assertRaises(RuntimeError):
            with managed_forward_hooks([(mod, lambda *a: None)]):
                raise RuntimeError("boom")
        self.assertTrue(mod.removed)

        class _DummyCfg:
            force_image_size = 336
            max_dynamic_patch = 2

        class _DummyLM:
            config = _DummyCfg()

        class _DummyModel:
            config = _DummyCfg()
            language_model = _DummyLM()
            num_image_token = 64

        adapter = VinternNEVFEAdapter(model=_DummyModel(), tokenizer=_OfflineMockTokenizer(), device="cpu")
        self.assertEqual(adapter.force_image_size, 336)

    def test_13_query_token_mask_utility(self):
        from src.nevfe.adapter.vintern_adapter import build_query_token_mask
        from transformers import AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                "5CD-AI/Vintern-1B-v3_5",
                trust_remote_code=True,
                local_files_only=True,
            )
        except Exception:
            tokenizer = _OfflineMockTokenizer()

        question = "ở đây có bán điện thoại của hãng nào ?"
        vocab_size = len(tokenizer)
        device = torch.device("cpu")

        q_mask = build_query_token_mask(question, tokenizer, vocab_size, device)
        self.assertTrue(q_mask[132774].item(), "Capitalized 'Ở' (132774) must be in q_mask")
        self.assertTrue(q_mask[78820].item(), "Lower-case 'ở' (78820) must be in q_mask")
        apple_ids = tokenizer.encode(" Apple", add_special_tokens=False)
        for aid in apple_ids:
            self.assertFalse(q_mask[aid].item(), f"Entity token {aid} (' Apple') must NOT be masked")

    def test_14_iter23_equivalence_across_ablation_masks(self):
        lp = torch.randn(3, 64)
        lz = torch.randn(3, 64)
        for mask in [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]:
            cfg = AlgorithmConfig(lambda_param=0.50, beta_param=0.05, ablation_mask=mask)
            engine = NEVFEEngine(config=cfg)
            for step_idx in (1, 2):
                got = engine.decode_step(lp, lz, step_idx=step_idx)
                ref = _reference_iter23_step(lp, lz, 0.50, 0.05, step_idx, bool(mask[0]), bool(mask[1]), bool(mask[2]))
                self.assertTrue(torch.equal(got[0], ref[0]))
                self.assertLess((got[1] - ref[1]).abs().max().item(), 1e-7)
                self.assertLess((got[2] - ref[2]).abs().max().item(), 1e-7)
                self.assertLess((got[3] - ref[3]).abs().max().item(), 1e-7)


    def test_15_wsr_msf_gamma_param_and_pipeline_preservation(self):
        from src.nevfe.pipeline import NEVFEPipeline

        self.assertIn("spec_iter_7.md@v1.0", SPEC_REF)
        cfg = AlgorithmConfig(gamma_param=0.35)
        self.assertEqual(cfg.gamma_param, 0.35)
        with self.assertRaises(ValueError):
            AlgorithmConfig(gamma_param=-0.05)
        with self.assertRaises(ValueError):
            AlgorithmConfig(gamma_param=1.05)
        with self.assertRaises(ValueError):
            AlgorithmConfig(ablation_mask=(1, 1))

        class _MockCfg:
            force_image_size = 448
            max_dynamic_patch = 2

        class _MockModel:
            config = _MockCfg()
            language_model = type("_LM", (), {"config": _MockCfg()})()
            num_image_token = 16

        pipe = NEVFEPipeline(model=_MockModel(), tokenizer=_OfflineMockTokenizer(), device="cpu", gamma_param=0.72)
        self.assertEqual(pipe.gamma_param, 0.72)
        pipe.set_ablation_switches(comp_a=False, comp_b=True, comp_c=True)
        self.assertEqual(pipe.config.gamma_param, 0.72)
        self.assertEqual(pipe.adapter.engine.gamma_param, 0.72)
        self.assertEqual(pipe.config.ablation_mask, (0, 1, 1))

    def test_16_wsr_msf_eq1_kl_specificity_soft_or_and_edge_cases(self):
        with self.assertRaises(ValueError):
            compute_optical_anchor_posterior(torch.zeros((4, 4)), torch.zeros((2, 4)))

        const_v = torch.ones((4, 4, 8), dtype=torch.float32) * 3.0
        const_v[0, 0, 0] += 1e-7
        with self.assertRaises(ValueError):
            compute_optical_anchor_posterior(const_v, torch.randn(2, 5))

        unif_4x4 = torch.full((4, 4), 1.0 / 16.0)
        self.assertTrue(torch.allclose(compute_optical_anchor_posterior(const_v, None), unif_4x4, atol=1e-6))
        self.assertTrue(torch.allclose(compute_optical_anchor_posterior(const_v, torch.zeros((0, 8))), unif_4x4, atol=1e-6))
        self.assertTrue(torch.allclose(compute_optical_anchor_posterior(torch.ones((1, 1, 8)), torch.ones((2, 8))), torch.ones((1, 1)), atol=1e-6))
        self.assertTrue(torch.allclose(compute_optical_anchor_posterior(const_v, torch.randn(3, 8)), unif_4x4, atol=1e-6))
        self.assertTrue(torch.allclose(compute_optical_anchor_posterior(const_v, torch.zeros((2, 8))), unif_4x4, atol=1e-6))

        nan_v = torch.randn((4, 4, 8))
        nan_v[0, 0, 0] = float("nan")
        nan_v[1, 1, 1] = float("inf")
        p_nan = compute_optical_anchor_posterior(nan_v, torch.full((2, 8), float("nan")))
        self.assertTrue(torch.isfinite(p_nan).all())
        self.assertTrue(torch.allclose(p_nan, unif_4x4, atol=1e-6))

        # Clutter rejection: 2 target patches vs 64 clutter patches on 16x16 grid
        patch_v = torch.zeros((16, 16, 16), dtype=torch.float32)
        patch_v[:, :, 0] = 1.0
        patch_v[3, 4:6, 1] = 1e30  # FLT_MAX-scale immunity
        patch_v[8:16, 8:16, 2] = 1e30
        q_mixed = torch.zeros((2, 16), dtype=torch.float32)
        q_mixed[0, 1] = 1e30  # narrow specific target query token
        q_mixed[1, 2] = 1e30  # broad 64-patch clutter query token

        p_kl = compute_optical_anchor_posterior(patch_v, q_mixed, comp_a=True)
        p_max = compute_optical_anchor_posterior(patch_v, q_mixed, comp_a=False)
        self.assertTrue(torch.isfinite(p_kl).all())
        self.assertGreater(float(p_kl[3, 4:6].sum().item()), 0.90)
        self.assertLess(float(p_max[3, 4:6].sum().item()), 0.15)

        # Constant S columns hit sig_m <= tau_v branch cleanly
        flat_s_v = torch.zeros((4, 4, 8), dtype=torch.float32)
        flat_s_v[:, :, 0] = torch.linspace(1.0, 2.0, 16).view(4, 4)
        self.assertTrue(torch.allclose(compute_optical_anchor_posterior(flat_s_v, torch.tensor([[0.0, 1.0] + [0.0] * 6]), comp_a=True), unif_4x4, atol=1e-6))
        self.assertTrue(torch.allclose(compute_optical_anchor_posterior(flat_s_v, torch.tensor([[0.0, 1.0] + [0.0] * 6]), comp_a=False), unif_4x4, atol=1e-6))

    def test_17_wsr_msf_eq2_two_locus_conditional_exterior_and_boundaries(self):
        with self.assertRaises(ValueError):
            select_optical_foveal_loci(torch.zeros((2, 2, 2)))

        empty_loci = select_optical_foveal_loci(torch.zeros((0, 0)))
        self.assertEqual(empty_loci[0][0], (0, 0, 0, 0))
        one_loci = select_optical_foveal_loci(torch.ones((1, 1)))
        self.assertEqual(one_loci[0], ((0, 0, 1, 1), 0.0, 1.0, 1.0))

        unif_loci = select_optical_foveal_loci(torch.full((8, 8), 1.0 / 64.0))
        self.assertEqual(len(unif_loci), 1)
        self.assertEqual(unif_loci[0][1], 0.0)

        nan_post = torch.tensor([[float("nan"), float("inf")], [float("-inf"), 0.0]])
        self.assertEqual(select_optical_foveal_loci(nan_post)[0][1], 0.0)

        # Bimodal two-sign posterior -> 2 spatially disjoint loci under comp_a=True, 1 locus under comp_a=False
        bimodal = torch.zeros((16, 16), dtype=torch.float32)
        bimodal[2:4, 2:4] = 0.55
        bimodal[10:12, 11:13] = 0.45
        loci_2 = select_optical_foveal_loci(bimodal, comp_a=True)
        loci_1 = select_optical_foveal_loci(bimodal, comp_a=False)
        self.assertEqual(len(loci_2), 2)
        self.assertEqual(len(loci_1), 1)
        (b1, j1, _, _), (b2, j2, _, _) = loci_2
        self.assertGreater(j1, 0.0)
        self.assertGreaterEqual(j2, 0.5 * j1)
        disjoint = (b2[2] <= b1[0]) or (b2[0] >= b1[2]) or (b2[3] <= b1[1]) or (b2[1] >= b1[3])
        self.assertTrue(disjoint)

        # Single-peak posterior stops at 1 locus even when comp_a=True
        single = torch.zeros((8, 8), dtype=torch.float32)
        single[1:3, 2:4] = 1.0
        loci_s = select_optical_foveal_loci(single, comp_a=True)
        self.assertEqual(len(loci_s), 1)
        self.assertEqual(loci_s[0][0], (1, 2, 3, 4))

    def test_18_wsr_msf_eq3_multi_strip_and_eq4_wiener_restoration(self):
        self.assertEqual(plan_overlapped_strip_boxes((0, 0, 10, 10), (0, 100), 28.0), ([], (1, 1), 0.0))
        self.assertEqual(plan_overlapped_strip_boxes((0, 0, 10, 10), (100, 100), 28.0, max_tiles=0), ([], (1, 1), 0.0))

        # Horizontal 180 x 20 strip: comp_b=True -> (3, 1), sub_span=72.0 (6.22x at 448), 25% overlap keeps 11/11 digits unbroken
        boxes_b, grid_b, span_b = plan_overlapped_strip_boxes((10, 100, 190, 120), (400, 300), patch_stride=28.0, max_tiles=3, comp_b=True)
        self.assertEqual(grid_b, (3, 1))
        self.assertEqual(len(boxes_b), 3)
        self.assertAlmostEqual(span_b, 72.0, places=5)
        overlap_01 = boxes_b[0][2] - boxes_b[1][0]
        self.assertAlmostEqual(overlap_01 / float(boxes_b[0][2] - boxes_b[0][0]), 0.25, places=3)
        for d_x in [int(round(14 + k * (162.0 / 10.0))) for k in range(11)]:
            self.assertTrue(any(bx1 <= d_x - 3 and d_x + 3 <= bx2 for bx1, _, bx2, _ in boxes_b))

        boxes_no_b, grid_no_b, span_no_b = plan_overlapped_strip_boxes((10, 100, 190, 120), (400, 300), patch_stride=28.0, max_tiles=3, comp_b=False)
        self.assertEqual(grid_no_b, (1, 1))
        self.assertGreater(span_no_b, 179.0)

        # Tiny 42 x 31 glyph -> 56px floor (8.00x) vs 112px (4.00x)
        _, tiny_grid, tiny_span = plan_overlapped_strip_boxes((50, 50, 92, 81), (300, 300), patch_stride=28.0, max_tiles=3, comp_b=True)
        _, _, tiny_span_off = plan_overlapped_strip_boxes((50, 50, 92, 81), (300, 300), patch_stride=28.0, max_tiles=3, comp_b=False)
        self.assertEqual(tiny_grid, (1, 1))
        self.assertAlmostEqual(tiny_span, 56.0, places=5)
        self.assertAlmostEqual(tiny_span_off, 112.0, places=5)

        # Eq. (4) Wiener stroke restoration contracts & boundary guards
        with self.assertRaises(ValueError):
            restore_wiener_optical_strokes(torch.zeros((8, 8)))
        self.assertEqual(restore_wiener_optical_strokes(torch.zeros((0, 8, 8))).shape, (0, 8, 8))
        px1 = torch.rand((3, 1, 1))
        self.assertTrue(torch.equal(restore_wiener_optical_strokes(px1), px1))

        # 1xN / Nx1 strips exercise single-axis smoothing paths without error
        self.assertEqual(restore_wiener_optical_strokes(torch.rand((3, 1, 16)), magnification=4.0).shape, (3, 1, 16))
        self.assertEqual(restore_wiener_optical_strokes(torch.rand((3, 16, 1)), magnification=4.0).shape, (3, 16, 1))
        # Ultra-narrow image (< patch_stride) triggers adm empty -> [(1, 1)] fallback cleanly
        self.assertEqual(plan_overlapped_strip_boxes((0, 0, 12, 12), (16, 16), patch_stride=28.0, comp_b=True)[1], (1, 1))

        # Exact 0-bit identity at M=1.0, gamma=0.0, comp_c=False; NaN/Inf safe sanitization
        sample_rgb = torch.rand((3, 24, 24))
        self.assertTrue(torch.equal(restore_wiener_optical_strokes(sample_rgb, magnification=1.0, comp_c=True), sample_rgb))
        self.assertTrue(torch.equal(restore_wiener_optical_strokes(sample_rgb, magnification=4.0, gamma_param=0.0, comp_c=True), sample_rgb))
        self.assertTrue(torch.equal(restore_wiener_optical_strokes(sample_rgb, magnification=4.0, comp_c=False), sample_rgb))
        nan_rgb = sample_rgb.clone()
        nan_rgb[0, 0, 0], nan_rgb[1, 1, 1], nan_rgb[2, 2, 2] = float("nan"), float("inf"), float("-inf")
        self.assertTrue(torch.isfinite(restore_wiener_optical_strokes(nan_rgb, magnification=4.0)).all())

        # Boosts inner glyph counter contrast while keeping low-level noisy flat background < 0.5/255 & preserving chromaticity
        torch.manual_seed(0)
        glyph = torch.full((3, 32, 32), 0.85, dtype=torch.float32)
        glyph[0, :, :] = 0.90  # non-neutral R>G>B tint to test opponent chromaticity invariance
        glyph[:, 8:24, 8:24] = 0.15  # dark glyph body
        glyph[:, 13:19, 13:19] = 0.75  # inner '9' counter hole
        # Add mild sensor noise on background, keeping interior away from clamping
        noise = torch.randn((3, 32, 32)) * 0.002
        noisy_glyph = torch.clamp(glyph + noise, 0.0, 1.0)
        restored = restore_wiener_optical_strokes(noisy_glyph, magnification=8.0, gamma_param=0.50, comp_c=True)
        # Inner counter brightens vs stroke rim
        pre_contrast = float(noisy_glyph[:, 15:17, 15:17].mean() - noisy_glyph[:, 9:11, 9:11].mean())
        post_contrast = float(restored[:, 15:17, 15:17].mean() - restored[:, 9:11, 9:11].mean())
        self.assertGreater(post_contrast, pre_contrast)
        # Flat noisy border perturbation stays sub-quantization (< 0.5 / 255)
        bg_delta = float((restored[:, 1:5, 1:5] - noisy_glyph[:, 1:5, 1:5]).abs().max().item())
        self.assertLess(bg_delta, 0.5 / 255.0)
        # Opponent chromaticity (R - G) preserved away from [0, 1] clamp
        rg_pre = noisy_glyph[0, 15:17, 15:17] - noisy_glyph[1, 15:17, 15:17]
        rg_post = restored[0, 15:17, 15:17] - restored[1, 15:17, 15:17]
        self.assertLess(float((rg_post - rg_pre).abs().max().item()), 1e-6)

    def test_19_adapter_immutability_ablation_masks_and_pipeline_diagnostics(self):
        import time
        from PIL import Image, ImageDraw
        from src.nevfe.adapter.vintern_adapter import VinternNEVFEAdapter
        from src.nevfe.pipeline import NEVFEPipeline

        class _MockCfg:
            force_image_size = 112
            max_dynamic_patch = 4

        class _MockLMCore:
            def __init__(self, vocab_size=64, hidden=16):
                self.vocab_size = vocab_size
                self.hidden = hidden
                self.seen_seq_lens = []

            def __call__(self, inputs_embeds, past_key_values=None, use_cache=True, return_dict=True):
                del past_key_values, use_cache, return_dict
                self.seen_seq_lens.append(int(inputs_embeds.shape[1]))
                s_len = int(inputs_embeds.shape[1])
                hidden_out = inputs_embeds[:, -1:, :]
                return type("_Out", (), {"last_hidden_state": hidden_out, "past_key_values": ("kv", s_len)})()

        class _MockLM:
            config = _MockCfg()

            def __init__(self, vocab_size=64, hidden=16):
                torch.manual_seed(7)
                self.emb = torch.nn.Embedding(vocab_size, hidden)
                self.model = _MockLMCore(vocab_size=vocab_size, hidden=hidden)
                self.vocab_size = vocab_size

            def get_input_embeddings(self):
                return self.emb

            def lm_head(self, x):
                out = torch.zeros((x.shape[0], x.shape[1], self.vocab_size), dtype=torch.float32)
                out[:, :, 2] = 5.0
                return out

        class _MockTok:
            eos_token_id = 2

            def encode(self, text, add_special_tokens=False):
                del add_special_tokens
                if not text:
                    return []
                return [5, 6] if ("bao nhiêu" in text or "ở đây" in text) else [7, 8]

            def convert_tokens_to_ids(self, tok):
                return {"<IMG_CONTEXT>": 3, "<|im_end|>": 2}.get(tok, 0)

            def __call__(self, text, return_tensors="pt", add_special_tokens=False):
                del return_tensors, add_special_tokens
                n_img = text.count("<IMG_CONTEXT>")
                ids = [1] + ([3] * n_img) + [4] if n_img > 0 else [5, 6, 7]
                return type("_Enc", (), {"input_ids": torch.tensor([ids], dtype=torch.long)})()

            def decode(self, ids, skip_special_tokens=True):
                del skip_special_tokens
                return "ok"

        class _MockModel:
            config = _MockCfg()
            num_image_token = 16

            def __init__(self):
                self.language_model = _MockLM(vocab_size=64, hidden=16)
                self.extract_calls = []

            def extract_feature(self, pixel_values):
                self.extract_calls.append(int(pixel_values.shape[0]))
                n_t = int(pixel_values.shape[0])
                feats = torch.zeros((n_t, 16, 16), dtype=torch.float32)
                # Inject localized target alignment matching token 5 embedding on patch 5
                tok5 = self.language_model.emb.weight[5].detach()
                feats[0, 5, :] = tok5 * 5.0
                return feats

        model = _MockModel()
        adapter = VinternNEVFEAdapter(model=model, tokenizer=_MockTok(), device="cpu")
        pipe = NEVFEPipeline(model=model, tokenizer=_MockTok(), device="cpu")

        # Non-square ViT token check
        with self.assertRaises(ValueError):
            adapter._extract_lattice(Image.new("RGB", (64, 64)), torch.zeros((1, 15, 16)))

        # Nyquist gate rejects full-image box, accepts focused banner box
        sq_img = Image.new("RGB", (224, 224), color=(240, 240, 240))
        calls_pre = len(model.extract_calls)
        self.assertIsNone(adapter.encode_transient_roi_crops(sq_img, [(0, 0, 224, 224)], max_tiles=3, min_magnification=0.5))
        self.assertEqual(len(model.extract_calls), calls_pre)

        banner_res = adapter.encode_transient_roi_crops(sq_img, [(10, 90, 210, 118)], max_tiles=3, min_magnification=0.5)
        self.assertIsNotNone(banner_res)
        self.assertEqual(model.extract_calls[-1], 3)

        # (0, 0, 0) skips foveation cleanly; all Stage-A LOO masks (111, 011, 101, 110) populate foveal_bboxes
        img = Image.new("RGB", (240, 200), color=(240, 240, 240))
        ImageDraw.Draw(img).rectangle([40, 40, 90, 70], fill=(20, 20, 20))
        v1 = pipe.extract_visual_features(img).clone()

        calls_zero = len(model.extract_calls)
        pipe.set_ablation_switches(comp_a=False, comp_b=False, comp_c=False)
        pipe.run(img, "ở đây ghi chữ gì?", max_new_tokens=2)
        self.assertEqual(len(model.extract_calls), calls_zero)
        self.assertEqual(pipe.adapter._last_foveal_bboxes, [])

        for mask in [(1, 1, 1), (0, 1, 1), (1, 0, 1), (1, 1, 0)]:
            pipe.set_ablation_switches(*mask)
            res = pipe.run(img, "giá bao nhiêu?", max_new_tokens=2)
            self.assertEqual(res.stage1_bbox, [0, 0, 240, 200])
            self.assertGreaterEqual(len(res.diagnostics["foveal_bboxes"]), 1)
        self.assertTrue(torch.equal(v1, pipe.adapter._cached_vit_embeds))

        # CPU single-step WSR-MSF engine + Wiener restoration latency < 5.0 ms/step
        p_test = torch.randn((16, 16, 64))
        q_test = torch.randn((4, 64))
        rgb_test = torch.rand((3, 48, 48))
        eng = NEVFEEngine()
        t0 = time.perf_counter()
        for _ in range(20):
            eng.foveate(p_test, q_test)
            plan_overlapped_strip_boxes((10, 10, 120, 30), (300, 300), 28.0)
            restore_wiener_optical_strokes(rgb_test, magnification=4.0)
        ms_per_step = ((time.perf_counter() - t0) * 1000.0) / 20.0
        self.assertLess(ms_per_step, 5.0)


if __name__ == "__main__":
    unittest.main()
