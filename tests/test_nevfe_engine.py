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
    compute_bhattacharyya_coefficient,
    compute_dynamic_coupling,
    compute_dynamic_plausibility_floor,
    compute_mean_field_reference_embeddings,
    compute_signboard_manifold_posterior,
    compute_spb_anchor_posterior,
    compute_unnormalized_entropy,
    nevfe_step_logits,
    select_submodular_bregman_loci,
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


    def test_15_gamma_param_default_sync_pruning_and_pipeline_preservation(self):
        import src.nevfe.core.engine as engine_mod
        from src.nevfe.pipeline import NEVFEPipeline

        self.assertEqual(AlgorithmConfig().gamma_param, 0.65)
        for pruned_name in (
            "compute_bregman_optical_distribution",
            "compute_lexical_spatial_distribution",
            "compute_variational_posterior",
            "select_poisson_bregman_subwindow",
        ):
            self.assertFalse(hasattr(engine_mod, pruned_name))

        cfg = AlgorithmConfig(gamma_param=0.35)
        self.assertEqual(cfg.gamma_param, 0.35)
        with self.assertRaises(ValueError):
            AlgorithmConfig(gamma_param=-0.05)
        with self.assertRaises(ValueError):
            AlgorithmConfig(gamma_param=1.05)

        class _MockCfg:
            force_image_size = 448
            max_dynamic_patch = 2

        class _MockModel:
            config = _MockCfg()
            language_model = type("_LM", (), {"config": _MockCfg()})()
            num_image_token = 16

        default_pipe = NEVFEPipeline(model=_MockModel(), tokenizer=_OfflineMockTokenizer(), device="cpu")
        self.assertEqual(default_pipe.gamma_param, 0.65)
        self.assertEqual(default_pipe.adapter.engine.gamma_param, 0.65)

        pipe = NEVFEPipeline(model=_MockModel(), tokenizer=_OfflineMockTokenizer(), device="cpu", gamma_param=0.72)
        self.assertEqual(pipe.gamma_param, 0.72)
        pipe.set_ablation_switches(comp_a=False, comp_b=True, comp_c=True)
        self.assertEqual(pipe.config.gamma_param, 0.72)
        self.assertEqual(pipe.adapter.engine.gamma_param, 0.72)
        self.assertEqual(pipe.config.ablation_mask, (0, 1, 1))

    def test_16_spb_mlf_eq1_empirical_cone_kl_specificity_and_soft_or(self):
        # 1. Constant / ULP-perturbed patches, None/empty queries, and degenerate dims hit uniform fallback
        const_v = torch.ones((4, 4, 8), dtype=torch.float32) * 3.0
        const_v[0, 0, 0] += 1e-7
        q_emb = torch.randn(3, 8)
        self.assertTrue(torch.allclose(compute_spb_anchor_posterior(const_v, q_emb), torch.full((4, 4), 1.0 / 16.0), atol=1e-6))
        self.assertTrue(torch.allclose(compute_spb_anchor_posterior(const_v, None), torch.full((4, 4), 1.0 / 16.0), atol=1e-6))
        self.assertTrue(torch.allclose(compute_spb_anchor_posterior(const_v, torch.empty((0, 8))), torch.full((4, 4), 1.0 / 16.0), atol=1e-6))
        self.assertTrue(torch.allclose(compute_spb_anchor_posterior(torch.zeros((1, 1, 8)), torch.ones((1, 8))), torch.ones((1, 1)), atol=1e-6))

        # 2. Adversarial NaN / Inf / subnormal 1e-38 / all-zero queries stay finite on simplex
        adv_v = torch.ones((2, 2, 4), dtype=torch.float32) * 1e-38
        adv_v[0, 0, 0] = float("nan")
        adv_v[0, 1, 1] = float("inf")
        adv_q = torch.tensor([[1.0, float("-inf"), float("nan"), 0.0]], dtype=torch.float32)
        p_adv = compute_spb_anchor_posterior(adv_v, adv_q)
        self.assertTrue(torch.isfinite(p_adv).all())
        self.assertAlmostEqual(float(p_adv.sum().item()), 1.0, places=5)
        p_all_nan_q = compute_spb_anchor_posterior(torch.randn((3, 3, 4)), torch.full((2, 4), float("nan")))
        self.assertTrue(torch.allclose(p_all_nan_q, torch.full((3, 3), 1.0 / 9.0), atol=1e-6))

        with self.assertRaises(ValueError):
            compute_spb_anchor_posterior(const_v, torch.randn(2, 5))
        with self.assertRaises(ValueError):
            compute_spb_anchor_posterior(torch.zeros((4, 4)), q_emb)

        # 3. 0-homogeneity & FLT_MAX overflow immunity
        disc_v = torch.zeros((4, 4, 8), dtype=torch.float32)
        disc_v[:, :, 0] = 1.0
        disc_v[2, 3, :] = torch.tensor([0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * 1e30
        q_target = torch.tensor([[0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]) * 1e30
        p_disc = compute_spb_anchor_posterior(disc_v, q_target)
        self.assertTrue(torch.isfinite(p_disc).all())
        self.assertEqual(tuple(torch.nonzero(p_disc == p_disc.max(), as_tuple=False)[0].tolist()), (2, 3))

        # 4. Narrow-cone sensitivity (no 1/D variance floor suppression when empirical dispersion << 1/sqrt(D))
        cone_v = torch.zeros((8, 8, 64), dtype=torch.float32)
        cone_v[:, :, 0] = 1.0
        cone_v[3, 4, 1] = 0.006  # tiny 0.006 cone perturbation << 1/sqrt(64) = 0.125
        q_cone = torch.zeros((1, 64), dtype=torch.float32)
        q_cone[0, 0] = 1.0
        q_cone[0, 1] = 0.006
        p_cone = compute_spb_anchor_posterior(cone_v, q_cone, comp_a=True)
        self.assertGreater(float(p_cone[3, 4].item()), 0.80)

        # 5. Repetitive clutter de-crowding + Soft-OR compound-entity recall (comp_a=True vs comp_a=False)
        clutter_v = torch.zeros((8, 8, 16), dtype=torch.float32)
        clutter_v[:, :, 3] = 1.0
        clutter_v[0:3, :, :] = 0.0
        clutter_v[0:3, :, 0] = 1.0  # 24 repetitive clutter patches along dim 0
        clutter_v[5, 2, :] = 0.0
        clutter_v[5, 2, 1] = 1.0  # compound token 1
        clutter_v[5, 3, :] = 0.0
        clutter_v[5, 3, 2] = 1.0  # compound token 2
        q_comp = torch.zeros((3, 16), dtype=torch.float32)
        q_comp[0, 0] = 1.0
        q_comp[1, 1] = 1.0
        q_comp[2, 2] = 1.0
        p_kl = compute_spb_anchor_posterior(clutter_v, q_comp, comp_a=True)
        p_raw = compute_spb_anchor_posterior(clutter_v, q_comp, comp_a=False)
        self.assertGreater(float(p_kl[5, 2:4].sum().item()), float(p_raw[5, 2:4].sum().item()))
        top2 = [divmod(int(idx), 8) for idx in torch.topk(p_kl.flatten(), 2).indices.tolist()]
        self.assertEqual(set(top2), {(5, 2), (5, 3)})

        # 6. Regression (#193/#201): N=256, D=896 narrow cone preserves 2-patch sparse target alongside 64-patch clutter
        vit_cone = torch.zeros((16, 16, 896), dtype=torch.float32)
        vit_cone[:, :, 0] = 1.0
        vit_cone[0:4, :, 1] = 0.01  # 64 broad clutter patches
        vit_cone[8, 4:6, 2] = 0.006  # 2 sparse narrow-cone target patches
        q_vit = torch.zeros((2, 896), dtype=torch.float32)
        q_vit[0, 0], q_vit[0, 1] = 1.0, 0.01
        q_vit[1, 0], q_vit[1, 2] = 1.0, 0.006
        p_vit_kl = compute_spb_anchor_posterior(vit_cone, q_vit, comp_a=True)
        p_vit_raw = compute_spb_anchor_posterior(vit_cone, q_vit, comp_a=False)
        self.assertGreater(float(p_vit_kl[8, 4:6].sum().item()), float(p_vit_raw[8, 4:6].sum().item()))
        self.assertGreater(float(p_vit_kl[8, 4:6].sum().item()), 0.50)

    def test_17_spb_mlf_eq2_coifman_lafon_and_eq3_submodular_2locus(self):
        # Eq. (2): Coifman-Lafon anisotropic reading-line resolvent at r_N = 1/N
        lat = torch.zeros((8, 8, 8), dtype=torch.float32)
        lat[:, :, 0] = 1.0
        lat[1, 1, :] = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # isolated 1x1 patch
        for col in (1, 2, 4, 5):  # reading-line row 3 with dj=2 gap at (3, 3)
            lat[3, col, :] = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        for col in (1, 2, 3, 4):  # disconnected parallel banner at row 6 (di == 3 > 1)
            lat[6, col, :] = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Isolated 1x1 patch retains 1.0000 mass with 0.0000 dilution
        anchor_iso = torch.zeros((8, 8), dtype=torch.float32)
        anchor_iso[1, 1] = 1.0
        self.assertTrue(torch.allclose(compute_signboard_manifold_posterior(anchor_iso, lat, comp_b=True), anchor_iso, atol=1e-6))

        # Reading line bridges dj=2 gap across (3, 2) -> (3, 4) with 0.0% cross-sign jump to row 6
        anchor_line = torch.zeros((8, 8), dtype=torch.float32)
        anchor_line[3, 1] = 1.0
        post_line = compute_signboard_manifold_posterior(anchor_line, lat, comp_b=True)
        self.assertTrue(torch.isfinite(post_line).all())
        self.assertGreater(float(post_line[3, 2].item()), 0.01)
        self.assertGreater(float(post_line[3, 4].item()), 0.005)
        self.assertAlmostEqual(float(post_line[6, :].sum().item()), 0.0, places=6)
        self.assertTrue(torch.allclose(compute_signboard_manifold_posterior(anchor_line, lat, comp_b=False), anchor_line, atol=1e-6))

        # Eq. (3): Unimodal 2x2 and 1x1 return strictly 1 locus (0.0% FP box2)
        p_uni = torch.zeros((8, 8), dtype=torch.float32)
        p_uni[1:3, 2:4] = 0.25
        loci_uni = select_submodular_bregman_loci(p_uni, comp_c=True)
        self.assertEqual(len(loci_uni), 1)
        self.assertEqual(loci_uni[0][0], (1, 2, 3, 4))
        self.assertAlmostEqual(loci_uni[0][1], math.log(16.0) - (1.0 - 4.0 / 64.0), places=4)

        p_tiny = torch.zeros((8, 8), dtype=torch.float32)
        p_tiny[4, 4] = 1.0
        loci_tiny = select_submodular_bregman_loci(p_tiny, comp_c=True)
        self.assertEqual(len(loci_tiny), 1)
        self.assertEqual(loci_tiny[0][0], (4, 4, 5, 5))

        # Uniform / comp_c=False cleanly abstain with j1_star == 0.0
        loci_unif = select_submodular_bregman_loci(torch.full((8, 8), 1.0 / 64.0), comp_c=True)
        self.assertEqual(loci_unif, [((0, 0, 8, 8), 0.0, 1.0, 1.0)])
        loci_off = select_submodular_bregman_loci(p_uni, comp_c=False)
        self.assertEqual(loci_off, [((0, 0, 8, 8), 0.0, 1.0, 1.0)])

        # Bimodal 2-locus posterior discovers strictly disjoint C_1* and C_2*
        p_bi = torch.full((8, 8), 0.08 / 56.0, dtype=torch.float32)
        p_bi[1:3, 1:3] = 0.12  # Locus 1: 0.48
        p_bi[5:7, 5:7] = 0.11  # Locus 2: 0.44
        loci_bi = select_submodular_bregman_loci(p_bi, comp_c=True)
        self.assertEqual(len(loci_bi), 2)
        b1, _, _, _ = loci_bi[0]
        b2, j2_ext, _, _ = loci_bi[1]
        self.assertEqual(b1, (1, 1, 3, 3))
        self.assertEqual(b2, (5, 5, 7, 7))
        self.assertGreater(j2_ext, 0.5)
        self.assertTrue((b2[2] <= b1[0]) or (b2[0] >= b1[2]) or (b2[3] <= b1[1]) or (b2[1] >= b1[3]))

        # End-to-end NEVFEEngine.foveate & latency budget (< 5.0 ms/step)
        import time
        engine = NEVFEEngine(config=AlgorithmConfig(gamma_param=0.65))
        fov_lat = torch.zeros((8, 8, 32), dtype=torch.float32)
        fov_lat[:, :, 0] = 1.0
        fov_lat[2:4, 3:5, :] = 0.0
        fov_lat[2:4, 3:5, 2] = 1.0
        q_fov = torch.zeros((1, 32), dtype=torch.float32)
        q_fov[0, 2] = 1.0
        fboxes, fj, _ = engine.foveate(fov_lat, query_embeds=q_fov)
        self.assertGreater(fj, 0.0)
        self.assertEqual(fboxes, [(2, 3, 4, 5)])
        nboxes, nj, _ = engine.foveate(fov_lat, query_embeds=None)
        self.assertEqual(nj, 0.0)
        self.assertEqual(nboxes, [(0, 0, 8, 8)])

        l_plus = torch.randn(1, 128)
        l_zero = torch.randn(1, 128)
        t_s = time.perf_counter()
        for _ in range(20):
            engine.decode_step(l_plus, l_zero, step_idx=1)
        self.assertLess(((time.perf_counter() - t_s) * 1000.0) / 20.0, 5.0)

    def test_18_adapter_extract_lattice_multi_locus_reservation_nyquist_and_immutability(self):
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
                _, s_len, _ = inputs_embeds.shape
                hidden_out = inputs_embeds[:, -1:, :]
                return type("_Out", (), {"last_hidden_state": hidden_out, "past_key_values": ("kv", s_len)})()

        class _MockLM:
            config = _MockCfg()

            def __init__(self, vocab_size=64, hidden=16):
                self.emb = torch.nn.Embedding(vocab_size, hidden)
                self.model = _MockLMCore(vocab_size=vocab_size, hidden=hidden)
                self.vocab_size = vocab_size

            def get_input_embeddings(self):
                return self.emb

            def lm_head(self, x):
                out = torch.zeros((x.shape[0], x.shape[1], self.vocab_size), dtype=torch.float32)
                out[:, :, 2] = 5.0  # immediate <|im_end|>
                return out

        class _MockTok:
            eos_token_id = 2

            def encode(self, text, add_special_tokens=False):
                del add_special_tokens
                return [5, 6] if text else []

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
                return torch.arange(n_t * 16 * 16, dtype=torch.float32).view(n_t, 16, 16) * 0.01

        model = _MockModel()
        adapter = VinternNEVFEAdapter(model=model, tokenizer=_MockTok(), device="cpu")
        self.assertEqual(adapter.engine.gamma_param, 0.65)
        self.assertFalse(hasattr(adapter, "_extract_lattice_and_energy"))

        # 1. _extract_lattice assembles local high-res grid and excludes thumbnail vit_embeds[-1]
        sq_img = Image.new("RGB", (224, 224), color=(240, 240, 240))
        multi_vit = torch.zeros((5, 16, 16), dtype=torch.float32)
        multi_vit[:4, :, :] = 1.0
        multi_vit[4, :, :] = 99.0
        lat_p, img_w, img_h = adapter._extract_lattice(sq_img, multi_vit)
        self.assertEqual((img_w, img_h), (224, 224))
        self.assertEqual(tuple(lat_p.shape), (8, 8, 16))
        self.assertTrue(torch.allclose(lat_p, torch.ones((8, 8, 16))))

        # 2. Nyquist gate + unimodal banner (<= 3 tiles) + bimodal reservation (both boxes encoded within <= 3 tiles) + anti-keyhole tiny box
        calls_pre = len(model.extract_calls)
        self.assertIsNone(adapter.encode_transient_roi_crops(sq_img, [(0, 0, 224, 224)], max_tiles=3, min_magnification=0.5))
        self.assertEqual(len(model.extract_calls), calls_pre)

        banner_res = adapter.encode_transient_roi_crops(sq_img, [(10, 90, 210, 118)], max_tiles=3, min_magnification=0.5)
        self.assertIsNotNone(banner_res)
        self.assertGreaterEqual(model.extract_calls[-1], 2)
        self.assertLessEqual(model.extract_calls[-1], 3)

        bimodal_res = adapter.encode_transient_roi_crops(
            sq_img, [(10, 20, 210, 48), (20, 170, 200, 198)], max_tiles=3, min_magnification=0.5
        )
        self.assertIsNotNone(bimodal_res)
        self.assertGreaterEqual(model.extract_calls[-1], 2)
        self.assertLessEqual(model.extract_calls[-1], 3)

        keyhole_res = adapter.encode_transient_roi_crops(sq_img, [(100, 100, 102, 102)], max_tiles=3, min_magnification=0.5)
        self.assertIsNotNone(keyhole_res)
        self.assertTrue(torch.isfinite(keyhole_res).all())

        # 3. Multi-question generation preserves _cached_vit_embeds immutably; NEVFEPipeline keeps full-scene stage1_bbox & exports diagnostics["foveal_bboxes"]
        img = Image.new("RGB", (200, 120), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        draw.rectangle([20, 20, 48, 42], fill=(10, 10, 10))
        draw.line([22, 22, 46, 40], fill=(250, 250, 250), width=2)
        v1 = adapter.extract_visual_features(img).clone()
        adapter.generate(img, "ở đây ghi chữ gì?", max_new_tokens=2)
        cached_after_q1 = adapter._cached_vit_embeds.clone()
        adapter.generate(img, "giá bao nhiêu?", max_new_tokens=2)
        self.assertTrue(torch.equal(v1, cached_after_q1))
        self.assertTrue(torch.equal(v1, adapter._cached_vit_embeds))

        pipe = NEVFEPipeline(model=model, tokenizer=_MockTok(), device="cpu")
        out_pipe = pipe.run(img, "ở đây ghi chữ gì?", max_new_tokens=2)
        self.assertEqual(out_pipe.stage1_bbox, [0, 0, 200, 120])
        self.assertEqual(out_pipe.diagnostics["foveal_bboxes"], list(pipe.adapter._last_foveal_bboxes))

        calls_before = len(model.extract_calls)
        adapter.set_ablation_switches(comp_a=True, comp_b=True, comp_c=False)
        adapter.generate(img, "ở đây ghi chữ gì?", max_new_tokens=2)
        self.assertEqual(len(model.extract_calls), calls_before)
        self.assertEqual(adapter._last_foveal_bboxes, [])


if __name__ == "__main__":
    unittest.main()
