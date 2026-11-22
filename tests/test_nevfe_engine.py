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
    COMPONENT_NAMES,
    NEVFEEngine,
    SPEC_REF,
    compute_bhattacharyya_coefficient,
    compute_dynamic_coupling,
    compute_dynamic_plausibility_floor,
    compute_unnormalized_entropy,
    nevfe_step_logits,
)
from src.nevfe.core.frozen_foveation import (
    compute_lexical_spatial_distribution,
    compute_mean_field_reference_embeddings,
    foveate_frozen_011,
    select_poisson_bregman_subwindow,
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
        self.assertEqual(cfg.ablation_mask, ())
        self.assertEqual(COMPONENT_NAMES, ())
        self.assertFalse(hasattr(cfg, "gamma_param"))
        self.assertIn("baseline_frsa011.md", SPEC_REF)
        with self.assertRaises(ValueError):
            AlgorithmConfig(ablation_mask=(1,))
        with self.assertRaises(ValueError):
            AlgorithmConfig(ablation_mask=(2,))
        with self.assertRaises(ValueError):
            AlgorithmConfig(lambda_param=-0.1)
        with self.assertRaises(ValueError):
            AlgorithmConfig(lambda_param=0.40)
        with self.assertRaises(ValueError):
            AlgorithmConfig(beta_param=0.0)
        with self.assertRaises(ValueError):
            AlgorithmConfig(beta_param=0.10)

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
        self.assertEqual(engine.ablation_mask, ())
        self.assertFalse(hasattr(engine, "gamma_param"))

        with self.assertRaises(ValueError):
            NEVFEEngine(lambda_param=-0.1)
        with self.assertRaises(ValueError):
            NEVFEEngine(lambda_param=1.6)
        with self.assertRaises(ValueError):
            NEVFEEngine(beta_param=-0.01)
        with self.assertRaises(ValueError):
            NEVFEEngine(ablation_mask=(1,))

        b_plus = torch.randn(2, 32)
        b_zero = torch.randn(2, 32)
        tokens, logits, dyn_lam, ent = engine.decode_step(
            b_plus, b_zero, step_context={"history_logits": b_plus}
        )
        self.assertEqual(tokens.shape, torch.Size([2]))
        self.assertEqual(logits.shape, torch.Size([2, 32]))
        self.assertEqual(dyn_lam.shape, torch.Size([2, 1]))
        self.assertEqual(ent.shape, torch.Size([2, 1]))

        engine.set_ablation_switches(ablation_mask=())
        self.assertEqual(engine.ablation_mask, ())

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
        engine = NEVFEEngine(config=AlgorithmConfig())
        for step_idx in (1, 2):
            got = engine.decode_step(lp, lz, step_idx=step_idx)
            ref = _reference_iter23_step(lp, lz, 0.50, 0.05, step_idx, True, True, True)
            self.assertTrue(torch.equal(got[0], ref[0]))
            self.assertLess((got[1] - ref[1]).abs().max().item(), 1e-7)
            self.assertLess((got[2] - ref[2]).abs().max().item(), 1e-7)
            self.assertLess((got[3] - ref[3]).abs().max().item(), 1e-7)

        for mask in [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]:
            for step_idx in (1, 2):
                got = nevfe_step_logits(
                    lp,
                    lz,
                    lambda_param=0.50,
                    beta_param=0.05,
                    step_idx=step_idx,
                    comp_a=bool(mask[0]),
                    comp_b=bool(mask[1]),
                    comp_c=bool(mask[2]),
                )
                ref = _reference_iter23_step(lp, lz, 0.50, 0.05, step_idx, bool(mask[0]), bool(mask[1]), bool(mask[2]))
                self.assertTrue(torch.equal(got[0], ref[0]))
                self.assertLess((got[1] - ref[1]).abs().max().item(), 1e-7)
                self.assertLess((got[2] - ref[2]).abs().max().item(), 1e-7)
                self.assertLess((got[3] - ref[3]).abs().max().item(), 1e-7)

    def test_15_pipeline_locked_constants_and_shared_config_resolver(self):
        import json
        import tempfile
        import os
        from src.nevfe.pipeline import NEVFEPipeline, parse_ablation_mask, resolve_algorithm_config

        self.assertEqual(parse_ablation_mask("base"), ())
        self.assertEqual(parse_ablation_mask(""), ())
        self.assertEqual(parse_ablation_mask("01"), (0, 1))
        self.assertEqual(parse_ablation_mask("1"), (1,))
        with self.assertRaises(ValueError):
            parse_ablation_mask("1,0")
        with self.assertRaises(ValueError):
            parse_ablation_mask("12")
        with self.assertRaises(ValueError):
            parse_ablation_mask("01", expected_len=1)

        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = os.path.join(tmp, "cfg.json")
            with open(cfg_path, "w", encoding="utf-8") as cf:
                json.dump({"lambda_param": 0.5, "beta_param": 0.05, "ablation_mask": "base"}, cf)
            resolved = resolve_algorithm_config(cfg_path, ablation_mask_str="base")
            self.assertEqual(resolved, AlgorithmConfig())

            bad_cfg = os.path.join(tmp, "bad.json")
            with open(bad_cfg, "w", encoding="utf-8") as cf:
                json.dump({"unknown_field": 1.0}, cf)
            with self.assertRaises(ValueError):
                resolve_algorithm_config(bad_cfg)

        with self.assertRaises(ValueError):
            resolve_algorithm_config(overrides={"unknown_override": 1.0})

        class _MockCfg:
            force_image_size = 448
            max_dynamic_patch = 2

        class _MockModel:
            config = _MockCfg()
            language_model = type("_LM", (), {"config": _MockCfg()})()
            num_image_token = 16

        with self.assertRaises(TypeError):
            NEVFEPipeline(model=_MockModel(), tokenizer=_OfflineMockTokenizer(), device="cpu", gamma_param=0.5)
        with self.assertRaises(ValueError):
            NEVFEPipeline(model=_MockModel(), tokenizer=_OfflineMockTokenizer(), device="cpu", lambda_param=0.4)
        bad_lam_cfg = AlgorithmConfig.__new__(AlgorithmConfig)
        object.__setattr__(bad_lam_cfg, "lambda_param", 0.4)
        with self.assertRaises(ValueError):
            NEVFEPipeline(model=_MockModel(), tokenizer=_OfflineMockTokenizer(), device="cpu", config=bad_lam_cfg)
        bad_bet_cfg = AlgorithmConfig.__new__(AlgorithmConfig)
        object.__setattr__(bad_bet_cfg, "beta_param", 0.1)
        with self.assertRaises(ValueError):
            NEVFEPipeline(model=_MockModel(), tokenizer=_OfflineMockTokenizer(), device="cpu", config=bad_bet_cfg)

        pipe = NEVFEPipeline(model=_MockModel(), tokenizer=_OfflineMockTokenizer(), device="cpu")
        self.assertEqual(pipe.lambda_param, 0.50)
        self.assertEqual(pipe.beta_param, 0.05)
        self.assertEqual(pipe.config.ablation_mask, ())
        pipe.set_ablation_switches(ablation_mask=())
        self.assertEqual(pipe.adapter.engine.ablation_mask, ())

    def test_16_frozen_foveation_lexical_boundaries(self):
        const_v = torch.ones((4, 4, 8), dtype=torch.float32) * 3.0
        const_v[0, 0, 0] += 1e-7
        q_emb = torch.randn(3, 8)
        p_lex_const = compute_lexical_spatial_distribution(const_v, q_emb)
        self.assertTrue(torch.allclose(p_lex_const, torch.full((4, 4), 1.0 / 16.0), atol=1e-6))
        self.assertTrue(torch.allclose(compute_lexical_spatial_distribution(const_v, None), torch.full((4, 4), 1.0 / 16.0), atol=1e-6))
        self.assertTrue(torch.allclose(compute_lexical_spatial_distribution(const_v, torch.empty((0, 8))), torch.full((4, 4), 1.0 / 16.0), atol=1e-6))

        with self.assertRaises(ValueError):
            compute_lexical_spatial_distribution(torch.zeros((4, 8)), q_emb)
        with self.assertRaises(ValueError):
            compute_lexical_spatial_distribution(const_v, torch.randn(2, 5))

        disc_v = torch.zeros((4, 4, 8), dtype=torch.float32)
        disc_v[:, :, 0] = 1.0
        disc_v[2, 3, :] = torch.tensor([0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * 1e30
        q_target = torch.tensor([[0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]) * 1e30
        p_lex_disc = compute_lexical_spatial_distribution(disc_v, q_target)
        self.assertTrue(torch.isfinite(p_lex_disc).all())
        self.assertEqual(tuple(torch.nonzero(p_lex_disc == p_lex_disc.max(), as_tuple=False)[0].tolist()), (2, 3))

        box, j_star, p_out = foveate_frozen_011(disc_v, q_target)
        self.assertEqual(box, (2, 3, 3, 4))
        self.assertGreater(j_star, 0.0)
        self.assertTrue(torch.allclose(p_out, p_lex_disc))

        engine = NEVFEEngine()
        e_box, e_j, _ = engine.foveate(disc_v, q_target)
        self.assertEqual(e_box, box)
        self.assertAlmostEqual(e_j, j_star, places=6)

    def test_17_frozen_foveation_poisson_bregman_subwindow(self):
        with self.assertRaises(ValueError):
            select_poisson_bregman_subwindow(torch.zeros((2, 2, 2)))
        self.assertEqual(select_poisson_bregman_subwindow(torch.zeros((0, 0))), ((0, 0, 0, 0), 0.0, 0.0, 0.0))
        self.assertEqual(select_poisson_bregman_subwindow(torch.ones((1, 1))), ((0, 0, 1, 1), 0.0, 1.0, 1.0))

        p_opt = torch.zeros((8, 8), dtype=torch.float32)
        p_opt[1:3, 2:4] = 0.25
        p_unif = torch.full((8, 8), 1.0 / 64.0, dtype=torch.float32)

        box, j_star, p_mass, area_frac = select_poisson_bregman_subwindow(p_opt, comp_c=True)
        self.assertTrue(math.isfinite(j_star))
        self.assertGreater(j_star, 0.0)
        self.assertEqual(box, (1, 2, 3, 4))
        self.assertAlmostEqual(p_mass, 1.0, places=5)
        self.assertAlmostEqual(area_frac, 4.0 / 64.0, places=5)
        self.assertAlmostEqual(j_star, math.log(16.0) - (1.0 - 4.0 / 64.0), places=4)

        spot = torch.full((8, 8), 0.005, dtype=torch.float32)
        spot[3:5, 4:6] = 0.18
        spot_box, spot_j, _, spot_area = select_poisson_bregman_subwindow(spot, comp_c=True)
        self.assertEqual(spot_box, (3, 4, 5, 6))
        self.assertLessEqual(spot_area, 0.1)
        self.assertGreater(spot_j, 0.0)

        box_u, j_u, _, _ = select_poisson_bregman_subwindow(p_unif, comp_c=True)
        self.assertEqual(j_u, 0.0)
        self.assertEqual(box_u, (0, 0, 8, 8))

        box_off, j_off, _, _ = select_poisson_bregman_subwindow(p_opt, comp_c=False)
        self.assertEqual(j_off, 0.0)
        self.assertEqual(box_off, (0, 0, 8, 8))

    def test_18_adapter_cst_isoluminant_nyquist_gate_and_immutability(self):
        import numpy as np
        from PIL import Image, ImageDraw
        from src.nevfe.adapter.vintern_adapter import VinternNEVFEAdapter, resolve_relative_decoder_layer_indices
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
                b_sz, s_len, _ = inputs_embeds.shape
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

        # 1. Rank-1 diagonal ramp (SO(2) invariance: det(J) == 0 -> optical_energy near 0 on interior cells)
        yy, xx = np.mgrid[0:64, 0:64]
        diag_ramp = np.clip((xx + yy) * 2, 0, 255).astype(np.uint8)
        diag_img = Image.fromarray(np.stack([diag_ramp, diag_ramp, diag_ramp], axis=-1))
        _, e_diag, _, _ = adapter._extract_lattice_and_energy(diag_img, torch.zeros(1, 16, 16))
        self.assertLess(float(e_diag[:-1, :-1].max().item()), 1e-6)

        # 2. Isoluminant colored corner junctions (R+G+B=300 everywhere -> grayscale blind, CST detects)
        iso_img = Image.new("RGB", (64, 64), color=(200, 50, 50))
        ImageDraw.Draw(iso_img).rectangle([16, 16, 36, 36], fill=(50, 200, 50))
        _, e_iso, _, _ = adapter._extract_lattice_and_energy(iso_img, torch.zeros(1, 16, 16))
        self.assertGreater(float(e_iso.max().item()), 0.02)
        self.assertAlmostEqual(float(e_iso.min().item()), 0.0, places=6)

        # 3. Nyquist super-resolution gate: reject crops with M_crop <= M_glob, allow focused banners up to 3 tiles
        sq_img = Image.new("RGB", (224, 224), color=(240, 240, 240))
        calls_pre = len(model.extract_calls)
        self.assertIsNone(adapter.encode_transient_roi_crops(sq_img, [(0, 0, 224, 224)], max_tiles=3, min_magnification=0.5))
        self.assertEqual(len(model.extract_calls), calls_pre)

        banner_res = adapter.encode_transient_roi_crops(sq_img, [(10, 90, 210, 118)], max_tiles=3, min_magnification=0.5)
        self.assertIsNotNone(banner_res)
        self.assertGreaterEqual(model.extract_calls[-1], 2)
        self.assertLessEqual(model.extract_calls[-1], 3)

        # 4. Multi-question generation preserves _cached_vit_embeds immutably & tracks decoder forward calls
        img = Image.new("RGB", (200, 120), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        draw.rectangle([20, 20, 48, 42], fill=(10, 10, 10))
        draw.line([22, 22, 46, 40], fill=(250, 250, 250), width=2)
        v1 = adapter.extract_visual_features(img).clone()
        adapter.generate(img, "ở đây ghi chữ gì?", max_new_tokens=2)
        self.assertEqual(adapter.last_decoder_forward_calls, 2)
        cached_after_q1 = adapter._cached_vit_embeds.clone()
        adapter.generate(img, "giá bao nhiêu?", max_new_tokens=2)
        self.assertTrue(torch.equal(v1, cached_after_q1))
        self.assertTrue(torch.equal(v1, adapter._cached_vit_embeds))

        pipe = NEVFEPipeline(model=model, tokenizer=_MockTok(), device="cpu")
        res = pipe.run(img, "ở đây ghi chữ gì?", max_new_tokens=2)
        self.assertEqual(res.diagnostics["component_names"], [])
        self.assertEqual(res.diagnostics["ablation_mask"], [])
        self.assertEqual(res.diagnostics["decoder_forward_calls"], 2)
        self.assertNotIn("gamma_param", res.diagnostics)

        self.assertEqual(resolve_relative_decoder_layer_indices(24, (0.5, 1.0)), list(range(12, 24)))
        with self.assertRaises(ValueError):
            resolve_relative_decoder_layer_indices(0, (0.5, 1.0))
        with self.assertRaises(ValueError):
            resolve_relative_decoder_layer_indices(24, (0.8, 0.5))


if __name__ == "__main__":
    unittest.main()
