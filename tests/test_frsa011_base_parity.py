# -*- coding: utf-8 -*-
"""
Phase-0 Frozen Baseline Parity & Authoritative Track 2 Runtime Verification Suite
(tests/test_frsa011_base_parity.py).

Verifies:
1. Bit-identical parity (max_abs_diff == 0.0) between base/frsa-011-frozen
   (foveate_frozen_011 + NEVFEEngine.decode_step + VinternNEVFEAdapter.generate)
   and an inline reference implementation of swarm/team-1 (4c8ee01) "011".
2. Authoritative Track 2 DynamicCache KV invariance (max_kv_diff == 0.0 across all
   24 layers, both streams, and multi-step decoding) for capture_upper_decoder_states
   on a tiny CPU Qwen2Model (24 layers, hidden_size=64, 4:2 GQA).
3. Single-step decoder forward telemetry (last_decoder_forward_calls == 2 * steps,
   last_decoder_forward_calls_per_step_max == 1.0) and rollout_fn KV isolation.
"""

import unittest
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2Config, Qwen2Model

from src.nevfe.adapter.vintern_adapter import (
    VinternNEVFEAdapter,
    capture_upper_decoder_states,
    clone_kv_cache,
    extract_kv_pairs,
    resolve_relative_decoder_layer_indices,
)
from src.nevfe.core.engine import AlgorithmConfig, NEVFEEngine, nevfe_step_logits
from src.nevfe.core.frozen_foveation import (
    compute_lexical_spatial_distribution,
    compute_mean_field_reference_embeddings,
    foveate_frozen_011,
    select_poisson_bregman_subwindow,
)


def _ref_4c8ee01_lexical(patch_embeds: torch.Tensor, query_embeds: torch.Tensor, tau_v: float = 1e-6) -> torch.Tensor:
    h_g, w_g, dim = patch_embeds.shape
    v_raw = torch.nan_to_num(patch_embeds.float(), nan=0.0, posinf=0.0, neginf=0.0)
    q_raw = torch.nan_to_num(query_embeds.float(), nan=0.0, posinf=0.0, neginf=0.0)
    v = v_raw / torch.clamp(v_raw.abs().max(), min=1.0)
    q = q_raw / torch.clamp(q_raw.abs().amax(dim=-1, keepdim=True), min=1.0)
    dv = v - v.mean(dim=(0, 1), keepdim=True)
    r_v = torch.linalg.vector_norm(dv, dim=-1, keepdim=True)
    r_e = torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    tau_d = tau_v * (float(max(1, dim)) ** 0.5)
    v_hat = torch.where(r_v > tau_d, dv / torch.clamp(r_v, min=tau_d), torch.zeros_like(dv))
    e_hat = torch.where(r_e > tau_v, q / torch.clamp(r_e, min=tau_v), torch.zeros_like(q))
    c_q = torch.matmul(v_hat, e_hat.transpose(0, 1)).max(dim=-1).values
    diff_c = c_q - c_q.mean()
    sigma_q = torch.sqrt(torch.clamp((diff_c * diff_c).mean(), min=0.0))
    z_q = torch.where(sigma_q > tau_v, diff_c / torch.clamp(sigma_q, min=tau_v), torch.zeros_like(diff_c))
    return F.softmax(z_q.view(-1), dim=-1).view(h_g, w_g)


def _ref_4c8ee01_ig_vfed(lp: torch.Tensor, lz: torch.Tensor, step_idx: int) -> tuple:
    lp32 = lp.float()
    lz32 = lz.float()
    p_p = F.softmax(lp32, dim=-1)
    p_z = F.softmax(lz32, dim=-1)
    bc = torch.clamp(torch.sum(torch.sqrt(torch.clamp(p_p * p_z, min=0.0)), dim=-1, keepdim=True), 0.0, 1.0)
    alpha_t = 0.50 * torch.sqrt(torch.clamp(1.0 - bc * bc, 0.0, 1.0))
    if step_idx == 1:
        alpha_t = alpha_t + 0.50 * bc
    pot = lp32 + alpha_t * (lp32 - lz32)
    beta_t = torch.clamp(0.05 * bc, 0.0, 1.0)
    max_p = torch.max(p_p, dim=-1, keepdim=True).values
    steered = torch.where(p_p >= beta_t * max_p, pot, torch.full_like(pot, -1e9))
    tok = torch.argmax(steered, dim=-1)
    return tok, steered, alpha_t


class _TinyTok:
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [5, 6, 7] if text else []

    def convert_tokens_to_ids(self, tok):
        return {"<IMG_CONTEXT>": 3, "<|im_end|>": 2}.get(tok, 0)

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        del return_tensors, add_special_tokens
        n_img = text.count("<IMG_CONTEXT>")
        ids = [1] + ([3] * n_img) + [4] if n_img > 0 else [5, 6, 7]
        return type("_Enc", (), {"input_ids": torch.tensor([ids], dtype=torch.long)})()

    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(str(i) for i in ids)


class _TinyVinternModel(nn.Module):
    def __init__(self, num_layers: int = 24, hidden_size: int = 64, vocab_size: int = 256):
        super().__init__()
        self.config = type("_Cfg", (), {"force_image_size": 112, "max_dynamic_patch": 2})()
        self.num_image_token = 16
        q_cfg = Qwen2Config(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 2,
            num_hidden_layers=num_layers,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        lm_core = Qwen2Model(q_cfg).eval()
        lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        with torch.no_grad():
            lm_head.weight.copy_(torch.randn_like(lm_head.weight) * 0.2)

        class _LM(nn.Module):
            def __init__(self, cfg, core, head):
                super().__init__()
                self.config = cfg
                self.model = core
                self.lm_head = head

            def get_input_embeddings(self):
                return self.model.embed_tokens

        self.language_model = _LM(q_cfg, lm_core, lm_head).eval()

    def extract_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
        n_t = int(pixel_values.shape[0])
        h_dim = int(self.language_model.config.hidden_size)
        base = torch.arange(n_t * 16 * h_dim, dtype=torch.float32).view(n_t, 16, h_dim) * 0.001
        return base + pixel_values.float().mean(dim=(1, 2, 3), keepdim=False).view(n_t, 1, 1)


class TestFRSA011BaseParity(unittest.TestCase):
    def test_01_bit_identical_foveation_and_decode_parity_vs_4c8ee01(self):
        torch.manual_seed(1234)
        patch_embeds = torch.randn(8, 8, 64)
        query_embeds = torch.randn(4, 64)

        p_lex_ref = _ref_4c8ee01_lexical(patch_embeds, query_embeds)
        box_ref, j_ref, _, _ = select_poisson_bregman_subwindow(p_lex_ref, comp_c=True)

        engine = NEVFEEngine(config=AlgorithmConfig())
        box_got, j_got, p_lex_got = engine.foveate(patch_embeds, query_embeds)
        self.assertEqual(box_got, box_ref)
        self.assertEqual(j_got, j_ref)
        self.assertEqual(float((p_lex_got - p_lex_ref).abs().max().item()), 0.0)

        box_foveate, j_foveate, _ = foveate_frozen_011(patch_embeds, query_embeds)
        self.assertEqual(box_foveate, box_ref)
        self.assertEqual(j_foveate, j_ref)

        for step_idx in range(1, 5):
            lp = torch.randn(2, 128)
            lz = torch.randn(2, 128)
            tok_ref, steered_ref, alpha_ref = _ref_4c8ee01_ig_vfed(lp, lz, step_idx=step_idx)
            tok_got, steered_got, alpha_got, _ = engine.decode_step(lp, lz, step_idx=step_idx)
            self.assertTrue(torch.equal(tok_got, tok_ref))
            self.assertEqual(float((steered_got - steered_ref).abs().max().item()), 0.0)
            self.assertEqual(float((alpha_got - alpha_ref).abs().max().item()), 0.0)

        # Adapter end-to-end generation parity against direct nevfe_step_logits(..., True, True, True)
        model = _TinyVinternModel(num_layers=4, hidden_size=32, vocab_size=64)
        tok = _TinyTok()
        img = Image.new("RGB", (112, 112), color=(220, 220, 220))
        ImageDraw.Draw(img).rectangle([10, 10, 48, 36], fill=(15, 15, 15))
        adapter = VinternNEVFEAdapter(model=model, tokenizer=tok, device="cpu")
        got_text = adapter.generate(img, "chữ gì?", max_new_tokens=3)

        with torch.no_grad():
            vit = adapter.extract_visual_features(img)
            extra = adapter.extract_foveated_roi_crops(img, vit, "chữ gì?")
            emb_p = adapter.build_multimodal_embeds(vit, "chữ gì?", extra_vit_embeds=extra)
            emb_z = adapter.build_mfvr_embeds(vit, "chữ gì?", extra_vit_embeds=extra)
            o_p = model.language_model.model(inputs_embeds=emb_p, use_cache=True, return_dict=True)
            o_z = model.language_model.model(inputs_embeds=emb_z, use_cache=True, return_dict=True)
            kv_p, kv_z = o_p.past_key_values, o_z.past_key_values
            lp = model.language_model.lm_head(o_p.last_hidden_state[:, -1:, :])[:, -1, :]
            lz = model.language_model.lm_head(o_z.last_hidden_state[:, -1:, :])[:, -1, :]
            ref_ids = []
            for step in range(1, 4):
                ntok, _, _, _ = nevfe_step_logits(
                    lp, lz, lambda_param=0.50, beta_param=0.05, step_idx=step, comp_a=True, comp_b=True, comp_c=True
                )
                nid = int(ntok.item())
                if nid == 2:
                    break
                ref_ids.append(nid)
                if step == 3:
                    break
                nemb = model.language_model.get_input_embeddings()(torch.tensor([[nid]], dtype=torch.long))
                o_p = model.language_model.model(inputs_embeds=nemb, past_key_values=kv_p, use_cache=True, return_dict=True)
                o_z = model.language_model.model(inputs_embeds=nemb, past_key_values=kv_z, use_cache=True, return_dict=True)
                kv_p, kv_z = o_p.past_key_values, o_z.past_key_values
                lp = model.language_model.lm_head(o_p.last_hidden_state[:, -1:, :])[:, -1, :]
                lz = model.language_model.lm_head(o_z.last_hidden_state[:, -1:, :])[:, -1, :]
            ref_text = tok.decode(ref_ids, skip_special_tokens=True)

        self.assertEqual(got_text, ref_text)
        self.assertEqual(adapter.last_decode_steps, 3)
        self.assertEqual(adapter.last_decoder_forward_calls, 6)
        self.assertEqual(adapter.last_decoder_forward_calls_per_step_max, 1.0)

    def test_02_track2_24_layer_dynamic_cache_kv_invariance_and_single_step(self):
        torch.manual_seed(77)
        model = _TinyVinternModel(num_layers=24, hidden_size=64, vocab_size=256)
        lm_core = model.language_model.model
        lm_head = model.language_model.lm_head
        embed_layer = model.language_model.get_input_embeddings()

        seq_plus = torch.randn(1, 8, 64)
        seq_zero = torch.randn(1, 8, 64)
        steps = 3

        def _run_unhooked_stream(init_embeds: torch.Tensor):
            step_kvs = []
            step_logits = []
            with torch.no_grad():
                out = lm_core(inputs_embeds=init_embeds, use_cache=True, output_attentions=False, return_dict=True)
                kv = out.past_key_values
                step_kvs.append([(k.clone(), v.clone()) for k, v in extract_kv_pairs(kv)])
                lg = lm_head(out.last_hidden_state[:, -1:, :])[:, -1, :]
                step_logits.append(lg.clone())
                for s in range(1, steps):
                    tok_t = torch.tensor([[s + 10]], dtype=torch.long)
                    out = lm_core(
                        inputs_embeds=embed_layer(tok_t),
                        past_key_values=kv,
                        use_cache=True,
                        output_attentions=False,
                        return_dict=True,
                    )
                    kv = out.past_key_values
                    step_kvs.append([(k.clone(), v.clone()) for k, v in extract_kv_pairs(kv)])
                    step_logits.append(lm_head(out.last_hidden_state[:, -1:, :])[:, -1, :].clone())
            return step_kvs, step_logits

        ref_kv_plus, ref_lg_plus = _run_unhooked_stream(seq_plus)
        ref_kv_zero, ref_lg_zero = _run_unhooked_stream(seq_zero)

        max_kv_diff = 0.0
        max_logit_diff = 0.0
        expected_upper = resolve_relative_decoder_layer_indices(24, (0.5, 1.0))
        self.assertEqual(expected_upper, list(range(12, 24)))

        with capture_upper_decoder_states(model, fraction=(0.5, 1.0)) as cap:
            self.assertEqual(cap["upper_indices"], expected_upper)
            for stream_embeds, ref_kvs, ref_lgs in (
                (seq_plus, ref_kv_plus, ref_lg_plus),
                (seq_zero, ref_kv_zero, ref_lg_zero),
            ):
                with torch.no_grad():
                    cap["clear"]()
                    out = lm_core(
                        inputs_embeds=stream_embeds,
                        use_cache=True,
                        output_attentions=False,
                        return_dict=True,
                    )
                    kv = out.past_key_values
                    lg = lm_head(out.last_hidden_state[:, -1:, :])[:, -1, :]
                    self.assertIsNotNone(cap["h_in_upper"])
                    self.assertEqual(tuple(cap["h_in_upper"].shape), (1, 1, 64))
                    self.assertEqual(sorted(cap["h_post_attn_upper"].keys()), expected_upper)
                    self.assertEqual(sorted(cap["h_post_ffn_upper"].keys()), expected_upper)

                    for s_idx in range(steps):
                        if s_idx > 0:
                            cap["clear"]()
                            tok_t = torch.tensor([[s_idx + 10]], dtype=torch.long)
                            out = lm_core(
                                inputs_embeds=embed_layer(tok_t),
                                past_key_values=kv,
                                use_cache=True,
                                output_attentions=False,
                                return_dict=True,
                            )
                            kv = out.past_key_values
                            lg = lm_head(out.last_hidden_state[:, -1:, :])[:, -1, :]
                        cur_pairs = extract_kv_pairs(kv)
                        self.assertEqual(len(cur_pairs), 24)
                        for (k_got, v_got), (k_ref, v_ref) in zip(cur_pairs, ref_kvs[s_idx]):
                            max_kv_diff = max(
                                max_kv_diff,
                                float((k_got - k_ref).abs().max().item()),
                                float((v_got - v_ref).abs().max().item()),
                            )
                        max_logit_diff = max(
                            max_logit_diff,
                            float((lg - ref_lgs[s_idx]).abs().max().item()),
                        )

        self.assertEqual(max_kv_diff, 0.0)
        self.assertEqual(max_logit_diff, 0.0)

        # Verify clone_kv_cache isolates mutations and rollout_fn executes cleanly
        with torch.no_grad():
            o = lm_core(inputs_embeds=seq_plus, use_cache=True, output_attentions=False, return_dict=True)
            kv_orig = o.past_key_values
            kv_clone = clone_kv_cache(kv_orig, max_len=4)
            orig_pairs = extract_kv_pairs(kv_orig)
            clone_pairs = extract_kv_pairs(kv_clone)
            self.assertEqual(clone_pairs[0][0].shape[2], 4)
            clone_pairs[0][0].zero_()
            self.assertGreater(float(orig_pairs[0][0].abs().max().item()), 0.0)

        # Verify adapter.generate step_context rollout_fn callback works with real Qwen2Model DynamicCache
        captured_ctx = {}

        class _RolloutCheckEngine(NEVFEEngine):
            def decode_step(self, logits_plus, logits_zero, q_mask=None, step_idx=1, step_context=None):
                if step_idx == 1 and step_context and callable(step_context.get("rollout_fn")):
                    rp, rz = step_context["rollout_fn"](torch.tensor([5, 6]), steps=1)
                    captured_ctx["shape"] = tuple(rp.shape)
                    captured_ctx["rz_shape"] = tuple(rz.shape)
                return super().decode_step(logits_plus, logits_zero, q_mask=q_mask, step_idx=step_idx, step_context=step_context)

        adapter = VinternNEVFEAdapter(model=model, tokenizer=_TinyTok(), device="cpu")
        adapter.engine = _RolloutCheckEngine(config=AlgorithmConfig())
        img = Image.new("RGB", (112, 112), color=(180, 180, 180))
        adapter.generate(img, "số mấy?", max_new_tokens=2)
        self.assertEqual(captured_ctx["shape"], (2, 256))
        self.assertEqual(captured_ctx["rz_shape"], (2, 256))
        self.assertGreater(adapter.last_decoder_forward_calls, 4)
        self.assertGreater(adapter.last_decoder_forward_calls_per_step_max, 1.0)


if __name__ == "__main__":
    unittest.main()
