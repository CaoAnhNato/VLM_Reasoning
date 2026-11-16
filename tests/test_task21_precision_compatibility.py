# -*- coding: utf-8 -*-
"""
Task 21 Phase A & Phase B Verification Suite:
  - Explicit precision selection (`bf16` | `fp16`) & no implicit fallback
  - Deterministic CPU synthetic numerical comparison across FP32, BF16, and FP16:
      * Canonical R-03 signed-logit formula
      * Dynamic alpha_t, entropy H(P_pos), and D_JS calculations
      * Positive and negative branch logits & top-k / argmax agreement
      * Candidate-support (CAPC) and valid_mask behavior
      * EOS and special-token selection
      * KV-cache shape/dtype contract
      * OPT-01 batched (B=2) vs sequential (B=1) numerical parity in FP16 & BF16
      * FP16 overflow, underflow (`1e-12`/`1e-9` clamp flush to zero -> NaN without FP32 promotion), and non-finite handling
      * Stage 1 attention extraction & ROI computation across FP32, BF16, FP16
  - Precision-aware Colab preflight unit checks (FP16 admission & conditional BF16 Owner exception)
"""

import math
import os
import sys
import types
import unittest
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.benchmark.colab_gate_b_preflight import (
    audit_hardware_and_environment,
    format_human_readable_report,
)
from scripts.benchmark.sg_vcd.attention_extractor import PointwiseKLAttentionExtractor
from scripts.benchmark.sg_vcd.bbox_selector import KLDivergenceRegionSelector
from scripts.benchmark.sg_vcd.dynamic_cd import DynamicContrastiveDecoder
from scripts.benchmark.sg_vcd.feature_cache import (
    VisualFeatureCache,
    resolve_precision_and_dtype,
)
from scripts.benchmark.sg_vcd.pipeline import SGVCDPipeline


def _identity_transform(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.resize((32, 32), Image.Resampling.BILINEAR).convert("RGB"), dtype=np.float32)
    return torch.from_numpy(arr / 255.0).permute(2, 0, 1).contiguous()


def _rel_error(ref: torch.Tensor, approx: torch.Tensor, eps: float = 1e-6) -> float:
    denom = torch.clamp(ref.abs(), min=eps)
    return float(((approx.float() - ref.float()).abs() / denom).max().item())


class _TinyTokenizer:
    def __init__(self, vocab_size: int = 128):
        self.vocab_size = vocab_size
        self.eos_token_id = 2
        self._special = {"<|im_end|>": 2, "<IMG_CONTEXT>": 3}

    def convert_tokens_to_ids(self, tok: str) -> int:
        return self._special.get(tok, 1)

    def __call__(self, text: str, return_tensors: str = "pt", add_special_tokens: bool = False):
        ids = []
        # Preserve exact <IMG_CONTEXT> tokens when present
        parts = text.split("<IMG_CONTEXT>")
        for idx, p in enumerate(parts):
            for ch in p:
                ids.append(10 + (ord(ch) % 40))
            if idx < len(parts) - 1:
                ids.append(3)
        if not ids:
            ids = [10, 11]
        tensor = torch.tensor([ids], dtype=torch.long)
        return types.SimpleNamespace(input_ids=tensor)

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        out = []
        for i in ids:
            if skip_special_tokens and i in (2, 3):
                continue
            out.append(chr(ord("a") + (i % 26)))
        return "".join(out)


class _SyntheticAttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int = 64, num_heads: int = 4, num_kv_heads: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim, bias=False),
        )
        self.self_attn = types.SimpleNamespace(config=types.SimpleNamespace(_attn_implementation="sdpa"))

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
        B, L, D = hidden_states.shape
        x = self.ln1(hidden_states)
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        new_kv = (k, v)

        rep = self.num_heads // self.num_kv_heads
        k_exp = k.repeat_interleave(rep, dim=1)
        v_exp = v.repeat_interleave(rep, dim=1)

        S_kv = k_exp.shape[2]
        past_len = S_kv - L
        scores = torch.matmul(q, k_exp.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if L > 1:
            q_pos = torch.arange(past_len, S_kv, device=hidden_states.device).unsqueeze(-1)
            k_pos = torch.arange(0, S_kv, device=hidden_states.device).unsqueeze(0)
            causal = k_pos <= q_pos
            scores = scores.masked_fill(~causal.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn_probs = F.softmax(scores.float(), dim=-1).to(hidden_states.dtype)
        attn_out = torch.matmul(attn_probs, v_exp)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        h = hidden_states + self.o_proj(attn_out)
        h = h + self.mlp(self.ln2(h))
        return h, new_kv, (attn_probs if output_attentions else None)


class _SyntheticLLMBackbone(nn.Module):
    def __init__(self, vocab_size: int = 128, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([_SyntheticAttentionLayer(hidden_dim=hidden_dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.config = types.SimpleNamespace(_attn_implementation="sdpa")

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values=None,
        use_cache: bool = True,
        output_attentions: bool = False,
        return_dict: bool = True,
        attention_mask=None,
        position_ids=None,
    ):
        h = inputs_embeds
        new_kvs = []
        attentions = []
        for idx, layer in enumerate(self.layers):
            p_kv = past_key_values[idx] if past_key_values is not None else None
            h, layer_kv, attn_w = layer(h, past_kv=p_kv, output_attentions=output_attentions)
            new_kvs.append(layer_kv)
            if output_attentions:
                attentions.append(attn_w)
        h = self.norm(h)
        return types.SimpleNamespace(
            last_hidden_state=h,
            past_key_values=tuple(new_kvs) if use_cache else None,
            attentions=tuple(attentions) if output_attentions else None,
        )


class _SyntheticVLM(nn.Module):
    def __init__(self, vocab_size: int = 128, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.num_image_token = 256
        self.img_context_token_id = 3
        self.config = types.SimpleNamespace(max_dynamic_patch=6)
        self.vision_proj = nn.Linear(3 * 32 * 32, hidden_dim, bias=False)
        lm_model = _SyntheticLLMBackbone(vocab_size=vocab_size, hidden_dim=hidden_dim, num_layers=num_layers)
        lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.language_model = types.SimpleNamespace(
            model=lm_model,
            lm_head=lm_head,
            config=lm_model.config,
            get_input_embeddings=lambda: lm_model.embed,
        )

    def to_dtype(self, dtype: torch.dtype):
        self.vision_proj.to(dtype=dtype)
        self.language_model.model.to(dtype=dtype)
        self.language_model.lm_head.to(dtype=dtype)
        return self

    def extract_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
        N = pixel_values.shape[0]
        flat = pixel_values.reshape(N, -1)
        vec = self.vision_proj(flat)  # (N, D)
        # Broadcast to (N, 256, D) with subtle deterministic spatial variation
        spatial = torch.linspace(0.8, 1.2, self.num_image_token, device=vec.device, dtype=vec.dtype).view(1, 256, 1)
        return vec.unsqueeze(1) * spatial


class TestTask21PrecisionCompatibility(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)

    def test_01_explicit_precision_resolution_and_no_fallback(self):
        """Explicit --precision bf16|fp16 resolver preserves bf16 default and rejects unsupported inputs."""
        self.assertEqual(resolve_precision_and_dtype(), ("bf16", torch.bfloat16))
        self.assertEqual(resolve_precision_and_dtype(precision="bf16"), ("bf16", torch.bfloat16))
        self.assertEqual(resolve_precision_and_dtype(precision="fp16"), ("fp16", torch.float16))
        self.assertEqual(resolve_precision_and_dtype(dtype=torch.float16), ("fp16", torch.float16))
        self.assertEqual(resolve_precision_and_dtype(dtype=torch.bfloat16), ("bf16", torch.bfloat16))

        for bad in ("fp32", "float32", "int8", "auto", ""):
            with self.assertRaises(ValueError):
                resolve_precision_and_dtype(precision=bad)

        with self.assertRaises(ValueError):
            resolve_precision_and_dtype(precision="fp16", dtype=torch.bfloat16)

    def test_02_pipeline_and_cache_precision_propagation_no_mixed_dtype(self):
        """Verify precision propagates cleanly to SGVCDPipeline, extractor, cache, and rejects mixed branches."""
        model = _SyntheticVLM()
        tok = _TinyTokenizer()
        cache = VisualFeatureCache(max_entries=8)
        pipe_fp16 = SGVCDPipeline(
            model=model,
            tokenizer=tok,
            transform=_identity_transform,
            feature_cache=cache,
            precision="fp16",
            device="cpu",
        )
        self.assertEqual(pipe_fp16.precision, "fp16")
        self.assertEqual(pipe_fp16.torch_dtype, torch.float16)
        self.assertEqual(pipe_fp16.extractor.torch_dtype, torch.float16)

        # Verify VisualFeatureCache does NOT return BF16 cached tensor when FP16 is requested
        model.to_dtype(torch.bfloat16)
        img = Image.new("RGB", (64, 64), (100, 140, 180))
        emb_bf16 = cache.get_or_compute(model, _identity_transform, [img], image=img, device="cpu", dtype=torch.bfloat16, image_id="imgA")
        self.assertEqual(emb_bf16.dtype, torch.bfloat16)

        model.to_dtype(torch.float16)
        emb_fp16 = cache.get_or_compute(model, _identity_transform, [img], image=img, device="cpu", dtype=torch.float16, image_id="imgA")
        self.assertEqual(emb_fp16.dtype, torch.float16)

        # Mixed branch input dtypes must raise ValueError (no silent conversion)
        pos_in = torch.randn(1, 5, 64, dtype=torch.float16)
        neg_in = torch.randn(1, 5, 64, dtype=torch.bfloat16)
        with self.assertRaises(ValueError):
            pipe_fp16._validate_dual_branch_inputs(pos_in, neg_in)

        dec = DynamicContrastiveDecoder()
        with self.assertRaises(ValueError):
            dec.decode_step(torch.randn(1, 16, dtype=torch.float16), torch.randn(1, 16, dtype=torch.bfloat16), step=0, spatial_confidence=0.9)

    def test_03_synthetic_numerical_comparison_fp32_bf16_fp16_stage2(self):
        """
        Compare FP32, BF16, and FP16 Stage 2 behavior on deterministic logits over 50 steps:
        - Canonical R-03 score, alpha_t, entropy_t, D_JS, valid_mask, EOS, top-k & argmax agreement.
        """
        torch.manual_seed(1234)
        dec_f32 = DynamicContrastiveDecoder(scoring_mode="canonical_signed", kappa_eos=0.25, n_min_eos=1)
        dec_bf16 = DynamicContrastiveDecoder(scoring_mode="canonical_signed", kappa_eos=0.25, n_min_eos=1)
        dec_fp16 = DynamicContrastiveDecoder(scoring_mode="canonical_signed", kappa_eos=0.25, n_min_eos=1)

        N_trials = 50
        V = 256
        eos_id = 2

        stats = {
            "bf16": {"alpha_abs": [], "alpha_rel": [], "ent_abs": [], "js_abs": [], "score_abs": [], "score_rel": [], "argmax_match": 0, "top5_match": 0, "mask_match": 0, "eos_match": 0},
            "fp16": {"alpha_abs": [], "alpha_rel": [], "ent_abs": [], "js_abs": [], "score_abs": [], "score_rel": [], "argmax_match": 0, "top5_match": 0, "mask_match": 0, "eos_match": 0},
        }

        for step in range(N_trials):
            pos_32 = torch.randn(1, V, dtype=torch.float32) * 3.5
            neg_32 = pos_32 + torch.randn(1, V, dtype=torch.float32) * 1.2
            # Force realistic visual peak on token 7 every 5 steps, EOS peak occasionally
            if step % 5 == 0:
                pos_32[0, 7] += 4.0
                neg_32[0, 7] -= 2.5
            if step % 10 == 9:
                pos_32[0, eos_id] = pos_32.max() - 0.1

            tok32, a32, e32, js32 = dec_f32.decode_step(pos_32, neg_32, step=step, spatial_confidence=0.82, eos_id=eos_id)
            score32 = (1.0 + a32) * pos_32 - a32 * neg_32
            top5_32 = torch.topk(pos_32, k=5, dim=-1).indices[0].tolist()
            p_pos32 = F.softmax(pos_32, dim=-1)
            beta32 = dec_f32.beta_0 * math.exp(-min(4.0, max(0.0, e32)))
            dz32 = torch.relu(pos_32 - neg_32)
            mask32 = (p_pos32 >= beta32 * p_pos32.max()) | ((dz32 >= dec_f32.theta_c * dz32.max()) & (dz32.max() > 1e-6))

            for tag, dt, dec_x in [("bf16", torch.bfloat16, dec_bf16), ("fp16", torch.float16, dec_fp16)]:
                pos_x = pos_32.to(dt)
                neg_x = neg_32.to(dt)
                tok_x, ax, ex, jsx = dec_x.decode_step(pos_x, neg_x, step=step, spatial_confidence=0.82, eos_id=eos_id)
                score_x = (1.0 + ax) * pos_x.float() - ax * neg_x.float()

                stats[tag]["alpha_abs"].append(abs(ax - a32))
                stats[tag]["alpha_rel"].append(abs(ax - a32) / max(1e-6, abs(a32)))
                stats[tag]["ent_abs"].append(abs(ex - e32))
                stats[tag]["js_abs"].append(abs(jsx - js32))
                stats[tag]["score_abs"].append(float((score_x - score32).abs().max().item()))
                stats[tag]["score_rel"].append(_rel_error(score32, score_x))
                if tok_x == tok32:
                    stats[tag]["argmax_match"] += 1
                top5_x = torch.topk(pos_x.float(), k=5, dim=-1).indices[0].tolist()
                if top5_x == top5_32:
                    stats[tag]["top5_match"] += 1
                p_pos_x = F.softmax(pos_x.float(), dim=-1)
                betax = dec_x.beta_0 * math.exp(-min(4.0, max(0.0, ex)))
                dzx = torch.relu(pos_x.float() - neg_x.float())
                mask_x = (p_pos_x >= betax * p_pos_x.max()) | ((dzx >= dec_x.theta_c * dzx.max()) & (dzx.max() > 1e-6))
                if torch.equal(mask_x, mask32):
                    stats[tag]["mask_match"] += 1
                if (tok_x == eos_id) == (tok32 == eos_id):
                    stats[tag]["eos_match"] += 1

        print("\n[PHASE A STAGE-2 LOGIT COMPARISON vs FP32 OVER 50 STEPS]")
        for tag in ("bf16", "fp16"):
            s = stats[tag]
            print(
                f"  {tag.upper()}: max_alpha_abs={max(s['alpha_abs']):.6e}, max_ent_abs={max(s['ent_abs']):.6e}, "
                f"max_js_abs={max(s['js_abs']):.6e}, max_score_abs={max(s['score_abs']):.6e}, "
                f"max_score_rel={max(s['score_rel']):.6e}, argmax_agree={s['argmax_match']}/{N_trials}, "
                f"top5_exact={s['top5_match']}/{N_trials}, mask_exact={s['mask_match']}/{N_trials}, "
                f"eos_agree={s['eos_match']}/{N_trials}"
            )

        # Structural parity invariants (report raw measurements without inventing formal thresholds):
        # FP16 has 10-bit mantissa vs BF16's 7-bit mantissa, so error relative to FP32 is smaller for FP16.
        self.assertLess(max(stats["fp16"]["score_abs"]), max(stats["bf16"]["score_abs"]))
        self.assertLess(max(stats["fp16"]["alpha_abs"]), max(stats["bf16"]["alpha_abs"]))
        self.assertEqual(stats["fp16"]["argmax_match"], N_trials)
        self.assertEqual(stats["fp16"]["eos_match"], N_trials)
        self.assertTrue(all(math.isfinite(x) for x in stats["fp16"]["alpha_abs"]))
        self.assertTrue(all(math.isfinite(x) for x in stats["fp16"]["score_abs"]))

    def test_04_fp16_underflow_overflow_and_nonfinite_handling(self):
        """
        Verify:
        1. Raw FP16 clamp(min=1e-12) or clamp(min=1e-9) flushes to 0.0 in FP16 -> NaN if not promoted to FP32.
        2. DynamicContrastiveDecoder's FP32 promotion eliminates this underflow NaN cleanly.
        3. FP16 overflow (> 65504 -> +inf) or NaN is explicitly caught with ValueError (never silent token 0).
        """
        # Demonstrate underflow hazard in raw float16 without FP32 promotion
        raw_1e9_fp16 = torch.tensor(1e-9, dtype=torch.float16)
        raw_1e12_fp16 = torch.tensor(1e-12, dtype=torch.float16)
        self.assertEqual(float(raw_1e9_fp16.item()), 0.0)
        self.assertEqual(float(raw_1e12_fp16.item()), 0.0)

        # Extreme negative logits in FP16 where smallest probabilities underflow in FP16
        pos_ext = torch.tensor([[10.0, -25.0, -30.0, 2.0]], dtype=torch.float16)
        neg_ext = torch.tensor([[1.0, -25.0, -30.0, 8.0]], dtype=torch.float16)
        dec = DynamicContrastiveDecoder()
        tok, alpha_t, ent, d_js = dec.decode_step(pos_ext, neg_ext, step=0, spatial_confidence=0.9)
        self.assertFalse(math.isnan(alpha_t))
        self.assertFalse(math.isnan(ent))
        self.assertFalse(math.isnan(d_js))
        self.assertEqual(tok, 0)

        # FP16 overflow (> 65504.0 becomes inf)
        overflow_fp16 = torch.tensor([[70000.0, 1.0, 0.0]], dtype=torch.float16)
        self.assertTrue(torch.isposinf(overflow_fp16).any().item())
        with self.assertRaises(ValueError):
            dec.decode_step(overflow_fp16, torch.zeros_like(overflow_fp16), step=0, spatial_confidence=1.0)

        # BF16 handles 70000.0 without overflow
        no_overflow_bf16 = torch.tensor([[70000.0, 1.0, 0.0]], dtype=torch.bfloat16)
        self.assertFalse(torch.isposinf(no_overflow_bf16).any().item())

        # NaN input rejected explicitly
        nan_fp16 = torch.tensor([[float("nan"), 1.0, 0.0]], dtype=torch.float16)
        with self.assertRaises(ValueError):
            dec.decode_step(nan_fp16, torch.zeros_like(nan_fp16), step=0, spatial_confidence=1.0)

    def test_05_synthetic_2layer_transformer_end_to_end_and_opt01_parity(self):
        """
        Run synthetic 2-layer causal transformer across FP32, BF16, FP16:
        - Measure pos/neg/contrastive logit error vs FP32, per-step token agreement, top-k overlap
        - Verify KV-cache shape/dtype contract
        - Verify OPT-01 batched (B=2) vs sequential (B=1) numerical parity in FP16 and BF16
        """
        torch.manual_seed(77)
        base_model = _SyntheticVLM(vocab_size=128, hidden_dim=64, num_layers=2)
        base_sd = {k: v.clone() for k, v in base_model.language_model.model.state_dict().items()}
        head_sd = {k: v.clone() for k, v in base_model.language_model.lm_head.state_dict().items()}
        vis_sd = {k: v.clone() for k, v in base_model.vision_proj.state_dict().items()}
        tok = _TinyTokenizer(vocab_size=128)

        img_pos = Image.new("RGB", (64, 64), (210, 40, 40))
        img_neg = Image.new("RGB", (64, 64), (105, 95, 95))

        results_by_dtype: Dict[str, Dict[str, Any]] = {}
        for tag, prec, dt in [
            ("fp32", None, torch.float32),
            ("bf16", "bf16", torch.bfloat16),
            ("fp16", "fp16", torch.float16),
        ]:
            m = _SyntheticVLM(vocab_size=128, hidden_dim=64, num_layers=2)
            m.language_model.model.load_state_dict(base_sd)
            m.language_model.lm_head.load_state_dict(head_sd)
            m.vision_proj.load_state_dict(vis_sd)
            m.to_dtype(dt)

            # Construct pipeline (bypass resolver for pure FP32 baseline reference)
            if prec is None:
                pipe = SGVCDPipeline(model=m, tokenizer=tok, transform=_identity_transform, precision="bf16", device="cpu")
                pipe.torch_dtype = torch.float32
            else:
                pipe = SGVCDPipeline(model=m, tokenizer=tok, transform=_identity_transform, precision=prec, device="cpu")

            pv_pos = _identity_transform(img_pos).unsqueeze(0).to(dt)
            pv_neg = _identity_transform(img_neg).unsqueeze(0).to(dt)
            with torch.no_grad():
                v_pos = m.extract_feature(pv_pos)
                v_neg = m.extract_feature(pv_neg)
                p_pre, sq, _, _ = pipe.extractor._build_prefix_and_suffix_embeds(v_pos, "cửa hàng tên gì ?")
                n_pre, _, _, _ = pipe.extractor._build_prefix_and_suffix_embeds(v_neg, "cửa hàng tên gì ?")
                pos_in = torch.cat([p_pre, sq], dim=1)
                neg_in = torch.cat([n_pre, sq], dim=1)

            self.assertEqual(pos_in.dtype, dt)
            self.assertEqual(neg_in.dtype, dt)

            bat_trace: List[Dict[str, Any]] = []
            seq_trace: List[Dict[str, Any]] = []
            txt_bat, _, _ = pipe._run_batched_dual_branch_decode(
                pos_in, neg_in, spatial_confidence=0.88, max_new_tokens=10, step_trace=bat_trace
            )
            txt_seq, _, _ = pipe._run_sequential_dual_branch_decode_reference(
                pos_in, neg_in, spatial_confidence=0.88, max_new_tokens=10, step_trace=seq_trace
            )

            # Assert OPT-01 batched vs sequential parity within the same dtype
            self.assertEqual(txt_bat, txt_seq)
            self.assertEqual(len(bat_trace), len(seq_trace))
            max_bat_seq_diff = 0.0
            for b_step, s_step in zip(bat_trace, seq_trace):
                self.assertEqual(b_step["next_token_id"], s_step["next_token_id"])
                diff_p = float((b_step["pos_logits"] - s_step["pos_logits"]).abs().max().item())
                diff_n = float((b_step["neg_logits"] - s_step["neg_logits"]).abs().max().item())
                diff_c = float((b_step["contrastive_logits_pre_mask"] - s_step["contrastive_logits_pre_mask"]).abs().max().item())
                max_bat_seq_diff = max(max_bat_seq_diff, diff_p, diff_n, diff_c)
            self.assertLessEqual(max_bat_seq_diff, 1e-4, f"OPT-01 batched vs sequential drift too high in {tag}: {max_bat_seq_diff}")

            # Verify KV-cache dtype
            with torch.no_grad():
                out = m.language_model.model(inputs_embeds=torch.cat([pos_in, neg_in], dim=0), use_cache=True)
                for k, v in out.past_key_values:
                    self.assertEqual(k.dtype, dt)
                    self.assertEqual(v.dtype, dt)
                    self.assertEqual(k.shape[0], 2)

            results_by_dtype[tag] = {
                "trace": bat_trace,
                "text": txt_bat,
                "opt01_max_diff": max_bat_seq_diff,
            }

        # Compare BF16 & FP16 multi-step traces against FP32 reference
        ref_trace = results_by_dtype["fp32"]["trace"]
        print("\n[PHASE A 2-LAYER TRANSFORMER MULTI-STEP PARITY vs FP32]")
        for tag in ("bf16", "fp16"):
            tr = results_by_dtype[tag]["trace"]
            steps_cmp = min(len(ref_trace), len(tr))
            max_pos_err = max(float((tr[i]["pos_logits"] - ref_trace[i]["pos_logits"]).abs().max().item()) for i in range(steps_cmp))
            max_neg_err = max(float((tr[i]["neg_logits"] - ref_trace[i]["neg_logits"]).abs().max().item()) for i in range(steps_cmp))
            max_cnt_err = max(float((tr[i]["contrastive_logits_pre_mask"] - ref_trace[i]["contrastive_logits_pre_mask"]).abs().max().item()) for i in range(steps_cmp))
            rel_cnt_err = max(_rel_error(ref_trace[i]["contrastive_logits_pre_mask"], tr[i]["contrastive_logits_pre_mask"]) for i in range(steps_cmp))
            tok_agree = sum(1 for i in range(steps_cmp) if tr[i]["next_token_id"] == ref_trace[i]["next_token_id"])
            top5_agree = sum(
                1 for i in range(steps_cmp)
                if torch.topk(tr[i]["pos_logits"], k=5, dim=-1).indices[0].tolist() == torch.topk(ref_trace[i]["pos_logits"], k=5, dim=-1).indices[0].tolist()
            )
            print(
                f"  {tag.upper()}: pos_abs={max_pos_err:.6e}, neg_abs={max_neg_err:.6e}, "
                f"contrastive_abs={max_cnt_err:.6e}, contrastive_rel={rel_cnt_err:.6e}, "
                f"token_agree={tok_agree}/{steps_cmp}, top5_agree={top5_agree}/{steps_cmp}, "
                f"opt01_parity_diff={results_by_dtype[tag]['opt01_max_diff']:.6e}"
            )
            self.assertLess(max_pos_err, 5e-2)
            self.assertLess(max_cnt_err, 5e-2)

    def test_06_stage1_attention_extraction_fp16_vs_bf16_vs_fp32(self):
        """Verify Stage 1 PointwiseKLAttentionExtractor + ROI selection produces consistent maps across FP32, BF16, FP16."""
        torch.manual_seed(99)
        base_model = _SyntheticVLM(vocab_size=128, hidden_dim=64, num_layers=2)
        base_sd = {k: v.clone() for k, v in base_model.language_model.model.state_dict().items()}
        head_sd = {k: v.clone() for k, v in base_model.language_model.lm_head.state_dict().items()}
        vis_sd = {k: v.clone() for k, v in base_model.vision_proj.state_dict().items()}
        tok = _TinyTokenizer(vocab_size=128)
        img = Image.new("RGB", (320, 240), (45, 50, 58))
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 40, 240, 95], fill=(25, 95, 190))
        for x in range(60, 230, 10):
            draw.rectangle([x, 50, x + 4, 85], fill=(245, 245, 240))

        selector = KLDivergenceRegionSelector()
        ext_outputs = {}
        sel_outputs = {}
        for tag, prec, dt in [("fp32", None, torch.float32), ("bf16", "bf16", torch.bfloat16), ("fp16", "fp16", torch.float16)]:
            m = _SyntheticVLM(vocab_size=128, hidden_dim=64, num_layers=2)
            m.language_model.model.load_state_dict(base_sd)
            m.language_model.lm_head.load_state_dict(head_sd)
            m.vision_proj.load_state_dict(vis_sd)
            m.to_dtype(dt)

            ext = PointwiseKLAttentionExtractor(
                model=m,
                tokenizer=tok,
                transform=_identity_transform,
                layer_start=0,
                layer_end=1,
                precision=prec or "bf16",
                device="cpu",
            )
            if prec is None:
                ext.torch_dtype = torch.float32

            res = ext.extract(img, "giá bao nhiêu ?", image_id="stage1_test")
            sel = selector.select(res.phi_map, img.size, res.grid_ratio)
            self.assertEqual(res.vit_embeds.dtype, dt)
            self.assertEqual(res.phi_map.dtype, np.float32)
            self.assertFalse(np.isnan(res.phi_map).any())
            self.assertFalse(np.isnan(res.r_map).any())
            self.assertTrue(math.isfinite(sel.kl_score))
            ext_outputs[tag] = res
            sel_outputs[tag] = sel

        ref = ext_outputs["fp32"]
        ref_sel = sel_outputs["fp32"]
        print("\n[PHASE A STAGE-1 ATTENTION EXTRACTION & ROI COMPARISON vs FP32]")
        for tag in ("bf16", "fp16"):
            cur = ext_outputs[tag]
            cur_sel = sel_outputs[tag]
            phi_abs = float(np.max(np.abs(cur.phi_map - ref.phi_map)))
            r_abs = float(np.max(np.abs(cur.r_map - ref.r_map)))
            aq_abs = float(np.max(np.abs(cur.a_query_map - ref.a_query_map)))
            kl_abs = abs(float(cur_sel.kl_score) - float(ref_sel.kl_score))
            print(
                f"  {tag.upper()}: phi_max_abs={phi_abs:.6e}, r_max_abs={r_abs:.6e}, "
                f"a_query_max_abs={aq_abs:.6e}, bbox={cur_sel.pixel_bbox} (ref={ref_sel.pixel_bbox}), "
                f"kl_score_abs={kl_abs:.6e}"
            )
            self.assertLess(aq_abs, 1e-2)
            self.assertTrue(np.isfinite(cur.phi_map).all())

    def test_07_preflight_fp16_and_bf16_conditional_exception_policy(self):
        """
        Verify Phase B preflight behavior on mocked Tesla T4 (CC 7.5, 15GB VRAM):
        - FP16 passes when runtime FP16 probe succeeds and hard gates pass
        - BF16 stays REJECT without exception AND stays verdict='REJECT' even when Owner exception is set
        - Hard resource failures block even when --allow-bf16-exception is enabled
        """
        fake_props = types.SimpleNamespace(
            name="Tesla T4",
            major=7,
            minor=5,
            total_memory=int(15.0 * (1024 ** 3)),
            multi_processor_count=40,
        )
        fake_vm = types.SimpleNamespace(total=int(12.7 * (1024 ** 3)), available=int(10.0 * (1024 ** 3)), percent=21.0)
        fake_disk = types.SimpleNamespace(total=int(100.0 * (1024 ** 3)), free=int(50.0 * (1024 ** 3)))

        fake_cuda = MagicMock()
        fake_cuda.is_available.return_value = True
        fake_cuda.device_count.return_value = 1
        fake_cuda.get_device_properties.return_value = fake_props
        fake_cuda.mem_get_info.return_value = (int(14.5 * (1024 ** 3)), int(15.0 * (1024 ** 3)))
        fake_cuda.memory_allocated.return_value = 0
        fake_cuda.memory_reserved.return_value = 0
        fake_cuda.is_bf16_supported.return_value = False

        with patch("torch.cuda", fake_cuda), \
             patch("torch.version.cuda", "12.1"), \
             patch("psutil.virtual_memory", return_value=fake_vm), \
             patch("shutil.disk_usage", return_value=fake_disk), \
             patch("scripts.benchmark.colab_gate_b_preflight.probe_fp16_runtime_support", return_value=("PASS", "Mock T4 FP16 OK", True)):
            # 1. FP16 admission on T4 -> PASS
            d_fp16 = audit_hardware_and_environment(precision="fp16")
            self.assertEqual(d_fp16["admission"]["verdict"], "PASS")
            self.assertTrue(d_fp16["admission"]["checks"]["fp16_hardware_support"]["passed"])
            self.assertFalse(d_fp16["admission"]["native_bf16_support"])

            # 2. BF16 admission on T4 without exception -> REJECT
            d_bf16_no_exc = audit_hardware_and_environment(precision="bf16", allow_bf16_exception=False)
            self.assertEqual(d_bf16_no_exc["admission"]["verdict"], "REJECT")
            self.assertFalse(d_bf16_no_exc["admission"]["checks"]["native_bf16_support"]["passed"])
            self.assertFalse(d_bf16_no_exc["admission"]["conditional_exception"]["proceed_under_exception"])

            # 3. BF16 with Owner exception -> verdict STILL REJECT, but proceed_under_exception=True
            d_bf16_exc = audit_hardware_and_environment(precision="bf16", allow_bf16_exception=True)
            self.assertEqual(d_bf16_exc["admission"]["verdict"], "REJECT")
            self.assertFalse(d_bf16_exc["admission"]["native_bf16_support"])
            self.assertTrue(d_bf16_exc["admission"]["conditional_exception"]["authorized"])
            self.assertTrue(d_bf16_exc["admission"]["conditional_exception"]["hard_resource_gates_passed"])
            self.assertTrue(d_bf16_exc["admission"]["conditional_exception"]["proceed_under_exception"])
            rep = format_human_readable_report(d_bf16_exc)
            self.assertIn("OVERALL ADMISSION VERDICT: REJECT", rep)
            self.assertIn("OWNER BF16 EXCEPTION", rep)

            # 4. BF16 with Owner exception when free VRAM < 6GB -> hard gate blocks!
            fake_cuda.mem_get_info.return_value = (int(4.0 * (1024 ** 3)), int(15.0 * (1024 ** 3)))
            d_bf16_oom = audit_hardware_and_environment(precision="bf16", allow_bf16_exception=True)
            self.assertEqual(d_bf16_oom["admission"]["verdict"], "REJECT")
            self.assertFalse(d_bf16_oom["admission"]["conditional_exception"]["hard_resource_gates_passed"])
            self.assertFalse(d_bf16_oom["admission"]["conditional_exception"]["proceed_under_exception"])

    def test_08_inspect_model_dtypes_reports_mixed_buffers(self):
        """
        Verify inspect_model_dtypes does NOT falsely claim every parameter/buffer is FP16
        when submodules retain FP32 buffers (e.g., rotary inv_freq or norm stats).
        """
        from scripts.benchmark.evaluate_vitextvqa import inspect_model_dtypes

        m = _SyntheticVLM(vocab_size=32, hidden_dim=32, num_layers=1)
        m.to_dtype(torch.float16)
        m.vision_model = nn.Linear(4, 4).to(torch.float16)
        m.vision_model.register_buffer("rotary_inv_freq_f32", torch.ones(16, dtype=torch.float32))

        rep = inspect_model_dtypes(m)
        self.assertEqual(rep["status"], "measured")
        self.assertIn("torch.float16", rep["parameters_by_dtype"])
        self.assertIn("torch.float32", rep["buffers_by_dtype"])
        self.assertEqual(rep["buffers_by_dtype"]["torch.float32"], 16)
        self.assertIn("vision_model", rep["modules"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
