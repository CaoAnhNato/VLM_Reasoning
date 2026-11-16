# -*- coding: utf-8 -*-
"""
Unit and Mock Integration Test Suite for TICKET-OPT-01:
Batched Positive/Negative Decoding (B=2) in `SGVCDPipeline._decode_spatial_pmi_contrastive`.

Covers mandatory Gate A local synthetic/mock tests:
- Test A: Two-layer causal transformer numerical parity (pos, neg, and contrastive logits <= 1e-5)
- Test B: Token sequence identity (100% token ID match at every step under greedy decoding)
- Test C: Branch separation and KV-cache correctness (zero cross-branch leakage, seq_len advancement)
- Test D: Transformer forward-call reduction (1 forward pass per decode step vs 2 in reference)
- Test E: Masks, positions, and cache API coverage (HF DynamicCache, .key_cache/.value_cache, legacy tuple)
- Test F: EOS and stopping behavior (step-0 EOS, mid-sequence EOS, max_new_tokens cap)
- Test G: Error and shape handling (mismatched branch shapes, dtypes, masks, and invalid cache states)
- Test H: Existing end-to-end pipeline regression (Stage 1 bboxes/crops + Stage 2 outputs + OPT-02/03/06)
"""

import ast
import copy
import inspect
import math
import os
import sys
import unittest
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.benchmark.sg_vcd.adaptive_propagator import AdaptiveScalePropagator
from scripts.benchmark.sg_vcd.attention_extractor import (
    BaseAttentionExtractor,
    ExtractionResult,
)
from scripts.benchmark.sg_vcd.dynamic_cd import DynamicContrastiveDecoder
from scripts.benchmark.sg_vcd.feature_cache import (
    VisualFeatureCache,
    default_feature_cache,
)
from scripts.benchmark.sg_vcd.pipeline import SGVCDPipeline


def _identity_transform(img: Image.Image) -> torch.Tensor:
    """Deterministic lightweight transform mapping PIL Image -> (3, 32, 32) float32 tensor."""
    arr = np.asarray(img.resize((32, 32), Image.Resampling.BILINEAR).convert("RGB"), dtype=np.float32)
    return torch.from_numpy(arr / 255.0).permute(2, 0, 1).contiguous()


def _make_synthetic_signboard_image(
    width: int = 320,
    height: int = 240,
    bg_color: Tuple[int, int, int] = (45, 50, 58),
    board_bg: Tuple[int, int, int] = (25, 95, 190),
    stroke_color: Tuple[int, int, int] = (245, 245, 240),
) -> Image.Image:
    """Create a deterministic synthetic signboard image with high-contrast strokes."""
    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = int(0.18 * width), int(0.14 * height), int(0.76 * width), int(0.32 * height)
    draw.rectangle([x1, y1, x2, y2], fill=board_bg)
    for x in range(x1 + 10, x2 - 10, 9):
        draw.rectangle([x, y1 + 8, x + 3, y2 - 8], fill=stroke_color)
    return img


class _KeyValueListCache:
    """Cache wrapper exposing `.key_cache` and `.value_cache` lists (Format 2)."""

    def __init__(self, num_layers: int):
        self.key_cache: List[Optional[torch.Tensor]] = [None for _ in range(num_layers)]
        self.value_cache: List[Optional[torch.Tensor]] = [None for _ in range(num_layers)]
        self._seen_tokens: int = 0

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if len(self.key_cache) == 0 or self.key_cache[layer_idx] is None:
            return 0
        return int(self.key_cache[layer_idx].shape[2])

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)
        if layer_idx == 0:
            self._seen_tokens = int(self.key_cache[0].shape[2])
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


class _SyntheticTransformerOutput:
    """Return object mimicking Hugging Face BaseModelOutputWithPast / CausalLMOutputWithPast."""

    def __init__(
        self,
        last_hidden_state: torch.Tensor,
        past_key_values: Any,
        logits: Optional[torch.Tensor] = None,
    ):
        self.last_hidden_state = last_hidden_state
        self.past_key_values = past_key_values
        if logits is not None:
            self.logits = logits


class _SyntheticCausalDecoderLayer(nn.Module):
    """
    Deterministic causal self-attention + SwiGLU MLP transformer block that genuinely computes
    multi-head QKV projections, position-dependent rotary-like sinusoidal phase shifts,
    causal attention masking, and autoregressive KV-cache updates.
    """

    def __init__(self, hidden_dim: int = 32, num_heads: int = 4):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.ln2 = nn.LayerNorm(hidden_dim)
        self.gate_proj = nn.Linear(hidden_dim, hidden_dim * 2, bias=False)
        self.up_proj = nn.Linear(hidden_dim, hidden_dim * 2, bias=False)
        self.down_proj = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)

    def _apply_positional_phase(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply deterministic sinusoidal position-dependent rotation to (B, H, S, D_h)
        so token positions genuinely affect attention scores and hidden states.
        """
        bsz, n_heads, seq_len, head_dim = x.shape
        pos = position_ids.to(device=x.device, dtype=torch.float32).unsqueeze(1).unsqueeze(-1)  # (B, 1, S, 1)
        freq_idx = torch.arange(0, head_dim, device=x.device, dtype=torch.float32).view(1, 1, 1, head_dim)
        angles = pos / torch.pow(100.0, freq_idx / float(max(1, head_dim)))
        cos_v = torch.cos(angles).to(dtype=x.dtype)
        sin_v = torch.sin(angles).to(dtype=x.dtype)
        x_rev = torch.flip(x, dims=[-1])
        return x * cos_v + x_rev * (0.15 * sin_v)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        past_layer_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        bsz, q_len, _ = hidden_states.shape
        normed = self.ln1(hidden_states)

        q = self.q_proj(normed).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(normed).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(normed).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        q = self._apply_positional_phase(q, position_ids)
        k = self._apply_positional_phase(k, position_ids)

        if past_layer_kv is not None:
            past_k, past_v = past_layer_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_kv = (k, v)
        total_len = k.shape[2]
        past_len = total_len - q_len

        # Compute attention in float32 for numerical stability, then cast back
        q_f = q.float()
        k_f = k.float()
        v_f = v.float()
        scores = torch.matmul(q_f, k_f.transpose(-2, -1)) / math.sqrt(float(self.head_dim))

        # Causal mask: query at index i in [0..q_len-1] corresponds to sequence position past_len + i
        q_Idx = torch.arange(past_len, total_len, device=hidden_states.device).unsqueeze(1)
        k_Idx = torch.arange(0, total_len, device=hidden_states.device).unsqueeze(0)
        causal_allowed = k_Idx <= q_Idx  # (q_len, total_len)
        scores = scores.masked_fill(~causal_allowed.unsqueeze(0).unsqueeze(0), float("-inf"))

        if attention_mask is not None:
            # 2D mask (B, total_len) where 1 = keep, 0 = mask
            key_keep = (attention_mask[:, :total_len] > 0).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, total_len)
            scores = scores.masked_fill(~key_keep, float("-inf"))

        attn_probs = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0)
        attn_out = torch.matmul(attn_probs, v_f).to(dtype=hidden_states.dtype)
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_dim)
        hidden_states = hidden_states + self.o_proj(attn_out)

        normed2 = self.ln2(hidden_states)
        mlp_out = self.down_proj(F.silu(self.gate_proj(normed2)) * self.up_proj(normed2))
        hidden_states = hidden_states + mlp_out
        return hidden_states, present_kv


class _SyntheticTwoLayerCausalTransformer(nn.Module):
    """
    Deterministic two-layer causal transformer backbone supporting both legacy tuple KV-caches
    and `.key_cache`/`.value_cache` objects, plus call instrumentation for Test D.
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        vocab_size: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        cache_mode: str = "tuple",
        dtype: torch.dtype = torch.float32,
        seed: int = 42,
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.cache_mode = cache_mode
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList(
            [_SyntheticCausalDecoderLayer(hidden_dim=hidden_dim, num_heads=num_heads) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.config = type("Cfg", (), {"_attn_implementation": "sdpa"})()
        self.to(dtype=dtype)
        self.eval()

        self.call_log: List[Dict[str, Any]] = []

    def reset_call_log(self) -> None:
        self.call_log.clear()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(
        self,
        inputs_embeds: torch.Tensor = None,
        past_key_values: Any = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        return_dict: bool = True,
        **kwargs,
    ) -> _SyntheticTransformerOutput:
        bsz, q_len, _ = inputs_embeds.shape
        is_prefill = past_key_values is None
        past_len = 0
        if past_key_values is not None:
            if isinstance(past_key_values, _KeyValueListCache):
                past_len = past_key_values.get_seq_length(0)
            elif isinstance(past_key_values, (tuple, list)) and len(past_key_values) > 0:
                past_len = int(past_key_values[0][0].shape[2])

        self.call_log.append(
            {
                "batch_size": int(bsz),
                "q_len": int(q_len),
                "past_len": int(past_len),
                "is_prefill": bool(is_prefill),
            }
        )

        if position_ids is None:
            position_ids = (
                torch.arange(past_len, past_len + q_len, device=inputs_embeds.device, dtype=torch.long)
                .unsqueeze(0)
                .expand(bsz, q_len)
            )
        elif position_ids.shape[0] == 1 and bsz > 1:
            position_ids = position_ids.expand(bsz, q_len)

        hidden = inputs_embeds
        if self.cache_mode == "key_value_list":
            kv_obj = past_key_values if isinstance(past_key_values, _KeyValueListCache) else _KeyValueListCache(self.num_layers)
            for idx, layer in enumerate(self.layers):
                past_pair = None
                if kv_obj.key_cache[idx] is not None:
                    past_pair = (kv_obj.key_cache[idx], kv_obj.value_cache[idx])
                hidden, (new_k, new_v) = layer(
                    hidden,
                    position_ids=position_ids,
                    past_layer_kv=past_pair,
                    attention_mask=None if attention_mask is None else attention_mask,
                )
                if new_k is not None:
                    kv_obj.key_cache[idx] = new_k
                    kv_obj.value_cache[idx] = new_v
            kv_obj._seen_tokens = int(kv_obj.key_cache[0].shape[2])
            next_cache = kv_obj if use_cache else None
        else:
            next_layers = []
            for idx, layer in enumerate(self.layers):
                past_pair = past_key_values[idx] if past_key_values is not None else None
                hidden, present_pair = layer(
                    hidden,
                    position_ids=position_ids,
                    past_layer_kv=past_pair,
                    attention_mask=attention_mask,
                )
                next_layers.append(present_pair)
            next_cache = tuple(next_layers) if use_cache else None

        hidden = self.final_norm(hidden)
        return _SyntheticTransformerOutput(last_hidden_state=hidden, past_key_values=next_cache)


class _SyntheticLanguageModel(nn.Module):
    """Wraps `_SyntheticTwoLayerCausalTransformer` + `lm_head` matching Vintern/Qwen2 structure."""

    def __init__(
        self,
        hidden_dim: int = 32,
        vocab_size: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        cache_mode: str = "tuple",
        dtype: torch.dtype = torch.float32,
        seed: int = 42,
    ):
        super().__init__()
        self.model = _SyntheticTwoLayerCausalTransformer(
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            num_layers=num_layers,
            num_heads=num_heads,
            cache_mode=cache_mode,
            dtype=dtype,
            seed=seed,
        )
        torch.manual_seed(seed + 100)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=True).to(dtype=dtype)
        with torch.no_grad():
            self.lm_head.bias.zero_()
        self.config = self.model.config
        self.eval()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def forward(
        self,
        inputs_embeds: torch.Tensor = None,
        past_key_values: Any = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        return_dict: bool = True,
        **kwargs,
    ) -> _SyntheticTransformerOutput:
        out = self.model(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            return_dict=return_dict,
            **kwargs,
        )
        logits = self.lm_head(out.last_hidden_state)
        return _SyntheticTransformerOutput(
            last_hidden_state=out.last_hidden_state,
            past_key_values=out.past_key_values,
            logits=logits,
        )


class _SyntheticVinternModel(nn.Module):
    """Synthetic VLM with content-sensitive ViT feature projection and 2-layer causal LLM."""

    def __init__(
        self,
        hidden_dim: int = 32,
        vocab_size: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        cache_mode: str = "tuple",
        dtype: torch.dtype = torch.bfloat16,
        seed: int = 42,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.dtype = dtype
        self.num_image_token = 16
        self.img_context_token_id = 4
        self.language_model = _SyntheticLanguageModel(
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            num_layers=num_layers,
            num_heads=num_heads,
            cache_mode=cache_mode,
            dtype=dtype,
            seed=seed,
        )
        torch.manual_seed(seed + 200)
        self.vit_proj = nn.Linear(48, hidden_dim, bias=True).to(dtype=dtype)
        self.eval()

    def extract_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Deterministically project each input image tile `(B, 3, H, W)` into
        `(B, num_image_token=16, hidden_dim)` so sharp positive ROI tiles and
        Gaussian-blurred negative ROI tiles produce distinct visual embeddings.
        """
        bsz = pixel_values.shape[0]
        pooled = F.adaptive_avg_pool2d(pixel_values.float(), (4, 4))  # (B, 3, 4, 4)
        high_freq = (pixel_values.float() - F.avg_pool2d(pixel_values.float(), kernel_size=3, stride=1, padding=1)).abs()
        hf_pooled = F.adaptive_avg_pool2d(high_freq, (4, 4))  # (B, 3, 4, 4)
        patches = (pooled + 2.5 * hf_pooled).permute(0, 2, 3, 1).reshape(bsz, 16, 3)
        # Expand 3 channels to 48 features via deterministic harmonic basis
        scales = torch.arange(1, 17, device=pixel_values.device, dtype=torch.float32).view(1, 1, 16, 1)
        feat48 = torch.sin(patches.unsqueeze(2) * scales).reshape(bsz, 16, 48).to(dtype=self.dtype)
        return self.vit_proj(feat48)


class _SyntheticTokenizer:
    """Deterministic vocabulary tokenizer for testing Stage 2 decoding."""

    eos_token_id = 2
    im_end_id = 3
    img_ctx_id = 4

    def __init__(self, delimiter_token_id: int = 59):
        self.delimiter_token_id = delimiter_token_id
        self._id_to_tok = {
            0: "<pad>",
            1: "<bos>",
            2: "<eos>",
            3: "<|im_end|>",
            4: "<IMG_CONTEXT>",
            self.delimiter_token_id: ":",
        }
        for tid in range(5, 64):
            if tid not in self._id_to_tok:
                self._id_to_tok[tid] = f"tok{tid}"

    def __call__(self, text: str, return_tensors: str = "pt", add_special_tokens: bool = False):
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        return type("TokOut", (), {"input_ids": torch.tensor([ids], dtype=torch.long)})()

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        if not text:
            return [10, 11]
        ctx_count = text.count("<IMG_CONTEXT>")
        if ctx_count > 0:
            return [8, 9] + ([self.img_ctx_id] * ctx_count) + [10, 11]
        return [min(58, 12 + ((ord(c) + idx * 3) % 45)) for idx, c in enumerate(text[:12])] or [10]

    def convert_tokens_to_ids(self, token: str) -> int:
        if token == "<IMG_CONTEXT>":
            return self.img_ctx_id
        if token == "<|im_end|>":
            return self.im_end_id
        return 3

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        parts = []
        for tid in token_ids:
            t_int = int(tid)
            if skip_special_tokens and t_int in (0, 1, 2, 3, 4):
                continue
            parts.append(self._id_to_tok.get(t_int, f"tok{t_int}"))
        return " ".join(parts)


class _DeterministicStage1Extractor(BaseAttentionExtractor):
    """Lightweight deterministic Stage 1 extractor that also builds real prefix/suffix embeddings."""

    def __init__(self, model: _SyntheticVinternModel, tokenizer: _SyntheticTokenizer, grid_ratio=(2, 2)):
        self.model = model
        self.tokenizer = tokenizer
        self.grid_ratio = grid_ratio
        self._cached_img_key = None
        self._cached_img_state = None

    def clear_cache(self) -> None:
        self._cached_img_key = None
        self._cached_img_state = None

    def _build_prefix_and_suffix_embeds(
        self, vit_embeds: torch.Tensor, question: str
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        num_patches = int(vit_embeds.shape[0])
        num_img_tok = int(vit_embeds.shape[1])
        total_vis = num_patches * num_img_tok

        embed_layer = self.model.language_model.get_input_embeddings()
        prefix_ids = torch.tensor(
            [[8, 9] + ([self.tokenizer.img_ctx_id] * total_vis) + [10, 11]],
            dtype=torch.long,
            device=vit_embeds.device,
        )
        q_ids = torch.tensor(
            [self.tokenizer.encode(question)],
            dtype=torch.long,
            device=vit_embeds.device,
        )
        prefix_embeds = embed_layer(prefix_ids).clone()
        suffix_embeds = embed_layer(q_ids)

        vis_start = 2
        vis_end = 2 + total_vis
        flat_vit = vit_embeds.reshape(-1, vit_embeds.shape[-1]).to(dtype=prefix_embeds.dtype)
        prefix_embeds[0, vis_start:vis_end, :] = flat_vit
        return prefix_embeds, suffix_embeds, vis_start, vis_end

    def extract(
        self,
        image: Image.Image,
        question: str,
        *,
        image_key=None,
        **kwargs,
    ) -> ExtractionResult:
        gx, gy = self.grid_ratio
        H_g, W_g = 16 * gy, 16 * gx
        phi = np.full((H_g, W_g), 0.02, dtype=np.float32)
        hy0, hy1 = int(0.14 * H_g), int(0.32 * H_g)
        hx0, hx1 = int(0.18 * W_g), int(0.76 * W_g)
        phi[hy0:hy1, hx0:hx1] = 2.5
        phi[(hy0 + hy1) // 2, (hx0 + hx1) // 2] = 4.2
        return ExtractionResult(
            phi_map=phi,
            r_map=phi.copy(),
            aff_map=np.ones_like(phi),
            a_query_map=phi.copy(),
            a_gen_map=np.full_like(phi, 1.0 / float(H_g * W_g)),
            grid_ratio=self.grid_ratio,
        )


class TestOPT01BatchedDualBranchDecoding(unittest.TestCase):
    """Comprehensive verification of TICKET-OPT-01: Batched Positive/Negative Decoding (B=2)."""

    def setUp(self):
        default_feature_cache.clear()
        self.tokenizer = _SyntheticTokenizer()

    def tearDown(self):
        default_feature_cache.clear()

    def _make_pipeline(
        self,
        dtype: torch.dtype = torch.float32,
        cache_mode: str = "tuple",
        seed: int = 42,
    ) -> SGVCDPipeline:
        model = _SyntheticVinternModel(
            hidden_dim=32,
            vocab_size=64,
            num_layers=2,
            num_heads=4,
            cache_mode=cache_mode,
            dtype=dtype,
            seed=seed,
        )
        # Suppress EOS tokens (2 and 3) by default so multi-step decoding runs cleanly
        with torch.no_grad():
            model.language_model.lm_head.weight[2, :].zero_()
            model.language_model.lm_head.weight[3, :].zero_()
            model.language_model.lm_head.bias[2] = -50.0
            model.language_model.lm_head.bias[3] = -50.0

        extractor = _DeterministicStage1Extractor(model=model, tokenizer=self.tokenizer)
        return SGVCDPipeline(
            model=model,
            tokenizer=self.tokenizer,
            transform=_identity_transform,
            extractor=extractor,
            propagator=AdaptiveScalePropagator(work_max_side=320, device="cpu"),
            feature_cache=VisualFeatureCache(max_entries=8),
            device="cpu",
        )

    def _assert_step_traces_match(
        self,
        trace_bat: List[Dict[str, Any]],
        trace_seq: List[Dict[str, Any]],
        tol: float = 1e-5,
    ) -> Dict[str, float]:
        """
        Compare positive logits, negative logits, contrastive scores, and selected token IDs
        at every decoding step against the strict `<= 1e-5` float32 tolerance.
        """
        self.assertEqual(len(trace_bat), len(trace_seq))
        self.assertGreater(len(trace_bat), 0)

        max_pos_diff = 0.0
        max_neg_diff = 0.0
        max_pre_mask_diff = 0.0
        max_signed_diff = 0.0
        max_final_diff = 0.0

        for idx, (b_step, s_step) in enumerate(zip(trace_bat, trace_seq)):
            self.assertEqual(b_step["step"], idx)
            self.assertEqual(s_step["step"], idx)

            pos_diff = float((b_step["pos_logits"] - s_step["pos_logits"]).abs().max().item())
            neg_diff = float((b_step["neg_logits"] - s_step["neg_logits"]).abs().max().item())
            pre_diff = float(
                (b_step["contrastive_logits_pre_mask"] - s_step["contrastive_logits_pre_mask"])
                .abs()
                .max()
                .item()
            )
            sig_diff = float(
                (b_step["contrastive_logits_signed"] - s_step["contrastive_logits_signed"])
                .abs()
                .max()
                .item()
            )

            # For final masked logits, verify identical -inf mask and compare finite entries
            b_final = b_step["contrastive_logits_final"]
            s_final = s_step["contrastive_logits_final"]
            b_finite = torch.isfinite(b_final)
            s_finite = torch.isfinite(s_final)
            self.assertTrue(
                torch.equal(b_finite, s_finite),
                f"Step {idx} candidate mask mismatch between batched and sequential paths",
            )
            fin_diff = (
                float((b_final[b_finite] - s_final[s_finite]).abs().max().item())
                if bool(b_finite.any())
                else 0.0
            )

            max_pos_diff = max(max_pos_diff, pos_diff)
            max_neg_diff = max(max_neg_diff, neg_diff)
            max_pre_mask_diff = max(max_pre_mask_diff, pre_diff)
            max_signed_diff = max(max_signed_diff, sig_diff)
            max_final_diff = max(max_final_diff, fin_diff)

            self.assertLessEqual(pos_diff, tol, f"Step {idx} pos_logits diff {pos_diff} > {tol}")
            self.assertLessEqual(neg_diff, tol, f"Step {idx} neg_logits diff {neg_diff} > {tol}")
            self.assertLessEqual(pre_diff, tol, f"Step {idx} contrastive pre-mask diff {pre_diff} > {tol}")
            self.assertLessEqual(sig_diff, tol, f"Step {idx} contrastive signed diff {sig_diff} > {tol}")
            self.assertLessEqual(fin_diff, tol, f"Step {idx} contrastive final diff {fin_diff} > {tol}")

            self.assertEqual(
                b_step["next_token_id"],
                s_step["next_token_id"],
                f"Step {idx} token deviation: batched={b_step['next_token_id']} vs seq={s_step['next_token_id']}",
            )
            self.assertEqual(b_step["cache_seq_len"], s_step["cache_seq_len"])
            self.assertAlmostEqual(b_step["alpha_t"], s_step["alpha_t"], places=6)
            self.assertAlmostEqual(b_step["entropy_t"], s_step["entropy_t"], places=6)
            self.assertAlmostEqual(b_step["d_js_val"], s_step["d_js_val"], places=6)

        return {
            "max_pos_diff": max_pos_diff,
            "max_neg_diff": max_neg_diff,
            "max_pre_mask_diff": max_pre_mask_diff,
            "max_signed_diff": max_signed_diff,
            "max_final_diff": max_final_diff,
        }

    # ------------------------------------------------------------------
    # Test A — Two-layer transformer numerical parity
    # ------------------------------------------------------------------
    def test_A_two_layer_transformer_numerical_parity(self):
        """
        Test A — Two-layer transformer numerical parity:
        Verify positive logits, negative logits, and contrastive logits match within <= 1e-5
        across every decoding step on both the synthetic 2-layer causal transformer and a
        2-layer Hugging Face Qwen2ForCausalLM in both float32 and bfloat16.
        """
        for dtype in (torch.float32, torch.bfloat16):
            pipe = self._make_pipeline(dtype=dtype, seed=101)
            torch.manual_seed(202)
            pos_inputs = torch.randn((1, 20, 32), dtype=dtype)
            neg_inputs = torch.randn((1, 20, 32), dtype=dtype)

            trace_bat: List[Dict[str, Any]] = []
            trace_seq: List[Dict[str, Any]] = []

            text_bat, alpha_bat, ent_bat = pipe._run_batched_dual_branch_decode(
                pos_inputs=pos_inputs.clone(),
                neg_inputs=neg_inputs.clone(),
                spatial_confidence=0.82,
                max_new_tokens=8,
                step_trace=trace_bat,
            )
            text_seq, alpha_seq, ent_seq = pipe._run_sequential_dual_branch_decode_reference(
                pos_inputs=pos_inputs.clone(),
                neg_inputs=neg_inputs.clone(),
                spatial_confidence=0.82,
                max_new_tokens=8,
                step_trace=trace_seq,
            )

            self.assertEqual(len(trace_bat), 8)
            diffs = self._assert_step_traces_match(trace_bat, trace_seq, tol=1e-5)
            self.assertLessEqual(diffs["max_pos_diff"], 1e-5)
            self.assertLessEqual(diffs["max_neg_diff"], 1e-5)
            self.assertLessEqual(diffs["max_pre_mask_diff"], 1e-5)
            self.assertEqual(text_bat, text_seq)
            self.assertAlmostEqual(alpha_bat, alpha_seq, places=6)
            self.assertAlmostEqual(ent_bat, ent_seq, places=6)

        # Also verify on actual installed Hugging Face 2-layer Qwen2ForCausalLM (with DynamicCache)
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Config, Qwen2ForCausalLM

        for dtype in (torch.float32, torch.bfloat16):
            torch.manual_seed(303)
            cfg = Qwen2Config(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=256,
                use_cache=True,
            )
            qwen_lm = Qwen2ForCausalLM(cfg).to(dtype=dtype).eval()
            qwen_lm.lm_head = nn.Linear(32, 64, bias=True).to(dtype=dtype)
            with torch.no_grad():
                qwen_lm.lm_head.bias.zero_()
                qwen_lm.lm_head.weight[2, :].zero_()
                qwen_lm.lm_head.weight[3, :].zero_()
                qwen_lm.lm_head.bias[2] = -50.0
                qwen_lm.lm_head.bias[3] = -50.0

            pipe_qwen = self._make_pipeline(dtype=dtype, seed=303)
            pipe_qwen.model.language_model = qwen_lm

            pos_in = torch.randn((1, 18, 32), dtype=dtype)
            neg_in = torch.randn((1, 18, 32), dtype=dtype)
            tr_b: List[Dict[str, Any]] = []
            tr_s: List[Dict[str, Any]] = []

            tb, ab, eb = pipe_qwen._run_batched_dual_branch_decode(
                pos_in.clone(), neg_in.clone(), spatial_confidence=0.75, max_new_tokens=6, step_trace=tr_b
            )
            ts, as_, es = pipe_qwen._run_sequential_dual_branch_decode_reference(
                pos_in.clone(), neg_in.clone(), spatial_confidence=0.75, max_new_tokens=6, step_trace=tr_s
            )
            self.assertEqual(len(tr_b), 6)
            self._assert_step_traces_match(tr_b, tr_s, tol=1e-5)
            self.assertEqual(tb, ts)
            self.assertAlmostEqual(ab, as_, places=6)
            self.assertAlmostEqual(eb, es, places=6)

    # ------------------------------------------------------------------
    # Test B — Token sequence identity
    # ------------------------------------------------------------------
    def test_B_token_sequence_identity_across_diverse_workloads(self):
        """
        Test B — Token sequence identity:
        Run deterministic greedy decoding across 6 distinct synthetic visual/question scenarios
        (including multi-tile ROI, context tile, ESSC cooldown, and two-phase colon delimiter)
        and assert 100% token ID identity at every step and for the final generated sequence.
        """
        pipe = self._make_pipeline(dtype=torch.bfloat16, seed=77)
        img_main = _make_synthetic_signboard_image()
        img_alt = _make_synthetic_signboard_image(
            board_bg=(160, 35, 40), stroke_color=(250, 240, 190)
        )
        thumb = img_main.resize((448, 448), Image.Resampling.BICUBIC)
        roi1 = img_main.crop((40, 20, 260, 90)).resize((448, 448), Image.Resampling.BICUBIC)
        roi2 = img_alt.crop((30, 30, 250, 100)).resize((448, 448), Image.Resampling.BICUBIC)

        scenarios = [
            dict(
                question="Tên nhà thuốc trên biển hiệu là gì?",
                stage1_kl_score=1.4,
                max_new_tokens=10,
                roi_tiles=[roi1],
                magnifications=[2.2],
                char_heights=[16.0],
                context_tile=None,
            ),
            dict(
                question="Địa chỉ và số điện thoại liên hệ?",
                stage1_kl_score=0.9,
                max_new_tokens=12,
                roi_tiles=[roi1, roi2],
                magnifications=[2.0, 1.8],
                char_heights=[14.0, 18.0],
                context_tile=None,
            ),
            dict(
                question="Biển hiệu phụ ghi nội dung gì?",
                stage1_kl_score=0.35,
                max_new_tokens=8,
                roi_tiles=[roi2],
                magnifications=[1.5],
                char_heights=[None],
                context_tile=thumb,
            ),
            dict(
                question="Low spatial confidence fallback test?",
                stage1_kl_score=0.0,
                max_new_tokens=7,
                roi_tiles=[roi1],
                magnifications=[1.0],
                char_heights=[22.0],
                context_tile=None,
            ),
        ]

        for idx, sc in enumerate(scenarios):
            with self.subTest(scenario=idx):
                text_bat, alpha_bat, ent_bat = pipe._decode_spatial_pmi_contrastive(
                    roi_tile=roi1,
                    global_thumb=thumb,
                    question=sc["question"],
                    stage1_kl_score=sc["stage1_kl_score"],
                    max_new_tokens=sc["max_new_tokens"],
                    char_height_px=sc["char_heights"][0],
                    magnification=sc["magnifications"][0],
                    context_tile=sc["context_tile"],
                    roi_tiles=sc["roi_tiles"],
                    magnifications=sc["magnifications"],
                    char_heights=sc["char_heights"],
                )
                text_seq, alpha_seq, ent_seq = pipe._decode_spatial_pmi_contrastive_sequential_reference(
                    roi_tile=roi1,
                    global_thumb=thumb,
                    question=sc["question"],
                    stage1_kl_score=sc["stage1_kl_score"],
                    max_new_tokens=sc["max_new_tokens"],
                    char_height_px=sc["char_heights"][0],
                    magnification=sc["magnifications"][0],
                    context_tile=sc["context_tile"],
                    roi_tiles=sc["roi_tiles"],
                    magnifications=sc["magnifications"],
                    char_heights=sc["char_heights"],
                )
                self.assertEqual(text_bat, text_seq)
                self.assertAlmostEqual(alpha_bat, alpha_seq, places=6)
                self.assertAlmostEqual(ent_bat, ent_seq, places=6)

        # Explicitly test two-phase delimiter reasoning transition (colon token id 59 emitted at step 0)
        pipe_delim = self._make_pipeline(dtype=torch.float32, seed=88)
        with torch.no_grad():
            pipe_delim.model.language_model.lm_head.bias[59] = 25.0
        pos_in = torch.randn((1, 14, 32), dtype=torch.float32)
        neg_in = torch.randn((1, 14, 32), dtype=torch.float32)
        tr_b, tr_s = [], []
        tb, ab, eb = pipe_delim._run_batched_dual_branch_decode(
            pos_in.clone(), neg_in.clone(), spatial_confidence=0.9, max_new_tokens=6, step_trace=tr_b
        )
        ts, as_, es = pipe_delim._run_sequential_dual_branch_decode_reference(
            pos_in.clone(), neg_in.clone(), spatial_confidence=0.9, max_new_tokens=6, step_trace=tr_s
        )
        self.assertEqual([x["next_token_id"] for x in tr_b], [x["next_token_id"] for x in tr_s])
        self.assertEqual(tr_b[0]["next_token_id"], 59)
        # After colon delimiter at step 0, step >= 1 must have step_spatial_conf == 0.0 -> alpha_t == 0.0
        for s_idx in range(1, len(tr_b)):
            self.assertEqual(tr_b[s_idx]["alpha_t"], 0.0)
            self.assertEqual(tr_s[s_idx]["alpha_t"], 0.0)
        self.assertEqual(tb, ts)

    # ------------------------------------------------------------------
    # Test C — Branch separation and cache correctness
    # ------------------------------------------------------------------
    def test_C_branch_separation_and_cache_correctness(self):
        """
        Test C — Branch separation and cache correctness:
        1. Positive cache remains at batch index 0; negative cache remains at batch index 1.
        2. Different positive/negative prefixes produce distinct branch outputs.
        3. Perturbing negative prefix produces 0.0 change in batch index 0 K/V cache and pos_logits;
           perturbing positive prefix produces 0.0 change in batch index 1 K/V cache and neg_logits.
        4. Cache sequence length advances by exactly +1 for each generated token.
        """
        pipe = self._make_pipeline(dtype=torch.float32, seed=55)
        llm = pipe.model.language_model.model
        lm_head = pipe.model.language_model.lm_head
        embed_layer = pipe.model.language_model.get_input_embeddings()

        torch.manual_seed(550)
        pos_A = torch.randn((1, 15, 32), dtype=torch.float32)
        neg_B = torch.randn((1, 15, 32), dtype=torch.float32)
        neg_C = torch.randn((1, 15, 32), dtype=torch.float32) * 3.0 + 1.5
        pos_D = torch.randn((1, 15, 32), dtype=torch.float32) * -2.5 - 1.0

        with torch.no_grad():
            # Run batched (pos_A, neg_B) vs (pos_A, neg_C) vs (pos_D, neg_B) across prefill + 4 fixed token steps
            bat_AB_out = llm(inputs_embeds=torch.cat([pos_A, neg_B], dim=0), use_cache=True, return_dict=True)
            bat_AC_out = llm(inputs_embeds=torch.cat([pos_A, neg_C], dim=0), use_cache=True, return_dict=True)
            bat_DB_out = llm(inputs_embeds=torch.cat([pos_D, neg_B], dim=0), use_cache=True, return_dict=True)

            # Separate single-branch reference runs
            ref_pos_out = llm(inputs_embeds=pos_A, use_cache=True, return_dict=True)
            ref_neg_out = llm(inputs_embeds=neg_B, use_cache=True, return_dict=True)

            kv_AB = bat_AB_out.past_key_values
            kv_AC = bat_AC_out.past_key_values
            kv_DB = bat_DB_out.past_key_values
            kv_pos = ref_pos_out.past_key_values
            kv_neg = ref_neg_out.past_key_values

            # Verify prefill sequence length is 15 and distinct prefixes give distinct logits
            self.assertEqual(pipe._validate_kv_cache(kv_AB, expected_batch_size=2, expected_seq_len=15), 15)
            logits_AB = lm_head(bat_AB_out.last_hidden_state[:, -1:, :])[:, -1, :]
            self.assertFalse(torch.allclose(logits_AB[0:1], logits_AB[1:2]))

            fixed_tokens = [14, 21, 33, 42]
            for step_idx, tok_id in enumerate(fixed_tokens):
                nxt = embed_layer(torch.tensor([[tok_id]], dtype=torch.long))
                nxt_bat = torch.cat([nxt, nxt], dim=0)

                s_AB = llm(inputs_embeds=nxt_bat, past_key_values=kv_AB, use_cache=True, return_dict=True)
                s_AC = llm(inputs_embeds=nxt_bat, past_key_values=kv_AC, use_cache=True, return_dict=True)
                s_DB = llm(inputs_embeds=nxt_bat, past_key_values=kv_DB, use_cache=True, return_dict=True)
                s_pos = llm(inputs_embeds=nxt, past_key_values=kv_pos, use_cache=True, return_dict=True)
                s_neg = llm(inputs_embeds=nxt, past_key_values=kv_neg, use_cache=True, return_dict=True)

                kv_AB = s_AB.past_key_values
                kv_AC = s_AC.past_key_values
                kv_DB = s_DB.past_key_values
                kv_pos = s_pos.past_key_values
                kv_neg = s_neg.past_key_values

                expected_len = 15 + step_idx + 1
                self.assertEqual(
                    pipe._validate_kv_cache(kv_AB, expected_batch_size=2, expected_seq_len=expected_len),
                    expected_len,
                )

                for layer_i in range(len(kv_AB)):
                    k_ab, v_ab = kv_AB[layer_i]
                    k_ac, v_ac = kv_AC[layer_i]
                    k_db, v_db = kv_DB[layer_i]
                    k_p, v_p = kv_pos[layer_i]
                    k_n, v_n = kv_neg[layer_i]

                    # Index 0 matches standalone positive branch; Index 1 matches standalone negative branch
                    self.assertLessEqual((k_ab[0:1] - k_p).abs().max().item(), 1e-5)
                    self.assertLessEqual((v_ab[0:1] - v_p).abs().max().item(), 1e-5)
                    self.assertLessEqual((k_ab[1:2] - k_n).abs().max().item(), 1e-5)
                    self.assertLessEqual((v_ab[1:2] - v_n).abs().max().item(), 1e-5)

                    # Zero cross-branch leakage: changing neg_B -> neg_C leaves batch index 0 bitwise identical
                    self.assertEqual((k_ab[0:1] - k_ac[0:1]).abs().max().item(), 0.0)
                    self.assertEqual((v_ab[0:1] - v_ac[0:1]).abs().max().item(), 0.0)
                    # Changing pos_A -> pos_D leaves batch index 1 bitwise identical
                    self.assertEqual((k_ab[1:2] - k_db[1:2]).abs().max().item(), 0.0)
                    self.assertEqual((v_ab[1:2] - v_db[1:2]).abs().max().item(), 0.0)

    # ------------------------------------------------------------------
    # Test D — Forward-call reduction
    # ------------------------------------------------------------------
    def test_D_forward_call_reduction(self):
        """
        Test D — Forward-call reduction:
        Instrument the synthetic transformer forward method. Verify that for M generated tokens
        (stopping at max_new_tokens=M without early EOS):
        - Batched path uses 1 prefill forward call (B=2) + M decode-step forward calls (B=2) = 1 + M total.
        - Sequential reference path uses 2 prefill forward calls (B=1) + 2*M decode-step forward calls (B=1) = 2 + 2*M total.
        """
        pipe = self._make_pipeline(dtype=torch.bfloat16, seed=123)
        llm = pipe.model.language_model.model
        pos_in = torch.randn((1, 22, 32), dtype=torch.bfloat16)
        neg_in = torch.randn((1, 22, 32), dtype=torch.bfloat16)

        max_tokens = 6

        # 1. Batched path
        llm.reset_call_log()
        _ = pipe._run_batched_dual_branch_decode(
            pos_inputs=pos_in.clone(),
            neg_inputs=neg_in.clone(),
            spatial_confidence=0.85,
            max_new_tokens=max_tokens,
        )
        bat_calls = list(llm.call_log)
        bat_prefill = [c for c in bat_calls if c["is_prefill"]]
        bat_decode = [c for c in bat_calls if not c["is_prefill"]]

        self.assertEqual(len(bat_prefill), 1, "Batched path must execute exactly 1 prefill forward pass")
        self.assertEqual(bat_prefill[0]["batch_size"], 2)
        self.assertEqual(bat_prefill[0]["q_len"], 22)
        self.assertEqual(
            len(bat_decode),
            max_tokens,
            f"Batched path must execute 1 forward pass per decode step ({max_tokens} steps)",
        )
        for step_idx, c in enumerate(bat_decode):
            self.assertEqual(c["batch_size"], 2)
            self.assertEqual(c["q_len"], 1)
            self.assertEqual(c["past_len"], 22 + step_idx)

        # 2. Sequential reference path
        llm.reset_call_log()
        _ = pipe._run_sequential_dual_branch_decode_reference(
            pos_inputs=pos_in.clone(),
            neg_inputs=neg_in.clone(),
            spatial_confidence=0.85,
            max_new_tokens=max_tokens,
        )
        seq_calls = list(llm.call_log)
        seq_prefill = [c for c in seq_calls if c["is_prefill"]]
        seq_decode = [c for c in seq_calls if not c["is_prefill"]]

        self.assertEqual(len(seq_prefill), 2, "Reference path must execute 2 prefill forward passes")
        self.assertEqual(
            len(seq_decode),
            2 * max_tokens,
            f"Reference path must execute 2 forward passes per decode step ({2 * max_tokens} calls)",
        )
        for c in seq_calls:
            self.assertEqual(c["batch_size"], 1)

        # Exact 2x reduction in both prefill and decode steps
        self.assertEqual(len(seq_prefill), 2 * len(bat_prefill))
        self.assertEqual(len(seq_decode), 2 * len(bat_decode))

    # ------------------------------------------------------------------
    # Test E — Masks, positions and cache API
    # ------------------------------------------------------------------
    def test_E_masks_positions_and_cache_apis(self):
        """
        Test E — Masks, positions and cache API:
        Verify all supported KV-cache representations:
        1. Hugging Face DynamicCache (`.layers`)
        2. `.key_cache` / `.value_cache` list cache wrapper
        3. Legacy tuple `(K, V)` cache
        4. Direct `language_model(...)` CausalLM path (`has_backbone == False`)
        Plus explicit `attention_mask` and `position_ids` propagation across prefill and decode steps.
        """
        pos_in = torch.randn((1, 16, 32), dtype=torch.float32)
        neg_in = torch.randn((1, 16, 32), dtype=torch.float32)

        # 1. `.key_cache` / `.value_cache` cache format + explicit attention_mask & position_ids
        pipe_kvlist = self._make_pipeline(dtype=torch.float32, cache_mode="key_value_list", seed=61)
        attn_mask = torch.ones((2, 16), dtype=torch.long)
        attn_mask[:, 2] = 0  # Mask out index 2 on both branches identically
        pos_ids = torch.arange(10, 26, dtype=torch.long).unsqueeze(0).repeat(2, 1)

        tr_b, tr_s = [], []
        tb, ab, eb = pipe_kvlist._run_batched_dual_branch_decode(
            pos_in.clone(),
            neg_in.clone(),
            spatial_confidence=0.7,
            max_new_tokens=5,
            attention_mask=attn_mask,
            position_ids=pos_ids,
            step_trace=tr_b,
        )
        ts, as_, es = pipe_kvlist._run_sequential_dual_branch_decode_reference(
            pos_in.clone(),
            neg_in.clone(),
            spatial_confidence=0.7,
            max_new_tokens=5,
            attention_mask=attn_mask,
            position_ids=pos_ids,
            step_trace=tr_s,
        )
        self._assert_step_traces_match(tr_b, tr_s, tol=1e-5)
        self.assertEqual(tb, ts)

        # 2. Direct CausalLM path (`has_backbone == False`, where `language_model` has no `.model` or `.lm_head`)
        class _DirectCausalLMWrapper(nn.Module):
            def __init__(self, base_lm: _SyntheticLanguageModel):
                super().__init__()
                self._base_lm = base_lm
                self.config = base_lm.config

            def get_input_embeddings(self):
                return self._base_lm.get_input_embeddings()

            def forward(self, **kwargs):
                return self._base_lm(**kwargs)

        pipe_direct = self._make_pipeline(dtype=torch.float32, cache_mode="tuple", seed=61)
        pipe_direct.model.language_model = _DirectCausalLMWrapper(pipe_direct.model.language_model)
        self.assertFalse(hasattr(pipe_direct.model.language_model, "model"))
        self.assertFalse(hasattr(pipe_direct.model.language_model, "lm_head"))

        tr_bd, tr_sd = [], []
        tbd, _, _ = pipe_direct._run_batched_dual_branch_decode(
            pos_in.clone(),
            neg_in.clone(),
            spatial_confidence=0.7,
            max_new_tokens=5,
            attention_mask=attn_mask[0:1],
            position_ids=pos_ids[0:1],
            step_trace=tr_bd,
        )
        tsd, _, _ = pipe_direct._run_sequential_dual_branch_decode_reference(
            pos_in.clone(),
            neg_in.clone(),
            spatial_confidence=0.7,
            max_new_tokens=5,
            attention_mask=attn_mask[0:1],
            position_ids=pos_ids[0:1],
            step_trace=tr_sd,
        )
        self._assert_step_traces_match(tr_bd, tr_sd, tol=1e-5)
        self.assertEqual(tbd, tsd)

    # ------------------------------------------------------------------
    # Test F — EOS and stopping behavior
    # ------------------------------------------------------------------
    def test_F_eos_and_stopping_behavior(self):
        """
        Test F — EOS and stopping behavior:
        Test (1) immediate step-0 EOS, (2) mid-sequence EOS at step 3, and (3) max_new_tokens limit.
        Batched and sequential implementations must stop at the exact same step and emit identical sequences.
        """
        pos_in = torch.randn((1, 12, 32), dtype=torch.float32)
        neg_in = torch.randn((1, 12, 32), dtype=torch.float32)

        # Case 1: Immediate EOS at step 0 (`<|im_end|>` token id 3)
        pipe_eos0 = self._make_pipeline(dtype=torch.float32, seed=91)
        with torch.no_grad():
            pipe_eos0.model.language_model.lm_head.weight.zero_()
            pipe_eos0.model.language_model.lm_head.bias.zero_()
            pipe_eos0.model.language_model.lm_head.bias[3] = 25.0
        llm0 = pipe_eos0.model.language_model.model
        llm0.reset_call_log()
        tr_b0, tr_s0 = [], []
        tb0, _, _ = pipe_eos0._run_batched_dual_branch_decode(
            pos_in.clone(), neg_in.clone(), spatial_confidence=0.8, max_new_tokens=8, step_trace=tr_b0
        )
        # Only 1 prefill call and 0 decode step calls when EOS is emitted at step 0
        self.assertEqual(len(llm0.call_log), 1)
        ts0, _, _ = pipe_eos0._run_sequential_dual_branch_decode_reference(
            pos_in.clone(), neg_in.clone(), spatial_confidence=0.8, max_new_tokens=8, step_trace=tr_s0
        )
        self.assertEqual(len(tr_b0), 1)
        self.assertEqual(len(tr_s0), 1)
        self.assertEqual(tr_b0[0]["next_token_id"], 3)
        self.assertEqual(tb0, "")
        self.assertEqual(ts0, "")

        # Case 2: Mid-sequence EOS at step 3 (emit tokens [15, 22, 31] then EOS=2 at step 3)
        pipe_mid = self._make_pipeline(dtype=torch.float32, seed=92)
        planned_ids = [15, 22, 31, 2]  # token 2 is tokenizer.eos_token_id

        orig_decode_step = pipe_mid.decoder.decode_step

        def _scripted_decode_step(pos_logits, neg_logits, step, **kwargs):
            _, alpha_t, ent_t, d_js = orig_decode_step(
                pos_logits=pos_logits, neg_logits=neg_logits, step=step, **kwargs
            )
            return planned_ids[min(step, len(planned_ids) - 1)], alpha_t, ent_t, d_js

        with patch.object(pipe_mid.decoder, "decode_step", side_effect=_scripted_decode_step):
            tr_bm, tr_sm = [], []
            tbm, _, _ = pipe_mid._run_batched_dual_branch_decode(
                pos_in.clone(), neg_in.clone(), spatial_confidence=0.8, max_new_tokens=10, step_trace=tr_bm
            )
            tsm, _, _ = pipe_mid._run_sequential_dual_branch_decode_reference(
                pos_in.clone(), neg_in.clone(), spatial_confidence=0.8, max_new_tokens=10, step_trace=tr_sm
            )
            self.assertEqual(len(tr_bm), 4)
            self.assertEqual(len(tr_sm), 4)
            self._assert_step_traces_match(tr_bm, tr_sm, tol=1e-5)
            self.assertEqual(tbm, "tok15 tok22 tok31")
            self.assertEqual(tsm, "tok15 tok22 tok31")

        # Case 3: Maximum generated length cap (`max_new_tokens=5`, no EOS emitted)
        pipe_max = self._make_pipeline(dtype=torch.float32, seed=93)
        tr_bx, tr_sx = [], []
        tbx, _, _ = pipe_max._run_batched_dual_branch_decode(
            pos_in.clone(), neg_in.clone(), spatial_confidence=0.8, max_new_tokens=5, step_trace=tr_bx
        )
        tsx, _, _ = pipe_max._run_sequential_dual_branch_decode_reference(
            pos_in.clone(), neg_in.clone(), spatial_confidence=0.8, max_new_tokens=5, step_trace=tr_sx
        )
        self.assertEqual(len(tr_bx), 5)
        self.assertEqual(len(tr_sx), 5)
        self._assert_step_traces_match(tr_bx, tr_sx, tol=1e-5)
        self.assertEqual(tbx, tsx)
        self.assertEqual(len(tbx.split()), 5)

    # ------------------------------------------------------------------
    # Test G — Error and shape handling
    # ------------------------------------------------------------------
    def test_G_error_and_shape_handling(self):
        """
        Test G — Error and shape handling:
        Verify that unsupported branch shape combinations, mismatched dtypes, asymmetric masks,
        and invalid cache states fail clearly with ValueError instead of silently altering semantics.
        """
        pipe = self._make_pipeline(dtype=torch.float32, seed=44)

        # 1. Mismatched sequence lengths between pos_inputs and neg_inputs
        with self.assertRaises(ValueError):
            pipe._run_batched_dual_branch_decode(
                torch.randn(1, 12, 32), torch.randn(1, 14, 32), spatial_confidence=0.5
            )

        # 2. Mismatched hidden dimensions
        with self.assertRaises(ValueError):
            pipe._run_batched_dual_branch_decode(
                torch.randn(1, 12, 32), torch.randn(1, 12, 64), spatial_confidence=0.5
            )

        # 3. Invalid batch size on branch inputs (B != 1) or wrong rank (2D) or empty sequence
        with self.assertRaises(ValueError):
            pipe._run_batched_dual_branch_decode(
                torch.randn(2, 12, 32), torch.randn(2, 12, 32), spatial_confidence=0.5
            )
        with self.assertRaises(ValueError):
            pipe._run_batched_dual_branch_decode(
                torch.randn(12, 32), torch.randn(12, 32), spatial_confidence=0.5
            )
        with self.assertRaises(ValueError):
            pipe._run_batched_dual_branch_decode(
                torch.randn(1, 0, 32), torch.randn(1, 0, 32), spatial_confidence=0.5
            )

        # 4. Mismatched dtypes
        with self.assertRaises(ValueError):
            pipe._run_batched_dual_branch_decode(
                torch.randn(1, 12, 32, dtype=torch.float32),
                torch.randn(1, 12, 32, dtype=torch.bfloat16),
                spatial_confidence=0.5,
            )

        # 5. Asymmetric attention_mask or divergent position_ids between branches
        pos_ok = torch.randn(1, 12, 32)
        neg_ok = torch.randn(1, 12, 32)
        bad_mask = torch.ones((2, 12), dtype=torch.long)
        bad_mask[1, 0] = 0
        with self.assertRaises(ValueError):
            pipe._run_batched_dual_branch_decode(
                pos_ok, neg_ok, spatial_confidence=0.5, attention_mask=bad_mask
            )

        bad_pos = torch.arange(0, 12, dtype=torch.long).unsqueeze(0).repeat(2, 1)
        bad_pos[1, -1] += 1
        with self.assertRaises(ValueError):
            pipe._run_batched_dual_branch_decode(
                pos_ok, neg_ok, spatial_confidence=0.5, position_ids=bad_pos
            )

        # 6. Invalid KV cache states
        with self.assertRaises(ValueError):
            pipe._validate_kv_cache(None, expected_batch_size=2)
        with self.assertRaises(ValueError):
            pipe._validate_kv_cache((), expected_batch_size=2)
        # Wrong batch dimension in cache (B=1 instead of B=2)
        bad_kv_batch = ((torch.ones(1, 4, 12, 8), torch.ones(1, 4, 12, 8)),)
        with self.assertRaises(ValueError):
            pipe._validate_kv_cache(bad_kv_batch, expected_batch_size=2)
        # Mismatched K and V shapes
        bad_kv_shape = ((torch.ones(2, 4, 12, 8), torch.ones(2, 4, 10, 8)),)
        with self.assertRaises(ValueError):
            pipe._validate_kv_cache(bad_kv_shape, expected_batch_size=2)
        # Non-advancing sequence length during decoding step
        with patch.object(
            pipe.model.language_model.model,
            "forward",
            return_value=_SyntheticTransformerOutput(
                last_hidden_state=torch.ones(2, 12, 32),
                past_key_values=((torch.ones(2, 4, 10, 8), torch.ones(2, 4, 10, 8)),),
            ),
        ):
            with self.assertRaises(ValueError):
                pipe._run_batched_dual_branch_decode(pos_ok, neg_ok, spatial_confidence=0.5, max_new_tokens=2)

    # ------------------------------------------------------------------
    # Test H — Existing pipeline regression
    # ------------------------------------------------------------------
    def test_H_existing_pipeline_regression_and_opt02_03_06_preservation(self):
        """
        Test H — Existing pipeline regression:
        Confirm the B=2 change does not modify Stage 1 outputs, ROI coordinates, bounding boxes,
        or surrounding Stage 2 formula, and preserves OPT-02 (no per-question gc.collect / empty_cache),
        OPT-03 (working-map cache reuse), and OPT-06 (image cache identity).
        """
        pipe = self._make_pipeline(dtype=torch.bfloat16, seed=144)
        img = _make_synthetic_signboard_image()

        # Run full pipeline with B=2 batched decoding
        res_bat = pipe.run(
            img,
            "Tên nhà thuốc trên biển hiệu là gì?",
            max_new_tokens=6,
            image_id=22,
            image_path="data/vitextvqa/st_images/22.jpg",
            dataset_version="vitextvqa:dev",
        )

        # Run full pipeline with sequential reference decoding on fresh cache state
        pipe.reset_cache()
        with patch.object(
            pipe,
            "_decode_spatial_pmi_contrastive",
            side_effect=pipe._decode_spatial_pmi_contrastive_sequential_reference,
        ):
            res_seq = pipe.run(
                img,
                "Tên nhà thuốc trên biển hiệu là gì?",
                max_new_tokens=6,
                image_id=22,
                image_path="data/vitextvqa/st_images/22.jpg",
                dataset_version="vitextvqa:dev",
            )

        # Stage 1 & Stage 2 exact equality
        self.assertEqual(res_bat.stage1_bbox, res_seq.stage1_bbox)
        self.assertEqual(res_bat.stage1_bboxes, res_seq.stage1_bboxes)
        self.assertEqual(res_bat.selection.pixel_bbox, res_seq.selection.pixel_bbox)
        self.assertEqual(res_bat.selection.grid_bbox, res_seq.selection.grid_bbox)
        self.assertAlmostEqual(res_bat.selection.kl_score, res_seq.selection.kl_score, places=6)
        self.assertEqual(res_bat.crop.isotropic_box, res_seq.crop.isotropic_box)
        self.assertAlmostEqual(res_bat.crop.magnification, res_seq.crop.magnification, places=6)
        self.assertEqual(res_bat.pred_text, res_seq.pred_text)
        self.assertEqual(res_bat.pred_raw_text, res_seq.pred_raw_text)
        self.assertEqual(res_bat.pred_sg_vcd, res_seq.pred_sg_vcd)
        self.assertAlmostEqual(res_bat.vcd_alpha_mean, res_seq.vcd_alpha_mean, places=6)
        self.assertAlmostEqual(res_bat.entropy_mean, res_seq.entropy_mean, places=6)

        # AST check: neither `_decode_spatial_pmi_contrastive` nor `_run_batched_dual_branch_decode`
        # introduces `gc.collect()` or `torch.cuda.empty_cache()` (OPT-02 preservation)
        for method in (
            SGVCDPipeline._decode_spatial_pmi_contrastive,
            SGVCDPipeline._run_batched_dual_branch_decode,
            SGVCDPipeline.run,
        ):
            src = inspect.getsource(method)
            tree = ast.parse(src.lstrip())
            forbidden = [
                node.func.attr
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("collect", "empty_cache")
            ]
            self.assertEqual(forbidden, [], f"Forbidden cleanup call in {method.__name__}: {forbidden}")


if __name__ == "__main__":
    unittest.main()
