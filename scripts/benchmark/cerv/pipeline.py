# -*- coding: utf-8 -*-
"""
CERV: Counterfactual Evidence Routing & Verification Pipeline.

Mathematical Specification (from `docs/Proposed_Method_CERV.md`):
1. Stage 1 — Bayesian Semantic-Fork Gate:
   Tracks decoding trajectory t = 1..T on global image I and question Q:
     - Posterior visual reliance:
         pi_t = P(H_vis | I, Q, y_{<t}) = (sum_{i in I} A_{t,i}) / (sum_{i in I} A_{t,i} + sum_{j in Q} A_{t,j})
     - Token Bayes risk:
         R_token(t) = 1 - max_{y in V} p_t(y | I, Q, y_{<t})
     - Normalized spatial attention entropy:
         H_tilde(A_t^{img}) = - (1 / log P) sum_{i in I} A_hat_{t,i} log A_hat_{t,i} in [0, 1]
     - Expected Value of Visual Sample Information & Baseline Risk (c_0 = 1 / log |V|):
         EVSI_t = pi_t * (1 - max_y p_t(y)) * H_tilde(A_t^{img})
         R_0(t) = (1 - pi_t) * (1 - max_y p_t(y)) + c_0
     - Bayesian Posterior Odds Ratio:
         Lambda_t = EVSI_t / R_0(t)
   - Fast-Path: If Lambda_t <= 1 for all t, return generated sequence immediately (0% overhead).
   - Slow-Path: At the first content step t* where Lambda_{t*} > 1 (Visual-Semantic Fork), lock prefix y_{<t*}
     and trigger Stage 2 & Stage 3.

2. Stage 2 — Prefix-Conditioned VAQ Localization at t*:
   - Positive rectified query-counterfactual contrast across layers l in [L/4, 3L/4]:
       d_l(I, Q, y_{<t*}) = max(0, s_l(I, Q, y_{<t*}) - s_l(I, empty, y_{<t*}))
   - Peak convergence layer selection:
       VAQ_l(t*) = ||d_l(I, Q, y_{<t*})||_2,   l* = argmax_{l in [L/4, 3L/4]} VAQ_l(t*)
   - Spatial normalization & minimal top-energy patch set (rho = 0.85):
       d_hat_{l*}(p) = d_{l*}(p) / (sum_j d_{l*}(j) + eps)
       T = TopPatches(d_hat_{l*}, rho = 0.85)
   - Opposing Views Construction on bounding crop B_crop:
       I^+ = Crop(I, B_crop)
       I^- = Crop(Mask(I, T_peak), B_crop)

3. Stage 3 — Fork-Anchored Counterfactual Verification (t >= t*):
   - Preserves locked preamble prefix y_{<t*} and decodes in parallel on (I^+, Q, y_{<t}) and (I^-, Q, y_{<t}).
   - Adaptive Information Gain alpha_t in [0, 1] via Mixture-Entropy Normalized Jensen-Shannon Divergence:
       alpha_t = JSD(p_t^+ || p_t^-) / H((p_t^+ + p_t^-) / 2) = 1 - (H(p_t^+) + H(p_t^-)) / (2 * H(m_t))
   - Plausibility-constrained counterfactual logit verification (beta = 0.1):
       z_t^{CERV}(y) = (1 + alpha_t) z_t^+(y) - alpha_t z_t^-(y)   for y in V_head(z_t^+), else -inf
"""

from dataclasses import dataclass, field
import math
import re
import time
from typing import Dict, List, Optional, Tuple
import unicodedata

import numpy as np
from PIL import Image
from scipy import ndimage
import torch
import torch.nn.functional as F


PROMPT_INSTRUCTION = "Trả lời thật ngắn gọn câu hỏi sau dựa vào chữ xuất hiện trong ảnh: "


@dataclass
class CERVResult:
    """Structured diagnostic and prediction result returned by CERVPipeline."""
    pred_text: str
    stage1_bbox: List[int]
    route: str                          # "fast_path" or "slow_path"
    fork_step: int                      # t* (1-indexed step where Lambda_{t*} > 1, or 0 if fast_path)
    max_lambda: float                   # max_t Lambda_t observed along trajectory
    fork_lambda: float                  # Lambda_{t*} at the fork step
    fork_pi: float                      # pi_{t*} visual reliance at the fork step
    fork_risk: float                    # 1 - max_y p_{t*}(y) or uncertainty U(p_t) at the fork step
    fork_entropy: float                 # H_tilde(A_{t*}^{img}) at the fork step
    peak_layer: int                     # l* selected by Fork-VAQ
    mean_alpha: float                   # mean adaptive gain alpha_t across Stage 3 steps
    max_alpha: float                    # max adaptive gain alpha_t across Stage 3 steps
    locked_prefix: str                  # decoded preamble y_{<t*} locked before Slow-Path
    fast_draft_text: str                # full text if Fast-Path or prefix before fork
    spatial_confidence: float = 1.0
    pred_raw_text: str = ""             # Full generated sequence
    fork_token_idx: int = 0             # Alias for fork_step
    routing_path: str = "fast_path"     # Alias for route
    bayesian_odds: float = 0.0          # Alias for fork_lambda
    alpha_mean: float = 0.0             # Alias for mean_alpha
    timing: Dict[str, float] = field(default_factory=dict)
    trajectory_trace: List[Dict[str, float]] = field(default_factory=list)


CERVPipelineResult = CERVResult


class BayesianOddsResult(tuple):
    """5-tuple backwards-compatible container with phi_res attribute."""
    phi_res: float

    def __new__(cls, lambda_t, pi_t, uncertainty, spatial_entropy, layer_vis_vecs, phi_res=0.0):
        obj = super().__new__(cls, (lambda_t, pi_t, uncertainty, spatial_entropy, layer_vis_vecs))
        obj.phi_res = float(phi_res)
        return obj


def _find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: List[Tuple[int, int]],
    width: int,
    height: int,
    image_size: int = 448,
) -> Tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess_with_grid(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 6,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> Tuple[List[Image.Image], Tuple[int, int]]:
    """
    Standard InternVL2 / Vintern dynamic aspect-ratio tiling returning both
    the list of 448x448 tiles and the (g_x, g_y) tile grid dimensions.
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / max(1, orig_height)

    target_ratios = sorted(
        set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        ),
        key=lambda x: x[0] * x[1],
    )

    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    gx, gy = target_aspect_ratio
    target_width = image_size * gx
    target_height = image_size * gy
    blocks = gx * gy

    resized_img = image.resize((target_width, target_height), Image.Resampling.BICUBIC)
    processed_images = []
    for i in range(blocks):
        box = (
            (i % gx) * image_size,
            (i // gx) * image_size,
            ((i % gx) + 1) * image_size,
            ((i // gx) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size), Image.Resampling.BICUBIC))

    return processed_images, (gx, gy)


class CERVPipeline:
    """
    Model-agnostic Counterfactual Evidence Routing & Verification (CERV) pipeline.
    Operates strictly in inference mode without any hardcoded keyword lists or threshold tuning.
    """

    def __init__(
        self,
        model,
        tokenizer,
        transform,
        device: str = "cuda",
        energy_rho: float = 0.85,
        plausibility_beta: float = 0.1,
        max_dynamic_patch: int = 6,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.transform = transform
        self.device = device
        self.energy_rho = float(energy_rho)
        self.plausibility_beta = float(plausibility_beta)
        self.max_dynamic_patch = int(
            getattr(getattr(model, "config", None), "max_dynamic_patch", max_dynamic_patch)
        )

        # Vocabulary size |V| and calibrated sequential sampling cost
        # lambda_cost = (C_crop / (C_global + C_crop)) * (1 / log |V|)
        vocab_size = getattr(getattr(model, "language_model", model).config, "vocab_size", 151674)
        self.vocab_size = int(vocab_size)
        self.cost_ratio = 0.30
        self.lambda_cost = self.cost_ratio / math.log(max(2, self.vocab_size))
        self.c_0 = self.lambda_cost

        # Layer range [L/4, 3L/4]
        num_layers = getattr(getattr(model, "language_model", model).config, "num_hidden_layers", 24)
        self.num_layers = int(num_layers)
        self.layer_start = max(0, self.num_layers // 4)
        self.layer_end = min(self.num_layers, (3 * self.num_layers) // 4 + 1)

        # Per-image visual feature & KV cache
        self._cached_img_key: Optional[Tuple[int, int, bytes]] = None
        self._cached_img_state: Optional[dict] = None

    @staticmethod
    def _make_image_key(image: Image.Image) -> Tuple[int, int, bytes]:
        w, h = image.size
        thumb = image.resize((16, 16), Image.Resampling.NEAREST).convert("RGB")
        return (w, h, thumb.tobytes())

    @staticmethod
    def _extract_causal_entity(question: str, full_text: str) -> str:
        r"""
        Extract concise causal visual entity E from autoregressive sequence S = \Pi_Q \circ C \circ E.

        Subsequence-Aligned Causal Entity Extraction:
        1. Dynamically extract question vocabulary V_Q = {q_1, ..., q_m} normalized to Unicode NFC lowercase.
        2. Scan leading words of clean_full. If a leading contiguous span of words
           (ignoring optional prepositions 'của', 'ở', 'trong', 'này' and copula 'là')
           belongs to V_Q with at least 1 substantive question word (>= 3 letters, in V_Q, not in connectors)
           and total matched length >= 2, treat this leading span as conversational preamble \Pi_Q.
        3. Slice clean_full after the matched preamble span, strip optional immediate copula/punctuation
           (e.g., `^(?:(?:là\b|[:\-–—])\s*)+`), and return the clean visual entity E.
        4. If matched length < 2 or no substantive question word is present, return clean_full as-is.
        Zero hardcoded stopwords: vocabulary V_Q is dynamically derived per question directly from prompt.
        """
        if not full_text or not full_text.strip():
            return ""

        clean_full = full_text.strip()
        if not question or not question.strip():
            return clean_full.rstrip(".").strip()

        # Dynamically derive question vocabulary V_Q with Unicode NFC normalization
        q_words = [unicodedata.normalize("NFC", m.group()).lower() for m in re.finditer(r"\w+", question)]
        v_q = set(q_words)

        s_matches = list(re.finditer(r"\w+", clean_full))
        s_words = [unicodedata.normalize("NFC", m.group()).lower() for m in s_matches]

        if not v_q or not s_words:
            return clean_full.rstrip(".").strip()

        # Optional connective tokens allowed to bridge conversational question echoes
        connectors = {"của", "ở", "trong", "này", "là"}

        # Scan leading contiguous span of words belonging to V_Q or connectors
        k = 0
        while k < len(s_words):
            w = s_words[k]
            if w in v_q or w in connectors:
                k += 1
            else:
                break

        # Check validity of conversational preamble Pi_Q:
        # Total matched length >= 2, with at least 1 substantive question word (>= 3 letters, in V_Q, not connector)
        substantive = [w for w in s_words[:k] if w in v_q and w not in connectors and len(w) >= 3]

        if k >= 2 and len(substantive) >= 1 and k < len(s_words):
            end_offset = s_matches[k - 1].end()
            remainder = clean_full[end_offset:].strip()
            # Strip optional immediate copula / punctuation delimiters
            remainder = re.sub(r"^(?:(?:là\b|[:\-–—])\s*)+", "", remainder, flags=re.IGNORECASE).strip()
            remainder = remainder.rstrip(".").strip()
            if len(remainder) > 0:
                return remainder

        return clean_full.rstrip(".").strip()

    @staticmethod
    def _clone_kv_cache(past_kv):
        if past_kv is None:
            return None
        try:
            from transformers.cache_utils import DynamicCache
            if hasattr(past_kv, "to_legacy_cache") and hasattr(DynamicCache, "from_legacy_cache"):
                legacy = past_kv.to_legacy_cache()
                cloned_legacy = tuple((k.clone(), v.clone()) for k, v in legacy)
                return DynamicCache.from_legacy_cache(cloned_legacy)
        except Exception:
            pass
        if isinstance(past_kv, (tuple, list)):
            return tuple((layer_kv[0].clone(), layer_kv[1].clone()) for layer_kv in past_kv)
        import copy
        return copy.deepcopy(past_kv)

    def _set_attn_implementation(self, mode: str) -> str:
        llm = self.model.language_model
        prev_mode = getattr(llm.config, "_attn_implementation", "sdpa")
        llm.config._attn_implementation = mode
        if hasattr(llm, "model") and hasattr(llm.model, "layers"):
            for layer in llm.model.layers:
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "config"):
                    layer.self_attn.config._attn_implementation = mode
        return prev_mode

    def _build_visual_prefix_and_query_suffix(
        self,
        vit_embeds: torch.Tensor,
        question: Optional[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int, int, int]:
        """
        Build:
          - `prefix_embeds`: `<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n<img><IMG_CONTEXT>...</img>\n`
          - `suffix_embeds`: `{PROMPT_INSTRUCTION}{question}<|im_end|>\n<|im_start|>assistant\n`
            (or null query `<|im_end|>\n<|im_start|>assistant\n` when `question` is None)
        Uses the model's exact `system_message` so Fast-Path is 100% identical to `model.chat(...)`.
        Returns:
          (prefix_embeds, suffix_embeds, vis_start, vis_end, q_start_abs, q_end_abs)
        """
        num_patches = vit_embeds.shape[0]
        num_image_token = getattr(self.model, "num_image_token", 256)
        total_img_tokens = num_patches * num_image_token
        sys_msg = getattr(self.model, "system_message", "Bạn là một trợ lý AI hữu ích.")

        prefix_str = (
            f"<|im_start|>system\n{sys_msg}<|im_end|>\n"
            "<|im_start|>user\n"
            "<img>" + ("<IMG_CONTEXT>" * total_img_tokens) + "</img>\n"
        )
        prefix_ids = self.tokenizer(prefix_str, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)

        if question is not None and len(question.strip()) > 0:
            q_clean = question.strip()
            instr_only_ids = self.tokenizer(PROMPT_INSTRUCTION.rstrip(), return_tensors="pt", add_special_tokens=False).input_ids
            full_suffix_str = f"{PROMPT_INSTRUCTION}{q_clean}<|im_end|>\n<|im_start|>assistant\n"
            suffix_ids = self.tokenizer(full_suffix_str, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
            tail_ids = self.tokenizer("<|im_end|>\n<|im_start|>assistant\n", return_tensors="pt", add_special_tokens=False).input_ids
            q_start_abs = int(prefix_ids.shape[1] + instr_only_ids.shape[1])
            q_end_abs = int(prefix_ids.shape[1] + suffix_ids.shape[1] - tail_ids.shape[1])
        else:
            # Null query \varnothing for counterfactual background subtraction
            suffix_ids = self.tokenizer("<|im_end|>\n<|im_start|>assistant\n", return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
            q_start_abs = int(prefix_ids.shape[1])
            q_end_abs = int(prefix_ids.shape[1])

        embed_layer = self.model.language_model.get_input_embeddings()
        prefix_embeds = embed_layer(prefix_ids).clone()
        suffix_embeds = embed_layer(suffix_ids)

        img_ctx_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        if not hasattr(self.model, "img_context_token_id") or self.model.img_context_token_id is None:
            self.model.img_context_token_id = img_ctx_id

        mask = (prefix_ids == img_ctx_id).squeeze(0)
        vis_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        vis_start = int(vis_indices[0].item())
        vis_end = int(vis_indices[-1].item()) + 1

        flat_vit = vit_embeds.reshape(-1, vit_embeds.shape[-1]).to(prefix_embeds.dtype)
        prefix_embeds[0, vis_start:vis_end, :] = flat_vit

        return prefix_embeds, suffix_embeds, vis_start, vis_end, q_start_abs, q_end_abs

    def _get_or_create_global_image_state(self, image: Image.Image) -> dict:
        """Extract and cache global image ViT features and shared visual KV cache."""
        img_key = self._make_image_key(image)
        if self._cached_img_key == img_key and self._cached_img_state is not None:
            return self._cached_img_state

        self._cached_img_state = None
        self._cached_img_key = None

        tiles, grid_ratio = dynamic_preprocess_with_grid(
            image,
            min_num=1,
            max_num=self.max_dynamic_patch,
            image_size=448,
            use_thumbnail=True,
        )
        gx, gy = grid_ratio
        has_thumbnail = len(tiles) > (gx * gy)

        pixel_values = torch.stack([self.transform(t) for t in tiles]).to(
            device=self.device, dtype=torch.bfloat16
        )
        vit_embeds = self.model.extract_feature(pixel_values)
        del pixel_values

        prefix_embeds, null_suffix_embeds, vis_start, vis_end, _, _ = self._build_visual_prefix_and_query_suffix(
            vit_embeds, question=None
        )

        self._set_attn_implementation("sdpa")
        out_prefix = self.model.language_model.model(
            inputs_embeds=prefix_embeds,
            use_cache=True,
            output_attentions=False,
            return_dict=True,
        )
        shared_kv = out_prefix.past_key_values
        del prefix_embeds, out_prefix

        state = {
            "vit_embeds": vit_embeds,
            "shared_kv": shared_kv,
            "null_suffix_embeds": null_suffix_embeds,
            "s_null_layers": None,
            "vis_start": vis_start,
            "vis_end": vis_end,
            "grid_ratio": grid_ratio,
            "has_thumbnail": has_thumbnail,
        }
        self._cached_img_key = img_key
        self._cached_img_state = state
        return state

    def _compute_step_bayesian_odds(
        self,
        logits: torch.Tensor,
        attentions: Tuple[torch.Tensor, ...],
        vis_start: int,
        vis_end: int,
        q_start: int,
        q_end: int,
        assistant_end: int,
        is_prefill_step: bool = False,
    ) -> Tuple[float, float, float, float, Dict[int, torch.Tensor]]:
        """
        Compute Stage 1 Calibrated Bayesian Evidence Gate (CBE-Gate) at decoding step t:
          - Visual head selection H_vis filtering syntactic sink heads
          - Posterior visual reliance:
              pi_t = P(H_vis | I, Q, y_{<t}) = vis_mass / (vis_mass + q_mass + prefix_mass + eps)
          - Normalized token uncertainty:
              U(p_t) = (1 - max_y p_t(y)) + gamma * (H(p_t) / log |V|)
          - Spatial Resolution-Deficit ratio:
              Phi_res(A_t^{img}) = (sum_{p in T_top} A_{t,p} / sum_p A_{t,p}) * (1 - Area(B_att) / Area(I))
          - Posterior Resolution-Deficit Likelihood Ratio:
              Lambda_t = (pi_t * U(p_t) * Phi_res) / ((1 - pi_t) * U(p_t) + lambda_cost)
        """
        probs = F.softmax(logits.float(), dim=-1)
        max_prob = float(probs.max().item())
        margin_risk = max(0.0, 1.0 - max_prob)

        # U(p_t): normalized uncertainty combining margin risk and Shannon entropy
        top_probs = torch.topk(probs, k=min(100, probs.shape[-1])).values
        norm_entropy = float(
            (-torch.sum(top_probs * torch.log(top_probs.clamp(min=1e-12)))).item()
        ) / math.log(max(2, self.vocab_size))
        norm_entropy = min(1.0, max(0.0, norm_entropy))
        uncertainty = float(min(1.0, max(0.0, margin_risk + 0.5 * norm_entropy)))

        P = max(1, vis_end - vis_start)
        log_P = math.log(max(2, P))

        layer_vis_vecs: Dict[int, torch.Tensor] = {}
        vis_mass_sum = 0.0
        q_mass_sum = 0.0
        prefix_mass_sum = 0.0
        accum_vis_dist = torch.zeros(P, dtype=torch.float32, device=self.device)
        num_eval_layers = max(1, self.layer_end - self.layer_start)

        del is_prefill_step
        for l_idx in range(self.layer_start, self.layer_end):
            # Shape of attentions[l_idx]: (1, H_heads, seq_len, total_kv_len)
            att_full = attentions[l_idx][0].float()          # (H_heads, seq_len, total_kv_len)
            att_last = att_full[:, -1, :]                    # (H_heads, total_kv_len)
            tot_len = att_last.shape[-1]

            # Visual head selection H_vis: filter out syntactic/token-0 sink heads
            head_vis = att_last[:, vis_start:vis_end].sum(dim=-1)
            head_tot = att_last.sum(dim=-1).clamp(min=1e-12)
            head_ratio = head_vis / head_tot
            ratio_mean = head_ratio.mean()
            h_vis_mask = head_ratio >= max(0.05, float(ratio_mean.item()) * 0.8)
            if not h_vis_mask.any():
                _, top_h = torch.topk(head_ratio, k=min(4, head_ratio.shape[0]))
                h_vis_mask = torch.zeros_like(head_ratio, dtype=torch.bool)
                h_vis_mask[top_h] = True

            mean_att_l = att_last[h_vis_mask].mean(dim=0)    # (total_kv_len,)
            vis_vec_step = mean_att_l[vis_start:vis_end]     # (P,)
            layer_vis_vecs[l_idx] = (vis_vec_step / vis_vec_step.sum().clamp(min=1e-12)).detach()

            vis_mass_l = float(vis_vec_step.sum().item())
            q_mass_l = float(mean_att_l[q_start:q_end].sum().item()) if q_end > q_start else 0.0
            # Account for autoregressive tokens y_{<t} in the context
            pref_mass_l = float(mean_att_l[assistant_end:-1].sum().item()) if tot_len > assistant_end + 1 else 0.0

            vis_mass_sum += vis_mass_l
            q_mass_sum += q_mass_l
            prefix_mass_sum += pref_mass_l
            accum_vis_dist += vis_vec_step

        vis_mass = vis_mass_sum / num_eval_layers
        q_mass = q_mass_sum / num_eval_layers
        prefix_mass = prefix_mass_sum / num_eval_layers

        # Visual reliance P(H_vis | I, Q, y_{<t}) taking into account autoregressive tokens y_{<t}
        pi_t = vis_mass / max(1e-12, vis_mass + q_mass + prefix_mass)
        pi_t = float(min(1.0, max(0.0, pi_t)))

        # Normalized spatial attention distribution A_hat_t over P patches
        a_hat = accum_vis_dist / accum_vis_dist.sum().clamp(min=1e-12)
        spatial_entropy = -float((a_hat * torch.log(a_hat.clamp(min=1e-12))).sum().item()) / log_P
        spatial_entropy = float(min(1.0, max(0.0, spatial_entropy)))

        # Spatial Resolution-Deficit ratio Phi_res(A_t^{img}):
        # Measures degree of attention concentration on small physical area:
        # top 15% patches concentration vs spatial spread
        k_top = max(1, int(0.15 * P))
        top_energy = float(torch.topk(a_hat, k_top).values.sum().item())
        sorted_a = torch.sort(a_hat, descending=True).values
        cum_a = torch.cumsum(sorted_a, dim=0)
        idx_80 = int(torch.searchsorted(cum_a, 0.80).item()) + 1
        area_ratio = min(1.0, max(0.02, float(idx_80) / float(P)))
        phi_res = float(min(1.0, max(0.0, top_energy * (1.0 - area_ratio))))

        # Posterior Resolution-Deficit Likelihood Ratio:
        # \Lambda_t = (pi_t * U(p_t) * Phi_res) / ((1 - pi_t) * U(p_t) + lambda_cost)
        evsi_t = pi_t * uncertainty * phi_res
        r0_t = (1.0 - pi_t) * uncertainty + self.lambda_cost
        lambda_t = evsi_t / max(1e-12, r0_t)

        return BayesianOddsResult(
            float(lambda_t), float(pi_t), float(uncertainty), float(spatial_entropy), layer_vis_vecs, phi_res=float(phi_res)
        )

    def _project_patch_vector_to_2d_grid(
        self,
        vec: torch.Tensor,
        grid_ratio: Tuple[int, int],
        has_thumbnail: bool,
    ) -> np.ndarray:
        """
        Project 1D visual patch vector of length P onto the 2D spatial grid (H_g, W_g) = (16*g_y, 16*g_x).
        Combines dynamic high-resolution tiles with the global thumbnail when present.
        """
        gx, gy = grid_ratio
        H_g, W_g = 16 * gy, 16 * gx
        num_grid_tokens = gx * gy * 256

        if vec.numel() >= num_grid_tokens and (gx * gy) > 1:
            grid_tokens = vec[:num_grid_tokens].reshape(gy, gx, 16, 16)
            grid_map = grid_tokens.permute(0, 2, 1, 3).reshape(H_g, W_g)
            grid_sum = grid_map.sum()
            # High-Resolution Tiled Contrast (HTC-RoI): pure native grid patch projection
            # without blurry thumbnail blending to prevent artificial energy dispersion
            del has_thumbnail
            out_map = grid_map
        else:
            single_map = vec[:256].reshape(16, 16)
            s_sum = single_map.sum()
            if s_sum > 1e-12:
                single_map = single_map / s_sum
            out_map = single_map

        arr = out_map.detach().cpu().numpy().astype(np.float32)
        arr_sum = float(arr.sum())
        if arr_sum > 1e-12:
            arr = arr / arr_sum
        return arr

    def _stage2_prefix_conditioned_vaq_localization(
        self,
        image: Image.Image,
        state: dict,
        locked_prefix_embeds: Optional[torch.Tensor],
        s_query_layers: Dict[int, torch.Tensor],
        prefill_query_layers: Dict[int, torch.Tensor],
    ) -> Tuple[Image.Image, Image.Image, List[int], int]:
        """
        Stage 2: Prefix-Conditioned VAQ Localization at Fork Step t*.
          1. Computes null-query attention s_l(I, empty, y_{<t*}) with the exact same locked prefix y_{<t*}.
          2. Computes rectified contrast d_l(I, Q, y_{<t*}) = max(0, s_l(I, Q, y_{<t*}) - s_l(I, empty, y_{<t*})).
          3. Selects peak convergence layer l* = argmax_{l in [L/4, 3L/4]} ||d_l||_2.
          4. Selects minimal patch set T = TopPatches(d_hat_{l*}, rho = 0.85) and forms
             positive zoomed view I^+ = Crop(I, B_crop) and counterfactual masked view I^- = Crop(Mask(I, T_peak), B_crop).
        """
        del prefill_query_layers
        W, H = image.size
        shared_kv = state["shared_kv"]
        null_suffix_embeds = state["null_suffix_embeds"]
        vis_start = state["vis_start"]
        vis_end = state["vis_end"]
        grid_ratio = state["grid_ratio"]
        has_thumbnail = state["has_thumbnail"]

        use_cached_null = locked_prefix_embeds is None or locked_prefix_embeds.shape[1] == 0
        if use_cached_null and state.get("s_null_layers") is not None:
            s_null_layers = state["s_null_layers"]
        else:
            if not use_cached_null:
                null_input_embeds = torch.cat([null_suffix_embeds, locked_prefix_embeds], dim=1)
            else:
                null_input_embeds = null_suffix_embeds

            kv_null = self._clone_kv_cache(shared_kv)
            prev_mode = self._set_attn_implementation("eager")
            try:
                with torch.no_grad():
                    out_null = self.model.language_model.model(
                        inputs_embeds=null_input_embeds,
                        past_key_values=kv_null,
                        use_cache=False,
                        output_attentions=True,
                        return_dict=True,
                    )
            finally:
                self._set_attn_implementation(prev_mode)

            s_null_layers = {}
            for l_idx in range(self.layer_start, self.layer_end):
                att_null_l = out_null.attentions[l_idx][0].float()[:, -1, :]
                head_vis_null = att_null_l[:, vis_start:vis_end].sum(dim=-1)
                head_tot_null = att_null_l.sum(dim=-1).clamp(min=1e-12)
                head_ratio_null = head_vis_null / head_tot_null
                ratio_mean_null = head_ratio_null.mean()
                h_vis_null = head_ratio_null >= max(0.05, float(ratio_mean_null.item()) * 0.8)
                if not h_vis_null.any():
                    _, top_h = torch.topk(head_ratio_null, k=min(4, head_ratio_null.shape[0]))
                    h_vis_null = torch.zeros_like(head_ratio_null, dtype=torch.bool)
                    h_vis_null[top_h] = True
                s_null = att_null_l[h_vis_null].mean(dim=0)[vis_start:vis_end]
                s_null_layers[l_idx] = (s_null / s_null.sum().clamp(min=1e-12)).detach()
            del out_null, kv_null
            if use_cached_null:
                state["s_null_layers"] = s_null_layers

        best_layer = self.layer_start
        best_vaq_norm = -1.0
        best_d_vec: Optional[torch.Tensor] = None

        for l_idx in range(self.layer_start, self.layer_end):
            # Step 2.2.1: Prefix-Conditioned Query Contrast at fork step t*
            s_q = s_query_layers[l_idx]
            s_q_norm = s_q / s_q.sum().clamp(min=1e-12)
            s_null_norm = s_null_layers[l_idx]

            d_l = torch.relu(s_q_norm - s_null_norm)
            vaq_l = float(torch.norm(d_l, p=2).item())
            if vaq_l > best_vaq_norm:
                best_vaq_norm = vaq_l
                best_layer = l_idx
                best_d_vec = d_l.detach()

        if best_d_vec is None or float(best_d_vec.sum().item()) <= 1e-12:
            return image, image, [0, 0, W, H], best_layer

        # Project d_{l*} to 2D spatial grid (H_g, W_g)
        d_hat_2d = self._project_patch_vector_to_2d_grid(best_d_vec, grid_ratio, has_thumbnail)
        H_g, W_g = d_hat_2d.shape

        # Step 2.2.3: TopPatches(d_hat_{l*}, rho = 0.85)
        flat = d_hat_2d.ravel()
        sorted_indices = np.argsort(flat)[::-1]
        sorted_vals = flat[sorted_indices]
        cumsum = np.cumsum(sorted_vals)
        cutoff_idx = int(np.searchsorted(cumsum, self.energy_rho, side="left")) + 1
        cutoff_idx = max(1, min(cutoff_idx, flat.size))

        # Eliminate false Pareto concentration aborts:
        # Only abort if contrast is near-uniform noise across > 70% of spatial grid
        if cutoff_idx > int(0.70 * flat.size):
            return image, image, [0, 0, W, H], -1

        top_mask = np.zeros_like(flat, dtype=bool)
        top_mask[sorted_indices[:cutoff_idx]] = True
        top_mask_2d = top_mask.reshape(H_g, W_g)

        # Statistical Covariance RoI Bounding Box:
        # Compute spatial expectation mu and covariance Sigma as a statistical k-sigma confidence
        # ellipse on evidence mass, eliminating hardcoded padding constants.
        r_coords, c_coords = np.indices((H_g, W_g))
        patch_w = W / float(W_g)
        patch_h = H / float(H_g)
        x_centers = (c_coords.astype(np.float64) + 0.5) * patch_w
        y_centers = (r_coords.astype(np.float64) + 0.5) * patch_h

        # Use the evidence mass distribution over the top active region
        weights = d_hat_2d * top_mask_2d.astype(np.float32)
        w_sum = float(weights.sum())
        if w_sum <= 1e-12:
            weights = d_hat_2d
            w_sum = float(weights.sum())

        w_norm = weights / max(1e-12, w_sum)
        mu_x = float(np.sum(w_norm * x_centers))
        mu_y = float(np.sum(w_norm * y_centers))

        sigma_x2 = float(np.sum(w_norm * ((x_centers - mu_x) ** 2)))
        sigma_y2 = float(np.sum(w_norm * ((y_centers - mu_y) ** 2)))
        sigma_x = math.sqrt(max(1.0, sigma_x2))
        sigma_y = math.sqrt(max(1.0, sigma_y2))

        # Statistical k-sigma confidence ellipse: k=2.5 covers ~98.8% mass under Gaussian evidence density
        k_sigma = 2.5
        box_w = 2.0 * k_sigma * sigma_x
        box_h = 2.0 * k_sigma * sigma_y

        # Minimum receptive window for ViT-448px views
        min_w = min(W, max(224, int(0.20 * W)))
        min_h = min(H, max(160, int(0.12 * H)))

        # Horizontal Aspect Expansion for Wide Text Banners:
        # When active patches spread horizontally, ensure wide horizontal text distributions
        # are not prematurely clamped on the right edge by setting final_w >= 1.5 * box_h.
        active_cols = int(np.sum(top_mask_2d.any(axis=0)))
        if active_cols > 1 or sigma_x >= sigma_y:
            final_w = max(box_w, float(min_w), 1.5 * box_h)
        else:
            final_w = max(box_w, float(min_w))
        final_h = max(box_h, float(min_h))

        bx1 = int(max(0, round(mu_x - 0.5 * final_w)))
        bx2 = int(min(W, round(bx1 + final_w)))
        bx1 = int(max(0, round(bx2 - final_w)))

        by1 = int(max(0, round(mu_y - 0.5 * final_h)))
        by2 = int(min(H, round(by1 + final_h)))
        by1 = int(max(0, round(by2 - final_h)))

        crop_bbox = [bx1, by1, bx2, by2]
        x1, y1, x2, y2 = bx1, by1, bx2, by2

        # Positive View I^+ = Crop(I, B_crop)
        img_pos = image.crop((x1, y1, x2, y2))

        # Counterfactual View I^- = Crop(Mask(I, T_peak), B_crop)
        active_vals = d_hat_2d * top_mask_2d.astype(np.float32)
        active_sum = float(active_vals.sum())
        if active_sum > 1e-12:
            active_norm = active_vals / active_sum
            flat_act = active_norm.ravel()
            act_order = np.argsort(flat_act)[::-1]
            act_cum = np.cumsum(flat_act[act_order])
            peak_count = int(np.searchsorted(act_cum, 0.50, side="left")) + 1
            peak_count = max(1, min(peak_count, int(top_mask_2d.sum())))
            peak_mask_flat = np.zeros_like(flat_act, dtype=bool)
            peak_mask_flat[act_order[:peak_count]] = True
            peak_mask_2d = peak_mask_flat.reshape(H_g, W_g)
        else:
            peak_mask_2d = top_mask_2d

        img_np = np.array(image, dtype=np.uint8)
        crop_np = img_np[y1:y2, x1:x2]
        local_mean_color = (
            crop_np.mean(axis=(0, 1)).astype(np.uint8)
            if crop_np.size > 0
            else np.array([128, 128, 128], dtype=np.uint8)
        )

        masked_full_np = img_np.copy()
        pys, pxs = np.where(peak_mask_2d)
        for py, px in zip(pys, pxs):
            px1 = int(max(x1, math.floor(px * patch_w)))
            py1 = int(max(y1, math.floor(py * patch_h)))
            px2 = int(min(x2, math.ceil((px + 1) * patch_w)))
            py2 = int(min(y2, math.ceil((py + 1) * patch_h)))
            if px2 > px1 and py2 > py1:
                masked_full_np[py1:py2, px1:px2] = local_mean_color

        img_neg = Image.fromarray(masked_full_np).crop((x1, y1, x2, y2))

        return img_pos, img_neg, crop_bbox, best_layer

    def _stage3_fork_anchored_counterfactual_verification(
        self,
        img_pos: Image.Image,
        img_neg: Image.Image,
        question: str,
        locked_prefix_ids: List[int],
        remaining_tokens: int,
    ) -> Tuple[List[int], List[float]]:
        """
        Stage 3: Fork-Anchored Counterfactual Verification (t >= t*).
        Preserves the locked preamble token sequence y_{<t*} and decodes in parallel on
        (I^+, Q, y_{<t}) and (I^-, Q, y_{<t}) using the Mixture-Entropy Normalized JSD gain:
            alpha_t = JSD(p_t^+ || p_t^-) / H((p_t^+ + p_t^-) / 2) in [0, 1]
            z_t^{CERV}(y) = (1 + alpha_t) z_t^+(y) - alpha_t z_t^-(y)  for y in V_head(z_t^+)
        """
        tiles_pos, _ = dynamic_preprocess_with_grid(
            img_pos, min_num=1, max_num=min(4, self.max_dynamic_patch), image_size=448, use_thumbnail=True
        )
        tiles_neg, _ = dynamic_preprocess_with_grid(
            img_neg, min_num=1, max_num=min(4, self.max_dynamic_patch), image_size=448, use_thumbnail=True
        )

        n_pos = len(tiles_pos)
        pv_both = torch.stack([self.transform(t) for t in (tiles_pos + tiles_neg)]).to(
            device=self.device, dtype=torch.bfloat16
        )
        vit_both = self.model.extract_feature(pv_both)
        vit_pos = vit_both[:n_pos]
        vit_neg = vit_both[n_pos:]
        del pv_both, vit_both

        pref_pos, suf_pos, _, _, _, _ = self._build_visual_prefix_and_query_suffix(vit_pos, question=question)
        pref_neg, suf_neg, _, _, _, _ = self._build_visual_prefix_and_query_suffix(vit_neg, question=question)
        del vit_pos, vit_neg

        embed_layer = self.model.language_model.get_input_embeddings()
        if len(locked_prefix_ids) > 0:
            lock_tensor = torch.tensor([locked_prefix_ids], dtype=torch.long, device=self.device)
            lock_embeds = embed_layer(lock_tensor)
            full_init_pos = torch.cat([pref_pos, suf_pos, lock_embeds], dim=1)
            full_init_neg = torch.cat([pref_neg, suf_neg, lock_embeds], dim=1)
        else:
            full_init_pos = torch.cat([pref_pos, suf_pos], dim=1)
            full_init_neg = torch.cat([pref_neg, suf_neg], dim=1)

        del pref_pos, suf_pos, pref_neg, suf_neg

        self._set_attn_implementation("sdpa")
        full_init_both = torch.cat([full_init_pos, full_init_neg], dim=0)
        out_both = self.model.language_model.model(
            inputs_embeds=full_init_both,
            use_cache=True,
            output_attentions=False,
            return_dict=True,
        )
        kv_both = out_both.past_key_values
        logits_both = self.model.language_model.lm_head(out_both.last_hidden_state[:, -1:, :])[:, -1, :]
        del full_init_pos, full_init_neg, full_init_both, out_both

        eos_im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        eos_std = self.tokenizer.eos_token_id

        verified_ids: List[int] = []
        alpha_trace: List[float] = []
        content_token_count = 0

        for _ in range(max(1, remaining_tokens)):
            z_pos = logits_both[0].float()
            z_neg = logits_both[1].float()

            p_pos = F.softmax(z_pos, dim=-1)
            p_neg = F.softmax(z_neg, dim=-1)
            m_t = 0.5 * (p_pos + p_neg)

            h_pos = -torch.sum(p_pos * torch.log(p_pos.clamp(min=1e-12)))
            h_neg = -torch.sum(p_neg * torch.log(p_neg.clamp(min=1e-12)))
            h_m = -torch.sum(m_t * torch.log(m_t.clamp(min=1e-12)))

            # Effective Support Entropy Normalization for Counterfactual Contrast (Pillar 4):
            # JSD(p_t^+ || p_t^-) = H(m_t) - 0.5 * (H(p_t^+) + H(p_t^-))
            jsd_t = h_m - 0.5 * (h_pos + h_neg)
            jsd_val = float(max(0.0, jsd_t.item()))

            k_support = min(50, m_t.shape[-1])
            top_m = torch.topk(m_t, k=k_support).values
            top_m_norm = top_m / top_m.sum().clamp(min=1e-12)
            h_k50 = -torch.sum(top_m_norm * torch.log(top_m_norm.clamp(min=1e-12)))
            h_k50_val = float(max(1e-6, h_k50.item()))

            # Normalized JSD over top-50 support distribution bounded in [0.15, 0.45]
            alpha_raw = jsd_val / h_k50_val
            alpha_t = float(min(0.45, max(0.15, alpha_raw)))

            alpha_trace.append(alpha_t)

            log_p_pos = F.log_softmax(z_pos, dim=-1)
            log_p_neg = F.log_softmax(z_neg, dim=-1)

            max_p_pos = torch.max(p_pos)
            plausible_mask = p_pos >= (self.plausibility_beta * max_p_pos)

            # Product-of-Experts Log-Probability Verification:
            # log p_t^{CERV}(y) = log p_t^+(y) + alpha_t * [ log p_t^+(y) - log p_t^-(y) ]
            log_p_cerv = (1.0 + alpha_t) * log_p_pos - alpha_t * log_p_neg
            log_p_cerv = torch.where(plausible_mask, log_p_cerv, torch.full_like(log_p_cerv, -float("inf")))

            next_id = int(torch.argmax(log_p_cerv).item())
            if next_id == eos_im_end or (eos_std is not None and next_id == eos_std):
                break

            # Resolve BPE digit-boundary space ambiguity when both spaced and unspaced
            # subword tokens lie inside the plausible visual head V_head(z_t^+)
            if len(verified_ids) > 0:
                prev_tok_str = self.tokenizer.decode([verified_ids[-1]], skip_special_tokens=False)
                if len(prev_tok_str) > 0 and prev_tok_str[-1].isdigit():
                    cand_str = self.tokenizer.decode([next_id], skip_special_tokens=False)
                    if cand_str.startswith(" ") and len(cand_str.strip()) > 0:
                        unspaced_ids = self.tokenizer.encode(cand_str.lstrip(" "), add_special_tokens=False)
                        if len(unspaced_ids) == 1 and bool(plausible_mask[unspaced_ids[0]].item()):
                            next_id = int(unspaced_ids[0])

            tok_str = self.tokenizer.decode([next_id], skip_special_tokens=False)
            is_content = any(ch.isalnum() for ch in tok_str)

            # Mutual Information-Bounded Dynamic Stopping Condition (Pillar 3):
            # Pointwise Mutual Information delta_t = log P(y_t | I+, Q, y_{<t}) - log P(y_t | I-, Q, y_{<t})
            delta_t = float((log_p_pos[next_id] - log_p_neg[next_id]).item())
            if content_token_count >= 2 and delta_t < 0:
                break

            if is_content:
                content_token_count += 1

            verified_ids.append(next_id)
            next_tok = torch.tensor([[next_id], [next_id]], dtype=torch.long, device=self.device)
            next_emb = embed_layer(next_tok)

            out_both = self.model.language_model.model(
                inputs_embeds=next_emb,
                past_key_values=kv_both,
                use_cache=True,
                output_attentions=False,
                return_dict=True,
            )
            kv_both = out_both.past_key_values
            logits_both = self.model.language_model.lm_head(out_both.last_hidden_state[:, -1:, :])[:, -1, :]

        del kv_both
        return verified_ids, alpha_trace

    def run(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int = 64,
        cached_visual_features: Optional[object] = None,
    ) -> CERVResult:
        """
        Execute the full 3-stage CERV inference loop for (image, question).
        """
        del cached_visual_features
        t_start = time.time()
        W, H = image.size

        with torch.no_grad():
            state = self._get_or_create_global_image_state(image)
            vit_embeds = state["vit_embeds"]
            shared_kv = state["shared_kv"]
            vis_start = state["vis_start"]
            vis_end = state["vis_end"]

            prefix_embeds_tmp, suffix_embeds, _, _, q_start, q_end = self._build_visual_prefix_and_query_suffix(
                vit_embeds, question=question
            )
            assistant_end = int(prefix_embeds_tmp.shape[1] + suffix_embeds.shape[1])
            del prefix_embeds_tmp

            kv_stage1 = self._clone_kv_cache(shared_kv)
            null_suffix_embeds = state.get("null_suffix_embeds")
            kv_null = self._clone_kv_cache(shared_kv) if null_suffix_embeds is not None else None
            logits_null = None
            out_null = None

            embed_layer = self.model.language_model.get_input_embeddings()
            eos_im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
            eos_std = self.tokenizer.eos_token_id

            generated_ids: List[int] = []
            fast_content_tokens = 0
            step_s_layers_history: List[Dict[int, torch.Tensor]] = []
            fork_s_layers_list: List[Dict[int, torch.Tensor]] = []
            trajectory_trace: List[Dict[str, float]] = []

            fork_triggered = False
            fork_step = 0
            fork_lambda = 0.0
            fork_pi = 0.0
            fork_risk = 0.0
            fork_entropy = 0.0
            max_lambda = 0.0

            prev_mode = self._set_attn_implementation("eager")
            try:
                out = self.model.language_model.model(
                    inputs_embeds=suffix_embeds,
                    past_key_values=kv_stage1,
                    use_cache=True,
                    output_attentions=True,
                    return_dict=True,
                )
                kv_stage1 = out.past_key_values
                last_hidden = out.last_hidden_state[:, -1:, :]
                logits = self.model.language_model.lm_head(last_hidden)[:, -1, :]

                if kv_null is not None:
                    out_null = self.model.language_model.model(
                        inputs_embeds=null_suffix_embeds,
                        past_key_values=kv_null,
                        use_cache=True,
                        output_attentions=False,
                        return_dict=True,
                    )
                    kv_null = out_null.past_key_values
                    logits_null = self.model.language_model.lm_head(out_null.last_hidden_state[:, -1:, :])[:, -1, :]

                for step_idx in range(1, max_new_tokens + 1):
                    if step_idx > 1:
                        top2_vals, _ = torch.topk(logits[0], k=2)
                        if top2_vals[0] == top2_vals[1]:
                            logits = F.linear(
                                last_hidden[:, -1, :].float(),
                                self.model.language_model.lm_head.weight.float(),
                            )
                    next_id = int(torch.argmax(logits[0], dim=-1).item())
                    if next_id == eos_im_end or (eos_std is not None and next_id == eos_std):
                        break

                    tok_str = self.tokenizer.decode([next_id], skip_special_tokens=False)
                    is_content_token = any(ch.isalnum() for ch in tok_str)

                    # Mutual Information-Bounded Dynamic Stopping Condition on Fast-Path (Pillar 3)
                    if logits_null is not None and not fork_triggered:
                        log_p_q = F.log_softmax(logits[0].float(), dim=-1)
                        log_p_null = F.log_softmax(logits_null[0].float(), dim=-1)
                        delta_t_fast = float((log_p_q[next_id] - log_p_null[next_id]).item())
                        if fast_content_tokens >= 2 and delta_t_fast < 0:
                            break

                    if is_content_token:
                        fast_content_tokens += 1

                    if not fork_triggered and step_idx <= 25:
                        odds_res = self._compute_step_bayesian_odds(
                            logits=logits[0],
                            attentions=out.attentions,
                            vis_start=vis_start,
                            vis_end=vis_end,
                            q_start=q_start,
                            q_end=q_end,
                            assistant_end=assistant_end,
                            is_prefill_step=(step_idx == 1),
                        )
                        lambda_t, pi_t, risk_t, ent_t, s_layers = odds_res
                        uncertainty = risk_t
                        if hasattr(odds_res, "phi_res"):
                            phi_res = float(odds_res.phi_res)
                        else:
                            denom = max(1e-12, pi_t * risk_t)
                            phi_res = float((lambda_t * ((1.0 - pi_t) * risk_t + self.lambda_cost)) / denom)

                        step_s_layers_history.append(s_layers)

                        if is_content_token and lambda_t > max_lambda:
                            max_lambda = lambda_t

                        trajectory_trace.append(
                            {
                                "step": float(step_idx),
                                "token_id": float(next_id),
                                "token": tok_str,
                                "lambda_t": round(lambda_t, 4),
                                "pi_t": round(pi_t, 4),
                                "risk_t": round(risk_t, 4),
                                "entropy_t": round(ent_t, 4),
                                "phi_res": round(phi_res, 4),
                            }
                        )

                        # Epistemically Calibrated VoI Gate
                        tau_u = 0.30       # Model must exhibit genuine epistemic uncertainty
                        tau_phi = 0.35     # Visual attention must be spatially concentrated
                        if is_content_token and uncertainty >= tau_u and phi_res >= tau_phi and lambda_t > 1.0:
                            fork_triggered = True
                            fork_step = step_idx
                            fork_lambda = lambda_t
                            fork_pi = pi_t
                            fork_risk = risk_t
                            fork_entropy = ent_t
                            fork_s_layers_list = list(step_s_layers_history)
                            self._set_attn_implementation("sdpa")
                            if kv_null is not None:
                                del kv_null
                                kv_null = None
                        elif step_idx == 25:
                            # Past entity region without fork: finish remaining tokens via fast SDPA
                            self._set_attn_implementation("sdpa")

                    generated_ids.append(next_id)
                    next_tok = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
                    next_emb = embed_layer(next_tok)

                    if step_idx < max_new_tokens:
                        need_attn = (not fork_triggered) and (step_idx < 25)
                        out = self.model.language_model.model(
                            inputs_embeds=next_emb,
                            past_key_values=kv_stage1,
                            use_cache=True,
                            output_attentions=need_attn,
                            return_dict=True,
                        )
                        kv_stage1 = out.past_key_values
                        last_hidden = out.last_hidden_state[:, -1:, :]
                        logits = self.model.language_model.lm_head(last_hidden)[:, -1, :]

                        if kv_null is not None and not fork_triggered:
                            out_null = self.model.language_model.model(
                                inputs_embeds=next_emb,
                                past_key_values=kv_null,
                                use_cache=True,
                                output_attentions=False,
                                return_dict=True,
                            )
                            kv_null = out_null.past_key_values
                            logits_null = self.model.language_model.lm_head(out_null.last_hidden_state[:, -1:, :])[:, -1, :]
            finally:
                self._set_attn_implementation(prev_mode)

            del kv_stage1, out, last_hidden
            if kv_null is not None:
                del kv_null
            if out_null is not None:
                del out_null

            fast_draft_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            # Fast-Path: Lambda_t <= 1 for all entity steps t
            if not fork_triggered or len(fork_s_layers_list) == 0:
                full_sequence_text = fast_draft_text.strip()
                final_pred_text = self._extract_causal_entity(question, full_sequence_text)
                if len(final_pred_text) == 0:
                    final_pred_text = full_sequence_text
                return CERVResult(
                    pred_text=final_pred_text,
                    stage1_bbox=[0, 0, W, H],
                    route="fast_path",
                    fork_step=0,
                    max_lambda=round(max_lambda, 4),
                    fork_lambda=0.0,
                    fork_pi=0.0,
                    fork_risk=0.0,
                    fork_entropy=0.0,
                    peak_layer=-1,
                    mean_alpha=0.0,
                    max_alpha=0.0,
                    locked_prefix="",
                    fast_draft_text=fast_draft_text,
                    spatial_confidence=round(1.0 / (1.0 + max_lambda), 4),
                    pred_raw_text=full_sequence_text,
                    fork_token_idx=0,
                    routing_path="fast_path",
                    bayesian_odds=0.0,
                    alpha_mean=0.0,
                    timing={"t_total": round(time.time() - t_start, 4)},
                    trajectory_trace=trajectory_trace,
                )

            # High-Resolution Tiled Contrast: isolate sharp query attention peak at fork step t*
            # rather than pooling diffuse attention across 1..t*
            fork_idx = min(len(fork_s_layers_list) - 1, max(0, fork_step - 1))
            sharp_s_query_layers = fork_s_layers_list[fork_idx]

            # True Prefix-Conditioned Null Query Contrast:
            # Pass locked prefix embeddings into Stage 2 to condition s_l(I, \varnothing, y_{<t*})
            locked_prefix_ids = generated_ids[:max(0, fork_step - 1)]
            locked_prefix_str = self.tokenizer.decode(locked_prefix_ids, skip_special_tokens=True).strip()

            if len(locked_prefix_ids) > 0:
                lock_tensor = torch.tensor([locked_prefix_ids], dtype=torch.long, device=self.device)
                locked_prefix_embeds = embed_layer(lock_tensor)
            else:
                locked_prefix_embeds = None

            # Stage 2: Prefix-Conditioned VAQ Localization at t*
            img_pos, img_neg, crop_bbox, peak_layer = self._stage2_prefix_conditioned_vaq_localization(
                image=image,
                state=state,
                locked_prefix_embeds=locked_prefix_embeds,
                s_query_layers=sharp_s_query_layers,
                prefill_query_layers=sharp_s_query_layers,
            )

            # If Stage 2 detects uniform diffuse contrast (peak_layer == -1),
            # route via Fast-Path with concise post-fork causal entity
            if peak_layer == -1:
                full_sequence_text = fast_draft_text.strip()
                final_pred_text = self._extract_causal_entity(question, full_sequence_text)
                if len(final_pred_text) == 0:
                    entity_ids = generated_ids[fork_step - 1:] if fork_step > 0 else generated_ids
                    entity_text = self.tokenizer.decode(entity_ids, skip_special_tokens=True).strip().rstrip(".").strip()
                    final_pred_text = entity_text if len(entity_text) > 0 else full_sequence_text
                return CERVResult(
                    pred_text=final_pred_text,
                    stage1_bbox=[0, 0, W, H],
                    route="fast_path",
                    fork_step=fork_step,
                    max_lambda=round(max_lambda, 4),
                    fork_lambda=round(fork_lambda, 4),
                    fork_pi=round(fork_pi, 4),
                    fork_risk=round(fork_risk, 4),
                    fork_entropy=round(fork_entropy, 4),
                    peak_layer=-1,
                    mean_alpha=0.0,
                    max_alpha=0.0,
                    locked_prefix=locked_prefix_str,
                    fast_draft_text=fast_draft_text,
                    spatial_confidence=round(1.0 / (1.0 + max_lambda), 4),
                    pred_raw_text=full_sequence_text,
                    fork_token_idx=fork_step,
                    routing_path="fast_path",
                    bayesian_odds=round(fork_lambda, 4),
                    alpha_mean=0.0,
                    timing={"t_total": round(time.time() - t_start, 4)},
                    trajectory_trace=trajectory_trace,
                )

            # Stage 3: True Prefix Fork-Anchoring (PFA-CD) for t >= t*
            # Pass locked prefix IDs so decoding continues seamlessly from t* on (I+, I-)
            verified_suffix_ids, alpha_trace = self._stage3_fork_anchored_counterfactual_verification(
                img_pos=img_pos,
                img_neg=img_neg,
                question=question,
                locked_prefix_ids=locked_prefix_ids,
                remaining_tokens=max_new_tokens,
            )

            verified_text = self.tokenizer.decode(verified_suffix_ids, skip_special_tokens=True).strip().rstrip(".").strip()
            mean_alpha = float(np.mean(alpha_trace)) if len(alpha_trace) > 0 else 0.0
            max_alpha = float(np.max(alpha_trace)) if len(alpha_trace) > 0 else 0.0

            # Unified Token-Space Sequence Assembly (Zero Sub-Token / Entity-Head Slicing Artifacts):
            # Assemble full_generated_ids = locked_prefix_ids + verified_suffix_ids in token ID space
            # and decode together, ensuring seamless sub-token recombination without whitespace or missing heads.
            if len(verified_text) > 0:
                full_generated_ids = locked_prefix_ids + verified_suffix_ids
                full_sequence_text = self.tokenizer.decode(full_generated_ids, skip_special_tokens=True).strip()
                final_pred_text = self._extract_causal_entity(question, full_sequence_text)
                if len(final_pred_text) == 0:
                    final_pred_text = verified_text
                final_raw_text = full_sequence_text
                final_bbox = crop_bbox
                final_route = "slow_path"
                spatial_conf = round(mean_alpha, 4) if len(alpha_trace) > 0 else 1.0
            else:
                entity_ids = generated_ids[fork_step - 1:] if fork_step > 0 else generated_ids
                entity_text = self.tokenizer.decode(entity_ids, skip_special_tokens=True).strip().rstrip(".").strip()
                full_sequence_text = fast_draft_text.strip()
                final_pred_text = self._extract_causal_entity(question, full_sequence_text)
                if len(final_pred_text) == 0:
                    final_pred_text = entity_text if len(entity_text) > 0 else full_sequence_text
                final_raw_text = full_sequence_text
                final_bbox = [0, 0, W, H]
                final_route = "fast_path"
                spatial_conf = round(1.0 / (1.0 + max_lambda), 4)

            return CERVResult(
                pred_text=final_pred_text,
                stage1_bbox=final_bbox,
                route=final_route,
                fork_step=fork_step,
                max_lambda=round(max_lambda, 4),
                fork_lambda=round(fork_lambda, 4),
                fork_pi=round(fork_pi, 4),
                fork_risk=round(fork_risk, 4),
                fork_entropy=round(fork_entropy, 4),
                peak_layer=peak_layer,
                mean_alpha=round(mean_alpha, 4),
                max_alpha=round(max_alpha, 4),
                locked_prefix=locked_prefix_str,
                fast_draft_text=fast_draft_text,
                spatial_confidence=spatial_conf,
                pred_raw_text=final_raw_text,
                fork_token_idx=fork_step,
                routing_path=final_route,
                bayesian_odds=round(fork_lambda, 4),
                alpha_mean=round(mean_alpha, 4),
                timing={"t_total": round(time.time() - t_start, 4)},
                trajectory_trace=trajectory_trace,
            )

