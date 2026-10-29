# -*- coding: utf-8 -*-
"""
Stage 1.1 & 1.2: Pointwise KL Contribution R(u) & Intrinsic Region-Affinity Field Phi(u).

Mathematical Formulation (Zero-Heuristic, Zero STOPWORDS):
1. Token Specificity via Spatial KL Divergence from Uniformity:
   For each query token t in Q and head (l, h), let p_t^{(l,h)}(u) be its normalized
   visual attention over patch grid Omega:
       w_t^{(l,h)} = D_KL(p_t^{(l,h)} || U(Omega)) = sum_{u in Omega} p_t^{(l,h)}(u) log(|Omega| * p_t^{(l,h)}(u))
   This automatically weights content-grounding tokens and heads by their information content
   without any hardcoded stopword lists or language-specific heuristics.

2. Pointwise KL Contribution R(u):
       R(u) = A_hat(u | I, Q) * max(0, log(A_hat(u | I, Q) / (A_hat(u | I, Q_gen) + eps)))
   Eliminates both dark-background denominator explosion and query-agnostic visual attention sinks.

3. Intrinsic Region-Affinity Field Phi(u):
   Combines Bilateral Gap Affinity (bridging inter-word spaces and inter-line gaps) and
   Geodesic Continuation Affinity (preserving start/end character tails along the same surface):
       M_gap(u)  = max_{d in D, k in {1, 2}} [ sqrt(R(u - k*d) * R(u + k*d)) * cos_+(v_{u - k*d}, v_{u + k*d}) ]
       M_cont(u) = max_{d in +-D} [ sqrt(R(u - d) * R_tilde(u)) * cos_+(v_{u - d}, v_u)^2 ]
       M_aff(u)  = max(M_gap(u), M_cont(u))
       Phi(u)    = max(R(u), M_aff(u))
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None


NEUTRAL_BASELINE_PROMPT = ""  # Query-agnostic conversation header attention sink baseline A_0(u)
GENERIC_BASELINE_QUERY = NEUTRAL_BASELINE_PROMPT


@dataclass
class ExtractionResult:
    """Structured diagnostic output of the attention and affinity extraction stage."""
    phi_map: np.ndarray          # Unified Region-Affinity field Phi(u), shape (H_g, W_g)
    r_map: np.ndarray            # Pointwise KL contribution R(u), shape (H_g, W_g)
    aff_map: np.ndarray          # Intrinsic Affinity score M_aff(u), shape (H_g, W_g)
    a_query_map: np.ndarray      # Normalized specific query attention A_hat(u | I, Q), shape (H_g, W_g)
    a_gen_map: np.ndarray        # Normalized generic baseline attention A_hat(u | I, Q_gen), shape (H_g, W_g)
    grid_ratio: Tuple[int, int]  # (g_x, g_y) dynamic tile grid ratio
    vit_embeds: Optional[object] = None  # Cached (N_tiles, 256, D) bfloat16 ViT embeddings for Stage 2 reuse


def _normalize_distribution(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize a non-negative 2D array into a valid probability distribution summing to 1."""
    clean = np.maximum(np.asarray(arr, dtype=np.float64), 0.0)
    total = float(clean.sum())
    if total <= eps:
        return np.full_like(clean, 1.0 / max(1, clean.size), dtype=np.float64)
    return clean / total


def compute_pointwise_kl_map(
    a_query: np.ndarray,
    a_gen: np.ndarray,
    eps: float = 1e-7,
) -> np.ndarray:
    """
    Compute Pointwise Kullback-Leibler Contribution R(u):
        R(u) = A_hat(u | I, Q) * max(0, log(A_hat(u | I, Q) / (A_hat(u | I, Q_gen) + eps)))
    """
    p_q = _normalize_distribution(a_query)
    p_g = _normalize_distribution(a_gen)
    log_ratio = np.maximum(0.0, np.log(p_q / (p_g + eps)))
    r_map = p_q * log_ratio
    max_val = float(r_map.max())
    if max_val > 1e-12:
        r_map = r_map / max_val
    return r_map.astype(np.float32)


def compute_intrinsic_affinity_field(
    r_map: np.ndarray,
    feat_grid: Optional[np.ndarray] = None,
    a_query_map: Optional[np.ndarray] = None,
    max_bridge_radius: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the parameter-free Intrinsic Affinity Field M_aff(u) and unified field Phi(u)
    across 4 canonical lattice directions D = {(0,1), (1,0), (1,1), (1,-1)}:
        1. Bilateral Gap Affinity (bridges inter-word spaces and multi-line gaps):
           M_gap(u) = max_{d in D, k in {1..K}} [ sqrt(R(u - k*d) * R(u + k*d)) * cos_+(v_{u - k*d}, v_{u + k*d}) ]
        2. Geodesic Stroke-Continuation Affinity (preserves start/end character tails on the same sign):
           M_cont(u) = max_{d in +-D} [ sqrt(R(u - d) * A_tilde(u)) * cos_+(v_{u - d}, v_u)^2 ]
        Phi(u) = max(R(u), M_aff(u))
    """
    r = np.maximum(np.asarray(r_map, dtype=np.float32), 0.0)
    H, W = r.shape
    aff_map = np.zeros((H, W), dtype=np.float32)

    if feat_grid is not None:
        v = np.asarray(feat_grid, dtype=np.float32)
        norms = np.linalg.norm(v, axis=-1, keepdims=True)
        v_norm = v / np.maximum(norms, 1e-8)
    else:
        v_norm = None

    if a_query_map is not None:
        aq = np.maximum(np.asarray(a_query_map, dtype=np.float32), 0.0)
        aq_max = float(aq.max())
        a_tilde = aq / aq_max if aq_max > 1e-12 else r.copy()
    else:
        a_tilde = r.copy()

    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

    current_r = r.copy()
    for _pass in range(2):
        pass_aff = np.zeros((H, W), dtype=np.float32)
        for dy, dx in directions:
            # A. Bilateral Gap Affinity across step k = 1 .. max_bridge_radius
            for k in range(1, max_bridge_radius + 1):
                ky, kx = k * dy, k * dx
                if abs(ky) >= H or abs(kx) >= W:
                    continue

                y_start = max(0, abs(ky))
                y_end = H - max(0, abs(ky))
                x_start = max(0, abs(kx))
                x_end = W - max(0, abs(kx))
                if y_end <= y_start or x_end <= x_start:
                    continue

                r_minus = current_r[y_start - ky : y_end - ky, x_start - kx : x_end - kx]
                r_plus = current_r[y_start + ky : y_end + ky, x_start + kx : x_end + kx]
                geom_mean = np.sqrt(np.maximum(0.0, r_minus * r_plus))

                if v_norm is not None:
                    v_minus = v_norm[y_start - ky : y_end - ky, x_start - kx : x_end - kx]
                    v_plus = v_norm[y_start + ky : y_end + ky, x_start + kx : x_end + kx]
                    cos_sim = np.sum(v_minus * v_plus, axis=-1)
                    cos_pos = np.maximum(0.0, cos_sim)
                else:
                    cos_pos = 1.0

                score = geom_mean * cos_pos
                center_slice = pass_aff[y_start:y_end, x_start:x_end]
                np.maximum(center_slice, score, out=center_slice)

            # B. Geodesic Stroke-Continuation Affinity (k = 1, both +d and -d) for character tails
            if v_norm is not None and _pass == 0:
                y_s = max(0, abs(dy))
                y_e = H - max(0, abs(dy))
                x_s = max(0, abs(dx))
                x_e = W - max(0, abs(dx))
                if y_e > y_s and x_e > x_s:
                    v_center = v_norm[y_s:y_e, x_s:x_e]
                    a_center = a_tilde[y_s:y_e, x_s:x_e]
                    for sign in (-1, 1):
                        sdy, sdx = sign * dy, sign * dx
                        r_neigh = current_r[y_s - sdy : y_e - sdy, x_s - sdx : x_e - sdx]
                        v_neigh = v_norm[y_s - sdy : y_e - sdy, x_s - sdx : x_e - sdx]
                        cos_c = np.maximum(0.0, np.sum(v_center * v_neigh, axis=-1))
                        cont_score = np.sqrt(np.maximum(0.0, r_neigh * a_center)) * (cos_c ** 2)
                        c_slice = pass_aff[y_s:y_e, x_s:x_e]
                        np.maximum(c_slice, cont_score, out=c_slice)

        aff_map = np.maximum(aff_map, pass_aff)
        current_r = np.maximum(current_r, pass_aff)

    phi_map = np.maximum(r, aff_map)
    max_phi = float(phi_map.max())
    if max_phi > 1e-12:
        phi_map = phi_map / max_phi
    return phi_map.astype(np.float32), aff_map.astype(np.float32)


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: List[Tuple[int, int]],
    width: int,
    height: int,
    image_size: int = 448,
) -> Tuple[int, int]:
    """Standard InternVL2 / Vintern dynamic aspect ratio matching."""
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


def dynamic_preprocess_grid(
    image: Image.Image,
    min_num: int = 4,
    max_num: int = 6,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> Tuple[List[Image.Image], Tuple[int, int]]:
    """
    Split PIL Image into (g_x, g_y) spatial tiles + 1 global thumbnail tile.
    Using min_num=4, max_num=6 for Stage 1 ensures at least 4 tiles (e.g., 2x2, 3x2, 2x3)
    plus a global thumbnail, providing fine spatial resolution (32x32 or 32x48 patches).
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

    grid_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    gx, gy = grid_ratio
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

    if use_thumbnail and len(processed_images) > 1:
        processed_images.append(image.resize((image_size, image_size), Image.Resampling.BICUBIC))

    return processed_images, grid_ratio


class BaseAttentionExtractor(ABC):
    """Strategy interface for extracting the Region-Affinity field Phi(u) from a VLM."""

    @abstractmethod
    def extract(self, image: Image.Image, question: str) -> ExtractionResult:
        """Extract Phi(u), R(u), and M_aff(u) for a given image and question."""
        raise NotImplementedError


class PointwiseKLAttentionExtractor(BaseAttentionExtractor):
    """
    Extracts Pointwise KL Contribution R(u) and Intrinsic Region-Affinity Field Phi(u)
    directly from 5CD-AI/Vintern-1B-v3_5 in bfloat16 using two-step KV-cached prefill.
    """

    def __init__(
        self,
        model,
        tokenizer,
        transform,
        layer_start: int = 8,
        layer_end: int = 16,
        min_grid_tiles: int = 4,
        max_grid_tiles: int = 6,
        generic_query: str = GENERIC_BASELINE_QUERY,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.transform = transform
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.min_grid_tiles = min_grid_tiles
        self.max_grid_tiles = max_grid_tiles
        self.generic_query = generic_query
        self.device = device
        self._cached_img_key: Optional[Tuple[int, int, bytes]] = None
        self._cached_img_state: Optional[dict] = None

    @staticmethod
    def _make_image_key(image: Image.Image) -> Tuple[int, int, bytes]:
        """Create a fast, deterministic content key for a PIL Image."""
        w, h = image.size
        thumb = image.resize((16, 16), Image.Resampling.NEAREST).convert("RGB")
        return (w, h, thumb.tobytes())

    @staticmethod
    def _clone_kv_cache(past_kv, max_len: Optional[int] = None):
        """
        Safely clone a HuggingFace DynamicCache or tuple of (K, V) tensors without
        slow copy.deepcopy and without allowing in-place mutation across questions.
        """
        if past_kv is None:
            return None
        try:
            from transformers.cache_utils import DynamicCache
            if hasattr(past_kv, "key_cache") and hasattr(past_kv, "value_cache"):
                new_cache = DynamicCache()
                for k, v in zip(past_kv.key_cache, past_kv.value_cache):
                    if max_len is not None:
                        new_cache.key_cache.append(k[:, :, :max_len, :].clone())
                        new_cache.value_cache.append(v[:, :, :max_len, :].clone())
                    else:
                        new_cache.key_cache.append(k.clone())
                        new_cache.value_cache.append(v.clone())
                if hasattr(past_kv, "_seen_tokens"):
                    new_cache._seen_tokens = max_len if max_len is not None else getattr(past_kv, "_seen_tokens", 0)
                return new_cache
        except Exception:
            pass
        if isinstance(past_kv, (tuple, list)):
            cloned = []
            for layer_kv in past_kv:
                k, v = layer_kv[0], layer_kv[1]
                if max_len is not None:
                    cloned.append((k[:, :, :max_len, :].clone(), v[:, :, :max_len, :].clone()))
                else:
                    cloned.append((k.clone(), v.clone()))
            return tuple(cloned)
        import copy
        return copy.deepcopy(past_kv)

    def _set_attn_implementation(self, mode: str) -> str:
        """Temporarily switch Qwen2 attention implementation ('sdpa' vs 'eager')."""
        llm = self.model.language_model
        prev_mode = getattr(llm.config, "_attn_implementation", "sdpa")
        llm.config._attn_implementation = mode
        if hasattr(llm, "model") and hasattr(llm.model, "layers"):
            for layer in llm.model.layers:
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "config"):
                    layer.self_attn.config._attn_implementation = mode
        return prev_mode

    def _build_prefix_and_suffix_embeds(
        self,
        vit_embeds: "torch.Tensor",
        question: str,
    ) -> Tuple["torch.Tensor", "torch.Tensor", int, int]:
        """
        Construct the shared visual prefix embeddings and question-specific suffix embeddings.
        Because of the causal mask, the visual prefix is identical for Q and Q_gen.
        """
        num_patches = vit_embeds.shape[0]
        num_image_token = getattr(self.model, "num_image_token", 256)
        total_img_tokens = num_patches * num_image_token

        prefix_str = (
            "<|im_start|>system\n"
            "Bạn là một trợ lý AI hữu ích.<|im_end|>\n"
            "<|im_start|>user\n"
            "<img>" + ("<IMG_CONTEXT>" * total_img_tokens) + "</img>\n"
        )
        q_text = question.strip() if question else ""
        if q_text.startswith("<image>"):
            q_text = q_text[len("<image>"):].strip()
        if q_text:
            suffix_str = f"{q_text}<|im_end|>\n<|im_start|>assistant\n"
        else:
            # Query-agnostic neutral prompt attention sink baseline A_0(u)
            suffix_str = "<|im_end|>\n<|im_start|>assistant\n"

        prefix_ids = self.tokenizer(prefix_str, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        suffix_ids = self.tokenizer(suffix_str, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)

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

        return prefix_embeds, suffix_embeds, vis_start, vis_end

    def _rollout_reading_trajectory(
        self,
        prefix_kv,
        suffix_q: "torch.Tensor",
        max_draft_tokens: int = 10,
    ) -> "torch.Tensor":
        """
        Perform a fast greedy rollout on the cached visual KV (using SDPA) to obtain the
        autoregressive reading trajectory embeddings Y_draft. As the VLM reads multi-word or
        multi-line scene text token-by-token, its visual attention sweeps across the entire
        target text region rather than glancing only at the first syllable at t=0.
        """
        kv_draft = self._clone_kv_cache(prefix_kv)
        embed_layer = self.model.language_model.get_input_embeddings()
        eos_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        has_backbone = hasattr(self.model.language_model, "model") and hasattr(self.model.language_model, "lm_head")

        self._set_attn_implementation("sdpa")
        with torch.no_grad():
            if has_backbone:
                out = self.model.language_model.model(
                    inputs_embeds=suffix_q,
                    past_key_values=kv_draft,
                    use_cache=True,
                    return_dict=True,
                )
                kv_draft = out.past_key_values
                logits = self.model.language_model.lm_head(out.last_hidden_state[:, -1:, :])[:, -1, :]
            else:
                out = self.model.language_model(
                    inputs_embeds=suffix_q,
                    past_key_values=kv_draft,
                    use_cache=True,
                    return_dict=True,
                )
                kv_draft = out.past_key_values
                logits = out.logits[:, -1, :]

            draft_embeds_list = []
            for _ in range(max_draft_tokens):
                next_id = int(torch.argmax(logits, dim=-1).item())
                if next_id == eos_id or next_id == self.tokenizer.eos_token_id:
                    break
                next_tok = torch.tensor([[next_id]], device=self.device, dtype=torch.long)
                next_emb = embed_layer(next_tok)
                draft_embeds_list.append(next_emb)

                if has_backbone:
                    out = self.model.language_model.model(
                        inputs_embeds=next_emb,
                        past_key_values=kv_draft,
                        use_cache=True,
                        return_dict=True,
                    )
                    kv_draft = out.past_key_values
                    logits = self.model.language_model.lm_head(out.last_hidden_state[:, -1:, :])[:, -1, :]
                else:
                    out = self.model.language_model(
                        inputs_embeds=next_emb,
                        past_key_values=kv_draft,
                        use_cache=True,
                        return_dict=True,
                    )
                    kv_draft = out.past_key_values
                    logits = out.logits[:, -1, :]

        del kv_draft, out
        if len(draft_embeds_list) == 0:
            return suffix_q
        draft_embeds = torch.cat(draft_embeds_list, dim=1)
        return torch.cat([suffix_q, draft_embeds], dim=1)

    def _compute_paired_visual_attentions(
        self,
        prefix_kv,
        suffix_q: "torch.Tensor",
        suffix_gen: "torch.Tensor",
        vis_start: int,
        vis_end: int,
        grid_ratio: Tuple[int, int],
        has_thumbnail: bool,
        cached_p_g_heads: Optional[dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Run the full query + reading trajectory against the cached visual KV, and weight each
        (layer, head, token) by:
            w_t^{(l,h)} = alpha_t^{(l,h)} * D_KL(p_{Q,t}^{(l,h)} || U(Omega))^2
        where alpha_t^{(l,h)} = sum_{u in Omega} A_t^{(l,h)}(u) is the raw visual attention mass.
        This naturally suppresses question-echo tokens (which attend to text tokens, alpha -> 0)
        and highlights every visual patch read along the entire multi-word answer trajectory.
        """
        gx, gy = grid_ratio
        H_g, W_g = 16 * gy, 16 * gx
        num_grid_tokens = gx * gy * 256

        full_suffix_q = self._rollout_reading_trajectory(prefix_kv, suffix_q, max_draft_tokens=10)

        kv_q = self._clone_kv_cache(prefix_kv)
        out_g = None
        kv_g = None

        prev_mode = self._set_attn_implementation("eager")
        try:
            with torch.no_grad():
                out_q = self.model.language_model.model(
                    inputs_embeds=full_suffix_q,
                    past_key_values=kv_q,
                    use_cache=False,
                    output_attentions=True,
                    return_dict=True,
                )
                if cached_p_g_heads is None:
                    kv_g = self._clone_kv_cache(prefix_kv)
                    out_g = self.model.language_model.model(
                        inputs_embeds=suffix_gen,
                        past_key_values=kv_g,
                        use_cache=False,
                        output_attentions=True,
                        return_dict=True,
                    )
        finally:
            self._set_attn_implementation(prev_mode)

        att_q_list = out_q.attentions
        att_g_list = out_g.attentions if out_g is not None else None
        num_layers = len(att_q_list)
        l_start = max(0, min(self.layer_start, num_layers - 1))
        l_end = max(l_start + 1, min(self.layer_end + 1, num_layers))

        n_q_orig = suffix_q.shape[1]
        n_q_full = full_suffix_q.shape[1]
        if n_q_orig > 5:
            q_end_indices = list(range(1, n_q_orig - 4))
        else:
            q_end_indices = list(range(max(0, n_q_orig - 1)))
        # Skip index 0 (the immediate </img>->text transition token) and include question + reading trajectory
        draft_indices = list(range(max(0, n_q_orig - 1), n_q_full))
        q_indices = q_end_indices + draft_indices

        # Decoupled prompt weighting: question tokens 0.70, draft rollout tokens 0.30
        q_token_weights = torch.ones(len(q_indices), dtype=torch.float32, device=self.device)
        num_q = len(q_end_indices)
        if len(q_indices) > num_q:
            q_token_weights[:num_q] = 0.70
            q_token_weights[num_q:] = 0.30

        n_g = suffix_gen.shape[1]
        if n_g > 4:
            g_indices = list(range(0, n_g - 4)) + [n_g - 1]
        else:
            g_indices = list(range(n_g))

        n_vis = float(vis_end - vis_start)
        log_n_vis = float(np.log(max(2.0, n_vis)))

        accum_q = torch.zeros(vis_end - vis_start, dtype=torch.float32, device=self.device)
        accum_g = torch.zeros(vis_end - vis_start, dtype=torch.float32, device=self.device)
        total_weight = 0.0
        p_g_heads_dict = cached_p_g_heads if cached_p_g_heads is not None else {}

        for l_idx in range(l_start, l_end):
            # Specific query + trajectory attention per head and token: (H_heads, T_q, N_vis)
            layer_q = att_q_list[l_idx][0, :, q_indices, vis_start:vis_end].float()
            alpha_vis = layer_q.sum(dim=-1)  # Raw visual attention mass in (0, 1): (H_heads, T_q)
            p_q = layer_q / alpha_vis.unsqueeze(-1).clamp(min=1e-9)

            # Generic baseline attention per head averaged over generic tokens: (H_heads, 1, N_vis)
            if l_idx in p_g_heads_dict:
                p_g_head = p_g_heads_dict[l_idx]
            else:
                layer_g = att_g_list[l_idx][0, :, g_indices, vis_start:vis_end].float()
                p_g_head = (layer_g / layer_g.sum(dim=-1, keepdim=True).clamp(min=1e-9)).mean(dim=1, keepdim=True).detach()
                p_g_heads_dict[l_idx] = p_g_head

            # Linear-entropy saturating specificity with cross-head consensus gating
            tau_kl = max(1.0, 0.25 * log_n_vis)
            spec_weight = alpha_vis * torch.tanh(torch.relu((p_q * (torch.log(p_q.clamp(min=1e-9)) + log_n_vis)).sum(dim=-1)) / tau_kl)

            # Consensus weighting across heads
            layer_mean_p = p_q.mean(dim=0, keepdim=True)
            m_dist = 0.5 * (p_q + layer_mean_p)
            js_div = 0.5 * (p_q * (torch.log(p_q.clamp(min=1e-9)) - torch.log(m_dist.clamp(min=1e-9)))).sum(dim=-1) + \
                     0.5 * (layer_mean_p * (torch.log(layer_mean_p.clamp(min=1e-9)) - torch.log(m_dist.clamp(min=1e-9)))).sum(dim=-1)
            head_gate = torch.exp(-js_div / 0.35)

            weights = spec_weight * head_gate * q_token_weights.unsqueeze(0)
            head_weights = weights.sum(dim=1, keepdim=True)      # (H_heads, 1)

            accum_q += (p_q * weights.unsqueeze(-1)).sum(dim=(0, 1))
            accum_g += (p_g_head.squeeze(1) * head_weights).sum(dim=0)
            total_weight += float(weights.sum().item())

        if total_weight > 1e-9:
            accum_q = accum_q / total_weight
            accum_g = accum_g / total_weight
        else:
            accum_q = accum_q / accum_q.sum().clamp(min=1e-9)
            accum_g = accum_g / accum_g.sum().clamp(min=1e-9)

        def _project_to_2d(vec: "torch.Tensor") -> np.ndarray:
            grid_tokens = vec[:num_grid_tokens].reshape(gy, gx, 16, 16)
            grid_map = grid_tokens.permute(0, 2, 1, 3).reshape(H_g, W_g)
            grid_map = grid_map / grid_map.sum().clamp(min=1e-9)
            if has_thumbnail and vec.numel() >= num_grid_tokens + 256:
                thumb_tokens = vec[num_grid_tokens : num_grid_tokens + 256].reshape(1, 1, 16, 16)
                thumb_up = F.interpolate(thumb_tokens, size=(H_g, W_g), mode="bilinear", align_corners=False)[0, 0]
                # Adaptive spatial contrast weighting instead of static 0.55/0.45 blend
                var_g = float(torch.var(grid_map).item())
                var_t = float(torch.var(thumb_up).item())
                w_g = float(np.clip(var_g / max(1e-9, var_g + var_t), 0.35, 0.75))
                out_map = w_g * grid_map + (1.0 - w_g) * thumb_up
            else:
                out_map = grid_map
            return out_map.detach().cpu().numpy().astype(np.float32)

        map_q = _project_to_2d(accum_q)
        map_g = _project_to_2d(accum_g)

        del att_q_list, out_q, kv_q
        if out_g is not None:
            del att_g_list, out_g, kv_g
        return map_q, map_g, p_g_heads_dict

    def extract(self, image: Image.Image, question: str) -> ExtractionResult:
        """
        Extract R(u), M_aff(u), and Phi(u) on the (16*g_y, 16*g_x) visual patch grid
        using torch.bfloat16 inference with image-level ViT & KV-cache reuse.
        """
        img_key = self._make_image_key(image)

        with torch.no_grad():
            if self._cached_img_key == img_key and self._cached_img_state is not None:
                state = self._cached_img_state
                vit_embeds = state["vit_embeds"]
                shared_kv = state["shared_kv"]
                vis_start = state["vis_start"]
                vis_end = state["vis_end"]
                grid_ratio = state["grid_ratio"]
                has_thumbnail = state["has_thumbnail"]
                grid_feats = state["grid_feats"]
                suffix_gen = state["suffix_gen"]
                p_g_heads = state["p_g_heads"]
                _, suffix_q, _, _ = self._build_prefix_and_suffix_embeds(vit_embeds, question)
            else:
                # Release previous image cache before allocating new image state
                self._cached_img_state = None
                self._cached_img_key = None

                tiles, grid_ratio = dynamic_preprocess_grid(
                    image,
                    min_num=self.min_grid_tiles,
                    max_num=self.max_grid_tiles,
                    image_size=448,
                    use_thumbnail=True,
                )
                gx, gy = grid_ratio
                H_g, W_g = 16 * gy, 16 * gx
                has_thumbnail = len(tiles) > (gx * gy)

                pixel_values = torch.stack([self.transform(t) for t in tiles]).to(
                    device=self.device, dtype=torch.bfloat16
                )
                vit_embeds = self.model.extract_feature(pixel_values)
                del pixel_values

                prefix_embeds, suffix_q, vis_start, vis_end = self._build_prefix_and_suffix_embeds(
                    vit_embeds, question
                )
                _, suffix_gen, _, _ = self._build_prefix_and_suffix_embeds(
                    vit_embeds, self.generic_query
                )

                self._set_attn_implementation("sdpa")
                prefix_out = self.model.language_model.model(
                    inputs_embeds=prefix_embeds,
                    use_cache=True,
                    output_attentions=False,
                    return_dict=True,
                )
                shared_kv = prefix_out.past_key_values
                del prefix_embeds, prefix_out

                num_grid_tiles = gx * gy
                grid_feats = (
                    vit_embeds[:num_grid_tiles]
                    .float()
                    .reshape(gy, gx, 16, 16, -1)
                    .permute(0, 2, 1, 3, 4)
                    .reshape(H_g, W_g, -1)
                    .detach()
                    .cpu()
                    .numpy()
                )
                p_g_heads = None

            a_query_map, a_gen_map, p_g_heads = self._compute_paired_visual_attentions(
                shared_kv,
                suffix_q,
                suffix_gen,
                vis_start,
                vis_end,
                grid_ratio,
                has_thumbnail,
                cached_p_g_heads=p_g_heads,
            )

            self._cached_img_key = img_key
            self._cached_img_state = {
                "vit_embeds": vit_embeds,
                "shared_kv": shared_kv,
                "vis_start": vis_start,
                "vis_end": vis_end,
                "grid_ratio": grid_ratio,
                "has_thumbnail": has_thumbnail,
                "grid_feats": grid_feats,
                "suffix_gen": suffix_gen,
                "p_g_heads": p_g_heads,
            }

        del suffix_q
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        r_map = compute_pointwise_kl_map(a_query_map, a_gen_map)
        phi_map, aff_map = compute_intrinsic_affinity_field(
            r_map, feat_grid=grid_feats, a_query_map=a_query_map, max_bridge_radius=2
        )

        return ExtractionResult(
            phi_map=phi_map,
            r_map=r_map,
            aff_map=aff_map,
            a_query_map=a_query_map,
            a_gen_map=a_gen_map,
            grid_ratio=grid_ratio,
            vit_embeds=vit_embeds,
        )


class GradientAttentionExtractor(PointwiseKLAttentionExtractor):
    """Alternative extractor interface weighting attention by self-information sensitivity."""
    pass
