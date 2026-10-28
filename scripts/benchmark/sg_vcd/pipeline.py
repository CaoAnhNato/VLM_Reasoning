# -*- coding: utf-8 -*-
"""
SG-VCD End-to-End Pipeline (Stage 1 Spatial Grounding + Stage 2 Dynamic Spatial PMI Decoding).

Follows Pipe-and-Filter Architecture, Strategy Pattern, and Dependency Injection:
- Stage 1:
  1. PointwiseKLAttentionExtractor -> ExtractionResult (Phi(u), R(u), M_aff(u))
  2. KLDivergenceRegionSelector    -> RegionSelectionResult (C* maximizing Bernoulli J_KL(C))
  3. IsotropicROICropper           -> CropResult (448x448 isotropic square ROI tile)
- Stage 2:
  Dynamic Spatio-Temporal Calibrated Contrastive Decoding in torch.bfloat16:
  - Positive High-Resolution Evidence Branch: [I_global_thumb, I_ROI]
  - Negative Low-Pass Prior Branch: [I_global_thumb, I_ROI_lowpass]
  - Dynamic Calibration: alpha_t = alpha_0 * S_spatial * M_decay(t) * U_divergence(t)
    with Jensen-Shannon divergence syntax gating (D_JS(P_pos || P_neg) <= tau_JS => alpha_t -> 0).
  - Entropy-Adaptive Plausibility Truncation: beta_t = beta_0 * exp(-H(P_t^(+))).
  - Visual Feature Caching for global thumbnail ViT representations.
"""

from dataclasses import dataclass, field
import gc
import math
import re
import time
from typing import Any, Dict, List, Optional, Tuple
import unicodedata

import numpy as np
from PIL import Image, ImageFilter

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None

from .adaptive_propagator import (
    AdaptiveScalePropagator,
    BaseSeedPropagator,
    verify_and_resolve_candidate_boxes,
)
from .attention_extractor import (
    BaseAttentionExtractor,
    ExtractionResult,
    PointwiseKLAttentionExtractor,
)
from .bbox_selector import (
    BaseRegionSelector,
    KLDivergenceRegionSelector,
    MultiRegionSelectionResult,
    RegionSelectionResult,
)
from .dynamic_cd import (
    DynamicContrastiveDecoder,
    compute_adaptive_lowpass_radius,
)
from .feature_cache import (
    VisualFeatureCache,
    default_feature_cache,
)
from .roi_cropper import (
    BaseROICropper,
    CropResult,
    IsotropicROICropper,
    apply_adaptive_laplacian_edge_reemphasis,
)


@dataclass
class PipelineResult:
    """
    Unified pipeline result meeting both PROJECT.md interface contract and
    diagnostic backward-compatibility requirements for run_stage1_vicrop.py.
    """
    # Authoritative PROJECT.md interface contract fields:
    pred_text: str                     # Clean, concise answer
    pred_raw_text: str                 # Raw generated tokens
    stage1_bbox: List[int]             # [x1, y1, x2, y2]
    spatial_confidence: float          # In [0.0, 1.0]
    vcd_alpha_mean: float              # Average alpha_t across decoding steps
    entropy_mean: float                # Average predictive entropy
    timing: Dict[str, float]           # Latency breakdown

    # Diagnostic / backward-compatibility fields:
    extraction: Optional[ExtractionResult] = None
    selection: Optional[RegionSelectionResult] = None
    crop: Optional[CropResult] = None
    pred_roi_only: str = ""
    pred_roi_plus_global: str = ""
    pred_sg_vcd: str = ""
    stage1_bboxes: Optional[List[List[int]]] = None
    multi_selection: Optional[MultiRegionSelectionResult] = None
    multi_crops: Optional[List[CropResult]] = None


def strip_conversational_preamble(text: str, question: Optional[str] = None) -> str:
    """
    Zero-heuristic conversational preamble strip & entity normalizer:
    Isolates the concise answer entity by stripping standard conversational
    preambles, outer quotes, trailing punctuation, and question-echo framing.
    Preserves raw generated tokens in PipelineResult.pred_raw_text.
    """
    if not text:
        return ""

    clean = unicodedata.normalize("NFC", text).strip()

    # Generic conversational introductory clause patterns
    intro_patterns = [
        r"^(?:bức\s+)?(?:hình\s+ảnh|ảnh|hình)(?:\s+này)?\s+(?:được\s+chụp\s+vào\s+dịp|có\s+nguồn\s+(?:từ|gốc\s+từ)|lấy\s+nguồn\s+từ|nguồn\s+từ|chụp|cho\s+thấy|ghi\s+lại|có\s+chữ|chỉ\s+ra|là|có)\s*[:,\-]?\s*",
        r"^(?:câu\s+trả\s+lời|đáp\s+án)(?:\s+(?:của\s+tôi|chính\s+xác|cho\s+câu\s+hỏi))?\s*(?:là\s*[:,\-]?|[:,\-])\s*",
        r"^(?:trong\s+(?:bức\s+)?(?:hình\s+ảnh|ảnh|hình|khung\s+hình)\s+(?:có\s+(?:chữ|từ|dòng\s+chữ|chi\s+tiết)?|là|thấy|xuất\s+hiện|ghi))\s*[:,\-]?\s*",
        r"^(?:chữ\s+(?:trong\s+ảnh|trên\s+ảnh|xuất\s+hiện)\s+(?:là|ghi))\s*[:,\-]?\s*",
        r"^(?:được\s+chụp\s+tại|chụp\s+tại|được\s+chụp\s+vào\s+dịp)\s*[:,\-]?\s*",
        r"^(?:đây\s+là|đó\s+là|nó\s+là)\s*[:,\-]\s*",
        r"^(?:có\s+thể\s+thấy\s+(?:rằng)?)\s*[:,\-]?\s*",
        r"^(?:theo\s+(?:thông\s+tin\s+trong\s+)?(?:bức\s+)?(?:hình\s+ảnh|ảnh|hình)(?:[:,\s]*)?)\s*",
        r"^(?:dựa\s+vào\s+(?:bức\s+)?(?:hình\s+ảnh|ảnh|hình)(?:[:,\s]*)?)\s*",
        r"^(?:kết\s+quả\s+là)\s*[:,\-]?\s*",
    ]

    # Iteratively strip prompt instructions, outer quotes, and introductory clauses
    for _ in range(6):
        prev_len = len(clean)

        # Strip prompt-instruction echo if present
        clean = re.sub(
            r"^(?:chỉ\s+)?(?:ghi|trả\s+lời|nêu)(?:\s+thật|\s+đúng|\s+ngắn\s+gọn|\s+từ|\s+hoặc|\s+cụm\s+từ|\s+đáp\s+án)*\s*[:,\-]?\s*",
            "",
            clean,
            flags=re.IGNORECASE,
        ).strip()

        # Strip enclosing quotation marks
        if len(clean) >= 2 and ((clean[0] in ('"', "'", '“', '‘') and clean[-1] in ('"', "'", '”', '’'))):
            clean = clean[1:-1].strip()

        for pattern in intro_patterns:
            m = re.match(pattern, clean, flags=re.IGNORECASE)
            if m:
                candidate = clean[m.end():].strip()
                if candidate:
                    clean = candidate
                break

        if len(clean) == prev_len:
            break

    # If question is provided, check for question-echo subject framing before copula / relation verbs
    if question and clean:
        q_words = set(re.findall(r"\w+", question.lower()))
        copula_pattern = r"\s+(?:đang\s+diễn\s+ra\s+trên\s+kênh|diễn\s+ra\s+trên\s+kênh|được\s+diễn\s+ra\s+tại|diễn\s+ra\s+tại|được\s+chụp\s+vào\s+dịp|tên\s+là|có\s+tên\s+là|có\s+tên|tên|chuyên\s+về|chuyên|thuộc\s+về|lấy\s+nguồn\s+từ|nguồn\s+từ|là|được\s+gọi\s+là|bán\s+hàng|bán|có)\s+"
        copula_match = re.search(copula_pattern, clean, flags=re.IGNORECASE)
        if copula_match:
            prefix_part = clean[:copula_match.start()].strip()
            p_words = set(re.findall(r"\w+", prefix_part.lower()))
            candidate_entity = clean[copula_match.end():].strip()
            if p_words and len(p_words & q_words) / len(p_words) >= 0.5 and candidate_entity:
                clean = candidate_entity

    # Strip Vietnamese organizational classifier prefix ("Nhà thuốc", "Cửa hàng", "Công ty", etc.)
    # when the question asks for an entity name or explicitly mentions the classifier (Refinement D).
    classifier_patterns = [
        r"^(?:nhà\s+thuốc|hiệu\s+thuốc|cửa\s+hàng|tiệm\s+cà\s+phê|tiệm|quán\s+cà\s+phê|quán\s+ăn|quán|công\s+ty(?:\s+tnhh|\s+cổ\s+phần)?|doanh\s+nghiệp|trường|phòng(?:\s+khám|\s+giao\s+dịch)?|sở|bệnh\s+viện|khách\s+sạn|ngân\s+hàng)\s+",
    ]
    if question and clean:
        q_lower = question.lower()
        is_name_query = any(k in q_lower for k in ["tên", "gọi là", "hiệu gì", "nhãn hiệu", "thương hiệu"])
        for cp in classifier_patterns:
            m = re.match(cp, clean, flags=re.IGNORECASE)
            if m:
                matched_prefix = m.group(0).strip().lower()
                rem = clean[m.end():].strip()
                if rem and is_name_query:
                    if matched_prefix in q_lower or any(w in q_lower for w in matched_prefix.split()):
                        clean = rem
                        break

    # Phase-switched delimiter extraction (e.g. "Chữ: ...\nĐáp án: ...")
    lines = clean.split("\n")
    if len(lines) > 1:
        for line in reversed(lines):
            line_str = line.strip()
            if re.match(r"^(?:đáp\s+án|câu\s+trả\s+lời|kết\s+luận|trả\s+lời)\s*[:,\-]?\s*", line_str, flags=re.IGNORECASE):
                extracted = re.sub(r"^(?:đáp\s+án|câu\s+trả\s+lời|kết\s+luận|trả\s+lời)\s*[:,\-]?\s*", "", line_str, flags=re.IGNORECASE).strip()
                if extracted:
                    clean = extracted
                    break

    # Strip trailing periods and whitespace (sentence terminator)
    clean = re.sub(r"[\.,;:!\?\s]+$", "", clean).strip()

    # Strip enclosing quotes again if newly exposed
    if len(clean) >= 2 and ((clean[0] in ('"', "'", '“', '‘') and clean[-1] in ('"', "'", '”', '’'))):
        clean = clean[1:-1].strip()

    return clean if clean else unicodedata.normalize("NFC", text).strip()


def format_concise_decoding_prompt(question: str) -> str:
    """
    Format decoding prompt aligned with Vintern SFT distribution for concise entity answering (Refinement D).
    Uses a single-turn instruction without a trailing colon line to prevent prompt-instruction echoing.
    """
    q_clean = question.strip()
    if q_clean.startswith("<image>"):
        return q_clean
    if q_clean.startswith("Câu hỏi:") or q_clean.startswith("Trả lời") or q_clean.startswith("Dựa vào chữ"):
        return f"<image>\n{q_clean}"
    return f"<image>\nTrả lời thật ngắn gọn câu hỏi sau dựa vào chữ xuất hiện trong ảnh (chỉ ghi đúng từ hoặc cụm từ đáp án): {q_clean}"


class SGVCDPipeline:
    """
    Unified Zero-Heuristic SG-VCD Pipeline for 5CD-AI/Vintern-1B-v3_5 (in torch.bfloat16).
    """

    def __init__(
        self,
        model,
        tokenizer,
        transform,
        extractor: Optional[BaseAttentionExtractor] = None,
        selector: Optional[BaseRegionSelector] = None,
        cropper: Optional[BaseROICropper] = None,
        propagator: Optional[BaseSeedPropagator] = None,
        decoder: Optional[DynamicContrastiveDecoder] = None,
        feature_cache: Optional[VisualFeatureCache] = None,
        alpha_cd: float = 0.65,
        beta_apc: float = 0.035,
        tau_js: float = 0.04,
        decode_auxiliary_branches: bool = False,
        max_visual_tiles: int = 2,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.transform = transform
        self.device = device
        self.alpha_cd = alpha_cd
        self.beta_apc = beta_apc
        self.decode_auxiliary_branches = decode_auxiliary_branches
        self.max_visual_tiles = int(max_visual_tiles)

        self.extractor = extractor or PointwiseKLAttentionExtractor(
            model=model,
            tokenizer=tokenizer,
            transform=transform,
            layer_start=8,
            layer_end=16,
            device=device,
        )
        self.selector = selector or KLDivergenceRegionSelector()
        self.cropper = cropper or IsotropicROICropper(target_size=448)
        self.propagator = propagator or AdaptiveScalePropagator(device=device)
        self.decoder = decoder or DynamicContrastiveDecoder(
            alpha_0=alpha_cd,
            beta_0=beta_apc,
            tau_js=tau_js,
        )
        self.feature_cache = feature_cache or default_feature_cache

    def extract_visual_features(self, image: Image.Image) -> torch.Tensor:
        """
        Extract or retrieve cached visual features for an image's global thumbnail (448x448)
        in torch.bfloat16, eliminating redundant ViT forward passes across multiple questions.
        """
        global_thumb = image.resize((448, 448), Image.Resampling.BICUBIC)
        return self.feature_cache.get_or_compute(
            model=self.model,
            transform=self.transform,
            tiles=[global_thumb],
            image=image,
            device=self.device,
        )

    def _decode_with_tiles(
        self,
        tiles: list,
        question: str,
        max_new_tokens: int = 64,
    ) -> str:
        """Run standard greedy decoding in bfloat16 on a list of 448x448 PIL Image tiles."""
        decoding_question = format_concise_decoding_prompt(question)

        pixel_values = torch.stack([self.transform(t) for t in tiles]).to(
            device=self.device, dtype=torch.bfloat16
        )
        gen_config = dict(max_new_tokens=max_new_tokens, do_sample=False)
        with torch.no_grad():
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                decoding_question,
                generation_config=gen_config,
            )
        del pixel_values
        return str(response).strip()

    def _decode_spatial_pmi_contrastive(
        self,
        roi_tile: Image.Image,
        global_thumb: Image.Image,
        question: str,
        stage1_kl_score: float = 1.0,
        max_new_tokens: int = 64,
        char_height_px: Optional[float] = None,
        magnification: float = 1.0,
        cached_thumb_vit: Optional[torch.Tensor] = None,
        context_tile: Optional[Image.Image] = None,
        roi_tiles: Optional[List[Image.Image]] = None,
        magnifications: Optional[List[float]] = None,
        char_heights: Optional[List[Optional[float]]] = None,
    ) -> Tuple[str, float, float]:
        """
        Stage 2: Dynamic Spatio-Temporal Calibrated Contrastive Decoding in torch.bfloat16.
        Contrasts the high-resolution evidence view [global_thumb, roi_pos, ...] against the
        low-pass salient-distractor prior [global_thumb, roi_lowpass, ...].
        Supports Top-K dual candidate tiles E_pos = [v_thumb, v_pos_1, v_pos_2] and
        E_neg = [v_thumb, v_neg_1, v_neg_2], Pre-RoI adaptive Laplacian edge re-emphasis,
        and two-phase delimiter reasoning.
        """
        if roi_tiles is None or len(roi_tiles) == 0:
            roi_tiles = [roi_tile]
            magnifications = [magnification]
            char_heights = [char_height_px]
        else:
            if magnifications is None:
                magnifications = [magnification] * len(roi_tiles)
            if char_heights is None:
                char_heights = [char_height_px] * len(roi_tiles)

        # For each crop:
        # Positive view gets adaptive Pre-RoI Laplacian edge re-emphasis
        # Negative view gets adaptive Gaussian lowpass blur
        pos_crops = []
        neg_crops = []
        for i, tile in enumerate(roi_tiles):
            mag_i = magnifications[i] if i < len(magnifications) else 1.0
            ch_i = char_heights[i] if i < len(char_heights) else None
            # 1. Positive enhanced crop (Group D optical blur resolution)
            pos_tile = apply_adaptive_laplacian_edge_reemphasis(tile, magnification=mag_i)
            pos_crops.append(pos_tile)
            # 2. Negative lowpass crop
            blur_radius = compute_adaptive_lowpass_radius(
                magnification=mag_i,
                char_height_px=ch_i,
            )
            neg_tile = tile.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            neg_crops.append(neg_tile)

        spatial_confidence = float(math.tanh(max(0.0, float(stage1_kl_score))))

        with torch.no_grad():
            # 1. Feature extraction with global_thumb reuse
            if cached_thumb_vit is not None:
                thumb_vit = cached_thumb_vit
            else:
                thumb_pv = self.transform(global_thumb).unsqueeze(0).to(
                    device=self.device, dtype=torch.bfloat16
                )
                thumb_vit = self.model.extract_feature(thumb_pv)
                del thumb_pv

            # Extract roi_tiles and lowpass tiles
            if len(roi_tiles) >= 2:
                # Top-2 candidate dual crops:
                # E_pos = [v_thumb, v_pos_1, v_pos_2]
                # E_neg = [v_thumb, v_neg_1, v_neg_2]
                roi_pv = torch.stack([
                    self.transform(pos_crops[0]),
                    self.transform(neg_crops[0]),
                    self.transform(pos_crops[1]),
                    self.transform(neg_crops[1]),
                ]).to(device=self.device, dtype=torch.bfloat16)
                roi_vit_all = self.model.extract_feature(roi_pv)
                del roi_pv

                pos_vit = torch.cat([thumb_vit, roi_vit_all[0:1], roi_vit_all[2:3]], dim=0)
                neg_vit = torch.cat([thumb_vit, roi_vit_all[1:2], roi_vit_all[3:4]], dim=0)
                del roi_vit_all
            elif context_tile is not None:
                roi_pair_pv = torch.stack([
                    self.transform(pos_crops[0]),
                    self.transform(neg_crops[0]),
                    self.transform(context_tile),
                ]).to(device=self.device, dtype=torch.bfloat16)
                roi_pair_vit = self.model.extract_feature(roi_pair_pv)
                del roi_pair_pv

                roi_vit = roi_pair_vit[0:1]
                roi_lowpass_vit = roi_pair_vit[1:2]
                ctx_vit = roi_pair_vit[2:3]

                # Multi-tile visual evidence: [global_thumb, roi_tile, context_tile]
                pos_vit = torch.cat([thumb_vit, roi_vit, ctx_vit], dim=0)
                neg_vit = torch.cat([thumb_vit, roi_lowpass_vit, ctx_vit], dim=0)
                del roi_pair_vit
            else:
                roi_pair_pv = torch.stack([
                    self.transform(pos_crops[0]),
                    self.transform(neg_crops[0]),
                ]).to(device=self.device, dtype=torch.bfloat16)
                roi_pair_vit = self.model.extract_feature(roi_pair_pv)
                del roi_pair_pv

                roi_vit = roi_pair_vit[0:1]
                roi_lowpass_vit = roi_pair_vit[1:2]

                # Dual-tile views [global_thumb, roi_tile] and [global_thumb, roi_lowpass]
                pos_vit = torch.cat([thumb_vit, roi_vit], dim=0)
                neg_vit = torch.cat([thumb_vit, roi_lowpass_vit], dim=0)
                del roi_pair_vit

            # Calibrate question prompt for concise VQA generation
            decoding_question = format_concise_decoding_prompt(question)

            # Build token embeddings for question
            pos_prefix, suffix_q, _, _ = self.extractor._build_prefix_and_suffix_embeds(
                pos_vit, decoding_question
            )
            neg_prefix, _, _, _ = self.extractor._build_prefix_and_suffix_embeds(
                neg_vit, decoding_question
            )

            pos_inputs = torch.cat([pos_prefix, suffix_q], dim=1)
            neg_inputs = torch.cat([neg_prefix, suffix_q], dim=1)

            eos_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
            embed_layer = self.model.language_model.get_input_embeddings()
            has_backbone = hasattr(self.model.language_model, "model") and hasattr(self.model.language_model, "lm_head")

            if has_backbone:
                pos_out = self.model.language_model.model(
                    inputs_embeds=pos_inputs,
                    use_cache=True,
                    return_dict=True,
                )
                pos_kv = pos_out.past_key_values
                pos_logits = self.model.language_model.lm_head(pos_out.last_hidden_state[:, -1:, :])[:, -1, :].float()
                del pos_out, pos_inputs

                neg_out = self.model.language_model.model(
                    inputs_embeds=neg_inputs,
                    use_cache=True,
                    return_dict=True,
                )
                neg_kv = neg_out.past_key_values
                neg_logits = self.model.language_model.lm_head(neg_out.last_hidden_state[:, -1:, :])[:, -1, :].float()
                del neg_out, neg_inputs
            else:
                pos_out = self.model.language_model(
                    inputs_embeds=pos_inputs,
                    use_cache=True,
                    return_dict=True,
                )
                neg_out = self.model.language_model(
                    inputs_embeds=neg_inputs,
                    use_cache=True,
                    return_dict=True,
                )
                pos_kv = pos_out.past_key_values
                neg_kv = neg_out.past_key_values
                pos_logits = pos_out.logits[:, -1, :].float()
                neg_logits = neg_out.logits[:, -1, :].float()
                del pos_out, neg_out, pos_inputs, neg_inputs

            generated_ids: List[int] = []
            alpha_history: List[float] = []
            entropy_history: List[float] = []
            phase_reasoning = False

            for step in range(max_new_tokens):
                step_spatial_conf = 0.0 if phase_reasoning else spatial_confidence
                next_token_id, alpha_t, entropy_t, _ = self.decoder.decode_step(
                    pos_logits=pos_logits,
                    neg_logits=neg_logits,
                    step=step,
                    spatial_confidence=step_spatial_conf,
                )

                alpha_history.append(alpha_t)
                entropy_history.append(entropy_t)

                if next_token_id == eos_id or next_token_id == self.tokenizer.eos_token_id:
                    break
                generated_ids.append(next_token_id)

                # Two-phase reasoning transition:
                # If delimiter (e.g. newline or colon) is emitted, switch off contrastive subtraction (alpha_t -> 0)
                # to allow unconstrained world-knowledge language reasoning for Group C resolution.
                if not phase_reasoning and len(generated_ids) >= 1:
                    token_str = self.tokenizer.decode([next_token_id], skip_special_tokens=True)
                    if "\n" in token_str or ":" in token_str:
                        phase_reasoning = True

                next_token_tensor = torch.tensor([[next_token_id]], device=self.device, dtype=torch.long)
                next_embed = embed_layer(next_token_tensor)

                if has_backbone:
                    pos_step = self.model.language_model.model(
                        inputs_embeds=next_embed,
                        past_key_values=pos_kv,
                        use_cache=True,
                        return_dict=True,
                    )
                    pos_kv = pos_step.past_key_values
                    pos_logits = self.model.language_model.lm_head(pos_step.last_hidden_state[:, -1:, :])[:, -1, :].float()

                    neg_step = self.model.language_model.model(
                        inputs_embeds=next_embed,
                        past_key_values=neg_kv,
                        use_cache=True,
                        return_dict=True,
                    )
                    neg_kv = neg_step.past_key_values
                    neg_logits = self.model.language_model.lm_head(neg_step.last_hidden_state[:, -1:, :])[:, -1, :].float()
                    del pos_step, neg_step
                else:
                    pos_out = self.model.language_model(
                        inputs_embeds=next_embed,
                        past_key_values=pos_kv,
                        use_cache=True,
                        return_dict=True,
                    )
                    neg_out = self.model.language_model(
                        inputs_embeds=next_embed,
                        past_key_values=neg_kv,
                        use_cache=True,
                        return_dict=True,
                    )
                    pos_kv = pos_out.past_key_values
                    neg_kv = neg_out.past_key_values
                    pos_logits = pos_out.logits[:, -1, :].float()
                    neg_logits = neg_out.logits[:, -1, :].float()
                    del pos_out, neg_out

        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        alpha_mean = float(sum(alpha_history) / max(1, len(alpha_history)))
        entropy_mean = float(sum(entropy_history) / max(1, len(entropy_history)))

        del (pos_vit, neg_vit, pos_prefix, neg_prefix, pos_kv, neg_kv)
        if cached_thumb_vit is None:
            del thumb_vit

        return text, alpha_mean, entropy_mean

    def run_stage1_only(
        self,
        image: Image.Image,
        question: str,
    ) -> Tuple[ExtractionResult, RegionSelectionResult, CropResult]:
        """Execute Stage 1 (Attention + Affinity + KL Region Selection + Seed Propagation + Isotropic Crop)."""
        extraction = self.extractor.extract(image, question)
        selection = self.selector.select(extraction.phi_map, image.size)
        stage1_bbox = list(selection.pixel_bbox)
        if self.propagator is not None:
            try:
                gx1, gy1, gx2, gy2 = selection.grid_bbox
                sub_phi = extraction.phi_map[gy1:gy2, gx1:gx2]
                if sub_phi.size > 0:
                    rel_y, rel_x = np.unravel_index(np.argmax(sub_phi), sub_phi.shape)
                    seed_patch = (int(gx1 + rel_x), int(gy1 + rel_y))
                else:
                    seed_patch = (int((gx1 + gx2) // 2), int((gy1 + gy2) // 2))

                prop_res = self.propagator.propagate(
                    image=image,
                    seed_patch=seed_patch,
                    grid_ratio=extraction.grid_ratio,
                )
                if prop_res is not None and prop_res.propagated_pixel_bbox is not None:
                    px1, py1, px2, py2 = prop_res.propagated_pixel_bbox
                    sx1, sy1, sx2, sy2 = selection.pixel_bbox
                    stage1_bbox = [
                        min(sx1, px1),
                        min(sy1, py1),
                        max(sx2, px2),
                        max(sy2, py2),
                    ]
            except Exception:
                stage1_bbox = list(selection.pixel_bbox)

        crop = self.cropper.crop(image, tuple(stage1_bbox))
        return extraction, selection, crop

    def _fuse_seed_propagation(
        self,
        image_size: Tuple[int, int],
        seed_bbox: Tuple[int, int, int, int],
        prop_bbox: Tuple[int, int, int, int],
    ) -> List[int]:
        """
        Information-Density Gated Fusion (Anti-Magnification Collapse M >= 1.0x).
        Clamps horizontal and vertical expansion around the high-density KL anchor core.
        """
        sx1, sy1, sx2, sy2 = seed_bbox
        px1, py1, px2, py2 = prop_bbox
        W_img, H_img = image_size
        max_dim = max(W_img, H_img)
        max_allowed_side = min(448, int(round(0.50 * max_dim)))

        w_seed = max(2, sx2 - sx1)
        h_seed = max(2, sy2 - sy1)

        exp_left = max(0, sx1 - px1)
        exp_right = max(0, px2 - sx2)
        total_exp_x = exp_left + exp_right
        if (w_seed + total_exp_x) > max_allowed_side and total_exp_x > 0:
            allowed_x = max(0, max_allowed_side - w_seed)
            scale_x = allowed_x / float(total_exp_x)
            fused_x1 = int(round(sx1 - exp_left * scale_x))
            fused_x2 = int(round(sx2 + exp_right * scale_x))
        else:
            fused_x1 = min(sx1, px1)
            fused_x2 = max(sx2, px2)

        exp_top = max(0, sy1 - py1)
        exp_bottom = max(0, py2 - sy2)
        total_exp_y = exp_top + exp_bottom
        if (h_seed + total_exp_y) > max_allowed_side and total_exp_y > 0:
            allowed_y = max(0, max_allowed_side - h_seed)
            scale_y = allowed_y / float(total_exp_y)
            fused_y1 = int(round(sy1 - exp_top * scale_y))
            fused_y2 = int(round(sy2 + exp_bottom * scale_y))
        else:
            fused_y1 = min(sy1, py1)
            fused_y2 = max(sy2, py2)

        pad_y_diacritic = max(int(round(0.15 * h_seed)), 2)
        fused_y1 = max(0, fused_y1 - pad_y_diacritic)
        fused_y2 = min(H_img, fused_y2 + pad_y_diacritic)

        fw = fused_x2 - fused_x1
        fh = fused_y2 - fused_y1
        if fw > 4.5 * max(1, fh):
            target_fw = int(round(4.5 * fh))
            if target_fw >= w_seed:
                diff_x = fw - target_fw
                fused_x1 = min(sx1, fused_x1 + diff_x // 2)
                fused_x2 = max(sx2, fused_x2 - (diff_x - diff_x // 2))
        elif fh > 4.5 * max(1, fw):
            target_fh = int(round(4.5 * fw))
            if target_fh >= h_seed:
                diff_y = fh - target_fh
                fused_y1 = min(sy1, fused_y1 + diff_y // 2)
                fused_y2 = max(sy2, fused_y2 - (diff_y - diff_y // 2))

        fused_x1 = max(0, min(W_img - 2, fused_x1))
        fused_y1 = max(0, min(H_img - 2, fused_y1))
        fused_x2 = max(fused_x1 + 2, min(W_img, fused_x2))
        fused_y2 = max(fused_y1 + 2, min(H_img, fused_y2))

        return [fused_x1, fused_y1, fused_x2, fused_y2]

    def run(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int = 64,
        cached_visual_features: Optional[torch.Tensor] = None,
    ) -> PipelineResult:
        """
        Execute full SG-VCD pipeline adhering strictly to PROJECT.md interface contract.
        Supports Top-K Candidate Grounding (K in {1, 2}) and Pre-RoI adaptive Laplacian edge re-emphasis.
        
        Args:
            image: PIL Image input
            question: Natural language question
            max_new_tokens: Maximum tokens to generate
            cached_visual_features: Optional cached ViT embeddings for global thumbnail
            
        Returns:
            PipelineResult with predictions, bboxes, metrics, and latency breakdown.
        """
        t0 = time.time()
        if cached_visual_features is None:
            cached_visual_features = self.extract_visual_features(image)

        # Stage 1.1: Information-Theoretic Attention Grounding
        t_start_ext = time.time()
        extraction = self.extractor.extract(image, question)
        t_ext = time.time() - t_start_ext

        # Stage 1.2: Scale-Invariant KL Region Selection (with High-Frequency Gradient Prior Guidance)
        t_start_sel = time.time()
        if hasattr(self.selector, "select_top_k"):
            multi_selection = self.selector.select_top_k(
                extraction.phi_map, image.size, image=image, max_k=2
            )
            selection = RegionSelectionResult(
                pixel_bbox=multi_selection.pixel_bboxes[0],
                grid_bbox=multi_selection.grid_bboxes[0],
                kl_score=multi_selection.kl_scores[0],
                captured_mass=multi_selection.captured_masses[0],
                area_fraction=multi_selection.area_fractions[0],
            )
        else:
            selection = self.selector.select(extraction.phi_map, image.size, image=image)
            multi_selection = MultiRegionSelectionResult(
                pixel_bboxes=[selection.pixel_bbox],
                grid_bboxes=[selection.grid_bbox],
                kl_scores=[selection.kl_score],
                captured_masses=[selection.captured_mass],
                area_fractions=[selection.area_fraction],
                k=1,
            )
        t_sel = time.time() - t_start_sel

        # Stage 1.3: Adaptive Scale Self-Calibration & Diacritic Belt Propagation (Refinement C)
        stage1_bboxes = [list(selection.pixel_bbox)]
        char_heights = [None]
        if self.propagator is not None:
            try:
                guided_phi = getattr(self.selector, "last_guided_phi", extraction.phi_map)
                phi_for_seeds = guided_phi if guided_phi is not None else extraction.phi_map
                seed_patches = []
                for k in range(multi_selection.k):
                    gx1, gy1, gx2, gy2 = multi_selection.grid_bboxes[k]
                    sub_phi = phi_for_seeds[gy1:gy2, gx1:gx2]
                    if sub_phi.size > 0:
                        rel_y, rel_x = np.unravel_index(np.argmax(sub_phi), sub_phi.shape)
                        seed_patch = (int(gx1 + rel_x), int(gy1 + rel_y))
                    else:
                        seed_patch = (int((gx1 + gx2) // 2), int((gy1 + gy2) // 2))
                    seed_patches.append(seed_patch)

                if hasattr(self.propagator, "propagate_multi"):
                    prop_results = self.propagator.propagate_multi(
                        image=image,
                        seed_patches=seed_patches,
                        grid_ratio=extraction.grid_ratio,
                    )
                else:
                    prop_results = [
                        self.propagator.propagate(image=image, seed_patch=sp, grid_ratio=extraction.grid_ratio)
                        for sp in seed_patches
                    ]

                fused_candidates = []
                fused_char_heights = []
                for k, prop_res in enumerate(prop_results):
                    sx1, sy1, sx2, sy2 = multi_selection.pixel_bboxes[k]
                    if prop_res is not None and prop_res.propagated_pixel_bbox is not None:
                        px1, py1, px2, py2 = prop_res.propagated_pixel_bbox
                        fused_box = self._fuse_seed_propagation(image.size, (sx1, sy1, sx2, sy2), (px1, py1, px2, py2))
                        fused_candidates.append(fused_box)
                        ch = getattr(prop_res.prototype, "char_height_px", None) if prop_res.prototype is not None else None
                        fused_char_heights.append(ch)
                    else:
                        fused_candidates.append([sx1, sy1, sx2, sy2])
                        fused_char_heights.append(None)

                # Post-propagation collision verification:
                # If IoU(B1, B2) >= 0.45, merge into K=1 (B1 union B2); otherwise keep K=2
                if len(fused_candidates) >= 2:
                    resolved = verify_and_resolve_candidate_boxes(fused_candidates, iou_threshold=0.45)
                    if len(resolved) == 1:
                        stage1_bboxes = resolved
                        char_heights = [fused_char_heights[0]]
                    else:
                        stage1_bboxes = fused_candidates
                        char_heights = fused_char_heights
                else:
                    stage1_bboxes = fused_candidates
                    char_heights = fused_char_heights
            except Exception:
                stage1_bboxes = [list(selection.pixel_bbox)]
                char_heights = [None]

        # Stage 1.4: Multi-Scale Isotropic Crop (Guarded against magnification collapse M < 1.0x)
        t_start_crop = time.time()
        W_img, H_img = image.size
        crops = []
        for b in stage1_bboxes:
            s_prop = max(b[2] - b[0], b[3] - b[1])
            if s_prop > 448 or s_prop > 0.50 * max(W_img, H_img):
                focal_crop_bbox = tuple(selection.pixel_bbox)
            else:
                focal_crop_bbox = tuple(b)
            c = self.cropper.crop(image, focal_crop_bbox)
            crops.append(c)

        crop = crops[0]
        stage1_bbox = stage1_bboxes[0]
        char_height_px = char_heights[0]
        t_crop = time.time() - t_start_crop

        global_thumb = image.resize((448, 448), Image.Resampling.BICUBIC)

        # Stage 2: Dynamic Spatio-Temporal Contrastive Decoding
        t_start_dec = time.time()
        context_tile = None
        if len(crops) == 1 and getattr(self, "max_visual_tiles", 2) >= 3 and hasattr(self.cropper, "get_multi_tiles"):
            try:
                multi_tiles, _ = self.cropper.get_multi_tiles(image, tuple(stage1_bbox), max_tiles=self.max_visual_tiles)
                if len(multi_tiles) >= 3:
                    context_tile = multi_tiles[2]
            except Exception:
                context_tile = None

        pred_sg_vcd, alpha_mean, entropy_mean = self._decode_spatial_pmi_contrastive(
            roi_tile=crop.roi_tile,
            global_thumb=global_thumb,
            question=question,
            stage1_kl_score=selection.kl_score,
            max_new_tokens=max_new_tokens,
            char_height_px=char_height_px,
            magnification=crop.magnification,
            cached_thumb_vit=cached_visual_features,
            context_tile=context_tile,
            roi_tiles=[c.roi_tile for c in crops],
            magnifications=[c.magnification for c in crops],
            char_heights=char_heights,
        )
        t_dec = time.time() - t_start_dec

        pred_roi_only = ""
        pred_roi_plus_global = ""
        if self.decode_auxiliary_branches:
            pred_roi_only = self._decode_with_tiles(
                [crop.roi_tile], question, max_new_tokens=max_new_tokens
            )
            pred_roi_plus_global = self._decode_with_tiles(
                [crop.roi_tile, global_thumb], question, max_new_tokens=max_new_tokens
            )

        t_total = time.time() - t0

        timing = {
            "stage1_extraction": round(t_ext, 3),
            "stage1_selection": round(t_sel, 3),
            "stage1_crop": round(t_crop, 3),
            "stage2_decoding": round(t_dec, 3),
            "total": round(t_total, 3),
        }

        pred_text = strip_conversational_preamble(pred_sg_vcd, question=question)

        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return PipelineResult(
            pred_text=pred_text,
            pred_raw_text=pred_sg_vcd,
            stage1_bbox=list(stage1_bbox),
            spatial_confidence=float(math.tanh(max(0.0, float(selection.kl_score)))),
            vcd_alpha_mean=round(alpha_mean, 4),
            entropy_mean=round(entropy_mean, 4),
            timing=timing,
            extraction=extraction,
            selection=selection,
            crop=crop,
            pred_roi_only=pred_roi_only,
            pred_roi_plus_global=pred_roi_plus_global,
            pred_sg_vcd=pred_sg_vcd,
            stage1_bboxes=stage1_bboxes,
            multi_selection=multi_selection,
            multi_crops=crops,
        )
