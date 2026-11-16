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
import inspect
import math
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
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
    _is_valid_working_maps,
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
    merge_aligned_candidates,
)
from .dynamic_cd import (
    DynamicContrastiveDecoder,
    compute_adaptive_lowpass_radius,
)
from .feature_cache import (
    ImageCacheIdentity,
    VisualFeatureCache,
    build_image_cache_identity,
    default_feature_cache,
    resolve_precision_and_dtype,
)
from .roi_cropper import (
    BaseROICropper,
    CropResult,
    DynamicMultiTileROICropper,
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


class SyntacticSpanRefiner:
    r"""
    Principled Syntactic & Information-Theoretic Answer Span Refiner.
    Formulation:
        \hat{y}_{entity} = \arg\max_{s \sqsubseteq y_{pred}} \mathcal{S}_{info}(s | Q, I_{ROI})
    Preserves generation fidelity: eliminates conversational preambles, question-echo
    prefixes before copulas, and trailing punctuation without destructive token-stripping.
    """

    def __init__(self):
        # Structural colonic delimiter: partitions structural labels (Schema : Instance)
        # Negative lookahead/lookbehind strictly preserves numeric times (10:30) and ratios (2:1).
        self.colonic_pattern = re.compile(r"(?<!\d)[:|](?!\d)")
        # Trailing sentence terminator and whitespace pattern
        self.trailing_punct_pattern = re.compile(r"[\.,;:!\?\s…]+$")
        # Unicode quotation marks
        self.quote_chars = ('"', "'", '“', '‘', '«', '”', '’', '»')
        # Guarded multiline selection header pattern
        self.answer_header_pattern = re.compile(
            r"^(?:đáp\s+án|câu\s+trả\s+lời|kết\s+luận|trả\s+lời)\s*[:,\-]?\s*",
            flags=re.IGNORECASE | re.UNICODE,
        )
        # Conversational introductory clause patterns
        self.intro_patterns = [
            r"^(?:yêu\s+cầu|cụm\s+từ|trả\s+lời(?:\s+ngắn\s+gọn)?)\s*[:,\-]?\s*",
            r"^(?:bức\s+)?(?:hình\s+ảnh|ảnh|hình)(?:\s+này)?\s+(?:được\s+chụp\s+(?:tại|vào\s+dịp)|được\s+chụp|chụp\s+tại|có\s+nguồn\s+(?:từ|gốc\s+từ)|lấy\s+nguồn\s+từ|nguồn\s+từ|chụp|cho\s+thấy|ghi\s+lại|có\s+chữ|chỉ\s+ra|là|có)\s*[:,\-]?\s*",
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
        # Metadiscourse introductory clause pattern (image / answer modality framing)
        self.metadiscourse_pattern = re.compile(
            r"^(?:bức\s+)?(?:ảnh|hình\s+ảnh|hình)(?:\s+này)?\s+(?:được\s+chụp\s+(?:tại|vào\s+dịp)|được\s+chụp|chụp\s+tại|có\s+nguồn\s+(?:từ|gốc\s+từ)|lấy\s+nguồn\s+từ|nguồn\s+từ|chụp|cho\s+thấy|ghi\s+lại|có\s+chữ|chỉ\s+ra|là|có)\s*[:,\-]?\s*"
            r"|^(?:câu\s+trả\s+lời|đáp\s+án|kết\s+quả)(?:\s+(?:của\s+tôi|chính\s+xác|cho\s+câu\s+hỏi))?\s*(?:là\s*[:,\-]?|[:,\-])\s*"
            r"|^(?:theo\s+(?:thông\s+tin\s+trong\s+)?(?:bức\s+)?(?:hình\s+ảnh|ảnh|hình)(?:[:,\s]*)?)\s*"
            r"|^(?:có\s+thể\s+thấy\s+(?:rằng)?)\s*[:,\-]?\s*",
            flags=re.IGNORECASE | re.UNICODE,
        )
        self.copula_pattern = re.compile(
            r"\s+(?:đang\s+diễn\s+ra\s+trên\s+kênh|diễn\s+ra\s+trên\s+kênh|được\s+diễn\s+ra\s+tại|diễn\s+ra\s+tại|"
            r"được\s+chụp\s+vào\s+dịp|tên\s+là|có\s+tên\s+là|có\s+tên|tên|chuyên\s+về|chuyên|thuộc\s+về|"
            r"lấy\s+nguồn\s+từ|nguồn\s+từ|là|được\s+gọi\s+là|bán\s+hàng|bán|có)\s+",
            flags=re.IGNORECASE | re.UNICODE,
        )
        self.classifier_patterns = [
            r"^(?:nhà\s+thuốc|hiệu\s+thuốc|cửa\s+hàng|tiệm\s+cà\s+phê|tiệm|quán\s+cà\s+phê|quán\s+ăn|quán|công\s+ty(?:\s+tnhh|\s+cổ\s+phần)?|doanh\s+nghiệp|trường|phòng(?:\s+khám|\s+giao\s+dịch)?|sở|bệnh\s+viện|khách\s+sạn|ngân\s+hàng)\s+",
        ]

    def refine(self, text: Optional[str], question: Optional[str] = None) -> str:
        if not text:
            return ""

        # Step 0: Unicode NFC normalization & initial paired quote peeling
        s = unicodedata.normalize("NFC", str(text)).strip()
        if not s:
            return ""

        while len(s) >= 2 and s[0] in self.quote_chars and s[-1] in self.quote_chars:
            s = s[1:-1].strip()

        # Step 1: Guarded multiline selection (only take terminal line if explicitly beginning with standard answer header)
        lines = [line.strip() for line in s.split("\n") if line.strip()]
        if len(lines) > 1:
            for line in reversed(lines):
                m_header = self.answer_header_pattern.match(line)
                if m_header:
                    extracted = line[m_header.end():].strip()
                    if extracted:
                        s = extracted
                    break

        # Step 2: Structural Colonic & Punctuation Disentanglement Operator (SCPDO)
        colonic_matches = list(self.colonic_pattern.finditer(s))
        if colonic_matches:
            last_delim = colonic_matches[-1]
            right_payload = s[last_delim.end():].strip()
            if right_payload:
                s = right_payload

        # Step 3: Quoted focal entity extraction (e.g. Trong bức ảnh có chữ 'Siêu Thị Big C')
        if any(q in s for q in self.quote_chars):
            for q_open, q_close in (('“', '”'), ('«', '»'), ('"', '"'), ('‘', '’'), ("'", "'")):
                if q_open in s and q_close in s:
                    if q_open == "'" and q_close == "'":
                        m_quote = re.search(
                            r"(?:^|[^\w])'((?:[^\'\n]|(?<=\w)'(?=\w))+?)'(?:[^\w]|$)",
                            s,
                            flags=re.UNICODE,
                        )
                    elif q_open == '‘' and q_close == '’':
                        m_quote = re.search(
                            r"(?:^|[^\w])‘((?:[^‘’\n]|(?<=\w)[’'](?=\w))+?)’(?:[^\w]|$)",
                            s,
                            flags=re.UNICODE,
                        )
                    else:
                        m_quote = re.search(
                            rf'(?:^|[^\w]){re.escape(q_open)}([^{re.escape(q_open)}{re.escape(q_close)}\n]+?){re.escape(q_close)}(?:[^\w]|$)',
                            s,
                            flags=re.UNICODE,
                        )
                    if m_quote:
                        quoted_span = m_quote.group(1).strip()
                        if quoted_span:
                            s = quoted_span
                            break

        # Step 4: Iteratively strip prompt instructions, outer quotes, and introductory clauses
        for _ in range(8):
            prev_len = len(s)

            # Strip prompt-instruction echo if present
            s = re.sub(
                r"^(?:yêu\s+cầu|cụm\s+từ|trả\s+lời(?:\s+ngắn\s+gọn)?|(?:chỉ\s+)?(?:ghi|trả\s+lời|nêu))(?:\s+thật|\s+đúng|\s+ngắn\s+gọn|\s+từ|\s+hoặc|\s+cụm\s+từ|\s+đáp\s+án)*\s*[:,\-]?\s*",
                "",
                s,
                flags=re.IGNORECASE | re.UNICODE,
            ).strip()

            # Strip enclosing quotation marks
            if len(s) >= 2 and ((s[0] in ('"', "'", '“', '‘', '«') and s[-1] in ('"', "'", '”', '’', '»'))):
                s = s[1:-1].strip()

            for pattern in self.intro_patterns:
                m = re.match(pattern, s, flags=re.IGNORECASE | re.UNICODE)
                if m:
                    candidate = s[m.end():].strip()
                    if candidate:
                        s = candidate
                    break

            # Metadiscourse pattern peeling
            m_meta = self.metadiscourse_pattern.match(s)
            if m_meta:
                cand = s[m_meta.end():].strip()
                if cand:
                    s = cand

            # Also strip single quotes or boundary quotation marks if left
            s = s.strip("\"'“”‘’«»").strip()

            if len(s) == prev_len:
                break

        # Step 5: Question-echo subject framing before copula / relation verbs (Round 5 verified)
        if question and s:
            q_words = set(re.findall(r"\w+", question.lower(), flags=re.UNICODE))
            copula_match = self.copula_pattern.search(s)
            if copula_match:
                prefix_part = s[:copula_match.start()].strip()
                p_words = set(re.findall(r"\w+", prefix_part.lower(), flags=re.UNICODE))
                candidate_entity = s[copula_match.end():].strip()
                if p_words and len(p_words & q_words) / len(p_words) >= 0.5 and candidate_entity:
                    s = candidate_entity

        # Step 6: Vietnamese organizational classifier prefix peeling for name queries (Round 5 verified)
        if question and s:
            q_lower = question.lower()
            is_name_query = any(k in q_lower for k in ["tên", "gọi là", "hiệu gì", "nhãn hiệu", "thương hiệu"])
            for cp in self.classifier_patterns:
                m = re.match(cp, s, flags=re.IGNORECASE | re.UNICODE)
                if m:
                    matched_prefix = m.group(0).strip().lower()
                    rem = s[m.end():].strip()
                    if rem and is_name_query:
                        if matched_prefix in q_lower or any(w in q_lower for w in matched_prefix.split()):
                            s = rem
                            break

        # Step 7: Terminal punctuation and residual quotes cleanup
        s = self.trailing_punct_pattern.sub("", s).strip()
        while len(s) >= 2 and s[0] in self.quote_chars and s[-1] in self.quote_chars:
            s = s[1:-1].strip()
        s = s.strip("\"'“”‘’«»").strip()

        return s if s else unicodedata.normalize("NFC", str(text)).strip()


_default_refiner = SyntacticSpanRefiner()


def strip_conversational_preamble(text: Optional[str], question: Optional[str] = None) -> str:
    """Drop-in zero-heuristic replacement for legacy strip_conversational_preamble using SyntacticSpanRefiner."""
    return _default_refiner.refine(text, question=question)


def format_concise_decoding_prompt(question: str) -> str:
    """
    Format recency-anchored prompt without dangling <image> tags.
    Anchors brevity constraints at the tail to prevent question echoing and prompt dilution.
    """
    q_clean = question.strip()
    if q_clean.startswith("<image>"):
        q_clean = q_clean[len("<image>"):].strip()
    return f"{q_clean}\nTrả lời ngắn gọn: "


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
        scoring_mode: str = "canonical_signed",
        precision: Optional[str] = None,
        torch_dtype: Optional["torch.dtype"] = None,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.transform = transform
        self.device = device
        self.precision, self.torch_dtype = resolve_precision_and_dtype(
            precision=precision, dtype=torch_dtype
        )
        self.alpha_cd = alpha_cd
        self.beta_apc = beta_apc
        self.decode_auxiliary_branches = decode_auxiliary_branches
        self.max_visual_tiles = int(max_visual_tiles)
        self.scoring_mode = scoring_mode

        self.extractor = extractor or PointwiseKLAttentionExtractor(
            model=model,
            tokenizer=tokenizer,
            transform=transform,
            layer_start=8,
            layer_end=16,
            precision=self.precision,
            torch_dtype=self.torch_dtype,
            device=device,
        )
        self.selector = selector or KLDivergenceRegionSelector()
        self.cropper = cropper or DynamicMultiTileROICropper(target_size=448)
        self.propagator = propagator or AdaptiveScalePropagator(device=device)
        self.decoder = decoder or DynamicContrastiveDecoder(
            alpha_0=alpha_cd,
            beta_0=beta_apc,
            tau_js=tau_js,
            scoring_mode=scoring_mode,
        )
        self.feature_cache = (
            feature_cache if feature_cache is not None else VisualFeatureCache(max_entries=16)
        )
        self._cached_working_maps: Optional[Tuple[Any, ...]] = None
        self._cached_working_maps_key: Optional[ImageCacheIdentity] = None

    def reset_cache(self) -> None:
        """
        Clear all image-level caches held by this pipeline instance (VisualFeatureCache,
        attention extractor state, and Stage 1 working maps) so starting a fresh evaluation
        run or transitioning across image boundaries cannot retain stale references
        (OPT-06 / OPT-03 / OPT-02). Does not invoke unconditional `torch.cuda.empty_cache()`.
        """
        self._cached_working_maps = None
        self._cached_working_maps_key = None
        if self.feature_cache is not None:
            cache_dict = getattr(self.feature_cache, "_cache", None)
            if isinstance(cache_dict, dict):
                for k in list(cache_dict.keys()):
                    tensor = cache_dict.pop(k, None)
                    del tensor
                cache_dict.clear()
            elif hasattr(self.feature_cache, "clear"):
                self.feature_cache.clear()
        if hasattr(self.extractor, "clear_cache"):
            self.extractor.clear_cache()
        elif hasattr(self.extractor, "_cached_img_state"):
            self.extractor._cached_img_state = None
            self.extractor._cached_img_key = None

    def extract_visual_features(
        self,
        image: Image.Image,
        *,
        image_key: Optional[Union[ImageCacheIdentity, Tuple[Any, ...], str]] = None,
        image_id: Optional[Union[str, int]] = None,
        image_path: Optional[str] = None,
        dataset_root: Optional[str] = None,
        dataset_version: Optional[str] = None,
        dataset_namespace: Optional[str] = None,
        content_fingerprint: Optional[str] = None,
    ) -> torch.Tensor:
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
            dtype=self.torch_dtype,
            image_key=image_key,
            image_id=image_id,
            image_path=image_path,
            dataset_root=dataset_root,
            dataset_version=dataset_version,
            dataset_namespace=dataset_namespace,
            content_fingerprint=content_fingerprint,
        )

    def _decode_with_tiles(
        self,
        tiles: list,
        question: str,
        max_new_tokens: int = 64,
    ) -> str:
        """Run standard greedy decoding in configured precision on a list of 448x448 PIL Image tiles."""
        decoding_question = format_concise_decoding_prompt(question)

        pixel_values = torch.stack([self.transform(t) for t in tiles]).to(
            device=self.device, dtype=self.torch_dtype
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

    @staticmethod
    def _validate_dual_branch_inputs(
        pos_inputs: Any,
        neg_inputs: Any,
    ) -> Tuple[int, int]:
        """
        Validate that positive and negative branch input embeddings satisfy the
        strict B=2 stacking contract (3D tensors of shape (1, L, D) with identical
        sequence length, hidden dimension, dtype, and device). Never pads or truncates.
        """
        if not isinstance(pos_inputs, torch.Tensor) or not isinstance(neg_inputs, torch.Tensor):
            raise ValueError(
                f"pos_inputs and neg_inputs must both be torch.Tensor, "
                f"got {type(pos_inputs).__name__} and {type(neg_inputs).__name__}."
            )
        if pos_inputs.ndim != 3 or neg_inputs.ndim != 3:
            raise ValueError(
                f"pos_inputs and neg_inputs must be 3D tensors (B, L, D), "
                f"got shapes {tuple(pos_inputs.shape)} and {tuple(neg_inputs.shape)}."
            )
        if pos_inputs.shape[0] != 1 or neg_inputs.shape[0] != 1:
            raise ValueError(
                f"pos_inputs and neg_inputs must each have batch size 1 before B=2 stacking, "
                f"got batch sizes {pos_inputs.shape[0]} and {neg_inputs.shape[0]}."
            )
        if pos_inputs.shape[1] <= 0 or pos_inputs.shape[2] <= 0:
            raise ValueError(
                f"pos_inputs must have positive sequence length and hidden dim, "
                f"got shape {tuple(pos_inputs.shape)}."
            )
        if pos_inputs.shape != neg_inputs.shape:
            raise ValueError(
                f"Cannot batch positive and negative branches with mismatched shapes: "
                f"pos_inputs={tuple(pos_inputs.shape)} vs neg_inputs={tuple(neg_inputs.shape)}. "
                f"Silent padding or truncation is prohibited (OPT-01)."
            )
        if pos_inputs.dtype != neg_inputs.dtype:
            raise ValueError(
                f"pos_inputs and neg_inputs must have identical dtype, "
                f"got {pos_inputs.dtype} vs {neg_inputs.dtype}."
            )
        if pos_inputs.device != neg_inputs.device:
            raise ValueError(
                f"pos_inputs and neg_inputs must be on the same device, "
                f"got {pos_inputs.device} vs {neg_inputs.device}."
            )
        return int(pos_inputs.shape[1]), int(pos_inputs.shape[2])

    @staticmethod
    def _extract_kv_cache_tensors(
        past_kv: Any,
    ) -> List[Tuple["torch.Tensor", "torch.Tensor"]]:
        """
        Extract per-layer (key, value) tensor pairs from supported KV-cache formats:
        1. Hugging Face DynamicCache / Cache with `.layers` (layer.keys, layer.values)
        2. Hugging Face DynamicCache with `.key_cache` and `.value_cache`
        3. Legacy tuple/list of (key, value) tensor pairs
        """
        if past_kv is None:
            raise ValueError("past_key_values is None; expected initialized KV cache.")

        # Format 1: HF DynamicCache / Cache with .layers list
        layers_attr = getattr(past_kv, "layers", None)
        if isinstance(layers_attr, list):
            if len(layers_attr) == 0:
                raise ValueError("DynamicCache.layers is empty; KV cache has no populated layers.")
            pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
            for idx, layer in enumerate(layers_attr):
                k = getattr(layer, "keys", None)
                v = getattr(layer, "values", None)
                if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
                    raise ValueError(
                        f"Cache layer {idx} does not contain valid key/value tensors."
                    )
                pairs.append((k, v))
            return pairs

        # Format 2: HF DynamicCache with .key_cache and .value_cache lists
        key_cache = getattr(past_kv, "key_cache", None)
        value_cache = getattr(past_kv, "value_cache", None)
        if isinstance(key_cache, list) and isinstance(value_cache, list):
            if len(key_cache) == 0 or len(key_cache) != len(value_cache):
                raise ValueError(
                    f"Invalid key_cache/value_cache lengths: {len(key_cache)} vs {len(value_cache)}."
                )
            pairs = []
            for idx, (k, v) in enumerate(zip(key_cache, value_cache)):
                if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
                    raise ValueError(
                        f"key_cache/value_cache layer {idx} does not contain valid tensors."
                    )
                pairs.append((k, v))
            return pairs

        # Format 3: Legacy tuple/list of (K, V) tensor pairs
        if isinstance(past_kv, (tuple, list)):
            if len(past_kv) == 0:
                raise ValueError("Tuple/list past_key_values is empty.")
            pairs = []
            for idx, layer_kv in enumerate(past_kv):
                if not isinstance(layer_kv, (tuple, list)) or len(layer_kv) < 2:
                    raise ValueError(
                        f"Legacy cache layer {idx} must be a (key, value) pair."
                    )
                k, v = layer_kv[0], layer_kv[1]
                if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
                    raise ValueError(
                        f"Legacy cache layer {idx} elements must be torch.Tensors."
                    )
                pairs.append((k, v))
            return pairs

        raise ValueError(
            f"Unsupported KV cache representation: {type(past_kv).__name__}."
        )

    @classmethod
    def _validate_kv_cache(
        cls,
        past_kv: Any,
        *,
        expected_batch_size: int,
        expected_seq_len: Optional[int] = None,
    ) -> int:
        """
        Verify that every layer of `past_kv` retains the expected batch dimension and
        consistent sequence length across all key and value tensors.
        """
        pairs = cls._extract_kv_cache_tensors(past_kv)
        observed_seq_len: Optional[int] = None
        for idx, (k, v) in enumerate(pairs):
            if k.ndim != 4 or v.ndim != 4:
                raise ValueError(
                    f"KV cache layer {idx} tensors must be 4D (B, H, S, D_h), "
                    f"got key={tuple(k.shape)}, value={tuple(v.shape)}."
                )
            if k.shape != v.shape:
                raise ValueError(
                    f"KV cache layer {idx} key/value shape mismatch: "
                    f"key={tuple(k.shape)} vs value={tuple(v.shape)}."
                )
            if k.dtype != v.dtype:
                raise ValueError(
                    f"KV cache layer {idx} key/value dtype mismatch: "
                    f"key={k.dtype} vs value={v.dtype}."
                )
            if k.shape[0] != expected_batch_size:
                raise ValueError(
                    f"KV cache layer {idx} batch dimension mismatch: "
                    f"expected B={expected_batch_size}, got {k.shape[0]}."
                )
            if k.shape[1] <= 0 or k.shape[2] <= 0 or k.shape[3] <= 0:
                raise ValueError(
                    f"KV cache layer {idx} has non-positive dimensions: {tuple(k.shape)}."
                )
            layer_seq_len = int(k.shape[2])
            if observed_seq_len is None:
                observed_seq_len = layer_seq_len
            elif layer_seq_len != observed_seq_len:
                raise ValueError(
                    f"KV cache sequence length mismatch across layers: "
                    f"layer 0 has {observed_seq_len}, layer {idx} has {layer_seq_len}."
                )

        assert observed_seq_len is not None
        if expected_seq_len is not None and observed_seq_len != expected_seq_len:
            raise ValueError(
                f"KV cache sequence length mismatch: expected {expected_seq_len}, "
                f"got {observed_seq_len}."
            )
        return observed_seq_len

    @staticmethod
    def _prepare_optional_mask_and_positions(
        seq_len: int,
        device: Any,
        attention_mask: Optional["torch.Tensor"] = None,
        position_ids: Optional["torch.Tensor"] = None,
    ) -> Tuple[
        Optional["torch.Tensor"],
        Optional["torch.Tensor"],
        Optional["torch.Tensor"],
        Optional["torch.Tensor"],
        Optional["torch.Tensor"],
        Optional["torch.Tensor"],
    ]:
        """
        Validate and format optional `attention_mask` and `position_ids` for both
        B=2 batched execution and B=1 sequential reference execution.
        Disallows asymmetric branch masks or divergent position IDs.
        """
        bat_mask = pos_mask = neg_mask = None
        if attention_mask is not None:
            if not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 2:
                raise ValueError(
                    "attention_mask must be a 2D torch.Tensor of shape (1, L) or (2, L)."
                )
            if attention_mask.shape[1] != seq_len:
                raise ValueError(
                    f"attention_mask sequence length {attention_mask.shape[1]} "
                    f"does not match input sequence length {seq_len}."
                )
            if attention_mask.shape[0] == 1:
                pos_mask = attention_mask.to(device=device)
                neg_mask = pos_mask.clone()
                bat_mask = torch.cat([pos_mask, neg_mask], dim=0)
            elif attention_mask.shape[0] == 2:
                if not torch.equal(attention_mask[0], attention_mask[1]):
                    raise ValueError(
                        "Asymmetric attention_mask between positive and negative branches "
                        "is unsupported; both branches must share identical mask geometry."
                    )
                bat_mask = attention_mask.to(device=device)
                pos_mask = bat_mask[0:1].clone()
                neg_mask = bat_mask[1:2].clone()
            else:
                raise ValueError(
                    f"attention_mask batch dimension must be 1 or 2, got {attention_mask.shape[0]}."
                )

        bat_pos = pos_pos = neg_pos = None
        if position_ids is not None:
            if not isinstance(position_ids, torch.Tensor) or position_ids.ndim != 2:
                raise ValueError(
                    "position_ids must be a 2D torch.Tensor of shape (1, L) or (2, L)."
                )
            if position_ids.shape[1] != seq_len:
                raise ValueError(
                    f"position_ids sequence length {position_ids.shape[1]} "
                    f"does not match input sequence length {seq_len}."
                )
            if position_ids.shape[0] == 1:
                pos_pos = position_ids.to(device=device, dtype=torch.long)
                neg_pos = pos_pos.clone()
                bat_pos = torch.cat([pos_pos, neg_pos], dim=0)
            elif position_ids.shape[0] == 2:
                if not torch.equal(position_ids[0], position_ids[1]):
                    raise ValueError(
                        "Divergent position_ids between positive and negative branches "
                        "are unsupported; both branches must share identical positions."
                    )
                bat_pos = position_ids.to(device=device, dtype=torch.long)
                pos_pos = bat_pos[0:1].clone()
                neg_pos = bat_pos[1:2].clone()
            else:
                raise ValueError(
                    f"position_ids batch dimension must be 1 or 2, got {position_ids.shape[0]}."
                )

        return bat_mask, pos_mask, neg_mask, bat_pos, pos_pos, neg_pos

    def _record_decode_step_trace(
        self,
        step_trace: List[Dict[str, Any]],
        *,
        step: int,
        pos_logits: "torch.Tensor",
        neg_logits: "torch.Tensor",
        alpha_t: float,
        entropy_t: float,
        d_js_val: float,
        next_token_id: int,
        cache_seq_len: int,
    ) -> None:
        """Record per-step float32 logits, contrastive scores, selected token, and cache state for parity tests."""
        p_pos = F.softmax(pos_logits, dim=-1)
        beta_t = self.decoder.beta_0 * math.exp(-min(4.0, max(0.0, float(entropy_t))))
        max_p_pos = p_pos.max(dim=-1, keepdim=True).values
        apc_mask = p_pos >= (beta_t * max_p_pos)

        delta_z = torch.relu(pos_logits - neg_logits)
        max_delta_z = delta_z.max(dim=-1, keepdim=True).values
        has_contrast = max_delta_z > 1e-6
        contrast_mask = (delta_z >= (self.decoder.theta_c * max_delta_z)) & has_contrast
        valid_mask = apc_mask | contrast_mask

        t_entropy = max(
            0.05, 1.0 - self.decoder.eta_temp * math.tanh(float(entropy_t) / self.decoder.tau_h)
        )
        if getattr(self.decoder, "scoring_mode", "canonical_signed") == "canonical_signed":
            raw_score = (1.0 + float(alpha_t)) * pos_logits - float(alpha_t) * neg_logits
        else:
            raw_score = pos_logits + float(alpha_t) * delta_z
        pre_mask_logits = raw_score / t_entropy
        signed_logits = (1.0 + float(alpha_t)) * pos_logits - float(alpha_t) * neg_logits
        final_logits = pre_mask_logits.masked_fill(~valid_mask, float("-inf"))

        step_trace.append(
            {
                "step": int(step),
                "pos_logits": pos_logits.detach().cpu().clone(),
                "neg_logits": neg_logits.detach().cpu().clone(),
                "contrastive_logits_pre_mask": pre_mask_logits.detach().cpu().clone(),
                "contrastive_logits_signed": signed_logits.detach().cpu().clone(),
                "contrastive_logits_final": final_logits.detach().cpu().clone(),
                "next_token_id": int(next_token_id),
                "alpha_t": float(alpha_t),
                "entropy_t": float(entropy_t),
                "d_js_val": float(d_js_val),
                "cache_seq_len": int(cache_seq_len),
            }
        )

    def _run_batched_dual_branch_decode(
        self,
        pos_inputs: "torch.Tensor",
        neg_inputs: "torch.Tensor",
        spatial_confidence: float,
        max_new_tokens: int = 64,
        *,
        attention_mask: Optional["torch.Tensor"] = None,
        position_ids: Optional["torch.Tensor"] = None,
        step_trace: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, float, float]:
        """
        TICKET-OPT-01: Batched Positive/Negative Decoding (B=2).
        Combines positive (batch index 0) and negative (batch index 1) branches into
        a single batch of size 2 for both prefill and each autoregressive decoding step.
        """
        prefill_seq_len, _ = self._validate_dual_branch_inputs(pos_inputs, neg_inputs)
        bat_mask, _, _, bat_pos, _, _ = self._prepare_optional_mask_and_positions(
            prefill_seq_len,
            pos_inputs.device,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        eos_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        embed_layer = self.model.language_model.get_input_embeddings()
        has_backbone = hasattr(self.model.language_model, "model") and hasattr(
            self.model.language_model, "lm_head"
        )

        with torch.no_grad():
            # Batch index 0: positive branch; Batch index 1: negative branch
            batched_inputs = torch.cat([pos_inputs, neg_inputs], dim=0)
            prefill_kwargs: Dict[str, Any] = {
                "inputs_embeds": batched_inputs,
                "use_cache": True,
                "return_dict": True,
            }
            if bat_mask is not None:
                prefill_kwargs["attention_mask"] = bat_mask
            if bat_pos is not None:
                prefill_kwargs["position_ids"] = bat_pos

            if has_backbone:
                prefill_out = self.model.language_model.model(**prefill_kwargs)
                batch_kv = prefill_out.past_key_values
                last_hidden = getattr(prefill_out, "last_hidden_state", None)
                if (
                    not isinstance(last_hidden, torch.Tensor)
                    or last_hidden.ndim != 3
                    or last_hidden.shape[0] != 2
                    or last_hidden.shape[1] != prefill_seq_len
                ):
                    raise ValueError(
                        f"Batched prefill last_hidden_state must have shape (2, {prefill_seq_len}, D), "
                        f"got {None if last_hidden is None else tuple(last_hidden.shape)}."
                    )
                cache_seq_len = self._validate_kv_cache(
                    batch_kv, expected_batch_size=2, expected_seq_len=prefill_seq_len
                )
                logits = self.model.language_model.lm_head(last_hidden[:, -1:, :])[:, -1, :]
                if logits.ndim != 2 or logits.shape[0] != 2:
                    raise ValueError(
                        f"Batched prefill lm_head output must have shape (2, V), got {tuple(logits.shape)}."
                    )
                pos_logits = logits[0:1].float()
                neg_logits = logits[1:2].float()
                del prefill_out, last_hidden, logits, batched_inputs
            else:
                prefill_out = self.model.language_model(**prefill_kwargs)
                batch_kv = prefill_out.past_key_values
                cache_seq_len = self._validate_kv_cache(
                    batch_kv, expected_batch_size=2, expected_seq_len=prefill_seq_len
                )
                out_logits = getattr(prefill_out, "logits", None)
                if (
                    not isinstance(out_logits, torch.Tensor)
                    or out_logits.ndim != 3
                    or out_logits.shape[0] != 2
                    or out_logits.shape[1] != prefill_seq_len
                ):
                    raise ValueError(
                        f"Batched prefill logits must have shape (2, {prefill_seq_len}, V), "
                        f"got {None if out_logits is None else tuple(out_logits.shape)}."
                    )
                logits = out_logits[:, -1, :]
                pos_logits = logits[0:1].float()
                neg_logits = logits[1:2].float()
                del prefill_out, out_logits, logits, batched_inputs

            generated_ids: List[int] = []
            alpha_history: List[float] = []
            entropy_history: List[float] = []
            phase_reasoning = False
            phase_cooldown_step = 0
            visual_reading_active = True

            # Step-0 initial visual divergence for QIACF
            p_pos0 = F.softmax(pos_logits, dim=-1)
            p_neg0 = F.softmax(neg_logits, dim=-1)
            initial_d_js = float(
                self.decoder.compute_js_divergence(p_pos0, p_neg0).squeeze().item()
            )

            for step in range(max_new_tokens):
                step_spatial_conf = 0.0 if phase_reasoning else spatial_confidence
                next_token_id, alpha_t, entropy_t, d_js_val = self.decoder.decode_step(
                    pos_logits=pos_logits,
                    neg_logits=neg_logits,
                    step=step,
                    spatial_confidence=step_spatial_conf,
                    phase_cooldown_step=phase_cooldown_step,
                    initial_d_js=initial_d_js,
                    eos_id=[
                        eid
                        for eid in [eos_id, getattr(self.tokenizer, "eos_token_id", None)]
                        if eid is not None
                    ],
                )

                if step_trace is not None:
                    self._record_decode_step_trace(
                        step_trace,
                        step=step,
                        pos_logits=pos_logits,
                        neg_logits=neg_logits,
                        alpha_t=alpha_t,
                        entropy_t=entropy_t,
                        d_js_val=d_js_val,
                        next_token_id=next_token_id,
                        cache_seq_len=cache_seq_len,
                    )

                alpha_history.append(alpha_t)
                entropy_history.append(entropy_t)

                if next_token_id == eos_id or next_token_id == self.tokenizer.eos_token_id:
                    break
                generated_ids.append(next_token_id)

                # ESSC (Entity-Sustained Soft Cooldown):
                if step >= 4 and d_js_val <= self.decoder.tau_js:
                    visual_reading_active = False
                    phase_cooldown_step += 1

                # Two-phase reasoning transition fallback:
                if not phase_reasoning and len(generated_ids) >= 1:
                    token_str = self.tokenizer.decode([next_token_id], skip_special_tokens=True)
                    if "\n" in token_str or ":" in token_str:
                        phase_reasoning = True
                        phase_cooldown_step = max(phase_cooldown_step, 4)

                # Validate cache state before advancing decoding step
                expected_prev_seq_len = prefill_seq_len + step
                self._validate_kv_cache(
                    batch_kv, expected_batch_size=2, expected_seq_len=expected_prev_seq_len
                )

                next_token_tensor = torch.tensor(
                    [[next_token_id]], device=self.device, dtype=torch.long
                )
                next_embed = embed_layer(next_token_tensor)
                next_embed_batched = torch.cat([next_embed, next_embed], dim=0)

                step_kwargs: Dict[str, Any] = {
                    "inputs_embeds": next_embed_batched,
                    "past_key_values": batch_kv,
                    "use_cache": True,
                    "return_dict": True,
                }
                if bat_mask is not None:
                    bat_mask = torch.cat(
                        [
                            bat_mask,
                            torch.ones((2, 1), dtype=bat_mask.dtype, device=bat_mask.device),
                        ],
                        dim=1,
                    )
                    step_kwargs["attention_mask"] = bat_mask
                if bat_pos is not None:
                    next_bat_pos = bat_pos[:, -1:] + 1
                    bat_pos = torch.cat([bat_pos, next_bat_pos], dim=1)
                    step_kwargs["position_ids"] = next_bat_pos

                expected_next_seq_len = expected_prev_seq_len + 1
                if has_backbone:
                    step_out = self.model.language_model.model(**step_kwargs)
                    batch_kv = step_out.past_key_values
                    step_hidden = getattr(step_out, "last_hidden_state", None)
                    if (
                        not isinstance(step_hidden, torch.Tensor)
                        or step_hidden.ndim != 3
                        or step_hidden.shape[0] != 2
                        or step_hidden.shape[1] != 1
                    ):
                        raise ValueError(
                            f"Batched step {step} last_hidden_state must have shape (2, 1, D), "
                            f"got {None if step_hidden is None else tuple(step_hidden.shape)}."
                        )
                    cache_seq_len = self._validate_kv_cache(
                        batch_kv, expected_batch_size=2, expected_seq_len=expected_next_seq_len
                    )
                    logits = self.model.language_model.lm_head(step_hidden[:, -1:, :])[:, -1, :]
                    if logits.ndim != 2 or logits.shape[0] != 2:
                        raise ValueError(
                            f"Batched step {step} lm_head output must have shape (2, V), got {tuple(logits.shape)}."
                        )
                    pos_logits = logits[0:1].float()
                    neg_logits = logits[1:2].float()
                    del step_out, step_hidden, logits
                else:
                    step_out = self.model.language_model(**step_kwargs)
                    batch_kv = step_out.past_key_values
                    cache_seq_len = self._validate_kv_cache(
                        batch_kv, expected_batch_size=2, expected_seq_len=expected_next_seq_len
                    )
                    out_logits = getattr(step_out, "logits", None)
                    if (
                        not isinstance(out_logits, torch.Tensor)
                        or out_logits.ndim != 3
                        or out_logits.shape[0] != 2
                        or out_logits.shape[1] != 1
                    ):
                        raise ValueError(
                            f"Batched step {step} logits must have shape (2, 1, V), "
                            f"got {None if out_logits is None else tuple(out_logits.shape)}."
                        )
                    logits = out_logits[:, -1, :]
                    pos_logits = logits[0:1].float()
                    neg_logits = logits[1:2].float()
                    del step_out, out_logits, logits

                del next_token_tensor, next_embed, next_embed_batched

        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        alpha_mean = float(sum(alpha_history) / max(1, len(alpha_history)))
        entropy_mean = float(sum(entropy_history) / max(1, len(entropy_history)))

        del batch_kv, pos_logits, neg_logits
        return text, alpha_mean, entropy_mean

    def _run_sequential_dual_branch_decode_reference(
        self,
        pos_inputs: "torch.Tensor",
        neg_inputs: "torch.Tensor",
        spatial_confidence: float,
        max_new_tokens: int = 64,
        *,
        attention_mask: Optional["torch.Tensor"] = None,
        position_ids: Optional["torch.Tensor"] = None,
        step_trace: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, float, float]:
        """
        Pre-change sequential (B=1) dual-branch decoding reference path preserved for
        strict numerical and token parity verification (Section 5.5).
        """
        prefill_seq_len, _ = self._validate_dual_branch_inputs(pos_inputs, neg_inputs)
        _, pos_mask, neg_mask, _, pos_pos, neg_pos = self._prepare_optional_mask_and_positions(
            prefill_seq_len,
            pos_inputs.device,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        eos_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        embed_layer = self.model.language_model.get_input_embeddings()
        has_backbone = hasattr(self.model.language_model, "model") and hasattr(
            self.model.language_model, "lm_head"
        )

        with torch.no_grad():
            pos_prefill_kwargs: Dict[str, Any] = {
                "inputs_embeds": pos_inputs,
                "use_cache": True,
                "return_dict": True,
            }
            neg_prefill_kwargs: Dict[str, Any] = {
                "inputs_embeds": neg_inputs,
                "use_cache": True,
                "return_dict": True,
            }
            if pos_mask is not None and neg_mask is not None:
                pos_prefill_kwargs["attention_mask"] = pos_mask
                neg_prefill_kwargs["attention_mask"] = neg_mask
            if pos_pos is not None and neg_pos is not None:
                pos_prefill_kwargs["position_ids"] = pos_pos
                neg_prefill_kwargs["position_ids"] = neg_pos

            if has_backbone:
                pos_out = self.model.language_model.model(**pos_prefill_kwargs)
                pos_kv = pos_out.past_key_values
                pos_logits = self.model.language_model.lm_head(
                    pos_out.last_hidden_state[:, -1:, :]
                )[:, -1, :].float()
                del pos_out

                neg_out = self.model.language_model.model(**neg_prefill_kwargs)
                neg_kv = neg_out.past_key_values
                neg_logits = self.model.language_model.lm_head(
                    neg_out.last_hidden_state[:, -1:, :]
                )[:, -1, :].float()
                del neg_out
            else:
                pos_out = self.model.language_model(**pos_prefill_kwargs)
                neg_out = self.model.language_model(**neg_prefill_kwargs)
                pos_kv = pos_out.past_key_values
                neg_kv = neg_out.past_key_values
                pos_logits = pos_out.logits[:, -1, :].float()
                neg_logits = neg_out.logits[:, -1, :].float()
                del pos_out, neg_out

            cache_seq_len = self._validate_kv_cache(
                pos_kv, expected_batch_size=1, expected_seq_len=prefill_seq_len
            )
            self._validate_kv_cache(
                neg_kv, expected_batch_size=1, expected_seq_len=prefill_seq_len
            )

            generated_ids: List[int] = []
            alpha_history: List[float] = []
            entropy_history: List[float] = []
            phase_reasoning = False
            phase_cooldown_step = 0
            visual_reading_active = True

            # Step-0 initial visual divergence for QIACF
            p_pos0 = F.softmax(pos_logits, dim=-1)
            p_neg0 = F.softmax(neg_logits, dim=-1)
            initial_d_js = float(
                self.decoder.compute_js_divergence(p_pos0, p_neg0).squeeze().item()
            )

            for step in range(max_new_tokens):
                step_spatial_conf = 0.0 if phase_reasoning else spatial_confidence
                next_token_id, alpha_t, entropy_t, d_js_val = self.decoder.decode_step(
                    pos_logits=pos_logits,
                    neg_logits=neg_logits,
                    step=step,
                    spatial_confidence=step_spatial_conf,
                    phase_cooldown_step=phase_cooldown_step,
                    initial_d_js=initial_d_js,
                    eos_id=[
                        eid
                        for eid in [eos_id, getattr(self.tokenizer, "eos_token_id", None)]
                        if eid is not None
                    ],
                )

                if step_trace is not None:
                    self._record_decode_step_trace(
                        step_trace,
                        step=step,
                        pos_logits=pos_logits,
                        neg_logits=neg_logits,
                        alpha_t=alpha_t,
                        entropy_t=entropy_t,
                        d_js_val=d_js_val,
                        next_token_id=next_token_id,
                        cache_seq_len=cache_seq_len,
                    )

                alpha_history.append(alpha_t)
                entropy_history.append(entropy_t)

                if next_token_id == eos_id or next_token_id == self.tokenizer.eos_token_id:
                    break
                generated_ids.append(next_token_id)

                # ESSC (Entity-Sustained Soft Cooldown):
                if step >= 4 and d_js_val <= self.decoder.tau_js:
                    visual_reading_active = False
                    phase_cooldown_step += 1

                # Two-phase reasoning transition fallback:
                if not phase_reasoning and len(generated_ids) >= 1:
                    token_str = self.tokenizer.decode([next_token_id], skip_special_tokens=True)
                    if "\n" in token_str or ":" in token_str:
                        phase_reasoning = True
                        phase_cooldown_step = max(phase_cooldown_step, 4)

                next_token_tensor = torch.tensor(
                    [[next_token_id]], device=self.device, dtype=torch.long
                )
                next_embed = embed_layer(next_token_tensor)

                pos_step_kwargs: Dict[str, Any] = {
                    "inputs_embeds": next_embed,
                    "past_key_values": pos_kv,
                    "use_cache": True,
                    "return_dict": True,
                }
                neg_step_kwargs: Dict[str, Any] = {
                    "inputs_embeds": next_embed,
                    "past_key_values": neg_kv,
                    "use_cache": True,
                    "return_dict": True,
                }
                if pos_mask is not None and neg_mask is not None:
                    pos_mask = torch.cat(
                        [
                            pos_mask,
                            torch.ones((1, 1), dtype=pos_mask.dtype, device=pos_mask.device),
                        ],
                        dim=1,
                    )
                    neg_mask = torch.cat(
                        [
                            neg_mask,
                            torch.ones((1, 1), dtype=neg_mask.dtype, device=neg_mask.device),
                        ],
                        dim=1,
                    )
                    pos_step_kwargs["attention_mask"] = pos_mask
                    neg_step_kwargs["attention_mask"] = neg_mask
                if pos_pos is not None and neg_pos is not None:
                    next_pos_id = pos_pos[:, -1:] + 1
                    next_neg_id = neg_pos[:, -1:] + 1
                    pos_pos = torch.cat([pos_pos, next_pos_id], dim=1)
                    neg_pos = torch.cat([neg_pos, next_neg_id], dim=1)
                    pos_step_kwargs["position_ids"] = next_pos_id
                    neg_step_kwargs["position_ids"] = next_neg_id

                expected_next_seq_len = prefill_seq_len + step + 1
                if has_backbone:
                    pos_step = self.model.language_model.model(**pos_step_kwargs)
                    pos_kv = pos_step.past_key_values
                    pos_logits = self.model.language_model.lm_head(
                        pos_step.last_hidden_state[:, -1:, :]
                    )[:, -1, :].float()

                    neg_step = self.model.language_model.model(**neg_step_kwargs)
                    neg_kv = neg_step.past_key_values
                    neg_logits = self.model.language_model.lm_head(
                        neg_step.last_hidden_state[:, -1:, :]
                    )[:, -1, :].float()
                    del pos_step, neg_step
                else:
                    pos_out = self.model.language_model(**pos_step_kwargs)
                    neg_out = self.model.language_model(**neg_step_kwargs)
                    pos_kv = pos_out.past_key_values
                    neg_kv = neg_out.past_key_values
                    pos_logits = pos_out.logits[:, -1, :].float()
                    neg_logits = neg_out.logits[:, -1, :].float()
                    del pos_out, neg_out

                cache_seq_len = self._validate_kv_cache(
                    pos_kv, expected_batch_size=1, expected_seq_len=expected_next_seq_len
                )
                self._validate_kv_cache(
                    neg_kv, expected_batch_size=1, expected_seq_len=expected_next_seq_len
                )
                del next_token_tensor, next_embed

        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        alpha_mean = float(sum(alpha_history) / max(1, len(alpha_history)))
        entropy_mean = float(sum(entropy_history) / max(1, len(entropy_history)))

        del pos_kv, neg_kv, pos_logits, neg_logits
        return text, alpha_mean, entropy_mean

    def _prepare_dual_branch_stage2_inputs(
        self,
        roi_tile: Image.Image,
        global_thumb: Image.Image,
        question: str,
        stage1_kl_score: float = 1.0,
        char_height_px: Optional[float] = None,
        magnification: float = 1.0,
        cached_thumb_vit: Optional[torch.Tensor] = None,
        context_tile: Optional[Image.Image] = None,
        roi_tiles: Optional[List[Image.Image]] = None,
        magnifications: Optional[List[float]] = None,
        char_heights: Optional[List[Optional[float]]] = None,
    ) -> Tuple["torch.Tensor", "torch.Tensor", float]:
        """
        Build the positive and negative branch input embedding sequences `(pos_inputs, neg_inputs)`
        and `spatial_confidence` for Stage 2 contrastive decoding.
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

        pos_crops = []
        neg_crops = []
        for i, tile in enumerate(roi_tiles):
            mag_i = magnifications[i] if i < len(magnifications) else 1.0
            ch_i = char_heights[i] if i < len(char_heights) else None
            pos_tile = apply_adaptive_laplacian_edge_reemphasis(tile, magnification=mag_i)
            pos_crops.append(pos_tile)
            blur_radius = compute_adaptive_lowpass_radius(
                magnification=mag_i,
                char_height_px=ch_i,
            )
            neg_tile = tile.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            neg_crops.append(neg_tile)

        spatial_confidence = float(math.tanh(max(0.0, float(stage1_kl_score))))

        with torch.no_grad():
            if cached_thumb_vit is not None:
                thumb_vit = cached_thumb_vit
            else:
                thumb_pv = self.transform(global_thumb).unsqueeze(0).to(
                    device=self.device, dtype=self.torch_dtype
                )
                thumb_vit = self.model.extract_feature(thumb_pv)
                del thumb_pv

            num_roi = len(pos_crops)
            all_roi_crops = pos_crops + neg_crops
            all_roi_pv = torch.stack([self.transform(c) for c in all_roi_crops]).to(
                device=self.device, dtype=self.torch_dtype
            )
            all_roi_vit = self.model.extract_feature(all_roi_pv)
            del all_roi_pv

            pos_sub_vits = [all_roi_vit[k : k + 1] for k in range(num_roi)]
            neg_sub_vits = [all_roi_vit[num_roi + k : num_roi + k + 1] for k in range(num_roi)]

            if context_tile is not None and num_roi == 1:
                ctx_pv = self.transform(context_tile).unsqueeze(0).to(
                    device=self.device, dtype=self.torch_dtype
                )
                ctx_vit = self.model.extract_feature(ctx_pv)
                del ctx_pv
                pos_sub_vits.append(ctx_vit)
                neg_sub_vits.append(ctx_vit)

            pos_vit = torch.cat([thumb_vit] + pos_sub_vits, dim=0)
            neg_vit = torch.cat([thumb_vit] + neg_sub_vits, dim=0)
            del all_roi_vit

            decoding_question = format_concise_decoding_prompt(question)

            pos_prefix, suffix_q, _, _ = self.extractor._build_prefix_and_suffix_embeds(
                pos_vit, decoding_question
            )
            neg_prefix, neg_suffix_q, _, _ = self.extractor._build_prefix_and_suffix_embeds(
                neg_vit, decoding_question
            )
            if (
                isinstance(neg_suffix_q, torch.Tensor)
                and isinstance(suffix_q, torch.Tensor)
                and neg_suffix_q.shape != suffix_q.shape
            ):
                raise ValueError(
                    f"Positive and negative question suffix embeddings must have identical shapes, "
                    f"got {tuple(suffix_q.shape)} vs {tuple(neg_suffix_q.shape)}."
                )

            pos_inputs = torch.cat([pos_prefix, suffix_q], dim=1)
            neg_inputs = torch.cat([neg_prefix, suffix_q], dim=1)

            del pos_vit, neg_vit, pos_prefix, neg_prefix, suffix_q
            if cached_thumb_vit is None:
                del thumb_vit

        return pos_inputs, neg_inputs, spatial_confidence

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
        Executes batched B=2 positive/negative decoding (TICKET-OPT-01).
        """
        pos_inputs, neg_inputs, spatial_confidence = self._prepare_dual_branch_stage2_inputs(
            roi_tile=roi_tile,
            global_thumb=global_thumb,
            question=question,
            stage1_kl_score=stage1_kl_score,
            char_height_px=char_height_px,
            magnification=magnification,
            cached_thumb_vit=cached_thumb_vit,
            context_tile=context_tile,
            roi_tiles=roi_tiles,
            magnifications=magnifications,
            char_heights=char_heights,
        )
        try:
            return self._run_batched_dual_branch_decode(
                pos_inputs=pos_inputs,
                neg_inputs=neg_inputs,
                spatial_confidence=spatial_confidence,
                max_new_tokens=max_new_tokens,
            )
        finally:
            del pos_inputs, neg_inputs

    def _decode_spatial_pmi_contrastive_sequential_reference(
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
        Stage 2 unbatched (B=1) sequential reference implementation preserved for parity tests
        (TICKET-OPT-01 Section 5.5).
        """
        pos_inputs, neg_inputs, spatial_confidence = self._prepare_dual_branch_stage2_inputs(
            roi_tile=roi_tile,
            global_thumb=global_thumb,
            question=question,
            stage1_kl_score=stage1_kl_score,
            char_height_px=char_height_px,
            magnification=magnification,
            cached_thumb_vit=cached_thumb_vit,
            context_tile=context_tile,
            roi_tiles=roi_tiles,
            magnifications=magnifications,
            char_heights=char_heights,
        )
        try:
            return self._run_sequential_dual_branch_decode_reference(
                pos_inputs=pos_inputs,
                neg_inputs=neg_inputs,
                spatial_confidence=spatial_confidence,
                max_new_tokens=max_new_tokens,
            )
        finally:
            del pos_inputs, neg_inputs

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
        max_allowed_side = int(round(0.85 * max_dim))

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

        pad_y_diacritic = max(int(round(0.25 * h_seed)), 4)
        fused_y1 = max(0, fused_y1 - pad_y_diacritic)
        fused_y2 = min(H_img, fused_y2 + pad_y_diacritic)

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
        *,
        oracle_bbox: Optional[Sequence[int]] = None,
        image_key: Optional[Union[ImageCacheIdentity, Tuple[Any, ...], str]] = None,
        image_id: Optional[Union[str, int]] = None,
        image_path: Optional[str] = None,
        dataset_root: Optional[str] = None,
        dataset_version: Optional[str] = None,
        dataset_namespace: Optional[str] = None,
        content_fingerprint: Optional[str] = None,
    ) -> PipelineResult:
        """
        Execute full SG-VCD pipeline adhering strictly to PROJECT.md interface contract.
        Supports Top-K Candidate Grounding (K in {1, 2}) and Pre-RoI adaptive Laplacian edge re-emphasis.
        
        Args:
            image: PIL Image input
            question: Natural language question
            max_new_tokens: Maximum tokens to generate
            cached_visual_features: Optional cached ViT embeddings for global thumbnail
            image_key: Optional precomputed ImageCacheIdentity or key (OPT-06)
            image_id: Optional stable image identifier within dataset
            image_path: Optional path to image file
            dataset_root: Optional dataset root directory
            dataset_version: Optional dataset version/namespace identifier
            dataset_namespace: Alias for dataset_version
            content_fingerprint: Optional content/version fingerprint
            
        Returns:
            PipelineResult with predictions, bboxes, metrics, and latency breakdown.
        """
        t0 = time.time()
        resolved_image_key = build_image_cache_identity(
            image,
            image_key=image_key,
            image_id=image_id,
            image_path=image_path,
            dataset_root=dataset_root,
            dataset_version=dataset_version,
            dataset_namespace=dataset_namespace,
            content_fingerprint=content_fingerprint,
        )
        if cached_visual_features is None:
            cached_visual_features = self.extract_visual_features(
                image, image_key=resolved_image_key
            )

        if oracle_bbox is not None:
            # Oracle ROI Diagnostic Arm (OD-01 / OD-05): Use GT bounding box directly
            ox1, oy1, ox2, oy2 = [int(v) for v in oracle_bbox]
            W_img, H_img = image.size
            if ox2 <= ox1 or oy2 <= oy1 or ox1 < 0 or oy1 < 0 or ox2 > W_img or oy2 > H_img:
                raise ValueError(
                    f"Invalid Oracle ROI bounding box {oracle_bbox} for image dimensions ({W_img}, {H_img})"
                )
            stage1_bbox = [ox1, oy1, ox2, oy2]
            stage1_bboxes = [stage1_bbox]
            char_heights = [None]
            area_frac = float((ox2 - ox1) * (oy2 - oy1)) / max(1.0, float(W_img * H_img))
            selection = RegionSelectionResult(
                pixel_bbox=tuple(stage1_bbox),
                grid_bbox=(0, 0, 0, 0),
                kl_score=1.0,
                captured_mass=1.0,
                area_fraction=area_frac,
            )
            multi_selection = MultiRegionSelectionResult(
                pixel_bboxes=[selection.pixel_bbox],
                grid_bboxes=[selection.grid_bbox],
                kl_scores=[1.0],
                captured_masses=[1.0],
                area_fractions=[selection.area_fraction],
                k=1,
            )
            extraction = ExtractionResult(
                phi_map=np.zeros((1, 1), dtype=np.float32),
                r_map=np.zeros((1, 1), dtype=np.float32),
                aff_map=np.zeros((1, 1), dtype=np.float32),
                a_query_map=np.zeros((1, 1), dtype=np.float32),
                a_gen_map=np.zeros((1, 1), dtype=np.float32),
                grid_ratio=(1, 1),
                vit_embeds=None,
            )
            t_ext = 0.0
            t_sel = 0.0
        else:
            # Stage 1.1: Information-Theoretic Attention Grounding
            t_start_ext = time.time()
            ext_params = inspect.signature(self.extractor.extract).parameters
            if "image_key" in ext_params or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in ext_params.values()
            ):
                extraction = self.extractor.extract(
                    image, question, image_key=resolved_image_key
                )
            else:
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
    
                    # OPT-03: Single-image Stage 1 working-map cache keyed by resolved_image_key
                    working_maps = None
                    if (
                        self._cached_working_maps_key == resolved_image_key
                        and _is_valid_working_maps(self._cached_working_maps)
                    ):
                        working_maps = self._cached_working_maps
                    else:
                        # Release previous cached working maps and key before computing new maps
                        self._cached_working_maps = None
                        self._cached_working_maps_key = None
                        if hasattr(self.propagator, "_compute_working_maps"):
                            computed_maps = self.propagator._compute_working_maps(image)
                            if _is_valid_working_maps(computed_maps):
                                self._cached_working_maps = computed_maps
                                self._cached_working_maps_key = resolved_image_key
                            working_maps = computed_maps
    
                    if hasattr(self.propagator, "propagate_multi"):
                        pm_params = inspect.signature(self.propagator.propagate_multi).parameters
                        if "working_maps" in pm_params or any(
                            p.kind == inspect.Parameter.VAR_KEYWORD for p in pm_params.values()
                        ):
                            prop_results = self.propagator.propagate_multi(
                                image=image,
                                seed_patches=seed_patches,
                                grid_ratio=extraction.grid_ratio,
                                working_maps=working_maps,
                            )
                        else:
                            prop_results = self.propagator.propagate_multi(
                                image=image,
                                seed_patches=seed_patches,
                                grid_ratio=extraction.grid_ratio,
                            )
                    else:
                        p_params = inspect.signature(self.propagator.propagate).parameters
                        if "working_maps" in p_params or any(
                            p.kind == inspect.Parameter.VAR_KEYWORD for p in p_params.values()
                        ):
                            prop_results = [
                                self.propagator.propagate(
                                    image=image,
                                    seed_patch=sp,
                                    grid_ratio=extraction.grid_ratio,
                                    working_maps=working_maps,
                                )
                                for sp in seed_patches
                            ]
                        else:
                            prop_results = [
                                self.propagator.propagate(
                                    image=image, seed_patch=sp, grid_ratio=extraction.grid_ratio
                                )
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
                            aligned_merged = merge_aligned_candidates([tuple(b) for b in fused_candidates])
                            if len(aligned_merged) < len(fused_candidates):
                                stage1_bboxes = [list(b) for b in aligned_merged]
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
            c = self.cropper.crop(image, tuple(b))
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

        # Flatten all dynamic multi-patch sub-tiles across crops
        all_roi_tiles = []
        all_magnifications = []
        all_char_heights = []
        for idx, c in enumerate(crops):
            sub_tiles = getattr(c, "roi_tiles", None) or [c.roi_tile]
            ch = char_heights[idx] if idx < len(char_heights) else None
            for st in sub_tiles:
                all_roi_tiles.append(st)
                all_magnifications.append(c.magnification)
                all_char_heights.append(ch)

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
            roi_tiles=all_roi_tiles,
            magnifications=all_magnifications,
            char_heights=all_char_heights,
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

        # Memory cleanup (gc.collect / conditional CUDA watchdog) is coordinated at
        # image boundaries by the benchmark runner (TICKET-OPT-02).

        spatial_conf_val = 1.0 if oracle_bbox is not None else float(math.tanh(max(0.0, float(selection.kl_score))))

        return PipelineResult(
            pred_text=pred_text,
            pred_raw_text=pred_sg_vcd,
            stage1_bbox=list(stage1_bbox),
            spatial_confidence=spatial_conf_val,
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
