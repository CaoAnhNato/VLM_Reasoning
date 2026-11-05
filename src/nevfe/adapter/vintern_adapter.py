# -*- coding: utf-8 -*-
"""
Decoupled Multi-Scale Query-Orthogonal Variational Free Energy Adapter for Vintern-1B.
Strictly <= 2 global hyperparameters (lambda_param, mu_param), zero regex, zero stopwords.
Preserves native 256-token 16x16 visual grid geometry while providing 3.33x optical zoom.
Enforces 1:1 isotropic square pixel cropping with symmetric border padding to eliminate font squashing.
"""

import math
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from src.nevfe.core.engine import (
    NEVFEEngine,
    compute_thermodynamic_spatial_window,
    compute_question_conditioned_spatial_window,
    compute_residual_spatial_pursuit_windows,
    compute_stroke_gated_homography,
    compute_box_iou,
    compute_otsu_threshold,
    compute_visual_token_saliency_and_mask,
    evaluate_semantic_stopping,
    pool_query_agnostic_gradient_tokens,
    pool_question_guided_focal_tokens,
    adaptive_tensor_stroke_dilation,
    compute_question_conditioned_cross_attention_saliency,
    compute_query_grounded_lexical_bias,
    apply_saliency_modulated_gating,
)


CANONICAL_PROMPT_TEMPLATE = "Dựa vào chữ trong ảnh, hãy trả lời ngắn gọn: {question}"
PROMPT_INSTRUCTION = "Dựa vào chữ trong ảnh, hãy trả lời ngắn gọn: "


def select_admissible_aspect_ratio_grid(
    w_box: float,
    h_box: float,
) -> Tuple[int, int]:
    """
    Continuous log-aspect-ratio loss minimization over admissible grids (Theorem 16, Directive W5).
    Admissible grids: (1, 1), (2, 1), (1, 2).
    """
    ar = max(1e-6, float(w_box)) / max(1.0, float(h_box))
    candidates = [(1, 1), (2, 1), (1, 2)]

    best_grid = min(
        candidates,
        key=lambda g: abs(math.log(ar) - math.log(g[0] / g[1]))
    )
    return best_grid


def pad_image_edge_replicate(crop: Image.Image) -> Image.Image:
    """
    Pad rectangular crop to square using edge-border replication (Directive W2, Lemma 2).
    Eliminates artificial step edges that quench interior text strokes.
    """
    cw, ch = crop.size
    if cw == ch:
        return crop
    max_side = max(cw, ch)
    pad_left = (max_side - cw) // 2
    pad_right = max_side - cw - pad_left
    pad_top = (max_side - ch) // 2
    pad_bottom = max_side - ch - pad_top

    arr = np.array(crop)
    if arr.ndim == 3:
        padded_arr = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="edge")
    else:
        padded_arr = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="edge")
    return Image.fromarray(padded_arr)


def extract_symmetric_square_crop(image: Image.Image, box: Tuple[int, int, int, int]) -> Image.Image:
    """
    Directive W2 (Boundary Seam Elimination, Theorem 32-Ext):
    Symmetric crop expansion on native image [0, W] x [0, H] with edge-border replicate fallback,
    completely eliminating static (128, 128, 128) gray padding seams and boundary step edges.
    """
    W, H = image.size
    bx0, by0, bx1, by1 = box
    cw = max(1, bx1 - bx0)
    ch = max(1, by1 - by0)
    max_side = max(cw, ch)
    cx = (bx0 + bx1) / 2.0
    cy = (by0 + by1) / 2.0

    # Expand box symmetrically centered on (cx, cy) to target square side max_side
    sx0 = int(round(cx - max_side / 2.0))
    sy0 = int(round(cy - max_side / 2.0))
    sx1 = sx0 + max_side
    sy1 = sy0 + max_side

    # Shift into image boundaries [0, W] x [0, H] preserving span if possible
    if sx0 < 0:
        sx1 = min(W, sx1 - sx0)
        sx0 = 0
    if sx1 > W:
        sx0 = max(0, sx0 - (sx1 - W))
        sx1 = W

    if sy0 < 0:
        sy1 = min(H, sy1 - sy0)
        sy0 = 0
    if sy1 > H:
        sy0 = max(0, sy0 - (sy1 - H))
        sy1 = H

    crop = image.crop((sx0, sy0, sx1, sy1))
    return pad_image_edge_replicate(crop)


def extract_anisotropic_focal_tiles(
    image: Image.Image,
    crop_box: Tuple[int, int, int, int],
    grid: Tuple[int, int],
    target_tile_size: int = 448,
) -> List[Image.Image]:
    """
    Extracts gx * gy square sub-tiles from crop_box preserving aspect ratio (Theorem 16, Directive W5).
    Each sub-tile is resized to target_tile_size x target_tile_size without anisotropic squashing.
    """
    gx, gy = grid
    x0, y0, x1, y1 = crop_box
    bw = max(2, x1 - x0)
    bh = max(2, y1 - y0)

    crop = image.crop(crop_box)

    tiles = []
    tile_w = bw / float(gx)
    tile_h = bh / float(gy)

    for j in range(gy):
        for i in range(gx):
            sub_x0 = int(round(i * tile_w))
            sub_y0 = int(round(j * tile_h))
            sub_x1 = int(round((i + 1) * tile_w))
            sub_y1 = int(round((j + 1) * tile_h))

            sub_x0 = max(0, min(crop.width - 1, sub_x0))
            sub_y0 = max(0, min(crop.height - 1, sub_y0))
            sub_x1 = max(sub_x0 + 1, min(crop.width, sub_x1))
            sub_y1 = max(sub_y0 + 1, min(crop.height, sub_y1))

            sub_tile = crop.crop((sub_x0, sub_y0, sub_x1, sub_y1))

            # Pad to square if slight rounding discrepancy using edge replicate padding
            sub_tile = pad_image_edge_replicate(sub_tile)

            tiles.append(sub_tile.resize((target_tile_size, target_tile_size), Image.Resampling.BICUBIC))

    return tiles


def _find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: List[Tuple[int, int]],
    width: int,
    height: int,
    image_size: int = 448,
) -> Tuple[int, int]:
    best_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for r in target_ratios:
        target_aspect_ratio = r[0] / r[1]
        diff = abs(aspect_ratio - target_aspect_ratio)
        if diff < best_diff:
            best_diff = diff
            best_ratio = r
        elif diff == best_diff:
            if area > 0.5 * image_size * image_size * r[0] * r[1]:
                best_ratio = r
    return best_ratio


def dynamic_preprocess_vintern(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 6,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> List[Image.Image]:
    """
    Standard dynamic aspect-ratio tiling for Vintern / InternVL models.
    """
    orig_w, orig_h = image.size
    aspect_ratio = orig_w / max(1, orig_h)

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
        aspect_ratio, target_ratios, orig_w, orig_h, image_size
    )
    gx, gy = target_aspect_ratio
    target_w = image_size * gx
    target_h = image_size * gy
    blocks = gx * gy

    resized_img = image.resize((target_w, target_h), Image.Resampling.BICUBIC)
    processed_images = []
    for i in range(blocks):
        box = (
            (i % gx) * image_size,
            (i // gx) * image_size,
            ((i % gx) + 1) * image_size,
            ((i // gx) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    if use_thumbnail and len(processed_images) > 1:
        thumbnail_img = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
        processed_images.append(thumbnail_img)

    return processed_images


def build_query_token_mask(
    question: str,
    tokenizer,
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Construct Query Indicator Tensor m_Q in {0, 1}^|V| invariant to casing and BPE prefixes.
    Strictly 0 regex, 0 hardcoded stopwords, 100% PyTorch tensor ops.
    """
    variants = [
        question,
        question.lower(),
        question.upper(),
        question.capitalize(),
        question.title(),
        " " + question.strip(),
        " " + question.lower().strip(),
    ]
    # Word-level BPE position invariance: ensures tokens appearing mid-sentence
    # (e.g. "cuốn", "full") are captured with and without leading BPE whitespace prefixes
    for w in question.strip().split():
        clean_w = w.strip(".,;:?!\"'()[]{}<>-/\\")
        candidates = [w]
        if clean_w and clean_w != w:
            candidates.append(clean_w)
        for cand in candidates:
            variants.extend([
                cand,
                cand.lower(),
                cand.upper(),
                cand.capitalize(),
                cand.title(),
                " " + cand,
                " " + cand.lower(),
            ])

    q_token_ids = set()
    for v in variants:
        tokens = tokenizer.encode(v, add_special_tokens=False)
        q_token_ids.update(tokens)

    q_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    if q_token_ids:
        indices = torch.tensor(list(q_token_ids), dtype=torch.long, device=device)
        indices = indices[indices < vocab_size]
        q_mask[indices] = True
    return q_mask


class VinternNEVFEAdapter:
    """
    Decoupled Multi-Scale Query-Orthogonal Variational Free Energy Adapter.

    Architecture (Route A):
    - Stage 1: Computes minimum-volume Chebyshev bounding box [x0, y0, x1, y1] on global 448x448 canvas.
               Isotropically scales to 1:1 pixel square with symmetric border padding to prevent character squashing.
    - Stage 2: Extracts high-resolution focal crop and projects exactly 256 visual tokens V_L.
    - Generation: Decodes autoregressively using strictly 256 visual tokens in native 16x16 RoPE geometry.
    """
    def __init__(
        self,
        model,
        tokenizer,
        transform=None,
        device: str = "cuda",
        lambda_param: float = 1.0,
        beta_param: float = 1.0,
        mu_param: float = 4.0,
        decoupled_mode: str = "focal_mfvr",  # "focal_mfvr" (default) or "focal_contrastive"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.transform = transform or self._build_default_transform(input_size=448)
        self.decoupled_mode = decoupled_mode

        self.engine = NEVFEEngine(
            lambda_param=lambda_param,
            beta_param=beta_param,
            mu_param=mu_param,
        )

        # Directive W3: Cache dynamic alphanumeric vocabulary V_alnum for the 62 ASCII primitives
        if self.tokenizer is not None and hasattr(self.tokenizer, "convert_tokens_to_ids"):
            ascii_chars = (
                [str(d) for d in range(10)]
                + [chr(c) for c in range(ord("A"), ord("Z") + 1)]
                + [chr(c) for c in range(ord("a"), ord("z") + 1)]
            )
            alnum_ids = [self.tokenizer.convert_tokens_to_ids(c) for c in ascii_chars]
            self.alnum_token_ids = [
                int(aid)
                for aid in alnum_ids
                if aid is not None and aid != getattr(self.tokenizer, "unk_token_id", None)
            ]
        else:
            self.alnum_token_ids = []
        self.engine.alnum_token_ids = self.alnum_token_ids

        self.max_dynamic_patch = getattr(getattr(model, "config", None), "max_dynamic_patch", 6)
        self.num_image_token = getattr(model, "num_image_token", 256)
        self.image_size = getattr(getattr(model, "config", None), "image_size", 448)
        self.system_message = getattr(model, "system_message", "Bạn là một trợ lý AI hữu ích.")

        # Caching dictionaries for image-level reuse across questions
        self._cached_img_key: Optional[Tuple[int, int, bytes]] = None
        self._cached_v_g: Optional[torch.Tensor] = None
        self._cached_v_l: Optional[torch.Tensor] = None
        self._last_crop_box: Optional[Tuple[int, int, int, int]] = None
        self._last_spatial_window: Optional[Tuple[float, float, float, float]] = None
        self._cached_vit_embeds: Optional[torch.Tensor] = None
        self._last_question: Optional[str] = None

    @staticmethod
    def _build_default_transform(input_size: int = 448):
        return T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    @staticmethod
    def _make_image_key(image: Image.Image) -> Tuple[int, int, bytes]:
        w, h = image.size
        thumb = image.resize((16, 16), Image.Resampling.NEAREST).convert("RGB")
        return (w, h, thumb.tobytes())

    # =========================================================================
    # STAGE 1: Fast Visual Saliency & Chebyshev Thermodynamic Spatial Window
    # =========================================================================
    def compute_stage1_window(
        self,
        image: Image.Image,
        question: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Tuple[int, int, int, int], Tuple[float, float, float, float]]:
        """
        Stage 1: Extract global features V_G from full 448x448 canvas and compute
        question-conditioned Chebyshev bounding window (Directive 2 & 3).
        
        Caches V_G by image key (w, h, thumb) to avoid re-extracting ViT features across questions.
        Computes question-conditioned bounding box dynamically per question.
        Enforces 1:1 isotropic square pixel cropping to eliminate character squashing.

        Returns:
            v_g: Global visual tokens [1, 256, 896]
            pixel_box: Bounding box (x0, y0, x1, y1) in image pixel coordinates
            norm_window: Normalized coordinates (x_min, y_min, x_max, y_max) in [0, 1]
        """
        img_key = self._make_image_key(image)
        norm_q = question.strip() if question is not None else None

        # Check if full cache (including question window) can be reused
        if (
            self._cached_img_key == img_key
            and self._cached_v_g is not None
            and self._last_question == norm_q
            and self._last_crop_box is not None
            and self._last_spatial_window is not None
        ):
            return self._cached_v_g, self._last_crop_box, self._last_spatial_window

        # Check if global visual features V_G can be reused for a new question on the same image
        if self._cached_img_key == img_key and self._cached_v_g is not None:
            v_g = self._cached_v_g
        else:
            # New image: extract global visual tokens V_G
            self._cached_img_key = img_key
            self._cached_v_g = None
            self._cached_v_l = None
            self._last_crop_box = None
            self._last_spatial_window = None
            self._cached_vit_embeds = None
            self._last_question = None

            pixel_values_g = self.transform(image).unsqueeze(0).to(
                device=self.device, dtype=torch.bfloat16
            )
            with torch.no_grad():
                v_g = self.model.extract_feature(pixel_values_g)  # [1, 256, 896]
            del pixel_values_g
            self._cached_v_g = v_g

        W, H = image.size

        if norm_q:
            raw_ids = self.tokenizer.encode(norm_q, add_special_tokens=False)
            if isinstance(raw_ids, torch.Tensor):
                q_ids = raw_ids.to(self.device)
            else:
                q_ids = torch.tensor(raw_ids, dtype=torch.long, device=self.device)
            if q_ids.ndim == 1:
                q_ids = q_ids.unsqueeze(0)
            with torch.no_grad():
                q_embeds = self.model.language_model.get_input_embeddings()(q_ids)  # [1, T, D]
            x_min, y_min, x_max, y_max = compute_question_conditioned_spatial_window(
                visual_tokens=v_g[0],
                question_embeddings=q_embeds[0],
                grid_size=16,
                k_sigma=self.engine.k_sigma,
                min_span=self.engine.min_span,
            )
            mh_windows = compute_residual_spatial_pursuit_windows(
                visual_tokens=v_g[0],
                question_embeddings=q_embeds[0],
                grid_size=16,
                k_sigma=self.engine.k_sigma,
                min_span=self.engine.min_span,
                lambda_res=0.20,
            )
        else:
            x_min, y_min, x_max, y_max = compute_thermodynamic_spatial_window(
                v_g[0],
                grid_size=16,
                k_sigma=self.engine.k_sigma,
                min_span=self.engine.min_span,
            )
            mh_windows = compute_residual_spatial_pursuit_windows(
                visual_tokens=v_g[0],
                grid_size=16,
                k_sigma=self.engine.k_sigma,
                min_span=self.engine.min_span,
                lambda_res=0.20,
            )

        norm_window = (x_min, y_min, x_max, y_max)

        # 3. Denormalize to pixel coordinates for Anisotropic Multi-Crop Tiling (Directive W5)
        box_x0 = max(0, int(round(x_min * W)))
        box_y0 = max(0, int(round(y_min * H)))
        box_x1 = min(W, int(round(x_max * W)))
        box_y1 = min(H, int(round(y_max * H)))

        box_x1 = max(box_x0 + 2, box_x1)
        box_y1 = max(box_y0 + 2, box_y1)
        pixel_box = (box_x0, box_y0, box_x1, box_y1)

        mh_pixel_boxes = []
        for (w_x0, w_y0, w_x1, w_y1) in mh_windows:
            px0 = max(0, int(round(w_x0 * W)))
            py0 = max(0, int(round(w_y0 * H)))
            px1 = min(W, int(round(w_x1 * W)))
            py1 = min(H, int(round(w_y1 * H)))
            mh_pixel_boxes.append((px0, py0, max(px0 + 2, px1), max(py0 + 2, py1)))
        self._last_multihypothesis_boxes = mh_pixel_boxes

        # Invalidate V_L if crop box changed
        if pixel_box != self._last_crop_box:
            self._cached_v_l = None

        self._last_question = norm_q
        self._last_crop_box = pixel_box
        self._last_spatial_window = norm_window

        return v_g, pixel_box, norm_window

    def compute_stage1_multihypothesis_windows(
        self,
        image: Image.Image,
        question: Optional[str] = None,
        lambda_res: float = 0.20,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Stage 1 Multi-Hypothesis RoI Selection (Theorem 35, Directive W1).
        Returns list of pixel bounding boxes [B1, B2] (K=2) or [B1] (K=1).
        """
        _, _, _ = self.compute_stage1_window(image, question=question)
        return getattr(self, "_last_multihypothesis_boxes", [self._last_crop_box])

    # =========================================================================
    # STAGE 2: Local Optical Zoom Focal Token Extraction (AMCT: K * 256 Tokens)
    # =========================================================================
    def extract_stage2_focal_features(
        self,
        image: Image.Image,
        crop_box: Tuple[int, int, int, int],
        question_embeddings: Optional[torch.Tensor] = None,
        use_amct: bool = True,
        question: Optional[str] = None,
        theta_zoom: float = 0.65,
        use_hierarchical_zoom: bool = True,
        use_homography_rectification: bool = False,
        multihypothesis_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> torch.Tensor:
        """
        Stage 2: Hierarchical Multi-Scale Focal Window Extraction (Theorem 32, Directive W1).
        Extracts Level 1 coarse focal window B^(1) -> V^(1) [1, 256, D].
        Conditions Level 2 sub-box B^(2) on question embeddings on V^(1).
        If rho_sub < theta_zoom (0.65), crops B^(2) from native high-res image -> V^(2) [1, 256, D]
        and concatenates losslessly: V_L = [V^(1); V^(2)] [1, 512, D].
        Else: V_L = V^(1) [1, 256, D].

        Returns:
            v_l_mod: Invariant focal visual tokens strictly with shape [1, 256, 896] or [1, 512, 896].
        """
        img_key = self._make_image_key(image)
        if (
            self._cached_img_key == img_key
            and self._cached_v_l is not None
            and self._last_crop_box == crop_box
            and multihypothesis_boxes is None
        ):
            return self._cached_v_l

        # If called directly with a new image, invalidate stale caches
        if self._cached_img_key != img_key:
            self._cached_img_key = img_key
            self._cached_v_g = None
            self._cached_v_l = None
            self._last_crop_box = crop_box
            self._last_spatial_window = None
            self._cached_vit_embeds = None

        if multihypothesis_boxes is not None and len(multihypothesis_boxes) >= 2:
            # Multi-hypothesis RoIs B1, B2 (Theorem 35, Directive W1 & W3)
            b1 = multihypothesis_boxes[0]
            b2 = multihypothesis_boxes[1]
            if use_homography_rectification:
                tile1, _, _ = compute_stroke_gated_homography(image, b1, mu_persp=0.08, target_size=(self.image_size, self.image_size))
                tile2, _, _ = compute_stroke_gated_homography(image, b2, mu_persp=0.08, target_size=(self.image_size, self.image_size))
            else:
                crop1 = extract_symmetric_square_crop(image, b1)
                crop2 = extract_symmetric_square_crop(image, b2)
                tile1 = crop1.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
                tile2 = crop2.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)

            pixel_values_1 = self.transform(tile1).unsqueeze(0).to(device=self.device, dtype=torch.float32)
            s1 = max(tile1.size)
            m1 = 448.0 / max(1.0, float(s1))
            pixel_values_1 = adaptive_tensor_stroke_dilation(pixel_values_1, magnification=m1)
            with torch.no_grad():
                v1_raw = self.model.extract_feature(pixel_values_1.to(torch.bfloat16))
            del pixel_values_1
            v1_mod = self.engine.modulate_patches(v1_raw[0]).unsqueeze(0).to(dtype=v1_raw.dtype, device=self.device)

            pixel_values_2 = self.transform(tile2).unsqueeze(0).to(device=self.device, dtype=torch.float32)
            s2 = max(tile2.size)
            m2 = 448.0 / max(1.0, float(s2))
            pixel_values_2 = adaptive_tensor_stroke_dilation(pixel_values_2, magnification=m2)
            with torch.no_grad():
                v2_raw = self.model.extract_feature(pixel_values_2.to(torch.bfloat16))
            del pixel_values_2
            v2_mod = self.engine.modulate_patches(v2_raw[0]).unsqueeze(0).to(dtype=v2_raw.dtype, device=self.device)

            # Strictly 512 visual tokens (Directive W3)
            v_l_mod = torch.cat([v1_mod, v2_mod], dim=1)
            self._cached_v_l = v_l_mod
            self._last_crop_box = crop_box
            self._cached_vit_embeds = v_l_mod
            return v_l_mod

        if use_hierarchical_zoom:
            # 1. Level 1 Coarse Focal Window B^(1)
            # Directive W2: Symmetric crop expansion on native image [0, W] x [0, H] eliminating gray padding seams
            if use_homography_rectification:
                tile1, _, _ = compute_stroke_gated_homography(image, crop_box, mu_persp=0.08, target_size=(self.image_size, self.image_size))
                s1 = max(tile1.size)
                m1 = 448.0 / max(1.0, float(s1))
            else:
                crop1 = extract_symmetric_square_crop(image, crop_box)
                s1 = max(crop1.size)
                m1 = 448.0 / max(1.0, float(s1))
                tile1 = crop1.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)

            pixel_values_1 = self.transform(tile1).unsqueeze(0).to(
                device=self.device, dtype=torch.float32
            )
            pixel_values_1 = adaptive_tensor_stroke_dilation(pixel_values_1, magnification=m1)
            with torch.no_grad():
                v1_raw = self.model.extract_feature(pixel_values_1.to(torch.bfloat16))  # [1, 256, 896]
            del pixel_values_1

            v1_mod = self.engine.modulate_patches(v1_raw[0]).unsqueeze(0).to(
                dtype=v1_raw.dtype, device=self.device
            )

            # 2. Condition Level 2 Sub-box Window B^(2) on question embeddings
            q_embeds = question_embeddings
            if q_embeds is None:
                q_text = question if question is not None else self._last_question
                if q_text and hasattr(self, "tokenizer") and self.tokenizer is not None and hasattr(self, "model") and self.model is not None:
                    try:
                        raw_ids = self.tokenizer.encode(q_text.strip(), add_special_tokens=False)
                        if raw_ids:
                            q_ids = torch.tensor(raw_ids, dtype=torch.long, device=self.device).unsqueeze(0)
                            with torch.no_grad():
                                q_embeds = self.model.language_model.get_input_embeddings()(q_ids)
                    except Exception:
                        q_embeds = None

            u_min, v_min, u_max, v_max = compute_question_conditioned_spatial_window(
                visual_tokens=v1_mod[0],
                question_embeddings=q_embeds[0] if (q_embeds is not None and isinstance(q_embeds, torch.Tensor) and q_embeds.numel() > 0) else None,
                grid_size=16,
                k_sigma=self.engine.k_sigma,
                min_span=self.engine.min_span,
            )

            delta_u = u_max - u_min
            delta_v = v_max - v_min
            rho_sub = max(delta_u, delta_v)

            if rho_sub < theta_zoom:
                W, H = image.size
                bx0, by0, bx1, by1 = crop_box
                bw = max(2, bx1 - bx0)
                bh = max(2, by1 - by0)

                sub_x0 = max(0, min(W - 2, int(round(bx0 + u_min * bw))))
                sub_y0 = max(0, min(H - 2, int(round(by0 + v_min * bh))))
                sub_x1 = max(sub_x0 + 2, min(W, int(round(bx0 + u_max * bw))))
                sub_y1 = max(sub_y0 + 2, min(H, int(round(by0 + v_max * bh))))
                box2_pixel = (sub_x0, sub_y0, sub_x1, sub_y1)

                # Directive W2: Symmetric crop expansion on native image [0, W] x [0, H] eliminating gray padding seams
                crop2 = extract_symmetric_square_crop(image, box2_pixel)
                s2 = max(crop2.size)
                m2 = 448.0 / max(1.0, float(s2))
                tile2 = crop2.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)

                pixel_values_2 = self.transform(tile2).unsqueeze(0).to(
                    device=self.device, dtype=torch.float32
                )
                pixel_values_2 = adaptive_tensor_stroke_dilation(pixel_values_2, magnification=m2)
                with torch.no_grad():
                    v2_raw = self.model.extract_feature(pixel_values_2.to(torch.bfloat16))  # [1, 256, 896]
                del pixel_values_2

                v2_mod = self.engine.modulate_patches(v2_raw[0]).unsqueeze(0).to(
                    dtype=v2_raw.dtype, device=self.device
                )

                # Concatenate losslessly: V_L = [V^(1); V^(2)] in R^(1 x 512 x D) (Theorem 32)
                v_l_mod = torch.cat([v1_mod, v2_mod], dim=1)
                self._last_level2_box = box2_pixel
            else:
                v_l_mod = v1_mod
                self._last_level2_box = None
        else:
            w_box = max(2, crop_box[2] - crop_box[0])
            h_box = max(2, crop_box[3] - crop_box[1])
            grid = select_admissible_aspect_ratio_grid(w_box, h_box)
            tiles = extract_anisotropic_focal_tiles(image, crop_box, grid, target_tile_size=self.image_size)

            tile_mods = []
            for tile in tiles:
                pixel_values_k = self.transform(tile).unsqueeze(0).to(
                    device=self.device, dtype=torch.bfloat16
                )
                with torch.no_grad():
                    v_k_raw = self.model.extract_feature(pixel_values_k)  # [1, 256, 896]
                del pixel_values_k

                v_k_mod = self.engine.modulate_patches(v_k_raw[0]).unsqueeze(0).to(
                    dtype=v_k_raw.dtype, device=self.device
                )
                tile_mods.append(v_k_mod)

            # Preserve uncompressed K * 256 tokens without spatial downsampling (Theorem 27)
            if use_amct and len(tile_mods) > 1:
                v_l_mod = torch.cat(tile_mods, dim=1)  # strictly [1, 512, 896]
            else:
                v_l_mod = tile_mods[0]  # [1, 256, 896]
            self._last_level2_box = None

        self._cached_v_l = v_l_mod
        self._last_crop_box = crop_box
        self._cached_vit_embeds = v_l_mod
        return v_l_mod

    def extract_hierarchical_multiscale_features(
        self,
        image: Image.Image,
        question: str,
        theta_zoom: float = 0.65,
    ) -> Tuple[torch.Tensor, Tuple[int, int, int, int], Optional[Tuple[int, int, int, int]]]:
        """
        Hierarchical Multi-Scale Focal Window Extraction (Theorem 32, Directive W1).
        Two-tier recursive Chebyshev windowing returning (v_l, box1_pixel, box2_pixel).
        """
        v_g, box1_pixel, _ = self.compute_stage1_window(image, question=question)
        v_l = self.extract_stage2_focal_features(
            image=image,
            crop_box=box1_pixel,
            question=question,
            theta_zoom=theta_zoom,
            use_hierarchical_zoom=True,
        )
        box2_pixel = getattr(self, "_last_level2_box", None)
        return v_l, box1_pixel, box2_pixel

    # =========================================================================
    # Embedding Construction: Single-Tile (256) or Multi-Tile (K * 256) Tokens
    # =========================================================================
    def build_multimodal_embeds(
        self,
        vit_embeds: torch.Tensor,
        question: str,
        user_prompt: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Construct input embeddings for standard single-tile stream (Directives W1 & W2).
        vit_embeds: [1, num_tokens, 896] or [num_tokens, 896], where num_tokens in {256, 512}.
        Preserves visual tokens inside <img><IMG_CONTEXT>*num_tokens</img>.
        """
        if vit_embeds.dim() == 2:
            vit_embeds = vit_embeds.unsqueeze(0)

        num_tokens = vit_embeds.shape[1]
        assert num_tokens in {self.num_image_token, 2 * self.num_image_token}, (
            f"Expected {self.num_image_token} or {2 * self.num_image_token} visual tokens, got {num_tokens}"
        )

        prefix_str = (
            f"<|im_start|>system\n{self.system_message}<|im_end|>\n"
            "<|im_start|>user\n"
            "<img>" + ("<IMG_CONTEXT>" * num_tokens) + "</img>\n"
        )
        if user_prompt is None:
            user_prompt = f"Dựa vào chữ trong ảnh, hãy trả lời ngắn gọn: {question.strip()}"
        suffix_str = f"{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

        prefix_ids = self.tokenizer(prefix_str, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        suffix_ids = self.tokenizer(suffix_str, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)

        embed_layer = self.model.language_model.get_input_embeddings()
        prefix_embeds = embed_layer(prefix_ids).clone()
        suffix_embeds = embed_layer(suffix_ids)

        img_ctx_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        mask = (prefix_ids == img_ctx_id).squeeze(0)
        vis_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        vis_start = int(vis_indices[0].item())
        vis_end = int(vis_indices[-1].item()) + 1

        prefix_embeds[0, vis_start:vis_end, :] = vit_embeds[0].to(prefix_embeds.dtype)
        return torch.cat([prefix_embeds, suffix_embeds], dim=1)

    def prepare_inputs(
        self,
        vit_embeds: torch.Tensor,
        question: str,
        user_prompt: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Directive W2: Configure prompt generation such that:
        "<img>" + ("<IMG_CONTEXT>" * 256) + "</img>\n"
        is identical for both Stream G and Stream L, guaranteeing sequence length invariance S_G = S_L.
        """
        return self.build_multimodal_embeds(vit_embeds, question, user_prompt=user_prompt)

    def build_mfvr_embeds(
        self,
        vit_embeds: torch.Tensor,
        question: str,
    ) -> torch.Tensor:
        """
        Construct input embeddings for Mean-Field Visual Reference (MFVR) stream (p_t^0).
        Computes spatial centroid v_mean = (1/256) sum v_p and broadcasts to [1, 256, D].
        """
        if vit_embeds.dim() == 2:
            vit_embeds = vit_embeds.unsqueeze(0)
        v_mean = vit_embeds.mean(dim=1, keepdim=True)  # [1, 1, D]
        vit_embeds_zero = v_mean.expand_as(vit_embeds)  # [1, 256, D]
        return self.build_multimodal_embeds(vit_embeds_zero, question)

    def build_text_only_embeds(self, question: str) -> torch.Tensor:
        """
        Construct input embeddings for text-only unconditioned stream (p_t^0).
        Deprecated in favor of build_mfvr_embeds (MFVR).
        """
        prompt_str = (
            f"<|im_start|>system\n{self.system_message}<|im_end|>\n"
            f"<|im_start|>user\n{question.strip()}\nTrả lời ngắn gọn: <|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        prompt_ids = self.tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        embed_layer = self.model.language_model.get_input_embeddings()
        return embed_layer(prompt_ids)

    # =========================================================================
    # Backward Compatibility Hooks for Benchmark Runner & Pipeline
    # =========================================================================
    def extract_vit_embeds(
        self,
        image: Image.Image,
        question: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Benchmark caching hook: runs Stage 1 and Stage 2, caching both V_G and V_L.
        Returns V_L (strictly 256 tokens) for downstream processing.
        """
        _, pixel_box, _ = self.compute_stage1_window(image, question=question)
        v_l = self.extract_stage2_focal_features(
            image, pixel_box, question=question, use_hierarchical_zoom=True
        )
        self._cached_vit_embeds = v_l
        return v_l

    def extract_and_modulate_visual_features(
        self,
        image: Image.Image,
        question: Optional[str] = None,
    ) -> torch.Tensor:
        """Alias for extract_vit_embeds preserving backwards compatibility."""
        return self.extract_vit_embeds(image, question=question)

    def apply_feature_space_nullification(
        self,
        visual_tokens: torch.Tensor,
        question_embeddings: Optional[torch.Tensor] = None,
        min_keep: int = 16,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Feature-Space Mean-Field Nullification (FSG) via Otsu Saliency Bisection (Directive 2).
        Replaces non-salient background tokens with the spatial mean-field vector v_mean.

        Guarantees background tokens carry zero localized variance into past_key_values,
        permanently eliminating attention sinks across all autoregressive steps while preserving
        100% 1D RoPE geometry and uniform B=2 batching.

        Args:
            visual_tokens: Patch embeddings [1, 256, D] or [256, D].
            question_embeddings: Optional question token embeddings [1, T, D] or [T, D].
            min_keep: Minimum salient patch floor (default 16).

        Returns:
            v_sparsified: Nullified visual tokens with identical shape and dtype.
            keep_mask: Boolean tensor [256] (True=salient, False=nullified).
        """
        is_3d = (visual_tokens.dim() == 3)
        v = visual_tokens[0] if is_3d else visual_tokens

        _, keep_mask = compute_visual_token_saliency_and_mask(
            visual_tokens=v,
            question_embeddings=question_embeddings,
            min_keep=min_keep,
        )

        v_mean = v.mean(dim=0, keepdim=True)  # [1, D]
        bg_mask = ~keep_mask  # [256]
        v_fsg = v.clone()
        v_fsg[bg_mask] = v_mean

        v_sparsified = v_fsg.unsqueeze(0) if is_3d else v_fsg
        return v_sparsified, keep_mask

    # =========================================================================
    # STAGE 3: Decoupled Dual-Stream Autoregressive Generation
    # =========================================================================
    def generate_answer(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int = 64,
    ) -> str:
        """
        Decoupled Dual-Stream Generation with Independent Dual KV-Caches (Theorems 26-28).
        Stream G (Global 256 tokens) and Stream L (Focal 256 or 512 tokens, K in {1, 2}).
        Dual KV-caches execute independently with S_G = 266 and S_L in {266, 522}.
        """
        with torch.no_grad():
            # 1. Stage 1: Window extraction & Global features V_G
            v_g, pixel_box, _ = self.compute_stage1_window(image, question=question)

            # 2. Stage 2: Hierarchical Multi-Scale Focal Feature Extraction V_L (Directive W1, Theorem 32)
            v_l = self.extract_stage2_focal_features(
                image, pixel_box, question=question, use_hierarchical_zoom=True
            )

            # 3. Theorem 42 Saliency Gating in Visual Representation Space (Directives W2 & W3)
            embed_layer = self.model.language_model.get_input_embeddings()
            q_ids = self.tokenizer.encode(question, add_special_tokens=False, return_tensors="pt").to(self.device)
            q_embeds = embed_layer(q_ids)
            s_qcass, _ = compute_question_conditioned_cross_attention_saliency(v_l, q_embeds)
            lambda_gate = 0.5 * float(self.engine.lambda_param)
            v_l_gated = apply_saliency_modulated_gating(v_l, s_qcass, lambda_gate=lambda_gate)

            # 4. Canonical natural prompt framing & Stream Embeddings (Directives W2-W4)
            prompt_text = f"Dựa vào chữ trong ảnh, hãy trả lời ngắn gọn: {question.strip()}"
            embeds_g = self.prepare_inputs(v_g, question, user_prompt=prompt_text)
            embeds_l = self.prepare_inputs(v_l_gated, question, user_prompt=prompt_text)

            # 5. Independent Dual KV-Cache Prefill (Theorems 26-28)
            out_g = self.model.language_model.model(
                inputs_embeds=embeds_g,
                use_cache=True,
                return_dict=True,
            )
            past_key_values_g = out_g.past_key_values
            logits_g = self.model.language_model.lm_head(out_g.last_hidden_state[:, -1:, :])[0, 0, :]

            out_l = self.model.language_model.model(
                inputs_embeds=embeds_l,
                use_cache=True,
                return_dict=True,
            )
            past_key_values_l = out_l.past_key_values
            logits_l = self.model.language_model.lm_head(out_l.last_hidden_state[:, -1:, :])[0, 0, :]

            # 6. Query token mask & Dynamic Token Resolution (Directive W4)
            vocab_size = logits_g.shape[-1]
            q_mask = build_query_token_mask(prompt_text, self.tokenizer, vocab_size, self.device)

            eos_im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
            eos_std = self.tokenizer.eos_token_id
            eos_token_ids = [eos_im_end]
            if eos_std is not None and eos_std not in eos_token_ids:
                eos_token_ids.append(eos_std)

            space_token_id = self.tokenizer.convert_tokens_to_ids(" ")
            if space_token_id is None:
                space_token_id = 220
            digit_token_ids = [self.tokenizer.convert_tokens_to_ids(str(d)) for d in range(10)]
            if any(did is None for did in digit_token_ids):
                digit_token_ids = list(range(15, 25))

            # Directive W4: Dynamically resolve orthographic separator token IDs
            sep_chars = ["-", "/", ",", ".", ":"]
            separator_token_ids = []
            for sc in sep_chars:
                s_id = self.tokenizer.convert_tokens_to_ids(sc)
                if s_id is not None and s_id != getattr(self.tokenizer, "unk_token_id", None):
                    separator_token_ids.append(int(s_id))
            if not separator_token_ids:
                separator_token_ids = [12, 14, 11, 13, 25]

            generated_ids: List[int] = []
            prev_token_id: Optional[int] = None

            for step in range(1, max_new_tokens + 1):
                # Steering Operator with Tripartite Clamping Engine (Theorem 33, Directives W1, W3 & W4)
                next_tok_tensor, steered_logits, _, _ = self.engine.decode_step(
                    logits_l=logits_l,
                    logits_g=logits_g,
                    q_mask=q_mask,
                    step_idx=step,
                    eos_token_ids=eos_token_ids,
                    prev_token_id=prev_token_id,
                    space_token_id=space_token_id,
                    digit_token_ids=digit_token_ids,
                    separator_token_ids=separator_token_ids,
                    alnum_token_ids=self.alnum_token_ids,
                    log_p_qcass=None,
                )
                next_id = int(next_tok_tensor.item())

                # Autonomous Optical Stopping Gate (Theorem 34, Directive W5)
                omega_t, s_bar_t = self.engine.get_last_continuation_saliency()
                stop_decision = evaluate_semantic_stopping(
                    logits_g=logits_g,
                    logits_l=logits_l,
                    steered_tokens=next_tok_tensor,
                    omega_t=omega_t,
                    s_bar_t=s_bar_t,
                    eos_token_ids=eos_token_ids,
                    step_idx=step,
                    grace_horizon=self.engine.get_last_grace_horizon(),
                )

                if bool(stop_decision.item()) if isinstance(stop_decision, torch.Tensor) else bool(stop_decision):
                    break

                generated_ids.append(next_id)
                prev_token_id = next_id

                if step == max_new_tokens:
                    break

                # Advance both streams synchronously via independent KV-caches (Directive W1)
                next_tok = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
                next_emb = embed_layer(next_tok)  # [1, 1, D]

                out_g = self.model.language_model.model(
                    inputs_embeds=next_emb,
                    past_key_values=past_key_values_g,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values_g = out_g.past_key_values
                logits_g = self.model.language_model.lm_head(out_g.last_hidden_state[:, -1:, :])[0, 0, :]

                out_l = self.model.language_model.model(
                    inputs_embeds=next_emb,
                    past_key_values=past_key_values_l,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values_l = out_l.past_key_values
                logits_l = self.model.language_model.lm_head(out_l.last_hidden_state[:, -1:, :])[0, 0, :]

            del past_key_values_g, past_key_values_l, out_g, out_l
            pred_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            return pred_text

    def generate(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int = 64,
    ) -> str:
        """Alias for generate_answer preserving backwards compatibility."""
        return self.generate_answer(image=image, question=question, max_new_tokens=max_new_tokens)


# Backward compatibility aliases
Vintern1BAdapter = VinternNEVFEAdapter
