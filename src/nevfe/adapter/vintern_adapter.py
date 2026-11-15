# -*- coding: utf-8 -*-
r"""
Vintern-1B-v3_5 Adapter for Information-Geometric Variational Free Energy Decoding (IG-VFED).

Handles:
- Dynamic Aspect-Ratio Tiling & 100% Native Unwarped Patch Extraction.
- Positive Stream (p_t^+): 100% native unwarped visual patch embeddings entering LLM Layer 0.
- Reference Stream (p_t^0): Mean-Field Visual Reference (MFVR, V^0 = 1_P \bar{v}^\top).
- Synchronized KV-cache progression across both streams.
- Safe try...finally forward-hook lifecycle for Qwen2 sdpa introspection.
"""

from contextlib import contextmanager
import math
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from src.nevfe.core.engine import (
    AlgorithmConfig,
    NEVFEEngine,
    assemble_foveated_visual_sequence,
    compute_mean_field_reference_embeddings,
    nevfe_step_logits,
)


PROMPT_INSTRUCTION = "Trả lời thật ngắn gọn câu hỏi sau dựa vào chữ xuất hiện trong ảnh: "


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


def _build_target_ratios(min_num: int, max_num: int) -> List[Tuple[int, int]]:
    return sorted(
        set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        ),
        key=lambda x: x[0] * x[1],
    )


def dynamic_preprocess_vintern(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 6,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> List[Image.Image]:
    """Standard dynamic aspect-ratio tiling for Vintern / InternVL models."""
    orig_w, orig_h = image.size
    aspect_ratio = orig_w / max(1, orig_h)
    target_ratios = _build_target_ratios(min_num, max_num)

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
    """Construct Query Indicator Tensor m_Q in {0, 1}^|V| invariant to casing and BPE prefixes."""
    variants = [
        question,
        question.lower(),
        question.upper(),
        question.capitalize(),
        question.title(),
        " " + question.strip(),
        " " + question.lower().strip(),
    ]
    for w in question.strip().split():
        variants.extend([
            w,
            w.lower(),
            w.upper(),
            w.capitalize(),
            w.title(),
            " " + w,
            " " + w.lower(),
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


@contextmanager
def managed_forward_hooks(
    hook_specs: List[Tuple[Any, Callable[..., Any]]],
) -> Iterator[List[Any]]:
    """
    Register forward hooks inside a try...finally block so handles are always removed,
    even under Qwen2 sdpa (attentions=None) or inference exceptions.
    Reference: scripts/benchmark/sg_vcd/attention_extractor.py.
    """
    handles: List[Any] = []
    try:
        for module, hook_fn in hook_specs:
            if hasattr(module, "register_forward_hook"):
                handles.append(module.register_forward_hook(hook_fn))
        yield handles
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass


class VinternNEVFEAdapter:
    """
    Vintern-1B-v3_5 VLM Adapter driving Information-Geometric Variational Free Energy Decoding.
    """

    def __init__(
        self,
        model,
        tokenizer,
        transform=None,
        device: str = "cuda",
        lambda_param: float = 0.50,
        beta_param: float = 0.05,
        gamma_param: float = 0.50,
        config: Optional[AlgorithmConfig] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        model_cfg = getattr(model, "config", None)
        vision_cfg = getattr(model_cfg, "vision_config", None)
        self.decoder_config = getattr(
            getattr(model, "language_model", model),
            "config",
            model_cfg,
        )
        self.force_image_size = int(
            getattr(model_cfg, "force_image_size", None)
            or getattr(vision_cfg, "image_size", None)
            or 448
        )
        self.transform = transform or self._build_default_transform(input_size=self.force_image_size)

        resolved_cfg = (
            config
            if config is not None
            else AlgorithmConfig(
                lambda_param=lambda_param,
                beta_param=beta_param,
                gamma_param=gamma_param,
            )
        )
        self.engine = NEVFEEngine(config=resolved_cfg)

        self.max_dynamic_patch = getattr(model_cfg, "max_dynamic_patch", 6)
        self.num_image_token = getattr(model, "num_image_token", 256)
        self.system_message = getattr(model, "system_message", "Bạn là một trợ lý AI hữu ích.")

        # Immutable per-image visual cache
        self._cached_img_key: Optional[Tuple[int, int, bytes]] = None
        self._cached_vit_embeds: Optional[torch.Tensor] = None

    def set_ablation_switches(
        self,
        comp_a: bool = True,
        comp_b: bool = True,
        comp_c: bool = True,
    ) -> None:
        """Dynamically configure micro-ablation toggles for Factorial Interaction testing."""
        self.engine.set_ablation_switches(comp_a=comp_a, comp_b=comp_b, comp_c=comp_c)

    @staticmethod
    def _build_default_transform(input_size: int = 448):
        try:
            import torchvision.transforms as T
            from torchvision.transforms.functional import InterpolationMode

            return T.Compose([
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        except ImportError:
            import numpy as np

            mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

            def _fallback_transform(img: Image.Image) -> torch.Tensor:
                rgb = img.convert("RGB") if img.mode != "RGB" else img
                resized = rgb.resize((input_size, input_size), Image.Resampling.BICUBIC)
                arr = torch.from_numpy(np.asarray(resized, dtype=np.float32) / 255.0).permute(2, 0, 1)
                return (arr - mean) / std

            return _fallback_transform

    @staticmethod
    def _make_image_key(image: Image.Image) -> Tuple[int, int, bytes]:
        w, h = image.size
        thumb = image.resize((16, 16), Image.Resampling.NEAREST).convert("RGB")
        return (w, h, thumb.tobytes())

    def extract_visual_features(self, image: Image.Image) -> torch.Tensor:
        """
        Extract 100% native unwarped visual patch embeddings from InternViT-300M.
        Preserves pretrained vision-language manifold entering LLM Layer 0.
        """
        img_key = self._make_image_key(image)
        if self._cached_img_key == img_key and self._cached_vit_embeds is not None:
            return self._cached_vit_embeds.detach()

        tiles = dynamic_preprocess_vintern(
            image,
            min_num=1,
            max_num=self.max_dynamic_patch,
            image_size=self.force_image_size,
            use_thumbnail=True,
        )

        pixel_values = torch.stack([self.transform(t) for t in tiles]).to(
            device=self.device, dtype=torch.bfloat16
        )

        with torch.no_grad():
            raw_vit_embeds = self.model.extract_feature(pixel_values)

        del pixel_values

        self._cached_img_key = img_key
        self._cached_vit_embeds = raw_vit_embeds.detach()
        return self._cached_vit_embeds.detach()

    def extract_and_modulate_visual_features(self, image: Image.Image) -> torch.Tensor:
        """Alias for backward compatibility; preserves unwarped native patch embeddings."""
        return self.extract_visual_features(image)

    def encode_transient_roi_crops(
        self,
        image: Image.Image,
        bboxes: List[Tuple[int, int, int, int]],
        max_tiles: int = 3,
        min_magnification: float = 0.0,
    ) -> Optional[torch.Tensor]:
        """
        Encode unblurred isotropic single-locus native RGB crops via Nyquist-gated multi-tile slicing
        using DynamicMultiTileROICropper.select_grid without mutating self._cached_vit_embeds.
        """
        if not bboxes or max_tiles <= 0:
            return None
        from scripts.benchmark.sg_vcd.roi_cropper import DynamicMultiTileROICropper

        rgb_img = image.convert("RGB")
        w_img, h_img = rgb_img.size
        if w_img <= 0 or h_img <= 0:
            return None

        remaining = int(max_tiles)
        min_tile_dim = max(1, self.force_image_size // 4)
        crop_tiles: List[Image.Image] = []

        for bx1, by1, bx2, by2 in bboxes:
            if remaining <= 0:
                break
            cbx1, cbx2 = round(min(bx1, bx2)), round(max(bx1, bx2))
            cby1, cby2 = round(min(by1, by2)), round(max(by1, by2))
            raw_w = max(2, cbx2 - cbx1)
            raw_h = max(2, cby2 - cby1)
            pad_x = round(0.15 * raw_w)
            pad_y = max(round(0.15 * raw_h), 2)
            min_dx = min(2, w_img)
            min_dy = min(2, h_img)
            x1 = max(0, min(w_img - min_dx, cbx1 - pad_x))
            y1 = max(0, min(h_img - min_dy, cby1 - pad_y))
            x2 = max(x1 + min_dx, min(w_img, cbx2 + pad_x))
            y2 = max(y1 + min_dy, min(h_img, cby2 + pad_y))
            dx = max(min_dx, x2 - x1)
            dy = max(min_dy, y2 - y1)

            cropper = DynamicMultiTileROICropper(
                target_size=self.force_image_size,
                max_tiles=remaining,
                min_tile_dim=min_tile_dim,
            )
            gx_star, gy_star = cropper.select_grid(dx, dy)

            s_unit = max(dx / gx_star, dy / gy_star)
            s_min = min(32.0, float(min(w_img, h_img)))
            s_floor = s_min + (0.05 * float(min(w_img, h_img))) / (
                1.0 + math.exp(-0.05 * (max(dx, dy) - s_min))
            )
            s_unit = min(max(s_unit, s_floor), float(w_img) / gx_star, float(h_img) / gy_star)
            s_x = min(float(w_img), max(float(dx), s_unit * gx_star))
            s_y = min(float(h_img), max(float(dy), s_unit * gy_star))

            m_crop = min(
                (self.force_image_size * gx_star) / max(1.0, s_x),
                (self.force_image_size * gy_star) / max(1.0, s_y),
            )
            if min_magnification > 0.0 and m_crop <= min_magnification:
                continue

            mu_x = 0.5 * (x1 + x2)
            mu_y = 0.5 * (y1 + y2)
            sx1 = max(0.0, min(float(w_img) - s_x, mu_x - 0.5 * s_x))
            sy1 = max(0.0, min(float(h_img) - s_y, mu_y - 0.5 * s_y))
            tile_w = s_x / gx_star
            tile_h = s_y / gy_star

            for ty in range(gy_star):
                for tx in range(gx_star):
                    tx1 = max(0, min(w_img - 1, int(round(sx1 + tx * tile_w))))
                    ty1 = max(0, min(h_img - 1, int(round(sy1 + ty * tile_h))))
                    tx2 = max(tx1 + 1, min(w_img, int(round(sx1 + (tx + 1) * tile_w))))
                    ty2 = max(ty1 + 1, min(h_img, int(round(sy1 + (ty + 1) * tile_h))))
                    sub = rgb_img.crop((tx1, ty1, tx2, ty2)).resize(
                        (self.force_image_size, self.force_image_size),
                        Image.BICUBIC,
                    )
                    crop_tiles.append(sub)
            remaining -= gx_star * gy_star

        if not crop_tiles:
            return None
        pixel_values = torch.stack([self.transform(t) for t in crop_tiles]).to(
            device=self.device, dtype=torch.bfloat16
        )
        with torch.no_grad():
            crop_embeds = self.model.extract_feature(pixel_values)
        del pixel_values
        return crop_embeds.detach()

    def _resolve_tile_grid(
        self,
        image: Image.Image,
        n_tiles: int,
    ) -> Tuple[int, int, int]:
        """Resolve spatial tile count B and grid (g_x, g_y) with mock fallback (B, 1)."""
        b = 1 if n_tiles == 1 else n_tiles - 1
        if b == 1:
            return 1, 1, 1
        orig_w, orig_h = image.size
        aspect_ratio = orig_w / max(1, orig_h)
        target_ratios = _build_target_ratios(1, self.max_dynamic_patch)
        gx, gy = _find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_w, orig_h, self.force_image_size
        )
        if b != gx * gy:
            return b, b, 1
        return b, gx, gy

    def _extract_lattice(
        self,
        image: Image.Image,
        vit_embeds: torch.Tensor,
    ) -> Tuple[torch.Tensor, int, int]:
        """Assemble unwarped pure-semantic spatial patch lattice [H_g, W_g, D] without pixel filtering."""
        b, gx, gy = self._resolve_tile_grid(image, int(vit_embeds.shape[0]))
        p_tile, dim = int(vit_embeds.shape[1]), int(vit_embeds.shape[2])
        p_side = int(round(math.sqrt(p_tile)))
        if p_side * p_side != p_tile:
            raise ValueError(f"Expected square patch token count per tile, got {p_tile}")
        h_g, w_g = p_side * gy, p_side * gx
        patch_embeds = (
            vit_embeds[:b]
            .detach()
            .reshape(gy, gx, p_side, p_side, dim)
            .permute(0, 2, 1, 3, 4)
            .reshape(h_g, w_g, dim)
        )
        img_w, img_h = image.size
        return patch_embeds, int(img_w), int(img_h)

    def _extract_query_embeds(
        self,
        question: str,
        target_device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Extract prompt-centered Layer-0 query embeddings [M, D] in float32."""
        q_text = question.strip()
        if not q_text:
            return None
        q_tok = self.tokenizer.encode(q_text, add_special_tokens=False)
        if not q_tok:
            return None
        lm = getattr(self.model, "language_model", None)
        get_emb = getattr(lm, "get_input_embeddings", None)
        if not callable(get_emb):
            return None
        embed_layer = get_emb()
        q_ids = torch.tensor(q_tok, dtype=torch.long, device=target_device).unsqueeze(0)
        q_raw = embed_layer(q_ids).squeeze(0).float()
        p_tok = self.tokenizer.encode(PROMPT_INSTRUCTION, add_special_tokens=False)
        if p_tok:
            p_ids = torch.tensor(p_tok, dtype=torch.long, device=target_device).unsqueeze(0)
            p_mean = embed_layer(p_ids).squeeze(0).float().mean(dim=0, keepdim=True)
            return q_raw - p_mean
        return q_raw

    def extract_foveated_roi_crops(
        self,
        image: Image.Image,
        vit_embeds: torch.Tensor,
        question: str,
    ) -> Optional[torch.Tensor]:
        """Run SKG-TTAF pure-semantic active sensing and extract transient RGB sub-tiles without mutating vit_embeds."""
        patch_embeds, img_w, img_h = self._extract_lattice(image, vit_embeds)
        query_embeds = self._extract_query_embeds(question, vit_embeds.device)
        box, j_star, _ = self.engine.foveate(patch_embeds, query_embeds=query_embeds)
        if j_star <= 1e-6:
            return None
        _, gx_glob, gy_glob = self._resolve_tile_grid(image, int(vit_embeds.shape[0]))
        m_glob = min(
            (self.force_image_size * gx_glob) / max(1, img_w),
            (self.force_image_size * gy_glob) / max(1, img_h),
        )
        h_g, w_g = int(patch_embeds.shape[0]), int(patch_embeds.shape[1])
        i1, j1, i2, j2 = box
        x1 = int(round(j1 * img_w / w_g))
        y1 = int(round(i1 * img_h / h_g))
        x2 = int(round(j2 * img_w / w_g))
        y2 = int(round(i2 * img_h / h_g))
        return self.encode_transient_roi_crops(
            image, [(x1, y1, x2, y2)], max_tiles=3, min_magnification=m_glob
        )

    def build_multimodal_embeds(
        self,
        vit_embeds: torch.Tensor,
        question: str,
        extra_vit_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Construct input embeddings for positive multimodal stream (p_t^+).
        Splices transient crop features per question without mutating vit_embeds.
        """
        effective_vit = assemble_foveated_visual_sequence(
            vit_embeds, extra_vit_embeds, comp_c=self.engine.comp_c
        )
        n_tiles = effective_vit.shape[0]
        total_img_tokens = n_tiles * self.num_image_token

        prefix_str = (
            f"<|im_start|>system\n{self.system_message}<|im_end|>\n"
            "<|im_start|>user\n"
            "<img>" + ("<IMG_CONTEXT>" * total_img_tokens) + "</img>\n"
        )
        suffix_str = f"{PROMPT_INSTRUCTION}{question.strip()}<|im_end|>\n<|im_start|>assistant\n"

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

        flat_vit = effective_vit.reshape(-1, effective_vit.shape[-1]).to(prefix_embeds.dtype)
        prefix_embeds[0, vis_start:vis_end, :] = flat_vit

        return torch.cat([prefix_embeds, suffix_embeds], dim=1)

    def build_mfvr_embeds(
        self,
        vit_embeds: torch.Tensor,
        question: str,
        extra_vit_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Construct input embeddings for Mean-Field Visual Reference (MFVR) stream (p_t^0)."""
        effective_vit = assemble_foveated_visual_sequence(
            vit_embeds, extra_vit_embeds, comp_c=self.engine.comp_c
        )
        vit_embeds_zero = compute_mean_field_reference_embeddings(effective_vit)
        return self.build_multimodal_embeds(vit_embeds_zero, question)

    def build_text_only_embeds(self, question: str) -> torch.Tensor:
        """Construct input embeddings for text-only unconditioned stream (legacy helper)."""
        prompt_str = (
            f"<|im_start|>system\n{self.system_message}<|im_end|>\n"
            f"<|im_start|>user\n{PROMPT_INSTRUCTION}{question.strip()}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        prompt_ids = self.tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        embed_layer = self.model.language_model.get_input_embeddings()
        return embed_layer(prompt_ids)

    def generate(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int = 64,
    ) -> str:
        """Dual-Stream Autoregressive Generation with VRE-OF + locked IG-VFED."""
        with torch.no_grad():
            vit_embeds = self.extract_visual_features(image)
            extra_vit_embeds = self.extract_foveated_roi_crops(image, vit_embeds, question)

            multimodal_embeds = self.build_multimodal_embeds(
                vit_embeds, question, extra_vit_embeds=extra_vit_embeds
            )
            mfvr_embeds = self.build_mfvr_embeds(
                vit_embeds, question, extra_vit_embeds=extra_vit_embeds
            )

            embed_layer = self.model.language_model.get_input_embeddings()
            eos_im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
            eos_std = self.tokenizer.eos_token_id

            out_plus = self.model.language_model.model(
                inputs_embeds=multimodal_embeds,
                use_cache=True,
                return_dict=True,
            )
            kv_plus = out_plus.past_key_values
            logits_plus = self.model.language_model.lm_head(out_plus.last_hidden_state[:, -1:, :])[:, -1, :]

            out_zero = self.model.language_model.model(
                inputs_embeds=mfvr_embeds,
                use_cache=True,
                return_dict=True,
            )
            kv_zero = out_zero.past_key_values
            logits_zero = self.model.language_model.lm_head(out_zero.last_hidden_state[:, -1:, :])[:, -1, :]

            generated_ids: List[int] = []

            for step in range(1, max_new_tokens + 1):
                next_tok_tensor, _, _, _ = nevfe_step_logits(
                    logits_plus,
                    logits_zero,
                    lambda_param=self.engine.lambda_param,
                    beta_param=self.engine.beta_param,
                    q_mask=None,
                    step_idx=step,
                    comp_a=True,
                    comp_b=True,
                    comp_c=True,
                )
                next_id = int(next_tok_tensor.item())

                if next_id == eos_im_end or (eos_std is not None and next_id == eos_std):
                    break

                generated_ids.append(next_id)

                if step == max_new_tokens:
                    break

                next_tok = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
                next_emb = embed_layer(next_tok)

                out_plus = self.model.language_model.model(
                    inputs_embeds=next_emb,
                    past_key_values=kv_plus,
                    use_cache=True,
                    return_dict=True,
                )
                kv_plus = out_plus.past_key_values
                logits_plus = self.model.language_model.lm_head(out_plus.last_hidden_state[:, -1:, :])[:, -1, :]

                out_zero = self.model.language_model.model(
                    inputs_embeds=next_emb,
                    past_key_values=kv_zero,
                    use_cache=True,
                    return_dict=True,
                )
                kv_zero = out_zero.past_key_values
                logits_zero = self.model.language_model.lm_head(out_zero.last_hidden_state[:, -1:, :])[:, -1, :]

            del kv_plus, kv_zero, out_plus, out_zero

            pred_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            return pred_text
