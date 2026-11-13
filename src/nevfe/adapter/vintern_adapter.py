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
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple
from PIL import Image
import torch

from src.nevfe.core.engine import (
    AlgorithmConfig,
    NEVFEEngine,
    compute_mean_field_reference_embeddings,
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
            else AlgorithmConfig(lambda_param=lambda_param, beta_param=beta_param)
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
    ) -> Optional[torch.Tensor]:
        """
        Encode transient question-guided native RGB crops via scripts/benchmark/sg_vcd/roi_cropper.py
        without mutating self._cached_vit_embeds.
        """
        if not bboxes:
            return None
        from scripts.benchmark.sg_vcd.roi_cropper import DynamicMultiTileROICropper

        cropper = DynamicMultiTileROICropper(target_size=self.force_image_size)
        crop_tiles: List[Image.Image] = []
        for box in bboxes:
            res = cropper.crop(image, box)
            crop_tiles.extend(res.roi_tiles)
        if not crop_tiles:
            return None
        pixel_values = torch.stack([self.transform(t) for t in crop_tiles]).to(
            device=self.device, dtype=torch.bfloat16
        )
        with torch.no_grad():
            crop_embeds = self.model.extract_feature(pixel_values)
        del pixel_values
        return crop_embeds.detach()

    def build_multimodal_embeds(
        self,
        vit_embeds: torch.Tensor,
        question: str,
        extra_vit_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Construct input embeddings for positive multimodal stream (p_t^+).
        Concatenates any transient crop features per question without mutating vit_embeds.
        """
        effective_vit = (
            torch.cat([vit_embeds, extra_vit_embeds], dim=0)
            if extra_vit_embeds is not None
            else vit_embeds
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
        effective_vit = (
            torch.cat([vit_embeds, extra_vit_embeds], dim=0)
            if extra_vit_embeds is not None
            else vit_embeds
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
        """Dual-Stream Autoregressive Generation with IG-VFED."""
        with torch.no_grad():
            vit_embeds = self.extract_visual_features(image)

            multimodal_embeds = self.build_multimodal_embeds(vit_embeds, question)
            mfvr_embeds = self.build_mfvr_embeds(vit_embeds, question)

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
                next_tok_tensor, _, _, _ = self.engine.decode_step(
                    logits_plus, logits_zero, q_mask=None, step_idx=step
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
