# -*- coding: utf-8 -*-
"""
Vintern-1B-v3_5 Adapter for NE-VFE Trajectory Steering.

Handles:
- Dynamic Aspect-Ratio Tiling & Visual Patch Feature Extraction.
- Continuous Fisher-Saliency Patch Energy Modulation on visual tokens.
- Dual-Stream Autoregressive Decoding:
  * Stream 1 (p_t^+): Multimodal prompt with modulated visual tokens.
  * Stream 2 (p_t^0): Pure text-only unconditioned prompt with ZERO visual markers (Directive 5).
- Synchronized KV-cache progression across both streams.
- Zero regex, zero stopwords, strictly <= 2 hyperparameters (lambda, beta).
"""

from typing import Dict, List, Optional, Tuple
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from src.nevfe.core.engine import (
    NEVFEEngine,
    compute_thermodynamic_spatial_window,
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


class VinternNEVFEAdapter:
    """
    Vintern-1B-v3_5 VLM Adapter driving QO-VFE Trajectory Steering with MFVR.
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
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.transform = transform or self._build_default_transform(input_size=448)

        self.engine = NEVFEEngine(
            lambda_param=lambda_param,
            beta_param=beta_param,
            mu_param=mu_param,
        )

        self.max_dynamic_patch = getattr(getattr(model, "config", None), "max_dynamic_patch", 6)
        self.num_image_token = getattr(model, "num_image_token", 256)
        self.system_message = getattr(model, "system_message", "Bạn là một trợ lý AI hữu ích.")

        # Cache for visual features per image
        self._cached_img_key: Optional[Tuple[int, int, bytes]] = None
        self._cached_vit_embeds: Optional[torch.Tensor] = None
        self._last_crop_box: Optional[Tuple[int, int, int, int]] = None
        self._last_spatial_window: Optional[Tuple[float, float, float, float]] = None

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

    def extract_vit_embeds(
        self,
        image: Image.Image,
        question: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Extract dual-scale visual features from InternViT-300M (Directive 2):
        1. Global scene tile V_G (full canvas downsampled to 448x448) -> [1, 256, 896].
        2. Thermodynamic Chebyshev spatial bounding window computed from V_G.
        3. Local focal crop V_L (optical magnification resized to 448x448) -> [1, 256, 896].
        4. Stack into dual-tile representation V in R^{2 x 256 x 896}.
        5. Modulate flattened patches [512, 896] via continuous Fisher-Saliency energy.
        """
        img_key = self._make_image_key(image)
        if self._cached_img_key == img_key and self._cached_vit_embeds is not None:
            return self._cached_vit_embeds

        W, H = image.size

        # a) Extract global features V_G from full image
        pixel_values_g = self.transform(image).unsqueeze(0).to(
            device=self.device, dtype=torch.bfloat16
        )
        with torch.no_grad():
            v_g = self.model.extract_feature(pixel_values_g)  # [1, 256, 896]
        del pixel_values_g

        # b) Compute Chebyshev crop window from V_G using compute_thermodynamic_spatial_window
        x_min, y_min, x_max, y_max = compute_thermodynamic_spatial_window(
            v_g[0],
            grid_size=16,
            k_sigma=self.engine.k_sigma,
            min_span=self.engine.min_span,
        )
        self._last_spatial_window = (x_min, y_min, x_max, y_max)

        # c) Crop PIL image to [x_min, y_min, x_max, y_max] and resize to 448x448
        x0 = max(0, min(W - 1, int(round(x_min * W))))
        y0 = max(0, min(H - 1, int(round(y_min * H))))
        x1 = max(x0 + 1, min(W, int(round(x_max * W))))
        y1 = max(y0 + 1, min(H, int(round(y_max * H))))
        self._last_crop_box = (x0, y0, x1, y1)

        focal_crop = image.crop((x0, y0, x1, y1))
        pixel_values_l = self.transform(focal_crop).unsqueeze(0).to(
            device=self.device, dtype=torch.bfloat16
        )

        # d) Extract focal features V_L
        with torch.no_grad():
            v_l = self.model.extract_feature(pixel_values_l)  # [1, 256, 896]
        del pixel_values_l

        # e) Stack into V in R^{2 x 256 x 896}
        v_raw = torch.cat([v_g, v_l], dim=0)  # [2, 256, 896]

        # Modulate flattened patches [512, 896] using self.engine.modulate_patches(flat_v)
        n_tiles, n_tokens, d_dim = v_raw.shape
        flat_v = v_raw.reshape(n_tiles * n_tokens, d_dim)
        modulated_flat = self.engine.modulate_patches(flat_v)
        modulated_vit = modulated_flat.reshape(n_tiles, n_tokens, d_dim).to(
            dtype=v_raw.dtype, device=self.device
        )

        self._cached_img_key = img_key
        self._cached_vit_embeds = modulated_vit
        return modulated_vit

    def extract_and_modulate_visual_features(
        self,
        image: Image.Image,
        question: Optional[str] = None,
    ) -> torch.Tensor:
        """Alias for extract_vit_embeds preserving backwards compatibility."""
        return self.extract_vit_embeds(image, question=question)

    def build_multimodal_embeds(
        self,
        vit_embeds: torch.Tensor,
        question: str,
    ) -> torch.Tensor:
        """
        Construct input embeddings for multimodal stream (p_t^+).
        """
        n_tiles = vit_embeds.shape[0]
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

        flat_vit = vit_embeds.reshape(-1, vit_embeds.shape[-1]).to(prefix_embeds.dtype)
        prefix_embeds[0, vis_start:vis_end, :] = flat_vit

        return torch.cat([prefix_embeds, suffix_embeds], dim=1)

    def build_mfvr_embeds(
        self,
        vit_embeds: torch.Tensor,
        question: str,
    ) -> torch.Tensor:
        """
        Construct input embeddings for Mean-Field Visual Reference (MFVR) stream (p_t^0).

        Evaluates the spatial centroid across all tiles and patch tokens:
            v_mean = (1 / P) \\sum_{p=1}^P v_p
        and broadcasts it across all P visual token positions:
            V^0 = v_mean \\mathbf{1}^\\top \\in \\mathbb{R}^{P \\times D}

        Guarantees:
        - Exact structural and positional alignment (|E^0| = |E^+|).
        - Persona Retention: VLM remains in VQA answer mode (I_1(preamble) ~ 0).
        - High-Frequency Oblivion: Spatial character strokes destroyed (I_1(entity) >> 0).
        - Zero extra hyperparameters, zero regex, zero stopwords.
        """
        # vit_embeds: [N_tiles, 256, D]
        # Compute spatial centroid across tiles (dim=0) and patch tokens (dim=1)
        v_mean = vit_embeds.mean(dim=(0, 1), keepdim=True) # [1, 1, D]
        vit_embeds_zero = v_mean.expand_as(vit_embeds) # [N_tiles, 256, D]
        return self.build_multimodal_embeds(vit_embeds_zero, question)

    def build_text_only_embeds(self, question: str) -> torch.Tensor:
        """
        Construct input embeddings for text-only unconditioned stream (p_t^0).
        Deprecated in favor of build_mfvr_embeds (MFVR).
        """
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
        """
        Dual-Stream Autoregressive Generation with NE-VFE Trajectory Steering and MFVR Reference.
        """
        with torch.no_grad():
            vit_embeds = self.extract_vit_embeds(image, question=question)

            multimodal_embeds = self.build_multimodal_embeds(vit_embeds, question)
            # Mean-Field Visual Reference (MFVR) stream embeddings
            mfvr_embeds = self.build_mfvr_embeds(vit_embeds, question)

            embed_layer = self.model.language_model.get_input_embeddings()
            eos_im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
            eos_std = self.tokenizer.eos_token_id

            # Prefill multimodal branch (p_t^+)
            out_plus = self.model.language_model.model(
                inputs_embeds=multimodal_embeds,
                use_cache=True,
                return_dict=True,
            )
            kv_plus = out_plus.past_key_values
            logits_plus = self.model.language_model.lm_head(out_plus.last_hidden_state[:, -1:, :])[:, -1, :]

            # Prefill Mean-Field Visual Reference unconditioned branch (p_t^0)
            out_zero = self.model.language_model.model(
                inputs_embeds=mfvr_embeds,
                use_cache=True,
                return_dict=True,
            )
            kv_zero = out_zero.past_key_values
            logits_zero = self.model.language_model.lm_head(out_zero.last_hidden_state[:, -1:, :])[:, -1, :]

            # Construct Query Indicator Tensor m_Q once before decoding loop (Directive 2 & 3)
            vocab_size = logits_plus.shape[-1]
            q_mask = build_query_token_mask(question, self.tokenizer, vocab_size, self.device)

            generated_ids: List[int] = []

            for step in range(1, max_new_tokens + 1):
                # Query-Orthogonal Variational Free Energy Single-Step Steering Operator
                next_tok_tensor, _, _, _ = self.engine.decode_step(
                    logits_plus, logits_zero, q_mask=q_mask, step_idx=step
                )
                next_id = int(next_tok_tensor.item())

                if next_id == eos_im_end or (eos_std is not None and next_id == eos_std):
                    break

                generated_ids.append(next_id)

                if step == max_new_tokens:
                    break

                # Advance both streams synchronously on the chosen token
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
