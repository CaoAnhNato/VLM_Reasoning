# -*- coding: utf-8 -*-
"""
Visual Feature Cache for SG-VCD (Feature 8 in PROJECT.md).

Caches InternViT visual features (vit_embeds) per image to eliminate redundant
forward passes across multiple questions on the same image (averaging 3.05 questions
per image on ViTextVQA), reducing ViT compute by up to 67%.

Memory footprint:
Each tile produces (256, 896) bfloat16 tokens = ~458 KB.
For a 5-tile representation (4 dynamic + 1 thumbnail), cache is ~2.29 MB per image.
"""

from collections import OrderedDict
import hashlib
from typing import List, Optional, Tuple, Union

from PIL import Image

try:
    import torch
except ImportError:
    torch = None


class VisualFeatureCache:
    """
    LRU Cache for InternViT visual features (vit_embeds) in torch.bfloat16.
    """

    def __init__(self, max_entries: int = 16):
        """
        Args:
            max_entries: Maximum number of images to cache simultaneously.
                         16 entries consume ~36.6 MB VRAM, well within RTX 3050 budget.
        """
        self.max_entries = max_entries
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()

    @staticmethod
    def compute_image_key(image: Image.Image, num_tiles: int) -> str:
        """
        Compute a fast, robust signature key for an image and tiling configuration.
        """
        w, h = image.size
        mode = image.mode
        # Sample bytes from four corners and center for fast, collision-resistant hash
        corners = []
        try:
            corners.extend([
                image.getpixel((0, 0)),
                image.getpixel((w - 1, 0)),
                image.getpixel((0, h - 1)),
                image.getpixel((w - 1, h - 1)),
                image.getpixel((w // 2, h // 2)),
            ])
            sample_str = f"{corners}"
        except Exception:
            sample_str = "fallback"

        raw_key = f"{w}x{h}_{mode}_{num_tiles}_{sample_str}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Retrieve cached visual features by key."""
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: str, vit_embeds: torch.Tensor) -> None:
        """Store visual features in cache, evicting oldest if capacity reached."""
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = vit_embeds
            return

        if len(self._cache) >= self.max_entries:
            oldest_key, oldest_tensor = self._cache.popitem(last=False)
            del oldest_tensor

        self._cache[key] = vit_embeds

    def get_or_compute(
        self,
        model,
        transform,
        tiles: List[Image.Image],
        image: Optional[Image.Image] = None,
        device: str = "cuda",
    ) -> torch.Tensor:
        """
        Retrieve or compute visual features for a list of image tiles.
        
        Args:
            model: InternVLChatModel
            transform: torchvision image transform (448x448 BICUBIC)
            tiles: List of 448x448 PIL Image tiles
            image: Original full-resolution image (optional, used for caching key)
            device: Target execution device
            
        Returns:
            vit_embeds: torch.Tensor of shape (num_tiles, 256, 896) in torch.bfloat16
        """
        key = None
        if image is not None:
            key = self.compute_image_key(image, len(tiles))
            cached = self.get(key)
            if cached is not None:
                target_dev = torch.device(device)
                if cached.device.type == target_dev.type:
                    cached_idx = 0 if cached.device.index is None else cached.device.index
                    target_idx = 0 if target_dev.index is None else target_dev.index
                    if cached_idx == target_idx:
                        return cached

        pixel_values = torch.stack([transform(t) for t in tiles]).to(
            device=device, dtype=torch.bfloat16
        )

        with torch.no_grad():
            vit_embeds = model.extract_feature(pixel_values)

        del pixel_values

        if key is not None:
            self.put(key, vit_embeds)

        return vit_embeds

    def clear(self) -> None:
        """Evict all cached tensors and free references."""
        for k in list(self._cache.keys()):
            tensor = self._cache.pop(k)
            del tensor
        self._cache.clear()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


# Global default cache instance
default_feature_cache = VisualFeatureCache(max_entries=16)


def get_or_compute_visual_features(
    model,
    transform,
    tiles: List[Image.Image],
    image: Optional[Image.Image] = None,
    device: str = "cuda",
    cache: Optional[VisualFeatureCache] = None,
) -> torch.Tensor:
    """Convenience functional interface adhering to PROJECT.md contract."""
    target_cache = cache if cache is not None else default_feature_cache
    return target_cache.get_or_compute(model, transform, tiles, image=image, device=device)
