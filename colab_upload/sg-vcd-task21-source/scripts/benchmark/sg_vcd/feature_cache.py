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
from dataclasses import dataclass
import hashlib
import os
from typing import Any, List, Optional, Tuple, Union

from PIL import Image

try:
    import torch
except ImportError:
    torch = None


UNSCOPED_DATASET_NAMESPACE = "__unscoped_dataset__"
FALLBACK_IMAGE_IDENTITY = "__content_fallback__"


def resolve_precision_and_dtype(
    precision: Optional[str] = None,
    dtype: Optional["torch.dtype"] = None,
) -> Tuple[str, "torch.dtype"]:
    """
    Resolve explicit model precision ('bf16' | 'fp16') to canonical `(precision_name, torch.dtype)`.
    Preserves `bf16` / `torch.bfloat16` as the default when neither is provided.
    Never performs implicit fallback on unsupported precision specifications.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required to resolve model precision dtypes.")

    resolved_from_prec: Optional[Tuple[str, "torch.dtype"]] = None
    if precision is not None:
        norm = str(precision).strip().lower()
        if norm in ("bf16", "bfloat16"):
            resolved_from_prec = ("bf16", torch.bfloat16)
        elif norm in ("fp16", "float16"):
            resolved_from_prec = ("fp16", torch.float16)
        else:
            raise ValueError(
                f"Unsupported precision {precision!r}. Allowed values are 'bf16' and 'fp16' "
                f"(no implicit fallback permitted)."
            )

    resolved_from_dtype: Optional[Tuple[str, "torch.dtype"]] = None
    if dtype is not None:
        if dtype == torch.bfloat16:
            resolved_from_dtype = ("bf16", torch.bfloat16)
        elif dtype == torch.float16:
            resolved_from_dtype = ("fp16", torch.float16)
        else:
            raise ValueError(
                f"Unsupported torch_dtype {dtype!r}. Allowed dtypes are torch.bfloat16 and torch.float16 "
                f"(no implicit fallback permitted)."
            )

    if resolved_from_prec is not None and resolved_from_dtype is not None:
        if resolved_from_prec != resolved_from_dtype:
            raise ValueError(
                f"Conflicting precision ({precision!r}) and torch_dtype ({dtype!r})."
            )
        return resolved_from_prec

    if resolved_from_prec is not None:
        return resolved_from_prec
    if resolved_from_dtype is not None:
        return resolved_from_dtype
    return ("bf16", torch.bfloat16)


@dataclass(frozen=True)
class ImageCacheIdentity:
    """
    Canonical collision-safe cache identity contract for an image (OPT-06).

    Conceptual structure:
        (dataset_version, image_id, width, height, content_fingerprint)

    When stable identity metadata (`dataset_version` / `dataset_namespace`, `image_id`,
    or directory-scoped `image_path`) is omitted, `is_fallback` is set to True and
    the key falls back to content-based thumbnail hashing without claiming
    collision-proof cross-dataset identity.
    """

    dataset_version: str
    image_identity: str
    width: int
    height: int
    content_fingerprint: str
    is_fallback: bool = False

    def as_tuple(self) -> Tuple[str, str, int, int, str]:
        """Return the canonical 5-tuple `(dataset_version, image_id, width, height, content_fingerprint)`."""
        return (
            self.dataset_version,
            self.image_identity,
            self.width,
            self.height,
            self.content_fingerprint,
        )

    def __iter__(self):
        return iter(self.as_tuple())

    def __len__(self) -> int:
        return 5

    def __getitem__(self, idx: int) -> Any:
        return self.as_tuple()[idx]

    def to_cache_key(self, num_tiles: Optional[int] = None) -> str:
        """Serialize identity into a deterministic string key for dictionary lookup."""
        base = (
            f"ds={self.dataset_version}|img={self.image_identity}|"
            f"dim={self.width}x{self.height}|fp={self.content_fingerprint}|"
            f"fb={int(self.is_fallback)}"
        )
        if num_tiles is not None:
            return f"{base}|tiles={int(num_tiles)}"
        return base


def _normalize_relative_image_path(
    image_path: Optional[str],
    dataset_root: Optional[str] = None,
) -> Tuple[Optional[str], bool]:
    """
    Normalize an image path into a portable dataset-relative POSIX path without
    relying on machine-specific absolute root prefixes or collapsing subdirectories
    into a bare basename (OPT-06 / M-01).

    Behavior:
    1. When `dataset_root` is supplied and `image_path` resolves inside `dataset_root`
       (without escaping via `..`), returns the complete relative POSIX path directly
       at full directory depth without applying anchor searches or depth truncation.
    2. When `dataset_root` is omitted or `image_path` lies outside `dataset_root`,
       strips drive/root prefixes, anchors at known project directories (`data/` or
       `results/`) if present, and otherwise preserves all remaining directory segments
       at full depth so distinct parent directories never collide.

    Returns:
        (normalized_path, has_directory_scope)
    """
    if image_path is None:
        return None, False
    raw = str(image_path).strip()
    if not raw:
        return None, False

    norm_raw = raw.replace("\\", "/")

    if dataset_root is not None and str(dataset_root).strip():
        norm_root = str(dataset_root).strip().replace("\\", "/")
        in_root_rel: Optional[str] = None

        # Try POSIX-normalized relpath first when drive prefixes match (handles mixed Windows/POSIX paths)
        raw_drive = norm_raw[:2].lower() if (len(norm_raw) >= 2 and norm_raw[1] == ":") else ""
        root_drive = norm_root[:2].lower() if (len(norm_root) >= 2 and norm_root[1] == ":") else ""
        raw_no_drive = norm_raw[2:] if raw_drive else norm_raw
        root_no_drive = norm_root[2:] if root_drive else norm_root

        raw_is_abs = bool(raw_drive) or raw_no_drive.startswith("/")
        root_is_abs = bool(root_drive) or root_no_drive.startswith("/")

        if raw_drive == root_drive and raw_is_abs == root_is_abs:
            import posixpath

            try:
                candidate = posixpath.relpath(
                    raw_no_drive.lstrip("/") or ".",
                    root_no_drive.lstrip("/") or ".",
                )
                if candidate != "." and not candidate.startswith(".."):
                    in_root_rel = candidate
            except ValueError:
                in_root_rel = None

        if in_root_rel is None:
            try:
                candidate = os.path.relpath(raw, str(dataset_root).strip()).replace("\\", "/")
                if candidate != "." and not candidate.startswith(".."):
                    in_root_rel = candidate
            except ValueError:
                in_root_rel = None

        if in_root_rel is not None:
            rel_parts = [p for p in in_root_rel.split("/") if p and p != "."]
            if rel_parts and rel_parts[0] != "..":
                normalized_in_root = "/".join(rel_parts)
                return normalized_in_root, len(rel_parts) > 1

    norm = norm_raw
    # Strip Windows drive letters or leading slashes when dataset_root is omitted or path is outside root
    if len(norm) >= 2 and norm[1] == ":":
        norm = norm[2:]
    norm = norm.lstrip("/")

    parts = [p for p in norm.split("/") if p and p not in (".", "..")]
    if not parts:
        return None, False

    # If path contains known project-relative anchors ('data/' or 'results/'), anchor from there;
    # otherwise preserve full directory hierarchy so deep parent directories never collide.
    for anchor in ("data", "results"):
        if anchor in parts:
            idx = parts.index(anchor)
            parts = parts[idx:]
            break

    if not parts:
        return None, False

    normalized = "/".join(parts)
    has_dir_scope = len(parts) > 1
    return normalized, has_dir_scope


def _sample_nine_point_pixels(image: Image.Image) -> str:
    """Sample 9 deterministic coordinates (corners, edge midpoints, center) in O(1) without resizing."""
    w, h = image.size
    if w <= 0 or h <= 0:
        return "empty"
    coords = [
        (0, 0),
        (w // 2, 0),
        (w - 1, 0),
        (0, h // 2),
        (w // 2, h // 2),
        (w - 1, h // 2),
        (0, h - 1),
        (w // 2, h - 1),
        (w - 1, h - 1),
    ]
    samples = []
    try:
        for cx, cy in coords:
            samples.append(image.getpixel((cx, cy)))
        return repr(samples)
    except Exception:
        return "unreadable"


def compute_image_content_fingerprint(
    image: Image.Image,
    *,
    image_path: Optional[str] = None,
    content_fingerprint: Optional[str] = None,
    use_thumbnail_bytes: bool = False,
) -> str:
    """
    Compute a content/version fingerprint for `image`.

    - If `content_fingerprint` is explicitly provided (e.g., from a manifest digest or
      precomputed once per loaded image), it is reused directly with zero pixel work.
    - When `use_thumbnail_bytes=False` (stable identity mode), combines on-disk file
      metadata (`st_size`, `st_mtime_ns`) when `image_path` exists on disk with an O(1)
      9-point pixel sample so no `(16 x 16)` thumbnail resize is performed.
    - When `use_thumbnail_bytes=True` (fallback mode without stable identity), hashes
      the `(16 x 16)` nearest-neighbor RGB thumbnail bytes + 9-point pixel sample.
    """
    if content_fingerprint is not None and str(content_fingerprint).strip():
        return f"explicit:{str(content_fingerprint).strip()}"

    w, h = image.size
    mode = str(getattr(image, "mode", "RGB"))
    nine_pt = _sample_nine_point_pixels(image)

    stat_part = "nostat"
    if image_path is not None and str(image_path).strip():
        raw_path = str(image_path).strip()
        if os.path.isfile(raw_path):
            try:
                st = os.stat(raw_path)
                stat_part = f"stat:{st.st_size}:{st.st_mtime_ns}"
            except OSError:
                stat_part = "stat_err"

    hasher = hashlib.sha256()
    hasher.update(f"{w}x{h}|{mode}|{stat_part}|{nine_pt}".encode("utf-8"))

    if use_thumbnail_bytes:
        try:
            resample_mode = getattr(getattr(Image, "Resampling", Image), "NEAREST")
            small = image.resize((16, 16), resample=resample_mode).convert("RGB")
            hasher.update(b"|thumb16:")
            hasher.update(small.tobytes())
        except Exception:
            hasher.update(b"|thumb16_err")

    return f"{'thumb' if use_thumbnail_bytes else 'probe'}:{hasher.hexdigest()[:24]}"


def build_image_cache_identity(
    image: Image.Image,
    *,
    image_key: Optional[Union["ImageCacheIdentity", Tuple[Any, ...], str]] = None,
    image_id: Optional[Union[str, int]] = None,
    image_path: Optional[str] = None,
    dataset_root: Optional[str] = None,
    dataset_version: Optional[str] = None,
    dataset_namespace: Optional[str] = None,
    content_fingerprint: Optional[str] = None,
) -> ImageCacheIdentity:
    """
    Build or validate a canonical `ImageCacheIdentity` for `image`.

    Rules enforced (OPT-06):
    1. A bare basename alone (without `image_id`, directory scope, or `dataset_version`)
       is NEVER treated as a globally unique stable identity; it triggers content-based
       thumbnail fallback (`is_fallback=True`).
    2. When `image_id` and/or directory-scoped relative path is provided, both are
       incorporated into `image_identity` so same-basename files in different directories
       or distinct `image_id` records never collide.
    3. `dataset_version` (or alias `dataset_namespace`) scopes the identity across datasets.
    4. `width`, `height`, and `content_fingerprint` are always included so changed image
       content or dimensions invalidate cached state immediately.
    5. If a precomputed `ImageCacheIdentity` is passed via `image_key` and matches the
       current `image.size`, it is returned directly without recomputing any hash.
    """
    w, h = image.size
    raw_id = image_id

    has_ver = dataset_version is not None and bool(str(dataset_version).strip())
    has_ns = dataset_namespace is not None and bool(str(dataset_namespace).strip())
    if has_ver and has_ns and str(dataset_version).strip() != str(dataset_namespace).strip():
        raise ValueError(
            f"Conflicting dataset_version ({dataset_version!r}) and "
            f"dataset_namespace ({dataset_namespace!r})."
        )

    ds_raw = dataset_version if has_ver else (dataset_namespace if has_ns else None)
    has_explicit_ds = ds_raw is not None
    has_explicit_id = None is not raw_id and bool(str(raw_id).strip())

    if isinstance(image_key, ImageCacheIdentity):
        if image_key.width != w or image_key.height != h:
            raise ValueError(
                f"Precomputed ImageCacheIdentity dimensions ({image_key.width}x{image_key.height}) "
                f"contradict image.size ({w}x{h})."
            )
        if has_explicit_ds:
            expected_ds = str(ds_raw).strip()
            if image_key.dataset_version != expected_ds:
                raise ValueError(
                    f"Explicit dataset_version/dataset_namespace ({expected_ds!r}) contradicts "
                    f"precomputed ImageCacheIdentity.dataset_version ({image_key.dataset_version!r})."
                )
        key_tokens = image_key.image_identity.split("|") if image_key.image_identity else []
        if has_explicit_id:
            expected_id_token = f"id={str(raw_id).strip()}"
            id_tokens = [t for t in key_tokens if t.startswith("id=")]
            if not id_tokens or id_tokens[0] != expected_id_token:
                raise ValueError(
                    f"Explicit image_id ({raw_id!r}) contradicts precomputed "
                    f"ImageCacheIdentity.image_identity ({image_key.image_identity!r})."
                )
        if image_path is not None and bool(str(image_path).strip()):
            check_relpath, _ = _normalize_relative_image_path(
                image_path, dataset_root=dataset_root
            )
            if check_relpath is not None:
                expected_path_token = f"path={check_relpath}"
                path_tokens = [t for t in key_tokens if t.startswith("path=")]
                if not path_tokens or path_tokens[0] != expected_path_token:
                    raise ValueError(
                        f"Explicit image_path ({image_path!r}, normalized={check_relpath!r}) "
                        f"contradicts precomputed ImageCacheIdentity.image_identity "
                        f"({image_key.image_identity!r})."
                    )
        if content_fingerprint is not None and bool(str(content_fingerprint).strip()):
            raw_fp = str(content_fingerprint).strip()
            expected_fp = (
                raw_fp
                if raw_fp.startswith(("explicit:", "probe:", "thumb:"))
                else f"explicit:{raw_fp}"
            )
            if image_key.content_fingerprint not in (expected_fp, raw_fp):
                raise ValueError(
                    f"Explicit content_fingerprint ({content_fingerprint!r}) contradicts "
                    f"precomputed ImageCacheIdentity.content_fingerprint "
                    f"({image_key.content_fingerprint!r})."
                )
        return image_key

    resolved_ds = str(ds_raw).strip() if has_explicit_ds else UNSCOPED_DATASET_NAMESPACE
    norm_relpath, has_dir_scope = _normalize_relative_image_path(
        image_path, dataset_root=dataset_root
    )

    id_parts: List[str] = []
    if has_explicit_id:
        id_parts.append(f"id={str(raw_id).strip()}")
    if norm_relpath is not None:
        id_parts.append(f"path={norm_relpath}")

    # Stable identity requires either an explicit image_id or a directory-scoped relative path
    # (a bare filename without image_id or directory scope is NOT sufficient on its own).
    has_stable_id = has_explicit_id or has_dir_scope

    if id_parts:
        resolved_img_identity = "|".join(id_parts)
    elif isinstance(image_key, str) and image_key.strip():
        resolved_img_identity = f"key={image_key.strip()}"
        has_stable_id = True
    else:
        resolved_img_identity = FALLBACK_IMAGE_IDENTITY

    is_fallback = not has_stable_id

    fp = compute_image_content_fingerprint(
        image,
        image_path=image_path,
        content_fingerprint=content_fingerprint,
        use_thumbnail_bytes=is_fallback,
    )

    return ImageCacheIdentity(
        dataset_version=resolved_ds,
        image_identity=resolved_img_identity,
        width=w,
        height=h,
        content_fingerprint=fp,
        is_fallback=is_fallback,
    )


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
    def compute_image_key(
        image: Image.Image,
        num_tiles: int = 1,
        *,
        image_key: Optional[Union[ImageCacheIdentity, Tuple[Any, ...], str]] = None,
        image_id: Optional[Union[str, int]] = None,
        image_path: Optional[str] = None,
        dataset_root: Optional[str] = None,
        dataset_version: Optional[str] = None,
        dataset_namespace: Optional[str] = None,
        content_fingerprint: Optional[str] = None,
    ) -> str:
        """
        Compute a collision-safe signature key for an image and tiling configuration (OPT-06).
        """
        identity = build_image_cache_identity(
            image,
            image_key=image_key,
            image_id=image_id,
            image_path=image_path,
            dataset_root=dataset_root,
            dataset_version=dataset_version,
            dataset_namespace=dataset_namespace,
            content_fingerprint=content_fingerprint,
        )
        return identity.to_cache_key(num_tiles=num_tiles)

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
        *,
        dtype: Optional["torch.dtype"] = None,
        precision: Optional[str] = None,
        image_key: Optional[Union[ImageCacheIdentity, Tuple[Any, ...], str]] = None,
        image_id: Optional[Union[str, int]] = None,
        image_path: Optional[str] = None,
        dataset_root: Optional[str] = None,
        dataset_version: Optional[str] = None,
        dataset_namespace: Optional[str] = None,
        content_fingerprint: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Retrieve or compute visual features for a list of image tiles.
        
        Args:
            model: InternVLChatModel
            transform: torchvision image transform (448x448 BICUBIC)
            tiles: List of 448x448 PIL Image tiles
            image: Original full-resolution image (optional, used for caching key)
            device: Target execution device
            dtype: Explicit target torch.dtype (torch.bfloat16 or torch.float16)
            precision: Explicit precision string ('bf16' or 'fp16')
            image_key: Optional precomputed ImageCacheIdentity or key
            image_id: Optional stable image identifier within dataset
            image_path: Optional path to image file (relative or absolute)
            dataset_root: Optional dataset root directory for relative path normalization
            dataset_version: Optional dataset version/namespace identifier
            dataset_namespace: Alias for dataset_version
            content_fingerprint: Optional content/version fingerprint
            
        Returns:
            vit_embeds: torch.Tensor of shape (num_tiles, 256, 896) in target precision
        """
        _, target_dtype = resolve_precision_and_dtype(precision=precision, dtype=dtype)
        key = None
        if image is not None:
            key = self.compute_image_key(
                image,
                len(tiles),
                image_key=image_key,
                image_id=image_id,
                image_path=image_path,
                dataset_root=dataset_root,
                dataset_version=dataset_version,
                dataset_namespace=dataset_namespace,
                content_fingerprint=content_fingerprint,
            )
            cached = self.get(key)
            if cached is not None:
                target_dev = torch.device(device)
                if cached.device.type == target_dev.type and cached.dtype == target_dtype:
                    cached_idx = 0 if cached.device.index is None else cached.device.index
                    target_idx = 0 if target_dev.index is None else target_dev.index
                    if cached_idx == target_idx:
                        return cached

        pixel_values = torch.stack([transform(t) for t in tiles]).to(
            device=device, dtype=target_dtype
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
    *,
    dtype: Optional["torch.dtype"] = None,
    precision: Optional[str] = None,
    image_key: Optional[Union[ImageCacheIdentity, Tuple[Any, ...], str]] = None,
    image_id: Optional[Union[str, int]] = None,
    image_path: Optional[str] = None,
    dataset_root: Optional[str] = None,
    dataset_version: Optional[str] = None,
    dataset_namespace: Optional[str] = None,
    content_fingerprint: Optional[str] = None,
) -> torch.Tensor:
    """Convenience functional interface adhering to PROJECT.md contract."""
    target_cache = cache if cache is not None else default_feature_cache
    return target_cache.get_or_compute(
        model,
        transform,
        tiles,
        image=image,
        device=device,
        dtype=dtype,
        precision=precision,
        image_key=image_key,
        image_id=image_id,
        image_path=image_path,
        dataset_root=dataset_root,
        dataset_version=dataset_version,
        dataset_namespace=dataset_namespace,
        content_fingerprint=content_fingerprint,
    )
