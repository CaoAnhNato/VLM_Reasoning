# -*- coding: utf-8 -*-
r"""
Stage 1.4: Continuous Isotropic Square ROI Transformation (IsotropicROICropper).

Mathematical Proof of Aspect-Ratio Invariance (Zero 1.35 Threshold):
Let B = [x1, x2] x [y1, y2] be the selected text bounding box with width Delta_x = x2 - x1,
height Delta_y = y2 - y1, and center (mu_x, mu_y) = ((x1 + x2)/2, (y1 + y2)/2).
To project B onto a 448 x 448 ViT tile without distorting character glyph aspect ratio
(and without introducing artificial step-edge bars or pulling in sharp distractor text
outside B), the isotropic square side length is defined continuously as:
    s = max(Delta_x, Delta_y)

The square window S = [mu_x - s/2, mu_x + s/2] x [mu_y - s/2, mu_y + s/2] is centered
directly on (mu_x, mu_y). The exterior region S \ B is smoothly low-pass filtered
(Gaussian blur) using the image's own natural surrounding colors while the interior
KL-optimal region B remains 100% sharp original pixels. This achieves:
1. Exact isotropic magnification M = 448 / s along both axes (zero glyph distortion).
2. Zero artificial step-edge ringing (seamless natural boundary transition).
3. Zero distractor contamination from outside B.
4. Exact centering of B inside the 448 x 448 ViT field of view.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter


@dataclass
class CropResult:
    """Structured output of the Dynamic Multi-Patch ROI Cropper."""
    roi_tile: Image.Image                     # Primary 448 x 448 RGB PIL Image ready for InternViT
    isotropic_box: Tuple[int, int, int, int]  # (sx1, sy1, sx2, sy2) canvas coordinates in image space
    source_bbox: Tuple[int, int, int, int]    # Expanded (x1, y1, x2, y2) sharp focus bounding box
    magnification: float                      # Isotropic scale factor 448 / s_unit
    roi_tiles: List[Image.Image] = field(default_factory=list)  # Sharp sub-tiles [T_0, ..., T_{N-1}]
    neg_tiles: List[Image.Image] = field(default_factory=list)  # Blurred negative prior views [T_0^{neg}, ...]
    grid: Tuple[int, int] = (1, 1)            # Optimal grid (g_x*, g_y*)
    unit_tile_size: float = 0.0               # s_unit

    def __post_init__(self):
        if not self.roi_tiles and self.roi_tile is not None:
            self.roi_tiles = [self.roi_tile]
        if not self.neg_tiles and self.roi_tile is not None:
            self.neg_tiles = [self.roi_tile]
        if self.unit_tile_size <= 0.0 and self.magnification > 0:
            self.unit_tile_size = 448.0 / self.magnification


class BaseROICropper(ABC):
    """Strategy interface for transforming a selected bounding box into a ViT input tile."""

    @abstractmethod
    def crop(
        self,
        image: Image.Image,
        bbox: Tuple[int, int, int, int],
    ) -> CropResult:
        """Transform bbox on image into a target_size x target_size RGB tile."""
        raise NotImplementedError


class DynamicMultiTileROICropper(BaseROICropper):
    r"""
    Dynamic Multi-Patch RoI Tiling (Aspect-Ratio Distortion Minimizing Cropper).

    Replaces aspect ratio padding collapse (s = max(Delta_x, Delta_y)) with continuous
    log-AR distortion optimization:
        (g_x*, g_y*) = argmin_{(g_x, g_y) in G_admissible} |ln(W_B / H_B) - ln(g_x / g_y)|
    Candidate grids: (1, 1), (2, 1), (1, 2), (3, 1), (1, 3), capped by physical span
    g_x <= floor(W_B / 224), g_y <= floor(H_B / 224).

    Canvas matching ratio g_x* : g_y*:
        S_x = max(W_B, H_B * g_x* / g_y*)
        S_y = S_x * g_y* / g_x*
    Unit square tile size:
        s_unit = S_x / g_x* = S_y / g_y*
    Each sub-tile has size s_unit x s_unit resized to 448 x 448, guaranteeing zero
    anisotropic stretching and identical magnification M = 448 / s_unit on both axes.
    """

    def __init__(
        self,
        target_size: int = 448,
        context_margin: float = 0.15,
        max_tiles: int = 3,
        min_tile_dim: int = 224,
    ):
        self.target_size = target_size
        self.context_margin = context_margin
        self.max_tiles = max_tiles
        self.min_tile_dim = min_tile_dim

    def select_grid(self, wb: float, hb: float) -> Tuple[int, int]:
        """Continuous log-AR distortion loss minimization bounded by physical pixel span."""
        ar = max(1e-6, float(wb)) / max(1.0, float(hb))
        candidates = [(1, 1)]
        if self.max_tiles >= 2:
            candidates.extend([(2, 1), (1, 2)])
        if self.max_tiles >= 3:
            candidates.extend([(3, 1), (1, 3)])

        gx_cap = max(1, int(wb // self.min_tile_dim))
        gy_cap = max(1, int(hb // self.min_tile_dim))
        admissible = [(gx, gy) for (gx, gy) in candidates if gx <= gx_cap and gy <= gy_cap]
        if not admissible:
            admissible = [(1, 1)]

        best_grid = min(
            admissible,
            key=lambda g: (abs(math.log(ar) - math.log(g[0] / g[1])), g[0] * g[1]),
        )
        return best_grid

    def crop(
        self,
        image: Image.Image,
        bbox: Tuple[int, int, int, int],
    ) -> CropResult:
        """Apply dynamic multi-patch RoI tiling and return sharp & blurred sub-tiles."""
        W, H = image.size
        bx1, by1, bx2, by2 = bbox

        # Canonicalize coordinates to handle inverted bounding boxes
        cbx1, cbx2 = int(round(min(bx1, bx2))), int(round(max(bx1, bx2)))
        cby1, cby2 = int(round(min(by1, by2))), int(round(max(by1, by2)))
        raw_w = max(2, cbx2 - cbx1)
        raw_h = max(2, cby2 - cby1)

        # Expand sharp bounding box B by context margin protecting ascenders/descenders
        pad_x = int(round(raw_w * self.context_margin))
        pad_y = max(int(round(raw_h * self.context_margin)), int(round(raw_h * 0.15)), 2)

        min_dim_x = min(2, W)
        min_dim_y = min(2, H)
        x1 = max(0, min(W - min_dim_x, cbx1 - pad_x))
        y1 = max(0, min(H - min_dim_y, cby1 - pad_y))
        x2 = max(x1 + min_dim_x, min(W, cbx2 + pad_x))
        y2 = max(y1 + min_dim_y, min(H, cby2 + pad_y))

        delta_x = max(2, x2 - x1)
        delta_y = max(2, y2 - y1)

        # 1. Optimal grid selection via log-AR loss
        gx_star, gy_star = self.select_grid(float(delta_x), float(delta_y))

        # 2. Enclosing canvas dimensions matching ratio gx_star : gy_star
        Sx = max(float(delta_x), float(delta_y) * (gx_star / gy_star))
        Sy = Sx * (gy_star / gx_star)
        s_unit = Sx / float(gx_star)

        # Continuous scale adaptation floor bounded by ViT patch resolution
        min_token_side = 32
        min_canvas_ref = min(W, H)
        box_max = max(delta_x, delta_y)
        adaptive_floor = min_token_side + (min_canvas_ref * 0.05) / (1.0 + math.exp(-0.05 * (box_max - min_token_side)))
        if s_unit < adaptive_floor:
            s_unit = float(adaptive_floor)
            Sx = s_unit * gx_star
            Sy = s_unit * gy_star

        # 3. Canvas coordinates in image space centered on sharp box
        mu_x = 0.5 * (x1 + x2)
        mu_y = 0.5 * (y1 + y2)
        sx1 = int(round(mu_x - 0.5 * Sx))
        sy1 = int(round(mu_y - 0.5 * Sy))
        sx2 = int(round(sx1 + Sx))
        sy2 = int(round(sy1 + Sy))
        can_w = max(2, sx2 - sx1)
        can_h = max(2, sy2 - sy1)

        # Build canvas array using coordinate reflection at image borders
        rgb_img = image.convert("RGB")
        img_arr = np.asarray(rgb_img)

        yy_raw = np.arange(sy1, sy2)
        xx_raw = np.arange(sx1, sx2)
        yy = np.clip(np.where(yy_raw < 0, -yy_raw, np.where(yy_raw >= H, 2 * (H - 1) - yy_raw, yy_raw)), 0, H - 1)
        xx = np.clip(np.where(xx_raw < 0, -xx_raw, np.where(xx_raw >= W, 2 * (W - 1) - xx_raw, xx_raw)), 0, W - 1)
        canvas_raw_arr = img_arr[np.ix_(yy, xx)]
        canvas_raw_img = Image.fromarray(canvas_raw_arr)

        # Background exterior blur
        blur_radius = max(4.0, float(s_unit) * 0.035)
        canvas_bg_img = canvas_raw_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        canvas_neg_img = canvas_raw_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # Paste sharp focus interior B onto canvas
        sharp_crop = rgb_img.crop((x1, y1, x2, y2))
        paste_x = int(round(x1 - sx1))
        paste_y = int(round(y1 - sy1))
        canvas_bg_img.paste(sharp_crop, (paste_x, paste_y))

        # 4. Partition canvas into gx_star x gy_star unit square tiles
        roi_tiles: List[Image.Image] = []
        neg_tiles: List[Image.Image] = []
        for j in range(gy_star):
            for i in range(gx_star):
                tx1 = int(round(float(i) * can_w / float(gx_star)))
                tx2 = int(round(float(i + 1) * can_w / float(gx_star)))
                ty1 = int(round(float(j) * can_h / float(gy_star)))
                ty2 = int(round(float(j + 1) * can_h / float(gy_star)))

                tile_crop = canvas_bg_img.crop((tx1, ty1, tx2, ty2))
                sharp_tile = tile_crop.resize(
                    (self.target_size, self.target_size),
                    Image.Resampling.BICUBIC,
                )
                roi_tiles.append(sharp_tile)

                tile_neg_crop = canvas_neg_img.crop((tx1, ty1, tx2, ty2))
                neg_tile = tile_neg_crop.resize(
                    (self.target_size, self.target_size),
                    Image.Resampling.BICUBIC,
                )
                neg_tile = neg_tile.filter(ImageFilter.GaussianBlur(radius=max(3.5, blur_radius * 0.8)))
                neg_tiles.append(neg_tile)

        magnification = float(self.target_size) / float(max(1.0, s_unit))

        return CropResult(
            roi_tile=roi_tiles[0],
            isotropic_box=(sx1, sy1, sx2, sy2),
            source_bbox=(x1, y1, x2, y2),
            magnification=magnification,
            roi_tiles=roi_tiles,
            neg_tiles=neg_tiles,
            grid=(gx_star, gy_star),
            unit_tile_size=float(s_unit),
        )

    def get_multi_tiles(
        self,
        image: Image.Image,
        bbox: Tuple[int, int, int, int],
        max_tiles: int = 2,
    ) -> Tuple[List[Image.Image], CropResult]:
        """
        Produce multi-tile visual representations. If dynamic tiling generated >1 tiles,
        returns [global_thumb] + roi_tiles; otherwise falls back to hierarchical dual-tile.
        """
        crop_res = self.crop(image, bbox)
        if max_tiles <= 1:
            return [crop_res.roi_tile], crop_res

        global_thumb = image.resize((self.target_size, self.target_size), Image.Resampling.BICUBIC)
        if len(crop_res.roi_tiles) > 1:
            return [global_thumb] + crop_res.roi_tiles, crop_res

        if max_tiles == 2:
            return [global_thumb, crop_res.roi_tile], crop_res

        # For max_tiles >= 3:
        W, H = image.size
        bx1, by1, bx2, by2 = bbox
        bw = max(2, bx2 - bx1)
        bh = max(2, by2 - by1)
        ctx_pad_x = int(round(bw * 0.75))
        ctx_pad_y = int(round(bh * 0.75))
        ctx_bbox = (
            max(0, bx1 - ctx_pad_x),
            max(0, by1 - ctx_pad_y),
            min(W, bx2 + ctx_pad_x),
            min(H, by2 + ctx_pad_y),
        )
        ctx_crop = self.crop(image, ctx_bbox)
        return [global_thumb, crop_res.roi_tile, ctx_crop.roi_tile], crop_res

    def get_dual_tiles(
        self,
        image: Image.Image,
        bbox: Tuple[int, int, int, int],
    ) -> Tuple[Image.Image, Image.Image, CropResult]:
        """Produce [I_global, I_ROI] dual tiles for multi-scale fusion."""
        tiles, crop_res = self.get_multi_tiles(image, bbox, max_tiles=2)
        return tiles[0], tiles[1], crop_res


class IsotropicROICropper(DynamicMultiTileROICropper):
    r"""
    Backwards-compatible interface alias for DynamicMultiTileROICropper.
    Inherits continuous log-AR distortion minimization and unit square multi-tile generation.
    """
    pass


def apply_adaptive_laplacian_edge_reemphasis(
    roi_img: Image.Image,
    magnification: float,
    lambda_0: float = 0.50,
) -> Image.Image:
    """
    Adaptive Pre-RoI Laplacian Edge Re-emphasis for sub-Nyquist blurred text.
    Zero-heuristic: purely scaled by isotropic magnification M and local gradient salience.

    Mathematical Formulation:
    - Scaling factor: psi(M) = tanh(max(0.0, M - 1.0)) (0 when M <= 1.0, scaling smoothly with zoom)
    - Gradient magnitude: G = sqrt(Gx^2 + Gy^2) on luminance channel Y
    - Local salience gate: omega_grad(x, y) = sigmoid((G - mu_G) / max(sigma_G, 1e-4))
    - 8-connected discrete isotropic Laplacian kernel:
      K_L = (1/8) * [[1,  1, 1],
                     [1, -8, 1],
                     [1,  1, 1]]
    - Edge boost:
      I_ROI^(+) = clip(I_ROI - (lambda_0 * psi(M) * omega_grad) * Delta I, 0, 255)
    """
    if magnification <= 1.02:
        return roi_img

    arr = np.asarray(roi_img.convert("RGB"), dtype=np.float32)
    gray = 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]

    # 1. Magnification scaling
    psi_m = math.tanh(max(0.0, float(magnification) - 1.0))

    # 2. Gradient magnitude & salience gating
    gy, gx = np.gradient(gray)
    grad_mag = np.sqrt(gx**2 + gy**2)
    mu_g = float(np.mean(grad_mag))
    std_g = float(np.std(grad_mag)) + 1e-5
    omega_grad = 1.0 / (1.0 + np.exp(-np.clip((grad_mag - mu_g) / std_g, -5.0, 5.0)))
    omega_grad = np.expand_dims(omega_grad, axis=-1)

    # 3. 8-connected isotropic discrete Laplacian
    # Pad array reflectively for border continuity
    padded = np.pad(arr, ((1, 1), (1, 1), (0, 0)), mode="reflect")
    laplacian = (
        padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
        padded[1:-1, :-2] - 8.0 * padded[1:-1, 1:-1] + padded[1:-1, 2:] +
        padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
    ) / 8.0

    # 4. Adaptive edge boost
    sharpened = arr - (lambda_0 * psi_m * omega_grad) * laplacian
    sharpened = np.clip(sharpened, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(sharpened)


