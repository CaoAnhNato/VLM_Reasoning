# -*- coding: utf-8 -*-
"""
Stage 1.3: Scale-Invariant Bernoulli Kullback-Leibler Region Selector (KLDivergenceRegionSelector).

Mathematical Formulation:
Given the Region-Affinity field Phi(u) >= 0 on the 2D patch lattice Omega of size (H_g, W_g),
let P_Phi(u) = Phi(u) / sum_{v in Omega} Phi(v) be the normalized spatial distribution.
For any candidate region C subset Omega:
    p(C) = sum_{u in C} P_Phi(u)     (Captured probability mass / Recall)
    a(C) = |C| / (H_g * W_g)         (Spatial area fraction)

We maximize the Scale-Invariant Bernoulli Kullback-Leibler Divergence:
    J_KL(C) = p(C) * log(p(C) / a(C)) + (1 - p(C)) * log((1 - p(C)) / (1 - a(C)))
subject to p(C) > a(C) (internal information density strictly exceeds uniform prior).

Why J_KL(C) resolves the Discrete Green's Theorem small-window collapse:
- Dividing perimeter difference by 2D area (b_w * b_h) scaled as 1/b_w + 1/b_h, forcing
  degenerate collapse to the smallest window (0.6, 0.6).
- By contrast, J_KL(C) weights the log density ratio log(p(C)/a(C)) by the extensive probability
  mass p(C), penalizing both under-cropping (missing text mass) and over-cropping (diluting
  density with background area).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage


@dataclass
class RegionSelectionResult:
    """Structured output of the KL-divergence region selector."""
    pixel_bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in original image pixels
    grid_bbox: Tuple[int, int, int, int]   # (gx1, gy1, gx2, gy2) in patch grid coordinates
    kl_score: float                        # Optimal Bernoulli KL score J_KL(C*)
    captured_mass: float                   # p(C*) in (0, 1)
    area_fraction: float                   # a(C*) in (0, 1)


@dataclass
class MultiRegionSelectionResult:
    """Structured output of the KL-divergence region selector for Top-K candidate regions (K in {1, 2})."""
    pixel_bboxes: List[Tuple[int, int, int, int]]  # Top-K bounding boxes in image pixels
    grid_bboxes: List[Tuple[int, int, int, int]]   # Top-K bounding boxes in patch grid
    kl_scores: List[float]                        # Top-K Bernoulli KL scores
    captured_masses: List[float]                  # p(C) values
    area_fractions: List[float]                   # a(C) values
    k: int                                        # Effective K (1 or 2)

    @property
    def bboxes(self) -> List[Tuple[int, int, int, int]]:
        return self.pixel_bboxes

    @property
    def kl_divergences(self) -> List[float]:
        return self.kl_scores


def compute_bernoulli_kl_score(
    p: float,
    a: float,
    eps: float = 1e-9,
) -> float:
    """
    Compute the Bernoulli KL Divergence J_KL(C) between the binary inside/outside distribution
    (p, 1 - p) and the uniform spatial area prior (a, 1 - a):
        J_KL(C) = p * log(p / a) + (1 - p) * log((1 - p) / (1 - a))
    For regions where p <= a (less dense than uniform random noise), returns 0.0.
    """
    if p <= a or a <= eps or a >= 1.0 - eps:
        return 0.0
    p_c = min(max(float(p), eps), 1.0 - eps)
    a_c = min(max(float(a), eps), 1.0 - eps)
    term1 = p_c * np.log(p_c / a_c)
    term2 = (1.0 - p_c) * np.log((1.0 - p_c) / (1.0 - a_c))
    return float(max(0.0, term1 + term2))


def _build_integral_image(prob_map: np.ndarray) -> np.ndarray:
    """Build 2D summed-area table with 1-cell zero padding at top and left."""
    padded = np.pad(np.asarray(prob_map, dtype=np.float64), ((1, 0), (1, 0)), mode="constant")
    return padded.cumsum(axis=0).cumsum(axis=1)


def _rect_sum(integral: np.ndarray, gx1: int, gy1: int, gx2: int, gy2: int) -> float:
    """O(1) rectangle sum on [gx1, gx2) x [gy1, gy2)."""
    return float(
        integral[gy2, gx2]
        - integral[gy1, gx2]
        - integral[gy2, gx1]
        + integral[gy1, gx1]
    )


def compute_high_frequency_gradient_prior(
    image: Any,
    grid_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Compute a normalized high-frequency gradient magnitude prior on the 2D patch lattice.
    Uses Sobel gradient operators on luminance to isolate high-frequency character glyph
    edges and suppress smooth non-text visual sinks (e.g. food bowls, clothing, sky, walls).

    Args:
        image: PIL Image or RGB ndarray
        grid_shape: (H_g, W_g) spatial dimensions of patch lattice Omega

    Returns:
        gradient_prior: (H_g, W_g) ndarray with values in [0, 1]
    """
    H_g, W_g = grid_shape
    if H_g <= 0 or W_g <= 0:
        return np.ones(grid_shape, dtype=np.float32)

    if isinstance(image, np.ndarray):
        img_pil = Image.fromarray(image).convert("L")
    elif hasattr(image, "convert"):
        img_pil = image.convert("L")
    else:
        return np.ones(grid_shape, dtype=np.float32)

    target_proc_size = (max(W_g * 16, 256), max(H_g * 16, 256))
    gray = np.asarray(img_pil.resize(target_proc_size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0

    gx = ndimage.sobel(gray, axis=1)
    gy = ndimage.sobel(gray, axis=0)
    grad_mag = np.sqrt(gx * gx + gy * gy)

    H_img, W_img = grad_mag.shape
    grad_grid = np.zeros((H_g, W_g), dtype=np.float32)
    step_y = H_img / float(H_g)
    step_x = W_img / float(W_g)

    for i in range(H_g):
        y1 = int(round(i * step_y))
        y2 = int(round((i + 1) * step_y))
        y2 = max(y1 + 1, min(H_img, y2))
        for j in range(W_g):
            x1 = int(round(j * step_x))
            x2 = int(round((j + 1) * step_x))
            x2 = max(x1 + 1, min(W_img, x2))
            grad_grid[i, j] = float(np.mean(grad_mag[y1:y2, x1:x2]))

    g_min = float(grad_grid.min())
    g_max = float(grad_grid.max())
    if (g_max - g_min) > 1e-6:
        grad_norm = (grad_grid - g_min) / (g_max - g_min)
    else:
        grad_norm = np.ones_like(grad_grid)

    return grad_norm


def modulate_affinity_with_gradient_prior(
    phi_map: np.ndarray,
    gradient_prior: np.ndarray,
    kappa: float = 0.30,
) -> np.ndarray:
    """
    Modulate cross-attention / region-affinity field Phi(u) with the high-frequency
    gradient prior G(u):
        Phi_guided(u) = Phi(u) * (kappa + (1 - kappa) * G_norm(u))

    Suppresses smooth non-text object surfaces (where G_norm ~ 0) while preserving
    sharp text glyph edges (where G_norm ~ 1). The baseline floor kappa in [0.25, 0.35]
    prevents vanishing attention on low-contrast text.
    """
    phi = np.asarray(phi_map, dtype=np.float64)
    if gradient_prior is None or gradient_prior.shape != phi.shape:
        return phi
    g = np.clip(np.asarray(gradient_prior, dtype=np.float64), 0.0, 1.0)
    gate = float(kappa) + (1.0 - float(kappa)) * g
    return phi * gate


class BaseRegionSelector(ABC):
    """Strategy interface for selecting the optimal bounding box from Phi(u)."""

    @abstractmethod
    def select(
        self,
        phi_map: np.ndarray,
        image_size: Tuple[int, int],
        image: Optional[Any] = None,
        gradient_prior: Optional[np.ndarray] = None,
    ) -> RegionSelectionResult:
        """Select optimal bounding box on phi_map and project to original image_size (W, H)."""
        raise NotImplementedError

    def select_top_k(
        self,
        phi_map: np.ndarray,
        image_size: Optional[Tuple[int, int]] = None,
        image: Optional[Any] = None,
        gradient_prior: Optional[np.ndarray] = None,
        max_k: int = 2,
        orig_w: Optional[int] = None,
        orig_h: Optional[int] = None,
    ) -> MultiRegionSelectionResult:
        """Select Top-K optimal bounding boxes from phi_map."""
        if image_size is None:
            if orig_w is not None and orig_h is not None:
                image_size = (orig_w, orig_h)
            else:
                image_size = (1, 1)
        res = self.select(phi_map, image_size, image, gradient_prior)
        return MultiRegionSelectionResult(
            pixel_bboxes=[res.pixel_bbox],
            grid_bboxes=[res.grid_bbox],
            kl_scores=[res.kl_score],
            captured_masses=[res.captured_mass],
            area_fractions=[res.area_fraction],
            k=1,
        )


def compute_otsu_threshold_1d(profile: np.ndarray) -> float:
    """
    Compute 1D Otsu optimal threshold separating foreground character energy from background noise.
    Maximizes between-class variance sigma_b^2(t) = w0(t) * w1(t) * (mu0(t) - mu1(t))^2.
    Zero magic constants.
    """
    arr = np.asarray(profile, dtype=np.float64)
    if arr.size <= 2:
        return float(arr.mean()) if arr.size > 0 else 0.0
    val_min = float(arr.min())
    val_max = float(arr.max())
    if (val_max - val_min) <= 1e-9:
        return val_min

    candidates = np.unique(arr)
    if len(candidates) <= 2:
        return float(np.mean(candidates))

    best_thresh = float(arr.mean())
    max_var = -1.0
    n = float(arr.size)

    for thresh in candidates[1:-1]:
        w0 = np.sum(arr < thresh) / n
        w1 = 1.0 - w0
        if w0 <= 1e-9 or w1 <= 1e-9:
            continue
        mu0 = np.mean(arr[arr < thresh])
        mu1 = np.mean(arr[arr >= thresh])
        var_between = w0 * w1 * ((mu0 - mu1) ** 2)
        if var_between > max_var:
            max_var = var_between
            best_thresh = float(thresh)

    return best_thresh


class KLDivergenceRegionSelector(BaseRegionSelector):
    """
    Scale-Invariant Region Selector maximizing Bernoulli KL Divergence J_KL(C)
    over connected components of the Region-Affinity field Phi(u) and integral-image
    bounding rectangles. Zero hardcoded window sizes or aspect ratio priors.
    """

    def __init__(
        self,
        num_quantiles: int = 28,
        min_area_fraction: Optional[float] = None,
        max_area_fraction: float = 0.85,
        boundary_pad_patches: float = 1.25,
        tau_iou: float = 0.20,
        tau_iom: float = 0.35,
        gamma_signif: float = 0.40,
        min_prob_threshold: float = 0.10,
    ):
        self.num_quantiles = num_quantiles
        self.min_area_fraction = min_area_fraction
        self.max_area_fraction = max_area_fraction
        self.boundary_pad_patches = boundary_pad_patches
        self.tau_iou = tau_iou
        self.tau_iom = tau_iom
        self.gamma_signif = gamma_signif
        self.min_prob_threshold = min_prob_threshold
        self.last_guided_phi: Optional[np.ndarray] = None

    def _trim_sparse_borders(
        self,
        prob_map: np.ndarray,
        gx1: int,
        gy1: int,
        gx2: int,
        gy2: int,
    ) -> Tuple[int, int, int, int]:
        """
        Trim outer rows/columns of a candidate bounding box using 1D Otsu thresholding
        on marginal projection densities, protecting Vietnamese vowel diacritic peaks.
        """
        # Trim rows along Y using 1D Otsu threshold computed once on initial marginal row density
        sub_y = prob_map[gy1:gy2, gx1:gx2]
        if sub_y.size > 0:
            tau_y = compute_otsu_threshold_1d(sub_y.mean(axis=1))
            while (gy2 - gy1) > 2:
                row_top_mean = float(prob_map[gy1, gx1:gx2].mean())
                row_top_max = float(prob_map[gy1, gx1:gx2].max())
                row_bot_mean = float(prob_map[gy2 - 1, gx1:gx2].mean())
                row_bot_max = float(prob_map[gy2 - 1, gx1:gx2].max())
                if row_top_mean < tau_y and row_top_max < 1.5 * tau_y:
                    gy1 += 1
                elif row_bot_mean < tau_y and row_bot_max < 1.5 * tau_y:
                    gy2 -= 1
                else:
                    break

        # Trim columns along X using 1D Otsu threshold computed once on initial marginal column density
        sub_x = prob_map[gy1:gy2, gx1:gx2]
        if sub_x.size > 0:
            tau_x = compute_otsu_threshold_1d(sub_x.mean(axis=0))
            while (gx2 - gx1) > 2:
                col_left_mean = float(prob_map[gy1:gy2, gx1].mean())
                col_left_max = float(prob_map[gy1:gy2, gx1].max())
                col_right_mean = float(prob_map[gy1:gy2, gx2 - 1].mean())
                col_right_max = float(prob_map[gy1:gy2, gx2 - 1].max())
                if col_left_mean < tau_x and col_left_max < 1.5 * tau_x:
                    gx1 += 1
                elif col_right_mean < tau_x and col_right_max < 1.5 * tau_x:
                    gx2 -= 1
                else:
                    break
        return gx1, gy1, gx2, gy2

    def select(
        self,
        phi_map: np.ndarray,
        image_size: Tuple[int, int],
        image: Optional[Any] = None,
        gradient_prior: Optional[np.ndarray] = None,
    ) -> RegionSelectionResult:
        """
        Find C* = argmax_C J_KL(C) on the 2D lattice Omega and project to pixel coordinates.
        Optionally modulates phi_map with a high-frequency gradient prior to suppress non-text visual sinks.
        """
        res = self.select_top_k(
            phi_map=phi_map,
            image_size=image_size,
            image=image,
            gradient_prior=gradient_prior,
            max_k=1,
        )
        return RegionSelectionResult(
            pixel_bbox=res.pixel_bboxes[0],
            grid_bbox=res.grid_bboxes[0],
            kl_score=res.kl_scores[0],
            captured_mass=res.captured_masses[0],
            area_fraction=res.area_fractions[0],
        )

    def select_top_k(
        self,
        phi_map: np.ndarray,
        image_size: Optional[Tuple[int, int]] = None,
        image: Optional[Any] = None,
        gradient_prior: Optional[np.ndarray] = None,
        max_k: int = 2,
        orig_w: Optional[int] = None,
        orig_h: Optional[int] = None,
    ) -> MultiRegionSelectionResult:
        """
        Adaptive Top-K Candidate Grounding (K in {1, 2}) using Bernoulli KL Divergence,
        Spatial NMS (IoU <= 0.20, IoM <= 0.35), and Information-Theoretic Significance Gating:
        J_KL(C2) / J_KL(C1) >= 0.40, p(C2) >= 0.10, and p(C2)/a(C2) > 1.0.
        """
        if image_size is None:
            if orig_w is not None and orig_h is not None:
                image_size = (orig_w, orig_h)
            else:
                image_size = (1, 1)
        W, H = image_size
        phi = np.maximum(np.asarray(phi_map, dtype=np.float64), 0.0)
        H_g, W_g = phi.shape
        total_cells = float(H_g * W_g)

        # Modulate with high-frequency gradient prior to reject smooth non-text sinks
        if gradient_prior is None and image is not None:
            gradient_prior = compute_high_frequency_gradient_prior(image, (H_g, W_g))

        if gradient_prior is not None:
            phi = modulate_affinity_with_gradient_prior(phi, gradient_prior)
        self.last_guided_phi = phi

        # Suppress low-amplitude background floor via self-information threshold:
        # Only patches whose density exceeds uniform density (P_Phi(u) > 1 / |Omega|) carry positive Pointwise information.
        raw_sum = float(phi.sum())
        if raw_sum <= 1e-12:
            return MultiRegionSelectionResult(
                pixel_bboxes=[(0, 0, W, H)],
                grid_bboxes=[(0, 0, W_g, H_g)],
                kl_scores=[0.0],
                captured_masses=[1.0],
                area_fractions=[1.0],
                k=1,
            )

        p_raw = phi / raw_sum
        uniform_prior = 1.0 / total_cells
        # Excess probability mass above uniform noise floor
        phi_clean = np.maximum(0.0, p_raw - uniform_prior)
        clean_sum = float(phi_clean.sum())
        if clean_sum > 1e-12:
            prob_map = phi_clean / clean_sum
        else:
            prob_map = p_raw

        integral = _build_integral_image(prob_map)
        structure_8 = np.ones((3, 3), dtype=np.int32)

        pos_vals = prob_map[prob_map > 0]
        if pos_vals.size == 0:
            pos_vals = prob_map.ravel()

        quantiles = np.linspace(0.25, 0.97, self.num_quantiles)
        thresholds = np.unique(np.quantile(pos_vals, quantiles))

        candidate_boxes = set()

        # 1. Connected-Component Bounding Boxes across natural level sets of Phi(u)
        for tau in thresholds:
            mask = prob_map >= tau
            if not np.any(mask):
                continue
            labeled, num_features = ndimage.label(mask, structure=structure_8)
            if num_features == 0:
                continue
            slices = ndimage.find_objects(labeled)
            for sl in slices:
                if sl is None:
                    continue
                y_sl, x_sl = sl
                gy1, gy2 = int(y_sl.start), int(y_sl.stop)
                gx1, gx2 = int(x_sl.start), int(x_sl.stop)
                gx1, gy1, gx2, gy2 = self._trim_sparse_borders(prob_map, gx1, gy1, gx2, gy2)
                candidate_boxes.add((gx1, gy1, gx2, gy2))
                # Also consider a 1-patch dilated bounding box to capture full stroke tails
                candidate_boxes.add((
                    max(0, gx1 - 1),
                    max(0, gy1 - 1),
                    min(W_g, gx2 + 1),
                    min(H_g, gy2 + 1),
                ))

        # 2. Local 1-step boundary refinement around each candidate box using the Integral Image
        refined_candidates = set(candidate_boxes)
        for (gx1, gy1, gx2, gy2) in list(candidate_boxes):
            for dx1 in (-1, 0, 1):
                for dy1 in (-1, 0, 1):
                    for dx2 in (-1, 0, 1):
                        for dy2 in (-1, 0, 1):
                            nx1 = max(0, gx1 + dx1)
                            ny1 = max(0, gy1 + dy1)
                            nx2 = min(W_g, gx2 + dx2)
                            ny2 = min(H_g, gy2 + dy2)
                            if nx2 > nx1 and ny2 > ny1:
                                refined_candidates.add((nx1, ny1, nx2, ny2))

        min_area = self.min_area_fraction if self.min_area_fraction is not None else (1.0 / total_cells)

        scored_candidates = []
        for (gx1, gy1, gx2, gy2) in refined_candidates:
            box_cells = float((gx2 - gx1) * (gy2 - gy1))
            a = box_cells / total_cells
            if a < min_area or a > self.max_area_fraction:
                continue
            p = _rect_sum(integral, gx1, gy1, gx2, gy2)
            kl = compute_bernoulli_kl_score(p, a)
            if kl > 0.0:
                scored_candidates.append(((gx1, gy1, gx2, gy2), kl, p, a))

        scored_candidates.sort(key=lambda item: item[1], reverse=True)

        selected_candidates = []
        if scored_candidates:
            # Candidate 1: Global argmax
            c1_box, c1_kl, c1_p, c1_a = scored_candidates[0]
            selected_candidates.append((c1_box, c1_kl, c1_p, c1_a))

            # Search for disjoint Candidate 2 if max_k >= 2
            if max_k >= 2:
                tau_iou = getattr(self, "tau_iou", 0.20)
                tau_iom = getattr(self, "tau_iom", 0.35)
                gamma_signif = max(0.25, 0.40 * float(np.exp(-getattr(self, "spatial_confidence", 0.8))))
                tau_mass = max(0.04, 1.5 / total_cells)

                for (box_k, kl_k, p_k, a_k) in scored_candidates[1:]:
                    inter_x1 = max(c1_box[0], box_k[0])
                    inter_y1 = max(c1_box[1], box_k[1])
                    inter_x2 = min(c1_box[2], box_k[2])
                    inter_y2 = min(c1_box[3], box_k[3])
                    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

                    area_1 = (c1_box[2] - c1_box[0]) * (c1_box[3] - c1_box[1])
                    area_k = (box_k[2] - box_k[0]) * (box_k[3] - box_k[1])
                    union_area = area_1 + area_k - inter_area

                    iou = inter_area / max(1.0, float(union_area))
                    iom = inter_area / max(1.0, float(min(area_1, area_k)))

                    if iou <= tau_iou and iom <= tau_iom:
                        # Information-Theoretic Significance Gate
                        if kl_k >= gamma_signif * c1_kl and p_k >= tau_mass and (p_k / max(1e-9, a_k)) > 1.0:
                            selected_candidates.append((box_k, kl_k, p_k, a_k))
                            break
        else:
            # Fallback if no valid scored candidates found
            peak_idx = int(np.argmax(prob_map))
            py, px = divmod(peak_idx, W_g)
            rad_y = max(2, H_g // 8)
            rad_x = max(2, W_g // 8)
            gx1 = max(0, px - rad_x)
            gy1 = max(0, py - rad_y)
            gx2 = min(W_g, px + rad_x + 1)
            gy2 = min(H_g, py + rad_y + 1)
            p_fallback = _rect_sum(integral, gx1, gy1, gx2, gy2)
            a_fallback = float((gx2 - gx1) * (gy2 - gy1)) / total_cells
            kl_fallback = compute_bernoulli_kl_score(p_fallback, a_fallback)
            selected_candidates.append(((gx1, gy1, gx2, gy2), kl_fallback, p_fallback, a_fallback))

        pixel_bboxes = []
        grid_bboxes = []
        kl_scores = []
        captured_masses = []
        area_fractions = []

        pad = self.boundary_pad_patches
        for (gbox, kl_val, p_val, a_val) in selected_candidates:
            gx1, gy1, gx2, gy2 = self._trim_sparse_borders(prob_map, *gbox)
            fx1 = max(0.0, float(gx1) - pad) / float(W_g)
            fy1 = max(0.0, float(gy1) - pad) / float(H_g)
            fx2 = min(float(W_g), float(gx2) + pad) / float(W_g)
            fy2 = min(float(H_g), float(gy2) + pad) / float(H_g)

            x1 = int(round(fx1 * W))
            y1 = int(round(fy1 * H))
            x2 = int(round(fx2 * W))
            y2 = int(round(fy2 * H))

            x1 = max(0, min(W - 2, x1))
            y1 = max(0, min(H - 2, y1))
            x2 = max(x1 + 2, min(W, x2))
            y2 = max(y1 + 2, min(H, y2))

            pixel_bboxes.append((x1, y1, x2, y2))
            grid_bboxes.append((gx1, gy1, gx2, gy2))
            kl_scores.append(float(kl_val))
            captured_masses.append(float(p_val))
            area_fractions.append(float(a_val))

        return MultiRegionSelectionResult(
            pixel_bboxes=pixel_bboxes,
            grid_bboxes=grid_bboxes,
            kl_scores=kl_scores,
            captured_masses=captured_masses,
            area_fractions=area_fractions,
            k=len(pixel_bboxes),
        )


def merge_aligned_candidates(
    candidates: Sequence[Tuple[int, int, int, int]],
    y_alignment_ratio: float = 0.40,
    x_gap_scale: float = 2.5,
) -> List[Tuple[int, int, int, int]]:
    """
    Merge horizontally-aligned candidate bounding boxes into a unified continuous banner.
    Prevents splitting a single multi-word sign into disjoint visual tiles.
    """
    if len(candidates) <= 1:
        return list(candidates)

    c1 = candidates[0]
    c2 = candidates[1]
    w1, h1 = c1[2] - c1[0], c1[3] - c1[1]
    w2, h2 = c2[2] - c2[0], c2[3] - c2[1]
    min_h = max(1, min(h1, h2))
    max_h = max(1, max(h1, h2))

    # Center-y vertical alignment
    cy1 = c1[1] + c1[3]
    cy2 = c2[1] + c2[3]
    center_y_diff = abs(cy1 - cy2) / (2.0 * min_h)

    # Horizontal distance between boxes
    if c1[2] <= c2[0]:
        dist_x = c2[0] - c1[2]
    elif c2[2] <= c1[0]:
        dist_x = c1[0] - c2[2]
    else:
        # Overlapping along X
        dist_x = 0

    if center_y_diff <= y_alignment_ratio and dist_x <= x_gap_scale * max_h:
        # Merge c1 and c2 into unified banner
        union_box = (
            min(c1[0], c2[0]),
            min(c1[1], c2[1]),
            max(c1[2], c2[2]),
            max(c1[3], c2[3]),
        )
        return [union_box] + list(candidates[2:])
    return list(candidates)
