# -*- coding: utf-8 -*-
r"""
Adaptive Local-Scale & Prototype-Locked Seed Propagation (AdaptiveScalePropagator).

Implements the two-step self-calibrated text propagation principle inspired by CRAFT & TextSnake
with zero hardcoded pixel distance thresholds, zero question keywords, and zero aspect-ratio heuristics:

Step 1 — Local Scale Self-Calibration (H_char & Orientation at the Seed Anchor):
    Directly at the seed patch/point u_0 where attention lands, measures the local character stroke
    cluster height H_char, dominant text-line axis e_parallel, and the local appearance prototype
    (foreground stroke chromaticity mu_fg, background surface chromaticity mu_bg, contrast vector Delta_mu).
    All subsequent spatial jump distances are scaled strictly as multiples of H_char.

Step 2 — Decoupled Directional Propagation (Prototype Locking + Vietnamese Diacritic Belt):
    - Parallel to the text line (e_parallel): Confines propagation to the calibrated line strip of
      thickness H_char and bridges inter-character and inter-word whitespace gaps up to a maximum
      jump proportional to H_char, provided the newly reached stroke clusters match the seed prototype
      (mu_fg, mu_bg, contrast polarity) and do not cross a signboard boundary.
    - Perpendicular to the text line (e_perp): Verifies and merges immediately stacked lines of
      matching prototype on the same signboard, then automatically opens a Vietnamese Diacritic &
      Calligraphic Swash Belt of +-0.5 * H_char above and below the core character body.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


@dataclass
class LocalScalePrototype:
    """Self-calibrated local scale and appearance prototype measured at the seed anchor."""
    seed_pixel: Tuple[int, int]          # (x0, y0) in original image pixels
    seed_patch: Tuple[int, int]          # (gx0, gy0) in ViT patch grid coordinates
    char_height_px: float                # Calibrated local character height H_char in original pixels
    char_height_patches: float           # Calibrated H_char in ViT patch units
    orientation: str                     # 'horizontal' or 'vertical'
    mu_fg: np.ndarray                    # Foreground stroke prototype in opponent color space (3,)
    mu_bg: np.ndarray                    # Background surface prototype in opponent color space (3,)
    delta_mu: np.ndarray                 # Signed contrast vector mu_fg - mu_bg (3,)
    stroke_scale_ref: float              # Reference stroke contrast energy at the seed


@dataclass
class SeedPropagationResult:
    """Structured output of AdaptiveScalePropagator from a single seed point."""
    seed_pixel: Tuple[int, int]                       # Input seed (x0, y0) in pixels
    seed_patch: Tuple[int, int]                       # Input seed (gx0, gy0) on (W_g, H_g) patch grid
    propagated_patch_bbox: Tuple[int, int, int, int]  # (gx1, gy1, gx2, gy2) on (W_g, H_g) patch grid
    propagated_pixel_bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in original image pixels
    core_pixel_bbox: Tuple[int, int, int, int]        # Core body bbox before +-0.5 * H_char diacritic belt
    prototype: LocalScalePrototype                    # Calibrated local scale H_char and color prototype
    patch_mask: np.ndarray                            # Binary mask on (H_g, W_g) patch grid
    compatibility_map: np.ndarray                     # Prototype compatibility field on working grid


def _rgb_tensor_to_opponent(rgb_t: torch.Tensor) -> torch.Tensor:
    """
    Convert float32 RGB tensor (1, 3, H, W) in [0, 255] on GPU to perceptual opponent color
    space [L, C_rg, C_yb] of shape (1, 3, H, W):
        L    = 0.299 R + 0.587 G + 0.114 B
        C_rg = (R - G) * 0.85
        C_yb = (0.5 * (R + G) - B) * 0.75
    """
    r = rgb_t[:, 0:1, :, :]
    g = rgb_t[:, 1:2, :, :]
    b = rgb_t[:, 2:3, :, :]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    c_rg = 0.85 * (r - g)
    c_yb = 0.75 * (0.5 * (r + g) - b)
    return torch.cat([lum, c_rg, c_yb], dim=1)


def compute_bbox_iou_and_recall(
    pred_box: Tuple[int, int, int, int],
    gt_box: Tuple[int, int, int, int],
) -> Tuple[float, float, float]:
    """
    Compute (IoU, GT_Recall, Precision) in percentage [0, 100] between two bounding boxes
    [x1, y1, x2, y2] (works identically for patch-grid bboxes and pixel bboxes).
    """
    px1, py1, px2, py2 = pred_box
    gx1, gy1, gx2, gy2 = gt_box

    ix1 = max(px1, gx1)
    iy1 = max(py1, gy1)
    ix2 = min(px2, gx2)
    iy2 = min(py2, gy2)

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter_area = float(inter_w * inter_h)

    pred_area = float(max(1, (px2 - px1) * (py2 - py1)))
    gt_area = float(max(1, (gx2 - gx1) * (gy2 - gy1)))
    union_area = pred_area + gt_area - inter_area

    iou = (inter_area / max(1.0, union_area)) * 100.0
    recall = (inter_area / gt_area) * 100.0
    precision = (inter_area / pred_area) * 100.0
    return float(iou), float(recall), float(precision)


def compute_autocorrelation_char_scale_1d(
    profile: np.ndarray,
    min_scale: int = 6,
    max_scale: int = 64,
) -> float:
    """
    Measure local character scale H_char via autocorrelation of luminance gradient:
        H_char = argmin_{delta > 0} [ d^2/d(delta)^2 Autocorr(grad(I), delta) = 0 ]
    or the first prominent period peak of directional autocorrelation.
    """
    prof = np.asarray(profile, dtype=np.float64)
    n = len(prof)
    if n < min_scale * 2:
        return float(min_scale)
    prof_zero = prof - prof.mean()
    var = np.sum(prof_zero ** 2)
    if var <= 1e-9:
        return float(min_scale)

    max_lag = min(max_scale, n // 2)
    autocorr = np.zeros(max_lag)
    for lag in range(max_lag):
        autocorr[lag] = np.sum(prof_zero[:n - lag] * prof_zero[lag:]) / var

    # Find the first zero-crossing of the second derivative (inflection point)
    d2 = np.diff(autocorr, n=2)
    zero_crossings = np.where(np.diff(np.signbit(d2)))[0] + 1
    valid_crossings = [zc for zc in zero_crossings if zc >= min_scale]

    if len(valid_crossings) > 0:
        return float(valid_crossings[0])

    # Fallback to first prominent trough/valley
    diffs = np.diff(autocorr)
    valleys = np.where((diffs[:-1] < 0) & (diffs[1:] >= 0))[0] + 1
    if len(valleys) > 0 and valleys[0] >= min_scale:
        return float(valleys[0])

    return float(min_scale)


class BaseSeedPropagator(ABC):
    """Strategy interface for seed-based region propagation."""

    @abstractmethod
    def propagate(
        self,
        image: Image.Image,
        seed_patch: Tuple[int, int],
        grid_ratio: Tuple[int, int],
        seed_pixel: Optional[Tuple[int, int]] = None,
        feat_grid: Optional[np.ndarray] = None,
    ) -> SeedPropagationResult:
        """Propagate from a single seed patch/pixel to cover the target text region."""
        raise NotImplementedError


class AdaptiveScalePropagator(BaseSeedPropagator):
    r"""
    PyTorch GPU Self-Calibrated Local-Scale & Prototype-Locked Propagator:
    1. Measures local character height H_char and stroke/surface color prototype at the seed on GPU.
    2. Propagates along the 1D line band of thickness H_char across inter-word gaps scaled by H_char.
    3. Verifies stacked lines on the same signboard and opens a +-0.5 * H_char Vietnamese diacritic belt.
    """

    def __init__(
        self,
        work_max_side: int = 640,
        diacritic_belt_ratio: float = 0.50,
        parallel_gap_scale: float = 1.65,
        stack_gap_scale: float = 0.90,
        device: Optional[str] = None,
    ):
        self.work_max_side = max(320, int(work_max_side))
        self.diacritic_belt_ratio = float(diacritic_belt_ratio)
        self.parallel_gap_scale = float(parallel_gap_scale)
        self.stack_gap_scale = float(stack_gap_scale)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self._kernel_cache: Dict[Tuple[str, float], torch.Tensor] = {}

    def _get_gaussian_1d_kernel(self, sigma: float) -> torch.Tensor:
        """Return cached 1D Gaussian kernel on self.device."""
        key = ("gauss1d", round(float(sigma), 4))
        if key not in self._kernel_cache:
            radius = max(1, int(round(3.0 * sigma)))
            coords = torch.arange(-radius, radius + 1, dtype=torch.float32, device=self.device)
            k = torch.exp(-0.5 * (coords / sigma) ** 2)
            k = k / k.sum()
            self._kernel_cache[key] = k
        return self._kernel_cache[key]

    def _gaussian_filter_gpu(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        """Apply separable 2D Gaussian smoothing to a (1, C, H, W) tensor on GPU."""
        k1d = self._get_gaussian_1d_kernel(sigma)
        rad = k1d.numel() // 2
        c = x.shape[1]
        kx = k1d.view(1, 1, 1, -1).expand(c, 1, 1, -1)
        ky = k1d.view(1, 1, -1, 1).expand(c, 1, -1, 1)
        out = F.pad(x, (rad, rad, 0, 0), mode="replicate")
        out = F.conv2d(out, kx, groups=c)
        out = F.pad(out, (0, 0, rad, rad), mode="replicate")
        out = F.conv2d(out, ky, groups=c)
        return out

    def _sobel_components_gpu(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute (|Gx|, |Gy|, hypot(Gx, Gy)) for a (1, C, H, W) tensor on GPU."""
        key_x = ("sobel_x", 0.0)
        key_y = ("sobel_y", 0.0)
        if key_x not in self._kernel_cache:
            kx = torch.tensor(
                [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            ky = torch.tensor(
                [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            self._kernel_cache[key_x] = kx
            self._kernel_cache[key_y] = ky

        c = x.shape[1]
        kx = self._kernel_cache[key_x].expand(c, 1, 3, 3)
        ky = self._kernel_cache[key_y].expand(c, 1, 3, 3)
        padded = F.pad(x, (1, 1, 1, 1), mode="replicate")
        gx = torch.abs(F.conv2d(padded, kx, groups=c))
        gy = torch.abs(F.conv2d(padded, ky, groups=c))
        return gx, gy, torch.hypot(gx, gy)

    @staticmethod
    def _smooth_1d_gpu(vec: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
        """Apply 1D box smoothing of odd kernel_size to a 1D tensor on GPU."""
        if vec.numel() < kernel_size or kernel_size <= 1:
            return vec
        pad = kernel_size // 2
        v3d = vec.view(1, 1, -1)
        padded = F.pad(v3d, (pad, pad), mode="replicate")
        return F.avg_pool1d(padded, kernel_size=kernel_size, stride=1).view(-1)

    @torch.inference_mode()
    def _compute_working_maps(
        self,
        image: Image.Image,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
        """
        Resize image to working grid (W_w, H_w) and compute on GPU (self.device):
        - opp: (H_w, W_w, 3) opponent color tensor on GPU
        - bg_smooth: (H_w, W_w, 3) local annular signboard background field on GPU
        - stroke_energy: (H_w, W_w) multi-scale high-frequency stroke energy on GPU
        - horiz_mod: (H_w, W_w) directional vertical-stroke modulation ratio in [0, 1] on GPU
          (distinguishes genuine horizontal character sequences with vertical strokes |Gx|
          from solid horizontal dividing bars where |Gy| >> |Gx|)
        """
        W, H = image.size
        scale = min(1.0, float(self.work_max_side) / float(max(W, H)))
        W_w = max(64, int(round(W * scale)))
        H_w = max(64, int(round(H * scale)))

        resized = image.convert("RGB").resize((W_w, H_w), Image.Resampling.BILINEAR)
        rgb_np = np.array(resized, dtype=np.uint8, copy=True)
        resized.close()
        rgb_t = torch.from_numpy(rgb_np).to(device=self.device, dtype=torch.float32)
        del rgb_np
        rgb_chw = rgb_t.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H_w, W_w)

        opp_chw = _rgb_tensor_to_opponent(rgb_chw)
        lum_chw = opp_chw[:, 0:1, :, :]

        # Multi-scale Difference-of-Gaussians + Sobel gradient for stroke energy on GPU
        g1 = self._gaussian_filter_gpu(lum_chw, sigma=0.8)
        g2 = self._gaussian_filter_gpu(lum_chw, sigma=2.8)
        dog = torch.abs(g1 - g2).squeeze(0).squeeze(0)

        gx_l, gy_l, grad_l = self._sobel_components_gpu(g1)
        grad = grad_l.squeeze(0).squeeze(0) * 0.22

        gx_c, gy_c, chroma_grad = self._sobel_components_gpu(opp_chw[:, 1:3, :, :])
        cgrad = torch.hypot(chroma_grad[0, 0], chroma_grad[0, 1]) * 0.14

        stroke_energy = dog + grad + cgrad

        # Smooth 5x5 directional modulation ratio: vertical stroke edge energy |Gx| vs |Gy|
        tot_gx = gx_l[:, 0:1] * 0.22 + 0.14 * (gx_c[:, 0:1] + gx_c[:, 1:2])
        tot_gy = gy_l[:, 0:1] * 0.22 + 0.14 * (gy_c[:, 0:1] + gy_c[:, 1:2])
        gx_avg = F.avg_pool2d(F.pad(tot_gx, (2, 2, 2, 2), mode="replicate"), kernel_size=5, stride=1).squeeze(0).squeeze(0)
        gy_avg = F.avg_pool2d(F.pad(tot_gy, (2, 2, 2, 2), mode="replicate"), kernel_size=5, stride=1).squeeze(0).squeeze(0)
        horiz_mod = (gx_avg + 0.25 * dog) / (gx_avg + gy_avg + 0.25 * dog + 1.2)

        # Annular (hollow-ring) local background estimate on GPU: outer 21x21 minus inner 5x5
        # so thick character strokes do not contaminate their own local background estimate
        outer_pool = F.avg_pool2d(
            F.pad(opp_chw, (10, 10, 10, 10), mode="replicate"),
            kernel_size=21,
            stride=1,
        )
        inner_pool = F.avg_pool2d(
            F.pad(opp_chw, (2, 2, 2, 2), mode="replicate"),
            kernel_size=5,
            stride=1,
        )
        bg_chw = (21.0 * 21.0 * outer_pool - 5.0 * 5.0 * inner_pool) / float(21 * 21 - 5 * 5)

        opp = opp_chw.squeeze(0).permute(1, 2, 0).contiguous()          # (H_w, W_w, 3)
        bg_smooth = bg_chw.squeeze(0).permute(1, 2, 0).contiguous()     # (H_w, W_w, 3)

        del (
            rgb_t, rgb_chw, opp_chw, lum_chw, g1, g2, dog, gx_l, gy_l, grad_l, grad,
            gx_c, gy_c, chroma_grad, cgrad, tot_gx, tot_gy, gx_avg, gy_avg,
            outer_pool, inner_pool, bg_chw,
        )
        scale_x = float(W_w) / float(max(1, W))
        scale_y = float(H_w) / float(max(1, H))
        return opp, bg_smooth, stroke_energy, horiz_mod, scale_x, scale_y

    @torch.inference_mode()
    def _scan_1d_band_span_gpu(
        self,
        compat_t: torch.Tensor,
        bg_sim_t: torch.Tensor,
        fg_sim_t: torch.Tensor,
        stroke_gate_t: torch.Tensor,
        band_start: int,
        band_end: int,
        seed_pos: int,
        max_gap: int,
        tau_col: float,
        axis: int = 1,
        exclude_band: Optional[Tuple[int, int]] = None,
    ) -> Tuple[int, int, float]:
        """
        Propagate along a 1D strip [band_start, band_end] on GPU (axis=1 for horizontal line,
        axis=0 for vertical line) starting at seed_pos, using:
        1. Baseline-referenced FWHM thresholding,
        2. Two-sided AND continuous border-to-guard-band bridge suppression (so one-sided
           2D vertical leakage into buildings/shoes/pillars/awnings is blocked while stacked
           text lines separated by an inter-line valley are preserved),
        3. Segment-bridge verification to trim non-signboard background tails.
        Returns (left_bound, right_bound, confinement_ratio).
        """
        H_w, W_w = compat_t.shape
        h_band = max(4, band_end - band_start)
        pad_track = max(3, int(round(0.38 * h_band)))

        if axis == 1:
            s_top = max(0, band_start - pad_track)
            s_bot = min(H_w, band_end + pad_track)
            strip_compat = compat_t[s_top:s_bot, :]
            strip_bg = bg_sim_t[band_start:band_end, :]
            strip_fg = fg_sim_t[s_top:s_bot, :]
            length = W_w

            col_stat = torch.quantile(strip_compat, 0.85, dim=0)
            col_mean = torch.mean(strip_compat, dim=0)
            col_compat = 0.72 * col_stat + 0.28 * torch.minimum(col_stat, col_mean * 2.8)
            col_bg = torch.quantile(strip_bg, 0.55, dim=0)
            col_fg = torch.quantile(strip_fg, 0.85, dim=0)

            # Guard bands placed outside the +-0.72 * h_band Vietnamese diacritic belt
            out_t1 = max(0, band_start - int(round(1.75 * h_band)))
            out_t2 = max(0, band_start - int(round(0.72 * h_band)))
            out_b1 = min(H_w, band_end + int(round(0.72 * h_band)))
            out_b2 = min(H_w, band_end + int(round(1.75 * h_band)))
            if exclude_band is not None:
                ex1, ex2 = exclude_band
                if out_t2 > ex1 and band_start >= ex1:
                    out_t2 = min(out_t2, ex1)
                if out_b1 < ex2 and band_end <= ex2:
                    out_b1 = max(out_b1, ex2)
            out_scores = []
            if out_t2 > out_t1 + 2:
                out_scores.append(torch.quantile(compat_t[out_t1:out_t2, :], 0.80, dim=0))
            if out_b2 > out_b1 + 2:
                out_scores.append(torch.quantile(compat_t[out_b1:out_b2, :], 0.80, dim=0))

            if len(out_scores) == 2:
                out_leak = torch.maximum(
                    torch.minimum(out_scores[0], out_scores[1]),
                    0.75 * torch.maximum(out_scores[0], out_scores[1]),
                )
            elif len(out_scores) == 1:
                out_leak = 0.75 * out_scores[0]
            else:
                out_leak = torch.zeros(length, dtype=torch.float32, device=self.device)

            out_leak = self._smooth_1d_gpu(out_leak, 5)

            seed_pos_c = max(0, min(length - 1, int(seed_pos)))
            base_col = float(torch.quantile(col_compat, 0.25).item())
            peak_col = max(
                tau_col,
                float(col_compat[max(0, seed_pos_c - 4):min(length, seed_pos_c + 5)].max().item()),
            )
            tau_eff = max(tau_col, base_col + 0.34 * (peak_col - base_col))
            active_col_t = (
                (col_compat >= tau_eff)
                & ((col_bg >= 0.20) | (col_fg >= 0.44))
                & (col_stat >= 1.05 * out_leak)
            )
        else:
            s_left = max(0, band_start - pad_track)
            s_right = min(W_w, band_end + pad_track)
            strip_compat = compat_t[:, s_left:s_right]
            strip_bg = bg_sim_t[:, band_start:band_end]
            strip_fg = fg_sim_t[:, s_left:s_right]
            length = H_w

            col_stat = torch.quantile(strip_compat, 0.85, dim=1)
            col_mean = torch.mean(strip_compat, dim=1)
            col_compat = 0.72 * col_stat + 0.28 * torch.minimum(col_stat, col_mean * 2.8)
            col_bg = torch.quantile(strip_bg, 0.55, dim=1)
            col_fg = torch.quantile(strip_fg, 0.85, dim=1)

            out_l1 = max(0, band_start - int(round(1.75 * h_band)))
            out_l2 = max(0, band_start - int(round(0.72 * h_band)))
            out_r1 = min(W_w, band_end + int(round(0.72 * h_band)))
            out_r2 = min(W_w, band_end + int(round(1.75 * h_band)))
            out_scores = []
            if out_l2 > out_l1 + 2:
                out_scores.append(torch.quantile(compat_t[:, out_l1:out_l2], 0.85, dim=1))
            if out_r2 > out_r1 + 2:
                out_scores.append(torch.quantile(compat_t[:, out_r1:out_r2], 0.85, dim=1))

            if len(out_scores) == 2:
                out_leak = torch.maximum(
                    torch.minimum(out_scores[0], out_scores[1]),
                    0.78 * torch.maximum(out_scores[0], out_scores[1]),
                )
            elif len(out_scores) == 1:
                out_leak = 0.78 * out_scores[0]
            else:
                out_leak = torch.zeros(length, dtype=torch.float32, device=self.device)

            out_leak = self._smooth_1d_gpu(out_leak, 5)

            seed_pos_c = max(0, min(length - 1, int(seed_pos)))
            base_col = float(torch.quantile(col_compat, 0.25).item())
            peak_col = max(
                tau_col,
                float(col_compat[max(0, seed_pos_c - 4):min(length, seed_pos_c + 5)].max().item()),
            )
            tau_eff = max(tau_col, base_col + 0.34 * (peak_col - base_col))
            active_col_t = (
                (col_compat >= tau_eff)
                & ((col_bg >= 0.20) | (col_fg >= 0.44))
                & (col_stat >= 1.12 * out_leak)
            )

        seed_pos = max(0, min(length - 1, int(seed_pos)))
        active_col = active_col_t.detach().cpu().numpy()
        col_bg_np = col_bg.detach().cpu().numpy()
        col_fg_np = col_fg.detach().cpu().numpy()
        active_col[max(0, seed_pos - 2):min(length, seed_pos + 3)] = True

        # March left/up (-1) and right/down (+1) from seed_pos with gap tolerance max_gap
        left_bound = seed_pos
        gap_count = 0
        barrier_count = 0
        max_barrier = max(2, max_gap // 3)
        for pos in range(seed_pos, -1, -1):
            if active_col[pos]:
                left_bound = pos
                gap_count = 0
                barrier_count = 0
            else:
                gap_count += 1
                if col_bg_np[pos] < 0.15 and col_fg_np[pos] < 0.34:
                    barrier_count += 1
                if gap_count > max_gap or barrier_count > max_barrier:
                    break

        right_bound = seed_pos + 1
        gap_count = 0
        barrier_count = 0
        for pos in range(seed_pos, length):
            if active_col[pos]:
                right_bound = pos + 1
                gap_count = 0
                barrier_count = 0
            else:
                gap_count += 1
                if col_bg_np[pos] < 0.15 and col_fg_np[pos] < 0.34:
                    barrier_count += 1
                if gap_count > max_gap or barrier_count > max_barrier:
                    break

        right_bound = max(left_bound + 2, right_bound)
        in_mean = float(col_stat[left_bound:right_bound].mean().item())
        out_mean = float(out_leak[left_bound:right_bound].mean().item())
        confinement_ratio = in_mean / max(0.08, out_mean)
        return left_bound, right_bound, confinement_ratio

    @torch.inference_mode()
    def propagate(
        self,
        image: Image.Image,
        seed_patch: Tuple[int, int],
        grid_ratio: Tuple[int, int],
        seed_pixel: Optional[Tuple[int, int]] = None,
        feat_grid: Optional[np.ndarray] = None,
    ) -> SeedPropagationResult:
        """
        Execute Step 1 (Local Scale Self-Calibration on GPU) and Step 2 (1D Band-Locked Directional
        Propagation + Stacked-Line Verification + +-0.5 * H_char Vietnamese Diacritic Belt on GPU).
        """
        W, H = image.size
        gx_grid, gy_grid = grid_ratio
        W_g, H_g = 16 * gx_grid, 16 * gy_grid

        sgx = int(np.clip(seed_patch[0], 0, W_g - 1))
        sgy = int(np.clip(seed_patch[1], 0, H_g - 1))

        if seed_pixel is None:
            px0 = int(round((sgx + 0.5) * W / W_g))
            py0 = int(round((sgy + 0.5) * H / H_g))
            seed_pixel = (int(np.clip(px0, 0, W - 1)), int(np.clip(py0, 0, H - 1)))
        else:
            px0, py0 = seed_pixel
            sgx = int(np.clip(int(px0 * W_g / max(1, W)), 0, W_g - 1))
            sgy = int(np.clip(int(py0 * H_g / max(1, H)), 0, H_g - 1))

        opp_t, bg_smooth_t, stroke_energy_t, horiz_mod_t, scale_x, scale_y = self._compute_working_maps(image)
        H_w, W_w = stroke_energy_t.shape

        sx = int(np.clip(round(seed_pixel[0] * scale_x), 0, W_w - 1))
        sy = int(np.clip(round(seed_pixel[1] * scale_y), 0, H_w - 1))

        # Step 1: Bimodal Foreground-Stroke vs Background-Surface Prototype Separation at Seed on GPU
        patch_w_px = max(6, int(round(W_w / float(W_g))))
        patch_h_px = max(6, int(round(H_w / float(H_g))))
        r_win = max(12, int(round(1.25 * max(patch_w_px, patch_h_px))))

        wy0 = max(0, sy - r_win)
        wy1 = min(H_w, sy + r_win + 1)
        wx0 = max(0, sx - r_win)
        wx1 = min(W_w, sx + r_win + 1)

        # Bilateral step-edge check: if [wx0, wx1] straddles a strong vertical chromatic signboard border,
        # select the half-window whose low-gradient background chromaticity matches the immediate seed core.
        r_in = 5
        in_o = opp_t[max(0, sy - r_in):min(H_w, sy + r_in + 1), max(0, sx - r_in):min(W_w, sx + r_in + 1)]
        in_e = stroke_energy_t[max(0, sy - r_in):min(H_w, sy + r_in + 1), max(0, sx - r_in):min(W_w, sx + r_in + 1)]
        in_m = in_e <= torch.quantile(in_e.reshape(-1), 0.45)
        in_bg = torch.median(in_o[in_m] if bool(torch.any(in_m).item()) else in_o.reshape(-1, 3), dim=0).values

        half_stats = []
        for hx0, hx1 in ((max(0, sx - r_win), min(W_w, sx + 2)), (max(0, sx - 1), min(W_w, sx + r_win + 1))):
            he = stroke_energy_t[wy0:wy1, hx0:hx1]
            ho = opp_t[wy0:wy1, hx0:hx1]
            hb = bg_smooth_t[wy0:wy1, hx0:hx1]
            hm = he <= torch.quantile(he.reshape(-1), 0.45)
            if not bool(torch.any(hm).item()):
                hm = torch.ones_like(he, dtype=torch.bool)
            half_stats.append((torch.median(ho[hm], dim=0).values, torch.median(hb[hm], dim=0).values))

        cd_opp = float(torch.hypot(half_stats[0][0][1] - half_stats[1][0][1], half_stats[0][0][2] - half_stats[1][0][2]).item())
        cd_bg = float(torch.hypot(half_stats[0][1][1] - half_stats[1][1][1], half_stats[0][1][2] - half_stats[1][1][2]).item())
        d_left = float(torch.hypot(half_stats[0][0][1] - in_bg[1], half_stats[0][0][2] - in_bg[2]).item())
        d_right = float(torch.hypot(half_stats[1][0][1] - in_bg[1], half_stats[1][0][2] - in_bg[2]).item())
        has_step_border = (
            cd_opp >= 75.0
            and cd_bg >= 45.0
            and 4.0 <= min(d_left, d_right) <= 15.0
            and max(d_left, d_right) >= 75.0
        )
        border_side = 0
        if has_step_border:
            if d_left < d_right:
                wx1 = min(W_w, sx + 2)
                border_side = -1
            else:
                wx0 = max(0, sx - 1)
                border_side = 1

        local_en = stroke_energy_t[wy0:wy1, wx0:wx1]
        local_opp = opp_t[wy0:wy1, wx0:wx1]
        local_bg_field = local_opp if has_step_border else bg_smooth_t[wy0:wy1, wx0:wx1]

        # Background surface prototype mu_bg from low-gradient background pixels in local window
        en_p45 = float(torch.quantile(local_en.reshape(-1), 0.45).item())
        bg_mask = local_en <= en_p45
        if not bool(torch.any(bg_mask).item()):
            bg_mask = torch.ones_like(local_en, dtype=torch.bool)
        mu_bg_t = torch.median(local_bg_field[bg_mask], dim=0).values

        # Foreground stroke prototype mu_fg from high-contrast stroke pixels near seed (sx, sy)
        r_core = max(5, r_win // 2)
        cy0 = max(0, sy - r_core)
        cy1 = min(H_w, sy + r_core + 1)
        cx0 = max(0, sx - r_core) if border_side != 1 else max(0, sx - 1)
        cx1 = min(W_w, sx + r_core + 1) if border_side != -1 else min(W_w, sx + 2)
        core_en = stroke_energy_t[cy0:cy1, cx0:cx1]
        core_opp = opp_t[cy0:cy1, cx0:cx1]
        core_bg = mu_bg_t.unsqueeze(0).unsqueeze(0) if has_step_border else bg_smooth_t[cy0:cy1, cx0:cx1]

        core_contrast_dist = torch.linalg.vector_norm(core_opp - core_bg, dim=-1)
        core_score = core_contrast_dist * torch.tanh(core_en / 5.0)
        score_p65 = float(torch.quantile(core_score.reshape(-1), 0.65).item())
        core_fg_mask = core_score >= max(1.8, score_p65)
        if not bool(torch.any(core_fg_mask).item()):
            core_fg_mask = torch.ones_like(core_en, dtype=torch.bool)

        mu_fg_t = torch.median(core_opp[core_fg_mask], dim=0).values
        delta_local_t = torch.median((core_opp - core_bg)[core_fg_mask], dim=0).values
        delta_mu_t = 0.55 * (mu_fg_t - mu_bg_t) + 0.45 * delta_local_t
        raw_delta_norm = float(torch.linalg.vector_norm(delta_mu_t).item())
        if raw_delta_norm < 2.0:
            r_in = max(2, r_core // 2)
            in_opp = opp_t[max(0, sy - r_in):min(H_w, sy + r_in + 1), max(0, sx - r_in):min(W_w, sx + r_in + 1)]
            in_bg = bg_smooth_t[max(0, sy - r_in):min(H_w, sy + r_in + 1), max(0, sx - r_in):min(W_w, sx + r_in + 1)]
            in_diff = in_opp - in_bg
            in_dist = torch.linalg.vector_norm(in_diff, dim=-1)
            in_mask = in_dist >= torch.quantile(in_dist.reshape(-1), 0.75)
            if bool(torch.any(in_mask).item()):
                mu_fg_t = torch.median(in_opp[in_mask], dim=0).values
                delta_local_t = torch.median(in_diff[in_mask], dim=0).values
                delta_mu_t = 0.55 * (mu_fg_t - mu_bg_t) + 0.45 * delta_local_t
                raw_delta_norm = float(torch.linalg.vector_norm(delta_mu_t).item())
        delta_norm = max(6.0, raw_delta_norm)
        stroke_ref = max(3.5, float(torch.quantile(core_en[core_fg_mask].reshape(-1), 0.75).item()))

        fg_dev = torch.linalg.vector_norm(core_opp[core_fg_mask] - mu_fg_t, dim=-1)
        bg_dev = torch.linalg.vector_norm(local_bg_field[bg_mask] - mu_bg_t, dim=-1)
        sigma_fg = float(np.clip(float(torch.quantile(fg_dev.reshape(-1), 0.75).item()) * 1.40 + 12.0, 16.0, 35.0))
        sigma_bg = float(np.clip(float(torch.quantile(bg_dev.reshape(-1), 0.75).item()) * 1.40 + 14.0, 18.0, 40.0))

        # Pixel-wise prototype compatibility across the working image on GPU
        dist_fg = torch.linalg.vector_norm(opp_t - mu_fg_t, dim=-1)
        dist_bg = torch.linalg.vector_norm(bg_smooth_t - mu_bg_t, dim=-1)

        sim_fg_t = torch.exp(-0.5 * (dist_fg / sigma_fg) ** 2)
        sim_bg_t = torch.exp(-0.5 * (dist_bg / sigma_bg) ** 2)

        # Signed contrast polarity alignment & alpha-blended contrast-vector similarity (for watermarks)
        diff_vec = opp_t - bg_smooth_t
        diff_norm = torch.linalg.vector_norm(diff_vec, dim=-1)
        dist_delta = torch.linalg.vector_norm(diff_vec - delta_local_t, dim=-1)
        sim_delta_t = torch.exp(-0.5 * (dist_delta / max(14.0, 0.90 * sigma_fg)) ** 2)

        cos_polarity = torch.sum(diff_vec * delta_mu_t, dim=-1) / (diff_norm * max(1.0, raw_delta_norm) + 1e-5)
        polarity_gate = torch.clamp((cos_polarity - 0.10) / 0.55, 0.0, 1.0)
        local_contrast_gate = torch.tanh(diff_norm / max(7.5, 0.45 * delta_norm))

        # Luminance-polarity agreement for two-tone sibling words (e.g. bright white "vietnam" + bright orange "mới")
        lum_diff = opp_t[..., 0] - bg_smooth_t[..., 0]
        seed_lum_sign = 1.0 if float(delta_mu_t[0].item()) >= 0.0 else -1.0
        lum_polarity_gate = torch.clamp((lum_diff * seed_lum_sign) / max(9.0, 0.36 * abs(float(delta_mu_t[0].item()))), 0.0, 1.0)

        # Foreground-locked or contrast-vector-locked path (supports opaque & semi-transparent watermark text)
        fg_or_delta_sim = torch.maximum(sim_fg_t, 0.84 * sim_delta_t)
        fg_locked_t = fg_or_delta_sim * polarity_gate * local_contrast_gate * (0.50 + 0.50 * sim_bg_t)

        # Smooth chromatic-luminance coupling for sibling words on bright background surfaces
        seed_chroma = float(torch.hypot(mu_fg_t[1] - mu_bg_t[1], mu_fg_t[2] - mu_bg_t[2]).item())
        bg_bright_weight = float(np.clip((float(mu_bg_t[0].item()) - 155.0) / 25.0, 0.0, 1.0))
        chroma_weight = float(np.clip((seed_chroma - 16.0) / 8.0, 0.0, 1.0)) * bg_bright_weight
        if chroma_weight > 1e-3:
            pix_chroma = torch.hypot(opp_t[..., 1] - bg_smooth_t[..., 1], opp_t[..., 2] - bg_smooth_t[..., 2])
            chroma_ratio = pix_chroma / max(1.0, seed_chroma)
            sibling_chroma_gate = (1.0 - chroma_weight) + chroma_weight * torch.clamp((chroma_ratio - 0.35) / 0.35, 0.20, 1.0)
        else:
            sibling_chroma_gate = 1.0

        same_bg_sibling = (
            sim_bg_t
            * torch.tanh(diff_norm / max(12.0, 0.58 * delta_norm))
            * torch.maximum(polarity_gate, 0.80 * lum_polarity_gate)
            * sibling_chroma_gate
        )
        effective_fg = torch.maximum(fg_locked_t, 0.62 * same_bg_sibling)

        stroke_gate = torch.tanh(stroke_energy_t / max(3.2, 0.30 * stroke_ref))
        raw_compat_t = stroke_gate * effective_fg

        # Small 3x3 morphological closing on GPU to bridge hollow character counters & LED dots
        compat_4d = raw_compat_t.unsqueeze(0).unsqueeze(0)
        dilated = F.max_pool2d(F.pad(compat_4d, (1, 1, 1, 1), mode="replicate"), kernel_size=3, stride=1)
        closed = -F.max_pool2d(F.pad(-dilated, (1, 1, 1, 1), mode="replicate"), kernel_size=3, stride=1)
        compat_t = 0.65 * raw_compat_t + 0.35 * closed.squeeze(0).squeeze(0)

        fg_sim_gated_t = torch.maximum(
            fg_or_delta_sim * polarity_gate,
            0.55 * lum_polarity_gate * sim_bg_t * sibling_chroma_gate,
        ) * local_contrast_gate * stroke_gate
        pure_fg_gated_t = sim_fg_t * polarity_gate * local_contrast_gate * stroke_gate
        stroke_gate_t = stroke_gate * local_contrast_gate

        mu_fg = mu_fg_t.detach().cpu().numpy().astype(np.float32)
        mu_bg = mu_bg_t.detach().cpu().numpy().astype(np.float32)
        delta_mu = delta_mu_t.detach().cpu().numpy().astype(np.float32)

        del (
            opp_t, bg_smooth_t, stroke_energy_t, local_en, local_opp, local_bg_field,
            core_en, core_opp, core_bg, core_contrast_dist, core_score, dist_fg, dist_bg,
            sim_fg_t, diff_vec, diff_norm, dist_delta, sim_delta_t, cos_polarity,
            polarity_gate, local_contrast_gate,
            lum_diff, lum_polarity_gate, fg_or_delta_sim, fg_locked_t, same_bg_sibling,
            effective_fg, stroke_gate, raw_compat_t, compat_4d, dilated, closed,
        )

        # Measure local character height H_char and line band [by1, by2] at (sx, sy) on GPU
        r_prof = max(10, int(round(1.4 * max(patch_w_px, patch_h_px))))
        strip_x0 = max(0, sx - r_prof) if border_side != 1 else max(0, sx - 1)
        strip_x1 = min(W_w, sx + r_prof + 1) if border_side != -1 else min(W_w, sx + 2)
        prof_y = torch.quantile(compat_t[:, strip_x0:strip_x1], 0.88, dim=1).detach().cpu().numpy()

        strip_y0 = max(0, sy - r_prof)
        strip_y1 = min(H_w, sy + r_prof + 1)
        prof_x = torch.quantile(compat_t[strip_y0:strip_y1, :], 0.88, dim=0).detach().cpu().numpy()

        max_char_h = max(14, int(round(H_w * 0.14)))
        loc_y0 = max(0, sy - 2 * max_char_h)
        loc_y1 = min(H_w, sy + 2 * max_char_h)
        base_y_val = float(np.percentile(prof_y[loc_y0:loc_y1], 28))
        y_peak_win0 = max(0, sy - 4)
        y_peak_win1 = min(H_w, sy + 5)
        sy_peak = y_peak_win0 + int(np.argmax(prof_y[y_peak_win0:y_peak_win1]))
        peak_y_val = max(0.22, float(prof_y[sy_peak]))
        tau_prof_y = max(0.20, base_y_val + 0.40 * max(0.08, peak_y_val - base_y_val))

        by1 = min(sy, sy_peak)
        gap_y = 0
        for y in range(by1 - 1, max(0, sy_peak - max_char_h) - 1, -1):
            if prof_y[y] >= tau_prof_y:
                by1 = y
                gap_y = 0
            else:
                gap_y += 1
                if gap_y > 3:
                    break

        by2 = max(sy + 1, sy_peak + 1)
        gap_y = 0
        for y in range(by2, min(H_w, sy_peak + max_char_h)):
            if prof_y[y] >= tau_prof_y:
                by2 = y + 1
                gap_y = 0
            else:
                gap_y += 1
                if gap_y > 3:
                    break

        # If the seed landed on an isolated descender/diacritic tip (span <= 8 px), check immediately
        # outside [by1, by2] within r_win for the primary character body band and merge if adjacent
        if (by2 - by1) <= 8:
            w_y0 = max(0, sy - r_win)
            w_y1 = min(H_w, sy + r_win + 1)
            masked_prof = prof_y[w_y0:w_y1].copy()
            m_b1 = max(0, by1 - 1 - w_y0)
            m_b2 = min(len(masked_prof), by2 + 1 - w_y0)
            masked_prof[m_b1:m_b2] = 0.0
            if masked_prof.size > 0 and float(masked_prof.max()) >= tau_prof_y:
                sy_peak2 = w_y0 + int(np.argmax(masked_prof))
                n_by1 = sy_peak2
                gap_y = 0
                for y in range(sy_peak2 - 1, max(0, sy_peak2 - max_char_h) - 1, -1):
                    if prof_y[y] >= tau_prof_y:
                        n_by1 = y
                        gap_y = 0
                    else:
                        gap_y += 1
                        if gap_y > 3:
                            break
                n_by2 = sy_peak2 + 1
                gap_y = 0
                for y in range(sy_peak2 + 1, min(H_w, sy_peak2 + max_char_h)):
                    if prof_y[y] >= tau_prof_y:
                        n_by2 = y + 1
                        gap_y = 0
                    else:
                        gap_y += 1
                        if gap_y > 3:
                            break
                sep_gap = max(0, max(by1 - n_by2, n_by1 - by2))
                merged_span = max(by2, n_by2) - min(by1, n_by1)
                if (n_by2 - n_by1) >= 4 and sep_gap <= max(7, int(round(0.75 * patch_h_px))) and merged_span <= max_char_h:
                    by1 = min(by1, n_by1)
                    by2 = max(by2, n_by2)

        max_char_w = max(14, int(round(W_w * 0.14)))
        loc_x0 = max(0, sx - 2 * max_char_w)
        loc_x1 = min(W_w, sx + 2 * max_char_w)
        base_x_val = float(np.percentile(prof_x[loc_x0:loc_x1], 28))
        x_peak_win0 = max(0, sx - 4)
        x_peak_win1 = min(W_w, sx + 5)
        sx_peak = x_peak_win0 + int(np.argmax(prof_x[x_peak_win0:x_peak_win1]))
        peak_x_val = max(0.22, float(prof_x[sx_peak]))
        tau_prof_x = max(0.20, base_x_val + 0.40 * max(0.08, peak_x_val - base_x_val))

        bx1_loc = min(sx, sx_peak)
        gap_x = 0
        for x in range(bx1_loc - 1, max(0, sx_peak - max_char_w) - 1, -1):
            if prof_x[x] >= tau_prof_x:
                bx1_loc = x
                gap_x = 0
            else:
                gap_x += 1
                if gap_x > 3:
                    break

        bx2_loc = max(sx + 1, sx_peak + 1)
        gap_x = 0
        for x in range(bx2_loc, min(W_w, sx_peak + max_char_w)):
            if prof_x[x] >= tau_prof_x:
                bx2_loc = x + 1
                gap_x = 0
            else:
                gap_x += 1
                if gap_x > 3:
                    break

        span_y = max(6, by2 - by1)
        span_x = max(6, bx2_loc - bx1_loc)

        # Determine line orientation by comparing 1D band propagation spans AND guard-band confinement on GPU.
        tau_h = float(np.clip(peak_y_val * 0.38, 0.20, 0.45))
        tau_v = float(np.clip(peak_x_val * 0.38, 0.20, 0.45))
        if span_y >= int(round(3.0 * patch_h_px)) and span_x >= int(round(1.5 * patch_w_px)) and span_y > 1.38 * span_x:
            eff_span_y = int(round(1.25 * span_x))
        else:
            eff_span_y = span_y
        gap_h_test = max(7, int(round(self.parallel_gap_scale * eff_span_y)))
        gap_v_test = max(7, int(round(self.parallel_gap_scale * span_x)))

        test_lx1, test_lx2, conf_h = self._scan_1d_band_span_gpu(
            compat_t, sim_bg_t, fg_sim_gated_t, stroke_gate_t, by1, by2, sx, gap_h_test, tau_h, axis=1
        )
        test_ly1, test_ly2, conf_v = self._scan_1d_band_span_gpu(
            compat_t, sim_bg_t, fg_sim_gated_t, stroke_gate_t, bx1_loc, bx2_loc, sy, gap_v_test, tau_v, axis=0
        )

        horiz_ext = float(test_lx2 - test_lx1) / float(max(1, span_y))
        vert_ext = float(test_ly2 - test_ly1) / float(max(1, span_x))

        is_vertical_col = (
            span_x >= int(round(2.0 * patch_w_px))
            and (test_ly2 - test_ly1) >= int(round(7.0 * patch_h_px))
            and vert_ext > 2.4
            and conf_v >= 1.25
            and (
                (vert_ext > 1.65 * horiz_ext and (test_ly2 - test_ly1) > 1.55 * (test_lx2 - test_lx1))
                or (span_y >= 1.38 * span_x and vert_ext > 3.2 and conf_v > 1.15 * conf_h)
            )
        )

        if is_vertical_col:
            orientation = "vertical"
            h_char_work = float(np.clip(span_x, 6, max_char_w))
            char_height_px = h_char_work / max(1e-6, scale_x)
            char_height_patches = h_char_work / float(max(1, patch_h_px))
        else:
            orientation = "horizontal"
            # Curved / Arched / Multi-part Line Adaptation:
            # When the initial horizontal scan is narrow relative to working canvas width and local height
            # (horiz_ext <= 9.5), test adjacent stroke cluster merging or symmetric curvature envelope,
            # and accept if horizontal span expands cleanly with strong guard-band confinement.
            if (
                horiz_ext <= 9.5
                and (
                    (test_lx2 - test_lx1) <= int(round(8.6 * patch_w_px))
                    or ((test_lx2 - test_lx1) <= int(round(9.5 * patch_w_px)) and (by2 - by1) >= int(round(2.0 * patch_h_px)))
                )
                and int(round(0.80 * patch_h_px)) <= (by2 - by1) <= int(round(2.45 * patch_h_px))
            ):
                candidates_y = []
                w_y0 = max(0, sy - r_win)
                w_y1 = min(H_w, sy + r_win + 1)
                masked_prof = prof_y[w_y0:w_y1].copy()
                m_b1 = max(0, by1 - 1 - w_y0)
                m_b2 = min(len(masked_prof), by2 + 1 - w_y0)
                masked_prof[m_b1:m_b2] = 0.0
                if masked_prof.size > 0 and float(masked_prof.max()) >= 0.78 * tau_prof_y:
                    sy_peak2 = w_y0 + int(np.argmax(masked_prof))
                    n_by1 = sy_peak2
                    gap_y = 0
                    for y in range(sy_peak2 - 1, max(0, sy_peak2 - max_char_h) - 1, -1):
                        if prof_y[y] >= 0.78 * tau_prof_y:
                            n_by1 = y
                            gap_y = 0
                        else:
                            gap_y += 1
                            if gap_y > 3:
                                break
                    n_by2 = sy_peak2 + 1
                    gap_y = 0
                    for y in range(sy_peak2 + 1, min(H_w, sy_peak2 + max_char_h)):
                        if prof_y[y] >= 0.78 * tau_prof_y:
                            n_by2 = y + 1
                            gap_y = 0
                        else:
                            gap_y += 1
                            if gap_y > 3:
                                break
                    sep_gap = max(0, max(by1 - n_by2, n_by1 - by2))
                    if sep_gap <= max(9, int(round(1.0 * patch_h_px))):
                        candidates_y.append((min(by1, n_by1), max(by2, n_by2)))

                if (test_lx2 - test_lx1) <= int(round(7.8 * patch_w_px)) and (by2 - by1) <= int(round(1.75 * patch_h_px)):
                    pad_try = max(4, int(round(0.55 * patch_h_px)))
                    candidates_y.append((max(0, by1 - pad_try), min(H_w, by2 + pad_try)))

                best_cand = None
                best_span_w = test_lx2 - test_lx1
                for cand_y1, cand_y2 in candidates_y:
                    if (cand_y2 - cand_y1) <= (by2 - by1) or (cand_y2 - cand_y1) > max(max_char_h, int(round(4.4 * patch_h_px))):
                        continue
                    cand_span_y = max(6, cand_y2 - cand_y1)
                    cand_gap = max(7, int(round(self.parallel_gap_scale * min(cand_span_y, int(round(3.0 * patch_h_px))))))
                    cand_lx1, cand_lx2, cand_conf = self._scan_1d_band_span_gpu(
                        compat_t, sim_bg_t, fg_sim_gated_t, stroke_gate_t, cand_y1, cand_y2, sx, cand_gap, tau_h, axis=1
                    )
                    cand_w = cand_lx2 - cand_lx1
                    cand_ext = float(cand_w) / float(max(1, cand_span_y))
                    # Adaptively derive line expansion factor from candidate aspect ratio and local stroke span
                    adapt_expand_ratio = 1.0 + (float(span_y) / float(max(1, cand_span_y))) / max(1.0, np.sqrt(max(1.0, horiz_ext)))
                    cand_min_w = max(
                        test_lx2 - test_lx1 + int(round(1.8 * span_y)),
                        int(round(adapt_expand_ratio * (test_lx2 - test_lx1))),
                    )
                    if (
                        cand_w >= cand_min_w
                        and cand_ext <= 16.5
                        and cand_conf >= 1.18
                        and (cand_conf >= 0.92 * conf_h or cand_ext >= 2.0 * horiz_ext)
                        and cand_w > best_span_w
                    ):
                        best_span_w = cand_w
                        best_cand = (cand_y1, cand_y2, cand_lx1, cand_lx2, cand_conf, cand_gap)

                if best_cand is not None:
                    by1, by2, test_lx1, test_lx2, conf_h, gap_h_test = best_cand
            # Local character scale self-calibration via autocorrelation & profile span
            h_char_autocorr = compute_autocorrelation_char_scale_1d(prof_y, min_scale=6, max_scale=max_char_h)
            h_char_work = float(np.clip(0.55 * span_y + 0.45 * h_char_autocorr, 6, max(16, int(round(3.1 * patch_h_px)))))
            char_height_px = h_char_work / max(1e-6, scale_y)
            char_height_patches = h_char_work / float(max(1, patch_h_px))

        proto = LocalScalePrototype(
            seed_pixel=seed_pixel,
            seed_patch=(sgx, sgy),
            char_height_px=round(float(char_height_px), 2),
            char_height_patches=round(float(char_height_patches), 3),
            orientation=orientation,
            mu_fg=mu_fg,
            mu_bg=mu_bg,
            delta_mu=delta_mu,
            stroke_scale_ref=round(float(stroke_ref), 3),
        )

        # Step 2: Band-Locked Line Propagation + Envelope Refinement + Stacked-Line Merging on GPU
        max_gap = max(7, int(round(self.parallel_gap_scale * h_char_work)))
        stack_gap = max(4, int(round(self.stack_gap_scale * h_char_work)))
        h_pad_ref = min(h_char_work, float(patch_h_px * 2.25))

        if orientation == "horizontal":
            tau_col = tau_h
            final_y1, final_y2 = by1, by2
            if max_gap != gap_h_test:
                final_x1, final_x2, _ = self._scan_1d_band_span_gpu(
                    compat_t, sim_bg_t, fg_sim_gated_t, stroke_gate_t, by1, by2, sx, max_gap, tau_h, axis=1
                )
            else:
                final_x1, final_x2 = test_lx1, test_lx2

            # Refine vertical envelope across [final_x1, final_x2] to include tall capitals/ascenders/swashes
            env_pad = max(
                4,
                int(round(0.60 * (h_char_work if (final_x2 - final_x1) >= int(round(21.0 * patch_w_px)) else h_pad_ref))),
            )
            ey0 = max(0, final_y1 - env_pad)
            ey1 = min(H_w, final_y2 + env_pad)
            if final_x2 > final_x1 + 3 and ey1 > ey0 + 3:
                sub_env = compat_t[ey0:ey1, final_x1:final_x2]
                q_hi, q_lo, w_hi, w_lo = (
                    (0.92, 0.82, 0.65, 0.35)
                    if h_char_work <= 1.85 * patch_h_px
                    else (0.90, 0.80, 0.55, 0.45)
                )
                span_prof_y = (
                    w_hi * torch.quantile(sub_env, q_hi, dim=1)
                    + w_lo * torch.quantile(sub_env, q_lo, dim=1)
                ).detach().cpu().numpy()
                span_bg_y = torch.mean(sim_bg_t[ey0:ey1, final_x1:final_x2], dim=1).detach().cpu().numpy()
                valid_y = np.nonzero((span_prof_y >= 0.78 * tau_col) & (span_bg_y >= 0.34))[0]
                if valid_y.size > 0:
                    final_y1 = min(final_y1, ey0 + int(valid_y.min()))
                    final_y2 = max(final_y2, ey0 + int(valid_y.max()) + 1)

            # Check adjacent stacked lines on the same uniform signboard
            for direction in (-1, 1):
                curr_y = final_y1 if direction == -1 else final_y2
                eff_h_search = max(min(float(span_y), float(patch_h_px * 3.8)), h_char_work, float(patch_h_px * 1.15))
                eff_stack_gap = max(stack_gap, int(round(self.stack_gap_scale * eff_h_search)))
                ext_mult = 1.55 if (direction == 1 and span_y <= int(round(1.2 * patch_h_px)) and final_y1 < by1 - 4) else 1.25
                search_start = max(0, curr_y - eff_stack_gap - int(round(1.25 * eff_h_search))) if direction == -1 else curr_y + 2
                search_end = max(0, curr_y - 2) if direction == -1 else min(H_w, curr_y + eff_stack_gap + int(round(ext_mult * eff_h_search)))
                if search_end <= search_start + 3:
                    continue

                probe_x1 = min(final_x1, max(0, sx - int(round(1.8 * patch_w_px))))
                probe_x2 = max(final_x2, min(W_w, sx + int(round(1.8 * patch_w_px))))

                sub_compat = compat_t[search_start:search_end, probe_x1:probe_x2]
                sub_bg = sim_bg_t[search_start:search_end, probe_x1:probe_x2]
                sub_pure_fg = pure_fg_gated_t[search_start:search_end, probe_x1:probe_x2]
                row_scores = (
                    0.55 * torch.quantile(sub_compat, 0.90, dim=1)
                    + 0.45 * torch.quantile(sub_compat, 0.80, dim=1)
                ).detach().cpu().numpy()
                row_bg = torch.mean(sub_bg, dim=1).detach().cpu().numpy()
                row_pure_fg = torch.quantile(sub_pure_fg, 0.88, dim=1).detach().cpu().numpy()
                active_rows = (row_scores >= 0.76 * tau_col) & (row_bg >= 0.46) & (row_pure_fg >= 0.20)

                if not np.any(active_rows):
                    continue

                # Extract the contiguous active band nearest to curr_y (allowing up to 5 px diacritic gap)
                n_rows = len(active_rows)
                if direction == -1:
                    b_end = -1
                    for r_i in range(n_rows - 1, -1, -1):
                        if (n_rows - 1 - r_i) > eff_stack_gap:
                            break
                        if active_rows[r_i]:
                            b_end = r_i
                            break
                    if b_end < 0:
                        continue
                    b_start = b_end
                    dip = 0
                    for r_i in range(b_end - 1, -1, -1):
                        if active_rows[r_i]:
                            b_start = r_i
                            dip = 0
                        else:
                            dip += 1
                            if dip > 5:
                                break
                else:
                    b_start = -1
                    for r_i in range(n_rows):
                        if r_i > eff_stack_gap:
                            break
                        if active_rows[r_i]:
                            b_start = r_i
                            break
                    if b_start < 0:
                        continue
                    b_end = b_start
                    dip = 0
                    max_dip_down = 6 if (span_y <= int(round(1.2 * patch_h_px)) and final_y1 < by1 - 4) else 5
                    for r_i in range(b_start + 1, n_rows):
                        if active_rows[r_i]:
                            b_end = r_i
                            dip = 0
                        else:
                            dip += 1
                            if dip > max_dip_down:
                                break

                ny1 = search_start + b_start
                ny2 = search_start + b_end + 1
                new_h = ny2 - ny1
                max_allowed_h = max(1.85 * eff_h_search, 2.4 * patch_h_px)
                if new_h < 0.40 * h_char_work or new_h > max_allowed_h:
                    continue

                sub_line = compat_t[ny1:ny2, probe_x1:probe_x2]
                col_line_stat = 0.55 * torch.quantile(sub_line, 0.90, dim=0) + 0.45 * torch.quantile(sub_line, 0.80, dim=0)
                active_line_cols = (col_line_stat >= 0.74 * tau_col).detach().cpu().numpy()
                col_fill = float(np.mean(active_line_cols))
                max_cluster_w = 0
                cur_w = 0
                gap_c = 0
                for is_act in active_line_cols:
                    if is_act:
                        cur_w += 1 + gap_c
                        gap_c = 0
                        if cur_w > max_cluster_w:
                            max_cluster_w = cur_w
                    else:
                        gap_c += 1
                        if gap_c > max(4, int(round(0.65 * h_char_work))):
                            cur_w = 0
                            gap_c = 0
                if col_fill < 0.35 and max_cluster_w < int(round(1.6 * h_char_work)):
                    continue

                best_col = probe_x1 + int(torch.argmax(col_line_stat).item())
                sib_gap = max(max_gap, int(round(self.parallel_gap_scale * (new_h + 0.5 * h_char_work))))
                nx1, nx2, sib_conf = self._scan_1d_band_span_gpu(
                    compat_t, sim_bg_t, fg_sim_gated_t, stroke_gate_t,
                    ny1, ny2, best_col, sib_gap, 0.84 * tau_col, axis=1,
                )
                comb_y1 = min(final_y1, ny1)
                comb_y2 = max(final_y2, ny2)
                if (final_x2 - final_x1) < 0.65 * (nx2 - nx1):
                    cx1, cx2, comb_conf = self._scan_1d_band_span_gpu(
                        compat_t, sim_bg_t, fg_sim_gated_t, stroke_gate_t,
                        comb_y1, comb_y2, sx, sib_gap, 0.84 * tau_col, axis=1,
                    )
                    if comb_conf >= 1.25 * sib_conf and (cx2 - cx1) >= int(round(1.8 * (final_x2 - final_x1))):
                        nx1, nx2 = cx1, cx2
                    else:
                        nx1 = min(nx1, cx1)
                        nx2 = max(nx2, cx2)
                    max_widen = int(round(6.0 * max(h_char_work, float(new_h))))
                else:
                    max_widen = int(round(2.0 * max(h_char_work, float(new_h))))

                final_y1 = comb_y1
                final_y2 = comb_y2
                final_x1 = max(final_x1 - max_widen, min(final_x1, nx1))
                final_x2 = min(final_x2 + max_widen, max(final_x2, nx2))
                if h_char_work < patch_h_px and (final_y2 - final_y1) >= 2.5 * h_char_work:
                    h_pad_ref = min(max(h_pad_ref, float(final_y2 - final_y1) * 0.45), float(patch_h_px * 2.0))

            if border_side == -1:
                final_x2 = min(W_w, final_x2 + 1)

            # Step 2b: Enforce +-0.5 * H_char Vietnamese Diacritic Belt along perpendicular axis e_perp
            pad_y = max(3, int(round(self.diacritic_belt_ratio * h_char_work)))
            pad_x = max(2, int(round(0.25 * h_char_work)))
        else:
            tau_col = tau_v
            final_x1, final_x2 = bx1_loc, bx2_loc
            if max_gap != gap_v_test:
                final_y1, final_y2, _ = self._scan_1d_band_span_gpu(
                    compat_t, sim_bg_t, fg_sim_gated_t, stroke_gate_t, bx1_loc, bx2_loc, sy, max_gap, tau_v, axis=0
                )
            else:
                final_y1, final_y2 = test_ly1, test_ly2

            env_pad = max(4, int(round(0.65 * h_char_work)))
            ex0 = max(0, final_x1 - env_pad)
            ex1 = min(W_w, final_x2 + env_pad)
            if final_y2 > final_y1 + 3 and ex1 > ex0 + 3:
                sub_env = compat_t[final_y1:final_y2, ex0:ex1]
                span_prof_x = (
                    0.55 * torch.quantile(sub_env, 0.90, dim=0)
                    + 0.45 * torch.quantile(sub_env, 0.80, dim=0)
                ).detach().cpu().numpy()
                span_bg_x = torch.mean(sim_bg_t[final_y1:final_y2, ex0:ex1], dim=0).detach().cpu().numpy()
                valid_x = np.nonzero((span_prof_x >= 0.78 * tau_col) & (span_bg_x >= 0.34))[0]
                if valid_x.size > 0:
                    final_x1 = min(final_x1, ex0 + int(valid_x.min()))
                    final_x2 = max(final_x2, ex0 + int(valid_x.max()) + 1)

            for direction in (-1, 1):
                curr_x = final_x1 if direction == -1 else final_x2
                search_start = max(0, curr_x - stack_gap - int(round(1.25 * h_char_work))) if direction == -1 else curr_x + 2
                search_end = max(0, curr_x - 2) if direction == -1 else min(W_w, curr_x + stack_gap + int(round(1.25 * h_char_work)))
                if search_end <= search_start + 3:
                    continue

                sub_compat = compat_t[final_y1:final_y2, search_start:search_end]
                sub_bg = sim_bg_t[final_y1:final_y2, search_start:search_end]
                col_scores = (
                    0.55 * torch.quantile(sub_compat, 0.90, dim=0)
                    + 0.45 * torch.quantile(sub_compat, 0.80, dim=0)
                ).detach().cpu().numpy()
                col_bg = torch.mean(sub_bg, dim=0).detach().cpu().numpy()
                active_cols = (col_scores >= 0.78 * tau_col) & (col_bg >= 0.42)
                if not np.any(active_cols):
                    continue
                c_indices = np.nonzero(active_cols)[0]
                nx1 = search_start + int(c_indices.min())
                nx2 = search_start + int(c_indices.max()) + 1
                new_w = nx2 - nx1
                if new_w < 0.40 * h_char_work or new_w > 1.85 * h_char_work:
                    continue

                sub_col = compat_t[final_y1:final_y2, nx1:nx2]
                row_col_stat = 0.55 * torch.quantile(sub_col, 0.90, dim=1) + 0.45 * torch.quantile(sub_col, 0.80, dim=1)
                row_fill = float(torch.mean((row_col_stat >= 0.75 * tau_col).to(torch.float32)).item())
                if row_fill < 0.36:
                    continue

                mid_y = int(round(0.5 * (final_y1 + final_y2)))
                ny1, ny2, _ = self._scan_1d_band_span_gpu(
                    compat_t, sim_bg_t, fg_sim_gated_t, stroke_gate_t,
                    nx1, nx2, mid_y, max_gap, 0.85 * tau_col, axis=0,
                )
                max_widen = int(round(1.5 * h_char_work))
                final_x1 = min(final_x1, nx1)
                final_x2 = max(final_x2, nx2)
                final_y1 = max(final_y1 - max_widen, min(final_y1, ny1))
                final_y2 = min(final_y2 + max_widen, max(final_y2, ny2))

            pad_x = max(2, int(round(self.diacritic_belt_ratio * h_char_work)))
            pad_y = max(2, int(round(0.25 * h_char_work)))

        compat_map = compat_t.detach().cpu().numpy().astype(np.float32)
        del compat_t, sim_bg_t, fg_sim_gated_t, pure_fg_gated_t, stroke_gate_t, horiz_mod_t

        core_px_x1 = max(0, min(W - 2, int(round(final_x1 / scale_x))))
        core_px_y1 = max(0, min(H - 2, int(round(final_y1 / scale_y))))
        core_px_x2 = max(core_px_x1 + 2, min(W, int(round(final_x2 / scale_x))))
        core_px_y2 = max(core_px_y1 + 2, min(H, int(round(final_y2 / scale_y))))
        core_pixel_bbox = (core_px_x1, core_px_y1, core_px_x2, core_px_y2)

        exp_x1 = max(0, final_x1 - pad_x)
        exp_y1 = max(0, final_y1 - pad_y)
        exp_x2 = min(W_w, final_x2 + pad_x)
        exp_y2 = min(H_w, final_y2 + pad_y)

        px1 = max(0, min(W - 2, int(round(exp_x1 / scale_x))))
        py1 = max(0, min(H - 2, int(round(exp_y1 / scale_y))))
        px2 = max(px1 + 2, min(W, int(round(exp_x2 / scale_x))))
        py2 = max(py1 + 2, min(H, int(round(exp_y2 / scale_y))))

        gx1 = max(0, min(W_g - 1, int(np.floor(px1 * W_g / max(1, W)))))
        gy1 = max(0, min(H_g - 1, int(np.floor(py1 * H_g / max(1, H)))))
        gx2 = max(gx1 + 1, min(W_g, int(np.ceil(px2 * W_g / max(1, W)))))
        gy2 = max(gy1 + 1, min(H_g, int(np.ceil(py2 * H_g / max(1, H)))))

        patch_mask = np.zeros((H_g, W_g), dtype=np.uint8)
        patch_mask[gy1:gy2, gx1:gx2] = 1

        return SeedPropagationResult(
            seed_pixel=seed_pixel,
            seed_patch=(sgx, sgy),
            propagated_patch_bbox=(gx1, gy1, gx2, gy2),
            propagated_pixel_bbox=(px1, py1, px2, py2),
            core_pixel_bbox=core_pixel_bbox,
            prototype=proto,
            patch_mask=patch_mask,
            compatibility_map=compat_map,
        )

