# -*- coding: utf-8 -*-
"""
Gate A Unit & Mock Integration Tests for TICKET-OPT-04:
GPU Affinity Field Vectorization (`compute_intrinsic_affinity_field_gpu`).

Validates:
- Test A: Direct numerical parity (`phi_map` and `aff_map`, max abs diff <= 1e-4)
  across all 10 deterministic synthetic attention/feature-grid regimes.
- Test B: Device & dtype contract verification (`float32` and `bfloat16` promotion,
  no `.cpu()` / `.numpy()` transfer on `feat_grid`).
- Test C: Boundary & no-wrap-around verification (opposing borders do not bridge).
- Test D: Pass-0 (`M_cont` + `M_gap`) vs Pass-1 (`M_gap` only) 2-pass semantics.
- Test E: Stage 1 `KLDivergenceRegionSelector.select_top_k` bounding-box parity
  (`IoU = 1.0` and exact box coordinates across all 10 synthetic regimes).
- Test F: `PointwiseKLAttentionExtractor.extract()` mock end-to-end integration,
  image-level `grid_feats` tensor caching across Q1/Q2, and `clear_cache()` lifecycle.
"""

import types
import unittest
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from scripts.benchmark.sg_vcd.adaptive_propagator import compute_bbox_iou_and_recall
from scripts.benchmark.sg_vcd.attention_extractor import (
    PointwiseKLAttentionExtractor,
    _compute_intrinsic_affinity_field_numpy_reference,
    compute_intrinsic_affinity_field,
    compute_intrinsic_affinity_field_gpu,
)
from scripts.benchmark.sg_vcd.bbox_selector import KLDivergenceRegionSelector


def _build_10_synthetic_regimes() -> List[Dict[str, Any]]:
    """
    Construct the 10 deterministic synthetic attention/feature-grid regimes
    specified in TASK 14 Section 6.
    """
    regimes: List[Dict[str, Any]] = []

    # 1. Single compact text block (16 x 16, D = 64)
    rng1 = np.random.RandomState(101)
    r1 = np.zeros((16, 16), dtype=np.float32)
    r1[5:10, 4:12] = rng1.uniform(0.5, 1.0, size=(5, 8)).astype(np.float32)
    aq1 = r1 * 0.9 + 0.05
    feat1 = rng1.normal(0.0, 0.2, size=(16, 16, 64)).astype(np.float32)
    proto1 = rng1.normal(1.0, 0.1, size=(64,)).astype(np.float32)
    feat1[4:11, 3:13] += proto1
    regimes.append(
        {
            "name": "01_single_compact_text_block_16x16_D64",
            "r_map": r1,
            "feat_grid": feat1,
            "a_query_map": aq1,
            "max_bridge_radius": 2,
            "image_size": (320, 320),
        }
    )

    # 2. Multi-word horizontal line with 1-patch and 2-patch gaps (16 x 32, D = 64)
    rng2 = np.random.RandomState(102)
    r2 = np.zeros((16, 32), dtype=np.float32)
    # Word 1: cols 3..7, 1-patch gap at col 8, Word 2: cols 9..14, 2-patch gap at cols 15..16, Word 3: cols 17..23
    r2[6:9, 3:8] = 0.85
    r2[6:9, 9:15] = 0.90
    r2[6:9, 17:24] = 0.80
    aq2 = np.copy(r2)
    aq2[6:9, 2:26] = np.maximum(aq2[6:9, 2:26], 0.45)
    feat2 = rng2.normal(0.0, 0.05, size=(16, 32, 64)).astype(np.float32)
    line_proto = rng2.uniform(0.5, 1.5, size=(64,)).astype(np.float32)
    feat2[6:9, 2:26] += line_proto
    regimes.append(
        {
            "name": "02_multi_word_horizontal_gaps_16x32_D64",
            "r_map": r2,
            "feat_grid": feat2,
            "a_query_map": aq2,
            "max_bridge_radius": 2,
            "image_size": (640, 320),
        }
    )

    # 3. Multi-line vertical stack with 1-patch and 2-patch row gaps (32 x 16, D = 64)
    rng3 = np.random.RandomState(103)
    r3 = np.zeros((32, 16), dtype=np.float32)
    # Line 1: rows 4..6, gap row 7, Line 2: rows 8..10, 2-row gap 11..12, Line 3: rows 13..16
    r3[4:7, 4:12] = 0.88
    r3[8:11, 4:12] = 0.92
    r3[13:17, 4:12] = 0.84
    aq3 = np.copy(r3)
    aq3[3:18, 4:12] = np.maximum(aq3[3:18, 4:12], 0.40)
    feat3 = rng3.normal(0.0, 0.05, size=(32, 16, 64)).astype(np.float32)
    stack_proto = rng3.uniform(0.4, 1.2, size=(64,)).astype(np.float32)
    feat3[3:18, 4:12] += stack_proto
    regimes.append(
        {
            "name": "03_multi_line_vertical_stack_32x16_D64",
            "r_map": r3,
            "feat_grid": feat3,
            "a_query_map": aq3,
            "max_bridge_radius": 2,
            "image_size": (320, 640),
        }
    )

    # 4. Diagonal stroke / tilted signboard pattern (32 x 32, D = 128)
    rng4 = np.random.RandomState(104)
    r4 = np.zeros((32, 32), dtype=np.float32)
    aq4 = np.zeros((32, 32), dtype=np.float32)
    feat4 = rng4.normal(0.0, 0.05, size=(32, 32, 128)).astype(np.float32)
    diag_proto = rng4.uniform(0.5, 1.5, size=(128,)).astype(np.float32)
    for t in range(5, 25):
        if t not in (11, 17, 18):  # 1-step and 2-step diagonal gaps
            r4[t, t] = 0.9
            r4[t, min(31, t + 1)] = 0.75
        aq4[t, t] = 0.65
        feat4[t, t] += diag_proto
        feat4[t, min(31, t + 1)] += diag_proto
    regimes.append(
        {
            "name": "04_diagonal_stroke_tilted_signboard_32x32_D128",
            "r_map": r4,
            "feat_grid": feat4,
            "a_query_map": aq4,
            "max_bridge_radius": 2,
            "image_size": (512, 512),
        }
    )

    # 5. Anti-diagonal ((1, -1)) stroke with tail continuation (32 x 32, D = 128)
    rng5 = np.random.RandomState(105)
    r5 = np.zeros((32, 32), dtype=np.float32)
    aq5 = np.zeros((32, 32), dtype=np.float32)
    feat5 = rng5.normal(0.0, 0.04, size=(32, 32, 128)).astype(np.float32)
    anti_proto = rng5.uniform(0.6, 1.4, size=(128,)).astype(np.float32)
    for step in range(6, 24):
        y = step
        x = 31 - step
        if step not in (6, 13, 23):  # Endpoints 6 & 23 rely on M_cont; 13 relies on M_gap
            r5[y, x] = 0.85
        aq5[y, x] = 0.70
        feat5[y, x] += anti_proto
    regimes.append(
        {
            "name": "05_antidiagonal_stroke_tail_continuation_32x32_D128",
            "r_map": r5,
            "feat_grid": feat5,
            "a_query_map": aq5,
            "max_bridge_radius": 2,
            "image_size": (512, 512),
        }
    )

    # 6. Border/corner-touching salient tokens (16 x 16, D = 64)
    rng6 = np.random.RandomState(106)
    r6 = np.zeros((16, 16), dtype=np.float32)
    r6[0, 0:5] = 0.95
    r6[15, 11:16] = 0.90
    r6[0:5, 15] = 0.85
    r6[11:16, 0] = 0.80
    aq6 = r6.copy()
    feat6 = rng6.normal(0.0, 0.1, size=(16, 16, 64)).astype(np.float32)
    border_proto = rng6.uniform(0.5, 1.2, size=(64,)).astype(np.float32)
    feat6[0, :] += border_proto
    feat6[15, :] += border_proto
    feat6[:, 0] += border_proto
    feat6[:, 15] += border_proto
    regimes.append(
        {
            "name": "06_border_corner_touching_tokens_16x16_D64",
            "r_map": r6,
            "feat_grid": feat6,
            "a_query_map": aq6,
            "max_bridge_radius": 2,
            "image_size": (320, 320),
        }
    )

    # 7. Multi-candidate disjoint regions (K = 3 separated text islands, 32 x 32, D = 128)
    rng7 = np.random.RandomState(107)
    r7 = np.zeros((32, 32), dtype=np.float32)
    feat7 = rng7.normal(0.0, 0.05, size=(32, 32, 128)).astype(np.float32)
    # Island 1 (top-left)
    r7[3:7, 3:9] = 0.92
    feat7[3:7, 3:9] += rng7.uniform(0.8, 1.2, size=(128,)).astype(np.float32)
    # Island 2 (middle-right)
    r7[13:17, 20:28] = 0.86
    feat7[13:17, 20:28] += rng7.uniform(0.8, 1.2, size=(128,)).astype(np.float32)
    # Island 3 (bottom-left)
    r7[24:29, 5:13] = 0.79
    feat7[24:29, 5:13] += rng7.uniform(0.8, 1.2, size=(128,)).astype(np.float32)
    aq7 = r7 * 0.95
    regimes.append(
        {
            "name": "07_multi_candidate_disjoint_regions_32x32_D128",
            "r_map": r7,
            "feat_grid": feat7,
            "a_query_map": aq7,
            "max_bridge_radius": 2,
            "image_size": (512, 512),
        }
    )

    # 8. Feature-free mode (feat_grid = None, 24 x 24)
    rng8 = np.random.RandomState(108)
    r8 = np.zeros((24, 24), dtype=np.float32)
    r8[8:14, 6:10] = rng8.uniform(0.6, 0.95, size=(6, 4)).astype(np.float32)
    r8[8:14, 12:17] = rng8.uniform(0.6, 0.95, size=(6, 5)).astype(np.float32)
    regimes.append(
        {
            "name": "08_feature_free_mode_24x24_None",
            "r_map": r8,
            "feat_grid": None,
            "a_query_map": None,
            "max_bridge_radius": 2,
            "image_size": (480, 480),
        }
    )

    # 9. Degenerate zero / near-zero maps (r_map = 0, a_query_map = 0, 16 x 16, D = 32)
    r9 = np.zeros((16, 16), dtype=np.float32)
    aq9 = np.zeros((16, 16), dtype=np.float32)
    feat9 = np.zeros((16, 16, 32), dtype=np.float32)
    regimes.append(
        {
            "name": "09_degenerate_zero_map_16x16_D32",
            "r_map": r9,
            "feat_grid": feat9,
            "a_query_map": aq9,
            "max_bridge_radius": 2,
            "image_size": (320, 320),
        }
    )

    # 10. Realistic Vintern patch-grid scale (32 x 48, D = 896 in bfloat16 promoted to float32)
    rng10 = np.random.RandomState(110)
    r10 = np.zeros((32, 48), dtype=np.float32)
    r10[10:15, 8:18] = rng10.uniform(0.65, 1.0, size=(5, 10)).astype(np.float32)
    r10[10:15, 20:32] = rng10.uniform(0.65, 1.0, size=(5, 12)).astype(np.float32)
    aq10 = np.copy(r10)
    aq10[10:15, 7:34] = np.maximum(aq10[10:15, 7:34], 0.50)
    feat10_f32 = rng10.normal(0.0, 0.08, size=(32, 48, 896)).astype(np.float32)
    vintern_proto = rng10.uniform(0.3, 0.9, size=(896,)).astype(np.float32)
    feat10_f32[10:15, 7:34] += vintern_proto
    feat10_bf16 = torch.from_numpy(feat10_f32).to(dtype=torch.bfloat16)
    regimes.append(
        {
            "name": "10_realistic_vintern_grid_32x48_D896_bfloat16",
            "r_map": r10,
            "feat_grid": feat10_bf16,
            "a_query_map": aq10,
            "max_bridge_radius": 2,
            "image_size": (960, 640),
        }
    )

    return regimes


def _make_synthetic_image_for_regime(r_map: np.ndarray, size: Tuple[int, int]) -> Image.Image:
    """Create a deterministic high-contrast synthetic RGB image aligned with `r_map`."""
    W_img, H_img = size
    H_g, W_g = r_map.shape
    canvas = np.full((H_img, W_img, 3), 235, dtype=np.uint8)
    for gy in range(H_g):
        for gx in range(W_g):
            if r_map[gy, gx] > 0.1:
                y0 = int(round(gy * H_img / H_g))
                y1 = int(round((gy + 1) * H_img / H_g))
                x0 = int(round(gx * W_img / W_g))
                x1 = int(round((gx + 1) * W_img / W_g))
                # High-frequency synthetic stroke pattern inside salient patches
                canvas[y0:y1, x0:x1] = 25
                canvas[y0:y1:2, x0:x1:2] = 245
    return Image.fromarray(canvas, mode="RGB")


class _NoCpuTransferTensor(torch.Tensor):
    """Tensor wrapper that raises AssertionError if `.cpu()` or `.numpy()` is called on 3D `feat_grid`."""

    @staticmethod
    def __new__(cls, data: torch.Tensor):
        return torch.Tensor._make_subclass(cls, data, data.requires_grad)

    def cpu(self, *args, **kwargs):
        if self.ndim == 3:
            raise AssertionError("Forbidden `.cpu()` transfer invoked on 3D `feat_grid`!")
        return super().cpu(*args, **kwargs).as_subclass(torch.Tensor)

    def numpy(self, *args, **kwargs):
        if self.ndim == 3:
            raise AssertionError("Forbidden `.numpy()` conversion invoked on 3D `feat_grid`!")
        return super().as_subclass(torch.Tensor).numpy(*args, **kwargs)


class TestOpt04GpuAffinityVectorization(unittest.TestCase):
    """Gate A test suite for TICKET-OPT-04: GPU Affinity Field Vectorization."""

    def setUp(self):
        self.regimes = _build_10_synthetic_regimes()
        self.selector = KLDivergenceRegionSelector()

    def test_a_direct_numerical_parity_across_10_synthetic_regimes(self):
        """
        Test A: Verify max_u |Phi_GPU(u) - Phi_CPU(u)| <= 1e-4 and
        max_u |M_aff,GPU(u) - M_aff,CPU(u)| <= 1e-4 across all 10 synthetic regimes.
        """
        self.assertEqual(len(self.regimes), 10)
        for regime in self.regimes:
            with self.subTest(regime=regime["name"]):
                phi_cpu, aff_cpu = _compute_intrinsic_affinity_field_numpy_reference(
                    r_map=regime["r_map"],
                    feat_grid=regime["feat_grid"],
                    a_query_map=regime["a_query_map"],
                    max_bridge_radius=regime["max_bridge_radius"],
                )
                phi_gpu_t, aff_gpu_t = compute_intrinsic_affinity_field_gpu(
                    r_map=regime["r_map"],
                    feat_grid=regime["feat_grid"],
                    a_query_map=regime["a_query_map"],
                    max_bridge_radius=regime["max_bridge_radius"],
                )
                phi_pub, aff_pub = compute_intrinsic_affinity_field(
                    r_map=regime["r_map"],
                    feat_grid=regime["feat_grid"],
                    a_query_map=regime["a_query_map"],
                    max_bridge_radius=regime["max_bridge_radius"],
                )

                phi_gpu = phi_gpu_t.detach().cpu().numpy()
                aff_gpu = aff_gpu_t.detach().cpu().numpy()

                max_diff_phi = float(np.max(np.abs(phi_gpu - phi_cpu)))
                max_diff_aff = float(np.max(np.abs(aff_gpu - aff_cpu)))

                self.assertLessEqual(
                    max_diff_phi,
                    1e-4,
                    f"phi_map parity failed on {regime['name']}: max_diff={max_diff_phi}",
                )
                self.assertLessEqual(
                    max_diff_aff,
                    1e-4,
                    f"aff_map parity failed on {regime['name']}: max_diff={max_diff_aff}",
                )
                np.testing.assert_allclose(phi_pub, phi_gpu, rtol=0.0, atol=1e-7)
                np.testing.assert_allclose(aff_pub, aff_gpu, rtol=0.0, atol=1e-7)

    def test_b_device_and_dtype_contract_verification(self):
        """
        Test B: Verify `compute_intrinsic_affinity_field_gpu` keeps `feat_grid` on its
        device, promotes `bfloat16`/`float16` to `float32`, and never calls `.cpu()` or
        `.numpy()` on `feat_grid`.
        """
        regime = self.regimes[9]  # 32x48x896 bfloat16 regime
        r_t = torch.from_numpy(regime["r_map"])
        aq_t = torch.from_numpy(regime["a_query_map"])

        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            raw_feat = regime["feat_grid"].to(dtype=dtype)
            guarded_feat = _NoCpuTransferTensor(raw_feat)

            phi_t, aff_t = compute_intrinsic_affinity_field_gpu(
                r_map=r_t,
                feat_grid=guarded_feat,
                a_query_map=aq_t,
                max_bridge_radius=2,
            )

            self.assertIsInstance(phi_t, torch.Tensor)
            self.assertIsInstance(aff_t, torch.Tensor)
            self.assertEqual(phi_t.device, raw_feat.device)
            self.assertEqual(aff_t.device, raw_feat.device)
            self.assertEqual(phi_t.dtype, torch.float32)
            self.assertEqual(aff_t.dtype, torch.float32)
            self.assertFalse(phi_t.requires_grad)
            self.assertFalse(aff_t.requires_grad)

    def test_c_boundary_and_no_wrap_around_verification(self):
        """
        Test C: Verify that high-intensity activations on opposing borders (top/bottom,
        left/right, and opposite corners) do NOT wrap around across image boundaries.
        """
        H, W, D = 12, 12, 32
        r_map = np.zeros((H, W), dtype=np.float32)
        # Opposing left/right borders at row 5
        r_map[5, 0] = 1.0
        r_map[5, W - 1] = 1.0
        # Opposing top/bottom borders at col 6
        r_map[0, 6] = 1.0
        r_map[H - 1, 6] = 1.0
        # Opposing corners
        r_map[0, 0] = 1.0
        r_map[H - 1, W - 1] = 1.0

        # Identical unit feature vectors everywhere so any wrap-around would produce score 1.0
        feat_grid = np.ones((H, W, D), dtype=np.float32)
        a_query_map = r_map.copy()

        phi_cpu, aff_cpu = _compute_intrinsic_affinity_field_numpy_reference(
            r_map=r_map,
            feat_grid=feat_grid,
            a_query_map=a_query_map,
            max_bridge_radius=2,
        )
        phi_gpu_t, aff_gpu_t = compute_intrinsic_affinity_field_gpu(
            r_map=r_map,
            feat_grid=feat_grid,
            a_query_map=a_query_map,
            max_bridge_radius=2,
        )
        phi_gpu = phi_gpu_t.cpu().numpy()
        aff_gpu = aff_gpu_t.cpu().numpy()

        # Border pixels must not bridge across opposite edges
        self.assertEqual(float(aff_gpu[5, 0]), 0.0)
        self.assertEqual(float(aff_gpu[5, W - 1]), 0.0)
        self.assertEqual(float(aff_gpu[0, 6]), 0.0)
        self.assertEqual(float(aff_gpu[H - 1, 6]), 0.0)
        self.assertEqual(float(aff_gpu[5, 1]), 0.0)
        self.assertEqual(float(aff_gpu[5, W - 2]), 0.0)

        np.testing.assert_allclose(phi_gpu, phi_cpu, rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(aff_gpu, aff_cpu, rtol=0.0, atol=1e-6)

    def test_d_pass0_vs_pass1_two_pass_semantics(self):
        """
        Test D: Verify Pass-0 geodesic stroke continuation (`M_cont`) followed by
        Pass-1 bilateral gap bridging (`M_gap`), and confirm `M_cont` is NOT re-run in Pass 1.
        """
        H, W, D = 10, 14, 16
        r_map = np.zeros((H, W), dtype=np.float32)
        a_query_map = np.zeros((H, W), dtype=np.float32)
        feat_grid = np.zeros((H, W, D), dtype=np.float32)
        feat_grid[:, :, 0] = 1.0  # Orthogonal background feature

        # Construct a chain along row 5:
        # Anchor at (5, 3) with r=1.0.
        # Tail candidate at (5, 4) has r=0.0, a_query=1.0 -> activated in Pass 0 via M_cont from (5, 3)!
        # Second anchor at (5, 6) has r=1.0.
        # Gap pixel at (5, 5) has r=0.0, a_query=0.0 -> bridged in Pass 1 via M_gap between
        # newly activated (5, 4) and anchor (5, 6)!
        # Further pixel at (5, 7) has r=0.0, a_query=1.0, and is adjacent only to a pixel that
        # would require Pass-1 M_cont if M_cont were erroneously run in Pass 1:
        # Specifically, place isolated anchor at (5, 9) and (5, 11) with gap at (5, 10) (filled in Pass 0),
        # and test if (5, 10) triggers M_cont in Pass 1 onto (5, 10)'s vertical neighbor (4, 10) where a_query=1.0.
        r_map[5, 3] = 1.0
        a_query_map[5, 3] = 1.0
        a_query_map[5, 4] = 1.0
        r_map[5, 6] = 1.0
        a_query_map[5, 6] = 1.0

        for x in (3, 4, 5, 6):
            feat_grid[5, x, :] = 0.0
            feat_grid[5, x, 1] = 1.0

        # Separate structure testing that Pass 1 does NOT run M_cont:
        # (2, 7) and (2, 11) have r=1.0 -> with max_bridge_radius=2, (2, 9) gets M_gap in Pass 0
        # (so current_r[2, 9] > 0 in Pass 1).
        # (3, 9) has r=0.0, a_query=1.0, and shares feature with (2, 9).
        # Note that (3, 9) is at (|dy|=1, |dx|=2) from (2, 7) and (2, 11), so it is NOT a k=1 neighbor
        # of (2, 7) or (2, 11) along any of the 4 lattice directions.
        # Because (2, 9) had r=0.0 in Pass 0, M_cont at (3, 9) from (2, 9) is 0 in Pass 0.
        # Since M_cont only runs in Pass 0, (3, 9) must remain 0 after Pass 1!
        r_map[2, 7] = 1.0
        r_map[2, 11] = 1.0
        a_query_map[2, 7] = 1.0
        a_query_map[2, 11] = 1.0
        a_query_map[3, 9] = 1.0
        for y, x in [(2, 7), (2, 9), (2, 11), (3, 9)]:
            feat_grid[y, x, :] = 0.0
            feat_grid[y, x, 2] = 1.0

        phi_cpu, aff_cpu = _compute_intrinsic_affinity_field_numpy_reference(
            r_map=r_map,
            feat_grid=feat_grid,
            a_query_map=a_query_map,
            max_bridge_radius=2,
        )
        phi_gpu_t, aff_gpu_t = compute_intrinsic_affinity_field_gpu(
            r_map=r_map,
            feat_grid=feat_grid,
            a_query_map=a_query_map,
            max_bridge_radius=2,
        )
        phi_gpu = phi_gpu_t.cpu().numpy()
        aff_gpu = aff_gpu_t.cpu().numpy()

        # Verify Pass 0 M_cont activated (5, 4) and Pass 1 M_gap bridged (5, 5)
        self.assertGreater(float(aff_gpu[5, 4]), 0.9)
        self.assertGreater(float(aff_gpu[5, 5]), 0.9)
        # Verify (2, 9) was bridged by Pass 0 M_gap, while (3, 9) stayed 0 (no Pass 1 M_cont)
        self.assertGreater(float(aff_gpu[2, 9]), 0.9)
        self.assertEqual(float(aff_gpu[3, 9]), 0.0)

        np.testing.assert_allclose(phi_gpu, phi_cpu, rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(aff_gpu, aff_cpu, rtol=0.0, atol=1e-6)

    def test_e_stage1_bounding_box_iou_parity_across_10_synthetic_regimes(self):
        """
        Test E: Pass `phi_map_gpu` and `phi_map_cpu` for all 10 synthetic regimes
        through `KLDivergenceRegionSelector.select_top_k(...)` and verify exact
        Stage 1 bounding-box parity (IoU = 1.0 and identical box coordinates).
        """
        for regime in self.regimes:
            with self.subTest(regime=regime["name"]):
                phi_cpu, _ = _compute_intrinsic_affinity_field_numpy_reference(
                    r_map=regime["r_map"],
                    feat_grid=regime["feat_grid"],
                    a_query_map=regime["a_query_map"],
                    max_bridge_radius=regime["max_bridge_radius"],
                )
                phi_gpu_t, _ = compute_intrinsic_affinity_field_gpu(
                    r_map=regime["r_map"],
                    feat_grid=regime["feat_grid"],
                    a_query_map=regime["a_query_map"],
                    max_bridge_radius=regime["max_bridge_radius"],
                )
                phi_gpu = phi_gpu_t.detach().cpu().numpy()

                syn_img = _make_synthetic_image_for_regime(
                    regime["r_map"], regime["image_size"]
                )

                sel_cpu = self.selector.select_top_k(
                    phi_map=phi_cpu,
                    image_size=syn_img.size,
                    image=syn_img,
                )
                sel_gpu = self.selector.select_top_k(
                    phi_map=phi_gpu,
                    image_size=syn_img.size,
                    image=syn_img,
                )

                self.assertEqual(
                    len(sel_gpu.pixel_bboxes),
                    len(sel_cpu.pixel_bboxes),
                    f"Candidate count mismatch on {regime['name']}",
                )
                for idx, (bbox_gpu, bbox_cpu, gbox_gpu, gbox_cpu, kl_gpu, kl_cpu) in enumerate(
                    zip(
                        sel_gpu.pixel_bboxes,
                        sel_cpu.pixel_bboxes,
                        sel_gpu.grid_bboxes,
                        sel_cpu.grid_bboxes,
                        sel_gpu.kl_scores,
                        sel_cpu.kl_scores,
                    )
                ):
                    iou, recall, precision = compute_bbox_iou_and_recall(
                        bbox_gpu, bbox_cpu
                    )
                    self.assertEqual(
                        bbox_gpu,
                        bbox_cpu,
                        f"Pixel BBox mismatch on {regime['name']} candidate {idx}: "
                        f"GPU={bbox_gpu} vs CPU={bbox_cpu}",
                    )
                    self.assertEqual(
                        gbox_gpu,
                        gbox_cpu,
                        f"Grid BBox mismatch on {regime['name']} candidate {idx}: "
                        f"GPU={gbox_gpu} vs CPU={gbox_cpu}",
                    )
                    self.assertAlmostEqual(
                        iou,
                        100.0,
                        places=5,
                        msg=f"IoU != 100% on {regime['name']} candidate {idx}",
                    )
                    self.assertAlmostEqual(
                        kl_gpu,
                        kl_cpu,
                        delta=1e-5,
                    )

    def test_f_public_api_contract_and_no_feat_grid_cpu_roundtrip(self):
        """
        Test F: Verify `compute_intrinsic_affinity_field` returns `(np.ndarray, np.ndarray)`
        of dtype `float32` while `compute_intrinsic_affinity_field_gpu` returns
        `(torch.Tensor, torch.Tensor)` on device without calling `.cpu()` or `.numpy()`
        on `feat_grid`.
        """
        regime = self.regimes[1]
        guarded_feat = _NoCpuTransferTensor(torch.from_numpy(regime["feat_grid"]))
        r_t = torch.from_numpy(regime["r_map"])
        aq_t = torch.from_numpy(regime["a_query_map"])

        phi_np, aff_np = compute_intrinsic_affinity_field(
            r_map=r_t,
            feat_grid=guarded_feat,
            a_query_map=aq_t,
            max_bridge_radius=2,
        )
        self.assertIsInstance(phi_np, np.ndarray)
        self.assertIsInstance(aff_np, np.ndarray)
        self.assertEqual(phi_np.dtype, np.float32)
        self.assertEqual(aff_np.dtype, np.float32)

        phi_cpu, aff_cpu = _compute_intrinsic_affinity_field_numpy_reference(
            r_map=regime["r_map"],
            feat_grid=regime["feat_grid"],
            a_query_map=regime["a_query_map"],
            max_bridge_radius=2,
        )
        np.testing.assert_allclose(phi_np, phi_cpu, rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(aff_np, aff_cpu, rtol=0.0, atol=1e-6)

    def test_g_edge_cases_zero_none_tiny_and_nonsquare_grids(self):
        """
        Test G: Verify edge cases including 1x1, 2x2, 3x3, 5x19, 19x5 grids,
        `feat_grid=None`, `a_query_map=None`, `max_bridge_radius=0` and `4`,
        negative input values, and zero-norm feature vectors.
        """
        rng = np.random.RandomState(707)
        shapes = [(1, 1), (2, 2), (3, 3), (5, 19), (19, 5)]
        for H, W in shapes:
            for radius in (0, 1, 2, 4):
                for use_feat in (True, False):
                    for use_aq in (True, False):
                        r_map = rng.uniform(-0.3, 1.0, size=(H, W)).astype(np.float32)
                        feat_grid = (
                            rng.normal(0.0, 1.0, size=(H, W, 16)).astype(np.float32)
                            if use_feat
                            else None
                        )
                        if feat_grid is not None and H >= 2 and W >= 2:
                            # Inject a zero-norm feature vector to verify 1e-8 clamp safety
                            feat_grid[0, 0, :] = 0.0
                        a_query_map = (
                            rng.uniform(-0.2, 0.8, size=(H, W)).astype(np.float32)
                            if use_aq
                            else None
                        )

                        phi_cpu, aff_cpu = _compute_intrinsic_affinity_field_numpy_reference(
                            r_map=r_map,
                            feat_grid=feat_grid,
                            a_query_map=a_query_map,
                            max_bridge_radius=radius,
                        )
                        phi_gpu_t, aff_gpu_t = compute_intrinsic_affinity_field_gpu(
                            r_map=r_map,
                            feat_grid=feat_grid,
                            a_query_map=a_query_map,
                            max_bridge_radius=radius,
                        )
                        phi_gpu = phi_gpu_t.cpu().numpy()
                        aff_gpu = aff_gpu_t.cpu().numpy()

                        self.assertFalse(np.isnan(phi_gpu).any())
                        self.assertFalse(np.isnan(aff_gpu).any())
                        np.testing.assert_allclose(phi_gpu, phi_cpu, rtol=0.0, atol=1e-5)
                        np.testing.assert_allclose(aff_gpu, aff_cpu, rtol=0.0, atol=1e-5)

    def test_h_extractor_mock_integration_and_cache_lifecycle(self):
        """
        Test H: Verify `PointwiseKLAttentionExtractor.extract()` end-to-end with a
        synthetic mock model/tokenizer (no real Vintern weights), confirming:
        1. Valid `ExtractionResult` with `np.float32` maps matching the CPU reference.
        2. `self._cached_img_state["grid_feats"]` is cached as a detached `torch.float32`
           tensor across same-image multi-question calls (`Q1`, `Q2`) without re-running ViT.
        3. `extractor.clear_cache()` releases `_cached_img_state` and `_cached_img_key`.
        """
        hidden_dim = 32
        num_heads = 2
        num_layers = 16
        img_ctx_id = 251
        eos_id = 252

        class _MockInnerLLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.prefix_calls = 0
                self.suffix_calls = 0

            def forward(
                self,
                inputs_embeds: torch.Tensor,
                past_key_values: Optional[Any] = None,
                use_cache: bool = True,
                output_attentions: bool = False,
                return_dict: bool = True,
            ):
                B, L, D = inputs_embeds.shape
                prev_len = past_key_values[0][0].shape[2] if past_key_values is not None else 0
                total_len = prev_len + L
                new_kv = [
                    (
                        torch.zeros((B, num_heads, total_len, 4), device=inputs_embeds.device),
                        torch.zeros((B, num_heads, total_len, 4), device=inputs_embeds.device),
                    )
                ]
                last_hidden = torch.zeros((B, L, D), device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                if not output_attentions:
                    if past_key_values is None:
                        self.prefix_calls += 1
                    return types.SimpleNamespace(
                        last_hidden_state=last_hidden,
                        past_key_values=new_kv,
                    )

                self.suffix_calls += 1
                gen = torch.Generator(device=inputs_embeds.device)
                seed_offset = int(torch.round(inputs_embeds.float().abs().sum() * 100).item()) % 1000
                gen.manual_seed(2026 + seed_offset)
                attentions = []
                for _ in range(num_layers):
                    logits = torch.randn(
                        (B, num_heads, L, total_len),
                        generator=gen,
                        device=inputs_embeds.device,
                        dtype=torch.float32,
                    )
                    if total_len > 120:
                        logits[:, :, :, 20:90] += 1.5 if (seed_offset % 2 == 0) else 0.2
                    attn = torch.softmax(logits, dim=-1)
                    attentions.append(attn)
                return types.SimpleNamespace(
                    last_hidden_state=last_hidden,
                    attentions=attentions,
                    past_key_values=new_kv,
                )

        class _MockVinternModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.extract_calls = 0
                self.emb = torch.nn.Embedding(260, hidden_dim)
                torch.manual_seed(42)
                torch.nn.init.normal_(self.emb.weight, mean=0.0, std=0.2)
                self.inner = _MockInnerLLM()

                def _lm_head(hidden: torch.Tensor) -> torch.Tensor:
                    B, L, _ = hidden.shape
                    out = torch.full((B, L, 260), -5.0, device=hidden.device, dtype=torch.float32)
                    # Immediately emit EOS to keep greedy trajectory fast and deterministic
                    out[:, :, eos_id] = 10.0
                    return out

                self.language_model = types.SimpleNamespace(
                    model=self.inner,
                    lm_head=_lm_head,
                    get_input_embeddings=lambda: self.emb,
                    config=types.SimpleNamespace(_attn_implementation="sdpa"),
                )
                self.config = types.SimpleNamespace(_attn_implementation="sdpa")
                self.img_context_token_id = img_ctx_id

            def extract_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
                self.extract_calls += 1
                num_tiles = pixel_values.shape[0]
                gen = torch.Generator(device=pixel_values.device)
                gen.manual_seed(777)
                feats = torch.randn(
                    (num_tiles, 256, hidden_dim),
                    generator=gen,
                    device=pixel_values.device,
                    dtype=torch.float32,
                )
                return feats.to(dtype=pixel_values.dtype)

            def get_input_embeddings(self):
                return self.emb

        class _MockTokenizer:
            eos_token_id = eos_id

            def convert_tokens_to_ids(self, token: str) -> int:
                if token == "<IMG_CONTEXT>":
                    return img_ctx_id
                if token == "<|im_end|>":
                    return eos_id
                return 10

            def __call__(
                self,
                text: str,
                return_tensors: str = "pt",
                add_special_tokens: bool = False,
            ):
                ids: List[int] = [2, 3, 4]
                n_img = text.count("<IMG_CONTEXT>")
                if n_img > 0:
                    ids.extend([img_ctx_id] * n_img)
                    ids.extend([5, 6])
                else:
                    ids.extend([min(240, max(10, ord(ch) % 240)) for ch in text[:16]] or [11])
                return types.SimpleNamespace(
                    input_ids=torch.tensor([ids], dtype=torch.long)
                )

        mock_model = _MockVinternModel()
        mock_tokenizer = _MockTokenizer()
        extractor = PointwiseKLAttentionExtractor(
            model=mock_model,
            tokenizer=mock_tokenizer,
            transform=lambda pil_img: torch.zeros((3, 448, 448), dtype=torch.float32),
            device="cpu",
            min_grid_tiles=1,
            max_grid_tiles=2,
        )

        img = Image.new("RGB", (448, 448), color=(180, 180, 180))
        res_q1 = extractor.extract(img, "Biển hiệu ghi nội dung gì?", image_id="img_001")

        self.assertEqual(mock_model.extract_calls, 1)
        self.assertIsNotNone(extractor._cached_img_state)
        cached_grid_feats = extractor._cached_img_state["grid_feats"]
        self.assertIsInstance(cached_grid_feats, torch.Tensor)
        self.assertEqual(cached_grid_feats.dtype, torch.float32)
        self.assertFalse(cached_grid_feats.requires_grad)

        self.assertEqual(res_q1.phi_map.dtype, np.float32)
        self.assertEqual(res_q1.r_map.dtype, np.float32)
        self.assertEqual(res_q1.aff_map.dtype, np.float32)

        # Verify parity of extractor output against CPU NumPy reference
        ref_phi_q1, ref_aff_q1 = _compute_intrinsic_affinity_field_numpy_reference(
            r_map=res_q1.r_map,
            feat_grid=cached_grid_feats,
            a_query_map=res_q1.a_query_map,
            max_bridge_radius=2,
        )
        np.testing.assert_allclose(res_q1.phi_map, ref_phi_q1, rtol=0.0, atol=1e-5)
        np.testing.assert_allclose(res_q1.aff_map, ref_aff_q1, rtol=0.0, atol=1e-5)

        # Q2 on the same image must reuse cached `grid_feats` without calling extract_feature again
        res_q2 = extractor.extract(img, "Số điện thoại trên bảng là gì?", image_id="img_001")
        self.assertEqual(mock_model.extract_calls, 1)
        self.assertIs(extractor._cached_img_state["grid_feats"], cached_grid_feats)

        ref_phi_q2, ref_aff_q2 = _compute_intrinsic_affinity_field_numpy_reference(
            r_map=res_q2.r_map,
            feat_grid=cached_grid_feats,
            a_query_map=res_q2.a_query_map,
            max_bridge_radius=2,
        )
        np.testing.assert_allclose(res_q2.phi_map, ref_phi_q2, rtol=0.0, atol=1e-5)
        np.testing.assert_allclose(res_q2.aff_map, ref_aff_q2, rtol=0.0, atol=1e-5)

        # Verify clear_cache() releases cached state completely
        extractor.clear_cache()
        self.assertIsNone(extractor._cached_img_state)
        self.assertIsNone(extractor._cached_img_key)


if __name__ == "__main__":
    unittest.main()
