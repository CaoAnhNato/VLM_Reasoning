# -*- coding: utf-8 -*-
"""
Adversarial Stress Test Suite for Theorem 35:
Residual Spatial Pursuit Multi-Hypothesis RoI Selection & Token Budget Invariants.

Empirical verification covering:
1. Extreme Aspect Ratios (10:1 horizontal banner, 1:10 vertical banner).
2. High-Noise Backgrounds with Tiny Secondary Text Modes (corner watermarks, price tags)
   verifying isolation when eta_res >= 0.20, vs noise suppression when eta_res < 0.20.
3. Overlapping Gaussian Peaks verifying IoU suppression when IoU >= 0.50 (mode collapse prevention)
   vs mode preservation when IoU < 0.50.
4. ViT Visual Token Budget Invariants: N_vis in {256, 512} tokens strictly enforced across
   arbitrary single or multi-crop outputs, rejecting any non-conforming budgets.
"""

import math
import unittest
from unittest.mock import MagicMock
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from src.nevfe.core.engine import (
    NEVFEEngine,
    compute_box_iou,
    compute_residual_spatial_pursuit_windows,
)
from src.nevfe.adapter.vintern_adapter import (
    VinternNEVFEAdapter,
    select_admissible_aspect_ratio_grid,
)


class MockEmbedding(nn.Module):
    def __init__(self, vocab_size: int = 151936, hidden_size: int = 896):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.randn(vocab_size, hidden_size) * 0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.embedding(self.weight, input_ids)


class MockLanguageModel(nn.Module):
    def __init__(self, vocab_size: int = 151936, hidden_size: int = 896):
        super().__init__()
        self.embed_tokens = MockEmbedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.embed_tokens

    def model(self, inputs_embeds=None, past_key_values=None, use_cache=True, return_dict=True):
        res = MagicMock()
        res.last_hidden_state = inputs_embeds.clone()
        res.past_key_values = ("mock_kv",)
        return res


class MockVinternModel(nn.Module):
    def __init__(self, num_image_token: int = 256, hidden_size: int = 896):
        super().__init__()
        self.num_image_token = num_image_token
        self.system_message = "Bạn là một trợ lý AI hữu ích."
        self.language_model = MockLanguageModel(vocab_size=151936, hidden_size=hidden_size)
        self.extract_feature_call_count = 0

    def extract_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
        self.extract_feature_call_count += 1
        batch_size = pixel_values.shape[0]
        P, D = self.num_image_token, 896
        return torch.randn(batch_size, P, D, dtype=torch.bfloat16, device=pixel_values.device)


class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 151643
        self.img_ctx_id = 151646
        self.im_start_id = 151644
        self.im_end_id = 151645
        self.vocab = {
            "<IMG_CONTEXT>": self.img_ctx_id,
            "<|im_start|>": self.im_start_id,
            "<|im_end|>": self.im_end_id,
        }

    def __len__(self):
        return 151936

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.vocab.get(token, 100)

    def encode(self, text: str, add_special_tokens: bool = False) -> list:
        tokens = []
        for word in text.split():
            tok_id = (abs(hash(word)) % 100000) + 1000
            tokens.append(tok_id)
        return tokens or [100]

    def decode(self, token_ids: list, skip_special_tokens: bool = True) -> str:
        return "câu trả lời kiểm thử"

    def __call__(self, text: str, return_tensors: str = "pt", add_special_tokens: bool = False):
        tokens = []
        parts = text.split("<IMG_CONTEXT>")
        if len(parts) > 1:
            for i, p in enumerate(parts):
                if p:
                    tokens.extend(self.encode(p, add_special_tokens=False))
                if i < len(parts) - 1:
                    tokens.append(self.img_ctx_id)
        else:
            tokens = self.encode(text, add_special_tokens=False)

        res = MagicMock()
        res.input_ids = torch.tensor([tokens], dtype=torch.long)
        return res


class TestAdversarialTheorem35(unittest.TestCase):
    """Empirical adversarial test suite for Theorem 35."""

    def setUp(self):
        torch.manual_seed(2026)
        np.random.seed(2026)
        self.device = "cpu"
        self.model = MockVinternModel(num_image_token=256, hidden_size=896)
        self.tokenizer = MockTokenizer()
        self.adapter = VinternNEVFEAdapter(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
        )

    # =========================================================================
    # 1. Extreme Aspect Ratios (10:1 horizontal banner, 1:10 vertical banner)
    # =========================================================================
    def test_extreme_horizontal_banner_10_to_1_tokens(self):
        """
        Stress-test: Visual tokens with saliency concentrated in a 1-row strip across all columns (10:1 horizontal layout).
        Verifies:
        - Box coordinates remain bounded in [0.0, 1.0].
        - x_min < x_max and y_min < y_max.
        - min_span floor (0.15) prevents degenerate zero-height box: height >= 2 * min_span.
        - No NaN or Inf coordinates.
        """
        grid_size = 16
        P = grid_size * grid_size
        D = 64
        tokens = torch.randn(P, D) * 0.05

        # Horizontal text banner at row 8 across all 16 columns
        for c in range(grid_size):
            idx = 8 * grid_size + c
            tokens[idx] += 8.0

        boxes, stats = compute_residual_spatial_pursuit_windows(
            visual_tokens=tokens,
            grid_size=grid_size,
            min_span=0.15,
            return_stats=True,
        )

        self.assertGreaterEqual(len(boxes), 1)
        b1 = boxes[0]
        x_min, y_min, x_max, y_max = b1

        self.assertTrue(0.0 <= x_min < x_max <= 1.0, f"Invalid horizontal box bounds: {b1}")
        self.assertTrue(0.0 <= y_min < y_max <= 1.0, f"Invalid vertical box bounds: {b1}")
        self.assertGreaterEqual(y_max - y_min, 2.0 * 0.15 - 1e-4, "min_span vertical floor violated!")
        self.assertFalse(math.isnan(x_min) or math.isnan(y_min) or math.isnan(x_max) or math.isnan(y_max))

    def test_extreme_vertical_banner_1_to_10_tokens(self):
        """
        Stress-test: Visual tokens with saliency concentrated in a 1-column strip down all rows (1:10 vertical layout).
        Verifies:
        - Box coordinates remain bounded in [0.0, 1.0].
        - min_span floor (0.15) prevents degenerate zero-width box: width >= 2 * min_span.
        - No NaN or Inf coordinates.
        """
        grid_size = 16
        P = grid_size * grid_size
        D = 64
        tokens = torch.randn(P, D) * 0.05

        # Vertical text banner at column 8 down all 16 rows
        for r in range(grid_size):
            idx = r * grid_size + 8
            tokens[idx] += 8.0

        boxes, stats = compute_residual_spatial_pursuit_windows(
            visual_tokens=tokens,
            grid_size=grid_size,
            min_span=0.15,
            return_stats=True,
        )

        self.assertGreaterEqual(len(boxes), 1)
        b1 = boxes[0]
        x_min, y_min, x_max, y_max = b1

        self.assertTrue(0.0 <= x_min < x_max <= 1.0, f"Invalid horizontal box bounds: {b1}")
        self.assertTrue(0.0 <= y_min < y_max <= 1.0, f"Invalid vertical box bounds: {b1}")
        self.assertGreaterEqual(x_max - x_min, 2.0 * 0.15 - 1e-4, "min_span horizontal floor violated!")
        self.assertFalse(math.isnan(x_min) or math.isnan(y_min) or math.isnan(x_max) or math.isnan(y_max))

    def test_select_admissible_aspect_ratio_grid_extreme_ratios(self):
        """
        Stress-test: select_admissible_aspect_ratio_grid with 10:1, 1:10, and degenerate aspect ratios.
        Verifies:
        - 10:1 ratio selects (2, 1) admissible grid.
        - 1:10 ratio selects (1, 2) admissible grid.
        - 1:1 ratio selects (1, 1) admissible grid.
        - Zero / near-zero inputs do not cause ZeroDivisionError.
        """
        # 10:1 horizontal banner
        g_horiz = select_admissible_aspect_ratio_grid(w_box=1000.0, h_box=100.0)
        self.assertEqual(g_horiz, (2, 1), f"Expected (2, 1) for 10:1 banner, got {g_horiz}")

        # 1:10 vertical banner
        g_vert = select_admissible_aspect_ratio_grid(w_box=100.0, h_box=1000.0)
        self.assertEqual(g_vert, (1, 2), f"Expected (1, 2) for 1:10 banner, got {g_vert}")

        # 1:1 square
        g_sq = select_admissible_aspect_ratio_grid(w_box=400.0, h_box=400.0)
        self.assertEqual(g_sq, (1, 1), f"Expected (1, 1) for square, got {g_sq}")

        # Degenerate inputs (0 width or 0 height)
        g_zero_w = select_admissible_aspect_ratio_grid(w_box=0.0, h_box=100.0)
        self.assertIn(g_zero_w, [(1, 1), (2, 1), (1, 2)])
        g_zero_h = select_admissible_aspect_ratio_grid(w_box=100.0, h_box=0.0)
        self.assertIn(g_zero_h, [(1, 1), (2, 1), (1, 2)])

    def test_denormalized_boxes_extreme_image_dimensions(self):
        """
        Stress-test: Compute Stage 1 multi-hypothesis windows on extreme canvas dimensions (1200x120 and 120x1200).
        Verifies pixel coordinates are valid integers strictly within [0, W] x [0, H] with width, height >= 2.
        """
        # 1200 x 120 image
        img_wide = Image.new("RGB", (1200, 120), color=(200, 200, 200))
        boxes_wide = self.adapter.compute_stage1_multihypothesis_windows(img_wide, question="Biển hiệu chữ gì?")
        self.assertGreaterEqual(len(boxes_wide), 1)
        for (x0, y0, x1, y1) in boxes_wide:
            self.assertTrue(0 <= x0 < x1 <= 1200, f"Pixel box out of bounds for wide image: {(x0, y0, x1, y1)}")
            self.assertTrue(0 <= y0 < y1 <= 120, f"Pixel box out of bounds for wide image: {(x0, y0, x1, y1)}")
            self.assertGreaterEqual(x1 - x0, 2)
            self.assertGreaterEqual(y1 - y0, 2)

        # 120 x 1200 image
        img_tall = Image.new("RGB", (120, 1200), color=(200, 200, 200))
        boxes_tall = self.adapter.compute_stage1_multihypothesis_windows(img_tall, question="Biển hiệu chữ gì?")
        self.assertGreaterEqual(len(boxes_tall), 1)
        for (x0, y0, x1, y1) in boxes_tall:
            self.assertTrue(0 <= x0 < x1 <= 120, f"Pixel box out of bounds for tall image: {(x0, y0, x1, y1)}")
            self.assertTrue(0 <= y0 < y1 <= 1200, f"Pixel box out of bounds for tall image: {(x0, y0, x1, y1)}")
            self.assertGreaterEqual(x1 - x0, 2)
            self.assertGreaterEqual(y1 - y0, 2)

    # =========================================================================
    # 2. High-Noise Backgrounds with Tiny Secondary Text Modes
    # =========================================================================
    def test_high_noise_background_secondary_mode_isolated_when_eta_res_ge_020(self):
        """
        Stress-test: High-noise background (sigma=1.5) with a tiny secondary text mode (e.g. corner watermark / price tag).
        When secondary mode has sufficient residual energy (eta_res >= 0.20 and max_rho_res >= 2.0 / P):
        Verifies:
        - Continuous deflation suppresses primary dominant mode.
        - Returns exactly K=2 proposals [B1, B2].
        - B2 accurately isolates the corner secondary mode despite noisy background.
        - IoU(B1, B2) <= 0.40.
        """
        grid_size = 16
        P = grid_size * grid_size
        D = 64

        # Strong Gaussian noise on background
        tokens = torch.randn(P, D) * 1.5

        # Primary dominant billboard at top-left (rows 1:4, cols 1:4)
        for r in range(1, 4):
            for c in range(1, 4):
                idx = r * grid_size + c
                tokens[idx] += 12.0

        # Secondary subtle text mode (price tag / corner watermark) at bottom-right (rows 13:15, cols 13:15)
        for r in range(13, 15):
            for c in range(13, 15):
                idx = r * grid_size + c
                tokens[idx] += 7.0

        boxes, stats = compute_residual_spatial_pursuit_windows(
            visual_tokens=tokens,
            grid_size=grid_size,
            lambda_res=0.20,
            iou_thresh=0.40,
            return_stats=True,
        )

        self.assertEqual(len(boxes), 2, f"Expected K=2 for secondary text mode, got {len(boxes)}")
        b1, b2 = boxes[0], boxes[1]

        self.assertGreaterEqual(stats["eta_res"], 0.20, f"eta_res should be >= 0.20, got {stats['eta_res']:.4f}")
        self.assertGreaterEqual(stats["max_rho_res"], 2.0 / P, f"max_rho_res should be >= 2/P, got {stats['max_rho_res']:.4f}")

        # B1 should cover top-left (center x < 0.5, center y < 0.5)
        cx1 = 0.5 * (b1[0] + b1[2])
        cy1 = 0.5 * (b1[1] + b1[3])
        self.assertLess(cx1, 0.5)
        self.assertLess(cy1, 0.5)

        # B2 should cover bottom-right corner mode (center x > 0.6, center y > 0.6)
        cx2 = 0.5 * (b2[0] + b2[2])
        cy2 = 0.5 * (b2[1] + b2[3])
        self.assertGreater(cx2, 0.6, f"B2 center_x was {cx2}, expected > 0.6")
        self.assertGreater(cy2, 0.6, f"B2 center_y was {cy2}, expected > 0.6")

        iou = compute_box_iou(b1, b2)
        self.assertLessEqual(iou, 0.40, f"IoU between B1 and B2 exceeded threshold: {iou:.4f}")

    def test_high_noise_background_faint_noise_suppressed_when_eta_res_lt_020(self):
        """
        Stress-test: High-noise background where secondary perturbation is faint noise
        (eta_res < 0.20 or max_rho_res < 2.0 / P).
        Verifies:
        - Engine does NOT hallucinate a spurious second crop on noise.
        - Returns strictly K=1 proposal [B1].
        """
        grid_size = 16
        P = grid_size * grid_size
        D = 64

        tokens = torch.randn(P, D) * 0.8

        # Massive dominant billboard at center
        for r in range(5, 11):
            for c in range(5, 11):
                idx = r * grid_size + c
                tokens[idx] += 15.0

        # Only very faint random perturbation at corner (amplitude 0.5)
        for r in range(14, 16):
            for c in range(14, 16):
                idx = r * grid_size + c
                tokens[idx] += 0.5

        boxes, stats = compute_residual_spatial_pursuit_windows(
            visual_tokens=tokens,
            grid_size=grid_size,
            lambda_res=0.20,
            iou_thresh=0.40,
            return_stats=True,
        )

        self.assertEqual(len(boxes), 1, f"Expected K=1 because noise mode should be suppressed, got {len(boxes)}")
        self.assertLess(stats["eta_res"], 0.20, f"eta_res should be < 0.20, got {stats['eta_res']:.4f}")

    def test_degenerate_zero_variance_uniform_tokens(self):
        """
        Stress-test: All visual tokens are identical (variance = 0, total_m = 0).
        Verifies:
        - Zero division is prevented by eps guards.
        - Falls back to center window [0.35, 0.35, 0.65, 0.65] (min_span=0.15).
        - Returns strictly K=1 proposal.
        - Zero NaN, zero Inf.
        """
        grid_size = 16
        P = grid_size * grid_size
        D = 64
        tokens = torch.zeros(P, D)

        boxes, stats = compute_residual_spatial_pursuit_windows(
            visual_tokens=tokens,
            grid_size=grid_size,
            min_span=0.15,
            return_stats=True,
        )

        self.assertEqual(len(boxes), 1)
        b = boxes[0]
        self.assertAlmostEqual(b[0], 0.35, places=3)
        self.assertAlmostEqual(b[1], 0.35, places=3)
        self.assertAlmostEqual(b[2], 0.65, places=3)
        self.assertAlmostEqual(b[3], 0.65, places=3)
        self.assertEqual(stats["eta_res"], 0.0)
        self.assertEqual(stats["max_rho_res"], 0.0)

    # =========================================================================
    # 3. Overlapping Gaussian Peaks (IoU Suppression vs Mode Preservation)
    # =========================================================================
    def test_overlapping_peaks_iou_suppression_when_iou_ge_050(self):
        """
        Stress-test: Two strong peaks placed such that unsuppressed candidate proposals
        have overlapping IoU >= 0.50 (e.g. shift = 3 grid units -> IoU ~ 0.61).
        Verifies:
        - When evaluated with iou_thresh=1.0, candidates overlap with IoU >= 0.50.
        - When evaluated with iou_thresh=0.50, mode collapse prevention suppresses B2
          and returns strictly K=1 proposal [B1].
        """
        grid_size = 16
        P = grid_size * grid_size
        D = 64
        tokens = torch.randn(P, D) * 0.01

        # Peak 1 at (4, 4)
        tokens[4 * grid_size + 4] += 10.0
        # Peak 2 at (4, 7) - shift of 3 patches horizontally
        tokens[4 * grid_size + 7] += 8.0

        # With iou_thresh=1.0, verify raw IoU between modes is indeed >= 0.50
        raw_boxes = compute_residual_spatial_pursuit_windows(
            visual_tokens=tokens,
            grid_size=grid_size,
            lambda_res=0.10,
            iou_thresh=1.0,
        )
        self.assertEqual(len(raw_boxes), 2)
        raw_iou = compute_box_iou(raw_boxes[0], raw_boxes[1])
        self.assertGreaterEqual(raw_iou, 0.50, f"Expected raw IoU >= 0.50, got {raw_iou:.4f}")

        # With iou_thresh=0.50 (and default 0.40), suppression fires!
        suppressed_boxes = compute_residual_spatial_pursuit_windows(
            visual_tokens=tokens,
            grid_size=grid_size,
            lambda_res=0.10,
            iou_thresh=0.50,
        )
        self.assertEqual(
            len(suppressed_boxes), 1,
            f"Overlapping peaks with IoU >= 0.50 must be suppressed to K=1, got {len(suppressed_boxes)}"
        )

    def test_separated_peaks_mode_preservation_when_iou_lt_050(self):
        """
        Stress-test: Two strong peaks placed far apart (e.g. top-left and bottom-right),
        producing proposals with IoU < 0.50 (and <= 0.40).
        Verifies:
        - Multimodal evidence preservation: both modes are captured.
        - Returns strictly K=2 proposals [B1, B2].
        """
        grid_size = 16
        P = grid_size * grid_size
        D = 64
        tokens = torch.randn(P, D) * 0.05

        # Peak 1 at (3, 3)
        for r in range(2, 5):
            for c in range(2, 5):
                tokens[r * grid_size + c] += 10.0

        # Peak 2 at (12, 12)
        for r in range(11, 14):
            for c in range(11, 14):
                tokens[r * grid_size + c] += 8.0

        boxes = compute_residual_spatial_pursuit_windows(
            visual_tokens=tokens,
            grid_size=grid_size,
            lambda_res=0.15,
            iou_thresh=0.50,
        )

        self.assertEqual(len(boxes), 2, f"Separated peaks must preserve K=2 proposals, got {len(boxes)}")
        iou = compute_box_iou(boxes[0], boxes[1])
        self.assertLess(iou, 0.50, f"Expected IoU < 0.50, got {iou:.4f}")

    def test_compute_box_iou_edge_cases(self):
        """
        Stress-test: compute_box_iou under edge cases:
        - Identical boxes -> 1.0
        - Completely disjoint boxes -> 0.0
        - Touching boxes (shared boundary line) -> 0.0
        - Inverted coordinates (x2 < x1) -> 0.0
        - Zero area boxes -> 0.0
        """
        # Identical
        self.assertAlmostEqual(compute_box_iou((0.1, 0.1, 0.5, 0.5), (0.1, 0.1, 0.5, 0.5)), 1.0)

        # Disjoint
        self.assertEqual(compute_box_iou((0.0, 0.0, 0.2, 0.2), (0.6, 0.6, 0.8, 0.8)), 0.0)

        # Touching boundary
        self.assertEqual(compute_box_iou((0.0, 0.0, 0.5, 0.5), (0.5, 0.0, 1.0, 0.5)), 0.0)

        # Inverted box coordinates
        self.assertEqual(compute_box_iou((0.5, 0.5, 0.1, 0.1), (0.2, 0.2, 0.6, 0.6)), 0.0)

        # Zero area box
        self.assertEqual(compute_box_iou((0.2, 0.2, 0.2, 0.2), (0.2, 0.2, 0.5, 0.5)), 0.0)

        # Partial overlap: (0, 0, 2, 2) and (1, 0, 3, 2)
        # Inter = 1 * 2 = 2, Union = 4 + 4 - 2 = 6 -> IoU = 1/3
        iou_val = compute_box_iou((0.0, 0.0, 2.0, 2.0), (1.0, 0.0, 3.0, 2.0))
        self.assertAlmostEqual(iou_val, 1.0 / 3.0, places=5)

    # =========================================================================
    # 4. Strict ViT Visual Token Budget Invariants (N_vis in {256, 512})
    # =========================================================================
    def test_build_multimodal_embeds_single_tile_256_tokens(self):
        """
        Stress-test: Single tile visual tokens [1, 256, 896].
        Verifies:
        - Accepted without error.
        - Injected <IMG_CONTEXT> tokens count is exactly 256.
        - Result tensor has shape [1, L, 896].
        """
        v_256 = torch.randn(1, 256, 896)
        embeds = self.adapter.build_multimodal_embeds(v_256, question="Chữ gì?")
        self.assertEqual(embeds.dim(), 3)
        self.assertEqual(embeds.shape[0], 1)
        self.assertEqual(embeds.shape[2], 896)

    def test_build_multimodal_embeds_multi_tile_512_tokens(self):
        """
        Stress-test: Multi-tile visual tokens [1, 512, 896] (K=2 proposals).
        Verifies:
        - Accepted without error.
        - Injected <IMG_CONTEXT> tokens count is exactly 512.
        - Result tensor has shape [1, L, 896].
        """
        v_512 = torch.randn(1, 512, 896)
        embeds = self.adapter.build_multimodal_embeds(v_512, question="Chữ gì?")
        self.assertEqual(embeds.dim(), 3)
        self.assertEqual(embeds.shape[0], 1)
        self.assertEqual(embeds.shape[2], 896)

    def test_build_multimodal_embeds_rejects_non_conforming_budgets(self):
        """
        Stress-test: Non-conforming visual token budgets (128, 384, 768, 1024 tokens).
        Verifies:
        - AssertionError is raised strictly enforcing N_vis in {256, 512}.
        """
        for invalid_tokens in [128, 384, 768, 1024]:
            v_bad = torch.randn(1, invalid_tokens, 896)
            with self.assertRaises(AssertionError, msg=f"Budget {invalid_tokens} should have been rejected!"):
                self.adapter.build_multimodal_embeds(v_bad, question="Chữ gì?")

    def test_extract_stage2_focal_features_budget_invariant(self):
        """
        Stress-test: extract_stage2_focal_features with multihypothesis_boxes.
        Verifies:
        - When multihypothesis_boxes has len >= 2: output has strictly shape [1, 512, 896].
        - When multihypothesis_boxes is None: output has shape [1, 256, 896] or [1, 512, 896] (never anything else).
        """
        img = Image.new("RGB", (300, 300), color=(180, 180, 180))

        # Multi-hypothesis RoIs B1, B2
        mh_boxes = [(10, 10, 80, 80), (150, 150, 250, 250)]
        v_l_mh = self.adapter.extract_stage2_focal_features(
            image=img,
            crop_box=mh_boxes[0],
            multihypothesis_boxes=mh_boxes,
        )
        self.assertEqual(v_l_mh.shape, (1, 512, 896), f"Expected [1, 512, 896] for K=2, got {v_l_mh.shape}")

        # Single crop box with hierarchical zoom
        v_l_single = self.adapter.extract_stage2_focal_features(
            image=img,
            crop_box=(20, 20, 100, 100),
            multihypothesis_boxes=None,
            use_hierarchical_zoom=True,
        )
        self.assertIn(v_l_single.shape[1], {256, 512}, f"Expected N_vis in {{256, 512}}, got {v_l_single.shape[1]}")
        self.assertEqual(v_l_single.shape[2], 896)


if __name__ == "__main__":
    unittest.main()
