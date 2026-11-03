# -*- coding: utf-8 -*-
"""
Adversarial Stress Test Suite for CERV Upgraded Pipeline (Milestone M2).

Focus Areas:
1. Degenerate & Singular Attention Maps (all-zeros, Dirac peak, flat uniform, collinear singular covariance).
2. Extreme Image Sizes & Aspect Ratios (1920x64, 64x1920, 32x32, 16x48, 4096x4096).
3. CBE-Gate Likelihood Ratio Lambda_t under Boundary Probability & Attention Conditions.
4. Robustness against NaN, Inf, and Singular Matrix collapse.
"""

import math
import unittest
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.benchmark.cerv.pipeline import CERVPipeline, dynamic_preprocess_with_grid


class MockConfig:
    vocab_size: int = 151674
    num_hidden_layers: int = 24
    max_dynamic_patch: int = 6
    _attn_implementation: str = "sdpa"


class MockLMModel(nn.Module):
    def __init__(self, vocab_size=151674, hidden_size=64):
        super().__init__()
        self.config = MockConfig()
        self.config.vocab_size = vocab_size
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([])

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, inputs_embeds=None, past_key_values=None, use_cache=True, output_attentions=False, return_dict=True):
        batch, seq_len, dim = inputs_embeds.shape
        last_hidden = inputs_embeds
        attentions = None
        if output_attentions:
            # 24 layers, 1 batch, 4 heads, seq_len, total_kv_len
            total_kv_len = seq_len
            if past_key_values is not None:
                # Add past length if simulated
                total_kv_len += 10
            att = torch.full((1, 4, seq_len, total_kv_len), 1.0 / total_kv_len, dtype=torch.float32)
            attentions = tuple(att for _ in range(24))
        
        class Output:
            def __init__(self, lh, pkv, atts):
                self.last_hidden_state = lh
                self.past_key_values = pkv or (torch.zeros(1, 4, 10, 16), torch.zeros(1, 4, 10, 16))
                self.attentions = atts
        return Output(last_hidden, past_key_values, attentions)


class MockLanguageModel(nn.Module):
    def __init__(self, vocab_size=151674, hidden_size=64):
        super().__init__()
        self.config = MockConfig()
        self.config.vocab_size = vocab_size
        self.model = MockLMModel(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()


class MockVinternModel(nn.Module):
    def __init__(self, vocab_size=151674, hidden_size=64):
        super().__init__()
        self.config = MockConfig()
        self.config.vocab_size = vocab_size
        self.language_model = MockLanguageModel(vocab_size, hidden_size)
        self.num_image_token = 256
        self.system_message = "Bạn là trợ lý AI hữu ích."
        self.img_context_token_id = 99999

    def extract_feature(self, pixel_values):
        # pixel_values: (num_tiles, 3, 448, 448)
        num_tiles = pixel_values.shape[0]
        # returns (num_tiles, 256, hidden_size)
        return torch.zeros(num_tiles, 256, 64, dtype=torch.bfloat16)


class MockTokenizer:
    def __init__(self, vocab_size=151674):
        self.vocab_size = vocab_size
        self.eos_token_id = 2
        self.img_context_token_id = 99999

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        # Generate dummy token IDs
        class Encoded:
            def __init__(self, ids):
                self.input_ids = ids
        
        count = max(4, len(text.split()))
        if "<IMG_CONTEXT>" in text:
            # Replicate IMG_CONTEXT count
            img_count = text.count("<IMG_CONTEXT>")
            prefix_ids = [10, 20]
            img_ids = [self.img_context_token_id] * img_count
            suffix_ids = [30, 40]
            ids = torch.tensor([prefix_ids + img_ids + suffix_ids], dtype=torch.long)
        else:
            ids = torch.randint(10, 1000, (1, count), dtype=torch.long)
        return Encoded(ids)

    def convert_tokens_to_ids(self, tok):
        if tok == "<IMG_CONTEXT>":
            return self.img_context_token_id
        if tok == "<|im_end|>":
            return 2
        return 50

    def decode(self, token_ids, skip_special_tokens=True):
        if not token_ids:
            return ""
        return "câu trả lời mẫu"


def mock_transform(img):
    return torch.zeros(3, 448, 448, dtype=torch.float32)


class TestCERVAdversarial(unittest.TestCase):
    """Adversarial stress-testing suite for CBE-Gate, HTC-RoI, and Covariance BBox."""

    def setUp(self):
        self.model = MockVinternModel()
        self.tokenizer = MockTokenizer()
        self.pipe = CERVPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            transform=mock_transform,
            device="cpu",
            energy_rho=0.85,
            plausibility_beta=0.1,
            max_dynamic_patch=6,
        )

    # =========================================================================
    # SUITE 1: Degenerate & Singular Attention Maps & Covariance Computation
    # =========================================================================

    def test_degenerate_all_zeros_attention_cbe_gate(self):
        """Verify CBE-Gate handles completely zeroed attention tensors without NaN, Inf, or crash."""
        vocab_size = self.pipe.vocab_size
        logits = torch.randn(1, vocab_size)
        
        # 24 layers, 1 batch, 4 heads, seq_len=1, total_kv_len=300
        # All attention values explicitly 0.0
        att_layer = torch.zeros(1, 4, 1, 300, dtype=torch.float32)
        attentions = tuple(att_layer for _ in range(24))

        vis_start, vis_end = 10, 266  # P = 256
        q_start, q_end = 267, 280
        assistant_end = 285

        lambda_t, pi_t, uncertainty, spatial_entropy, layer_vis_vecs = (
            self.pipe._compute_step_bayesian_odds(
                logits=logits,
                attentions=attentions,
                vis_start=vis_start,
                vis_end=vis_end,
                q_start=q_start,
                q_end=q_end,
                assistant_end=assistant_end,
            )
        )

        self.assertFalse(math.isnan(lambda_t), "lambda_t should not be NaN")
        self.assertFalse(math.isinf(lambda_t), "lambda_t should not be Inf")
        self.assertEqual(lambda_t, 0.0, "lambda_t must be exactly 0.0 when visual attention is zero")
        self.assertEqual(pi_t, 0.0, "pi_t must be 0.0 when visual attention is zero")
        self.assertFalse(math.isnan(uncertainty), "uncertainty should not be NaN")
        self.assertEqual(spatial_entropy, 0.0, "spatial_entropy should be 0.0 on zero mass")

    def test_degenerate_all_zeros_attention_stage2(self):
        """Verify Stage 2 localization safely handles all-zero contrast vector d_vec."""
        img = Image.new("RGB", (640, 480), color=(128, 128, 128))
        state = {
            "shared_kv": None,
            "null_suffix_embeds": torch.zeros(1, 10, 64),
            "vis_start": 0,
            "vis_end": 256,
            "grid_ratio": (1, 1),
            "has_thumbnail": True,
            "s_null_layers": {l: torch.zeros(256) for l in range(6, 19)},
        }
        # s_query_layers identical to null (so d_vec will be all zeros)
        s_query_layers = {l: torch.zeros(256) for l in range(6, 19)}

        img_pos, img_neg, bbox, peak_layer = self.pipe._stage2_prefix_conditioned_vaq_localization(
            image=img,
            state=state,
            locked_prefix_embeds=None,
            s_query_layers=s_query_layers,
            prefill_query_layers=s_query_layers,
        )

        self.assertEqual(bbox, [0, 0, 640, 480], "All-zero contrast must return full image bbox")
        self.assertEqual(img_pos.size, (640, 480))
        self.assertEqual(img_neg.size, (640, 480))

    def test_degenerate_dirac_delta_attention_cbe_gate(self):
        """Verify CBE-Gate with a single-patch Dirac delta peak (100% mass in 1 patch)."""
        vocab_size = self.pipe.vocab_size
        logits = torch.randn(1, vocab_size)

        vis_start, vis_end = 10, 266  # P = 256
        P = vis_end - vis_start
        att_layer = torch.zeros(1, 4, 1, 300, dtype=torch.float32)
        # Put all visual attention in patch 42
        att_layer[:, :, -1, vis_start + 42] = 1.0

        attentions = tuple(att_layer for _ in range(24))

        lambda_t, pi_t, uncertainty, spatial_entropy, layer_vis_vecs = (
            self.pipe._compute_step_bayesian_odds(
                logits=logits,
                attentions=attentions,
                vis_start=vis_start,
                vis_end=vis_end,
                q_start=267,
                q_end=280,
                assistant_end=285,
            )
        )

        self.assertFalse(math.isnan(lambda_t))
        self.assertFalse(math.isinf(lambda_t))
        self.assertEqual(pi_t, 1.0, "pi_t should be 1.0 when only visual mass exists")
        self.assertAlmostEqual(spatial_entropy, 0.0, places=5, msg="Dirac peak entropy must be ~0.0")

    def test_degenerate_dirac_delta_covariance_stage2(self):
        """Verify Dirac delta peak (zero intrinsic variance) does not collapse bbox to zero area."""
        img = Image.new("RGB", (640, 480), color=(100, 150, 200))
        state = {
            "shared_kv": None,
            "null_suffix_embeds": torch.zeros(1, 10, 64),
            "vis_start": 0,
            "vis_end": 256,
            "grid_ratio": (1, 1),
            "has_thumbnail": True,
            "s_null_layers": {l: torch.zeros(256) for l in range(6, 19)},
        }

        # Single patch peak at index (r=8, c=8) in 16x16 grid -> index = 8*16 + 8 = 136
        s_query_layers = {}
        for l in range(6, 19):
            vec = torch.zeros(256)
            vec[136] = 10.0
            s_query_layers[l] = vec

        img_pos, img_neg, bbox, peak_layer = self.pipe._stage2_prefix_conditioned_vaq_localization(
            image=img,
            state=state,
            locked_prefix_embeds=None,
            s_query_layers=s_query_layers,
            prefill_query_layers=s_query_layers,
        )

        x1, y1, x2, y2 = bbox
        self.assertGreater(x2, x1, f"x2 ({x2}) must be strictly greater than x1 ({x1})")
        self.assertGreater(y2, y1, f"y2 ({y2}) must be strictly greater than y1 ({y1})")
        self.assertGreaterEqual(x1, 0)
        self.assertGreaterEqual(y1, 0)
        self.assertLessEqual(x2, 640)
        self.assertLessEqual(y2, 480)
        # Patch 136 is at center (320, 240) in (640, 480). Check bbox covers it!
        self.assertTrue(x1 <= 320 <= x2, f"Center x=320 must be inside [{x1}, {x2}]")
        self.assertTrue(y1 <= 240 <= y2, f"Center y=240 must be inside [{y1}, {y2}]")

    def test_degenerate_uniform_flat_attention_cbe_gate(self):
        """Verify uniform flat attention across all patches produces maximum entropy (1.0)."""
        vocab_size = self.pipe.vocab_size
        logits = torch.randn(1, vocab_size)

        vis_start, vis_end = 10, 266  # P = 256
        P = vis_end - vis_start
        att_layer = torch.zeros(1, 4, 1, 300, dtype=torch.float32)
        att_layer[:, :, -1, vis_start:vis_end] = 1.0 / P  # Uniform

        attentions = tuple(att_layer for _ in range(24))

        lambda_t, pi_t, uncertainty, spatial_entropy, _ = self.pipe._compute_step_bayesian_odds(
            logits=logits,
            attentions=attentions,
            vis_start=vis_start,
            vis_end=vis_end,
            q_start=267,
            q_end=280,
            assistant_end=285,
        )

        self.assertAlmostEqual(spatial_entropy, 1.0, places=4, msg="Uniform distribution must have H_tilde = 1.0")

    def test_degenerate_uniform_flat_attention_stage2_pareto_abort(self):
        """Verify uniform flat attention across grid triggers Pareto abort and falls back to full image."""
        img = Image.new("RGB", (640, 480), color=(100, 100, 100))
        state = {
            "shared_kv": None,
            "null_suffix_embeds": torch.zeros(1, 10, 64),
            "vis_start": 0,
            "vis_end": 256,
            "grid_ratio": (1, 1),
            "has_thumbnail": True,
            "s_null_layers": {l: torch.zeros(256) for l in range(6, 19)},
        }

        # Perfectly uniform positive contrast across all 256 patches
        s_query_layers = {l: torch.ones(256) for l in range(6, 19)}

        img_pos, img_neg, bbox, peak_layer = self.pipe._stage2_prefix_conditioned_vaq_localization(
            image=img,
            state=state,
            locked_prefix_embeds=None,
            s_query_layers=s_query_layers,
            prefill_query_layers=s_query_layers,
        )

        self.assertEqual(peak_layer, -1, "Uniform attention must trigger Pareto abort (peak_layer == -1)")
        self.assertEqual(bbox, [0, 0, 640, 480], "Aborted localization must return [0, 0, W, H]")

    def test_degenerate_singular_covariance_collinear_horizontal(self):
        """Verify singular covariance matrix along vertical axis (sigma_y2 = 0) regularizes safely."""
        img = Image.new("RGB", (640, 480), color=(200, 200, 200))
        state = {
            "shared_kv": None,
            "null_suffix_embeds": torch.zeros(1, 10, 64),
            "vis_start": 0,
            "vis_end": 256,
            "grid_ratio": (1, 1),
            "has_thumbnail": True,
            "s_null_layers": {l: torch.zeros(256) for l in range(6, 19)},
        }

        # Single horizontal line of 4 patches: row 5, cols 6..9 (sigma_y2 = 0)
        s_query_layers = {}
        for l in range(6, 19):
            vec = torch.zeros(256)
            for c in range(6, 10):
                vec[5 * 16 + c] = 2.5
            s_query_layers[l] = vec

        img_pos, img_neg, bbox, peak_layer = self.pipe._stage2_prefix_conditioned_vaq_localization(
            image=img,
            state=state,
            locked_prefix_embeds=None,
            s_query_layers=s_query_layers,
            prefill_query_layers=s_query_layers,
        )

        x1, y1, x2, y2 = bbox
        self.assertGreater(x2, x1, "x2 must be > x1")
        self.assertGreater(y2, y1, "y2 must be > y1 (regularized, non-zero height)")
        self.assertGreaterEqual(x1, 0)
        self.assertGreaterEqual(y1, 0)
        self.assertLessEqual(x2, 640)
        self.assertLessEqual(y2, 480)

    def test_degenerate_singular_covariance_collinear_vertical(self):
        """Verify singular covariance matrix along horizontal axis (sigma_x2 = 0) regularizes safely."""
        img = Image.new("RGB", (640, 480), color=(200, 200, 200))
        state = {
            "shared_kv": None,
            "null_suffix_embeds": torch.zeros(1, 10, 64),
            "vis_start": 0,
            "vis_end": 256,
            "grid_ratio": (1, 1),
            "has_thumbnail": True,
            "s_null_layers": {l: torch.zeros(256) for l in range(6, 19)},
        }

        # Single vertical line of 4 patches: col 5, rows 6..9 (sigma_x2 = 0)
        s_query_layers = {}
        for l in range(6, 19):
            vec = torch.zeros(256)
            for r in range(6, 10):
                vec[r * 16 + 5] = 2.5
            s_query_layers[l] = vec

        img_pos, img_neg, bbox, peak_layer = self.pipe._stage2_prefix_conditioned_vaq_localization(
            image=img,
            state=state,
            locked_prefix_embeds=None,
            s_query_layers=s_query_layers,
            prefill_query_layers=s_query_layers,
        )

        x1, y1, x2, y2 = bbox
        self.assertGreater(x2, x1, "x2 must be > x1 (regularized, non-zero width)")
        self.assertGreater(y2, y1, "y2 must be > y1")
        self.assertGreaterEqual(x1, 0)
        self.assertGreaterEqual(y1, 0)
        self.assertLessEqual(x2, 640)
        self.assertLessEqual(y2, 480)

    # =========================================================================
    # SUITE 2: Extreme Image Sizes & Aspect Ratios
    # =========================================================================

    def _run_stage2_for_dims(self, width: int, height: int, grid_ratio: Tuple[int, int]):
        img = Image.new("RGB", (width, height), color=(50, 100, 150))
        gx, gy = grid_ratio
        total_tokens = gx * gy * 256
        state = {
            "shared_kv": None,
            "null_suffix_embeds": torch.zeros(1, 10, 64),
            "vis_start": 0,
            "vis_end": total_tokens,
            "grid_ratio": grid_ratio,
            "has_thumbnail": True,
            "s_null_layers": {l: torch.zeros(total_tokens) for l in range(6, 19)},
        }

        # Localized energy in a 2x2 patch cluster
        s_query_layers = {}
        for l in range(6, 19):
            vec = torch.zeros(total_tokens)
            vec[0] = 5.0
            vec[1] = 5.0
            s_query_layers[l] = vec

        img_pos, img_neg, bbox, peak_layer = self.pipe._stage2_prefix_conditioned_vaq_localization(
            image=img,
            state=state,
            locked_prefix_embeds=None,
            s_query_layers=s_query_layers,
            prefill_query_layers=s_query_layers,
        )

        x1, y1, x2, y2 = bbox
        # Strict boundary invariant assertions
        self.assertGreaterEqual(x1, 0, f"x1 ({x1}) must be >= 0")
        self.assertGreaterEqual(y1, 0, f"y1 ({y1}) must be >= 0")
        self.assertLessEqual(x2, width, f"x2 ({x2}) must be <= W ({width})")
        self.assertLessEqual(y2, height, f"y2 ({y2}) must be <= H ({height})")
        self.assertGreater(x2, x1, f"x2 ({x2}) must be > x1 ({x1})")
        self.assertGreater(y2, y1, f"y2 ({y2}) must be > y1 ({y1})")

        # Check crops
        self.assertEqual(img_pos.size, (x2 - x1, y2 - y1))
        self.assertEqual(img_neg.size, (x2 - x1, y2 - y1))
        return bbox

    def test_extreme_size_ultra_wide_1920x64(self):
        """Ultra-wide image (1920x64, aspect ratio 30:1)."""
        tiles, grid = dynamic_preprocess_with_grid(Image.new("RGB", (1920, 64)), min_num=1, max_num=6)
        self.assertEqual(grid, (6, 1), f"Aspect 30:1 should map to (6, 1), got {grid}")
        bbox = self._run_stage2_for_dims(1920, 64, grid)
        self.assertEqual(bbox[1], 0)
        self.assertEqual(bbox[3], 64)

    def test_extreme_size_ultra_tall_64x1920(self):
        """Ultra-tall image (64x1920, aspect ratio 1:30)."""
        tiles, grid = dynamic_preprocess_with_grid(Image.new("RGB", (64, 1920)), min_num=1, max_num=6)
        self.assertEqual(grid, (1, 6), f"Aspect 1:30 should map to (1, 6), got {grid}")
        bbox = self._run_stage2_for_dims(64, 1920, grid)
        self.assertEqual(bbox[0], 0)
        self.assertEqual(bbox[2], 64)

    def test_extreme_size_microscopic_32x32(self):
        """Microscopic tiny image (32x32)."""
        tiles, grid = dynamic_preprocess_with_grid(Image.new("RGB", (32, 32)), min_num=1, max_num=6)
        bbox = self._run_stage2_for_dims(32, 32, grid)
        self.assertEqual(bbox, [0, 0, 32, 32], "Microscopic image should span full [0, 0, 32, 32]")

    def test_extreme_size_microscopic_16x48_and_48x16(self):
        """Microscopic extreme aspect ratios (16x48 and 48x16)."""
        bbox1 = self._run_stage2_for_dims(16, 48, (1, 1))
        self.assertEqual(bbox1[0], 0)
        self.assertEqual(bbox1[2], 16)
        self.assertLessEqual(bbox1[3], 48)

        bbox2 = self._run_stage2_for_dims(48, 16, (1, 1))
        self.assertEqual(bbox2[1], 0)
        self.assertEqual(bbox2[3], 16)
        self.assertLessEqual(bbox2[2], 48)

    def test_extreme_size_large_4096x4096(self):
        """Large high-resolution image (4096x4096)."""
        bbox = self._run_stage2_for_dims(4096, 4096, (2, 2))
        self.assertGreaterEqual(bbox[0], 0)
        self.assertGreaterEqual(bbox[1], 0)
        self.assertLessEqual(bbox[2], 4096)
        self.assertLessEqual(bbox[3], 4096)
        self.assertGreater(bbox[2], bbox[0])
        self.assertGreater(bbox[3], bbox[1])

    # =========================================================================
    # SUITE 3: CBE-Gate Boundary Probability & Likelihood Ratio Conditions
    # =========================================================================

    def test_cbe_gate_top1_certainty_pt_1_0(self):
        """Verify CBE-Gate with top-1 probability p_t = 1.0 (zero uncertainty)."""
        vocab_size = self.pipe.vocab_size
        logits = torch.full((1, vocab_size), -100.0)
        logits[0, 42] = 100.0  # softmax(logits)[0, 42] == 1.0

        att_layer = torch.zeros(1, 4, 1, 300, dtype=torch.float32)
        # Non-zero visual attention
        att_layer[:, :, -1, 10:266] = 1.0 / 256.0
        attentions = tuple(att_layer for _ in range(24))

        lambda_t, pi_t, uncertainty, spatial_entropy, _ = self.pipe._compute_step_bayesian_odds(
            logits=logits,
            attentions=attentions,
            vis_start=10,
            vis_end=266,
            q_start=267,
            q_end=280,
            assistant_end=285,
        )

        self.assertEqual(uncertainty, 0.0, "Certain token must have U(p_t) = 0.0")
        self.assertEqual(lambda_t, 0.0, "Lambda_t must be 0.0 when uncertainty is 0.0")

    def test_cbe_gate_uniform_probability_pt_inv_V(self):
        """Verify CBE-Gate with maximum entropy uniform probability p_t = 1 / |V|."""
        vocab_size = self.pipe.vocab_size
        logits = torch.zeros(1, vocab_size)  # perfectly uniform

        att_layer = torch.zeros(1, 4, 1, 300, dtype=torch.float32)
        # Concentrated visual attention
        att_layer[:, :, -1, 10] = 1.0
        attentions = tuple(att_layer for _ in range(24))

        lambda_t, pi_t, uncertainty, spatial_entropy, _ = self.pipe._compute_step_bayesian_odds(
            logits=logits,
            attentions=attentions,
            vis_start=10,
            vis_end=266,
            q_start=267,
            q_end=280,
            assistant_end=285,
        )

        self.assertAlmostEqual(uncertainty, 1.0, places=3, msg="Uniform logits must have U(p_t) ~ 1.0")
        self.assertGreater(lambda_t, 1.0, "With concentrated vision & max uncertainty, Lambda_t should exceed 1.0")
        self.assertFalse(math.isnan(lambda_t))
        self.assertFalse(math.isinf(lambda_t))

    def test_cbe_gate_zero_visual_mass_vs_nonzero_question(self):
        """Zero visual attention mass vs non-zero question mass."""
        vocab_size = self.pipe.vocab_size
        logits = torch.randn(1, vocab_size)

        att_layer = torch.zeros(1, 4, 1, 300, dtype=torch.float32)
        # Visual tokens 10:266 are 0.0
        # Question tokens 267:280 are 1.0
        att_layer[:, :, -1, 267:280] = 1.0
        attentions = tuple(att_layer for _ in range(24))

        lambda_t, pi_t, uncertainty, spatial_entropy, _ = self.pipe._compute_step_bayesian_odds(
            logits=logits,
            attentions=attentions,
            vis_start=10,
            vis_end=266,
            q_start=267,
            q_end=280,
            assistant_end=285,
        )

        self.assertEqual(pi_t, 0.0, "Visual reliance must be 0.0")
        self.assertEqual(lambda_t, 0.0, "Lambda_t must be 0.0 when visual reliance is 0.0")

    def test_cbe_gate_zero_question_mass_vs_nonzero_visual(self):
        """Zero question attention mass vs non-zero visual mass."""
        vocab_size = self.pipe.vocab_size
        logits = torch.zeros(1, vocab_size)  # high uncertainty

        att_layer = torch.zeros(1, 4, 1, 300, dtype=torch.float32)
        # Visual tokens 10:20 are active
        att_layer[:, :, -1, 10:20] = 0.1
        # Question tokens are 0.0
        attentions = tuple(att_layer for _ in range(24))

        lambda_t, pi_t, uncertainty, spatial_entropy, _ = self.pipe._compute_step_bayesian_odds(
            logits=logits,
            attentions=attentions,
            vis_start=10,
            vis_end=266,
            q_start=267,
            q_end=280,
            assistant_end=285,
        )

        self.assertEqual(pi_t, 1.0, "Visual reliance must be 1.0 when question has zero mass")
        self.assertGreater(lambda_t, 0.0, "Lambda_t should be positive")

    def test_cbe_gate_all_zero_attention_masses(self):
        """All visual, question, and prefix attention masses are simultaneously 0.0."""
        vocab_size = self.pipe.vocab_size
        logits = torch.randn(1, vocab_size)

        att_layer = torch.zeros(1, 4, 1, 300, dtype=torch.float32)
        attentions = tuple(att_layer for _ in range(24))

        lambda_t, pi_t, uncertainty, spatial_entropy, _ = self.pipe._compute_step_bayesian_odds(
            logits=logits,
            attentions=attentions,
            vis_start=10,
            vis_end=266,
            q_start=267,
            q_end=280,
            assistant_end=285,
        )

        self.assertEqual(pi_t, 0.0)
        self.assertEqual(lambda_t, 0.0)
        self.assertFalse(math.isnan(lambda_t))

    def test_cbe_gate_extreme_logits_dynamic_range(self):
        """Softmax stability check with extreme logits (+1000.0 and -1000.0)."""
        vocab_size = self.pipe.vocab_size
        logits = torch.zeros(1, vocab_size)
        logits[0, 0] = 1000.0
        logits[0, 1] = -1000.0

        att_layer = torch.full((1, 4, 1, 300), 1.0 / 300.0, dtype=torch.float32)
        attentions = tuple(att_layer for _ in range(24))

        lambda_t, pi_t, uncertainty, spatial_entropy, _ = self.pipe._compute_step_bayesian_odds(
            logits=logits,
            attentions=attentions,
            vis_start=10,
            vis_end=266,
            q_start=267,
            q_end=280,
            assistant_end=285,
        )

        self.assertFalse(math.isnan(lambda_t))
        self.assertFalse(math.isinf(lambda_t))
        self.assertEqual(uncertainty, 0.0)

    # =========================================================================
    # SUITE 4: Multi-Modal Covariance, Image Modes, & End-to-End Invariants
    # =========================================================================

    def test_degenerate_bimodal_diagonal_corners_covariance(self):
        """Bimodal energy distribution at opposite diagonal corners (0,0) and (15,15)."""
        img = Image.new("RGB", (640, 480), color=(100, 100, 100))
        state = {
            "shared_kv": None,
            "null_suffix_embeds": torch.zeros(1, 10, 64),
            "vis_start": 0,
            "vis_end": 256,
            "grid_ratio": (1, 1),
            "has_thumbnail": True,
            "s_null_layers": {l: torch.zeros(256) for l in range(6, 19)},
        }

        # Energy at top-left (0) and bottom-right (255)
        s_query_layers = {}
        for l in range(6, 19):
            vec = torch.zeros(256)
            vec[0] = 5.0
            vec[255] = 5.0
            s_query_layers[l] = vec

        img_pos, img_neg, bbox, peak_layer = self.pipe._stage2_prefix_conditioned_vaq_localization(
            image=img,
            state=state,
            locked_prefix_embeds=None,
            s_query_layers=s_query_layers,
            prefill_query_layers=s_query_layers,
        )

        x1, y1, x2, y2 = bbox
        self.assertGreater(x2, x1)
        self.assertGreater(y2, y1)
        self.assertGreaterEqual(x1, 0)
        self.assertGreaterEqual(y1, 0)
        self.assertLessEqual(x2, 640)
        self.assertLessEqual(y2, 480)

    def test_extreme_image_modes_rgba_and_grayscale(self):
        """Image modes RGBA (4 channels) and Grayscale L (1 channel)."""
        state_tmpl = {
            "shared_kv": None,
            "null_suffix_embeds": torch.zeros(1, 10, 64),
            "vis_start": 0,
            "vis_end": 256,
            "grid_ratio": (1, 1),
            "has_thumbnail": True,
            "s_null_layers": {l: torch.zeros(256) for l in range(6, 19)},
        }
        s_query_layers = {l: torch.full((256,), 0.1) for l in range(6, 19)}
        s_query_layers[6][42] = 5.0

        # Mode RGBA
        img_rgba = Image.new("RGBA", (320, 240), color=(10, 20, 30, 255))
        _, _, bbox_rgba, _ = self.pipe._stage2_prefix_conditioned_vaq_localization(
            image=img_rgba,
            state=state_tmpl,
            locked_prefix_embeds=None,
            s_query_layers=s_query_layers,
            prefill_query_layers=s_query_layers,
        )
        self.assertGreater(bbox_rgba[2], bbox_rgba[0])
        self.assertGreater(bbox_rgba[3], bbox_rgba[1])

        # Mode L (Grayscale)
        img_l = Image.new("L", (320, 240), color=128)
        _, _, bbox_l, _ = self.pipe._stage2_prefix_conditioned_vaq_localization(
            image=img_l,
            state=state_tmpl,
            locked_prefix_embeds=None,
            s_query_layers=s_query_layers,
            prefill_query_layers=s_query_layers,
        )
        self.assertGreater(bbox_l[2], bbox_l[0])
        self.assertGreater(bbox_l[3], bbox_l[1])


if __name__ == "__main__":
    unittest.main()
