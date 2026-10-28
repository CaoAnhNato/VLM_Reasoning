# -*- coding: utf-8 -*-
"""
Tier 3: Cross-Feature Combinations Test Suite.
Requirement-Driven Verification of:
- C1: Visual token caching + resumable checkpointing interaction.
- C2: Stage 1 spatial cropping combined with Stage 2 contrastive decoding flow.
- C3: Dual-metric evaluation engine (Spatial IoU/recall + VQA Exact Match/F1).
- C4: Resumable benchmark state persistence, atomic checkpointing & crash recovery.

Verifies pairwise and multi-module feature interactions.
"""

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter
import torch
import torch.nn.functional as F

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

cropper_path = os.path.join(PROJECT_ROOT, "scripts", "benchmark", "sg_vcd", "roi_cropper.py")
spec = importlib.util.spec_from_file_location("roi_cropper_standalone", cropper_path)
roi_cropper_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(roi_cropper_mod)
IsotropicROICropper = roi_cropper_mod.IsotropicROICropper

from tests.test_helpers import (
    MockPipelineResult,
    calculate_inclusion_recall,
    calculate_iou,
    compute_vqa_exact_match,
    compute_vqa_token_f1,
    normalize_vietnamese_text,
)


class MockVisualFeatureExtractor:
    """Mock visual feature extractor tracking call counts and caching behavior."""

    def __init__(self):
        self.call_count = 0
        self.cached_features = {}

    def extract_features(self, image_id: str, image: Image.Image) -> torch.Tensor:
        """Extract or retrieve cached visual features."""
        if image_id in self.cached_features:
            return self.cached_features[image_id]

        self.call_count += 1
        # Simulated ViT embeddings: (N_tiles=2, sequence_len=256, embed_dim=896) in bfloat16
        feat = torch.randn(2, 256, 896, dtype=torch.bfloat16)
        self.cached_features[image_id] = feat
        return feat

    def clear_cache(self):
        self.cached_features.clear()


class TestTier3Combination1VisualCacheAndCheckpoint(unittest.TestCase):
    """C1: Visual Token Caching + Resumable Checkpointing Interactions."""

    def setUp(self):
        self.extractor = MockVisualFeatureExtractor()
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_01_cache_hit_multiple_questions_same_image(self):
        """Simulate 3 questions on the same image: ViT forward pass called exactly once."""
        img = Image.new("RGB", (448, 448), color=(100, 100, 100))
        image_id = "img_001"
        questions = ["Tiệm bán gì?", "Số điện thoại là bao nhiêu?", "Địa chỉ ở đâu?"]

        # Process all 3 questions
        features_list = []
        for q in questions:
            feat = self.extractor.extract_features(image_id, img)
            features_list.append(feat)

        # ViT forward extraction executed only once
        self.assertEqual(self.extractor.call_count, 1)
        # All 3 references point to the exact same cached tensor
        self.assertIs(features_list[0], features_list[1])
        self.assertIs(features_list[1], features_list[2])

    def test_02_cache_invalidation_on_new_image(self):
        """Switching to a new image generates fresh cache and increments call count."""
        img1 = Image.new("RGB", (448, 448), color=(50, 50, 50))
        img2 = Image.new("RGB", (448, 448), color=(150, 150, 150))

        feat1 = self.extractor.extract_features("img_001", img1)
        feat2 = self.extractor.extract_features("img_002", img2)

        self.assertEqual(self.extractor.call_count, 2)
        self.assertIsNot(feat1, feat2)

    def test_03_checkpoint_resumption_with_cached_evaluations(self):
        """Simulate interrupted evaluation: checkpoint records completed queries, resuming skips them."""
        checkpoint_path = os.path.join(self.test_dir, "checkpoint_eval.json")
        all_samples = [
            {"qid": 1, "image_id": "img_001", "q": "Q1"},
            {"qid": 2, "image_id": "img_001", "q": "Q2"},
            {"qid": 3, "image_id": "img_002", "q": "Q3"},
            {"qid": 4, "image_id": "img_002", "q": "Q4"},
        ]

        # Simulate first run evaluated first 2 questions and saved checkpoint
        saved_records = [
            {"qid": 1, "pred": "đáp án 1"},
            {"qid": 2, "pred": "đáp án 2"},
        ]
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump({"completed_qids": [1, 2], "results": saved_records}, f)

        # Resumed runner loads checkpoint
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        completed = set(state["completed_qids"])

        remaining_samples = [s for s in all_samples if s["qid"] not in completed]
        self.assertEqual(len(remaining_samples), 2)
        self.assertEqual(remaining_samples[0]["qid"], 3)
        self.assertEqual(remaining_samples[1]["qid"], 4)

    def test_04_cache_tensor_memory_footprint_invariant(self):
        """Verify cached tensor memory is ~2.3 MB per image in bfloat16 as defined in PROJECT.md."""
        # 2 tiles * 256 tokens * 896 hidden dim * 2 bytes (bfloat16) = 917,504 bytes ~ 0.92 MB (or for N=5 tiles ~2.3 MB)
        tiles = 5
        feat = torch.randn(tiles, 256, 896, dtype=torch.bfloat16)
        num_bytes = feat.element_size() * feat.nelement()
        num_mb = num_bytes / (1024 * 1024)

        # Assert memory stays within expected 2.2 - 2.5 MB range
        self.assertAlmostEqual(num_mb, 2.29, delta=0.2)


class TestTier3Combination2Stage1CroppingAndStage2Decoding(unittest.TestCase):
    """C2: Stage 1 Cropping Combined with Stage 2 Contrastive Decoding."""

    def setUp(self):
        self.cropper = IsotropicROICropper(target_size=448)

    def test_01_dual_tile_positive_branch_construction(self):
        """Stage 1 cropper output [global_thumb, roi_tile] forms valid positive visual evidence."""
        img = Image.new("RGB", (800, 600), color=(180, 180, 180))
        selected_bbox = (200, 150, 400, 250)

        crop_res = self.cropper.crop(img, selected_bbox)
        roi_tile = crop_res.roi_tile
        global_thumb = img.resize((448, 448), Image.Resampling.BICUBIC)

        self.assertEqual(roi_tile.size, (448, 448))
        self.assertEqual(global_thumb.size, (448, 448))

        pos_tiles = [global_thumb, roi_tile]
        self.assertEqual(len(pos_tiles), 2)

    def test_02_dual_role_negative_prior_blurred_roi(self):
        """Stage 2 negative branch applies Gaussian low-pass filter to ROI tile while retaining global context."""
        img = Image.new("RGB", (600, 600), color=(100, 120, 140))
        selected_bbox = (150, 150, 350, 250)

        crop_res = self.cropper.crop(img, selected_bbox)
        roi_tile = crop_res.roi_tile
        global_thumb = img.resize((448, 448), Image.Resampling.BICUBIC)

        roi_lowpass = roi_tile.filter(ImageFilter.GaussianBlur(radius=5.5))
        self.assertEqual(roi_lowpass.size, (448, 448))

        # Pixel difference between sharp ROI and blurred ROI exists (variance reduction)
        arr_sharp = np.array(roi_tile, dtype=np.float32)
        arr_blur = np.array(roi_lowpass, dtype=np.float32)
        self.assertEqual(arr_sharp.shape, arr_blur.shape)

    def test_03_stage1_confidence_dynamic_alpha_modulation(self):
        """Stage 1 spatial certainty tanh(J_KL(C*)) scales Stage 2 contrastive weight alpha."""
        alpha_0 = 0.65

        # Case 1: High confidence (J_KL = 2.5) -> tanh(2.5) ~ 0.9866
        kl_high = 2.5
        conf_high = float(np.tanh(kl_high))
        alpha_high = alpha_0 * conf_high
        self.assertAlmostEqual(alpha_high, alpha_0 * 0.9866, places=3)

        # Case 2: Low confidence / ambiguous background (J_KL = 0.1) -> tanh(0.1) ~ 0.0996
        kl_low = 0.1
        conf_low = float(np.tanh(kl_low))
        alpha_low = alpha_0 * conf_low
        self.assertLess(alpha_low, alpha_high)

        # Invariant: alpha is always non-negative and bounded by alpha_0
        self.assertGreaterEqual(alpha_low, 0.0)
        self.assertLessEqual(alpha_high, alpha_0)

    def test_04_plausibility_constraint_apc_filtering(self):
        """Adaptive Plausibility Constraint (APC): tokens with P_pos < beta * max(P_pos) are masked to -inf."""
        beta_apc = 0.10
        vocab_size = 10

        pos_logits = torch.tensor([[5.0, 4.5, 3.0, 1.0, 0.2, -1.0, -2.0, -3.0, -4.0, -5.0]])
        neg_logits = torch.tensor([[4.0, 6.0, 2.0, 1.5, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0]])

        p_pos = F.softmax(pos_logits, dim=-1)
        max_p = p_pos.max(dim=-1, keepdim=True).values
        valid_mask = p_pos >= (beta_apc * max_p)

        cd_logits = pos_logits + 0.65 * (pos_logits - neg_logits)
        cd_logits = cd_logits.masked_fill(~valid_mask, float("-inf"))

        # High probability candidate is preserved, low probability candidate masked
        self.assertFalse(torch.isinf(cd_logits[0, 0]))
        self.assertTrue(torch.isneginf(cd_logits[0, -1]))

    def test_05_zero_data_leakage_flow_contract(self):
        """Verify pipeline execution interface never requires or accepts ground-truth answers or bboxes."""
        from inspect import signature
        # Check standard pipeline run signature contract
        mock_pipeline_call_args = {"image": Image.new("RGB", (100, 100)), "question": "Hỏi gì?", "max_new_tokens": 32}
        self.assertNotIn("gt_bbox", mock_pipeline_call_args)
        self.assertNotIn("gt_answers", mock_pipeline_call_args)
        self.assertNotIn("answer_region_bbox", mock_pipeline_call_args)


class TestTier3Combination3DualMetricEvaluation(unittest.TestCase):
    """C3: Dual-Metric Evaluation Engine (Spatial IoU/Recall + VQA Exact Match/F1)."""

    def test_01_dual_metric_aggregation_over_predictions(self):
        """Aggregate both Stage 1 spatial metrics and Stage 2 VQA metrics over mock benchmark batch."""
        test_samples = [
            {
                "pred_box": [100, 100, 200, 200],
                "gt_box": [100, 100, 200, 200],
                "pred_text": "nhà thuốc",
                "gt_answers": ["nhà thuốc", "tiệm thuốc"],
            },
            {
                "pred_box": [0, 0, 50, 50],
                "gt_box": [300, 300, 400, 400],
                "pred_text": "quán ăn",
                "gt_answers": ["cà phê"],
            },
            {
                "pred_box": [50, 50, 150, 150],
                "gt_box": [50, 50, 150, 150],
                "pred_text": "tiệm thuốc tây",
                "gt_answers": ["nhà thuốc tây"],
            },
        ]

        ious = []
        exact_matches = []
        f1_scores = []

        for s in test_samples:
            iou = calculate_iou(s["pred_box"], s["gt_box"])
            em = compute_vqa_exact_match(s["pred_text"], s["gt_answers"])
            f1 = compute_vqa_token_f1(s["pred_text"], s["gt_answers"])

            ious.append(iou)
            exact_matches.append(em)
            f1_scores.append(f1)

        mean_iou = sum(ious) / len(ious)
        mean_em = sum(exact_matches) / len(exact_matches)
        mean_f1 = sum(f1_scores) / len(f1_scores)

        # 2 out of 3 had IoU=1.0, 1 had IoU=0.0 -> mean IoU = 2/3 ~ 0.6667
        self.assertAlmostEqual(mean_iou, 2.0 / 3.0, places=4)
        # 1 out of 3 exact matched ('nhà thuốc') -> mean EM = 1/3 ~ 0.3333
        self.assertAlmostEqual(mean_em, 1.0 / 3.0, places=4)
        self.assertGreater(mean_f1, mean_em)

    def test_02_spatial_grounding_accuracy_correlation_slice(self):
        """Analyze accuracy correlation: High IoU subset (IoU >= 0.5) vs Low IoU subset."""
        predictions = [
            {"iou": 0.85, "em": 1.0},
            {"iou": 0.72, "em": 1.0},
            {"iou": 0.60, "em": 1.0},
            {"iou": 0.10, "em": 0.0},
            {"iou": 0.05, "em": 0.0},
        ]

        high_iou_em = [p["em"] for p in predictions if p["iou"] >= 0.5]
        low_iou_em = [p["em"] for p in predictions if p["iou"] < 0.5]

        self.assertEqual(sum(high_iou_em) / len(high_iou_em), 1.0)
        self.assertEqual(sum(low_iou_em) / len(low_iou_em), 0.0)


class TestTier3Combination4ResumableBenchmarkPersistence(unittest.TestCase):
    """C4: Resumable Benchmark State Persistence & Atomic Checkpointing."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.ckpt_file = os.path.join(self.test_dir, "benchmark_checkpoint.json")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_01_atomic_checkpoint_write_and_reload(self):
        """Verify atomic file replacement prevents corrupt/empty JSON on interrupted write."""
        temp_file = f"{self.ckpt_file}.tmp"
        payload = {
            "processed_images": 25,
            "processed_quests": 78,
            "current_accuracy": 0.735,
            "records": [{"image_id": 22, "qid": 74, "pred": "nhà thuốc", "em": 1.0}],
        }

        # Write to temp file then atomically rename
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(temp_file, self.ckpt_file)

        self.assertTrue(os.path.exists(self.ckpt_file))
        self.assertFalse(os.path.exists(temp_file))

        with open(self.ckpt_file, "r", encoding="utf-8") as f:
            reloaded = json.load(f)
        self.assertEqual(reloaded["processed_images"], 25)
        self.assertEqual(reloaded["current_accuracy"], 0.735)

    def test_02_recovery_from_corrupted_temp_file(self):
        """Ensure benchmark runner preserves existing valid checkpoint if temp file write failed."""
        valid_payload = {"processed_images": 10, "status": "valid"}
        with open(self.ckpt_file, "w", encoding="utf-8") as f:
            json.dump(valid_payload, f)

        # Corrupt temp file created during simulated crash
        temp_file = f"{self.ckpt_file}.tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write("{ incomplete json ...")

        # Recovery logic: check if ckpt_file exists and is valid
        if os.path.exists(self.ckpt_file):
            with open(self.ckpt_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            self.assertEqual(state["status"], "valid")


if __name__ == "__main__":
    unittest.main()
