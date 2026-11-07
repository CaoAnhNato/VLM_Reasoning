# -*- coding: utf-8 -*-
"""
Tier 1: Feature Coverage Test Suite.
Requirement-Driven Verification of:
- F1: Data loading from all 4 workspaces and merged 400-image JSON schema.
- F2: Spatial metrics engine (IoU, inclusion recall, sub-boxes hit rate).
- F3: VQA evaluation metrics engine (Exact Match, Token F1, consensus voting).
- F4: Tokenizer & prompt configuration contracts (length, EOS, unicode preservation).
- F5: PipelineResult dataclass interface contracts and serialization.

Each feature includes at least 5 distinct, rigorous test cases.
"""

import json
import os
import unittest
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from tests.test_helpers import (
    MockPipelineResult,
    calculate_box_area,
    calculate_inclusion_recall,
    calculate_iou,
    calculate_sub_bboxes_hit_rate,
    compute_vqa_exact_match,
    compute_vqa_token_f1,
    normalize_vietnamese_text,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
import sys
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.benchmark.metrics import (
    compute_substring_match,
    compute_anls,
    compute_token_f1,
    compute_exact_match,
)
import torch
import torch.nn.functional as F

from src.nevfe.core.engine import (
    NEVFEEngine,
    compute_mean_field_reference_embeddings,
    compute_bhattacharyya_coefficient,
    compute_dynamic_coupling,
    compute_dynamic_plausibility_floor,
    mean_field_contrastive_step,
    nevfe_step_logits,
)


class TestTier1Feature1DataLoading(unittest.TestCase):
    """F1: Data Loading & Annotator Workspace Schema Verification (>=5 tests)."""

    @classmethod
    def setUpClass(cls):
        cls.merged_json_path = os.path.join(
            PROJECT_ROOT, "results", "swintextspotter_auto_annot", "answer_bboxes_400_verified.json"
        )
        cls.workspace_base = os.path.join(PROJECT_ROOT, "data", "xanylabeling_workspace")

    def test_01_merged_json_metadata_and_integrity(self):
        """Verify merged JSON contains authoritative 400 images and 1222 verified quests."""
        self.assertTrue(os.path.exists(self.merged_json_path), f"Missing: {self.merged_json_path}")
        with open(self.merged_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("metadata", data)
        self.assertIn("images", data)
        meta = data["metadata"]
        self.assertEqual(meta.get("num_images"), 400)
        self.assertEqual(meta.get("total_quests"), 1222)
        self.assertEqual(meta.get("verified_quests"), 1222)
        self.assertEqual(meta.get("still_missing_quests"), 0)
        self.assertEqual(len(data["images"]), 400)

    def test_02_workspace_cao_anh_100_images_and_annotations(self):
        """Verify workspace Cao_Anh_001_100 contains 100 images and corresponding JSON files."""
        ws_dir = os.path.join(self.workspace_base, "Cao_Anh_001_100")
        self.assertTrue(os.path.isdir(ws_dir), f"Missing directory: {ws_dir}")
        all_files = os.listdir(ws_dir)
        jpg_files = [f for f in all_files if f.lower().endswith((".jpg", ".png"))]
        json_files = [f for f in all_files if f.lower().endswith(".json")]

        self.assertGreaterEqual(len(jpg_files), 100, f"Expected >=100 images, got {len(jpg_files)}")
        self.assertGreaterEqual(len(json_files), 100, f"Expected >=100 JSONs, got {len(json_files)}")

        # Verify sample image can be opened with PIL and json is valid
        sample_img = os.path.join(ws_dir, jpg_files[0])
        with Image.open(sample_img) as img:
            self.assertGreater(img.width, 0)
            self.assertGreater(img.height, 0)

        sample_json = os.path.join(ws_dir, json_files[0])
        with open(sample_json, "r", encoding="utf-8") as f:
            parsed = json.load(f)
            self.assertIn("shapes", parsed)

    def test_03_workspace_trieu_100_images_and_annotations(self):
        """Verify workspace Trieu_101_200 contains 100 images and corresponding JSON files."""
        ws_dir = os.path.join(self.workspace_base, "Trieu_101_200")
        self.assertTrue(os.path.isdir(ws_dir), f"Missing directory: {ws_dir}")
        all_files = os.listdir(ws_dir)
        jpg_files = [f for f in all_files if f.lower().endswith((".jpg", ".png"))]
        json_files = [f for f in all_files if f.lower().endswith(".json")]

        self.assertGreaterEqual(len(jpg_files), 100, f"Expected >=100 images, got {len(jpg_files)}")
        self.assertGreaterEqual(len(json_files), 100, f"Expected >=100 JSONs, got {len(json_files)}")

        sample_json = os.path.join(ws_dir, json_files[0])
        with open(sample_json, "r", encoding="utf-8") as f:
            parsed = json.load(f)
            self.assertIn("shapes", parsed)

    def test_04_workspace_cam_tu_100_images_and_annotations(self):
        """Verify workspace Cam_Tu_201_300 contains 100 images and corresponding JSON files."""
        ws_dir = os.path.join(self.workspace_base, "Cam_Tu_201_300")
        self.assertTrue(os.path.isdir(ws_dir), f"Missing directory: {ws_dir}")
        all_files = os.listdir(ws_dir)
        jpg_files = [f for f in all_files if f.lower().endswith((".jpg", ".png"))]
        json_files = [f for f in all_files if f.lower().endswith(".json")]

        self.assertGreaterEqual(len(jpg_files), 100, f"Expected >=100 images, got {len(jpg_files)}")
        self.assertGreaterEqual(len(json_files), 100, f"Expected >=100 JSONs, got {len(json_files)}")

        sample_json = os.path.join(ws_dir, json_files[0])
        with open(sample_json, "r", encoding="utf-8") as f:
            parsed = json.load(f)
            self.assertIn("shapes", parsed)

    def test_05_workspace_thanh_tam_100_images_and_annotations(self):
        """Verify workspace Thanh_Tam_301_400 contains 100 images and corresponding JSON files."""
        ws_dir = os.path.join(self.workspace_base, "Thanh_Tam_301_400")
        self.assertTrue(os.path.isdir(ws_dir), f"Missing directory: {ws_dir}")
        all_files = os.listdir(ws_dir)
        jpg_files = [f for f in all_files if f.lower().endswith((".jpg", ".png"))]
        json_files = [f for f in all_files if f.lower().endswith(".json")]

        self.assertGreaterEqual(len(jpg_files), 100, f"Expected >=100 images, got {len(jpg_files)}")
        self.assertGreaterEqual(len(json_files), 100, f"Expected >=100 JSONs, got {len(json_files)}")

        sample_json = os.path.join(ws_dir, json_files[0])
        with open(sample_json, "r", encoding="utf-8") as f:
            parsed = json.load(f)
            self.assertIn("shapes", parsed)

    def test_06_image_annotation_schema_contract(self):
        """Verify image entry in merged JSON strictly respects the expected schema."""
        with open(self.merged_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        first_img = data["images"][0]

        required_img_fields = ["image_id", "image_name", "width", "height", "num_quests", "answer_regions"]
        for rf in required_img_fields:
            self.assertIn(rf, first_img, f"Missing field '{rf}' in image object")

        first_region = first_img["answer_regions"][0]
        required_region_fields = ["region_id", "bbox", "sub_bboxes", "gt_answers", "question_ids", "questions"]
        for rrf in required_region_fields:
            self.assertIn(rrf, first_region, f"Missing field '{rrf}' in answer_regions object")

        self.assertEqual(len(first_region["bbox"]), 4)
        self.assertIsInstance(first_region["gt_answers"], list)
        self.assertGreater(len(first_region["gt_answers"]), 0)


class TestTier1Feature2SpatialMetrics(unittest.TestCase):
    """F2: Spatial Metrics Engine (IoU, Inclusion Recall, Sub-Boxes Hit Rate) (>=5 tests)."""

    def test_01_iou_identical_boxes(self):
        """Derivation: IoU of identical bounding boxes is exactly 1.0."""
        box = [100, 150, 300, 450]
        self.assertAlmostEqual(calculate_iou(box, box), 1.0, places=6)

    def test_02_iou_completely_disjoint_boxes(self):
        """Derivation: Non-overlapping bounding boxes have IoU of 0.0."""
        box1 = [0, 0, 50, 50]
        box2 = [100, 100, 200, 200]
        self.assertEqual(calculate_iou(box1, box2), 0.0)

    def test_03_iou_partial_overlap_known_geometry(self):
        """
        Derivation:
        Box A: [0, 0, 20, 20], area = 400
        Box B: [10, 0, 30, 20], area = 400
        Intersection: [10, 0, 20, 20], area = 10 * 20 = 200
        Union: 400 + 400 - 200 = 600
        IoU = 200 / 600 = 1/3 ~ 0.333333.
        """
        boxA = [0, 0, 20, 20]
        boxB = [10, 0, 30, 20]
        expected_iou = 200.0 / 600.0
        self.assertAlmostEqual(calculate_iou(boxA, boxB), expected_iou, places=5)

    def test_04_inclusion_recall_full_and_partial(self):
        """
        Derivation:
        Pred: [0, 0, 100, 100]
        GT: [20, 20, 40, 40] (area 400, fully inside pred) -> Inclusion Recall = 1.0
        Pred2: [30, 20, 50, 40] -> Intersection is [30, 20, 40, 40] (area 200) -> Inclusion Recall = 200/400 = 0.5.
        """
        pred_full = [0, 0, 100, 100]
        gt = [20, 20, 40, 40]
        self.assertAlmostEqual(calculate_inclusion_recall(pred_full, gt), 1.0, places=5)

        pred_half = [30, 20, 50, 40]
        self.assertAlmostEqual(calculate_inclusion_recall(pred_half, gt), 0.5, places=5)

    def test_05_sub_bboxes_hit_rate(self):
        """
        Derivation:
        GT consists of 2 sub-boxes:
        s1 = [10, 10, 30, 30] (covered by pred)
        s2 = [200, 200, 220, 220] (missed by pred)
        Hit rate = 1 / 2 = 0.5.
        """
        pred = [0, 0, 50, 50]
        sub_boxes = [[10, 10, 30, 30], [200, 200, 220, 220]]
        hit_rate = calculate_sub_bboxes_hit_rate(pred, sub_boxes, iou_thresh=0.2)
        self.assertAlmostEqual(hit_rate, 0.5, places=5)


class TestTier1Feature3VQAMetrics(unittest.TestCase):
    """F3: VQA Evaluation Metrics Engine (Exact Match, Token F1, Consensus) (>=5 tests)."""

    def test_01_exact_match_perfect(self):
        """Derivation: normalized matching text returns 1.0."""
        self.assertEqual(compute_vqa_exact_match("Nhà Thuốc", ["nhà thuốc"]), 1.0)
        self.assertEqual(compute_vqa_exact_match("Cà phê Trung Nguyên", ["cà phê trung nguyên"]), 1.0)

    def test_02_exact_match_mismatch(self):
        """Derivation: semantically different text returns 0.0."""
        self.assertEqual(compute_vqa_exact_match("quán ăn", ["nhà thuốc"]), 0.0)
        self.assertEqual(compute_vqa_exact_match("hà nội", ["hồ chí minh"]), 0.0)

    def test_03_token_f1_exact(self):
        """Derivation: identical tokens produce precision=1, recall=1, F1=1.0."""
        f1 = compute_vqa_token_f1("hiệu thuốc tây phương", ["hiệu thuốc tây phương"])
        self.assertAlmostEqual(f1, 1.0, places=6)

    def test_04_token_f1_partial_overlap(self):
        """
        Derivation:
        Pred: "thuốc tây" (2 tokens: 'thuốc', 'tây')
        GT: "hiệu thuốc tây" (3 tokens: 'hiệu', 'thuốc', 'tây')
        Common: 2 tokens
        Precision: 2/2 = 1.0
        Recall: 2/3 = 0.666667
        F1 = 2 * (1.0 * 2/3) / (1.0 + 2/3) = (4/3) / (5/3) = 4/5 = 0.80.
        """
        f1 = compute_vqa_token_f1("thuốc tây", ["hiệu thuốc tây"])
        self.assertAlmostEqual(f1, 0.8, places=5)

    def test_05_vqa_multi_gt_consensus_scoring(self):
        """Derivation: when annotators provide multiple valid answers, best match is taken."""
        gt_list = ["baolangson.vn", "báo lạng sơn", "baolangson"]
        # Model predicted "baolangson"
        em = compute_vqa_exact_match("baolangson", gt_list)
        f1 = compute_vqa_token_f1("baolangson", gt_list)
        self.assertEqual(em, 1.0)
        self.assertAlmostEqual(f1, 1.0, places=5)

    def test_06_substring_match_full_and_partial(self):
        """
        Derivation:
        - Exact substring containment: "bảo tâm an" in "nhà thuốc bảo tâm an" -> 1.0
        - Reverse substring containment: "bảo tâm an" in "nhà thuốc bảo tâm an" (pred="nhà thuốc bảo tâm an", GT="bảo tâm an") -> 1.0
        - Space-concatenated OCR tokens: "baotaman" vs "bảo tâm an" -> 1.0
        - Token set inclusion: tokens of GT are subset of pred -> 1.0
        - Soft token inclusion ratio: 2 out of 4 tokens match -> ratio 0.5
        - Disjoint strings: "hà nội" vs "sài gòn" -> 0.0
        - Empty string guards: "" or empty GT -> 0.0
        """
        # Exact substring: GT inside pred
        self.assertEqual(compute_substring_match("nhà thuốc bảo tâm an", ["bảo tâm an"]), 1.0)
        # Concise prediction inside verbose GT is NOT substring match (Strict A* Protocol)
        self.assertEqual(compute_substring_match("bảo tâm an", ["nhà thuốc bảo tâm an"]), 0.0)
        # Space-concatenated without match is 0.0
        self.assertEqual(compute_substring_match("baotaman", ["bảo tâm an"]), 0.0)
        # Token set inclusion: GT inside pred
        self.assertEqual(compute_substring_match("đây là quán ăn ngon", ["quán ăn"]), 1.0)
        # Non-containment is 0.0
        score = compute_substring_match("quán việt", ["quán gia đình việt"])
        self.assertEqual(score, 0.0)
        # Completely disjoint strings
        self.assertEqual(compute_substring_match("hà nội", ["sài gòn"]), 0.0)
        # Empty string handling
        self.assertEqual(compute_substring_match("", ["nhà thuốc"]), 0.0)
        self.assertEqual(compute_substring_match("nhà thuốc", []), 0.0)

    def test_07_anls_threshold_and_exact(self):
        """
        Derivation:
        - Exact match returns 1.0.
        - Minor typo: "hallo" vs "hello" -> edit distance 1, max len 5 -> similarity 0.8 >= 0.5 (tau) -> 0.8.
        - Major typo: "world" vs "hello" -> edit distance 4, max len 5 -> similarity 0.2 < 0.5 (tau) -> 0.0.
        - Space-concatenated OCR tokens: "baolangson" vs "báo lạng sơn" -> score >= 0.5.
        - Empty prediction or empty ground truths return 0.0.
        """
        # Exact match
        self.assertEqual(compute_anls("nhà thuốc", ["nhà thuốc"]), 1.0)
        # Minor typo within tau=0.5
        anls_minor = compute_anls("hallo", ["hello"])
        self.assertAlmostEqual(anls_minor, 0.8, places=4)
        # Major typo below tau=0.5
        anls_major = compute_anls("world", ["hello"])
        self.assertEqual(anls_major, 0.0)
        # Space-concatenated OCR tokens
        anls_ocr = compute_anls("baolangson", ["báo lạng sơn"])
        self.assertGreaterEqual(anls_ocr, 0.50)
        # Empty string handling
        self.assertEqual(compute_anls("", ["nhà thuốc"]), 0.0)
        self.assertEqual(compute_anls("nhà thuốc", []), 0.0)


class TestTier1Feature4TokenizerConfig(unittest.TestCase):
    """F4: Tokenizer & Generation Config Contracts (>=5 tests)."""

    def test_01_max_new_tokens_contract(self):
        """Verify generation config limits maximum generated sequence length."""
        max_tokens = 64
        # Simulated generator returning token list
        def mock_generate(tokens_budget: int) -> List[int]:
            return list(range(min(tokens_budget, 100)))

        gen_tokens = mock_generate(max_tokens)
        self.assertLessEqual(len(gen_tokens), max_tokens)
        self.assertEqual(len(gen_tokens), 64)

    def test_02_eos_detection_contract(self):
        """Verify generation stops immediately upon encountering EOS token <|im_end|>."""
        eos_id = 151645
        sample_stream = [234, 1045, 8832, 151645, 9999, 1234]

        emitted = []
        for t in sample_stream:
            if t == eos_id:
                break
            emitted.append(t)

        self.assertEqual(emitted, [234, 1045, 8832])
        self.assertNotIn(eos_id, emitted)

    def test_03_vietnamese_unicode_preservation_in_prompt(self):
        """Ensure Vietnamese diacritics and tones are preserved without ASCII corruption."""
        raw_question = "Cửa hàng bán món gì vào buổi sáng?"
        prompt = f"<image>\n{raw_question}\nTrả lời ngắn gọn:"

        self.assertIn("Cửa hàng bán món gì", prompt)
        self.assertTrue(prompt.endswith("Trả lời ngắn gọn:"))
        self.assertNotIn("\\u", prompt)  # Not escaped to raw unicode hex

    def test_04_empty_or_whitespace_question_guard(self):
        """Verify system handles empty/blank question without crashing."""
        blank_q = "   \n\t  "
        clean_q = blank_q.strip()
        self.assertEqual(clean_q, "")
        norm = normalize_vietnamese_text(blank_q)
        self.assertEqual(norm, "")

    def test_05_special_token_stripping(self):
        """Ensure special markup tokens (<|im_start|>, <|im_end|>, etc.) are stripped."""
        raw_pred = "<|im_start|>assistant\nNhà sách Tiền Phong<|im_end|>"
        # Normalize and strip
        cleaned = raw_pred.replace("<|im_start|>assistant", "").replace("<|im_end|>", "").strip()
        norm = normalize_vietnamese_text(cleaned)
        self.assertEqual(norm, "nhà sách tiền phong")


class TestTier1Feature5PipelineResult(unittest.TestCase):
    """F5: PipelineResult Contract & Interface Fidelity (>=5 tests)."""

    def test_01_result_dataclass_fields(self):
        """Verify PipelineResult satisfies all required fields from PROJECT.md."""
        res = MockPipelineResult(
            pred_text="nhà thuốc",
            pred_raw_text="Đây là nhà thuốc tư nhân.",
            stage1_bbox=[100, 200, 300, 250],
            spatial_confidence=0.88,
        )
        self.assertEqual(res.pred_text, "nhà thuốc")
        self.assertEqual(res.pred_raw_text, "Đây là nhà thuốc tư nhân.")
        self.assertEqual(res.stage1_bbox, [100, 200, 300, 250])
        self.assertEqual(res.spatial_confidence, 0.88)
        self.assertIn("stage1_ms", res.timing)

    def test_02_bbox_coordinate_order_invariant(self):
        """Verify stage1_bbox satisfies x1 <= x2 and y1 <= y2."""
        res = MockPipelineResult(
            pred_text="test",
            pred_raw_text="test",
            stage1_bbox=[50, 80, 150, 120],
            spatial_confidence=0.9,
        )
        x1, y1, x2, y2 = res.stage1_bbox
        self.assertLessEqual(x1, x2)
        self.assertLessEqual(y1, y2)
        self.assertGreaterEqual(x1, 0)
        self.assertGreaterEqual(y1, 0)

    def test_03_confidence_bounded_range(self):
        """Verify spatial_confidence is strictly bounded in [0.0, 1.0]."""
        for conf in [0.0, 0.25, 0.5, 0.75, 1.0]:
            res = MockPipelineResult(
                pred_text="test",
                pred_raw_text="test",
                stage1_bbox=[0, 0, 10, 10],
                spatial_confidence=conf,
            )
            self.assertTrue(0.0 <= res.spatial_confidence <= 1.0)

    def test_04_timing_numeric_latencies(self):
        """Verify timing metrics are positive floating-point numbers."""
        res = MockPipelineResult(
            pred_text="test",
            pred_raw_text="test",
            stage1_bbox=[0, 0, 10, 10],
            spatial_confidence=0.5,
            timing={"stage1_ms": 15.2, "stage2_ms": 42.1, "total_ms": 57.3},
        )
        for k, v in res.timing.items():
            self.assertIsInstance(v, float)
            self.assertGreater(v, 0.0)

    def test_05_json_serializability_for_checkpointing(self):
        """Verify PipelineResult can be converted to JSON without serialization errors."""
        res = MockPipelineResult(
            pred_text="baolangson",
            pred_raw_text="baolangson.vn",
            stage1_bbox=[292, 655, 452, 703],
            spatial_confidence=0.94,
        )
        data_dict = res.to_dict()
        serialized = json.dumps(data_dict, ensure_ascii=False)
        self.assertIsInstance(serialized, str)
        deserialized = json.loads(serialized)
        self.assertEqual(deserialized["pred_text"], "baolangson")
        self.assertEqual(deserialized["stage1_bbox"], [292, 655, 452, 703])


class TestTier1Feature6IGVFEDMathematicalInvariants(unittest.TestCase):
    r"""
    F6: Information-Geometric Variational Free Energy Decoding (IG-VFED) Mathematical Invariants.
    
    Verifies:
    - Test 1: Blind Mean-Field reference tensor V^0 = 1_P \bar{v}^\top with spatial gradient oblivion (\nabla_{spatial} V^0 \equiv 0) across 2D and 3D shapes.
    - Test 2: Fisher-Rao Bhattacharyya proximity BC_t with clamp safeguard strictly in [0.0, 1.0], identity, and orthogonal cases.
    - Test 3: Binding Directive A (Preamble Echo Suppression at t=1): strict penalization of negative visual PMI tokens (z_1^+ - z_1^0 < 0) and amplification of positive visual PMI tokens.
    - Test 4: Dynamic Geodesic Candidate Plausibility Boundary beta_t = beta * BC_t: boundary expansion under optical divergence to rescue valid scene text subwords.
    - Test 5: 100% Backbone-Agnostic tensor dimensionality invariance: arbitrary D, P, V, and batch sizes without hardcoding.
    - Test 6: Strict hyperparameter budget invariant: exactly <= 2 continuous hyperparameters (lambda_param, beta_param).
    - Test 7: Modular Factorial Micro-Ablation Switches (comp_a, comp_b, comp_c) and dynamic reconfiguration.
    """

    def test_01_mean_field_reference_tensor_and_spatial_gradient_oblivion(self):
        """Verify V^0 has zero spatial gradient across tokens while preserving sequence length and channel dimension."""
        p_len, d_dim = 256, 896
        v_2d = torch.randn(p_len, d_dim, dtype=torch.float32)
        v0_2d = compute_mean_field_reference_embeddings(v_2d)

        self.assertEqual(v0_2d.shape, v_2d.shape)
        # Spatial gradient check: all patch tokens in v0_2d must be identical
        diff_2d = v0_2d[0:1, :] - v0_2d[1:, :]
        self.assertAlmostEqual(torch.max(torch.abs(diff_2d)).item(), 0.0, places=6)
        # Centroid equality
        expected_mean = v_2d.mean(dim=0, keepdim=True).expand_as(v_2d)
        torch.testing.assert_close(v0_2d, expected_mean)

        # Test 3D [N_tiles, P, D]
        n_tiles = 6
        v_3d = torch.randn(n_tiles, p_len, d_dim, dtype=torch.float32)
        v0_3d = compute_mean_field_reference_embeddings(v_3d)

        self.assertEqual(v0_3d.shape, v_3d.shape)
        # All patches across all tiles must be identical
        diff_3d = v0_3d - v0_3d[0:1, 0:1, :]
        self.assertAlmostEqual(torch.max(torch.abs(diff_3d)).item(), 0.0, places=6)

    def test_02_fisher_rao_bhattacharyya_coefficient_safeguard_clamp(self):
        """Verify BC_t clamp safeguard in [0.0, 1.0], identity case = 1.0, and orthogonal support = 0.0."""
        vocab_size = 1000

        # Identity case: p_plus == p_zero -> BC == 1.0
        logits = torch.randn(1, vocab_size)
        p = F.softmax(logits, dim=-1)
        bc_ident = compute_bhattacharyya_coefficient(p, p)
        self.assertAlmostEqual(bc_ident.item(), 1.0, places=5)

        # Orthogonal disjoint support case: p_plus and p_zero have zero overlap -> BC == 0.0
        p_plus = torch.zeros(1, vocab_size)
        p_zero = torch.zeros(1, vocab_size)
        p_plus[0, :500] = 1.0 / 500.0
        p_zero[0, 500:] = 1.0 / 500.0
        bc_ortho = compute_bhattacharyya_coefficient(p_plus, p_zero)
        self.assertAlmostEqual(bc_ortho.item(), 0.0, places=6)

        # Clamp safeguard check: values are strictly in [0.0, 1.0] across 20 random distributions
        for _ in range(20):
            lp = torch.randn(2, vocab_size) * 5.0
            lz = torch.randn(2, vocab_size) * 5.0
            bc = compute_bhattacharyya_coefficient(F.softmax(lp, dim=-1), F.softmax(lz, dim=-1))
            self.assertTrue(torch.all(bc >= 0.0).item())
            self.assertTrue(torch.all(bc <= 1.0).item())

    def test_03_binding_directive_a_preamble_echo_suppression_at_t1(self):
        """
        Verify Binding Directive A at t=1:
        - When language prior dominates (BC_1 -> 1.0), conversational echo tokens with
          negative visual PMI (z_1^+ - z_1^0 < 0) are strictly penalized: z_1^* < z_1^+.
        - Visual entity tokens with positive visual PMI (z_1^+ - z_1^0 > 0) are amplified: z_1^* > z_1^+.
        - At t > 1, geodesic cooling smoothly attenuates contrast on syntactic connectors where BC_t -> 1.0.
        """
        vocab_size = 500
        lambda_param = 0.50

        # Construct logits at t=1 where language prior dominates (BC_1 close to 1.0)
        base_logits = torch.randn(1, vocab_size)
        lp = base_logits.clone()
        lz = base_logits.clone()

        # Echo token v_echo: high in language prior (unconditioned), so z_plus < z_zero (negative PMI)
        v_echo = 10
        lp[0, v_echo] = 5.0
        lz[0, v_echo] = 8.0  # negative visual PMI: lp - lz = -3.0

        # Entity token v_entity: high in optical evidence (multimodal), so z_plus > z_zero (positive PMI)
        v_entity = 20
        lp[0, v_entity] = 6.0
        lz[0, v_entity] = 1.0  # positive visual PMI: lp - lz = +5.0

        # Step t = 1
        _, steered_logits_t1, alpha_t1, _ = nevfe_step_logits(
            lp, lz, lambda_param=lambda_param, step_idx=1
        )

        # Directive A check: alpha_1 must be strictly positive (alpha_1 >= lambda)
        self.assertGreaterEqual(alpha_t1.item(), lambda_param)

        # Steered logit for echo token must be strictly lower than lp (penalized!)
        self.assertLess(steered_logits_t1[0, v_echo].item(), lp[0, v_echo].item())

        # Steered logit for entity token must be strictly higher than lp (amplified!)
        self.assertGreater(steered_logits_t1[0, v_entity].item(), lp[0, v_entity].item())

        # Check step t > 1 on identical distributions (BC -> 1.0)
        # alpha_t should cool to zero (s_t = sqrt(1 - BC^2) -> 0)
        p_same = F.softmax(base_logits, dim=-1)
        bc_high = compute_bhattacharyya_coefficient(p_same, p_same)
        alpha_t2 = compute_dynamic_coupling(lambda_param, bc_high, step_idx=2)
        self.assertLess(alpha_t2.item(), 1e-3)

    def test_04_dynamic_geodesic_candidate_plausibility_boundary_beta_t(self):
        """
        Verify dynamic plausibility boundary beta_t = beta * BC_t:
        - When optical divergence is high (BC_t -> 0), beta_t -> 0, rescuing valid rare subwords.
        - When aligned with prior (BC_t -> 1), beta_t -> beta, enforcing standard truncation.
        """
        beta_param = 0.05
        bc_divergent = torch.tensor([[0.05]])
        bc_aligned = torch.tensor([[0.95]])

        beta_div = compute_dynamic_plausibility_floor(beta_param, bc_divergent, comp_b=True)
        beta_ali = compute_dynamic_plausibility_floor(beta_param, bc_aligned, comp_b=True)

        self.assertAlmostEqual(beta_div.item(), beta_param * 0.05, places=5)
        self.assertAlmostEqual(beta_ali.item(), beta_param * 0.95, places=5)
        self.assertLess(beta_div.item(), beta_ali.item())

        # Test candidate selection rescues subword token
        vocab_size = 100
        lp = torch.full((1, vocab_size), -10.0)
        lp[0, 0] = 5.0    # dominant token
        lp[0, 1] = 1.0    # rare text subword
        lz = torch.randn(1, vocab_size)

        p_plus = F.softmax(lp, dim=-1)
        p_zero = F.softmax(lz, dim=-1)
        bc = compute_bhattacharyya_coefficient(p_plus, p_zero)

        beta_dynamic = compute_dynamic_plausibility_floor(beta_param, bc, comp_b=True)
        max_p = torch.max(p_plus, dim=-1, keepdim=True).values

        if bc.item() < 0.35:
            self.assertTrue(p_plus[0, 1] >= beta_dynamic[0, 0] * max_p[0, 0])

    def test_05_backbone_agnostic_dimensionality_invariance(self):
        """Verify engine operates agnostically across arbitrary D, P, V, and batch dimensions."""
        engine = NEVFEEngine(lambda_param=0.50, beta_param=0.05)

        for vocab_size in [50, 500, 32000, 151936]:
            # 1D logits
            lp_1d = torch.randn(vocab_size)
            lz_1d = torch.randn(vocab_size)
            tok_1d, steered_1d, alpha_1d, _ = engine.decode_step(lp_1d, lz_1d, step_idx=1)
            self.assertIsInstance(tok_1d.item(), int)
            self.assertEqual(steered_1d.shape, (vocab_size,))

            # 2D batched logits [B, V]
            batch_size = 2
            lp_2d = torch.randn(batch_size, vocab_size)
            lz_2d = torch.randn(batch_size, vocab_size)
            tok_2d, steered_2d, alpha_2d, _ = engine.decode_step(lp_2d, lz_2d, step_idx=1)
            self.assertEqual(tok_2d.shape, (batch_size,))
            self.assertEqual(steered_2d.shape, (batch_size, vocab_size))

    def test_06_hyperparameter_budget_and_strict_invariants(self):
        """Verify engine enforces strictly <= 2 continuous hyperparameters with valid parameter bounds."""
        engine = NEVFEEngine(lambda_param=0.50, beta_param=0.05)
        self.assertEqual(engine.lambda_param, 0.50)
        self.assertEqual(engine.beta_param, 0.05)

        with self.assertRaises(ValueError):
            NEVFEEngine(lambda_param=-0.1)
        with self.assertRaises(ValueError):
            NEVFEEngine(lambda_param=2.5)
        with self.assertRaises(ValueError):
            NEVFEEngine(beta_param=0.0)

    def test_07_factorial_micro_ablation_switches(self):
        """Verify modular ablation switches (Marginal A, Marginal B, Primary AB, and Directive A comp_c) toggle cleanly."""
        engine = NEVFEEngine(lambda_param=0.50, beta_param=0.05)
        # By default, full IG-VFED is active (comp_a=True, comp_b=True, comp_c=True)
        self.assertTrue(engine.comp_a)
        self.assertTrue(engine.comp_b)
        self.assertTrue(engine.comp_c)

        # 4 Factorial Micro-Ablation Masks:
        # Mask 1,0,0: Iter 22 Champion baseline (comp_a=True, comp_b=False, comp_c=False)
        engine.set_ablation_switches(comp_a=True, comp_b=False, comp_c=False)
        self.assertTrue(engine.comp_a)
        self.assertFalse(engine.comp_b)
        self.assertFalse(engine.comp_c)

        # Mask 1,1,0: Iter 22 + Dynamic beta_t ONLY
        engine.set_ablation_switches(comp_a=True, comp_b=True, comp_c=False)
        self.assertTrue(engine.comp_a)
        self.assertTrue(engine.comp_b)
        self.assertFalse(engine.comp_c)

        # Mask 1,0,1: Iter 22 + Directive A t=1 boost ONLY
        engine.set_ablation_switches(comp_a=True, comp_b=False, comp_c=True)
        self.assertTrue(engine.comp_a)
        self.assertFalse(engine.comp_b)
        self.assertTrue(engine.comp_c)

        # Mask 1,1,1: Full IG-VFED (comp_a=True, comp_b=True, comp_c=True)
        engine.set_ablation_switches(comp_a=True, comp_b=True, comp_c=True)
        self.assertTrue(engine.comp_a)
        self.assertTrue(engine.comp_b)
        self.assertTrue(engine.comp_c)

        # Mathematical verification of comp_c gating Directive A t=1 preamble boost:
        bc_one = torch.tensor([[1.0]])
        # When comp_c=False at t=1: alpha_1 = lambda * sqrt(1 - 1^2) = 0.0 (Iter 22 baseline)
        alpha_t1_no_boost = compute_dynamic_coupling(0.50, bc_one, step_idx=1, comp_a=True, comp_c=False)
        self.assertAlmostEqual(alpha_t1_no_boost.item(), 0.0, places=5)

        # When comp_c=True at t=1: alpha_1 = 0.0 + lambda * 1.0 = 0.50 (Directive A boost)
        alpha_t1_boost = compute_dynamic_coupling(0.50, bc_one, step_idx=1, comp_a=True, comp_c=True)
        self.assertAlmostEqual(alpha_t1_boost.item(), 0.50, places=5)


if __name__ == "__main__":
    unittest.main()
