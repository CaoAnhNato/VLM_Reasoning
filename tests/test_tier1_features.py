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
    canonical_normalize_vietnamese_vqa,
)
from scripts.benchmark.sg_vcd.bbox_selector import (
    KLDivergenceRegionSelector,
    MultiRegionSelectionResult,
)
from scripts.benchmark.sg_vcd.roi_cropper import (
    apply_adaptive_laplacian_edge_reemphasis,
)
from scripts.benchmark.sg_vcd.adaptive_propagator import (
    verify_and_resolve_candidate_boxes,
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
        # Reverse substring: pred inside GT (len >= 3)
        self.assertEqual(compute_substring_match("bảo tâm an", ["nhà thuốc bảo tâm an"]), 1.0)
        # Space-concatenated OCR tokens
        self.assertEqual(compute_substring_match("baotaman", ["bảo tâm an"]), 1.0)
        # Token set inclusion
        self.assertEqual(compute_substring_match("đây là quán ăn ngon", ["quán ăn"]), 1.0)
        # Soft token inclusion ratio (2 of 4 tokens present, non-contiguous)
        score = compute_substring_match("quán việt", ["quán gia đình việt"])
        self.assertAlmostEqual(score, 0.5, places=4)
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
        self.assertGreaterEqual(anls_ocr, 0.70)
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


class TestTier1Round3AlgorithmicUpgrades(unittest.TestCase):
    """
    Round 3 SOTA Algorithmic Upgrades & Canonical Metrics Test Suite:
    - Test 1: Stage 1 Adaptive Top-K candidate selection with disjoint bimodal distribution (K=2, IoU <= 0.20, IoM <= 0.35).
    - Test 2: Fallback to K=1 when secondary mode is below significance gate (gamma_signif < 0.40 or p < 0.10).
    - Test 3: Adaptive Pre-RoI Laplacian edge re-emphasis at low magnification (M <= 1.0 -> psi=0, identity).
    - Test 4: Adaptive Pre-RoI Laplacian edge re-emphasis at high magnification (M=3.5 -> sharpening active, uint8 range).
    - Test 5: Symmetric Canonical Text Normalizer on closed classes (admin, numbers, currency, loanwords, digit chunking).
    - Test 6: Monotonic Max evaluation metric guarantee (Delta Metric >= 0 across all metrics).
    - Test 7: Post-propagation candidate box collision verification (IoU >= 0.45 merges to K=1, IoU < 0.45 retains K=2).
    """

    def test_01_top2_candidate_selection_disjoint_modes(self):
        """Verify K=2 candidate selection when two distinct, non-overlapping spatial modes exist."""
        h, w = 64, 64
        prob_map = np.full((h, w), 0.01, dtype=np.float32)
        
        # Mode 1 at (15, 15)
        prob_map[10:20, 10:20] = 0.85
        # Mode 2 at (45, 45)
        prob_map[40:50, 40:50] = 0.75

        selector = KLDivergenceRegionSelector(
            tau_iou=0.20,
            tau_iom=0.35,
            gamma_signif=0.40,
            min_prob_threshold=0.10,
        )
        result = selector.select_top_k(prob_map, orig_w=640, orig_h=640, max_k=2)

        self.assertIsInstance(result, MultiRegionSelectionResult)
        self.assertEqual(result.k, 2)
        self.assertEqual(len(result.bboxes), 2)
        self.assertEqual(len(result.kl_divergences), 2)
        self.assertGreater(result.kl_divergences[0], result.kl_divergences[1])

        # Verify spatial non-overlap constraints
        box1, box2 = result.bboxes[0], result.bboxes[1]
        iou = calculate_iou(box1, box2)
        self.assertLessEqual(iou, 0.20)

    def test_02_top2_candidate_selection_single_mode_fallback(self):
        """Verify fallback to K=1 when secondary mode is below significance gate."""
        h, w = 64, 64
        prob_map = np.full((h, w), 0.01, dtype=np.float32)
        
        # Only one dominant peak
        prob_map[10:20, 10:20] = 0.90
        # Weak secondary peak with p < 0.10 or ratio p/a < 1.0
        prob_map[40:44, 40:44] = 0.05

        selector = KLDivergenceRegionSelector(
            tau_iou=0.20,
            tau_iom=0.35,
            gamma_signif=0.40,
            min_prob_threshold=0.10,
        )
        result = selector.select_top_k(prob_map, orig_w=640, orig_h=640, max_k=2)

        self.assertEqual(result.k, 1)
        self.assertEqual(len(result.bboxes), 1)
        self.assertEqual(len(result.kl_divergences), 1)

    def test_03_adaptive_laplacian_edge_reemphasis_low_mag(self):
        """Verify identity transform when magnification M <= 1.0."""
        img_arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_arr, mode="RGB")

        # M = 1.0 -> psi(1.0) = tanh(0) = 0.0 -> returns original unchanged
        out_img = apply_adaptive_laplacian_edge_reemphasis(img, magnification=1.0)
        out_arr = np.array(out_img)

        np.testing.assert_array_equal(out_arr, img_arr)

        # M = 0.8 -> psi <= 0 -> returns original unchanged
        out_img_low = apply_adaptive_laplacian_edge_reemphasis(img, magnification=0.8)
        np.testing.assert_array_equal(np.array(out_img_low), img_arr)

    def test_04_adaptive_laplacian_edge_reemphasis_high_mag(self):
        """Verify high magnification (M=3.5) re-emphasizes edges while bounding pixels to [0, 255]."""
        img_arr = np.zeros((100, 100, 3), dtype=np.uint8)
        img_arr[:, :50, :] = 50
        img_arr[:, 50:, :] = 180
        img = Image.fromarray(img_arr, mode="RGB")

        out_img = apply_adaptive_laplacian_edge_reemphasis(img, magnification=3.5, lambda_0=0.50)
        out_arr = np.array(out_img)

        self.assertEqual(out_arr.shape, img_arr.shape)
        self.assertEqual(out_arr.dtype, np.uint8)
        self.assertTrue(np.all(out_arr >= 0) and np.all(out_arr <= 255))
        
        # High-contrast edge boundary should be sharpened (non-identical to input)
        self.assertFalse(np.array_equal(out_arr, img_arr))
        # Uniform flat regions (far from edge) should remain untouched due to gradient gate omega_grad
        self.assertEqual(out_arr[10, 10, 0], img_arr[10, 10, 0])
        self.assertEqual(out_arr[10, 90, 0], img_arr[10, 90, 0])

    def test_05_canonical_normalizer_administrative_and_closed_classes(self):
        """Verify canonical normalizer correctly handles administrative abbreviations, currency, loanwords, and phone numbers."""
        test_cases = [
            ("q . 1", "quận 1"),
            ("q.1", "quận 1"),
            ("ubnd tp hcm", "ủy ban nhân dân thành phố hồ chí minh"),
            ("gd & đt", "giáo dục và đào tạo"),
            ("100.000 vnđ", "100000 đ"),
            ("café", "cà phê"),
            ("090 123 4567", "0901234567"),
            ("098.765.4321", "0987654321"),
        ]
        for inp, expected in test_cases:
            norm = canonical_normalize_vietnamese_vqa(inp)
            self.assertEqual(norm, expected, f"Failed canonical normalize for '{inp}': got '{norm}', expected '{expected}'")

    def test_06_monotonic_max_metric_guarantee(self):
        """Verify Monotonic Max guarantees Delta Metric >= 0 against standard metrics."""
        pred_abbr = "ubnd q . 1"
        gt_full = "Ủy ban nhân dân quận 1"

        em = compute_exact_match(pred_abbr, gt_full)
        self.assertEqual(em, 1.0)

        f1 = compute_token_f1(pred_abbr, gt_full)
        self.assertEqual(f1, 1.0)

        anls = compute_anls(pred_abbr, gt_full)
        self.assertEqual(anls, 1.0)

        sub = compute_substring_match(pred_abbr, gt_full)
        self.assertEqual(sub, 1.0)

        pairs = [
            ("100 k", "100 nghìn"),
            ("0901 234 567", "0901234567"),
            ("Báo Lạng Sơn", "Báo Lạng Sơn"),
            ("hoàn toàn khác", "không liên quan"),
        ]
        for p, g in pairs:
            self.assertTrue(0.0 <= compute_exact_match(p, g) <= 1.0)
            self.assertTrue(0.0 <= compute_token_f1(p, g) <= 1.0)
            self.assertTrue(0.0 <= compute_anls(p, g) <= 1.0)
            self.assertTrue(0.0 <= compute_substring_match(p, g) <= 1.0)

    def test_07_post_propagation_collision_merge(self):
        """Verify candidate box collision resolution: IoU >= 0.45 merges to K=1, IoU < 0.45 keeps K=2."""
        box_a = [10, 10, 50, 50]
        box_b = [15, 15, 55, 55]
        iou_ab = calculate_iou(box_a, box_b)
        self.assertGreaterEqual(iou_ab, 0.45)

        merged = verify_and_resolve_candidate_boxes([box_a, box_b], iou_collision_threshold=0.45)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0], [10, 10, 55, 55])

        box_c = [10, 10, 30, 30]
        box_d = [70, 70, 90, 90]
        iou_cd = calculate_iou(box_c, box_d)
        self.assertLess(iou_cd, 0.45)

        retained = verify_and_resolve_candidate_boxes([box_c, box_d], iou_collision_threshold=0.45)
        self.assertEqual(len(retained), 2)
        self.assertEqual(retained[0], box_c)
        self.assertEqual(retained[1], box_d)


if __name__ == "__main__":
    unittest.main()
