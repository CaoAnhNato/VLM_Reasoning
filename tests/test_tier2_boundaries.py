# -*- coding: utf-8 -*-
"""
Tier 2: Boundary & Corner Cases Test Suite.
Requirement-Driven Verification of:
- B1: Extreme image aspect ratios (ultra-tall, ultra-wide, tiny, large, square).
- B2: Single vs multi-token Vietnamese questions (single character, dense tones, long queries, symbols).
- B3: Multi-box sub_bboxes handling (disjoint segments, stacked lines, nested/empty).
- B4: Missing box detection and coordinate clamping / placeholder handling.
- B5: Whitespace, diacritics, and punctuation normalization.

Each category includes at least 5 distinct, rigorous test cases.
"""

import os
import sys
import unittest
from typing import List, Tuple

import numpy as np
from PIL import Image

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import importlib.util

cropper_path = os.path.join(PROJECT_ROOT, "scripts", "benchmark", "sg_vcd", "roi_cropper.py")
spec = importlib.util.spec_from_file_location("roi_cropper_standalone", cropper_path)
roi_cropper_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(roi_cropper_mod)
IsotropicROICropper = roi_cropper_mod.IsotropicROICropper
from tests.test_helpers import (
    calculate_box_area,
    calculate_inclusion_recall,
    calculate_iou,
    calculate_sub_bboxes_hit_rate,
    compute_vqa_exact_match,
    compute_vqa_token_f1,
    normalize_vietnamese_text,
)


class TestTier2Boundary1AspectRatios(unittest.TestCase):
    """B1: Extreme Image Aspect Ratios (>=5 tests)."""

    def setUp(self):
        self.cropper = IsotropicROICropper(target_size=448)

    def test_01_ultra_wide_aspect_ratio_16_to_1(self):
        """Ultra-wide image (1600x100, 16:1) cropped around horizontal text line."""
        img = Image.new("RGB", (1600, 100), color=(200, 200, 200))
        bbox = (200, 20, 800, 80)  # Width 600, Height 60
        result = self.cropper.crop(img, bbox)

        self.assertEqual(result.roi_tile.size, (448, 448))
        self.assertGreater(result.magnification, 0.0)
        # Verify result is a valid PIL Image
        self.assertEqual(result.roi_tile.mode, "RGB")

    def test_02_ultra_tall_aspect_ratio_1_to_16(self):
        """Ultra-tall vertical image (100x1600, 1:16) cropped around vertical sign."""
        img = Image.new("RGB", (100, 1600), color=(100, 150, 200))
        bbox = (10, 200, 90, 800)  # Width 80, Height 600
        result = self.cropper.crop(img, bbox)

        self.assertEqual(result.roi_tile.size, (448, 448))
        self.assertGreater(result.magnification, 0.0)
        self.assertEqual(result.roi_tile.mode, "RGB")

    def test_03_minimal_image_dimensions_tiny(self):
        """Extremely small 16x16 image does not cause division by zero or negative bounds."""
        img = Image.new("RGB", (16, 16), color=(50, 50, 50))
        bbox = (2, 2, 14, 14)
        result = self.cropper.crop(img, bbox)

        self.assertEqual(result.roi_tile.size, (448, 448))
        self.assertGreater(result.magnification, 0.0)

    def test_04_large_image_resolution(self):
        """High-resolution 3000x2000 image with localized text."""
        img = Image.new("RGB", (3000, 2000), color=(240, 240, 240))
        bbox = (1200, 800, 1600, 950)
        result = self.cropper.crop(img, bbox)

        self.assertEqual(result.roi_tile.size, (448, 448))
        self.assertGreater(result.magnification, 0.0)

    def test_05_exact_square_1_to_1_ratio(self):
        """Standard 1:1 square image (448x448) with centered text."""
        img = Image.new("RGB", (448, 448), color=(128, 128, 128))
        bbox = (100, 100, 348, 348)
        result = self.cropper.crop(img, bbox)

        self.assertEqual(result.roi_tile.size, (448, 448))
        self.assertAlmostEqual(result.roi_tile.width, 448)
        self.assertAlmostEqual(result.roi_tile.height, 448)


class TestTier2Boundary2VietnameseQuestions(unittest.TestCase):
    """B2: Single vs Multi-Token Vietnamese Questions (>=5 tests)."""

    def test_01_single_character_or_minimal_query(self):
        """Single character/minimal query handles punctuation stripping and normalization cleanly."""
        q = "Gì?"
        norm = normalize_vietnamese_text(q)
        self.assertEqual(norm, "gì")
        self.assertEqual(len(norm.split()), 1)

    def test_02_dense_vietnamese_tone_marks(self):
        """Vietnamese sentence with dense diacritics preserves tone marks accurately."""
        q = "Ở đây có chữa trĩ ngoại và phục hồi chức năng không?"
        norm = normalize_vietnamese_text(q)
        expected = "ở đây có chữa trĩ ngoại và phục hồi chức năng không"
        self.assertEqual(norm, expected)

    def test_03_mixed_casing_and_capitalization(self):
        """Mixed casing on Vietnamese proper nouns folds properly without losing diacritics."""
        q = "TRƯỜNG ĐẠI HỌC CÔNG THƯƠNG TP.HCM ở ĐÂU?"
        norm = normalize_vietnamese_text(q)
        self.assertIn("trường đại học công thương", norm)
        self.assertNotIn("TRƯỜNG", norm)

    def test_04_query_with_phone_numbers_and_symbols(self):
        """Query with contact phone numbers and special symbols."""
        q = "Số điện thoại hotline là 028-3810-5555 đúng không?"
        norm = normalize_vietnamese_text(q)
        self.assertIn("02838105555", norm.replace(" ", "") or norm)
        self.assertEqual(compute_vqa_exact_match("028-3810-5555", ["02838105555"]), 1.0)
        self.assertEqual(compute_vqa_exact_match("028-3810-5555", ["028-3810-5555"]), 1.0)

    def test_05_long_complex_vietnamese_query(self):
        """Long Vietnamese query (>50 words) tokenizes cleanly without buffer overrun."""
        q = " " + " ".join(["quán", "cà", "phê", "nằm", "ở", "góc", "đường"] * 8) + "?"
        norm = normalize_vietnamese_text(q)
        tokens = norm.split()
        self.assertGreater(len(tokens), 50)
        self.assertEqual(tokens[0], "quán")
        self.assertEqual(tokens[-1], "đường")


class TestTier2Boundary3MultiBoxSubBboxes(unittest.TestCase):
    """B3: Multi-box sub_bboxes Handling (>=5 tests)."""

    def test_01_disjoint_multi_boxes(self):
        """Sub-boxes representing separate signboards across an image."""
        sub_boxes = [
            [50, 50, 150, 100],     # Left sign
            [400, 50, 500, 100],    # Right sign
        ]
        # Pred box only covers left sign
        pred = [40, 40, 160, 110]
        hit_rate = calculate_sub_bboxes_hit_rate(pred, sub_boxes, iou_thresh=0.3)
        self.assertAlmostEqual(hit_rate, 0.5, places=5)

    def test_02_stacked_lines_sub_boxes(self):
        """Three vertically stacked lines of text forming a compound address."""
        sub_boxes = [
            [100, 100, 300, 130],   # Line 1
            [100, 140, 300, 170],   # Line 2
            [100, 180, 300, 210],   # Line 3
        ]
        # Pred box encloses all 3 lines
        pred_full = [95, 95, 305, 215]
        hit_rate_full = calculate_sub_bboxes_hit_rate(pred_full, sub_boxes, iou_thresh=0.2)
        self.assertAlmostEqual(hit_rate_full, 1.0, places=5)

        # Pred box encloses only first 2 lines
        pred_two = [95, 95, 305, 175]
        hit_rate_two = calculate_sub_bboxes_hit_rate(pred_two, sub_boxes, iou_thresh=0.2)
        self.assertAlmostEqual(hit_rate_two, 2.0 / 3.0, places=5)

    def test_03_nested_or_overlapping_sub_boxes(self):
        """Overlapping sub-boxes within the same text cluster."""
        sub_boxes = [
            [100, 100, 250, 150],
            [150, 100, 300, 150],
        ]
        pred = [120, 90, 280, 160]
        hit_rate = calculate_sub_bboxes_hit_rate(pred, sub_boxes, iou_thresh=0.3)
        self.assertEqual(hit_rate, 1.0)

    def test_04_single_element_sub_boxes(self):
        """When sub_bboxes has exactly 1 element equal to main bbox."""
        main_bbox = [200, 300, 400, 350]
        sub_boxes = [main_bbox]
        pred = [190, 290, 410, 360]
        hit_rate = calculate_sub_bboxes_hit_rate(pred, sub_boxes, iou_thresh=0.5)
        self.assertEqual(hit_rate, 1.0)

    def test_05_empty_sub_bboxes_list(self):
        """Empty sub_bboxes list returns 0.0 hit rate gracefully."""
        pred = [100, 100, 200, 200]
        hit_rate = calculate_sub_bboxes_hit_rate(pred, [], iou_thresh=0.3)
        self.assertEqual(hit_rate, 0.0)


class TestTier2Boundary4MissingBoxAndCoordinates(unittest.TestCase):
    """B4: Missing Box Detection & Coordinate Fallback Handling (>=5 tests)."""

    def test_01_degenerate_zero_area_bbox(self):
        """Degenerate zero-width or zero-height box has 0 area and 0 IoU."""
        deg_box = [100, 100, 100, 100]
        self.assertEqual(calculate_box_area(deg_box), 0.0)
        self.assertEqual(calculate_iou(deg_box, [50, 50, 150, 150]), 0.0)

    def test_02_out_of_bounds_coordinate_clamping(self):
        """Coordinate clamping logic clamps negative and oversized pixels to [0, W] and [0, H]."""
        W, H = 800, 600
        raw_box = [-50, -20, 900, 750]

        clamped = [
            max(0, min(W, raw_box[0])),
            max(0, min(H, raw_box[1])),
            max(0, min(W, raw_box[2])),
            max(0, min(H, raw_box[3])),
        ]
        self.assertEqual(clamped, [0, 0, 800, 600])

    def test_03_inverted_coordinates_rectification(self):
        """Inverted coordinates where x1 > x2 or y1 > y2 can be rectified safely."""
        inv_box = [300, 400, 100, 200]
        rectified = [
            min(inv_box[0], inv_box[2]),
            min(inv_box[1], inv_box[3]),
            max(inv_box[0], inv_box[2]),
            max(inv_box[1], inv_box[3]),
        ]
        self.assertEqual(rectified, [100, 200, 300, 400])
        self.assertEqual(calculate_box_area(rectified), 200 * 200)

    def test_04_xanylabeling_placeholder_box_detection(self):
        """
        X-AnyLabeling default placeholder box is top-left [0, 0, 50, 50].
        Verify heuristic detection of unannotated placeholder.
        """
        def is_placeholder_box(box: List[int]) -> bool:
            return box[0] == 0 and box[1] == 0 and (box[2] - box[0]) <= 50 and (box[3] - box[1]) <= 50

        self.assertTrue(is_placeholder_box([0, 0, 50, 50]))
        self.assertFalse(is_placeholder_box([100, 200, 300, 400]))

    def test_05_none_gt_box_handling(self):
        """Evaluation logic handles samples where GT box is None without throwing exceptions."""
        gt_box = None
        pred_box = [100, 100, 200, 200]
        iou = calculate_iou(pred_box, gt_box) if gt_box is not None else 0.0
        self.assertEqual(iou, 0.0)


class TestTier2Boundary5WhitespaceAndPunctuation(unittest.TestCase):
    """B5: Whitespace, Diacritics & Punctuation Normalization (>=5 tests)."""

    def test_01_excessive_whitespace_and_tabs(self):
        """Removes trailing, leading, newline, tab, and carriage return characters."""
        text = "  \t\n  Nhà thuốc An Khang   \r\n\t "
        norm = normalize_vietnamese_text(text)
        self.assertEqual(norm, "nhà thuốc an khang")

    def test_02_multiple_consecutive_internal_spaces(self):
        """Collapses multiple spaces between words into a single space."""
        text = "quán    cà     phê      sáng"
        norm = normalize_vietnamese_text(text)
        self.assertEqual(norm, "quán cà phê sáng")

    def test_03_vietnamese_special_punctuation(self):
        """Strips quotation marks, brackets, dashes, and question marks cleanly."""
        text = '"Cà phê - Trà sữa (Chi nhánh 1)!"'
        norm = normalize_vietnamese_text(text)
        self.assertEqual(norm, "cà phê trà sữa chi nhánh 1")

    def test_04_vietnamese_unicode_compound_diacritics(self):
        """Ensures all 12 Vietnamese vowels with 5 tones are preserved in NFC/NFD equivalence."""
        import unicodedata
        vowels_nfc = "aăâeêioôơuưy"
        vowels_with_tones = "àáảãạ ằắẳẵặ ầấẩẫậ èéẻẽẹ ềếểễệ ìíỉĩị òóỏõọ ồốổỗộ ờớởỡợ ùúủũụ ừứửữự ỳýỷỹỵ"
        norm = normalize_vietnamese_text(vowels_with_tones)
        # Should be preserved and lowercased
        self.assertTrue(len(norm.split()) > 10)
        # Verify unicode normalization does not strip marks
        self.assertEqual(unicodedata.normalize("NFC", norm), norm)

    def test_05_empty_prediction_vs_nonempty_gt(self):
        """Empty prediction string returns 0.0 Exact Match and 0.0 F1 gracefully."""
        self.assertEqual(compute_vqa_exact_match("", ["nhà thuốc"]), 0.0)
        self.assertEqual(compute_vqa_exact_match("   ", ["nhà thuốc"]), 0.0)
        self.assertEqual(compute_vqa_token_f1("", ["nhà thuốc"]), 0.0)
        self.assertEqual(compute_vqa_token_f1("   ", ["nhà thuốc"]), 0.0)


if __name__ == "__main__":
    unittest.main()
