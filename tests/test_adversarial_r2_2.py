# -*- coding: utf-8 -*-
"""
Adversarial Stress Test Suite for Milestone R2-2.
Agent: Challenger R2-2 (challenger_r2_2)

Scope:
1. Adversarial Preamble Stripping:
   - Deeply nested conversational prefixes (3, 4, 5 levels; colon delimiter handling).
   - Legitimate answers containing prefix substrings ("đây là bí mật", "câu trả lời không có sẵn").
   - Unicode diacritics (Vietnamese tone marks in NFC vs NFD normalization).
   - Punctuation (trailing periods, commas, colons, exclamations; quotes).
   - Whitespace and degenerate inputs (empty, None, erratic spaces, numbers).

2. Adversarial Spatial Cropping (IsotropicROICropper.crop):
   - Extreme bounding boxes (W=1000, H=1; W=1, H=1000).
   - Degenerate boxes (0x0, 1x1, single-point).
   - Inverted coordinates (x2 < x1, y2 < y1, both inverted).
   - Out-of-bounds coordinates (negative, exceeding canvas).
   - Extreme canvas sizes (1x1, 2x2, 1000x1, 1x1000, 4000x4000).
   - Contract verification: zero ValueErrors, no inverted coords, source_bbox inside [0, W] x [0, H],
     valid 448x448 RGB output, valid positive finite magnification.
"""

import math
import os
import sys
import unicodedata
import unittest
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.benchmark.sg_vcd.pipeline import strip_conversational_preamble
from scripts.benchmark.sg_vcd.roi_cropper import IsotropicROICropper, CropResult


class TestAdversarialPreambleStripping(unittest.TestCase):
    """Adversarial stress tests for strip_conversational_preamble."""

    def test_01_nested_preambles_up_to_3_levels(self):
        """Verify peeling nested conversational preambles up to 3 levels."""
        # Level 1
        self.assertEqual(
            strip_conversational_preamble("Bức ảnh này là: 123"),
            "123"
        )
        # Level 2
        self.assertEqual(
            strip_conversational_preamble("Trong ảnh có chữ: Bức ảnh này là: 123"),
            "123"
        )
        # Level 3
        self.assertEqual(
            strip_conversational_preamble("Trong ảnh có chữ: Bức ảnh này là: Câu trả lời là: 123"),
            "123"
        )

    def test_02_deeply_nested_preambles_boundary_and_delimiters(self):
        """Stress test deeply nested prefixes (>3 passes) and colon handling in introductory patterns."""
        # Level 4 nesting
        nested_4 = "Có thể thấy rằng: Trong ảnh có chữ: Bức ảnh này là: Câu trả lời là: 123"
        res_4 = strip_conversational_preamble(nested_4)
        self.assertEqual(
            res_4, "123",
            f"Level 4 nesting failed to peel completely (loop limit issue): got '{res_4}'"
        )

        # Level 5 nesting with 'theo thông tin trong ảnh:' pattern
        nested_5 = "Theo thông tin trong ảnh: Có thể thấy rằng: Trong ảnh có chữ: Bức ảnh này là: Câu trả lời là: 123"
        res_5 = strip_conversational_preamble(nested_5)
        self.assertEqual(
            res_5, "123",
            f"Level 5 nesting failed to peel completely (colon delimiter unhandled): got '{res_5}'"
        )

    def test_03_legitimate_answers_and_prefix_substring_collisions(self):
        """Stress test whether legitimate entity answers containing prefix substrings are preserved."""
        # 1. Exact single prefix words as answers (must not become empty)
        single_words = [
            "đáp án",
            "câu trả lời",
            "bức ảnh",
            "hình ảnh",
            "kết quả",
            "thông tin",
        ]
        for word in single_words:
            res = strip_conversational_preamble(word)
            self.assertTrue(
                len(res) > 0,
                f"Single word answer '{word}' was stripped to empty string!"
            )

        # 2. Legitimate phrases that start with words like 'Trong', 'Theo', 'Dựa':
        phrases = [
            ("Trong lành và mát mẻ", "Trong lành và mát mẻ"),
            ("Theo đuổi ước mơ", "Theo đuổi ước mơ"),
            ("Dựa lưng vào tường", "Dựa lưng vào tường"),
            ("Chữ nổi trên bảng", "Chữ nổi trên bảng"),
        ]
        for text, expected in phrases:
            cleaned = strip_conversational_preamble(text)
            self.assertEqual(cleaned, expected, f"Over-stripped legitimate phrase '{text}' -> '{cleaned}'")

        # 3. Legitimate answers containing prefix phrases:
        # e.g., "đây là bí mật", "câu trả lời không có sẵn"
        # When unprompted, stripping 'đây là' turns 'đây là bí mật' into 'bí mật'
        # Check if full meaning is altered:
        ans_bi_mat = strip_conversational_preamble("đây là bí mật")
        self.assertEqual(
            ans_bi_mat, "đây là bí mật",
            f"Over-stripped legitimate answer 'đây là bí mật' -> '{ans_bi_mat}'"
        )
        ans_khong_co_san = strip_conversational_preamble("câu trả lời không có sẵn")
        self.assertEqual(
            ans_khong_co_san, "câu trả lời không có sẵn",
            f"Over-stripped legitimate answer 'câu trả lời không có sẵn' -> '{ans_khong_co_san}'"
        )

    def test_04_unicode_nfc_nfd_normalization(self):
        """Stress test Vietnamese tone marks across NFC and NFD normalization."""
        texts = [
            ("Bức ảnh này là: Phở Bò Tái", "Phở Bò Tái"),
            ("Trong ảnh có chữ: Bách Hóa Xanh", "Bách Hóa Xanh"),
            ("Đáp án là: Agribank", "Agribank"),
            ("Câu trả lời là: Bánh mì", "Bánh mì"),
        ]
        for t, expected in texts:
            # NFC test
            nfc_text = unicodedata.normalize("NFC", t)
            cleaned_nfc = strip_conversational_preamble(nfc_text)
            self.assertEqual(cleaned_nfc, expected, f"Failed on NFC: {nfc_text}")

            # NFD test (decomposed diacritics)
            nfd_text = unicodedata.normalize("NFD", t)
            cleaned_nfd = strip_conversational_preamble(nfd_text)
            # Must also equal expected when recomposed to NFC
            recomposed = unicodedata.normalize("NFC", cleaned_nfd)
            self.assertEqual(
                recomposed, expected,
                f"NFD input '{nfd_text}' failed preamble stripping: got '{recomposed}' (expected '{expected}')"
            )

    def test_05_punctuation_and_quotation_stress(self):
        """Test with complex quotation, punctuation, and trailing mark stripping."""
        cases = [
            # Standard quotes
            ('"Bệnh viện Chợ Rẫy"', "Bệnh viện Chợ Rẫy"),
            ("‘Trường THPT Lê Quý Đôn’", "Trường THPT Lê Quý Đôn"),
            ("“Công ty Cổ phần Sữa Việt Nam”", "Công ty Cổ phần Sữa Việt Nam"),
            # Nested quotes
            ('“"Vinamilk"”', "Vinamilk"),
            # Trailing periods and spaces
            ("Đáp án là: Hà Nội...  ", "Hà Nội"),
            # Trailing comma, colon, exclamation mark
            ("Câu trả lời là: Hà Nội,", "Hà Nội"),
            ("Đáp án là: Hà Nội:", "Hà Nội"),
            ("Bức ảnh này là: Hà Nội!", "Hà Nội"),
            # Punctuation inside numbers/dates should be preserved
            ("12:30", "12:30"),
            ("090-123-4567", "090-123-4567"),
            ("3.14159", "3.14159"),
        ]
        for inp, exp in cases:
            out = strip_conversational_preamble(inp)
            self.assertEqual(out, exp, f"Failed on input '{inp}': expected '{exp}', got '{out}'")

    def test_06_whitespace_and_empty_inputs(self):
        """Test boundary strings: empty, whitespace-only, tabs, newlines."""
        self.assertEqual(strip_conversational_preamble(""), "")
        self.assertEqual(strip_conversational_preamble("   "), "")
        self.assertEqual(strip_conversational_preamble("\t\n\r  "), "")
        self.assertEqual(strip_conversational_preamble(None), "")

        # Preamble with erratic spacing
        self.assertEqual(
            strip_conversational_preamble("Trong   ảnh    có   chữ :   Viettel"),
            "Viettel"
        )
        self.assertEqual(
            strip_conversational_preamble("\n\nCâu trả lời là: \t\t 42 \n"),
            "42"
        )

    def test_07_pure_numbers_and_special_symbols(self):
        """Numbers and symbols must be cleanly preserved."""
        cases = [
            ("Câu trả lời là: 0", "0"),
            ("Đáp án: -15", "-15"),
            ("Kết quả là: +84901234567", "+84901234567"),
            ("Trong ảnh có chữ: 50%", "50%"),
            ("Đây là: $1,000,000", "$1,000,000"),
        ]
        for inp, exp in cases:
            self.assertEqual(strip_conversational_preamble(inp), exp)


class TestAdversarialCropGeometries(unittest.TestCase):
    """Adversarial stress tests for IsotropicROICropper.crop."""

    def setUp(self):
        self.cropper = IsotropicROICropper(target_size=448, context_margin=0.15)

    def _verify_crop_contract(self, img: Image.Image, bbox, crop_res: CropResult):
        """Strict invariant verification on CropResult."""
        W, H = img.size
        self.assertIsInstance(crop_res, CropResult)

        # 1. Output tile shape and mode
        self.assertIsInstance(crop_res.roi_tile, Image.Image)
        self.assertEqual(crop_res.roi_tile.size, (448, 448))
        self.assertEqual(crop_res.roi_tile.mode, "RGB")

        # 2. Source bbox valid coordinates within canvas
        x1, y1, x2, y2 = crop_res.source_bbox
        self.assertGreaterEqual(x1, 0, f"x1 ({x1}) < 0")
        self.assertGreaterEqual(y1, 0, f"y1 ({y1}) < 0")
        self.assertLessEqual(x2, W, f"x2 ({x2}) > W ({W})")
        self.assertLessEqual(y2, H, f"y2 ({y2}) > H ({H})")
        self.assertLessEqual(x1, x2, f"Inverted coords: x1 ({x1}) > x2 ({x2})")
        self.assertLessEqual(y1, y2, f"Inverted coords: y1 ({y1}) > y2 ({y2})")

        # 3. Isotropic box non-inverted
        sx1, sy1, sx2, sy2 = crop_res.isotropic_box
        self.assertLessEqual(sx1, sx2, f"Inverted isotropic x: {sx1} > {sx2}")
        self.assertLessEqual(sy1, sy2, f"Inverted isotropic y: {sy1} > {sy2}")

        # 4. Magnification positive and finite
        self.assertIsInstance(crop_res.magnification, float)
        self.assertFalse(math.isnan(crop_res.magnification), "Magnification is NaN")
        self.assertFalse(math.isinf(crop_res.magnification), "Magnification is Inf")
        self.assertGreater(crop_res.magnification, 0.0, "Magnification <= 0")

    def test_08_spatial_extreme_aspect_ratios_w1000_h1_and_w1_h1000(self):
        """Extreme aspect ratios: W=1000, H=1 and W=1, H=1000."""
        # 1. Ultra-wide box (1000:1) on 1200x800 canvas
        canvas = Image.new("RGB", (1200, 800), color=(100, 150, 200))
        wide_bbox = (50, 300, 1050, 301)  # W=1000, H=1
        res_wide = self.cropper.crop(canvas, wide_bbox)
        self._verify_crop_contract(canvas, wide_bbox, res_wide)

        # 2. Ultra-tall box (1:1000) on 800x1200 canvas
        tall_canvas = Image.new("RGB", (800, 1200), color=(200, 150, 100))
        tall_bbox = (300, 50, 301, 1050)  # W=1, H=1000
        res_tall = self.cropper.crop(tall_canvas, tall_bbox)
        self._verify_crop_contract(tall_canvas, tall_bbox, res_tall)

        # 3. Ultra-wide box on canvas of exact height 1 (1000x1 canvas)
        canvas_1px_h = Image.new("RGB", (1000, 1), color=(255, 0, 0))
        bbox_1px_h = (0, 0, 1000, 1)
        res_1px_h = self.cropper.crop(canvas_1px_h, bbox_1px_h)
        self._verify_crop_contract(canvas_1px_h, bbox_1px_h, res_1px_h)

        # 4. Ultra-tall box on canvas of exact width 1 (1x1000 canvas)
        canvas_1px_w = Image.new("RGB", (1, 1000), color=(0, 255, 0))
        bbox_1px_w = (0, 0, 1, 1000)
        res_1px_w = self.cropper.crop(canvas_1px_w, bbox_1px_w)
        self._verify_crop_contract(canvas_1px_w, bbox_1px_w, res_1px_w)

    def test_09_spatial_degenerate_and_point_boxes(self):
        """Degenerate 0x0 and 1x1 bounding boxes."""
        canvas = Image.new("RGB", (640, 480), color=(50, 50, 50))
        cases = [
            (100, 100, 100, 100),  # 0x0 in middle
            (0, 0, 0, 0),          # 0x0 at top-left
            (640, 480, 640, 480),  # 0x0 at bottom-right
            (100, 100, 101, 101),  # 1x1 in middle
            (0, 0, 1, 1),          # 1x1 at top-left
            (639, 479, 640, 480),  # 1x1 at bottom-right
        ]
        for bbox in cases:
            res = self.cropper.crop(canvas, bbox)
            self._verify_crop_contract(canvas, bbox, res)

    def test_10_spatial_inverted_coordinates(self):
        """Inverted coordinates (x2 < x1, y2 < y1, both inverted)."""
        canvas = Image.new("RGB", (640, 480), color=(70, 80, 90))
        cases = [
            (300, 100, 100, 200),  # x2 < x1
            (100, 400, 200, 100),  # y2 < y1
            (500, 400, 200, 100),  # both inverted
            (1000, 100, 0, 99),    # extreme inverted
        ]
        for bbox in cases:
            res = self.cropper.crop(canvas, bbox)
            self._verify_crop_contract(canvas, bbox, res)

    def test_11_spatial_out_of_bounds_coordinates(self):
        """Bounding boxes completely or partially out of canvas."""
        canvas = Image.new("RGB", (640, 480), color=(10, 20, 30))
        cases = [
            (-200, -200, -50, -50),       # Fully negative
            (800, 600, 1000, 800),        # Fully beyond canvas
            (-100, -100, 800, 600),       # Enclosing canvas and extending beyond
            (-50, 100, 700, 200),         # Partially negative x, partially overflowing x
        ]
        for bbox in cases:
            res = self.cropper.crop(canvas, bbox)
            self._verify_crop_contract(canvas, bbox, res)

    def test_12_spatial_extreme_canvas_dimensions(self):
        """Extreme canvas dimensions: 1x1, 2x2, 4000x4000."""
        # 1x1 canvas
        c1 = Image.new("RGB", (1, 1), color=(128, 128, 128))
        res1 = self.cropper.crop(c1, (0, 0, 1, 1))
        self._verify_crop_contract(c1, (0, 0, 1, 1), res1)

        # 2x2 canvas
        c2 = Image.new("RGB", (2, 2), color=(128, 128, 128))
        res2 = self.cropper.crop(c2, (0, 0, 2, 2))
        self._verify_crop_contract(c2, (0, 0, 2, 2), res2)

        # 4000x4000 canvas
        c4000 = Image.new("RGB", (4000, 4000), color=(128, 128, 128))
        res4000 = self.cropper.crop(c4000, (1000, 1500, 3000, 2500))
        self._verify_crop_contract(c4000, (1000, 1500, 3000, 2500), res4000)


def run_adversarial_suite():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print("\n" + "=" * 50)
    print(f"ADVERSARIAL SUMMARY: Total={result.testsRun}, Passed={result.testsRun - len(result.failures) - len(result.errors)}, Failures={len(result.failures)}, Errors={len(result.errors)}")
    print("=" * 50)
    return result


if __name__ == "__main__":
    run_adversarial_suite()
