"""
Adversarial Stress Test Suite for SyntacticSpanRefiner in SG-VCD.
Validates:
1. Non-Collapsing Invariant (output never empty for non-empty input)
2. Structural delimiters (ratios, timestamps, multi-colons, spaced dashes)
3. Quotes, apostrophes, boundary punctuation, trailing ellipses
4. Vietnamese Unicode robustness (NFC vs NFD canonical equivalence)
5. Single-word interrogative answer preservation
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import unittest
import unicodedata
from scripts.benchmark.sg_vcd.pipeline import SyntacticSpanRefiner


class TestSyntacticSpanRefinerAdversarial(unittest.TestCase):
    def setUp(self):
        self.refiner = SyntacticSpanRefiner()

    # =========================================================================
    # 1. Non-Collapsing Invariant Tests (MUST NEVER COLLAPSE TO EMPTY)
    # =========================================================================
    def test_empty_and_whitespace_inputs(self):
        self.assertEqual(self.refiner.refine(""), "")
        self.assertEqual(self.refiner.refine("   "), "")
        self.assertEqual(self.refiner.refine(None), "")

    def test_100_percent_question_token_overlap(self):
        """Even when 100% of input words match question tokens, output must NEVER collapse to empty."""
        cases = [
            ("trong ảnh", "trong ảnh", "ảnh"),
            ("ảnh này có chữ gì", "ảnh này có chữ gì?", "gì"),
            ("có", "trong ảnh có gì không?", "có"),
            ("không", "trong ảnh có người không?", "không"),
            ("bức ảnh", "bức ảnh này", "ảnh"),
            ("xe máy", "trong ảnh có xe máy không?", "máy"),
            ("chữ trên biển", "chữ trên biển là gì?", "biển"),
        ]
        for pred, q, expected_substr in cases:
            out = self.refiner.refine(pred, question=q)
            self.assertTrue(len(out) > 0, f"Collapsing violation: refine('{pred}', '{q}') returned empty!")
            self.assertIn(expected_substr.lower(), out.lower(), f"Expected core token '{expected_substr}' in '{out}'")

    def test_exhaustive_question_subsets_non_collapsing(self):
        """Check all 1-word and 2-word contiguous combinations matching question words."""
        q = "bức ảnh này có bao nhiêu người đi bộ trên đường phố"
        words = q.split()
        for i in range(len(words)):
            single_word = words[i]
            out = self.refiner.refine(single_word, question=q)
            self.assertTrue(len(out) > 0, f"Single-word collapsed: '{single_word}' -> '{out}'")
            if i < len(words) - 1:
                two_words = f"{words[i]} {words[i+1]}"
                out2 = self.refiner.refine(two_words, question=q)
                self.assertTrue(len(out2) > 0, f"Two-word collapsed: '{two_words}' -> '{out2}'")

    # =========================================================================
    # 2. Structural Delimiters Tests
    # =========================================================================
    def test_ratio_preservation(self):
        """Preserve numeric ratios such as 3:2, 2:1, 16:9, 1:1000."""
        ratios = ["3:2", "2:1", "16:9", "1:1000", "4:3", "21:9"]
        for r in ratios:
            out = self.refiner.refine(r)
            self.assertEqual(out, r, f"Ratio mangled: '{r}' -> '{out}'")

            out_prefixed = self.refiner.refine(f"Tỉ số: {r}")
            self.assertEqual(out_prefixed, r, f"Prefixed ratio mangled: 'Tỉ số: {r}' -> '{out_prefixed}'")

    def test_timestamp_preservation(self):
        """Preserve HH:MM:SS and HH:MM timestamps."""
        times = ["08:30:00", "10:30", "12:45:15", "00:00:00", "23:59:59"]
        for t in times:
            out = self.refiner.refine(t)
            self.assertEqual(out, t, f"Timestamp mangled: '{t}' -> '{out}'")

            out_labeled = self.refiner.refine(f"Giờ mở cửa: {t}")
            self.assertEqual(out_labeled, t, f"Labeled timestamp mangled: 'Giờ mở cửa: {t}' -> '{out_labeled}'")

    def test_multi_colon_lines(self):
        """Verify handling of multiple colons in a single line (rightmost payload)."""
        self.assertEqual(self.refiner.refine("Thương hiệu: Nike: Chi nhánh 1"), "Chi nhánh 1")
        self.assertEqual(self.refiner.refine("A: B: C"), "C")
        self.assertEqual(self.refiner.refine("Mã: SP01: Giá: 50.000đ"), "50.000đ")

    def test_unspaced_dash_preservation(self):
        """Unspaced hyphens must never be split."""
        self.assertEqual(self.refiner.refine("COVID-19"), "COVID-19")
        self.assertEqual(self.refiner.refine("Wi-Fi"), "Wi-Fi")
        self.assertEqual(self.refiner.refine("TP.HCM-Hà Nội"), "TP.HCM-Hà Nội")

    def test_spaced_dash_brand_in_question(self):
        """When the brand from segment 0 appears in question, segment 1 is correctly selected."""
        text = "Highlands Coffee - Tầng 1"
        out = self.refiner.refine(text, question="Highlands Coffee ở tầng mấy?")
        self.assertEqual(out, "1")

    # =========================================================================
    # 3. Quotes, Punctuation & Ellipses Tests
    # =========================================================================
    def test_balanced_and_nested_quotes(self):
        """Verify balanced outer quotes, unicode quotes, and nested quotes."""
        self.assertEqual(self.refiner.refine('"Hà Nội"'), "Hà Nội")
        self.assertEqual(self.refiner.refine("'Hà Nội'"), "Hà Nội")
        self.assertEqual(self.refiner.refine("“Hà Nội”"), "Hà Nội")
        self.assertEqual(self.refiner.refine("«Hà Nội»"), "Hà Nội")
        self.assertEqual(self.refiner.refine("“‘Hà Nội’”"), "Hà Nội")
        self.assertEqual(self.refiner.refine('"\'Hà Nội\'"'), "Hà Nội")

    def test_internal_quoted_entity_extraction(self):
        """Verify internal quotes extraction from full conversational sentence."""
        s = "Trong bức ảnh có chữ 'Siêu Thị Big C' trên bảng hiệu."
        self.assertEqual(self.refiner.refine(s), "Siêu Thị Big C")

        s2 = 'Đáp án là "Phòng Khám Đa Khoa".'
        self.assertEqual(self.refiner.refine(s2), "Phòng Khám Đa Khoa")

    def test_single_apostrophe_brand(self):
        """Single apostrophe inside words is preserved."""
        self.assertEqual(self.refiner.refine("McDonald's"), "McDonald's")
        self.assertEqual(self.refiner.refine("L'Oreal"), "L'Oreal")
        self.assertEqual(self.refiner.refine("O'Connor"), "O'Connor")

    def test_trailing_ascii_punctuation(self):
        """Verify removal of trailing periods, exclamation marks, commas, colons, ascii ellipses."""
        self.assertEqual(self.refiner.refine("Hà Nội."), "Hà Nội")
        self.assertEqual(self.refiner.refine("Hà Nội!"), "Hà Nội")
        self.assertEqual(self.refiner.refine("Hà Nội?"), "Hà Nội")
        self.assertEqual(self.refiner.refine("Hà Nội,"), "Hà Nội")
        self.assertEqual(self.refiner.refine("Hà Nội:"), "Hà Nội")
        self.assertEqual(self.refiner.refine("Hà Nội..."), "Hà Nội")
        self.assertEqual(self.refiner.refine("Hà Nội!?...,:"), "Hà Nội")

    # =========================================================================
    # 4. Vietnamese Unicode Robustness Tests (NFC vs NFD)
    # =========================================================================
    def test_vietnamese_nfc_nfd_equivalence(self):
        """Verify that NFC and NFD input representations yield identical NFC output."""
        test_strings = [
            "Tiếng Việt giàu và đẹp",
            "Đường Bà Triệu, Quận Hai Bà Trưng",
            "Phở bò sốt vang gia truyền",
            "Trường Đại học Công Thương TP.HCM",
            "Nhà thuốc Bảo Tâm An",
            "Bệnh viện Đa khoa Quốc tế",
            "Cửa hàng Bách Hóa Xanh",
            "Trung tâm Huấn luyện Kỹ thuật Thể thao",
        ]
        for s in test_strings:
            nfc_input = unicodedata.normalize("NFC", s)
            nfd_input = unicodedata.normalize("NFD", s)

            out_nfc = self.refiner.refine(nfc_input)
            out_nfd = self.refiner.refine(nfd_input)

            self.assertEqual(out_nfc, out_nfd, f"NFC/NFD mismatch for: {s}")
            self.assertEqual(unicodedata.normalize("NFC", out_nfd), out_nfd, "Output must be NFC normalized!")

    def test_cross_normalized_question_matching(self):
        """Test NFD prediction against NFC question and vice-versa in QAMIF."""
        pred_base = "bức ảnh này có chữ VinMart"
        q_base = "bức ảnh này có chữ gì?"

        # NFD prediction, NFC question
        pred_nfd = unicodedata.normalize("NFD", pred_base)
        q_nfc = unicodedata.normalize("NFC", q_base)
        self.assertEqual(self.refiner.refine(pred_nfd, question=q_nfc), "VinMart")

        # NFC prediction, NFD question
        pred_nfc = unicodedata.normalize("NFC", pred_base)
        q_nfd = unicodedata.normalize("NFD", q_base)
        self.assertEqual(self.refiner.refine(pred_nfc, question=q_nfd), "VinMart")

    # =========================================================================
    # 5. Single-Word Answers Matching Question Interrogatives
    # =========================================================================
    def test_single_word_interrogative_answers(self):
        """True single-word answers matching question words must be strictly preserved."""
        cases = [
            ("Có", "Trong ảnh có ô tô không?", "Có"),
            ("Không", "Bức ảnh có biển báo không?", "Không"),
            ("Đúng", "Đây có đúng là trường học không?", "Đúng"),
            ("Sai", "Thông tin này đúng hay sai?", "Sai"),
            ("Gì", "Từ trên biển hiệu là gì?", "Gì"),
            ("Ai", "Người trong ảnh là ai?", "Ai"),
        ]
        for pred, q, expected in cases:
            out = self.refiner.refine(pred, question=q)
            self.assertEqual(out, expected, f"Single-word '{pred}' improperly stripped to '{out}' for Q: '{q}'")


if __name__ == "__main__":
    unittest.main()
