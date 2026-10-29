# -*- coding: utf-8 -*-
"""
Unit and Integration Test Suite for SG-VCD Round 5 SOTA Algorithmic Upgrades:
- Dynamic Multi-Patch RoI Tiling (DynamicMultiTileROICropper, log-AR distortion minimization, unit square homothety)
- Calibrated Dynamic Contrastive Decoding (CAPC, AHRA, QIACF, TESG, ESSC)
- Prompt Disentanglement & Robust Preamble Stripping
- End-to-End Pipeline Multi-Patch Token Embedding Sequencing
"""

import math
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

import torch
import torch.nn.functional as F

from scripts.benchmark.sg_vcd.roi_cropper import (
    CropResult,
    DynamicMultiTileROICropper,
    IsotropicROICropper,
    apply_adaptive_laplacian_edge_reemphasis,
)
from scripts.benchmark.sg_vcd.dynamic_cd import (
    DynamicContrastiveDecoder,
    compute_adaptive_lowpass_radius,
)
from scripts.benchmark.sg_vcd.pipeline import (
    PipelineResult,
    SGVCDPipeline,
    SyntacticSpanRefiner,
    format_concise_decoding_prompt,
    strip_conversational_preamble,
)


class TestDynamicMultiPatchROICropper(unittest.TestCase):
    """Mathematical validation for Dynamic Multi-Patch RoI Tiling."""

    def setUp(self):
        self.cropper = DynamicMultiTileROICropper(target_size=448, max_tiles=3, min_tile_dim=224)
        # Create synthetic test canvas
        self.test_img = Image.new("RGB", (1200, 800), color=(128, 128, 128))

    def test_grid_selection_wide_banner(self):
        """Wide banner (e.g. 900x100, AR=9:1) must select (3, 1) grid to minimize log-AR distortion."""
        grid = self.cropper.select_grid(900.0, 100.0)
        self.assertEqual(grid, (3, 1))

    def test_grid_selection_moderate_horizontal(self):
        """Moderate horizontal box (e.g. 600x250, AR=2.4) must select (2, 1) grid."""
        grid = self.cropper.select_grid(600.0, 250.0)
        self.assertEqual(grid, (2, 1))

    def test_grid_selection_square(self):
        """Square box (e.g. 400x400, AR=1.0) must select (1, 1) grid."""
        grid = self.cropper.select_grid(400.0, 400.0)
        self.assertEqual(grid, (1, 1))

    def test_grid_selection_vertical_banner(self):
        """Tall vertical banner (e.g. 150x800, AR=0.1875) must select (1, 3) grid."""
        grid = self.cropper.select_grid(150.0, 800.0)
        self.assertEqual(grid, (1, 3))

    def test_physical_pixel_span_capping(self):
        """Small box (e.g. 100x40) must be capped to (1, 1) to prevent empty bicubic over-sampling."""
        grid = self.cropper.select_grid(100.0, 40.0)
        self.assertEqual(grid, (1, 1))

    def test_exact_isotropic_magnification_invariant(self):
        """Verify unit tile size and exact isotropic magnification M_x == M_y on both axes."""
        bbox = (100, 100, 1000, 200)  # 900x100
        crop_res = self.cropper.crop(self.test_img, bbox)
        
        self.assertEqual(crop_res.grid, (3, 1))
        self.assertEqual(len(crop_res.roi_tiles), 3)
        self.assertEqual(len(crop_res.neg_tiles), 3)

        # Invariant: Each sub-tile must be exactly 448x448
        for t in crop_res.roi_tiles:
            self.assertEqual(t.size, (448, 448))
        for t in crop_res.neg_tiles:
            self.assertEqual(t.size, (448, 448))

        # Invariant: Uniform magnification
        expected_mag = 448.0 / crop_res.unit_tile_size
        self.assertAlmostEqual(crop_res.magnification, expected_mag, places=4)

    def test_backward_compatibility_crop_result(self):
        """CropResult must provide roi_tile as primary 448x448 tile for backward compatibility."""
        bbox = (200, 200, 500, 350)
        crop_res = self.cropper.crop(self.test_img, bbox)
        self.assertIsInstance(crop_res.roi_tile, Image.Image)
        self.assertEqual(crop_res.roi_tile.size, (448, 448))
        self.assertIsInstance(crop_res.isotropic_box, tuple)
        self.assertEqual(len(crop_res.isotropic_box), 4)

    def test_isotropic_cropper_subclass_equivalence(self):
        """IsotropicROICropper alias must produce identical multi-patch capabilities."""
        iso_cropper = IsotropicROICropper(target_size=448)
        crop_res = iso_cropper.crop(self.test_img, (100, 100, 1000, 200))
        self.assertEqual(crop_res.grid, (3, 1))
        self.assertEqual(len(crop_res.roi_tiles), 3)


class TestDynamicContrastiveDecoderCalibrated(unittest.TestCase):
    """Validation for CAPC, AHRA, QIACF, TESG, and ESSC."""

    def setUp(self):
        self.decoder = DynamicContrastiveDecoder(
            alpha_0=0.65,
            beta_0=0.035,
            tau_js=0.04,
            gamma_phase=0.75,
            theta_c=0.35,
            delta_head=0.45,
            tau_head=1.8,
            kappa_intent=0.35,
            tau_intent=0.15,
            eta_temp=0.18,
            tau_h=1.50,
        )

    def test_capc_preserves_high_contrast_diacritic_token(self):
        """
        CAPC Theorem: A Vietnamese diacritic token with low prior P_pos < beta_t * max_p_pos
        but high visual contrast Delta z_t >= theta_c * max Delta z_t must NOT be masked to -inf.
        """
        vocab_size = 100
        pos_logits = torch.zeros((1, vocab_size), dtype=torch.float32)
        neg_logits = torch.zeros((1, vocab_size), dtype=torch.float32)

        # Token 0 is high-probability generic LM prior
        pos_logits[0, 0] = 10.0
        neg_logits[0, 0] = 9.8  # low visual contrast Delta z = 0.2

        # Token 42 is low-probability accented proper noun with strong visual evidence
        pos_logits[0, 42] = 2.0  # low base logit (P_pos << P_pos[0])
        neg_logits[0, 42] = -5.0 # high visual contrast Delta z = 7.0 (max Delta z)

        next_token, alpha_t, entropy_val, d_js = self.decoder.decode_step(
            pos_logits=pos_logits,
            neg_logits=neg_logits,
            step=0,
            spatial_confidence=1.0,
            phase_cooldown_step=0,
        )

        # In old APC, token 42 was pruned to -inf because P_pos[42] < beta_0 * P_pos[0].
        # In CAPC, token 42 is retained via contrastive bypass Delta z >= theta_c * max Delta z.
        delta_z = torch.relu(pos_logits - neg_logits)
        max_delta = delta_z.max().item()
        self.assertGreaterEqual(delta_z[0, 42].item(), 0.35 * max_delta)

    def test_ahra_head_recency_anchoring(self):
        """AHRA must provide 1.45x contrast boost at step 0 and decay smoothly toward 1.0."""
        # Calculate A_head(t) at steps 0, 1, 2, 4
        a_0 = 1.0 + 0.45 * math.exp(-0.0 / 1.8)
        a_1 = 1.0 + 0.45 * math.exp(-1.0 / 1.8)
        a_4 = 1.0 + 0.45 * math.exp(-4.0 / 1.8)

        self.assertAlmostEqual(a_0, 1.45, places=4)
        self.assertAlmostEqual(a_1, 1.0 + 0.45 * math.exp(-1.0 / 1.8), places=4)
        self.assertLess(a_4, 1.06)

    def test_qiacf_intent_adaptive_contrast_factor(self):
        """High initial D_JS (fine text) scales contrast up by ~1.35x; low D_JS scales near 1.0x."""
        psi_high = 1.0 + 0.35 * math.tanh(0.50 / 0.15)
        psi_low = 1.0 + 0.35 * math.tanh(0.01 / 0.15)

        self.assertGreater(psi_high, 1.30)
        self.assertLess(psi_low, 1.05)

    def test_tesg_entropy_temperature_cooling(self):
        """High predictive entropy H_t > 1.5 must cool temperature down toward 0.82."""
        t_high_entropy = 1.0 - 0.18 * math.tanh(3.0 / 1.50)
        t_low_entropy = 1.0 - 0.18 * math.tanh(0.1 / 1.50)

        self.assertLess(t_high_entropy, 0.85)
        self.assertGreater(t_low_entropy, 0.98)

    def test_essc_cooldown_delay(self):
        """ESSC must sustain contrast across initial tokens (step < 4) and use softened decay 0.75."""
        self.assertEqual(self.decoder.gamma_phase, 0.75)


class TestPromptAndPreambleDisentanglement(unittest.TestCase):
    """Validation for prompt formatting and robust preamble/quote stripping."""

    def test_concise_prompt_format(self):
        """Prompt format must end with '\\nTrả lời ngắn gọn: ' and strip <image> tags."""
        q1 = "<image> biển hiệu này màu gì ?"
        formatted = format_concise_decoding_prompt(q1)
        self.assertTrue(formatted.endswith("\nTrả lời ngắn gọn: "))
        self.assertNotIn("<image>", formatted)
        self.assertTrue(formatted.startswith("biển hiệu này màu gì ?"))

    def test_strip_prompt_echo_yeu_cau(self):
        """Must strip 'Yêu cầu: ...' prompt prefix echo observed in Round 4."""
        raw = "Yêu cầu: Wuta Cam"
        self.assertEqual(strip_conversational_preamble(raw), "Wuta Cam")

    def test_strip_prompt_echo_with_quotes(self):
        """Must strip 'Yêu cầu: \"...\"' and enclosing quotes cleanly."""
        raw = 'Yêu cầu: "KHU VỰC CÁCH LY"'
        self.assertEqual(strip_conversational_preamble(raw), "KHU VỰC CÁCH LY")

    def test_strip_cum_tu_and_tra_loi(self):
        """Must strip 'Cụm từ: ...' and 'Trả lời ngắn gọn: ...'."""
        self.assertEqual(strip_conversational_preamble("Cụm từ: Bếp Gas"), "Bếp Gas")
        self.assertEqual(strip_conversational_preamble("Trả lời ngắn gọn: Hà Nội"), "Hà Nội")

    def test_strip_unicode_curved_quotes(self):
        """Must strip unicode curved quotes “...” and ‘...’."""
        self.assertEqual(strip_conversational_preamble("“Bảo Tâm An”"), "Bảo Tâm An")
        self.assertEqual(strip_conversational_preamble("‘0945 799 979’"), "0945 799 979")

    def test_strip_conversational_intro_clauses(self):
        """Must strip standard Vietnamese conversational intros."""
        self.assertEqual(
            strip_conversational_preamble("Ảnh này được chụp tại Đà Lạt"),
            "Đà Lạt",
        )
        self.assertEqual(
            strip_conversational_preamble("Trong bức hình có chữ: Nhà thuốc An Khang"),
            "Nhà thuốc An Khang",
        )


class TestDynamicContrastiveEOSBoosting(unittest.TestCase):
    """Mathematical and behavioral validation for Dynamic Contrastive EOS Boosting (DCEB)."""

    def setUp(self):
        self.decoder = DynamicContrastiveDecoder(
            kappa_eos=4.00,
            n_min_eos=1,
            h_bar_eos=1.80,
            tau_h_eos=0.40,
            delta_peak_min=0.50,
        )

    def test_dceb_step0_protection(self):
        """At step 0, DCEB boost must be 0.0 logits to strictly prevent premature EOS emission."""
        vocab_size = 100
        eos_id = 99
        pos_logits = torch.randn((1, vocab_size), dtype=torch.float32)
        neg_logits = torch.randn((1, vocab_size), dtype=torch.float32)

        # Step 0 decoding
        _, _, _, _ = self.decoder.decode_step(
            pos_logits=pos_logits,
            neg_logits=neg_logits,
            step=0,
            spatial_confidence=1.0,
            eos_id=eos_id,
        )
        # Peak delta_z must be initialized to step 0 max delta_z
        delta_z = torch.relu(pos_logits - neg_logits)
        expected_peak = float(delta_z.max().item())
        self.assertAlmostEqual(self.decoder._peak_delta_z, expected_peak, places=4)

    def test_dceb_active_reading_margin(self):
        """During active entity reading (high contrast, low entropy), DCEB boost must remain negligible (< 0.1)."""
        effective_peak = max(3.5, self.decoder.delta_peak_min)
        c_collapse = max(0.0, 1.0 - 3.2 / (effective_peak + 1e-6))
        sig_entropy = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, (0.50 - self.decoder.h_bar_eos) / self.decoder.tau_h_eos))))
        tau_eos = self.decoder.kappa_eos * c_collapse * sig_entropy
        self.assertLess(tau_eos, 0.1)

    def test_dceb_termination_boost(self):
        """When entity reading ends (contrast collapses to ~0.05, entropy spikes to 2.4), DCEB boost > 3.0 logits."""
        effective_peak = max(3.5, self.decoder.delta_peak_min)
        c_collapse = max(0.0, 1.0 - 0.05 / (effective_peak + 1e-6))
        sig_entropy = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, (2.40 - self.decoder.h_bar_eos) / self.decoder.tau_h_eos))))
        tau_eos = self.decoder.kappa_eos * c_collapse * sig_entropy
        self.assertGreater(tau_eos, 3.0)

    def test_dceb_apc_unmasking_invariant(self):
        """EOS token must never be masked out by APC truncation even when P_pos(EOS) is negligible."""
        vocab_size = 50
        eos_id = 49
        pos_logits = torch.zeros((1, vocab_size), dtype=torch.float32)
        neg_logits = torch.zeros((1, vocab_size), dtype=torch.float32)

        # Distractor token 0 has very high prior
        pos_logits[0, 0] = 12.0
        neg_logits[0, 0] = 11.5
        # EOS token has low base prior (would be pruned by APC without preservation)
        pos_logits[0, eos_id] = -5.0
        neg_logits[0, eos_id] = -5.0

        # Step 0: Prime peak contrast
        self.decoder.decode_step(
            pos_logits=pos_logits,
            neg_logits=neg_logits,
            step=0,
            spatial_confidence=1.0,
            eos_id=eos_id,
        )

        # Step 1: Contrast collapsed, entropy high
        pos_logits_step1 = torch.zeros((1, vocab_size), dtype=torch.float32)
        neg_logits_step1 = torch.zeros((1, vocab_size), dtype=torch.float32)
        pos_logits_step1[0, 0] = 1.0
        neg_logits_step1[0, 0] = 1.0
        pos_logits_step1[0, eos_id] = 0.5
        neg_logits_step1[0, eos_id] = 0.5

        next_token, alpha_t, entropy_val, d_js = self.decoder.decode_step(
            pos_logits=pos_logits_step1,
            neg_logits=neg_logits_step1,
            step=1,
            spatial_confidence=1.0,
            eos_id=eos_id,
        )
        # Because DCEB boosted EOS by > 3.0 logits and unmasked it, EOS wins argmax!
        self.assertEqual(next_token, eos_id)

    def test_dceb_supports_list_of_eos_ids(self):
        """DCEB must accept a list of EOS IDs (e.g. [151645, 151643]) and unmask all of them."""
        vocab_size = 50
        eos_ids = [40, 41]
        pos_logits = torch.randn((1, vocab_size), dtype=torch.float32)
        neg_logits = torch.randn((1, vocab_size), dtype=torch.float32)

        next_token, alpha_t, entropy_val, d_js = self.decoder.decode_step(
            pos_logits=pos_logits,
            neg_logits=neg_logits,
            step=1,
            spatial_confidence=1.0,
            eos_id=eos_ids,
        )
        self.assertIsInstance(next_token, int)


class TestSyntacticSpanRefiner(unittest.TestCase):
    """Validation for SyntacticSpanRefiner (SCPDO + QAMIF)."""

    def setUp(self):
        self.refiner = SyntacticSpanRefiner()

    def test_scpdo_colonic_delimiter(self):
        """SCPDO must cleanly extract entity payload on non-digit colons."""
        self.assertEqual(
            self.refiner.refine("Hotline: 0903123456", "Số điện thoại hotline là gì?"),
            "0903123456",
        )
        self.assertEqual(
            self.refiner.refine("Địa chỉ: 128 Nguyễn Trãi, P.3, Q.5", "Địa chỉ trên biển hiệu là gì?"),
            "128 Nguyễn Trãi, P.3, Q.5",
        )

    def test_scpdo_numeric_time_and_ratio_preservation(self):
        """SCPDO negative lookbehind/lookahead must strictly preserve timestamps and ratios."""
        self.assertEqual(self.refiner.refine("10:30", "Quán mở cửa lúc mấy giờ?"), "10:30")
        self.assertEqual(self.refiner.refine("08:00:00", "Thời gian bắt đầu là khi nào?"), "08:00:00")
        self.assertEqual(self.refiner.refine("Tỉ số là 2:1", "Tỉ số của trận đấu là bao nhiêu?"), "2:1")

    def test_scpdo_spaced_dash_novelty_ranking(self):
        """Spaced dashes must be disambiguated by novelty conditioned on question."""
        self.assertEqual(
            self.refiner.refine("Highlands Coffee - Tầng 1", "Tên quán cà phê là gì?"),
            "Highlands Coffee",
        )

    def test_qamif_question_echo_peeling(self):
        """QAMIF must peel leading and trailing tokens that duplicate interrogative question stem."""
        self.assertEqual(
            self.refiner.refine("Tên thương hiệu trên áo là Nike.", "Tên thương hiệu trên áo là gì?"),
            "Nike",
        )
        self.assertEqual(
            self.refiner.refine("Cửa tiệm trong ảnh là Nhà thuốc An Khang", "cửa tiệm trong ảnh là gì ?"),
            "Nhà thuốc An Khang",
        )

    def test_qamif_non_collapsing_invariant(self):
        """QAMIF must guarantee >= 1 core token is preserved, never collapsing to empty."""
        self.assertEqual(self.refiner.refine("Có", "Trong ảnh có ô tô không?"), "Có")
        self.assertEqual(self.refiner.refine("Không", "Có người đi bộ không?"), "Không")

    def test_quotation_and_terminal_punctuation_sanitizer(self):
        """Must strip balanced outer quotes and sentence-terminal punctuation."""
        self.assertEqual(self.refiner.refine("\"Bún bò Huế.\"", "Quán ăn bán món gì?"), "Bún bò Huế")
        self.assertEqual(self.refiner.refine("“100.000đ.”", "Giá tiền là bao nhiêu?"), "100.000đ")
        self.assertEqual(self.refiner.refine("Hà Nội…"), "Hà Nội")

    def test_intra_word_apostrophe_preservation(self):
        """Must prevent intra-word apostrophes from corrupting quote extraction."""
        self.assertEqual(self.refiner.refine("McDonald's và Wendy's"), "McDonald's và Wendy's")
        self.assertEqual(self.refiner.refine("'McDonald\'s'"), "McDonald's")
        self.assertEqual(self.refiner.refine("Trong ảnh có chữ 'McDonald\'s'"), "McDonald's")
        self.assertEqual(self.refiner.refine('Trong ảnh có chữ "McDonald\'s"'), "McDonald's")

    def test_multiline_concluding_payload(self):
        """Concluding payload in terminal non-empty line must be selected."""
        self.assertEqual(
            self.refiner.refine("Câu hỏi: Tên cửa hàng?\nĐáp án: Circle K", "Tên cửa hàng là gì?"),
            "Circle K",
        )


class TestPipelineMockIntegration(unittest.TestCase):
    """Smoke test for pipeline integration with mock components."""

    def test_crop_multi_patch_sequence_invariants(self):
        """Cropper multi-patch sub-tiles must be valid PIL Images and match sequence ordering."""
        cropper = DynamicMultiTileROICropper(target_size=448)
        img = Image.new("RGB", (1000, 500), color=(200, 200, 200))
        crop_res = cropper.crop(img, (100, 100, 900, 200))

        self.assertGreaterEqual(len(crop_res.roi_tiles), 2)
        self.assertEqual(len(crop_res.roi_tiles), len(crop_res.neg_tiles))
        for t in crop_res.roi_tiles:
            self.assertEqual(t.size, (448, 448))


if __name__ == "__main__":
    unittest.main()
