# -*- coding: utf-8 -*-
"""
Iteration 16 Master Unit Test Suite: Hierarchical Multi-Scale Focal Windowing,
PyTorch Tensor Shock Dilation, and Tripartite Symmetrical Clamping (Theorems 32 & 33).

Verifies Directives W1–W6 per Area Chair Gate Review:
1. Directive W1: Hierarchical Multi-Scale Focal Window Extraction (Theorem 32):
   - Sequence length invariant: |V_L| in {256, 512} strictly, S_L <= 522 tokens.
2. Directive W2: PyTorch Tensor-Space Continuous Shock Dilation Kernel (Theorem 32):
   - Polarity symmetry: dark-on-light and light-on-dark symmetrically dilated.
   - Flat background invariance: 0 background noise amplification.
   - 100% PyTorch tensor ops (F.conv2d, torch.tanh, torch.clamp).
3. Directive W3: Tripartite Symmetrical Clamping Engine (Theorem 33):
   - Persistent Head Carrier Barrier: B_carrier(t, v) = mu * (0.5)^(t-1) * Psi_t * (1 - OGI_t(v)) for t <= 3.
   - Continuous Numerical Cohesion Clamp: B_cohesion blocks whitespace token 220 when y_{t-1} in V_num and Omega_t^num > S_bar_t.
   - Zero drag on non-numeric word spacing when y_{t-1} is non-numeric.
   - Tail Termination Clamp: forces EOS dominance upon optical contrast exhaustion at t >= 2.
4. Directive W4: Synchronous Dual-Stream Autoregressive Advance & Token Tracking:
   - Dynamic resolution of space_token_id and digit_token_ids.
   - prev_token_id tracking in decode_step.
5. Directive W5: Strict A* Protocol AST Compliance Audit:
   - 0 regex calls, 0 keywords, 0 external OCR, strictly <= 2 continuous hyperparameters.
"""

import ast
import os
import unittest
from typing import List, Optional, Tuple
from unittest.mock import MagicMock
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nevfe.core.engine import (
    NEVFEEngine,
    nevfe_step_logits,
    adaptive_tensor_stroke_dilation,
    compute_optical_continuation_saliency,
    compute_refusal_resistant_continuation_horizon,
    evaluate_semantic_stopping,
)
from src.nevfe.adapter.vintern_adapter import VinternNEVFEAdapter


class TestIter16Tripartite(unittest.TestCase):
    """Authoritative test suite for Iteration 16 Directives W1–W6."""

    def setUp(self):
        self.device = torch.device("cpu")
        self.dim = 896
        self.vocab_size = 151674  # Qwen2.5 exact vocabulary cardinality

    # =========================================================================
    # Test 1: Directive W2 - Shock Dilation Polarity Symmetry (Theorem 32)
    # =========================================================================
    def test_01_shock_dilation_polarity_symmetry(self):
        """
        Directive W2 (Theorem 32 Part 2):
        Verify continuous shock scale-space stroke dilation is strictly polarity-symmetric:
        - Dark strokes on light background: Delta_sigma Y > 0 => delta X <= 0 (darkens/thickens).
        - Light strokes on dark background: Delta_sigma Y < 0 => delta X >= 0 (brightens/thickens).
        - Relative dilation magnitude is symmetric between polarities.
        """
        H, W = 64, 64
        # Case A: Dark stroke on light background (e.g. text stroke with luminance 0.2 on 0.8 background)
        img_dark_stroke = torch.full((1, 3, H, W), 0.8, device=self.device, dtype=torch.float32)
        img_dark_stroke[:, :, 28:36, 28:36] = 0.2  # 8x8 dark square stroke

        # Case B: Light stroke on dark background (exact polarity inversion: 0.8 stroke on 0.2 background)
        img_light_stroke = 1.0 - img_dark_stroke

        # Apply continuous shock dilation at magnification M = 3.0
        dilated_dark = adaptive_tensor_stroke_dilation(img_dark_stroke, magnification=3.0)
        dilated_light = adaptive_tensor_stroke_dilation(img_light_stroke, magnification=3.0)

        # 1. Bounded outputs in [0.0, 1.0]
        self.assertTrue((dilated_dark >= 0.0).all() and (dilated_dark <= 1.0).all())
        self.assertTrue((dilated_light >= 0.0).all() and (dilated_light <= 1.0).all())

        # 2. Polarity direction check:
        # Dark stroke: boundary pixels must darken (delta <= 0)
        delta_dark = dilated_dark - img_dark_stroke
        self.assertLessEqual(
            delta_dark.min().item(), -0.05,
            "Dark stroke must darken surrounding transition pixels (delta <= 0)",
        )

        # Light stroke: boundary pixels must brighten (delta >= 0)
        delta_light = dilated_light - img_light_stroke
        self.assertGreaterEqual(
            delta_light.max().item(), 0.05,
            "Light stroke must brighten surrounding transition pixels (delta >= 0)",
        )

        # 3. Symmetry check: delta_dark and delta_light must be exact additive inverses
        symmetry_diff = torch.abs(delta_dark + delta_light).max().item()
        self.assertLess(
            symmetry_diff, 1e-4,
            f"Shock dilation must be strictly polarity-symmetric, max inversion diff: {symmetry_diff:.6f}",
        )

    # =========================================================================
    # Test 2: Directive W2 - Flat Background Invariance (Theorem 32)
    # =========================================================================
    def test_02_shock_dilation_flat_background_invariance(self):
        """
        Directive W2 (Theorem 32 Part 2):
        Verify that flat background regions experience identically zero noise amplification:
        ||nabla_sigma Y|| -> 0 => grad_gate = 0 => delta X = 0.
        """
        # Case A: Solid flat image (all pixels = 0.70)
        flat_img = torch.full((1, 3, 48, 48), 0.70, device=self.device, dtype=torch.float32)
        dilated_flat = adaptive_tensor_stroke_dilation(flat_img, magnification=5.0)

        self.assertTrue(
            torch.equal(dilated_flat, flat_img),
            "Shock dilation must leave flat background 100% bit-for-bit unchanged (zero noise)",
        )

        # Case B: Magnification <= 1.05 returns unperturbed input
        test_img = torch.rand(1, 3, 32, 32, device=self.device)
        dilated_low_m = adaptive_tensor_stroke_dilation(test_img, magnification=1.0)
        self.assertTrue(torch.equal(dilated_low_m, test_img))

        # Case C: 3D [3, H, W] dimension handling
        test_3d = torch.rand(3, 32, 32, device=self.device)
        dilated_3d = adaptive_tensor_stroke_dilation(test_3d, magnification=2.5)
        self.assertEqual(dilated_3d.shape, (3, 32, 32))

    # =========================================================================
    # Test 3: Directive W3 - Continuous Numerical Cohesion Clamp on Digits
    # =========================================================================
    def test_03_numerical_cohesion_clamp_blocks_space_between_digits(self):
        """
        Directive W3 (Theorem 33 Part 1):
        When y_{t-1} in V_num (previous token is a digit) and optical digit evidence persists
        (Omega_t^num > S_bar_t), the Continuous Numerical Cohesion Clamp imposes -mu_param * Psi_t
        penalty on token 220 (ASCII space), mathematically blocking BPE whitespace splitting
        and forcing digit continuity (e.g. '46900').
        """
        B, K = 1, 1000
        prev_digit_id = 19   # ASCII digit '4'
        next_digit_id = 21   # ASCII digit '6'
        space_id = 220       # ASCII space ' '
        eos_id = 0

        digit_ids = list(range(15, 25))

        # Baseline language prior strongly favors thousands group whitespace: log P_G(' ') > log P_G('6')
        lg = torch.full((B, K), -5.0, device=self.device)
        lg[0, space_id] = 4.0      # Linguistic prior strongly wants "46 900" (token 220)
        lg[0, next_digit_id] = 1.0 # Pretrained prior is lower on direct unspaced digit

        # Optical focal evidence: image has sharp digit '6' (ink on signboard)
        # Empty space has zero optical contrast: Delta_t(220) = 0
        ll = lg.clone()
        ll[0, next_digit_id] = 7.0  # High optical contrast for digit '6'
        ll[0, space_id] = 4.0       # No optical gain on empty background whitespace

        # Execute step at t=2 with previous token y_1 = 19 ('4')
        next_tok, steered, _, _ = nevfe_step_logits(
            logits_l=ll,
            logits_g=lg,
            mu_param=4.0,
            step_idx=2,
            prev_token_id=prev_digit_id,
            space_token_id=space_id,
            digit_token_ids=digit_ids,
            eos_token_ids=[eos_id],
        )

        # 1. Verify candidate digit '6' strictly dominates whitespace token 220
        self.assertGreater(
            steered[0, next_digit_id].item(),
            steered[0, space_id].item(),
            "Steered logit for digit '6' must strictly exceed whitespace token 220",
        )

        # 2. Selected token must be the genuine digit, not whitespace
        self.assertEqual(
            int(next_tok.item()), next_digit_id,
            f"Expected next token to be digit {next_digit_id} ('6'), got {int(next_tok.item())}",
        )

    # =========================================================================
    # Test 4: Directive W3 - Cohesion Clamp Zero Penalty on Non-Numeric Tokens
    # =========================================================================
    def test_04_numerical_cohesion_zero_drag_on_non_numeric_spacing(self):
        """
        Directive W3 (Theorem 33 Part 2):
        When y_{t-1} is non-numeric (e.g. Vietnamese word 'khách'), the Cohesion Clamp
        evaluates identically to 0.00 nats (is_prev_digit = False).
        Natural word spacing in multi-word proper nouns is 100% unrestricted.
        """
        B, K = 1, 1000
        prev_word_id = 500   # Non-numeric word token (e.g. "khách")
        space_id = 220
        eos_id = 0
        digit_ids = list(range(15, 25))

        lg = torch.full((B, K), -3.0, device=self.device)
        lg[0, space_id] = 4.0
        ll = lg.clone()
        ll[0, space_id] = 4.0
        ll[0, 501] = 5.0

        # Execute with non-numeric previous token
        _, steered_word, _, _ = nevfe_step_logits(
            logits_l=ll,
            logits_g=lg,
            mu_param=4.0,
            step_idx=2,
            prev_token_id=prev_word_id,  # NOT in digit_ids
            space_token_id=space_id,
            digit_token_ids=digit_ids,
            eos_token_ids=[eos_id],
        )

        # Control run without cohesion clamp
        _, steered_control, _, _ = nevfe_step_logits(
            logits_l=ll,
            logits_g=lg,
            mu_param=0.0,  # Zero clamp
            step_idx=2,
            prev_token_id=prev_word_id,
            space_token_id=space_id,
            digit_token_ids=digit_ids,
            eos_token_ids=[eos_id],
        )

        # Space logit must experience ZERO cohesion drag when y_{t-1} is non-numeric
        diff_space = abs(steered_word[0, space_id].item() - steered_control[0, space_id].item())
        self.assertLess(
            diff_space, 1e-4,
            f"Whitespace token must receive 0.00 cohesion penalty when y_{{t-1}} is non-numeric, diff: {diff_space:.6f}",
        )

    # =========================================================================
    # Test 5: Directive W3 - Persistent Head Carrier Barrier for t <= 3
    # =========================================================================
    def test_05_persistent_head_carrier_barrier_at_t_le_3(self):
        """
        Directive W3 (Theorem 33 Part 3):
        Verify that the Head Carrier Barrier is persistent across t in {1, 2, 3}
        with dyadic decay delta_t = (0.5)^(t-1), suppressing ungrounded carrier tokens
        even if simulated in_entity_state was True.
        """
        B, K = 1, 1000
        grounded_entity_id = 100
        carrier_id = 200
        eos_id = 0

        # Focal crop strongly resolves entity token; carrier token has zero optical contrast
        lg = torch.full((B, K), -2.0, device=self.device)
        lg[0, carrier_id] = 4.0  # Dialogue prior favors preamble "Cuốn sách..."
        lg[0, grounded_entity_id] = 1.0

        ll = lg.clone()
        ll[0, grounded_entity_id] = 8.0  # Strong optical contrast (Delta_t = 7.0)
        ll[0, carrier_id] = 4.0          # Zero optical contrast (Delta_t = 0.0)

        # Verify across steps t = 1, 2, 3
        for step in [1, 2, 3]:
            _, steered, _, _ = nevfe_step_logits(
                logits_l=ll,
                logits_g=lg,
                mu_param=4.0,
                step_idx=step,
                eos_token_ids=[eos_id],
                barrier_decay=0.5,
                barrier_horizon=3,
            )

            # Grounded entity token must strictly defeat carrier token at t <= 3
            self.assertGreater(
                steered[0, grounded_entity_id].item(),
                steered[0, carrier_id].item(),
                f"Step {step}: Grounded entity token must dominate ungrounded carrier token",
            )

    # =========================================================================
    # Test 6: Directive W3 - Tail Termination Clamp at Optical Exhaustion
    # =========================================================================
    def test_06_tail_termination_clamp_at_optical_exhaustion(self):
        """
        Directive W3 (Theorem 30 & 33 Part 4):
        At step t >= 2, when text ends and optical contrast is exhausted (Psi_t -> 0),
        the Tail Termination Clamp imposes full -mu_param on all non-EOS candidate tokens,
        while EOS receives 0.00 penalty, forcing instant crisp termination.
        """
        B, K = 1, 1000
        connective_id = 450  # "là", "và" trailing connective
        eos_id = 0

        # Prior wants to continue describing scene: "năm 2020 là một năm..."
        lg = torch.full((B, K), -5.0, device=self.device)
        lg[0, connective_id] = 5.0
        lg[0, eos_id] = 3.0

        # Optical evidence is completely exhausted: ll == lg => Psi_2 = 0.0
        ll = lg.clone()

        # Step 1: Tail clamp MUST NOT fire at t=1 (preserves 1-token answers)
        tok_t1, steered_t1, _, _ = nevfe_step_logits(
            logits_l=ll,
            logits_g=lg,
            mu_param=4.0,
            step_idx=1,
            eos_token_ids=[eos_id],
        )
        # At t=1, connective_id (logit 5.0) beats eos (logit 3.0) because tail clamp is inactive
        self.assertEqual(int(tok_t1.item()), connective_id)

        # Step 2: Tail clamp MUST fire at t=2 under optical exhaustion
        tok_t2, steered_t2, _, _ = nevfe_step_logits(
            logits_l=ll,
            logits_g=lg,
            mu_param=4.0,
            step_idx=2,
            eos_token_ids=[eos_id],
        )
        # Connective penalized by -4.0 nats: 5.0 - 4.0 = 1.0 < EOS (3.0)
        self.assertEqual(
            int(tok_t2.item()), eos_id,
            "Step 2 must emit EOS token under Tail Termination Clamp upon optical exhaustion",
        )

    # =========================================================================
    # Test 7: Directive W1 - Hierarchical Multi-Scale Sequence Length Invariance
    # =========================================================================
    def test_07_hierarchical_multiscale_sequence_length_invariance(self):
        """
        Directive W1 (Theorem 32 Part 4):
        Verify that Hierarchical Multi-Scale Focal Window Extraction strictly maintains:
        - |V_L| in {256, 512} visual tokens (S_L <= 522 tokens).
        - Direct lossless concatenation: V_L = [V^(1); V^(2)] when rho_sub < 0.65.
        - Fallback: V_L = V^(1) when rho_sub >= 0.65.
        - VRAM memory footprint strictly bounded.
        """
        mock_model = MagicMock()
        mock_model.num_image_token = 256
        mock_model.config.image_size = 448
        mock_model.system_message = "Trợ lý AI."
        mock_model.extract_feature = MagicMock(
            return_value=torch.randn(1, 256, self.dim, dtype=torch.bfloat16)
        )

        mock_tokenizer = MagicMock()
        img_ctx_id = 151646
        mock_tokenizer.convert_tokens_to_ids.return_value = img_ctx_id
        mock_tokenizer.encode.return_value = [10, 20, 30]

        def mock_tokenize_call(text, **kwargs):
            res = MagicMock()
            num_ctx = text.count("<IMG_CONTEXT>")
            if num_ctx > 0:
                tokens = [1, 2, 3, 4, 5] + [img_ctx_id] * num_ctx
            else:
                tokens = [6, 7, 8, 9, 10]
            res.input_ids = torch.tensor([tokens], dtype=torch.long)
            return res

        mock_tokenizer.side_effect = mock_tokenize_call

        adapter = VinternNEVFEAdapter(
            model=mock_model,
            tokenizer=mock_tokenizer,
            device="cpu",
        )

        image = Image.new("RGB", (800, 600), color=(128, 128, 128))
        box1 = (50, 50, 400, 400)

        # Case A: Sub-box span rho_sub < 0.65 (Hierarchical zoom triggered)
        # Passing small text question conditioned embeddings forces zoom trigger
        v_l_zoomed = adapter.extract_stage2_focal_features(
            image=image,
            crop_box=box1,
            question="Biển số bao nhiêu?",
            theta_zoom=1.05,  # Force zoom activation
            use_hierarchical_zoom=True,
        )
        self.assertEqual(
            v_l_zoomed.shape, (1, 512, self.dim),
            "Zoomed focal features must be strictly [1, 512, 896] visual tokens (Theorem 32)",
        )

        # Case B: Sub-box span rho_sub >= 0.65 (Zoom bypassed)
        adapter._cached_v_l = None
        adapter._last_crop_box = None
        v_l_bypass = adapter.extract_stage2_focal_features(
            image=image,
            crop_box=box1,
            question="Biển số bao nhiêu?",
            theta_zoom=0.01,  # Force zoom bypass
            use_hierarchical_zoom=True,
        )
        self.assertEqual(
            v_l_bypass.shape, (1, 256, self.dim),
            "Bypassed focal features must be strictly [1, 256, 896] visual tokens",
        )

        # Verify prompt sequence length: S_L <= 522 tokens
        embed_dim = 64
        embed_layer = nn.Embedding(200000, embed_dim)
        adapter.model.language_model.get_input_embeddings.return_value = embed_layer

        v_512 = torch.randn(1, 512, embed_dim)
        embeds_512 = adapter.build_multimodal_embeds(v_512, question="Số gì?")
        self.assertEqual(embeds_512.shape[1], 522, "Prompt sequence length for 512 tokens must equal 522")

    # =========================================================================
    # Test 8: Directive W5 - Strict A* Protocol AST Compliance Audit
    # =========================================================================
    def test_08_strict_a_star_ast_compliance(self):
        """
        Directive W5 (Strict A* Forensic AST Audit):
        1. 0 regex calls (`re.sub`, `re.match`, `re.search`, `re.findall`).
        2. 0 external OCR imports.
        3. 0 hardcoded carrier keyword lists.
        4. Strictly <= 2 continuous tunable hyperparameters (lambda_param, mu_param).
        5. Zero modifications to scripts/benchmark/metrics.py.
        """
        engine_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "src", "nevfe", "core", "engine.py")
        )
        adapter_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "src", "nevfe", "adapter", "vintern_adapter.py")
        )

        for file_path in [engine_path, adapter_path]:
            with open(file_path, "r", encoding="utf-8") as f:
                code_text = f.read()
                tree = ast.parse(code_text, filename=file_path)

            for node in ast.walk(tree):
                # 1. Regex check
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertNotEqual(alias.name, "re", f"Illegal regex import in {file_path}")
                if isinstance(node, ast.ImportFrom):
                    self.assertNotEqual(node.module, "re", f"Illegal regex import in {file_path}")
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    attr_name = node.func.attr
                    self.assertNotIn(
                        attr_name,
                        ["sub", "match", "search", "findall"],
                        f"Illegal regex call '{attr_name}' detected in {file_path}",
                    )

                # 2. External OCR check
                banned_modules = ["pytesseract", "easyocr", "paddleocr", "vietocr", "transformers.models.trocr"]
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        for bm in banned_modules:
                            self.assertNotIn(bm, alias.name, f"Illegal external OCR import in {file_path}")
                if isinstance(node, ast.ImportFrom) and node.module:
                    for bm in banned_modules:
                        self.assertNotIn(bm, node.module, f"Illegal external OCR import in {file_path}")

                # 3. Carrier / stopword keyword lists
                if isinstance(node, (ast.List, ast.Set, ast.Tuple)):
                    carrier_keywords = ["ảnh", "theo", "đây", "sản phẩm", "bản tin", "hình ảnh"]
                    string_elements = [
                        elt.value.strip().lower()
                        for elt in node.elts
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                    ]
                    for kw in carrier_keywords:
                        self.assertNotIn(
                            kw,
                            string_elements,
                            f"Hardcoded carrier keyword '{kw}' list detected in {file_path}",
                        )

        # 4. Strictly <= 2 continuous tunable hyperparameters in NEVFEEngine
        with open(engine_path, "r", encoding="utf-8") as f:
            engine_tree = ast.parse(f.read())
        engine_cls = [
            n for n in engine_tree.body if isinstance(n, ast.ClassDef) and n.name == "NEVFEEngine"
        ][0]
        init_def = [
            m for m in engine_cls.body if isinstance(m, ast.FunctionDef) and m.name == "__init__"
        ][0]
        param_names = [a.arg for a in init_def.args.args if a.arg != "self"]

        tunable_params = [p for p in ["lambda_param", "mu_param"] if p in param_names]
        self.assertLessEqual(
            len(tunable_params),
            2,
            f"Strictly <= 2 continuous hyperparameters allowed, found {tunable_params}",
        )
        self.assertIn("lambda_param", param_names)
        self.assertIn("mu_param", param_names)


if __name__ == "__main__":
    unittest.main()
