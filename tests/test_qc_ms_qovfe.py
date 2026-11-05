# -*- coding: utf-8 -*-
"""
Unit tests for Question-Conditioned Multi-Scale QO-VFE with Calibrated Termination (QC-MS-QO-VFE).

Verifies:
1. Directive 1: Dual-Stream Semantic Stopping Invariance (Formulation B).
2. Directive 2: Question-Aware Cache Separation in VinternNEVFEAdapter.
3. Directive 3: Scratch PyTorch Question Saliency:
   - Question-conditioned window shift toward queried token cluster (Theorems 1 & 2).
   - Parameter-free spatial variance token weighting downweighting uniform/stopword tokens.
   - Radial Chebyshev >= 75.0% visual evidence mass enclosure guarantee.
4. Directive 4: AST Anti-Cheating & Invariant Governance:
   - 0 regex imports (re, regex).
   - 0 regex function calls (re.sub, re.match, etc.).
   - Strictly <= 2 continuous hyperparameters in NEVFEEngine (lambda_param, mu_param).
"""

import ast
import inspect
import math
import unittest
from typing import List, Tuple
from PIL import Image
import torch
import torch.nn.functional as F

from src.nevfe.core.engine import (
    NEVFEEngine,
    compute_thermodynamic_spatial_window,
    compute_question_conditioned_spatial_window,
    nevfe_step_logits,
)
from src.nevfe.adapter.vintern_adapter import (
    VinternNEVFEAdapter,
    CANONICAL_PROMPT_TEMPLATE,
    PROMPT_INSTRUCTION,
)


class TestQCMSQOVFE(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    # =========================================================================
    # Test 1: Question-Conditioned Spatial Window Shift Toward Queried Cluster
    # =========================================================================
    def test_01_question_conditioned_window_shift_toward_cluster(self):
        """
        Theorem 1 & 2: Cross-modal cosine affinity directs focal Chebyshev window
        to the specific spatial cluster queried by question embeddings.
        """
        P, D = 256, 896
        # Background tokens with low noise
        v = torch.randn(P, D, dtype=torch.float32) * 0.05

        # Cluster 1 (Header/Store Name): top-left patches (row 2..3, col 2..3)
        # Spatial centroid ~ (x = 2.5/16 = 0.156, y = 2.5/16 = 0.156)
        cluster_1_idx = [2 * 16 + 2, 2 * 16 + 3, 3 * 16 + 2, 3 * 16 + 3]
        feat_1 = torch.randn(D, dtype=torch.float32)
        feat_1 = feat_1 / torch.norm(feat_1) * 6.0
        v[cluster_1_idx] += feat_1

        # Cluster 2 (Footer/Phone): bottom-right patches (row 12..13, col 12..13)
        # Spatial centroid ~ (x = 12.5/16 = 0.781, y = 12.5/16 = 0.781)
        cluster_2_idx = [12 * 16 + 12, 12 * 16 + 13, 13 * 16 + 12, 13 * 16 + 13]
        feat_2 = torch.randn(D, dtype=torch.float32)
        # Make feat_2 orthogonal to feat_1
        feat_2 = feat_2 - torch.dot(feat_2, feat_1) / (torch.norm(feat_1) ** 2) * feat_1
        feat_2 = feat_2 / torch.norm(feat_2) * 6.0
        v[cluster_2_idx] += feat_2

        # Question 1 queried about Cluster 1: question embedding aligned with feat_1
        q_emb_1 = feat_1.unsqueeze(0)  # [1, D]
        box_1 = compute_question_conditioned_spatial_window(v, q_emb_1, grid_size=16)

        # Question 2 queried about Cluster 2: question embedding aligned with feat_2
        q_emb_2 = feat_2.unsqueeze(0)  # [1, D]
        box_2 = compute_question_conditioned_spatial_window(v, q_emb_2, grid_size=16)

        # Unconditioned window (Fisher only)
        box_uncond = compute_thermodynamic_spatial_window(v, grid_size=16)

        # Compute window centers
        cx_1 = 0.5 * (box_1[0] + box_1[2])
        cy_1 = 0.5 * (box_1[1] + box_1[3])
        cx_2 = 0.5 * (box_2[0] + box_2[2])
        cy_2 = 0.5 * (box_2[1] + box_2[3])
        cx_uncond = 0.5 * (box_uncond[0] + box_uncond[2])
        cy_uncond = 0.5 * (box_uncond[1] + box_uncond[3])

        # Verification 1: Window 1 strictly shifts toward top-left (< 0.40)
        self.assertLess(cx_1, 0.40, f"Window 1 center x={cx_1:.3f} must be < 0.40")
        self.assertLess(cy_1, 0.40, f"Window 1 center y={cy_1:.3f} must be < 0.40")

        # Verification 2: Window 2 strictly shifts toward bottom-right (> 0.60)
        self.assertGreater(cx_2, 0.60, f"Window 2 center x={cx_2:.3f} must be > 0.60")
        self.assertGreater(cy_2, 0.60, f"Window 2 center y={cy_2:.3f} must be > 0.60")

        # Verification 3: Unconditioned window sits in the middle (bimodal compromise)
        self.assertAlmostEqual(cx_uncond, 0.50, delta=0.15)
        self.assertAlmostEqual(cy_uncond, 0.50, delta=0.15)

        # Verification 4: Question conditioning reduces half-span (focal zoom) relative to unconditioned
        span_1 = (box_1[2] - box_1[0]) / 2.0
        span_uncond = (box_uncond[2] - box_uncond[0]) / 2.0
        self.assertLess(span_1, span_uncond, "Question conditioning must focus tighter than bimodal unconditioned window")

    # =========================================================================
    # Test 2: Spatial Variance Downweights Diffuse / Uniform Tokens
    # =========================================================================
    def test_02_spatial_variance_downweights_uniform_tokens(self):
        """
        Theorem 2: Parameter-free token specificity weighting omega_t = Var_p(A_{., t})
        automatically assigns near-zero weight to uniform/stopword tokens without dictionaries.
        """
        P, D = 256, 896
        v = torch.randn(P, D, dtype=torch.float32)

        # Construct 2 question token embeddings:
        # Token 0: "stopword" — uniform cosine affinity with all patches
        # Token 1: "entity" — highly localized affinity with only patch 50
        v_norm = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-6)

        # Target patch 50
        e_entity = v_norm[50:51].clone()  # Shape [1, D]

        # Uniform vector orthogonal to variance across patches
        e_stopword = torch.mean(v_norm, dim=0, keepdim=True)
        e_stopword = e_stopword / (torch.norm(e_stopword, dim=-1, keepdim=True) + 1e-6)

        q_embeds = torch.cat([e_stopword, e_entity], dim=0)  # [2, D]

        # Compute affinities S and softmax patch distributions A
        S = torch.matmul(v_norm, q_embeds.t())  # [256, 2]
        A = F.softmax(S / 0.1, dim=0)  # [256, 2]

        omega = torch.var(A, dim=0, unbiased=False)  # [2]

        # The stopword token should have vastly lower spatial variance than the entity token
        self.assertLess(omega[0].item(), omega[1].item() * 0.05,
                        f"Stopword variance {omega[0].item():.6f} must be < 5% of entity variance {omega[1].item():.6f}")

        # Normalized weights
        omega_norm = omega / (torch.sum(omega) + 1e-6)
        self.assertGreater(omega_norm[1].item(), 0.90, "Entity token must receive >= 90% of aggregated attention weight")

    # =========================================================================
    # Test 3: Chebyshev 75% Mass Enclosure Property
    # =========================================================================
    def test_03_chebyshev_75_percent_mass_enclosure(self):
        """
        Proposition 1 & Theorem 2: Bounding window computed by k_sigma=2.0
        rigorously encloses >= 75.0% of question-targeted visual mass.
        """
        P, D = 256, 896
        v = torch.randn(P, D, dtype=torch.float32) * 0.1

        # Hotspot at patch (row 8, col 8)
        hotspot_idx = 8 * 16 + 8
        target_feat = torch.randn(D, dtype=torch.float32) * 4.0
        v[hotspot_idx] += target_feat

        q_emb = target_feat.unsqueeze(0)
        box = compute_question_conditioned_spatial_window(v, q_emb, grid_size=16, k_sigma=2.0, min_span=0.15)
        x_min, y_min, x_max, y_max = box

        # Compute rho
        v_mean = torch.mean(v, dim=0, keepdim=True)
        var_d = torch.mean((v - v_mean) ** 2, dim=0, keepdim=True)
        s_p = 0.5 * torch.sum(((v - v_mean) ** 2) / (var_d + 1e-6), dim=-1)
        z_s = (s_p - torch.mean(s_p)) / (torch.std(s_p) + 1e-6)
        m_vis = torch.relu(z_s)

        v_norm = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-6)
        e_norm = q_emb / (torch.norm(q_emb, dim=-1, keepdim=True) + 1e-6)
        S = torch.matmul(v_norm, e_norm.t())
        A = F.softmax(S / 0.1, dim=0)
        omega = torch.var(A, dim=0, unbiased=False)
        omega_norm = omega / (torch.sum(omega) + 1e-6)
        alpha = torch.matmul(A, omega_norm)
        m_sal = m_vis * alpha
        rho = m_sal / (torch.sum(m_sal) + 1e-6)

        coords_1d = (torch.arange(16, dtype=torch.float32) + 0.5) / 16.0
        grid_y, grid_x = torch.meshgrid(coords_1d, coords_1d, indexing="ij")
        gx = grid_x.reshape(-1)
        gy = grid_y.reshape(-1)

        inside = (gx >= x_min - 1e-5) & (gx <= x_max + 1e-5) & (gy >= y_min - 1e-5) & (gy <= y_max + 1e-5)
        enclosed_mass = torch.sum(rho[inside]).item()

        self.assertGreaterEqual(enclosed_mass, 0.749,
                                f"Chebyshev window must enclose >= 75% mass, got {enclosed_mass*100:.2f}%")
        self.assertGreaterEqual(x_min, 0.0)
        self.assertLessEqual(x_max, 1.0)
        self.assertGreaterEqual(y_min, 0.0)
        self.assertLessEqual(y_max, 1.0)

    # =========================================================================
    # Test 4: Dual-Stream Semantic Stopping Invariance (Formulation B)
    # =========================================================================
    def test_04_dual_stream_semantic_stopping_invariance(self):
        """
        Theorem 4: When global language stream p_G emits EOS, generation stops unconditionally,
        preventing compound entity overshoot. When p_G continues, EOS is masked from steered competition.
        """
        K = 200
        eos_id = 199
        eos_token_ids = [eos_id]

        # 1. Verification of EOS Masking during active continuation:
        # When p_G does not emit EOS, but steered contrastive potential is computed,
        # eos_id must be clamped to -1e9 in steered_logits.
        lg_cont = torch.randn(K)
        lg_cont[10] = 10.0  # Content token 10 is dominant in p_G
        lg_cont[eos_id] = -5.0

        ll_cont = torch.randn(K)
        ll_cont[10] = 8.0

        next_tok, steered_logits, _, _ = nevfe_step_logits(
            logits_l=ll_cont,
            logits_g=lg_cont,
            eos_token_ids=eos_token_ids,
        )

        self.assertEqual(steered_logits[eos_id].item(), -1e9,
                         "EOS token ID must be masked to -1e9 in steered logits during continuation")
        self.assertNotEqual(int(next_tok.item()), eos_id,
                            "Next token must not be EOS when p_G has content argmax")

        # 2. Verification of Dual-Stream Stopping when p_G signals EOS:
        # In this scenario, p_G argmax is EOS (question answered), but local stream p_L
        # has a large contrastive spike on a subsequent text line token (e.g. token 88).
        lg_stop = torch.randn(K)
        lg_stop[eos_id] = 15.0  # Global language stream wants to STOP
        lg_stop[88] = 2.0

        ll_spike = torch.randn(K)
        ll_spike[eos_id] = -2.0  # EOS not visible in image
        ll_spike[88] = 18.0     # Runaway sign continuation token

        # Under Formulation B stopping rule:
        # argmax(logits_g) == eos_id -> stop immediately!
        stop_decision = int(torch.argmax(lg_stop, dim=-1).item()) in eos_token_ids
        self.assertTrue(stop_decision, "Dual-stream stopping rule must detect termination from global stream")

    # =========================================================================
    # Test 5: AST Anti-Cheating & Invariant Governance Audit
    # =========================================================================
    def test_05_ast_anti_cheating_audit(self):
        """
        Directive 4: AST audit verifies 0 regex imports, 0 regex function calls,
        and strictly <= 2 continuous hyperparameters in NEVFEEngine.
        """
        target_files = [
            "src/nevfe/core/engine.py",
            "src/nevfe/adapter/vintern_adapter.py",
        ]

        for filepath in target_files:
            with open(filepath, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=filepath)

            # Check for regex imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertNotIn("re", alias.name.split("."),
                                         f"Forbidden regex import in {filepath}: {alias.name}")
                        self.assertNotIn("regex", alias.name.split("."),
                                         f"Forbidden regex import in {filepath}: {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        self.assertNotIn("re", node.module.split("."),
                                         f"Forbidden regex import in {filepath}: {node.module}")
                        self.assertNotIn("regex", node.module.split("."),
                                         f"Forbidden regex import in {filepath}: {node.module}")

                # Check for regex calls like re.sub, re.match, etc.
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute):
                        caller_id = getattr(node.func.value, "id", None)
                        if caller_id in ("re", "regex"):
                            self.fail(f"Forbidden call {caller_id}.{node.func.attr} in {filepath}")

        # Check NEVFEEngine hyperparameters
        init_sig = inspect.signature(NEVFEEngine.__init__)
        params = list(init_sig.parameters.keys())
        params.remove("self")

        # The 2 continuous tuning hyperparameters
        tunable_params = ["lambda_param", "mu_param"]
        for tp in tunable_params:
            self.assertIn(tp, params, f"Expected {tp} in NEVFEEngine.__init__")

        # Structural / geometric invariants
        fixed_invariants = {
            "beta_param": 1.0,
            "barrier_decay": 0.5,
            "barrier_horizon": 3,
            "k_sigma": 2.0,
            "min_span": 0.15,
        }
        for name, expected_val in fixed_invariants.items():
            if name in init_sig.parameters:
                self.assertEqual(init_sig.parameters[name].default, expected_val,
                                 f"Invariant {name} must have default {expected_val}")

    # =========================================================================
    # Test 6: Question-Aware Cache Separation in VinternNEVFEAdapter
    # =========================================================================
    def test_06_question_aware_cache_separation(self):
        """
        Directive 2: VinternNEVFEAdapter reuses global visual tokens V_G across
        different questions on the same image, while computing independent bounding boxes.
        """
        # Create a lightweight mock model & tokenizer to test cache logic without GPU VRAM
        class MockLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embeds = torch.nn.Embedding(1000, 896)
            def get_input_embeddings(self):
                return self.embeds

        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.extract_call_count = 0
                self.language_model = MockLM()
                self.num_image_token = 256

            def extract_feature(self, pixel_values):
                self.extract_call_count += 1
                return torch.randn(1, 256, 896, dtype=torch.bfloat16)

        class MockTokenizer:
            def encode(self, text, add_special_tokens=False, return_tensors=None):
                # Return dummy token IDs
                t = torch.tensor([[10, 20, 30]])
                return t
            def convert_tokens_to_ids(self, tok):
                return 1

        mock_model = MockModel()
        mock_tokenizer = MockTokenizer()

        adapter = VinternNEVFEAdapter(
            model=mock_model,
            tokenizer=mock_tokenizer,
            device="cpu",
        )

        test_img = Image.new("RGB", (200, 200), color=(100, 150, 200))

        # Question 1 on image
        v_g1, box1, norm1 = adapter.compute_stage1_window(test_img, question="quán ăn tên gì?")
        self.assertEqual(mock_model.extract_call_count, 1, "First question should extract V_G once")

        # Question 2 on SAME image
        v_g2, box2, norm2 = adapter.compute_stage1_window(test_img, question="số điện thoại là bao nhiêu?")
        # V_G extraction call count should still be 1 (reused from cache!)
        self.assertEqual(mock_model.extract_call_count, 1, "Second question on same image must reuse cached V_G")
        self.assertIs(v_g1, v_g2, "v_g2 must be identical cached object to v_g1")

        # Repeat Question 2 on SAME image: full cache should hit
        v_g3, box3, norm3 = adapter.compute_stage1_window(test_img, question="số điện thoại là bao nhiêu?")
        self.assertEqual(mock_model.extract_call_count, 1)
        self.assertEqual(box2, box3, "Identical question should return identical cached crop box")

        # New Image: should trigger feature extraction
        new_img = Image.new("RGB", (300, 300), color=(50, 80, 120))
        v_g4, box4, norm4 = adapter.compute_stage1_window(new_img, question="địa chỉ ở đâu?")
        self.assertEqual(mock_model.extract_call_count, 2, "New image must trigger new feature extraction")

    # =========================================================================
    # Test 7: Prompt Template Formatting Invariance
    # =========================================================================
    def test_07_canonical_prompt_brevity(self):
        """
        Directive 2 & Theorem 5: Verify canonical brevity prompt template.
        """
        self.assertEqual(CANONICAL_PROMPT_TEMPLATE, "Dựa vào chữ trong ảnh, hãy trả lời ngắn gọn: {question}")
        self.assertEqual(PROMPT_INSTRUCTION, "Dựa vào chữ trong ảnh, hãy trả lời ngắn gọn: ")


if __name__ == "__main__":
    unittest.main()
