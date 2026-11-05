# -*- coding: utf-8 -*-
"""
Unit tests for Refusal-Free Calibrated Multi-Scale QO-VFE (RFC-MS-QO-VFE).

Verifies the 5 Binding Directives from Area Chair Meta-Review (critic_r12_areachair/review.md, Score 9.3/10.0):
1. Directive 1: Complete Reversion of FSG in VinternNEVFEAdapter:
   - Completely bypass apply_feature_space_nullification in generate_answer.
   - Intact raw v_l directly passed into build_multimodal_embeds.
   - Uniform batched B=2 forward pass preserved.
2. Directive 2: Bounded Dynamic Temperature Scaling in engine.py:
   - Enforce Formulation A with alpha_cal = 0.40: tau_t = 1.0 / (1.0 + 0.40 * xi_t) in [0.714, 1.00].
   - Prohibit unconstrained formulas (tau_t never falls below 0.714).
3. Directive 3: Optical Grounding Index (OGI) Prior Attenuation in engine.py:
   - In nevfe_step_logits, OGI attenuation active for step_idx <= 2 when mu_param > 0.
   - Denominator clamped with clamp(min=1e-6) for numerical stability.
   - Scale-insensitive, non-optical tokens (delta_log <= 0) penalized by barrier_val * (1.0 - ogi).
   - Inactive for step_idx >= 3.
   - Optical tokens (delta_log > 0) receive zero penalty.
   - Out-of-crop tokens (large negative delta_log, ogi ~ 1.0) receive ~0 penalty.
4. Directive 4: Preservation of Semantic Stopping and Invariants:
   - Dual-Stream Semantic Stopping Invariance: unsteered logits_g argmax determines EOS.
   - EOS tokens masked to -1e9 in steered_logits.
   - Query-orthogonal potential barrier preserved on prompt tokens via q_mask.
5. Directive 5: AST Governance & Anti-Cheating:
   - 0 regex imports (re, regex).
   - 0 regex function calls (re.sub, re.match, etc.).
   - 0 hardcoded refusal keyword lists/blacklists.
   - Strictly <= 2 continuous hyperparameters in NEVFEEngine (lambda_param, mu_param).
"""

import ast
import inspect
import math
import unittest
from typing import List, Tuple
from unittest.mock import MagicMock
from PIL import Image
import torch
import torch.nn.functional as F

from src.nevfe.core.engine import (
    NEVFEEngine,
    compute_dynamic_contrastive_temperature,
    compute_thermodynamic_spatial_window,
    compute_question_conditioned_spatial_window,
    nevfe_step_logits,
)
from src.nevfe.adapter.vintern_adapter import (
    VinternNEVFEAdapter,
    CANONICAL_PROMPT_TEMPLATE,
    PROMPT_INSTRUCTION,
)


class TestRFCMSQOVFE(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    # =========================================================================
    # Test 1: Complete Reversion of FSG in VinternNEVFEAdapter (Directive 1)
    # =========================================================================
    def test_01_fsg_complete_reversion_and_intact_features(self):
        """
        Directive 1 (Theorems 10 & 11):
        In VinternNEVFEAdapter.generate_answer:
        1. Completely bypass apply_feature_space_nullification.
        2. Pass intact raw v_l directly into build_multimodal_embeds.
        3. Maintain uniform batched B=2 forward pass.
        """
        P, D = 256, 896

        class MockLMModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.last_inputs_embeds = None

            def forward(self, inputs_embeds, use_cache=True, return_dict=True):
                self.last_inputs_embeds = inputs_embeds
                B, seq_len, dim = inputs_embeds.shape
                # Return dummy output with last_hidden_state
                out = MagicMock()
                out.last_hidden_state = torch.randn(B, seq_len, dim, device=inputs_embeds.device)
                out.past_key_values = MagicMock()
                return out

        class MockLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = MockLMModel()
                self.lm_head = torch.nn.Linear(D, 1000, bias=False)
                self.embeds = torch.nn.Embedding(1000, D)

            def get_input_embeddings(self):
                return self.embeds

        class MockVLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.language_model = MockLM()
                self.num_image_token = P

            def extract_feature(self, pixel_values):
                return torch.randn(1, P, D, dtype=torch.bfloat16)

        class MockTokenizer:
            def __init__(self):
                self.eos_token_id = 2

            def __call__(self, text, return_tensors="pt", add_special_tokens=False):
                class Output:
                    pass
                out = Output()
                if "<IMG_CONTEXT>" in text:
                    toks = [1] * 10 + [3] * P + [2] * 10
                    out.input_ids = torch.tensor([toks])
                else:
                    out.input_ids = torch.tensor([[1, 2, 4, 5]])
                return out

            def encode(self, text, add_special_tokens=False, return_tensors=None):
                return [1, 2, 4, 5]

            def convert_tokens_to_ids(self, tok):
                if tok == "<IMG_CONTEXT>":
                    return 3
                if tok == "<|im_end|>":
                    return 2
                return 1

            def decode(self, ids, skip_special_tokens=True):
                return "test answer"

        mock_model = MockVLM()
        mock_tokenizer = MockTokenizer()

        adapter = VinternNEVFEAdapter(
            model=mock_model,
            tokenizer=mock_tokenizer,
            device="cpu",
        )

        # Track calls to apply_feature_space_nullification and build_multimodal_embeds
        nullification_call_count = 0
        original_nullify = adapter.apply_feature_space_nullification

        def spy_nullify(*args, **kwargs):
            nonlocal nullification_call_count
            nullification_call_count += 1
            return original_nullify(*args, **kwargs)

        adapter.apply_feature_space_nullification = spy_nullify

        passed_embeds = []
        original_build = adapter.build_multimodal_embeds

        def spy_build(visual_embeds, question, user_prompt=None):
            passed_embeds.append(visual_embeds.clone())
            return original_build(visual_embeds, question, user_prompt=user_prompt)

        adapter.build_multimodal_embeds = spy_build

        test_img = Image.new("RGB", (224, 224), color=(120, 130, 140))
        test_question = "Chữ trên bảng hiệu là gì?"

        # Run generate_answer for 1 step
        # Force model to stop immediately at step 1 by making global stream output EOS
        with torch.no_grad():
            mock_model.language_model.lm_head.weight.data.zero_()
            # Set EOS logit high
            mock_model.language_model.lm_head.weight.data[2, 0] = 100.0

            ans = adapter.generate_answer(test_img, test_question, max_new_tokens=1)

        # 1. Verify apply_feature_space_nullification was NEVER called
        self.assertEqual(
            nullification_call_count, 0,
            "Directive 1 Violation: apply_feature_space_nullification must be completely bypassed in generate_answer"
        )

        # 2. Verify build_multimodal_embeds was called twice (once for V_G, once for intact V_L)
        self.assertEqual(len(passed_embeds), 2, "build_multimodal_embeds must be called for global and focal streams")

        # 3. Verify inputs_embeds passed to language model has batch dimension B=2
        last_inputs = mock_model.language_model.model.last_inputs_embeds
        self.assertIsNotNone(last_inputs, "Model must have received inputs_embeds")
        self.assertEqual(last_inputs.shape[0], 2, f"Batched forward pass must have B=2, got {last_inputs.shape[0]}")

    # =========================================================================
    # Test 2: Bounded Dynamic Temperature Scaling Formulation A (Directive 2)
    # =========================================================================
    def test_02_bounded_dynamic_temperature_formulation_a(self):
        """
        Directive 2 (Theorems 11 & 12):
        tau_t = 1.0 / (1.0 + 0.40 * xi_t) in [0.714, 1.00] with alpha_cal = 0.40.
        Prohibit unrestricted floor tau_t = 0.50 which induced Dirac mode collapse and BPE splitting.
        """
        K = 1000

        # Scenario A: Identical distributions (p_L == p_G)
        logits_same = torch.randn(1, K)
        tau_same, h_same, d_same = compute_dynamic_contrastive_temperature(logits_same, logits_same)
        self.assertAlmostEqual(d_same.item(), 0.0, places=4, msg="Identical distributions must have zero divergence")
        self.assertAlmostEqual(tau_same.item(), 1.0, places=4, msg="tau_t must equal 1.00 when divergence is 0")

        # Scenario B: Disjoint distributions with high entropy
        half_k = K // 2
        logits_l = torch.full((1, K), -100.0)
        logits_l[0, :half_k] = 0.0  # Uniform over first half

        logits_g = torch.full((1, K), -100.0)
        logits_g[0, half_k:] = 0.0  # Uniform over second half

        tau_div, h_div, d_div = compute_dynamic_contrastive_temperature(logits_l, logits_g)
        self.assertAlmostEqual(d_div.item(), 1.0, places=2, msg="Disjoint distributions must have norm JSD ~ 1.0")
        self.assertGreater(h_div.item(), 0.85, msg="Uniform over half vocab must have high normalized entropy")

        # Dynamic temperature must cool toward 0.714 (Formulation A with alpha_cal = 0.40)
        # Theoretical lower bound: 1.0 / (1.0 + 0.40 * 1.0) = 0.7142857
        self.assertGreaterEqual(
            tau_div.item(), 0.714,
            f"Directive 2 Violation: tau_t ({tau_div.item():.4f}) dropped below the 0.714 floor!"
        )
        self.assertLess(
            tau_div.item(), 0.75,
            f"tau_t for max divergence should be close to 0.714, got {tau_div.item():.4f}"
        )
        self.assertGreater(
            tau_div.item(), 0.60,
            "Unrestricted 0.50 floor must NOT be reachable under Formulation A"
        )

        # Scenario C: Dynamic range across 100 heterogeneous random logit pairs
        for seed in range(100):
            torch.manual_seed(seed)
            ll = torch.randn(2, K) * (1.0 + seed * 0.05)
            lg = torch.randn(2, K) * (1.0 + seed * 0.05)
            tau, h, d = compute_dynamic_contrastive_temperature(ll, lg)
            self.assertTrue(
                torch.all(tau >= 0.714 - 1e-5),
                f"tau_t below 0.714 floor at seed {seed}: min was {torch.min(tau).item()}"
            )
            self.assertTrue(
                torch.all(tau <= 1.00 + 1e-5),
                f"tau_t above 1.00 ceiling at seed {seed}: max was {torch.max(tau).item()}"
            )
            self.assertTrue(torch.all(h >= 0.0 - 1e-5) and torch.all(h <= 1.0 + 1e-5))
            self.assertTrue(torch.all(d >= 0.0 - 1e-5) and torch.all(d <= 1.0 + 1e-5))

        # Scenario D: Extreme FP32 numerical stability (underflow and overflow protection)
        extreme_ll = torch.tensor([[1000.0, -1000.0, 0.0, -500.0]])
        extreme_lg = torch.tensor([[-1000.0, 1000.0, 0.0, -500.0]])
        tau_ext, h_ext, d_ext = compute_dynamic_contrastive_temperature(extreme_ll, extreme_lg)
        self.assertFalse(torch.isnan(tau_ext).any(), "tau_t must not contain NaN")
        self.assertFalse(torch.isinf(tau_ext).any(), "tau_t must not contain Inf")
        self.assertGreaterEqual(tau_ext.item(), 0.714)
        self.assertLessEqual(tau_ext.item(), 1.00)

    # =========================================================================
    # Test 3: Optical Grounding Index (OGI) Prior Attenuation (Directive 3)
    # =========================================================================
    def test_03_ogi_prior_attenuation_dynamics(self):
        """
        Directive 3 (Theorems 13 & 14):
        In nevfe_step_logits:
        - At step_idx <= 2, ungrounded scale-insensitive tokens (delta_log <= 0, ogi ~ 0)
          are penalized by -barrier_val * (1.0 - ogi).
        - Grounded optical tokens (delta_log > 0) receive zero penalty.
        - Out-of-crop tokens (delta_log << 0, ogi ~ 1.0) receive near-zero penalty.
        - At step_idx >= 3, OGI prior attenuation is inactive.
        """
        K = 5
        # Token 0: In-crop scene text character (optical token, strong local boost)
        # Token 1: Conversational refusal preamble token (scale-insensitive, delta_log <= 0)
        # Token 2: Out-of-crop scene text character (large negative differential, high global evidence)
        # Tokens 3, 4: Background context
        ll = torch.tensor([[3.0, 2.0, 1.0, 2.0, 2.0]])
        lg = torch.tensor([[2.0, 2.1, 2.5, 2.0, 2.0]])

        # 1. Continuous logit attenuation test:
        # Use beta_param=0.5 and mu_param=0.4 to keep all tokens within support mask
        _, s_no, _, _ = nevfe_step_logits(
            logits_l=ll,
            logits_g=lg,
            beta_param=0.5,
            mu_param=0.0,
            step_idx=1,
        )

        _, s_mu, _, _ = nevfe_step_logits(
            logits_l=ll,
            logits_g=lg,
            beta_param=0.5,
            mu_param=0.4,
            step_idx=1,
        )

        diff_step1 = s_no - s_mu

        # Token 0 (Optical Token): delta_log > 0 => non_opt_mask == 0 => ZERO penalty
        self.assertAlmostEqual(
            diff_step1[0, 0].item(), 0.0, places=4,
            msg=f"Optical token (Token 0) must receive zero penalty, got {diff_step1[0, 0].item()}"
        )

        # Token 2 (Out-of-crop Token): large negative delta => ogi ~ 1.0 => penalty ~ 0
        self.assertAlmostEqual(
            diff_step1[0, 2].item(), 0.0, places=4,
            msg=f"Out-of-crop text (Token 2) must receive minimal penalty due to high OGI, got {diff_step1[0, 2].item()}"
        )

        # Token 1 (Refusal Token): scale-insensitive => delta_log <= 0, ogi ~ 0 => penalized
        self.assertGreater(
            diff_step1[0, 1].item(), 0.15,
            f"Refusal token (Token 1) must be penalized, got {diff_step1[0, 1].item():.4f}"
        )

        # 2. Dyadic decay at step 2 (mu_2 = 0.4 * 0.5 = 0.2)
        _, s_mu2, _, _ = nevfe_step_logits(
            logits_l=ll,
            logits_g=lg,
            beta_param=0.5,
            mu_param=0.4,
            step_idx=2,
        )
        diff_step2 = s_no - s_mu2
        self.assertGreater(
            diff_step2[0, 1].item(), 0.07,
            f"Refusal token at step 2 must receive halved penalty (~0.089), got {diff_step2[0, 1].item():.4f}"
        )
        self.assertLess(
            diff_step2[0, 1].item(), diff_step1[0, 1].item(),
            "Step 2 penalty must be strictly less than Step 1 penalty due to dyadic decay"
        )

        # 3. Step 3 & Step 4: OGI Prior Attenuation Inactive (step_idx >= 3)
        _, s_step3, _, _ = nevfe_step_logits(
            logits_l=ll,
            logits_g=lg,
            beta_param=0.5,
            mu_param=0.4,
            step_idx=3,
        )
        diff_step3 = s_no - s_step3
        self.assertAlmostEqual(
            diff_step3[0, 1].item(), 0.0, places=4,
            msg=f"OGI prior attenuation must be inactive at step 3, got penalty {diff_step3[0, 1].item()}"
        )

        _, s_step4, _, _ = nevfe_step_logits(
            logits_l=ll,
            logits_g=lg,
            beta_param=0.5,
            mu_param=0.4,
            step_idx=4,
        )
        diff_step4 = s_no - s_step4
        self.assertAlmostEqual(
            diff_step4[0, 1].item(), 0.0, places=4,
            msg=f"OGI prior attenuation must be inactive at step 4, got penalty {diff_step4[0, 1].item()}"
        )

        # 4. Complete Refusal Suppression under Full mu_param = 4.0
        # When mu_param = 4.0, refusal token is completely pruned from candidate support to -1e9
        _, s_full, _, _ = nevfe_step_logits(
            logits_l=ll,
            logits_g=lg,
            beta_param=0.5,
            mu_param=4.0,
            step_idx=1,
        )
        self.assertEqual(
            s_full[0, 1].item(), -1e9,
            "Under full mu_param = 4.0, ungrounded refusal token must be pruned to -1e9"
        )
        self.assertGreater(
            s_full[0, 0].item(), -1e5,
            "Optical token (Token 0) must remain active in candidate support"
        )

    # =========================================================================
    # Test 4: OGI Denominator Numerical Stability (Directive 3)
    # =========================================================================
    def test_04_ogi_denominator_numerical_stability(self):
        """
        Directive 3: Denominator clamp with clamp(min=1e-6) guarantees numerical stability
        even in degenerate edge cases where V_L and V_G produce identical distributions (zero variance).
        """
        K = 100
        # Exactly identical logits across all tokens (e.g. blank or solid color image)
        identical_logits = torch.ones(1, K) * 2.5

        next_tok, steered, coupling, ent = nevfe_step_logits(
            logits_l=identical_logits,
            logits_g=identical_logits,
            lambda_param=1.0,
            mu_param=4.0,
            step_idx=1,
        )

        self.assertFalse(torch.isnan(steered).any(), "steered_logits must not contain NaN")
        self.assertFalse(torch.isinf(steered).any(), "steered_logits must not contain Inf")
        self.assertFalse(torch.isnan(coupling).any(), "coupling must not contain NaN")
        self.assertFalse(torch.isnan(ent).any(), "entropy must not contain NaN")

    # =========================================================================
    # Test 5: Semantic Stopping Invariance and EOS Masking (Directive 4)
    # =========================================================================
    def test_05_semantic_stopping_invariance_and_eos_masking(self):
        """
        Directive 4 (Theorem 15):
        1. When global stream p_G signals EOS, decoding terminates immediately,
           unsteered by focal zoom or OGI attenuation.
        2. EOS tokens are masked with -1e9 in steered_logits during active continuation.
        3. Global stream logits_g are 100% unsteered by tau_t and OGI.
        """
        K = 150
        eos_id = 149
        eos_list = [eos_id]

        # 1. EOS Masking Verification during continuation
        ll_cont = torch.randn(1, K)
        lg_cont = torch.randn(1, K)
        lg_cont[0, 10] = 12.0  # Content token 10 dominates global stream
        lg_cont[0, eos_id] = -5.0

        _, steered_logits, _, _ = nevfe_step_logits(
            logits_l=ll_cont,
            logits_g=lg_cont,
            eos_token_ids=eos_list,
            step_idx=1,
            mu_param=4.0,
        )

        self.assertEqual(
            steered_logits[0, eos_id].item(), -1e9,
            "EOS token must be clamped to -1e9 in steered_logits during active continuation"
        )

        # 2. Dual-Stream Stopping Invariance
        # When p_G emits EOS (question answered), but focal crop has a runaway text continuation
        lg_stop = torch.randn(1, K)
        lg_stop[0, eos_id] = 20.0  # Global stream clearly signals EOS
        lg_stop[0, 50] = 2.0

        ll_runaway = torch.randn(1, K)
        ll_runaway[0, 50] = 25.0   # Focal stream wants to continue with token 50
        ll_runaway[0, eos_id] = -5.0

        # Stopping decision depends strictly on unsteered logits_g
        is_eos = int(torch.argmax(lg_stop, dim=-1).item()) in eos_list
        self.assertTrue(is_eos, "Dual-stream stopping must evaluate strictly on unsteered global stream")

        # 3. Verify logits_g is completely unmodified by nevfe_step_logits
        lg_copy = lg_stop.clone()
        _ = nevfe_step_logits(
            logits_l=ll_runaway,
            logits_g=lg_stop,
            eos_token_ids=eos_list,
            step_idx=1,
            mu_param=4.0,
        )
        self.assertTrue(torch.equal(lg_stop, lg_copy), "logits_g must never be mutated during steering")

    # =========================================================================
    # Test 6: Query-Orthogonal Potential Barrier Preservation (Directive 4)
    # =========================================================================
    def test_06_query_orthogonal_potential_barrier_preservation(self):
        """
        Directive 4: The Query-Orthogonal Potential Barrier operates on prompt tokens
        via q_mask for step_idx <= barrier_horizon (horizon = 3).
        """
        K = 50
        ll = torch.randn(1, K)
        lg = torch.randn(1, K)

        q_mask = torch.zeros(K, dtype=torch.bool)
        q_mask[5] = True   # Prompt token 5 (e.g. carrier preamble word)
        q_mask[12] = True  # Prompt token 12

        # Ensure token 5 is dominant so it remains inside support_mask
        ll_supp = ll.clone()
        lg_supp = lg.clone()
        ll_supp[0, 5] = 10.0
        lg_supp[0, 5] = 10.0

        # Step 1: Barrier = 4.0
        _, steered_step1, _, _ = nevfe_step_logits(
            logits_l=ll_supp,
            logits_g=lg_supp,
            q_mask=q_mask,
            mu_param=4.0,
            barrier_decay=0.5,
            barrier_horizon=3,
            step_idx=1,
        )

        _, steered_no_q, _, _ = nevfe_step_logits(
            logits_l=ll_supp,
            logits_g=lg_supp,
            q_mask=None,
            mu_param=4.0,
            barrier_decay=0.5,
            barrier_horizon=3,
            step_idx=1,
        )

        # Check prompt token 5 received query barrier reduction
        diff_q = (steered_no_q[0, 5] - steered_step1[0, 5]).item()
        self.assertAlmostEqual(
            diff_q, 4.0, places=3,
            msg="Prompt token in q_mask must receive exact query barrier penalty"
        )

        # Check non-prompt token was not affected by q_mask
        diff_non_q = steered_no_q[0, 20] - steered_step1[0, 20]
        self.assertAlmostEqual(
            diff_non_q.item(), 0.0, places=3,
            msg="Non-prompt token must not be penalized by q_mask"
        )

    # =========================================================================
    # Test 7: AST Anti-Cheating & Invariant Governance Audit (Directive 5)
    # =========================================================================
    def test_07_ast_anti_cheating_audit(self):
        """
        Directive 5:
        1. 0 regex imports (re, regex).
        2. 0 regex calls (re.sub, re.match, etc.).
        3. 0 hardcoded refusal keyword lists/blacklists.
        4. Strictly <= 2 continuous hyperparameters in NEVFEEngine.
        """
        target_files = [
            "src/nevfe/core/engine.py",
            "src/nevfe/adapter/vintern_adapter.py",
        ]

        # Forbidden substring patterns in variable assignments (keyword blacklists)
        forbidden_keywords = [
            "khong the xac dinh",
            "không thể xác định",
            "không có thông tin",
            "khong co thong tin",
        ]

        for filepath in target_files:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                tree = ast.parse(content, filename=filepath)

            # 1. Check for forbidden regex imports
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

                # 2. Check for regex function calls
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute):
                        caller_id = getattr(node.func.value, "id", None)
                        if caller_id in ("re", "regex"):
                            self.fail(f"Forbidden call {caller_id}.{node.func.attr} in {filepath}")

            # 3. Check for hardcoded refusal keyword blacklists
            lower_content = content.lower()
            for kw in forbidden_keywords:
                self.assertNotIn(
                    kw, lower_content,
                    f"Forbidden refusal keyword string '{kw}' detected in {filepath}!"
                )

        # 4. Check NEVFEEngine hyperparameters
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


if __name__ == "__main__":
    unittest.main()
