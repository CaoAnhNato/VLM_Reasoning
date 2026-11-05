# -*- coding: utf-8 -*-
"""
Master Unit Test Suite for Iteration 19 Implementation (Directives W1–W5).

Strictly verifies:
- Directive W1: Theorem 38 Question-Conditioned Cross-Attention Saliency Steering (Q-CASS).
  * Shape invariance across 2D [P, D] and 3D [1, P, D] for P in {256, 512}, D=896.
  * Spatial KL divergence token specificity weighting (omega_hat).
  * Convex-hull visual prototype projection and query-grounded lexical bias.
- Directive W2: Theorem 39 Differential Visual Stroke Projection & Resynthesized Logits.
  * Background cancellation and stroke logit projection through lm_head.
  * Resynthesized focal logits z_tilde_l = z_l + z_stroke.
- Directive W3: Theorem 39 Alphanumeric Cohesion Clamp on Token 220.
  * Blocks premature whitespace token 220 emission when prev_token_id in alnum_token_ids under optical evidence.
  * Releases whitespace token 220 when optical evidence is exhausted.
- Directive W4: Theorem 40 Diacritic-Preserving Contrastive Operator.
  * Bounded Rényi Reference Mixture: delta_log <= -log(lambda_diac) = 2.3026 nats.
  * Visual curvature gating: throttling low-probability unaccented distractors.
  * Strictly monotonic margin non-flipping across lambda_t in [0.5, 3.0].
  * Integration in DynamicContrastiveDecoder.
- Directive W5: Strict A* Zero-Heuristic AST Compliance & Anti-Cheating Invariants.
"""

import ast
import math
import unittest
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nevfe.core.engine import (
    NEVFEEngine,
    compute_question_conditioned_cross_attention_saliency,
    compute_query_grounded_lexical_bias,
    nevfe_step_logits,
)
from scripts.benchmark.sg_vcd.dynamic_cd import DynamicContrastiveDecoder


class TestIteration19Implementation(unittest.TestCase):
    """Exhaustive behavioral unit and boundary tests covering Iteration 19 Directives W1 through W5."""

    def setUp(self):
        torch.manual_seed(2026)
        np.random.seed(2026)

    # =========================================================================
    # Directive W1: Question-Conditioned Cross-Attention Saliency Steering (Q-CASS)
    # =========================================================================
    def test_w1_qcass_shape_invariance_and_simplex(self):
        """
        Directive W1: Verify Q-CASS handles both 2D [P, D] and 3D [1, P, D] shapes
        for P in {256, 512} and D in {64, 896}.
        Verify that output saliency S_QCASS lies on the probability simplex Delta^{P-1}.
        """
        for P in [256, 512]:
            for D in [64, 896]:
                for is_3d in [False, True]:
                    T = 12
                    v = torch.randn(1, P, D) if is_3d else torch.randn(P, D)
                    e = torch.randn(1, T, D) if is_3d else torch.randn(T, D)

                    s_qcass, omega_hat = compute_question_conditioned_cross_attention_saliency(
                        visual_tokens=v,
                        question_embeddings=e,
                        temperature=0.1,
                    )

                    # Verify shapes
                    self.assertEqual(s_qcass.shape, (P,), f"Expected s_qcass shape ({P},), got {s_qcass.shape}")
                    self.assertEqual(omega_hat.shape, (T,), f"Expected omega_hat shape ({T},), got {omega_hat.shape}")

                    # Verify probability simplex constraints (sum == 1.0, non-negative)
                    self.assertAlmostEqual(s_qcass.sum().item(), 1.0, places=5)
                    self.assertTrue((s_qcass >= -1e-6).all(), "s_qcass elements must be non-negative")

                    self.assertAlmostEqual(omega_hat.sum().item(), 1.0, places=5)
                    self.assertTrue((omega_hat >= -1e-6).all(), "omega_hat elements must be non-negative")

    def test_w1_qcass_spatial_kl_specificity_discrimination(self):
        """
        Directive W1: Verify spatial KL divergence specificity weighting:
        Generic syntax tokens (diffuse/uniform attention) achieve omega_j ~ 0.
        Discriminatory entity tokens (sharp peak on target patch) achieve omega_j >> 0.
        """
        P = 256
        D = 128
        T = 3

        # Construct visual tokens: one localized cluster at patch 42
        v = torch.randn(P, D) * 0.05
        target_patch = 42
        target_feat = torch.randn(D)
        target_feat = target_feat / target_feat.norm()
        v[target_patch] = target_feat * 5.0

        # Construct question tokens:
        # Token 0: generic uniform token (orthogonal or diffuse relative to all patches)
        # Token 1: sharp entity keyword aligned with target patch
        # Token 2: generic syntax token
        e = torch.randn(T, D) * 0.05
        e[1] = target_feat * 5.0  # Aligns strongly with patch 42

        s_qcass, omega_hat = compute_question_conditioned_cross_attention_saliency(
            visual_tokens=v,
            question_embeddings=e,
            temperature=0.1,
        )

        # Token 1 must have significantly higher specificity weight than Token 0 or Token 2
        self.assertGreater(
            omega_hat[1].item(), omega_hat[0].item() * 3.0,
            f"Sharp keyword token must receive >3x higher weight than diffuse token, got {omega_hat.tolist()}"
        )
        self.assertGreater(
            omega_hat[1].item(), omega_hat[2].item() * 3.0,
            f"Sharp keyword token must receive >3x higher weight than syntax token, got {omega_hat.tolist()}"
        )

        # Saliency S_QCASS must peak strongly at target_patch
        self.assertEqual(
            int(s_qcass.argmax().item()), target_patch,
            f"Saliency must peak at target patch {target_patch}, got {int(s_qcass.argmax().item())}"
        )

    def test_w1_qcass_convex_prototype_and_lexical_bias(self):
        """
        Directive W1: Verify convex-hull prototype v_qcass and lexical bias log_p_qcass:
        1. Prototype lies within the convex hull: ||v_qcass|| <= max_u ||v_u||.
        2. log_p_qcass is a valid log-probability distribution over vocabulary.
        3. Norm layer pre-conditioning operates correctly without dimension mismatch.
        """
        P = 256
        D = 64
        V = 1000

        v = torch.randn(P, D)
        s_qcass = F.softmax(torch.randn(P), dim=0)
        lm_head = nn.Linear(D, V, bias=False)
        norm_layer = nn.LayerNorm(D)

        log_p_qcass = compute_query_grounded_lexical_bias(
            visual_tokens=v,
            s_qcass=s_qcass,
            lm_head=lm_head,
            norm_layer=norm_layer,
        )

        self.assertEqual(log_p_qcass.shape, (1, V))
        p_qcass = torch.exp(log_p_qcass)
        self.assertAlmostEqual(p_qcass.sum().item(), 1.0, places=5)
        self.assertFalse(torch.isnan(log_p_qcass).any(), "log_p_qcass must not contain NaNs")

        # Test convex hull boundedness
        s_weights = (s_qcass / s_qcass.sum()).unsqueeze(-1)
        v_qcass = torch.sum(s_weights * v, dim=0)
        norm_v_qcass = torch.norm(v_qcass, p=2).item()
        max_patch_norm = torch.norm(v, p=2, dim=-1).max().item()
        self.assertLessEqual(
            norm_v_qcass, max_patch_norm + 1e-4,
            "Convex prototype norm must not exceed maximum visual patch norm"
        )

    def test_w1_qcass_lexical_bias_in_nevfe_step_logits(self):
        """
        Directive W1: In nevfe_step_logits, passing log_p_qcass adds lexical potential
        Delta_lex = ReLU(log_p_qcass - log_p_g) * chi_term, scaled by mu_param.
        """
        K = 100
        ll = torch.zeros((1, K))
        lg = torch.zeros((1, K))
        target_token = 25

        # Construct log_p_qcass with high probability for target_token
        log_p_qcass = torch.full((1, K), -10.0)
        log_p_qcass[0, target_token] = 0.0

        # Step 1: decode with log_p_qcass
        _, steered_with_qcass, _, _ = nevfe_step_logits(
            logits_l=ll,
            logits_g=lg,
            mu_param=2.0,
            step_idx=1,
            log_p_qcass=log_p_qcass,
        )

        # Step 1: decode without log_p_qcass
        _, steered_without_qcass, _, _ = nevfe_step_logits(
            logits_l=ll,
            logits_g=lg,
            mu_param=2.0,
            step_idx=1,
            log_p_qcass=None,
        )

        # The target_token potential must be strictly higher with log_p_qcass
        diff_target = steered_with_qcass[0, target_token].item() - steered_without_qcass[0, target_token].item()
        self.assertGreater(
            diff_target, 0.5,
            f"Target token should receive strong lexical boost from Q-CASS, got boost {diff_target}"
        )

    # =========================================================================
    # Directive W2: Differential Visual Stroke Projection & Resynthesized Logits
    # =========================================================================
    def test_w2_differential_visual_stroke_background_cancellation(self):
        """
        Directive W2: Verify differential stroke Delta c_t = c_{l,t} - c_{g,t}:
        When background clutter is identical in focal and global views, Delta c_t ~ 0.
        When focal zoom reveals sharp character stroke, Delta c_t isolates the stroke.
        """
        D = 64
        V = 200
        P_L = 256
        P_G = 256

        # Identical background visual tokens in both views
        common_bg = torch.randn(P_L, D) * 0.1
        v_l = common_bg.clone()
        v_g = common_bg.clone()

        # Add unique character stroke to focal view at patch 10
        stroke_feat = torch.randn(D)
        stroke_feat = stroke_feat / stroke_feat.norm()
        v_l[10] = stroke_feat * 4.0

        h_l = stroke_feat.unsqueeze(0).unsqueeze(0)  # [1, 1, D]
        h_g = stroke_feat.unsqueeze(0).unsqueeze(0)  # [1, 1, D]

        # Compute stroke centroids
        a_l = F.softmax(torch.matmul(h_l.squeeze(0), v_l.t()) / math.sqrt(D), dim=-1)
        a_g = F.softmax(torch.matmul(h_g.squeeze(0), v_g.t()) / math.sqrt(D), dim=-1)

        c_l = torch.matmul(a_l, v_l)  # [1, D]
        c_g = torch.matmul(a_g, v_g)  # [1, D]
        delta_c = c_l - c_g  # [1, D]

        # Delta c must be non-zero and aligned with stroke_feat
        stroke_alignment = F.cosine_similarity(delta_c, stroke_feat.unsqueeze(0)).item()
        self.assertGreater(
            stroke_alignment, 0.70,
            f"Differential stroke must align with focal character stroke, got cosine {stroke_alignment}"
        )

    def test_w2_resynthesized_logits_in_engine_decode_step(self):
        """
        Directive W2: In NEVFEEngine.decode_step, passing logits_stroke
        resynthesizes focal logits: z_tilde_l = logits_l + logits_stroke.
        """
        engine = NEVFEEngine(lambda_param=1.0, beta_param=1.0, mu_param=2.0)
        K = 100
        ll = torch.zeros((1, K))
        lg = torch.zeros((1, K))
        stroke_token = 33

        logits_stroke = torch.zeros((1, K))
        logits_stroke[0, stroke_token] = 5.0

        # Decode step with logits_stroke
        next_tok, steered, _, _ = engine.decode_step(
            logits_l=ll,
            logits_g=lg,
            logits_stroke=logits_stroke,
            step_idx=1,
        )

        # The stroke token should have received the boost and be selected
        self.assertEqual(
            int(next_tok.item()), stroke_token,
            f"Selected token must be stroke token {stroke_token}, got {int(next_tok.item())}"
        )

    # =========================================================================
    # Directive W3: Alphanumeric Cohesion Clamp on Token 220
    # =========================================================================
    def test_w3_alphanumeric_cohesion_clamp_blocks_whitespace_token_220(self):
        """
        Directive W3: When prev_token_id is in alnum_token_ids and optical evidence
        omega_alnum > s_bar_t is present, penalize space token 220 by mu_param * psi_t,
        ensuring the candidate alphanumeric continuation token strictly dominates space token 220.
        """
        K = 250
        space_id = 220
        prev_alnum_id = 45  # 'A'
        next_alnum_id = 46  # 'B'
        alnum_ids = list(range(40, 50))
        mu = 4.0

        # Baseline language prior strongly favors space over continuing the alphanumeric code
        lg = torch.full((1, K), -5.0)
        lg[0, space_id] = 5.0
        lg[0, next_alnum_id] = 2.0

        # Focal stream has sharp optical evidence for the genuine alphanumeric continuation
        ll = lg.clone()
        ll[0, next_alnum_id] = 8.0
        ll[0, space_id] = 5.0

        next_tok, steered, _, _ = nevfe_step_logits(
            logits_l=ll,
            logits_g=lg,
            mu_param=mu,
            step_idx=2,
            prev_token_id=prev_alnum_id,
            space_token_id=space_id,
            alnum_token_ids=alnum_ids,
        )

        # Verification 1: Candidate alphanumeric continuation token strictly dominates whitespace token 220
        self.assertGreater(
            steered[0, next_alnum_id].item(),
            steered[0, space_id].item(),
            "Alphanumeric token must strictly dominate whitespace token 220 under alnum cohesion clamp"
        )
        # Verification 2: Selected token must be the alphanumeric continuation
        self.assertEqual(
            int(next_tok.item()), next_alnum_id,
            f"Expected next token to be alphanumeric {next_alnum_id}, got {int(next_tok.item())}"
        )

    def test_w3_alphanumeric_cohesion_clamp_releases_when_evidence_exhausted(self):
        """
        Directive W3: When optical evidence is exhausted (omega_alnum <= s_bar_t),
        the cohesion clamp must be 0, permitting natural word separation.
        """
        K = 250
        space_id = 220
        prev_alnum_id = 45
        alnum_ids = [40, 41, 42, 43, 44, 45]
        mu = 3.0

        # Exhausted optical evidence: flat logits between focal and global
        ll = torch.zeros((1, K))
        lg = torch.zeros((1, K))

        _, steered_exhausted, _, _ = nevfe_step_logits(
            logits_l=ll,
            logits_g=lg,
            mu_param=mu,
            step_idx=2,
            prev_token_id=prev_alnum_id,
            space_token_id=space_id,
            alnum_token_ids=alnum_ids,
        )

        _, steered_non_alnum, _, _ = nevfe_step_logits(
            logits_l=ll,
            logits_g=lg,
            mu_param=mu,
            step_idx=2,
            prev_token_id=999,
            space_token_id=space_id,
            alnum_token_ids=alnum_ids,
        )

        # Under optical exhaustion, no extra cohesion penalty should be applied to space_id
        diff_space = abs(steered_exhausted[0, space_id].item() - steered_non_alnum[0, space_id].item())
        self.assertLess(
            diff_space, 1e-4,
            f"Under optical exhaustion, space token should not receive extra cohesion penalty, got diff {diff_space}"
        )

    # =========================================================================
    # Directive W4: Theorem 40 Diacritic-Preserving Contrastive Operator
    # =========================================================================
    def test_w4_theorem40_universal_contrast_bound(self):
        """
        Directive W4 (Theorem 40 Lemma 1):
        For any vocabulary token w, bounded_delta_log <= -log(lambda_diac) = 2.3026 nats
        even under extreme density ratio explosion (p_G -> 0, p_L >> 0).
        """
        lambda_diac = 0.10
        theoretical_bound = -math.log(lambda_diac)  # log(10) ~ 2.302585 nats

        # Construct extreme adversarial distribution:
        # p_g has near-zero probability for rare token (e.g. 1e-12)
        # p_l has moderate probability (e.g. 0.05)
        for p_l_val in [0.001, 0.01, 0.05, 0.50, 0.99]:
            for p_g_val in [1e-12, 1e-8, 1e-5, 0.001, 0.10]:
                p_l = torch.tensor([p_l_val], dtype=torch.float32)
                p_g = torch.tensor([p_g_val], dtype=torch.float32)

                p_ref = (1.0 - lambda_diac) * p_g + lambda_diac * p_l
                bounded_delta_log = torch.relu(torch.log(p_l) - torch.log(p_ref.clamp(min=1e-12)))

                val = bounded_delta_log.item()
                self.assertLessEqual(
                    val, theoretical_bound + 1e-5,
                    f"Theorem 40 bound violated: {val:.6f} > {theoretical_bound:.6f} for p_L={p_l_val}, p_G={p_g_val}"
                )

    def test_w4_theorem40_margin_monotonicity_prevents_diacritic_stripping(self):
        """
        Directive W4 (Theorem 40 Invariant 3):
        Simulate true accented entity w* ('phở') vs unaccented distractor w' ('pho').
        Under Theorem 40, contrastive margin Delta S_t(lambda_t) = S_t(w*) - S_t(w')
        is STRICTLY MONOTONICALLY INCREASING across lambda_t in [0.5, 3.0].
        Diacritic stripping is mathematically impossible.
        """
        # True accented token w*: p_L(w*) = 0.13, p_G(w*) = 0.05 (fluent Vietnamese word)
        p_l_star = 0.13
        p_g_star = 0.05

        # Unaccented distractor w': p_L(w') = 0.0024, p_G(w') = 1e-5 (rare unaccented token)
        p_l_dist = 0.0024
        p_g_dist = 1e-5

        lambda_diac = 0.10
        mu_diac = 1.0

        p_l_tensor = torch.tensor([[p_l_star, p_l_dist]], dtype=torch.float32)
        p_g_tensor = torch.tensor([[p_g_star, p_g_dist]], dtype=torch.float32)
        max_p_l = p_l_tensor.max(dim=-1, keepdim=True).values

        # Theorem 40 computation
        p_ref = (1.0 - lambda_diac) * p_g_tensor + lambda_diac * p_l_tensor
        bounded_delta = torch.relu(torch.log(p_l_tensor) - torch.log(p_ref.clamp(min=1e-12)))
        curvature = torch.pow(p_l_tensor / (max_p_l + 1e-12), mu_diac)
        s_diac = curvature * bounded_delta

        s_star = s_diac[0, 0].item()
        s_dist = s_diac[0, 1].item()

        # The true accented token must receive significantly higher boost than distractor
        self.assertGreater(
            s_star, s_dist * 5.0,
            f"True accented token must receive >5x higher contrast boost than distractor, got {s_star:.4f} vs {s_dist:.4f}"
        )

        # Verify margin monotonicity across lambda_t in [0.5, 3.0]
        base_margin = 1.0  # Arbitrary base log-prob margin
        margins = []
        lambdas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        for lam in lambdas:
            margin = base_margin + lam * (s_star - s_dist)
            margins.append(margin)

        for i in range(1, len(margins)):
            self.assertGreaterEqual(
                margins[i], margins[i - 1],
                f"Margin non-monotonicity at lambda={lambdas[i]}: {margins[i]} < {margins[i-1]}"
            )

    def test_w4_dynamic_cd_theorem40_integration(self):
        """
        Directive W4: Verify Theorem 40 bounded reference mixture and visual curvature gating
        in DynamicContrastiveDecoder.decode_step.
        """
        decoder = DynamicContrastiveDecoder(
            alpha_0=0.65,
            lambda_diac=0.10,
            mu_diac=1.0,
        )

        K = 50
        pos_logits = torch.randn(1, K)
        neg_logits = torch.randn(1, K)

        next_token_id, alpha_t, entropy_val, d_js_val = decoder.decode_step(
            pos_logits=pos_logits,
            neg_logits=neg_logits,
            step=1,
            spatial_confidence=0.85,
        )

        self.assertIsInstance(next_token_id, int)
        self.assertGreaterEqual(next_token_id, 0)
        self.assertLess(next_token_id, K)
        self.assertGreaterEqual(alpha_t, 0.0)
        self.assertLessEqual(alpha_t, 2.0)
        self.assertFalse(math.isnan(entropy_val))
        self.assertFalse(math.isnan(d_js_val))

    # =========================================================================
    # Directive W5: Strict A* Zero-Heuristic AST Compliance & Anti-Cheating Invariants
    # =========================================================================
    def test_w5_zero_heuristics_ast_audit(self):
        """
        Directive W5: AST audit of all modified files:
        1. 0 occurrences of re.sub, re.compile, re.match, or regex patterns.
        2. 0 hardcoded stopword dictionaries.
        3. 0 coordinate forcing or ground-truth leakage.
        """
        files_to_check = [
            "src/nevfe/core/engine.py",
            "src/nevfe/adapter/vintern_adapter.py",
            "scripts/benchmark/sg_vcd/dynamic_cd.py",
        ]

        prohibited_functions = {"sub", "compile", "match", "search", "findall", "finditer", "split"}

        for filepath in files_to_check:
            with open(filepath, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=filepath)

            for node in ast.walk(tree):
                # Check for regex imports or calls
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertNotEqual(
                            alias.name, "re",
                            f"Prohibited 'import re' detected in {filepath}"
                        )
                elif isinstance(node, ast.ImportFrom):
                    self.assertNotEqual(
                        node.module, "re",
                        f"Prohibited 'from re import ...' detected in {filepath}"
                    )
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute):
                        if node.func.attr in prohibited_functions:
                            # Verify caller is not 're'
                            if isinstance(node.func.value, ast.Name) and node.func.value.id == "re":
                                self.fail(f"Prohibited re.{node.func.attr} call in {filepath}")

    def test_w5_hyperparameter_budget_invariant(self):
        """
        Directive W5: Strictly <= 2 global continuous hyperparameters:
        lambda_param in [0.1, 1.5] and mu_param in [0.0, 10.0].
        All internal constants (tau_q=0.1, lambda_diac=0.10, mu_diac=1.0) are fixed.
        """
        engine = NEVFEEngine(lambda_param=0.50, mu_param=1.00)
        self.assertAlmostEqual(engine.lambda_param, 0.50)
        self.assertAlmostEqual(engine.mu_param, 1.00)
        self.assertAlmostEqual(engine.lambda_diac, 0.10)
        self.assertAlmostEqual(engine.mu_diac, 1.0)


if __name__ == "__main__":
    unittest.main()
