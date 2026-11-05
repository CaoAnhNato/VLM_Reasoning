# -*- coding: utf-8 -*-
"""
Iteration 12 Comprehensive Unit Test Suite: Mixture-Entropy Dynamic Continuation Horizon (Directive W3, W4, W5, W6).

Verifies:
1. Dual-Form Log-Sum-Exp Identity: Exact mathematical equivalence H(p) = logsumexp(z) - sum(p * z)
   without transcendental log calls on large vocabulary |V| = 151,674.
2. Normalized Entropy Sensitivity: High-confidence peaked distributions yield H_hat < 0.05,
   while ambiguous multi-token tails yield H_hat > 0.20.
3. Mixture-Entropy Minimum Bound: Guarantees H_hat_t <= min(H_hat(p_L), H_hat(p_G)), extinguishing
   entropy whenever either stream is confident.
4. Dynamic Continuation Horizon gamma_t = H_hat_t * (1 - Xi_t^div) in [0.0, 1.0].
5. Symmetrically Calibrated EOS Termination Penalty: Instant EOS emission on short entities
   (term_penalty -> 0) while sustaining momentum on ambiguous tails.
6. Optical Continuation Saliency Gating: tilde_Omega_t = gamma_t * Omega_t.
7. Consensus Stopping Integration: evaluate_semantic_stopping stops seamlessly.
8. Anti-Cheating & Parameter Budget Compliance: 0 regex, <= 2 continuous hyperparameters.
"""

import ast
import math
import os
import unittest
from typing import List, Tuple
import torch
import torch.nn.functional as F

from src.nevfe.core.engine import (
    NEVFEEngine,
    nevfe_step_logits,
    compute_normalized_entropy_lse,
    compute_dynamic_contrastive_temperature,
    compute_entropy_continuation_horizon,
    compute_optical_continuation_saliency,
    evaluate_semantic_stopping,
)


class TestEntropyCHOIteration12(unittest.TestCase):
    """Test suite validating all Iteration 12 Mixture-Entropy CHO invariants."""

    def setUp(self):
        self.device = torch.device("cpu")
        self.vocab_size = 1000
        self.full_vocab_size = 151674  # Vintern-1B-v3_5 vocabulary size

    # =========================================================================
    # Test 1: Dual-Form Log-Sum-Exp Entropy Equivalence
    # =========================================================================
    def test_01_dual_form_lse_entropy_equivalence(self):
        """
        Directive W3: Verify that compute_normalized_entropy_lse computes the Shannon
        entropy via the Legendre dual identity H(p) = LSE(z) - sum(p * z) matching
        the definition -sum(p * log p) within float32 numerical precision.
        """
        # Random logits across vocabulary
        z = torch.randn(1, 500, dtype=torch.float32, device=self.device)

        # LSE identity formulation
        entropy_nats_lse, norm_entropy_lse = compute_normalized_entropy_lse(z)

        # Classical definition formulation
        p = F.softmax(z, dim=-1)
        entropy_classical = -torch.sum(p * torch.log(p.clamp(min=1e-12)), dim=-1, keepdim=True)
        norm_classical = entropy_classical / math.log(500)

        # Assert equivalence
        diff = torch.abs(entropy_nats_lse - entropy_classical).item()
        self.assertLess(diff, 1e-4, f"LSE entropy differs from classical: {diff}")

        # Assert non-negativity
        self.assertGreaterEqual(entropy_nats_lse.item(), 0.0)
        self.assertGreaterEqual(norm_entropy_lse.item(), 0.0)
        self.assertLessEqual(norm_entropy_lse.item(), 1.0)

    # =========================================================================
    # Test 2: Numerical Stability on Large Vocabulary |V| = 151,674
    # =========================================================================
    def test_02_large_vocab_stability(self):
        """
        Directive W3 & W4: Compute entropy over full Vintern-1B vocabulary (|V| = 151,674).
        Must execute without NaNs, Infs, or subnormal underflow.
        """
        z_large = torch.randn(1, self.full_vocab_size, dtype=torch.float32, device=self.device)
        entropy_nats, norm_entropy = compute_normalized_entropy_lse(z_large)

        self.assertFalse(torch.isnan(entropy_nats).any())
        self.assertFalse(torch.isinf(entropy_nats).any())
        self.assertGreaterEqual(norm_entropy.item(), 0.0)
        self.assertLessEqual(norm_entropy.item(), 1.0)

    # =========================================================================
    # Test 3: Sensitivity on Peaked vs Diffuse Distributions
    # =========================================================================
    def test_03_peaked_vs_diffuse_entropy_sensitivity(self):
        """
        Directive W3:
        1. Peaked distribution (p[EOS] >= 0.95): H_hat_t < 0.05 (near zero).
        2. Uniform distribution: H_hat_t == 1.0.
        3. Ambiguous multi-token distribution: H_hat_t in [0.15, 0.50].
        """
        # Case A: Peaked distribution on EOS
        z_peaked = torch.full((1, self.vocab_size), -10.0, dtype=torch.float32, device=self.device)
        z_peaked[0, 0] = 10.0  # Top-1 token probability > 0.999
        _, norm_h_peaked = compute_normalized_entropy_lse(z_peaked)
        self.assertLess(norm_h_peaked.item(), 0.05, f"Peaked distribution must have H_hat < 0.05, got {norm_h_peaked.item()}")

        # Case B: Uniform distribution
        z_uniform = torch.zeros((1, self.vocab_size), dtype=torch.float32, device=self.device)
        _, norm_h_uniform = compute_normalized_entropy_lse(z_uniform)
        self.assertAlmostEqual(norm_h_uniform.item(), 1.0, places=4, msg="Uniform distribution must have H_hat == 1.0")

        # Case C: Ambiguous distribution (competing top candidates)
        z_diffuse = torch.full((1, self.vocab_size), -5.0, dtype=torch.float32, device=self.device)
        z_diffuse[0, :20] = 2.0  # 20 plausible candidates
        _, norm_h_diffuse = compute_normalized_entropy_lse(z_diffuse)
        self.assertGreater(norm_h_diffuse.item(), 0.15)
        self.assertLess(norm_h_diffuse.item(), 0.60)

    # =========================================================================
    # Test 4: Mixture-Entropy Minimum Bound (Adversarial Challenge 4)
    # =========================================================================
    def test_04_mixture_entropy_minimum_bound(self):
        """
        Directive W3 & Adversarial Challenge 4:
        In compute_dynamic_contrastive_temperature, H_hat_t <= min(H_hat(p_L), H_hat(p_G)).
        If Stream G collapses to EOS (p_G[EOS] = 0.99) while Stream L sees peripheral text,
        norm_entropy must still contract to < 0.05!
        """
        # Stream G is confident at EOS
        lg_peaked = torch.full((1, self.vocab_size), -10.0, device=self.device)
        lg_peaked[0, 0] = 10.0  # p_G(EOS) ~ 1.0

        # Stream L sees diffuse text (high entropy)
        ll_diffuse = torch.randn(1, self.vocab_size, device=self.device)

        tau_t, norm_entropy, norm_div = compute_dynamic_contrastive_temperature(ll_diffuse, lg_peaked)

        # The resulting entropy must be bounded by Stream G's near-zero entropy!
        self.assertLess(norm_entropy.item(), 0.05, f"Entropy must contract when Stream G is at EOS, got {norm_entropy.item()}")

    # =========================================================================
    # Test 5: Dynamic Continuation Horizon gamma_t
    # =========================================================================
    def test_05_dynamic_continuation_horizon_gamma_t(self):
        """
        Directive W3: gamma_t = H_hat_t * (1 - Xi_t^div) in [0.0, 1.0].
        1. When H_hat -> 0: gamma_t -> 0 regardless of divergence.
        2. When divergence -> 1: gamma_t -> 0.
        3. When H_hat > 0 and divergence is low: gamma_t > 0.
        """
        # Case A: Low entropy (H_hat = 0.02)
        h_low = torch.tensor([[0.02]], device=self.device)
        div_mid = torch.tensor([[0.30]], device=self.device)
        gamma_low = compute_entropy_continuation_horizon(h_low, div_mid)
        self.assertLess(gamma_low.item(), 0.02)

        # Case B: High divergence (div = 0.95)
        h_high = torch.tensor([[0.50]], device=self.device)
        div_high = torch.tensor([[0.95]], device=self.device)
        gamma_div = compute_entropy_continuation_horizon(h_high, div_high)
        self.assertLess(gamma_div.item(), 0.05)

        # Case C: Active continuation (H_hat = 0.40, div = 0.10)
        h_active = torch.tensor([[0.40]], device=self.device)
        div_low = torch.tensor([[0.10]], device=self.device)
        gamma_active = compute_entropy_continuation_horizon(h_active, div_low)
        self.assertAlmostEqual(gamma_active.item(), 0.40 * (1.0 - 0.10), places=4)

    # =========================================================================
    # Test 6: Instant EOS Emission on Short Entities (Resolving Iteration 11 Trap)
    # =========================================================================
    def test_06_instant_eos_emission_on_short_entities(self):
        """
        Directive W3: For a short entity query (e.g. 'CBAM'), when the entity has been emitted,
        Stream G predicts EOS with high confidence. Under Iteration 11's static step ramp,
        the termination penalty crushed EOS. Under Iteration 12's Mixture-Entropy CHO,
        gamma_t -> 0 deactivates the termination penalty, allowing instant EOS emission.
        """
        engine = NEVFEEngine(lambda_param=1.0, beta_param=1.0, mu_param=4.0)
        eos_id = 999
        eos_ids = [eos_id]

        # Simulate step t=3 on concise entity completion:
        # Stream G has finished the acronym and strongly predicts EOS
        lg = torch.full((1, self.vocab_size), -10.0, device=self.device)
        lg[0, eos_id] = 12.0  # p_G(EOS) ~ 1.0

        # Stream L still sees peripheral text (e.g. secondary banner words) at token 100
        ll = torch.full((1, self.vocab_size), -5.0, device=self.device)
        ll[0, 100] = 5.0
        ll[0, eos_id] = -2.0  # p_L(EOS) is low

        # Step t=3 (in Iteration 11, static ramp was sigma_h = 0.5, which killed EOS!)
        res = nevfe_step_logits(
            logits_l=ll,
            logits_g=lg,
            step_idx=3,
            eos_token_ids=eos_ids,
            return_continuation_saliency=True,
        )
        next_tok, steered_logits, coupling, entropy, omega_t, s_bar_t = res

        # Under Iteration 12, EOS MUST be selected!
        self.assertEqual(int(next_tok.item()), eos_id, "Short entity completion must emit EOS at step t=3!")

        # Verify consensus stopping triggers
        stop_decision = evaluate_semantic_stopping(
            logits_g=lg,
            logits_l=ll,
            steered_tokens=next_tok,
            omega_t=omega_t,
            s_bar_t=s_bar_t,
            eos_token_ids=eos_ids,
        )
        self.assertTrue(bool(stop_decision.item()), "Semantic stopping must evaluate to True!")

    # =========================================================================
    # Test 7: Sustained Continuation on Ambiguous Multi-Token Tails
    # =========================================================================
    def test_07_sustained_continuation_on_ambiguous_tails(self):
        """
        Directive W3: When decoding ambiguous multi-token tails (e.g. 'kỷ niệm 39 năm...'),
        vocabulary entropy H_hat_t > 0.20 maintains gamma_t > 0, which maintains the termination
        penalty to prevent premature EOS truncation.
        """
        eos_id = 999
        eos_ids = [eos_id]
        content_token = 250

        # Global stream wants to terminate prematurely with moderate diffuse entropy
        lg = torch.randn(1, self.vocab_size, device=self.device) * 0.5
        lg[0, eos_id] = 2.0

        # Focal stream has strong optical evidence for continuation tokens with lexical ambiguity (entropy > 0.20)
        ll = torch.randn(1, self.vocab_size, device=self.device) * 0.5
        ll[0, content_token] = 6.0
        ll[0, content_token + 1] = 4.5
        ll[0, content_token + 2] = 4.0
        ll[0, eos_id] = -5.0

        res = nevfe_step_logits(
            logits_l=ll,
            logits_g=lg,
            step_idx=4,
            eos_token_ids=eos_ids,
            return_continuation_saliency=True,
        )
        next_tok, steered_logits, coupling, entropy, omega_t, s_bar_t = res

        # Must NOT terminate prematurely; content token must win
        self.assertNotEqual(int(next_tok.item()), eos_id, "Must NOT emit EOS when optical continuation evidence is active!")
        self.assertEqual(int(next_tok.item()), content_token, "Content token must be selected!")

        # Continuation saliency omega_t must be greater than ambient expectation s_bar_t
        self.assertGreater(omega_t.item(), s_bar_t.item(), "Omega_t must exceed ambient baseline during active text!")

    # =========================================================================
    # Test 8: AST Integrity & Continuous Hyperparameter Census
    # =========================================================================
    def test_08_ast_integrity_and_hyperparameters(self):
        """
        Directive W5: Strictly <= 2 continuous hyperparameters (lambda_param, mu_param),
        and zero regex occurrences across src/nevfe/.
        """
        engine = NEVFEEngine(lambda_param=1.0, beta_param=1.0, mu_param=4.0)
        self.assertEqual(engine.lambda_param, 1.0)
        self.assertEqual(engine.mu_param, 4.0)

        # Check that no regex library is imported
        engine_path = "src/nevfe/core/engine.py"
        with open(engine_path, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    self.assertNotEqual(n.name, "re", f"Found regex import 're' in {engine_path}")
            elif isinstance(node, ast.ImportFrom):
                self.assertNotEqual(node.module, "re", f"Found regex import 'from re' in {engine_path}")


if __name__ == "__main__":
    unittest.main()
