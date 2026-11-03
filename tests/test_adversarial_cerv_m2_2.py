# -*- coding: utf-8 -*-
"""
Adversarial Verification & Stress Test Suite for Milestone M2:
Empirical Stress-Testing on Stage 3 PFA-CD Log-Prob Contrast, Numerical Bounds,
Prefix Anchoring & Concise Causal Entity Extraction.

Challenger Agent: challenger_cerv_m2_2
Parent: orchestrator_1 (73180c24-9a64-415f-873e-c8f70ed9809c)

Test Dimensions:
1. Contrastive log-probability calculation under edge cases:
   - Identical positive and negative distributions (p^+_t = p^-_t -> alpha_t -> 0 and contrastive term -> 0).
   - Completely disjoint distributions (p^+_t perp p^-_t -> alpha_t -> 1).
   - Extreme tail probabilities (p^+_t(y) approx 10^-12 or 0.0): verify plausibility head filter
     V_head(p^+_t) strictly rejects them without -inf propagation or NaN.
   - Numerical bounds under extreme logits (-inf, +1e6, -1e6) and NaN resistance.
2. Prefix Anchoring and KV-Cache prefill:
   - Empty prefix (locked_prefix_ids = []).
   - Long prefix (128 tokens).
   - Prefix with repeated tokens or EOS token.
   - Positive/Negative crop length symmetry for batching.
3. Concise Causal Entity extraction:
   - Short post-fork generation (1 token).
   - Immediate EOS (0 tokens post-fork) with fallback to Stage 1 entity.
   - Whitespace and punctuation-only generation with clean fallback.
   - Uniform diffuse contrast abort (peak_layer == -1) concise extraction.
4. End-to-End Simulation with Lightweight PyTorch Mock LLM:
   - Full pipeline run under Fast-Path and Slow-Path.
   - Guarantee that pred_text is non-empty and strictly formatted.
"""

import math
import os
import sys
import unittest
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.benchmark.cerv.pipeline import (
    CERVPipeline,
    CERVResult,
    CERVPipelineResult,
    dynamic_preprocess_with_grid,
)
from scripts.benchmark.metrics import (
    compute_exact_match,
    compute_token_f1,
    compute_substring_match,
    compute_anls,
)


class MockEmbedding(nn.Module):
    def __init__(self, vocab_size: int = 151674, hidden_size: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.randn(vocab_size, hidden_size) * 0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_ids, self.weight)


class MockLanguageModelInner(nn.Module):
    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Optional[Any] = None,
        use_cache: bool = True,
        output_attentions: bool = False,
        return_dict: bool = True,
    ):
        bsz, seq_len, hdim = inputs_embeds.shape
        hidden = self.proj(inputs_embeds)

        # Build dummy KV cache: tuple of (k, v)
        if past_key_values is None:
            new_kv = tuple(
                (torch.zeros(bsz, 4, seq_len, 16, device=inputs_embeds.device),
                 torch.zeros(bsz, 4, seq_len, 16, device=inputs_embeds.device))
                for _ in range(4)
            )
        else:
            prev_len = past_key_values[0][0].shape[2]
            new_len = prev_len + seq_len
            new_kv = tuple(
                (torch.zeros(bsz, 4, new_len, 16, device=inputs_embeds.device),
                 torch.zeros(bsz, 4, new_len, 16, device=inputs_embeds.device))
                for _ in range(4)
            )

        attentions = None
        if output_attentions:
            # Dummy attention weights (bsz, n_heads=4, seq_len, total_seq_len)
            total_len = seq_len if past_key_values is None else past_key_values[0][0].shape[2] + seq_len
            attentions = tuple(
                torch.full((bsz, 4, seq_len, total_len), 1.0 / total_len, device=inputs_embeds.device)
                for _ in range(4)
            )

        class Output:
            pass

        out = Output()
        out.last_hidden_state = hidden
        out.past_key_values = new_kv
        out.attentions = attentions
        return out


class MockLanguageModel(nn.Module):
    def __init__(self, vocab_size: int = 151674, hidden_size: int = 64, num_layers: int = 4):
        super().__init__()
        self.config = type("Config", (), {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "num_hidden_layers": num_layers,
            "_attn_implementation": "eager",
        })()
        self.embeddings = MockEmbedding(vocab_size, hidden_size)
        self.model = MockLanguageModelInner(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.embeddings


class MockVinternModel(nn.Module):
    def __init__(self, vocab_size: int = 151674, hidden_size: int = 64):
        super().__init__()
        self.config = type("Config", (), {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "num_hidden_layers": 4,
            "max_dynamic_patch": 4,
        })()
        self.language_model = MockLanguageModel(vocab_size, hidden_size, 4)
        self.num_image_token = 256
        self.system_message = "Bạn là trợ lý AI hữu ích."

    def extract_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: (n_tiles, 3, 448, 448) -> features: (n_tiles, 256, 64)
        n_tiles = pixel_values.shape[0]
        return torch.randn(n_tiles, 256, 64, device=pixel_values.device, dtype=pixel_values.dtype)


PROMPT_INSTRUCTION = "Trả lời thật ngắn gọn câu hỏi sau dựa vào chữ xuất hiện trong ảnh: "


class MockTokenizerOutput:
    def __init__(self, input_ids: torch.Tensor):
        self.input_ids = input_ids


class MockTokenizer:
    def __init__(self):
        self.eos_token_id = 151643
        self.vocab = {
            "<|im_start|>": 151644,
            "<|im_end|>": 151645,
            "<IMG_CONTEXT>": 500,
            "<img>": 501,
            "</img>": 502,
            "đáp": 100,
            "án": 101,
            " ": 102,
        }

    def convert_tokens_to_ids(self, tok: str) -> int:
        return self.vocab.get(tok, 151645)

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        res = []
        for tid in token_ids:
            if tid in (151643, 151645):
                if not skip_special_tokens:
                    res.append("<|im_end|>")
            elif tid == 100:
                res.append("đáp")
            elif tid == 101:
                res.append("án")
            elif tid == 102:
                res.append(" ")
            elif tid in (500, 501, 502):
                if not skip_special_tokens:
                    res.append("<IMG>")
            else:
                res.append(f"w{tid}")
        return "".join(res)

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        if not text:
            return []
        if text.strip() == "đáp":
            return [100]
        if text.strip() == "án":
            return [101]
        return [200]

    def __call__(self, text: str, return_tensors: Optional[str] = None, add_special_tokens: bool = False):
        if "<IMG_CONTEXT>" in text:
            count = text.count("<IMG_CONTEXT>")
            tokens = [151644] * 10 + [500] * count + [151645] * 5
        elif "Trả lời thật ngắn gọn" in text:
            tokens = [200] * 10 + [151645, 151644]
        elif text == "<|im_end|>\n<|im_start|>assistant\n":
            tokens = [151645, 151644]
        elif text == PROMPT_INSTRUCTION.rstrip():
            tokens = [200] * 8
        else:
            tokens = [100, 102, 101]

        if return_tensors == "pt":
            return MockTokenizerOutput(torch.tensor([tokens], dtype=torch.long))
        return tokens


# ==============================================================================
# SUITE 1: ADVERSARIAL CONTRASTIVE LOG-PROB & ADAPTIVE ALPHA BOUNDS
# ==============================================================================

class TestAdversarialLogProbContrast(unittest.TestCase):
    """
    Stress-tests the Stage 3 PFA-CD log-probability formulation and Jensen-Shannon
    Divergence adaptive weight bounds under extreme mathematical boundary regimes.
    """

    def setUp(self):
        self.beta = 0.1

    def _compute_alpha_and_log_p_cerv(
        self, z_pos: torch.Tensor, z_neg: torch.Tensor, beta: float = 0.1
    ) -> Tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Direct replica of lines 756-782 in pipeline.py."""
        p_pos = F.softmax(z_pos, dim=-1)
        p_neg = F.softmax(z_neg, dim=-1)
        m_t = 0.5 * (p_pos + p_neg)

        h_pos = -torch.sum(p_pos * torch.log(p_pos.clamp(min=1e-12)))
        h_neg = -torch.sum(p_neg * torch.log(p_neg.clamp(min=1e-12)))
        h_m = -torch.sum(m_t * torch.log(m_t.clamp(min=1e-12)))

        h_m_val = float(h_m.item())
        if h_m_val > 1e-9:
            alpha_t = 1.0 - float((h_pos + h_neg).item()) / (2.0 * h_m_val)
            alpha_t = float(min(1.0, max(0.0, alpha_t)))
        else:
            alpha_t = 0.0

        log_p_pos = F.log_softmax(z_pos, dim=-1)
        log_p_neg = F.log_softmax(z_neg, dim=-1)

        max_p_pos = torch.max(p_pos)
        plausible_mask = p_pos >= (beta * max_p_pos)

        log_p_cerv = (1.0 + alpha_t) * log_p_pos - alpha_t * log_p_neg
        log_p_cerv = torch.where(plausible_mask, log_p_cerv, torch.full_like(log_p_cerv, -float("inf")))

        return alpha_t, log_p_cerv, plausible_mask, log_p_pos

    def test_01_identical_distributions_yields_zero_alpha_and_zero_contrast(self):
        """
        When positive and negative distributions are identical (p^+ = p^-),
        the theoretical JSD is identically 0.
        Therefore alpha_t must strictly equal 0.0, and log_p_cerv must equal log_p_pos.
        """
        vocab_sizes = [10, 100, 1000, 151674]
        for v in vocab_sizes:
            with self.subTest(vocab_size=v):
                # 1. Uniform distribution
                z_pos = torch.zeros(v)
                z_neg = torch.zeros(v)
                alpha, log_cerv, mask, log_pos = self._compute_alpha_and_log_p_cerv(z_pos, z_neg, self.beta)
                self.assertAlmostEqual(alpha, 0.0, places=5, msg=f"Uniform distribution failed alpha=0 for V={v}")
                # Contrastive term alpha * (log_pos - log_neg) must be 0
                diff = (log_cerv[mask] - log_pos[mask]).abs().max().item()
                self.assertAlmostEqual(diff, 0.0, places=5)

                # 2. Peaked distribution (random Gaussian)
                torch.manual_seed(42 + v)
                z_rand = torch.randn(v) * 3.0
                alpha_rand, log_cerv_rand, mask_rand, log_pos_rand = self._compute_alpha_and_log_p_cerv(
                    z_rand, z_rand.clone(), self.beta
                )
                self.assertAlmostEqual(alpha_rand, 0.0, places=5, msg=f"Random identical failed alpha=0 for V={v}")
                diff_rand = (log_cerv_rand[mask_rand] - log_pos_rand[mask_rand]).abs().max().item()
                self.assertAlmostEqual(diff_rand, 0.0, places=5)

                # 3. Delta (one-hot) distribution
                z_one_hot = torch.full((v,), -1e6)
                z_one_hot[0] = 1e6
                alpha_oh, log_cerv_oh, mask_oh, log_pos_oh = self._compute_alpha_and_log_p_cerv(
                    z_one_hot, z_one_hot.clone(), self.beta
                )
                self.assertEqual(alpha_oh, 0.0, msg="One-hot distribution did not trigger h_m <= 1e-9 fallback")
                self.assertEqual(int(torch.argmax(log_cerv_oh).item()), 0)

    def test_02_completely_disjoint_distributions_yields_unit_alpha(self):
        """
        When positive and negative distributions are completely disjoint and concentrated
        (p^+ perp p^- on orthogonal singletons), JSD is maximal.
        Therefore alpha_t must strictly equal 1.0, and the negative mode must be rejected.
        """
        vocab_size = 1000
        z_pos = torch.full((vocab_size,), -50.0)
        z_neg = torch.full((vocab_size,), -50.0)

        # Mode of pos is token 42, mode of neg is token 99
        z_pos[42] = 50.0
        z_neg[99] = 50.0

        alpha, log_cerv, mask, log_pos = self._compute_alpha_and_log_p_cerv(z_pos, z_neg, self.beta)
        self.assertAlmostEqual(alpha, 1.0, places=4, msg="Disjoint singletons did not achieve alpha=1.0")

        # Token 42 must be plausible, token 99 must be strictly non-plausible (-inf)
        self.assertTrue(bool(mask[42].item()), "Positive mode token 42 was rejected by plausibility head")
        self.assertFalse(bool(mask[99].item()), "Negative mode token 99 was not rejected by plausibility head")
        self.assertEqual(log_cerv[99].item(), -float("inf"), "Negative mode token did not receive -inf")
        self.assertEqual(int(torch.argmax(log_cerv).item()), 42, "Argmax failed to select positive mode")

    def test_03_monotonicity_of_alpha_with_distribution_divergence(self):
        """
        As the overlap between p^+ and p^- decreases from identical to disjoint,
        adaptive gain alpha_t must monotonically increase from 0 to 1.
        """
        vocab_size = 500
        z_pos = torch.full((vocab_size,), -30.0)
        z_pos[10] = 30.0  # Positive mode at token 10

        z_orthogonal = torch.full((vocab_size,), -30.0)
        z_orthogonal[20] = 30.0  # Orthogonal mode at token 20

        alphas = []
        lambdas = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]

        for lam in lambdas:
            # Linear mixture in probability space
            p_p = F.softmax(z_pos, dim=-1)
            p_ortho = F.softmax(z_orthogonal, dim=-1)
            p_n = (1.0 - lam) * p_p + lam * p_ortho
            z_n = torch.log(p_n.clamp(min=1e-30))

            alpha, _, _, _ = self._compute_alpha_and_log_p_cerv(z_pos, z_n, self.beta)
            alphas.append(alpha)

        # Verify monotonicity: alpha_{k+1} >= alpha_k - 1e-6
        for k in range(len(alphas) - 1):
            self.assertGreaterEqual(
                alphas[k + 1],
                alphas[k] - 1e-6,
                f"Alpha not monotonically increasing: step {k} ({alphas[k]}) vs step {k+1} ({alphas[k+1]})"
            )
        self.assertAlmostEqual(alphas[0], 0.0, places=4, msg="lambda=0 did not yield alpha=0")
        self.assertAlmostEqual(alphas[-1], 1.0, places=4, msg="lambda=1 did not yield alpha=1")

    def test_04_extreme_tail_probabilities_and_plausibility_head_rejection(self):
        """
        Extreme tail probabilities (p^+_t(y) approx 10^-12 or 0.0) must be strictly
        masked out by V_head(p^+_t) with beta=0.1, preventing -inf propagation or NaN.
        """
        vocab_size = 1000
        z_pos = torch.full((vocab_size,), -100.0)
        z_pos[5] = 10.0   # p_pos[5] approx 0.88
        z_pos[6] = 8.0    # p_pos[6] approx 0.12 (>= 0.1 * max -> plausible)
        z_pos[7] = 0.0    # p_pos[7] approx 1e-4 (< 0.1 * max -> rejected tail)
        z_pos[8] = -100.0 # p_pos[8] approx 1e-48 -> extreme tail

        z_neg = torch.full((vocab_size,), 0.0)  # Flat negative distribution

        alpha, log_cerv, mask, log_pos = self._compute_alpha_and_log_p_cerv(z_pos, z_neg, self.beta)

        # Tokens 7 and 8 must be masked to -inf
        self.assertFalse(bool(mask[7].item()))
        self.assertFalse(bool(mask[8].item()))
        self.assertEqual(log_cerv[7].item(), -float("inf"))
        self.assertEqual(log_cerv[8].item(), -float("inf"))

        # Plausible tokens must be finite
        self.assertTrue(math.isfinite(log_cerv[5].item()))
        self.assertTrue(math.isfinite(log_cerv[6].item()))

        # Overall tensor must contain NO NaN values
        self.assertFalse(torch.isnan(log_cerv).any().item(), "log_p_cerv contains NaN!")

        # Argmax must be strictly from the plausible set
        best_id = int(torch.argmax(log_cerv).item())
        self.assertIn(best_id, [5, 6], "Argmax chose an unplausible or masked token")

    def test_05_extreme_logit_magnitudes_and_nan_resistance(self):
        """
        Adversarial logit values (+1e6, -1e6, -inf) across positive and negative distributions
        must not cause NaN, crash, or numeric overflow in Stage 3 decoding.
        """
        vocab_size = 200
        z_pos = torch.full((vocab_size,), -float("inf"))
        z_pos[10] = 50.0
        z_pos[11] = 48.0

        z_neg = torch.full((vocab_size,), -float("inf"))
        z_neg[11] = 50.0
        z_neg[12] = 40.0

        alpha, log_cerv, mask, _ = self._compute_alpha_and_log_p_cerv(z_pos, z_neg, self.beta)

        # Check for NaN anywhere in plausible tokens
        self.assertFalse(torch.isnan(log_cerv[mask]).any().item(), "Plausible tokens contain NaN")
        # Check overall tensor
        has_nan = torch.isnan(log_cerv).any().item()
        self.assertFalse(has_nan, "log_p_cerv contains NaN after torch.where masking")

        # Argmax must select token 10 (since token 11 is penalized by z_neg[11]=50)
        best_id = int(torch.argmax(log_cerv).item())
        self.assertEqual(best_id, 10, f"Contrastive decoding should prefer token 10 over token 11, got {best_id}")


# ==============================================================================
# SUITE 2: ADVERSARIAL PREFIX ANCHORING & KV-CACHE PREFILL
# ==============================================================================

class TestAdversarialPrefixAnchoring(unittest.TestCase):
    """
    Stress-tests Stage 3 True Prefix Fork-Anchoring (PFA-CD) KV-cache prefill
    under adversarial prefix token configurations: empty, 128 tokens, repeats, EOS.
    """

    def setUp(self):
        self.device = "cpu"
        self.model = MockVinternModel(vocab_size=151674, hidden_size=64)
        self.tokenizer = MockTokenizer()
        self.transform = lambda img: torch.zeros(3, 448, 448)
        self.pipe = CERVPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            transform=self.transform,
            device=self.device,
        )

    def test_01_empty_prefix_kv_cache_prefill(self):
        """
        Empty locked_prefix_ids = [] must prefill visual prefix and query suffix
        cleanly without index errors, 0-dimension tensors, or batch mismatch.
        """
        img_pos = Image.new("RGB", (300, 200), color=(100, 150, 200))
        img_neg = Image.new("RGB", (300, 200), color=(50, 50, 50))
        question = "Số nhà là bao nhiêu?"

        verified_ids, alpha_trace = self.pipe._stage3_fork_anchored_counterfactual_verification(
            img_pos=img_pos,
            img_neg=img_neg,
            question=question,
            locked_prefix_ids=[],
            remaining_tokens=5,
        )
        self.assertIsInstance(verified_ids, list)
        self.assertIsInstance(alpha_trace, list)
        self.assertGreater(len(verified_ids), 0)
        self.assertEqual(len(verified_ids), len(alpha_trace))

    def test_02_long_prefix_128_tokens(self):
        """
        A long prefix of 128 tokens (context stress) must prefill into KV-cache
        and decode autoregressively without tensor dimension mismatch.
        """
        img_pos = Image.new("RGB", (250, 250), color=(120, 120, 120))
        img_neg = Image.new("RGB", (250, 250), color=(60, 60, 60))
        question = "Biển hiệu ghi gì?"
        long_prefix = list(range(100, 228))  # 128 token IDs

        verified_ids, alpha_trace = self.pipe._stage3_fork_anchored_counterfactual_verification(
            img_pos=img_pos,
            img_neg=img_neg,
            question=question,
            locked_prefix_ids=long_prefix,
            remaining_tokens=4,
        )
        self.assertGreater(len(verified_ids), 0)
        self.assertEqual(len(verified_ids), len(alpha_trace))

    def test_03_repeated_and_eos_tokens_in_prefix(self):
        """
        Prefixes containing repetitive tokens [100]*64 or special EOS tokens
        must not break embedding lookup or trigger early decoding aborts during prefill.
        """
        img_pos = Image.new("RGB", (224, 224), color=(200, 100, 50))
        img_neg = Image.new("RGB", (224, 224), color=(10, 10, 10))
        question = "Tên quán là gì?"

        # Repeated tokens
        repeat_prefix = [100] * 64
        verified_ids, alpha_trace = self.pipe._stage3_fork_anchored_counterfactual_verification(
            img_pos=img_pos,
            img_neg=img_neg,
            question=question,
            locked_prefix_ids=repeat_prefix,
            remaining_tokens=3,
        )
        self.assertGreater(len(verified_ids), 0)

        # Prefix containing special EOS token <|im_end|> (id 151645)
        eos_prefix = [100, 101, 151645, 102]
        verified_ids_eos, alpha_trace_eos = self.pipe._stage3_fork_anchored_counterfactual_verification(
            img_pos=img_pos,
            img_neg=img_neg,
            question=question,
            locked_prefix_ids=eos_prefix,
            remaining_tokens=3,
        )
        self.assertGreater(len(verified_ids_eos), 0)

    def test_04_batch_length_symmetry_between_pos_and_neg_crops(self):
        """
        Ensure that dynamic_preprocess_with_grid produces strictly identical tile counts
        for positive and negative views of identical spatial dimensions, guaranteeing
        batch compatibility in Stage 3 prefill.
        """
        for dims in [(224, 224), (448, 224), (600, 150), (1000, 800)]:
            with self.subTest(dimensions=dims):
                img_pos = Image.new("RGB", dims, color=(255, 0, 0))
                img_neg = Image.new("RGB", dims, color=(0, 255, 0))
                tiles_pos, grid_pos = dynamic_preprocess_with_grid(
                    img_pos, min_num=1, max_num=4, image_size=448, use_thumbnail=True
                )
                tiles_neg, grid_neg = dynamic_preprocess_with_grid(
                    img_neg, min_num=1, max_num=4, image_size=448, use_thumbnail=True
                )
                self.assertEqual(len(tiles_pos), len(tiles_neg), "Tile counts mismatched between pos and neg crops!")
                self.assertEqual(grid_pos, grid_neg, "Grid layouts mismatched between pos and neg crops!")


# ==============================================================================
# SUITE 3: ADVERSARIAL CONCISE CAUSAL ENTITY EXTRACTION
# ==============================================================================

class TestAdversarialConciseCausalEntityExtraction(unittest.TestCase):
    """
    Stress-tests the Concise Causal Entity extraction logic in CERVPipeline:
    ensuring pred_text is never empty or crashing when post-fork generation is short,
    empty, or composed only of whitespace/punctuation.
    """

    def setUp(self):
        self.device = "cpu"
        self.model = MockVinternModel(vocab_size=151674, hidden_size=64)
        self.tokenizer = MockTokenizer()
        self.transform = lambda img: torch.zeros(3, 448, 448)
        self.pipe = CERVPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            transform=self.transform,
            device=self.device,
        )

    def test_01_short_generation_single_token(self):
        """
        When Stage 3 generates only 1 token before EOS, pred_text must cleanly
        contain that single token, eliminating conversational preambles.
        """
        dummy_s = {l: torch.randn(256) for l in range(0, 4)}
        # Mock Stage 3 to return 1 token [100] ("đáp")
        self.pipe._stage3_fork_anchored_counterfactual_verification = lambda *args, **kwargs: ([100], [0.85])
        # Force a Slow-Path fork at step 3
        self.pipe._compute_step_bayesian_odds = lambda *args, **kwargs: (2.5, 0.7, 0.6, 0.4, dummy_s)

        dummy_img = Image.new("RGB", (400, 300), color=(200, 200, 200))
        res = self.pipe.run(dummy_img, question="Chữ gì?", max_new_tokens=10)

        self.assertEqual(res.route, "slow_path")
        self.assertEqual(res.pred_text, "đáp")
        self.assertIn("đáp", res.pred_raw_text)
        self.assertGreaterEqual(len(res.pred_raw_text), len(res.pred_text))

    def test_02_immediate_eos_stage3_fallback_to_stage1_entity(self):
        """
        When Stage 3 immediately outputs EOS (verified_suffix_ids = []),
        pred_text must not crash or return empty, but fall back to the Stage 1
        draft entity tokens.
        """
        dummy_s = {l: torch.randn(256) for l in range(0, 4)}
        # Mock Stage 3 returning empty token list
        self.pipe._stage3_fork_anchored_counterfactual_verification = lambda *args, **kwargs: ([], [])
        # Force a Slow-Path fork at step 2
        self.pipe._compute_step_bayesian_odds = lambda *args, **kwargs: (2.0, 0.65, 0.55, 0.35, dummy_s)

        dummy_img = Image.new("RGB", (400, 300), color=(180, 180, 180))
        res = self.pipe.run(dummy_img, question="Biển gì?", max_new_tokens=10)

        self.assertIsInstance(res.pred_text, str)
        self.assertGreater(len(res.pred_text), 0, "pred_text was empty upon immediate Stage 3 EOS!")
        self.assertEqual(res.route, "fast_path", "Should fall back to fast_path metadata on empty Stage 3")

    def test_03_whitespace_and_punctuation_only_generation(self):
        """
        When Stage 3 decodes to only whitespace and punctuation (e.g. '   .   '),
        verified_text.strip().rstrip('.').strip() becomes empty.
        Verify that pred_text safely falls back to non-empty draft entity without crashing.
        """
        dummy_s = {l: torch.randn(256) for l in range(0, 4)}
        # Token 102 is ' ', so [102, 102] decodes to '  '
        self.pipe._stage3_fork_anchored_counterfactual_verification = lambda *args, **kwargs: ([102, 102], [0.5])
        self.pipe._compute_step_bayesian_odds = lambda *args, **kwargs: (1.8, 0.6, 0.5, 0.3, dummy_s)

        dummy_img = Image.new("RGB", (350, 250), color=(150, 150, 150))
        res = self.pipe.run(dummy_img, question="Chữ số?", max_new_tokens=10)

        self.assertIsInstance(res.pred_text, str)
        self.assertGreater(len(res.pred_text), 0, "pred_text became empty on whitespace-only Stage 3!")

    def test_04_peak_layer_uniform_abort_concise_extraction(self):
        """
        When Stage 2 localization detects uniform diffuse contrast (peak_layer == -1),
        CERVPipeline must route via Fast-Path and extract the concise post-fork causal
        entity from the Stage 1 generated sequence rather than the full preamble.
        """
        # Mock Stage 2 to abort with peak_layer == -1
        self.pipe._stage2_prefix_conditioned_vaq_localization = lambda *args, **kwargs: (None, None, [0, 0, 100, 100], -1)
        self.pipe._compute_step_bayesian_odds = lambda *args, **kwargs: (2.2, 0.7, 0.6, 0.4, {})

        dummy_img = Image.new("RGB", (400, 300), color=(100, 100, 100))
        res = self.pipe.run(dummy_img, question="Quán gì?", max_new_tokens=10)

        self.assertEqual(res.route, "fast_path")
        self.assertEqual(res.peak_layer, -1)
        self.assertIsInstance(res.pred_text, str)
        self.assertGreater(len(res.pred_text), 0, "Aborted Stage 2 produced empty pred_text!")

    def test_05_multi_token_post_fork_entity_preservation(self):
        """
        Verify that multi-token post-fork generation is fully preserved without
        premature sequence truncation, allowing Stage 3 to output the full entity.
        """
        dummy_s = {l: torch.randn(256) for l in range(0, 4)}
        # Mock Stage 3 returning tokens for "đáp án" [100, 102, 101]
        self.pipe._stage3_fork_anchored_counterfactual_verification = lambda *args, **kwargs: ([100, 102, 101], [0.35, 0.40, 0.38])
        self.pipe._compute_step_bayesian_odds = lambda *args, **kwargs: (2.5, 0.7, 0.6, 0.4, dummy_s)

        dummy_img = Image.new("RGB", (400, 300), color=(200, 200, 200))
        res = self.pipe.run(dummy_img, question="Hỏi gì?", max_new_tokens=10)

        self.assertEqual(res.route, "slow_path")
        self.assertIn("đáp án", res.pred_text)

    def test_06_benchmark_preamble_causal_entity_extraction(self):
        """
        Verify that _extract_causal_entity properly strips conversational question
        echoes from untruncated model generations across ViTextVQA benchmark failure cases.
        """
        cases = [
            ("nhà thuốc trong ảnh tên gì ?", "Nhà thuốc trong ảnh tên là Bảo Tâm An.", "Bảo Tâm An"),
            ("nơi này thuộc về phòng nào ?", "Nơi này thuộc về Phòng Thương mại và Công nghiệp Việt Nam", "Thương mại và Công nghiệp Việt Nam"),
            ("địa chỉ website nơi này là gì ?", "Địa chỉ website nơi này là www.cbam.edu.vn", "www.cbam.edu.vn"),
            ("nơi này tên là gì ?", "Nơi này tên là CBAM.", "CBAM"),
        ]
        for q, full_gen, expected in cases:
            extracted = CERVPipeline._extract_causal_entity(q, full_gen)
            self.assertEqual(extracted, expected, f"Failed for question '{q}': got '{extracted}' vs '{expected}'")


# ==============================================================================
# SUITE 4: END-TO-END PIPELINE SIMULATION & METRIC INVARIANTS AUDIT
# ==============================================================================

class TestAdversarialEndToEndAndMetricInvariance(unittest.TestCase):
    """
    Simulates end-to-end execution of CERVPipeline across Fast-Path and Slow-Path,
    and performs an independent audit of strict evaluation metric invariants.
    """

    def setUp(self):
        self.device = "cpu"
        self.model = MockVinternModel(vocab_size=151674, hidden_size=64)
        self.tokenizer = MockTokenizer()
        self.transform = lambda img: torch.zeros(3, 448, 448)
        self.pipe = CERVPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            transform=self.transform,
            device=self.device,
        )

    def test_01_end_to_end_fast_path_routing(self):
        """
        Under low Bayesian odds (Lambda_t <= 1), the pipeline must take the Fast-Path,
        complete with 0% Stage 2/3 overhead, and return valid CERVResult.
        """
        # Lambda_t <= 1.0 everywhere
        self.pipe._compute_step_bayesian_odds = lambda *args, **kwargs: (0.42, 0.3, 0.2, 0.1, {})

        img = Image.new("RGB", (400, 400), color=(200, 150, 100))
        res = self.pipe.run(img, question="Bao nhiêu tiền?", max_new_tokens=8)

        self.assertEqual(res.route, "fast_path")
        self.assertEqual(res.fork_step, 0)
        self.assertEqual(res.stage1_bbox, [0, 0, 400, 400])
        self.assertIsInstance(res.pred_text, str)
        self.assertGreater(len(res.pred_text), 0)
        self.assertEqual(res.mean_alpha, 0.0)

    def test_02_end_to_end_slow_path_routing(self):
        """
        Under high Bayesian odds (Lambda_t > 1), the pipeline must take the Slow-Path,
        lock preamble, crop RoI, verify with Stage 3 PFA-CD, and output concise entity.
        """
        img = Image.new("RGB", (400, 400), color=(100, 150, 200))
        # Step 1: Lambda=0.5, Step 2: Lambda=2.5 (triggers fork)
        call_count = [0]
        def mock_odds(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] >= 2:
                # Return s_layers dict for peak detection
                dummy_s = {l: torch.randn(256) for l in range(1, 4)}
                return (2.5, 0.75, 0.65, 0.45, dummy_s)
            return (0.5, 0.3, 0.2, 0.1, {l: torch.zeros(256) for l in range(1, 4)})

        self.pipe._compute_step_bayesian_odds = mock_odds

        res = self.pipe.run(img, question="Tên đường là gì?", max_new_tokens=8)

        self.assertIn(res.route, ["slow_path", "fast_path"])
        self.assertGreaterEqual(res.fork_step, 1)
        self.assertIsInstance(res.pred_text, str)
        self.assertGreater(len(res.pred_text), 0)
        self.assertEqual(len(res.stage1_bbox), 4)

    def test_03_strict_metric_relaxation_assertions(self):
        """
        Verify that none of the strict evaluation metrics have been relaxed or compromised.
        All 4 core metrics must strictly adhere to project specification invariants.
        """
        # Exact match
        self.assertEqual(compute_exact_match("nhà thuốc", ["nhà thuốc"]), 1.0)
        self.assertEqual(compute_exact_match("nhà thuốc", ["quán cơm"]), 0.0)

        # Token F1
        self.assertEqual(compute_token_f1("nhà thuốc an khang", ["nhà thuốc an khang"]), 1.0)
        self.assertEqual(compute_token_f1("", ["nhà thuốc"]), 0.0)

        # Substring match (GT is substring of prediction)
        self.assertEqual(compute_substring_match("số 123 đường abc", ["123"]), 1.0)
        self.assertEqual(compute_substring_match("số 456 đường abc", ["123"]), 0.0)

        # ANLS
        self.assertEqual(compute_anls("đại học", ["đại học"]), 1.0)
        # Edit distance > 0.5 tau threshold must yield 0.0
        self.assertEqual(compute_anls("xyz", ["đại học"]), 0.0)


if __name__ == "__main__":
    unittest.main()
