# -*- coding: utf-8 -*-
"""
Unit and Mock Integration Test Suite for TICKET-OPT-03: Image-Level Working-Map Caching.

Covers all mandatory verification gates (Tests A through H) in local synthetic/mock mode
without loading real VLM weights:
- Test A: First question / cache miss computes `_compute_working_maps` once and stores under `resolved_image_key`.
- Test B: Same image / multiple questions reuse cached working maps (`_compute_working_maps` count == 1)
          while recomputing question-conditioned attention, seed selection, and ROI propagation per question.
- Test C: Image transition invalidates previous maps, computes new maps once, and retains only one active image cache.
- Test D: Reset lifecycle (`reset_cache()`) clears cached maps and cache key, forcing recomputation on the next run.
- Test E: Cached vs. uncached exact parity across `SeedPropagationResult` (all fields, prototype arrays, masks,
          compatibility maps) and downstream `PipelineResult` (`stage1_bboxes`, `char_heights`, `crop`, `pred_text`).
- Test F: Repeated use and immutability — verifies cached tensors are never mutated in-place across questions
          with zero extra clone/copy operations required.
- Test G: Failure safety — simulated exception in `_compute_working_maps` during image transition clears prior
          cache and never reuses stale maps under the new image identity.
- Test H: Regression, backward compatibility for `propagate`/`propagate_multi` callers, and non-accumulation of
          memory references across a sequence of distinct synthetic images.
"""

import gc
import os
import sys
import unittest
from unittest.mock import patch
import weakref

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.benchmark.sg_vcd.adaptive_propagator import (
    AdaptiveScalePropagator,
    BaseSeedPropagator,
    SeedPropagationResult,
    _is_valid_working_maps,
)
from scripts.benchmark.sg_vcd.attention_extractor import (
    BaseAttentionExtractor,
    ExtractionResult,
)
from scripts.benchmark.sg_vcd.feature_cache import (
    VisualFeatureCache,
    build_image_cache_identity,
    default_feature_cache,
)
from scripts.benchmark.sg_vcd.pipeline import SGVCDPipeline


def _identity_transform(pil_img: Image.Image) -> torch.Tensor:
    """Fast synthetic transform returning a (3, 16, 16) tensor from a PIL Image."""
    arr = np.asarray(pil_img.convert("RGB"), dtype=np.float32)
    h, w, _ = arr.shape
    cy, cx = min(16, h), min(16, w)
    patch = arr[:cy, :cx, :] / 255.0
    return torch.from_numpy(patch).permute(2, 0, 1).contiguous()


def _make_synthetic_signboard_image(
    width: int = 320,
    height: int = 240,
    bg_color: tuple = (35, 40, 48),
    top_board_bg: tuple = (25, 95, 190),
    top_stroke_fg: tuple = (250, 245, 210),
    bot_board_bg: tuple = (175, 30, 35),
    bot_stroke_fg: tuple = (255, 255, 255),
) -> Image.Image:
    """
    Create a deterministic synthetic scene-text image with two distinct horizontal signboards:
    - Top signboard around y in [28, 76], x in [32, 270] with character-like vertical stroke bars.
    - Bottom signboard around y in [148, 198], x in [48, 284] with character-like vertical stroke bars.
    """
    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Top signboard surface + character stroke sequence
    draw.rectangle([28, 26, 276, 78], fill=top_board_bg)
    for x in range(42, 260, 12):
        draw.rectangle([x, 38, x + 5, 64], fill=top_stroke_fg)
        draw.rectangle([x, 38, x + 8, 42], fill=top_stroke_fg)

    # Bottom signboard surface + character stroke sequence
    draw.rectangle([44, 146, 290, 202], fill=bot_board_bg)
    for x in range(58, 274, 14):
        draw.rectangle([x, 160, x + 6, 188], fill=bot_stroke_fg)
        draw.rectangle([x, 172, x + 9, 176], fill=bot_stroke_fg)

    return img


class _QuestionConditionedMockExtractor(BaseAttentionExtractor):
    """
    Question-conditioned mock attention extractor that places the attention hotspot on:
    - Top signboard when question contains 'top' or 'q1' or 'q3'
    - Bottom signboard when question contains 'bot' or 'q2' or 'q4'
    Tracks exact `extract()` call counts and question arguments.
    """

    def __init__(self, grid_ratio=(2, 2)):
        self.grid_ratio = grid_ratio
        self.extract_calls = 0
        self.questions_seen = []
        self._cached_img_key = None
        self._cached_img_state = None

    def clear_cache(self) -> None:
        self._cached_img_key = None
        self._cached_img_state = None

    def extract(self, image: Image.Image, question: str, **kwargs) -> ExtractionResult:
        self.extract_calls += 1
        self.questions_seen.append(question)
        self._cached_img_key = kwargs.get("image_key")
        self._cached_img_state = {"question_count": self.extract_calls}

        gx, gy = self.grid_ratio
        H_g, W_g = 16 * gy, 16 * gx
        phi = np.full((H_g, W_g), 0.01, dtype=np.float32)

        q_low = question.lower()
        if any(tok in q_low for tok in ("bot", "q2", "q4")):
            # Hotspot on bottom signboard: y around 70% of height, x around 50% of width
            hy0, hy1 = int(0.65 * H_g), int(0.80 * H_g)
            hx0, hx1 = int(0.25 * W_g), int(0.75 * W_g)
            phi[hy0:hy1, hx0:hx1] = 2.6
            phi[(hy0 + hy1) // 2, (hx0 + hx1) // 2] = 4.2
        else:
            # Hotspot on top signboard: y around 22% of height, x around 45% of width
            hy0, hy1 = int(0.14 * H_g), int(0.30 * H_g)
            hx0, hx1 = int(0.18 * W_g), int(0.70 * W_g)
            phi[hy0:hy1, hx0:hx1] = 2.4
            phi[(hy0 + hy1) // 2, (hx0 + hx1) // 2] = 4.0

        r_map = phi.copy()
        m_aff = np.ones_like(phi)
        return ExtractionResult(
            phi_map=phi,
            r_map=r_map,
            aff_map=m_aff,
            a_query_map=phi.copy(),
            a_gen_map=np.full_like(phi, 1.0 / float(H_g * W_g)),
            grid_ratio=self.grid_ratio,
        )


class _DummyAttentionOutput:
    def __init__(
        self,
        seq_len: int,
        hidden_dim: int = 32,
        num_layers: int = 2,
        batch_size: int = 1,
        total_len: int = None,
    ):
        t_len = total_len if total_len is not None else seq_len
        self.last_hidden_state = torch.ones((batch_size, seq_len, hidden_dim), dtype=torch.bfloat16)
        self.past_key_values = tuple(
            (
                torch.ones((batch_size, 2, t_len, hidden_dim // 2), dtype=torch.bfloat16),
                torch.ones((batch_size, 2, t_len, hidden_dim // 2), dtype=torch.bfloat16),
            )
            for _ in range(num_layers)
        )


class _DummyInnerLLM(nn.Module):
    def __init__(self, hidden_dim: int = 32, vocab_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.embed_lookup = nn.Embedding(vocab_size, hidden_dim, dtype=torch.bfloat16)
        self.config = type("Cfg", (), {"_attn_implementation": "sdpa"})()

    def get_input_embeddings(self):
        return self.embed_lookup

    def forward(
        self,
        inputs_embeds=None,
        past_key_values=None,
        use_cache=True,
        output_attentions=False,
        return_dict=True,
    ):
        batch_size = int(inputs_embeds.shape[0])
        q_len = inputs_embeds.shape[1]
        past_len = 0
        if past_key_values is not None and len(past_key_values) > 0:
            past_len = past_key_values[0][0].shape[2]
        total_len = past_len + q_len
        return _DummyAttentionOutput(
            q_len,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            batch_size=batch_size,
            total_len=total_len,
        )


class _DummyLanguageModel(nn.Module):
    def __init__(self, hidden_dim: int = 32, vocab_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.model = _DummyInnerLLM(hidden_dim=hidden_dim, vocab_size=vocab_size, num_layers=num_layers)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False, dtype=torch.bfloat16)
        # Deterministic weights so greedy decoding finishes cleanly
        with torch.no_grad():
            self.lm_head.weight.zero_()
            self.lm_head.weight[3, :] = 5.0  # Favor EOS token id 3
        self.config = self.model.config

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()


class _DummyVinternModel(nn.Module):
    def __init__(self, hidden_dim: int = 32, vocab_size: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.extract_feature_calls = 0
        self.language_model = _DummyLanguageModel(hidden_dim=hidden_dim, vocab_size=vocab_size)
        self.vision_model = None

    def extract_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
        self.extract_feature_calls += 1
        num_tiles = pixel_values.shape[0]
        return torch.ones(
            (num_tiles, 256, self.hidden_dim),
            dtype=torch.bfloat16,
            device=pixel_values.device,
        )


class _DummyTokenizer:
    eos_token_id = 2
    im_end_id = 3
    img_ctx_id = 4

    def __call__(self, text: str, return_tensors: str = "pt", add_special_tokens: bool = False):
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        return type("TokOut", (), {"input_ids": torch.tensor([ids], dtype=torch.long)})()

    def encode(self, text: str, add_special_tokens: bool = False):
        if not text:
            return [10, 11]
        ctx_count = text.count("<IMG_CONTEXT>")
        if ctx_count > 0:
            return [8, 9] + ([self.img_ctx_id] * ctx_count) + [10, 11]
        return [min(60, 12 + (ord(c) % 40)) for c in text[:8]] or [10]

    def convert_tokens_to_ids(self, token: str) -> int:
        if token == "<IMG_CONTEXT>":
            return self.img_ctx_id
        return 3

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        return "nhà thuốc bảo tâm an"


class TestOPT03WorkingMapCache(unittest.TestCase):
    """Comprehensive verification of TICKET-OPT-03: Image-Level Working-Map Caching (Tests A–H)."""

    def setUp(self):
        default_feature_cache.clear()
        self.model = _DummyVinternModel()
        self.tokenizer = _DummyTokenizer()
        self.extractor = _QuestionConditionedMockExtractor(grid_ratio=(2, 2))
        self.propagator = AdaptiveScalePropagator(work_max_side=320, device="cpu")
        self.pipeline = SGVCDPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            transform=_identity_transform,
            device="cpu",
            extractor=self.extractor,
            propagator=self.propagator,
            feature_cache=VisualFeatureCache(max_entries=8),
        )
        # Provide _build_prefix_and_suffix_embeds on custom mock extractor for Stage 2
        self.extractor._build_prefix_and_suffix_embeds = (
            lambda vit_embeds, q: (
                torch.ones((1, 16, self.model.hidden_dim), dtype=torch.bfloat16),
                torch.ones((1, 8, self.model.hidden_dim), dtype=torch.bfloat16),
                2,
                10,
            )
        )

    def _assert_seed_propagation_result_equal(
        self, res_a: SeedPropagationResult, res_b: SeedPropagationResult
    ):
        """Assert exact bitwise/value equality across all fields of two SeedPropagationResult objects."""
        self.assertEqual(res_a.seed_pixel, res_b.seed_pixel)
        self.assertEqual(res_a.seed_patch, res_b.seed_patch)
        self.assertEqual(res_a.propagated_patch_bbox, res_b.propagated_patch_bbox)
        self.assertEqual(res_a.propagated_pixel_bbox, res_b.propagated_pixel_bbox)
        self.assertEqual(res_a.core_pixel_bbox, res_b.core_pixel_bbox)

        pa, pb = res_a.prototype, res_b.prototype
        self.assertEqual(pa.seed_pixel, pb.seed_pixel)
        self.assertEqual(pa.seed_patch, pb.seed_patch)
        self.assertEqual(pa.char_height_px, pb.char_height_px)
        self.assertEqual(pa.char_height_patches, pb.char_height_patches)
        self.assertEqual(pa.orientation, pb.orientation)
        self.assertEqual(pa.stroke_scale_ref, pb.stroke_scale_ref)
        self.assertTrue(np.array_equal(pa.mu_fg, pb.mu_fg))
        self.assertTrue(np.array_equal(pa.mu_bg, pb.mu_bg))
        self.assertTrue(np.array_equal(pa.delta_mu, pb.delta_mu))

        self.assertTrue(np.array_equal(res_a.patch_mask, res_b.patch_mask))
        self.assertTrue(np.array_equal(res_a.compatibility_map, res_b.compatibility_map))

    def test_A_first_question_cache_miss_stores_under_resolved_image_key(self):
        """
        Test A — First question/cache miss:
        - A new image causes `_compute_working_maps` to execute once.
        - The returned tuple is stored under the correct `resolved_image_key`.
        """
        img = _make_synthetic_signboard_image()
        expected_key = build_image_cache_identity(
            img,
            image_id=101,
            image_path="data/vitextvqa/st_images/101.jpg",
            dataset_version="vitextvqa:dev",
        )
        self.assertIsNone(self.pipeline._cached_working_maps)
        self.assertIsNone(self.pipeline._cached_working_maps_key)

        with patch.object(
            self.propagator,
            "_compute_working_maps",
            wraps=self.propagator._compute_working_maps,
        ) as spy_compute:
            res = self.pipeline.run(
                img,
                "Q1 top signboard?",
                max_new_tokens=4,
                image_id=101,
                image_path="data/vitextvqa/st_images/101.jpg",
                dataset_version="vitextvqa:dev",
            )
            self.assertEqual(spy_compute.call_count, 1)

        self.assertIsNotNone(res)
        self.assertTrue(_is_valid_working_maps(self.pipeline._cached_working_maps))
        self.assertEqual(self.pipeline._cached_working_maps_key, expected_key)

    def test_B_same_image_multiple_questions_reuses_cache_and_runs_question_specific_steps(self):
        """
        Test B — Same image/multiple questions:
        - Four questions using the same image identity reuse the same cached maps.
        - `_compute_working_maps` executes exactly once for that image.
        - Question-dependent attention, seed selection, and ROI propagation still execute for each question.
        """
        img = _make_synthetic_signboard_image()
        questions = [
            "Q1 top signboard text?",
            "Q2 bottom signboard text?",
            "Q3 top signboard phone?",
            "Q4 bottom signboard address?",
        ]

        with (
            patch.object(
                self.propagator,
                "_compute_working_maps",
                wraps=self.propagator._compute_working_maps,
            ) as spy_compute,
            patch.object(
                self.propagator,
                "propagate_multi",
                wraps=self.propagator.propagate_multi,
            ) as spy_prop_multi,
            patch.object(
                self.pipeline.selector,
                "select_top_k",
                wraps=self.pipeline.selector.select_top_k,
            ) as spy_select,
        ):
            results = []
            cached_ids = []
            for q in questions:
                r = self.pipeline.run(
                    img,
                    q,
                    max_new_tokens=4,
                    image_id=202,
                    image_path="data/vitextvqa/st_images/202.jpg",
                    dataset_version="vitextvqa:dev",
                )
                results.append(r)
                cached_ids.append(id(self.pipeline._cached_working_maps))

            # Working maps computed ONCE across all 4 questions on the same image
            self.assertEqual(spy_compute.call_count, 1)
            self.assertEqual(len(set(cached_ids)), 1)

            # Question-conditioned stages executed 4 times (once per question)
            self.assertEqual(self.extractor.extract_calls, 4)
            self.assertEqual(spy_select.call_count, 4)
            self.assertEqual(spy_prop_multi.call_count, 4)

            # Verify that top-question vs bottom-question produced different question-specific Stage 1 bboxes
            self.assertNotEqual(results[0].stage1_bbox, results[1].stage1_bbox)
            self.assertEqual(results[0].stage1_bbox, results[2].stage1_bbox)
            self.assertEqual(results[1].stage1_bbox, results[3].stage1_bbox)

    def test_C_image_transition_invalidates_previous_maps_and_retains_single_entry(self):
        """
        Test C — Image transition:
        - A different image identity invalidates the previous maps.
        - The new image's maps are computed exactly once.
        - Only one active image cache remains.
        """
        img1 = _make_synthetic_signboard_image(width=320, height=240)
        img2 = _make_synthetic_signboard_image(
            width=360,
            height=260,
            top_board_bg=(40, 140, 60),
            bot_board_bg=(90, 40, 150),
        )
        key1 = build_image_cache_identity(img1, image_id=1, dataset_version="v1")
        key2 = build_image_cache_identity(img2, image_id=2, dataset_version="v1")

        with patch.object(
            self.propagator,
            "_compute_working_maps",
            wraps=self.propagator._compute_working_maps,
        ) as spy_compute:
            # Image 1: 2 questions -> 1 working-map computation
            self.pipeline.run(img1, "Q1 top", max_new_tokens=4, image_key=key1)
            maps_img1_ref = weakref.ref(self.pipeline._cached_working_maps[0])
            self.pipeline.run(img1, "Q2 bot", max_new_tokens=4, image_key=key1)
            self.assertEqual(spy_compute.call_count, 1)
            self.assertEqual(self.pipeline._cached_working_maps_key, key1)

            # Transition to Image 2: 2 questions -> 1 new working-map computation (total = 2)
            self.pipeline.run(img2, "Q1 top", max_new_tokens=4, image_key=key2)
            self.pipeline.run(img2, "Q2 bot", max_new_tokens=4, image_key=key2)
            self.assertEqual(spy_compute.call_count, 2)
            self.assertEqual(self.pipeline._cached_working_maps_key, key2)

        gc.collect()
        # Confirm previous image's working-map tensor has no lingering reference in pipeline
        self.assertIsNone(maps_img1_ref())

    def test_D_reset_cache_clears_working_maps_and_key(self):
        """
        Test D — Reset lifecycle:
        - `reset_cache()` clears cached maps and the cache key.
        - The next run recomputes maps instead of reusing stale tensors.
        """
        img = _make_synthetic_signboard_image()
        key = build_image_cache_identity(img, image_id=77, dataset_version="v1")

        with patch.object(
            self.propagator,
            "_compute_working_maps",
            wraps=self.propagator._compute_working_maps,
        ) as spy_compute:
            self.pipeline.run(img, "Q1 top", max_new_tokens=4, image_key=key)
            self.assertEqual(spy_compute.call_count, 1)
            self.assertIsNotNone(self.pipeline._cached_working_maps)
            self.assertEqual(self.pipeline._cached_working_maps_key, key)

            self.pipeline.reset_cache()
            self.assertIsNone(self.pipeline._cached_working_maps)
            self.assertIsNone(self.pipeline._cached_working_maps_key)

            # Next run on the exact same image & key must recompute working maps
            self.pipeline.run(img, "Q1 top", max_new_tokens=4, image_key=key)
            self.assertEqual(spy_compute.call_count, 2)

    def test_E_cached_versus_uncached_exact_parity(self):
        """
        Test E — Cached versus uncached parity:
        - Run equivalent synthetic inputs with cache enabled and disabled.
        - Assert exact equality of `SeedPropagationResult` bounding boxes, character heights,
          prototypes, patch masks, compatibility maps, and downstream `PipelineResult` fields.
        """
        img = _make_synthetic_signboard_image()
        seeds = [(12, 7), (16, 22), (8, 7), (20, 23)]
        grid_ratio = (2, 2)

        # 1. Direct propagator parity: precomputed working_maps vs fresh working_maps per call
        shared_maps = self.propagator._compute_working_maps(img)
        cached_multi_res = self.propagator.propagate_multi(
            image=img,
            seed_patches=seeds,
            grid_ratio=grid_ratio,
            working_maps=shared_maps,
        )
        uncached_single_res = [
            self.propagator.propagate(
                image=img,
                seed_patch=sp,
                grid_ratio=grid_ratio,
                working_maps=None,
            )
            for sp in seeds
        ]
        self.assertEqual(len(cached_multi_res), len(uncached_single_res))
        for r_cached, r_uncached in zip(cached_multi_res, uncached_single_res):
            self._assert_seed_propagation_result_equal(r_cached, r_uncached)

        # 2. Full SGVCDPipeline.run() parity across 4 questions on the same image
        questions = [
            "Q1 top signboard?",
            "Q2 bottom signboard?",
            "Q3 top signboard second query?",
            "Q4 bottom signboard second query?",
        ]
        key = build_image_cache_identity(img, image_id=303, dataset_version="v1")

        # Run with working-map cache active across all 4 questions
        self.pipeline.reset_cache()
        cached_pipe_results = [
            self.pipeline.run(img, q, max_new_tokens=4, image_key=key)
            for q in questions
        ]

        # Run uncached by resetting cache before every single question
        uncached_pipe_results = []
        for q in questions:
            self.pipeline.reset_cache()
            uncached_pipe_results.append(
                self.pipeline.run(img, q, max_new_tokens=4, image_key=key)
            )

        for res_c, res_u in zip(cached_pipe_results, uncached_pipe_results):
            self.assertEqual(res_c.stage1_bbox, res_u.stage1_bbox)
            self.assertEqual(res_c.stage1_bboxes, res_u.stage1_bboxes)
            self.assertEqual(res_c.selection.pixel_bbox, res_u.selection.pixel_bbox)
            self.assertEqual(res_c.selection.grid_bbox, res_u.selection.grid_bbox)
            self.assertEqual(res_c.selection.kl_score, res_u.selection.kl_score)
            self.assertEqual(res_c.multi_selection.pixel_bboxes, res_u.multi_selection.pixel_bboxes)
            self.assertEqual(res_c.multi_selection.grid_bboxes, res_u.multi_selection.grid_bboxes)
            self.assertEqual(res_c.multi_selection.kl_scores, res_u.multi_selection.kl_scores)
            self.assertEqual(res_c.crop.isotropic_box, res_u.crop.isotropic_box)
            self.assertEqual(res_c.crop.magnification, res_u.crop.magnification)
            self.assertEqual(res_c.pred_text, res_u.pred_text)
            self.assertEqual(res_c.pred_sg_vcd, res_u.pred_sg_vcd)

    def test_F_repeated_use_and_immutability_zero_mutation_zero_clone(self):
        """
        Test F — Repeated use and immutability:
        - Reuse cached maps across multiple questions (top seed -> bottom seed -> top seed).
        - Confirm one call cannot mutate the cached tuple tensors and affect subsequent results.
        """
        img = _make_synthetic_signboard_image()
        key = build_image_cache_identity(img, image_id=404, dataset_version="v1")

        res_q1 = self.pipeline.run(img, "Q1 top signboard", max_new_tokens=4, image_key=key)
        wm = self.pipeline._cached_working_maps
        self.assertTrue(_is_valid_working_maps(wm))

        # Take deep snapshots of all 4 working-map tensors after Q1
        snap_opp = wm[0].clone()
        snap_bg = wm[1].clone()
        snap_stroke = wm[2].clone()
        snap_horiz = wm[3].clone()
        snap_sx, snap_sy = wm[4], wm[5]

        # Run Q2 (bottom signboard) and Q3 (top signboard again) reusing the exact same cached tuple
        res_q2 = self.pipeline.run(img, "Q2 bottom signboard", max_new_tokens=4, image_key=key)
        res_q3 = self.pipeline.run(img, "Q1 top signboard", max_new_tokens=4, image_key=key)

        wm_after = self.pipeline._cached_working_maps
        self.assertIs(wm_after, wm)
        self.assertTrue(torch.equal(wm_after[0], snap_opp))
        self.assertTrue(torch.equal(wm_after[1], snap_bg))
        self.assertTrue(torch.equal(wm_after[2], snap_stroke))
        self.assertTrue(torch.equal(wm_after[3], snap_horiz))
        self.assertEqual((wm_after[4], wm_after[5]), (snap_sx, snap_sy))

        # Q3 (after Q2) must produce identical localization to Q1
        self.assertEqual(res_q1.stage1_bbox, res_q3.stage1_bbox)
        self.assertEqual(res_q1.stage1_bboxes, res_q3.stage1_bboxes)
        self.assertNotEqual(res_q1.stage1_bbox, res_q2.stage1_bbox)

    def test_G_failure_safety_on_image_transition_exception(self):
        """
        Test G — Failure safety:
        - Simulate `_compute_working_maps` raising an exception during an image transition.
        - Verify that stale maps from the previous image are cleared before computation and
          never reused under the new image identity.
        """
        img1 = _make_synthetic_signboard_image(width=320, height=240)
        img2 = _make_synthetic_signboard_image(width=360, height=260)
        key1 = build_image_cache_identity(img1, image_id=501, dataset_version="v1")
        key2 = build_image_cache_identity(img2, image_id=502, dataset_version="v1")

        # 1. Populate valid cache for Image 1
        self.pipeline.run(img1, "Q1 top", max_new_tokens=4, image_key=key1)
        self.assertEqual(self.pipeline._cached_working_maps_key, key1)
        self.assertIsNotNone(self.pipeline._cached_working_maps)

        # 2. Simulate _compute_working_maps raising RuntimeError during transition to Image 2
        with patch.object(
            self.propagator,
            "_compute_working_maps",
            side_effect=RuntimeError("Simulated GPU working-map failure"),
        ):
            res_fail = self.pipeline.run(img2, "Q1 top", max_new_tokens=4, image_key=key2)
            # Pipeline gracefully falls back to selection.pixel_bbox
            self.assertEqual(res_fail.stage1_bbox, list(res_fail.selection.pixel_bbox))

        # Stale cache from Image 1 must have been cleared and NOT associated with key2 or key1
        self.assertIsNone(self.pipeline._cached_working_maps)
        self.assertIsNone(self.pipeline._cached_working_maps_key)

        # 3. Subsequent call on Image 2 with working propagator computes fresh maps for Image 2
        with patch.object(
            self.propagator,
            "_compute_working_maps",
            wraps=self.propagator._compute_working_maps,
        ) as spy_recover:
            self.pipeline.run(img2, "Q2 bot", max_new_tokens=4, image_key=key2)
            self.assertEqual(spy_recover.call_count, 1)
            self.assertEqual(self.pipeline._cached_working_maps_key, key2)
            self.assertTrue(_is_valid_working_maps(self.pipeline._cached_working_maps))

    def test_H_regression_backward_compatibility_and_non_accumulation_across_images(self):
        """
        Test H — Regression & Memory Non-Accumulation:
        - Verify callers of `propagate_multi` without `working_maps` work unchanged.
        - Verify custom `BaseSeedPropagator` subclasses without `_compute_working_maps` work unchanged.
        - Verify cache memory references do not accumulate across a sequence of 6 distinct images.
        """
        img = _make_synthetic_signboard_image()
        # 1. Direct propagate_multi without working_maps (legacy caller path)
        res_legacy = self.propagator.propagate_multi(
            image=img,
            seed_patches=[(12, 7), (16, 22)],
            grid_ratio=(2, 2),
        )
        self.assertEqual(len(res_legacy), 2)

        # 2. Custom BaseSeedPropagator subclass without _compute_working_maps
        class _MinimalCustomPropagator(BaseSeedPropagator):
            def __init__(self):
                self.calls = 0

            def propagate(
                self,
                image,
                seed_patch,
                grid_ratio,
                seed_pixel=None,
                feat_grid=None,
                working_maps=None,
            ):
                self.calls += 1
                return None

        custom_prop = _MinimalCustomPropagator()
        custom_pipe = SGVCDPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            transform=_identity_transform,
            device="cpu",
            extractor=self.extractor,
            propagator=custom_prop,
        )
        custom_pipe.run(img, "Q1 top", max_new_tokens=4, image_id=900, dataset_version="v1")
        self.assertGreaterEqual(custom_prop.calls, 1)
        self.assertIsNone(custom_pipe._cached_working_maps)

        # 3. Sequence of 6 distinct synthetic images: verify old working-map tensors are freed
        tensor_weakrefs = []
        for idx in range(6):
            seq_img = _make_synthetic_signboard_image(
                width=300 + idx * 8,
                height=220 + idx * 6,
                bg_color=(20 + idx * 5, 30, 40),
            )
            seq_key = build_image_cache_identity(seq_img, image_id=1000 + idx, dataset_version="v1")
            self.pipeline.run(seq_img, "Q1 top", max_new_tokens=4, image_key=seq_key)
            self.pipeline.run(seq_img, "Q2 bot", max_new_tokens=4, image_key=seq_key)
            tensor_weakrefs.append(weakref.ref(self.pipeline._cached_working_maps[0]))

        gc.collect()
        alive_flags = [ref() is not None for ref in tensor_weakrefs]
        # Exactly the last image's working-map tensor (index 5) remains alive; images 0..4 are freed
        self.assertEqual(alive_flags, [False, False, False, False, False, True])


if __name__ == "__main__":
    unittest.main()
