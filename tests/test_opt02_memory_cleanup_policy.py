# -*- coding: utf-8 -*-
"""
Unit and Mock Integration Test Suite for TICKET-OPT-02: Coordinated Memory Cleanup Policy.

Covers all mandatory verification gates (Tests A through I) in local synthetic/mock mode
without loading real VLM weights:
- Test A: No global cleanup per question (`SGVCDPipeline.run()` and `PointwiseKLAttentionExtractor.extract()`
          invoke `gc.collect()` 0 times and `torch.cuda.empty_cache()` 0 times per question).
- Test B: Cleanup at image boundary (`gc.collect()` and CUDA watchdog run once after the final question
          of an image group completes, and 0 times during the inner question loop).
- Test C: Watchdog below threshold (`allocated / total <= 0.85`, including `0.50` and exact `0.85` boundary,
          does NOT call `torch.cuda.empty_cache()`).
- Test D: Watchdog above threshold (`allocated / total > 0.85`, e.g., `0.86` and `0.92`, calls
          `torch.cuda.empty_cache()` once per check and logs the trigger).
- Test E: CPU-only and error-handling compatibility (`torch.cuda.is_available() == False`, `device='cpu'`,
          and simulated CUDA query errors / invalid total memory complete cleanly without crashing).
- Test F: OPT-03 and OPT-06 integration preservation (`_compute_working_maps` runs once per image across
          multiple questions, cached maps and visual features remain valid across all questions in the group,
          and `run_image_boundary_cleanup` clears cached state at the image boundary).
- Test G: Image transition and exception safety (a failed question inside an image group is recorded,
          subsequent questions in the same group still reuse the image cache, image-boundary cleanup runs
          once per image group, and the next image group executes cleanly).
- Test H: Synthetic 20-QA workload across 6 images verifying exact call counts, conditional watchdog triggers,
          OPT-03 cache hit/miss counts, cached-vs-uncached output parity, and zero cross-image tensor retention.
- Test I: AST verification that `SGVCDPipeline.run` and `PointwiseKLAttentionExtractor.extract` contain
          zero `gc.collect()` or `empty_cache()` calls.
"""

import argparse
import ast
import functools
import gc
import inspect
import json
import os
import sys
import tempfile
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

import scripts.benchmark.evaluate_vitextvqa as eval_mod
from scripts.benchmark.evaluate_vitextvqa import (
    CUDA_MEMORY_WATCHDOG_THRESHOLD,
    check_and_run_cuda_memory_watchdog,
    run_benchmark,
    run_image_boundary_cleanup,
)
from scripts.benchmark.sg_vcd.adaptive_propagator import (
    AdaptiveScalePropagator,
    _is_valid_working_maps,
)
from scripts.benchmark.sg_vcd.attention_extractor import (
    BaseAttentionExtractor,
    ExtractionResult,
    PointwiseKLAttentionExtractor,
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
    """Create a deterministic synthetic scene-text image with two distinct signboards."""
    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    draw.rectangle([28, 26, 276, 78], fill=top_board_bg)
    for x in range(42, 260, 12):
        draw.rectangle([x, 38, x + 5, 64], fill=top_stroke_fg)
        draw.rectangle([x, 38, x + 8, 42], fill=top_stroke_fg)

    draw.rectangle([44, 146, 290, 202], fill=bot_board_bg)
    for x in range(58, 274, 14):
        draw.rectangle([x, 160, x + 6, 188], fill=bot_stroke_fg)
        draw.rectangle([x, 172, x + 9, 176], fill=bot_stroke_fg)

    return img


class _QuestionConditionedMockExtractor(BaseAttentionExtractor):
    """Question-conditioned mock attention extractor for deterministic Stage 1 testing."""

    def __init__(self, grid_ratio=(2, 2)):
        self.grid_ratio = grid_ratio
        self.extract_calls = 0
        self.questions_seen = []
        self._cached_img_key = None
        self._cached_img_state = None

    def clear_cache(self) -> None:
        self._cached_img_key = None
        self._cached_img_state = None

    def _build_prefix_and_suffix_embeds(self, vit_embeds: torch.Tensor, question: str):
        hidden_dim = int(vit_embeds.shape[-1]) if vit_embeds is not None else 32
        return (
            torch.ones((1, 16, hidden_dim), dtype=torch.bfloat16),
            torch.ones((1, 8, hidden_dim), dtype=torch.bfloat16),
            2,
            10,
        )

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
            hy0, hy1 = int(0.65 * H_g), int(0.80 * H_g)
            hx0, hx1 = int(0.25 * W_g), int(0.75 * W_g)
            phi[hy0:hy1, hx0:hx1] = 2.6
            phi[(hy0 + hy1) // 2, (hx0 + hx1) // 2] = 4.2
        else:
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
        total_len: int = None,
        batch_size: int = 1,
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
        self.attentions = tuple(
            torch.full((batch_size, 2, seq_len, t_len), 1.0 / max(1, t_len), dtype=torch.bfloat16)
            for _ in range(num_layers)
        )


class _DummyInnerLLM(nn.Module):
    def __init__(self, hidden_dim: int = 32, vocab_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.embed_lookup = nn.Embedding(vocab_size, hidden_dim, dtype=torch.bfloat16)
        self.prefill_calls = 0
        self.forward_calls = 0
        self.config = type("Cfg", (), {"_attn_implementation": "sdpa"})()
        self.layers = []

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
        self.forward_calls += 1
        batch_size = int(inputs_embeds.shape[0])
        q_len = inputs_embeds.shape[1]
        past_len = 0
        if past_key_values is not None and len(past_key_values) > 0:
            past_len = past_key_values[0][0].shape[2]
        else:
            self.prefill_calls += 1
        total_len = past_len + q_len
        return _DummyAttentionOutput(
            q_len,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            total_len=total_len,
            batch_size=batch_size,
        )


class _DummyLanguageModel(nn.Module):
    def __init__(self, hidden_dim: int = 32, vocab_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.model = _DummyInnerLLM(hidden_dim=hidden_dim, vocab_size=vocab_size, num_layers=num_layers)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False, dtype=torch.bfloat16)
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


class TestOPT02MemoryCleanupPolicy(unittest.TestCase):
    """Comprehensive verification of TICKET-OPT-02: Coordinated Memory Cleanup Policy (Tests A–I)."""

    def setUp(self):
        default_feature_cache.clear()
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        self.model = _DummyVinternModel()
        self.tokenizer = _DummyTokenizer()
        self.img1 = _make_synthetic_signboard_image(
            top_board_bg=(25, 95, 190), bot_board_bg=(175, 30, 35)
        )
        self.img2 = _make_synthetic_signboard_image(
            bg_color=(55, 50, 42), top_board_bg=(30, 145, 75), bot_board_bg=(130, 45, 165)
        )

    def tearDown(self):
        for stream_name, orig_stream in (("stdout", self._orig_stdout), ("stderr", self._orig_stderr)):
            cur_stream = getattr(sys, stream_name, None)
            if hasattr(cur_stream, "file"):
                try:
                    cur_stream.file.close()
                except Exception:
                    pass
            setattr(sys, stream_name, orig_stream)
        default_feature_cache.clear()

    def _build_pipeline(self, extractor=None) -> SGVCDPipeline:
        ext = extractor if extractor is not None else _QuestionConditionedMockExtractor(grid_ratio=(2, 2))
        return SGVCDPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            transform=_identity_transform,
            device="cpu",
            extractor=ext,
            propagator=AdaptiveScalePropagator(),
            feature_cache=VisualFeatureCache(max_entries=8),
        )

    # ------------------------------------------------------------------
    # Test A — No global cleanup per question
    # ------------------------------------------------------------------
    def test_A_no_global_cleanup_per_question(self):
        """
        Test A — No global cleanup per question:
        Instrument `gc.collect` and `torch.cuda.empty_cache` (with `torch.cuda.is_available` mocked
        to True). Run multiple synthetic QA samples on one image through both
        `PointwiseKLAttentionExtractor.extract()` and `SGVCDPipeline.run()`.
        Verify `0` calls to `gc.collect()` and `0` calls to `torch.cuda.empty_cache()` per question.
        """
        kl_extractor = PointwiseKLAttentionExtractor(
            model=self.model,
            tokenizer=self.tokenizer,
            transform=_identity_transform,
            layer_start=0,
            layer_end=2,
            min_grid_tiles=1,
            max_grid_tiles=1,
            device="cpu",
        )
        pipeline = self._build_pipeline(extractor=kl_extractor)
        img_key = build_image_cache_identity(
            self.img1, image_id=101, dataset_version="vitextvqa_opt02:test_a"
        )

        questions = [
            "Q1 top: Biển hiệu phía trên ghi gì?",
            "Q2 bot: Biển hiệu phía dưới ghi gì?",
            "Q3 top: Tên cửa hàng ở trên là gì?",
            "Q4 bot: Địa chỉ ở bảng dưới là gì?",
        ]

        with patch("gc.collect") as mock_gc_collect, \
             patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.empty_cache") as mock_empty_cache:

            # 1. Direct PointwiseKLAttentionExtractor.extract() calls across multiple questions
            for q in questions:
                ext_res = kl_extractor.extract(self.img1, q, image_key=img_key)
                self.assertIsNotNone(ext_res.phi_map)

            self.assertEqual(
                mock_gc_collect.call_count,
                0,
                "PointwiseKLAttentionExtractor.extract() must not invoke gc.collect() per question",
            )
            self.assertEqual(
                mock_empty_cache.call_count,
                0,
                "PointwiseKLAttentionExtractor.extract() must not invoke torch.cuda.empty_cache() per question",
            )

            # 2. Full SGVCDPipeline.run() calls across multiple questions
            for q in questions:
                res = pipeline.run(self.img1, q, max_new_tokens=2, image_key=img_key)
                self.assertEqual(res.pred_text, "nhà thuốc bảo tâm an")

            self.assertEqual(
                mock_gc_collect.call_count,
                0,
                "SGVCDPipeline.run() must not invoke gc.collect() per question",
            )
            self.assertEqual(
                mock_empty_cache.call_count,
                0,
                "SGVCDPipeline.run() must not invoke torch.cuda.empty_cache() per question",
            )

    # ------------------------------------------------------------------
    # Test B — Cleanup at image boundary
    # ------------------------------------------------------------------
    def test_B_cleanup_at_image_boundary(self):
        """
        Test B — Cleanup at image boundary:
        Run a synthetic image group with 3 QA samples through `run_benchmark()`.
        Verify that image-boundary cleanup runs ONLY after the final QA sample of that image
        has completed (`0` cleanups during the inner question loop, `1` cleanup at image boundary).
        """
        tmpdir = tempfile.mkdtemp(prefix="opt02_test_b_")
        try:
            img_path = os.path.join(tmpdir, "img_boundary.jpg")
            self.img1.save(img_path)

            dataset_records = [
                {
                    "sample_key": f"img_boundary.jpg::{qid}",
                    "question_id": qid,
                    "image_id": 501,
                    "image_name": "img_boundary.jpg",
                    "image_path": img_path,
                    "dataset_version": "opt02_test_b:v1",
                    "content_fingerprint": None,
                    "question": f"Câu hỏi {qid} trên ảnh?",
                    "gt_answers": ["nhà thuốc bảo tâm an"],
                    "verified_all_answers": ["nhà thuốc bảo tâm an"],
                    "added_gt_answers": [],
                    "answer_region_bbox": [28, 26, 276, 78],
                    "sub_bboxes": [],
                }
                for qid in (1, 2, 3)
            ]

            pipeline = self._build_pipeline()
            timeline = []

            orig_run = pipeline.run

            @functools.wraps(orig_run)
            def _tracked_run(image, question, **kwargs):
                out = orig_run(image, question, **kwargs)
                timeline.append(("question_completed", question))
                return out

            pipeline.run = _tracked_run

            orig_cleanup = eval_mod.run_image_boundary_cleanup

            def _tracked_cleanup(**kwargs):
                timeline.append(("image_boundary_cleanup",))
                return orig_cleanup(**kwargs)

            args = argparse.Namespace(
                dataset="synthetic.json",
                data_dir=tmpdir,
                output_dir=os.path.join(tmpdir, "out"),
                checkpoint=None,
                overwrite=True,
                method="sg_vcd",
                model_name="synthetic-mock",
                device="cpu",
                test=False,
                test_images=0,
                test_samples=0,
                limit=0,
                num_workers=1,
                worker_id=0,
                mock=False,
            )

            with patch.object(eval_mod, "load_benchmark_dataset", return_value=dataset_records), \
                 patch.object(eval_mod, "load_vintern_model", return_value=(self.model, self.tokenizer, _identity_transform)), \
                 patch.object(eval_mod, "SGVCDPipeline", return_value=pipeline), \
                 patch.object(eval_mod, "run_image_boundary_cleanup", side_effect=_tracked_cleanup):
                run_benchmark(args)

            # Timeline must be: 3 questions completed in order, then 1 image_boundary_cleanup at the end
            self.assertEqual(len(timeline), 4)
            self.assertEqual(timeline[0][0], "question_completed")
            self.assertEqual(timeline[1][0], "question_completed")
            self.assertEqual(timeline[2][0], "question_completed")
            self.assertEqual(timeline[3], ("image_boundary_cleanup",))
        finally:
            self.tearDown()
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Test C — Watchdog below threshold
    # ------------------------------------------------------------------
    def test_C_watchdog_below_threshold(self):
        """
        Test C — Watchdog below threshold:
        Mock CUDA memory with `allocated / total <= 0.85` (testing both `0.50` and exact `0.85`).
        Verify `torch.cuda.empty_cache()` is NOT called (`0` calls) and the decision is logged.
        """
        total_bytes = 4000 * 1024 * 1024  # Exact multiple of 100 so 0.85 * total_bytes is an exact integer
        dummy_props = type("CudaProps", (), {"total_memory": total_bytes})()

        for ratio_target in (0.50, CUDA_MEMORY_WATCHDOG_THRESHOLD):
            alloc_bytes = int(ratio_target * total_bytes)
            logs = []
            with patch("torch.cuda.is_available", return_value=True), \
                 patch("torch.cuda.current_device", return_value=0), \
                 patch("torch.cuda.memory_allocated", return_value=alloc_bytes) as mock_alloc, \
                 patch("torch.cuda.get_device_properties", return_value=dummy_props) as mock_props, \
                 patch("torch.cuda.empty_cache") as mock_empty_cache:
                info = check_and_run_cuda_memory_watchdog(
                    device="cuda:0",
                    watermark_ratio=CUDA_MEMORY_WATCHDOG_THRESHOLD,
                    log_fn=logs.append,
                )

            self.assertTrue(info["cuda_available"])
            self.assertEqual(info["device_index"], 0)
            self.assertAlmostEqual(info["ratio"], ratio_target, places=6)
            self.assertFalse(info["triggered_empty_cache"])
            self.assertEqual(info["status"], "below_threshold")
            self.assertEqual(mock_alloc.call_count, 1)
            self.assertEqual(mock_props.call_count, 1)
            self.assertEqual(
                mock_empty_cache.call_count,
                0,
                f"empty_cache() must NOT be called when ratio={ratio_target:.2f} <= 0.85",
            )
            self.assertEqual(len(logs), 1)
            self.assertIn("skipped torch.cuda.empty_cache()", logs[0])

    # ------------------------------------------------------------------
    # Test D — Watchdog above threshold
    # ------------------------------------------------------------------
    def test_D_watchdog_above_threshold(self):
        """
        Test D — Watchdog above threshold:
        Mock CUDA memory with `allocated / total > 0.85` (`0.86` and `0.92`).
        Verify `torch.cuda.empty_cache()` is called once per check and logged.
        """
        total_bytes = 4000 * 1024 * 1024
        dummy_props = type("CudaProps", (), {"total_memory": total_bytes})()

        for ratio_target in (0.86, 0.92):
            alloc_bytes = int(round(ratio_target * total_bytes))
            logs = []
            with patch("torch.cuda.is_available", return_value=True), \
                 patch("torch.cuda.current_device", return_value=0), \
                 patch("torch.cuda.memory_allocated", return_value=alloc_bytes), \
                 patch("torch.cuda.get_device_properties", return_value=dummy_props), \
                 patch("torch.cuda.empty_cache") as mock_empty_cache:
                info = check_and_run_cuda_memory_watchdog(
                    device=torch.device("cuda:0"),
                    watermark_ratio=CUDA_MEMORY_WATCHDOG_THRESHOLD,
                    log_fn=logs.append,
                )

            self.assertTrue(info["cuda_available"])
            self.assertEqual(info["device_index"], 0)
            self.assertGreater(info["ratio"], CUDA_MEMORY_WATCHDOG_THRESHOLD)
            self.assertTrue(info["triggered_empty_cache"])
            self.assertEqual(info["status"], "triggered")
            self.assertEqual(
                mock_empty_cache.call_count,
                1,
                f"empty_cache() must be called once when ratio={ratio_target:.2f} > 0.85",
            )
            self.assertEqual(len(logs), 1)
            self.assertIn("triggered torch.cuda.empty_cache()", logs[0])

    # ------------------------------------------------------------------
    # Test E — CPU-only compatibility and safe error handling
    # ------------------------------------------------------------------
    def test_E_cpu_only_compatibility(self):
        """
        Test E — CPU-only compatibility:
        1. Simulate `torch.cuda.is_available() == False`: zero CUDA memory queries, zero crashes.
        2. Simulate `device='cpu'` while `torch.cuda.is_available() == True`: zero CUDA queries.
        3. Simulate CUDA query exception or invalid `total_memory <= 0`: handled safely without crashing.
        """
        # Case 1: torch.cuda.is_available() == False
        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.cuda.memory_allocated") as mock_alloc, \
             patch("torch.cuda.get_device_properties") as mock_props, \
             patch("torch.cuda.empty_cache") as mock_empty_cache:
            cleanup_res = run_image_boundary_cleanup(pipeline=None, device="cpu", log_fn=None)

        self.assertFalse(cleanup_res["watchdog"]["cuda_available"])
        self.assertFalse(cleanup_res["watchdog"]["triggered_empty_cache"])
        self.assertEqual(cleanup_res["watchdog"]["status"], "cpu_or_cuda_unavailable")
        self.assertEqual(mock_alloc.call_count, 0)
        self.assertEqual(mock_props.call_count, 0)
        self.assertEqual(mock_empty_cache.call_count, 0)

        # Case 2: device explicitly set to 'cpu' even if CUDA is available
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.memory_allocated") as mock_alloc, \
             patch("torch.cuda.empty_cache") as mock_empty_cache:
            info_cpu = check_and_run_cuda_memory_watchdog(device="cpu", log_fn=None)

        self.assertFalse(info_cpu["cuda_available"])
        self.assertFalse(info_cpu["triggered_empty_cache"])
        self.assertEqual(mock_alloc.call_count, 0)
        self.assertEqual(mock_empty_cache.call_count, 0)

        # Case 3: CUDA query raises RuntimeError
        logs_err = []
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.current_device", return_value=0), \
             patch("torch.cuda.memory_allocated", side_effect=RuntimeError("driver query failed")), \
             patch("torch.cuda.empty_cache") as mock_empty_cache:
            info_err = check_and_run_cuda_memory_watchdog(device="cuda:0", log_fn=logs_err.append)

        self.assertFalse(info_err["triggered_empty_cache"])
        self.assertTrue(info_err["status"].startswith("memory_query_unavailable:"))
        self.assertEqual(mock_empty_cache.call_count, 0)
        self.assertEqual(len(logs_err), 1)

        # Case 4: Invalid total_memory == 0
        zero_props = type("CudaProps", (), {"total_memory": 0})()
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.current_device", return_value=0), \
             patch("torch.cuda.memory_allocated", return_value=1024), \
             patch("torch.cuda.get_device_properties", return_value=zero_props), \
             patch("torch.cuda.empty_cache") as mock_empty_cache:
            info_zero = check_and_run_cuda_memory_watchdog(device="cuda:0", log_fn=None)

        self.assertFalse(info_zero["triggered_empty_cache"])
        self.assertEqual(info_zero["status"], "invalid_memory_values")
        self.assertEqual(mock_empty_cache.call_count, 0)

    # ------------------------------------------------------------------
    # Test F — OPT-03 and OPT-06 integration preservation
    # ------------------------------------------------------------------
    def test_F_opt03_integration_preservation(self):
        """
        Test F — OPT-03 integration preservation:
        Run 4 questions sharing the same image identity through `SGVCDPipeline.run()`.
        Verify `_compute_working_maps` is called once per image (1 miss, 3 hits),
        cached working maps remain valid across all 4 questions, and `run_image_boundary_cleanup`
        releases `_cached_working_maps` at the end of the image group.
        """
        pipeline = self._build_pipeline()
        img_key = build_image_cache_identity(
            self.img1, image_id=601, dataset_version="vitextvqa_opt02:test_f"
        )
        questions = [
            "Q1 top: Biển hiệu phía trên ghi gì?",
            "Q2 bot: Biển hiệu phía dưới ghi gì?",
            "Q3 top: Tên cửa hàng ở trên là gì?",
            "Q4 bot: Địa chỉ ở bảng dưới là gì?",
        ]

        with patch.object(
            pipeline.propagator,
            "_compute_working_maps",
            wraps=pipeline.propagator._compute_working_maps,
        ) as spy_compute_maps:
            cached_vis = pipeline.extract_visual_features(self.img1, image_key=img_key)
            for q in questions:
                res = pipeline.run(
                    self.img1,
                    q,
                    max_new_tokens=2,
                    cached_visual_features=cached_vis,
                    image_key=img_key,
                )
                self.assertTrue(_is_valid_working_maps(pipeline._cached_working_maps))
                self.assertEqual(pipeline._cached_working_maps_key, img_key)
                self.assertEqual(res.pred_text, "nhà thuốc bảo tâm an")

            self.assertEqual(
                spy_compute_maps.call_count,
                1,
                "OPT-03 working maps must be computed once across all 4 questions of the same image",
            )

        # Now run image-boundary cleanup and verify cache references are released
        with patch("torch.cuda.is_available", return_value=False):
            boundary_info = run_image_boundary_cleanup(pipeline=pipeline, device="cpu", log_fn=None)

        self.assertTrue(boundary_info["pipeline_cache_reset"])
        self.assertIsNone(pipeline._cached_working_maps)
        self.assertIsNone(pipeline._cached_working_maps_key)
        self.assertEqual(len(pipeline.feature_cache._cache), 0)

    # ------------------------------------------------------------------
    # Test G — Image transition and exception safety
    # ------------------------------------------------------------------
    def test_G_image_transition_and_exception_safety(self):
        """
        Test G — Image transition and exception safety:
        Simulate 2 image groups (`img_g1.jpg` with 3 questions where Q2 raises an exception
        inside `pipeline.run()`, followed by `img_g2.jpg` with 2 questions).
        Verify:
        - Q2's failure is gracefully caught and recorded in checkpoint.
        - Q3 on `img_g1.jpg` still executes and reuses `img_g1.jpg`'s cached working maps.
        - Image-boundary cleanup runs once after `img_g1.jpg` and once after `img_g2.jpg` (2 total).
        - `_compute_working_maps` is called once per image (2 total across 5 questions).
        """
        tmpdir = tempfile.mkdtemp(prefix="opt02_test_g_")
        try:
            p1 = os.path.join(tmpdir, "img_g1.jpg")
            p2 = os.path.join(tmpdir, "img_g2.jpg")
            self.img1.save(p1)
            self.img2.save(p2)

            dataset_records = [
                {
                    "sample_key": f"img_g1.jpg::{qid}",
                    "question_id": qid,
                    "image_id": 701,
                    "image_name": "img_g1.jpg",
                    "image_path": p1,
                    "dataset_version": "opt02_test_g:v1",
                    "content_fingerprint": None,
                    "question": f"Q{qid} top?",
                    "gt_answers": ["nhà thuốc bảo tâm an"],
                    "verified_all_answers": ["nhà thuốc bảo tâm an"],
                    "added_gt_answers": [],
                    "answer_region_bbox": [28, 26, 276, 78],
                    "sub_bboxes": [],
                }
                for qid in (1, 2, 3)
            ] + [
                {
                    "sample_key": f"img_g2.jpg::{qid}",
                    "question_id": qid,
                    "image_id": 702,
                    "image_name": "img_g2.jpg",
                    "image_path": p2,
                    "dataset_version": "opt02_test_g:v1",
                    "content_fingerprint": None,
                    "question": f"Q{qid} bot?",
                    "gt_answers": ["nhà thuốc bảo tâm an"],
                    "verified_all_answers": ["nhà thuốc bảo tâm an"],
                    "added_gt_answers": [],
                    "answer_region_bbox": [44, 146, 290, 202],
                    "sub_bboxes": [],
                }
                for qid in (4, 5)
            ]

            pipeline = self._build_pipeline()
            orig_run = pipeline.run

            @functools.wraps(orig_run)
            def _flaky_run(image, question, **kwargs):
                if "Q2" in question:
                    raise RuntimeError("Simulated per-question failure on Q2")
                return orig_run(image, question, **kwargs)

            pipeline.run = _flaky_run

            out_dir = os.path.join(tmpdir, "out_g")
            args = argparse.Namespace(
                dataset="synthetic.json",
                data_dir=tmpdir,
                output_dir=out_dir,
                checkpoint=None,
                overwrite=True,
                method="sg_vcd",
                model_name="synthetic-mock",
                device="cpu",
                test=False,
                test_images=0,
                test_samples=0,
                limit=0,
                num_workers=1,
                worker_id=0,
                mock=False,
            )

            with patch.object(eval_mod, "load_benchmark_dataset", return_value=dataset_records), \
                 patch.object(eval_mod, "load_vintern_model", return_value=(self.model, self.tokenizer, _identity_transform)), \
                 patch.object(eval_mod, "SGVCDPipeline", return_value=pipeline), \
                 patch.object(
                     pipeline.propagator,
                     "_compute_working_maps",
                     wraps=pipeline.propagator._compute_working_maps,
                 ) as spy_compute_maps, \
                 patch.object(
                     eval_mod,
                     "run_image_boundary_cleanup",
                     wraps=eval_mod.run_image_boundary_cleanup,
                 ) as spy_cleanup:
                run_benchmark(args)

            # 2 image groups -> 2 image-boundary cleanups and 2 _compute_working_maps calls
            self.assertEqual(spy_cleanup.call_count, 2)
            self.assertEqual(spy_compute_maps.call_count, 2)

            # Verify all 5 questions were recorded in checkpoint.json (including failed Q2 with empty pred_answer)
            ckpt_file = os.path.join(out_dir, "checkpoint.json")
            with open(ckpt_file, "r", encoding="utf-8") as f:
                ckpt_data = json.load(f)
            records = ckpt_data["records"]
            self.assertEqual(len(records), 5)
            q2_rec = [r for r in records if r["question_id"] == 2][0]
            self.assertEqual(q2_rec["pred_answer"], "")
            q3_rec = [r for r in records if r["question_id"] == 3][0]
            self.assertEqual(q3_rec["pred_answer"], "nhà thuốc bảo tâm an")
        finally:
            self.tearDown()
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Test H — Synthetic 20-QA workload across 6 images (Gate A)
    # ------------------------------------------------------------------
    def test_H_synthetic_20_qa_workload(self):
        """
        Test H — Synthetic 20-QA workload (Gate A):
        Execute a 20-QA synthetic workload grouped across 6 images (4 + 3 + 4 + 3 + 3 + 3 = 20 QA)
        through `run_benchmark()` with `SGVCDPipeline`.
        Verify:
        1. 20 QA samples processed across 6 image groups.
        2. 0 per-question `gc.collect()` or `torch.cuda.empty_cache()` calls inside `pipeline.run()`.
        3. 6 image-boundary `gc.collect()` calls and 6 CUDA watchdog checks (1 per image group).
        4. Under mocked CUDA memory ratios `[0.62, 0.78, 0.85, 0.88, 0.74, 0.91]`,
           `torch.cuda.empty_cache()` is triggered on exactly 2 image boundaries (`0.88` and `0.91`).
        5. `OPT-03` working maps are computed 6 times (6 misses, 14 cache hits = 70.0% reduction).
        6. Exact output parity between the image-grouped cached run and an uncached reference pipeline.
        7. `weakref` verification that no working-map arrays from completed images accumulate in memory.
        """
        qa_counts_per_image = [4, 3, 4, 3, 3, 3]  # 6 images, 20 QA total
        self.assertEqual(sum(qa_counts_per_image), 20)

        tmpdir = tempfile.mkdtemp(prefix="opt02_test_h_")
        try:
            dataset_records = []
            images_by_id = {}
            qid_counter = 1

            palette = [
                ((25, 95, 190), (175, 30, 35)),
                ((30, 145, 75), (130, 45, 165)),
                ((180, 90, 20), (20, 110, 150)),
                ((90, 40, 160), (160, 120, 25)),
                ((40, 130, 130), (170, 45, 90)),
                ((140, 50, 50), (45, 85, 170)),
            ]

            for img_idx, num_q in enumerate(qa_counts_per_image):
                img_id = 800 + img_idx
                img_name = f"synth_img_{img_idx:02d}.jpg"
                img_path = os.path.join(tmpdir, img_name)
                top_bg, bot_bg = palette[img_idx]
                pil_img = _make_synthetic_signboard_image(
                    width=320 + (img_idx * 8),
                    height=240 + (img_idx * 4),
                    top_board_bg=top_bg,
                    bot_board_bg=bot_bg,
                )
                pil_img.save(img_path)
                images_by_id[img_id] = (img_path, [
                    f"Q{q_i + 1} {'top' if (q_i % 2 == 0) else 'bot'} trên {img_name}?"
                    for q_i in range(num_q)
                ])

                for q_text in images_by_id[img_id][1]:
                    is_top = "top" in q_text
                    dataset_records.append(
                        {
                            "sample_key": f"{img_name}::{qid_counter}",
                            "question_id": qid_counter,
                            "image_id": img_id,
                            "image_name": img_name,
                            "image_path": img_path,
                            "dataset_version": "opt02_20qa_gate_a:v1",
                            "content_fingerprint": None,
                            "question": q_text,
                            "gt_answers": ["nhà thuốc bảo tâm an"],
                            "verified_all_answers": ["nhà thuốc bảo tâm an"],
                            "added_gt_answers": [],
                            "answer_region_bbox": [28, 26, 276, 78] if is_top else [44, 146, 290, 202],
                            "sub_bboxes": [],
                        }
                    )
                    qid_counter += 1

            # Compute uncached reference results first (calling reset_cache() before every question)
            ref_pipeline = self._build_pipeline()
            uncached_expected = {}
            for rec in dataset_records:
                ref_pipeline.reset_cache()
                with Image.open(rec["image_path"]) as raw_im:
                    im_rgb = raw_im.convert("RGB")
                ref_res = ref_pipeline.run(im_rgb, rec["question"], max_new_tokens=2, image_key=None)
                uncached_expected[rec["question_id"]] = (
                    list(ref_res.stage1_bbox),
                    ref_res.pred_text,
                    round(float(ref_res.spatial_confidence), 6),
                )

            # Prepare tracked pipeline and weakref tracker for working maps
            pipeline = self._build_pipeline()
            weak_refs = []
            orig_compute_maps = pipeline.propagator._compute_working_maps

            def _tracked_compute_maps(image):
                maps = orig_compute_maps(image)
                # maps is (img_work, L, C_rg, C_yb, W_w, H_w, sx, sy); index 1 is L (np.ndarray)
                weak_refs.append(weakref.ref(maps[1]))
                return maps

            # Track per-question vs image-boundary cleanup calls
            in_pipeline_run = {"active": False}
            per_question_gc_calls = []
            per_question_empty_cache_calls = []
            boundary_gc_calls = []
            boundary_watchdog_results = []

            orig_pipeline_run = pipeline.run

            @functools.wraps(orig_pipeline_run)
            def _wrapped_pipeline_run(image, question, max_new_tokens=64, cached_visual_features=None, *, image_key=None, **kwargs):
                in_pipeline_run["active"] = True
                try:
                    return orig_pipeline_run(
                        image,
                        question,
                        max_new_tokens=max_new_tokens,
                        cached_visual_features=cached_visual_features,
                        image_key=image_key,
                        **kwargs,
                    )
                finally:
                    in_pipeline_run["active"] = False

            pipeline.run = _wrapped_pipeline_run

            orig_gc_collect = gc.collect

            def _counting_gc_collect(*args, **kwargs):
                if in_pipeline_run["active"]:
                    per_question_gc_calls.append(1)
                else:
                    boundary_gc_calls.append(1)
                return orig_gc_collect(*args, **kwargs)

            # Mock CUDA memory schedule across the 6 image boundaries:
            # Ratios: [0.62, 0.78, 0.85, 0.88, 0.74, 0.91] -> only 0.88 and 0.91 exceed 0.85
            total_vram = 4000 * 1024 * 1024
            ratio_schedule = [0.62, 0.78, 0.85, 0.88, 0.74, 0.91]
            alloc_schedule = [int(r * total_vram) for r in ratio_schedule]
            dummy_props = type("CudaProps", (), {"total_memory": total_vram})()

            orig_watchdog = eval_mod.check_and_run_cuda_memory_watchdog

            def _tracked_watchdog(*args, **kwargs):
                res = orig_watchdog(*args, **kwargs)
                boundary_watchdog_results.append(res)
                return res

            out_dir = os.path.join(tmpdir, "out_20qa")
            args = argparse.Namespace(
                dataset="synthetic_20qa.json",
                data_dir=tmpdir,
                output_dir=out_dir,
                checkpoint=None,
                overwrite=True,
                method="sg_vcd",
                model_name="synthetic-mock",
                device="cuda:0",
                test=False,
                test_images=0,
                test_samples=0,
                limit=0,
                num_workers=1,
                worker_id=0,
                mock=False,
            )

            with patch.object(eval_mod, "load_benchmark_dataset", return_value=dataset_records), \
                 patch.object(eval_mod, "load_vintern_model", return_value=(self.model, self.tokenizer, _identity_transform)), \
                 patch.object(eval_mod, "SGVCDPipeline", return_value=pipeline), \
                 patch.object(pipeline.propagator, "_compute_working_maps", side_effect=_tracked_compute_maps) as spy_maps, \
                 patch.object(eval_mod, "check_and_run_cuda_memory_watchdog", side_effect=_tracked_watchdog), \
                 patch("gc.collect", side_effect=_counting_gc_collect), \
                 patch("torch.cuda.is_available", return_value=True), \
                 patch("torch.cuda.current_device", return_value=0), \
                 patch("torch.cuda.memory_allocated", side_effect=alloc_schedule), \
                 patch("torch.cuda.get_device_properties", return_value=dummy_props), \
                 patch("torch.cuda.max_memory_allocated", return_value=0), \
                 patch("torch.cuda.max_memory_reserved", return_value=0), \
                 patch("torch.cuda.empty_cache") as mock_empty_cache:

                def _counting_empty_cache():
                    if in_pipeline_run["active"]:
                        per_question_empty_cache_calls.append(1)

                mock_empty_cache.side_effect = _counting_empty_cache
                run_benchmark(args)

            # 1. Verify 0 per-question global cleanups
            self.assertEqual(len(per_question_gc_calls), 0)
            self.assertEqual(len(per_question_empty_cache_calls), 0)

            # 2. Verify 6 image-boundary gc.collect() calls and 6 watchdog evaluations
            self.assertEqual(len(boundary_gc_calls), 6)
            self.assertEqual(len(boundary_watchdog_results), 6)

            # 3. Verify conditional CUDA empty_cache() triggers (only for 0.88 and 0.91 -> 2 calls)
            self.assertEqual(mock_empty_cache.call_count, 2)
            triggered_flags = [w["triggered_empty_cache"] for w in boundary_watchdog_results]
            self.assertEqual(triggered_flags, [False, False, False, True, False, True])

            # 4. Verify OPT-03 working-map cache reuse: 6 computations across 20 questions (14 cache hits)
            self.assertEqual(spy_maps.call_count, 6)

            # 5. Verify all 20 QA records match uncached reference outputs
            ckpt_file = os.path.join(out_dir, "checkpoint.json")
            with open(ckpt_file, "r", encoding="utf-8") as f:
                ckpt_data = json.load(f)
            records = ckpt_data["records"]
            self.assertEqual(len(records), 20)
            for rec in records:
                qid = rec["question_id"]
                exp_bbox, exp_pred, exp_conf = uncached_expected[qid]
                self.assertEqual(rec["pred_bbox"], exp_bbox)
                self.assertEqual(rec["pred_answer"], exp_pred)
                self.assertAlmostEqual(rec["spatial_confidence"], exp_conf, places=5)

            # 6. Verify all 6 image-scoped working-map arrays were released and garbage-collected
            gc.collect()
            alive_refs = [r for r in weak_refs if r() is not None]
            self.assertEqual(len(alive_refs), 0)
        finally:
            self.tearDown()
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Test I — AST inspection of pipeline.run and extractor.extract
    # ------------------------------------------------------------------
    def test_I_ast_verification_no_per_question_cleanup_in_pipeline_or_extractor(self):
        """
        Test I — AST verification:
        Parse `SGVCDPipeline.run` and `PointwiseKLAttentionExtractor.extract` and verify
        neither method contains a call to `gc.collect()` or `torch.cuda.empty_cache()`.
        """
        for fn_obj, label in (
            (SGVCDPipeline.run, "SGVCDPipeline.run"),
            (PointwiseKLAttentionExtractor.extract, "PointwiseKLAttentionExtractor.extract"),
        ):
            src = inspect.getsource(fn_obj)
            tree = ast.parse(inspect.cleandoc(src))
            forbidden_calls = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute) and node.func.attr in ("collect", "empty_cache"):
                        forbidden_calls.append(node.func.attr)
            self.assertEqual(
                forbidden_calls,
                [],
                f"{label} must not contain per-question calls to gc.collect or empty_cache, found: {forbidden_calls}",
            )


if __name__ == "__main__":
    unittest.main()
