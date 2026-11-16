# -*- coding: utf-8 -*-
"""
Unit and Mock Test Suite for TICKET-OPT-06: Stable Image Identity and Cache Key.

Covers all mandatory verification gates (Tests A through H) in local mock/synthetic mode
without loading real VLM weights:
- Test A: Same image identity reuses cache key and valid cache entry without per-question thumbnail resize.
- Test B: Different image identities with identical dimensions/pixels do not collide.
- Test C: Same basename in different directories do not collide; bare basename alone is not treated as stable ID.
- Test D: Same image ID in different dataset namespaces produce distinct keys.
- Test E: Changed image content/fingerprint/file-stat invalidates cached state under the same logical ID.
- Test F: Missing stable identity safely falls back to content-based thumbnail key with `is_fallback=True`.
- Test G: Backward compatibility for legacy positional callers and custom mock extractors.
- Test H: Cache lifecycle invalidation on image change and fresh evaluation run isolation.
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.benchmark.evaluate_vitextvqa import (
    _sample_image_group_key,
    group_samples_by_image,
    load_benchmark_dataset,
    run_benchmark,
)
from scripts.benchmark.sg_vcd.attention_extractor import (
    BaseAttentionExtractor,
    ExtractionResult,
    PointwiseKLAttentionExtractor,
)
from scripts.benchmark.sg_vcd.feature_cache import (
    FALLBACK_IMAGE_IDENTITY,
    UNSCOPED_DATASET_NAMESPACE,
    ImageCacheIdentity,
    VisualFeatureCache,
    _normalize_relative_image_path,
    build_image_cache_identity,
    compute_image_content_fingerprint,
    default_feature_cache,
)
from scripts.benchmark.sg_vcd.pipeline import SGVCDPipeline


def _identity_transform(pil_img: Image.Image) -> torch.Tensor:
    """Fast synthetic transform returning a (3, 16, 16) tensor from a PIL Image."""
    arr = np.asarray(pil_img.convert("RGB"), dtype=np.float32)
    h, w, _ = arr.shape
    cy, cx = min(16, h), min(16, w)
    patch = arr[:cy, :cx, :] / 255.0
    tensor = torch.from_numpy(patch).permute(2, 0, 1).contiguous()
    return tensor


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
        # Uniform attention weights (batch=batch_size, heads=2, q_len=seq_len, kv_len=t_len)
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
        self.embed_Lookup = nn.Embedding(vocab_size, hidden_dim, dtype=torch.bfloat16)
        self.prefill_calls = 0
        self.forward_calls = 0
        self.config = type("Cfg", (), {"_attn_implementation": "sdpa"})()
        self.layers = []

    def get_input_embeddings(self):
        return self.embed_Lookup

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
        out = _DummyAttentionOutput(
            q_len,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            batch_size=batch_size,
            total_len=total_len,
        )
        if output_attentions:
            out.attentions = tuple(
                torch.full((batch_size, 2, q_len, total_len), 1.0 / max(1, total_len), dtype=torch.bfloat16)
                for _ in range(self.num_layers)
            )
        return out


class _DummyLanguageModel(nn.Module):
    def __init__(self, hidden_dim: int = 32, vocab_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.model = _DummyInnerLLM(hidden_dim=hidden_dim, vocab_size=vocab_size, num_layers=num_layers)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False, dtype=torch.bfloat16)
        self.config = self.model.config

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()


class _DummyVinternModel(nn.Module):
    """Lightweight synthetic model stub for testing caching without real weights."""

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
        # Return unique tensor per call scaled by call count so cache hits vs misses are verifiable
        val = float(self.extract_feature_calls)
        return torch.full(
            (num_tiles, 256, self.hidden_dim),
            val,
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


class TestOPT06ImageCacheIdentity(unittest.TestCase):
    """Comprehensive verification of TICKET-OPT-06 (Tests A through H)."""

    def setUp(self):
        default_feature_cache.clear()
        self.model = _DummyVinternModel()
        self.tokenizer = _DummyTokenizer()

    def tearDown(self):
        default_feature_cache.clear()

    def test_A_same_image_identity_reuses_cache_and_avoids_redundant_thumbnail_resize(self):
        """
        Test A — Same image identity:
        Two questions for the same image identity reuse the same cache key and valid cache entry,
        and do NOT invoke 16x16 thumbnail resizing on every question.
        """
        img = Image.new("RGB", (640, 480), color=(40, 90, 140))
        cache = VisualFeatureCache(max_entries=8)
        extractor = PointwiseKLAttentionExtractor(
            model=self.model,
            tokenizer=self.tokenizer,
            transform=_identity_transform,
            layer_start=0,
            layer_end=2,
            min_grid_tiles=1,
            max_grid_tiles=1,
            device="cpu",
        )

        orig_resize = img.resize
        resize_16x16_calls = []

        def _spy_resize(size, *args, **kwargs):
            if tuple(size) == (16, 16):
                resize_16x16_calls.append(size)
            return orig_resize(size, *args, **kwargs)

        with patch.object(img, "resize", side_effect=_spy_resize):
            key1 = build_image_cache_identity(
                img,
                image_id=22,
                image_path="data/vitextvqa/st_images/22.jpg",
                dataset_version="answer_bboxes_400_verified:dev",
            )
            key2 = build_image_cache_identity(
                img,
                image_id=22,
                image_path="data/vitextvqa/st_images/22.jpg",
                dataset_version="answer_bboxes_400_verified:dev",
            )
            self.assertEqual(key1, key2)
            self.assertFalse(key1.is_fallback)
            self.assertEqual(len(resize_16x16_calls), 0, "Stable identity must not resize to 16x16")

            # VisualFeatureCache reuse across 2 calls for same image identity
            f1 = cache.get_or_compute(
                self.model, _identity_transform, [img], image=img, device="cpu", image_key=key1
            )
            f2 = cache.get_or_compute(
                self.model, _identity_transform, [img], image=img, device="cpu", image_key=key2
            )
            self.assertIs(f1, f2)
            self.assertEqual(self.model.extract_feature_calls, 1)

            # PointwiseKLAttentionExtractor reuse across 2 questions for same image identity
            res_q1 = extractor.extract(img, "cửa tiệm màu xanh là gì ?", image_key=key1)
            self.assertEqual(self.model.extract_feature_calls, 2)  # 1 for cache + 1 for extractor Q1
            self.assertEqual(self.model.language_model.model.prefill_calls, 1)

            res_q2 = extractor.extract(img, "nhà thuốc trong ảnh tên gì ?", image_key=key2)
            # Q2 must hit extractor._cached_img_state without re-running ViT or prefix prefill
            self.assertEqual(self.model.extract_feature_calls, 2)
            self.assertEqual(self.model.language_model.model.prefill_calls, 1)
            self.assertIs(res_q1.vit_embeds, res_q2.vit_embeds)
            self.assertEqual(len(resize_16x16_calls), 0)

    def test_B_different_image_identities_same_dimensions_and_pixels_do_not_collide(self):
        """
        Test B — Different image identities:
        Two distinct image IDs with identical dimensions and identical pixel contents
        produce separate keys and do not reuse cached state across images.
        """
        img_a = Image.new("RGB", (640, 480), color=(128, 128, 128))
        img_b = Image.new("RGB", (640, 480), color=(128, 128, 128))

        key_a = build_image_cache_identity(
            img_a, image_id=101, dataset_version="answer_bboxes_400_verified:dev"
        )
        key_b = build_image_cache_identity(
            img_b, image_id=102, dataset_version="answer_bboxes_400_verified:dev"
        )
        self.assertNotEqual(key_a, key_b)
        self.assertNotEqual(key_a.to_cache_key(num_tiles=1), key_b.to_cache_key(num_tiles=1))

        cache = VisualFeatureCache(max_entries=8)
        feat_a = cache.get_or_compute(
            self.model, _identity_transform, [img_a], image=img_a, device="cpu", image_key=key_a
        )
        feat_b = cache.get_or_compute(
            self.model, _identity_transform, [img_b], image=img_b, device="cpu", image_key=key_b
        )
        self.assertEqual(self.model.extract_feature_calls, 2)
        self.assertFalse(torch.equal(feat_a, feat_b))

        extractor = PointwiseKLAttentionExtractor(
            model=self.model,
            tokenizer=self.tokenizer,
            transform=_identity_transform,
            layer_start=0,
            layer_end=2,
            min_grid_tiles=1,
            max_grid_tiles=1,
            device="cpu",
        )
        ext_a = extractor.extract(img_a, "Q1", image_key=key_a)
        ext_b = extractor.extract(img_b, "Q1", image_key=key_b)
        self.assertEqual(self.model.extract_feature_calls, 4)
        self.assertFalse(torch.equal(ext_a.vit_embeds, ext_b.vit_embeds))

    def test_C_same_basename_different_directories_do_not_collide(self):
        """
        Test C — Same basename, different directories:
        Two images with the same basename and dimensions but different dataset-relative
        locations or identities do not collide. Also verifies that a bare basename alone
        is never treated as a collision-proof stable identity.
        """
        img1 = Image.new("RGB", (500, 500), color=(200, 200, 200))
        img2 = Image.new("RGB", (500, 500), color=(200, 200, 200))

        # Bare basename without image_id or directory scope must fall back to content mode
        bare_key = build_image_cache_identity(img1, image_path="22.jpg")
        self.assertTrue(
            bare_key.is_fallback,
            "Bare basename without image_id or directory scope must be marked is_fallback=True",
        )

        # Same basename in different dataset-relative directories must not collide
        key_dir1 = build_image_cache_identity(
            img1,
            image_path="data/xanylabeling_workspace/Cao_Anh_001_100/22.jpg",
            dataset_version="vitextvqa:dev",
        )
        key_dir2 = build_image_cache_identity(
            img2,
            image_path="data/xanylabeling_workspace/Trieu_101_200/22.jpg",
            dataset_version="vitextvqa:dev",
        )
        self.assertFalse(key_dir1.is_fallback)
        self.assertFalse(key_dir2.is_fallback)
        self.assertNotEqual(key_dir1, key_dir2)
        self.assertNotEqual(key_dir1.to_cache_key(1), key_dir2.to_cache_key(1))

        # Verify group_samples_by_image keeps them in distinct groups
        samples = [
            {
                "question_id": 1,
                "image_name": "22.jpg",
                "image_path": "data/xanylabeling_workspace/Cao_Anh_001_100/22.jpg",
                "dataset_version": "vitextvqa:dev",
            },
            {
                "question_id": 2,
                "image_name": "22.jpg",
                "image_path": "data/xanylabeling_workspace/Trieu_101_200/22.jpg",
                "dataset_version": "vitextvqa:dev",
            },
        ]
        grouped = group_samples_by_image(samples)
        self.assertEqual(len(grouped), 2)

    def test_D_same_image_id_different_dataset_namespaces_do_not_collide(self):
        """
        Test D — Same image ID, different dataset namespace:
        Two records with the same image_id from different dataset namespaces produce different cache keys.
        """
        img = Image.new("RGB", (640, 480), color=(90, 90, 90))
        key_ds1 = build_image_cache_identity(
            img,
            image_id=22,
            image_path="data/vitextvqa/st_images/22.jpg",
            dataset_namespace="vitextvqa_20qa_gate_b",
        )
        key_ds2 = build_image_cache_identity(
            img,
            image_id=22,
            image_path="data/vitextvqa/st_images/22.jpg",
            dataset_namespace="vitextvqa_100img_gate_c",
        )
        self.assertNotEqual(key_ds1, key_ds2)
        self.assertNotEqual(key_ds1.to_cache_key(2), key_ds2.to_cache_key(2))

        cache = VisualFeatureCache(max_entries=8)
        f1 = cache.get_or_compute(
            self.model, _identity_transform, [img], image=img, device="cpu", image_key=key_ds1
        )
        f2 = cache.get_or_compute(
            self.model, _identity_transform, [img], image=img, device="cpu", image_key=key_ds2
        )
        self.assertEqual(self.model.extract_feature_calls, 2)
        self.assertFalse(torch.equal(f1, f2))

    def test_E_changed_image_content_or_fingerprint_invalidates_cached_state(self):
        """
        Test E — Image changes:
        If image content/version fingerprint changes while logical ID and dimensions remain the same,
        previous cached state is invalidated and not reused.
        """
        img_v1 = Image.new("RGB", (640, 480), color=(10, 20, 30))
        img_v2 = Image.new("RGB", (640, 480), color=(200, 210, 220))

        # 1. Pixel change under identical (dataset_version, image_id, width, height)
        key_v1 = build_image_cache_identity(
            img_v1, image_id=22, dataset_version="vitextvqa:dev"
        )
        key_v2 = build_image_cache_identity(
            img_v2, image_id=22, dataset_version="vitextvqa:dev"
        )
        self.assertEqual(key_v1.dataset_version, key_v2.dataset_version)
        self.assertEqual(key_v1.image_identity, key_v2.image_identity)
        self.assertEqual((key_v1.width, key_v1.height), (key_v2.width, key_v2.height))
        self.assertNotEqual(key_v1.content_fingerprint, key_v2.content_fingerprint)
        self.assertNotEqual(key_v1, key_v2)

        # 2. Explicit content_fingerprint change on identical image pixels & logical ID
        key_fp1 = build_image_cache_identity(
            img_v1, image_id=22, dataset_version="vitextvqa:dev", content_fingerprint="sha256:aaa111"
        )
        key_fp2 = build_image_cache_identity(
            img_v1, image_id=22, dataset_version="vitextvqa:dev", content_fingerprint="sha256:bbb222"
        )
        self.assertNotEqual(key_fp1, key_fp2)

        # 3. On-disk file modification invalidates cache
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "sub", "22.jpg")
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            img_v1.save(fpath, format="PNG")
            key_disk1 = build_image_cache_identity(
                img_v1, image_id=22, image_path=fpath, dataset_root=tmpdir, dataset_version="vitextvqa:dev"
            )
            # Rewrite file with different size/mtime
            with open(fpath, "ab") as f:
                f.write(b"\x00" * 64)
            key_disk2 = build_image_cache_identity(
                img_v1, image_id=22, image_path=fpath, dataset_root=tmpdir, dataset_version="vitextvqa:dev"
            )
            self.assertNotEqual(key_disk1, key_disk2)

        # Verify extractor and VisualFeatureCache invalidate cached state when key_v1 -> key_v2
        extractor = PointwiseKLAttentionExtractor(
            model=self.model,
            tokenizer=self.tokenizer,
            transform=_identity_transform,
            layer_start=0,
            layer_end=2,
            min_grid_tiles=1,
            max_grid_tiles=1,
            device="cpu",
        )
        r1 = extractor.extract(img_v1, "Q1", image_id=22, dataset_version="vitextvqa:dev")
        r2 = extractor.extract(img_v2, "Q1", image_id=22, dataset_version="vitextvqa:dev")
        self.assertEqual(self.model.extract_feature_calls, 2)
        self.assertFalse(torch.equal(r1.vit_embeds, r2.vit_embeds))

    def test_F_missing_stable_identity_falls_back_to_content_hash_with_fallback_flag(self):
        """
        Test F — Missing stable identity:
        When stable identity is omitted, the code safely falls back to content-based
        behavior and explicitly sets `is_fallback=True` without claiming collision-proof identity.
        """
        img_a = Image.new("RGB", (320, 240), color=(15, 25, 35))
        img_b = Image.new("RGB", (320, 240), color=(15, 25, 35))
        # Modify an interior pixel of img_b that is outside the 5 corners/center points
        # but captured by the 16x16 fallback thumbnail
        for y in range(20, 80):
            for x in range(20, 80):
                img_b.putpixel((x, y), (250, 10, 10))

        fb_a = build_image_cache_identity(img_a)
        fb_b = build_image_cache_identity(img_b)

        self.assertTrue(fb_a.is_fallback)
        self.assertTrue(fb_b.is_fallback)
        self.assertEqual(fb_a.dataset_version, UNSCOPED_DATASET_NAMESPACE)
        self.assertEqual(fb_a.image_identity, FALLBACK_IMAGE_IDENTITY)
        self.assertTrue(fb_a.content_fingerprint.startswith("thumb:"))
        self.assertNotEqual(fb_a, fb_b)

    def test_G_backward_compatibility_positional_callers(self):
        """
        Test G — Backward compatibility:
        Existing callers that pass only positional `(image, question)` or `(image, num_tiles)`
        and custom `BaseAttentionExtractor` implementations with 2-arg `extract(image, question)`
        continue to work without breakage.
        """
        img = Image.new("RGB", (448, 448), color=(80, 120, 160))

        # 1. PointwiseKLAttentionExtractor._make_image_key(image) positional call
        legacy_key = PointwiseKLAttentionExtractor._make_image_key(img)
        self.assertIsInstance(legacy_key, ImageCacheIdentity)
        self.assertEqual(len(legacy_key), 5)

        # 2. VisualFeatureCache.compute_image_key(image, num_tiles) positional call
        cache = VisualFeatureCache(max_entries=4)
        str_key = cache.compute_image_key(img, 2)
        self.assertIsInstance(str_key, str)

        # 3. Custom legacy extractor with strict 2-arg signature `(self, image, question)`
        class _LegacyTwoArgExtractor(PointwiseKLAttentionExtractor):
            def extract(self, image: Image.Image, question: str) -> ExtractionResult:
                phi = np.full((16, 16), 0.1, dtype=np.float32)
                phi[4:8, 4:8] = 0.9
                return ExtractionResult(
                    phi_map=phi,
                    r_map=phi,
                    aff_map=phi,
                    a_query_map=phi / phi.sum(),
                    a_gen_map=np.full((16, 16), 1.0 / 256.0, dtype=np.float32),
                    grid_ratio=(1, 1),
                    vit_embeds=torch.ones((1, 256, 32), dtype=torch.bfloat16),
                )

        pipeline = SGVCDPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            transform=_identity_transform,
            extractor=_LegacyTwoArgExtractor(
                model=self.model,
                tokenizer=self.tokenizer,
                transform=_identity_transform,
                layer_start=0,
                layer_end=2,
                min_grid_tiles=1,
                max_grid_tiles=1,
                device="cpu",
            ),
            propagator=None,
            device="cpu",
        )
        # Call pipeline.extract_visual_features(img) and pipeline.run(img, question) positionally
        vf = pipeline.extract_visual_features(img)
        self.assertEqual(vf.shape, (1, 256, 32))
        result = pipeline.run(img, "cửa tiệm tên gì ?")
        self.assertTrue(hasattr(result, "pred_text"))
        self.assertEqual(len(result.stage1_bbox), 4)

    def test_H_cache_lifecycle_invalidation_and_fresh_run_isolation(self):
        """
        Test H — Cache lifecycle:
        1. Changing image identity invalidates image-specific state in PointwiseKLAttentionExtractor.
        2. Starting a fresh SGVCDPipeline instance or calling `reset_cache()` does not retain
           stale cache entries from a previous evaluation run.
        """
        img1 = Image.new("RGB", (448, 448), color=(30, 60, 90))
        img2 = Image.new("RGB", (448, 448), color=(90, 60, 30))

        pipeline_run1 = SGVCDPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            transform=_identity_transform,
            propagator=None,
            device="cpu",
        )
        pipeline_run1.extractor.layer_start = 0
        pipeline_run1.extractor.layer_end = 2
        pipeline_run1.extractor.min_grid_tiles = 1
        pipeline_run1.extractor.max_grid_tiles = 1

        # Run 1 caches img1
        _ = pipeline_run1.run(
            img1,
            "Q1",
            image_id=1,
            dataset_version="vitextvqa:dev",
        )
        self.assertEqual(len(pipeline_run1.feature_cache._cache), 1)
        first_cached_key = pipeline_run1.extractor._cached_img_key
        self.assertIsNotNone(first_cached_key)

        # Switching to img2 invalidates extractor's previous image state and replaces key
        _ = pipeline_run1.run(
            img2,
            "Q2",
            image_id=2,
            dataset_version="vitextvqa:dev",
        )
        second_cached_key = pipeline_run1.extractor._cached_img_key
        self.assertNotEqual(first_cached_key, second_cached_key)
        self.assertEqual(len(pipeline_run1.feature_cache._cache), 2)

        # Starting a fresh SGVCDPipeline (Run 2) must have an isolated, empty feature cache
        pipeline_run2 = SGVCDPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            transform=_identity_transform,
            propagator=None,
            device="cpu",
        )
        self.assertIsNot(pipeline_run1.feature_cache, pipeline_run2.feature_cache)
        self.assertEqual(len(pipeline_run2.feature_cache._cache), 0)
        self.assertIsNone(pipeline_run2.extractor._cached_img_state)

        # Explicit reset_cache() on pipeline_run1 clears both feature_cache and extractor state
        pipeline_run1.reset_cache()
        self.assertEqual(len(pipeline_run1.feature_cache._cache), 0)
        self.assertIsNone(pipeline_run1.extractor._cached_img_state)
        self.assertIsNone(pipeline_run1.extractor._cached_img_key)

    def test_M01_relative_path_normalization_deep_paths_in_root_and_fallback(self):
        """
        M-01 — Relative path normalization hardening:
        1. Two deep paths with different parent directories do not collide (with or without dataset_root).
        2. A path inside dataset_root retains its complete relative structure without anchor/depth truncation.
        3. A path outside dataset_root follows the documented fallback behavior.
        4. Existing data/ and results/ path handling remains compatible when dataset_root is omitted.
        5. Windows-style and POSIX-style paths normalize consistently.
        6. Bare basename returns (basename, False).
        """
        # 1 & 2. In-root deep paths preserve complete relative structure and do not truncate to parts[-3:]
        n1, d1 = _normalize_relative_image_path("root/x/p/q/r/22.jpg", dataset_root="root")
        n2, d2 = _normalize_relative_image_path("root/y/p/q/r/22.jpg", dataset_root="root")
        self.assertEqual((n1, d1), ("x/p/q/r/22.jpg", True))
        self.assertEqual((n2, d2), ("y/p/q/r/22.jpg", True))
        self.assertNotEqual(n1, n2)

        # In-root path containing 'data' inside a subdirectory must NOT strip the parent subdirectory
        in_root_data1, _ = _normalize_relative_image_path(
            "/workspace/root/exp_a/data/imgs/22.jpg", dataset_root="/workspace/root"
        )
        in_root_data2, _ = _normalize_relative_image_path(
            "/workspace/root/exp_b/data/imgs/22.jpg", dataset_root="/workspace/root"
        )
        self.assertEqual(in_root_data1, "exp_a/data/imgs/22.jpg")
        self.assertEqual(in_root_data2, "exp_b/data/imgs/22.jpg")
        self.assertNotEqual(in_root_data1, in_root_data2)

        # Deep paths without dataset_root and without 'data'/'results' anchors must not collide
        abs1, ad1 = _normalize_relative_image_path("/mnt/ds1/a/b/c/22.jpg", dataset_root=None)
        abs2, ad2 = _normalize_relative_image_path("/mnt/ds2/a/b/c/22.jpg", dataset_root=None)
        self.assertTrue(ad1 and ad2)
        self.assertEqual(abs1, "mnt/ds1/a/b/c/22.jpg")
        self.assertEqual(abs2, "mnt/ds2/a/b/c/22.jpg")
        self.assertNotEqual(abs1, abs2)

        # 3. Path outside dataset_root falls back cleanly
        out1, od1 = _normalize_relative_image_path(
            "/other_mount/data/vitextvqa/st_images/22.jpg", dataset_root="/workspace/root"
        )
        self.assertEqual((out1, od1), ("data/vitextvqa/st_images/22.jpg", True))
        out2, od2 = _normalize_relative_image_path(
            "/other_mount/custom_set/split_a/22.jpg", dataset_root="/workspace/root"
        )
        self.assertEqual((out2, od2), ("other_mount/custom_set/split_a/22.jpg", True))

        # 4. Existing data/ and results/ anchor compatibility when dataset_root is omitted
        data_p, data_d = _normalize_relative_image_path(
            "D:\\Users\\host\\VLM_Reasoning\\data\\vitextvqa\\st_images\\22.jpg", dataset_root=None
        )
        res_p, res_d = _normalize_relative_image_path(
            "/content/VLM_Reasoning/results/swintextspotter/22.jpg", dataset_root=None
        )
        self.assertEqual((data_p, data_d), ("data/vitextvqa/st_images/22.jpg", True))
        self.assertEqual((res_p, res_d), ("results/swintextspotter/22.jpg", True))

        # 5. Windows-style vs POSIX-style normalization consistency
        win_norm, win_d = _normalize_relative_image_path(
            "D:\\proj\\split1\\sub\\22.jpg", dataset_root="D:\\proj"
        )
        pos_norm, pos_d = _normalize_relative_image_path(
            "/proj/split1/sub/22.jpg", dataset_root="/proj"
        )
        self.assertEqual((win_norm, win_d), ("split1/sub/22.jpg", True))
        self.assertEqual((win_norm, win_d), (pos_norm, pos_d))

        # 6. Bare basename
        bare_norm, bare_d = _normalize_relative_image_path("22.jpg", dataset_root=None)
        self.assertEqual((bare_norm, bare_d), ("22.jpg", False))

    def test_M02_precomputed_identity_consistency_validation(self):
        """
        M-02 — Validate precomputed ImageCacheIdentity consistency:
        1. Consistent precomputed identity is accepted (with omitted optional fields or matching fields).
        2. Conflicting image_id raises ValueError.
        3. Conflicting dataset_version raises ValueError.
        4. Conflicting dataset_namespace raises ValueError.
        5. Conflicting image_path or content_fingerprint raises ValueError.
        6. Dimension mismatch raises ValueError (never silently rebuilds).
        """
        img = Image.new("RGB", (640, 480), color=(50, 100, 150))
        pre_key = build_image_cache_identity(
            img,
            image_id=22,
            image_path="data/vitextvqa/st_images/22.jpg",
            dataset_version="vitextvqa:dev",
            content_fingerprint="sha256:abc123",
        )

        # 1. Omitted optional fields preserve fast-path backward compatibility
        accepted_omitted = build_image_cache_identity(img, image_key=pre_key)
        self.assertIs(accepted_omitted, pre_key)

        # Matching optional fields are accepted and return exact precomputed identity
        accepted_matching = build_image_cache_identity(
            img,
            image_key=pre_key,
            image_id="22",
            image_path="data/vitextvqa/st_images/22.jpg",
            dataset_version="vitextvqa:dev",
            dataset_namespace="vitextvqa:dev",
            content_fingerprint="sha256:abc123",
        )
        self.assertIs(accepted_matching, pre_key)

        # 2. Conflicting image_id raises ValueError
        with self.assertRaises(ValueError):
            build_image_cache_identity(img, image_key=pre_key, image_id=99)

        # 3. Conflicting dataset_version raises ValueError
        with self.assertRaises(ValueError):
            build_image_cache_identity(img, image_key=pre_key, dataset_version="vitextvqa:test")

        # 4. Conflicting dataset_namespace raises ValueError
        with self.assertRaises(ValueError):
            build_image_cache_identity(img, image_key=pre_key, dataset_namespace="other_ns")

        # Conflicting dataset_version vs dataset_namespace on same call raises ValueError
        with self.assertRaises(ValueError):
            build_image_cache_identity(
                img, image_id=22, dataset_version="v1", dataset_namespace="v2"
            )

        # 5. Conflicting image_path or content_fingerprint raises ValueError
        with self.assertRaises(ValueError):
            build_image_cache_identity(
                img, image_key=pre_key, image_path="data/vitextvqa/st_images/99.jpg"
            )
        with self.assertRaises(ValueError):
            build_image_cache_identity(
                img, image_key=pre_key, content_fingerprint="sha256:different"
            )

        # 6. Dimension mismatch raises ValueError
        other_size_img = Image.new("RGB", (800, 600), color=(50, 100, 150))
        with self.assertRaises(ValueError):
            build_image_cache_identity(other_size_img, image_key=pre_key)

    def test_M03_unified_image_grouping_sampling_and_sharding(self):
        """
        M-03 — Unified image grouping key across group_samples_by_image, --test, --test-images,
        and multi-worker sharding:
        1. Same image with multiple QA remains one group.
        2. Different image IDs with the same basename remain separate groups.
        3. Same image ID in different dataset namespaces remains separate groups.
        4. --test-images selects the expected number of logical images even with shared basenames.
        5. --test selects 1 QA from each of the first 3 distinct logical images even with shared basenames.
        6. Multi-worker sharding covers every selected logical image exactly once across workers.
        7. Existing 400-image / 1,222-QA manifest grouping behavior remains unchanged.
        """
        # Construct 4 distinct logical images that share basenames ('22.jpg' x3, '30.jpg' x1), 2 QAs each = 8 QAs
        def _mk_s(qid: int, ds: str, iid: int, name: str, path: str, ans: str):
            return {
                "sample_key": f"q{qid}",
                "question_id": qid,
                "quest_idx": 1,
                "dataset_version": ds,
                "image_id": iid,
                "image_name": name,
                "image_path": path,
                "image_size": [200, 200],
                "question": f"Q{qid}",
                "gt_answers": [ans],
                "verified_all_answers": [ans],
                "added_gt_answers": [],
                "answer_region_bbox": [0, 0, 10, 10],
                "sub_bboxes": [],
            }

        synthetic_samples = [
            # Logical image 0: ds=v1, id=101, name=22.jpg (2 QAs)
            _mk_s(1, "v1", 101, "22.jpg", "dir_a/22.jpg", "a"),
            _mk_s(2, "v1", 101, "22.jpg", "dir_a/22.jpg", "a"),
            # Logical image 1: ds=v1, id=102, name=22.jpg (same basename, different image_id & path; 2 QAs)
            _mk_s(3, "v1", 102, "22.jpg", "dir_b/22.jpg", "b"),
            _mk_s(4, "v1", 102, "22.jpg", "dir_b/22.jpg", "b"),
            # Logical image 2: ds=v2, id=101, name=22.jpg (same image_id & basename, different dataset_version; 2 QAs)
            _mk_s(5, "v2", 101, "22.jpg", "dir_a/22.jpg", "c"),
            _mk_s(6, "v2", 101, "22.jpg", "dir_a/22.jpg", "c"),
            # Logical image 3: ds=v1, id=103, name=30.jpg (2 QAs)
            _mk_s(7, "v1", 103, "30.jpg", "dir_a/30.jpg", "d"),
            _mk_s(8, "v1", 103, "30.jpg", "dir_a/30.jpg", "d"),
        ]

        # 1, 2, 3. group_samples_by_image produces 4 distinct logical image groups of 2 QAs each
        grouped = group_samples_by_image(synthetic_samples)
        self.assertEqual(len(grouped), 4)
        for g_list in grouped.values():
            self.assertEqual(len(g_list), 2)

        # Legacy sample without dataset_version or image_id falls back cleanly to path_or_name
        legacy_s = {"image_name": "legacy.jpg", "image_path": None}
        self.assertEqual(_sample_image_group_key(legacy_s), "legacy.jpg")

        # 4, 5, 6. Verify run_benchmark --test, --test-images, and multi-worker sharding in mock mode
        import argparse
        import json
        import shutil

        tmpdir = tempfile.mkdtemp(prefix="opt06_m03_")
        try:
            with patch(
                "scripts.benchmark.evaluate_vitextvqa.load_benchmark_dataset",
                side_effect=lambda *a, **kw: [dict(s) for s in synthetic_samples],
            ):
                # 4. --test-images=2 must select exactly the first 2 logical images (q1..q4 = 4 QAs),
                # NOT all 6 QAs sharing image_name=='22.jpg'
                out_ti = os.path.join(tmpdir, "test_images_2")
                args_ti = argparse.Namespace(
                    dataset="dummy.json",
                    data_dir="data/vitextvqa",
                    output_dir=out_ti,
                    method="sg_vcd",
                    model_name="mock",
                    device="cpu",
                    test=False,
                    test_images=2,
                    test_samples=0,
                    limit=0,
                    num_workers=1,
                    worker_id=0,
                    checkpoint=None,
                    overwrite=True,
                    mock=True,
                )
                run_benchmark(args_ti)
                with open(os.path.join(out_ti, "checkpoint.json"), encoding="utf-8") as f:
                    ckpt_ti = json.load(f)
                self.assertEqual(
                    [r["question_id"] for r in ckpt_ti["records"]],
                    [1, 2, 3, 4],
                )

                # 5. --test must select 1 QA from each of the first 3 distinct logical images (q1, q3, q5)
                out_test = os.path.join(tmpdir, "test_mode")
                args_test = argparse.Namespace(**{**vars(args_ti), "output_dir": out_test, "test": True, "test_images": 0})
                run_benchmark(args_test)
                with open(os.path.join(out_test, "checkpoint.json"), encoding="utf-8") as f:
                    ckpt_test = json.load(f)
                self.assertEqual(
                    [r["question_id"] for r in ckpt_test["records"]],
                    [1, 3, 5],
                )

                # 6. Multi-worker sharding (num_workers=2) across 4 logical images:
                # Worker 0 gets logical images 0 & 2 (q1, q2, q5, q6); Worker 1 gets logical images 1 & 3 (q3, q4, q7, q8)
                out_sh = os.path.join(tmpdir, "sharded")
                for wid in (0, 1):
                    args_w = argparse.Namespace(
                        **{**vars(args_ti), "output_dir": out_sh, "test_images": 0, "num_workers": 2, "worker_id": wid}
                    )
                    run_benchmark(args_w)
                with open(os.path.join(out_sh, "checkpoint_shard_0.json"), encoding="utf-8") as f0:
                    w0_qids = [r["question_id"] for r in json.load(f0)["records"]]
                with open(os.path.join(out_sh, "checkpoint_shard_1.json"), encoding="utf-8") as f1:
                    w1_qids = [r["question_id"] for r in json.load(f1)["records"]]
                self.assertEqual(w0_qids, [1, 2, 5, 6])
                self.assertEqual(w1_qids, [3, 4, 7, 8])
                self.assertEqual(sorted(w0_qids + w1_qids), [1, 2, 3, 4, 5, 6, 7, 8])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        # 7. Existing 400-image / 1,222-QA manifest grouping behavior remains unchanged
        manifest_path = os.path.join(
            PROJECT_ROOT, "results", "swintextspotter_auto_annot", "answer_bboxes_400_verified.json"
        )
        if os.path.isfile(manifest_path):
            with open(manifest_path, encoding="utf-8") as mf:
                raw_manifest = json.load(mf)
            manifest_samples = []
            ds_ver = f"answer_bboxes_400_verified:{raw_manifest.get('metadata', {}).get('split', 'dev')}"
            for img_entry in raw_manifest["images"]:
                for q_entry in img_entry.get("quests", []):
                    manifest_samples.append({
                        "dataset_version": ds_ver,
                        "image_id": img_entry.get("image_id"),
                        "image_name": img_entry["image_name"],
                        "image_path": os.path.join("data", "vitextvqa", "st_images", img_entry["image_name"]),
                        "question_id": q_entry["question_id"],
                    })
            grouped_400 = group_samples_by_image(manifest_samples)
            self.assertEqual(len(grouped_400), 400)
            self.assertEqual(sum(len(v) for v in grouped_400.values()), 1222)


if __name__ == "__main__":
    unittest.main()

