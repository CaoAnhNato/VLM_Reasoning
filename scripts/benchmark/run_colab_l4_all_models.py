#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HIGH-THROUGHPUT COLAB L4 MULTI-MODEL BENCHMARK SUITE FOR ViTextVQA (400 VERIFIED IMAGES / 1,222 QUESTIONS)

Key Capabilities:
1. 7 Official Baseline Models Only (Zero Unsloth Derivatives; SG-VCD excluded from re-run as scores already verified):
   - [1] 5CD-AI/Vintern-1B-v3_5
   - [2] erax-ai/EraX-VL-2B-V1.5
   - [3] Qwen/Qwen2.5-VL-3B-Instruct
   - [4] 5CD-AI/Vintern-3B-beta
   - [5] Qwen/Qwen2.5-VL-7B-Instruct
   - [6] erax-ai/EraX-VL-7B-V1.5
   - [7] CohereLabs/aya-vision-8b
2. Authenticated Hugging Face Hub Model Downloader:
   - Uses HF API Key (--hf-token or $HF_TOKEN) with snapshot_download(max_workers=8)
   - Pre-downloads weights once per model before spawning worker processes (prevents race conditions)
3. Full RAM Image Caching (/dev/shm + In-Memory PIL RGB Cache):
   - Pre-loads all 400 verified images into RAM using 5 parallel threads with tqdm progress
   - Stages images into /dev/shm (Linux RAM disk on Colab) so all 5 subprocesses read from RAM with zero disk I/O
4. 5-Process Parallel Execution + Hardware-Level Speed Optimizations:
   - Multi-process sharding (--num-workers 5) with live unified tqdm progress monitoring
   - Smart L4 24GB VRAM Guard: Runs 5 GPU processes on 1B models, and auto-balances process count + 5-thread
     prefetching + image-grouped dynamic batching + FlashAttention-2/SDPA + TF32 on 2B-8B models (or 5 processes
     with NF4 quantization if --worker-mode force_5proc is selected)
5. Complete Output Artifacts for Manual Colab Download:
   - Per-model logs, predictions (.jsonl & .csv), summary JSONs, master comparison CSV/Markdown, and a single
     ready-to-download ZIP archive (results/colab_l4_benchmark_bundle.zip).
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import datetime
import functools
import gc
import glob
import inspect
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# Ensure UTF-8 console output on Windows & Linux
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from PIL import Image
from tqdm import tqdm
import torch

# Enable hardware TF32 & cuDNN benchmark on Ampere/Ada Lovelace (NVIDIA L4)
if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

# Compatibility patches for InternVLChatModel / Vintern on modern transformers
if hasattr(torch.nn.Module, "__getattr__") and not getattr(torch.nn.Module, "_all_tied_weights_patched", False):
    _orig_nn_getattr = torch.nn.Module.__getattr__

    def _safe_nn_getattr(self, name):
        if name == "all_tied_weights_keys":
            val = self.__dict__.get("all_tied_weights_keys") or getattr(self, "_tied_weights_keys", None)
            return val if isinstance(val, dict) else {}
        return _orig_nn_getattr(self, name)

    torch.nn.Module.__getattr__ = _safe_nn_getattr
    torch.nn.Module._all_tied_weights_patched = True

try:
    import transformers.configuration_utils as config_utils
    config_utils.PretrainedConfig.has_no_defaults_at_init = True
    if hasattr(config_utils.PretrainedConfig, "to_diff_dict") and not getattr(config_utils.PretrainedConfig, "_safe_diff_dict_patched", False):
        _orig_diff_dict = config_utils.PretrainedConfig.to_diff_dict

        def _safe_diff_dict(cfg_self):
            try:
                return _orig_diff_dict(cfg_self)
            except Exception:
                return cfg_self.to_dict()

        config_utils.PretrainedConfig.to_diff_dict = _safe_diff_dict
        config_utils.PretrainedConfig._safe_diff_dict_patched = True
except Exception:
    pass

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.benchmark.metrics import (
    MetricsAggregator,
    compute_exact_match,
    compute_stage1_metrics,
    compute_stage2_metrics,
    compute_token_f1,
    normalize_answer,
)
from scripts.benchmark.evaluate_vitextvqa import (
    build_vintern_transform,
    load_vintern_pixel_values,
)
from sg_vcd import SGVCDPipeline
from sg_vcd.pipeline import strip_conversational_preamble


# ==============================================================================
# 1. OFFICIAL MODEL REGISTRY (NO UNSLOTH DERIVATIVES)
# ==============================================================================

OFFICIAL_MODEL_SUITE: List[Dict[str, Any]] = [
    {
        "id": 1,
        "key": "5CD-AI_Vintern-1B-v3_5",
        "display_name": "5CD-AI/Vintern-1B-v3_5",
        "repo_id": "5CD-AI/Vintern-1B-v3_5",
        "method": "baseline",
        "arch": "vintern",
        "est_vram_bf16_gb": 2.3,
        "recommended_batch_size": 8,
    },
    {
        "id": 2,
        "key": "erax-ai_EraX-VL-2B-V1.5",
        "display_name": "erax-ai/EraX-VL-2B-V1.5",
        "repo_id": "erax-ai/EraX-VL-2B-V1.5",
        "method": "baseline",
        "arch": "transformers_vlm",
        "est_vram_bf16_gb": 4.5,
        "recommended_batch_size": 8,
    },
    {
        "id": 3,
        "key": "Qwen_Qwen2.5-VL-3B-Instruct",
        "display_name": "Qwen/Qwen2.5-VL-3B-Instruct",
        "repo_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "method": "baseline",
        "arch": "transformers_vlm",
        "est_vram_bf16_gb": 7.6,
        "recommended_batch_size": 8,
    },
    {
        "id": 4,
        "key": "5CD-AI_Vintern-3B-beta",
        "display_name": "5CD-AI/Vintern-3B-beta",
        "repo_id": "5CD-AI/Vintern-3B-beta",
        "method": "baseline",
        "arch": "vintern",
        "est_vram_bf16_gb": 7.5,
        "recommended_batch_size": 6,
    },
    {
        "id": 5,
        "key": "Qwen_Qwen2.5-VL-7B-Instruct",
        "display_name": "Qwen/Qwen2.5-VL-7B-Instruct",
        "repo_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "method": "baseline",
        "arch": "transformers_vlm",
        "est_vram_bf16_gb": 16.2,
        "recommended_batch_size": 4,
    },
    {
        "id": 6,
        "key": "erax-ai_EraX-VL-7B-V1.5",
        "display_name": "erax-ai/EraX-VL-7B-V1.5",
        "repo_id": "erax-ai/EraX-VL-7B-V1.5",
        "method": "baseline",
        "arch": "transformers_vlm",
        "est_vram_bf16_gb": 14.3,
        "recommended_batch_size": 4,
    },
    {
        "id": 7,
        "key": "CohereLabs_aya-vision-8b",
        "display_name": "CohereLabs/aya-vision-8b",
        "repo_id": "CohereLabs/aya-vision-8b",
        "method": "baseline",
        "arch": "transformers_vlm",
        "est_vram_bf16_gb": 16.8,
        "recommended_batch_size": 4,
    },
]


def resolve_selected_models(models_arg: str) -> List[Dict[str, Any]]:
    """Resolve '--models all' or comma-separated IDs/names into model configs (filtering out any unsloth aliases)."""
    arg_clean = models_arg.strip()
    if arg_clean.lower() in ("all", "*", "default"):
        return [dict(m) for m in OFFICIAL_MODEL_SUITE]

    selected: List[Dict[str, Any]] = []
    tokens = [t.strip() for t in arg_clean.split(",") if t.strip()]
    for tok in tokens:
        # Map any legacy unsloth name to official repo
        tok_norm = tok.lower()
        if "unsloth" in tok_norm and "3b" in tok_norm and "qwen" in tok_norm:
            tok_norm = "qwen/qwen2.5-vl-3b-instruct"
        elif "unsloth" in tok_norm and "7b" in tok_norm and "qwen" in tok_norm:
            tok_norm = "qwen/qwen2.5-vl-7b-instruct"

        if tok_norm.isdigit():
            mid = int(tok_norm)
            for m in OFFICIAL_MODEL_SUITE:
                if m["id"] == mid:
                    selected.append(dict(m))
                    break
            continue

        matched = False
        for m in OFFICIAL_MODEL_SUITE:
            if (
                tok_norm == m["key"].lower()
                or tok_norm == m["repo_id"].lower()
                or tok_norm == m["display_name"].lower()
                or tok_norm in m["key"].lower()
            ):
                if m["key"] not in {x["key"] for x in selected}:
                    selected.append(dict(m))
                matched = True
        if not matched:
            # Custom official HF repo
            safe_k = re.sub(r"[^a-zA-Z0-9_\-]", "_", tok)
            selected.append({
                "id": len(selected) + 1,
                "key": safe_k,
                "display_name": tok,
                "repo_id": tok,
                "method": "baseline",
                "arch": "vintern" if "vintern" in tok_norm else "transformers_vlm",
                "est_vram_bf16_gb": 8.0,
                "recommended_batch_size": 4,
            })
    return selected


# ==============================================================================
# 2. HUGGING FACE API AUTHENTICATION & FAST MODEL DOWNLOADER
# ==============================================================================

def authenticate_hf(hf_token: Optional[str] = None) -> Optional[str]:
    """Authenticate with Hugging Face Hub using API key and enable high-speed transfer."""
    token = (
        hf_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or None
    )
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    if token:
        os.environ["HF_TOKEN"] = token
        try:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)
            print("[HF-AUTH] Successfully authenticated with Hugging Face Hub API Token.")
        except Exception as e:
            print(f"[HF-AUTH] Token set in environment ({e}).")
    else:
        print("[HF-AUTH WARN] No HF_TOKEN provided! Gated models (e.g. CohereLabs/aya-vision-8b) may require --hf-token.")
    return token


def preload_model_weights_from_hf(repo_id: str, hf_token: Optional[str] = None, max_dl_workers: int = 8) -> str:
    """
    Pre-download model weights into local HF cache using snapshot_download with parallel workers
    and tqdm progress bar BEFORE spawning inference worker processes.
    """
    from huggingface_hub import snapshot_download

    print(f"\n[HF-DOWNLOAD] Pre-downloading official model '{repo_id}' via Hugging Face API (max_workers={max_dl_workers})...")
    t0 = time.perf_counter()
    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        token=hf_token,
        max_workers=max_dl_workers,
        ignore_patterns=["*.msgpack", "*.h5", "consolidated.safetensors"],
    )
    dt = time.perf_counter() - t0
    print(f"[HF-DOWNLOAD] Model '{repo_id}' ready in cache ({dt:.1f}s): {local_dir}")
    return local_dir


def evict_model_hf_cache(repo_id: str):
    """Delete cached weights of a finished model to prevent Colab disk exhaustion."""
    target_repo = "models--" + repo_id.replace("/", "--")
    candidate_dirs = [
        os.environ.get("HF_HOME", ""),
        os.path.join(os.environ.get("HF_HOME", ""), "hub") if os.environ.get("HF_HOME") else "",
        os.environ.get("HUGGINGFACE_HUB_CACHE", ""),
        os.path.expanduser("~/.cache/huggingface/hub"),
        "/root/.cache/huggingface/hub",
        "/content/.hf_cache/hub",
    ]
    for cdir in set(candidate_dirs):
        if not cdir or not os.path.isdir(cdir):
            continue
        p = os.path.join(cdir, target_repo)
        if os.path.isdir(p):
            print(f"[CACHE-CLEANUP] Removing cached weights: {p}")
            shutil.rmtree(p, ignore_errors=True)


# ==============================================================================
# 3. DATASET LOADER & IN-RAM IMAGE CACHE (/dev/shm + ThreadPool RGB Preloader)
# ==============================================================================

def load_verified_400_dataset(
    dataset_path: str,
    data_dir: str = "data/vitextvqa",
    ram_stage_dir: Optional[str] = None,
    preload_threads: int = 5,
     quiet: bool = False,
) -> List[Dict[str, Any]]:
    """
    Load the 400 verified images (1,222 questions) and optionally stage all image files
    into /dev/shm (Linux RAM disk) so all 5 worker processes read directly from RAM.
    """
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Verified dataset JSON not found: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    candidate_dirs = [
        os.path.join(PROJECT_ROOT, "data", "xanylabeling_workspace", "Cao_Anh_001_100"),
        os.path.join(PROJECT_ROOT, "data", "xanylabeling_workspace", "Trieu_101_200"),
        os.path.join(PROJECT_ROOT, "data", "xanylabeling_workspace", "Cam_Tu_201_300"),
        os.path.join(PROJECT_ROOT, "data", "xanylabeling_workspace", "Thanh_Tam_301_400"),
        os.path.join(PROJECT_ROOT, "data", "vitextvqa", "st_images"),
        os.path.join(PROJECT_ROOT, data_dir, "st_images"),
        os.path.join(PROJECT_ROOT, data_dir),
    ]
    if ram_stage_dir and os.path.isdir(ram_stage_dir):
        candidate_dirs.insert(0, ram_stage_dir)

    def _find_img(name: str) -> Optional[str]:
        for d in candidate_dirs:
            if d and os.path.isdir(d):
                p = os.path.join(d, name)
                if os.path.isfile(p) and os.path.getsize(p) > 0:
                    return p
        return None

    samples: List[Dict[str, Any]] = []
    images_list = raw_data.get("images", []) if isinstance(raw_data, dict) else []

    # Check if any images are missing locally and fetch from HF zip if needed
    missing_names: Set[str] = set()
    resolved_map: Dict[str, str] = {}
    for img_entry in images_list:
        iname = img_entry["image_name"]
        p = _find_img(iname)
        if p:
            resolved_map[iname] = p
        else:
            missing_names.add(iname)

    if missing_names:
        fallback_cache = os.path.join(PROJECT_ROOT, "data", "vitextvqa", "st_images")
        os.makedirs(fallback_cache, exist_ok=True)
        try:
            from huggingface_hub import HfFileSystem
            fs = HfFileSystem(token=os.environ.get("HF_TOKEN"))
            zip_uri = "datasets/nhonhoccode/ViTextVQA/images.zip"
            if not quiet:
                print(f"[DATA] Fetching {len(missing_names)} missing images from {zip_uri}...")
            with fs.open(zip_uri, "rb") as zf_stream:
                with zipfile.ZipFile(zf_stream) as zf:
                    name_lookup = {os.path.basename(n): n for n in zf.namelist() if not n.endswith("/")}
                    for target in sorted(missing_names):
                        if target in name_lookup:
                            dst = os.path.join(fallback_cache, target)
                            with open(dst, "wb") as out_f:
                                out_f.write(zf.read(name_lookup[target]))
                            resolved_map[target] = dst
        except Exception as e:
            if not quiet:
                print(f"[DATA WARN] Could not fetch missing images from remote zip: {e}")

    # Stage images into /dev/shm RAM disk if requested
    if ram_stage_dir:
        os.makedirs(ram_stage_dir, exist_ok=True)
        to_copy = []
        for iname, src_p in list(resolved_map.items()):
            dst_p = os.path.join(ram_stage_dir, iname)
            if not os.path.isfile(dst_p) or os.path.getsize(dst_p) == 0:
                to_copy.append((iname, src_p, dst_p))
            else:
                resolved_map[iname] = dst_p

        if to_copy and not quiet:
            def _copy_one(item):
                iname, s_p, d_p = item
                shutil.copy2(s_p, d_p)
                return iname, d_p

            with ThreadPoolExecutor(max_workers=max(1, preload_threads)) as ex:
                futs = [ex.submit(_copy_one, item) for item in to_copy]
                for fut in tqdm(as_completed(futs), total=len(futs), desc="[RAM-DISK] Staging 400 images to /dev/shm"):
                    iname, d_p = fut.result()
                    resolved_map[iname] = d_p

    global_idx = 0
    for img_entry in images_list:
        img_name = img_entry["image_name"]
        img_path = resolved_map.get(img_name) or _find_img(img_name)
        w = img_entry.get("width", 0)
        h = img_entry.get("height", 0)
        for q in img_entry.get("quests", []):
            qid = q["question_id"]
            gts = q.get("gt_answers", [])
            ver_gts = q.get("verified_all_answers") or gts
            samples.append({
                "global_index": global_idx,
                "sample_key": f"{qid}_{img_name}",
                "image_name": img_name,
                "image_path": img_path,
                "image_size": [w, h],
                "question_id": qid,
                "quest_idx": q.get("quest_idx", 1),
                "question": q["question"],
                "gt_answers": gts,
                "verified_all_answers": ver_gts,
                "added_gt_answers": q.get("added_gt_answers", []),
                "answer_region_bbox": q.get("answer_region_bbox", [0, 0, 0, 0]),
                "sub_bboxes": q.get("sub_bboxes", []),
            })
            global_idx += 1

    if not quiet:
        print(f"[DATA] Loaded {len(samples)} verified questions across {len(images_list)} images.")
    return samples


class InMemoryRGBCache:
    """
    Pre-decodes and caches all target PIL RGB images in System RAM using 5 parallel threads with tqdm.
    400 images in RGB occupy ~320-380 MB of System RAM, eliminating 100% of disk read & JPEG decode latency.
    """

    def __init__(self, image_paths: Sequence[str], num_threads: int = 5, show_progress: bool = True, worker_tag: str = ""):
        self.cache: Dict[str, Image.Image] = {}
        unique_paths = sorted({p for p in image_paths if p and os.path.isfile(p)})
        if not unique_paths:
            return

        def _decode_rgb(path: str) -> Tuple[str, Optional[Image.Image]]:
            try:
                with Image.open(path) as im:
                    rgb = im.convert("RGB")
                    rgb.load()
                return path, rgb
            except Exception:
                return path, None

        desc = f"{worker_tag}[RAM-CACHE] Pre-decoding {len(unique_paths)} images into RAM"
        with ThreadPoolExecutor(max_workers=max(1, num_threads)) as pool:
            futures = [pool.submit(_decode_rgb, p) for p in unique_paths]
            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=len(futures), desc=desc, leave=False)
            for fut in iterator:
                path, img = fut.result()
                if img is not None:
                    self.cache[path] = img
                    self.cache[os.path.basename(path)] = img

    def get(self, image_path: Optional[str], image_name: Optional[str] = None) -> Optional[Image.Image]:
        if image_path and image_path in self.cache:
            return self.cache[image_path]
        if image_name and image_name in self.cache:
            return self.cache[image_name]
        if image_path and os.path.isfile(image_path):
            try:
                with Image.open(image_path) as im:
                    rgb = im.convert("RGB")
                    rgb.load()
                self.cache[image_path] = rgb
                return rgb
            except Exception:
                return None
        return None


# ==============================================================================
# 4. UNIFIED MODEL ENGINE (VINTERN BASELINE, SG-VCD, QWEN2.5-VL, ERAX-VL, AYA-8B)
# ==============================================================================

class UnifiedModelWorkerEngine:
    """
    Loads any of the 8 target configurations using official Hugging Face Transformers / AutoModel
    with FlashAttention-2 / SDPA, optional 4-bit NF4 quantization (for 5-process heavy models),
    and image-grouped batching.
    """

    def __init__(
        self,
        repo_id: str,
        method: str = "baseline",
        arch: str = "transformers_vlm",
        device: str = "cuda",
        load_in_4bit: bool = False,
        hf_token: Optional[str] = None,
        mock: bool = False,
    ):
        self.repo_id = repo_id
        self.method = method
        self.arch = arch
        self.device = device if torch.cuda.is_available() else "cpu"
        self.load_in_4bit = load_in_4bit
        self.hf_token = hf_token or os.environ.get("HF_TOKEN") or None
        self.mock = mock

        self.model = None
        self.tokenizer = None
        self.processor = None
        self.transform = None
        self.sg_vcd_pipeline: Optional[SGVCDPipeline] = None

        if not self.mock:
            self._load()

    def _get_quant_config(self):
        if not self.load_in_4bit:
            return None
        try:
            from transformers import BitsAndBytesConfig
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        except Exception as e:
            print(f"[QUANT WARN] BitsAndBytesConfig unavailable ({e}), falling back to bfloat16.")
            return None

    def _load(self):
        hf_kwargs = {"token": self.hf_token} if self.hf_token else {}
        quant_cfg = self._get_quant_config()

        if "vintern" in self.repo_id.lower() or self.arch == "vintern":
            from transformers import AutoConfig, AutoModel, AutoTokenizer

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.repo_id, trust_remote_code=True, use_fast=False, **hf_kwargs)
            except Exception:
                self.tokenizer = AutoTokenizer.from_pretrained(self.repo_id, trust_remote_code=True, **hf_kwargs)

            config = AutoConfig.from_pretrained(self.repo_id, trust_remote_code=True, **hf_kwargs)
            config.has_no_defaults_at_init = True

            load_kwargs = dict(
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                **hf_kwargs,
            )
            if quant_cfg is not None and self.method != "sg_vcd":
                load_kwargs["quantization_config"] = quant_cfg
                load_kwargs["device_map"] = {"": self.device}
                self.model = AutoModel.from_pretrained(self.repo_id, **load_kwargs).eval()
            else:
                self.model = AutoModel.from_pretrained(self.repo_id, **load_kwargs).eval().to(self.device)

            # Patch duplicate return_dict on transformers >= 4.45
            if hasattr(self.model, "language_model") and hasattr(self.model.language_model, "generate"):
                _orig_llm_gen = self.model.language_model.generate

                @functools.wraps(_orig_llm_gen)
                def _safe_llm_gen(*args, **kwargs):
                    kwargs.pop("return_dict", None)
                    return _orig_llm_gen(*args, **kwargs)

                self.model.language_model.generate = _safe_llm_gen

            if hasattr(self.model, "generate"):
                _orig_model_gen = self.model.generate

                @functools.wraps(_orig_model_gen)
                def _safe_model_gen(*args, **kwargs):
                    kwargs.pop("return_dict", None)
                    return _orig_model_gen(*args, **kwargs)

                self.model.generate = _safe_model_gen

            self.transform = build_vintern_transform(input_size=448)
            if self.method == "sg_vcd":
                self.sg_vcd_pipeline = SGVCDPipeline(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    transform=self.transform,
                    device=self.device,
                )
            return

        # Standard Multimodal VLMs: Qwen2.5-VL, EraX-VL, CohereLabs/aya-vision-8b
        import transformers
        from transformers import AutoProcessor, AutoTokenizer

        try:
            self.processor = AutoProcessor.from_pretrained(self.repo_id, trust_remote_code=True, **hf_kwargs)
        except Exception:
            self.processor = AutoTokenizer.from_pretrained(self.repo_id, trust_remote_code=True, **hf_kwargs)

        # Configure Left Padding for batched generation
        if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
            self.processor.tokenizer.padding_side = "left"
            if self.processor.tokenizer.pad_token is None and self.processor.tokenizer.eos_token is not None:
                self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        candidate_classes = []
        for cls_name in [
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
            "Qwen2_5_VLForConditionalGeneration",
            "AutoModelForMultimodalLM",
            "AutoModelForCausalLM",
            "AutoModel",
        ]:
            if hasattr(transformers, cls_name):
                candidate_classes.append((cls_name, getattr(transformers, cls_name)))

        attn_impl = "sdpa"
        if torch.cuda.is_available():
            try:
                import flash_attn  # noqa: F401
                attn_impl = "flash_attention_2"
            except Exception:
                attn_impl = "sdpa"

        base_kwargs = dict(
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map={"": self.device} if torch.cuda.is_available() else "auto",
            **hf_kwargs,
        )
        if quant_cfg is not None:
            base_kwargs["quantization_config"] = quant_cfg

        last_err = None
        for cls_name, cls_obj in candidate_classes:
            for use_attn in (True, False):
                try:
                    call_kwargs = dict(base_kwargs)
                    if use_attn:
                        call_kwargs["attn_implementation"] = attn_impl
                    self.model = cls_obj.from_pretrained(self.repo_id, **call_kwargs).eval()
                    return
                except Exception as e:
                    last_err = e
                    continue

        raise RuntimeError(f"Failed to load model '{self.repo_id}': {last_err}")

    def infer_image_questions(
        self,
        image: Image.Image,
        items: List[Dict[str, Any]],
        batch_size: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Run inference on all questions belonging to a single pre-cached PIL RGB image.
        Reuses visual features/pixel_values across questions on the same image.
        """
        W, H = image.size
        results: List[Dict[str, Any]] = []

        # Case A: Mock mode for fast local testing
        if self.mock:
            for item in items:
                gts = item["gt_answers"]
                gt_bbox = item["answer_region_bbox"]
                results.append({
                    "pred_answer": gts[0] if gts else "đáp án",
                    "pred_bbox": gt_bbox if (gt_bbox and gt_bbox[2] > gt_bbox[0]) else [0, 0, W, H],
                    "spatial_confidence": 0.95,
                    "latency_s": 0.005,
                })
            return results

        # Case B: Proposed SG-VCD Pipeline on Vintern-1B
        if self.method == "sg_vcd" and self.sg_vcd_pipeline is not None:
            cached_vf = None
            if hasattr(self.sg_vcd_pipeline, "extract_visual_features"):
                try:
                    with torch.inference_mode():
                        cached_vf = self.sg_vcd_pipeline.extract_visual_features(image)
                except Exception:
                    cached_vf = None

            sig = inspect.signature(self.sg_vcd_pipeline.run)
            supports_cache = "cached_visual_features" in sig.parameters

            for item in items:
                t0 = time.perf_counter()
                pred_answer = ""
                pred_bbox = [0, 0, W, H]
                spatial_conf = 0.0
                try:
                    with torch.inference_mode():
                        kw = dict(max_new_tokens=64)
                        if supports_cache and cached_vf is not None:
                            kw["cached_visual_features"] = cached_vf
                        res = self.sg_vcd_pipeline.run(image, item["question"], **kw)
                        if hasattr(res, "pred_text"):
                            pred_answer = res.pred_text
                        elif hasattr(res, "pred_sg_vcd"):
                            pred_answer = res.pred_sg_vcd
                        else:
                            pred_answer = str(res)

                        if hasattr(res, "stage1_bbox"):
                            pred_bbox = list(res.stage1_bbox)
                        elif hasattr(res, "selection") and hasattr(res.selection, "pixel_bbox"):
                            pred_bbox = list(res.selection.pixel_bbox)
                        spatial_conf = float(getattr(res, "spatial_confidence", getattr(getattr(res, "selection", None), "kl_score", 0.0)))
                except Exception as e:
                    print(f"[SG-VCD WARN] QID {item['question_id']} error: {e}")
                dt = time.perf_counter() - t0
                results.append({
                    "pred_answer": strip_conversational_preamble(pred_answer, question=item["question"]),
                    "pred_bbox": pred_bbox,
                    "spatial_confidence": spatial_conf,
                    "latency_s": dt,
                })
            del cached_vf
            return results

        # Case C: Vintern Baseline (1B or 3B) — extract pixel_values once per image & batch_chat
        if "vintern" in self.repo_id.lower() or self.arch == "vintern":
            t_img_start = time.perf_counter()
            max_patch = getattr(getattr(self.model, "config", None), "max_dynamic_patch", 6)
            pv = load_vintern_pixel_values(image, self.transform, input_size=448, max_num=max_patch)
            if pv is not None:
                pv = pv.to(torch.bfloat16).to(self.device)
            pv_time = time.perf_counter() - t_img_start

            gen_cfg = dict(max_new_tokens=64, do_sample=False)
            gen_cfg.pop("return_dict", None)

            eff_bs = max(1, batch_size)
            for i in range(0, len(items), eff_bs):
                chunk = items[i : i + eff_bs]
                prompts = [
                    f"<image>\nTrả lời thật ngắn gọn câu hỏi sau dựa vào chữ xuất hiện trong ảnh: {c['question']}"
                    for c in chunk
                ]
                t0 = time.perf_counter()
                preds: List[str] = []
                if hasattr(self.model, "batch_chat") and pv is not None and len(chunk) > 1:
                    try:
                        num_patches_list = [pv.shape[0]] * len(chunk)
                        pv_batch = torch.cat([pv] * len(chunk), dim=0)
                        with torch.inference_mode():
                            res_list = self.model.batch_chat(
                                self.tokenizer,
                                pv_batch,
                                prompts,
                                gen_cfg,
                                num_patches_list=num_patches_list,
                            )
                        if isinstance(res_list, list) and len(res_list) == len(chunk):
                            preds = [str(r).strip() for r in res_list]
                    except Exception:
                        preds = []

                if not preds:
                    with torch.inference_mode():
                        for p_text in prompts:
                            try:
                                r = self.model.chat(self.tokenizer, pv, p_text, gen_cfg, history=None, return_history=False)
                                ans = r[0] if isinstance(r, (tuple, list)) else str(r)
                                preds.append(ans.strip())
                            except Exception:
                                preds.append("")

                dt = (time.perf_counter() - t0) + (pv_time if i == 0 else 0.0)
                per_lat = dt / max(1, len(chunk))
                for c, p_ans in zip(chunk, preds):
                    results.append({
                        "pred_answer": p_ans,
                        "pred_bbox": [0, 0, W, H],
                        "spatial_confidence": 1.0,
                        "latency_s": per_lat,
                    })
            del pv
            return results

        # Case D: Transformers Native VLMs (Qwen2.5-VL 3B/7B, EraX-VL 2B/7B, Aya-Vision 8B)
        eff_bs = max(1, batch_size)
        model_dev = getattr(self.model, "device", self.device)
        for i in range(0, len(items), eff_bs):
            chunk = items[i : i + eff_bs]
            prompts = [
                f"Trả lời thật ngắn gọn câu hỏi sau dựa vào chữ xuất hiện trong ảnh: {c['question']}"
                for c in chunk
            ]
            images_batch = [image] * len(chunk)
            t0 = time.perf_counter()
            preds: List[str] = []

            try:
                if hasattr(self.processor, "apply_chat_template"):
                    messages_batch = [
                        [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": p}]}]
                        for img, p in zip(images_batch, prompts)
                    ]
                    try:
                        inputs = self.processor.apply_chat_template(
                            messages_batch if len(chunk) > 1 else messages_batch[0],
                            padding=True,
                            add_generation_prompt=True,
                            tokenize=True,
                            return_dict=True,
                            return_tensors="pt",
                        ).to(model_dev)
                    except Exception:
                        text_inputs = [
                            self.processor.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
                            for m in messages_batch
                        ]
                        inputs = self.processor(
                            text=text_inputs,
                            images=images_batch,
                            padding=True,
                            return_tensors="pt",
                        ).to(model_dev)
                else:
                    inputs = self.processor(images=images_batch, text=prompts, padding=True, return_tensors="pt").to(model_dev)

                with torch.inference_mode():
                    out_ids = self.model.generate(**inputs, max_new_tokens=64, do_sample=False, use_cache=True)
                if hasattr(inputs, "input_ids"):
                    out_ids = out_ids[:, inputs.input_ids.shape[1] :]

                if hasattr(self.processor, "batch_decode"):
                    preds = [s.strip() for s in self.processor.batch_decode(out_ids, skip_special_tokens=True)]
                elif hasattr(self.processor, "tokenizer") and hasattr(self.processor.tokenizer, "batch_decode"):
                    preds = [s.strip() for s in self.processor.tokenizer.batch_decode(out_ids, skip_special_tokens=True)]
                else:
                    preds = [str(o).strip() for o in out_ids]
            except Exception:
                # Safe sequential fallback per question
                preds = []
                for p_text in prompts:
                    try:
                        msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": p_text}]}]
                        try:
                            inps = self.processor.apply_chat_template(
                                msgs, padding=True, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
                            ).to(model_dev)
                        except Exception:
                            t_in = self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
                            inps = self.processor(text=[t_in], images=[image], return_tensors="pt").to(model_dev)
                        with torch.inference_mode():
                            o_ids = self.model.generate(**inps, max_new_tokens=64, do_sample=False, use_cache=True)
                        if hasattr(inps, "input_ids"):
                            o_ids = o_ids[:, inps.input_ids.shape[1] :]
                        if hasattr(self.processor, "batch_decode"):
                            ans = self.processor.batch_decode(o_ids, skip_special_tokens=True)[0]
                        else:
                            ans = self.processor.tokenizer.batch_decode(o_ids, skip_special_tokens=True)[0]
                        preds.append(str(ans).strip())
                    except Exception as e_seq:
                        print(f"[VLM WARN] Fallback generation error: {e_seq}")
                        preds.append("")

            dt = time.perf_counter() - t0
            per_lat = dt / max(1, len(chunk))
            for c, p_ans in zip(chunk, preds):
                results.append({
                    "pred_answer": p_ans,
                    "pred_bbox": [0, 0, W, H],
                    "spatial_confidence": 1.0,
                    "latency_s": per_lat,
                })
        return results

    def unload(self):
        del self.sg_vcd_pipeline
        del self.model
        del self.processor
        del self.tokenizer
        self.sg_vcd_pipeline = None
        self.model = None
        self.processor = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


# ==============================================================================
# 5. WORKER SHARD EXECUTION (SINGLE SHARD SUBPROCESS)
# ==============================================================================

def run_worker_shard(args: argparse.Namespace):
    """
    Executes a single worker shard (0 <= worker_id < num_workers) for one model.
    Preloads its assigned images into RAM (using 5 threads) and streams predictions to shard JSONL.
    """
    os.makedirs(args.model_out_dir, exist_ok=True)
    log_path = os.path.join(args.model_out_dir, f"worker_{args.worker_id}.log")
    shard_jsonl = os.path.join(args.model_out_dir, f"predictions_shard_{args.worker_id}.jsonl")
    shard_meta = os.path.join(args.model_out_dir, f"meta_shard_{args.worker_id}.json")

    log_f = open(log_path, "a", encoding="utf-8")

    def _wlog(msg: str):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}][W{args.worker_id}/{args.num_workers}] {msg}"
        log_f.write(line + "\n")
        log_f.flush()
        if args.num_workers == 1:
            print(line)

    samples = load_verified_400_dataset(
        args.dataset,
        data_dir=args.data_dir,
        ram_stage_dir=args.ram_stage_dir,
        preload_threads=5,
        quiet=True,
    )
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]

    # Group by image first so all questions of an image stay on the same worker (100% visual cache hit!)
    grouped_all: Dict[str, List[Dict[str, Any]]] = {}
    for s in samples:
        grouped_all.setdefault(s["image_name"], []).append(s)

    unique_img_names = list(grouped_all.keys())
    my_img_names = [
        iname for idx, iname in enumerate(unique_img_names)
        if (idx % args.num_workers) == args.worker_id
    ]
    my_samples = [s for iname in my_img_names for s in grouped_all[iname]]

    # Check existing completed sample keys in shard_jsonl for crash-proof resume
    done_keys: Set[str] = set()
    if not args.overwrite and os.path.isfile(shard_jsonl):
        with open(shard_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        done_keys.add(rec["sample_key"])
                    except Exception:
                        pass
    elif args.overwrite and os.path.isfile(shard_jsonl):
        try:
            os.remove(shard_jsonl)
        except Exception:
            pass

    remaining_by_img: Dict[str, List[Dict[str, Any]]] = {}
    for iname in my_img_names:
        rem = [s for s in grouped_all[iname] if s["sample_key"] not in done_keys]
        if rem:
            remaining_by_img[iname] = rem

    total_rem_questions = sum(len(v) for v in remaining_by_img.values())
    _wlog(f"Assigned {len(my_img_names)} images ({len(my_samples)} questions). Remaining: {total_rem_questions} questions.")

    if total_rem_questions == 0:
        _wlog("Shard already 100% completed.")
        log_f.close()
        return

    # Pre-cache all assigned images into Worker RAM using 5 threads
    rem_paths = [
        remaining_by_img[iname][0]["image_path"]
        for iname in remaining_by_img
        if remaining_by_img[iname][0].get("image_path")
    ]
    ram_cache = InMemoryRGBCache(
        rem_paths,
        num_threads=5,
        show_progress=(args.num_workers == 1),
        worker_tag=f"[W{args.worker_id}] ",
    )
    _wlog(f"Cached {len(ram_cache.cache) // 2} RGB images in System RAM.")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t_load0 = time.perf_counter()
    engine = UnifiedModelWorkerEngine(
        repo_id=args.repo_id,
        method=args.method,
        arch=args.arch,
        device=args.device,
        load_in_4bit=args.load_in_4bit,
        hf_token=args.hf_token,
        mock=args.mock,
    )
    load_time_s = time.perf_counter() - t_load0
    _wlog(f"Loaded {args.repo_id} (method={args.method}, 4bit={args.load_in_4bit}) in {load_time_s:.2f}s.")

    pbar = tqdm(
        total=total_rem_questions,
        desc=f"[GPU] {args.display_name}",
        disable=(args.num_workers > 1),
        dynamic_ncols=True,
    )

    with open(shard_jsonl, "a", encoding="utf-8") as out_f:
        for iname, q_items in remaining_by_img.items():
            img_path = q_items[0].get("image_path")
            image = ram_cache.get(img_path, iname)
            if image is None:
                if args.mock:
                    image = Image.new("RGB", (224, 224), color=(128, 128, 128))
                else:
                    _wlog(f"[SKIP] Missing image {iname}")
                    continue

            W, H = image.size
            outs = engine.infer_image_questions(image, q_items, batch_size=args.batch_size)

            for item, out in zip(q_items, outs):
                pred_ans = out["pred_answer"]
                pred_bbox = out["pred_bbox"]
                lat_s = out["latency_s"]
                gts = item["gt_answers"]
                ver_gts = item["verified_all_answers"]

                s1 = compute_stage1_metrics(pred_bbox, item["answer_region_bbox"], item["sub_bboxes"])
                s2 = compute_stage2_metrics(pred_ans, gts, ver_gts, question=item["question"])

                # Also compute raw strict ViTextVQA F1/EM for backward-compatible summary
                raw_em = 1.0 if any(normalize_answer(pred_ans) == normalize_answer(g) for g in gts if normalize_answer(g)) else 0.0
                raw_f1 = compute_token_f1(pred_ans, gts)

                record = {
                    "global_index": item["global_index"],
                    "sample_key": item["sample_key"],
                    "question_id": item["question_id"],
                    "image_name": iname,
                    "image_size": [W, H],
                    "question": item["question"],
                    "gt_answers": gts,
                    "verified_all_answers": ver_gts,
                    "gt_bbox": item["answer_region_bbox"],
                    "pred_bbox": pred_bbox,
                    "sub_bboxes": item["sub_bboxes"],
                    "pred_answer": pred_ans,
                    "raw_em": raw_em,
                    "raw_f1": round(raw_f1 * 100.0, 2),
                    "stage1_metrics": s1,
                    "stage2_metrics": s2,
                    "spatial_confidence": out["spatial_confidence"],
                    "latency_s": round(lat_s, 4),
                    "worker_id": args.worker_id,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                _wlog(
                    f"QID={item['question_id']} ({iname}) | F1(Ver)={s2['f1_verified']*100:.1f}% | "
                    f"Sub={s2['substring_verified']*100:.1f}% | ANLS={s2['anls_verified']*100:.1f}% | "
                    f"EM={int(s2['em_verified'])} | {lat_s:.2f}s | Pred: {pred_ans}"
                )
                pbar.update(1)

    pbar.close()
    peak_vram_alloc = (torch.cuda.max_memory_allocated() / (1024 ** 3)) if torch.cuda.is_available() else 0.0
    peak_vram_res = (torch.cuda.max_memory_reserved() / (1024 ** 3)) if torch.cuda.is_available() else 0.0

    with open(shard_meta, "w", encoding="utf-8") as mf:
        json.dump(
            {
                "worker_id": args.worker_id,
                "load_time_s": round(load_time_s, 2),
                "peak_vram_allocated_gb": round(peak_vram_alloc, 2),
                "peak_vram_reserved_gb": round(peak_vram_res, 2),
            },
            mf,
            indent=2,
        )

    engine.unload()
    _wlog(f"Worker finished cleanly. Peak VRAM Allocated: {peak_vram_alloc:.2f} GB.")
    log_f.close()


# ==============================================================================
# 6. SMART VRAM PROCESS PLANNER & COORDINATOR WITH LIVE TQDM
# ==============================================================================

def plan_workers_for_model(
    model_cfg: Dict[str, Any],
    requested_workers: int = 5,
    worker_mode: str = "auto",
    force_4bit: bool = False,
) -> Tuple[int, bool, str]:
    """
    Compute (effective_num_workers, load_in_4bit, reason) for NVIDIA L4 (22.5 GB usable VRAM).
    - If worker_mode == 'force_5proc': Always spawns 5 processes; enables 4-bit NF4 automatically if 5 * BF16 > 20.5 GB.
    - If worker_mode == 'auto': Preserves 100% Native BF16 precision and scales process count up to requested_workers
      based on available GPU VRAM (e.g., 5 processes for 1B models, 3 for 2B, 2 for 3B, 1 process + batching for 7B/8B).
    """
    total_vram_gb = 22.5
    if torch.cuda.is_available():
        try:
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except Exception:
            pass

    usable_vram_gb = total_vram_gb * 0.88
    est_bf16 = float(model_cfg.get("est_vram_bf16_gb", 8.0))

    if worker_mode == "force_5proc":
        if requested_workers * est_bf16 <= usable_vram_gb and not force_4bit:
            return (
                requested_workers,
                False,
                f"{requested_workers} parallel GPU processes in Native BF16 (~{requested_workers * est_bf16:.1f} GB <= {usable_vram_gb:.1f} GB budget)",
            )
        else:
            return (
                requested_workers,
                True,
                f"{requested_workers} parallel GPU processes with 4-bit NF4 quantization to fit {total_vram_gb:.1f} GB VRAM",
            )

    # Default 'auto' mode: 100% Native BF16 fidelity + safe process count + 5-thread RAM image cache + dynamic batching
    if force_4bit:
        est_mem = max(1.8, est_bf16 * 0.38)
        w = max(1, min(requested_workers, int(usable_vram_gb // est_mem)))
        return w, True, f"{w} parallel GPU processes in 4-bit NF4 (~{w * est_mem:.1f} GB / {total_vram_gb:.1f} GB VRAM)"

    safe_w = max(1, min(requested_workers, int(usable_vram_gb // est_bf16)))
    return (
        safe_w,
        False,
        f"{safe_w} parallel GPU process(es) in Native BF16 + 5-thread RAM cache + batch_size={model_cfg.get('recommended_batch_size', 4)} (~{safe_w * est_bf16:.1f} GB / {total_vram_gb:.1f} GB VRAM)",
    )


def count_lines_fast(filepath: str) -> int:
    if not os.path.isfile(filepath):
        return 0
    try:
        with open(filepath, "rb") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def run_single_model_coordinated(
    model_cfg: Dict[str, Any],
    total_target_questions: int,
    args: argparse.Namespace,
    hf_token: Optional[str],
) -> Dict[str, Any]:
    """
    Coordinate pre-downloading, multi-process shard execution with live tqdm monitoring,
    and artifact aggregation for one model in OFFICIAL_MODEL_SUITE.
    """
    model_key = model_cfg["key"]
    repo_id = model_cfg["repo_id"]
    display_name = model_cfg["display_name"]
    method = model_cfg["method"]
    arch = model_cfg["arch"]
    batch_size = args.batch_size if args.batch_size > 0 else model_cfg.get("recommended_batch_size", 4)

    model_out_dir = os.path.join(args.output_dir, "models", model_key)
    preds_dir = os.path.join(args.output_dir, "predictions")
    logs_dir = os.path.join(args.output_dir, "logs")
    summaries_dir = os.path.join(args.output_dir, "summaries")
    for d in (model_out_dir, preds_dir, logs_dir, summaries_dir):
        os.makedirs(d, exist_ok=True)

    master_pred_jsonl = os.path.join(preds_dir, f"{model_key}_predictions.jsonl")
    master_pred_csv = os.path.join(preds_dir, f"{model_key}_predictions.csv")
    master_log_file = os.path.join(logs_dir, f"{model_key}.log")
    master_summary_json = os.path.join(summaries_dir, f"{model_key}_summary.json")

    # Check if already 100% completed
    if not args.overwrite and os.path.isfile(master_summary_json) and count_lines_fast(master_pred_jsonl) >= total_target_questions:
        with open(master_summary_json, "r", encoding="utf-8") as sf:
            cached_sum = json.load(sf)
        print(f"\n[RESUME-SKIP] {display_name} already completed ({total_target_questions} questions). Loaded summary from {master_summary_json}.")
        return cached_sum

    # 1. Pre-download weights once via HF API Key (unless in --mock mode)
    if not args.mock:
        preload_model_weights_from_hf(repo_id=repo_id, hf_token=hf_token, max_dl_workers=8)

    # 2. Plan process count & precision for L4 24GB VRAM
    eff_workers, use_4bit, plan_desc = plan_workers_for_model(
        model_cfg,
        requested_workers=args.num_workers,
        worker_mode=args.worker_mode,
        force_4bit=args.load_in_4bit,
    )
    if args.mock:
        eff_workers = args.num_workers

    print("\n" + "=" * 88)
    print(f" MODEL [{model_cfg['id']}/7]: {display_name}")
    print(f" Hugging Face Repo : {repo_id} | Method: {method.upper()}")
    print(f" Parallel Strategy : {plan_desc}")
    print(f" Target Questions  : {total_target_questions}")
    print("=" * 88)

    if args.overwrite:
        for pat in ("predictions_shard_*.jsonl", "meta_shard_*.json", "worker_*.log"):
            for f in glob.glob(os.path.join(model_out_dir, pat)):
                try:
                    os.remove(f)
                except Exception:
                    pass

    t_start = time.perf_counter()

    if eff_workers == 1:
        worker_args = argparse.Namespace(
            dataset=args.dataset,
            data_dir=args.data_dir,
            ram_stage_dir=args.ram_stage_dir,
            model_out_dir=model_out_dir,
            repo_id=repo_id,
            display_name=display_name,
            method=method,
            arch=arch,
            device=args.device,
            load_in_4bit=use_4bit,
            hf_token=hf_token,
            batch_size=batch_size,
            num_workers=1,
            worker_id=0,
            limit=args.limit,
            overwrite=args.overwrite,
            mock=args.mock,
        )
        run_worker_shard(worker_args)
    else:
        script_path = os.path.abspath(__file__)
        procs: List[subprocess.Popen] = []
        for wid in range(eff_workers):
            cmd = [
                sys.executable,
                script_path,
                "--internal-worker",
                "--worker-id", str(wid),
                "--num-workers", str(eff_workers),
                "--model-out-dir", model_out_dir,
                "--repo-id", repo_id,
                "--display-name", display_name,
                "--method", method,
                "--arch", arch,
                "--device", args.device,
                "--batch-size", str(batch_size),
                "--dataset", args.dataset,
                "--data-dir", args.data_dir,
            ]
            if args.ram_stage_dir:
                cmd += ["--ram-stage-dir", args.ram_stage_dir]
            if use_4bit:
                cmd.append("--load-in-4bit")
            if hf_token:
                cmd += ["--hf-token", hf_token]
            if args.limit and args.limit > 0:
                cmd += ["--limit", str(args.limit)]
            if args.overwrite:
                cmd.append("--overwrite")
            if args.mock:
                cmd.append("--mock")

            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            procs.append(subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))

        # Live unified tqdm progress bar across all parallel worker processes
        shard_files = [os.path.join(model_out_dir, f"predictions_shard_{w}.jsonl") for w in range(eff_workers)]
        with tqdm(total=total_target_questions, desc=f"[{eff_workers}-PROC] {display_name}", dynamic_ncols=True) as pbar:
            last_count = 0
            while any(p.poll() is None for p in procs):
                cur_count = sum(count_lines_fast(sf) for sf in shard_files)
                if cur_count > last_count:
                    pbar.update(cur_count - last_count)
                    last_count = cur_count
                time.sleep(0.4)

            cur_count = sum(count_lines_fast(sf) for sf in shard_files)
            if cur_count > last_count:
                pbar.update(cur_count - last_count)

        failed = [(w, p.returncode) for w, p in enumerate(procs) if p.returncode != 0]
        if failed:
            print(f"[ERROR] Worker subprocess(es) failed for {display_name}: {failed}")
            for w, _ in failed:
                wlog = os.path.join(model_out_dir, f"worker_{w}.log")
                if os.path.isfile(wlog):
                    with open(wlog, "r", encoding="utf-8", errors="replace") as wf:
                        print(f"--- Last 25 lines of worker_{w}.log ---\n" + "".join(wf.readlines()[-25:]))
            raise RuntimeError(f"Workers {failed} exited with error on {display_name}")

    wall_time_s = time.perf_counter() - t_start

    # 3. Merge shards, compute Stage 1 + Stage 2 + Raw metrics, and write logs/CSVs
    all_records: List[Dict[str, Any]] = []
    seen_keys: Set[str] = set()
    peak_vram_alloc = 0.0
    peak_vram_res = 0.0

    for wid in range(eff_workers):
        sf = os.path.join(model_out_dir, f"predictions_shard_{wid}.jsonl")
        if os.path.isfile(sf):
            with open(sf, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        if rec["sample_key"] not in seen_keys:
                            seen_keys.add(rec["sample_key"])
                            all_records.append(rec)
        mf = os.path.join(model_out_dir, f"meta_shard_{wid}.json")
        if os.path.isfile(mf):
            try:
                mdata = json.load(open(mf, "r", encoding="utf-8"))
                peak_vram_alloc += float(mdata.get("peak_vram_allocated_gb", 0.0))
                peak_vram_res += float(mdata.get("peak_vram_reserved_gb", 0.0))
            except Exception:
                pass

    all_records.sort(key=lambda r: (r.get("global_index", 0), r.get("question_id", 0)))

    aggregator = MetricsAggregator()
    raw_em_sum = 0.0
    raw_f1_sum = 0.0
    for r in all_records:
        aggregator.update(
            stage1_metrics=r.get("stage1_metrics"),
            stage2_metrics=r.get("stage2_metrics"),
            latency_s=r.get("latency_s"),
        )
        raw_em_sum += float(r.get("raw_em", 0.0))
        raw_f1_sum += float(r.get("raw_f1", 0.0))

    n_eval = max(1, len(all_records))
    agg_summary = aggregator.summary()
    agg_summary.update({
        "model_key": model_key,
        "display_name": display_name,
        "repo_id": repo_id,
        "method": method,
        "num_workers": eff_workers,
        "load_in_4bit": use_4bit,
        "batch_size": batch_size,
        "wall_time_s": round(wall_time_s, 2),
        "effective_throughput_q_per_min": round((len(all_records) / max(0.001, wall_time_s)) * 60.0, 2),
        "total_peak_vram_allocated_gb": round(peak_vram_alloc, 2),
        "total_peak_vram_reserved_gb": round(peak_vram_res, 2),
        "raw_full_em_pct": round((raw_em_sum / n_eval) * 100.0, 2),
        "raw_full_f1_pct": round(raw_f1_sum / n_eval, 2),
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })

    # Write master predictions JSONL
    with open(master_pred_jsonl, "w", encoding="utf-8") as jf:
        for r in all_records:
            jf.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Write master predictions CSV
    with open(master_pred_csv, "w", newline="", encoding="utf-8-sig") as cf:
        writer = csv.writer(cf)
        writer.writerow([
            "global_index", "question_id", "image_name", "question",
            "pred_answer", "gt_primary", "verified_answers",
            "f1_verified", "substring_verified", "anls_verified",
            "em_verified", "vqa_acc_verified", "f1_gt", "em_gt",
            "stage1_recall", "stage1_iou", "latency_s", "worker_id",
        ])
        for r in all_records:
            s1 = r.get("stage1_metrics", {})
            s2 = r.get("stage2_metrics", {})
            gts = r.get("gt_answers", [])
            ver = r.get("verified_all_answers", [])
            writer.writerow([
                r.get("global_index", 0),
                r.get("question_id", ""),
                r.get("image_name", ""),
                r.get("question", ""),
                r.get("pred_answer", ""),
                gts[0] if gts else "",
                "; ".join(ver) if ver else "",
                round(s2.get("f1_verified", 0.0) * 100.0, 2),
                round(s2.get("substring_verified", 0.0) * 100.0, 2),
                round(s2.get("anls_verified", 0.0) * 100.0, 2),
                round(s2.get("em_verified", 0.0) * 100.0, 2),
                round(s2.get("vqa_acc_verified", 0.0) * 100.0, 2),
                round(s2.get("f1_gt", 0.0) * 100.0, 2),
                round(s2.get("em_gt", 0.0) * 100.0, 2),
                round(s1.get("recall", 0.0) * 100.0, 2),
                round(s1.get("iou", 0.0) * 100.0, 2),
                r.get("latency_s", 0.0),
                r.get("worker_id", 0),
            ])

    # Consolidate worker logs into master_log_file
    with open(master_log_file, "w", encoding="utf-8") as lf:
        lf.write(f"=== BENCHMARK LOG: {display_name} ({repo_id}, method={method}) ===\n")
        lf.write(f"Wall Time: {wall_time_s:.2f}s | Workers: {eff_workers} | Questions: {len(all_records)}\n\n")
        for wid in range(eff_workers):
            wlog = os.path.join(model_out_dir, f"worker_{wid}.log")
            if os.path.isfile(wlog):
                lf.write(f"--- WORKER {wid} LOG ---\n")
                with open(wlog, "r", encoding="utf-8", errors="replace") as wf:
                    lf.write(wf.read())
                lf.write("\n")

    # Save model summary JSON
    with open(master_summary_json, "w", encoding="utf-8") as sf:
        json.dump(agg_summary, sf, ensure_ascii=False, indent=2)

    s2_sum = agg_summary["stage2"]
    print(
        f"[DONE] {display_name} ({wall_time_s:.1f}s) -> "
        f"Token F1(Ver): {s2_sum['f1_verified_pct']:.2f}% | "
        f"Substring(Ver): {s2_sum['substring_match_verified_pct']:.2f}% | "
        f"ANLS(Ver): {s2_sum['anls_verified_pct']:.2f}% | "
        f"EM(Ver): {s2_sum['exact_match_verified_pct']:.2f}%"
    )
    return agg_summary


# ==============================================================================
# 7. MASTER REPORT, CSV EXPORTER & COLAB ZIP BUNDLER
# ==============================================================================

def export_master_tables_and_zip(output_dir: str, model_summaries: List[Dict[str, Any]]) -> str:
    """
    Export master CSV tables, Markdown report, and bundle all logs + predictions + results into a ZIP file.
    """
    os.makedirs(output_dir, exist_ok=True)
    stage2_csv = os.path.join(output_dir, "stage2_400_verified_summary.csv")
    compat_csv = os.path.join(output_dir, "vitextvqa_benchmark_summary.csv")
    master_json = os.path.join(output_dir, "all_models_benchmark_summary.json")
    report_md = os.path.join(output_dir, "BENCHMARK_SUMMARY_REPORT.md")

    with open(master_json, "w", encoding="utf-8") as jf:
        json.dump(model_summaries, jf, ensure_ascii=False, indent=2)

    # 1. Stage 2 Full Benchmark Table CSV
    with open(stage2_csv, "w", newline="", encoding="utf-8-sig") as cf:
        writer = csv.writer(cf)
        writer.writerow([
            "model_key", "display_name", "repo_id", "method", "num_workers", "load_in_4bit",
            "num_samples", "f1_verified_pct", "substring_match_verified_pct", "anls_verified_pct",
            "exact_match_verified_pct", "vqa_acc_verified_pct",
            "f1_gt_pct", "substring_match_gt_pct", "anls_gt_pct", "exact_match_gt_pct",
            "stage1_recall_pct", "stage1_iou_pct",
            "peak_vram_gb", "avg_latency_s", "wall_time_s", "effective_throughput_q_per_min", "timestamp",
        ])
        for m in model_summaries:
            s1 = m.get("stage1", {})
            s2 = m.get("stage2", {})
            perf = m.get("performance", {})
            writer.writerow([
                m.get("model_key", ""),
                m.get("display_name", ""),
                m.get("repo_id", ""),
                m.get("method", ""),
                m.get("num_workers", 1),
                m.get("load_in_4bit", False),
                m.get("total_evaluated_samples", 0),
                s2.get("f1_verified_pct", 0.0),
                s2.get("substring_match_verified_pct", 0.0),
                s2.get("anls_verified_pct", 0.0),
                s2.get("exact_match_verified_pct", 0.0),
                s2.get("vqa_acc_verified_pct", 0.0),
                s2.get("f1_gt_pct", 0.0),
                s2.get("substring_match_gt_pct", 0.0),
                s2.get("anls_gt_pct", 0.0),
                s2.get("exact_match_gt_pct", 0.0),
                s1.get("mean_recall_pct", 0.0),
                s1.get("mean_iou_pct", 0.0),
                m.get("total_peak_vram_allocated_gb", 0.0),
                perf.get("avg_latency_s", 0.0),
                m.get("wall_time_s", 0.0),
                m.get("effective_throughput_q_per_min", 0.0),
                m.get("timestamp", ""),
            ])

    # 2. Backward-compatible vitextvqa_benchmark_summary.csv
    with open(compat_csv, "w", newline="", encoding="utf-8-sig") as cf:
        writer = csv.writer(cf)
        writer.writerow([
            "model_name", "method", "backend", "num_samples",
            "f1_verified", "substring_verified", "anls_verified", "em_verified", "vqa_acc_verified",
            "full_f1", "full_em", "peak_vram_gb", "elapsed_seconds", "timestamp",
        ])
        for m in model_summaries:
            s2 = m.get("stage2", {})
            writer.writerow([
                m.get("display_name", m.get("repo_id", "")),
                m.get("method", "baseline"),
                "4bit_nf4" if m.get("load_in_4bit") else "native_bf16",
                m.get("total_evaluated_samples", 0),
                s2.get("f1_verified_pct", 0.0),
                s2.get("substring_match_verified_pct", 0.0),
                s2.get("anls_verified_pct", 0.0),
                s2.get("exact_match_verified_pct", 0.0),
                s2.get("vqa_acc_verified_pct", 0.0),
                s2.get("f1_gt_pct", 0.0),
                s2.get("exact_match_gt_pct", 0.0),
                m.get("total_peak_vram_allocated_gb", 0.0),
                m.get("wall_time_s", 0.0),
                m.get("timestamp", ""),
            ])

    # 3. Markdown Summary Table
    lines = [
        "# ViTextVQA 400-Verified Multi-Model Benchmark Report (Google Colab L4)",
        "",
        f"- **Generated At**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Total Models Evaluated**: {len(model_summaries)}",
        "",
        "| Model | Method | Workers | Token F1 (Ver) | Substring (Ver) | ANLS (Ver) | Exact Match (Ver) | VQA Acc (Ver) | Stage 1 Recall | Peak VRAM (GB) | Wall Time (s) |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for m in model_summaries:
        s1 = m.get("stage1", {})
        s2 = m.get("stage2", {})
        lines.append(
            f"| `{m.get('display_name')}` | `{m.get('method')}` | {m.get('num_workers')} | "
            f"**{s2.get('f1_verified_pct', 0.0):.2f}%** | **{s2.get('substring_match_verified_pct', 0.0):.2f}%** | "
            f"**{s2.get('anls_verified_pct', 0.0):.2f}%** | **{s2.get('exact_match_verified_pct', 0.0):.2f}%** | "
            f"{s2.get('vqa_acc_verified_pct', 0.0):.2f}% | {s1.get('mean_recall_pct', 0.0):.2f}% | "
            f"{m.get('total_peak_vram_allocated_gb', 0.0):.2f} | {m.get('wall_time_s', 0.0):.1f}s |"
        )
    with open(report_md, "w", encoding="utf-8") as rf:
        rf.write("\n".join(lines) + "\n")

    # 4. Create ZIP archive of all logs, predictions, and summary tables for 1-click Colab download
    zip_path = os.path.abspath(os.path.join(output_dir, "..", "colab_l4_benchmark_bundle.zip"))
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(output_dir):
            for file in files:
                # Skip temporary per-worker shard files since consolidated files exist in predictions/ & logs/
                if "_shard_" in file or file.startswith("worker_"):
                    continue
                abs_p = os.path.join(root, file)
                rel_p = os.path.relpath(abs_p, os.path.dirname(output_dir))
                zf.write(abs_p, rel_p)

    print("\n" + "=" * 92)
    print(" ALL BENCHMARK ARTIFACTS EXPORTED SUCCESSFULLY!")
    print("=" * 92)
    print(f" - Stage 2 Summary CSV : {stage2_csv}")
    print(f" - Compatible Summary  : {compat_csv}")
    print(f" - Markdown Report     : {report_md}")
    print(f" - Predictions Folder  : {os.path.join(output_dir, 'predictions')}")
    print(f" - Logs Folder         : {os.path.join(output_dir, 'logs')}")
    print(f" - Ready-to-Download ZIP: {zip_path} ({os.path.getsize(zip_path) / 1024:.1f} KB)")
    print("=" * 92)
    return zip_path


# ==============================================================================
# 8. CLI ENTRYPOINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="High-Speed Google Colab L4 Multi-Model Benchmark Runner (Official Models + SG-VCD)"
    )
    parser.add_argument(
        "--hf-token", "--hf_token", "--token",
        dest="hf_token",
        type=str,
        default=None,
        help="Hugging Face API Key (or set HF_TOKEN env var).",
    )
    parser.add_argument(
        "--models", "--model-name",
        dest="models",
        type=str,
        default="all",
        help="Models to run: 'all' (default) or comma-separated IDs/names (e.g. '1,2,4' or 'Qwen/Qwen2.5-VL-7B-Instruct').",
    )
    parser.add_argument(
        "--num-workers", "--num_workers",
        dest="num_workers",
        type=int,
        default=5,
        help="Number of parallel worker processes / threads (default: 5).",
    )
    parser.add_argument(
        "--worker-mode",
        dest="worker_mode",
        type=str,
        default="auto",
        choices=["auto", "force_5proc"],
        help="'auto' scales GPU processes safely to preserve 100%% Native BF16 on L4 24GB; 'force_5proc' forces 5 OS processes (auto-enabling 4-bit NF4 for >=3B models).",
    )
    parser.add_argument(
        "--batch-size", "--batch_size",
        dest="batch_size",
        type=int,
        default=0,
        help="Batch size per image group (0 = auto optimal per model).",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Force 4-bit NF4 quantization.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="results/swintextspotter_auto_annot/answer_bboxes_400_verified.json",
        help="Path to the 400-image verified dataset JSON (1,222 questions).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/vitextvqa",
        help="Fallback ViTextVQA image directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/colab_l4_benchmark",
        help="Directory for logs, predictions, summaries, and tables.",
    )
    parser.add_argument(
        "--ram-stage-dir",
        type=str,
        default="/dev/shm/vitextvqa_400_ram" if os.path.isdir("/dev/shm") else None,
        help="RAM disk directory to stage all 400 images (defaults to /dev/shm/vitextvqa_400_ram on Linux/Colab).",
    )
    parser.add_argument(
        "--cleanup-cache",
        action="store_true",
        help="Delete Hugging Face model weights cache after each model finishes to save Colab disk space.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of questions per model (e.g. 15 for pilot smoke test).",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a quick 10-question smoke test.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run with synthetic mock inference (zero GPU/VRAM required for local pipeline verification).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore existing checkpoints and re-evaluate from scratch.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Target device (cuda or cpu).")

    # Internal flags used when spawning worker subprocesses
    parser.add_argument("--internal-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-id", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--model-out-dir", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--repo-id", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--display-name", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--method", type=str, default="baseline", help=argparse.SUPPRESS)
    parser.add_argument("--arch", type=str, default="transformers_vlm", help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Subprocess Worker Entry
    if args.internal_worker:
        run_worker_shard(args)
        return

    if args.test and args.limit <= 0:
        args.limit = 10

    hf_token = authenticate_hf(args.hf_token)

    # Stage all 400 images into RAM disk (/dev/shm on Colab) & verify dataset once
    samples = load_verified_400_dataset(
        args.dataset,
        data_dir=args.data_dir,
        ram_stage_dir=args.ram_stage_dir,
        preload_threads=args.num_workers,
        quiet=False,
    )
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]
    total_questions = len(samples)

    selected_models = resolve_selected_models(args.models)
    print("\n" + "#" * 92)
    print(" GOOGLE COLAB L4 MULTI-MODEL BENCHMARK SUITE (OFFICIAL MODELS + SG-VCD)")
    print(f" Total Models to Run : {len(selected_models)}")
    print(f" Questions per Model : {total_questions} (from {len({s['image_name'] for s in samples})} unique images cached in RAM)")
    print(f" Parallel Workers    : {args.num_workers} (Mode: {args.worker_mode})")
    print(f" Output Directory    : {args.output_dir}")
    for m in selected_models:
        print(f"   [{m['id']}] {m['display_name']} -> repo='{m['repo_id']}', method='{m['method']}'")
    print("#" * 92)

    all_summaries: List[Dict[str, Any]] = []
    suite_t0 = time.perf_counter()

    for idx, m_cfg in enumerate(selected_models, start=1):
        summary = run_single_model_coordinated(
            model_cfg=m_cfg,
            total_target_questions=total_questions,
            args=args,
            hf_token=hf_token,
        )
        all_summaries.append(summary)

        # Update master CSVs incrementally after every model
        export_master_tables_and_zip(args.output_dir, all_summaries)

        # Clean up HF cache if requested (skip if next model uses the exact same repo_id, e.g., Vintern-1B baseline -> SG-VCD)
        if args.cleanup_cache and not args.mock:
            next_repo = selected_models[idx]["repo_id"] if idx < len(selected_models) else None
            if next_repo != m_cfg["repo_id"]:
                evict_model_hf_cache(m_cfg["repo_id"])

    suite_dt = time.perf_counter() - suite_t0
    print(f"\n[SUITE COMPLETED] Evaluated {len(all_summaries)} model configurations in {suite_dt:.1f}s ({suite_dt/60:.2f} min).")


if __name__ == "__main__":
    main()
