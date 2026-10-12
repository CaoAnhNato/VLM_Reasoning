#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVALUATION PIPELINE: ZERO-SHOT VLM BENCHMARKING ON ViTextVQA
Hỗ trợ 7 mô hình VLM: Qwen2.5-VL (3B/7B), Vintern (1B/3B), EraX-VL (2B/7B), Aya Vision 8B.
Tối ưu VRAM 16GB, Unsloth Fast Inference, và giải phóng Disk Cache tuần tự.
"""

import argparse
import atexit
import csv
import gc
import glob
import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import queue
import re
import shutil
import string
import sys
import threading

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from PIL import Image
try:
    import torch
    # Tự động vá lỗi thiếu all_tied_weights_keys trên transformers >= 5.0 đối với các mô hình custom (InternVL, Vintern)
    if hasattr(torch.nn.Module, "__getattr__") and not getattr(torch.nn.Module, "_all_tied_weights_patched", False):
        _orig_nn_getattr = torch.nn.Module.__getattr__
        def _safe_nn_getattr(self, name):
            if name == "all_tied_weights_keys":
                return self.__dict__.get("all_tied_weights_keys", getattr(self, "_tied_weights_keys", {}))
            return _orig_nn_getattr(self, name)
        torch.nn.Module.__getattr__ = _safe_nn_getattr
        torch.nn.Module._all_tied_weights_patched = True
except ImportError:
    torch = None

# Tự động vá lỗi KeyError: 'architectures' khi transformers gọi to_diff_dict() / __repr__() trên InternVLChatConfig (OpenGVLab)
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
from tqdm import tqdm

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import psutil
except ImportError:
    psutil = None


# ==============================================================================
# HÀM CHUẨN HÓA VĂN BẢN VÀ TÍNH CHỈ SỐ HỌC THUẬT (METRICS)
# ==============================================================================

def normalize_answer(text: str) -> str:
    """Chuẩn hóa văn bản theo quy ước của bài báo ViTextVQA."""
    if not text:
        return ""
    text = str(text).lower()
    # Loại bỏ dấu câu tiếng Việt và tiếng Anh
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    # Loại bỏ khoảng trắng thừa
    text = " ".join(text.split())
    return text

def compute_exact_match(prediction: str, ground_truths: List[str]) -> float:
    """Tính điểm Exact Match (1.0 nếu trùng khớp chính xác 1 trong các đáp án)."""
    norm_pred = normalize_answer(prediction)
    if not norm_pred:
        return 0.0
    for gt in ground_truths:
        if norm_pred == normalize_answer(gt):
            return 1.0
    return 0.0

def compute_f1_score(prediction: str, ground_truths: List[str]) -> float:
    """Tính điểm Token-level F1-score so với tập đáp án đúng (Ground Truths)."""
    norm_pred = normalize_answer(prediction)
    pred_tokens = norm_pred.split()
    if not pred_tokens:
        return 0.0

    best_f1 = 0.0
    for gt in ground_truths:
        gt_tokens = normalize_answer(gt).split()
        if not gt_tokens:
            continue
        common = set(pred_tokens) & set(gt_tokens)
        num_same = sum(min(pred_tokens.count(w), gt_tokens.count(w)) for w in common)
        if num_same == 0:
            continue
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gt_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        if f1 > best_f1:
            best_f1 = f1
    return best_f1

# ==============================================================================
# BỘ NẠP VÀ TRÍCH XUẤT DỮ LIỆU ViTextVQA
# ==============================================================================

def locate_dataset_files(data_dir: str) -> Tuple[str, str]:
    """Tìm kiếm file test annotations và thư mục ảnh trong thư mục dataset."""
    candidates_anno = [
        # Ưu tiên các file có ground truth (chứa đáp án chuẩn)
        os.path.join(data_dir, "ViTextVQA_test_gt.json"),
        os.path.join(data_dir, "vitextvqa_test_gt.json"),
        os.path.join(data_dir, "test_gt.json"),
        os.path.join(data_dir, "test_annotations.json"),
        os.path.join(data_dir, "ViTextVQA_test.json"),
        os.path.join(data_dir, "vitextvqa_test.json"),
        os.path.join(data_dir, "test.json"),
        os.path.join(data_dir, "annotations", "test.json"),
    ]
    anno_path = None
    for cand in candidates_anno:
        if os.path.exists(cand):
            anno_path = cand
            break

    if not anno_path:
        # Tìm các file có chữ test_gt trước, sau đó mới đến test
        matches_gt = glob.glob(os.path.join(data_dir, "**", "*test*gt*.json"), recursive=True)
        if matches_gt:
            anno_path = matches_gt[0]
        else:
            matches = glob.glob(os.path.join(data_dir, "**", "*test*.json"), recursive=True)
            if matches:
                anno_path = matches[0]

    # Tìm thư mục ảnh (ưu tiên st_images được giải nén từ ViTextVQA_images.zip)
    candidates_img = [
        os.path.join(data_dir, "st_images"),
        os.path.join(data_dir, "ViTextVQA_images"),
        os.path.join(data_dir, "vitextvqa_images"),
        os.path.join(data_dir, "images"),
        os.path.join(data_dir, "test_images"),
        os.path.join(data_dir, "train_images"),
        data_dir,
    ]
    img_dir = data_dir
    found_img_dir = False
    for cand in candidates_img:
        if os.path.isdir(cand) and any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in os.listdir(cand)):
            img_dir = cand
            found_img_dir = True
            break

    if not found_img_dir:
        # Quét đệ quy tìm bất kỳ thư mục con nào chứa ảnh
        for root, dirs, files in os.walk(data_dir):
            if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files):
                img_dir = root
                break

    return anno_path, img_dir

def load_vitextvqa_test_data(anno_path: str, img_dir: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """Đọc dữ liệu kiểm thử từ file JSON và chuẩn hóa cấu trúc mẫu."""
    if not anno_path or not os.path.exists(anno_path):
        raise FileNotFoundError(f"Không tìm thấy file annotations tại: {anno_path}")

    with open(anno_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # raw_data có thể là dict hoặc list
    items = []
    image_map = {}

    if isinstance(raw_data, dict):
        # Trích xuất bảng ánh xạ id -> tên file ảnh nếu có danh sách "images" (chuẩn COCO)
        if "images" in raw_data and isinstance(raw_data["images"], list):
            for img_info in raw_data["images"]:
                if isinstance(img_info, dict):
                    iid = img_info.get("id") or img_info.get("image_id")
                    fname = img_info.get("file_name") or img_info.get("filename") or img_info.get("image_path")
                    if iid is not None and fname:
                        image_map[str(iid)] = fname
                        image_map[iid] = fname

        if "annotations" in raw_data:
            items = raw_data["annotations"]
        elif "data" in raw_data:
            items = raw_data["data"]
        else:
            items = list(raw_data.values())
    elif isinstance(raw_data, list):
        items = raw_data

    samples = []
    data_dir = os.path.dirname(img_dir)

    for item in items:
        # Trích xuất tên file ảnh
        img_name = item.get("image_path") or item.get("image") or item.get("filename") or item.get("file_name")
        if not img_name and item.get("image_id") is not None:
            img_name = image_map.get(str(item.get("image_id"))) or image_map.get(item.get("image_id")) or item.get("image_id")
        if not img_name:
            continue

        if isinstance(img_name, int) or (isinstance(img_name, str) and not img_name.endswith(('.jpg', '.png', '.jpeg'))):
            img_name = f"{img_name}.jpg"

        full_img_path = os.path.join(img_dir, os.path.basename(str(img_name)))
        if not os.path.exists(full_img_path):
            # Thử các trường hợp padding số 0
            base_no_ext = os.path.splitext(os.path.basename(str(img_name)))[0]
            if base_no_ext.isdigit():
                padded_cands = [
                    f"{int(base_no_ext):012d}.jpg",
                    f"{int(base_no_ext):06d}.jpg",
                    f"{int(base_no_ext)}.jpg",
                ]
                for p in padded_cands:
                    p_path = os.path.join(img_dir, p)
                    if os.path.exists(p_path):
                        full_img_path = p_path
                        break

        if not os.path.exists(full_img_path):
            # Thử tìm đệ quy nếu tên file không nằm ngay ngoài img_dir
            found = glob.glob(os.path.join(data_dir, "**", os.path.basename(str(img_name))), recursive=True)
            if found:
                full_img_path = found[0]

        question = item.get("question") or item.get("question_text") or ""
        answers = item.get("answers") or item.get("answer") or []
        if isinstance(answers, str):
            answers = [answers]
        elif isinstance(answers, list) and answers and isinstance(answers[0], dict):
            answers = [a.get("answer", "") for a in answers if "answer" in a]

        samples.append({
            "question_id": item.get("id") or item.get("question_id") or len(samples),
            "image_path": full_img_path,
            "image_name": os.path.basename(str(img_name)),
            "question": question,
            "ground_truths": [str(a) for a in answers if str(a).strip()]
        })

    if max_samples and max_samples > 0:
        samples = samples[:max_samples]

    print(f"[DATA] Đã nạp thành công {len(samples)} mẫu câu hỏi từ: {anno_path}")
    print(f"[DATA] Thư mục ảnh đang sử dụng: {img_dir}")
    return samples

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Chia nhỏ danh sách thành các chunks có kích thước tối đa chunk_size."""
    if chunk_size <= 0:
        chunk_size = 1
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def group_samples_by_image(samples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Gom nhóm danh sách mẫu kiểm thử theo đường dẫn ảnh duy nhất.
    Duy trì thứ tự xuất hiện đầu tiên của từng ảnh và bảo toàn _global_index cho từng câu hỏi.
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for s in samples:
        img_p = s["image_path"]
        if img_p not in grouped:
            grouped[img_p] = []
        grouped[img_p].append(s)
    return grouped

# ==============================================================================
# HÀM TIỀN XỬ LÝ ẢNH CHUYÊN BIỆT CHO VINTERN (5CD-AI / InternVL Architecture)
# ==============================================================================

def build_vintern_transform(input_size: int = 448):
    """Tạo bộ chuẩn hóa ảnh ImageNet chuyên biệt cho Vintern."""
    try:
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        return T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    except Exception:
        return None

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess_vintern(image: Image.Image, min_num: int = 1, max_num: int = 6, image_size: int = 448, use_thumbnail: bool = True):
    """Chia nhỏ ảnh theo tỉ lệ động (dynamic patch) theo chuẩn InternVL2.5 / Vintern-1B-v3.5."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_vintern_pixel_values(image: Image.Image, input_size: int = 448, max_num: int = 6):
    """Chuyển đổi PIL Image sang pixel_values Tensor 4D/5D chuẩn hóa cho Vintern."""
    transform = build_vintern_transform(input_size=input_size)
    if transform is None or torch is None:
        return None
    images = dynamic_preprocess_vintern(image, min_num=1, max_num=max_num, image_size=input_size, use_thumbnail=True)
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)


# ==============================================================================
# QUẢN LÝ MÔ HÌNH VLM VÀ INFERENCE ĐA KIẾN TRÚC
# ==============================================================================

class VLMRunner:
    def __init__(self, model_name: str, backend: str = "auto", load_in_4bit: bool = False, hf_token: Optional[str] = None):
        self.model_name = model_name
        self.backend = backend
        self.load_in_4bit = load_in_4bit or ("4bit" in model_name.lower())
        self.hf_token = hf_token or os.environ.get("HF_TOKEN") or None
        self.device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.actual_backend = None
        self._load_model()

    def _load_model(self):
        print(f"\n=======================================================")
        print(f" ĐANG NẠP MÔ HÌNH: {self.model_name}")
        print(f" Yêu cầu Backend: {self.backend}")
        if self.hf_token:
            print("[INFO] Đang sử dụng Hugging Face Access Token để nạp mô hình.")
        print(f"=======================================================")

        hf_kwargs = {"token": self.hf_token} if self.hf_token else {}

        # [1] Thử nghiệm nạp qua Unsloth nếu mô hình được Unsloth hỗ trợ và yêu cầu backend phù hợp
        is_unsloth_supported = (
            "qwen2.5-vl" in self.model_name.lower() or
            "aya-vision" in self.model_name.lower() or
            "unsloth" in self.model_name.lower()
        )
        if is_unsloth_supported and (self.backend in ["auto", "unsloth"]):
            try:
                from unsloth import FastVisionModel
                load_4bit = self.load_in_4bit or ("4bit" in self.model_name.lower())
                print(f"[INFO] Đang nạp qua Unsloth FastVisionModel (load_in_4bit={load_4bit}, Fast Inference)...")
                self.model, self.processor = FastVisionModel.from_pretrained(
                    self.model_name,
                    load_in_4bit=load_4bit,
                    use_gradient_checkpointing="unsloth",
                    **hf_kwargs
                )
                FastVisionModel.for_inference(self.model)
                self.actual_backend = "unsloth"
                print("[SUCCESS] Đã nạp thành công bằng Unsloth!")
                return
            except Exception as e:
                print(f"[WARN] Unsloth nạp không thành công ({e}). Tự động chuyển tiếp sang Transformers Native Fallback...")

        # [2] Nạp mô hình chuyên biệt tiếng Việt: Vintern (InternVL architecture)
        if "vintern" in self.model_name.lower():
            import transformers.configuration_utils as config_utils
            from transformers import AutoModel, AutoTokenizer, AutoConfig

            # Khắc phục triệt để lỗi KeyError: 'architectures' trên Vintern-3B-beta / InternVLChatConfig (OpenGVLab)
            config_utils.PretrainedConfig.has_no_defaults_at_init = True
            if hasattr(config_utils.PretrainedConfig, "to_diff_dict"):
                _orig_diff_dict = config_utils.PretrainedConfig.to_diff_dict
                def _safe_diff_dict(cfg_self):
                    try:
                        return _orig_diff_dict(cfg_self)
                    except Exception:
                        return cfg_self.to_dict()
                config_utils.PretrainedConfig.to_diff_dict = _safe_diff_dict

            print("[INFO] Đang nạp Vintern qua Hugging Face AutoModel (bfloat16)...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, use_fast=False, **hf_kwargs)
            except Exception:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, **hf_kwargs)

            # Nạp trước config an toàn
            config = None
            try:
                config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True, **hf_kwargs)
                config.has_no_defaults_at_init = True
            except Exception as cfg_err:
                print(f"[WARN] AutoConfig nạp trước gặp cảnh báo: {cfg_err}. Tiếp tục nạp AutoModel...")

            extra_model_kwargs = {"config": config} if config is not None else {}

            # Khởi tạo mô hình Vintern: ưu tiên nạp trực tiếp lên CUDA theo khuyến nghị của 5CD-AI
            # Tránh hoàn toàn lỗi 'AttributeError: all_tied_weights_keys' do accelerate infer_auto_device_map
            try:
                print(f"[INFO] Nạp Vintern trực tiếp lên thiết bị {self.device} (chuẩn khuyến nghị 5CD-AI)...")
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    **extra_model_kwargs,
                    **hf_kwargs
                ).eval().to(self.device)
            except Exception as direct_err:
                print(f"[WARN] Nạp trực tiếp lên {self.device} gặp lỗi: {direct_err}. Đang thử fallback với device_map='auto'...")
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    device_map="auto",
                    **extra_model_kwargs,
                    **hf_kwargs
                ).eval()

            # [CRITICAL FIX] Khắc phục triệt để lỗi "got multiple values for keyword argument 'return_dict'"
            # trên Vintern-3B-beta / InternVL khi chạy cùng transformers >= 4.45
            import functools
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

            self.actual_backend = "vintern_native"
            print("[SUCCESS] Đã nạp thành công Vintern (đã tích hợp safe return_dict patch)!")
            return

        # [3] Nạp các dòng VLM thông thường: Qwen2.5-VL, EraX-VL, Aya Vision
        import transformers
        from transformers import AutoProcessor

        print("[INFO] Đang nạp qua Hugging Face Transformers Native (bfloat16)...")
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True, **hf_kwargs)
        except Exception as e:
            # Fallback sang AutoTokenizer nếu Processor chuyên biệt chưa tải được
            from transformers import AutoTokenizer
            try:
                self.processor = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, **hf_kwargs)
            except Exception as tok_err:
                # Kiểm tra lỗi gated repo đối với Aya Vision
                if ("aya-vision" in self.model_name.lower()) and ("unsloth" not in self.model_name.lower()):
                    print(f"[WARN] Lỗi truy cập repo gated '{self.model_name}': {tok_err}")
                    print("[INFO] Tự động chuyển sang model cộng đồng công khai 100%: 'unsloth/aya-vision-8b'...")
                    self.model_name = "unsloth/aya-vision-8b"
                    self._load_model()
                    return
                raise tok_err

        # Danh sách class loader theo thứ tự ưu tiên đa phiên bản transformers
        candidate_classes = []
        for cls_name in [
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
            "Qwen2_5_VLForConditionalGeneration",
            "AutoModelForCausalLM",
            "AutoModel",
        ]:
            if hasattr(transformers, cls_name):
                candidate_classes.append((cls_name, getattr(transformers, cls_name)))

        model_loaded = False
        last_error = None

        # Cấu hình tối ưu hóa nhân Attention (Ưu tiên SDPA / FlashAttention-2 cho Ada Lovelace & Blackwell)
        extra_model_load_kwargs = dict(hf_kwargs)
        if torch is not None and torch.cuda.is_available():
            try:
                import flash_attn
                extra_model_load_kwargs["attn_implementation"] = "flash_attention_2"
            except Exception:
                extra_model_load_kwargs["attn_implementation"] = "sdpa"

        for cls_name, cls_obj in candidate_classes:
            try:
                attn_mode = extra_model_load_kwargs.get("attn_implementation", "default")
                print(f"[INFO] Đang thử khởi tạo bằng class: {cls_name} (attn={attn_mode})...")
                self.model = cls_obj.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    device_map="auto",
                    **extra_model_load_kwargs
                ).eval()
                model_loaded = True
                print(f"[SUCCESS] Đã nạp thành công bằng class {cls_name}!")
                break
            except Exception as e:
                # Nếu không nhận diện attn_implementation, fallback sang nạp mặc định
                if "attn_implementation" in extra_model_load_kwargs:
                    try:
                        fallback_kwargs = dict(extra_model_load_kwargs)
                        fallback_kwargs.pop("attn_implementation", None)
                        self.model = cls_obj.from_pretrained(
                            self.model_name,
                            torch_dtype=torch.bfloat16,
                            trust_remote_code=True,
                            device_map="auto",
                            **fallback_kwargs
                        ).eval()
                        model_loaded = True
                        print(f"[SUCCESS] Đã nạp thành công bằng class {cls_name} (default attn fallback)!")
                        break
                    except Exception:
                        pass
                last_error = e
                # Nếu gặp lỗi quyền truy cập gated repo đối với CohereLabs/aya-vision-8b
                err_str = str(e).lower()
                if ("gated" in err_str or "401" in err_str or "403" in err_str or "access" in err_str) and ("aya-vision" in self.model_name.lower()) and ("unsloth" not in self.model_name.lower()):
                    print(f"[WARN] Không có quyền truy cập repo gated '{self.model_name}'.")
                    print("[INFO] Tự động fallback sang repo công khai 'unsloth/aya-vision-8b'...")
                    self.model_name = "unsloth/aya-vision-8b"
                    self._load_model()
                    return

        if not model_loaded:
            raise RuntimeError(f"Không thể nạp mô hình {self.model_name} qua bất kỳ class nào! Lỗi: {last_error}")

        self.actual_backend = "transformers_native"
        print(f"[SUCCESS] Đã nạp thành công mô hình bằng {self.actual_backend}!")

    def generate(self, image_path: str, question: str) -> str:
        """Sinh câu trả lời Zero-shot dựa trên ảnh và câu hỏi."""
        if not os.path.exists(image_path):
            return ""

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[ERROR] Không đọc được ảnh {image_path}: {e}")
            return ""

        # Prompt chuẩn tối giản từ ViTextVQA paper
        prompt_text = f"Trả lời thật ngắn gọn câu hỏi sau dựa vào chữ xuất hiện trong ảnh: {question}"

        # 1. Cơ chế sinh cho Unsloth
        if self.actual_backend == "unsloth":
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text}
                ]}
            ]
            input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to("cuda")

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=64,
                    use_cache=True,
                    temperature=0.01,
                    do_sample=False
                )
            generated_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)]
            response = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
            return response.strip()

        # 2. Cơ chế sinh cho Vintern (InternVL architecture)
        elif self.actual_backend == "vintern_native":
            generation_config = dict(max_new_tokens=64, do_sample=False)
            generation_config.pop("return_dict", None)
            with torch.inference_mode():
                # Tiền xử lý pixel_values chuẩn theo hướng dẫn của 5CD-AI Vintern (phù hợp max_dynamic_patch)
                max_patch = getattr(getattr(self.model, "config", None), "max_dynamic_patch", 6)
                pixel_values = None
                try:
                    pixel_values = load_vintern_pixel_values(image, max_num=max_patch)
                    if pixel_values is not None:
                        pixel_values = pixel_values.to(torch.bfloat16).to(self.device)
                except Exception as pv_err:
                    print(f"[WARN] Tiền xử lý pixel_values Vintern gặp lỗi: {pv_err}")
                    pixel_values = None

                # Đảm bảo prompt có token <image>\n theo chuẩn InternVL
                vintern_prompt = prompt_text if "<image>" in prompt_text else f"<image>\n{prompt_text}"
                image_input = pixel_values if pixel_values is not None else image

                try:
                    # Gọi chuẩn khuyến nghị từ tác giả (return_history=False, trả về chuỗi str)
                    res = self.model.chat(
                        self.tokenizer,
                        image_input,
                        vintern_prompt,
                        generation_config,
                        history=None,
                        return_history=False
                    )
                    response = res[0] if isinstance(res, (tuple, list)) else res
                    return str(response).strip()
                except Exception as chat_err:
                    print(f"[WARN] Vintern model.chat (return_history=False) gặp lỗi: {chat_err}. Thử fallback return_history=True...")
                    try:
                        # Fallback 1: Thử với return_history=True
                        res = self.model.chat(
                            self.tokenizer,
                            image_input,
                            vintern_prompt,
                            generation_config,
                            history=None,
                            return_history=True
                        )
                        response = res[0] if isinstance(res, (tuple, list)) else res
                        return str(response).strip()
                    except Exception as fb1_err:
                        print(f"[WARN] Vintern fallback 1 gặp lỗi: {fb1_err}. Thử fallback gọi 4 tham số...")
                        try:
                            # Fallback 2: Thử gọi tối giản 4 tham số
                            res = self.model.chat(self.tokenizer, image_input, vintern_prompt, generation_config)
                            response = res[0] if isinstance(res, (tuple, list)) else res
                            return str(response).strip()
                        except Exception as fb2_err:
                            print(f"[ERROR] Tất cả các phương thức sinh Vintern đều thất bại: {fb2_err}")
                            return ""

        # 3. Cơ chế sinh cho Transformers Native (Qwen2.5-VL, EraX, Aya Vision)
        else:
            model_device = getattr(self.model, "device", "cuda" if torch.cuda.is_available() else "cpu")
            if hasattr(self.processor, "apply_chat_template"):
                messages = [
                    {"role": "user", "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt_text}
                    ]}
                ]
                try:
                    # Hỗ trợ chuẩn multimodal mới của AutoProcessor (Aya-Vision, Transformers 4.49+)
                    inputs = self.processor.apply_chat_template(
                        messages,
                        padding=True,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt"
                    ).to(model_device)
                except Exception:
                    # Định dạng phân tách text & images (Qwen2-VL, EraX)
                    text_input = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                    inputs = self.processor(text=[text_input], images=[image], return_tensors="pt").to(model_device)
            else:
                inputs = self.processor(images=image, text=prompt_text, return_tensors="pt").to(model_device)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=False
                )
            if hasattr(inputs, "input_ids"):
                output_ids = output_ids[:, inputs.input_ids.shape[1]:]

            if hasattr(self.processor, "decode"):
                response = self.processor.decode(output_ids[0], skip_special_tokens=True)
            elif hasattr(self.processor, "tokenizer") and hasattr(self.processor.tokenizer, "decode"):
                response = self.processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            else:
                response = str(output_ids[0])

            return response.strip()

    def get_recommended_batch_size(self) -> int:
        """Định đoạt kích thước lô (batch size) tối ưu dựa trên số lượng tham số của mô hình và dung lượng VRAM."""
        name = self.model_name.lower()
        has_large_vram = False
        if torch is not None and torch.cuda.is_available():
            try:
                total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                if total_vram_gb >= 20.0:
                    has_large_vram = True
            except Exception:
                pass

        if "1b" in name or "2b" in name:
            return 8
        elif "3b" in name:
            return 8 if has_large_vram else 4
        elif "7b" in name or "8b" in name:
            return 4 if (has_large_vram or getattr(self, "load_in_4bit", False) or "4bit" in name) else 2
        return 4

    def generate_multi(
        self,
        image_path: str,
        questions: List[str],
        max_chunk_size: int = 4,
        preloaded_image: Optional[Image.Image] = None
    ) -> List[str]:
        """Sinh câu trả lời cho danh sách nhiều câu hỏi cùng chung một hình ảnh (tối ưu gom nhóm & cache ảnh)."""
        if not questions:
            return []

        image = preloaded_image
        if image is None:
            if not os.path.exists(image_path):
                return ["" for _ in questions]
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"[ERROR] Không đọc được ảnh {image_path}: {e}")
                return ["" for _ in questions]

        all_responses: List[str] = []
        chunks = chunk_list(questions, max(1, max_chunk_size))

        for chunk_qs in chunks:
            # 1. Cơ chế cho Vintern (InternVL architecture)
            if self.actual_backend == "vintern_native":
                chunk_res = self._generate_vintern_batch(image, chunk_qs)
                all_responses.extend(chunk_res)

            # 2. Cơ chế cho Unsloth (Qwen2.5-VL, Aya)
            elif self.actual_backend == "unsloth":
                chunk_res = self._generate_unsloth_batch(image, chunk_qs)
                all_responses.extend(chunk_res)

            # 3. Cơ chế cho Transformers Native (EraX, Qwen)
            else:
                chunk_res = self._generate_transformers_batch(image, chunk_qs)
                all_responses.extend(chunk_res)

        return all_responses

    def _generate_vintern_batch(self, image: Image.Image, questions: List[str]) -> List[str]:
        """Tối ưu sinh đồng thời cho Vintern: trích xuất pixel_values 1 lần và batch inference."""
        max_patch = getattr(getattr(self.model, "config", None), "max_dynamic_patch", 6)
        generation_config = dict(max_new_tokens=64, do_sample=False)
        generation_config.pop("return_dict", None)

        pixel_values = None
        try:
            pixel_values = load_vintern_pixel_values(image, max_num=max_patch)
            if pixel_values is not None:
                pixel_values = pixel_values.to(torch.bfloat16).to(self.device)
        except Exception as pv_err:
            print(f"[WARN] Tiền xử lý pixel_values Vintern gặp lỗi: {pv_err}")
            pixel_values = None

        prompts = [
            f"<image>\nTrả lời thật ngắn gọn câu hỏi sau dựa vào chữ xuất hiện trong ảnh: {q}"
            for q in questions
        ]

        # Thử 1: Dùng model.batch_chat nếu có sẵn
        if hasattr(self.model, "batch_chat") and pixel_values is not None:
            try:
                num_patches_list = [pixel_values.shape[0]] * len(questions)
                pixel_values_batch = torch.cat([pixel_values] * len(questions), dim=0)
                with torch.inference_mode():
                    res = self.model.batch_chat(
                        self.tokenizer,
                        pixel_values_batch,
                        prompts,
                        generation_config,
                        num_patches_list=num_patches_list
                    )
                if isinstance(res, list) and len(res) == len(questions):
                    return [str(r).strip() for r in res]
            except Exception:
                pass

        # Fallback an toàn: Dùng lại pixel_values đã tính toán trong bộ nhớ cho từng câu hỏi
        responses = []
        image_input = pixel_values if pixel_values is not None else image
        with torch.inference_mode():
            for p in prompts:
                try:
                    res = self.model.chat(
                        self.tokenizer,
                        image_input,
                        p,
                        generation_config,
                        history=None,
                        return_history=False
                    )
                    r = res[0] if isinstance(res, (tuple, list)) else res
                    responses.append(str(r).strip())
                except Exception:
                    try:
                        res = self.model.chat(self.tokenizer, image_input, p, generation_config)
                        r = res[0] if isinstance(res, (tuple, list)) else res
                        responses.append(str(r).strip())
                    except Exception:
                        responses.append("")
        return responses

    def _generate_unsloth_batch(self, image: Image.Image, questions: List[str]) -> List[str]:
        """Tối ưu sinh đồng thời cho Unsloth: cùng 1 ảnh, nhân bản tham chiếu và batched forward."""
        if len(questions) == 1:
            q = questions[0]
            p = f"Trả lời thật ngắn gọn câu hỏi sau dựa vào chữ xuất hiện trong ảnh: {q}"
            msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": p}]}]
            try:
                in_text = self.processor.apply_chat_template(msgs, add_generation_prompt=True)
                inps = self.processor(image, in_text, add_special_tokens=False, return_tensors="pt").to("cuda")
                with torch.inference_mode():
                    outs = self.model.generate(**inps, max_new_tokens=64, use_cache=True, temperature=0.01, do_sample=False)
                g_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inps.input_ids, outs)]
                r = self.processor.batch_decode(g_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
                return [str(r).strip()]
            except Exception:
                return [""]

        prompts = [
            f"Trả lời thật ngắn gọn câu hỏi sau dựa vào chữ xuất hiện trong ảnh: {q}"
            for q in questions
        ]
        images = [image] * len(questions)
        messages_batch = [
            [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": p}
            ]}]
            for img, p in zip(images, prompts)
        ]
        try:
            input_texts = [
                self.processor.apply_chat_template(m, add_generation_prompt=True)
                for m in messages_batch
            ]
            inputs = self.processor(
                images=images,
                text=input_texts,
                add_special_tokens=False,
                return_tensors="pt",
                padding=True
            ).to("cuda")

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=64,
                    use_cache=True,
                    temperature=0.01,
                    do_sample=False
                )
            generated_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)]
            responses = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            return [str(r).strip() for r in responses]
        except Exception:
            # Fallback tuần tự an toàn
            res = []
            for p_text in prompts:
                try:
                    msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": p_text}]}]
                    in_text = self.processor.apply_chat_template(msgs, add_generation_prompt=True)
                    inps = self.processor(image, in_text, add_special_tokens=False, return_tensors="pt").to("cuda")
                    with torch.inference_mode():
                        outs = self.model.generate(**inps, max_new_tokens=64, use_cache=True, temperature=0.01, do_sample=False)
                    g_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inps.input_ids, outs)]
                    r = self.processor.batch_decode(g_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
                    res.append(str(r).strip())
                except Exception:
                    res.append("")
            return res

    def _generate_transformers_batch(self, image: Image.Image, questions: List[str]) -> List[str]:
        """Tối ưu sinh đồng thời cho Transformers Native (EraX, Qwen-VL)."""
        model_device = getattr(self.model, "device", "cuda" if torch.cuda.is_available() else "cpu")
        if len(questions) == 1:
            q = questions[0]
            p = f"Trả lời thật ngắn gọn câu hỏi sau dựa vào chữ xuất hiện trong ảnh: {q}"
            msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": p}]}]
            try:
                if hasattr(self.processor, "apply_chat_template"):
                    text_input = self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
                    inputs = self.processor(text=[text_input], images=[image], return_tensors="pt").to(model_device)
                else:
                    inputs = self.processor(images=image, text=p, return_tensors="pt").to(model_device)
                with torch.inference_mode():
                    output_ids = self.model.generate(**inputs, max_new_tokens=64, do_sample=False)
                if hasattr(inputs, "input_ids"):
                    output_ids = output_ids[:, inputs.input_ids.shape[1]:]
                if hasattr(self.processor, "batch_decode"):
                    r = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
                elif hasattr(self.processor, "tokenizer") and hasattr(self.processor.tokenizer, "batch_decode"):
                    r = self.processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                else:
                    r = str(output_ids[0])
                return [str(r).strip()]
            except Exception:
                return [""]

        prompts = [
            f"Trả lời thật ngắn gọn câu hỏi sau dựa vào chữ xuất hiện trong ảnh: {q}"
            for q in questions
        ]
        images = [image] * len(questions)
        try:
            if hasattr(self.processor, "apply_chat_template"):
                messages_batch = [
                    [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": p}]}]
                    for img, p in zip(images, prompts)
                ]
                try:
                    text_inputs = [
                        self.processor.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
                        for m in messages_batch
                    ]
                    inputs = self.processor(text=text_inputs, images=images, padding=True, return_tensors="pt").to(model_device)
                except Exception:
                    inputs = self.processor(images=images, text=prompts, padding=True, return_tensors="pt").to(model_device)
            else:
                inputs = self.processor(images=images, text=prompts, padding=True, return_tensors="pt").to(model_device)

            with torch.inference_mode():
                output_ids = self.model.generate(**inputs, max_new_tokens=64, do_sample=False)
            if hasattr(inputs, "input_ids"):
                output_ids = output_ids[:, inputs.input_ids.shape[1]:]

            if hasattr(self.processor, "batch_decode"):
                responses = self.processor.batch_decode(output_ids, skip_special_tokens=True)
            elif hasattr(self.processor, "tokenizer") and hasattr(self.processor.tokenizer, "batch_decode"):
                responses = self.processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            else:
                responses = [str(o) for o in output_ids]
            return [str(r).strip() for r in responses]
        except Exception:
            # Fallback tuần tự an toàn
            res = []
            for p in prompts:
                try:
                    inps = self.processor(images=image, text=p, return_tensors="pt").to(model_device)
                    with torch.inference_mode():
                        out_ids = self.model.generate(**inps, max_new_tokens=64, do_sample=False)
                    if hasattr(inps, "input_ids"):
                        out_ids = out_ids[:, inps.input_ids.shape[1]:]
                    if hasattr(self.processor, "decode"):
                        r = self.processor.decode(out_ids[0], skip_special_tokens=True)
                    else:
                        r = str(out_ids[0])
                    res.append(str(r).strip())
                except Exception:
                    res.append("")
            return res

    def unload(self, evict_hf_cache: bool = True):
        """Giải phóng toàn bộ VRAM GPU và xóa weights khỏi Hugging Face cache."""
        print(f"\n[CLEANUP] Đang giải phóng VRAM và dọn dẹp bộ nhớ cho {self.model_name}...")
        del self.model
        del self.processor
        del self.tokenizer
        self.model = None
        self.processor = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        if evict_hf_cache:
            candidate_cache_dirs = []
            if os.environ.get("HF_HOME"):
                candidate_cache_dirs.append(os.environ["HF_HOME"])
                candidate_cache_dirs.append(os.path.join(os.environ["HF_HOME"], "hub"))
            if os.environ.get("HUGGINGFACE_HUB_CACHE"):
                candidate_cache_dirs.append(os.environ["HUGGINGFACE_HUB_CACHE"])

            candidate_cache_dirs.extend([
                "/workspace/.hf_home/hub",
                "/workspace/.hf_home",
                os.path.expanduser("~/.cache/huggingface/hub"),
                os.path.expanduser("~/.cache/huggingface"),
                "/root/.cache/huggingface/hub",
                "/root/.cache/huggingface",
            ])

            try:
                from huggingface_hub import constants
                if hasattr(constants, "HF_HUB_CACHE"):
                    candidate_cache_dirs.append(str(constants.HF_HUB_CACHE))
                if hasattr(constants, "HF_HOME"):
                    candidate_cache_dirs.append(str(constants.HF_HOME))
            except Exception:
                pass

            target_repo = "models--" + self.model_name.replace("/", "--")
            deleted_any = False
            for cdir in set(candidate_cache_dirs):
                if not cdir or not os.path.exists(cdir):
                    continue
                target_path = os.path.join(cdir, target_repo)
                if os.path.exists(target_path):
                    print(f"[DISK CLEANUP] Đang xóa cache trọng số: {target_path}")
                    shutil.rmtree(target_path, ignore_errors=True)
                    deleted_any = True

            if not deleted_any:
                for cdir in set(candidate_cache_dirs):
                    if not cdir or not os.path.exists(cdir):
                        continue
                    found = glob.glob(os.path.join(cdir, "**", target_repo), recursive=True)
                    for f in found:
                        if os.path.isdir(f):
                            print(f"[DISK CLEANUP] Đã xóa cache trọng số tìm thấy: {f}")
                            shutil.rmtree(f, ignore_errors=True)

# ==============================================================================
# HỖ TRỢ RESUMPTION CẤP DỮ LIỆU (SAMPLE-LEVEL CHECKPOINTING & RESUME)
# ==============================================================================

def load_existing_predictions(output_jsonl_path: Optional[str]) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """
    Nạp các mẫu đã dự đoán từ trước trong file JSONL để resume chính xác từng mẫu.
    Trả về:
      - records: danh sách các bản ghi kết quả đã có.
      - evaluated_keys: tập hợp các key (question_id + image_name) để kiểm tra trùng lặp.
    """
    if not output_jsonl_path or not os.path.exists(output_jsonl_path):
        return [], set()

    records = []
    evaluated_keys = set()
    try:
        with open(output_jsonl_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    records.append(item)
                    key = f"{item.get('question_id')}_{item.get('image_name')}"
                    evaluated_keys.add(key)
                except Exception as e:
                    print(f"[CẢNH BÁO] Không parse được dòng JSON #{line_no} trong {output_jsonl_path}: {e}")
    except Exception as e:
        print(f"[CẢNH BÁO] Lỗi khi đọc file kết quả cũ {output_jsonl_path}: {e}")

    return records, evaluated_keys


def update_summary_csv(summary_csv_path: str, row: Dict[str, Any]):
    """
    Ghi hoặc cập nhật dòng kết quả của mô hình vào bảng CSV tổng hợp (tránh ghi trùng lặp).
    """
    os.makedirs(os.path.dirname(summary_csv_path), exist_ok=True)
    fieldnames = [
        "model_name", "backend", "num_samples",
        "full_f1", "full_em",
        "elapsed_seconds", "timestamp"
    ]

    rows = []
    found = False
    if os.path.exists(summary_csv_path):
        try:
            with open(summary_csv_path, "r", encoding="utf-8-sig") as csv_file:
                reader = csv.DictReader(csv_file)
                for r in reader:
                    if r.get("model_name") == row["model_name"]:
                        rows.append(row)
                        found = True
                    else:
                        rows.append(r)
        except Exception as e:
            print(f"[CẢNH BÁO] Lỗi khi đọc CSV tổng hợp cũ: {e}")

    if not found:
        rows.append(row)

    with open(summary_csv_path, "w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def update_profiling_csv(profiling_csv_path: str, row: Dict[str, Any]):
    """
    Ghi hoặc cập nhật dòng kết quả profiling của mô hình vào bảng CSV profiling (tránh ghi trùng lặp).
    """
    os.makedirs(os.path.dirname(profiling_csv_path), exist_ok=True)
    fieldnames = [
        "model_name", "backend", "profile_samples",
        "load_time_s", "avg_latency_s", "throughput_samples_per_min",
        "peak_vram_allocated_gb", "peak_vram_reserved_gb", "peak_ram_gb",
        "preliminary_f1", "preliminary_em",
        "est_full_10028_hours", "est_full_10028_cost_usd", "timestamp"
    ]

    rows = []
    found = False
    if os.path.exists(profiling_csv_path):
        try:
            with open(profiling_csv_path, "r", encoding="utf-8-sig") as csv_file:
                reader = csv.DictReader(csv_file)
                for r in reader:
                    if r.get("model_name") == row["model_name"]:
                        rows.append(row)
                        found = True
                    else:
                        rows.append(r)
        except Exception as e:
            print(f"[CẢNH BÁO] Lỗi khi đọc CSV profiling cũ: {e}")

    if not found:
        rows.append(row)

    with open(profiling_csv_path, "w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)



# ==============================================================================
# DANH MỤC 7 MÔ HÌNH VLM CHUẨN BENCHMARK TRÊN ViTextVQA
# ==============================================================================

BENCHMARK_MODEL_REGISTRY = [
    {
        "id": 1,
        "name": "unsloth/Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit",
        "backend": "unsloth",
        "load_in_4bit": True,
        "description": "Qwen2.5-VL 3B 4-bit qua Unsloth"
    },
    {
        "id": 2,
        "name": "unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit",
        "backend": "unsloth",
        "load_in_4bit": True,
        "description": "Qwen2.5-VL 7B 4-bit qua Unsloth"
    },
    {
        "id": 3,
        "name": "5CD-AI/Vintern-1B-v3_5",
        "backend": "native",
        "load_in_4bit": False,
        "description": "Vintern 1B v3.5 qua Native BF16"
    },
    {
        "id": 4,
        "name": "5CD-AI/Vintern-3B-beta",
        "backend": "native",
        "load_in_4bit": False,
        "description": "Vintern 3B beta qua Native BF16"
    },
    {
        "id": 5,
        "name": "erax-ai/EraX-VL-2B-V1.5",
        "backend": "native",
        "load_in_4bit": False,
        "description": "EraX-VL 2B V1.5 qua Native BF16"
    },
    {
        "id": 6,
        "name": "erax-ai/EraX-VL-7B-V1.5",
        "backend": "native",
        "load_in_4bit": False,
        "description": "EraX-VL 7B V1.5 qua Native BF16"
    },
    {
        "id": 7,
        "name": "unsloth/aya-vision-8b-unsloth-bnb-4bit",
        "backend": "auto",
        "load_in_4bit": True,
        "description": "Aya Vision 8B 4-bit qua Unsloth"
    },
]

def resolve_models_to_run(model_arg: str, user_backend: str = "auto", user_4bit: bool = False) -> List[Dict[str, Any]]:
    """Phân giải chuỗi tham số --model_name thành danh sách các mô hình cần đánh giá."""
    arg_clean = model_arg.strip().lower()

    # 1. Chạy tất cả mô hình trong danh mục chuẩn
    if arg_clean in ["all", "list", "all-models", "*"]:
        return [dict(m) for m in BENCHMARK_MODEL_REGISTRY]

    # 2. Chạy danh sách số thứ tự (ví dụ: "1,2,3" hoặc "3-5" hoặc "3..5")
    if re.match(r"^(\d+[\s,;-]+\d+)+$", arg_clean) or arg_clean.isdigit():
        nums = []
        for part in re.split(r"[\s,;]+", arg_clean):
            if ".." in part or "-" in part:
                sep = ".." if ".." in part else "-"
                sub = part.split(sep)
                if len(sub) == 2 and sub[0].isdigit() and sub[1].isdigit():
                    nums.extend(range(int(sub[0]), int(sub[1]) + 1))
            elif part.isdigit():
                nums.append(int(part))
        selected = []
        for m in BENCHMARK_MODEL_REGISTRY:
            if m["id"] in nums:
                selected.append(dict(m))
        if selected:
            return selected

    # 3. Danh sách tên mô hình phân tách bằng dấu phẩy
    if "," in model_arg:
        names = [n.strip() for n in model_arg.split(",") if n.strip()]
        resolved = []
        for n in names:
            match = next((m for m in BENCHMARK_MODEL_REGISTRY if n.lower() in m["name"].lower() or n.lower() == str(m["id"])), None)
            if match:
                resolved.append(dict(match))
            else:
                resolved.append({
                    "id": len(resolved) + 1,
                    "name": n,
                    "backend": user_backend,
                    "load_in_4bit": user_4bit,
                    "description": n
                })
        return resolved

    # 4. Lọc theo từ khóa (ví dụ: "vintern", "qwen", "erax")
    keyword_matches = [dict(m) for m in BENCHMARK_MODEL_REGISTRY if arg_clean in m["name"].lower()]
    if keyword_matches and len(keyword_matches) > 1 and arg_clean in ["vintern", "qwen", "erax", "unsloth"]:
        return keyword_matches

    # 5. Khớp chính xác ID hoặc tên trong registry
    match = next((m for m in BENCHMARK_MODEL_REGISTRY if arg_clean == m["name"].lower() or arg_clean == str(m["id"])), None)
    if match:
        cfg = dict(match)
        if user_backend != "auto":
            cfg["backend"] = user_backend
        if user_4bit:
            cfg["load_in_4bit"] = True
        return [cfg]

    # Mặc định: Trả về chính xác model_name do người dùng chỉ định
    return [{
        "id": 1,
        "name": model_arg.strip(),
        "backend": user_backend,
        "load_in_4bit": user_4bit,
        "description": model_arg.strip()
    }]


# ==============================================================================
# BỘ NẠP TRƯỚC BẤT ĐỒNG BỘ CHO DỮ LIỆU GOM NHÓM (ASYNC PREFETCH PIPELINE)
# ==============================================================================

class AsyncGroupedPrefetcher:
    """
    Bộ nạp trước bất đồng bộ (Producer-Consumer Double Buffering Queue) cho các lô câu hỏi gom nhóm theo ảnh.
    Một luồng nền (Worker Thread) độc lập chịu trách nhiệm:
      1. Đọc tệp ảnh từ đĩa NVMe và giải mã PIL.Image trước trong nền.
      2. Tái sử dụng đối tượng PIL.Image này cho tất cả các chunk thuộc về cùng một ảnh (không đọc lại đĩa).
      3. Đẩy tuple (img_path, image, chunk_items) vào hàng đợi có giới hạn kích thước (queue.Queue(maxsize=prefetch_depth)).
    
    Cơ chế này che giấu hoàn toàn độ trễ đọc đĩa (Disk I/O latency) và triệt tiêu hiện tượng
    trễ bong bóng (GPU Starvation / Pipeline Bubbles), duy trì GPU ở trạng thái bận rộn liên tục.
    """
    def __init__(self, grouped_samples: Dict[str, List[Dict[str, Any]]], effective_bs: int, prefetch_depth: int = 3):
        self.grouped_samples = grouped_samples
        self.effective_bs = max(1, effective_bs)
        self.prefetch_depth = max(1, prefetch_depth)
        self.queue: queue.Queue = queue.Queue(maxsize=self.prefetch_depth)
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._producer_loop, daemon=True)

    def _producer_loop(self):
        try:
            for img_path, items in self.grouped_samples.items():
                if self.stop_event.is_set():
                    break

                # Đọc ảnh một lần duy nhất trong RAM cho toàn bộ các chunk của ảnh này
                image = None
                if os.path.exists(img_path):
                    try:
                        image = Image.open(img_path).convert("RGB")
                    except Exception as e:
                        print(f"[PREFETCH WARN] Lỗi đọc ảnh {img_path}: {e}")
                        image = None

                chunks = chunk_list(items, self.effective_bs)
                for chunk_items in chunks:
                    if self.stop_event.is_set():
                        break
                    while not self.stop_event.is_set():
                        try:
                            self.queue.put((img_path, image, chunk_items), timeout=0.2)
                            break
                        except queue.Full:
                            continue
        except Exception as e:
            print(f"[PREFETCH ERROR] Ngoại lệ trong luồng nạp trước: {e}")
        finally:
            while not self.stop_event.is_set():
                try:
                    self.queue.put((None, None, None), timeout=0.2)
                    break
                except queue.Full:
                    continue

    def __enter__(self):
        self.worker_thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_event.set()
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        self.worker_thread.join(timeout=2.0)

    def __iter__(self):
        while True:
            try:
                img_path, image, chunk_items = self.queue.get(timeout=30.0)
            except queue.Empty:
                print("[PREFETCH TIMEOUT] Hàng đợi nạp trước bị quá hạn thời gian chờ!")
                break

            if img_path is None and chunk_items is None:
                break

            yield img_path, image, chunk_items


def evaluate_single_model(
    model_cfg: Dict[str, Any],
    samples: List[Dict[str, Any]],
    args: argparse.Namespace,
    hf_token: Optional[str] = None,
    is_multi_model: bool = False
) -> Dict[str, Any]:
    """Thực thi đánh giá hoặc profiling cho một mô hình VLM cụ thể."""
    model_name = model_cfg["name"]
    model_backend = model_cfg["backend"] if args.backend == "auto" else args.backend
    load_in_4bit = model_cfg.get("load_in_4bit", False) or args.load_in_4bit
    safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', model_name)

    is_profile_mode = (args.profile is not None and args.profile > 0)
    target_samples = samples[:args.profile] if is_profile_mode else samples
    total_samples_target = len(target_samples)

    # Xác định đường dẫn file JSONL đầu ra
    if args.output_jsonl and not is_multi_model:
        out_jsonl = args.output_jsonl
    else:
        if is_profile_mode:
            out_jsonl = f"results/profiling/{safe_name}_profile_{args.profile}_preds.jsonl"
        else:
            out_jsonl = f"results/predictions/{safe_name}_preds.jsonl"

    overwrite_flag = args.overwrite if not is_profile_mode else True

    # 1. Kiểm tra tiến độ đã hoàn thành trước đó (Sample-level Resumption Check)
    existing_predictions = []
    evaluated_keys = set()
    if out_jsonl and os.path.exists(out_jsonl) and not overwrite_flag:
        existing_predictions, evaluated_keys = load_existing_predictions(out_jsonl)

    remaining_samples = []
    for idx, item in enumerate(target_samples):
        key = f"{item.get('question_id')}_{item.get('image_name')}"
        if key not in evaluated_keys or overwrite_flag:
            item_with_idx = dict(item)
            item_with_idx["_global_index"] = idx
            remaining_samples.append(item_with_idx)

    num_already_done = total_samples_target - len(remaining_samples)

    # Trường hợp 1: Tất cả các mẫu đã hoàn thành 100%
    if len(remaining_samples) == 0:
        print("\n==================================================================")
        print(f" [RESUME - ĐÃ HOÀN TẤT 100%] Model: {model_name}")
        print(f" File kết quả {out_jsonl} đã có đủ {len(existing_predictions)} / {total_samples_target} mẫu.")
        print(f" Bỏ qua bước nạp trọng số và suy luận lại!")
        print("==================================================================")

        all_predictions = sorted(existing_predictions, key=lambda x: x.get("index", 0))
        total_em = sum(p["exact_match"] for p in all_predictions)
        total_f1 = sum(p["f1_score"] for p in all_predictions)
        avg_em = (total_em / len(all_predictions)) * 100 if all_predictions else 0.0
        avg_f1 = (total_f1 / len(all_predictions)) if all_predictions else 0.0

        print(f" Full Test Set F1:           {avg_f1:.2f}%")
        print(f" Full Test Set Exact Match:  {avg_em:.2f}%")
        print("==================================================================")

        if args.summary_csv and not is_profile_mode:
            update_summary_csv(args.summary_csv, {
                "model_name": model_name,
                "backend": model_backend,
                "num_samples": len(all_predictions),
                "full_f1": round(avg_f1, 2),
                "full_em": round(avg_em, 2),
                "elapsed_seconds": 0.0,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
        return {
            "model_name": model_name,
            "backend": model_backend,
            "f1": avg_f1,
            "em": avg_em,
            "throughput": 0.0,
            "latency": 0.0,
            "status": "COMPLETED_SKIPPED"
        }

    # Trường hợp 2: Cần thực thi (chạy mới hoặc tiếp tục dở dang)
    if num_already_done > 0:
        first_rem = remaining_samples[0]
        print("\n==================================================================")
        print(f" [SAMPLE RESUME DETECTED] Tìm thấy tiến độ dở dang cho: {model_name}")
        print(f" Đã hoàn thành trước đó:     {num_already_done} / {total_samples_target} mẫu")
        print(f" Số mẫu cần chạy tiếp:        {len(remaining_samples)} mẫu")
        print(f" Bắt đầu suy luận từ mẫu thứ: #{num_already_done + 1} (Dataset Index: {first_rem['_global_index']}, ID: {first_rem['question_id']})")
        print("==================================================================")
    else:
        mode_label = f"PROFILING ({total_samples_target} mẫu)" if is_profile_mode else f"ĐÁNH GIÁ MỚI ({len(target_samples)} mẫu)"
        print(f"\n[INFERENCE] Bắt đầu {mode_label} cho mô hình {model_name}...")

    # Khởi tạo Runner mô hình (Đo thời gian nạp)
    load_start_time = time.perf_counter()
    runner = VLMRunner(model_name, backend=model_backend, load_in_4bit=load_in_4bit, hf_token=hf_token)
    load_time_s = time.perf_counter() - load_start_time

    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Chuẩn bị file ghi streaming
    out_f = None
    if out_jsonl:
        os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
        file_mode = "w" if (overwrite_flag or not os.path.exists(out_jsonl)) else "a"
        out_f = open(out_jsonl, file_mode, encoding="utf-8")

    new_predictions = []
    sample_latencies = []
    start_time = time.time()

    try:
        desc_msg = f"Profiling {model_name}" if is_profile_mode else (f"Resuming {model_name}" if num_already_done > 0 else f"Evaluating {model_name}")

        if args.no_group_by_image:
            # Chế độ chạy tuần tự (Fallback)
            for item in tqdm(remaining_samples, desc=desc_msg):
                t_sample_start = time.perf_counter()
                pred = runner.generate(item["image_path"], item["question"])
                s_lat = time.perf_counter() - t_sample_start
                sample_latencies.append(s_lat)

                em = compute_exact_match(pred, item["ground_truths"])
                f1 = compute_f1_score(pred, item["ground_truths"])

                record = {
                    "index": item.get("_global_index", len(existing_predictions) + len(new_predictions)),
                    "question_id": item["question_id"],
                    "image_name": item["image_name"],
                    "question": item["question"],
                    "ground_truths": item["ground_truths"],
                    "prediction": pred,
                    "exact_match": em,
                    "f1_score": round(f1 * 100, 2),
                    "latency_s": round(s_lat, 3)
                }
                new_predictions.append(record)

                if out_f is not None:
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()
        else:
            # Chế độ tối ưu hóa gom nhóm theo ảnh & Asynchronous Prefetch Pipeline
            grouped_samples = group_samples_by_image(remaining_samples)
            effective_bs = args.batch_size if args.batch_size > 0 else runner.get_recommended_batch_size()
            print(f"\n[OPTIMIZATION] Bật cơ chế Image-Grouped Processing & Asynchronous Prefetch Pipeline!")
            print(f" - Tổng số câu hỏi cần chạy:      {len(remaining_samples)} câu")
            print(f" - Số lượng hình ảnh duy nhất:     {len(grouped_samples)} ảnh")
            print(f" - Kích thước lô tối đa (B_max):   {effective_bs} câu hỏi/lô")
            print(f" - Hàng đợi nạp trước Double Buffering: 3 ảnh (Ẩn hoàn toàn Disk I/O)")

            with AsyncGroupedPrefetcher(grouped_samples, effective_bs=effective_bs, prefetch_depth=3) as prefetcher:
                with tqdm(total=len(remaining_samples), desc=desc_msg) as pbar:
                    for img_path, image, chunk_items in prefetcher:
                        chunk_questions = [c["question"] for c in chunk_items]
                        t_chunk_start = time.perf_counter()
                        chunk_preds = runner.generate_multi(
                            img_path,
                            chunk_questions,
                            max_chunk_size=effective_bs,
                            preloaded_image=image
                        )
                        chunk_dur = time.perf_counter() - t_chunk_start

                        while len(chunk_preds) < len(chunk_items):
                            chunk_preds.append("")

                        avg_item_lat = chunk_dur / max(1, len(chunk_items))

                        for item, pred in zip(chunk_items, chunk_preds):
                            sample_latencies.append(avg_item_lat)
                            em = compute_exact_match(pred, item["ground_truths"])
                            f1 = compute_f1_score(pred, item["ground_truths"])

                            record = {
                                "index": item.get("_global_index", len(existing_predictions) + len(new_predictions)),
                                "question_id": item["question_id"],
                                "image_name": item["image_name"],
                                "question": item["question"],
                                "ground_truths": item["ground_truths"],
                                "prediction": pred,
                                "exact_match": em,
                                "f1_score": round(f1 * 100, 2),
                                "latency_s": round(avg_item_lat, 3)
                            }
                            new_predictions.append(record)

                            if out_f is not None:
                                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

                        if out_f is not None:
                            out_f.flush()
                        pbar.update(len(chunk_items))
    finally:
        if out_f is not None:
            out_f.close()

    elapsed_time = time.time() - start_time

    # Hợp nhất dữ liệu đã có và dữ liệu mới sinh
    all_predictions = (existing_predictions if not overwrite_flag else []) + new_predictions
    all_predictions.sort(key=lambda x: x.get("index", 0))

    total_em = sum(p["exact_match"] for p in all_predictions)
    total_f1 = sum(p["f1_score"] for p in all_predictions)
    avg_em = (total_em / len(all_predictions)) * 100 if all_predictions else 0.0
    avg_f1 = (total_f1 / len(all_predictions)) if all_predictions else 0.0

    avg_latency = (sum(sample_latencies) / len(sample_latencies)) if sample_latencies else 0.0
    throughput = (60.0 / avg_latency) if avg_latency > 0 else 0.0

    if is_profile_mode:
        num_prof = len(new_predictions)
        peak_vram_alloc = (torch.cuda.max_memory_allocated() / (1024 ** 3)) if (torch is not None and torch.cuda.is_available()) else 0.0
        peak_vram_res = (torch.cuda.max_memory_reserved() / (1024 ** 3)) if (torch is not None and torch.cuda.is_available()) else 0.0
        peak_ram = (psutil.Process().memory_info().rss / (1024 ** 3)) if psutil is not None else 0.0

        total_dataset_samples = 10028
        est_full_hours = ((avg_latency * total_dataset_samples) + load_time_s) / 3600.0
        est_cost_usd = est_full_hours * args.hourly_rate

        print("\n========================================================================================")
        print(f" BÁO CÁO PROFILING HIỆU NĂNG MÔ HÌNH: {model_name}")
        print(f" Backend thực tế: {runner.actual_backend}")
        print("========================================================================================")
        print(f" - Số mẫu profiling:               {num_prof} mẫu (indices 0..{num_prof - 1})")
        print(f" - Thời gian nạp mô hình:           {load_time_s:.2f} s")
        print(f" - Tốc độ suy luận trung bình:      {avg_latency:.2f} s/mẫu")
        print(f" - Thông lượng (Throughput):        {throughput:.2f} mẫu/phút")
        print(f" - VRAM tiêu thụ đỉnh (Allocated):  {peak_vram_alloc:.2f} GB")
        print(f" - VRAM tiêu thụ đỉnh (Reserved):   {peak_vram_res:.2f} GB")
        print(f" - RAM hệ thống tiêu thụ:           {peak_ram:.2f} GB")
        print(f" - Điểm sơ bộ (Preliminary):        F1: {avg_f1:.2f}% | Exact Match: {avg_em:.2f}%")
        print("----------------------------------------------------------------------------------------")
        print(" - DỰ ĐOÁN NGOẠI SUY TẬP TEST (10.028 MẪU):")
        print(f"   * Tổng thời gian ước tính:       {est_full_hours:.2f} giờ (~{est_full_hours * 60:.1f} phút)")
        print(f"   * Ước tính chi phí Vast.ai (${args.hourly_rate:.2f}/h): ${est_cost_usd:.2f} USD")
        print("========================================================================================")

        if args.profile_csv:
            prof_row = {
                "model_name": model_name,
                "backend": runner.actual_backend,
                "profile_samples": num_prof,
                "load_time_s": round(load_time_s, 2),
                "avg_latency_s": round(avg_latency, 2),
                "throughput_samples_per_min": round(throughput, 2),
                "peak_vram_allocated_gb": round(peak_vram_alloc, 2),
                "peak_vram_reserved_gb": round(peak_vram_res, 2),
                "peak_ram_gb": round(peak_ram, 2),
                "preliminary_f1": round(avg_f1, 2),
                "preliminary_em": round(avg_em, 2),
                "est_full_10028_hours": round(est_full_hours, 2),
                "est_full_10028_cost_usd": round(est_cost_usd, 2),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            update_profiling_csv(args.profile_csv, prof_row)
            print(f"[SAVED] Đã lưu kết quả profiling tại: {args.profile_csv}")
    else:
        print("\n==================================================================")
        print(f" KẾT QUẢ ZERO-SHOT BENCHMARK: {model_name}")
        print("==================================================================")
        print(f" Tổng số mẫu hoàn thành:    {len(all_predictions)} / {total_samples_target}")
        print(f" Mẫu mới hoàn thành phiên này: {len(new_predictions)}")
        print(f" Thời gian chạy phiên này:   {elapsed_time:.2f} giây")
        print(f" Full Test Set F1:           {avg_f1:.2f}%")
        print(f" Full Test Set Exact Match:  {avg_em:.2f}%")
        print("==================================================================")

        if args.summary_csv:
            update_summary_csv(args.summary_csv, {
                "model_name": model_name,
                "backend": runner.actual_backend,
                "num_samples": len(all_predictions),
                "full_f1": round(avg_f1, 2),
                "full_em": round(avg_em, 2),
                "elapsed_seconds": round(elapsed_time, 2),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
            print(f"[SAVED] Đã cập nhật bảng tổng kết tại: {args.summary_csv}")

    # Giải phóng VRAM và dọn dẹp cache đĩa của model này trước khi chuyển sang model kế tiếp
    runner.unload(evict_hf_cache=(not args.no_evict_cache))

    return {
        "model_name": model_name,
        "backend": runner.actual_backend,
        "f1": avg_f1,
        "em": avg_em,
        "throughput": throughput,
        "latency": avg_latency,
        "status": "COMPLETED_RUN"
    }


def main():
    parser = argparse.ArgumentParser(description="Zero-shot Evaluation of VLMs on ViTextVQA")
    parser.add_argument("--model_name", type=str, default="all",
                        help="Hugging Face model ID hoặc 'all' / '3,4' / 'vintern' để chạy 1 mạch toàn bộ danh sách")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "unsloth", "native"])
    parser.add_argument("--load_in_4bit", action="store_true", help="Nạp 4-bit qua Unsloth")
    parser.add_argument("--data_dir", type=str, default="data/vitextvqa", help="Thư mục chứa dataset ViTextVQA")
    parser.add_argument("--annotations", type=str, default=None, help="Đường dẫn file test annotations json")
    parser.add_argument("--image_dir", type=str, default=None, help="Đường dẫn thư mục chứa ảnh")
    parser.add_argument("--max_samples", type=int, default=None, help="Giới hạn số mẫu (ví dụ 10 để smoke test)")
    parser.add_argument("--output_jsonl", type=str, default=None, help="File lưu chi tiết kết quả dự đoán")
    parser.add_argument("--summary_csv", type=str, default="results/vitextvqa_benchmark_summary.csv", help="File tổng hợp điểm số")
    parser.add_argument("--no_evict_cache", action="store_true", help="Không xóa cache sau khi chạy xong")
    parser.add_argument("--overwrite", action="store_true", help="Ghi đè lại từ đầu, không resume từ kết quả cũ")
    parser.add_argument("--profile", type=int, default=None, help="Kích hoạt chế độ profiling tốc độ và tài nguyên (ví dụ: --profile 100)")
    parser.add_argument("--profile_csv", type=str, default="results/profiling_summary.csv", help="File tổng hợp thông số profiling")
    parser.add_argument("--hourly_rate", type=float, default=0.21, help="Chi phí thuê GPU trên Vast.ai ($/giờ) để ngoại suy chi phí")
    parser.add_argument("--hf_token", "--token", dest="hf_token", type=str, default=None,
                        help="Hugging Face Access Token (hoặc tự động nhận từ biến môi trường HF_TOKEN)")
    parser.add_argument("--batch_size", type=int, default=0,
                        help="Kích thước lô tối đa cho mỗi nhóm ảnh (0 = auto tự động xác định theo dung lượng VRAM)")
    parser.add_argument("--no_group_by_image", action="store_true",
                        help="Vô hiệu hóa tính năng gom nhóm câu hỏi theo ảnh (chạy tuần tự)")
    args = parser.parse_args()

    # 0. Thiết lập Hugging Face Token & Xác thực Hub (Kích hoạt Xet High Performance)
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or None
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        try:
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)
            print("[AUTH] Đã kích hoạt xác thực Hugging Face Hub thành công (tăng tốc tải & mở rộng rate limit)!")
        except Exception as e:
            print(f"[AUTH] Đã gán HF_TOKEN vào biến môi trường ({e})")
    else:
        print("[CẢNH BÁO] Chưa cấu hình HF_TOKEN! Các yêu cầu tải model từ Hugging Face Hub sẽ ở chế độ unauthenticated (có thể bị giới hạn tốc độ).")

    # 1. Định vị và nạp dữ liệu
    anno_path = args.annotations
    img_dir = args.image_dir
    if not anno_path or not img_dir:
        auto_anno, auto_img = locate_dataset_files(args.data_dir)
        anno_path = anno_path or auto_anno
        img_dir = img_dir or auto_img

    samples = load_vitextvqa_test_data(anno_path, img_dir, max_samples=args.max_samples)
    if not samples:
        print("[LỖI] Không tìm thấy mẫu kiểm thử nào! Vui lòng kiểm tra lại đường dẫn dataset.")
        sys.exit(1)

    # 2. Phân giải danh sách mô hình cần chạy
    models_to_run = resolve_models_to_run(args.model_name, args.backend, args.load_in_4bit)
    is_multi = len(models_to_run) > 1

    print("\n==================================================================")
    print(f" CHƯƠNG TRÌNH BENCHMARK VLM TRÊN ViTextVQA")
    print(f" Số lượng mô hình sẽ thực thi: {len(models_to_run)}")
    for idx, m in enumerate(models_to_run, 1):
        print(f"   {idx}. {m['name']} (Backend: {m['backend']})")
    print("==================================================================")

    results = []
    for idx, model_cfg in enumerate(models_to_run, 1):
        if is_multi:
            print(f"\n######################################################################")
            print(f">>> [TIẾN TRÌNH {idx}/{len(models_to_run)}] BẮT ĐẦU: {model_cfg['name']}")
            print(f"######################################################################")
        res = evaluate_single_model(model_cfg, samples, args, hf_token=hf_token, is_multi_model=is_multi)
        results.append(res)

    if is_multi:
        print("\n==================================================================")
        print(">>> TỔNG KẾT TOÀN BỘ CHUỖI BENCHMARK:")
        print("==================================================================")
        for r in results:
            print(f" - {r['model_name']}: F1 = {r.get('f1', 0.0):.2f}% | EM = {r.get('em', 0.0):.2f}% | Status: {r.get('status', 'DONE')}")
        print("==================================================================")


if __name__ == "__main__":
    main()
