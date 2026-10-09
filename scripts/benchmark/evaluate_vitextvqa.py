#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVALUATION PIPELINE: ZERO-SHOT VLM BENCHMARKING ON ViTextVQA
Hỗ trợ 7 mô hình VLM: Qwen2.5-VL (3B/7B), Vintern (1B/3B), EraX-VL (2B/7B), Aya Vision 8B.
Tối ưu VRAM 16GB, Unsloth Fast Inference, và giải phóng Disk Cache tuần tự.
"""

import argparse
import gc
import glob
import json
import os
import re
import shutil
import string
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import argparse
import csv
import gc
import glob
import json
import os
import re
import shutil
import string
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image
import torch
from tqdm import tqdm

try:
    import pandas as pd
except ImportError:
    pd = None


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
        os.path.join(data_dir, "test.json"),
        os.path.join(data_dir, "test_annotations.json"),
        os.path.join(data_dir, "vitextvqa_test.json"),
        os.path.join(data_dir, "annotations", "test.json"),
    ]
    anno_path = None
    for cand in candidates_anno:
        if os.path.exists(cand):
            anno_path = cand
            break

    if not anno_path:
        # Tìm bất kỳ file json nào có chữ test
        matches = glob.glob(os.path.join(data_dir, "**", "*test*.json"), recursive=True)
        if matches:
            anno_path = matches[0]

    # Tìm thư mục ảnh
    candidates_img = [
        data_dir,
        os.path.join(data_dir, "images"),
        os.path.join(data_dir, "test_images"),
        os.path.join(data_dir, "train_images"),
    ]
    img_dir = data_dir
    for cand in candidates_img:
        if os.path.isdir(cand) and any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in os.listdir(cand)):
            img_dir = cand
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
    if isinstance(raw_data, list):
        items = raw_data
    elif isinstance(raw_data, dict):
        if "data" in raw_data:
            items = raw_data["data"]
        elif "annotations" in raw_data:
            items = raw_data["annotations"]
        else:
            items = list(raw_data.values())

    samples = []
    for item in items:
        # Trích xuất image_path
        img_name = item.get("image_path") or item.get("image") or item.get("filename") or item.get("image_id")
        if not img_name:
            continue
        if isinstance(img_name, int) or (isinstance(img_name, str) and not img_name.endswith(('.jpg', '.png', '.jpeg'))):
            img_name = f"{img_name}.jpg"

        full_img_path = os.path.join(img_dir, os.path.basename(str(img_name)))
        if not os.path.exists(full_img_path):
            # Thử tìm đệ quy nếu tên file không nằm ngay ngoài img_dir
            found = glob.glob(os.path.join(img_dir, "**", os.path.basename(str(img_name))), recursive=True)
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
    return samples

# ==============================================================================
# QUẢN LÝ MÔ HÌNH VLM VÀ INFERENCE ĐA KIẾN TRÚC
# ==============================================================================

class VLMRunner:
    def __init__(self, model_name: str, backend: str = "auto", load_in_4bit: bool = False):
        self.model_name = model_name
        self.backend = backend
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.actual_backend = None
        self._load_model()

    def _load_model(self):
        print(f"\n=======================================================")
        print(f" ĐANG NẠP MÔ HÌNH: {self.model_name}")
        print(f" Yêu cầu Backend: {self.backend}")
        print(f"=======================================================")

        # [1] Thử nghiệm nạp qua Unsloth nếu là họ Qwen2.5-VL và có Unsloth
        if ("qwen2.5-vl" in self.model_name.lower()) and (self.backend in ["auto", "unsloth"]):
            try:
                from unsloth import FastVisionModel
                print("[INFO] Đang nạp qua Unsloth FastVisionModel (4-bit/16-bit Fast Inference)...")
                self.model, self.processor = FastVisionModel.from_pretrained(
                    self.model_name,
                    load_in_4bit=self.load_in_4bit,
                    use_gradient_checkpointing="unsloth",
                )
                FastVisionModel.for_inference(self.model)
                self.actual_backend = "unsloth"
                print("[SUCCESS] Đã nạp thành công bằng Unsloth!")
                return
            except Exception as e:
                print(f"[WARN] Unsloth nạp không thành công: {e}. Chuyển sang Transformers Native Fallback...")

        # [2] Nạp mô hình chuyên biệt tiếng Việt: Vintern (InternVL architecture)
        if "vintern" in self.model_name.lower():
            from transformers import AutoModel, AutoTokenizer
            print("[INFO] Đang nạp Vintern qua Hugging Face AutoModel (bfloat16)...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map="auto"
            ).eval()
            self.actual_backend = "vintern_native"
            print("[SUCCESS] Đã nạp thành công Vintern!")
            return

        # [3] Nạp các dòng VLM thông thường: Qwen2.5-VL, EraX-VL, Aya Vision
        from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM
        print("[INFO] Đang nạp qua Hugging Face Transformers Native (bfloat16)...")
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        except Exception:
            from transformers import AutoTokenizer
            self.processor = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

        try:
            # Ưu tiên AutoModelForVision2Seq
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto"
            ).eval()
        except Exception:
            # Fallback sang AutoModelForCausalLM (ví dụ Qwen2.5-VL)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto"
            ).eval()

        self.actual_backend = "transformers_native"
        print(f"[SUCCESS] Đã nạp thành công bằng {self.actual_backend}!")

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

        # 1. Cơ chế sinh cho Unsloth (Qwen2.5-VL)
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

        # 2. Cơ chế sinh cho Vintern
        elif self.actual_backend == "vintern_native":
            generation_config = dict(max_new_tokens=64, do_sample=False)
            with torch.inference_mode():
                # Vintern hỗ trợ hàm chat trực tiếp
                try:
                    response, _ = self.model.chat(
                        self.tokenizer,
                        image,
                        prompt_text,
                        generation_config,
                        history=None,
                        return_history=True
                    )
                    return response.strip()
                except Exception:
                    # Nếu model cần pixel_values chuẩn
                    pixel_values = self.model.extract_feature(image) if hasattr(self.model, 'extract_feature') else None
                    response = self.model.chat(self.tokenizer, pixel_values, prompt_text, generation_config)
                    return response.strip()

        # 3. Cơ chế sinh cho Transformers Native (Qwen2.5-VL, EraX, Aya)
        else:
            # Kiểm tra nếu là chat template chuẩn
            if hasattr(self.processor, "apply_chat_template"):
                messages = [
                    {"role": "user", "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt_text}
                    ]}
                ]
                text_input = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                inputs = self.processor(text=[text_input], images=[image], return_tensors="pt").to("cuda")
            else:
                inputs = self.processor(images=image, text=prompt_text, return_tensors="pt").to("cuda")

            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=False
                )
            # Cắt bớt input tokens nếu có
            if hasattr(inputs, "input_ids"):
                output_ids = output_ids[:, inputs.input_ids.shape[1]:]
            response = self.processor.decode(output_ids[0], skip_special_tokens=True)
            return response.strip()

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
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            target_repo = "models--" + self.model_name.replace("/", "--")
            target_path = os.path.join(cache_dir, target_repo)
            if os.path.exists(target_path):
                print(f"[DISK CLEANUP] Đã xóa cache trọng số: {target_path}")
                shutil.rmtree(target_path, ignore_errors=True)

# ==============================================================================
# HÀM THỰC THI CHÍNH
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Zero-shot Evaluation of VLMs on ViTextVQA")
    parser.add_argument("--model_name", type=str, required=True, help="Hugging Face model ID")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "unsloth", "native"])
    parser.add_argument("--load_in_4bit", action="store_true", help="Nạp 4-bit qua Unsloth")
    parser.add_argument("--data_dir", type=str, default="data/vitextvqa", help="Thư mục chứa dataset ViTextVQA")
    parser.add_argument("--annotations", type=str, default=None, help="Đường dẫn file test annotations json")
    parser.add_argument("--image_dir", type=str, default=None, help="Đường dẫn thư mục chứa ảnh")
    parser.add_argument("--max_samples", type=int, default=None, help="Giới hạn số mẫu (ví dụ 10 để smoke test)")
    parser.add_argument("--output_jsonl", type=str, default=None, help="File lưu chi tiết kết quả dự đoán")
    parser.add_argument("--summary_csv", type=str, default="results/vitextvqa_benchmark_summary.csv", help="File tổng hợp điểm số")
    parser.add_argument("--no_evict_cache", action="store_true", help="Không xóa cache sau khi chạy xong")
    args = parser.parse_args()

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

    # 2. Khởi tạo Runner mô hình
    runner = VLMRunner(args.model_name, backend=args.backend, load_in_4bit=args.load_in_4bit)

    # 3. Chạy suy luận và tính điểm
    print(f"\n[INFERENCE] Bắt đầu đánh giá Zero-shot trên {len(samples)} mẫu...")
    predictions = []
    total_em = 0.0
    total_f1 = 0.0
    start_time = time.time()

    for idx, item in enumerate(tqdm(samples, desc=f"Evaluating {args.model_name}")):
        pred = runner.generate(item["image_path"], item["question"])
        em = compute_exact_match(pred, item["ground_truths"])
        f1 = compute_f1_score(pred, item["ground_truths"])

        total_em += em
        total_f1 += f1

        record = {
            "index": idx,
            "question_id": item["question_id"],
            "image_name": item["image_name"],
            "question": item["question"],
            "ground_truths": item["ground_truths"],
            "prediction": pred,
            "exact_match": em,
            "f1_score": round(f1 * 100, 2)
        }
        predictions.append(record)

    elapsed_time = time.time() - start_time
    avg_em = (total_em / len(samples)) * 100
    avg_f1 = (total_f1 / len(samples)) * 100

    # Tính điểm riêng cho 100 mẫu đầu tiên (để đối chiếu với Paper gốc Table 5.6)
    subset_100_em = (sum(p["exact_match"] for p in predictions[:100]) / min(100, len(predictions))) * 100
    subset_100_f1 = (sum(p["f1_score"] for p in predictions[:100]) / min(100, len(predictions)))

    print("\n==================================================================")
    print(f" KẾT QUẢ ZERO-SHOT BENCHMARK: {args.model_name}")
    print("==================================================================")
    print(f" Tổng số mẫu đánh giá:      {len(samples)}")
    print(f" Thời gian thực thi:         {elapsed_time:.2f} giây ({elapsed_time/len(samples):.3f}s / mẫu)")
    print(f" Full Test Set F1:           {avg_f1:.2f}%")
    print(f" Full Test Set Exact Match:  {avg_em:.2f}%")
    print(f" Subset 100 samples F1:      {subset_100_f1:.2f}% (Paper GPT-4o: 55.51%)")
    print(f" Subset 100 samples EM:      {subset_100_em:.2f}% (Paper GPT-4o: 25.00%)")
    print("==================================================================")

    # 4. Lưu kết quả chi tiết
    if args.output_jsonl:
        os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)
        with open(args.output_jsonl, "w", encoding="utf-8") as f:
            for pred in predictions:
                f.write(json.dumps(pred, ensure_ascii=False) + "\n")
        print(f"[SAVED] Đã lưu chi tiết từng câu hỏi tại: {args.output_jsonl}")

    # 5. Cập nhật bảng tổng kết CSV
    if args.summary_csv:
        os.makedirs(os.path.dirname(args.summary_csv), exist_ok=True)
        row = {
            "model_name": args.model_name,
            "backend": runner.actual_backend,
            "num_samples": len(samples),
            "full_f1": round(avg_f1, 2),
            "full_em": round(avg_em, 2),
            "subset_100_f1": round(subset_100_f1, 2),
            "subset_100_em": round(subset_100_em, 2),
            "elapsed_seconds": round(elapsed_time, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        fieldnames = ["model_name", "backend", "num_samples", "full_f1", "full_em", "subset_100_f1", "subset_100_em", "elapsed_seconds", "timestamp"]
        file_exists = os.path.exists(args.summary_csv)
        with open(args.summary_csv, "a", newline="", encoding="utf-8-sig") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        print(f"[SAVED] Đã cập nhật bảng tổng kết tại: {args.summary_csv}")

    # 6. Dọn dẹp cache
    runner.unload(evict_hf_cache=(not args.no_evict_cache))

if __name__ == "__main__":
    main()
