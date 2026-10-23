#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
CLI Runner for SwinTextSpotter (Swin Transformer Backbone - SWINTS-swin-vintext)
Automated Answer-Region Bounding Box Detection Pipeline on ViTextVQA (Target: 400 Images).

Features:
- Detects generous Answer Region BBoxes (`answer_region_bbox`) covering the GT answer plus
  surrounding local context/noisy text on the same line/cluster (since VLMs can read & filter
  surrounding noisy text, avoiding clipped answers).
- Explicitly supports multiple questions / multiple answer regions per image (`answer_regions` & `quests`).
- Out-of-the-box ready for Google Colab (L4 24GB GPU) and Local GPU:
  1. Auto-downloads `SWINTS-swin-vintext.pth` and PCA `.npz` matrix if missing.
  2. Auto-loads QA pairs from `data/vitextvqa/ViTextVQA_dev.json` or Git-tracked
     `results/predictions/5CD-AI_Vintern-1B-v3_5_preds.jsonl`.
  3. Auto-streams missing target `.jpg` images from HuggingFace `datasets/nhonhoccode/ViTextVQA/images.zip`.
  4. Saves an easy-to-download & easy-to-edit JSON file (`answer_bboxes_400.json`) + resumable JSONL.

Usage:
  1. Quick local GPU test (8 images):
     .\venv\Scripts\python.exe scripts/benchmark/run_swin_auto_annotate.py --test

  2. Run on Google Colab (L4 24GB GPU) for 400 images:
     python scripts/benchmark/run_swin_auto_annotate.py --num-images 400
"""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import os
import sys
import time
import urllib.request
import zipfile
from typing import Any, Dict, List, Optional, Set, Tuple

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
from tqdm.auto import tqdm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from swin_text_spotter import MemoryGuard, RAMPrefetchCache, SwinTextSpotterAnnotator


DEFAULT_CKPT = os.path.join(
    PROJECT_ROOT, "data", "weights", "swintextspotter", "SWINTS-swin-vintext.pth"
)
DEFAULT_NPZ = os.path.join(
    PROJECT_ROOT,
    "data",
    "weights",
    "swintextspotter",
    "coco_2017_train_class_agnosticTrue_whitenTrue_sigmoidTrue_60_siz28.npz",
)
DEFAULT_IMAGE_DIR = os.path.join(PROJECT_ROOT, "data", "vitextvqa", "st_images")
DEFAULT_OUT_DIR = os.path.join(PROJECT_ROOT, "results", "swintextspotter_auto_annot")
FALLBACK_PREDS_JSONL = os.path.join(
    PROJECT_ROOT, "results", "predictions", "5CD-AI_Vintern-1B-v3_5_preds.jsonl"
)

NPZ_RAW_URL = (
    "https://raw.githubusercontent.com/mxin262/SwinTextSpotter/main/"
    "projects/SWINTS/LME/coco_2017_train_class_agnosticTrue_whitenTrue_sigmoidTrue_60_siz28.npz"
)
CKPT_GDRIVE_ID = "1IfyPrYFnQOWoY8pPg-GIN5ofuALU15yD"


def ensure_swin_weights(ckpt_path: str, npz_path: str) -> None:
    """Automatically download SWINTS-swin-vintext weights & PCA matrix if missing (e.g., on Colab)."""
    os.makedirs(os.path.dirname(os.path.abspath(npz_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(ckpt_path)), exist_ok=True)

    if not os.path.exists(npz_path) or os.path.getsize(npz_path) < 100_000:
        print(f"[WEIGHTS] Downloading PCA mask matrix to {npz_path}...")
        urllib.request.urlretrieve(NPZ_RAW_URL, npz_path)

    if not os.path.exists(ckpt_path) or os.path.getsize(ckpt_path) < 550_000_000:
        print(f"[WEIGHTS] Downloading SWINTS-swin-vintext.pth (577 MB) to {ckpt_path}...")
        try:
            import gdown

            gdown.download(id=CKPT_GDRIVE_ID, output=ckpt_path, quiet=False)
        except Exception as e:
            print(f"[WEIGHTS] gdown fallback to direct HTTP stream ({e})...")
            url = (
                f"https://drive.usercontent.google.com/download?"
                f"id={CKPT_GDRIVE_ID}&export=download&confirm=t"
            )
            urllib.request.urlretrieve(url, ckpt_path)

        if not os.path.exists(ckpt_path) or os.path.getsize(ckpt_path) < 550_000_000:
            raise RuntimeError(f"Failed to download valid checkpoint at {ckpt_path}")


def load_qa_annotations(
    split: str,
    hf_token: Optional[str] = None,
) -> Tuple[Dict[int, str], Dict[int, List[Dict[str, Any]]]]:
    """
    Load ViTextVQA image filenames and multi-question annotations from `data/vitextvqa/ViTextVQA_<split>.json`.
    If missing locally and `hf_token` is provided, automatically downloads `ViTextVQA_<split>.json`
    from `minhquan6203/ViTextVQA`; otherwise falls back to Git-tracked `results/predictions/5CD-AI_Vintern-1B-v3_5_preds.jsonl`.
    """
    split_file_map = {
        "dev": "ViTextVQA_dev.json",
        "test_gt": "ViTextVQA_test_gt.json",
        "train": "ViTextVQA_train.json",
    }
    vitext_dir = os.path.join(PROJECT_ROOT, "data", "vitextvqa")
    json_path = os.path.join(vitext_dir, split_file_map[split])

    if not os.path.exists(json_path) and hf_token:
        try:
            from huggingface_hub import hf_hub_download

            os.makedirs(vitext_dir, exist_ok=True)
            print(
                f"[DATA] Downloading {split_file_map[split]} from minhquan6203/ViTextVQA with HF token..."
            )
            hf_hub_download(
                repo_id="minhquan6203/ViTextVQA",
                repo_type="dataset",
                filename=split_file_map[split],
                local_dir=vitext_dir,
                token=hf_token,
            )
        except Exception as e:
            print(f"[DATA] Direct JSON download skipped ({e}); checking fallback...")

    id_to_filename: Dict[int, str] = {}
    img_to_qas: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    if os.path.exists(json_path):
        print(f"[DATA] Loading QA annotations from {json_path}...")
        with open(json_path, "r", encoding="utf-8") as f:
            raw_split = json.load(f)

        for img in raw_split.get("images", []):
            id_to_filename[int(img["id"])] = img["filename"]

        for ann in raw_split.get("annotations", []):
            img_id = int(ann["image_id"])
            img_to_qas[img_id].append(
                {
                    "id": ann.get("id"),
                    "image_id": img_id,
                    "question": ann.get("question", ""),
                    "answers": ann.get("answers", []),
                }
            )
        del raw_split
        return id_to_filename, img_to_qas

    if os.path.exists(FALLBACK_PREDS_JSONL):
        print(
            f"[DATA] {json_path} not found; loading QA annotations from Git-tracked {FALLBACK_PREDS_JSONL}..."
        )
        with open(FALLBACK_PREDS_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                fname = rec.get("image_name", "")
                if not fname:
                    continue
                stem = os.path.splitext(fname)[0]
                img_id = int(stem) if stem.isdigit() else hash(fname)
                id_to_filename[img_id] = fname
                img_to_qas[img_id].append(
                    {
                        "id": rec.get("question_id"),
                        "image_id": img_id,
                        "question": rec.get("question", ""),
                        "answers": rec.get("ground_truths", []),
                    }
                )
        return id_to_filename, img_to_qas

    raise FileNotFoundError(
        f"Neither {json_path} nor {FALLBACK_PREDS_JSONL} found. Please clone full repo or pass --hf-token."
    )


def ensure_target_images(
    image_names: Set[str],
    image_dir: str,
    hf_token: Optional[str] = None,
    num_workers: int = 8,
) -> None:
    """
    Ensure all target `.jpg` files exist in `image_dir`.
    Combines authenticated HuggingFace token (`hf_token`, lifting anonymous rate limits)
    with multi-threaded HTTP Range streaming (`ThreadPoolExecutor`) so fetching 400 target images
    completes in seconds without downloading the entire 5.3GB archive.
    """
    os.makedirs(image_dir, exist_ok=True)
    missing = sorted(
        name
        for name in image_names
        if not (
            os.path.exists(os.path.join(image_dir, name))
            and os.path.getsize(os.path.join(image_dir, name)) > 0
        )
    )
    if not missing:
        return

    auth_mode = "Authenticated (HF Token)" if hf_token else "Anonymous"
    print(
        f"[DATA] Streaming {len(missing)} missing target images from HuggingFace [{auth_mode}, {num_workers} threads]..."
    )
    from huggingface_hub import HfFileSystem

    zip_uri = "datasets/nhonhoccode/ViTextVQA/images.zip"
    fs_main = HfFileSystem(token=hf_token)
    with fs_main.open(zip_uri, "rb") as fp:
        with zipfile.ZipFile(fp) as zf:
            name_map = {os.path.basename(n): n for n in zf.namelist() if not n.endswith("/")}

    valid_targets = [t for t in missing if t in name_map]
    if not valid_targets:
        return

    workers = max(1, min(num_workers, len(valid_targets)))
    chunks = [valid_targets[i::workers] for i in range(workers)]

    t_dl = time.perf_counter()
    with tqdm(
        total=len(valid_targets),
        desc="[DATA] Streaming HF images",
        unit="img",
        dynamic_ncols=True,
    ) as dl_pbar:
        def _fetch_chunk(chunk_targets: List[str]) -> int:
            if not chunk_targets:
                return 0
            local_fs = HfFileSystem(token=hf_token)
            count = 0
            with local_fs.open(zip_uri, "rb") as fp_w:
                with zipfile.ZipFile(fp_w) as zf_w:
                    for target_name in chunk_targets:
                        data = zf_w.read(name_map[target_name])
                        out_p = os.path.join(image_dir, target_name)
                        with open(out_p, "wb") as out_f:
                            out_f.write(data)
                        count += 1
                        dl_pbar.update(1)
            return count

        with ThreadPoolExecutor(max_workers=workers) as pool:
            fetched_total = sum(pool.map(_fetch_chunk, chunks))
    print(
        f"[DATA] Cached {fetched_total}/{len(missing)} images in {time.perf_counter() - t_dl:.1f}s"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SwinTextSpotter Answer-Region BBox Auto-Annotation Pipeline (Colab L4 & Local GPU)"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.environ.get("HF_TOKEN", None),
        help="HuggingFace token (or set HF_TOKEN env var) for authenticated high-speed dataset streaming.",
    )
    parser.add_argument(
        "--colab-l4",
        action="store_true",
        help=(
            "Enable dedicated Colab L4 High-Speed Profile: "
            "(1) Remove torch.cuda.empty_cache() stalls, "
            "(2) Enable native bfloat16 AMP autocast + TF32, "
            "(3) Enable RAMPrefetchCache with pin_memory=True & async non_blocking=True H2D DMA."
        ),
    )
    parser.add_argument(
        "--prefetch-workers",
        type=int,
        default=4,
        help="Number of CPU threads for RAMPrefetchCache (default: 4).",
    )
    parser.add_argument(
        "--prefetch-size",
        type=int,
        default=0,
        help="Max images held in pinned RAMPrefetchCache (0 = auto: 400 on Colab L4, 32 on local PC).",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run quick verification test on 8 images with visual overlays.",
    )
    parser.add_argument(
        "--num-images",
        "--target-images",
        dest="num_images",
        type=int,
        default=400,
        help="Target number of images to auto-annotate in batch mode (default: 400).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-image detailed log lines alongside the tqdm progress bar.",
    )
    parser.add_argument(
        "--num-test-images",
        type=int,
        default=8,
        help="Number of images to evaluate in --test mode (default: 8).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        choices=["dev", "test_gt", "train"],
        help="ViTextVQA split to load annotations from (default: dev).",
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.40,
        help="Confidence threshold for SwinTextSpotter proposals (default: 0.40).",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=1000,
        help="Shortest edge resize for inference (default: 1000).",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=0,
        help="Longest edge cap (0 = auto: 1800 on >=16GB Colab L4 GPU, 1600 on 4GB local GPU).",
    )
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="Save visual bbox overlay images even in batch mode.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh instead of resuming from existing JSONL output.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUT_DIR,
        help="Directory to save bbox JSON/JSONL files and visual overlays.",
    )
    return parser.parse_args()


def build_compact_editable_record(res: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a clean, human-editable record for each image containing:
    - `answer_regions`: 1 or more detected answer region bboxes (`[x1, y1, x2, y2]`) on this image.
    - `quests`: Per-question grounding (handling 2+ questions per image) with `answer_region_bbox`.
    - `candidate_text_lines`: All proximate merged text lines spotted on the image for easy reference.
    """
    return {
        "image_id": res["image_id"],
        "image_name": res["image_name"],
        "width": res["width"],
        "height": res["height"],
        "num_quests": len(res["qa_groundings"]),
        "num_answer_regions": res["num_answer_regions"],
        "answer_regions": [
            {
                "region_id": r["region_id"],
                "bbox": r["bbox"],
                "exact_answer_bbox": r["exact_answer_bbox"],
                "context_line_bbox": r["context_line_bbox"],
                "ocr_text_in_region": r["ocr_text_in_region"],
                "gt_answers": r["gt_answers"],
                "question_ids": r["question_ids"],
                "questions": r["questions"],
                "match_score": r["match_score"],
                "needs_manual_edit": r["needs_manual_edit"],
            }
            for r in res["answer_regions"]
        ],
        "quests": [
            {
                "quest_idx": q["quest_idx"],
                "question_id": q["question_id"],
                "question": q["question"],
                "gt_answers": q["answers"],
                "answer_region_bbox": q["answer_region_bbox"],
                "gt_context_line_bbox": q["gt_context_line_bbox"],
                "gt_answer_union_bbox": q["gt_answer_union_bbox"],
                "region_ocr_text": q["region_ocr_text"],
                "matched_ocr_phrase": q["matched_ocr_phrase"],
                "match_score": q["match_score"],
                "needs_manual_edit": q["needs_manual_edit"],
            }
            for q in res["qa_groundings"]
        ],
        "candidate_text_lines": [
            {
                "line_id": l["line_id"],
                "role": l.get("role", "auxiliary"),
                "text": l["text"],
                "bbox": l["bbox"],
                "score": l["score"],
            }
            for l in res["all_merged_lines_tl_br"]
        ],
    }


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    save_vis = args.test or args.save_vis
    vis_dir = os.path.join(
        args.output_dir, "vis_test" if args.test else f"vis_{args.split}_{args.num_images}"
    )
    if save_vis:
        os.makedirs(vis_dir, exist_ok=True)

    # Detect GPU & auto-configure Colab L4 High-Speed Profile vs Local 4GB GPU
    total_vram_mb = 0.0
    if torch.cuda.is_available():
        total_vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    is_l4_class_gpu = total_vram_mb >= 15000
    fast_l4_mode = bool(args.colab_l4 or is_l4_class_gpu)

    max_vram_guard_mb = max(3400.0, total_vram_mb * 0.85) if total_vram_mb > 0 else 3400.0
    max_size_test = (
        args.max_size
        if args.max_size > 0
        else (1800 if is_l4_class_gpu else 1600)
    )

    memory_guard = MemoryGuard(max_ram_percent=85.0, max_vram_mb=max_vram_guard_mb)
    init_snap = memory_guard.snapshot()
    gpu_label = (
        f"cuda ({torch.cuda.get_device_name(0)}, {total_vram_mb / 1024:.1f} GB VRAM)"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(
        f"[INIT] Device: {gpu_label} | max_size={max_size_test} | "
        f"Colab-L4 Fast Mode: {'ON (no empty_cache + AMP bf16 + RAMPrefetch pin_memory=True)' if fast_l4_mode else 'OFF'} | "
        f"System RAM: {init_snap.ram_used_gb:.2f}/{init_snap.ram_total_gb:.2f} GB ({init_snap.ram_percent:.1f}%)"
    )

    # 1. Ensure SwinTextSpotter weights are present (auto-downloads on Colab if missing)
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    ensure_swin_weights(DEFAULT_CKPT, DEFAULT_NPZ)

    # 2. Load QA pairs and select target images (default: 400 images)
    id_to_filename, img_to_qas = load_qa_annotations(args.split, hf_token=args.hf_token)
    if not fast_l4_mode:
        memory_guard.cleanup(force_cuda_empty=False)

    all_sorted_ids = sorted(img_to_qas.keys())
    target_count = args.num_test_images if args.test else args.num_images
    target_img_ids = all_sorted_ids[:target_count]

    # 3. Ensure target `.jpg` files exist locally (streams from HF zip on Colab if missing)
    needed_filenames = {id_to_filename.get(iid, f"{iid}.jpg") for iid in target_img_ids}
    ensure_target_images(needed_filenames, DEFAULT_IMAGE_DIR, hf_token=args.hf_token)

    # Filter to existing images
    target_img_ids = [
        iid
        for iid in target_img_ids
        if os.path.exists(os.path.join(DEFAULT_IMAGE_DIR, id_to_filename.get(iid, f"{iid}.jpg")))
    ]

    print(f"[LOAD] Initializing SwinTextSpotter (SWINTS-swin-vintext) on {gpu_label}...")
    annotator = SwinTextSpotterAnnotator(
        ckpt_path=DEFAULT_CKPT,
        npz_path=DEFAULT_NPZ,
        device="cuda" if torch.cuda.is_available() else "cpu",
        min_size_test=args.min_size,
        max_size_test=max_size_test,
        confidence_threshold=args.conf_thresh,
        use_amp_bf16=fast_l4_mode,
        disable_empty_cache=fast_l4_mode,
        pin_memory=fast_l4_mode,
        memory_guard=memory_guard,
    )
    post_load_snap = memory_guard.snapshot()
    print(
        f"[MODEL READY] System RAM: {post_load_snap.ram_used_gb:.2f}/{post_load_snap.ram_total_gb:.2f} GB "
        f"({post_load_snap.ram_percent:.1f}%) | GPU VRAM Allocated: {post_load_snap.vram_allocated_mb:.1f} MB"
    )

    tag = "test" if args.test else f"{args.split}_{len(target_img_ids)}"
    jsonl_path = os.path.join(
        args.output_dir,
        "test_annotations.jsonl" if args.test else f"annotations_{tag}.jsonl",
    )
    editable_bboxes_path = os.path.join(
        args.output_dir,
        "test_answer_bboxes.json" if args.test else f"answer_bboxes_{len(target_img_ids)}.json",
    )
    summary_path = os.path.join(
        args.output_dir,
        "test_summary.json" if args.test else f"summary_{tag}.json",
    )

    # Support resuming in batch mode
    completed_records: Dict[int, Dict[str, Any]] = {}
    if not args.test and not args.no_resume and os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as rf:
            for line in rf:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        completed_records[int(rec["image_id"])] = rec
                    except Exception:
                        pass
        if completed_records:
            print(f"[RESUME] Found {len(completed_records)} already-annotated images in {jsonl_path}")

    # Initialize RAMPrefetchCache (pin_memory=True) for pending images
    prefetch_cache: Optional[RAMPrefetchCache] = None
    if fast_l4_mode:
        auto_prefetch_cap = (
            args.prefetch_size
            if args.prefetch_size > 0
            else (400 if init_snap.ram_total_gb >= 28.0 else 32)
        )
        pending_paths = [
            os.path.join(DEFAULT_IMAGE_DIR, id_to_filename.get(iid, f"{iid}.jpg"))
            for iid in target_img_ids
            if iid not in completed_records
        ]
        prefetch_cache = RAMPrefetchCache(
            min_size_test=args.min_size,
            max_size_test=max_size_test,
            pin_memory=True,
            num_workers=args.prefetch_workers,
            max_prefetched=auto_prefetch_cap,
            keep_rgb_for_vis=save_vis,
        )
        prefetch_cache.enqueue(pending_paths)
        print(
            f"[PREFETCH] RAMPrefetchCache active (pin_memory=True, workers={args.prefetch_workers}, window={auto_prefetch_cap})"
        )

    editable_records: List[Dict[str, Any]] = []
    results_summary: List[Dict[str, Any]] = []
    total_qas = 0
    matched_qas = 0
    total_answer_regions = 0
    t_start = time.perf_counter()

    mode_flag = "w" if (args.test or args.no_resume or not completed_records) else "a"
    try:
        with open(jsonl_path, mode_flag, encoding="utf-8") as jsonl_fp, tqdm(
            total=len(target_img_ids),
            desc="[DETECT] SwinTextSpotter",
            unit="img",
            dynamic_ncols=True,
        ) as pbar:
            for idx, img_id in enumerate(target_img_ids, 1):
                fname = id_to_filename.get(img_id, f"{img_id}.jpg")
                img_path = os.path.join(DEFAULT_IMAGE_DIR, fname)
                vis_path = os.path.join(vis_dir, f"annot_{fname}") if save_vis else None

                if img_id in completed_records:
                    res = completed_records[img_id]
                else:
                    res = annotator.annotate_image(
                        image_path=img_path,
                        image_id=img_id,
                        qa_annotations=img_to_qas[img_id],
                        vis_output_path=vis_path,
                        prefetch_cache=prefetch_cache,
                    )
                    jsonl_fp.write(json.dumps(res, ensure_ascii=False) + "\n")
                    jsonl_fp.flush()

                compact_rec = build_compact_editable_record(res)
                editable_records.append(compact_rec)

                qas = res["qa_groundings"]
                n_qas = len(qas)
                n_matched = sum(1 for q in qas if q.get("answer_region_bbox") is not None)
                n_regions = res.get("num_answer_regions", len(res.get("answer_regions", [])))
                total_qas += n_qas
                matched_qas += n_matched
                total_answer_regions += n_regions

                pbar.update(1)
                pbar.set_postfix_str(
                    f"img={fname} | matched={matched_qas}/{total_qas} | "
                    f"bboxes={total_answer_regions} | {res['latency_ms']:.0f}ms | "
                    f"VRAM={res['memory']['vram_peak_mb']:.0f}MB"
                )

                if args.test or args.verbose:
                    pbar.write(
                        f"[{idx:03d}/{len(target_img_ids):03d}] {fname} ({res['width']}x{res['height']}) | "
                        f"Quests: {n_qas} (Matched: {n_matched}/{n_qas}) -> Answer BBoxes: {n_regions} | "
                        f"Lines: {res['num_total_lines']} | "
                        f"Time: {res['latency_ms']:.0f}ms | "
                        f"RAM: {res['memory']['ram_used_gb']:.2f}GB ({res['memory']['ram_percent']:.1f}%) | "
                        f"VRAM Peak: {res['memory']['vram_peak_mb']:.0f}MB"
                    )

                results_summary.append(
                    {
                        "image_id": img_id,
                        "image_name": fname,
                        "resolution": f"{res['width']}x{res['height']}",
                        "latency_ms": res["latency_ms"],
                        "num_quests": n_qas,
                        "num_matched_quests": n_matched,
                        "num_answer_regions": n_regions,
                        "num_total_lines": res["num_total_lines"],
                        "answer_regions": compact_rec["answer_regions"],
                        "quests": compact_rec["quests"],
                        "vis_path": vis_path,
                        "memory": res["memory"],
                    }
                )
    finally:
        if prefetch_cache is not None:
            prefetch_cache.close()

    total_time_s = time.perf_counter() - t_start
    final_snap = memory_guard.cleanup(force_cuda_empty=True)

    # Save easy-to-download & editable Answer BBoxes JSON file
    editable_payload = {
        "metadata": {
            "model": "SwinTextSpotter (SWINTS-swin-vintext)",
            "split": args.split,
            "num_images": len(editable_records),
            "total_quests": total_qas,
            "matched_quests": matched_qas,
            "quest_match_rate_pct": round(100.0 * matched_qas / max(1, total_qas), 2),
            "total_answer_region_bboxes": total_answer_regions,
            "bbox_format": "[x1, y1, x2, y2] in original pixel coordinates",
            "note": (
                "Each image can have multiple questions (`quests`) and multiple `answer_regions`. "
                "`answer_region_bbox` generously encloses the answer + local context/noisy text on the same line/cluster."
            ),
        },
        "images": editable_records,
    }
    with open(editable_bboxes_path, "w", encoding="utf-8") as f:
        json.dump(editable_payload, f, ensure_ascii=False, indent=2)

    summary_payload = {
        "model": "SwinTextSpotter (SWINTS-swin-vintext, Swin-Transformer-Tiny + FPN)",
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "split": args.split,
        "num_images_processed": len(target_img_ids),
        "total_qas": total_qas,
        "matched_qas": matched_qas,
        "qa_match_rate_pct": round(100.0 * matched_qas / max(1, total_qas), 2),
        "total_answer_regions": total_answer_regions,
        "avg_latency_ms": round(
            sum(r["latency_ms"] for r in results_summary) / max(1, len(results_summary)), 1
        ),
        "total_time_s": round(total_time_s, 2),
        "editable_bboxes_file": editable_bboxes_path,
        "memory_safety": {
            "init_ram_gb": init_snap.ram_used_gb,
            "peak_ram_gb": round(memory_guard.peak_ram_used_gb, 2),
            "total_ram_gb": final_snap.ram_total_gb,
            "peak_ram_percent": round(memory_guard.peak_ram_percent, 1),
            "peak_vram_mb": round(memory_guard.peak_vram_mb, 1),
            "final_vram_allocated_mb": final_snap.vram_allocated_mb,
        },
        "results": results_summary,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)

    print(
        f"\n[DONE] Processed {len(target_img_ids)} images in {total_time_s:.2f}s | "
        f"Quests Matched: {matched_qas}/{total_qas} ({summary_payload['qa_match_rate_pct']}%) | "
        f"Total Answer Region BBoxes: {total_answer_regions} | "
        f"Saved Editable BBoxes: {editable_bboxes_path} | "
        f"Peak RAM: {summary_payload['memory_safety']['peak_ram_gb']:.2f}/{final_snap.ram_total_gb:.2f} GB "
        f"({summary_payload['memory_safety']['peak_ram_percent']:.1f}%) | "
        f"Peak VRAM: {summary_payload['memory_safety']['peak_vram_mb']:.1f} MB"
    )


if __name__ == "__main__":
    main()
