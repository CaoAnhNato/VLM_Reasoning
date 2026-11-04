#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TIER 1 FAST FALSIFICATION PROBE RUNNER (N=100)

Executes rapid empirical falsification on the curated 100-question probe
from results/cerv_400_benchmark/predictions.jsonl across Classes A through E.

Strict A* Evaluation Protocol:
- Uses exact unrelaxed SQuAD / ViTextVQA metrics from scripts.benchmark.metrics.
- Compares against CERV baseline (Token F1: 41.18%, Substring Match: 18.00%).
- Verification Gate: Delta Token F1 >= +3.0% and Delta Substring Match >= +5.0%.
"""

import argparse
import datetime
import gc
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from PIL import Image
import torch

# Clean UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.benchmark.metrics import (
    compute_token_f1,
    compute_exact_match,
    compute_vqa_accuracy,
    compute_anls,
    compute_substring_match,
    compute_stage2_metrics,
)
from scripts.benchmark.evaluate_vitextvqa import load_vintern_model
from src.nevfe.pipeline import NEVFEPipeline


DEFAULT_MANIFEST_PATH = os.path.join(
    PROJECT_ROOT, ".agents", "teamwork", "explorer_r5_survey_1", "probe_100_manifest.json"
)
DEFAULT_VERIFIED_DATASET = os.path.join(
    PROJECT_ROOT, "results", "swintextspotter_auto_annot", "answer_bboxes_400_verified.json"
)
DEFAULT_IMAGE_DIR = os.path.join(PROJECT_ROOT, "data", "vitextvqa", "st_images")


def resolve_image_path(image_name: str, candidate_dirs: List[str]) -> Optional[str]:
    for d in candidate_dirs:
        if not d or not os.path.isdir(d):
            continue
        p = os.path.join(d, image_name)
        if os.path.isfile(p) and os.path.getsize(p) > 0:
            return p
    return None


def run_tier1_probe(args: argparse.Namespace) -> Dict[str, Any]:
    os.makedirs(args.output_dir, exist_ok=True)
    manifest_path = args.probe_manifest

    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Probe manifest not found at: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_items: List[Dict[str, Any]] = json.load(f)

    if args.limit and args.limit > 0:
        manifest_items = manifest_items[:args.limit]

    total_probe = len(manifest_items)
    print("=" * 80)
    print(" TIER 1 FAST FALSIFICATION PROBE (N=100)")
    print(f" Timestamp:      {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Pipeline Type:  {args.pipeline_type.upper()}")
    print(f" Model:          {args.model_name}")
    print(f" Device:         {args.device}")
    print(f" Hyperparams:    lambda = {args.lambda_param}, beta = {args.beta_param}, mu = {args.mu_param}")
    print(f" Probe Samples:  {total_probe}")
    print("=" * 80)

    # Load verified answers dictionary if available
    verified_answer_map: Dict[int, List[str]] = {}
    if os.path.isfile(args.verified_dataset):
        try:
            with open(args.verified_dataset, "r", encoding="utf-8") as vf:
                v_data = json.load(vf)
            for img_info in v_data.get("images", []):
                for q_info in img_info.get("quests", []):
                    qid = q_info.get("question_id")
                    ver_ans = q_info.get("verified_all_answers") or q_info.get("gt_answers")
                    if qid is not None and ver_ans:
                        verified_answer_map[qid] = ver_ans
        except Exception as e:
            print(f"[WARN] Could not load verified dataset: {e}")

    candidate_image_dirs = [
        args.image_dir,
        os.path.join(PROJECT_ROOT, "data", "vitextvqa", "st_images"),
        os.path.join(PROJECT_ROOT, "data", "vitextvqa"),
    ]

    # Group probe items by image for visual feature caching
    grouped_by_image: Dict[str, List[Dict[str, Any]]] = {}
    for item in manifest_items:
        img_name = item["image_name"]
        if img_name not in grouped_by_image:
            grouped_by_image[img_name] = []
        grouped_by_image[img_name].append(item)

    print(f"[PROBE] Grouped {total_probe} questions into {len(grouped_by_image)} unique images.")

    # Load model and NE-VFE Pipeline
    device = args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    model, tokenizer, transform = load_vintern_model(args.model_name, device=device)
    pipeline = NEVFEPipeline(
        model=model,
        tokenizer=tokenizer,
        transform=transform,
        device=device,
        lambda_param=args.lambda_param,
        beta_param=args.beta_param,
        mu_param=args.mu_param,
    )

    results_records: List[Dict[str, Any]] = []
    start_total_time = time.time()
    evaluated_count = 0

    jsonl_output_path = os.path.join(args.output_dir, "predictions.jsonl")
    jsonl_file = open(jsonl_output_path, "w", encoding="utf-8")

    for img_name, q_list in grouped_by_image.items():
        img_path = resolve_image_path(img_name, candidate_image_dirs)
        if not img_path:
            print(f"[ERROR] Could not resolve image: {img_name}")
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[ERROR] Failed to open image {img_path}: {e}")
            continue

        # Extract & cache modulated visual features for this image
        try:
            with torch.inference_mode():
                _ = pipeline.extract_visual_features(image)
        except Exception as e:
            print(f"[WARN] Feature pre-extraction exception for {img_name}: {e}")

        for item in q_list:
            evaluated_count += 1
            qid = item["question_id"]
            question = item["question"]
            error_class = item.get("error_class", "UNKNOWN")
            gt_answers = item.get("gt_answers", [])
            verified_answers = verified_answer_map.get(qid) or gt_answers

            t0 = time.time()
            pred_text = ""
            try:
                with torch.inference_mode():
                    res = pipeline.run(image, question, max_new_tokens=args.max_new_tokens)
                    pred_text = res.pred_text
            except Exception as e:
                print(f"[ERROR] Inference exception on qid {qid}: {e}")
                pred_text = ""

            elapsed = time.time() - t0

            # Strict A* Metric Computation
            s2_metrics = compute_stage2_metrics(
                pred_answer=pred_text,
                gt_answers=gt_answers,
                verified_answers=verified_answers,
                question=question,
            )

            rec = {
                "question_id": qid,
                "image_name": img_name,
                "error_class": error_class,
                "question": question,
                "gt_answers": gt_answers,
                "verified_all_answers": verified_answers,
                "cerv_baseline_pred": item.get("pred_answer", ""),
                "cerv_baseline_f1": float(item.get("f1_verified", 0.0)),
                "cerv_baseline_substring": float(item.get("substring_verified", 0.0)),
                "cerv_baseline_anls": float(item.get("anls_verified", 0.0)),
                "nevfe_pred": pred_text,
                "nevfe_f1": float(s2_metrics["f1_verified"]),
                "nevfe_substring": float(s2_metrics["substring_verified"]),
                "nevfe_anls": float(s2_metrics["anls_verified"]),
                "nevfe_em": float(s2_metrics["em_verified"]),
                "nevfe_vqa_acc": float(s2_metrics["vqa_acc_verified"]),
                "latency_s": round(elapsed, 3),
            }
            results_records.append(rec)
            jsonl_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
            jsonl_file.flush()

            if evaluated_count % 10 == 0 or evaluated_count == total_probe:
                cur_f1 = sum(r["nevfe_f1"] for r in results_records) / len(results_records) * 100
                cur_sub = sum(r["nevfe_substring"] for r in results_records) / len(results_records) * 100
                print(f"[{evaluated_count:3d}/{total_probe:3d}] "
                      f"Running F1: {cur_f1:5.2f}% | Running Substring: {cur_sub:5.2f}% | "
                      f"Last: {elapsed:.2f}s | Q: {question[:35]}... -> {pred_text[:25]}")

            if device == "cuda":
                torch.cuda.empty_cache()

    jsonl_file.close()
    total_time = time.time() - start_total_time

    # Compute Overall & Per-Class Summary
    overall_nevfe_f1 = sum(r["nevfe_f1"] for r in results_records) / max(1, len(results_records)) * 100
    overall_nevfe_sub = sum(r["nevfe_substring"] for r in results_records) / max(1, len(results_records)) * 100
    overall_nevfe_anls = sum(r["nevfe_anls"] for r in results_records) / max(1, len(results_records)) * 100
    overall_nevfe_em = sum(r["nevfe_em"] for r in results_records) / max(1, len(results_records)) * 100

    overall_cerv_f1 = sum(r["cerv_baseline_f1"] for r in results_records) / max(1, len(results_records)) * 100
    overall_cerv_sub = sum(r["cerv_baseline_substring"] for r in results_records) / max(1, len(results_records)) * 100
    overall_cerv_anls = sum(r["cerv_baseline_anls"] for r in results_records) / max(1, len(results_records)) * 100

    delta_f1 = overall_nevfe_f1 - overall_cerv_f1
    delta_sub = overall_nevfe_sub - overall_cerv_sub
    delta_anls = overall_nevfe_anls - overall_cerv_anls

    # Class breakdown
    classes = ["CLASS_A", "CLASS_B", "CLASS_C", "CLASS_D", "CLASS_E"]
    class_stats: Dict[str, Dict[str, float]] = {}
    for c in classes:
        c_recs = [r for r in results_records if r["error_class"] == c]
        if c_recs:
            c_f1 = sum(r["nevfe_f1"] for r in c_recs) / len(c_recs) * 100
            c_sub = sum(r["nevfe_substring"] for r in c_recs) / len(c_recs) * 100
            c_anls = sum(r["nevfe_anls"] for r in c_recs) / len(c_recs) * 100
            c_base_f1 = sum(r["cerv_baseline_f1"] for r in c_recs) / len(c_recs) * 100
            c_base_sub = sum(r["cerv_baseline_substring"] for r in c_recs) / len(c_recs) * 100
            class_stats[c] = {
                "count": len(c_recs),
                "nevfe_f1": round(c_f1, 2),
                "cerv_f1": round(c_base_f1, 2),
                "delta_f1": round(c_f1 - c_base_f1, 2),
                "nevfe_substring": round(c_sub, 2),
                "cerv_substring": round(c_base_sub, 2),
                "delta_substring": round(c_sub - c_base_sub, 2),
                "nevfe_anls": round(c_anls, 2),
            }

    gate_passed = (delta_f1 >= 3.0) and (delta_sub >= 5.0)

    summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "total_probe_samples": len(results_records),
        "total_runtime_s": round(total_time, 2),
        "avg_latency_s": round(total_time / max(1, len(results_records)), 3),
        "overall": {
            "nevfe_token_f1": round(overall_nevfe_f1, 2),
            "cerv_baseline_token_f1": round(overall_cerv_f1, 2),
            "delta_token_f1": round(delta_f1, 2),
            "nevfe_substring_match": round(overall_nevfe_sub, 2),
            "cerv_baseline_substring_match": round(overall_cerv_sub, 2),
            "delta_substring_match": round(delta_sub, 2),
            "nevfe_anls": round(overall_nevfe_anls, 2),
            "cerv_baseline_anls": round(overall_cerv_anls, 2),
            "delta_anls": round(delta_anls, 2),
            "nevfe_exact_match": round(overall_nevfe_em, 2),
        },
        "per_class_diagnostics": class_stats,
        "falsification_gate": {
            "required_delta_f1": 3.0,
            "achieved_delta_f1": round(delta_f1, 2),
            "required_delta_substring": 5.0,
            "achieved_delta_substring": round(delta_sub, 2),
            "gate_passed": gate_passed,
        },
    }

    summary_json_path = os.path.join(args.output_dir, "summary_metrics.json")
    with open(summary_json_path, "w", encoding="utf-8") as sf:
        json.dump(summary, sf, ensure_ascii=False, indent=2)

    # Print publication-grade summary report
    print("\n" + "=" * 80)
    print(" TIER 1 FAST FALSIFICATION PROBE: EMPIRICAL VALIDATION REPORT")
    print("=" * 80)
    print(f" Total Evaluated:            {len(results_records)} questions across {len(grouped_by_image)} images")
    print(f" Total Runtime:              {total_time:.1f} s ({total_time/len(results_records):.2f} s/question)")
    print("-" * 80)
    print(f" Token F1 Score:             NE-VFE: {overall_nevfe_f1:5.2f}% | CERV: {overall_cerv_f1:5.2f}% | Delta: {delta_f1:+5.2f}%")
    print(f" Substring Match:            NE-VFE: {overall_nevfe_sub:5.2f}% | CERV: {overall_cerv_sub:5.2f}% | Delta: {delta_sub:+5.2f}%")
    print(f" ANLS (Relaxed Accuracy):    NE-VFE: {overall_nevfe_anls:5.2f}% | CERV: {overall_cerv_anls:5.2f}% | Delta: {delta_anls:+5.2f}%")
    print(f" Exact Match:                NE-VFE: {overall_nevfe_em:5.2f}%")
    print("-" * 80)
    print(" PER-CLASS PERFORMANCE BREAKDOWN:")
    for c, stats in class_stats.items():
        print(f"   [{c}] (N={stats['count']:2d}): "
              f"F1: {stats['nevfe_f1']:5.2f}% (d: {stats['delta_f1']:+5.2f}%) | "
              f"Sub: {stats['nevfe_substring']:5.2f}% (d: {stats['delta_substring']:+5.2f}%) | "
              f"ANLS: {stats['nevfe_anls']:5.2f}%")
    print("-" * 80)
    print(f" FALSIFICATION GATE STATUS:   {'PASSED (CLEAR FOR TIER 2 FULL RUN)' if gate_passed else 'FAILED (REJECTED)'}")
    print(f" Target Delta F1: >= +3.0%    -> Achieved: {delta_f1:+5.2f}%")
    print(f" Target Delta Sub: >= +5.0%   -> Achieved: {delta_sub:+5.2f}%")
    print("=" * 80)
    print(f" Detailed predictions saved: {jsonl_output_path}")
    print(f" Summary metrics saved:      {summary_json_path}")
    print("=" * 80 + "\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Tier 1 Fast Falsification Probe Runner")
    parser.add_argument("--probe-manifest", dest="probe_manifest", type=str, default=DEFAULT_MANIFEST_PATH, help="Path to probe_100_manifest.json")
    parser.add_argument("--verified-dataset", dest="verified_dataset", type=str, default=DEFAULT_VERIFIED_DATASET, help="Path to answer_bboxes_400_verified.json")
    parser.add_argument("--image-dir", dest="image_dir", type=str, default=DEFAULT_IMAGE_DIR, help="Path to image directory")
    parser.add_argument("--pipeline-type", "--method", dest="pipeline_type", type=str, default="nevfe", choices=["nevfe"], help="Pipeline type")
    parser.add_argument("--model-name", dest="model_name", type=str, default="5CD-AI/Vintern-1B-v3_5", help="Hugging Face model ID")
    parser.add_argument("--device", dest="device", type=str, default="cuda", help="Execution device (cuda or cpu)")
    parser.add_argument("--lambda-param", "--lambda", dest="lambda_param", type=float, default=1.0, help="Visual evidence coupling constant")
    parser.add_argument("--beta-param", "--beta", dest="beta_param", type=float, default=1.0, help="Thermodynamic precision")
    parser.add_argument("--mu-param", "--mu", dest="mu_param", type=float, default=3.0, help="QO-VFE query barrier height parameter")
    parser.add_argument("--output-dir", dest="output_dir", type=str, default="results/tier1_probe_nevfe", help="Output directory")
    parser.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=64, help="Max new tokens")
    parser.add_argument("--limit", dest="limit", type=int, default=0, help="Limit number of probe questions (for testing)")
    args = parser.parse_args()

    run_tier1_probe(args)


if __name__ == "__main__":
    main()
