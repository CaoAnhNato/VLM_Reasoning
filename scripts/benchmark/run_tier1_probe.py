#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TIER 1 FAST FALSIFICATION PROBE RUNNER (N=100) — ITERATION 23 CALIBRATED
Multi-Worker Sharded Architecture for Google Colab L4 GPU (~40s execution)

Key Calibrations for Iteration 23:
1. Ratcheted Falsification Baseline:
   Updated to Reigning Probe Record (from results/tier1_probe_nevfe/summary_metrics.json):
     * Champion_Probe_F1:        41.94%
     * Champion_Probe_Substring: 34.00%
     * Champion_Probe_ANLS:      30.17%
     * Champion_Probe_EM:         9.00%
2. Fast Falsification Gate:
   Candidate_Probe_F1 > 41.94% AND Candidate_Probe_Substring >= 34.00%.
   If either fails -> Immediate VETO and rollback to reigning champion.
3. Modular Micro-Ablation Switches:
   Supports Factorial Micro-Ablation Strategy (Directive #3) via:
     * --ablation-mask "1,1,1" (or "1,0,0", "0,1,0", "0,0,1", "1,0,1", etc.)
     * --enable-comp-a / --enable-comp-b / --enable-comp-c flags.
4. Directive #5 & #6 Compliance:
   Turnkey multi-worker sharding (4 workers), zero debug on live GPU,
   pure mock verification capability locally.
"""

import argparse
import datetime
import gc
import glob
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from PIL import Image
import torch

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

# Ratcheted Baseline Invariants for Reigning Champion on N=100 probe (Iteration 22 record)
CHAMPION_PROBE_SCORES = {
    "token_f1": 41.94,
    "substring_match": 34.00,
    "anls": 30.17,
    "exact_match": 9.00,
}

# Per-Class Scores achieved by Reigning Champion on N=100 probe
CHAMPION_CLASS_SCORES = {
    "CLASS_A": {"count": 30, "f1": 15.60, "substring": 26.67, "anls": 7.93, "em": 0.00},
    "CLASS_B": {"count": 35, "f1": 62.88, "substring": 37.14, "anls": 50.93, "em": 11.43},
    "CLASS_C": {"count": 12, "f1": 42.09, "substring": 33.33, "anls": 23.44, "em": 0.00},
    "CLASS_D": {"count": 20, "f1": 45.58, "substring": 40.00, "anls": 30.78, "em": 20.00},
    "CLASS_E": {"count": 3, "f1": 36.23, "substring": 33.33, "anls": 33.33, "em": 33.33},
}


def resolve_image_path(image_name: str, candidate_dirs: List[str]) -> Optional[str]:
    for d in candidate_dirs:
        if not d or not os.path.isdir(d):
            continue
        p = os.path.join(d, image_name)
        if os.path.isfile(p) and os.path.getsize(p) > 0:
            return p
    return None


def parse_ablation_mask(mask_str: str) -> Tuple[bool, bool, bool]:
    """Parse comma-separated mask e.g. '1,0,1' into boolean flags."""
    parts = [p.strip() for p in mask_str.split(",")]
    b_a = bool(int(parts[0])) if len(parts) > 0 else True
    b_b = bool(int(parts[1])) if len(parts) > 1 else True
    b_c = bool(int(parts[2])) if len(parts) > 2 else True
    return b_a, b_b, b_c


def run_tier1_probe_worker(args: argparse.Namespace) -> Dict[str, Any]:
    os.makedirs(args.output_dir, exist_ok=True)
    manifest_path = args.probe_manifest

    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Probe manifest not found at: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_items: List[Dict[str, Any]] = json.load(f)

    effective_limit = 0
    if getattr(args, "test_samples", 0) and args.test_samples > 0:
        effective_limit = args.test_samples
    elif getattr(args, "limit", 0) and args.limit > 0:
        effective_limit = args.limit

    if effective_limit > 0:
        manifest_items = manifest_items[:effective_limit]

    # Shard by unique images deterministically
    unique_images: List[str] = []
    seen = set()
    for item in manifest_items:
        img_name = item["image_name"]
        if img_name not in seen:
            seen.add(img_name)
            unique_images.append(img_name)

    if args.num_workers > 1:
        sharded_images = {
            img for idx, img in enumerate(unique_images)
            if idx % args.num_workers == args.worker_id
        }
        manifest_items = [item for item in manifest_items if item["image_name"] in sharded_images]

    shard_info = f" (Worker {args.worker_id}/{args.num_workers})" if args.num_workers > 1 else ""
    is_mock = getattr(args, "mock", False)

    # Resolve ablation switches
    comp_a, comp_b, comp_c = parse_ablation_mask(args.ablation_mask)
    if getattr(args, "enable_comp_a", None) is not None:
        comp_a = args.enable_comp_a
    if getattr(args, "enable_comp_b", None) is not None:
        comp_b = args.enable_comp_b
    if getattr(args, "enable_comp_c", None) is not None:
        comp_c = args.enable_comp_c

    ablation_str = f"A={int(comp_a)}, B={int(comp_b)}, C={int(comp_c)}"

    print("=" * 80)
    print(f" TIER 1 FAST FALSIFICATION PROBE (N={len(manifest_items)}){shard_info}")
    print(f" Timestamp:        {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Pipeline Type:    {args.pipeline_type.upper()}{' [MOCK MODE]' if is_mock else ''}")
    print(f" Ablation Mask:    [{ablation_str}]")
    print(f" Hyperparams:      lambda = {args.lambda_param}, beta = {args.beta_param}, mu = {args.mu_param}")
    print(f" Ratchet Baseline: F1 > {CHAMPION_PROBE_SCORES['token_f1']}%, Substring >= {CHAMPION_PROBE_SCORES['substring_match']}%")
    print("=" * 80)

    # Load verified answers dictionary
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

    grouped_by_image: Dict[str, List[Dict[str, Any]]] = {}
    for item in manifest_items:
        img_name = item["image_name"]
        if img_name not in grouped_by_image:
            grouped_by_image[img_name] = []
        grouped_by_image[img_name].append(item)

    device = args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    pipeline = None

    if not is_mock:
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
        # Pass ablation flags dynamically to pipeline adapter/engine if supported
        if hasattr(pipeline, "set_ablation_switches"):
            pipeline.set_ablation_switches(comp_a=comp_a, comp_b=comp_b, comp_c=comp_c)

    results_records: List[Dict[str, Any]] = []
    start_total_time = time.time()
    evaluated_count = 0

    if args.num_workers > 1:
        jsonl_output_path = os.path.join(args.output_dir, f"predictions_shard_{args.worker_id}.jsonl")
    else:
        jsonl_output_path = os.path.join(args.output_dir, "predictions.jsonl")

    jsonl_file = open(jsonl_output_path, "w", encoding="utf-8")

    for img_name, q_list in grouped_by_image.items():
        image = None
        if not is_mock:
            img_path = resolve_image_path(img_name, candidate_image_dirs)
            if not img_path:
                print(f"[ERROR] Could not resolve image: {img_name}")
                continue
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"[ERROR] Failed to open image {img_path}: {e}")
                continue

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

            if is_mock:
                pred_text = verified_answers[0] if verified_answers else "đáp án"
            else:
                try:
                    with torch.inference_mode():
                        res = pipeline.run(image, question, max_new_tokens=args.max_new_tokens)
                        pred_text = res.pred_text
                except Exception as e:
                    print(f"[ERROR] Inference exception on qid {qid}: {e}")
                    pred_text = ""

            elapsed = time.time() - t0

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
                "candidate_pred": pred_text,
                "candidate_f1": float(s2_metrics["f1_verified"]),
                "candidate_substring": float(s2_metrics["substring_verified"]),
                "candidate_anls": float(s2_metrics["anls_verified"]),
                "candidate_em": float(s2_metrics["em_verified"]),
                "candidate_vqa_acc": float(s2_metrics["vqa_acc_verified"]),
                "latency_s": round(elapsed, 3),
            }
            results_records.append(rec)
            jsonl_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
            jsonl_file.flush()

            if evaluated_count % 5 == 0 or evaluated_count == len(manifest_items):
                cur_f1 = sum(r["candidate_f1"] for r in results_records) / len(results_records) * 100
                cur_sub = sum(r["candidate_substring"] for r in results_records) / len(results_records) * 100
                print(f"[{shard_info.strip()} {evaluated_count:2d}/{len(manifest_items):2d}] "
                      f"Running F1: {cur_f1:5.2f}% | Substring: {cur_sub:5.2f}% | "
                      f"Last: {elapsed:.2f}s | Q: {question[:30]}... -> {pred_text[:20]}")

            if device == "cuda" and not is_mock:
                torch.cuda.empty_cache()

    jsonl_file.close()
    total_time = time.time() - start_total_time

    if args.num_workers > 1:
        shard_summary_path = os.path.join(args.output_dir, f"summary_shard_{args.worker_id}.json")
        with open(shard_summary_path, "w", encoding="utf-8") as sf:
            json.dump({"worker_id": args.worker_id, "count": len(results_records), "records": results_records}, sf)
        return {"worker_id": args.worker_id, "count": len(results_records)}

    return aggregate_and_report_probe(results_records, total_time, args.output_dir, ablation_str)


def aggregate_and_report_probe(
    results_records: List[Dict[str, Any]],
    total_time: float,
    output_dir: str,
    ablation_str: str = "A=1, B=1, C=1",
) -> Dict[str, Any]:
    n = max(1, len(results_records))
    overall_f1 = sum(r["candidate_f1"] for r in results_records) / n * 100
    overall_sub = sum(r["candidate_substring"] for r in results_records) / n * 100
    overall_anls = sum(r["candidate_anls"] for r in results_records) / n * 100
    overall_em = sum(r["candidate_em"] for r in results_records) / n * 100

    delta_f1_vs_champion = overall_f1 - CHAMPION_PROBE_SCORES["token_f1"]
    delta_sub_vs_champion = overall_sub - CHAMPION_PROBE_SCORES["substring_match"]
    delta_anls_vs_champion = overall_anls - CHAMPION_PROBE_SCORES["anls"]

    classes = ["CLASS_A", "CLASS_B", "CLASS_C", "CLASS_D", "CLASS_E"]
    class_stats: Dict[str, Dict[str, float]] = {}
    for c in classes:
        c_recs = [r for r in results_records if r["error_class"] == c]
        if c_recs:
            cn = len(c_recs)
            c_f1 = sum(r["candidate_f1"] for r in c_recs) / cn * 100
            c_sub = sum(r["candidate_substring"] for r in c_recs) / cn * 100
            c_anls = sum(r["candidate_anls"] for r in c_recs) / cn * 100
            c_em = sum(r["candidate_em"] for r in c_recs) / cn * 100
            champ_c = CHAMPION_CLASS_SCORES.get(c, {})
            class_stats[c] = {
                "count": cn,
                "candidate_f1": round(c_f1, 2),
                "champion_f1": champ_c.get("f1", 0.0),
                "delta_f1": round(c_f1 - champ_c.get("f1", 0.0), 2),
                "candidate_substring": round(c_sub, 2),
                "champion_substring": champ_c.get("substring", 0.0),
                "delta_substring": round(c_sub - champ_c.get("substring", 0.0), 2),
                "candidate_anls": round(c_anls, 2),
                "candidate_em": round(c_em, 2),
            }

    # Strict Monotonic Ratchet Gate: MUST BEAT Iteration 22 Record
    gate_passed = (overall_f1 > CHAMPION_PROBE_SCORES["token_f1"]) and (overall_sub >= CHAMPION_PROBE_SCORES["substring_match"])

    summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "total_probe_samples": len(results_records),
        "total_runtime_s": round(total_time, 2),
        "avg_latency_s": round(total_time / n, 3),
        "ablation_configuration": ablation_str,
        "overall": {
            "candidate_token_f1": round(overall_f1, 2),
            "champion_token_f1": CHAMPION_PROBE_SCORES["token_f1"],
            "delta_token_f1_vs_champion": round(delta_f1_vs_champion, 2),
            "candidate_substring_match": round(overall_sub, 2),
            "champion_substring_match": CHAMPION_PROBE_SCORES["substring_match"],
            "delta_substring_vs_champion": round(delta_sub_vs_champion, 2),
            "candidate_anls": round(overall_anls, 2),
            "champion_anls": CHAMPION_PROBE_SCORES["anls"],
            "delta_anls_vs_champion": round(delta_anls_vs_champion, 2),
            "candidate_exact_match": round(overall_em, 2),
            "champion_exact_match": CHAMPION_PROBE_SCORES["exact_match"],
        },
        "per_class_diagnostics": class_stats,
        "falsification_gate": {
            "criterion": f"Candidate_F1 > {CHAMPION_PROBE_SCORES['token_f1']}% AND Candidate_Substring >= {CHAMPION_PROBE_SCORES['substring_match']}%",
            "champion_probe_f1": CHAMPION_PROBE_SCORES["token_f1"],
            "achieved_probe_f1": round(overall_f1, 2),
            "champion_probe_substring": CHAMPION_PROBE_SCORES["substring_match"],
            "achieved_probe_substring": round(overall_sub, 2),
            "gate_passed": gate_passed,
            "action": "CLEAR FOR TIER 2 FULL RUN" if gate_passed else "IMMEDIATE VETO & GIT ROLLBACK TO REIGNING CHAMPION",
        },
    }

    summary_json_path = os.path.join(output_dir, "summary_metrics.json")
    with open(summary_json_path, "w", encoding="utf-8") as sf:
        json.dump(summary, sf, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print(f" TIER 1 FAST FALSIFICATION PROBE REPORT [{ablation_str}]")
    print("=" * 80)
    print(f" Total Evaluated:            {len(results_records)} questions")
    print(f" Total Runtime:              {total_time:.1f} s ({total_time/n:.2f} s/question)")
    print("-" * 80)
    print(f" Token F1 Score:             Candidate: {overall_f1:5.2f}% | Champion: {CHAMPION_PROBE_SCORES['token_f1']:5.2f}% | Delta: {delta_f1_vs_champion:+5.2f}%")
    print(f" Substring Match:            Candidate: {overall_sub:5.2f}% | Champion: {CHAMPION_PROBE_SCORES['substring_match']:5.2f}% | Delta: {delta_sub_vs_champion:+5.2f}%")
    print(f" ANLS (Relaxed Accuracy):    Candidate: {overall_anls:5.2f}% | Champion: {CHAMPION_PROBE_SCORES['anls']:5.2f}% | Delta: {delta_anls_vs_champion:+5.2f}%")
    print(f" Exact Match:                Candidate: {overall_em:5.2f}% | Champion: {CHAMPION_PROBE_SCORES['exact_match']:5.2f}%")
    print("-" * 80)
    print(f" FALSIFICATION GATE STATUS:   {'PASSED (CLEAR FOR TIER 2 FULL RUN)' if gate_passed else 'FAILED (VETO & ROLLBACK TO CHAMPION)'}")
    print(f" Gate Requirement: Candidate F1 > {CHAMPION_PROBE_SCORES['token_f1']}% AND Candidate Substring >= {CHAMPION_PROBE_SCORES['substring_match']}%")
    print(f" Achieved: F1 = {overall_f1:.2f}%, Substring = {overall_sub:.2f}%")
    print("=" * 80 + "\n")

    return summary


def run_probe_multiprocess_coordinator(args: argparse.Namespace) -> Dict[str, Any]:
    num_workers = args.multi_proc
    os.makedirs(args.output_dir, exist_ok=True)
    start_time = time.time()

    print("\n" + "=" * 80)
    print(f" MULTI-WORKER TIER 1 PROBE COORDINATOR ({num_workers} Workers)")
    print(f" Target Device:     {args.device}")
    print(f" Output Directory:  {args.output_dir}")
    print("=" * 80)

    for pattern in ["predictions_shard_*.jsonl*", "summary_shard_*.json*"]:
        for f in glob.glob(os.path.join(args.output_dir, pattern)):
            try:
                os.remove(f)
            except Exception:
                pass

    script_path = os.path.abspath(__file__)
    procs: List[subprocess.Popen] = []

    for k in range(num_workers):
        cmd = [
            sys.executable,
            script_path,
            "--num-workers", str(num_workers),
            "--worker-id", str(k),
            "--multi-proc", "0",
            "--device", str(args.device),
            "--pipeline-type", str(args.pipeline_type),
            "--model-name", str(args.model_name),
            "--probe-manifest", str(args.probe_manifest),
            "--verified-dataset", str(args.verified_dataset),
            "--image-dir", str(args.image_dir),
            "--output-dir", str(args.output_dir),
            "--ablation-mask", str(args.ablation_mask),
        ]
        if getattr(args, "enable_comp_a", None) is not None:
            cmd += ["--enable-comp-a", str(args.enable_comp_a)]
        if getattr(args, "enable_comp_b", None) is not None:
            cmd += ["--enable-comp-b", str(args.enable_comp_b)]
        if getattr(args, "enable_comp_c", None) is not None:
            cmd += ["--enable-comp-c", str(args.enable_comp_c)]
        if getattr(args, "mock", False):
            cmd.append("--mock")
        if getattr(args, "test_samples", 0) and args.test_samples > 0:
            cmd += ["--test-samples", str(args.test_samples)]
        elif args.limit and args.limit > 0:
            cmd += ["--limit", str(args.limit)]
        if getattr(args, "lambda_param", None) is not None:
            cmd += ["--lambda-param", str(args.lambda_param)]
        if getattr(args, "beta_param", None) is not None:
            cmd += ["--beta-param", str(args.beta_param)]
        if getattr(args, "mu_param", None) is not None:
            cmd += ["--mu-param", str(args.mu_param)]

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        print(f"[COORDINATOR] Spawning Worker {k}/{num_workers}...")
        p = subprocess.Popen(cmd, env=env)
        procs.append(p)

    failed_workers = []
    for k, p in enumerate(procs):
        ret = p.wait()
        if ret != 0:
            failed_workers.append((k, ret))

    if failed_workers:
        print(f"\n[ERROR] Probe worker(s) failed: {failed_workers}")
        sys.exit(1)

    total_time = time.time() - start_time
    print(f"\n[COORDINATOR] All {num_workers} probe workers finished in {total_time:.2f} s!")

    master_jsonl = os.path.join(args.output_dir, "predictions.jsonl")
    all_records = []
    seen_qids = set()

    with open(master_jsonl, "w", encoding="utf-8") as out_f:
        for k in range(num_workers):
            shard_f = os.path.join(args.output_dir, f"predictions_shard_{k}.jsonl")
            if os.path.exists(shard_f):
                with open(shard_f, "r", encoding="utf-8") as sf:
                    for line in sf:
                        if line.strip():
                            rec = json.loads(line)
                            qid = rec.get("question_id")
                            if qid not in seen_qids:
                                seen_qids.add(qid)
                                all_records.append(rec)
                                out_f.write(line)

    print(f"[COORDINATOR] Aggregated {len(all_records)} unique records into {master_jsonl}")
    return aggregate_and_report_probe(all_records, total_time, args.output_dir, args.ablation_mask)


def main():
    parser = argparse.ArgumentParser(description="Tier 1 Fast Falsification Probe Runner (Calibrated)")
    parser.add_argument("--probe-manifest", dest="probe_manifest", type=str, default=DEFAULT_MANIFEST_PATH, help="Path to probe_100_manifest.json")
    parser.add_argument("--verified-dataset", dest="verified_dataset", type=str, default=DEFAULT_VERIFIED_DATASET, help="Path to answer_bboxes_400_verified.json")
    parser.add_argument("--image-dir", dest="image_dir", type=str, default=DEFAULT_IMAGE_DIR, help="Path to image directory")
    parser.add_argument("--pipeline-type", "--method", dest="pipeline_type", type=str, default="nevfe", choices=["nevfe"], help="Pipeline type")
    parser.add_argument("--model-name", dest="model_name", type=str, default="5CD-AI/Vintern-1B-v3_5", help="Hugging Face model ID")
    parser.add_argument("--device", dest="device", type=str, default="cuda", help="Execution device (cuda or cpu)")
    parser.add_argument("--lambda-param", "--lambda", dest="lambda_param", type=float, default=0.50, help="Visual coupling alpha")
    parser.add_argument("--beta-param", "--beta", dest="beta_param", type=float, default=0.05, help="Plausibility floor beta")
    parser.add_argument("--mu-param", "--mu", dest="mu_param", type=float, default=0.0, help="Query barrier height parameter")
    parser.add_argument("--ablation-mask", dest="ablation_mask", type=str, default="1,1,1", help="Ablation mask e.g. 1,0,1")
    parser.add_argument("--enable-comp-a", dest="enable_comp_a", type=lambda x: (str(x).lower() == 'true'), default=None, help="Explicit toggle for component A")
    parser.add_argument("--enable-comp-b", dest="enable_comp_b", type=lambda x: (str(x).lower() == 'true'), default=None, help="Explicit toggle for component B")
    parser.add_argument("--enable-comp-c", dest="enable_comp_c", type=lambda x: (str(x).lower() == 'true'), default=None, help="Explicit toggle for component C")
    parser.add_argument("--output-dir", dest="output_dir", type=str, default="results/tier1_probe_nevfe", help="Output directory")
    parser.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=64, help="Max new tokens")
    parser.add_argument("--limit", dest="limit", type=int, default=0, help="Limit number of probe questions")
    parser.add_argument("--test-samples", dest="test_samples", type=int, default=0, help="Alias for limit")
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--worker-id", dest="worker_id", type=int, default=0, help="Worker shard ID")
    parser.add_argument("--multi-proc", dest="multi_proc", type=int, default=0, help="Launch N parallel worker processes")
    parser.add_argument("--mock", dest="mock", action="store_true", help="Synthetic mock mode")
    args = parser.parse_args()

    if args.multi_proc and args.multi_proc > 1:
        run_probe_multiprocess_coordinator(args)
    else:
        run_tier1_probe_worker(args)


if __name__ == "__main__":
    main()
