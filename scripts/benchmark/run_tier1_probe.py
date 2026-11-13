#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TIER 1 FAST FALSIFICATION PROBE RUNNER (N=100; 30 cal + 70 held-out)
Multi-Worker Sharded Architecture with 1x model load per worker across --ablation-masks.
"""

import argparse
import datetime
import glob
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
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

from scripts.benchmark.metrics import compute_stage2_metrics
from scripts.benchmark.evaluate_vitextvqa import load_vintern_model
from src.nevfe.core.engine import AlgorithmConfig
from src.nevfe.pipeline import NEVFEPipeline

DEFAULT_MANIFEST_PATH = os.path.join(PROJECT_ROOT, "configs", "probe_100_manifest.json")
DEFAULT_BASELINE_PROBE_JSON = os.path.join(PROJECT_ROOT, "configs", "iter23_baseline_summary.json")
DEFAULT_VERIFIED_DATASET = os.path.join(
    PROJECT_ROOT, "results", "swintextspotter_auto_annot", "answer_bboxes_400_verified.json"
)
DEFAULT_IMAGE_DIR = os.path.join(PROJECT_ROOT, "data", "vitextvqa", "st_images")

CHAMPION_PROBE_SCORES = {
    "token_f1": 44.17,
    "substring_match": 35.00,
    "comp_100": 39.59,
    "comp_70": 37.19,
    "anls": 30.86,
    "exact_match": 10.00,
}

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
    """Parse either compact bits ('101') or comma-separated ('1,0,1') into 3 bool flags."""
    cleaned = str(mask_str).strip()
    if "," in cleaned:
        parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    elif len(cleaned) >= 3 and all(ch in "01" for ch in cleaned):
        parts = list(cleaned)
    else:
        raise ValueError(f"Invalid ablation mask: {mask_str!r}")
    if len(parts) < 3:
        raise ValueError(f"Expected >= 3 ablation mask bits, got {mask_str!r}")
    return bool(int(parts[0])), bool(int(parts[1])), bool(int(parts[2]))


def normalize_mask_bits(mask_str: str) -> str:
    b_a, b_b, b_c = parse_ablation_mask(mask_str)
    return f"{int(b_a)}{int(b_b)}{int(b_c)}"


def load_baseline_probe_scores(baseline_json_path: Optional[str]) -> Dict[str, float]:
    scores = dict(CHAMPION_PROBE_SCORES)
    if baseline_json_path and os.path.isfile(baseline_json_path):
        with open(baseline_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        p70 = data.get("probe_70_heldout", {})
        p100 = data.get("probe_100_full", {})
        if "composite_pct" in p70:
            scores["comp_70"] = float(p70["composite_pct"])
        if "token_f1_pct" in p100:
            scores["token_f1"] = float(p100["token_f1_pct"])
        if "substring_match_pct" in p100:
            scores["substring_match"] = float(p100["substring_match_pct"])
        if "composite_pct" in p100:
            scores["comp_100"] = float(p100["composite_pct"])
    return scores


def _resolve_algorithm_config(args: argparse.Namespace, mask_bits: str) -> AlgorithmConfig:
    defaults = AlgorithmConfig()
    lam = defaults.lambda_param
    beta = defaults.beta_param
    if getattr(args, "config_json", None) and os.path.isfile(args.config_json):
        with open(args.config_json, "r", encoding="utf-8") as cf:
            cfg = json.load(cf)
        if cfg.get("lambda_param") is not None:
            lam = float(cfg["lambda_param"])
        if cfg.get("beta_param") is not None:
            beta = float(cfg["beta_param"])
    if getattr(args, "lambda_param", None) is not None:
        lam = float(args.lambda_param)
    if getattr(args, "beta_param", None) is not None:
        beta = float(args.beta_param)
    b_a, b_b, b_c = parse_ablation_mask(mask_bits)
    if getattr(args, "enable_comp_a", None) is not None:
        b_a = bool(args.enable_comp_a)
    if getattr(args, "enable_comp_b", None) is not None:
        b_b = bool(args.enable_comp_b)
    if getattr(args, "enable_comp_c", None) is not None:
        b_c = bool(args.enable_comp_c)
    return AlgorithmConfig(
        lambda_param=lam,
        beta_param=beta,
        ablation_mask=(int(b_a), int(b_b), int(b_c)),
    )


def _resolve_mask_list(args: argparse.Namespace) -> List[str]:
    if getattr(args, "ablation_masks", None):
        raw = [m.strip() for m in str(args.ablation_masks).split(",") if m.strip()]
        seen: List[str] = []
        for m in raw:
            bits = normalize_mask_bits(m)
            if bits not in seen:
                seen.append(bits)
        if seen:
            return seen
    return [normalize_mask_bits(args.ablation_mask)]


def _preflight_images(manifest_items: List[Dict[str, Any]], candidate_image_dirs: List[str]) -> None:
    missing: List[str] = []
    seen_imgs = set()
    for item in manifest_items:
        img_name = item["image_name"]
        if img_name in seen_imgs:
            continue
        seen_imgs.add(img_name)
        if not resolve_image_path(img_name, candidate_image_dirs):
            missing.append(img_name)
    if missing:
        raise FileNotFoundError(
            f"Preflight failed: {len(missing)}/{len(seen_imgs)} probe images missing: {missing[:5]}"
        )


def _compute_split_metrics(records: List[Dict[str, Any]]) -> Dict[str, float]:
    n = len(records)
    if n == 0:
        return {"f1": 0.0, "sub": 0.0, "comp": 0.0, "anls": 0.0, "em": 0.0}
    f1 = sum(r["candidate_f1"] for r in records) / n * 100.0
    sub = sum(r["candidate_substring"] for r in records) / n * 100.0
    anls = sum(r["candidate_anls"] for r in records) / n * 100.0
    em = sum(r["candidate_em"] for r in records) / n * 100.0
    comp = (f1 + sub) / 2.0
    return {
        "f1": round(f1, 2),
        "sub": round(sub, 2),
        "comp": round(comp, 2),
        "anls": round(anls, 2),
        "em": round(em, 2),
    }


def run_tier1_probe_worker(args: argparse.Namespace) -> Dict[str, Any]:
    os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.isfile(args.probe_manifest):
        raise FileNotFoundError(f"Probe manifest not found at: {args.probe_manifest}")

    with open(args.probe_manifest, "r", encoding="utf-8") as f:
        manifest_items: List[Dict[str, Any]] = json.load(f)

    effective_limit = 0
    if getattr(args, "test_samples", 0) and args.test_samples > 0:
        effective_limit = args.test_samples
    elif getattr(args, "limit", 0) and args.limit > 0:
        effective_limit = args.limit
    if effective_limit > 0:
        manifest_items = manifest_items[:effective_limit]

    candidate_image_dirs = [
        args.image_dir,
        os.path.join(PROJECT_ROOT, "data", "vitextvqa", "st_images"),
        os.path.join(PROJECT_ROOT, "data", "vitextvqa"),
    ]
    is_mock = getattr(args, "mock", False)

    unique_images: List[str] = []
    seen = set()
    for item in manifest_items:
        img_name = item["image_name"]
        if img_name not in seen:
            seen.add(img_name)
            unique_images.append(img_name)

    if args.num_workers > 1:
        sharded_images = {
            img for idx, img in enumerate(unique_images) if idx % args.num_workers == args.worker_id
        }
        manifest_items = [item for item in manifest_items if item["image_name"] in sharded_images]

    if not is_mock:
        _preflight_images(manifest_items, candidate_image_dirs)

    verified_answer_map: Dict[int, List[str]] = {}
    if os.path.isfile(args.verified_dataset):
        with open(args.verified_dataset, "r", encoding="utf-8") as vf:
            v_data = json.load(vf)
        for img_info in v_data.get("images", []):
            for q_info in img_info.get("quests", []):
                qid = q_info.get("question_id")
                ver_ans = q_info.get("verified_all_answers") or q_info.get("gt_answers")
                if qid is not None and ver_ans:
                    verified_answer_map[qid] = ver_ans

    grouped_by_image: Dict[str, List[Dict[str, Any]]] = {}
    for item in manifest_items:
        grouped_by_image.setdefault(item["image_name"], []).append(item)

    mask_list = _resolve_mask_list(args)
    multi_mask_mode = bool(getattr(args, "ablation_masks", None))
    baseline_scores = load_baseline_probe_scores(getattr(args, "baseline_probe_json", None))
    device = args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"

    pipeline = None
    if not is_mock:
        init_cfg = _resolve_algorithm_config(args, mask_list[0])
        model, tokenizer, transform = load_vintern_model(args.model_name, device=device)
        pipeline = NEVFEPipeline(
            model=model,
            tokenizer=tokenizer,
            transform=transform,
            device=device,
            config=init_cfg,
        )
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

    shard_info = f"(Worker {args.worker_id}/{args.num_workers})" if args.num_workers > 1 else ""
    last_summary: Dict[str, Any] = {}

    for mask_bits in mask_list:
        cfg = _resolve_algorithm_config(args, mask_bits)
        b_a, b_b, b_c = (bool(x) for x in cfg.ablation_mask[:3])
        if pipeline is not None and hasattr(pipeline, "set_ablation_switches"):
            pipeline.set_ablation_switches(comp_a=b_a, comp_b=b_b, comp_c=b_c)

        mask_dir = os.path.join(args.output_dir, f"mask_{mask_bits}") if multi_mask_mode else args.output_dir
        os.makedirs(mask_dir, exist_ok=True)
        jsonl_output_path = (
            os.path.join(mask_dir, f"predictions_shard_{args.worker_id}.jsonl")
            if args.num_workers > 1
            else os.path.join(mask_dir, "predictions.jsonl")
        )

        print(
            f"[PROBE {shard_info}] Running mask={mask_bits} N={len(manifest_items)} "
            f"lambda={cfg.lambda_param} beta={cfg.beta_param}"
        )
        results_records: List[Dict[str, Any]] = []
        t_start = time.time()

        with open(jsonl_output_path, "w", encoding="utf-8") as jsonl_file:
            for img_name, q_list in grouped_by_image.items():
                image = None
                if not is_mock:
                    img_path = resolve_image_path(img_name, candidate_image_dirs)
                    if not img_path:
                        raise FileNotFoundError(f"Could not resolve image: {img_name}")
                    image = Image.open(img_path).convert("RGB")
                    with torch.inference_mode():
                        _ = pipeline.extract_visual_features(image)

                for item in q_list:
                    qid = item["question_id"]
                    question = item["question"]
                    error_class = item.get("error_class", "UNKNOWN")
                    gt_answers = item.get("gt_answers", [])
                    verified_answers = verified_answer_map.get(qid) or gt_answers
                    is_cal = bool(item.get("is_calibration", False))

                    t0 = time.time()
                    if is_mock:
                        pred_text = verified_answers[0] if verified_answers else "đáp án"
                    else:
                        with torch.inference_mode():
                            res = pipeline.run(image, question, max_new_tokens=args.max_new_tokens)
                            pred_text = res.pred_text
                    elapsed = time.time() - t0

                    s2 = compute_stage2_metrics(
                        pred_answer=pred_text,
                        gt_answers=gt_answers,
                        verified_answers=verified_answers,
                        question=question,
                    )
                    rec = {
                        "question_id": qid,
                        "image_name": img_name,
                        "error_class": error_class,
                        "is_calibration": is_cal,
                        "question": question,
                        "gt_answers": gt_answers,
                        "verified_all_answers": verified_answers,
                        "candidate_pred": pred_text,
                        "candidate_f1": float(s2["f1_verified"]),
                        "candidate_substring": float(s2["substring_verified"]),
                        "candidate_anls": float(s2["anls_verified"]),
                        "candidate_em": float(s2["em_verified"]),
                        "candidate_vqa_acc": float(s2["vqa_acc_verified"]),
                        "latency_s": round(elapsed, 3),
                    }
                    results_records.append(rec)
                    jsonl_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    jsonl_file.flush()

                    if device == "cuda" and not is_mock:
                        torch.cuda.empty_cache()

        total_time = time.time() - t_start
        peak_vram_gb = (
            round(torch.cuda.max_memory_allocated() / (1024**3), 3)
            if (device == "cuda" and torch.cuda.is_available() and not is_mock)
            else 0.0
        )

        if args.num_workers > 1:
            shard_summary_path = os.path.join(mask_dir, f"summary_shard_{args.worker_id}.json")
            with open(shard_summary_path, "w", encoding="utf-8") as sf:
                json.dump(
                    {
                        "worker_id": args.worker_id,
                        "mask": mask_bits,
                        "count": len(results_records),
                        "peak_vram_gb": peak_vram_gb,
                        "records": results_records,
                    },
                    sf,
                )
            last_summary = {"worker_id": args.worker_id, "count": len(results_records), "peak_vram_gb": peak_vram_gb}
        else:
            last_summary = aggregate_and_report_probe(
                results_records, total_time, mask_dir, mask_bits, baseline_scores, peak_vram_gb
            )

    if multi_mask_mode and args.num_workers <= 1:
        _write_ablation_summary(args.output_dir, mask_list, baseline_scores)
    return last_summary


def aggregate_and_report_probe(
    results_records: List[Dict[str, Any]],
    total_time: float,
    output_dir: str,
    ablation_str: str = "111",
    baseline_scores: Optional[Dict[str, float]] = None,
    peak_vram_gb: float = 0.0,
) -> Dict[str, Any]:
    base = baseline_scores or CHAMPION_PROBE_SCORES
    m100 = _compute_split_metrics(results_records)
    held70_recs = [r for r in results_records if not bool(r.get("is_calibration", False))]
    m70 = _compute_split_metrics(held70_recs) if held70_recs else m100

    classes = ["CLASS_A", "CLASS_B", "CLASS_C", "CLASS_D", "CLASS_E"]
    class_stats: Dict[str, Dict[str, float]] = {}
    for c in classes:
        c_recs = [r for r in results_records if r.get("error_class") == c]
        if c_recs:
            cm = _compute_split_metrics(c_recs)
            champ_c = CHAMPION_CLASS_SCORES.get(c, {})
            class_stats[c] = {
                "count": len(c_recs),
                "candidate_f1": cm["f1"],
                "champion_f1": champ_c.get("f1", 0.0),
                "delta_f1": round(cm["f1"] - champ_c.get("f1", 0.0), 2),
                "candidate_substring": cm["sub"],
                "champion_substring": champ_c.get("substring", 0.0),
                "delta_substring": round(cm["sub"] - champ_c.get("substring", 0.0), 2),
                "candidate_anls": cm["anls"],
                "candidate_em": cm["em"],
            }

    gate_passed = m70["comp"] >= (base["comp_70"] - 0.75)
    summary = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_probe_samples": len(results_records),
        "heldout_70_samples": len(held70_recs),
        "total_runtime_s": round(total_time, 2),
        "avg_latency_s": round(total_time / max(1, len(results_records)), 3),
        "peak_vram_gb": peak_vram_gb,
        "ablation_configuration": ablation_str,
        "heldout_70": {
            "f1_70": m70["f1"],
            "sub_70": m70["sub"],
            "comp_70": m70["comp"],
            "baseline_comp_70": base["comp_70"],
        },
        "overall": {
            "candidate_token_f1": m100["f1"],
            "champion_token_f1": base["token_f1"],
            "delta_token_f1_vs_champion": round(m100["f1"] - base["token_f1"], 2),
            "candidate_substring_match": m100["sub"],
            "champion_substring_match": base["substring_match"],
            "delta_substring_vs_champion": round(m100["sub"] - base["substring_match"], 2),
            "candidate_comp_100": m100["comp"],
            "candidate_anls": m100["anls"],
            "champion_anls": base["anls"],
            "candidate_exact_match": m100["em"],
            "champion_exact_match": base["exact_match"],
        },
        "per_class_diagnostics": class_stats,
        "falsification_gate": {
            "criterion": f"comp_70 >= {base['comp_70']:.2f} - 0.75 ({base['comp_70'] - 0.75:.2f})",
            "achieved_comp_70": m70["comp"],
            "gate_passed": gate_passed,
            "action": "CLEAR FOR TIER 2 FULL RUN" if gate_passed else "IMMEDIATE VETO & GIT ROLLBACK TO REIGNING CHAMPION",
        },
    }

    summary_json_path = os.path.join(output_dir, "summary_metrics.json")
    with open(summary_json_path, "w", encoding="utf-8") as sf:
        json.dump(summary, sf, ensure_ascii=False, indent=2)
    print(
        f"[PROBE {ablation_str}] N={len(results_records)} | 70-Q comp={m70['comp']:.2f}% "
        f"(base {base['comp_70']:.2f}%) | 100-Q F1={m100['f1']:.2f}% Sub={m100['sub']:.2f}% | "
        f"Gate={'PASS' if gate_passed else 'FAIL'}"
    )
    return summary


def _write_ablation_summary(
    output_dir: str,
    mask_list: List[str],
    baseline_scores: Dict[str, float],
    peak_vram_gb: float = 0.0,
) -> Dict[str, Any]:
    masks_out: Dict[str, Dict[str, Any]] = {}
    best_mask = None
    best_comp = -1.0
    for bits in mask_list:
        s_path = os.path.join(output_dir, f"mask_{bits}", "summary_metrics.json")
        if not os.path.isfile(s_path):
            continue
        with open(s_path, "r", encoding="utf-8") as f:
            s = json.load(f)
        h70 = s.get("heldout_70", {})
        ov = s.get("overall", {})
        entry = {
            "f1_70": h70.get("f1_70", 0.0),
            "sub_70": h70.get("sub_70", 0.0),
            "comp_70": h70.get("comp_70", 0.0),
            "f1_100": ov.get("candidate_token_f1", 0.0),
            "sub_100": ov.get("candidate_substring_match", 0.0),
        }
        masks_out[bits] = entry
        peak_vram_gb = max(peak_vram_gb, float(s.get("peak_vram_gb", 0.0)))
        if entry["comp_70"] > best_comp:
            best_comp = entry["comp_70"]
            best_mask = bits

    pass_stage_a = bool(masks_out) and (best_comp >= (baseline_scores["comp_70"] - 0.75))
    ablation_summary = {
        "masks": masks_out,
        "best_mask": best_mask,
        "best_comp_70": round(best_comp, 2) if best_mask else 0.0,
        "baseline_comp_70": baseline_scores["comp_70"],
        "pass_stage_a": pass_stage_a,
        "peak_vram_gb": round(peak_vram_gb, 3),
    }
    with open(os.path.join(output_dir, "probe_ablation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(ablation_summary, f, indent=2, ensure_ascii=False)
    return ablation_summary


def run_probe_multiprocess_coordinator(args: argparse.Namespace) -> Dict[str, Any]:
    num_workers = args.multi_proc
    os.makedirs(args.output_dir, exist_ok=True)
    mask_list = _resolve_mask_list(args)
    multi_mask_mode = bool(getattr(args, "ablation_masks", None))
    baseline_scores = load_baseline_probe_scores(getattr(args, "baseline_probe_json", None))
    start_time = time.time()

    with open(args.probe_manifest, "r", encoding="utf-8") as f:
        manifest_items = json.load(f)
    if getattr(args, "test_samples", 0) and args.test_samples > 0:
        manifest_items = manifest_items[: args.test_samples]
    elif getattr(args, "limit", 0) and args.limit > 0:
        manifest_items = manifest_items[: args.limit]

    if not getattr(args, "mock", False):
        _preflight_images(
            manifest_items,
            [
                args.image_dir,
                os.path.join(PROJECT_ROOT, "data", "vitextvqa", "st_images"),
                os.path.join(PROJECT_ROOT, "data", "vitextvqa"),
            ],
        )

    for bits in mask_list:
        mdir = os.path.join(args.output_dir, f"mask_{bits}") if multi_mask_mode else args.output_dir
        for pattern in ["predictions_shard_*.jsonl*", "summary_shard_*.json*"]:
            for p in glob.glob(os.path.join(mdir, pattern)):
                try:
                    os.remove(p)
                except OSError:
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
            "--baseline-probe-json", str(args.baseline_probe_json),
            "--verified-dataset", str(args.verified_dataset),
            "--image-dir", str(args.image_dir),
            "--output-dir", str(args.output_dir),
            "--ablation-mask", str(args.ablation_mask),
            "--max-new-tokens", str(args.max_new_tokens),
        ]
        if getattr(args, "ablation_masks", None):
            cmd += ["--ablation-masks", str(args.ablation_masks)]
        if getattr(args, "config_json", None):
            cmd += ["--config-json", str(args.config_json)]
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
        elif getattr(args, "limit", 0) and args.limit > 0:
            cmd += ["--limit", str(args.limit)]
        if getattr(args, "lambda_param", None) is not None:
            cmd += ["--lambda-param", str(args.lambda_param)]
        if getattr(args, "beta_param", None) is not None:
            cmd += ["--beta-param", str(args.beta_param)]

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        procs.append(subprocess.Popen(cmd, env=env))

    failed_workers = [(k, p.wait()) for k, p in enumerate(procs)]
    failed_workers = [(k, ret) for k, ret in failed_workers if ret != 0]
    if failed_workers:
        raise RuntimeError(f"Probe worker(s) failed: {failed_workers}")

    total_time = time.time() - start_time
    peak_vram_gb = 0.0
    last_summary: Dict[str, Any] = {}

    for bits in mask_list:
        mdir = os.path.join(args.output_dir, f"mask_{bits}") if multi_mask_mode else args.output_dir
        master_jsonl = os.path.join(mdir, "predictions.jsonl")
        all_records: List[Dict[str, Any]] = []
        seen_qids = set()
        for k in range(num_workers):
            s_json = os.path.join(mdir, f"summary_shard_{k}.json")
            if os.path.isfile(s_json):
                with open(s_json, "r", encoding="utf-8") as sf:
                    peak_vram_gb = max(peak_vram_gb, float(json.load(sf).get("peak_vram_gb", 0.0)))
            shard_f = os.path.join(mdir, f"predictions_shard_{k}.jsonl")
            if os.path.isfile(shard_f):
                with open(shard_f, "r", encoding="utf-8") as pf:
                    for line in pf:
                        if line.strip():
                            rec = json.loads(line)
                            qid = rec.get("question_id")
                            if qid not in seen_qids:
                                seen_qids.add(qid)
                                all_records.append(rec)
        with open(master_jsonl, "w", encoding="utf-8") as out_f:
            for rec in all_records:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        last_summary = aggregate_and_report_probe(
            all_records, total_time, mdir, bits, baseline_scores, peak_vram_gb
        )

    if multi_mask_mode:
        return _write_ablation_summary(args.output_dir, mask_list, baseline_scores, peak_vram_gb)
    return last_summary


def main():
    parser = argparse.ArgumentParser(description="Tier 1 Fast Falsification Probe Runner (Calibrated)")
    parser.add_argument("--probe-manifest", "--manifest", dest="probe_manifest", type=str, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--baseline-probe-json", dest="baseline_probe_json", type=str, default=DEFAULT_BASELINE_PROBE_JSON)
    parser.add_argument("--verified-dataset", dest="verified_dataset", type=str, default=DEFAULT_VERIFIED_DATASET)
    parser.add_argument("--image-dir", dest="image_dir", type=str, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--pipeline-type", "--method", dest="pipeline_type", type=str, default="nevfe", choices=["nevfe"])
    parser.add_argument("--model-name", dest="model_name", type=str, default="5CD-AI/Vintern-1B-v3_5")
    parser.add_argument("--device", dest="device", type=str, default="cuda")
    parser.add_argument("--config-json", dest="config_json", type=str, default=None)
    parser.add_argument("--lambda-param", "--lambda", dest="lambda_param", type=float, default=None)
    parser.add_argument("--beta-param", "--beta", dest="beta_param", type=float, default=None)
    parser.add_argument("--mu-param", "--mu", dest="mu_param", type=float, default=0.0)
    parser.add_argument("--ablation-mask", dest="ablation_mask", type=str, default="1,1,1")
    parser.add_argument("--ablation-masks", dest="ablation_masks", type=str, default="")
    parser.add_argument("--enable-comp-a", dest="enable_comp_a", type=lambda x: (str(x).lower() == "true"), default=None)
    parser.add_argument("--enable-comp-b", dest="enable_comp_b", type=lambda x: (str(x).lower() == "true"), default=None)
    parser.add_argument("--enable-comp-c", dest="enable_comp_c", type=lambda x: (str(x).lower() == "true"), default=None)
    parser.add_argument("--output-dir", dest="output_dir", type=str, default="results/tier1_probe_nevfe")
    parser.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=64)
    parser.add_argument("--limit", dest="limit", type=int, default=0)
    parser.add_argument("--test-samples", dest="test_samples", type=int, default=0)
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=1)
    parser.add_argument("--worker-id", dest="worker_id", type=int, default=0)
    parser.add_argument("--multi-proc", dest="multi_proc", type=int, default=0)
    parser.add_argument("--mock", dest="mock", action="store_true")
    args = parser.parse_args()

    if args.multi_proc and args.multi_proc > 1:
        run_probe_multiprocess_coordinator(args)
    else:
        run_tier1_probe_worker(args)


if __name__ == "__main__":
    main()
