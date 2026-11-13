# -*- coding: utf-8 -*-
"""Export and evaluate the 30-sample local CPU calibration bank."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

DEFAULT_MANIFEST_PATH = os.path.join(PROJECT_ROOT, "configs", "probe_100_manifest.json")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "local_calibration_bank")
MAX_BANK_BYTES = 200 * 1024 * 1024


def load_calibration_items(manifest_path: str) -> List[Dict[str, Any]]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        items = data.get("samples") or data.get("items") or []
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError(f"Unsupported manifest structure in {manifest_path}")

    cal_items = [it for it in items if bool(it.get("is_calibration", False))]
    cal_items.sort(key=lambda it: int(it.get("question_id", it.get("qid", 0))))
    return cal_items


def _is_scalar_finite(val: Any) -> bool:
    if isinstance(val, torch.Tensor):
        return bool(torch.isfinite(val).all().item())
    return bool(torch.isfinite(torch.as_tensor(val, dtype=torch.float32)).all().item())


def run_calibration_eval(engine: Any, bank_dir: str) -> Dict[str, float]:
    """Run CPU-only decoding-step calibration metrics on the saved bank."""
    if not os.path.isdir(bank_dir):
        raise FileNotFoundError(f"Calibration bank directory not found: {bank_dir}")

    sample_files = sorted(
        f for f in os.listdir(bank_dir) if f.endswith(".pt") and f.startswith("qid_")
    )
    if not sample_files:
        raise ValueError(f"No qid_*.pt calibration files found in {bank_dir}")

    t0 = time.perf_counter()
    n_samples = len(sample_files)
    finite_count = 0
    shift_count = 0
    kl_sum = 0.0

    with torch.no_grad():
        for fname in sample_files:
            fpath = os.path.join(bank_dir, fname)
            payload = torch.load(fpath, map_location="cpu", weights_only=True)
            logits_plus = payload["logits_plus_t1"].to(dtype=torch.float32)
            logits_zero = payload["logits_zero_t1"].to(dtype=torch.float32)
            q_mask = payload.get("q_mask")
            if q_mask is not None and isinstance(q_mask, torch.Tensor):
                q_mask = q_mask.to(device="cpu")

            _, steered_logits, alpha_t, entropy_nats = engine.decode_step(
                logits_plus, logits_zero, q_mask=q_mask, step_idx=1
            )
            steered_f = steered_logits.to(dtype=torch.float32)

            is_finite = (
                bool(torch.isfinite(steered_f).all().item())
                and _is_scalar_finite(alpha_t)
                and _is_scalar_finite(entropy_nats)
            )
            if is_finite:
                finite_count += 1
                log_p_steer = F.log_softmax(steered_f, dim=-1)
                log_p_plus = F.log_softmax(logits_plus, dim=-1)
                p_steer = log_p_steer.exp()
                kl_val = float((p_steer * (log_p_steer - log_p_plus)).sum(dim=-1).mean().item())
                kl_sum += max(0.0, kl_val)

            top1_steer = int(steered_f.argmax(dim=-1).reshape(-1)[0].item())
            top1_plus = int(logits_plus.argmax(dim=-1).reshape(-1)[0].item())
            if top1_steer != top1_plus:
                shift_count += 1

    wall_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "num_samples": float(n_samples),
        "finite_ratio": finite_count / float(n_samples),
        "simplex_kl_mean": kl_sum / float(max(finite_count, 1)),
        "top1_prior_shift_rate": shift_count / float(n_samples),
        "wall_ms": wall_ms,
    }


def _finalize_bank_manifest(output_dir: str, entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_bytes = sum(int(e["size_bytes"]) for e in entries)
    if total_bytes > MAX_BANK_BYTES:
        raise RuntimeError(
            f"Calibration bank size {total_bytes / (1024 * 1024):.2f} MB exceeds 200 MB limit."
        )
    summary = {
        "num_samples": len(entries),
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / (1024 * 1024), 4),
        "entries": entries,
    }
    manifest_out = os.path.join(output_dir, "manifest.json")
    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def export_synthetic_calibration_bank(
    manifest_path: str,
    output_dir: str,
    vocab_size: int = 1024,
    num_tiles: int = 4,
    seq_len: int = 32,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate deterministic CPU-safe float16 tensors for the 30 calibration QIDs."""
    cal_items = load_calibration_items(manifest_path)
    if not cal_items:
        raise ValueError(f"No is_calibration=true items found in {manifest_path}")

    os.makedirs(output_dir, exist_ok=True)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    entries: List[Dict[str, Any]] = []

    for item in cal_items:
        qid = int(item.get("question_id", item.get("qid")))
        class_tag = str(item.get("error_class", item.get("class_tag", "CLASS_A")))
        logits_plus = (torch.randn((1, vocab_size), generator=gen, dtype=torch.float32) * 2.0).to(torch.float16)
        logits_zero = (torch.randn((1, vocab_size), generator=gen, dtype=torch.float32) * 2.0).to(torch.float16)
        vit_norms = torch.rand((num_tiles, 256), generator=gen, dtype=torch.float32).add(0.5).to(torch.float16)
        attn_sum = torch.rand((24, 14), generator=gen, dtype=torch.float32).to(torch.float16)
        q_mask = torch.zeros((vocab_size,), dtype=torch.bool)
        q_mask[2: min(16, vocab_size)] = True

        payload = {
            "qid": qid,
            "class_tag": class_tag,
            "logits_plus_t1": logits_plus,
            "logits_zero_t1": logits_zero,
            "vit_embed_norms": vit_norms,
            "sample_attn_summary": attn_sum,
            "q_mask": q_mask,
        }
        fname = f"qid_{qid}.pt"
        fpath = os.path.join(output_dir, fname)
        torch.save(payload, fpath)
        entries.append(
            {
                "qid": qid,
                "class_tag": class_tag,
                "image_name": str(item.get("image_name", item.get("image", ""))),
                "file": fname,
                "size_bytes": os.path.getsize(fpath),
                "vocab_size": vocab_size,
            }
        )

    return _finalize_bank_manifest(output_dir, entries)


def _summarize_attentions(attentions: Optional[Tuple[torch.Tensor, ...]]) -> torch.Tensor:
    """Pool decoder layer/head attentions into a compact [24, 14] float16 tensor."""
    if not attentions:
        return torch.zeros((24, 14), dtype=torch.float16)
    rows: List[torch.Tensor] = []
    for layer_attn in attentions[:24]:
        # layer_attn: [B, H, Q, K] -> take last query token across first 14 heads
        last_q = layer_attn[0, :, -1, :].mean(dim=-1).to(dtype=torch.float32).cpu()
        if last_q.numel() < 14:
            padded = torch.zeros((14,), dtype=torch.float32)
            padded[: last_q.numel()] = last_q
            last_q = padded
        else:
            last_q = last_q[:14]
        rows.append(last_q)
    while len(rows) < 24:
        rows.append(torch.zeros((14,), dtype=torch.float32))
    return torch.stack(rows, dim=0).to(dtype=torch.float16)


def export_live_calibration_bank(
    manifest_path: str,
    image_dir: str,
    output_dir: str,
    model_name: str = "5CD-AI/Vintern-1B-v3_5",
    device: str = "cuda",
) -> Dict[str, Any]:
    """Extract real step-1 baseline tensors from Vintern on GPU for the 30 calibration QIDs."""
    from PIL import Image
    from scripts.benchmark.evaluate_vitextvqa import load_vintern_model
    from src.nevfe.adapter.vintern_adapter import VinternNEVFEAdapter, build_query_token_mask

    cal_items = load_calibration_items(manifest_path)
    if not cal_items:
        raise ValueError(f"No is_calibration=true items found in {manifest_path}")

    model, tokenizer, transform = load_vintern_model(model_name, device=device)
    model.eval()
    adapter = VinternNEVFEAdapter(model=model, tokenizer=tokenizer, transform=transform, device=device)

    os.makedirs(output_dir, exist_ok=True)
    entries: List[Dict[str, Any]] = []

    with torch.no_grad():
        for item in cal_items:
            qid = int(item.get("question_id", item.get("qid")))
            img_name = str(item.get("image_name", item.get("image")))
            class_tag = str(item.get("error_class", item.get("class_tag", "CLASS_A")))
            question = str(item.get("question", ""))
            img_path = os.path.join(image_dir, img_name)
            if not os.path.isfile(img_path):
                raise FileNotFoundError(f"Calibration image missing: {img_path}")

            pil_img = Image.open(img_path).convert("RGB")
            vit_embeds = adapter.extract_visual_features(pil_img)
            vit_norms = vit_embeds.norm(dim=-1).to(dtype=torch.float16).cpu()

            embeds_plus = adapter.build_multimodal_embeds(vit_embeds, question)
            embeds_zero = adapter.build_mfvr_embeds(vit_embeds, question)
            attn_mask = torch.ones((1, embeds_plus.shape[1]), dtype=torch.long, device=embeds_plus.device)

            out_plus = model.language_model(
                inputs_embeds=embeds_plus,
                attention_mask=attn_mask,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )
            out_zero = model.language_model(
                inputs_embeds=embeds_zero,
                attention_mask=attn_mask,
                output_attentions=False,
                use_cache=False,
                return_dict=True,
            )
            logits_plus = out_plus.logits[:, -1, :].to(dtype=torch.float16).cpu()
            logits_zero = out_zero.logits[:, -1, :].to(dtype=torch.float16).cpu()
            attn_sum = _summarize_attentions(getattr(out_plus, "attentions", None))
            q_mask = build_query_token_mask(
                question, tokenizer, vocab_size=int(logits_plus.shape[-1]), device="cpu"
            )

            payload = {
                "qid": qid,
                "class_tag": class_tag,
                "logits_plus_t1": logits_plus,
                "logits_zero_t1": logits_zero,
                "vit_embed_norms": vit_norms,
                "sample_attn_summary": attn_sum,
                "q_mask": q_mask,
            }
            fname = f"qid_{qid}.pt"
            fpath = os.path.join(output_dir, fname)
            torch.save(payload, fpath)
            entries.append(
                {
                    "qid": qid,
                    "class_tag": class_tag,
                    "image_name": img_name,
                    "file": fname,
                    "size_bytes": os.path.getsize(fpath),
                    "vocab_size": int(logits_plus.shape[-1]),
                }
            )

    return _finalize_bank_manifest(output_dir, entries)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Export and evaluate the 30-sample local calibration bank.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--image-dir", default=os.path.join(PROJECT_ROOT, "data", "vitextvqa", "st_images"))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default="5CD-AI/Vintern-1B-v3_5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mock", action="store_true", help="Export deterministic synthetic tensors offline.")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate an existing calibration bank.")
    args = parser.parse_args(argv)

    if not args.eval_only:
        if args.mock:
            summary = export_synthetic_calibration_bank(args.manifest, args.output_dir)
        else:
            summary = export_live_calibration_bank(
                manifest_path=args.manifest,
                image_dir=args.image_dir,
                output_dir=args.output_dir,
                model_name=args.model_name,
                device=args.device,
            )
        print(
            f"[CAL_BANK] Exported {summary['num_samples']} samples "
            f"({summary['total_mb']:.2f} MB) -> {args.output_dir}"
        )

    from nevfe.core.engine import AlgorithmConfig, NEVFEEngine

    engine = NEVFEEngine(config=AlgorithmConfig())
    metrics = run_calibration_eval(engine, args.output_dir)
    print(f"[CAL_BANK] CPU eval: {json.dumps(metrics)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
