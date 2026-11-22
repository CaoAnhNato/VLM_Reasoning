# -*- coding: utf-8 -*-
"""Export, validate, and evaluate the 30-sample local CPU calibration bank (v1 & Factored v2)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
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
DEFAULT_OUTPUT_DIR_V2 = os.path.join(PROJECT_ROOT, "data", "calibration_bank_v2")
MAX_BANK_BYTES = 200 * 1024 * 1024
TOPK_UNION_CAP = 2048
TOPK_PER_STREAM = 1024
ACTIVE_VOCAB_CAP = 256


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_git_head(repo_root: str = PROJECT_ROOT) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", repo_root, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    except Exception:
        pass
    return "unknown"


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


def _rms_norm_project(
    hidden: torch.Tensor,
    norm_weight: torch.Tensor,
    rms_norm_eps: float,
    lm_head_slice: torch.Tensor,
) -> torch.Tensor:
    """Apply final RMSNorm and project onto the QID active vocabulary slice on CPU."""
    h32 = hidden.to(dtype=torch.float32)
    w32 = norm_weight.to(dtype=torch.float32)
    var = (h32 * h32).mean(dim=-1, keepdim=True)
    normed = h32 * torch.rsqrt(var + float(rms_norm_eps)) * w32
    return torch.matmul(normed, lm_head_slice.to(dtype=torch.float32).transpose(0, 1))


def run_calibration_eval(engine: Any, bank_dir: str) -> Dict[str, float]:
    """Run CPU-only step-1 calibration metrics on a v1 calibration bank."""
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


def validate_cal_bank_v2(
    bank_dir: str,
    expected_base_sha: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate Calibration Bank v2 schema, size cap (< 200 MB), KV invariance, and SHA-256 digests."""
    manifest_path = os.path.join(bank_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Calibration Bank v2 manifest missing: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if str(manifest.get("schema_version")) != "2.0":
        raise ValueError(f"Unsupported Calibration Bank schema_version: {manifest.get('schema_version')!r}")
    if expected_base_sha and manifest.get("base_commit_sha") != expected_base_sha:
        raise ValueError(
            f"Calibration Bank v2 base_commit_sha mismatch: expected {expected_base_sha}, got {manifest.get('base_commit_sha')}"
        )
    max_kv_diff = float(manifest.get("max_kv_diff", -1.0))
    if max_kv_diff != 0.0:
        raise ValueError(f"Calibration Bank v2 max_kv_diff must be 0.0, got {max_kv_diff}")

    files_map = manifest.get("files")
    if not isinstance(files_map, dict) or "shared_readout.pt" not in files_map:
        raise ValueError("Calibration Bank v2 manifest missing 'shared_readout.pt' entry in 'files'")
    qid_keys = sorted(k for k in files_map if k.startswith("qid_") and k.endswith(".pt"))
    if not qid_keys:
        raise ValueError("Calibration Bank v2 manifest contains no qid_*.pt entries")

    computed_bytes = 0
    for rel_name, meta in files_map.items():
        fpath = os.path.join(bank_dir, rel_name)
        if not os.path.isfile(fpath):
            raise FileNotFoundError(f"Calibration Bank v2 file missing: {fpath}")
        actual_size = os.path.getsize(fpath)
        computed_bytes += actual_size
        if int(meta.get("size_bytes", -1)) != actual_size:
            raise ValueError(
                f"Size mismatch for {rel_name}: manifest={meta.get('size_bytes')} actual={actual_size}"
            )
        actual_sha = _sha256_file(fpath)
        expected_sha = str(meta.get("sha256", "")).lower()
        if actual_sha.lower() != expected_sha:
            raise ValueError(
                f"SHA-256 mismatch for {rel_name}: manifest={expected_sha} actual={actual_sha}"
            )

    if computed_bytes > MAX_BANK_BYTES:
        raise RuntimeError(
            f"Calibration Bank v2 size {computed_bytes / (1024 * 1024):.2f} MB exceeds 200 MB cap"
        )
    return {
        "valid": True,
        "schema_version": "2.0",
        "num_samples": len(qid_keys),
        "total_bytes": computed_bytes,
        "total_mb": round(computed_bytes / (1024 * 1024), 4),
        "max_kv_diff": max_kv_diff,
        "base_commit_sha": manifest.get("base_commit_sha"),
    }


def run_calibration_eval_v2(
    engine: Any,
    bank_dir: str,
    track: int = 1,
) -> Dict[str, Any]:
    """
    Standalone CPU Calibration Bank v2 evaluator living outside src/nevfe/.
    Calls only unmodified NEVFEEngine.decode_step() across t=1..T trajectories.
    """
    bank_meta = validate_cal_bank_v2(bank_dir)
    readout = torch.load(os.path.join(bank_dir, "shared_readout.pt"), map_location="cpu", weights_only=True)
    norm_weight = readout["norm_weight"]
    rms_norm_eps = float(readout["rms_norm_eps"])

    sample_files = sorted(
        f for f in os.listdir(bank_dir) if f.endswith(".pt") and f.startswith("qid_")
    )
    t0 = time.perf_counter()
    total_steps = 0
    finite_steps = 0
    nan_count = 0
    inf_count = 0
    shift_count = 0
    kl_sum = 0.0

    with torch.no_grad():
        for fname in sample_files:
            payload = torch.load(os.path.join(bank_dir, fname), map_location="cpu", weights_only=True)
            lp_t1 = payload["logits_plus_t1"].to(dtype=torch.float32)
            lz_t1 = payload["logits_zero_t1"].to(dtype=torch.float32)
            vocab_size = int(lp_t1.shape[-1])
            q_mask = payload.get("q_mask")
            if isinstance(q_mask, torch.Tensor):
                q_mask = q_mask.to(device="cpu")

            topk_ids = payload["step_topk_ids"].to(dtype=torch.long)
            topk_plus = payload["step_topk_plus"].to(dtype=torch.float32)
            topk_zero = payload["step_topk_zero"].to(dtype=torch.float32)
            invariants = payload["step_simplex_invariants"].to(dtype=torch.float32)
            n_steps = int(invariants.shape[0])

            lm_head_slice = payload["lm_head_slice"]
            active_vocab_ids = payload["active_vocab_ids"]
            h_in_p = payload["h_in_upper_plus"]
            h_in_z = payload["h_in_upper_zero"]
            h_attn_p = payload["h_post_attn_upper_plus"]
            h_attn_z = payload["h_post_attn_upper_zero"]
            h_ffn_p = payload["h_post_ffn_upper_plus"]
            h_ffn_z = payload["h_post_ffn_upper_zero"]

            history_steps: List[torch.Tensor] = []

            for step_idx in range(1, n_steps + 1):
                if step_idx == 1:
                    lp = lp_t1.unsqueeze(0)
                    lz = lz_t1.unsqueeze(0)
                else:
                    row_idx = step_idx - 2
                    ids_row = topk_ids[row_idx]
                    valid_mask = ids_row >= 0
                    ids_valid = ids_row[valid_mask]
                    lp = torch.full((1, vocab_size), -1e4, dtype=torch.float32)
                    lz = torch.full((1, vocab_size), -1e4, dtype=torch.float32)
                    if ids_valid.numel() > 0:
                        lp[0, ids_valid] = topk_plus[row_idx][valid_mask]
                        lz[0, ids_valid] = topk_zero[row_idx][valid_mask]

                history_steps.append(lp.squeeze(0))
                step_context: Dict[str, Any] = {
                    "history_logits": torch.stack(history_steps, dim=0),
                    "step_simplex_invariants": invariants[:step_idx],
                }
                if int(track) == 2:
                    t_i = step_idx - 1
                    step_context["upper_hidden_states"] = {
                        "active_vocab_ids": active_vocab_ids,
                        "logits_in_plus": _rms_norm_project(h_in_p[t_i], norm_weight, rms_norm_eps, lm_head_slice),
                        "logits_in_zero": _rms_norm_project(h_in_z[t_i], norm_weight, rms_norm_eps, lm_head_slice),
                        "logits_post_attn_plus": _rms_norm_project(h_attn_p[t_i], norm_weight, rms_norm_eps, lm_head_slice),
                        "logits_post_attn_zero": _rms_norm_project(h_attn_z[t_i], norm_weight, rms_norm_eps, lm_head_slice),
                        "logits_post_ffn_plus": _rms_norm_project(h_ffn_p[t_i], norm_weight, rms_norm_eps, lm_head_slice),
                        "logits_post_ffn_zero": _rms_norm_project(h_ffn_z[t_i], norm_weight, rms_norm_eps, lm_head_slice),
                    }
                else:
                    step_context["upper_hidden_states"] = None

                _, steered_logits, alpha_t, entropy_nats = engine.decode_step(
                    lp,
                    lz,
                    q_mask=q_mask,
                    step_idx=step_idx,
                    step_context=step_context,
                )
                steered_f = steered_logits.to(dtype=torch.float32)
                total_steps += 1
                nan_count += int(torch.isnan(steered_f).sum().item())
                inf_count += int(torch.isinf(steered_f).sum().item())

                is_finite = (
                    bool(torch.isfinite(steered_f).all().item())
                    and _is_scalar_finite(alpha_t)
                    and _is_scalar_finite(entropy_nats)
                )
                if is_finite:
                    finite_steps += 1
                    log_p_steer = F.log_softmax(steered_f, dim=-1)
                    log_p_plus = F.log_softmax(lp, dim=-1)
                    p_steer = log_p_steer.exp()
                    kl_val = float((p_steer * (log_p_steer - log_p_plus)).sum(dim=-1).mean().item())
                    kl_sum += max(0.0, kl_val)

                top1_steer = int(steered_f.argmax(dim=-1).reshape(-1)[0].item())
                top1_plus = int(lp.argmax(dim=-1).reshape(-1)[0].item())
                if top1_steer != top1_plus:
                    shift_count += 1

    wall_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "schema_version": "2.0",
        "track": int(track),
        "num_samples": float(len(sample_files)),
        "total_steps": int(total_steps),
        "finite_ratio": finite_steps / float(max(1, total_steps)),
        "nan_count": int(nan_count),
        "inf_count": int(inf_count),
        "simplex_kl_mean": kl_sum / float(max(finite_steps, 1)),
        "top1_prior_shift_rate": shift_count / float(max(1, total_steps)),
        "max_kv_diff": float(bank_meta["max_kv_diff"]),
        "total_mb": float(bank_meta["total_mb"]),
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
    with open(manifest_out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def _finalize_bank_v2_manifest(
    output_dir: str,
    file_records: Dict[str, Dict[str, Any]],
    base_commit_sha: str,
    backbone_id: str,
    num_hidden_layers: int,
    max_steps: int,
    max_kv_diff: float,
) -> Dict[str, Any]:
    total_bytes = sum(int(meta["size_bytes"]) for meta in file_records.values())
    if total_bytes > MAX_BANK_BYTES:
        raise RuntimeError(
            f"Calibration Bank v2 size {total_bytes / (1024 * 1024):.2f} MB exceeds 200 MB limit."
        )
    qid_count = sum(1 for k in file_records if k.startswith("qid_") and k.endswith(".pt"))
    summary = {
        "schema_version": "2.0",
        "base_commit_sha": base_commit_sha,
        "backbone_id": backbone_id,
        "num_hidden_layers": int(num_hidden_layers),
        "max_steps": int(max_steps),
        "max_kv_diff": float(max_kv_diff),
        "num_samples": int(qid_count),
        "total_bytes": int(total_bytes),
        "total_mb": round(total_bytes / (1024 * 1024), 4),
        "files": file_records,
    }
    manifest_out = os.path.join(output_dir, "manifest.json")
    with open(manifest_out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def _compute_simplex_invariant_row(lp_1d: torch.Tensor, lz_1d: torch.Tensor) -> torch.Tensor:
    """Compute [lse_plus, lse_zero, H_plus, BC, p_max_plus, greedy_id] in float32."""
    lp32 = lp_1d.to(dtype=torch.float32)
    lz32 = lz_1d.to(dtype=torch.float32)
    lse_p = torch.logsumexp(lp32, dim=-1)
    lse_z = torch.logsumexp(lz32, dim=-1)
    log_p = lp32 - lse_p
    p_p = torch.exp(log_p)
    p_z = F.softmax(lz32, dim=-1)
    h_p = -torch.sum(torch.nan_to_num(p_p * log_p, nan=0.0, posinf=0.0, neginf=0.0))
    bc = torch.clamp(torch.sum(torch.sqrt(torch.clamp(p_p * p_z, min=0.0))), min=0.0, max=1.0)
    p_max, greedy = torch.max(p_p, dim=-1)
    return torch.tensor(
        [float(lse_p.item()), float(lse_z.item()), float(h_p.item()), float(bc.item()), float(p_max.item()), float(greedy.item())],
        dtype=torch.float32,
    )


def _build_topk_union_row(lp_1d: torch.Tensor, lz_1d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build padded [2048] deduplicated union of top-1024 plus and top-1024 zero logits."""
    v_dim = int(lp_1d.shape[-1])
    k_each = min(TOPK_PER_STREAM, v_dim)
    top_p = torch.topk(lp_1d.float(), k=k_each, dim=-1).indices.tolist()
    top_z = torch.topk(lz_1d.float(), k=k_each, dim=-1).indices.tolist()
    merged: List[int] = []
    seen = set()
    for tid in top_p + top_z:
        if tid not in seen:
            seen.add(tid)
            merged.append(int(tid))
        if len(merged) >= TOPK_UNION_CAP:
            break
    ids_out = torch.full((TOPK_UNION_CAP,), -1, dtype=torch.int32)
    vals_p = torch.full((TOPK_UNION_CAP,), -1e4, dtype=torch.bfloat16)
    vals_z = torch.full((TOPK_UNION_CAP,), -1e4, dtype=torch.bfloat16)
    if merged:
        idx_t = torch.tensor(merged, dtype=torch.long, device=lp_1d.device)
        n_m = len(merged)
        ids_out[:n_m] = idx_t.to(dtype=torch.int32).cpu()
        vals_p[:n_m] = lp_1d[idx_t].to(dtype=torch.bfloat16).cpu()
        vals_z[:n_m] = lz_1d[idx_t].to(dtype=torch.bfloat16).cpu()
    return ids_out, vals_p, vals_z


def export_synthetic_calibration_bank(
    manifest_path: str,
    output_dir: str,
    vocab_size: int = 1024,
    num_tiles: int = 4,
    seq_len: int = 32,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate deterministic CPU-safe float16 tensors for the 30 calibration QIDs (v1)."""
    del seq_len
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


def export_synthetic_calibration_bank_v2(
    manifest_path: str,
    output_dir: str,
    max_steps: int = 16,
    vocab_size: int = 4096,
    hidden_size: int = 64,
    num_hidden_layers: int = 24,
    seed: int = 42,
    base_commit_sha: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate deterministic CPU-safe Factored Calibration Bank v2 tensors (< 200 MB)."""
    cal_items = load_calibration_items(manifest_path)
    if not cal_items:
        raise ValueError(f"No is_calibration=true items found in {manifest_path}")

    os.makedirs(output_dir, exist_ok=True)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    upper_layers = num_hidden_layers // 2
    file_records: Dict[str, Dict[str, Any]] = {}

    readout_path = os.path.join(output_dir, "shared_readout.pt")
    torch.save(
        {
            "norm_weight": torch.ones((hidden_size,), dtype=torch.bfloat16),
            "rms_norm_eps": 1e-6,
        },
        readout_path,
    )
    file_records["shared_readout.pt"] = {
        "size_bytes": os.path.getsize(readout_path),
        "sha256": _sha256_file(readout_path),
    }

    s_q = min(64, vocab_size, ACTIVE_VOCAB_CAP)
    for item in cal_items:
        qid = int(item.get("question_id", item.get("qid")))
        class_tag = str(item.get("error_class", item.get("class_tag", "CLASS_A")))
        all_lp = (torch.randn((max_steps, vocab_size), generator=gen, dtype=torch.float32) * 2.0)
        all_lz = (torch.randn((max_steps, vocab_size), generator=gen, dtype=torch.float32) * 2.0)

        invariants = torch.stack(
            [_compute_simplex_invariant_row(all_lp[t], all_lz[t]) for t in range(max_steps)],
            dim=0,
        )
        topk_ids_list: List[torch.Tensor] = []
        topk_p_list: List[torch.Tensor] = []
        topk_z_list: List[torch.Tensor] = []
        for t in range(1, max_steps):
            ids_row, vp_row, vz_row = _build_topk_union_row(all_lp[t], all_lz[t])
            topk_ids_list.append(ids_row)
            topk_p_list.append(vp_row)
            topk_z_list.append(vz_row)

        q_mask = torch.zeros((vocab_size,), dtype=torch.bool)
        q_mask[2: min(16, vocab_size)] = True
        active_ids = torch.arange(s_q, dtype=torch.int32)
        lm_head_slice = (torch.randn((s_q, hidden_size), generator=gen, dtype=torch.float32) * 0.1).to(torch.bfloat16)

        payload = {
            "qid": qid,
            "class_tag": class_tag,
            "logits_plus_t1": all_lp[0].to(torch.bfloat16),
            "logits_zero_t1": all_lz[0].to(torch.bfloat16),
            "step_topk_ids": torch.stack(topk_ids_list, dim=0) if topk_ids_list else torch.empty((0, TOPK_UNION_CAP), dtype=torch.int32),
            "step_topk_plus": torch.stack(topk_p_list, dim=0) if topk_p_list else torch.empty((0, TOPK_UNION_CAP), dtype=torch.bfloat16),
            "step_topk_zero": torch.stack(topk_z_list, dim=0) if topk_z_list else torch.empty((0, TOPK_UNION_CAP), dtype=torch.bfloat16),
            "step_simplex_invariants": invariants,
            "h_in_upper_plus": torch.randn((max_steps, 1, hidden_size), generator=gen, dtype=torch.float32).to(torch.bfloat16),
            "h_in_upper_zero": torch.randn((max_steps, 1, hidden_size), generator=gen, dtype=torch.float32).to(torch.bfloat16),
            "h_post_attn_upper_plus": torch.randn((max_steps, upper_layers, hidden_size), generator=gen, dtype=torch.float32).to(torch.bfloat16),
            "h_post_attn_upper_zero": torch.randn((max_steps, upper_layers, hidden_size), generator=gen, dtype=torch.float32).to(torch.bfloat16),
            "h_post_ffn_upper_plus": torch.randn((max_steps, upper_layers, hidden_size), generator=gen, dtype=torch.float32).to(torch.bfloat16),
            "h_post_ffn_upper_zero": torch.randn((max_steps, upper_layers, hidden_size), generator=gen, dtype=torch.float32).to(torch.bfloat16),
            "active_vocab_ids": active_ids,
            "lm_head_slice": lm_head_slice,
            "q_mask": q_mask,
        }
        fname = f"qid_{qid}.pt"
        fpath = os.path.join(output_dir, fname)
        torch.save(payload, fpath)
        file_records[fname] = {
            "qid": qid,
            "class_tag": class_tag,
            "image_name": str(item.get("image_name", item.get("image", ""))),
            "size_bytes": os.path.getsize(fpath),
            "sha256": _sha256_file(fpath),
            "vocab_size": vocab_size,
        }

    return _finalize_bank_v2_manifest(
        output_dir=output_dir,
        file_records=file_records,
        base_commit_sha=base_commit_sha or _resolve_git_head(),
        backbone_id="mock-vintern-v2",
        num_hidden_layers=num_hidden_layers,
        max_steps=max_steps,
        max_kv_diff=0.0,
    )


def _summarize_attentions(attentions: Optional[Tuple[torch.Tensor, ...]]) -> torch.Tensor:
    """Pool decoder layer/head attentions into a compact [24, 14] float16 tensor."""
    if not attentions:
        return torch.zeros((24, 14), dtype=torch.float16)
    rows: List[torch.Tensor] = []
    for layer_attn in attentions[:24]:
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
    """Extract real step-1 baseline tensors from Vintern on GPU for the 30 calibration QIDs (v1)."""
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


def export_live_calibration_bank_v2(
    manifest_path: str,
    image_dir: str,
    output_dir: str,
    model_name: str = "5CD-AI/Vintern-1B-v3_5",
    device: str = "cuda",
    max_steps: int = 16,
    base_commit_sha: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Export Factored Calibration Bank v2 on GPU under frozen FRSA-PBCF ROI zoom,
    enforcing max_kv_diff == 0.0 against unhooked reference passes.
    """
    from PIL import Image
    from scripts.benchmark.evaluate_vitextvqa import load_vintern_model
    from src.nevfe.adapter.vintern_adapter import (
        VinternNEVFEAdapter,
        build_query_token_mask,
        capture_upper_decoder_states,
        extract_kv_pairs,
    )

    cal_items = load_calibration_items(manifest_path)
    if not cal_items:
        raise ValueError(f"No is_calibration=true items found in {manifest_path}")

    model, tokenizer, transform = load_vintern_model(model_name, device=device)
    model.eval()
    adapter = VinternNEVFEAdapter(model=model, tokenizer=tokenizer, transform=transform, device=device)
    lm_core = model.language_model.model
    lm_head = model.language_model.lm_head
    embed_layer = model.language_model.get_input_embeddings()
    lm_norm = getattr(lm_core, "norm")
    norm_weight = lm_norm.weight.detach().to(dtype=torch.bfloat16).cpu()
    rms_norm_eps = float(getattr(lm_norm, "variance_epsilon", getattr(model.language_model.config, "rms_norm_eps", 1e-6)))
    num_layers = int(len(lm_core.layers))
    eos_ids: List[int] = []
    for eid in (tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.eos_token_id):
        if eid is not None and int(eid) >= 0 and int(eid) not in eos_ids:
            eos_ids.append(int(eid))

    os.makedirs(output_dir, exist_ok=True)
    file_records: Dict[str, Dict[str, Any]] = {}
    readout_path = os.path.join(output_dir, "shared_readout.pt")
    torch.save({"norm_weight": norm_weight, "rms_norm_eps": rms_norm_eps}, readout_path)
    file_records["shared_readout.pt"] = {
        "size_bytes": os.path.getsize(readout_path),
        "sha256": _sha256_file(readout_path),
    }

    max_kv_diff = 0.0
    lm_head_weight = lm_head.weight.detach()

    with torch.no_grad():
        for idx, item in enumerate(cal_items):
            qid = int(item.get("question_id", item.get("qid")))
            img_name = str(item.get("image_name", item.get("image")))
            class_tag = str(item.get("error_class", item.get("class_tag", "CLASS_A")))
            question = str(item.get("question", ""))
            gt_answers = item.get("verified_all_answers") or item.get("gt_answers") or []
            img_path = os.path.join(image_dir, img_name)
            if not os.path.isfile(img_path):
                raise FileNotFoundError(f"Calibration image missing: {img_path}")

            pil_img = Image.open(img_path).convert("RGB")
            vit_embeds = adapter.extract_visual_features(pil_img)
            extra_vit = adapter.extract_foveated_roi_crops(pil_img, vit_embeds, question)
            embeds_plus = adapter.build_multimodal_embeds(vit_embeds, question, extra_vit_embeds=extra_vit)
            embeds_zero = adapter.build_mfvr_embeds(vit_embeds, question, extra_vit_embeds=extra_vit)

            # Verify KV invariance against unhooked reference pass on calibration stream
            if idx == 0:
                ref_p = lm_core(inputs_embeds=embeds_plus, use_cache=True, output_attentions=False, return_dict=True)
                ref_z = lm_core(inputs_embeds=embeds_zero, use_cache=True, output_attentions=False, return_dict=True)
                ref_kv_p = extract_kv_pairs(ref_p.past_key_values)
                ref_kv_z = extract_kv_pairs(ref_z.past_key_values)
            else:
                ref_kv_p = None
                ref_kv_z = None

            logits_plus_t1: Optional[torch.Tensor] = None
            logits_zero_t1: Optional[torch.Tensor] = None
            topk_ids_rows: List[torch.Tensor] = []
            topk_p_rows: List[torch.Tensor] = []
            topk_z_rows: List[torch.Tensor] = []
            invariant_rows: List[torch.Tensor] = []
            h_in_p_rows: List[torch.Tensor] = []
            h_in_z_rows: List[torch.Tensor] = []
            h_attn_p_rows: List[torch.Tensor] = []
            h_attn_z_rows: List[torch.Tensor] = []
            h_ffn_p_rows: List[torch.Tensor] = []
            h_ffn_z_rows: List[torch.Tensor] = []
            active_set: List[int] = []
            active_seen = set()

            def _add_active(tok_ids: List[int]) -> None:
                for tid in tok_ids:
                    if 0 <= int(tid) and int(tid) not in active_seen and len(active_set) < ACTIVE_VOCAB_CAP:
                        active_seen.add(int(tid))
                        active_set.append(int(tid))

            with capture_upper_decoder_states(model, fraction=(0.5, 1.0)) as cap:
                upper_indices = cap["upper_indices"]
                cap["clear"]()
                out_p = lm_core(inputs_embeds=embeds_plus, use_cache=True, output_attentions=False, return_dict=True)
                h_in_p_rows.append(cap["h_in_upper"].squeeze(0).to(dtype=torch.bfloat16).cpu())
                h_attn_p_rows.append(
                    torch.cat([cap["h_post_attn_upper"][li] for li in upper_indices], dim=1).squeeze(0).to(dtype=torch.bfloat16).cpu()
                )
                h_ffn_p_rows.append(
                    torch.cat([cap["h_post_ffn_upper"][li] for li in upper_indices], dim=1).squeeze(0).to(dtype=torch.bfloat16).cpu()
                )
                if ref_kv_p is not None:
                    for (k_h, v_h), (k_r, v_r) in zip(extract_kv_pairs(out_p.past_key_values), ref_kv_p):
                        max_kv_diff = max(
                            max_kv_diff,
                            float((k_h - k_r).abs().max().item()),
                            float((v_h - v_r).abs().max().item()),
                        )

                cap["clear"]()
                out_z = lm_core(inputs_embeds=embeds_zero, use_cache=True, output_attentions=False, return_dict=True)
                h_in_z_rows.append(cap["h_in_upper"].squeeze(0).to(dtype=torch.bfloat16).cpu())
                h_attn_z_rows.append(
                    torch.cat([cap["h_post_attn_upper"][li] for li in upper_indices], dim=1).squeeze(0).to(dtype=torch.bfloat16).cpu()
                )
                h_ffn_z_rows.append(
                    torch.cat([cap["h_post_ffn_upper"][li] for li in upper_indices], dim=1).squeeze(0).to(dtype=torch.bfloat16).cpu()
                )
                if ref_kv_z is not None:
                    for (k_h, v_h), (k_r, v_r) in zip(extract_kv_pairs(out_z.past_key_values), ref_kv_z):
                        max_kv_diff = max(
                            max_kv_diff,
                            float((k_h - k_r).abs().max().item()),
                            float((v_h - v_r).abs().max().item()),
                        )

                if max_kv_diff > 0.0:
                    raise RuntimeError(f"Non-zero KV perturbation detected during Bank v2 export: max_kv_diff={max_kv_diff}")

                kv_p = out_p.past_key_values
                kv_z = out_z.past_key_values
                lp = lm_head(out_p.last_hidden_state[:, -1:, :])[:, -1, :]
                lz = lm_head(out_z.last_hidden_state[:, -1:, :])[:, -1, :]

                for step_idx in range(1, max_steps + 1):
                    lp_1d = lp.squeeze(0)
                    lz_1d = lz.squeeze(0)
                    if step_idx == 1:
                        logits_plus_t1 = lp_1d.to(dtype=torch.bfloat16).cpu()
                        logits_zero_t1 = lz_1d.to(dtype=torch.bfloat16).cpu()
                    else:
                        ids_row, vp_row, vz_row = _build_topk_union_row(lp_1d, lz_1d)
                        topk_ids_rows.append(ids_row)
                        topk_p_rows.append(vp_row)
                        topk_z_rows.append(vz_row)

                    invariant_rows.append(_compute_simplex_invariant_row(lp_1d, lz_1d))
                    next_tok_tensor, _, _, _ = adapter.engine.decode_step(lp, lz, q_mask=None, step_idx=step_idx)
                    next_id = int(next_tok_tensor.item())
                    _add_active([next_id])
                    _add_active(torch.topk(lp_1d.float(), k=min(8, int(lp_1d.shape[-1]))).indices.tolist())

                    if step_idx == max_steps:
                        break

                    next_emb = embed_layer(torch.tensor([[next_id]], dtype=torch.long, device=device))
                    cap["clear"]()
                    out_p = lm_core(inputs_embeds=next_emb, past_key_values=kv_p, use_cache=True, output_attentions=False, return_dict=True)
                    kv_p = out_p.past_key_values
                    h_in_p_rows.append(cap["h_in_upper"].squeeze(0).to(dtype=torch.bfloat16).cpu())
                    h_attn_p_rows.append(
                        torch.cat([cap["h_post_attn_upper"][li] for li in upper_indices], dim=1).squeeze(0).to(dtype=torch.bfloat16).cpu()
                    )
                    h_ffn_p_rows.append(
                        torch.cat([cap["h_post_ffn_upper"][li] for li in upper_indices], dim=1).squeeze(0).to(dtype=torch.bfloat16).cpu()
                    )

                    cap["clear"]()
                    out_z = lm_core(inputs_embeds=next_emb, past_key_values=kv_z, use_cache=True, output_attentions=False, return_dict=True)
                    kv_z = out_z.past_key_values
                    h_in_z_rows.append(cap["h_in_upper"].squeeze(0).to(dtype=torch.bfloat16).cpu())
                    h_attn_z_rows.append(
                        torch.cat([cap["h_post_attn_upper"][li] for li in upper_indices], dim=1).squeeze(0).to(dtype=torch.bfloat16).cpu()
                    )
                    h_ffn_z_rows.append(
                        torch.cat([cap["h_post_ffn_upper"][li] for li in upper_indices], dim=1).squeeze(0).to(dtype=torch.bfloat16).cpu()
                    )

                    lp = lm_head(out_p.last_hidden_state[:, -1:, :])[:, -1, :]
                    lz = lm_head(out_z.last_hidden_state[:, -1:, :])[:, -1, :]

            for gt_ans in gt_answers:
                _add_active(tokenizer.encode(str(gt_ans), add_special_tokens=False))
            _add_active(eos_ids)

            active_ids_t = torch.tensor(active_set, dtype=torch.int32)
            lm_slice = lm_head_weight[active_ids_t.to(device=lm_head_weight.device, dtype=torch.long)].to(dtype=torch.bfloat16).cpu()
            vocab_size = int(logits_plus_t1.shape[-1])
            q_mask = build_query_token_mask(question, tokenizer, vocab_size=vocab_size, device=torch.device("cpu"))

            payload = {
                "qid": qid,
                "class_tag": class_tag,
                "logits_plus_t1": logits_plus_t1,
                "logits_zero_t1": logits_zero_t1,
                "step_topk_ids": torch.stack(topk_ids_rows, dim=0) if topk_ids_rows else torch.empty((0, TOPK_UNION_CAP), dtype=torch.int32),
                "step_topk_plus": torch.stack(topk_p_rows, dim=0) if topk_p_rows else torch.empty((0, TOPK_UNION_CAP), dtype=torch.bfloat16),
                "step_topk_zero": torch.stack(topk_z_rows, dim=0) if topk_z_rows else torch.empty((0, TOPK_UNION_CAP), dtype=torch.bfloat16),
                "step_simplex_invariants": torch.stack(invariant_rows, dim=0),
                "h_in_upper_plus": torch.stack(h_in_p_rows, dim=0),
                "h_in_upper_zero": torch.stack(h_in_z_rows, dim=0),
                "h_post_attn_upper_plus": torch.stack(h_attn_p_rows, dim=0),
                "h_post_attn_upper_zero": torch.stack(h_attn_z_rows, dim=0),
                "h_post_ffn_upper_plus": torch.stack(h_ffn_p_rows, dim=0),
                "h_post_ffn_upper_zero": torch.stack(h_ffn_z_rows, dim=0),
                "active_vocab_ids": active_ids_t,
                "lm_head_slice": lm_slice,
                "q_mask": q_mask,
            }
            fname = f"qid_{qid}.pt"
            fpath = os.path.join(output_dir, fname)
            torch.save(payload, fpath)
            file_records[fname] = {
                "qid": qid,
                "class_tag": class_tag,
                "image_name": img_name,
                "size_bytes": os.path.getsize(fpath),
                "sha256": _sha256_file(fpath),
                "vocab_size": vocab_size,
            }

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return _finalize_bank_v2_manifest(
        output_dir=output_dir,
        file_records=file_records,
        base_commit_sha=base_commit_sha or _resolve_git_head(),
        backbone_id=model_name,
        num_hidden_layers=num_layers,
        max_steps=max_steps,
        max_kv_diff=max_kv_diff,
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Export and evaluate the 30-sample local calibration bank.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--image-dir", default=os.path.join(PROJECT_ROOT, "data", "vitextvqa", "st_images"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model-name", default="5CD-AI/Vintern-1B-v3_5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bank-version", type=int, choices=[1, 2], default=2)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--track", type=int, choices=[1, 2], default=1)
    parser.add_argument("--base-commit-sha", default=None)
    parser.add_argument("--mock", action="store_true", help="Export deterministic synthetic tensors offline.")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate an existing calibration bank.")
    args = parser.parse_args(argv)

    out_dir = args.output_dir or (DEFAULT_OUTPUT_DIR_V2 if args.bank_version == 2 else DEFAULT_OUTPUT_DIR)

    if not args.eval_only:
        if args.bank_version == 2:
            if args.mock:
                summary = export_synthetic_calibration_bank_v2(
                    args.manifest,
                    out_dir,
                    max_steps=args.max_steps,
                    base_commit_sha=args.base_commit_sha,
                )
            else:
                summary = export_live_calibration_bank_v2(
                    manifest_path=args.manifest,
                    image_dir=args.image_dir,
                    output_dir=out_dir,
                    model_name=args.model_name,
                    device=args.device,
                    max_steps=args.max_steps,
                    base_commit_sha=args.base_commit_sha,
                )
        else:
            if args.mock:
                summary = export_synthetic_calibration_bank(args.manifest, out_dir)
            else:
                summary = export_live_calibration_bank(
                    manifest_path=args.manifest,
                    image_dir=args.image_dir,
                    output_dir=out_dir,
                    model_name=args.model_name,
                    device=args.device,
                )
        print(
            f"[CAL_BANK v{args.bank_version}] Exported {summary['num_samples']} samples "
            f"({summary['total_mb']:.2f} MB) -> {out_dir}"
        )

    from src.nevfe.core.engine import AlgorithmConfig, NEVFEEngine

    engine = NEVFEEngine(config=AlgorithmConfig())
    if args.bank_version == 2:
        metrics = run_calibration_eval_v2(engine, out_dir, track=args.track)
    else:
        metrics = run_calibration_eval(engine, out_dir)
    print(f"[CAL_BANK v{args.bank_version}] CPU eval: {json.dumps(metrics)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
