# -*- coding: utf-8 -*-
"""Forensic prediction diff & anchored promotion ratchet for 400-image benchmark."""
from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_BASELINE_SUMMARY = os.path.join(PROJECT_ROOT, "configs", "frsa011_baseline_summary.json")
DEFAULT_BASELINE_PREDS = os.path.join(PROJECT_ROOT, "configs", "frsa011_baseline_predictions.jsonl")
DEFAULT_PROBE_MANIFEST = os.path.join(PROJECT_ROOT, "configs", "probe_100_manifest.json")
DEFAULT_BASE_F1 = 64.12
DEFAULT_BASE_SUB = 62.19
MAX_FORENSIC_MD_BYTES = 8192

PREAMBLE_PREFIXES = (
    "dựa vào hình ảnh",
    "theo hình ảnh",
    "trong bức ảnh",
    "câu trả lời là",
    "đáp án là",
    "bức ảnh cho thấy",
    "có thể thấy",
    "rất tiếc",
    "xin lỗi",
)


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.lower().strip())
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn").replace("đ", "d")


def _norm(text: str) -> str:
    return " ".join((text or "").lower().strip().split())


def classify_failure_mode(
    pred_answer: str,
    gt_answers: List[str],
    f1_score: float,
    em_score: float = 0.0,
) -> str:
    """Strict-priority failure taxonomy: MATCHED > PREAMBLE_ECHO > TRUNCATION > GLYPH_CONFUSION > PRIOR_HALLUCINATION."""
    if em_score >= 1.0 or f1_score >= 0.99:
        return "MATCHED"

    pred_n = _norm(pred_answer)
    best_gt = _norm(gt_answers[0]) if gt_answers else ""

    if pred_n and any(prefix in pred_n for prefix in PREAMBLE_PREFIXES):
        return "PREAMBLE_ECHO"

    is_prefix_trunc = bool(pred_n and best_gt and best_gt.startswith(pred_n) and len(pred_n) + 2 < len(best_gt))
    if not pred_n or (len(best_gt) >= 6 and len(pred_n) < 3) or is_prefix_trunc:
        return "TRUNCATION"

    pred_flat = _strip_accents(pred_n)
    for gt in gt_answers:
        gt_flat = _strip_accents(_norm(gt))
        if gt_flat and SequenceMatcher(None, pred_flat, gt_flat).ratio() >= 0.65:
            return "GLYPH_CONFUSION"

    return "PRIOR_HALLUCINATION"


def check_promotion_ratchet(
    cand_f1: float,
    cand_sub: float,
    base_f1: float = DEFAULT_BASE_F1,
    base_sub: float = DEFAULT_BASE_SUB,
    best_f1: Optional[float] = None,
    best_sub: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Enforce anchored monotonic promotion ratchet:
      d_comp = cand_comp - eff_best_comp >= +0.25,
      cand_f1 >= max(base_f1, eff_best_f1 - 0.50),
      cand_sub >= max(base_sub, eff_best_sub - 0.50).
    """
    eff_best_f1 = float(base_f1) if best_f1 is None else float(best_f1)
    eff_best_sub = float(base_sub) if best_sub is None else float(best_sub)

    cand_comp = (float(cand_f1) + float(cand_sub)) / 2.0
    base_comp = (float(base_f1) + float(base_sub)) / 2.0
    eff_best_comp = (eff_best_f1 + eff_best_sub) / 2.0
    d_comp = cand_comp - eff_best_comp

    f1_floor = max(float(base_f1), eff_best_f1 - 0.50)
    sub_floor = max(float(base_sub), eff_best_sub - 0.50)

    reasons: List[str] = []
    if d_comp < 0.25 - 1e-9:
        reasons.append(f"d_comp={d_comp:+.2f} < +0.25 vs eff_best_comp {eff_best_comp:.2f}")
    if float(cand_f1) < f1_floor - 1e-9:
        reasons.append(f"cand_f1={cand_f1:.2f} < floor {f1_floor:.2f}")
    if float(cand_sub) < sub_floor - 1e-9:
        reasons.append(f"cand_sub={cand_sub:.2f} < floor {sub_floor:.2f}")

    return {
        "promoted": len(reasons) == 0,
        "cand_f1": round(float(cand_f1), 2),
        "cand_sub": round(float(cand_sub), 2),
        "cand_comp": round(cand_comp, 2),
        "base_f1": round(float(base_f1), 2),
        "base_sub": round(float(base_sub), 2),
        "base_comp": round(base_comp, 2),
        "eff_best_f1": round(eff_best_f1, 2),
        "eff_best_sub": round(eff_best_sub, 2),
        "eff_best_comp": round(eff_best_comp, 2),
        "d_comp": round(d_comp, 2),
        "f1_floor": round(f1_floor, 2),
        "sub_floor": round(sub_floor, 2),
        "reasons": reasons,
    }


def _load_jsonl_map(path: str) -> Dict[int, Dict[str, Any]]:
    records: Dict[int, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                qid = obj.get("question_id", obj.get("qid"))
                if qid is not None:
                    records[int(qid)] = obj
    return records


def _extract_scores(rec: Dict[str, Any]) -> Tuple[str, float, float, float, List[str]]:
    s2 = rec.get("stage2_metrics", {})
    pred = rec.get("pred_answer", rec.get("candidate_pred", ""))
    f1 = float(s2.get("f1_verified", rec.get("candidate_f1", rec.get("f1_verified", 0.0))))
    sub = float(s2.get("substring_verified", rec.get("candidate_substring", rec.get("substring_verified", 0.0))))
    em = float(s2.get("em_verified", rec.get("candidate_em", rec.get("em_verified", 0.0))))
    gts = rec.get("verified_all_answers") or rec.get("gt_answers") or []
    return pred, f1, sub, em, gts


def _extract_candidate_summary_scores(candidate_summary_path: str) -> Tuple[float, float, Optional[float]]:
    if not os.path.isfile(candidate_summary_path):
        raise FileNotFoundError(f"Candidate summary not found: {candidate_summary_path}")
    with open(candidate_summary_path, "r", encoding="utf-8") as f:
        csum = json.load(f)
    if not isinstance(csum, dict):
        raise ValueError(f"Candidate summary must be a JSON object: {candidate_summary_path}")
    s2 = csum.get("stage2") if "stage2" in csum else csum.get("overall")
    if not isinstance(s2, dict):
        raise KeyError(f"Candidate summary missing required 'stage2' or 'overall' section: {candidate_summary_path}")

    if "f1_verified_pct" in s2:
        cand_f1 = float(s2["f1_verified_pct"])
    elif "candidate_token_f1" in s2:
        cand_f1 = float(s2["candidate_token_f1"])
    else:
        raise KeyError(f"Candidate summary missing 'f1_verified_pct'/'candidate_token_f1': {candidate_summary_path}")

    if "substring_match_verified_pct" in s2:
        cand_sub = float(s2["substring_match_verified_pct"])
    elif "candidate_substring_match" in s2:
        cand_sub = float(s2["candidate_substring_match"])
    else:
        raise KeyError(
            f"Candidate summary missing 'substring_match_verified_pct'/'candidate_substring_match': {candidate_summary_path}"
        )

    perf = csum.get("performance", {})
    avg_lat = float(perf["avg_latency_s"]) if isinstance(perf, dict) and "avg_latency_s" in perf else None
    return cand_f1, cand_sub, avg_lat


def _extract_baseline_summary_scores(baseline_summary_path: Optional[str]) -> Tuple[float, float]:
    if not baseline_summary_path:
        return DEFAULT_BASE_F1, DEFAULT_BASE_SUB
    if not os.path.isfile(baseline_summary_path):
        raise FileNotFoundError(f"Baseline summary not found: {baseline_summary_path}")
    with open(baseline_summary_path, "r", encoding="utf-8") as f:
        bsum = json.load(f)
    if not isinstance(bsum, dict) or not isinstance(bsum.get("benchmark_1222"), dict):
        raise KeyError(f"Baseline summary missing required 'benchmark_1222' section: {baseline_summary_path}")
    b1222 = bsum["benchmark_1222"]
    if "token_f1_pct" not in b1222 or "substring_match_pct" not in b1222:
        raise KeyError(
            f"Baseline summary 'benchmark_1222' missing 'token_f1_pct' or 'substring_match_pct': {baseline_summary_path}"
        )
    return float(b1222["token_f1_pct"]), float(b1222["substring_match_pct"])


def _load_optional_json(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path or not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {"entries": data}


def _render_forensic_markdown(
    ratchet: Dict[str, Any],
    win_count: int,
    loss_count: int,
    tie_count: int,
    taxonomy_counts: Dict[str, int],
    per_class_summary: Dict[str, Dict[str, float]],
    calibration_diffs: List[Dict[str, Any]],
    provenance_data: Optional[Dict[str, Any]] = None,
    probe_data: Optional[Dict[str, Any]] = None,
    telemetry_data: Optional[Dict[str, Any]] = None,
    avg_latency_s: Optional[float] = None,
    max_kv_diff: Optional[float] = None,
) -> str:
    header_lines: List[str] = [
        "# Forensic Delta Report",
        "",
        f"- **Promotion Verdict**: `{'PROMOTED' if ratchet['promoted'] else 'REJECTED'}`",
        f"- **Candidate**: F1={ratchet['cand_f1']:.2f}%, Sub={ratchet['cand_sub']:.2f}%, Comp={ratchet['cand_comp']:.2f}% (delta={ratchet['d_comp']:+.2f} vs eff_best={ratchet['eff_best_comp']:.2f}%)",
        f"- **Anchor Floors**: Comp >= {ratchet['eff_best_comp'] + 0.25:.2f}%, F1 >= {ratchet['f1_floor']:.2f}%, Sub >= {ratchet['sub_floor']:.2f}%",
        f"- **Win / Loss / Tie vs Baseline**: {win_count} / {loss_count} / {tie_count}",
    ]

    if avg_latency_s is not None or telemetry_data is not None or max_kv_diff is not None:
        lat_val = avg_latency_s
        vram_val = None
        kv_val = max_kv_diff
        fwd_max = None
        if telemetry_data:
            vram_val = telemetry_data.get("peak_vram_gb")
            if kv_val is None:
                kv_val = telemetry_data.get("max_kv_diff")
            fwd_max = telemetry_data.get("decoder_forward_calls_per_step_max")
        parts: List[str] = []
        if lat_val is not None:
            parts.append(f"avg_latency_s={float(lat_val):.3f}s")
        if vram_val is not None:
            parts.append(f"peak_vram_gb={float(vram_val):.2f}GB")
        if kv_val is not None:
            parts.append(f"max_kv_diff={float(kv_val):.6g}")
        if fwd_max is not None:
            parts.append(f"fwd_calls_per_step_max={float(fwd_max):.2f}")
        if parts:
            header_lines.append(f"- **Runtime Telemetry**: {', '.join(parts)}")

    if provenance_data:
        papers = provenance_data.get("papers") or provenance_data.get("entries") or []
        if isinstance(papers, list) and papers:
            p_tags = [
                f"{p.get('paper_id', p.get('id', 'unknown'))} ({p.get('status', 'UNKNOWN')}@{p.get('section', p.get('target_section', '?'))})"
                for p in papers[:3]
            ]
            header_lines.append(f"- **Provenance**: {'; '.join(p_tags)}")

    if probe_data:
        header_lines.extend(
            [
                "",
                "## Stage A LOO Matrix",
                f"- Decision: `{probe_data.get('loo_decision', 'UNKNOWN')}` | selected=`{probe_data.get('selected_mask')}` | submask_hint=`{probe_data.get('submask_hint')}` | full_comp_70={probe_data.get('full_comp_70')} vs best_probe_70={probe_data.get('best_probe_70_comp')}",
            ]
        )
        for m_name, m_vals in sorted((probe_data.get("masks") or {}).items()):
            header_lines.append(
                f"  - mask `{m_name}`: comp_70={m_vals.get('comp_70', 0.0):.2f} (f1_70={m_vals.get('f1_70', 0.0):.2f}, sub_70={m_vals.get('sub_70', 0.0):.2f})"
            )

    if per_class_summary:
        header_lines.extend(["", "## Probe Subset Class Deltas (`CLASS_A..CLASS_E`)"])
        for cls_name, cstats in sorted(per_class_summary.items()):
            header_lines.append(
                f"- `{cls_name}` (n={cstats['count']}): cand_f1={cstats['cand_f1_pct']:.2f}%, base_f1={cstats['base_f1_pct']:.2f}%, delta={cstats['delta_f1_pct']:+.2f}%"
            )

    header_lines.extend(["", "## Failure Taxonomy Counts"])
    for k, v in taxonomy_counts.items():
        header_lines.append(f"- `{k}`: {v}")

    # Filter calibration-only diffs to changed or failing QIDs sorted by |cand_f1 - base_f1| desc
    filtered: List[Dict[str, Any]] = []
    for d in calibration_diffs:
        bf1 = d.get("base_f1")
        cf1 = float(d["cand_f1"])
        changed = bf1 is not None and abs(cf1 - float(bf1)) > 1e-6
        failing = cf1 < 99.0
        if changed or failing:
            abs_delta = abs(cf1 - float(bf1)) if bf1 is not None else (100.0 - cf1)
            filtered.append((abs_delta, cf1, int(d["question_id"]), d))
    filtered.sort(key=lambda item: (-item[0], item[1], item[2]))

    omitted = len(calibration_diffs) - len(filtered)
    section_hdr = f"## Calibration-Only Changed/Failing Diffs (Shown<= {len(filtered)} / {len(calibration_diffs)})"
    lines = list(header_lines) + ["", section_hdr]

    kept = 0
    for _, _, _, d in filtered:
        q_short = str(d["question"])[:140]
        gt_short = str(d["gt_primary"])[:80]
        pred_short = str(d["cand_pred"])[:80]
        b_str = f"{d['base_f1']:.1f}%" if d.get("base_f1") is not None else "N/A"
        cand_line = (
            f"- QID `{d['question_id']}` [{d['error_class']} | {d['failure_mode']}] "
            f"F1={d['cand_f1']:.1f}% (base {b_str}) | Q={q_short} | GT={gt_short} | Pred={pred_short}"
        )
        rem_after = omitted + (len(filtered) - kept - 1)
        reserve_note = f"- (+{rem_after} unchanged/omitted calibration QIDs)" if rem_after > 0 else ""
        candidate_doc = "\n".join(lines + [cand_line] + ([reserve_note] if reserve_note else [])) + "\n"
        if len(candidate_doc.encode("utf-8")) > MAX_FORENSIC_MD_BYTES:
            omitted += len(filtered) - kept
            break
        lines.append(cand_line)
        kept += 1

    if omitted > 0:
        note_line = f"- (+{omitted} unchanged/omitted calibration QIDs)"
        test_doc = "\n".join(lines + [note_line]) + "\n"
        if len(test_doc.encode("utf-8")) <= MAX_FORENSIC_MD_BYTES:
            lines.append(note_line)

    rendered = "\n".join(lines) + "\n"
    encoded = rendered.encode("utf-8")
    if len(encoded) > MAX_FORENSIC_MD_BYTES:
        rendered = encoded[:MAX_FORENSIC_MD_BYTES].decode("utf-8", errors="ignore").rstrip("\n") + "\n"
    return rendered


def run_forensic_diff(
    candidate_preds_path: str,
    candidate_summary_path: Optional[str] = None,
    baseline_preds_path: Optional[str] = None,
    baseline_summary_path: Optional[str] = DEFAULT_BASELINE_SUMMARY,
    probe_manifest_path: str = DEFAULT_PROBE_MANIFEST,
    best_f1: Optional[float] = None,
    best_sub: Optional[float] = None,
    out_json_path: Optional[str] = None,
    out_md_path: Optional[str] = None,
    provenance_json_path: Optional[str] = None,
    probe_summary_path: Optional[str] = None,
    colab_telemetry_path: Optional[str] = None,
    max_kv_diff: Optional[float] = None,
) -> Dict[str, Any]:
    cand_map = _load_jsonl_map(candidate_preds_path)
    resolved_base_preds = baseline_preds_path
    if resolved_base_preds is None and os.path.isfile(DEFAULT_BASELINE_PREDS):
        resolved_base_preds = DEFAULT_BASELINE_PREDS
    base_map = _load_jsonl_map(resolved_base_preds) if (resolved_base_preds and os.path.isfile(resolved_base_preds)) else {}

    cal_info: Dict[int, Dict[str, Any]] = {}
    if probe_manifest_path and os.path.isfile(probe_manifest_path):
        with open(probe_manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        for item in manifest:
            qid = int(item.get("question_id", item.get("qid")))
            cal_info[qid] = item

    avg_latency_s: Optional[float] = None
    if candidate_summary_path is not None:
        cand_f1, cand_sub, avg_latency_s = _extract_candidate_summary_scores(candidate_summary_path)
    else:
        n_c = max(1, len(cand_map))
        cand_f1 = sum(_extract_scores(r)[1] for r in cand_map.values()) / n_c * 100.0
        cand_sub = sum(_extract_scores(r)[2] for r in cand_map.values()) / n_c * 100.0

    base_f1, base_sub = _extract_baseline_summary_scores(baseline_summary_path)

    ratchet = check_promotion_ratchet(
        cand_f1=cand_f1,
        cand_sub=cand_sub,
        base_f1=base_f1,
        base_sub=base_sub,
        best_f1=best_f1,
        best_sub=best_sub,
    )

    taxonomy_counts = {
        "MATCHED": 0,
        "PREAMBLE_ECHO": 0,
        "TRUNCATION": 0,
        "GLYPH_CONFUSION": 0,
        "PRIOR_HALLUCINATION": 0,
    }
    class_deltas: Dict[str, Dict[str, float]] = {}
    win_count = 0
    loss_count = 0
    tie_count = 0
    calibration_diffs: List[Dict[str, Any]] = []

    for qid in sorted(cand_map.keys()):
        crec = cand_map[qid]
        c_pred, c_f1, _, c_em, gts = _extract_scores(crec)
        mode = classify_failure_mode(c_pred, gts, c_f1, c_em)
        taxonomy_counts[mode] += 1

        meta = cal_info.get(qid, {})
        in_probe_subset = (qid in cal_info) if cal_info else bool(crec.get("error_class"))
        err_cls = str(crec.get("error_class") or meta.get("error_class") or "UNLABELED")
        if in_probe_subset:
            bucket = class_deltas.setdefault(err_cls, {"count": 0.0, "cand_f1_sum": 0.0, "base_f1_sum": 0.0})
            bucket["count"] += 1.0
            bucket["cand_f1_sum"] += c_f1 * 100.0

        b_pred = ""
        b_f1 = 0.0
        if qid in base_map:
            b_pred, b_f1, _, _, _ = _extract_scores(base_map[qid])
            if in_probe_subset:
                bucket["base_f1_sum"] += b_f1 * 100.0
            diff_val = c_f1 - b_f1
            if diff_val > 1e-6:
                win_count += 1
            elif diff_val < -1e-6:
                loss_count += 1
            else:
                tie_count += 1

        # Anti-Goodhart blindfold: ONLY emit sample-level diffs for the 30 is_calibration QIDs
        if bool(meta.get("is_calibration", False)):
            calibration_diffs.append(
                {
                    "question_id": qid,
                    "error_class": err_cls,
                    "failure_mode": mode,
                    "cand_f1": round(c_f1 * 100.0, 2),
                    "base_f1": round(b_f1 * 100.0, 2) if qid in base_map else None,
                    "question": crec.get("question", meta.get("question", "")),
                    "gt_primary": gts[0] if gts else "",
                    "cand_pred": c_pred,
                    "base_pred": b_pred if qid in base_map else None,
                }
            )

    per_class_summary: Dict[str, Dict[str, float]] = {}
    for cls_name, b in sorted(class_deltas.items()):
        cnt = max(1.0, b["count"])
        cf1 = round(b["cand_f1_sum"] / cnt, 2)
        bf1 = round(b["base_f1_sum"] / cnt, 2) if base_map else 0.0
        per_class_summary[cls_name] = {
            "count": int(b["count"]),
            "cand_f1_pct": cf1,
            "base_f1_pct": bf1,
            "delta_f1_pct": round(cf1 - bf1, 2),
        }

    provenance_data = _load_optional_json(provenance_json_path)
    probe_data = _load_optional_json(probe_summary_path)
    telemetry_data = _load_optional_json(colab_telemetry_path)

    report = {
        "total_samples": len(cand_map),
        "ratchet": ratchet,
        "taxonomy_counts": taxonomy_counts,
        "vs_baseline": {"wins": win_count, "losses": loss_count, "ties": tie_count},
        "per_class": per_class_summary,
        "calibration_diffs": calibration_diffs,
        "provenance": provenance_data,
        "probe_summary": probe_data,
        "colab_telemetry": telemetry_data,
    }

    if out_json_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_json_path)), exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    if out_md_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_md_path)), exist_ok=True)
        md_content = _render_forensic_markdown(
            ratchet=ratchet,
            win_count=win_count,
            loss_count=loss_count,
            tie_count=tie_count,
            taxonomy_counts=taxonomy_counts,
            per_class_summary=per_class_summary,
            calibration_diffs=calibration_diffs,
            provenance_data=provenance_data,
            probe_data=probe_data,
            telemetry_data=telemetry_data,
            avg_latency_s=avg_latency_s,
            max_kv_diff=max_kv_diff,
        )
        with open(out_md_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(md_content)

    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Forensic prediction diff and anchored ratchet.")
    parser.add_argument("--candidate-preds", required=True)
    parser.add_argument("--candidate-summary", default=None)
    parser.add_argument("--baseline-preds", "--baseline", dest="baseline_preds", default=None)
    parser.add_argument("--baseline-summary", default=DEFAULT_BASELINE_SUMMARY)
    parser.add_argument("--probe-manifest", default=DEFAULT_PROBE_MANIFEST)
    parser.add_argument("--best-f1", type=float, default=None)
    parser.add_argument("--best-sub", type=float, default=None)
    parser.add_argument("--provenance-json", default=None)
    parser.add_argument("--probe-summary", default=None)
    parser.add_argument("--colab-telemetry", default=None)
    parser.add_argument("--max-kv-diff", type=float, default=None)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    args = parser.parse_args(argv)

    cand_dir = os.path.dirname(os.path.abspath(args.candidate_preds))
    out_json = args.out_json or os.path.join(cand_dir, "forensic_delta.json")
    out_md = args.out_md or os.path.join(cand_dir, "forensic_delta.md")

    report = run_forensic_diff(
        candidate_preds_path=args.candidate_preds,
        candidate_summary_path=args.candidate_summary,
        baseline_preds_path=args.baseline_preds,
        baseline_summary_path=args.baseline_summary,
        probe_manifest_path=args.probe_manifest,
        best_f1=args.best_f1,
        best_sub=args.best_sub,
        out_json_path=out_json,
        out_md_path=out_md,
        provenance_json_path=args.provenance_json,
        probe_summary_path=args.probe_summary,
        colab_telemetry_path=args.colab_telemetry,
        max_kv_diff=args.max_kv_diff,
    )
    print(f"[DIFF] Promoted={report['ratchet']['promoted']} d_comp={report['ratchet']['d_comp']:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
