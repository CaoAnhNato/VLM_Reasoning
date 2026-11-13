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

DEFAULT_BASELINE_SUMMARY = os.path.join(PROJECT_ROOT, "configs", "iter23_baseline_summary.json")
DEFAULT_PROBE_MANIFEST = os.path.join(PROJECT_ROOT, "configs", "probe_100_manifest.json")

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
    base_f1: float = 60.19,
    base_sub: float = 61.46,
    best_f1: Optional[float] = None,
    best_sub: Optional[float] = None,
) -> Dict[str, Any]:
    """Enforce anchored promotion ratchet: d_comp >= +0.25, f1 >= max(base, best-0.50), sub >= max(base, best-0.50)."""
    eff_best_f1 = base_f1 if best_f1 is None else float(best_f1)
    eff_best_sub = base_sub if best_sub is None else float(best_sub)

    cand_comp = (float(cand_f1) + float(cand_sub)) / 2.0
    base_comp = (float(base_f1) + float(base_sub)) / 2.0
    d_comp = cand_comp - base_comp

    f1_floor = max(float(base_f1), eff_best_f1 - 0.50)
    sub_floor = max(float(base_sub), eff_best_sub - 0.50)

    reasons: List[str] = []
    if d_comp < 0.25 - 1e-9:
        reasons.append(f"d_comp={d_comp:+.2f} < +0.25 vs anchor {base_comp:.2f}")
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


def run_forensic_diff(
    candidate_preds_path: str,
    candidate_summary_path: Optional[str] = None,
    baseline_preds_path: Optional[str] = None,
    baseline_summary_path: str = DEFAULT_BASELINE_SUMMARY,
    probe_manifest_path: str = DEFAULT_PROBE_MANIFEST,
    best_f1: Optional[float] = None,
    best_sub: Optional[float] = None,
    out_json_path: Optional[str] = None,
    out_md_path: Optional[str] = None,
) -> Dict[str, Any]:
    cand_map = _load_jsonl_map(candidate_preds_path)
    base_map = _load_jsonl_map(baseline_preds_path) if (baseline_preds_path and os.path.isfile(baseline_preds_path)) else {}

    cal_info: Dict[int, Dict[str, Any]] = {}
    if probe_manifest_path and os.path.isfile(probe_manifest_path):
        with open(probe_manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        for item in manifest:
            qid = int(item.get("question_id", item.get("qid")))
            cal_info[qid] = item

    # Resolve aggregate F1 / Sub metrics
    if candidate_summary_path and os.path.isfile(candidate_summary_path):
        with open(candidate_summary_path, "r", encoding="utf-8") as f:
            csum = json.load(f)
        s2 = csum.get("stage2", csum.get("overall", {}))
        cand_f1 = float(s2.get("f1_verified_pct", s2.get("candidate_token_f1", 0.0)))
        cand_sub = float(s2.get("substring_match_verified_pct", s2.get("candidate_substring_match", 0.0)))
    else:
        n_c = max(1, len(cand_map))
        cand_f1 = sum(_extract_scores(r)[1] for r in cand_map.values()) / n_c * 100.0
        cand_sub = sum(_extract_scores(r)[2] for r in cand_map.values()) / n_c * 100.0

    base_f1 = 60.19
    base_sub = 61.46
    if baseline_summary_path and os.path.isfile(baseline_summary_path):
        with open(baseline_summary_path, "r", encoding="utf-8") as f:
            bsum = json.load(f)
        b1222 = bsum.get("benchmark_1222", {})
        base_f1 = float(b1222.get("token_f1_pct", base_f1))
        base_sub = float(b1222.get("substring_match_pct", base_sub))

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
        c_pred, c_f1, c_sub, c_em, gts = _extract_scores(crec)
        mode = classify_failure_mode(c_pred, gts, c_f1, c_em)
        taxonomy_counts[mode] += 1

        meta = cal_info.get(qid, {})
        err_cls = str(crec.get("error_class") or meta.get("error_class") or "UNLABELED")
        bucket = class_deltas.setdefault(err_cls, {"count": 0.0, "cand_f1_sum": 0.0, "base_f1_sum": 0.0})
        bucket["count"] += 1.0
        bucket["cand_f1_sum"] += c_f1 * 100.0

        b_pred = ""
        b_f1 = 0.0
        if qid in base_map:
            b_pred, b_f1, _, _, _ = _extract_scores(base_map[qid])
            bucket["base_f1_sum"] += b_f1 * 100.0
            diff_val = c_f1 - b_f1
            if diff_val > 1e-6:
                win_count += 1
            elif diff_val < -1e-6:
                loss_count += 1
            else:
                tie_count += 1

        # Anti-Goodhart rule: ONLY emit sample-level diffs for the 30 is_calibration QIDs
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

    report = {
        "total_samples": len(cand_map),
        "ratchet": ratchet,
        "taxonomy_counts": taxonomy_counts,
        "vs_baseline": {"wins": win_count, "losses": loss_count, "ties": tie_count},
        "per_class": per_class_summary,
        "calibration_diffs": calibration_diffs,
    }

    if out_json_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_json_path)), exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    if out_md_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_md_path)), exist_ok=True)
        lines = [
            "# Forensic Delta Report",
            "",
            f"- **Promotion Verdict**: `{'PROMOTED' if ratchet['promoted'] else 'REJECTED'}`",
            f"- **Candidate**: F1={ratchet['cand_f1']:.2f}%, Sub={ratchet['cand_sub']:.2f}%, Comp={ratchet['cand_comp']:.2f}% (delta={ratchet['d_comp']:+.2f})",
            f"- **Anchor Floors**: Comp >= {ratchet['base_comp'] + 0.25:.2f}%, F1 >= {ratchet['f1_floor']:.2f}%, Sub >= {ratchet['sub_floor']:.2f}%",
            f"- **Win / Loss / Tie vs Baseline**: {win_count} / {loss_count} / {tie_count}",
            "",
            "## Failure Taxonomy Counts",
        ]
        for k, v in taxonomy_counts.items():
            lines.append(f"- `{k}`: {v}")
        lines.extend(["", f"## Calibration-Only Sample Diffs (N={len(calibration_diffs)})"])
        for d in calibration_diffs:
            lines.append(
                f"- QID `{d['question_id']}` [{d['error_class']} | {d['failure_mode']}] "
                f"F1={d['cand_f1']:.1f}% | Q={d['question']} | GT={d['gt_primary']} | Pred={d['cand_pred']}"
            )
        with open(out_md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Forensic prediction diff and anchored ratchet.")
    parser.add_argument("--candidate-preds", required=True)
    parser.add_argument("--candidate-summary", default=None)
    parser.add_argument("--baseline-preds", default=None)
    parser.add_argument("--baseline-summary", default=DEFAULT_BASELINE_SUMMARY)
    parser.add_argument("--probe-manifest", default=DEFAULT_PROBE_MANIFEST)
    parser.add_argument("--best-f1", type=float, default=None)
    parser.add_argument("--best-sub", type=float, default=None)
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
    )
    print(f"[DIFF] Promoted={report['ratchet']['promoted']} d_comp={report['ratchet']['d_comp']:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
