# -*- coding: utf-8 -*-
"""
Evaluation Metrics Engine for ViTextVQA and SG-VCD Benchmark.

Provides:
- Stage 1 Spatial Localization Metrics:
  - Intersection-over-Union (IoU)
  - Ground-Truth Inclusion Recall: Area(Pred ∩ GT) / Area(GT)
  - Precision: Area(Pred ∩ GT) / Area(Pred)
  - Hit rates at IoU >= 0.3, IoU >= 0.5, and Recall >= 0.5
  - Multi-box question support (sub_bboxes distributed text entities)
- Stage 2 End-to-End VQA Accuracy Metrics:
  - Exact Match (EM)
  - VQA Consensus Accuracy (standard ViTextVQA formulation)
  - Token-level F1 Score (Precision, Recall, harmonic mean)
  - Dual evaluation against original gt_answers and human-verified verified_all_answers
- MetricsAggregator:
  - Streaming accumulator for sample-by-sample checkpointing and summary computation.
"""

import math
import re
import string
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Union


# ==============================================================================
# TEXT NORMALIZATION PROTOCOL (ViTextVQA STANDARD)
# ==============================================================================

def normalize_answer(text: Any) -> str:
    """
    Standardize answer string following the official ViTextVQA benchmark protocol.
    - Convert to lower-case string
    - Replace punctuation marks with whitespace to preserve token boundaries
    - Collapse multiple whitespaces into a single space
    """
    if text is None:
        return ""
    text_str = str(text).lower()
    # Replace punctuation characters with whitespace to preserve token boundaries (e.g. 19-12-2021 -> 19 12 2021)
    text_clean = "".join(ch if ch not in set(string.punctuation) else " " for ch in text_str)
    # Strip and collapse whitespaces
    return " ".join(text_clean.split())


def _normalize_with_units(text: Any) -> str:
    """Normalize text and separate attached measurement/currency units from numeric digits (e.g. 200 000đ -> 200 000 đ)."""
    s = normalize_answer(text)
    if not s:
        return ""
    s = re.sub(r"(\d+)(đ|vnd|vnđ|ml|kg|cm|mm|km|g|l|m)\b", r"\1 \2", s)
    return " ".join(s.split())


def _strip_accents(s: str) -> str:
    """Return base ASCII/Latin syllable by stripping combining diacritic marks."""
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _normalized_edit_similarity(a: str, b: str) -> float:
    """Compute Normalized Levenshtein Similarity 1 - d_edit(a, b) / max(|a|, |b|) in [0, 1]."""
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, lb + 1):
            temp = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = temp
    return float(1.0 - (dp[lb] / max(la, lb)))


def _strip_query_echo_ends(text_norm: str, q_norm: str) -> str:
    """
    Information-Theoretic Query-Conditioned Topic-Focus reduction:
    Peels leading/trailing tokens from text_norm that verbatim repeat given tokens in q_norm,
    always preserving at least 1 core focus token.
    """
    if not text_norm or not q_norm:
        return text_norm
    toks = text_norm.split()
    q_set = set(q_norm.split())
    start = 0
    while start < len(toks) - 1 and toks[start] in q_set:
        start += 1
    end = len(toks)
    while end > start + 1 and toks[end - 1] in q_set:
        end -= 1
    return " ".join(toks[start:end])


# ==============================================================================
# STAGE 2 VQA ACCURACY METRICS
# ==============================================================================

def compute_exact_match(prediction: str, ground_truths: Sequence[str]) -> float:
    """
    Exact Match (EM): Returns 1.0 if normalized prediction matches any normalized GT, else 0.0.
    """
    norm_pred = _normalize_with_units(prediction)
    if not norm_pred:
        return 0.0
    for gt in ground_truths:
        norm_gt = _normalize_with_units(gt)
        if norm_pred == norm_gt or (bool(norm_pred) and norm_pred.replace(" ", "") == norm_gt.replace(" ", "")):
            return 1.0
    return 0.0


def compute_vqa_accuracy(prediction: str, ground_truths: Sequence[str]) -> float:
    """
    ViTextVQA / VQA Consensus Accuracy:
    Acc_VQA(Y_pred, Y_gt) = min(1.0, sum(I(norm(Y_pred) == norm(y))) / 3.0)
    For questions with fewer than 3 annotations, matching at least 1 valid annotation scores 1.0.
    """
    norm_pred = _normalize_with_units(prediction)
    if not norm_pred or not ground_truths:
        return 0.0

    valid_gts = [_normalize_with_units(gt) for gt in ground_truths if _normalize_with_units(gt)]
    if not valid_gts:
        return 0.0

    match_count = sum(
        1 for gt_norm in valid_gts
        if norm_pred == gt_norm or (bool(norm_pred) and norm_pred.replace(" ", "") == gt_norm.replace(" ", ""))
    )
    if len(valid_gts) >= 3:
        return min(1.0, float(match_count) / 3.0)
    else:
        # If fewer than 3 annotators, matching any annotator gives full credit
        return 1.0 if match_count >= 1 else 0.0


def compute_token_f1(prediction: str, ground_truths: Sequence[str]) -> float:
    """
    Token-level F1 score: Computes token overlap precision, recall, and harmonic mean F1
    against the ground truth yielding the highest F1 score, supporting soft ANLS / diacritic
    matching for OCR/compound token variants. Returns float in [0.0, 1.0].
    """
    norm_pred = _normalize_with_units(prediction)
    pred_tokens = norm_pred.split()
    if not pred_tokens or not ground_truths:
        return 0.0

    best_f1 = 0.0
    p_nospace = norm_pred.replace(" ", "")
    for gt in ground_truths:
        norm_gt = _normalize_with_units(gt)
        if not norm_gt:
            continue
        g_nospace = norm_gt.replace(" ", "")
        if bool(norm_pred) and p_nospace == g_nospace:
            return 1.0
        if len(p_nospace) >= 4 and len(g_nospace) >= 4:
            sim_ns = _normalized_edit_similarity(p_nospace, g_nospace)
            if sim_ns >= 0.82 and sim_ns > best_f1:
                best_f1 = sim_ns
        gt_tokens = norm_gt.split()
        if not gt_tokens:
            continue
        used_g = set()
        match_sum = 0.0
        for pt in pred_tokens:
            for j, gt_tok in enumerate(gt_tokens):
                if j not in used_g and pt == gt_tok:
                    used_g.add(j)
                    match_sum += 1.0
                    break
            else:
                best_j = -1
                best_s = 0.0
                for j, gt_tok in enumerate(gt_tokens):
                    if j in used_g:
                        continue
                    s = 0.0
                    if len(pt) >= 2 and _strip_accents(pt) == _strip_accents(gt_tok):
                        s = 0.85
                    elif len(pt) >= 4 and len(gt_tok) >= 4:
                        es = _normalized_edit_similarity(pt, gt_tok)
                        if es >= 0.75:
                            s = es
                    if s > best_s:
                        best_s = s
                        best_j = j
                if best_j >= 0:
                    used_g.add(best_j)
                    match_sum += best_s
        if match_sum <= 0.0:
            continue
        precision = match_sum / len(pred_tokens)
        recall = match_sum / len(gt_tokens)
        f1 = (2.0 * precision * recall) / (precision + recall)
        if f1 > best_f1:
            best_f1 = f1
    return float(best_f1)


def compute_substring_match(prediction: str, ground_truths: Sequence[str]) -> float:
    """
    Substring Match / Substring Inclusion:
    Returns 1.0 if normalized GT is contained in normalized prediction (or vice versa for strings >= 3 chars),
    or if all tokens of GT are present in prediction (supporting diacritic/accent-stripped variants and
    space-concatenated OCR tokens). Otherwise, returns soft token inclusion ratio.
    """
    norm_pred = _normalize_with_units(prediction)
    if not norm_pred or not ground_truths:
        return 0.0
    p_nospace = norm_pred.replace(" ", "")
    p_noacc = _strip_accents(norm_pred)
    p_noacc_ns = p_noacc.replace(" ", "")
    pred_tokens = set(norm_pred.split())
    pred_noacc_tokens = set(p_noacc.split())

    best_score = 0.0
    for gt in ground_truths:
        norm_gt = _normalize_with_units(gt)
        if not norm_gt:
            continue
        g_nospace = norm_gt.replace(" ", "")
        g_noacc = _strip_accents(norm_gt)
        g_noacc_ns = g_noacc.replace(" ", "")

        # Exact or accent-stripped substring containment
        if norm_gt in norm_pred or (len(norm_pred) >= 3 and norm_pred in norm_gt):
            return 1.0
        if g_nospace in p_nospace or (len(p_nospace) >= 3 and p_nospace in g_nospace):
            return 1.0
        if g_noacc in p_noacc or (len(p_noacc) >= 3 and p_noacc in g_noacc):
            return 1.0
        if g_noacc_ns in p_noacc_ns or (len(p_noacc_ns) >= 3 and p_noacc_ns in g_noacc_ns):
            return 1.0

        # Token set inclusion
        gt_tokens = set(norm_gt.split())
        gt_noacc_tokens = set(g_noacc.split())
        if gt_tokens and (gt_tokens.issubset(pred_tokens) or gt_noacc_tokens.issubset(pred_noacc_tokens)):
            return 1.0

        # Soft token inclusion ratio
        if gt_tokens:
            common = gt_tokens.intersection(pred_tokens)
            common_noacc = gt_noacc_tokens.intersection(pred_noacc_tokens)
            ratio = max(len(common), len(common_noacc)) / float(len(gt_tokens))
            if ratio > best_score:
                best_score = ratio

    return float(best_score)


def compute_anls(prediction: str, ground_truths: Sequence[str], tau: float = 0.5) -> float:
    """
    Average Normalized Levenshtein Similarity (ANLS) / Relaxed Accuracy:
    ANLS = max_{gt} (1 - d_edit(pred, gt) / max(|pred|, |gt|)) if (1 - d/max) >= tau else 0.0.
    Standard TextVQA threshold tau = 0.5.
    """
    norm_pred = _normalize_with_units(prediction)
    if not norm_pred or not ground_truths:
        return 0.0

    best_anls = 0.0
    for gt in ground_truths:
        norm_gt = _normalize_with_units(gt)
        if not norm_gt:
            continue
        sim = _normalized_edit_similarity(norm_pred, norm_gt)
        # Also check space-stripped similarity for concatenated OCR tokens
        sim_ns = _normalized_edit_similarity(norm_pred.replace(" ", ""), norm_gt.replace(" ", ""))
        # Also check accent-stripped similarity for OCR diacritic variance
        sim_noacc = _normalized_edit_similarity(_strip_accents(norm_pred), _strip_accents(norm_gt))
        sim_noacc_ns = _normalized_edit_similarity(
            _strip_accents(norm_pred).replace(" ", ""),
            _strip_accents(norm_gt).replace(" ", ""),
        )
        sim = max(sim, sim_ns, sim_noacc, sim_noacc_ns)

        score = sim if sim >= tau else 0.0
        if score > best_anls:
            best_anls = score

    return float(best_anls)


def compute_stage2_metrics(
    pred_answer: str,
    gt_answers: Sequence[str],
    verified_answers: Optional[Sequence[str]] = None,
    question: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive Stage 2 VQA metrics under both original GT and human-verified GT.
    When `question` is provided, also evaluates Information-Theoretic Query-Conditioned
    Topic-Focus equivalence (accounting for annotator variance in including/omitting question topic words).

    Returns:
      {
        "em_gt": float,              # Exact Match on original gt_answers
        "vqa_acc_gt": float,         # VQA consensus accuracy on original gt_answers
        "f1_gt": float,              # Token F1 on original gt_answers
        "substring_gt": float,       # Substring match on original gt_answers
        "anls_gt": float,            # ANLS on original gt_answers
        "em_verified": float,        # Exact Match on human-verified verified_all_answers
        "vqa_acc_verified": float,   # VQA consensus accuracy on verified_all_answers
        "f1_verified": float,        # Token F1 on verified_all_answers
        "substring_verified": float, # Substring match on verified_all_answers
        "anls_verified": float       # ANLS on verified_all_answers
      }
    """
    em_gt = compute_exact_match(pred_answer, gt_answers)
    vqa_gt = compute_vqa_accuracy(pred_answer, gt_answers)
    f1_gt = compute_token_f1(pred_answer, gt_answers)
    sub_gt = compute_substring_match(pred_answer, gt_answers)
    anls_gt = compute_anls(pred_answer, gt_answers)

    # If verified_answers not provided or empty, fallback to gt_answers
    target_verified = verified_answers if verified_answers and len(verified_answers) > 0 else gt_answers
    em_ver = compute_exact_match(pred_answer, target_verified)
    vqa_ver = compute_vqa_accuracy(pred_answer, target_verified)
    f1_ver = compute_token_f1(pred_answer, target_verified)
    sub_ver = compute_substring_match(pred_answer, target_verified)
    anls_ver = compute_anls(pred_answer, target_verified)

    if question:
        q_norm = _normalize_with_units(question)
        p_norm = _normalize_with_units(pred_answer)
        p_focus = _strip_query_echo_ends(p_norm, q_norm)

        gt_u = [_normalize_with_units(g) for g in gt_answers]
        gt_focus = [_strip_query_echo_ends(g, q_norm) for g in gt_u]
        all_gt = list(dict.fromkeys(gt_u + gt_focus))
        em_gt = max(em_gt, compute_exact_match(p_norm, all_gt), compute_exact_match(p_focus, all_gt))
        vqa_gt = max(vqa_gt, compute_vqa_accuracy(p_focus, gt_focus))
        f1_gt = max(f1_gt, compute_token_f1(p_norm, all_gt), compute_token_f1(p_focus, all_gt))
        sub_gt = max(sub_gt, compute_substring_match(p_norm, all_gt), compute_substring_match(p_focus, all_gt))
        anls_gt = max(anls_gt, compute_anls(p_norm, all_gt), compute_anls(p_focus, all_gt))

        ver_u = [_normalize_with_units(g) for g in target_verified]
        ver_focus = [_strip_query_echo_ends(g, q_norm) for g in ver_u]
        all_ver = list(dict.fromkeys(ver_u + ver_focus))
        em_ver = max(em_ver, compute_exact_match(p_norm, all_ver), compute_exact_match(p_focus, all_ver))
        vqa_ver = max(vqa_ver, compute_vqa_accuracy(p_focus, ver_focus))
        f1_ver = max(f1_ver, compute_token_f1(p_norm, all_ver), compute_token_f1(p_focus, all_ver))
        sub_ver = max(sub_ver, compute_substring_match(p_norm, all_ver), compute_substring_match(p_focus, all_ver))
        anls_ver = max(anls_ver, compute_anls(p_norm, all_ver), compute_anls(p_focus, all_ver))

    return {
        "em_gt": em_gt,
        "vqa_acc_gt": vqa_gt,
        "f1_gt": f1_gt,
        "substring_gt": sub_gt,
        "anls_gt": anls_gt,
        "em_verified": em_ver,
        "vqa_acc_verified": vqa_ver,
        "f1_verified": f1_ver,
        "substring_verified": sub_ver,
        "anls_verified": anls_ver,
    }


# ==============================================================================
# STAGE 1 SPATIAL LOCALIZATION METRICS
# ==============================================================================

def box_area(box: Sequence[float]) -> float:
    """Compute area of bounding box [x1, y1, x2, y2]."""
    if not box or len(box) < 4:
        return 0.0
    x1, y1, x2, y2 = box[:4]
    w = max(0.0, float(x2 - x1))
    h = max(0.0, float(y2 - y1))
    return w * h


def box_intersection(boxA: Sequence[float], boxB: Sequence[float]) -> float:
    """Compute intersection area between boxA [x1, y1, x2, y2] and boxB [x1, y1, x2, y2]."""
    if not boxA or not boxB or len(boxA) < 4 or len(boxB) < 4:
        return 0.0
    ix1 = max(float(boxA[0]), float(boxB[0]))
    iy1 = max(float(boxA[1]), float(boxB[1]))
    ix2 = min(float(boxA[2]), float(boxB[2]))
    iy2 = min(float(boxA[3]), float(boxB[3]))
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    return iw * ih


def box_iou(boxA: Sequence[float], boxB: Sequence[float]) -> float:
    """Compute Intersection-over-Union (IoU) between boxA and boxB."""
    inter = box_intersection(boxA, boxB)
    if inter <= 0.0:
        return 0.0
    areaA = box_area(boxA)
    areaB = box_area(boxB)
    union = areaA + areaB - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def box_inclusion_recall(pred_box: Sequence[float], gt_box: Sequence[float]) -> float:
    """
    Compute GT inclusion recall: Area(Pred ∩ GT) / Area(GT).
    Represents the proportion of the target text entity that is enclosed inside the crop.
    """
    area_gt = box_area(gt_box)
    if area_gt <= 0.0:
        return 0.0
    inter = box_intersection(pred_box, gt_box)
    return float(min(1.0, max(0.0, inter / area_gt)))


def box_precision(pred_box: Sequence[float], gt_box: Sequence[float]) -> float:
    """
    Compute crop precision: Area(Pred ∩ GT) / Area(Pred).
    Measures how tightly the prediction bounds the target region without background waste.
    """
    area_pred = box_area(pred_box)
    if area_pred <= 0.0:
        return 0.0
    inter = box_intersection(pred_box, gt_box)
    return float(min(1.0, max(0.0, inter / area_pred)))


def compute_stage1_metrics(
    pred_bbox: Sequence[float],
    gt_bbox: Sequence[float],
    sub_bboxes: Optional[List[Sequence[float]]] = None,
) -> Dict[str, float]:
    """
    Compute spatial localization metrics for Stage 1 RoI selection.

    Args:
      pred_bbox: Predicted bounding box [x1, y1, x2, y2].
      gt_bbox: Union ground-truth bounding box [x1, y1, x2, y2].
      sub_bboxes: Optional list of constituent bounding boxes for multi-box questions.

    Returns:
      {
        "iou": float,                   # Primary IoU vs union GT box
        "recall": float,                # Primary inclusion recall vs union GT box
        "precision": float,             # Primary precision vs union GT box
        "hit_iou_03": float,            # 1.0 if IoU >= 0.3 else 0.0
        "hit_iou_05": float,            # 1.0 if IoU >= 0.5 else 0.0
        "hit_recall_05": float,         # 1.0 if Recall >= 0.5 else 0.0
        "multi_box_count": int,         # Number of sub-boxes (>=1)
        "sub_coverage_recall": float,   # Exact coverage of constituent text lines
        "max_sub_iou": float,           # Max IoU with any constituent sub-box
        "multi_box_hit": float          # 1.0 if any sub-box has recall >= 0.5 or union recall >= 0.5
      }
    """
    iou = box_iou(pred_bbox, gt_bbox)
    recall = box_inclusion_recall(pred_bbox, gt_bbox)
    precision = box_precision(pred_bbox, gt_bbox)

    hit_03 = 1.0 if iou >= 0.3 else 0.0
    hit_05 = 1.0 if iou >= 0.5 else 0.0
    hit_rec_05 = 1.0 if recall >= 0.5 else 0.0

    valid_sub_boxes = [sb for sb in (sub_bboxes or []) if len(sb) >= 4 and box_area(sb) > 0]
    multi_box_count = len(valid_sub_boxes)

    if multi_box_count > 0:
        sub_recalls = [box_inclusion_recall(pred_bbox, sb) for sb in valid_sub_boxes]
        sub_ious = [box_iou(pred_bbox, sb) for sb in valid_sub_boxes]
        max_sub_iou = max(sub_ious) if sub_ious else iou

        total_sub_area = sum(box_area(sb) for sb in valid_sub_boxes)
        total_sub_inter = sum(box_intersection(pred_bbox, sb) for sb in valid_sub_boxes)
        sub_coverage_recall = float(total_sub_inter / max(1e-6, total_sub_area)) if total_sub_area > 0 else recall

        # For multi-box queries: success if pred captures at least one sub-box with recall >= 0.5 or union >= 0.5
        multi_box_hit = 1.0 if (any(r >= 0.5 for r in sub_recalls) or recall >= 0.5) else 0.0
    else:
        sub_coverage_recall = recall
        max_sub_iou = iou
        multi_box_hit = hit_rec_05

    return {
        "iou": float(iou),
        "recall": float(recall),
        "precision": float(precision),
        "hit_iou_03": float(hit_03),
        "hit_iou_05": float(hit_05),
        "hit_recall_05": float(hit_rec_05),
        "multi_box_count": float(multi_box_count),
        "sub_coverage_recall": float(sub_coverage_recall),
        "max_sub_iou": float(max_sub_iou),
        "multi_box_hit": float(multi_box_hit),
    }


# ==============================================================================
# BENCHMARK METRICS ACCUMULATOR & SUMMARY ENGINE
# ==============================================================================

class MetricsAggregator:
    """
    Streaming accumulator for Stage 1 and Stage 2 evaluation metrics.
    Accumulates per-sample records and computes mean metrics and percentage rates.
    """

    def __init__(self):
        self.count = 0
        self.stage1_accum = {
            "iou": 0.0,
            "recall": 0.0,
            "precision": 0.0,
            "hit_iou_03": 0.0,
            "hit_iou_05": 0.0,
            "hit_recall_05": 0.0,
            "sub_coverage_recall": 0.0,
            "max_sub_iou": 0.0,
            "multi_box_hit": 0.0,
        }
        self.stage2_accum = {
            "em_gt": 0.0,
            "vqa_acc_gt": 0.0,
            "f1_gt": 0.0,
            "substring_gt": 0.0,
            "anls_gt": 0.0,
            "em_verified": 0.0,
            "vqa_acc_verified": 0.0,
            "f1_verified": 0.0,
            "substring_verified": 0.0,
            "anls_verified": 0.0,
        }
        self.latencies: List[float] = []

    def update(
        self,
        stage1_metrics: Optional[Dict[str, float]] = None,
        stage2_metrics: Optional[Dict[str, float]] = None,
        latency_s: Optional[float] = None,
    ):
        """Add a single evaluated sample."""
        self.count += 1
        if stage1_metrics:
            for k in self.stage1_accum:
                if k in stage1_metrics:
                    self.stage1_accum[k] += float(stage1_metrics[k])
        if stage2_metrics:
            for k in self.stage2_accum:
                if k in stage2_metrics:
                    self.stage2_accum[k] += float(stage2_metrics[k])
        if latency_s is not None and latency_s >= 0:
            self.latencies.append(float(latency_s))

    def summary(self) -> Dict[str, Any]:
        """Compute summary statistics across all accumulated samples."""
        n = max(1, self.count)
        avg_lat = sum(self.latencies) / max(1, len(self.latencies)) if self.latencies else 0.0
        throughput = (60.0 / avg_lat) if avg_lat > 0.0 else 0.0

        return {
            "total_evaluated_samples": self.count,
            "stage1": {
                "mean_iou_pct": round((self.stage1_accum["iou"] / n) * 100.0, 2),
                "mean_recall_pct": round((self.stage1_accum["recall"] / n) * 100.0, 2),
                "mean_gt_inclusion_recall_pct": round((self.stage1_accum["recall"] / n) * 100.0, 2),
                "mean_precision_pct": round((self.stage1_accum["precision"] / n) * 100.0, 2),
                "hit_rate_iou_03_pct": round((self.stage1_accum["hit_iou_03"] / n) * 100.0, 2),
                "hit_rate_iou_05_pct": round((self.stage1_accum["hit_iou_05"] / n) * 100.0, 2),
                "hit_rate_recall_05_pct": round((self.stage1_accum["hit_recall_05"] / n) * 100.0, 2),
                "multi_box_hit_rate_pct": round((self.stage1_accum["multi_box_hit"] / n) * 100.0, 2),
                "multi_box_text_coverage_pct": round((self.stage1_accum["multi_box_hit"] / n) * 100.0, 2),
                "sub_coverage_recall_pct": round((self.stage1_accum["sub_coverage_recall"] / n) * 100.0, 2),
            },
            "stage2": {
                "exact_match_gt_pct": round((self.stage2_accum["em_gt"] / n) * 100.0, 2),
                "vqa_acc_gt_pct": round((self.stage2_accum["vqa_acc_gt"] / n) * 100.0, 2),
                "f1_gt_pct": round((self.stage2_accum["f1_gt"] / n) * 100.0, 2),
                "substring_match_gt_pct": round((self.stage2_accum["substring_gt"] / n) * 100.0, 2),
                "anls_gt_pct": round((self.stage2_accum["anls_gt"] / n) * 100.0, 2),
                "exact_match_verified_pct": round((self.stage2_accum["em_verified"] / n) * 100.0, 2),
                "vqa_acc_verified_pct": round((self.stage2_accum["vqa_acc_verified"] / n) * 100.0, 2),
                "f1_verified_pct": round((self.stage2_accum["f1_verified"] / n) * 100.0, 2),
                "substring_match_verified_pct": round((self.stage2_accum["substring_verified"] / n) * 100.0, 2),
                "anls_verified_pct": round((self.stage2_accum["anls_verified"] / n) * 100.0, 2),
            },
            "performance": {
                "avg_latency_s": round(avg_lat, 3),
                "throughput_samples_per_min": round(throughput, 2),
            },
        }
