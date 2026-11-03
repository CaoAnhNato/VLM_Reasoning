# -*- coding: utf-8 -*-
"""
Strict A* Evaluation Metrics Engine for ViTextVQA and CERV Benchmark.

Provides:
- Stage 1 Spatial Localization Metrics:
  - Intersection-over-Union (IoU)
  - Ground-Truth Inclusion Recall: Area(Pred ∩ GT) / Area(GT)
  - Precision: Area(Pred ∩ GT) / Area(Pred)
  - Hit rates at IoU >= 0.3, IoU >= 0.5, and Recall >= 0.5
  - Multi-box question support (sub_bboxes distributed text entities)
- Stage 2 End-to-End VQA Accuracy Metrics (Strict A* Protocol — Zero Heuristic Relaxation):
  - Exact Match (EM): strict normalized equality (norm_pred == norm_gt)
  - VQA Consensus Accuracy: min(1.0, match_count / 3.0)
  - Token-level F1 Score: standard SQuAD bag-of-words Counter harmonic mean (2*P*R / (P+R))
  - ANLS (Average Normalized Levenshtein Similarity): standard ICDAR DocVQA edit similarity with tau=0.5
  - Substring Match: independent containment indicator (1.0 if norm_gt in norm_pred else 0.0)
  - Dual evaluation against original gt_answers and human-verified verified_all_answers
- MetricsAggregator:
  - Streaming accumulator for sample-by-sample checkpointing and summary computation.
"""

from collections import Counter
import string
import unicodedata
from typing import Any, Dict, List, Optional, Sequence


# ==============================================================================
# TEXT NORMALIZATION PROTOCOL (OFFICIAL SQuAD / ViTextVQA STANDARD)
# ==============================================================================

def normalize_answer(text: Any) -> str:
    """
    Standardize answer string following the official ViTextVQA / SQuAD benchmark protocol:
    - Unicode NFC normalization
    - Convert to lower-case string
    - Replace punctuation marks with whitespace to preserve token boundaries
    - Collapse multiple whitespaces into a single space
    """
    if text is None:
        return ""
    text_str = unicodedata.normalize("NFC", str(text)).lower()
    text_clean = "".join(ch if ch not in set(string.punctuation) else " " for ch in text_str)
    return " ".join(text_clean.split())


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


# ==============================================================================
# STAGE 2 VQA ACCURACY METRICS (STRICT A* EVALUATION PROTOCOL)
# ==============================================================================

def compute_exact_match(prediction: str, ground_truths: Sequence[str]) -> float:
    """
    Exact Match (EM): Returns 1.0 if normalized prediction exactly equals any normalized GT, else 0.0.
    """
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    norm_pred = normalize_answer(prediction)
    if not norm_pred or not ground_truths:
        return 0.0
    for gt in ground_truths:
        norm_gt = normalize_answer(gt)
        if norm_gt and norm_pred == norm_gt:
            return 1.0
    return 0.0


def compute_vqa_accuracy(prediction: str, ground_truths: Sequence[str]) -> float:
    """
    Standard VQA Consensus Accuracy:
    Acc_VQA(pred, GTs) = min(1.0, (# matching GTs) / 3.0) when >= 3 GTs are present,
    or 1.0 if >= 1 match when fewer than 3 GTs are present.
    """
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    norm_pred = normalize_answer(prediction)
    if not norm_pred or not ground_truths:
        return 0.0

    valid_gts = [normalize_answer(gt) for gt in ground_truths if normalize_answer(gt)]
    if not valid_gts:
        return 0.0

    match_count = sum(1 for gt_norm in valid_gts if norm_pred == gt_norm)
    if len(valid_gts) >= 3:
        return min(1.0, float(match_count) / 3.0)
    return 1.0 if match_count >= 1 else 0.0


def compute_token_f1(prediction: str, ground_truths: Sequence[str]) -> float:
    """
    Standard SQuAD / ViTextVQA bag-of-words Token F1 score:
    Uses multiset (Counter) token intersection between normalized prediction and normalized GT:
      P = |Pred ∩ GT| / |Pred|,  R = |Pred ∩ GT| / |GT|,  F1 = 2*P*R / (P + R)
    No substring override, no accent stripping, and no length/recall threshold relaxation.
    """
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    norm_pred = normalize_answer(prediction)
    pred_tokens = norm_pred.split()
    if not pred_tokens or not ground_truths:
        return 0.0

    best_f1 = 0.0
    pred_counter = Counter(pred_tokens)

    for gt in ground_truths:
        norm_gt = normalize_answer(gt)
        gt_tokens = norm_gt.split()
        if not gt_tokens:
            continue
        common = pred_counter & Counter(gt_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = float(num_same) / len(pred_tokens)
        recall = float(num_same) / len(gt_tokens)
        f1 = (2.0 * precision * recall) / (precision + recall)
        if f1 > best_f1:
            best_f1 = f1

    return float(best_f1)


def compute_anls(prediction: str, ground_truths: Sequence[str], tau: float = 0.5) -> float:
    """
    Standard ICDAR DocVQA / TextVQA Average Normalized Levenshtein Similarity (ANLS):
      s(pred, gt) = 1 - d_edit(norm_pred, norm_gt) / max(|norm_pred|, |norm_gt|)
      ANLS = max_{gt} (s(pred, gt) if s(pred, gt) >= tau else 0.0)
    No substring override and no sub-span sliding window inflation.
    """
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    norm_pred = normalize_answer(prediction)
    if not norm_pred or not ground_truths:
        return 0.0

    best_anls = 0.0
    for gt in ground_truths:
        norm_gt = normalize_answer(gt)
        if not norm_gt:
            continue
        sim = _normalized_edit_similarity(norm_pred, norm_gt)
        score = sim if sim >= tau else 0.0
        if score > best_anls:
            best_anls = score

    return float(best_anls)


def compute_substring_match(prediction: str, ground_truths: Sequence[str]) -> float:
    """
    Independent Substring Containment Indicator:
    Returns 1.0 if any non-empty normalized ground truth is a contiguous substring
    of the normalized prediction, else 0.0.
    Reported strictly as an auxiliary diagnostic metric without inflating EM, F1, or ANLS.
    """
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    norm_pred = normalize_answer(prediction)
    if not norm_pred or not ground_truths:
        return 0.0

    for gt in ground_truths:
        norm_gt = normalize_answer(gt)
        if norm_gt and norm_gt in norm_pred:
            return 1.0

    return 0.0


def compute_stage2_metrics(
    pred_answer: str,
    gt_answers: Sequence[str],
    verified_answers: Optional[Sequence[str]] = None,
    question: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute Stage 2 VQA metrics under both original GT and human-verified GT
    using the Strict A* Evaluation Protocol (no query-echo stripping or regex rewriting).

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
    del question  # Strictly unused: evaluation never mutates prediction based on question tokens

    em_gt = compute_exact_match(pred_answer, gt_answers)
    vqa_gt = compute_vqa_accuracy(pred_answer, gt_answers)
    f1_gt = compute_token_f1(pred_answer, gt_answers)
    sub_gt = compute_substring_match(pred_answer, gt_answers)
    anls_gt = compute_anls(pred_answer, gt_answers)

    target_verified = verified_answers if verified_answers and len(verified_answers) > 0 else gt_answers
    em_ver = compute_exact_match(pred_answer, target_verified)
    vqa_ver = compute_vqa_accuracy(pred_answer, target_verified)
    f1_ver = compute_token_f1(pred_answer, target_verified)
    sub_ver = compute_substring_match(pred_answer, target_verified)
    anls_ver = compute_anls(pred_answer, target_verified)

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
