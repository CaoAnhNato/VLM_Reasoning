# -*- coding: utf-8 -*-
"""
Shared Helper Utilities, Canonical Mathematical Oracles, and Mock Contracts for E2E Tests.
"""

from dataclasses import dataclass, field
import json
import os
import string
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))


# ==============================================================================
# CANONICAL MATHEMATICAL ORACLES FOR SPATIAL METRICS
# ==============================================================================

def calculate_box_area(box: List[Union[int, float]]) -> float:
    """Calculate area of [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = box
    w = max(0.0, float(x2 - x1))
    h = max(0.0, float(y2 - y1))
    return w * h


def calculate_iou(
    boxA: List[Union[int, float]],
    boxB: List[Union[int, float]],
) -> float:
    """
    Authoritative spatial Intersection-over-Union (IoU) calculation:
    IoU = Area(Intersection) / Area(Union).
    Returns 0.0 if union is 0.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0.0, float(xB - xA))
    inter_h = max(0.0, float(yB - yA))
    inter_area = inter_w * inter_h

    areaA = calculate_box_area(boxA)
    areaB = calculate_box_area(boxB)
    union_area = areaA + areaB - inter_area

    if union_area <= 1e-9:
        return 0.0
    return float(inter_area / union_area)


def calculate_inclusion_recall(
    pred_box: List[Union[int, float]],
    gt_box: List[Union[int, float]],
) -> float:
    """
    Calculate GT Inclusion Recall:
    Fraction of ground truth box area enclosed inside pred_box:
    Recall = Area(Intersection(pred, gt)) / Area(gt).
    """
    xA = max(pred_box[0], gt_box[0])
    yA = max(pred_box[1], gt_box[1])
    xB = min(pred_box[2], gt_box[2])
    yB = min(pred_box[3], gt_box[3])

    inter_w = max(0.0, float(xB - xA))
    inter_h = max(0.0, float(yB - yA))
    inter_area = inter_w * inter_h

    gt_area = calculate_box_area(gt_box)
    if gt_area <= 1e-9:
        return 0.0
    return float(min(1.0, max(0.0, inter_area / gt_area)))


def calculate_sub_bboxes_hit_rate(
    pred_box: List[Union[int, float]],
    sub_bboxes: List[List[Union[int, float]]],
    iou_thresh: float = 0.3,
) -> float:
    """
    Calculate hit rate for composite ground truths:
    Fraction of sub-boxes having IoU or inclusion >= iou_thresh with pred_box.
    """
    if not sub_bboxes:
        return 0.0
    hits = 0
    for sbox in sub_bboxes:
        iou = calculate_iou(pred_box, sbox)
        rec = calculate_inclusion_recall(pred_box, sbox)
        if iou >= iou_thresh or rec >= 0.5:
            hits += 1
    return float(hits / len(sub_bboxes))


# ==============================================================================
# CANONICAL TEXT NORMALIZATION & VQA METRICS
# ==============================================================================

def normalize_vietnamese_text(text: str) -> str:
    """Standard ViTextVQA text normalizer: lowercase, replace punctuation with whitespace, normalize spaces."""
    if not text:
        return ""
    text = str(text).lower()
    text = "".join(ch if ch not in set(string.punctuation) else " " for ch in text)
    return " ".join(text.split())


def compute_vqa_exact_match(prediction: str, ground_truths: List[str]) -> float:
    """Computes exact match score (1.0 if match, 0.0 otherwise)."""
    norm_pred = normalize_vietnamese_text(prediction)
    if not norm_pred:
        return 0.0
    for gt in ground_truths:
        norm_gt = normalize_vietnamese_text(gt)
        if norm_pred == norm_gt or (bool(norm_pred) and norm_pred.replace(" ", "") == norm_gt.replace(" ", "")):
            return 1.0
    return 0.0


def compute_vqa_token_f1(prediction: str, ground_truths: List[str]) -> float:
    """Token-level F1 score against list of ground truth answers."""
    norm_pred = normalize_vietnamese_text(prediction)
    pred_tokens = norm_pred.split()
    if not pred_tokens:
        return 0.0

    best_f1 = 0.0
    for gt in ground_truths:
        norm_gt = normalize_vietnamese_text(gt)
        if bool(norm_pred) and norm_pred.replace(" ", "") == norm_gt.replace(" ", ""):
            return 1.0
        gt_tokens = norm_gt.split()
        if not gt_tokens:
            continue
        common = set(pred_tokens) & set(gt_tokens)
        num_same = sum(min(pred_tokens.count(w), gt_tokens.count(w)) for w in common)
        if num_same == 0:
            continue
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gt_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        if f1 > best_f1:
            best_f1 = f1
    return float(best_f1)


# ==============================================================================
# MOCK CONTRACTS FOR FAST, DETERMINISTIC INTEGRATION TESTING
# ==============================================================================

@dataclass
class MockPipelineResult:
    """Mirror of the PROJECT.md PipelineResult interface contract."""
    pred_text: str
    pred_raw_text: str
    stage1_bbox: List[int]
    spatial_confidence: float
    vcd_alpha_mean: float = 0.65
    entropy_mean: float = 0.42
    timing: Dict[str, float] = field(default_factory=lambda: {"stage1_ms": 12.5, "stage2_ms": 35.0, "total_ms": 47.5})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pred_text": self.pred_text,
            "pred_raw_text": self.pred_raw_text,
            "stage1_bbox": self.stage1_bbox,
            "spatial_confidence": self.spatial_confidence,
            "vcd_alpha_mean": self.vcd_alpha_mean,
            "entropy_mean": self.entropy_mean,
            "timing": self.timing,
        }
