# -*- coding: utf-8 -*-
"""
Official Vietnamese Character/Diacritic Decoder, Tight-Crop Mask-to-Polygon Extractor,
Polygon NMS, and Top-Left to Bottom-Right (TL-BR) Spatial Sorting for SwinTextSpotter.

Source:
- https://github.com/mxin262/SwinTextSpotter (detectron2/utils/visualizer_vintext.py & text_evaluation.py)
- ViTextVQA paper (arXiv:2404.10652, Section 3.1, 4.1, 6.4)
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from shapely.geometry import Polygon
import torch
import torch.nn.functional as F


# 106 character + diacritic/tone tokens (indices 1..106; 0 is EOS/SOS)
CTLABELS: List[str] = [
    " ", "!", '"', "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";", "<", "=", ">", "?",
    "@", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O",
    "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "[", "\\", "]", "^", "_",
    "`", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o",
    "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "{", "|", "}", "~", "ˋ",
    "ˊ", "﹒", "ˀ", "˜", "ˇ", "ˆ", "˒", "‑", "´", "~",
]

DICTIONARY_STR = (
    "aàáạảãâầấậẩẫăằắặẳẵAÀÁẠẢÃĂẰẮẶẲẴÂẦẤẬẨẪ"
    "eèéẹẻẽêềếệểễEÈÉẸẺẼÊỀẾỆỂỄ"
    "oòóọỏõôồốộổỗơờớợởỡOÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ"
    "iìíịỉĩIÌÍỊỈĨuùúụủũưừứựửữƯỪỨỰỬỮUÙÚỤỦŨyỳýỵỷỹYỲÝỴỶỸ"
)


def _make_groups() -> List[List[str]]:
    groups: List[List[str]] = []
    i = 0
    while i < len(DICTIONARY_STR) - 5:
        groups.append([c for c in DICTIONARY_STR[i : i + 6]])
        i += 6
    return groups


GROUPS: List[List[str]] = _make_groups()
TONES: List[str] = ["", "ˋ", "ˊ", "﹒", "ˀ", "˜"]
SOURCES: List[str] = ["ă", "â", "Ă", "Â", "ê", "Ê", "ô", "ơ", "Ô", "Ơ", "ư", "Ư", "Đ", "đ"]
TARGETS: List[str] = ["aˇ", "aˆ", "Aˇ", "Aˆ", "eˆ", "Eˆ", "oˆ", "o˒", "Oˆ", "O˒", "u˒", "U˒", "D-", "d‑"]


def correct_tone_position(word: str) -> str:
    """
    Official SwinTextSpotter (text_evaluation.py) tone vowel selector.
    Identifies which vowel character in `word` receives the trailing tone mark.
    """
    word = word[:-1]
    first_ord_char = ""
    second_order_char = ""
    for char in word:
        for group in GROUPS:
            if char in group:
                second_order_char = first_ord_char
                first_ord_char = group[0]
    if len(word) >= 1 and word[-1] == first_ord_char and second_order_char != "":
        pair_chars = ["qu", "Qu", "qU", "QU", "gi", "Gi", "gI", "GI"]
        for pair in pair_chars:
            if pair in word and second_order_char in ["u", "U", "i", "I"]:
                return first_ord_char
        return second_order_char
    return first_ord_char


def vintext_decoder(recognition: str) -> str:
    """
    Official SwinTextSpotter (text_evaluation.py lines 152-166) Vietnamese sequence decoder.
    """
    for idx, char in enumerate(TARGETS):
        recognition = recognition.replace(char, SOURCES[idx])
    if len(recognition) < 1:
        return recognition
    if recognition[-1] in TONES:
        if len(recognition) < 2:
            return recognition
        replace_char = correct_tone_position(recognition)
        tone = recognition[-1]
        recognition = recognition[:-1]
        for group in GROUPS:
            if replace_char in group:
                recognition = recognition.replace(replace_char, group[TONES.index(tone)])
    return recognition


def decode_recognition_tokens(rec_tensor: torch.Tensor) -> str:
    """Decode a (max_length,) integer tensor from SequencePredictor into Vietnamese text."""
    chars: List[str] = []
    for c_val in rec_tensor.tolist():
        c = int(c_val)
        if 0 < c < len(CTLABELS):
            chars.append(CTLABELS[c - 1])
        elif c == 0:
            break
    return vintext_decoder("".join(chars)).strip()


def extract_tight_polygon_and_bbox(
    mask_28x28: torch.Tensor,
    box_xyxy: torch.Tensor,
    img_h: int,
    img_w: int,
    threshold: float = 0.5,
) -> Optional[Tuple[List[List[int]], List[int]]]:
    """
    Paste a single 28x28 soft mask ONLY within its tight bounding box slice
    (mathematically identical to Detectron2 _do_paste_mask with skip_empty=True),
    avoiding full-image mask allocation and saving >99% RAM.

    Returns:
        (polygon_pts, tight_bbox_xyxy) where polygon_pts is [[x, y], ...] and
        tight_bbox_xyxy is [x1, y1, x2, y2], or None if contour is invalid.
    """
    x0, y0, x1, y1 = [float(v) for v in box_xyxy.tolist()]
    x0_int = max(0, int(np.floor(x0) - 1))
    y0_int = max(0, int(np.floor(y0) - 1))
    x1_int = min(img_w, int(np.ceil(x1) + 1))
    y1_int = min(img_h, int(np.ceil(y1) + 1))

    crop_w = x1_int - x0_int
    crop_h = y1_int - y0_int
    if crop_w < 2 or crop_h < 2 or (x1 - x0) < 1.0 or (y1 - y0) < 1.0:
        return None

    img_y = (torch.arange(y0_int, y1_int, dtype=torch.float32) + 0.5 - y0) / (y1 - y0) * 2.0 - 1.0
    img_x = (torch.arange(x0_int, x1_int, dtype=torch.float32) + 0.5 - x0) / (x1 - x0) * 2.0 - 1.0

    gx = img_x[None, :].expand(crop_h, crop_w)
    gy = img_y[:, None].expand(crop_h, crop_w)
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)  # (1, crop_h, crop_w, 2)

    mask_crop = F.grid_sample(
        mask_28x28.float().unsqueeze(0).unsqueeze(0),
        grid,
        align_corners=False,
    )[0, 0]

    binary_crop = (mask_crop >= threshold).numpy().astype(np.uint8)
    contours, _ = cv2.findContours(binary_crop, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    # Pick the largest contour if multiple fragments exist
    contour = max(contours, key=cv2.contourArea)
    pts = contour.reshape(-1, 2)
    if len(pts) < 4:
        return None

    # Shift back to original image coordinates
    pts[:, 0] += x0_int
    pts[:, 1] += y0_int

    tx1 = int(np.clip(pts[:, 0].min(), 0, img_w))
    ty1 = int(np.clip(pts[:, 1].min(), 0, img_h))
    tx2 = int(np.clip(pts[:, 0].max(), 0, img_w))
    ty2 = int(np.clip(pts[:, 1].max(), 0, img_h))
    if tx2 <= tx1 or ty2 <= ty1:
        return None

    return pts.tolist(), [tx1, ty1, tx2, ty2]


def polygon_nms(
    detections: List[Dict[str, Any]],
    iou_thresh: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Polygon Non-Maximum Suppression matching visualizer_vintext.py (py_cpu_pnms),
    also suppressing nearly-contained duplicate text fragments (IoMin >= 0.85).
    """
    if not detections:
        return []

    ordered = sorted(detections, key=lambda d: d["score"], reverse=True)
    polys: List[Optional[Polygon]] = []
    for d in ordered:
        try:
            p = Polygon(d["polygon"])
            if not p.is_valid:
                p = p.buffer(0)
            polys.append(p if (p.is_valid and p.area > 1.0) else None)
        except Exception:
            polys.append(None)

    keep_indices: List[int] = []
    suppressed = [False] * len(ordered)

    for i in range(len(ordered)):
        if suppressed[i] or polys[i] is None:
            continue
        keep_indices.append(i)
        pi = polys[i]
        area_i = pi.area
        for j in range(i + 1, len(ordered)):
            if suppressed[j] or polys[j] is None:
                continue
            pj = polys[j]
            if not pi.bounds or not pj.bounds:
                continue
            # Fast AABB reject before exact polygon intersection
            bx1, by1, bx2, by2 = ordered[i]["bbox"]
            qx1, qy1, qx2, qy2 = ordered[j]["bbox"]
            if bx2 <= qx1 or qx2 <= bx1 or by2 <= qy1 or qy2 <= by1:
                continue
            try:
                inter_area = pi.intersection(pj).area
            except Exception:
                inter_area = 0.0
            if inter_area <= 0.0:
                continue
            area_j = pj.area
            union_area = area_i + area_j - inter_area
            iou = inter_area / max(union_area, 1e-6)
            io_min = inter_area / max(min(area_i, area_j), 1e-6)
            if iou >= iou_thresh or (io_min >= 0.85 and ordered[i]["text"].lower() == ordered[j]["text"].lower()):
                suppressed[j] = True

    return [ordered[idx] for idx in keep_indices]


def sort_detections_tl_br(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort OCR text boxes from Top-Left to Bottom-Right (TL-BR) reading order
    as specified in ViTextVQA (arXiv:2404.10652, Section 6.4).
    Groups boxes into horizontal lines when vertical center distance is within 0.6 * median height,
    then sorts left-to-right within each line.
    """
    if len(detections) <= 1:
        return list(detections)

    items = []
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        h = max(1.0, float(y2 - y1))
        items.append((cy, cx, h, d))

    items.sort(key=lambda t: t[0])
    lines: List[List[Tuple[float, float, float, Dict[str, Any]]]] = []
    for item in items:
        cy, cx, h, d = item
        placed = False
        for line in lines:
            mean_cy = sum(x[0] for x in line) / len(line)
            mean_h = sum(x[2] for x in line) / len(line)
            if abs(cy - mean_cy) <= 0.6 * max(h, mean_h):
                line.append(item)
                placed = True
                break
        if not placed:
            lines.append([item])

    lines.sort(key=lambda line: sum(x[0] for x in line) / len(line))
    sorted_dets: List[Dict[str, Any]] = []
    for line in lines:
        line.sort(key=lambda x: x[1])
        for _, _, _, d in line:
            sorted_dets.append(d)

    for rank_idx, d in enumerate(sorted_dets):
        d["reading_order"] = rank_idx
    return sorted_dets


def _are_words_proximate_same_line(
    box_left: List[int],
    box_right: List[int],
    max_gap_height_ratio: float = 1.35,
    min_vertical_overlap: float = 0.35,
    max_cy_diff_ratio: float = 0.55,
    min_scale_ratio: float = 0.42,
) -> bool:
    """
    Check if two word bounding boxes (`box_left` to the left of `box_right`) belong to
    the same proximate horizontal/slightly-inclined phrase segment without jumping across
    distant columns or unrelated signboards.
    """
    lx1, ly1, lx2, ly2 = box_left
    rx1, ry1, rx2, ry2 = box_right

    lh = max(1.0, float(ly2 - ly1))
    rh = max(1.0, float(ry2 - ry1))
    lw = max(1.0, float(lx2 - lx1))
    rw = max(1.0, float(rx2 - rx1))
    max_h = max(lh, rh)
    min_h = min(lh, rh)

    # 1. Character height scale consistency (accounting for Vietnamese diacritics)
    if (min_h / max_h) < min_scale_ratio:
        return False

    # 2. Vertical alignment (vertical overlap + center-Y alignment)
    lcy = 0.5 * (ly1 + ly2)
    rcy = 0.5 * (ry1 + ry2)
    if abs(lcy - rcy) > max_cy_diff_ratio * max_h:
        return False

    inter_y = max(0.0, float(min(ly2, ry2) - max(ly1, ry1)))
    if (inter_y / min_h) < min_vertical_overlap:
        return False

    # 3. Strict horizontal proximity (only merge nearby words, never distant columns)
    gap_x = float(rx1 - lx2)
    if gap_x < -0.45 * min(lw, rw):
        return False
    if gap_x > max_gap_height_ratio * max_h:
        return False

    return True


def group_proximate_text_lines(
    sorted_detections: List[Dict[str, Any]],
    max_gap_height_ratio: float = 1.35,
) -> List[Dict[str, Any]]:
    """
    Group word-level OCR detections into proximate phrase / line bounding boxes.
    Only merges spatially close words on the same line (`gap_x <= max_gap_height_ratio * max_h`),
    keeping distant text blocks strictly separate.
    """
    n = len(sorted_detections)
    if n == 0:
        return []

    # Sort candidates primarily by vertical band then left-to-right x-center
    indexed = list(enumerate(sorted_detections))

    # Build right-neighbor chaining on the same line
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # For each word i, link it to its nearest valid right-hand neighbor j on the same line
    for i in range(n):
        bi = sorted_detections[i]["bbox"]
        icx = 0.5 * (bi[0] + bi[2])
        best_j = -1
        best_gap = float("inf")
        for j in range(n):
            if i == j:
                continue
            bj = sorted_detections[j]["bbox"]
            jcx = 0.5 * (bj[0] + bj[2])
            if jcx < icx or (jcx == icx and j < i):
                continue
            if _are_words_proximate_same_line(
                bi, bj, max_gap_height_ratio=max_gap_height_ratio
            ):
                gap = float(bj[0] - bi[2])
                if gap < best_gap:
                    best_gap = gap
                    best_j = j
        if best_j != -1:
            union(i, best_j)

    clusters_map: Dict[int, List[int]] = {}
    for idx in range(n):
        root = find(idx)
        clusters_map.setdefault(root, []).append(idx)

    merged_lines: List[Dict[str, Any]] = []
    for word_indices in clusters_map.values():
        # Sort words left-to-right within the merged phrase/line
        word_indices.sort(
            key=lambda idx: (
                0.5 * (sorted_detections[idx]["bbox"][0] + sorted_detections[idx]["bbox"][2]),
                sorted_detections[idx]["bbox"][0],
            )
        )
        words = [sorted_detections[idx] for idx in word_indices]
        x1 = min(w["bbox"][0] for w in words)
        y1 = min(w["bbox"][1] for w in words)
        x2 = max(w["bbox"][2] for w in words)
        y2 = max(w["bbox"][3] for w in words)
        phrase_text = " ".join(w["text"] for w in words)
        mean_score = sum(float(w["score"]) for w in words) / len(words)

        merged_lines.append(
            {
                "text": phrase_text,
                "score": round(mean_score, 4),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "word_indices": word_indices,
                "word_texts": [w["text"] for w in words],
                "num_words": len(words),
            }
        )

    # Sort merged lines in Top-Left to Bottom-Right reading order
    merged_lines = sort_detections_tl_br(merged_lines)
    for line_id, line_dict in enumerate(merged_lines):
        line_dict["line_id"] = line_id
        for w_idx in line_dict["word_indices"]:
            sorted_detections[w_idx]["line_id"] = line_id

    return merged_lines

