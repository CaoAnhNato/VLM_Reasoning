# -*- coding: utf-8 -*-
"""
Strategy-based Ground-Truth Answer Region Detector & Scene-Text Classifier.

Aligns word-level OCR detections and proximate phrase/line clusters from SwinTextSpotter
with multi-question Vietnamese Ground-Truth QA pairs from ViTextVQA.
Designed for VLM ROI optimization (SG-VCD) & rapid human-in-the-loop bbox refinement:
- Supports multiple questions per image (1 image -> multiple `answer_regions`).
- Expands each matched answer into a generous `answer_region_bbox` covering the answer
  plus adjacent/proximate text on the same line/cluster (accepting surrounding noisy text
  that the VLM can naturally read and filter, while preventing clipped answers).
"""

from abc import ABC, abstractmethod
import math
import string
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple


def normalize_vi_text(text: str) -> str:
    """Normalize Vietnamese text (NFC, lowercase, strip punctuation) per ViTextVQA convention."""
    if not text:
        return ""
    s = unicodedata.normalize("NFC", str(text)).lower()
    s = "".join(ch if ch not in string.punctuation else " " for ch in s)
    return " ".join(s.split())


def strip_accents(text: str) -> str:
    """Strip Vietnamese diacritics for secondary lexical similarity computation."""
    s = unicodedata.normalize("NFD", text)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s.replace("đ", "d").replace("Đ", "D")


def levenshtein_similarity(a: str, b: str) -> float:
    """Normalized Levenshtein similarity in [0, 1]."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, lb + 1):
            temp = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = temp
    return 1.0 - (dp[lb] / float(max(la, lb)))


def lexical_token_similarity(ocr_token: str, gt_token: str) -> float:
    """
    Combined diacritic-exact (0.7 weight) and base-character (0.3 weight) Levenshtein similarity.
    Also checks sub-tokens and >=4-char stems when OCR merges or partially spots a compound token.
    """
    n_gt = normalize_vi_text(gt_token).replace(" ", "")
    if not n_gt:
        return 0.0

    norm_ocr_full = normalize_vi_text(ocr_token)
    if not norm_ocr_full:
        return 0.0

    candidates = [norm_ocr_full.replace(" ", "")]
    sub_parts = norm_ocr_full.split()
    if len(sub_parts) > 1:
        candidates.extend(sub_parts)

    best_sim = 0.0
    for cand in candidates:
        sim_exact = levenshtein_similarity(cand, n_gt)
        sim_base = levenshtein_similarity(strip_accents(cand), strip_accents(n_gt))
        sim = 0.7 * sim_exact + 0.3 * sim_base
        # Substring/prefix credit when OCR spots the main stem (>= 4 chars) of a compound token (e.g. 'wuta' in 'wutacam')
        if len(cand) >= 4 and len(n_gt) > len(cand) and (
            n_gt.startswith(cand)
            or strip_accents(n_gt).startswith(strip_accents(cand))
            or cand in n_gt
        ):
            stem_sim = 0.82 * (len(cand) / float(len(n_gt))) + 0.18
            sim = max(sim, stem_sim)
        if cand != norm_ocr_full.replace(" ", ""):
            sim *= 0.95
        if sim > best_sim:
            best_sim = sim
    return best_sim


def spatial_transition_score(box_a: List[int], box_b: List[int]) -> float:
    """
    Continuous spatial coherence score in [0, 1] between two consecutive word boxes (box_a -> box_b),
    strictly rewarding proximate left-to-right adjacency on the same line or tight line wrap within a signboard,
    and returning 0.0 for distant jumps across columns or signboards.
    """
    if box_a == box_b:
        return 1.0
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ah = max(1.0, float(ay2 - ay1))
    bh = max(1.0, float(by2 - by1))
    avg_h = 0.5 * (ah + bh)

    acx, acy = 0.5 * (ax1 + ax2), 0.5 * (ay1 + ay2)
    bcx, bcy = 0.5 * (bx1 + bx2), 0.5 * (by1 + by2)

    scale_ratio = min(ah, bh) / max(ah, bh)
    if scale_ratio < 0.35:
        return 0.0

    dy = abs(bcy - acy) / avg_h
    # Same horizontal line (b must be to the right of a and within local proximity)
    if dy <= 0.75:
        if bcx < acx - 0.25 * avg_h:
            return 0.0
        gap_x = max(0.0, float(bx1 - ax2)) / avg_h
        if gap_x > 2.6:
            return 0.0
        return float(scale_ratio * math.exp(-0.28 * gap_x - 0.35 * dy))

    # Next line wrap (b is directly below a within 2.5 line heights and horizontally aligned)
    if 0.6 <= (bcy - acy) / avg_h <= 2.5:
        horiz_span = abs(bx1 - ax1) / max(avg_h, float(ax2 - ax1))
        if horiz_span <= 3.0:
            return float(0.75 * scale_ratio * math.exp(-0.25 * horiz_span))

    return 0.0


def bbox_iou(box_a: List[int], box_b: List[int]) -> float:
    """Compute axis-aligned IoU between two [x1, y1, x2, y2] bounding boxes."""
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(1, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(1, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    return float(inter) / float(area_a + area_b - inter)


class BaseTextRoleClassifier(ABC):
    """Abstract Strategy interface for classifying spotted OCR regions against Ground-Truth QA pairs."""

    @abstractmethod
    def classify(
        self,
        detections: List[Dict[str, Any]],
        qa_annotations: List[Dict[str, Any]],
        merged_lines: Optional[List[Dict[str, Any]]] = None,
        img_w: Optional[int] = None,
        img_h: Optional[int] = None,
    ) -> Dict[str, Any]:
        pass


class SpatioLexicalGTClassifier(BaseTextRoleClassifier):
    """
    Aligns multi-word ViTextVQA Ground-Truth answers with SwinTextSpotter word-level detections
    via Viterbi sequence alignment over lexical similarity and spatial continuity, then expands
    each matched answer into a generous local `answer_region_bbox` (accepting nearby noisy text
    on the same line/cluster so the VLM can read the full context without answer clipping).
    """

    def __init__(
        self,
        min_token_sim: float = 0.60,
        fallback_token_sim: float = 0.45,
        min_chain_score: float = 0.62,
    ):
        self.min_token_sim = min_token_sim
        self.fallback_token_sim = fallback_token_sim
        self.min_chain_score = min_chain_score

    def _align_with_threshold(
        self,
        detections: List[Dict[str, Any]],
        answer_str: str,
        token_thresh: float,
    ) -> Tuple[List[int], float]:
        norm_ans = normalize_vi_text(answer_str)
        gt_tokens = norm_ans.split()
        if not gt_tokens or not detections:
            return [], 0.0

        M = len(detections)
        L = len(gt_tokens)

        best_single_idx = -1
        best_single_sim = 0.0
        joined_gt = "".join(gt_tokens)
        for i, det in enumerate(detections):
            sim = lexical_token_similarity(det["text"], joined_gt)
            if sim > best_single_sim:
                best_single_sim = sim
                best_single_idx = i

        if L == 1:
            if best_single_sim >= token_thresh:
                return [best_single_idx], float(best_single_sim)
            return [], 0.0

        sim_matrix = [
            [lexical_token_similarity(detections[i]["text"], gt_tokens[t]) for i in range(M)]
            for t in range(L)
        ]

        dp_scores = [[0.0] * M for _ in range(L)]
        dp_paths: List[List[List[int]]] = [[[] for _ in range(M)] for _ in range(L)]

        for i in range(M):
            if sim_matrix[0][i] >= token_thresh:
                dp_scores[0][i] = sim_matrix[0][i]
                dp_paths[0][i] = [i]

        for t in range(1, L):
            for j in range(M):
                sim_tj = sim_matrix[t][j]
                if sim_tj < token_thresh:
                    continue

                best_prev_score = -1.0
                best_prev_path: List[int] = []

                for t_prev in range(max(0, t - 2), t):
                    skip_penalty = 1.0 if t_prev == t - 1 else 0.90
                    for i in range(M):
                        prev_path = dp_paths[t_prev][i]
                        if not prev_path:
                            continue
                        if j in prev_path and not (
                            j == i and len(normalize_vi_text(detections[j]["text"]).split()) > 1
                        ):
                            continue
                        sp_score = spatial_transition_score(
                            detections[i]["bbox"], detections[j]["bbox"]
                        )
                        if sp_score <= 0.0:
                            continue
                        step_score = dp_scores[t_prev][i] + skip_penalty * sim_tj * (
                            0.70 + 0.30 * sp_score
                        )
                        if step_score > best_prev_score:
                            best_prev_score = step_score
                            best_prev_path = prev_path if j == prev_path[-1] else (prev_path + [j])

                if best_prev_score > 0:
                    dp_scores[t][j] = best_prev_score
                    dp_paths[t][j] = best_prev_path
                else:
                    dp_scores[t][j] = sim_tj
                    dp_paths[t][j] = [j]

        best_chain: List[int] = []
        best_norm_score = 0.0
        best_selection_metric = -1.0

        for t in range(L):
            for i in range(M):
                path = dp_paths[t][i]
                if not path:
                    continue
                unique_path: List[int] = []
                for idx in path:
                    if idx not in unique_path:
                        unique_path.append(idx)

                avg_lex = sum(
                    max(sim_matrix[k][idx] for k in range(L)) for idx in unique_path
                ) / len(unique_path)
                if avg_lex < token_thresh:
                    continue

                if len(unique_path) > 1:
                    sp_transitions = [
                        spatial_transition_score(
                            detections[unique_path[m]]["bbox"],
                            detections[unique_path[m + 1]]["bbox"],
                        )
                        for m in range(len(unique_path) - 1)
                    ]
                    if any(s <= 0.0 for s in sp_transitions):
                        continue
                    mean_sp = sum(sp_transitions) / len(sp_transitions)
                else:
                    mean_sp = 1.0

                matched_token_count = sum(
                    min(2, max(1, len(normalize_vi_text(detections[idx]["text"]).split())))
                    for idx in unique_path
                )
                coverage = min(1.0, matched_token_count / float(L))
                report_score = avg_lex * (0.65 + 0.35 * coverage)
                selection_metric = report_score * (0.80 + 0.20 * mean_sp)

                if selection_metric > best_selection_metric:
                    best_selection_metric = selection_metric
                    best_norm_score = report_score
                    best_chain = unique_path

        if best_single_sim > best_norm_score and best_single_sim >= self.min_chain_score:
            return [best_single_idx], float(best_single_sim)

        if best_norm_score >= token_thresh:
            return best_chain, float(min(1.0, best_norm_score))
        return [], 0.0

    def _align_single_answer(
        self,
        detections: List[Dict[str, Any]],
        answer_str: str,
    ) -> Tuple[List[int], float]:
        chain, score = self._align_with_threshold(detections, answer_str, self.min_token_sim)
        if chain:
            return chain, score
        # Relaxed fallback pass so slightly noisy OCR detections still propose a candidate region
        return self._align_with_threshold(detections, answer_str, self.fallback_token_sim)

    def _expand_to_answer_region_bbox(
        self,
        best_chain: List[int],
        matched_line_ids: List[int],
        detections: List[Dict[str, Any]],
        merged_lines: Optional[List[Dict[str, Any]]],
        img_w: Optional[int],
        img_h: Optional[int],
    ) -> Tuple[List[int], List[int], str]:
        """
        Expand from the matched answer words (`best_chain`) and their proximate line(s)
        (`matched_line_ids`) to include any same-line / local-cluster words within a generous
        local neighborhood (handling cases where an intermediate word was missed by OCR or
        where surrounding context/noisy words sit right next to the answer), plus safety padding.
        """
        seed_word_indices: Set[int] = set(best_chain)
        if merged_lines and matched_line_ids:
            for lid in matched_line_ids:
                for w_idx in merged_lines[lid].get("word_indices", []):
                    seed_word_indices.add(int(w_idx))

        seed_boxes = [detections[idx]["bbox"] for idx in seed_word_indices]
        rx1 = min(b[0] for b in seed_boxes)
        ry1 = min(b[1] for b in seed_boxes)
        rx2 = max(b[2] for b in seed_boxes)
        ry2 = max(b[3] for b in seed_boxes)

        # Median character height of the matched answer words
        ans_heights = sorted(
            max(1.0, float(detections[idx]["bbox"][3] - detections[idx]["bbox"][1]))
            for idx in best_chain
        )
        ref_h = ans_heights[len(ans_heights) // 2]

        # Absorb nearby same-scale words on the exact same line band (up to 1.65 * ref_h gap,
        # e.g., bridging across a single missed word on the same banner line without jumping to adjacent storefronts)
        changed = True
        while changed:
            changed = False
            rcy = 0.5 * (ry1 + ry2)
            for idx, det in enumerate(detections):
                if idx in seed_word_indices:
                    continue
                bx1, by1, bx2, by2 = det["bbox"]
                bh = max(1.0, float(by2 - by1))
                scale_ratio = min(ref_h, bh) / max(ref_h, bh)
                if scale_ratio < 0.68:
                    continue
                bcy = 0.5 * (by1 + by2)
                inter_y = max(0.0, float(min(ry2, by2) - max(ry1, by1)))
                same_band = (inter_y / min(ref_h, bh) >= 0.55) and (
                    abs(bcy - rcy) <= 0.35 * max(ref_h, bh)
                )
                if not same_band:
                    continue
                if bx2 < rx1:
                    horiz_gap = float(rx1 - bx2)
                elif bx1 > rx2:
                    horiz_gap = float(bx1 - rx2)
                else:
                    horiz_gap = 0.0

                if horiz_gap <= 1.65 * ref_h:
                    seed_word_indices.add(idx)
                    rx1 = min(rx1, bx1)
                    ry1 = min(ry1, by1)
                    rx2 = max(rx2, bx2)
                    ry2 = max(ry2, by2)
                    changed = True

        # Sort absorbed words left-to-right (and top-to-bottom if multi-line)
        ordered_indices = sorted(
            seed_word_indices,
            key=lambda idx: (
                detections[idx].get("line_id", 0),
                0.5 * (detections[idx]["bbox"][0] + detections[idx]["bbox"][2]),
            ),
        )
        region_text = " ".join(detections[idx]["text"] for idx in ordered_indices)

        # Add small safety padding around the text region so strokes/diacritics are never clipped
        pad_x = max(6, int(round(0.25 * ref_h)))
        pad_y = max(4, int(round(0.20 * ref_h)))
        px1 = max(0, rx1 - pad_x)
        py1 = max(0, ry1 - pad_y)
        px2 = min(img_w, rx2 + pad_x) if img_w else (rx2 + pad_x)
        py2 = min(img_h, ry2 + pad_y) if img_h else (ry2 + pad_y)

        return [int(px1), int(py1), int(px2), int(py2)], ordered_indices, region_text

    def classify(
        self,
        detections: List[Dict[str, Any]],
        qa_annotations: List[Dict[str, Any]],
        merged_lines: Optional[List[Dict[str, Any]]] = None,
        img_w: Optional[int] = None,
        img_h: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Detect answer-containing text regions for every question on the image (supporting
        multiple questions / multiple answer regions per image) and classify proximate text lines.
        """
        gt_det_indices: Set[int] = set()
        det_to_qas: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(len(detections))}
        line_to_qas: Dict[int, List[Dict[str, Any]]] = {
            i: [] for i in range(len(merged_lines or []))
        }
        line_to_exact_bboxes: Dict[int, List[List[int]]] = {
            i: [] for i in range(len(merged_lines or []))
        }
        qa_groundings: List[Dict[str, Any]] = []
        image_answer_regions: List[Dict[str, Any]] = []

        for q_order, qa in enumerate(qa_annotations, 1):
            qid = qa.get("id", qa.get("question_id"))
            question = qa.get("question", "")
            answers = qa.get("answers", qa.get("ground_truths", []))

            best_chain: List[int] = []
            best_score = 0.0
            best_ans = answers[0] if answers else ""

            for ans in answers:
                chain, score = self._align_single_answer(detections, ans)
                if score > best_score:
                    best_score = score
                    best_chain = chain
                    best_ans = ans

            union_bbox: Optional[List[int]] = None
            context_line_bbox: Optional[List[int]] = None
            answer_region_bbox: Optional[List[int]] = None
            matched_line_ids: List[int] = []
            matched_line_phrase: str = ""
            region_ocr_text: str = ""
            matched_texts: List[str] = []

            if best_chain:
                xs1 = [detections[idx]["bbox"][0] for idx in best_chain]
                ys1 = [detections[idx]["bbox"][1] for idx in best_chain]
                xs2 = [detections[idx]["bbox"][2] for idx in best_chain]
                ys2 = [detections[idx]["bbox"][3] for idx in best_chain]
                union_bbox = [min(xs1), min(ys1), max(xs2), max(ys2)]
                matched_texts = [detections[idx]["text"] for idx in best_chain]

                for idx in best_chain:
                    gt_det_indices.add(idx)
                    det_to_qas[idx].append(
                        {
                            "question_id": qid,
                            "answer": best_ans,
                            "match_score": round(best_score, 4),
                        }
                    )
                    lid = detections[idx].get("line_id")
                    if lid is not None and lid not in matched_line_ids:
                        matched_line_ids.append(int(lid))

                answer_region_bbox, absorbed_word_indices, region_ocr_text = (
                    self._expand_to_answer_region_bbox(
                        best_chain=best_chain,
                        matched_line_ids=matched_line_ids,
                        detections=detections,
                        merged_lines=merged_lines,
                        img_w=img_w,
                        img_h=img_h,
                    )
                )
                for idx in absorbed_word_indices:
                    gt_det_indices.add(idx)
                    lid = detections[idx].get("line_id")
                    if lid is not None and lid not in matched_line_ids:
                        matched_line_ids.append(int(lid))

                if merged_lines and matched_line_ids:
                    l_xs1 = [merged_lines[lid]["bbox"][0] for lid in matched_line_ids]
                    l_ys1 = [merged_lines[lid]["bbox"][1] for lid in matched_line_ids]
                    l_xs2 = [merged_lines[lid]["bbox"][2] for lid in matched_line_ids]
                    l_ys2 = [merged_lines[lid]["bbox"][3] for lid in matched_line_ids]
                    context_line_bbox = [min(l_xs1), min(l_ys1), max(l_xs2), max(l_ys2)]
                    matched_line_phrase = " | ".join(
                        merged_lines[lid]["text"] for lid in matched_line_ids
                    )
                    for lid in matched_line_ids:
                        line_to_qas[lid].append(
                            {
                                "question_id": qid,
                                "answer": best_ans,
                                "match_score": round(best_score, 4),
                                "exact_answer_bbox": union_bbox,
                            }
                        )
                        if union_bbox not in line_to_exact_bboxes[lid]:
                            line_to_exact_bboxes[lid].append(union_bbox)
                else:
                    context_line_bbox = union_bbox
                    matched_line_phrase = " ".join(matched_texts)

                # Group into image-level `answer_regions` (keep separate bboxes for distinct regions,
                # merge only if two questions map to nearly the exact same region with IoU >= 0.80)
                merged_into_existing = False
                for reg in image_answer_regions:
                    if bbox_iou(reg["bbox"], answer_region_bbox) >= 0.80:
                        reg["bbox"] = [
                            min(reg["bbox"][0], answer_region_bbox[0]),
                            min(reg["bbox"][1], answer_region_bbox[1]),
                            max(reg["bbox"][2], answer_region_bbox[2]),
                            max(reg["bbox"][3], answer_region_bbox[3]),
                        ]
                        reg["question_ids"].append(qid)
                        reg["questions"].append(question)
                        reg["gt_answers"].append(best_ans)
                        reg["match_score"] = round(max(reg["match_score"], best_score), 4)
                        merged_into_existing = True
                        break

                if not merged_into_existing:
                    image_answer_regions.append(
                        {
                            "region_id": len(image_answer_regions) + 1,
                            "bbox": answer_region_bbox,
                            "exact_answer_bbox": union_bbox,
                            "context_line_bbox": context_line_bbox,
                            "ocr_text_in_region": region_ocr_text,
                            "matched_ocr_phrase": " ".join(matched_texts),
                            "match_score": round(best_score, 4),
                            "question_ids": [qid],
                            "questions": [question],
                            "gt_answers": [best_ans],
                            "needs_manual_edit": bool(best_score < 0.70),
                        }
                    )

            qa_groundings.append(
                {
                    "quest_idx": q_order,
                    "question_id": qid,
                    "question": question,
                    "answers": answers,
                    "answer_region_bbox": answer_region_bbox,
                    "gt_context_line_bbox": context_line_bbox,
                    "gt_answer_union_bbox": union_bbox,
                    "region_ocr_text": region_ocr_text,
                    "matched_ocr_phrase": " ".join(matched_texts),
                    "matched_ocr_indices": best_chain,
                    "matched_ocr_texts": matched_texts,
                    "matched_line_ids": matched_line_ids,
                    "matched_line_phrase": matched_line_phrase,
                    "match_score": round(best_score, 4),
                    "needs_manual_edit": bool(answer_region_bbox is None or best_score < 0.70),
                }
            )

        gt_answer_boxes: List[Dict[str, Any]] = []
        auxiliary_boxes: List[Dict[str, Any]] = []

        for i, det in enumerate(detections):
            if i in gt_det_indices:
                det["role"] = "gt_answer"
                det["matched_qas"] = det_to_qas[i]
                gt_answer_boxes.append(det)
            else:
                det["role"] = "auxiliary"
                det["matched_qas"] = []
                auxiliary_boxes.append(det)

        gt_answer_lines: List[Dict[str, Any]] = []
        auxiliary_lines: List[Dict[str, Any]] = []

        if merged_lines is not None:
            for lid, line_obj in enumerate(merged_lines):
                if line_to_qas.get(lid):
                    line_obj["role"] = "gt_answer"
                    line_obj["matched_qas"] = line_to_qas[lid]
                    line_obj["exact_answer_bboxes"] = line_to_exact_bboxes[lid]
                    gt_answer_lines.append(line_obj)
                else:
                    line_obj["role"] = "auxiliary"
                    line_obj["matched_qas"] = []
                    line_obj["exact_answer_bboxes"] = []
                    auxiliary_lines.append(line_obj)

        return {
            "answer_regions": image_answer_regions,
            "qa_groundings": qa_groundings,
            "gt_answer_lines": gt_answer_lines,
            "auxiliary_lines": auxiliary_lines,
            "gt_answer_regions": gt_answer_boxes,
            "auxiliary_text_regions": auxiliary_boxes,
            "num_answer_regions": len(image_answer_regions),
            "num_total_lines": len(merged_lines or []),
            "num_gt_answer_lines": len(gt_answer_lines),
            "num_auxiliary_lines": len(auxiliary_lines),
            "num_total_ocr": len(detections),
            "num_gt_answer_ocr": len(gt_answer_boxes),
            "num_auxiliary_ocr": len(auxiliary_boxes),
        }
