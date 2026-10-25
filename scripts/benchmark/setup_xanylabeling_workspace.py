#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X-AnyLabeling 4-Annotator Collaborative Workspace Builder & Merger for ViTextVQA (400 Images).

Solves the Multi-Question Ground-Truth Disambiguation problem in X-AnyLabeling via a
3-Layer Zero-Typing Binding Mechanism:
1. Quest-Indexed Class Labels (`Q1_gt`, `Q2_gt`, ..., `Q10_gt`):
   - Each question `quest_idx = k` on an image gets its own distinct label `Q{k}_gt`,
     which X-AnyLabeling automatically colors with a unique color on the canvas.
2. Deterministic `group_id = question_id` & Rich `shape.description`:
   - Clicking any box in X-AnyLabeling displays its exact question, `question_id`,
     and Ground-Truth answers in the right-hand Shape Description box:
     `[Q1 | qid=74] Hỏi: cửa tiệm màu xanh là gì ? => ĐA: nhà thuốc | OCR: nhà thuốc`
   - Image-level `description` lists all questions & answers on that image as a checklist.
3. Auto-Generated Top-Left Placeholder Boxes for Unmatched Quests (`difficult=True`):
   - When SwinTextSpotter could not match a question (`answer_region_bbox is None`),
     a dashed placeholder box (`difficult: true`) with label `Q{k}_gt` and `group_id=question_id`
     is pre-created at the top-left corner of the image. The annotator simply drags its
     corners over the target text—zero typing, zero missing questions.

Usage:
  1. Test & generate leader sample pack + 4 annotator folders (100 images each):
     .\\venv\\Scripts\\python.exe scripts/benchmark/setup_xanylabeling_workspace.py --test --export-all

  2. Merge completed annotations from the 4 annotators back into a verified dataset:
     .\\venv\\Scripts\\python.exe scripts/benchmark/setup_xanylabeling_workspace.py --merge
"""

import argparse
from collections import defaultdict
import json
import os
import shutil
import sys
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))

DEFAULT_BBOXES_JSON = os.path.join(
    PROJECT_ROOT, "results", "swintextspotter_auto_annot", "answer_bboxes_400.json"
)
DEFAULT_IMAGE_DIR = os.path.join(PROJECT_ROOT, "data", "vitextvqa", "st_images")
DEFAULT_WORKSPACE_DIR = os.path.join(PROJECT_ROOT, "data", "xanylabeling_workspace")
DEFAULT_MERGED_JSON = os.path.join(
    PROJECT_ROOT, "results", "swintextspotter_auto_annot", "answer_bboxes_400_verified.json"
)

MAX_QUEST_CLASSES = 10
QUEST_CLASSES = [f"Q{i}_gt" for i in range(1, MAX_QUEST_CLASSES + 1)] + ["aux_text"]

# Distinct RGB colors for rendering guide visualizations per Quest index
QUEST_COLORS_RGB = [
    (0, 230, 90),    # Q1_gt: Emerald Green
    (0, 190, 255),   # Q2_gt: Cyan Blue
    (255, 180, 0),   # Q3_gt: Amber Gold
    (255, 75, 145),  # Q4_gt: Magenta Pink
    (165, 105, 255), # Q5_gt: Violet
    (255, 110, 50),  # Q6_gt: Coral Orange
    (50, 225, 200),  # Q7_gt: Turquoise
    (220, 235, 50),  # Q8_gt: Lime Yellow
    (255, 90, 90),   # Q9_gt: Crimson Red
    (140, 200, 255), # Q10_gt: Sky Blue
]


def bbox_to_xany_points(bbox: List[int], img_w: int, img_h: int) -> List[List[float]]:
    """
    Convert `[x1, y1, x2, y2]` to X-AnyLabeling v2.2+ 4-point rectangle format:
    `[[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]`.
    """
    x1, y1, x2, y2 = [float(v) for v in bbox]
    xmin = max(0.0, min(x1, x2, float(img_w - 1)))
    ymin = max(0.0, min(y1, y2, float(img_h - 1)))
    xmax = max(xmin + 2.0, min(max(x1, x2), float(img_w)))
    ymax = max(ymin + 2.0, min(max(y1, y2), float(img_h)))
    return [
        [round(xmin, 1), round(ymin, 1)],
        [round(xmax, 1), round(ymin, 1)],
        [round(xmax, 1), round(ymax, 1)],
        [round(xmin, 1), round(ymax, 1)],
    ]


def xany_points_to_bbox(points: List[List[float]], img_w: int, img_h: int) -> List[int]:
    """Convert 2-point or 4-point X-AnyLabeling rectangle/polygon points back to `[x1, y1, x2, y2]`."""
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    x1 = max(0, min(int(round(min(xs))), img_w - 1))
    y1 = max(0, min(int(round(min(ys))), img_h - 1))
    x2 = max(x1 + 1, min(int(round(max(xs))), img_w))
    y2 = max(y1 + 1, min(int(round(max(ys))), img_h))
    return [x1, y1, x2, y2]


def is_unmoved_placeholder(bbox: List[int], slot_idx: int) -> bool:
    """Check if a top-left placeholder box has not been dragged by the annotator yet."""
    px1 = 15
    py1 = 15 + slot_idx * 52
    px2 = 235
    py2 = py1 + 44
    return (
        abs(bbox[0] - px1) <= 3
        and abs(bbox[1] - py1) <= 3
        and abs(bbox[2] - px2) <= 3
        and abs(bbox[3] - py2) <= 3
    )


def are_bboxes_spatially_contiguous(box_a: List[int], box_b: List[int]) -> bool:
    """
    Determine whether two bounding boxes `[x1, y1, x2, y2]` belong to the same
    contiguous signboard/text block without swallowing large unrelated intermediate areas.
    """
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    aw, ah = max(1.0, ax2 - ax1), max(1.0, ay2 - ay1)
    bw, bh = max(1.0, bx2 - bx1), max(1.0, by2 - by1)
    min_h = min(ah, bh)

    dx = max(0.0, max(ax1, bx1) - min(ax2, bx2))
    dy = max(0.0, max(ay1, by1) - min(ay2, by2))
    if dx > 0.85 * min_h or dy > 0.85 * min_h:
        return False

    ux1, uy1 = min(ax1, bx1), min(ay1, by1)
    ux2, uy2 = max(ax2, bx2), max(ay2, by2)
    union_area = max(1.0, (ux2 - ux1) * (uy2 - uy1))
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter_area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    sum_area = (aw * ah) + (bw * bh) - inter_area

    return union_area <= 1.45 * max(1.0, sum_area)


def group_identical_quests_for_image(quests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Group quests on the same image that have the exact same normalized question text
    under the same `quest_idx` (`Q{k}_gt`), then cluster their bounding boxes by
    spatial contiguity:
    - Spatially contiguous boxes on the same sign (`30.jpg`: 'bếp gas', 'bếp từ', 'hút mùi')
      are merged into 1 single union box.
    - Spatially disjoint boxes (`126.jpg`: stores separated along a street) are kept as
      separate sub-boxes that all share the same `Q{k}_gt` label and color.
    """
    grouped_map: Dict[str, Dict[str, Any]] = {}
    ordered_keys: List[str] = []

    for q in quests:
        q_text = str(q.get("question", "")).strip()
        norm_key = " ".join(q_text.lower().split())
        q_id = int(q["question_id"])
        ans_list = [str(a).strip() for a in q.get("gt_answers", []) if str(a).strip()]
        raw_bbox = q.get("answer_region_bbox")
        exact_bbox = q.get("gt_answer_union_bbox") or raw_bbox
        score = float(q.get("match_score", 0.0))
        ocr_text = str(q.get("region_ocr_text", "")).strip()

        if norm_key not in grouped_map:
            ordered_keys.append(norm_key)
            grouped_map[norm_key] = {
                "quest_idx": len(ordered_keys),
                "question": q_text,
                "items": [],
            }
        grouped_map[norm_key]["items"].append(
            {
                "question_id": q_id,
                "gt_answers": list(dict.fromkeys(ans_list)),
                "bbox": [int(v) for v in raw_bbox] if (raw_bbox and len(raw_bbox) == 4) else None,
                "exact_bbox": [int(v) for v in exact_bbox] if (exact_bbox and len(exact_bbox) == 4) else None,
                "ocr_text": ocr_text,
                "score": score,
            }
        )

    merged_quests: List[Dict[str, Any]] = []
    for key in ordered_keys:
        entry = grouped_map[key]
        q_idx = entry["quest_idx"]
        q_text = entry["question"]
        items = entry["items"]

        # Connected-components clustering of spatially contiguous bboxes (using exact_bbox to avoid false bridges)
        n_items = len(items)
        visited = [False] * n_items
        clusters: List[List[Dict[str, Any]]] = []

        for i in range(n_items):
            if visited[i]:
                continue
            visited[i] = True
            comp = [items[i]]
            if items[i]["exact_bbox"] is not None:
                changed = True
                while changed:
                    changed = False
                    for j in range(n_items):
                        if visited[j] or items[j]["exact_bbox"] is None:
                            continue
                        if any(
                            c["exact_bbox"] is not None
                            and are_bboxes_spatially_contiguous(c["exact_bbox"], items[j]["exact_bbox"])
                            for c in comp
                        ):
                            visited[j] = True
                            comp.append(items[j])
                            changed = True
            clusters.append(comp)

        all_qids = [it["question_id"] for it in items]
        all_answers: List[str] = []
        for it in items:
            for a in it["gt_answers"]:
                if a not in all_answers:
                    all_answers.append(a)

        for comp in clusters:
            comp_qids = [c["question_id"] for c in comp]
            comp_answers: List[str] = []
            comp_ocrs: List[str] = []
            comp_boxes: List[List[int]] = []
            for c in comp:
                for a in c["gt_answers"]:
                    if a not in comp_answers:
                        comp_answers.append(a)
                if c["ocr_text"] and c["ocr_text"] not in comp_ocrs:
                    comp_ocrs.append(c["ocr_text"])
                b = c["bbox"]
                eb = c["exact_bbox"]
                # If a disjoint sub-item's region bbox is >1.8x wider than its exact answer bbox,
                # use padded exact_bbox so it doesn't swallow neighboring stores (e.g., SMART SHOES on 126.jpg)
                if len(clusters) > 1 and b is not None and eb is not None:
                    if (b[2] - b[0]) > 1.8 * max(1, eb[2] - eb[0]):
                        pad = 6
                        b = [max(0, eb[0] - pad), max(0, eb[1] - pad), eb[2] + pad, eb[3] + pad]
                if b is not None:
                    comp_boxes.append(b)

            if comp_boxes:
                union_bbox: Optional[List[int]] = [
                    min(int(b[0]) for b in comp_boxes),
                    min(int(b[1]) for b in comp_boxes),
                    max(int(b[2]) for b in comp_boxes),
                    max(int(b[3]) for b in comp_boxes),
                ]
            else:
                union_bbox = None

            merged_quests.append(
                {
                    "quest_idx": q_idx,
                    "question_id": comp_qids[0],
                    "question_ids": comp_qids,
                    "all_group_question_ids": all_qids,
                    "all_group_answers": all_answers,
                    "question": q_text,
                    "gt_answers": comp_answers,
                    "answer_region_bbox": union_bbox,
                    "region_ocr_text": " | ".join(comp_ocrs),
                    "match_score": sum(c["score"] for c in comp) / max(1, len(comp)),
                }
            )
    return merged_quests


def build_xanylabeling_json_for_image(
    img_record: Dict[str, Any],
    include_aux: bool = False,
) -> Dict[str, Any]:
    """
    Build an X-AnyLabeling compatible `.json` dictionary for a single image record.
    - Spatially contiguous answers for the same question merge into 1 union bbox (`30.jpg`).
    - Spatially disjoint answers for the same question stay as separate sub-boxes sharing
      the same `Q{k}_gt` label (`126.jpg`).
    """
    img_w = int(img_record["width"])
    img_h = int(img_record["height"])
    image_name = img_record["image_name"]
    quests = group_identical_quests_for_image(img_record.get("quests", []))
    unique_q_indices = sorted({int(q["quest_idx"]) for q in quests})

    shapes: List[Dict[str, Any]] = []
    image_desc_lines: List[str] = [
        f"Ảnh: {image_name} (ID={img_record['image_id']}) | Tổng số câu hỏi: {len(unique_q_indices)}"
    ]
    logged_q_indices = set()

    seen_bbox_counts: Dict[Tuple[int, int, int, int], int] = defaultdict(int)
    unmatched_slot = 0

    for q in quests:
        q_idx = int(q["quest_idx"])
        q_id = int(q["question_id"])
        q_ids = q.get("question_ids", [q_id])
        q_text = str(q.get("question", "")).strip()
        gt_ans_list = [str(a).strip() for a in q.get("gt_answers", []) if str(a).strip()]
        gt_ans_str = " / ".join(dict.fromkeys(gt_ans_list))
        all_group_ans_str = " / ".join(dict.fromkeys(q.get("all_group_answers", gt_ans_list)))
        match_score = float(q.get("match_score", 0.0))
        raw_bbox = q.get("answer_region_bbox")

        label_name = f"Q{q_idx}_gt"

        if q_idx not in logged_q_indices:
            logged_q_indices.add(q_idx)
            image_desc_lines.append(
                f"• Q{q_idx}_gt: Hỏi: {q_text} || ĐA: {all_group_ans_str}"
            )

        if raw_bbox is not None and len(raw_bbox) == 4:
            base_tuple = (int(raw_bbox[0]), int(raw_bbox[1]), int(raw_bbox[2]), int(raw_bbox[3]))
            dup_count = seen_bbox_counts[base_tuple]
            seen_bbox_counts[base_tuple] += 1

            # Offset identical shared boxes by 3px so both colored borders are visible & clickable
            offset = dup_count * 3
            adj_bbox = [
                base_tuple[0] + offset,
                base_tuple[1] + offset,
                min(img_w, base_tuple[2] + offset),
                min(img_h, base_tuple[3] + offset),
            ]
            points = bbox_to_xany_points(adj_bbox, img_w, img_h)
            difficult_flag = False
            shape_desc = f"Hỏi: {q_text} || ĐA: {gt_ans_str}"
        else:
            # Create a top-left placeholder box so the annotator only needs to drag it to the target text
            py1 = min(img_h - 50, 15 + unmatched_slot * 52)
            py2 = min(img_h - 4, py1 + 44)
            px1 = 15
            px2 = min(img_w - 4, 235)
            unmatched_slot += 1

            points = bbox_to_xany_points([px1, py1, px2, py2], img_w, img_h)
            difficult_flag = True
            shape_desc = f"Hỏi: {q_text} || ĐA: {gt_ans_str}"

        shapes.append(
            {
                "label": label_name,
                "score": round(match_score, 4),
                "points": points,
                "group_id": q_id,
                "description": shape_desc,
                "difficult": difficult_flag,
                "shape_type": "rectangle",
                "flags": {
                    "needs_manual_edit": bool(raw_bbox is None or match_score < 0.70),
                },
                "attributes": {
                    "quest_idx": q_idx,
                    "question_id": q_id,
                    "question_ids": q_ids,
                    "question": q_text,
                    "gt_answer": gt_ans_str,
                },
                "kie_linking": [],
            }
        )

    if include_aux:
        for line in img_record.get("candidate_text_lines", []):
            if line.get("role") != "auxiliary":
                continue
            lb = line.get("bbox")
            if not lb or len(lb) != 4:
                continue
            shapes.append(
                {
                    "label": "aux_text",
                    "score": round(float(line.get("score", 0.0)), 4),
                    "points": bbox_to_xany_points(lb, img_w, img_h),
                    "group_id": None,
                    "description": f"[AUX] OCR: {line.get('text', '')}",
                    "difficult": False,
                    "shape_type": "rectangle",
                    "flags": {},
                    "attributes": {},
                    "kie_linking": [],
                }
            )

    return {
        "version": "2.4.0",
        "flags": {
            "has_unmatched_quest": unmatched_slot > 0,
            "multi_quest": len(quests) >= 2,
        },
        "checked": False,
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": img_h,
        "imageWidth": img_w,
        "description": "\n".join(image_desc_lines),
    }


def write_batch_folder(
    batch_dir: str,
    batch_name: str,
    records: List[Dict[str, Any]],
    image_src_dir: str,
    include_aux: bool = False,
) -> Dict[str, Any]:
    """Write `.jpg`, `.json`, `classes.txt`, and `ASSIGNMENT_CHECKLIST.md` into `batch_dir`."""
    os.makedirs(batch_dir, exist_ok=True)

    # Write classes.txt for X-AnyLabeling
    classes_path = os.path.join(batch_dir, "classes.txt")
    with open(classes_path, "w", encoding="utf-8") as cf:
        for cls_name in QUEST_CLASSES:
            cf.write(cls_name + "\n")

    num_quests_total = 0
    num_unmatched_quests = 0
    priority_images: List[str] = []
    shared_region_images: List[str] = []

    checklist_rows: List[str] = []
    for idx, rec in enumerate(records, 1):
        img_name = rec["image_name"]
        stem = os.path.splitext(img_name)[0]
        src_img_path = os.path.join(image_src_dir, img_name)
        dst_img_path = os.path.join(batch_dir, img_name)
        dst_json_path = os.path.join(batch_dir, f"{stem}.json")

        if os.path.exists(src_img_path) and not os.path.exists(dst_img_path):
            shutil.copy2(src_img_path, dst_img_path)

        xany_payload = build_xanylabeling_json_for_image(rec, include_aux=include_aux)
        # Preserve manually edited shapes/descriptions/custom added boxes from X-AnyLabeling
        if os.path.exists(dst_json_path):
            try:
                with open(dst_json_path, "r", encoding="utf-8") as old_jf:
                    old_data = json.load(old_jf)
                if old_data.get("version") == "4.2.0" or len(old_data.get("shapes", [])) > len(xany_payload.get("shapes", [])):
                    xany_payload["shapes"] = old_data["shapes"]
                    xany_payload["version"] = old_data.get("version", "4.2.0")
                else:
                    old_by_qid = {
                        s.get("group_id"): s
                        for s in old_data.get("shapes", [])
                        if s.get("group_id") is not None
                        and len(s.get("attributes", {}).get("question_ids", [s.get("group_id")])) <= 1
                    }
                    for new_s in xany_payload.get("shapes", []):
                        q_ids = new_s.get("attributes", {}).get("question_ids", [])
                        old_s = old_by_qid.get(new_s.get("group_id"))
                        if old_s and old_s.get("points") and len(q_ids) <= 1:
                            new_s["points"] = old_s["points"]
                if old_data.get("checked"):
                    xany_payload["checked"] = True
            except Exception:
                pass
        with open(dst_json_path, "w", encoding="utf-8") as jf:
            json.dump(xany_payload, jf, ensure_ascii=False, indent=2)

        quests = group_identical_quests_for_image(rec.get("quests", []))
        unique_q_map: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for q in quests:
            unique_q_map[int(q["quest_idx"])].append(q)

        n_q = len(unique_q_map)
        n_unmatched = sum(1 for q in quests if q.get("answer_region_bbox") is None)
        num_quests_total += n_q
        num_unmatched_quests += n_unmatched

        if n_unmatched > 0:
            priority_images.append(img_name)
        if n_q > rec.get("num_answer_regions", n_q) and n_unmatched == 0:
            shared_region_images.append(img_name)

        summary_parts: List[str] = []
        for q_idx in sorted(unique_q_map.keys()):
            sub_q_list = unique_q_map[q_idx]
            first_q = sub_q_list[0]
            all_ans = first_q.get("all_group_answers", first_q.get("gt_answers", []))
            has_unmatched = any(sq.get("answer_region_bbox") is None for sq in sub_q_list)
            multi_box_tag = f" *({len(sub_q_list)} box rời)*" if len(sub_q_list) > 1 else ""
            summary_parts.append(
                f"**`Q{q_idx}_gt` (`qid={first_q['question_id']}`){multi_box_tag}**: "
                f"{first_q['question']} $\\rightarrow$ `{' / '.join(all_ans)}` "
                f"{'⚠️ **[CẦN KÉO BOX]**' if has_unmatched else '✅'}"
            )
        quest_summary = " <br> ".join(summary_parts)
        checklist_rows.append(
            f"| {idx} | `{img_name}` | {n_q} | {'⚠️ Cần kéo ' + str(n_unmatched) + ' box' if n_unmatched else '✅ Đã có sẵn'} | {quest_summary} |"
        )

    checklist_md_path = os.path.join(batch_dir, "ASSIGNMENT_CHECKLIST.md")
    with open(checklist_md_path, "w", encoding="utf-8") as mf:
        mf.write(f"# Danh sách Gán nhãn X-AnyLabeling — `{batch_name}`\n\n")
        mf.write(f"- **Tổng số ảnh:** `{len(records)}`\n")
        mf.write(f"- **Tổng số câu hỏi (Quests):** `{num_quests_total}`\n")
        mf.write(
            f"- **Số câu hỏi cần kéo box từ góc trái trên (`[CẦN KÉO BOX]`):** `{num_unmatched_quests}` "
            f"(trên `{len(priority_images)}` ảnh)\n"
        )
        if priority_images:
            mf.write(
                f"- **Các ảnh ưu tiên kiểm tra trước (có box chờ kéo):** {', '.join(f'`{x}`' for x in priority_images)}\n"
            )
        mf.write("\n## Hướng dẫn bổ sung thêm BBox & Đáp án bị thiếu trong bộ dữ liệu gốc\n\n")
        mf.write(
            "Khi kiểm tra ảnh có câu hỏi liệt kê (ví dụ `126.jpg`: *\"ở đây có những cửa hàng gì ?\"* có 6 cửa hàng trên phố nhưng GT gốc chỉ ghi 3):\n"
            "1. **Vẽ thêm BBox cho các đáp án bị thiếu**: Nhấn phím **`R`**, khoanh vào biển hiệu bị thiếu (ví dụ `Like Kid's`, `SMART SHOES`, `Yến Kids`), chọn đúng nhãn **`Q{k}_gt`** của câu hỏi đó (ví dụ `Q2_gt`).\n"
            "2. **Ghi bổ sung đáp án**: Ở ô **Description** (góc dưới bên phải X-AnyLabeling) của box vừa vẽ, nhập tên đáp án mới (ví dụ: `like kid's` hoặc `Hỏi: ở đây có những cửa hàng gì ? || ĐA: like kid's`). Hoặc sửa trực tiếp phần sau `|| ĐA:` trên box có sẵn (ngăn cách nhiều đáp án bằng dấu `/`).\n"
            "3. Khi chạy `--merge`, hệ thống sẽ tự động gom toàn bộ các BBox cùng nhãn `Q{k}_gt` và cập nhật danh sách đáp án mới vào dữ liệu chuẩn.\n"
        )
        mf.write("\n## Bảng tra cứu nhanh Câu hỏi & Đáp án theo từng ảnh\n\n")
        mf.write("| STT | Tên ảnh | Số Quest | Trạng thái Auto-Detect | Chi tiết `Qk_gt` (Câu hỏi $\\rightarrow$ Đáp án GT) |\n")
        mf.write("| :---: | :--- | :---: | :--- | :--- |\n")
        for row in checklist_rows:
            mf.write(row + "\n")

    return {
        "batch_name": batch_name,
        "batch_dir": batch_dir,
        "num_images": len(records),
        "num_quests": num_quests_total,
        "num_unmatched_quests": num_unmatched_quests,
        "priority_images": priority_images,
        "shared_region_images": shared_region_images,
    }


def select_diverse_sample_records(images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Select 8 representative sample images for the team leader to test & create examples:
    - Case 1: Multi-quest with 1 unmatched quest needing drag (`22.jpg`: 4 quests, Q4 'wutacam' unmatched)
    - Case 2: Multi-quest where multiple questions share the same text region (`38.jpg`: 6 quests, 4 regions)
    - Case 3: Multi-quest with distinct regions on a store sign (`30.jpg`, `68.jpg`, `76.jpg`, `77.jpg`, `111.jpg`, `126.jpg`)
    """
    preferred_names = ["22.jpg", "30.jpg", "38.jpg", "68.jpg", "76.jpg", "77.jpg", "111.jpg", "126.jpg"]
    name_to_rec = {r["image_name"]: r for r in images}
    selected = [name_to_rec[n] for n in preferred_names if n in name_to_rec]
    if len(selected) < 8:
        for r in images:
            if r not in selected:
                selected.append(r)
            if len(selected) >= 8:
                break
    return selected


def render_xanylabeling_preview(
    img_path: str,
    xany_json_path: str,
    out_vis_path: str,
) -> None:
    """
    Render a high-clarity visual illustration of an X-AnyLabeling `.json` file
    showing `Q1_gt`, `Q2_gt`, ... with distinct colors and quest labels for the `.md` guide.
    """
    with open(xany_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with Image.open(img_path) as pil_img:
        canvas = pil_img.convert("RGBA")

    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    img_w, img_h = canvas.size

    font_size = max(14, min(26, int(min(img_w, img_h) * 0.022)))
    font = None
    for font_candidate in [
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
    ]:
        if os.path.exists(font_candidate):
            try:
                font = ImageFont.truetype(font_candidate, font_size)
                break
            except Exception:
                pass
    if font is None:
        font = ImageFont.load_default()

    placed_labels: List[Tuple[int, int, int, int]] = []

    for shape in data.get("shapes", []):
        label = str(shape.get("label", ""))
        if not label.startswith("Q"):
            continue
        try:
            q_num = int(label.split("_")[0][1:])
        except Exception:
            q_num = 1

        color_rgb = QUEST_COLORS_RGB[(q_num - 1) % len(QUEST_COLORS_RGB)]
        bbox = xany_points_to_bbox(shape["points"], img_w, img_h)
        x1, y1, x2, y2 = bbox
        is_difficult = bool(shape.get("difficult", False))
        qid = shape.get("group_id")
        attrs = shape.get("attributes", {})
        gt_ans = attrs.get("gt_answer", "")
        desc_str = str(shape.get("description", "") or "").strip()
        if "|| ĐA:" in desc_str:
            gt_ans = desc_str.split("|| ĐA:")[1].split("||")[0].strip()
        elif not gt_ans and desc_str:
            gt_ans = desc_str

        fill_alpha = 35 if not is_difficult else 75
        draw.rectangle([x1, y1, x2, y2], fill=(*color_rgb, fill_alpha), outline=(*color_rgb, 255), width=3)

        tag_prefix = "[CẦN KÉO] " if is_difficult else ""
        qid_tag = f" (qid={qid})" if qid is not None else " [Bổ sung]"
        tag_text = f"{tag_prefix}{label}{qid_tag}: {gt_ans}"
        tx = max(4, min(x1, img_w - 180))
        ty = max(4, y1 - font_size - 6)
        tb = draw.textbbox((tx, ty), tag_text, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        if tx + tw > img_w - 4:
            tx = max(4, img_w - tw - 6)

        for _ in range(5):
            cand = (tx, ty, tx + tw + 6, ty + th + 4)
            if any(not (cand[2] <= ox1 or cand[0] >= ox2 or cand[3] <= oy1 or cand[1] >= oy2) for ox1, oy1, ox2, oy2 in placed_labels):
                ty = max(4, ty - th - 4)
            else:
                break
        final_box = (tx, ty, tx + tw + 6, ty + th + 4)
        placed_labels.append(final_box)
        draw.rectangle(final_box, fill=(20, 24, 30, 230), outline=(*color_rgb, 255), width=1)
        draw.text((tx + 3, ty + 2), tag_text, fill=(*color_rgb, 255), font=font)

    os.makedirs(os.path.dirname(os.path.abspath(out_vis_path)), exist_ok=True)
    Image.alpha_composite(canvas, overlay).convert("RGB").save(out_vis_path, quality=92)


def extract_answers_from_shape(shape: Dict[str, Any]) -> List[str]:
    """
    Extract Ground-Truth answer strings from an X-AnyLabeling shape's `description`.
    Supports:
    - Standard format: `'Hỏi: ... || ĐA: ans1 / ans2'`
    - Short format: `'ĐA: ans1 / ans2'`
    - Direct text entered by annotator on a newly drawn `Q{k}_gt` box: `'like kid's'`
    """
    desc = str(shape.get("description", "") or "").strip()
    if not desc or desc.startswith("[AUX]"):
        return []
    if "|| ĐA:" in desc:
        ans_part = desc.split("|| ĐA:", 1)[1].split("||", 1)[0].strip()
    elif desc.lower().startswith("đa:"):
        ans_part = desc.split(":", 1)[1].strip()
    elif desc.lower().startswith("hỏi:"):
        return []
    else:
        ans_part = desc

    return [a.strip() for a in ans_part.split("/") if a.strip()]


def merge_workspace_annotations(
    bboxes_json_path: str,
    workspace_dir: str,
    output_json_path: str,
) -> Dict[str, Any]:
    """
    Merge human-edited X-AnyLabeling `.json` files from `00_sample_for_leader` and
    `member_1_001_100` .. `member_4_301_400` back into `answer_bboxes_400_verified.json`.
    Supports:
    - Matching by `label == f"Q{quest_idx}_gt"` OR `group_id == question_id`.
    - Multiple boxes drawn for a single question (stores `sub_bboxes` and union `answer_region_bbox`).
    - Extracting newly added answers typed by annotators in `shape.description`.
    """
    with open(bboxes_json_path, "r", encoding="utf-8") as f:
        master_data = json.load(f)

    # Scan member folders first, then allow `00_sample_for_leader` to override if edited
    subfolders = [
        "Cao_Anh_001_100",
        "Trieu_101_200",
        "Cam_Tu_201_300",
        "Thanh_Tam_301_400",
        "00_sample_for_leader",
    ]
    stem_to_json_path: Dict[str, str] = {}
    for sub in subfolders:
        sub_dir = os.path.join(workspace_dir, sub)
        if not os.path.isdir(sub_dir):
            continue
        for fname in os.listdir(sub_dir):
            if fname.endswith(".json") and fname != "README_MANIFEST.json":
                stem = os.path.splitext(fname)[0]
                cand_path = os.path.join(sub_dir, fname)
                if stem not in stem_to_json_path or sub == "00_sample_for_leader":
                    stem_to_json_path[stem] = cand_path

    total_quests = 0
    verified_quests = 0
    still_missing_quests = 0

    for img_rec in master_data.get("images", []):
        img_name = img_rec["image_name"]
        img_w = int(img_rec["width"])
        img_h = int(img_rec["height"])
        stem = os.path.splitext(img_name)[0]
        json_p = stem_to_json_path.get(stem)
        if not json_p or not os.path.exists(json_p):
            continue

        with open(json_p, "r", encoding="utf-8") as jf:
            xany_data = json.load(jf)

        # Map each raw question_id to its grouped logical quest_idx
        grouped_quests = group_identical_quests_for_image(img_rec.get("quests", []))
        qid_to_logical_qidx: Dict[int, int] = {}
        for gq in grouped_quests:
            for qid_member in gq.get("all_group_question_ids", gq.get("question_ids", [gq["question_id"]])):
                qid_to_logical_qidx[int(qid_member)] = int(gq["quest_idx"])

        by_qid: Dict[int, List[List[int]]] = defaultdict(list)
        by_qidx: Dict[int, List[List[int]]] = defaultdict(list)
        answers_by_qidx: Dict[int, List[str]] = defaultdict(list)

        unmatched_slot_counter = 0
        for shape in xany_data.get("shapes", []):
            label = str(shape.get("label", "")).strip()
            if label == "aux_text" or not shape.get("points"):
                continue
            bbox = xany_points_to_bbox(shape["points"], img_w, img_h)

            if bool(shape.get("difficult", False)) and is_unmoved_placeholder(
                bbox, unmatched_slot_counter
            ):
                unmatched_slot_counter += 1
                continue

            shape_answers = extract_answers_from_shape(shape)

            grp_id = shape.get("group_id")
            if grp_id is not None:
                try:
                    by_qid[int(grp_id)].append(bbox)
                except Exception:
                    pass

            for extra_qid in shape.get("attributes", {}).get("question_ids", []):
                try:
                    if bbox not in by_qid[int(extra_qid)]:
                        by_qid[int(extra_qid)].append(bbox)
                except Exception:
                    pass

            if label.startswith("Q") and "_gt" in label:
                try:
                    q_idx = int(label.split("_")[0][1:])
                    if bbox not in by_qidx[q_idx]:
                        by_qidx[q_idx].append(bbox)
                    for sa in shape_answers:
                        if sa not in answers_by_qidx[q_idx]:
                            answers_by_qidx[q_idx].append(sa)
                except Exception:
                    pass

        updated_regions: List[Dict[str, Any]] = []
        for q in img_rec.get("quests", []):
            total_quests += 1
            raw_q_idx = int(q["quest_idx"])
            q_id = int(q["question_id"])
            logical_q_idx = qid_to_logical_qidx.get(q_id, raw_q_idx)

            matched_boxes = by_qid.get(q_id) or by_qidx.get(logical_q_idx) or []
            all_qidx_boxes = by_qidx.get(logical_q_idx) or matched_boxes

            if matched_boxes:
                union_box = [
                    min(b[0] for b in matched_boxes),
                    min(b[1] for b in matched_boxes),
                    max(b[2] for b in matched_boxes),
                    max(b[3] for b in matched_boxes),
                ]
                q["answer_region_bbox"] = union_box
                q["sub_bboxes"] = all_qidx_boxes

                # Merge any newly added answers from annotator shapes on this logical quest
                orig_answers = [str(a).strip() for a in q.get("gt_answers", []) if str(a).strip()]
                merged_ans = list(dict.fromkeys(orig_answers + answers_by_qidx.get(logical_q_idx, [])))
                added_ans = [a for a in merged_ans if a.lower() not in {oa.lower() for oa in orig_answers}]
                if added_ans:
                    q["added_gt_answers"] = added_ans
                    q["verified_all_answers"] = merged_ans

                q["needs_manual_edit"] = False
                verified_quests += 1
                updated_regions.append(
                    {
                        "region_id": raw_q_idx,
                        "bbox": union_box,
                        "sub_bboxes": all_qidx_boxes,
                        "gt_answers": merged_ans if added_ans else orig_answers,
                        "added_gt_answers": added_ans,
                        "question_ids": [q_id],
                        "questions": [q.get("question", "")],
                        "needs_manual_edit": False,
                    }
                )
            else:
                still_missing_quests += 1
                q["needs_manual_edit"] = True

        img_rec["answer_regions"] = updated_regions
        img_rec["num_answer_regions"] = len(updated_regions)

    master_data["metadata"]["verified_quests"] = verified_quests
    master_data["metadata"]["quest_match_rate_pct"] = round(
        100.0 * verified_quests / max(1, total_quests), 2
    )
    master_data["metadata"]["still_missing_quests"] = still_missing_quests

    os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as out_f:
        json.dump(master_data, out_f, ensure_ascii=False, indent=2)

    return {
        "total_quests": total_quests,
        "verified_quests": verified_quests,
        "still_missing_quests": still_missing_quests,
        "output_json_path": output_json_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Setup & Merge X-AnyLabeling 4-Person Collaborative Workspace for ViTextVQA (400 Images)"
    )
    parser.add_argument(
        "--bboxes-json",
        type=str,
        default=DEFAULT_BBOXES_JSON,
        help="Path to answer_bboxes_400.json from Colab L4.",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=DEFAULT_IMAGE_DIR,
        help="Directory containing original ViTextVQA .jpg images.",
    )
    parser.add_argument(
        "--workspace-dir",
        type=str,
        default=DEFAULT_WORKSPACE_DIR,
        help="Output root directory for X-AnyLabeling folders.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Create the 8-image leader sample folder (`00_sample_for_leader`) and render preview overlays.",
    )
    parser.add_argument(
        "--export-all",
        action="store_true",
        help="Export all 4 annotator folders (`member_1_001_100` .. `member_4_301_400`, 100 images each).",
    )
    parser.add_argument(
        "--include-aux",
        action="store_true",
        help="Also include `aux_text` distraction boxes in X-AnyLabeling JSONs (default: False for clean canvas).",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge edited X-AnyLabeling JSON files from the workspace back into answer_bboxes_400_verified.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.bboxes_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("images", [])
    print(
        f"[LOAD] Loaded {len(images)} images ({data['metadata']['total_quests']} quests, "
        f"{data['metadata']['matched_quests']} pre-matched = {data['metadata']['quest_match_rate_pct']}%) "
        f"from {args.bboxes_json}"
    )

    if args.test or not (args.export_all or args.merge):
        sample_records = select_diverse_sample_records(images)
        sample_dir = os.path.join(args.workspace_dir, "00_sample_for_leader")
        sample_info = write_batch_folder(
            batch_dir=sample_dir,
            batch_name="00_sample_for_leader",
            records=sample_records,
            image_src_dir=args.image_dir,
            include_aux=args.include_aux,
        )
        preview_dir = os.path.join(args.workspace_dir, "guide_previews")
        for rec in sample_records:
            img_name = rec["image_name"]
            stem = os.path.splitext(img_name)[0]
            render_xanylabeling_preview(
                img_path=os.path.join(sample_dir, img_name),
                xany_json_path=os.path.join(sample_dir, f"{stem}.json"),
                out_vis_path=os.path.join(preview_dir, f"xany_preview_{img_name}"),
            )
        print(
            f"[SAMPLE READY] Created {sample_info['num_images']} sample images "
            f"({sample_info['num_quests']} quests) at: {sample_dir}\n"
            f"[PREVIEWS] Rendered guide preview images at: {preview_dir}"
        )

    if args.export_all:
        batch_specs = [
            ("Cao_Anh_001_100", images[0:100]),
            ("Trieu_101_200", images[100:200]),
            ("Cam_Tu_201_300", images[200:300]),
            ("Thanh_Tam_301_400", images[300:400]),
        ]
        # Sync leader's already-edited sample JSONs into Cao_Anh_001_100 before exporting
        sample_dir = os.path.join(args.workspace_dir, "00_sample_for_leader")
        cao_anh_dir = os.path.join(args.workspace_dir, "Cao_Anh_001_100")
        os.makedirs(cao_anh_dir, exist_ok=True)
        if os.path.isdir(sample_dir):
            for fname in os.listdir(sample_dir):
                if fname.endswith(".json"):
                    shutil.copy2(os.path.join(sample_dir, fname), os.path.join(cao_anh_dir, fname))
        manifest_summary = []
        for batch_name, batch_records in batch_specs:
            batch_dir = os.path.join(args.workspace_dir, batch_name)
            info = write_batch_folder(
                batch_dir=batch_dir,
                batch_name=batch_name,
                records=batch_records,
                image_src_dir=args.image_dir,
                include_aux=args.include_aux,
            )
            manifest_summary.append(info)
            print(
                f"[EXPORT] {batch_name}: {info['num_images']} images | "
                f"{info['num_quests']} quests | "
                f"{info['num_unmatched_quests']} unmatched quests needing drag "
                f"({len(info['priority_images'])} priority images)"
            )

        manifest_path = os.path.join(args.workspace_dir, "WORKSPACE_MANIFEST.json")
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(manifest_summary, mf, ensure_ascii=False, indent=2)
        print(f"[MANIFEST] Saved 4-member assignment summary to: {manifest_path}")

    if args.merge:
        merge_stats = merge_workspace_annotations(
            bboxes_json_path=args.bboxes_json,
            workspace_dir=args.workspace_dir,
            output_json_path=DEFAULT_MERGED_JSON,
        )
        print(
            f"[MERGE] Verified Quests: {merge_stats['verified_quests']}/{merge_stats['total_quests']} | "
            f"Still Unmoved Placeholders: {merge_stats['still_missing_quests']} | "
            f"Saved to: {merge_stats['output_json_path']}"
        )


if __name__ == "__main__":
    main()
