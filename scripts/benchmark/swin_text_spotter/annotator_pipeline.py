# -*- coding: utf-8 -*-
"""
Pipe-and-Filter Auto-Annotation Pipeline & Memory Guard for SwinTextSpotter (SWINTS-swin-vintext).

Enforces strict System RAM and GPU VRAM bounds on local hardware (RTX 3050 Ti 4GB GPU)
while automatically detecting, recognizing, and classifying scene text into:
1. Ground-Truth Answer Text Regions (`role = "gt_answer"`)
2. Auxiliary / Distractor Scene Text Regions (`role = "auxiliary"`)
"""

from dataclasses import dataclass, field
import gc
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import psutil
import torch

from .gt_classifier import BaseTextRoleClassifier, SpatioLexicalGTClassifier
from .modeling import SWINTS
from .vintext_decoder import (
    decode_recognition_tokens,
    extract_tight_polygon_and_bbox,
    group_proximate_text_lines,
    polygon_nms,
    sort_detections_tl_br,
)

logger = logging.getLogger("SwinTextSpotterAnnotator")


@dataclass
class MemorySnapshot:
    ram_used_gb: float
    ram_total_gb: float
    ram_percent: float
    vram_allocated_mb: float
    vram_reserved_mb: float
    vram_peak_mb: float


class MemoryGuard:
    """
    Observability & Resource Guard preventing System RAM or GPU VRAM from becoming full.
    """

    def __init__(self, max_ram_percent: float = 82.0, max_vram_mb: float = 3400.0):
        self.max_ram_percent = max_ram_percent
        self.max_vram_mb = max_vram_mb
        self.peak_ram_percent: float = 0.0
        self.peak_ram_used_gb: float = 0.0
        self.peak_vram_mb: float = 0.0

    def snapshot(self) -> MemorySnapshot:
        vm = psutil.virtual_memory()
        ram_used_gb = vm.used / (1024 ** 3)
        ram_total_gb = vm.total / (1024 ** 3)
        ram_pct = float(vm.percent)

        if torch.cuda.is_available():
            vram_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
            vram_res = torch.cuda.memory_reserved() / (1024 ** 2)
            vram_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            vram_alloc = vram_res = vram_peak = 0.0

        self.peak_ram_percent = max(self.peak_ram_percent, ram_pct)
        self.peak_ram_used_gb = max(self.peak_ram_used_gb, ram_used_gb)
        self.peak_vram_mb = max(self.peak_vram_mb, vram_peak)

        return MemorySnapshot(
            ram_used_gb=round(ram_used_gb, 2),
            ram_total_gb=round(ram_total_gb, 2),
            ram_percent=round(ram_pct, 1),
            vram_allocated_mb=round(vram_alloc, 1),
            vram_reserved_mb=round(vram_res, 1),
            vram_peak_mb=round(vram_peak, 1),
        )

    def cleanup(self, force_cuda_empty: bool = True) -> MemorySnapshot:
        gc.collect()
        if force_cuda_empty and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self.snapshot()

    def check_and_reclaim(self) -> MemorySnapshot:
        snap = self.snapshot()
        if snap.ram_percent >= self.max_ram_percent or snap.vram_reserved_mb >= self.max_vram_mb:
            snap = self.cleanup(force_cuda_empty=True)
        return snap


class SwinTextSpotterAnnotator:
    """
    Modular Pipe-and-Filter Auto-Annotation Pipeline using official SWINTS-swin-vintext.
    """

    def __init__(
        self,
        ckpt_path: str,
        npz_path: str,
        device: Optional[str] = None,
        min_size_test: int = 1000,
        max_size_test: int = 1600,
        confidence_threshold: float = 0.40,
        nms_iou_threshold: float = 0.50,
        classifier: Optional[BaseTextRoleClassifier] = None,
        memory_guard: Optional[MemoryGuard] = None,
    ):
        self.ckpt_path = ckpt_path
        self.npz_path = npz_path
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.min_size_test = min_size_test
        self.max_size_test = max_size_test
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.classifier = classifier or SpatioLexicalGTClassifier()
        self.memory_guard = memory_guard or MemoryGuard()

        self.model = self._load_model()

    def _load_model(self) -> SWINTS:
        self.memory_guard.cleanup(force_cuda_empty=True)
        model = SWINTS(npz_path=self.npz_path)

        ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"State dict mismatch: missing={missing}, unexpected={unexpected}")

        # Immediately free CPU checkpoint dicts before moving to GPU
        del state_dict
        del ckpt
        self.memory_guard.cleanup(force_cuda_empty=True)

        model.to(self.device)
        model.eval()
        self.memory_guard.snapshot()
        return model

    def _preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, int, int, np.ndarray]:
        """
        Read image in RGB (matching Base-SWINTS_swin.yaml INPUT.FORMAT: 'RGB') and resize
        following Detectron2 ResizeShortestEdge(min_size_test, max_size_test).
        """
        with Image.open(image_path) as pil_img:
            rgb_img = np.asarray(pil_img.convert("RGB"))

        orig_h, orig_w = rgb_img.shape[:2]
        scale = float(self.min_size_test) / float(min(orig_h, orig_w))
        if round(scale * max(orig_h, orig_w)) > self.max_size_test:
            scale = float(self.max_size_test) / float(max(orig_h, orig_w))

        new_h = max(32, int(round(orig_h * scale)))
        new_w = max(32, int(round(orig_w * scale)))

        if (new_h, new_w) != (orig_h, orig_w):
            resized_rgb = cv2.resize(rgb_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized_rgb = rgb_img

        img_tensor = (
            torch.from_numpy(resized_rgb).permute(2, 0, 1).contiguous().to(
                device=self.device, dtype=torch.float32
            )
        )
        return img_tensor, orig_h, orig_w, rgb_img

    @torch.inference_mode()
    def annotate_image(
        self,
        image_path: str,
        image_id: Any,
        qa_annotations: Optional[List[Dict[str, Any]]] = None,
        vis_output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run full auto-annotation pipeline on a single image with strict RAM/VRAM reclamation.
        """
        t0 = time.perf_counter()
        self.memory_guard.check_and_reclaim()

        img_tensor, orig_h, orig_w, rgb_img = self._preprocess_image(image_path)
        raw_out = self.model.inference_single_image(
            img_tensor,
            orig_h=orig_h,
            orig_w=orig_w,
            confidence_threshold=self.confidence_threshold,
        )

        # Free GPU input tensor immediately after forward pass
        del img_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        boxes = raw_out["boxes"]
        scores = raw_out["scores"]
        masks = raw_out["masks"]
        recs = raw_out["recs"]

        candidate_detections: List[Dict[str, Any]] = []
        for idx in range(len(scores)):
            score_val = float(scores[idx].item())
            text_str = decode_recognition_tokens(recs[idx])
            if not text_str:
                continue

            poly_and_bbox = extract_tight_polygon_and_bbox(
                masks[idx], boxes[idx], img_h=orig_h, img_w=orig_w, threshold=0.5
            )
            if poly_and_bbox is None:
                continue
            poly_pts, tight_bbox = poly_and_bbox

            candidate_detections.append(
                {
                    "text": text_str,
                    "score": round(score_val, 4),
                    "bbox": tight_bbox,
                    "raw_bbox": [int(round(v)) for v in boxes[idx].tolist()],
                    "polygon": poly_pts,
                }
            )

        del raw_out, boxes, scores, masks, recs

        # Polygon NMS + Top-Left to Bottom-Right (TL-BR) reading order sorting
        nms_detections = polygon_nms(candidate_detections, iou_thresh=self.nms_iou_threshold)
        sorted_detections = sort_detections_tl_br(nms_detections)

        # Group proximate word boxes on the same line into phrase/line bounding boxes
        merged_lines = group_proximate_text_lines(sorted_detections, max_gap_height_ratio=1.35)

        # Classify into Ground-Truth Answer Regions (`R_target`) vs Auxiliary Scene Text Lines (`R_distractor`)
        role_summary = self.classifier.classify(
            sorted_detections,
            qa_annotations or [],
            merged_lines=merged_lines,
            img_w=orig_w,
            img_h=orig_h,
        )

        if vis_output_path:
            self._render_visualization(
                rgb_img=rgb_img,
                detections=sorted_detections,
                merged_lines=merged_lines,
                answer_regions=role_summary["answer_regions"],
                qa_groundings=role_summary["qa_groundings"],
                vis_output_path=vis_output_path,
            )

        del rgb_img
        mem_snap = self.memory_guard.cleanup(force_cuda_empty=True)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "image_id": image_id,
            "image_name": os.path.basename(image_path),
            "width": orig_w,
            "height": orig_h,
            "latency_ms": round(elapsed_ms, 1),
            "num_answer_regions": role_summary["num_answer_regions"],
            "num_total_lines": role_summary["num_total_lines"],
            "num_gt_answer_lines": role_summary["num_gt_answer_lines"],
            "num_auxiliary_lines": role_summary["num_auxiliary_lines"],
            "num_total_ocr": role_summary["num_total_ocr"],
            "num_gt_answer_ocr": role_summary["num_gt_answer_ocr"],
            "num_auxiliary_ocr": role_summary["num_auxiliary_ocr"],
            "answer_regions": role_summary["answer_regions"],
            "qa_groundings": role_summary["qa_groundings"],
            "gt_answer_lines": role_summary["gt_answer_lines"],
            "auxiliary_lines": role_summary["auxiliary_lines"],
            "gt_answer_regions": role_summary["gt_answer_regions"],
            "auxiliary_text_regions": role_summary["auxiliary_text_regions"],
            "all_merged_lines_tl_br": merged_lines,
            "all_detections_tl_br": sorted_detections,
            "memory": {
                "ram_used_gb": mem_snap.ram_used_gb,
                "ram_percent": mem_snap.ram_percent,
                "vram_allocated_mb": mem_snap.vram_allocated_mb,
                "vram_peak_mb": mem_snap.vram_peak_mb,
            },
        }

    def _render_visualization(
        self,
        rgb_img: np.ndarray,
        detections: List[Dict[str, Any]],
        merged_lines: List[Dict[str, Any]],
        answer_regions: List[Dict[str, Any]],
        qa_groundings: List[Dict[str, Any]],
        vis_output_path: str,
    ) -> None:
        """
        Render clean phrase/region-level annotated visual overlay for rapid manual inspection:
        - Solid Green BBox (`[GT-R1]`, `[GT-R2]`...): Expanded Answer Region BBoxes (1 image can have multiple
          answer regions for multiple questions; includes adjacent local context/noisy text).
        - Inner Cyan BBox: Exact matched answer span within the region.
        - Solid Orange BBox (`[AUX]`): Proximate auxiliary/distractor text lines.
        """
        os.makedirs(os.path.dirname(os.path.abspath(vis_output_path)), exist_ok=True)
        pil_canvas = Image.fromarray(rgb_img).convert("RGBA")
        overlay = Image.new("RGBA", pil_canvas.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        img_w, img_h = pil_canvas.size
        font_size = max(13, min(24, int(min(img_w, img_h) * 0.020)))
        font = None
        for font_candidate in [
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/tahoma.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]:
            if os.path.exists(font_candidate):
                try:
                    font = ImageFont.truetype(font_candidate, font_size)
                    break
                except Exception:
                    pass
        if font is None:
            font = ImageFont.load_default()

        placed_label_boxes: List[Tuple[int, int, int, int]] = []

        def _place_non_overlapping_label(
            x1: int,
            y1: int,
            label_text: str,
            bg_color: Tuple[int, int, int, int],
            fg_color: Tuple[int, int, int, int],
        ) -> None:
            tx = max(2, min(x1, img_w - 120))
            ty = max(2, y1 - font_size - 5)
            tb = draw.textbbox((tx, ty), label_text, font=font)
            tw = tb[2] - tb[0]
            th = tb[3] - tb[1]
            if tx + tw > img_w - 2:
                tx = max(2, img_w - tw - 4)
            for _ in range(4):
                cand = (tx, ty, tx + tw + 4, ty + th + 2)
                collision = False
                for ox1, oy1, ox2, oy2 in placed_label_boxes:
                    if not (cand[2] <= ox1 or cand[0] >= ox2 or cand[3] <= oy1 or cand[1] >= oy2):
                        collision = True
                        ty = max(2, oy1 - th - 3)
                        break
                if not collision:
                    break
            final_box = (tx, ty, tx + tw + 4, ty + th + 2)
            placed_label_boxes.append(final_box)
            draw.rectangle(final_box, fill=bg_color)
            draw.text((tx + 2, ty + 1), label_text, fill=fg_color, font=font)

        # 1. Draw subtle inner word masks first
        for line_obj in merged_lines:
            is_gt = line_obj.get("role") == "gt_answer"
            fill_col = (0, 230, 90, 50) if is_gt else (255, 165, 0, 30)
            for w_idx in line_obj.get("word_indices", []):
                pts = [tuple(p) for p in detections[w_idx]["polygon"]]
                if len(pts) >= 3:
                    draw.polygon(pts, fill=fill_col)

        # 2. Draw merged Auxiliary phrase/line bounding boxes (`R_distractor`)
        for line_obj in merged_lines:
            if line_obj.get("role") != "auxiliary":
                continue
            x1, y1, x2, y2 = line_obj["bbox"]
            draw.rectangle([x1, y1, x2, y2], outline=(255, 145, 0, 210), width=2)
            if float(line_obj.get("score", 0.0)) >= 0.55:
                label = f"[AUX] {line_obj['text']} ({line_obj['score']:.2f})"
                _place_non_overlapping_label(
                    x1, y1, label, bg_color=(35, 30, 20, 195), fg_color=(255, 205, 95, 255)
                )

        # 3. Draw each Answer Region BBox (`answer_regions` - supports multiple quests per image)
        for reg in answer_regions:
            rx1, ry1, rx2, ry2 = reg["bbox"]
            draw.rectangle([rx1, ry1, rx2, ry2], outline=(0, 255, 100, 255), width=3)
            ebox = reg.get("exact_answer_bbox")
            if ebox and ebox != [rx1, ry1, rx2, ry2]:
                ex1, ey1, ex2, ey2 = ebox
                draw.rectangle([ex1, ey1, ex2, ey2], outline=(0, 240, 255, 255), width=2)
            ans_str = " / ".join(dict.fromkeys(reg.get("gt_answers", [])))
            label = f"[GT-R{reg['region_id']}] {reg['ocr_text_in_region']} -> ĐA: {ans_str}"
            _place_non_overlapping_label(
                rx1, ry1, label, bg_color=(0, 115, 45, 235), fg_color=(255, 255, 255, 255)
            )

        combined = Image.alpha_composite(pil_canvas, overlay).convert("RGB")
        combined.save(vis_output_path, quality=92)
