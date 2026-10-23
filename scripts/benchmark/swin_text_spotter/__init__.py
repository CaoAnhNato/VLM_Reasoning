# -*- coding: utf-8 -*-
"""
SwinTextSpotter (SWINTS-swin-vintext) Automated Scene-Text Annotation Package for ViTextVQA.
"""

from .annotator_pipeline import (
    MemoryGuard,
    MemorySnapshot,
    RAMPrefetchCache,
    SwinTextSpotterAnnotator,
)
from .gt_classifier import BaseTextRoleClassifier, SpatioLexicalGTClassifier
from .modeling import SWINTS
from .vintext_decoder import (
    CTLABELS,
    decode_recognition_tokens,
    extract_tight_polygon_and_bbox,
    group_proximate_text_lines,
    polygon_nms,
    sort_detections_tl_br,
    vintext_decoder,
)

__all__ = [
    "SWINTS",
    "MemoryGuard",
    "MemorySnapshot",
    "RAMPrefetchCache",
    "SwinTextSpotterAnnotator",
    "BaseTextRoleClassifier",
    "SpatioLexicalGTClassifier",
    "CTLABELS",
    "vintext_decoder",
    "decode_recognition_tokens",
    "extract_tight_polygon_and_bbox",
    "group_proximate_text_lines",
    "polygon_nms",
    "sort_detections_tl_br",
]
