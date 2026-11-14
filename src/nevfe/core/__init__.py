# -*- coding: utf-8 -*-
"""
NE-VFE Core Mathematical Package: Information-Geometric Variational Free Energy Decoding (IG-VFED).
"""

from src.nevfe.core.engine import (
    AlgorithmConfig,
    NEVFEEngine,
    SPEC_REF,
    compute_bhattacharyya_coefficient,
    compute_dynamic_coupling,
    compute_dynamic_plausibility_floor,
    compute_mean_field_reference_embeddings,
    compute_optical_anchor_posterior,
    compute_unnormalized_entropy,
    nevfe_step_logits,
    plan_overlapped_strip_boxes,
    restore_wiener_optical_strokes,
    select_optical_foveal_loci,
)

__all__ = [
    "AlgorithmConfig",
    "NEVFEEngine",
    "SPEC_REF",
    "compute_optical_anchor_posterior",
    "select_optical_foveal_loci",
    "plan_overlapped_strip_boxes",
    "restore_wiener_optical_strokes",
    "compute_mean_field_reference_embeddings",
    "compute_bhattacharyya_coefficient",
    "compute_dynamic_coupling",
    "compute_dynamic_plausibility_floor",
    "compute_unnormalized_entropy",
    "nevfe_step_logits",
]
