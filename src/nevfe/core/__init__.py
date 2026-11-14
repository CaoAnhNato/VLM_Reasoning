# -*- coding: utf-8 -*-
"""
NE-VFE Core Mathematical Package: Information-Geometric Variational Free Energy Decoding (IG-VFED).
"""

from src.nevfe.core.engine import (
    AlgorithmConfig,
    NEVFEEngine,
    SPEC_REF,
    SPEC_SHA256,
    compute_bhattacharyya_coefficient,
    compute_dynamic_coupling,
    compute_dynamic_plausibility_floor,
    compute_mean_field_reference_embeddings,
    compute_optical_anchor_posterior,
    compute_unnormalized_entropy,
    nevfe_step_logits,
    plan_isotropic_strip_boxes,
    restore_extremum_wiener_strokes,
    select_optical_foveal_locus,
)

__all__ = [
    "AlgorithmConfig",
    "NEVFEEngine",
    "SPEC_REF",
    "SPEC_SHA256",
    "compute_optical_anchor_posterior",
    "select_optical_foveal_locus",
    "plan_isotropic_strip_boxes",
    "restore_extremum_wiener_strokes",
    "compute_mean_field_reference_embeddings",
    "compute_bhattacharyya_coefficient",
    "compute_dynamic_coupling",
    "compute_dynamic_plausibility_floor",
    "compute_unnormalized_entropy",
    "nevfe_step_logits",
]
