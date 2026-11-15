# -*- coding: utf-8 -*-
"""
NE-VFE Core Mathematical Package: Information-Geometric Variational Free Energy Decoding (IG-VFED).
"""

from src.nevfe.core.engine import (
    AlgorithmConfig,
    NEVFEEngine,
    SPEC_REF,
    assemble_foveated_visual_sequence,
    compute_bhattacharyya_coefficient,
    compute_dynamic_coupling,
    compute_dynamic_plausibility_floor,
    compute_mean_field_reference_embeddings,
    compute_semantic_specificity_posterior,
    compute_unnormalized_entropy,
    nevfe_step_logits,
    select_active_foveal_subwindow,
)

__all__ = [
    "AlgorithmConfig",
    "NEVFEEngine",
    "SPEC_REF",
    "assemble_foveated_visual_sequence",
    "compute_mean_field_reference_embeddings",
    "compute_semantic_specificity_posterior",
    "select_active_foveal_subwindow",
    "compute_bhattacharyya_coefficient",
    "compute_dynamic_coupling",
    "compute_dynamic_plausibility_floor",
    "compute_unnormalized_entropy",
    "nevfe_step_logits",
]
