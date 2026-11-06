# -*- coding: utf-8 -*-
"""
NE-VFE Core Mathematical Package: Mean-Field Contrastive Reference Decoding (MF-CD).
"""

from src.nevfe.core.engine import (
    NEVFEEngine,
    compute_mean_field_reference_embeddings,
    mean_field_contrastive_step,
    modulate_patch_embeddings,
    compute_bhattacharyya_coefficient,
    compute_dynamic_coupling,
    compute_unnormalized_entropy,
    compute_candidate_support_mask,
    compute_free_energy_support_mask,
    nevfe_step_logits,
)

__all__ = [
    "NEVFEEngine",
    "compute_mean_field_reference_embeddings",
    "mean_field_contrastive_step",
    "modulate_patch_embeddings",
    "compute_bhattacharyya_coefficient",
    "compute_dynamic_coupling",
    "compute_unnormalized_entropy",
    "compute_candidate_support_mask",
    "compute_free_energy_support_mask",
    "nevfe_step_logits",
]
