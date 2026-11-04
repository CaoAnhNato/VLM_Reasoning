# -*- coding: utf-8 -*-
"""
NE-VFE Core Mathematical Package.
"""

from src.nevfe.core.engine import (
    NEVFEEngine,
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
    "modulate_patch_embeddings",
    "compute_bhattacharyya_coefficient",
    "compute_dynamic_coupling",
    "compute_unnormalized_entropy",
    "compute_candidate_support_mask",
    "compute_free_energy_support_mask",
    "nevfe_step_logits",
]
