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
    pool_query_agnostic_gradient_tokens,
    compute_refusal_resistant_continuation_horizon,
    adaptive_tensor_stroke_dilation,
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
    "pool_query_agnostic_gradient_tokens",
    "compute_refusal_resistant_continuation_horizon",
    "adaptive_tensor_stroke_dilation",
]
