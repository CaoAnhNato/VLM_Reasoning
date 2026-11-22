# -*- coding: utf-8 -*-
"""
Non-Equilibrium Variational Free Energy (NE-VFE) Package.
"""

from src.nevfe.core import (
    AlgorithmConfig,
    COMPONENT_NAMES,
    NEVFEEngine,
    SPEC_REF,
    compute_bhattacharyya_coefficient,
    compute_dynamic_coupling,
    compute_dynamic_plausibility_floor,
    compute_lexical_spatial_distribution,
    compute_mean_field_reference_embeddings,
    compute_unnormalized_entropy,
    foveate_frozen_011,
    nevfe_step_logits,
    select_poisson_bregman_subwindow,
)
from src.nevfe.adapter.vintern_adapter import VinternNEVFEAdapter
from src.nevfe.pipeline import NEVFEPipeline, NEVFEResult, parse_ablation_mask, resolve_algorithm_config

__all__ = [
    "AlgorithmConfig",
    "COMPONENT_NAMES",
    "NEVFEEngine",
    "SPEC_REF",
    "compute_bhattacharyya_coefficient",
    "compute_dynamic_coupling",
    "compute_dynamic_plausibility_floor",
    "compute_lexical_spatial_distribution",
    "compute_mean_field_reference_embeddings",
    "compute_unnormalized_entropy",
    "foveate_frozen_011",
    "nevfe_step_logits",
    "select_poisson_bregman_subwindow",
    "VinternNEVFEAdapter",
    "NEVFEPipeline",
    "NEVFEResult",
    "parse_ablation_mask",
    "resolve_algorithm_config",
]
