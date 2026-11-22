# -*- coding: utf-8 -*-
"""
NE-VFE Core Mathematical Package: Frozen FRSA-PBCF ("011") + IG-VFED (1, 1, 1).
"""

from src.nevfe.core.engine import (
    AlgorithmConfig,
    COMPONENT_NAMES,
    NEVFEEngine,
    SPEC_REF,
    compute_bhattacharyya_coefficient,
    compute_dynamic_coupling,
    compute_dynamic_plausibility_floor,
    compute_unnormalized_entropy,
    nevfe_step_logits,
)
from src.nevfe.core.frozen_foveation import (
    compute_lexical_spatial_distribution,
    compute_mean_field_reference_embeddings,
    foveate_frozen_011,
    select_poisson_bregman_subwindow,
)

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
]
