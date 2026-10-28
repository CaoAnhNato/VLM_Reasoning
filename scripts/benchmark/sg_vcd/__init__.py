# -*- coding: utf-8 -*-
"""
SG-VCD (Spatial-Grounding & Information-Gain Contrastive Decoding) Package.

Implements a unified, zero-heuristic, training-free pipeline grounded in
Information Theory, Kullback-Leibler Divergence (D_KL), and Dynamic PMI Calibration:
1. Stage 1 (Spatial Grounding & Multi-Scale Dual-Tile Fusion):
   - Information-theoretic query weighting w_t = D_KL(p_t || U(Omega))
   - Neutral prompt attention sink baseline subtraction A_0(u) for layers 8-16
   - Local character scale H_char self-calibration via 2D/1D autocorrelation
   - Prototype-locked seed propagation in opponent space (L, C_rg, C_yb)
   - +-0.5 * H_char Vietnamese diacritic belt expansion
   - Scale-Invariant Bernoulli KL Divergence Region Selection J_KL(C) with 1D Otsu border trimming
   - Continuous Isotropic Square ROI Transformation s = max(Delta_x, Delta_y) and dual-tile fusion [I_global, I_ROI]
2. Stage 2 (Prior-Calibrated Contrastive Decoding):
   - Dual-role negative prior [I_global, I_ROI_lowpass] to simultaneously cancel salient distractors and isolate language priors
   - Dynamic spatio-temporal calibration alpha_t with Jensen-Shannon divergence syntax gating
   - Entropy-adaptive plausibility truncation beta_t = beta_0 * exp(-H(P_t^(+)))
   - Visual token caching (FeatureCache) for shared representations
"""

from .adaptive_propagator import (
    AdaptiveScalePropagator,
    BaseSeedPropagator,
    LocalScalePrototype,
    SeedPropagationResult,
    compute_autocorrelation_char_scale_1d,
    compute_bbox_iou_and_recall,
)
from .attention_extractor import (
    BaseAttentionExtractor,
    ExtractionResult,
    GradientAttentionExtractor,
    PointwiseKLAttentionExtractor,
    compute_intrinsic_affinity_field,
    compute_pointwise_kl_map,
)
from .bbox_selector import (
    BaseRegionSelector,
    KLDivergenceRegionSelector,
    RegionSelectionResult,
    compute_bernoulli_kl_score,
    compute_high_frequency_gradient_prior,
    compute_otsu_threshold_1d,
    modulate_affinity_with_gradient_prior,
)
from .dynamic_cd import (
    DecodingDiagnostics,
    DynamicContrastiveDecoder,
    compute_adaptive_lowpass_radius,
)
from .feature_cache import (
    VisualFeatureCache,
    default_feature_cache,
    get_or_compute_visual_features,
)
from .pipeline import (
    PipelineResult,
    SGVCDPipeline,
    strip_conversational_preamble,
)
from .roi_cropper import (
    BaseROICropper,
    CropResult,
    IsotropicROICropper,
)

__all__ = [
    "BaseAttentionExtractor",
    "ExtractionResult",
    "PointwiseKLAttentionExtractor",
    "GradientAttentionExtractor",
    "compute_pointwise_kl_map",
    "compute_intrinsic_affinity_field",
    "BaseRegionSelector",
    "RegionSelectionResult",
    "KLDivergenceRegionSelector",
    "compute_bernoulli_kl_score",
    "compute_high_frequency_gradient_prior",
    "compute_otsu_threshold_1d",
    "modulate_affinity_with_gradient_prior",
    "BaseROICropper",
    "CropResult",
    "IsotropicROICropper",
    "BaseSeedPropagator",
    "AdaptiveScalePropagator",
    "LocalScalePrototype",
    "SeedPropagationResult",
    "compute_autocorrelation_char_scale_1d",
    "compute_bbox_iou_and_recall",
    "DynamicContrastiveDecoder",
    "compute_adaptive_lowpass_radius",
    "DecodingDiagnostics",
    "VisualFeatureCache",
    "default_feature_cache",
    "get_or_compute_visual_features",
    "PipelineResult",
    "SGVCDPipeline",
    "strip_conversational_preamble",
]
