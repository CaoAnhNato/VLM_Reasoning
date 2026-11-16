# -*- coding: utf-8 -*-
"""
CERV (Counterfactual Evidence Routing & Verification) Package.

Implements the 3-stage training-free causal inference framework specified in
`docs/Proposed_Method_CERV.md`:
  - Stage 1: Bayesian Semantic-Fork Gate (Lambda_t = EVSI_t / R_0(t) > 1)
  - Stage 2: Prefix-Conditioned VAQ Localization at t* (Peak layer l*, TopPatches rho=0.85, I+ & I-)
  - Stage 3: Fork-Anchored Counterfactual Verification (t >= t*) with Adaptive Information Gain alpha_t = JSD / H(m_t)
"""

from .pipeline import CERVPipeline, CERVResult, CERVPipelineResult

__all__ = ["CERVPipeline", "CERVResult", "CERVPipelineResult"]
