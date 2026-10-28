# -*- coding: utf-8 -*-
"""
Stage 2: Dynamic Spatio-Temporal Calibration & Dual-Role Contrastive Decoding (Features 6, 7 in PROJECT.md).

Mathematical Formulation:
1. Dual-Role Negative Prior View:
   Contrasts high-resolution evidence [I_global, I_ROI] against [I_global, I_ROI_lowpass].
   - Small target text: lowpass filtering dissolves sub-pixel character strokes, isolating
     unsupported language priors and hallucinations (Case 1a, 1b).
   - Salient distractor signboards: legible on both branches, cancelling out in
     z_t^(+) - z_t^(-) (Case 2b).

2. Dynamic Spatio-Temporal Calibration:
   alpha_t = alpha_0 * S_spatial * M_decay(t) * U_divergence(t)
   - S_spatial = tanh(max(0, S_Stage1)) in [0, 1)
   - M_decay(t) = 1 + gamma * (1 - exp(-lambda * t)) / (exp(-lambda * t) + eps)
   - U_divergence(t) = tanh(D_KL(P_pos || P_neg)) * I(D_JS(P_pos || P_neg) > tau_JS)
     where Jensen-Shannon divergence gating enforces alpha_t -> 0 on syntactic connectors,
     preserving 100% SFT grammatical fluency.

3. Entropy-Adaptive Plausibility Truncation (EAPT):
   beta_t = beta_0 * exp(-H(P_t^(+)))
   V_head(t) = { w in V | P_t^(+)(w) >= beta_t * max_v P_t^(+)(v) }
   Dynamic expansion under visual ambiguity prevents discarding rare proper-name tokens.
"""

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageFilter

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None


@dataclass
class DecodingDiagnostics:
    """Diagnostic statistics gathered during dynamic contrastive decoding."""
    alpha_mean: float
    entropy_mean: float
    num_tokens: int
    js_divergence_mean: float
    syntax_gated_fraction: float


def compute_adaptive_lowpass_radius(
    magnification: float = 1.0,
    char_height_px: Optional[float] = None,
    base_radius: float = 5.5,
) -> float:
    """
    Calculate an adaptive Gaussian blur radius for the negative ROI tile.
    
    Rather than fixing radius=5.5 px, this scales with character scale and magnification:
    - High magnification / small text needs a moderate blur (3.5-5.0 px) to dissolve sub-pixel strokes.
    - Large text needs a proportionally larger blur radius (6.5-8.5 px) to dissolve thick stroke centers.
    """
    if char_height_px is not None and char_height_px > 0:
        h_proj = char_height_px * magnification
        radius = float(np_clip(h_proj * 0.08 + 2.8, 3.0, 8.5))
        return radius

    # Fallback scaling with magnification: higher magnification -> smaller original text
    mag_clamped = max(0.5, min(6.0, float(magnification)))
    radius = base_radius + (1.0 / mag_clamped - 1.0) * 1.5
    return float(max(3.0, min(8.0, radius)))


def np_clip(val: float, low: float, high: float) -> float:
    """Helper for clipping scalar floats."""
    return max(low, min(high, val))


class DynamicContrastiveDecoder:
    """
    Executes dynamic spatio-temporal calibrated contrastive decoding on top of
    Qwen2.5 language model in torch.bfloat16.
    """

    def __init__(
        self,
        alpha_0: float = 0.65,
        beta_0: float = 0.035,
        tau_js: float = 0.04,
        gamma_decay: float = 0.50,
        lambda_decay: float = 0.08,
    ):
        """
        Args:
            alpha_0: Base contrastive weight
            beta_0: Base APC truncation cutoff
            tau_js: Jensen-Shannon divergence syntax gate threshold
            gamma_decay: Visual information decay compensation strength
            lambda_decay: Exponential rate of visual information decay
        """
        self.alpha_0 = float(alpha_0)
        self.beta_0 = float(beta_0)
        self.tau_js = float(tau_js)
        self.gamma_decay = float(gamma_decay)
        self.lambda_decay = float(lambda_decay)

    def compute_js_divergence(
        self,
        p_pos: "torch.Tensor",
        p_neg: "torch.Tensor",
        eps: float = 1e-9,
    ) -> "torch.Tensor":
        """
        Compute Jensen-Shannon Divergence D_JS(P || Q) = 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M)
        where M = 0.5 * (P + Q). Guaranteed symmetric and bounded in [0, log(2)].
        """
        m = 0.5 * (p_pos + p_neg)
        log_m = torch.log(m.clamp(min=eps))
        kl_p_m = (p_pos * (torch.log(p_pos.clamp(min=eps)) - log_m)).sum(dim=-1, keepdim=True)
        kl_q_m = (p_neg * (torch.log(p_neg.clamp(min=eps)) - log_m)).sum(dim=-1, keepdim=True)
        d_js = 0.5 * (kl_p_m + kl_q_m)
        return torch.clamp(d_js, min=0.0)

    def decode_step(
        self,
        pos_logits: "torch.Tensor",
        neg_logits: "torch.Tensor",
        step: int,
        spatial_confidence: float,
    ) -> Tuple[int, float, float, float]:
        """
        Execute a single calibrated decoding step.
        
        Returns:
            next_token_id: Selected token ID
            alpha_t: Calibrated contrast factor used at this step
            entropy_t: Predictive entropy of positive distribution
            d_js_val: Jensen-Shannon divergence
        """
        p_pos = F.softmax(pos_logits, dim=-1)
        p_neg = F.softmax(neg_logits, dim=-1)

        # 1. Predictive Entropy H(P_pos)
        log_p_pos = torch.log(p_pos.clamp(min=1e-12))
        entropy_t = -torch.sum(p_pos * log_p_pos, dim=-1, keepdim=True)
        entropy_val = float(entropy_t.squeeze().item())

        # 2. Entropy-Adaptive Plausibility Truncation (EAPT): beta_t = beta_0 * exp(-H(P_pos))
        beta_t = self.beta_0 * math.exp(-min(5.0, max(0.0, entropy_val)))
        max_p_pos = p_pos.max(dim=-1, keepdim=True).values
        valid_mask = p_pos >= (beta_t * max_p_pos)

        # 3. Dynamic Spatio-Temporal Calibration alpha_t
        # S_spatial = tanh(max(0, S_Stage1)), directly passed without redundant second tanh
        s_spatial = min(1.0, max(0.0, float(spatial_confidence)))

        # M_decay(t) = 1 + gamma * (1 - exp(-lambda * t)) / (exp(-lambda * t) + eps), bounded <= 1.50
        decay_exp = math.exp(-self.lambda_decay * step)
        m_decay = min(1.50, 1.0 + self.gamma_decay * ((1.0 - decay_exp) / (decay_exp + 1e-6)))

        # D_KL and D_JS
        kl_div = (p_pos * (torch.log(p_pos.clamp(min=1e-9)) - torch.log(p_neg.clamp(min=1e-9)))).sum(dim=-1, keepdim=True)
        d_js = self.compute_js_divergence(p_pos, p_neg)
        d_js_val = float(d_js.squeeze().item())

        # Syntax gating: if D_JS <= tau_js, both branches agree (syntactic connector) -> gate = 0
        is_semantic = float(d_js_val > self.tau_js)
        u_div = torch.tanh(torch.relu(kl_div)) * is_semantic

        alpha_t = float(self.alpha_0 * s_spatial * m_decay * float(u_div.squeeze().item()))
        alpha_t = max(0.0, min(1.8, alpha_t))

        # Calibrated logits
        calibrated_logits = pos_logits + alpha_t * (pos_logits - neg_logits)
        calibrated_logits = calibrated_logits.masked_fill(~valid_mask, float("-inf"))

        next_token_id = int(torch.argmax(calibrated_logits, dim=-1).item())
        return next_token_id, alpha_t, entropy_val, d_js_val
