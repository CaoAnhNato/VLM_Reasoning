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
from typing import Dict, List, Optional, Tuple, Union

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

    Round 5 SOTA Algorithmic Upgrades:
    - CAPC (Contrast-Aware Plausibility Constraint) with contrastive retention floor theta_c = 0.35.
    - AHRA (Answer-Head Recency Anchoring) boosting initial entity head contrast.
    - QIACF (Question-Intent Adaptive Contrast Factor) scaling via step-0 visual divergence D_JS^(0).
    - TESG (Temperature & Entropy Synergistic Gating) sharpening logits under high entropy.
    - ESSC (Entity-Sustained Soft Cooldown) with softened geometric decay gamma_phase = 0.75.
    """

    def __init__(
        self,
        alpha_0: float = 0.65,
        beta_0: float = 0.035,
        tau_js: float = 0.04,
        gamma_decay: float = 0.50,
        lambda_decay: float = 0.08,
        tau_entropy_low: float = 0.35,
        tau_entropy_high: float = 2.20,
        gamma_phase: float = 0.75,
        theta_c: float = 0.35,
        delta_head: float = 0.45,
        tau_head: float = 1.8,
        kappa_intent: float = 0.35,
        tau_intent: float = 0.15,
        eta_temp: float = 0.18,
        tau_h: float = 1.50,
        # Round 6 DCEB Parameters:
        kappa_eos: float = 4.00,
        n_min_eos: int = 1,
        h_bar_eos: float = 1.80,
        tau_h_eos: float = 0.40,
        delta_peak_min: float = 0.50,
    ):
        """
        Args:
            alpha_0: Base contrastive weight
            beta_0: Base APC truncation cutoff
            tau_js: Jensen-Shannon divergence syntax gate threshold
            gamma_decay: Visual information decay compensation strength
            lambda_decay: Exponential rate of visual information decay
            tau_entropy_low: Predictive entropy lower bound for visual certainty
            tau_entropy_high: Predictive entropy upper bound for noise gating
            gamma_phase: Exponential cooldown rate for soft reasoning transition (ESSC, 0.75)
            theta_c: Contrast-Aware Plausibility Constraint (CAPC) retention floor (0.35)
            delta_head: Answer-Head Recency Anchoring (AHRA) boost factor (0.45)
            tau_head: Answer-Head Recency Anchoring (AHRA) decay rate (1.8)
            kappa_intent: Question-Intent Adaptive Contrast Factor (QIACF) scale (0.35)
            tau_intent: Question-Intent Adaptive Contrast Factor (QIACF) tau (0.15)
            eta_temp: Temperature & Entropy Synergistic Gating (TESG) cooling scale (0.18)
            tau_h: Temperature & Entropy Synergistic Gating (TESG) entropy tau (1.50)
            kappa_eos: Dynamic Contrastive EOS Boosting (DCEB) maximum logit boost scale (4.00)
            n_min_eos: DCEB minimum generation step before EOS boosting is enabled (1)
            h_bar_eos: DCEB predictive entropy midpoint threshold (1.80 nats)
            tau_h_eos: DCEB predictive entropy logistic sigmoid scale (0.40)
            delta_peak_min: DCEB minimum historical peak contrast floor (0.50)
        """
        self.alpha_0 = float(alpha_0)
        self.beta_0 = float(beta_0)
        self.tau_js = float(tau_js)
        self.gamma_decay = float(gamma_decay)
        self.lambda_decay = float(lambda_decay)
        self.tau_entropy_low = float(tau_entropy_low)
        self.tau_entropy_high = float(tau_entropy_high)
        self.gamma_phase = float(gamma_phase)
        self.theta_c = float(theta_c)
        self.delta_head = float(delta_head)
        self.tau_head = float(tau_head)
        self.kappa_intent = float(kappa_intent)
        self.tau_intent = float(tau_intent)
        self.eta_temp = float(eta_temp)
        self.tau_h = float(tau_h)
        self.kappa_eos = float(kappa_eos)
        self.n_min_eos = int(n_min_eos)
        self.h_bar_eos = float(h_bar_eos)
        self.tau_h_eos = float(tau_h_eos)
        self.delta_peak_min = float(delta_peak_min)
        self._current_step0_d_js: Optional[float] = None
        self._peak_delta_z: float = 0.0

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
        phase_cooldown_step: int = 0,
        initial_d_js: Optional[float] = None,
        eos_id: Optional[Union[int, List[int]]] = None,
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

        # 2. CAPC (Contrast-Aware Plausibility Constraint):
        # V_head(t) = { w in V | P_pos(w) >= beta_t * max_v P_pos(v) or Delta z_t(w) >= theta_c * max_v Delta z_t(v) }
        beta_t = self.beta_0 * math.exp(-min(4.0, max(0.0, entropy_val)))
        max_p_pos = p_pos.max(dim=-1, keepdim=True).values
        apc_mask = p_pos >= (beta_t * max_p_pos)

        delta_z = torch.relu(pos_logits - neg_logits)
        max_delta_z = delta_z.max(dim=-1, keepdim=True).values
        has_contrast = max_delta_z > 1e-6
        contrast_mask = (delta_z >= (self.theta_c * max_delta_z)) & has_contrast
        valid_mask = apc_mask | contrast_mask

        # Track historical peak visual contrast across generation steps (DCEB sensor)
        m_t = float(max_delta_z.squeeze().item())
        if step == 0:
            self._peak_delta_z = m_t
        else:
            self._peak_delta_z = max(self._peak_delta_z, m_t)

        # 3. Dynamic Spatio-Temporal Calibration alpha_t
        s_spatial = min(1.0, max(0.0, float(spatial_confidence)))

        # M_decay(t) = 1 + gamma * (1 - exp(-lambda * t)) / (exp(-lambda * t) + eps), bounded <= 1.50
        decay_exp = math.exp(-self.lambda_decay * step)
        m_decay = min(1.50, 1.0 + self.gamma_decay * ((1.0 - decay_exp) / (decay_exp + 1e-6)))

        # D_KL and D_JS
        kl_div = (p_pos * (torch.log(p_pos.clamp(min=1e-9)) - torch.log(p_neg.clamp(min=1e-9)))).sum(dim=-1, keepdim=True)
        d_js = self.compute_js_divergence(p_pos, p_neg)
        d_js_val = float(d_js.squeeze().item())

        # Track or use step-0 visual divergence for QIACF
        if step == 0 or initial_d_js is not None:
            d_js_0 = d_js_val if initial_d_js is None else float(initial_d_js)
            self._current_step0_d_js = d_js_0
        else:
            d_js_0 = self._current_step0_d_js if self._current_step0_d_js is not None else d_js_val

        # QIACF: Psi_intent = 1.0 + kappa_intent * tanh(D_JS^(0) / tau_intent)
        psi_intent = 1.0 + self.kappa_intent * math.tanh(d_js_0 / self.tau_intent)

        # AHRA: A_head(t) = 1.0 + delta_head * exp(-t / tau_head)
        a_head = 1.0 + self.delta_head * math.exp(-float(step) / self.tau_head)

        # Syntax gating: if D_JS <= tau_js, both branches agree (syntactic connector) -> gate = 0
        is_semantic = float(d_js_val > self.tau_js)
        u_div = torch.tanh(torch.relu(kl_div)) * is_semantic

        # Entropy Modulation Gate: G_entropy(H) in [0, 1]
        g_entropy_low = math.tanh(max(0.0, entropy_val) / self.tau_entropy_low)
        g_entropy_high = 1.0 - math.tanh(max(0.0, entropy_val - self.tau_entropy_high))
        g_entropy = float(np_clip(g_entropy_low * g_entropy_high, 0.0, 1.0))

        alpha_t = float(
            self.alpha_0
            * s_spatial
            * m_decay
            * float(u_div.squeeze().item())
            * g_entropy
            * a_head
            * psi_intent
        )
        if phase_cooldown_step > 0:
            alpha_t *= (self.gamma_phase ** phase_cooldown_step)
        alpha_t = max(0.0, min(2.0, alpha_t))

        # TESG: z_t^*(w) = (z_t^(+)(w) + alpha_t * Delta z_t(w)) / T(H_t)
        # where T(H_t) = 1.0 - eta_T * tanh(H_t / tau_H)
        t_entropy = max(0.05, 1.0 - self.eta_temp * math.tanh(entropy_val / self.tau_h))
        calibrated_logits = (pos_logits + alpha_t * delta_z) / t_entropy

        # 4. Dynamic Contrastive EOS Boosting (DCEB) & CAPC EOS Preservation:
        effective_peak = max(self._peak_delta_z, self.delta_peak_min)
        c_collapse = max(0.0, 1.0 - m_t / (effective_peak + 1e-6))
        sig_entropy = 1.0 / (1.0 + math.exp(-np_clip((entropy_val - self.h_bar_eos) / self.tau_h_eos, -20.0, 20.0)))
        tau_eos = float(self.kappa_eos * c_collapse * sig_entropy) if step >= self.n_min_eos else 0.0

        if eos_id is not None:
            eos_ids = [eos_id] if isinstance(eos_id, int) else list(eos_id)
            for eid in eos_ids:
                valid_mask[..., eid] = True  # Ensure EOS is never truncated by APC/contrast mask
                if tau_eos > 0.0:
                    calibrated_logits[..., eid] += tau_eos

        calibrated_logits = calibrated_logits.masked_fill(~valid_mask, float("-inf"))
        next_token_id = int(torch.argmax(calibrated_logits, dim=-1).item())
        return next_token_id, alpha_t, entropy_val, d_js_val
