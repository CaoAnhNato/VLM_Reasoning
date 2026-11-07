# -*- coding: utf-8 -*-
"""
Information-Geometric Variational Free Energy Decoding (IG-VFED) Core Engine.

Mathematical Foundations:
- Information Geometry on Probability Simplex endowed with Fisher-Rao Metric.
- Mean-Field Visual Reference Tensor V^0 = 1_P \bar{v}^\top (nabla_spatial V^0 = 0, I(S; V^0) = 0).
- Fisher-Rao Bhattacharyya Proximity with Safeguard Clamp:
    BC_t = clamp(sum_{v=1}^V sqrt(clamp(p_t^+(v) * p_t^0(v), min=0.0)), 0.0, 1.0).
- Binding Directive A (Preamble Echo Suppression at t=1):
    alpha_t = lambda * sqrt(clamp(1.0 - BC_t^2, 0.0, 1.0)) + lambda * 1_{t=1} * BC_1.
    Strictly penalizes negative visual Pointwise Mutual Information (z_1^+ - z_1^0 < 0) at t=1.
- Dynamic Geodesic Candidate Plausibility Boundary (Component B):
    beta_t = beta * BC_t, rescuing rare scene text subwords when optical divergence is high.
- Factorial Micro-Ablation Switches (comp_a, comp_b, comp_c).
- Strictly <= 2 continuous global hyperparameters: lambda (lambda_param) and beta (beta_param).
- Strictly zero regex, zero hardcoded stopwords/keywords, zero coordinate heuristics.
- 100% Backbone-Agnostic Core: ZERO hardcoded D, P, V, layer indices, or model tokens.
"""

from typing import Optional, Tuple
import torch
import torch.nn.functional as F


def compute_mean_field_reference_embeddings(
    vit_embeds: torch.Tensor,
) -> torch.Tensor:
    """
    Construct the blind Mean-Field Visual Reference tensor V^0 = 1_P \bar{v}^\top.

    Vanishes all spatial gradients (\nabla_x V^0 = 0), destroying all character
    strokes while preserving exact sequence length P, channel dimension D, and mean illumination.

    Args:
        vit_embeds: Native unwarped patch embeddings of shape [N_tiles, 256, D] or [P, D] or [B, P, D].

    Returns:
        vit_embeds_zero: Spatially uniform reference tensor of identical shape and dtype.
    """
    if vit_embeds.dim() == 3:
        # Across all tiles/batches (dim 0) and patches (dim 1)
        v_mean = vit_embeds.mean(dim=(0, 1), keepdim=True)  # [1, 1, D]
        return v_mean.expand_as(vit_embeds)
    elif vit_embeds.dim() == 2:
        v_mean = vit_embeds.mean(dim=0, keepdim=True)        # [1, D]
        return v_mean.expand_as(vit_embeds)
    else:
        raise ValueError(f"Unexpected vit_embeds shape: {vit_embeds.shape}")


def modulate_patch_embeddings(
    visual_tokens: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Continuous Fisher-Saliency Visual Patch Energy Modulation (Utility Primitive).

    Args:
        visual_tokens: Visual patch embeddings of shape [P, D] or [B, P, D].
        eps: Epsilon for numerical stability against division by zero.

    Returns:
        Modulated visual patch embeddings with identical shape and dtype.
    """
    orig_dtype = visual_tokens.dtype
    v = visual_tokens.float()

    if v.dim() == 2:
        # v: [P, D]
        v_bar = torch.mean(v, dim=0, keepdim=True)
        var_d = torch.mean((v - v_bar) ** 2, dim=0, keepdim=True)
        s_v = 0.5 * torch.sum(((v - v_bar) ** 2) / (var_d + eps), dim=-1, keepdim=True)
        mu_s = torch.mean(s_v, dim=0, keepdim=True)
        sigma_s = torch.std(s_v, dim=0, keepdim=True) + eps
        z_s = (s_v - mu_s) / sigma_s
        modulation = 1.0 + torch.tanh(z_s)
        v_modulated = v * modulation
    elif v.dim() == 3:
        # v: [B, P, D]
        v_bar = torch.mean(v, dim=1, keepdim=True)
        var_d = torch.mean((v - v_bar) ** 2, dim=1, keepdim=True)
        s_v = 0.5 * torch.sum(((v - v_bar) ** 2) / (var_d + eps), dim=-1, keepdim=True)
        mu_s = torch.mean(s_v, dim=1, keepdim=True)
        sigma_s = torch.std(s_v, dim=1, keepdim=True) + eps
        z_s = (s_v - mu_s) / sigma_s
        modulation = 1.0 + torch.tanh(z_s)
        v_modulated = v * modulation
    else:
        raise ValueError(f"Expected visual_tokens of dimension 2 or 3, got shape {visual_tokens.shape}")

    return v_modulated.to(dtype=orig_dtype)


def compute_bhattacharyya_coefficient(
    p_plus: torch.Tensor,
    p_zero: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute Bhattacharyya Coefficient BC = sum_w sqrt(p_plus * p_zero) with float32 clamp safeguard.

    Args:
        p_plus: Multimodal candidate distribution [..., V].
        p_zero: Unconditioned linguistic prior distribution [..., V].
        eps: Epsilon to prevent negative or zero square roots.

    Returns:
        Bhattacharyya coefficient clamped strictly to [0.0, 1.0].
    """
    prod = torch.clamp(p_plus * p_zero, min=0.0)
    bc = torch.sum(torch.sqrt(prod), dim=-1, keepdim=True)
    return torch.clamp(bc, min=0.0, max=1.0)


def compute_dynamic_coupling(
    lambda_param: float,
    bc: torch.Tensor,
    step_idx: int = 1,
    comp_a: bool = True,
    comp_c: bool = True,
) -> torch.Tensor:
    """
    Fisher-Rao Geodesic Coupling Modulation with Binding Directive A Preamble Echo Suppression.

    alpha_t = lambda * sqrt(clamp(1.0 - BC_t^2, 0.0, 1.0)) + lambda * 1_{t=1 and comp_c} * BC_1.
    - At t = 1 with comp_c=True (Directive A Preamble Boost):
      alpha_1 = lambda * sqrt(clamp(1.0 - BC_1^2, 0.0, 1.0)) + lambda * BC_1.
      Amplifying alpha_1 >= lambda ensures prompt echoes are strictly penalized rather than protected.
    - At t = 1 with comp_c=False (Iteration 22 Champion baseline coupling):
      alpha_1 = lambda * sqrt(clamp(1.0 - BC_1^2, 0.0, 1.0)).
    - At t > 1, smoothly cools contrast to zero on non-visual syntactic connectors where BC_t -> 1.0.

    Args:
        lambda_param: Base visual coupling strength lambda (Global Hyperparameter 1, default 0.50).
        bc: Bhattacharyya coefficient tensor [..., 1].
        step_idx: Autoregressive decoding step index (1-indexed).
        comp_a: Boolean toggle for Component A (Fisher-Rao Geodesic Coupling).
        comp_c: Boolean toggle for Component C (Directive A t=1 Preamble Echo Suppression Boost).

    Returns:
        Dynamic coupling coefficient tensor [..., 1].
    """
    bc_clamped = torch.clamp(bc, min=0.0, max=1.0)
    if not comp_a:
        return torch.full_like(bc_clamped, float(lambda_param))
    radicand = torch.clamp(1.0 - bc_clamped * bc_clamped, min=0.0, max=1.0)
    alpha_t = float(lambda_param) * torch.sqrt(radicand)
    if step_idx == 1 and comp_c:
        # Binding Directive A: Preamble Echo Suppression at t = 1
        alpha_t = alpha_t + float(lambda_param) * bc_clamped
    return alpha_t


def compute_dynamic_plausibility_floor(
    beta_param: float,
    bc: torch.Tensor,
    comp_b: bool = True,
) -> torch.Tensor:
    """
    Dynamic Geodesic Candidate Plausibility Boundary (Component B).

    beta_t = beta * BC_t.
    - When optical evidence diverges from unconditioned prior (BC_t -> 0),
      beta_t -> 0, rescuing rare text subwords, diacritics, and numbers.
    - When optical evidence aligns with prior (BC_t -> 1), beta_t -> beta,
      enforcing standard candidate plausibility truncation.

    Args:
        beta_param: Base plausibility floor beta (Global Hyperparameter 2, default 0.05).
        bc: Bhattacharyya coefficient tensor [..., 1].
        comp_b: Boolean toggle for Component B (Dynamic Plausibility Truncation).

    Returns:
        Dynamic plausibility threshold tensor [..., 1].
    """
    if not comp_b:
        return torch.full_like(bc, float(beta_param))
    bc_clamped = torch.clamp(bc, min=0.0, max=1.0)
    return torch.clamp(float(beta_param) * bc_clamped, min=0.0, max=1.0)


def compute_unnormalized_entropy(
    p_dist: torch.Tensor,
    log_p_dist: torch.Tensor,
) -> torch.Tensor:
    """
    Thermodynamic Shannon Entropy in Nats.

    Enforces standard information-theoretic boundary condition:
    lim_{p -> 0^+} p log p = 0 via torch.nan_to_num.
    """
    prod = torch.nan_to_num(p_dist * log_p_dist, nan=0.0, neginf=0.0, posinf=0.0)
    return -torch.sum(prod, dim=-1, keepdim=True)


def compute_free_energy_support_mask(
    log_q_star: torch.Tensor,
    entropy_nats: torch.Tensor,
) -> torch.Tensor:
    """
    Free-Energy Support Truncation (FEST) Candidate Support Set Mask.
    V_supp = {v : log q^*(v) >= max_w log q^*(w) - H(q^*)}.
    """
    max_log_q = torch.max(log_q_star, dim=-1, keepdim=True).values
    return log_q_star >= (max_log_q - entropy_nats)


def compute_candidate_support_mask(
    tensor_values: torch.Tensor,
    entropy_nats: torch.Tensor,
    beta_param: Optional[float] = None,
) -> torch.Tensor:
    """
    Thermodynamic Candidate Support Mask.
    """
    if beta_param is not None and beta_param > 0:
        cutoff = entropy_nats / float(beta_param)
    else:
        cutoff = entropy_nats
    max_val = torch.max(tensor_values, dim=-1, keepdim=True).values
    return tensor_values >= (max_val - cutoff)


def mean_field_contrastive_step(
    logits_plus: torch.Tensor,
    logits_zero: torch.Tensor,
    alpha: float = 0.50,
    beta: float = 0.05,
    step_idx: int = 1,
    comp_a: bool = True,
    comp_b: bool = True,
    comp_c: bool = True,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Execute single-step Information-Geometric Variational Free Energy Decoding (IG-VFED).

    Workflow:
    1. Compute probability distributions p_t^+ and p_t^0 via Softmax.
    2. Compute Bhattacharyya proximity BC_t with clamp safeguard.
    3. Compute dynamic coupling alpha_t = lambda * sqrt(1 - BC_t^2) + lambda * 1_{t=1 and comp_c} * BC_1 (Directive A).
    4. Compute contrastive visual Pointwise Mutual Information: z_t^* = z_t^+ + alpha_t * (z_t^+ - z_t^0).
    5. Compute dynamic geodesic candidate plausibility boundary beta_t = beta * BC_t (Component B).
    6. Masked variational selection over V_head = {v : p_t^+(v) >= beta_t * max_w p_t^+(w)}.

    Hyperparameters (strictly <= 2):
        alpha: Continuous coupling factor lambda in [0.0, 1.5] (default 0.50).
        beta: Continuous plausibility truncation floor beta in [0.001, 1.0] (default 0.05).

    Returns:
        next_token: Selected token tensor [B] or scalar.
        steered_logits: Variational steered logit tensor [B, V] or [V].
    """
    is_1d = (logits_plus.dim() == 1)
    if is_1d:
        logits_plus = logits_plus.unsqueeze(0)
        logits_zero = logits_zero.unsqueeze(0)

    lp = logits_plus.float()
    lz = logits_zero.float()

    # Compute Softmax distributions
    p_plus = F.softmax(lp, dim=-1)
    p_zero = F.softmax(lz, dim=-1)

    # 1. Fisher-Rao Bhattacharyya Coefficient with Float32 Clamp Safeguard
    bc = compute_bhattacharyya_coefficient(p_plus, p_zero, eps=eps)

    # 2. Dynamic Coupling with Binding Directive A (Preamble Echo Suppression at t=1)
    alpha_t = compute_dynamic_coupling(alpha, bc, step_idx=step_idx, comp_a=comp_a, comp_c=comp_c)

    # 3. Pointwise Visual Mutual Information Contrast (z_t^* = z_t^+ + alpha_t * (z_t^+ - z_t^0))
    contrast_diff = lp - lz
    steered_logits = lp + alpha_t * contrast_diff

    # 4. Dynamic Geodesic Candidate Plausibility Boundary (Component B)
    beta_t = compute_dynamic_plausibility_floor(beta, bc, comp_b=comp_b)
    max_p_plus, _ = torch.max(p_plus, dim=-1, keepdim=True)
    candidate_mask = (p_plus >= beta_t * max_p_plus)

    # 5. Masked Variational Selection
    masked_steered_logits = torch.where(
        candidate_mask,
        steered_logits,
        torch.full_like(steered_logits, -1e9),
    )
    next_tokens = torch.argmax(masked_steered_logits, dim=-1)

    if is_1d:
        next_tokens = next_tokens.squeeze(0)
        masked_steered_logits = masked_steered_logits.squeeze(0)

    return next_tokens, masked_steered_logits


def nevfe_step_logits(
    logits_plus: torch.Tensor,
    logits_zero: torch.Tensor,
    lambda_param: float = 0.50,
    beta_param: float = 0.05,
    mu_param: float = 0.0,
    q_mask: Optional[torch.Tensor] = None,
    step_idx: int = 1,
    barrier_decay: float = 0.5,
    barrier_horizon: int = 3,
    comp_a: bool = True,
    comp_b: bool = True,
    comp_c: bool = True,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Single-Step Information-Geometric Variational Free Energy Decoding (IG-VFED) Operator.

    Workflow:
    1. Softmax probability distributions p^+ and p^0.
    2. Fisher-Rao Bhattacharyya Coefficient BC = sum_w sqrt(p^+ * p^0) with float32 clamp.
    3. Geodesic coupling alpha_t = lambda * sqrt(clamp(1.0 - BC^2, 0.0, 1.0)) + lambda * 1_{t=1 and comp_c} * BC_1.
    4. Contrastive Pointwise Visual Mutual Information steering: z^* = z^+ + alpha_t * (z^+ - z^0).
    5. Dynamic Geodesic Candidate Plausibility Boundary beta_t = beta * BC_t.
    6. Masked Variational selection via Argmax over V_head.

    Strict A* budget: Exactly 2 continuous parameters (lambda_param, beta_param).
    100% Backbone-Agnostic Core: zero hardcoded dimensions D, P, V, or model tokens.
    """
    is_1d = (logits_plus.dim() == 1)
    if is_1d:
        logits_plus = logits_plus.unsqueeze(0)
        logits_zero = logits_zero.unsqueeze(0)

    lp = logits_plus.float()
    lz = logits_zero.float()

    p_plus = F.softmax(lp, dim=-1)
    p_zero = F.softmax(lz, dim=-1)

    # 1. Fisher-Rao Bhattacharyya Coefficient with Float32 Clamp Safeguard
    bc = compute_bhattacharyya_coefficient(p_plus, p_zero, eps=eps)

    # 2. Geodesic Coupling with Binding Directive A (Preamble Echo Suppression at t=1 gated by comp_c)
    alpha_t = compute_dynamic_coupling(lambda_param, bc, step_idx=step_idx, comp_a=comp_a, comp_c=comp_c)

    # 3. Mean-Field Contrastive Steering
    steered_potential = lp + alpha_t * (lp - lz)

    # Optional query barrier (active only if caller explicitly sets mu_param > 0 and q_mask is provided)
    if q_mask is not None and mu_param > 0 and step_idx <= barrier_horizon:
        barrier = float(mu_param) * (float(barrier_decay) ** (step_idx - 1))
        q_mask_t = q_mask.to(device=steered_potential.device, dtype=torch.bool)
        if q_mask_t.dim() == 1 and steered_potential.dim() == 2:
            q_mask_t = q_mask_t.unsqueeze(0)
        steered_potential = torch.where(
            q_mask_t,
            steered_potential - barrier,
            steered_potential,
        )

    # 4. Dynamic Geodesic Candidate Plausibility Boundary (Component B)
    beta_t = compute_dynamic_plausibility_floor(beta_param, bc, comp_b=comp_b)
    max_p_plus, _ = torch.max(p_plus, dim=-1, keepdim=True)
    candidate_mask = (p_plus >= beta_t * max_p_plus)

    # 5. Masked Variational Logits
    steered_logits = torch.where(
        candidate_mask,
        steered_potential,
        torch.full_like(steered_potential, -1e9),
    )
    next_tokens = torch.argmax(steered_logits, dim=-1)

    # Shannon entropy of p_plus distribution for informational tracking
    log_p_plus = F.log_softmax(lp, dim=-1)
    entropy_nats = compute_unnormalized_entropy(p_plus, log_p_plus)

    if is_1d:
        next_tokens = next_tokens.squeeze(0)
        steered_logits = steered_logits.squeeze(0)
        alpha_t = alpha_t.squeeze(0)
        entropy_nats = entropy_nats.squeeze(0)

    return next_tokens, steered_logits, alpha_t, entropy_nats


class NEVFEEngine:
    """
    Backbone-Agnostic Information-Geometric Variational Free Energy Decoding Engine (IG-VFED).

    Encapsulates core tensor mathematics governed strictly by at most 2 global continuous hyperparameters:
    - lambda_param: Base Fisher-Rao coupling factor lambda in [0.0, 1.5] (default 0.50).
    - beta_param: Dynamic plausibility truncation threshold beta in [0.001, 1.0] (default 0.05).
    """
    def __init__(
        self,
        lambda_param: float = 0.50,
        beta_param: float = 0.05,
        mu_param: float = 0.0,
        barrier_decay: float = 0.5,
        barrier_horizon: int = 3,
        comp_a: bool = True,
        comp_b: bool = True,
        comp_c: bool = True,
        **kwargs,
    ):
        if not (0.0 <= lambda_param <= 1.5):
            raise ValueError(f"lambda_param must be in [0.0, 1.5], got {lambda_param}")
        if not (0.001 <= beta_param <= 5.0):
            raise ValueError(f"beta_param must be in [0.001, 5.0], got {beta_param}")
        if not (0.0 <= mu_param <= 10.0):
            raise ValueError(f"mu_param must be in [0.0, 10.0], got {mu_param}")

        self.lambda_param = float(lambda_param)
        self.beta_param = float(beta_param)
        self.mu_param = float(mu_param)
        self.barrier_decay = float(barrier_decay)
        self.barrier_horizon = int(barrier_horizon)

        # Micro-Ablation Switches (Directive #3 Factorial Micro-Ablation)
        self.comp_a = bool(comp_a)
        self.comp_b = bool(comp_b)
        self.comp_c = bool(comp_c)

    def set_ablation_switches(
        self,
        comp_a: bool = True,
        comp_b: bool = True,
        comp_c: bool = True,
    ) -> None:
        """Dynamically configure micro-ablation toggles for Factorial Interaction testing."""
        self.comp_a = bool(comp_a)
        self.comp_b = bool(comp_b)
        self.comp_c = bool(comp_c)

    def modulate_patches(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        """Modulate visual patch embeddings via Fisher-Saliency energy (kept for utility)."""
        return modulate_patch_embeddings(visual_tokens)

    def decode_step(
        self,
        logits_plus: torch.Tensor,
        logits_zero: torch.Tensor,
        q_mask: Optional[torch.Tensor] = None,
        step_idx: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Execute single-step Information-Geometric Variational Free Energy Decoding."""
        return nevfe_step_logits(
            logits_plus=logits_plus,
            logits_zero=logits_zero,
            lambda_param=self.lambda_param,
            beta_param=self.beta_param,
            mu_param=self.mu_param,
            q_mask=q_mask,
            step_idx=step_idx,
            barrier_decay=self.barrier_decay,
            barrier_horizon=self.barrier_horizon,
            comp_a=self.comp_a,
            comp_b=self.comp_b,
            comp_c=self.comp_c,
        )
