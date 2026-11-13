# -*- coding: utf-8 -*-
# SPEC_REF: docs/specs/baseline_iter23.md@v1.0 (sha256=7e9032f488d9c4ff1f754276c1cff1f5fbc207b602410627a3939d7cf3a97edd)
"""
Information-Geometric Variational Free Energy Decoding (IG-VFED) Core Engine.

Backbone-agnostic mathematical core (<= 250 AST statement lines, <= 3 tunable parameters).
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn.functional as F

SPEC_REF = "docs/specs/baseline_iter23.md@v1.0 (sha256=7e9032f488d9c4ff1f754276c1cff1f5fbc207b602410627a3939d7cf3a97edd)"


@dataclass(frozen=True)
class AlgorithmConfig:
    """Strictly <= 3 tunable hyperparameters plus factorial ablation_mask."""

    lambda_param: float = 0.50
    beta_param: float = 0.05
    ablation_mask: Tuple[int, ...] = (1, 1, 1)

    def __post_init__(self) -> None:
        if not (0.0 <= float(self.lambda_param) <= 1.5):
            raise ValueError(f"lambda_param must be in [0.0, 1.5], got {self.lambda_param}")
        if not (0.001 <= float(self.beta_param) <= 5.0):
            raise ValueError(f"beta_param must be in [0.001, 5.0], got {self.beta_param}")
        mask = tuple(int(bool(x)) for x in self.ablation_mask)
        if len(mask) != 3:
            raise ValueError(f"ablation_mask must have length 3, got {self.ablation_mask}")
        object.__setattr__(self, "lambda_param", float(self.lambda_param))
        object.__setattr__(self, "beta_param", float(self.beta_param))
        object.__setattr__(self, "ablation_mask", mask)


def compute_mean_field_reference_embeddings(vit_embeds: torch.Tensor) -> torch.Tensor:
    """Construct blind Mean-Field Visual Reference tensor V^0 = 1_P v_bar^T."""
    if vit_embeds.dim() == 3:
        return vit_embeds.mean(dim=(0, 1), keepdim=True).expand_as(vit_embeds)
    if vit_embeds.dim() == 2:
        return vit_embeds.mean(dim=0, keepdim=True).expand_as(vit_embeds)
    raise ValueError(f"Unexpected vit_embeds shape: {vit_embeds.shape}")


def compute_bhattacharyya_coefficient(
    p_plus: torch.Tensor,
    p_zero: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute Bhattacharyya Coefficient BC = sum_w sqrt(p_plus * p_zero) clamped to [0, 1]."""
    del eps
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
    """Fisher-Rao geodesic coupling with t=1 preamble echo suppression."""
    bc_clamped = torch.clamp(bc, min=0.0, max=1.0)
    if not comp_a:
        return torch.full_like(bc_clamped, float(lambda_param))
    radicand = torch.clamp(1.0 - bc_clamped * bc_clamped, min=0.0, max=1.0)
    alpha_t = float(lambda_param) * torch.sqrt(radicand)
    if step_idx == 1 and comp_c:
        alpha_t = alpha_t + float(lambda_param) * bc_clamped
    return alpha_t


def compute_dynamic_plausibility_floor(
    beta_param: float,
    bc: torch.Tensor,
    comp_b: bool = True,
) -> torch.Tensor:
    """Dynamic geodesic candidate plausibility boundary beta_t = beta * BC_t."""
    if not comp_b:
        return torch.full_like(bc, float(beta_param))
    bc_clamped = torch.clamp(bc, min=0.0, max=1.0)
    return torch.clamp(float(beta_param) * bc_clamped, min=0.0, max=1.0)


def compute_unnormalized_entropy(
    p_dist: torch.Tensor,
    log_p_dist: torch.Tensor,
) -> torch.Tensor:
    """Thermodynamic Shannon entropy in nats with 0 * log(0) = 0 boundary."""
    prod = torch.nan_to_num(p_dist * log_p_dist, nan=0.0, neginf=0.0, posinf=0.0)
    return -torch.sum(prod, dim=-1, keepdim=True)


def nevfe_step_logits(
    logits_plus: torch.Tensor,
    logits_zero: torch.Tensor,
    lambda_param: Optional[float] = None,
    beta_param: Optional[float] = None,
    q_mask: Optional[torch.Tensor] = None,
    step_idx: int = 1,
    comp_a: bool = True,
    comp_b: bool = True,
    comp_c: bool = True,
    eps: float = 1e-12,
    config: Optional[AlgorithmConfig] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-step Information-Geometric Variational Free Energy Decoding (IG-VFED)."""
    del q_mask
    cfg = config if config is not None else AlgorithmConfig()
    lam = cfg.lambda_param if lambda_param is None else float(lambda_param)
    bet = cfg.beta_param if beta_param is None else float(beta_param)

    is_1d = logits_plus.dim() == 1
    if is_1d:
        logits_plus = logits_plus.unsqueeze(0)
        logits_zero = logits_zero.unsqueeze(0)

    lp = logits_plus.float()
    lz = logits_zero.float()
    p_plus = F.softmax(lp, dim=-1)
    p_zero = F.softmax(lz, dim=-1)

    bc = compute_bhattacharyya_coefficient(p_plus, p_zero, eps=eps)
    alpha_t = compute_dynamic_coupling(lam, bc, step_idx=step_idx, comp_a=comp_a, comp_c=comp_c)
    steered_potential = lp + alpha_t * (lp - lz)

    beta_t = compute_dynamic_plausibility_floor(bet, bc, comp_b=comp_b)
    max_p_plus, _ = torch.max(p_plus, dim=-1, keepdim=True)
    candidate_mask = p_plus >= beta_t * max_p_plus
    steered_logits = torch.where(
        candidate_mask,
        steered_potential,
        torch.full_like(steered_potential, -1e9),
    )
    next_tokens = torch.argmax(steered_logits, dim=-1)
    entropy_nats = compute_unnormalized_entropy(p_plus, F.log_softmax(lp, dim=-1))

    if is_1d:
        next_tokens = next_tokens.squeeze(0)
        steered_logits = steered_logits.squeeze(0)
        alpha_t = alpha_t.squeeze(0)
        entropy_nats = entropy_nats.squeeze(0)

    return next_tokens, steered_logits, alpha_t, entropy_nats


class NEVFEEngine:
    """Backbone-agnostic IG-VFED engine configured by AlgorithmConfig."""

    def __init__(
        self,
        lambda_param: Optional[float] = None,
        beta_param: Optional[float] = None,
        comp_a: Optional[bool] = None,
        comp_b: Optional[bool] = None,
        comp_c: Optional[bool] = None,
        config: Optional[AlgorithmConfig] = None,
    ) -> None:
        base = config if config is not None else AlgorithmConfig()
        mask = base.ablation_mask
        resolved_mask = (
            int(bool(comp_a)) if comp_a is not None else int(mask[0]),
            int(bool(comp_b)) if comp_b is not None else int(mask[1]),
            int(bool(comp_c)) if comp_c is not None else int(mask[2]),
        )
        self.config = AlgorithmConfig(
            lambda_param=base.lambda_param if lambda_param is None else float(lambda_param),
            beta_param=base.beta_param if beta_param is None else float(beta_param),
            ablation_mask=resolved_mask,
        )
        self.lambda_param = self.config.lambda_param
        self.beta_param = self.config.beta_param
        self.comp_a = bool(self.config.ablation_mask[0])
        self.comp_b = bool(self.config.ablation_mask[1])
        self.comp_c = bool(self.config.ablation_mask[2])

    def set_ablation_switches(
        self,
        comp_a: bool = True,
        comp_b: bool = True,
        comp_c: bool = True,
    ) -> None:
        """Dynamically configure micro-ablation toggles for factorial probing."""
        self.comp_a = bool(comp_a)
        self.comp_b = bool(comp_b)
        self.comp_c = bool(comp_c)
        self.config = AlgorithmConfig(
            lambda_param=self.lambda_param,
            beta_param=self.beta_param,
            ablation_mask=(int(self.comp_a), int(self.comp_b), int(self.comp_c)),
        )

    def decode_step(
        self,
        logits_plus: torch.Tensor,
        logits_zero: torch.Tensor,
        q_mask: Optional[torch.Tensor] = None,
        step_idx: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Execute single-step IG-VFED decoding."""
        return nevfe_step_logits(
            logits_plus=logits_plus,
            logits_zero=logits_zero,
            lambda_param=self.lambda_param,
            beta_param=self.beta_param,
            q_mask=q_mask,
            step_idx=step_idx,
            comp_a=self.comp_a,
            comp_b=self.comp_b,
            comp_c=self.comp_c,
        )
