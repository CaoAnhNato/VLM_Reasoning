# -*- coding: utf-8 -*-
# SPEC_REF: docs/specs/baseline_frsa011.md@v1.0 (sha256=8da794eb5204770e371860806d1f4467d0718f8209920c614a756c5bcf781c35)
"""
Frozen FRSA-PBCF ("011") + Base IG-VFED (1, 1, 1) Core Decoding Engine.

Backbone-agnostic mathematical core (<= 250 AST statement lines, locked base constants
lambda_param=0.50 and beta_param=0.05 plus <= 2 cumulative new tunable scalars).
"""

from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional, Tuple
import torch
import torch.nn.functional as F

from src.nevfe.core.frozen_foveation import (
    compute_mean_field_reference_embeddings,
    foveate_frozen_011,
)

SPEC_REF = "docs/specs/baseline_frsa011.md@v1.0 (sha256=8da794eb5204770e371860806d1f4467d0718f8209920c614a756c5bcf781c35)"
COMPONENT_NAMES: Tuple[str, ...] = ()


@dataclass(frozen=True)
class AlgorithmConfig:
    """Locked base parameters (lambda_param=0.50, beta_param=0.05) + variable ablation_mask."""

    lambda_param: float = 0.50
    beta_param: float = 0.05
    ablation_mask: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        lam = float(self.lambda_param)
        bet = float(self.beta_param)
        if abs(lam - 0.50) > 1e-12:
            raise ValueError(f"lambda_param is locked to 0.50, got {self.lambda_param}")
        if abs(bet - 0.05) > 1e-12:
            raise ValueError(f"beta_param is locked to 0.05, got {self.beta_param}")
        raw_mask = tuple(self.ablation_mask)
        if any(not isinstance(x, (bool, int)) or int(x) not in (0, 1) for x in raw_mask):
            raise ValueError(f"ablation_mask entries must be binary (0 or 1), got {self.ablation_mask}")
        mask = tuple(int(x) for x in raw_mask)
        if len(mask) != len(COMPONENT_NAMES):
            raise ValueError(
                f"ablation_mask length {len(mask)} != len(COMPONENT_NAMES) ({len(COMPONENT_NAMES)}): {self.ablation_mask}"
            )
        object.__setattr__(self, "lambda_param", lam)
        object.__setattr__(self, "beta_param", bet)
        object.__setattr__(self, "ablation_mask", mask)


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
    """Backbone-agnostic frozen FRSA-PBCF + base IG-VFED (1, 1, 1) engine."""

    def __init__(
        self,
        lambda_param: Optional[float] = None,
        beta_param: Optional[float] = None,
        ablation_mask: Optional[Tuple[int, ...]] = None,
        config: Optional[AlgorithmConfig] = None,
    ) -> None:
        base = config if config is not None else AlgorithmConfig()
        updates = {}
        if lambda_param is not None:
            updates["lambda_param"] = float(lambda_param)
        if beta_param is not None:
            updates["beta_param"] = float(beta_param)
        if ablation_mask is not None:
            updates["ablation_mask"] = tuple(int(bool(x)) for x in ablation_mask)
        self.config = replace(base, **updates) if updates else base
        self.lambda_param = self.config.lambda_param
        self.beta_param = self.config.beta_param
        self.ablation_mask = self.config.ablation_mask

    def set_ablation_switches(
        self,
        *bits: Any,
        ablation_mask: Optional[Tuple[int, ...]] = None,
    ) -> None:
        """Dynamically configure variable-length ablation toggles via dataclasses.replace."""
        raw = ablation_mask if ablation_mask is not None else bits
        new_mask = tuple(int(bool(x)) for x in raw)
        self.config = replace(self.config, ablation_mask=new_mask)
        self.ablation_mask = self.config.ablation_mask

    def foveate(
        self,
        patch_embeds: torch.Tensor,
        query_embeds: Optional[torch.Tensor] = None,
    ) -> Tuple[Tuple[int, int, int, int], float, torch.Tensor]:
        """Delegate foveation to immutable FRSA-PBCF ('011') pure p_lex + Poisson-Bregman module."""
        return foveate_frozen_011(patch_embeds, query_embeds=query_embeds)

    def decode_step(
        self,
        logits_plus: torch.Tensor,
        logits_zero: torch.Tensor,
        q_mask: Optional[torch.Tensor] = None,
        step_idx: int = 1,
        step_context: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Execute single-step base IG-VFED (1, 1, 1) decoding.

        Optional `step_context` keys for downstream track operators:
            - `history_logits`: Optional[Tensor] prior-step dual-stream logits history
            - `upper_hidden_states`: Optional[Dict[str, Tensor]] read-only upper-layer states
            - `rollout_fn`: Optional[Callable] adapter-supplied multi-step rollout callback
        """
        del step_context
        return nevfe_step_logits(
            logits_plus=logits_plus,
            logits_zero=logits_zero,
            lambda_param=self.config.lambda_param,
            beta_param=self.config.beta_param,
            q_mask=q_mask,
            step_idx=step_idx,
            comp_a=True,
            comp_b=True,
            comp_c=True,
            config=self.config,
        )
