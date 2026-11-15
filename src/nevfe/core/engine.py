# -*- coding: utf-8 -*-
# SPEC_REF: docs/specs/team_1/spec_iter_10.md@v1.0 (sha256=b0ec66d5ea611f724e06cc525c83477b08deb299069695f46dd80f79fed6a66f)
"""
Spherical KL-Specificity Gated & Thumbnail-Terminal Active Foveation (SKG-TTAF) + IG-VFED Core Engine.

Backbone-agnostic mathematical core (<= 250 AST statement lines, <= 3 tunable parameters).
"""

from dataclasses import dataclass
import math
from typing import Optional, Tuple
import torch
import torch.nn.functional as F

SPEC_REF = "docs/specs/team_1/spec_iter_10.md@v1.0 (sha256=b0ec66d5ea611f724e06cc525c83477b08deb299069695f46dd80f79fed6a66f)"


@dataclass(frozen=True)
class AlgorithmConfig:
    """Strictly <= 3 tunable hyperparameters plus factorial ablation_mask."""

    lambda_param: float = 0.50
    beta_param: float = 0.05
    gamma_param: float = 0.50
    ablation_mask: Tuple[int, ...] = (1, 1, 1)

    def __post_init__(self) -> None:
        if not (0.0 <= float(self.lambda_param) <= 1.5):
            raise ValueError(f"lambda_param must be in [0.0, 1.5], got {self.lambda_param}")
        if not (0.001 <= float(self.beta_param) <= 5.0):
            raise ValueError(f"beta_param must be in [0.001, 5.0], got {self.beta_param}")
        if not (0.0 <= float(self.gamma_param) <= 1.0):
            raise ValueError(f"gamma_param must be in [0.0, 1.0], got {self.gamma_param}")
        mask = tuple(int(bool(x)) for x in self.ablation_mask)
        if len(mask) != 3:
            raise ValueError(f"ablation_mask must have length 3, got {self.ablation_mask}")
        object.__setattr__(self, "lambda_param", float(self.lambda_param))
        object.__setattr__(self, "beta_param", float(self.beta_param))
        object.__setattr__(self, "gamma_param", float(self.gamma_param))
        object.__setattr__(self, "ablation_mask", mask)


def compute_semantic_specificity_posterior(
    patch_embeds: torch.Tensor,
    query_embeds: Optional[torch.Tensor] = None,
    comp_a: bool = True,
    tau_v: float = 1e-6,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Eq. (1): Spherical von Mises-Fisher KL-specificity Soft-OR spatial posterior."""
    if patch_embeds.dim() != 2 + 1:
        raise ValueError(f"Expected 3D patch_embeds, got {patch_embeds.shape}")
    h_g, w_g, dim = patch_embeds.shape
    if query_embeds is not None and query_embeds.dim() == 2 and query_embeds.shape[1] != dim:
        raise ValueError(f"Expected query_embeds with dim {dim}, got {query_embeds.shape}")
    n_lat = h_g * w_g
    unif = torch.full((h_g, w_g), 1.0 / float(max(1, n_lat)), dtype=torch.float32, device=patch_embeds.device)
    if query_embeds is None or query_embeds.dim() != 2 or query_embeds.shape[0] == 0 or dim == 0 or n_lat == 0:
        return unif
    v_raw = torch.nan_to_num(patch_embeds.float(), nan=0.0, posinf=0.0, neginf=0.0).reshape(n_lat, dim)
    q_raw = torch.nan_to_num(query_embeds.float(), nan=0.0, posinf=0.0, neginf=0.0)
    v_safe = v_raw / torch.clamp(v_raw.abs().max(), min=1.0)
    q_safe = q_raw / torch.clamp(q_raw.abs().amax(dim=-1, keepdim=True), min=1.0)
    dv = v_safe - v_safe.mean(dim=0, keepdim=True)
    r_v = torch.linalg.vector_norm(dv, dim=-1, keepdim=True)
    r_e = torch.linalg.vector_norm(q_safe, dim=-1, keepdim=True)
    tau_d = tau_v * math.sqrt(float(max(1, dim)))
    v_hat = torch.where(r_v > tau_d, dv / torch.clamp(r_v, min=tau_d), torch.zeros_like(dv))
    e_hat = torch.where(r_e > tau_v, q_safe / torch.clamp(r_e, min=tau_v), torch.zeros_like(q_safe))
    s_mat = torch.matmul(v_hat, e_hat.transpose(0, 1))
    if not comp_a:
        c_vec = s_mat.max(dim=-1).values
    else:
        s_til = s_mat - s_mat.mean(dim=0, keepdim=True)
        sig_s = torch.sqrt(torch.clamp((s_til * s_til).mean(), min=0.0))
        if float(sig_s.item()) <= tau_v:
            return unif
        z_mat = s_til / torch.clamp(sig_s, min=tau_v)
        q_prob = F.softmax(z_mat, dim=0)
        kl = torch.nan_to_num(q_prob * (torch.log(torch.clamp(q_prob, min=eps)) + math.log(float(max(1, n_lat)))), nan=0.0, posinf=0.0, neginf=0.0)
        i_m = torch.clamp(kl.sum(dim=0), min=0.0)
        w_sq = i_m * i_m
        w_sum = w_sq.sum()
        w_m = (w_sq / torch.clamp(w_sum, min=eps)) if float(w_sum.item()) > eps else torch.full_like(w_sq, 1.0 / float(query_embeds.shape[0]))
        z_star = z_mat.max(dim=-1, keepdim=True).values
        c_vec = (z_star + torch.log(torch.clamp((w_m.unsqueeze(0) * torch.exp(z_mat - z_star)).sum(dim=-1, keepdim=True), min=eps))).squeeze(-1)
    c_til = c_vec - c_vec.mean()
    sig_c = torch.sqrt(torch.clamp((c_til * c_til).mean(), min=0.0))
    if float(sig_c.item()) <= tau_v:
        return unif
    return F.softmax(c_til / torch.clamp(sig_c, min=tau_v), dim=-1).view(h_g, w_g)


def select_active_foveal_subwindow(
    posterior_2d: torch.Tensor,
    gamma_param: Optional[float] = None,
    comp_b: bool = True,
    eps_k: float = 1e-6,
    eps: float = 1e-12,
) -> Tuple[Tuple[int, int, int, int], float, float, float]:
    """Eq. (2): Exact multiscale Poisson-Bregman scan-statistic active subwindow gate."""
    if posterior_2d.dim() != 2:
        raise ValueError(f"Expected 2D posterior_2d, got {posterior_2d.shape}")
    h_g, w_g = posterior_2d.shape
    n_lat = h_g * w_g
    p0 = torch.nan_to_num(posterior_2d.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
    if n_lat == 0:
        return (0, 0, h_g, w_g), 0.0, 0.0, 0.0
    p1 = p0 / torch.clamp(p0.max(), min=1.0)
    p_sum = p1.sum()
    p = (p1 / p_sum) if float(p_sum.item()) > eps else torch.full_like(p1, 1.0 / float(n_lat))
    if n_lat <= 1:
        return (0, 0, h_g, w_g), 0.0, 1.0, 1.0
    g_val = AlgorithmConfig().gamma_param if gamma_param is None else float(gamma_param)
    if not math.isfinite(g_val):
        g_val = AlgorithmConfig().gamma_param
    gamma_eff = min(1.0, max(0.0, g_val))
    tau_scan = max(eps_k, 2.0 * gamma_eff * math.sqrt(math.log(float(n_lat)) / float(n_lat))) if comp_b else eps_k
    sat = F.pad(p, (1, 0, 1, 0)).cumsum(dim=0).cumsum(dim=1)
    ri = torch.triu_indices(h_g + 1, h_g + 1, offset=1, device=p.device)
    ci = torch.triu_indices(w_g + 1, w_g + 1, offset=1, device=p.device)
    i1, i2, j1, j2 = ri[0], ri[1], ci[0], ci[1]
    mass = torch.clamp(
        sat[i2.unsqueeze(1), j2.unsqueeze(0)]
        - sat[i1.unsqueeze(1), j2.unsqueeze(0)]
        - sat[i2.unsqueeze(1), j1.unsqueeze(0)]
        + sat[i1.unsqueeze(1), j1.unsqueeze(0)],
        min=0.0,
        max=1.0,
    )
    area = ((i2 - i1).float().unsqueeze(1) * (j2 - j1).float().unsqueeze(0)) / float(n_lat)
    adm = (mass > area + eps_k) & (area >= eps_k) & (area <= 0.5)
    p_cl = torch.clamp(mass, min=eps_k, max=1.0)
    a_cl = torch.clamp(area, min=eps_k, max=1.0)
    j_raw = torch.nan_to_num(p_cl * torch.log(p_cl / a_cl) - (p_cl - a_cl), nan=0.0, posinf=0.0, neginf=0.0)
    j_pois = torch.where(adm, torch.clamp(j_raw, min=0.0), torch.zeros_like(j_raw))
    flat_idx = int(torch.argmax(j_pois.view(-1)).item())
    n_cols = int(j1.numel())
    r_idx, c_idx = flat_idx // n_cols, flat_idx % n_cols
    j_star = float(j_pois[r_idx, c_idx].item())
    if j_star <= tau_scan:
        return (0, 0, h_g, w_g), 0.0, 1.0, 1.0
    box = (int(i1[r_idx].item()), int(j1[c_idx].item()), int(i2[r_idx].item()), int(j2[c_idx].item()))
    return box, j_star, float(mass[r_idx, c_idx].item()), float(area[r_idx, c_idx].item())


def assemble_foveated_visual_sequence(
    vit_embeds: torch.Tensor,
    extra_vit_embeds: Optional[torch.Tensor] = None,
    comp_c: bool = True,
) -> torch.Tensor:
    """Eq. (3): Thumbnail-terminal InternVL pretrained visual tile splice without mutating vit_embeds."""
    if vit_embeds.dim() != 2 + 1:
        raise ValueError(f"Expected 3D vit_embeds, got {vit_embeds.shape}")
    if extra_vit_embeds is not None and (
        extra_vit_embeds.dim() != 2 + 1 or extra_vit_embeds.shape[1:] != vit_embeds.shape[1:]
    ):
        raise ValueError(f"Expected 3D extra_vit_embeds matching {vit_embeds.shape[1:]}, got {extra_vit_embeds.shape}")
    if extra_vit_embeds is None or int(extra_vit_embeds.shape[0]) == 0 or int(vit_embeds.shape[0]) == 0:
        return vit_embeds
    crops = extra_vit_embeds.to(device=vit_embeds.device, dtype=vit_embeds.dtype)
    if not comp_c:
        return torch.cat([vit_embeds, crops], dim=0)
    if int(vit_embeds.shape[0]) == 1:
        return torch.cat([crops, vit_embeds], dim=0)
    return torch.cat([vit_embeds[:-1], crops, vit_embeds[-1:]], dim=0)


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
    """Backbone-agnostic SKG-TTAF + IG-VFED engine configured by AlgorithmConfig."""

    def __init__(
        self,
        lambda_param: Optional[float] = None,
        beta_param: Optional[float] = None,
        gamma_param: Optional[float] = None,
        comp_a: Optional[bool] = None,
        comp_b: Optional[bool] = None,
        comp_c: Optional[bool] = None,
        config: Optional[AlgorithmConfig] = None,
    ) -> None:
        base = config if config is not None else AlgorithmConfig()
        ca_cfg, cb_cfg, cc_cfg = base.ablation_mask
        resolved_mask = (
            int(bool(comp_a)) if comp_a is not None else int(ca_cfg),
            int(bool(comp_b)) if comp_b is not None else int(cb_cfg),
            int(bool(comp_c)) if comp_c is not None else int(cc_cfg),
        )
        self.config = AlgorithmConfig(
            lambda_param=base.lambda_param if lambda_param is None else float(lambda_param),
            beta_param=base.beta_param if beta_param is None else float(beta_param),
            gamma_param=base.gamma_param if gamma_param is None else float(gamma_param),
            ablation_mask=resolved_mask,
        )
        ca_val, cb_val, cc_val = self.config.ablation_mask
        self.lambda_param = self.config.lambda_param
        self.beta_param = self.config.beta_param
        self.gamma_param = self.config.gamma_param
        self.comp_a = bool(ca_val)
        self.comp_b = bool(cb_val)
        self.comp_c = bool(cc_val)

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
            gamma_param=self.gamma_param,
            ablation_mask=(int(self.comp_a), int(self.comp_b), int(self.comp_c)),
        )

    def foveate(
        self,
        patch_embeds: torch.Tensor,
        query_embeds: Optional[torch.Tensor] = None,
    ) -> Tuple[Tuple[int, int, int, int], float, torch.Tensor]:
        """Execute unified SKG-TTAF (Eq. 1)-(Eq. 2) pure semantic active subwindow selection."""
        posterior_2d = compute_semantic_specificity_posterior(
            patch_embeds, query_embeds=query_embeds, comp_a=self.comp_a
        )
        box, j_star, _, _ = select_active_foveal_subwindow(
            posterior_2d, gamma_param=self.gamma_param, comp_b=self.comp_b
        )
        return box, j_star, posterior_2d

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
