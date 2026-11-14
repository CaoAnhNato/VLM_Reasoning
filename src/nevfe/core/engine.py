# -*- coding: utf-8 -*-
# SPEC_REF: docs/specs/team_1/spec_iter_3.md@v1.0 (sha256=6abf2c0322e675e9afa76c1e180a78a757af387c95e798a9e45a02a0dc67456b)
"""
Fisher-Rao Spherical Amplitude & Poisson-Bregman Chromatic Foveation (FRSA-PBCF) + IG-VFED Core Engine.

Backbone-agnostic mathematical core (<= 250 AST statement lines, <= 3 tunable parameters).
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn.functional as F

SPEC_REF = "docs/specs/team_1/spec_iter_3.md@v1.0 (sha256=6abf2c0322e675e9afa76c1e180a78a757af387c95e798a9e45a02a0dc67456b)"


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


def compute_bregman_optical_distribution(
    optical_energy: torch.Tensor,
    tau_v: float = 1e-6,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Eq. (1): 0-homogeneous rectified Bregman root-density optical distribution over Omega."""
    if optical_energy.dim() != 2:
        raise ValueError(f"Expected 2D optical_energy, got {optical_energy.shape}")
    e = torch.nan_to_num(optical_energy.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
    if e.numel() == 0:
        return e
    unif = torch.full_like(e, 1.0 / float(max(1, e.numel())))
    m_e = e.max()
    if float(m_e.item()) <= eps:
        return unif
    e_tilde = e / torch.clamp(m_e, min=eps)
    if float((1.0 - e_tilde.min()).item()) <= tau_v:
        return unif
    psi_raw = torch.clamp(e_tilde - e_tilde.mean(), min=0.0)
    sq_excess = psi_raw * psi_raw
    z_opt = sq_excess.sum()
    return torch.where(z_opt > eps, sq_excess / torch.clamp(z_opt, min=eps), unif)


def compute_lexical_spatial_distribution(
    patch_embeds: torch.Tensor,
    query_embeds: Optional[torch.Tensor] = None,
    tau_v: float = 1e-6,
) -> torch.Tensor:
    """Eq. (2): Bregman-centered Layer-0 lexical spatial distribution over Omega."""
    if patch_embeds.dim() != 3:
        raise ValueError(f"Expected 3D patch_embeds, got {patch_embeds.shape}")
    h_g, w_g, dim = patch_embeds.shape
    unif = torch.full((h_g, w_g), 1.0 / float(max(1, h_g * w_g)), dtype=torch.float32, device=patch_embeds.device)
    if query_embeds is None or query_embeds.dim() != 2:
        return unif
    if query_embeds.shape[1] != dim:
        raise ValueError(f"Expected query_embeds with dim {dim}, got {query_embeds.shape}")
    if query_embeds.shape[0] == 0 or dim == 0 or h_g * w_g == 0:
        return unif
    v_raw = torch.nan_to_num(patch_embeds.float(), nan=0.0, posinf=0.0, neginf=0.0)
    q_raw = torch.nan_to_num(query_embeds.float(), nan=0.0, posinf=0.0, neginf=0.0)
    v = v_raw / torch.clamp(v_raw.abs().max(), min=1.0)
    q = q_raw / torch.clamp(q_raw.abs().amax(dim=-1, keepdim=True), min=1.0)
    dv = v - v.mean(dim=(0, 1), keepdim=True)
    r_v = torch.linalg.vector_norm(dv, dim=-1, keepdim=True)
    r_e = torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    tau_d = tau_v * (float(max(1, dim)) ** 0.5)
    v_hat = torch.where(r_v > tau_d, dv / torch.clamp(r_v, min=tau_d), torch.zeros_like(dv))
    e_hat = torch.where(r_e > tau_v, q / torch.clamp(r_e, min=tau_v), torch.zeros_like(q))
    c_q = torch.matmul(v_hat, e_hat.transpose(0, 1)).max(dim=-1).values
    diff_c = c_q - c_q.mean()
    sigma_q = torch.sqrt(torch.clamp((diff_c * diff_c).mean(), min=0.0))
    z_q = torch.where(sigma_q > tau_v, diff_c / torch.clamp(sigma_q, min=tau_v), torch.zeros_like(diff_c))
    return F.softmax(z_q.view(-1), dim=-1).view(h_g, w_g)


def compute_variational_posterior(
    p_opt: torch.Tensor,
    p_lex: torch.Tensor,
    gamma_param: Optional[float] = None,
    comp_a: bool = True,
    comp_b: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Eq. (3): Riemannian Fisher-Rao spherical amplitude geodesic posterior."""
    if p_opt.dim() != 2 or p_opt.shape != p_lex.shape:
        raise ValueError(f"Expected matching 2D distributions, got {p_opt.shape} and {p_lex.shape}")
    h_g, w_g = p_opt.shape
    unif = torch.full((h_g, w_g), 1.0 / float(max(1, h_g * w_g)), dtype=torch.float32, device=p_opt.device)
    if h_g * w_g == 0 or (not comp_a and not comp_b):
        return unif
    g_val = AlgorithmConfig().gamma_param if gamma_param is None else float(gamma_param)
    if comp_a and comp_b:
        gamma_eff = min(1.0, max(0.0, g_val))
    elif comp_a:
        gamma_eff = 1.0
    else:
        gamma_eff = 0.0
    po = torch.nan_to_num(p_opt.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
    pl = torch.nan_to_num(p_lex.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
    po = po / torch.clamp(po.max(), min=1.0)
    pl = pl / torch.clamp(pl.max(), min=1.0)
    po_sum, pl_sum = po.sum(), pl.sum()
    po = po / po_sum if float(po_sum.item()) > eps else unif
    pl = pl / pl_sum if float(pl_sum.item()) > eps else unif
    psi = gamma_eff * torch.sqrt(po) + (1.0 - gamma_eff) * torch.sqrt(pl)
    sq_psi = psi * psi
    z_gamma = sq_psi.sum()
    return torch.where(z_gamma > eps, sq_psi / torch.clamp(z_gamma, min=eps), unif)


def select_poisson_bregman_subwindow(
    posterior_2d: torch.Tensor,
    comp_c: bool = True,
    eps_k: float = 1e-6,
    eps: float = 1e-12,
) -> Tuple[Tuple[int, int, int, int], float, float, float]:
    """Eq. (4): Exact Generalized KL / Poisson-Bregman (beta=1) subwindow selection."""
    if posterior_2d.dim() != 2:
        raise ValueError(f"Expected 2D posterior_2d, got {posterior_2d.shape}")
    h_g, w_g = posterior_2d.shape
    p = torch.nan_to_num(posterior_2d.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
    if h_g * w_g == 0:
        return (0, 0, h_g, w_g), 0.0, 0.0, 0.0
    p = p / torch.clamp(p.max(), min=1.0)
    p_sum = p.sum()
    p = p / p_sum if float(p_sum.item()) > eps else torch.full_like(p, 1.0 / float(h_g * w_g))
    if not comp_c or h_g * w_g <= 1:
        return (0, 0, h_g, w_g), 0.0, 1.0, 1.0
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
    area = ((i2 - i1).float().unsqueeze(1) * (j2 - j1).float().unsqueeze(0)) / float(h_g * w_g)
    adm = (mass > area + eps_k) & (area >= eps_k) & (area <= 0.5)
    p_cl = torch.clamp(mass, min=eps_k, max=1.0)
    a_cl = torch.clamp(area, min=eps_k, max=1.0)
    j_raw = torch.nan_to_num(
        p_cl * torch.log(p_cl / a_cl) - (p_cl - a_cl),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    j_pois = torch.where(adm, torch.clamp(j_raw, min=0.0), torch.zeros_like(j_raw))
    flat_idx = int(torch.argmax(j_pois.view(-1)).item())
    n_cols = int(j1.numel())
    r_idx, c_idx = flat_idx // n_cols, flat_idx % n_cols
    j_star = float(j_pois[r_idx, c_idx].item())
    if j_star <= eps_k:
        return (0, 0, h_g, w_g), 0.0, 1.0, 1.0
    box = (int(i1[r_idx].item()), int(j1[c_idx].item()), int(i2[r_idx].item()), int(j2[c_idx].item()))
    return box, j_star, float(mass[r_idx, c_idx].item()), float(area[r_idx, c_idx].item())


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
    """Backbone-agnostic VRE-OF + IG-VFED engine configured by AlgorithmConfig."""

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
        mask = base.ablation_mask
        resolved_mask = (
            int(bool(comp_a)) if comp_a is not None else int(mask[0]),
            int(bool(comp_b)) if comp_b is not None else int(mask[1]),
            int(bool(comp_c)) if comp_c is not None else int(mask[2]),
        )
        self.config = AlgorithmConfig(
            lambda_param=base.lambda_param if lambda_param is None else float(lambda_param),
            beta_param=base.beta_param if beta_param is None else float(beta_param),
            gamma_param=base.gamma_param if gamma_param is None else float(gamma_param),
            ablation_mask=resolved_mask,
        )
        self.lambda_param = self.config.lambda_param
        self.beta_param = self.config.beta_param
        self.gamma_param = self.config.gamma_param
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
            gamma_param=self.gamma_param,
            ablation_mask=(int(self.comp_a), int(self.comp_b), int(self.comp_c)),
        )

    def foveate(
        self,
        optical_energy: torch.Tensor,
        patch_embeds: torch.Tensor,
        query_embeds: Optional[torch.Tensor] = None,
    ) -> Tuple[Tuple[int, int, int, int], float, torch.Tensor]:
        """Execute unified FRSA-PBCF (Eq. 1)-(Eq. 4) variational optical foveation."""
        p_opt = compute_bregman_optical_distribution(optical_energy)
        p_lex = compute_lexical_spatial_distribution(patch_embeds, query_embeds)
        posterior_2d = compute_variational_posterior(
            p_opt, p_lex, gamma_param=self.gamma_param, comp_a=self.comp_a, comp_b=self.comp_b
        )
        box, j_star, _, _ = select_poisson_bregman_subwindow(posterior_2d, comp_c=self.comp_c)
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
