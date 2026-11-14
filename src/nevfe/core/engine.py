# -*- coding: utf-8 -*-
# SPEC_REF: docs/specs/team_1/spec_iter_6.md@v1.1 (sha256=b58c0f8eb71f551d109b637ae2e22737a8f03e153d2156b20b776caac305d6a4)
"""
Submodular Poisson-Bregman Multi-Locus Foveation (SPB-MLF) + IG-VFED Core Engine.

Backbone-agnostic mathematical core (<= 250 AST statement lines, <= 3 tunable parameters).
"""

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F

SPEC_REF = "docs/specs/team_1/spec_iter_6.md@v1.1 (sha256=b58c0f8eb71f551d109b637ae2e22737a8f03e153d2156b20b776caac305d6a4)"


@dataclass(frozen=True)
class AlgorithmConfig:
    """Strictly <= 3 tunable hyperparameters plus factorial ablation_mask."""

    lambda_param: float = 0.50
    beta_param: float = 0.05
    gamma_param: float = 0.65
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


def _normalize_patch_sphere(
    patch_embeds: torch.Tensor,
    tau_v: float = 1e-6,
) -> Tuple[torch.Tensor, float]:
    """Sanitize, peak-scale, spatially center, and unit-normalize ViT patch vectors."""
    _, _, dim = patch_embeds.shape
    v_raw = torch.nan_to_num(patch_embeds.float(), nan=0.0, posinf=0.0, neginf=0.0).reshape(-1, dim)
    if v_raw.numel() == 0:
        return v_raw, -1.0
    v = v_raw / torch.clamp(v_raw.abs().max(), min=1.0)
    dv = v - v.mean(dim=0, keepdim=True)
    tau_d = tau_v * (float(max(1, dim)) ** 0.5)
    r_v = torch.linalg.vector_norm(dv, dim=-1, keepdim=True)
    v_hat = torch.where(r_v > tau_d, dv / torch.clamp(r_v, min=tau_d), torch.zeros_like(dv))
    return v_hat, float(r_v.max().item()) - tau_d


def compute_spb_anchor_posterior(
    patch_embeds: torch.Tensor,
    query_embeds: Optional[torch.Tensor] = None,
    comp_a: bool = True,
    tau_v: float = 1e-6,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Eq. (1): Per-column empirical-standardized KL-specificity Soft-OR anchor posterior."""
    if patch_embeds.dim() != 3:
        raise ValueError(f"Expected 3D patch_embeds, got {patch_embeds.shape}")
    h_g, w_g, dim = patch_embeds.shape
    n_p = h_g * w_g
    unif = torch.full((h_g, w_g), 1.0 / float(max(1, n_p)), dtype=torch.float32, device=patch_embeds.device)
    if query_embeds is not None and query_embeds.dim() == 2 and query_embeds.shape[1] != dim:
        raise ValueError(f"Expected query_embeds with dim {dim}, got {query_embeds.shape}")
    if query_embeds is None or query_embeds.dim() != 2 or query_embeds.shape[0] == 0 or dim == 0 or n_p <= 1:
        return unif
    v_hat, margin = _normalize_patch_sphere(patch_embeds, tau_v=tau_v)
    if margin <= 0.0:
        return unif
    q_raw = torch.nan_to_num(query_embeds.float(), nan=0.0, posinf=0.0, neginf=0.0)
    q = q_raw / torch.clamp(q_raw.abs().amax(dim=-1, keepdim=True), min=1.0)
    r_e = torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    e_hat = torch.where(r_e > tau_v, q / torch.clamp(r_e, min=tau_v), torch.zeros_like(q))
    if float(e_hat.abs().max().item()) <= eps:
        return unif
    s_mat = torch.matmul(v_hat, e_hat.transpose(0, 1))
    if not comp_a:
        c_q = s_mat.max(dim=-1).values
        dc = c_q - c_q.mean()
        sigma_c = torch.sqrt(torch.clamp((dc * dc).mean(), min=0.0))
        return unif if float(sigma_c.item()) <= tau_v else F.softmax(dc / torch.clamp(sigma_c, min=tau_v), dim=-1).view(h_g, w_g)
    ds = s_mat - s_mat.mean(dim=0, keepdim=True)
    sigma_m = torch.sqrt(torch.clamp((ds * ds).mean(dim=0, keepdim=True), min=0.0))
    if float(sigma_m.max().item()) <= tau_v:
        return unif
    z_spec = torch.where(sigma_m > tau_v, ds / torch.clamp(sigma_m, min=tau_v), torch.zeros_like(ds))
    q_m = F.softmax(z_spec, dim=0)
    kl_terms = torch.nan_to_num(q_m * (torch.log(torch.clamp(q_m, min=eps)) + math.log(float(n_p))), nan=0.0, posinf=0.0, neginf=0.0)
    info_m = torch.clamp(kl_terms.sum(dim=0), min=0.0)
    w_raw = info_m * info_m
    w_sum = w_raw.sum()
    if float(w_sum.item()) <= eps:
        return unif
    w_m = w_raw / torch.clamp(w_sum, min=eps)
    z_max = z_spec.max(dim=1).values
    mix = torch.matmul(torch.exp(z_spec - z_max.unsqueeze(1)), w_m.unsqueeze(1)).squeeze(1)
    c_kl = z_max + torch.log(torch.clamp(mix, min=eps))
    dc_kl = c_kl - c_kl.mean()
    sigma_kl = torch.sqrt(torch.clamp((dc_kl * dc_kl).mean(), min=0.0))
    return unif if float(sigma_kl.item()) <= tau_v else F.softmax(dc_kl / torch.clamp(sigma_kl, min=tau_v), dim=-1).view(h_g, w_g)


def compute_signboard_manifold_posterior(
    p_anchor: torch.Tensor,
    patch_embeds: torch.Tensor,
    gamma_param: Optional[float] = None,
    comp_b: bool = True,
    tau_v: float = 1e-6,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Eq. (2): Anisotropic Coifman-Lafon horizontal-signboard Markov resolvent."""
    if p_anchor.dim() != 2 or patch_embeds.dim() != 3 or p_anchor.shape != patch_embeds.shape[:2]:
        raise ValueError(f"Expected 2D p_anchor matching patch_embeds[:2], got {p_anchor.shape} and {patch_embeds.shape}")
    h_g, w_g, dim = patch_embeds.shape
    n_p = h_g * w_g
    if n_p == 0:
        return torch.zeros_like(p_anchor, dtype=torch.float32)
    unif = torch.full((n_p,), 1.0 / float(n_p), dtype=torch.float32, device=p_anchor.device)
    pa = torch.nan_to_num(p_anchor.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0).reshape(-1)
    pa = pa / torch.clamp(pa.max(), min=1.0)
    pa_sum = pa.sum()
    p_hat = pa / pa_sum if float(pa_sum.item()) > eps else unif
    g_val = AlgorithmConfig().gamma_param if gamma_param is None else float(gamma_param)
    gamma_eff = min(1.0, max(0.0, g_val)) if comp_b else 0.0
    if gamma_eff <= eps or n_p <= 1 or dim == 0 or float((p_hat.max() - p_hat.min()).item()) <= tau_v:
        return p_hat.view(h_g, w_g)
    v_hat, margin = _normalize_patch_sphere(patch_embeds, tau_v=tau_v)
    if margin <= 0.0:
        return p_hat.view(h_g, w_g)
    rows = torch.arange(h_g, device=patch_embeds.device).unsqueeze(1).expand(h_g, w_g).reshape(-1)
    cols = torch.arange(w_g, device=patch_embeds.device).unsqueeze(0).expand(h_g, w_g).reshape(-1)
    di = (rows.unsqueeze(1) - rows.unsqueeze(0)).abs()
    dj = (cols.unsqueeze(1) - cols.unsqueeze(0)).abs()
    m_line = torch.where((di == 0) & (dj >= 1) & (dj <= 2), 1.0, torch.where((di == 1) & (dj <= 1), 0.5, 0.0))
    w_raw = torch.clamp(torch.matmul(v_hat, v_hat.transpose(0, 1)) - 0.5, min=0.0)
    w_raw = (w_raw * w_raw) * m_line
    d_raw = w_raw.sum(dim=-1)
    d_prod = d_raw.unsqueeze(1) * d_raw.unsqueeze(0)
    w_tilde = torch.where((d_raw.unsqueeze(1) > eps) & (d_raw.unsqueeze(0) > eps), w_raw / torch.clamp(d_prod, min=eps), torch.zeros_like(w_raw))
    d_tilde = w_tilde.sum(dim=-1, keepdim=True)
    eye = torch.eye(n_p, dtype=torch.float32, device=patch_embeds.device)
    k_mat = torch.where(d_tilde > eps, w_tilde / torch.clamp(d_tilde, min=eps), eye)
    r_n = 1.0 / float(n_p)
    p_line = torch.nan_to_num(torch.linalg.solve(eye - (1.0 - r_n) * k_mat.transpose(0, 1), r_n * p_hat), nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
    pl_sum = p_line.sum()
    p_line_hat = p_line / pl_sum if float(pl_sum.item()) > eps else p_hat
    p_mix = (1.0 - gamma_eff) * p_hat + gamma_eff * p_line_hat
    pm_sum = p_mix.sum()
    return (p_mix / pm_sum if float(pm_sum.item()) > eps else p_hat).view(h_g, w_g)


def _poisson_bregman_divergence(pm: torch.Tensor, am: torch.Tensor, eps_k: float) -> torch.Tensor:
    """Compute Generalized KL / Poisson-Bregman divergence J_Pois(pm || am)."""
    pc = torch.clamp(pm, min=eps_k, max=1.0)
    ac = torch.clamp(am, min=eps_k, max=1.0)
    return torch.clamp(torch.nan_to_num(pc * torch.log(pc / ac) - (pc - ac), nan=0.0, posinf=0.0, neginf=0.0), min=0.0)


def select_submodular_bregman_loci(
    posterior_2d: torch.Tensor,
    comp_c: bool = True,
    eps_k: float = 1e-6,
    eps: float = 1e-12,
) -> List[Tuple[Tuple[int, int, int, int], float, float, float]]:
    """Eq. (3): Submodular conditional-exterior 2-locus Poisson-Bregman subwindow selection."""
    if posterior_2d.dim() != 2:
        raise ValueError(f"Expected 2D posterior_2d, got {posterior_2d.shape}")
    h_g, w_g = posterior_2d.shape
    n_p = h_g * w_g
    if n_p == 0:
        return [((0, 0, h_g, w_g), 0.0, 0.0, 0.0)]
    p = torch.nan_to_num(posterior_2d.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
    p = p / torch.clamp(p.max(), min=1.0)
    p_sum = p.sum()
    p = p / p_sum if float(p_sum.item()) > eps else torch.full_like(p, 1.0 / float(n_p))
    if not comp_c or n_p <= 1:
        return [((0, 0, h_g, w_g), 0.0, 1.0, 1.0)]
    sat = F.pad(p, (1, 0, 1, 0)).cumsum(dim=0).cumsum(dim=1)
    ri = torch.triu_indices(h_g + 1, h_g + 1, offset=1, device=p.device)
    ci = torch.triu_indices(w_g + 1, w_g + 1, offset=1, device=p.device)
    i1, i2, j1, j2 = ri[0], ri[1], ci[0], ci[1]
    mass = torch.clamp(sat[i2.unsqueeze(1), j2.unsqueeze(0)] - sat[i1.unsqueeze(1), j2.unsqueeze(0)] - sat[i2.unsqueeze(1), j1.unsqueeze(0)] + sat[i1.unsqueeze(1), j1.unsqueeze(0)], min=0.0, max=1.0)
    area = ((i2 - i1).float().unsqueeze(1) * (j2 - j1).float().unsqueeze(0)) / float(n_p)
    j1_mat = torch.where((mass > area + eps_k) & (area >= eps_k) & (area <= 0.5), _poisson_bregman_divergence(mass, area, eps_k), torch.zeros_like(mass))
    idx1, n_cols = int(torch.argmax(j1_mat.view(-1)).item()), int(j1.numel())
    r1, c1 = idx1 // n_cols, idx1 % n_cols
    j1_star = float(j1_mat[r1, c1].item())
    if j1_star <= eps_k:
        return [((0, 0, h_g, w_g), 0.0, 1.0, 1.0)]
    bi1, bj1, bi2, bj2 = int(i1[r1].item()), int(j1[c1].item()), int(i2[r1].item()), int(j2[c1].item())
    p1, a1 = float(mass[r1, c1].item()), float(area[r1, c1].item())
    loci = [((bi1, bj1, bi2, bj2), j1_star, p1, a1)]
    m_rem, a_rem = 1.0 - p1, 1.0 - a1
    if m_rem > eps_k and a_rem > eps_k:
        disj = ((i2 <= bi1) | (i1 >= bi2)).unsqueeze(1) | ((j2 <= bj1) | (j1 >= bj2)).unsqueeze(0)
        adm2 = disj & (mass > 2.0 * area + eps_k) & (p1 + mass > 0.5) & (area >= eps_k) & (area <= 0.5 * a_rem)
        j2_mat = torch.where(adm2, _poisson_bregman_divergence(mass / m_rem, area / a_rem, eps_k), torch.zeros_like(mass))
        idx2 = int(torch.argmax(j2_mat.view(-1)).item())
        r2, c2 = idx2 // n_cols, idx2 % n_cols
        j2_ext = float(j2_mat[r2, c2].item())
        if j2_ext >= 0.5 * j1_star and j2_ext > eps_k:
            b2 = (int(i1[r2].item()), int(j1[c2].item()), int(i2[r2].item()), int(j2[c2].item()))
            loci.append((b2, j2_ext, float(mass[r2, c2].item()), float(area[r2, c2].item())))
    return loci


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
    """Backbone-agnostic SPB-MLF + IG-VFED engine configured by AlgorithmConfig."""

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
        m1, m2, m3 = base.ablation_mask
        resolved_mask = (
            int(bool(comp_a)) if comp_a is not None else int(m1),
            int(bool(comp_b)) if comp_b is not None else int(m2),
            int(bool(comp_c)) if comp_c is not None else int(m3),
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
        patch_embeds: torch.Tensor,
        query_embeds: Optional[torch.Tensor] = None,
    ) -> Tuple[List[Tuple[int, int, int, int]], float, torch.Tensor]:
        """Execute unified SPB-MLF (Eq. 1)-(Eq. 3) submodular multi-locus foveation."""
        p_anchor = compute_spb_anchor_posterior(patch_embeds, query_embeds, comp_a=self.comp_a)
        p_post = compute_signboard_manifold_posterior(
            p_anchor, patch_embeds, gamma_param=self.gamma_param, comp_b=self.comp_b
        )
        loci = select_submodular_bregman_loci(p_post, comp_c=self.comp_c)
        _, j1, _, _ = loci[0]
        h_g, w_g, _ = patch_embeds.shape
        return ([loc[0] for loc in loci] if j1 > 1e-6 else [(0, 0, h_g, w_g)]), j1, p_post

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
