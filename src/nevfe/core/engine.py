# -*- coding: utf-8 -*-
# SPEC_REF: docs/specs/team_1/spec_iter_9.md@v1.0 (sha256=54743664938d12abb12e0ef54faab0ac9d5a21adc85562a283fc85ba79601b5b)
"""
Stroke-Gated KL-Specificity & Seam-Overlapped Isotropic Foveation (SKS-OIF) + IG-VFED Core Engine.

Backbone-agnostic mathematical core (<= 250 AST statement lines, <= 3 tunable parameters).
"""

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F

SPEC_REF = "docs/specs/team_1/spec_iter_9.md@v1.0 (sha256=54743664938d12abb12e0ef54faab0ac9d5a21adc85562a283fc85ba79601b5b)"


@dataclass(frozen=True)
class AlgorithmConfig:
    """Strictly <= 3 tunable hyperparameters plus factorial ablation_mask."""

    lambda_param: float = 0.50
    beta_param: float = 0.05
    gamma_param: float = 0.85
    ablation_mask: Tuple[int, ...] = (1, 1, 1)

    def __post_init__(self) -> None:
        if not (0.0 <= float(self.lambda_param) <= 1.5):
            raise ValueError(f"lambda_param must be in [0.0, 1.5], got {self.lambda_param}")
        if not (0.001 <= float(self.beta_param) <= 5.0):
            raise ValueError(f"beta_param must be in [0.001, 5.0], got {self.beta_param}")
        if not (0.0 <= float(self.gamma_param) <= 1.0):
            raise ValueError(f"gamma_param must be in [0.0, 1.0], got {self.gamma_param}")
        mask = tuple(int(bool(x)) for x in self.ablation_mask)
        if len(mask) != 2 + 1:
            raise ValueError(f"ablation_mask must have length 3, got {self.ablation_mask}")
        object.__setattr__(self, "lambda_param", float(self.lambda_param))
        object.__setattr__(self, "beta_param", float(self.beta_param))
        object.__setattr__(self, "gamma_param", float(self.gamma_param))
        object.__setattr__(self, "ablation_mask", mask)


def compute_chromatic_stroke_prior(
    optical_energy: torch.Tensor,
    gamma_param: Optional[float] = None,
    comp_a: bool = True,
    tau_v: float = 1e-6,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Eq. (1): 0-homogeneous Naka-Rushton chromatic SO(2) stroke-support Bayesian prior."""
    if optical_energy.dim() != 2:
        raise ValueError(f"Expected 2D optical_energy, got {optical_energy.shape}")
    e = torch.nan_to_num(optical_energy.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
    if e.numel() == 0:
        return e
    g_raw = AlgorithmConfig().gamma_param if gamma_param is None else float(gamma_param)
    g_eff = 0.0 if (math.isnan(g_raw) or not comp_a) else min(1.0, max(0.0, g_raw))
    m_e, ones = e.max(), torch.ones_like(e)
    if g_eff <= eps or float(m_e.item()) <= eps:
        return ones
    e_tilde = e / torch.clamp(m_e, min=eps)
    if float((1.0 - e_tilde.min()).item()) <= tau_v:
        return ones
    g_stroke = torch.clamp(e_tilde / (e_tilde + e_tilde.mean() + eps), min=tau_v, max=1.0)
    return torch.pow(g_stroke, g_eff)


def compute_stroke_specificity_posterior(
    patch_embeds: torch.Tensor,
    stroke_prior: torch.Tensor,
    query_embeds: Optional[torch.Tensor] = None,
    comp_b: bool = True,
    tau_v: float = 1e-6,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Eq. (2): Stroke-gated shared-RMS KL-specificity Soft-OR spatial posterior."""
    if patch_embeds.dim() != 2 + 1:
        raise ValueError(f"Expected 3D patch_embeds, got {patch_embeds.shape}")
    h_g, w_g, dim = patch_embeds.shape
    if stroke_prior.dim() != 2 or stroke_prior.shape != (h_g, w_g):
        raise ValueError(f"Expected 2D stroke_prior matching {(h_g, w_g)}, got {stroke_prior.shape}")
    n_p = h_g * w_g
    unif = torch.full((h_g, w_g), 1.0 / float(max(1, n_p)), dtype=torch.float32, device=patch_embeds.device)
    if query_embeds is None or query_embeds.dim() != 2:
        return unif
    if query_embeds.shape[1] != dim:
        raise ValueError(f"Expected query_embeds with dim {dim}, got {query_embeds.shape}")
    if query_embeds.shape[0] == 0 or dim == 0 or n_p <= 1:
        return unif
    v_raw = torch.nan_to_num(patch_embeds.float(), nan=0.0, posinf=0.0, neginf=0.0).view(n_p, dim)
    q_raw = torch.nan_to_num(query_embeds.float(), nan=0.0, posinf=0.0, neginf=0.0)
    pi_raw = torch.nan_to_num(stroke_prior.float(), nan=1.0, posinf=1.0, neginf=0.0).clamp(min=0.0).view(n_p)
    pi = (pi_raw / torch.clamp(pi_raw.max(), min=1.0)) if float(pi_raw.max().item()) > eps else torch.ones_like(pi_raw)
    dv = v_raw / torch.clamp(v_raw.abs().max(), min=1.0)
    dv = dv - dv.mean(dim=0, keepdim=True)
    q = q_raw / torch.clamp(q_raw.abs().amax(dim=-1, keepdim=True), min=1.0)
    r_v, r_e = torch.linalg.vector_norm(dv, dim=-1, keepdim=True), torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    tau_d = tau_v * math.sqrt(float(max(1, dim)))
    if float(r_v.max().item()) <= tau_d or float(r_e.max().item()) <= tau_v:
        return unif
    v_hat = torch.where(r_v > tau_d, dv / torch.clamp(r_v, min=tau_d), torch.zeros_like(dv))
    e_hat = torch.where(r_e > tau_v, q / torch.clamp(r_e, min=tau_v), torch.zeros_like(q))
    s_mat = torch.matmul(v_hat, e_hat.transpose(0, 1))
    if not comp_b:
        c_raw = s_mat.max(dim=-1).values
    else:
        ds = s_mat - s_mat.mean(dim=0, keepdim=True)
        sig_s = torch.sqrt(torch.clamp((ds * ds).mean(), min=0.0))
        if float(sig_s.item()) <= tau_v:
            return unif
        z_mat = ds / torch.clamp(sig_s, min=tau_v)
        q_prob = F.softmax(z_mat, dim=0)
        kl = torch.nan_to_num(q_prob * (torch.log(torch.clamp(q_prob, min=eps)) + math.log(float(n_p))), nan=0.0, posinf=0.0, neginf=0.0)
        s_m = torch.matmul(pi.unsqueeze(0), q_prob).squeeze(0) * torch.clamp(kl.sum(dim=0), min=0.0)
        w_raw = s_m * s_m
        if float(w_raw.sum().item()) <= eps:
            return unif
        w_m = (w_raw / torch.clamp(w_raw.sum(), min=eps)).unsqueeze(0)
        z_max = z_mat.max(dim=-1, keepdim=True).values
        c_raw = (z_max + torch.log(torch.clamp((w_m * torch.exp(z_mat - z_max)).sum(dim=-1, keepdim=True), min=eps))).squeeze(-1)
    dc = c_raw - c_raw.mean()
    sig_c = torch.sqrt(torch.clamp((dc * dc).mean(), min=0.0))
    if float(sig_c.item()) <= tau_v:
        return unif
    w_post = pi * F.softmax(dc / torch.clamp(sig_c, min=tau_v), dim=-1)
    z_post = w_post.sum()
    return (w_post / torch.clamp(z_post, min=eps)).view(h_g, w_g) if float(z_post.item()) > eps else unif


def select_optical_foveal_locus(
    posterior_2d: torch.Tensor,
    eps_k: float = 1e-6,
    eps: float = 1e-12,
) -> Tuple[Tuple[int, int, int, int], float, float, float]:
    """Eq. (3): Single-locus Generalized KL / Poisson-Bregman subwindow selection."""
    if posterior_2d.dim() != 2:
        raise ValueError(f"Expected 2D posterior_2d, got {posterior_2d.shape}")
    h_g, w_g = posterior_2d.shape
    n_p = h_g * w_g
    if n_p == 0:
        return (0, 0, h_g, w_g), 0.0, 0.0, 0.0
    p = torch.nan_to_num(posterior_2d.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
    p = p / torch.clamp(p.max(), min=1.0)
    p = p / p.sum() if float(p.sum().item()) > eps else torch.full_like(p, 1.0 / float(n_p))
    if n_p <= 1:
        return (0, 0, h_g, w_g), 0.0, 1.0, 1.0
    sat = F.pad(p, (1, 0, 1, 0)).cumsum(dim=0).cumsum(dim=1)
    ri, ci = torch.triu_indices(h_g + 1, h_g + 1, offset=1, device=p.device), torch.triu_indices(w_g + 1, w_g + 1, offset=1, device=p.device)
    i1, i2, j1, j2 = ri[0], ri[1], ci[0], ci[1]
    mass = torch.clamp(sat[i2.unsqueeze(1), j2.unsqueeze(0)] - sat[i1.unsqueeze(1), j2.unsqueeze(0)] - sat[i2.unsqueeze(1), j1.unsqueeze(0)] + sat[i1.unsqueeze(1), j1.unsqueeze(0)], min=0.0, max=1.0)
    area = ((i2 - i1).float().unsqueeze(1) * (j2 - j1).float().unsqueeze(0)) / float(n_p)
    p_cl, a_cl = torch.clamp(mass, min=eps_k, max=1.0), torch.clamp(area, min=eps_k, max=1.0)
    j_raw = torch.nan_to_num(p_cl * torch.log(p_cl / a_cl) - (p_cl - a_cl), nan=0.0, posinf=0.0, neginf=0.0)
    j_mat = torch.where((mass > area + eps_k) & (area >= eps_k) & (area <= 0.5), torch.clamp(j_raw, min=0.0), torch.zeros_like(j_raw))
    r_idx, c_idx = divmod(int(torch.argmax(j_mat.view(-1)).item()), int(j1.numel()))
    j_star = float(j_mat[r_idx, c_idx].item())
    if j_star <= eps_k:
        return (0, 0, h_g, w_g), 0.0, 1.0, 1.0
    box = (int(i1[r_idx].item()), int(j1[c_idx].item()), int(i2[r_idx].item()), int(j2[c_idx].item()))
    return box, j_star, float(mass[r_idx, c_idx].item()), float(area[r_idx, c_idx].item())


def plan_isotropic_strip_boxes(
    bbox: Tuple[int, int, int, int],
    img_size: Tuple[int, int],
    patch_stride: float,
    max_tiles: int = 2 + 1,
    comp_c: bool = True,
    eps: float = 1e-6,
) -> Tuple[List[Tuple[int, int, int, int]], Tuple[int, int], float]:
    """Eq. (4): 112px-gated 25%-seam-overlapped exact 1:1 square sub-tile planner."""
    w_r, h_r = img_size
    w_img, h_img = int(w_r), int(h_r)
    if w_img <= 0 or h_img <= 0 or int(max_tiles) <= 0:
        return [], (1, 1), 0.0
    bx1_i, by1_i, bx2_i, by2_i = bbox
    cx1, cx2 = max(0.0, min(float(w_img), float(min(bx1_i, bx2_i)))), max(0.0, min(float(w_img), float(max(bx1_i, bx2_i))))
    cy1, cy2 = max(0.0, min(float(h_img), float(min(by1_i, by2_i)))), max(0.0, min(float(h_img), float(max(by1_i, by2_i))))
    dx, dy = max(min(2.0, float(w_img)), min(float(w_img), cx2 - cx1)), max(min(2.0, float(h_img)), min(float(h_img), cy2 - cy1))
    s_p = max(1.0, float(patch_stride))
    step_f, d_min = ((1.0 - 0.5 * 0.5) if comp_c else 1.0), (2.0 + 2.0) * s_p
    cands = [(1, 1)] + ([(2, 1), (1, 2)] if max_tiles >= 2 else []) + ([(2 + 1, 1), (1, 2 + 1)] if max_tiles >= 2 + 1 else []) + ([(2, 2)] if max_tiles >= 2 + 2 else [])
    adm = [
        g for g in cands
        if (g[0] == 1 or dx >= (1.0 + step_f * float(g[0] - 1)) * d_min)
        and (g[1] == 1 or dy >= (1.0 + step_f * float(g[1] - 1)) * d_min)
        and (1.0 + step_f * float(g[0] - 1)) * s_p <= float(w_img) + eps
        and (1.0 + step_f * float(g[1] - 1)) * s_p <= float(h_img) + eps
    ] or [(1, 1)]
    gx_s, gy_s = min(adm, key=lambda g: (abs(math.log(max(eps, dx) / max(eps, dy)) - math.log((1.0 + step_f * float(g[0] - 1)) / (1.0 + step_f * float(g[1] - 1)))), g[0] * g[1]))
    fx_s, fy_s = 1.0 + step_f * float(gx_s - 1), 1.0 + step_f * float(gy_s - 1)
    s_fl = min(2.0 * s_p, float(min(w_img, h_img)))
    s_unit = min(max(dx / fx_s, dy / fy_s, s_fl), float(w_img) / fx_s, float(h_img) / fy_s)
    s_int = max(1, min(w_img, h_img, int(round(s_unit))))
    s_x, s_y = min(float(w_img), max(float(s_int), s_unit * fx_s)), min(float(h_img), max(float(s_int), s_unit * fy_s))
    sx1, sy1 = max(0.0, min(float(w_img) - s_x, 0.5 * (cx1 + cx2 - s_x))), max(0.0, min(float(h_img) - s_y, 0.5 * (cy1 + cy2 - s_y)))
    dx_step = (s_x - float(s_int)) / float(max(1, gx_s - 1)) if gx_s > 1 else 0.0
    dy_step = (s_y - float(s_int)) / float(max(1, gy_s - 1)) if gy_s > 1 else 0.0
    boxes = []
    for ty in range(gy_s):
        for tx in range(gx_s):
            x_0 = max(0, min(w_img - s_int, int(round(sx1 + float(tx) * dx_step))))
            y_0 = max(0, min(h_img - s_int, int(round(sy1 + float(ty) * dy_step))))
            boxes.append((x_0, y_0, x_0 + s_int, y_0 + s_int))
    return boxes, (gx_s, gy_s), float(s_int)


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
    """Backbone-agnostic SKS-OIF + IG-VFED engine configured by AlgorithmConfig."""

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
        m_a, m_b, m_c = base.ablation_mask
        resolved_mask = (
            int(bool(comp_a)) if comp_a is not None else int(m_a),
            int(bool(comp_b)) if comp_b is not None else int(m_b),
            int(bool(comp_c)) if comp_c is not None else int(m_c),
        )
        self.config = AlgorithmConfig(
            lambda_param=base.lambda_param if lambda_param is None else float(lambda_param),
            beta_param=base.beta_param if beta_param is None else float(beta_param),
            gamma_param=base.gamma_param if gamma_param is None else float(gamma_param),
            ablation_mask=resolved_mask,
        )
        self.lambda_param, self.beta_param, self.gamma_param = (
            self.config.lambda_param,
            self.config.beta_param,
            self.config.gamma_param,
        )
        c_a, c_b, c_c = self.config.ablation_mask
        self.comp_a, self.comp_b, self.comp_c = bool(c_a), bool(c_b), bool(c_c)

    def set_ablation_switches(
        self,
        comp_a: bool = True,
        comp_b: bool = True,
        comp_c: bool = True,
    ) -> None:
        """Dynamically configure micro-ablation toggles for factorial probing."""
        self.comp_a, self.comp_b, self.comp_c = bool(comp_a), bool(comp_b), bool(comp_c)
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
        """Execute SKS-OIF (Eq. 1)-(Eq. 3) single-locus stroke-gated foveation."""
        stroke_prior = compute_chromatic_stroke_prior(
            optical_energy, gamma_param=self.gamma_param, comp_a=self.comp_a
        )
        posterior_2d = compute_stroke_specificity_posterior(
            patch_embeds, stroke_prior, query_embeds=query_embeds, comp_b=self.comp_b
        )
        box, j_star, _, _ = select_optical_foveal_locus(posterior_2d)
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
