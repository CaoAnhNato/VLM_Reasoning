# -*- coding: utf-8 -*-
# SPEC_REF: docs/specs/team_1/spec_iter_7.md@v1.0 (sha256=32eecd8fede0baf6e60bcf0682722172c0f15f851c99e3851fc61fee7dfa3395)
"""
Wiener Stroke-Restored Aspect-Adaptive Multi-Strip Foveation (WSR-MSF) + IG-VFED Core Engine.

Backbone-agnostic mathematical core (<= 250 AST statement lines, <= 3 tunable parameters).
"""

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F

SPEC_REF = "docs/specs/team_1/spec_iter_7.md@v1.0 (sha256=32eecd8fede0baf6e60bcf0682722172c0f15f851c99e3851fc61fee7dfa3395)"


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


def compute_optical_anchor_posterior(
    patch_embeds: torch.Tensor,
    query_embeds: Optional[torch.Tensor] = None,
    comp_a: bool = True,
    tau_v: float = 1e-6,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Eq. (1): Diffusion-free KL-specificity Soft-OR spatial anchor (or max_m fallback)."""
    if patch_embeds.dim() != 2 + 1:
        raise ValueError(f"Expected 3D patch_embeds, got {patch_embeds.shape}")
    h_g, w_g, dim = patch_embeds.shape
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
    dv = v_raw / torch.clamp(v_raw.abs().max(), min=1.0)
    dv = dv - dv.mean(dim=0, keepdim=True)
    q = q_raw / torch.clamp(q_raw.abs().amax(dim=-1, keepdim=True), min=1.0)
    r_v, r_e = torch.linalg.vector_norm(dv, dim=-1, keepdim=True), torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    tau_d = tau_v * math.sqrt(float(max(1, dim)))
    if float(r_v.max().item()) <= tau_d:
        return unif
    v_hat = torch.where(r_v > tau_d, dv / torch.clamp(r_v, min=tau_d), torch.zeros_like(dv))
    e_hat = torch.where(r_e > tau_v, q / torch.clamp(r_e, min=tau_v), torch.zeros_like(q))
    s_mat = torch.matmul(v_hat, e_hat.transpose(0, 1))
    if not comp_a:
        c_val = s_mat.max(dim=-1).values
    else:
        ds = s_mat - s_mat.mean(dim=0, keepdim=True)
        sig_m = torch.sqrt(torch.clamp((ds * ds).mean(dim=0, keepdim=True), min=0.0))
        if float(sig_m.max().item()) <= tau_v:
            return unif
        z_mat = torch.where(sig_m > tau_v, ds / torch.clamp(sig_m, min=tau_v), torch.zeros_like(ds))
        q_prob = F.softmax(z_mat, dim=0)
        kl = torch.nan_to_num(q_prob * (torch.log(torch.clamp(q_prob, min=eps)) + math.log(float(n_p))), nan=0.0, posinf=0.0, neginf=0.0)
        info_m = torch.clamp(kl.sum(dim=0), min=0.0)
        w_raw = info_m * info_m
        if float(w_raw.sum().item()) <= eps:
            return unif
        w_m = (w_raw / torch.clamp(w_raw.sum(), min=eps)).unsqueeze(0)
        z_max = z_mat.max(dim=-1, keepdim=True).values
        c_val = (z_max + torch.log(torch.clamp((w_m * torch.exp(z_mat - z_max)).sum(dim=-1, keepdim=True), min=eps))).squeeze(-1)
    dc = c_val - c_val.mean()
    sig_c = torch.sqrt(torch.clamp((dc * dc).mean(), min=0.0))
    if float(sig_c.item()) <= tau_v:
        return unif
    return F.softmax(dc / torch.clamp(sig_c, min=tau_v), dim=-1).view(h_g, w_g)


def _j_pois(mass_val: torch.Tensor, area_val: torch.Tensor, adm: torch.Tensor, eps_k: float) -> torch.Tensor:
    p_cl, a_cl = torch.clamp(mass_val, min=eps_k, max=1.0), torch.clamp(area_val, min=eps_k, max=1.0)
    j_raw = torch.nan_to_num(p_cl * torch.log(p_cl / a_cl) - (p_cl - a_cl), nan=0.0, posinf=0.0, neginf=0.0)
    return torch.where(adm, torch.clamp(j_raw, min=0.0), torch.zeros_like(j_raw))


def select_optical_foveal_loci(
    posterior_2d: torch.Tensor,
    comp_a: bool = True,
    eps_k: float = 1e-6,
    eps: float = 1e-12,
) -> List[Tuple[Tuple[int, int, int, int], float, float, float]]:
    """Eq. (2): Diffusion-free conditional-exterior Poisson-Bregman locus selection."""
    if posterior_2d.dim() != 2:
        raise ValueError(f"Expected 2D posterior_2d, got {posterior_2d.shape}")
    h_g, w_g = posterior_2d.shape
    n_p = h_g * w_g
    if n_p == 0:
        return [((0, 0, h_g, w_g), 0.0, 0.0, 0.0)]
    p = torch.nan_to_num(posterior_2d.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
    p = p / torch.clamp(p.max(), min=1.0)
    p = p / p.sum() if float(p.sum().item()) > eps else torch.full_like(p, 1.0 / float(n_p))
    if n_p <= 1:
        return [((0, 0, h_g, w_g), 0.0, 1.0, 1.0)]
    sat = F.pad(p, (1, 0, 1, 0)).cumsum(dim=0).cumsum(dim=1)
    ri, ci = torch.triu_indices(h_g + 1, h_g + 1, offset=1, device=p.device), torch.triu_indices(w_g + 1, w_g + 1, offset=1, device=p.device)
    i1, i2, j1, j2 = ri[0], ri[1], ci[0], ci[1]
    mass = torch.clamp(sat[i2.unsqueeze(1), j2.unsqueeze(0)] - sat[i1.unsqueeze(1), j2.unsqueeze(0)] - sat[i2.unsqueeze(1), j1.unsqueeze(0)] + sat[i1.unsqueeze(1), j1.unsqueeze(0)], min=0.0, max=1.0)
    area = ((i2 - i1).float().unsqueeze(1) * (j2 - j1).float().unsqueeze(0)) / float(n_p)
    j1_mat = _j_pois(mass, area, (mass > area + eps_k) & (area >= eps_k) & (area <= 0.5), eps_k)
    n_cols = int(j1.numel())
    r1, c1 = divmod(int(torch.argmax(j1_mat.view(-1)).item()), n_cols)
    j1_star = float(j1_mat[r1, c1].item())
    if j1_star <= eps_k:
        return [((0, 0, h_g, w_g), 0.0, 1.0, 1.0)]
    bi1, bj1, bi2, bj2 = int(i1[r1].item()), int(j1[c1].item()), int(i2[r1].item()), int(j2[c1].item())
    p1, a1 = float(mass[r1, c1].item()), float(area[r1, c1].item())
    loci = [((bi1, bj1, bi2, bj2), j1_star, p1, a1)]
    m_rem, a_rem = 1.0 - p1, 1.0 - a1
    if comp_a and m_rem > eps_k and a_rem > eps_k:
        disj = ((i2 <= bi1) | (i1 >= bi2)).unsqueeze(1) | ((j2 <= bj1) | (j1 >= bj2)).unsqueeze(0)
        j2_mat = _j_pois(mass / m_rem, area / a_rem, disj & (mass > 2.0 * area + eps_k) & (p1 + mass > 0.5) & (area >= eps_k) & (area <= 0.5 * a_rem), eps_k)
        r2, c2 = divmod(int(torch.argmax(j2_mat.view(-1)).item()), n_cols)
        j2_star = float(j2_mat[r2, c2].item())
        if j2_star >= 0.5 * j1_star and j2_star > eps_k:
            loci.append(((int(i1[r2].item()), int(j1[c2].item()), int(i2[r2].item()), int(j2[c2].item())), j2_star, float(mass[r2, c2].item()), float(area[r2, c2].item())))
    return loci


def plan_overlapped_strip_boxes(
    bbox: Tuple[int, int, int, int],
    img_size: Tuple[int, int],
    patch_stride: float,
    max_tiles: int = 2 + 1,
    comp_b: bool = True,
    eps: float = 1e-6,
) -> Tuple[List[Tuple[int, int, int, int]], Tuple[int, int], float]:
    """Eq. (3): 25%-seam-overlapped ViT-patch-stride aspect-adaptive strip planner."""
    w_img, h_img = img_size
    if w_img <= 0 or h_img <= 0 or max_tiles <= 0:
        return [], (1, 1), 0.0
    bx1, by1, bx2, by2 = bbox
    cx1, cx2 = max(0.0, min(float(w_img), float(min(bx1, bx2)))), max(0.0, min(float(w_img), float(max(bx1, bx2))))
    cy1, cy2 = max(0.0, min(float(h_img), float(min(by1, by2)))), max(0.0, min(float(h_img), float(max(by1, by2))))
    dx, dy = max(min(2.0, float(w_img)), min(float(w_img), cx2 - cx1)), max(min(2.0, float(h_img)), min(float(h_img), cy2 - cy1))
    s_p, sig_step = max(1.0, float(patch_stride)), (1.0 - 0.5 * 0.5 if comp_b else 1.0)
    cands = [(1, 1)] + ([(2, 1), (1, 2)] if max_tiles >= 2 else []) + ([(2 + 1, 1), (1, 2 + 1)] if max_tiles >= 2 + 1 else [])
    d_min = (2.0 + 2.0) * s_p

    def _ok(gx: int, gy: int) -> bool:
        fx, fy = 1.0 + sig_step * float(gx - 1), 1.0 + sig_step * float(gy - 1)
        return ((gx == 1 or dx >= fx * s_p) and (gy == 1 or dy >= fy * s_p) and fx * s_p <= float(w_img) + eps and fy * s_p <= float(h_img) + eps) if comp_b else (gx <= max(1, int(dx // d_min)) and gy <= max(1, int(dy // d_min)))

    adm = [g for g in cands if _ok(g[0], g[1])] or [(1, 1)]
    gx_s, gy_s = min(adm, key=lambda g: (abs(math.log(max(eps, dx) / max(eps, dy)) - math.log((1.0 + sig_step * float(g[0] - 1)) / (1.0 + sig_step * float(g[1] - 1)))), g[0] * g[1]))
    fx, fy = 1.0 + sig_step * float(gx_s - 1), 1.0 + sig_step * float(gy_s - 1)
    s_fl = min((2.0 if comp_b else (2.0 + 2.0)) * s_p, float(min(w_img, h_img)))
    s_unit = min(max(dx / fx, dy / fy, s_fl), float(w_img) / fx, float(h_img) / fy)
    s_x, s_y = min(float(w_img), max(dx, s_unit * fx)), min(float(h_img), max(dy, s_unit * fy))
    sx1, sy1 = max(0.0, min(float(w_img) - s_x, 0.5 * (cx1 + cx2 - s_x))), max(0.0, min(float(h_img) - s_y, 0.5 * (cy1 + cy2 - s_y)))
    w_sub, h_sub = s_x / fx, s_y / fy
    step_x = (s_x - w_sub) / float(max(1, gx_s - 1)) if gx_s > 1 else 0.0
    step_y = (s_y - h_sub) / float(max(1, gy_s - 1)) if gy_s > 1 else 0.0
    boxes = []
    for iy in range(gy_s):
        for ix in range(gx_s):
            x0, y0 = sx1 + float(ix) * step_x, sy1 + float(iy) * step_y
            tx1, ty1 = max(0, min(w_img - 1, int(round(x0)))), max(0, min(h_img - 1, int(round(y0))))
            boxes.append((tx1, ty1, max(tx1 + 1, min(w_img, int(round(x0 + w_sub)))), max(ty1 + 1, min(h_img, int(round(y0 + h_sub))))))
    return boxes, (gx_s, gy_s), max(w_sub, h_sub)


def _binomial_smooth_2d(x: torch.Tensor) -> torch.Tensor:
    out, (_, h_t, w_t) = x, x.shape
    for _ in range(2):
        if w_t > 1:
            wp = torch.cat([out[:, :, 1:2], out, out[:, :, w_t - 2 : w_t - 1]], dim=-1)
            out = 0.5 * 0.5 * (wp[:, :, :w_t] + 2.0 * wp[:, :, 1 : w_t + 1] + wp[:, :, 2 : w_t + 2])
        if h_t > 1:
            hp = torch.cat([out[:, 1:2, :], out, out[:, h_t - 2 : h_t - 1, :]], dim=1)
            out = 0.5 * 0.5 * (hp[:, :h_t, :] + 2.0 * hp[:, 1 : h_t + 1, :] + hp[:, 2 : h_t + 2, :])
    return out


def restore_wiener_optical_strokes(
    rgb_chw: torch.Tensor,
    magnification: float = 2.0,
    gamma_param: Optional[float] = None,
    comp_c: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Eq. (4): Median-noise-gated MMSE local-Wiener achromatic stroke restoration."""
    if rgb_chw.dim() != 2 + 1:
        raise ValueError(f"Expected 3D [C, H, W] tensor, got {rgb_chw.shape}")
    img = torch.nan_to_num(rgb_chw.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    c_t, h_t, w_t = img.shape
    g_val = AlgorithmConfig().gamma_param if gamma_param is None else float(gamma_param)
    gamma_eff = min(1.0, max(0.0, g_val)) if comp_c else 0.0
    eta_m = 1.0 - 1.0 / (max(1.0, float(magnification)) ** 2)
    if c_t == 0 or h_t * w_t <= 1 or gamma_eff * eta_m <= eps:
        return img
    lum = img.mean(dim=0, keepdim=True)
    d = lum - _binomial_smooth_2d(lum)
    v_loc = _binomial_smooth_2d(d * d)
    nu0 = torch.median(v_loc)
    g_w = torch.clamp(v_loc - 2.0 * nu0, min=0.0) / (v_loc + 2.0 * nu0 + eps)
    return torch.clamp(img + 2.0 * gamma_eff * eta_m * g_w * d, 0.0, 1.0)


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
    """Backbone-agnostic WSR-MSF + IG-VFED engine configured by AlgorithmConfig."""

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
        self.lambda_param, self.beta_param, self.gamma_param = (
            self.config.lambda_param,
            self.config.beta_param,
            self.config.gamma_param,
        )
        c1, c2, c3 = self.config.ablation_mask
        self.comp_a, self.comp_b, self.comp_c = bool(c1), bool(c2), bool(c3)

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
        patch_embeds: torch.Tensor,
        query_embeds: Optional[torch.Tensor] = None,
    ) -> Tuple[List[Tuple[int, int, int, int]], float, torch.Tensor]:
        """Execute WSR-MSF (Eq. 1)-(Eq. 2) diffusion-free foveation."""
        h_g, w_g, _ = patch_embeds.shape
        p_post = compute_optical_anchor_posterior(patch_embeds, query_embeds=query_embeds, comp_a=self.comp_a)
        loci = select_optical_foveal_loci(p_post, comp_a=self.comp_a)
        _, j1, _, _ = loci[0]
        boxes = [loc[0] for loc in loci] if j1 > 1e-6 else [(0, 0, h_g, w_g)]
        return boxes, j1, p_post

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
