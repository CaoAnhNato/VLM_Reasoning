# -*- coding: utf-8 -*-
"""
Immutable FRSA-PBCF ("011") Optical Foveation & Mean-Field Reference Core.

Frozen at base/frsa-011-frozen from swarm/team-1 (0f8a2d4d / 4c8ee01).
Must remain bit-identical across all Track 1 and Track 2 iterations.
"""

from typing import Optional, Tuple
import torch
import torch.nn.functional as F


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


def foveate_frozen_011(
    patch_embeds: torch.Tensor,
    query_embeds: Optional[torch.Tensor] = None,
) -> Tuple[Tuple[int, int, int, int], float, torch.Tensor]:
    """Execute immutable FRSA-PBCF ('011') pure Layer-0 lexical + Poisson-Bregman foveation."""
    p_lex = compute_lexical_spatial_distribution(patch_embeds, query_embeds=query_embeds)
    box, j_star, _, _ = select_poisson_bregman_subwindow(p_lex, comp_c=True)
    return box, j_star, p_lex
