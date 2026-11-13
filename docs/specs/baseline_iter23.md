# Baseline Specification: Iteration-23 IG-VFED (`baseline_iter23.md`)

## 1. Header
- **team_id**: `swarm_base`
- **direction_id**: `direction_0`
- **iteration**: `23`
- **spec_version**: `v1.0`
- **base_best_commit**: `148f870`

## 2. Backbone Tensor Invariants
- **Vision Encoder (`InternViT-300M`)**: Native unwarped patch embeddings `vit_embeds` of shape `[N_tiles, 256, 896]` (`bfloat16`), immutable per image and entering LLM Layer 0 unmodified.
- **Language Decoder (`Qwen2.5-0.5B-Instruct`)**: `24` layers, `14` query heads, `2` KV heads (`7:1` GQA ratio), hidden dimension `896`, dynamic `vocab_size` (never hardcoded in `src/nevfe/core/engine.py`).
- **Dual Logit Streams**: Visual-conditioned `logits_plus` (`[B, V]` or `[V]`) and Mean-Field Visual Reference `logits_zero` (`[B, V]` or `[V]`).

## 3. Open-Loop Calibration Bank Observation
- Baseline `148f870` (`IG-VFED`, `lambda_param=0.50`, `beta_param=0.05`, `ablation_mask=(1, 1, 1)`) establishes the campaign anchor (`60.19%` Token F1, `61.46%` Substring Match, `48.86%` ANLS, `31.59%` Exact Match on 1,222 ViTextVQA questions; `42.94%` F1 / `31.43%` Substring on the 70 held-out probe items).

## 4. Numbered Governing Equations & Boundary Conditions
1. **Mean-Field Visual Reference (`V^0`)**: For 3D `[N_tiles, P, D]` (`2D` `[P, D]`), `v_bar = mean(V, dim=(0, 1), keepdim=True)` (`dim=0` for 2D), `V^0 = expand_as(v_bar, V)`, guaranteeing zero spatial gradient.
2. **Simplex Projection & Bhattacharyya Coefficient (`BC_t`)**: Promote logits to `float32`: `p_t^+ = softmax(z_t^+, dim=-1)`, `p_t^0 = softmax(z_t^0, dim=-1)`. Compute `BC_t = clamp(sum_v sqrt(clamp(p_t^+(v) * p_t^0(v), min=0.0)), min=0.0, max=1.0)`.
3. **Fisher-Rao Geodesic Coupling (`alpha_t`)**: If `comp_a` is False, `alpha_t = lambda_param`. Otherwise `alpha_t = lambda_param * sqrt(clamp(1.0 - BC_t^2, min=0.0, max=1.0)) + lambda_param * 1_{step_idx == 1 and comp_c} * BC_t`.
4. **Pointwise Visual Mutual Information Steering**: `z_t^* = z_t^+ + alpha_t * (z_t^+ - z_t^0)`.
5. **Dynamic Geodesic Plausibility Boundary (`beta_t`)**: If `comp_b` is False, `beta_t = beta_param`; else `beta_t = clamp(beta_param * BC_t, min=0.0, max=1.0)`. Retain vocabulary indices `{v : p_t^+(v) >= beta_t * max_w p_t^+(w)}` and mask excluded indices to `-1e9`.
6. **Thermodynamic Shannon Entropy (`H_t`)**: `H_t = -sum_v nan_to_num(p_t^+(v) * log_softmax(z_t^+, dim=-1)(v), 0.0)`.

## 5. `AlgorithmConfig` Parameter Table (`<= 3` Total Tunable Parameters)
| Field Name | Type | Valid Domain | Default | Physical Role |
| :--- | :--- | :--- | :--- | :--- |
| `lambda_param` | `float` | `[0.0, 1.5]` | `0.50` | Base Fisher-Rao geodesic visual coupling strength |
| `beta_param` | `float` | `[0.001, 5.0]` | `0.05` | Dynamic geodesic candidate plausibility truncation floor |

## 6. Full + Leave-One-Out Ablation Mask Table
| Mask (`ablation_mask`) | Active Components | Hypothesis Tested |
| :--- | :--- | :--- |
| `(1, 1, 1)` | Full `IG-VFED` | Fisher-Rao coupling + dynamic `beta_t` + `t=1` preamble boost |
| `(0, 1, 1)` | `comp_a=0` | Static coupling `lambda_param` baseline |
| `(1, 0, 1)` | `comp_b=0` | Static plausibility threshold `beta_param` |
| `(1, 1, 0)` | `comp_c=0` | Iteration-22 `MF-CD` coupling without `t=1` preamble boost |

## 7. Structural Independence Matrix
| BibTeX Key | Bucket [A/B/C/D] | Extracted Principle | Structural Difference vs. Prior SOTA & Other Teams |
| :--- | :--- | :--- | :--- |
| `iter23_ig_vfed_anchor` | `D` | Fisher-Rao geodesic arc `sqrt(1 - BC^2)` + `t=1` preamble echo boost | Baseline campaign anchor (`148f870`) |

## 8. Mechanical Revision Changelog (`v1.0 -> v1.3`)
- `v1.0`: Initial baseline specification matching `swarm/base` (`148f870` `IG-VFED` parity).
