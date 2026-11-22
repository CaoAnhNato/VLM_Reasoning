# Baseline Specification: Frozen FRSA-PBCF ("011") + IG-VFED (1, 1, 1) (`baseline_frsa011.md`)

## 1. Header
- **track_id**: `base`
- **iteration**: `0`
- **spec_version**: `v1.0`
- **base_best_commit**: `0f8a2d4d` (`4c8ee01`)

## 2. Backbone Tensor Invariants
- **Vision Encoder (`InternViT-300M`)**: Native unwarped patch embeddings `vit_embeds` of shape `[N_tiles, 256, 896]` (`bfloat16`), immutable per image, plus transient single-locus ROI tiles extracted by `FRSA-PBCF`.
- **Language Decoder (`Qwen2.5-0.5B-Instruct`)**: `24` transformer layers, `14` query heads, `2` KV heads (`7:1` GQA ratio), hidden dimension `896`, dynamic `vocab_size` (never hardcoded in `src/nevfe/core/engine.py`).
- **Dual Logit Streams**: Positive visual-conditioned logits `logits_plus` (`[B, V]` or `[V]`) and Mean-Field Visual Reference logits `logits_zero` (`[B, V]` or `[V]`, constructed with `V^0 = 1_P \bar{v}^\top`).

## 3. Baseline Anchor Metrics (`0f8a2d4d` `"011"`)
- **Stage B (1,222 ViTextVQA QIDs)**: `64.12%` Token F1, `62.19%` Substring Match, `63.16%` Composite, `36.99%` Exact Match, `55.55%` ANLS, `1.932 s/Q`.
- **Stage A (100-Q Probe)**: `46.56%` Token F1 (`42.78%` Composite on 100 QIDs), `40.92%` Composite on 70 held-out QIDs (`46.12%` F1, `35.71%` Substring).

## 4. Numbered Governing Equations & Boundary Conditions
1. **Immutable Optical Foveation (`src/nevfe/core/frozen_foveation.py`)**:
   - **Eq. (1) Bregman-Centered Lexical Spatial Distribution (`p_lex`)**: Normalize patch embeddings `v` and prompt-centered Layer-0 query embeddings `q`, compute unit-direction cosine alignment `c_q = max_m <v_hat, e_hat_m>`, standardize `z_q`, and project via `softmax(z_q)` over `[H_g, W_g]`.
   - **Eq. (2) Poisson-Bregman Subwindow Selection**: Maximize `J(C) = p(C) log(p(C)/A(C)) - (p(C) - A(C))` over admissible axis-aligned subwindows `p(C) > A(C) + eps_k`, `eps_k <= A(C) <= 0.5`.
   - **Eq. (3) Mean-Field Visual Reference (`V^0`)**: `V^0 = mean(V, dim=(0, 1), keepdim=True).expand_as(V)`.
2. **Base `IG-VFED (1, 1, 1)` Decoding (`src/nevfe/core/engine.py`)**:
   - Promote logits to `float32`: `p_t^+ = softmax(z_t^+, dim=-1)`, `p_t^0 = softmax(z_t^0, dim=-1)`.
   - Compute Bhattacharyya coefficient `BC_t = clamp(sum_v sqrt(clamp(p_t^+(v) * p_t^0(v), min=0.0)), min=0.0, max=1.0)`.
   - Fisher-Rao geodesic coupling with `t = 1` preamble boost: `alpha_t = lambda_param * sqrt(clamp(1.0 - BC_t^2, min=0.0, max=1.0)) + lambda_param * 1_{step_idx == 1} * BC_t`.
   - Steered potential: `z_t^* = z_t^+ + alpha_t * (z_t^+ - z_t^0)`.
   - Dynamic plausibility truncation: `beta_t = clamp(beta_param * BC_t, min=0.0, max=1.0)`, retaining `{v : p_t^+(v) >= beta_t * max_w p_t^+(w)}` and masking excluded indices to `-1e9`.
3. **Unified Step Context Contract**:
   - `NEVFEEngine.decode_step(logits_plus, logits_zero, q_mask=None, step_idx=1, step_context=None)` accepts optional `step_context` (`history_logits`, `upper_hidden_states`, `rollout_fn`).
   - `VinternNEVFEAdapter.generate()` routes every decoding step through `self.engine.decode_step()`.

## 5. `AlgorithmConfig` Parameter Table (Locked Baseline)
| Field Name | Type | Valid Domain | Default | Physical Role |
| :--- | :--- | :--- | :--- | :--- |
| `lambda_param` | `float` | `== 0.50` (locked) | `0.50` | Base Fisher-Rao geodesic visual coupling strength |
| `beta_param` | `float` | `== 0.05` (locked) | `0.05` | Dynamic geodesic candidate plausibility truncation floor |

*(Track iterations may add `<= 2` cumulative new tunable scalars; `COMPONENT_NAMES = ()`, `ablation_mask = ()`.)*
