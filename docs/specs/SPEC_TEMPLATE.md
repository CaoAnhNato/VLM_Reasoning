# Specification Template (`spec_iter_<k>.md`)

> **Note**: Do NOT embed a `spec_sha256` field inside this file. The SHA-256 digest is computed externally over the UTF-8 LF-normalized bytes of this file (`sha256(bytes.replace(b"\r\n", b"\n"))`) and bound on line 2 of `src/nevfe/core/engine.py`.

## 1. Header
- **track_id**: `track_<1|2>`
- **iteration**: `<k>`
- **spec_version**: `v1.0`
- **base_best_commit**: `<track_best_commit_sha>`

## 2. Backbone Tensor Invariants
- **Vision Encoder (`InternViT-300M`)**: Unwarped patch embeddings `vit_embeds` of shape `[N_tiles, 256, 896]` (`bfloat16`), paired with immutable `FRSA-PBCF` ("011") foveation in `src/nevfe/core/frozen_foveation.py`.
- **Language Decoder (`Qwen2.5-0.5B-Instruct`)**: `24` transformer layers, `14` query heads, `2` KV heads (`7:1` GQA group ratio), hidden dimension `896`, dynamic vocabulary size `vocab_size` (never hardcoded in `src/nevfe/core/engine.py`). Track 2 upper-layer introspection uses relative depth `[0.5, 1.0)` with zero KV perturbation (`max_kv_diff == 0.0`) and single-step decoding (`1` decoder forward call per stream per step).
- **Two-Stream Contract**: Positive visual-conditioned logits `logits_plus` (`[B, V]` or `[V]`) and Mean-Field Visual Reference prior logits `logits_zero` (`[B, V]` or `[V]`), evaluated at each step via `NEVFEEngine.decode_step(..., step_context=...)`.

## 3. Calibration Bank v2 & Forensic Observation
- Summarize CPU-only (`CUDA_VISIBLE_DEVICES=""`) diagnostics on the 30-sample `data/calibration_bank_v2/` (`finite_ratio`, `simplex_kl_mean`, `top1_prior_shift_rate`, `max_kv_diff`, `wall_ms`) and the latest `forensic_delta.md` probe-subset `CLASS_A..CLASS_E` observations.

## 4. Numbered Governing Equations & Boundary Conditions
- **(Eq. 1)** Define every new tensor operator using explicit shapes, reduction axes, `float32` promotion where required, and `NaN`/`Inf`/zero-denominator boundary conditions on top of locked base `IG-VFED (1, 1, 1)`.
- **(Eq. 2)** Explicitly state `t = 1` vs. `t > 1` behavior and named component ablation switch gating (`COMPONENT_NAMES`).

## 5. `AlgorithmConfig` Parameter Table (`lambda_param=0.50`, `beta_param=0.05` Locked + `<= 2` Cumulative New Scalars)
| Field Name | Type | Valid Domain | Default | Physical Role |
| :--- | :--- | :--- | :--- | :--- |
| `lambda_param` | `float` | `== 0.50` (locked) | `0.50` | Frozen base geodesic coupling |
| `beta_param` | `float` | `== 0.05` (locked) | `0.05` | Frozen base plausibility truncation floor |
| `<new_param_1>` | `float` | `[low, high]` | `val` | `<First adaptive or scalar field (<= 2 cumulative new scalars)>` |

*(Count excludes `ablation_mask: Tuple[int, ...]`. Total scalar fields in `AlgorithmConfig` MUST be `<= 4`, i.e., locked `lambda_param`, `beta_param`, and `<= 2` cumulative new scalars.)*

## 6. Named Component & Leave-One-Out Ablation Mask Table
- `COMPONENT_NAMES = ("<comp_1>", ...)` (`M = len(COMPONENT_NAMES)` components).

| Compact Bitstring (`ablation_mask`) | Active Components | Hypothesis Tested |
| :--- | :--- | :--- |
| `base` (when `M = 0`) or `"0"*M` | Frozen base `IG-VFED` | Anchor baseline |
| `"1"*M` | Full candidate (`M` components) | Combined mechanism vs. `best_probe_70_comp` |
| Proper sub-masks (e.g., `"01"`, `"10"`) | Leave-One-Out sub-masks | Net marginal contribution of each component |

## 7. Verified Core Paper Provenance (`core_paper_verification.json`)
| Source ID | Source Type (`VERIFIED_TEX` / `VERIFIED_PDF` / `WAIVER`) | SHA-256 | Section & Equation Anchor | Principle Borrowed |
| :--- | :--- | :--- | :--- | :--- |
| `<arxiv_id_or_doi>` | `VERIFIED_TEX` | `<sha256>` | `Sec. X, Eq. (Y)` | `...` |

*(Must list `1–3` verified core papers driving the specification, validated by `track_ops.py papers verify`.)*

## 8. Mechanical Revision Changelog (`v1.0 -> v1.3`)
- `v1.0`: Initial track iteration specification derived from verified core paper primitives.
- `v1.1..v1.3`: Mechanical shape/dtype/`NaN`/audit repairs inside `IMPLEMENT_AUDIT` (no equation/parameter drift).
