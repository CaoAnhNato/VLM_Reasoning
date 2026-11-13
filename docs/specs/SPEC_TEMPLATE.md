# Specification Template (`spec_iter_<k>.md`)

> **Note**: Do NOT embed a `spec_sha256` field inside this file. The SHA-256 digest is computed externally over the UTF-8 LF-normalized bytes of this file (`sha256(bytes.replace(b"\r\n", b"\n"))`) and bound on line 2 of `src/nevfe/core/engine.py`.

## 1. Header
- **team_id**: `team_<1|2|3>`
- **direction_id**: `direction_<d>`
- **iteration**: `<k>`
- **spec_version**: `v1.0`
- **base_best_commit**: `<team_best_commit_sha>`

## 2. Backbone Tensor Invariants
- **Vision Encoder (`InternViT-300M`)**: Unwarped patch embeddings `vit_embeds` of shape `[N_tiles, 256, 896]` (`bfloat16`), entering LLM Layer 0 unmodified.
- **Language Decoder (`Qwen2.5-0.5B-Instruct`)**: `24` transformer layers, `14` query heads, `2` KV heads (`7:1` GQA group ratio), hidden dimension `896`, dynamic vocabulary size `vocab_size` (e.g., `151,936` on `Vintern-1B-v3_5`; never hardcoded in `src/nevfe/core/engine.py`).
- **Two-Stream Contract**: Positive visual-conditioned logits `logits_plus` (`[B, V]` or `[V]`) and Mean-Field Visual Reference prior logits `logits_zero` (`[B, V]` or `[V]`) computed with zeroed spatial gradient (`V^0 = 1_P \bar{v}^\top`).

## 3. Open-Loop Calibration Bank Observation
- Summarize CPU-only (`CUDA_VISIBLE_DEVICES=""`) open-loop measurements on the 30-sample `data/local_calibration_bank/` (`finite_ratio`, `simplex_kl_mean`, `top1_prior_shift_rate`, `wall_ms`, and `CLASS_A..CLASS_E` observations).

## 4. Numbered Governing Equations & Boundary Conditions
- **(Eq. 1)** Define every tensor operator using explicit shapes, reduction axes, `float32` promotion where required, and `NaN`/`Inf`/zero-denominator boundary conditions.
- **(Eq. 2)** Explicitly state `t = 1` vs. `t > 1` behavior and ablation switch gating (`comp_a`, `comp_b`, `comp_c`).

## 5. `AlgorithmConfig` Parameter Table (`<= 3` Total Tunable Parameters)
| Field Name | Type | Valid Domain | Default | Physical Role |
| :--- | :--- | :--- | :--- | :--- |
| `param_1` | `float` | `[low, high]` | `val` | `...` |

*(Count excludes `ablation_mask: Tuple[int, ...]`. Total tunable fields in `AlgorithmConfig` MUST be `<= 3`.)*

## 6. Full + Leave-One-Out Ablation Mask Table
| Mask (`ablation_mask`) | Active Components | Hypothesis Tested |
| :--- | :--- | :--- |
| `(1, 1, 1)` | Full operator | Combined mechanism |
| `(0, 1, 1)` | Leave-One-Out (`comp_a=0`) | Marginal contribution of Component A |
| `(1, 0, 1)` | Leave-One-Out (`comp_b=0`) | Marginal contribution of Component B |
| `(1, 1, 0)` | Leave-One-Out (`comp_c=0`) | Marginal contribution of Component C |

## 7. Structural Independence Matrix
| BibTeX Key | Bucket [A/B/C/D] | Extracted Principle | Structural Difference vs. Prior SOTA & Other Teams |
| :--- | :--- | :--- | :--- |
| `author2025key` | `D` | `...` | `...` |

## 8. Mechanical Revision Changelog (`v1.0 -> v1.3`)
- `v1.0`: Initial `@critic`-cleared specification.
- `v1.1..v1.3`: Mechanical shape/dtype/`NaN` clarifications triggered by `SPEC_DEFECT_REPORT` (no equation/parameter drift without `@critic` re-approval).
