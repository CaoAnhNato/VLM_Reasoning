# TASK 15 — INDEPENDENT ACCEPTANCE QA: TICKET-OPT-04 (GPU AFFINITY FIELD VECTORIZATION)

- **Independent QA Reviewer:** `Antigravity (Anti)` — Senior ML Systems Reviewer
- **Implementer:** `4argon` — Task 14
- **Methodology Reviewer:** `ChatGPT`
- **Final Approval Authority:** `Project Owner`
- **Ticket:** `TICKET-OPT-04 — GPU Affinity Field Vectorization`
- **QA Mode:** Strict read-only independent acceptance — **Gate A only**
- **Date:** `2026-11-09`

---

## 1. Executive Summary & Final Acceptance Verdict

- **Final Acceptance Verdict:** **`PASS WITH NON-BLOCKING FINDINGS`**
- **Gate A Technical Status:** **`ACCEPTED`** (All Gate A numerical, spatial localization, cache lifecycle, and regression criteria are fully met; Gate B / Colab / real Vintern model execution remains `NOT AUTHORIZED` and was not executed).
- **Core Verification Summary:**
  1. **Numerical Parity Criterion ($\max_i |\Phi_{\mathrm{PyTorch},i} - \Phi_{\mathrm{NumPy},i}| \le 10^{-4}$):** **`VERIFIED`**. Across all 10 synthetic attention and feature-grid regimes, the worst-case absolute error was $6.557 \times 10^{-7}$, well within the mandatory $1.0 \times 10^{-4}$ bound ($>150\times$ margin). Across 15 degenerate boundary and edge cases, max absolute difference was $\le 1.192 \times 10^{-7}$.
  2. **Stage 1 Downstream Localization Parity:** **`VERIFIED`**. Both [`KLDivergenceRegionSelector.select_top_k()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/bbox_selector.py#L357-L561) and downstream [`AdaptiveScalePropagator.propagate()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/adaptive_propagator.py#L388-L510) produced **`100.0% IoU` (`1.0`)** and identical bounding-box coordinates between CPU NumPy and GPU PyTorch outputs across all 10 regimes.
  3. **No-Wrap-Around Boundary Semantics:** **`VERIFIED`**. Opposing border activations (top/bottom, left/right, and opposite corners) produced `0.0` bridging affinity, matching the CPU reference.
  4. **Device & No-CPU-Round-Trip Contract:** **`VERIFIED`**. The 3D visual feature grid `feat_grid` remains on its native device without host transfers (`.cpu()` or `.numpy()` calls on `feat_grid` are completely eliminated). Only the final 2D $(H_g, W_g)$ scalar maps `phi_map` and `aff_map` ($\sim 4\text{ KB}$ each) are converted to NumPy for [`ExtractionResult`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/attention_extractor.py#L56-L64).
  5. **Independent Test Execution:** **`VERIFIED`**. All 7 required test suites plus the mandatory Tier-4 anti-hardcoding audit executed cleanly with zero failures (**`99/99 tests PASS`**, exit code `0`).

---

## 2. Mandatory Governance Files Verified

The following binding governance documents and technical sources were read and verified:
1. `AGENTS.md` (verified absent at repository root).
2. [`.github/copilot-instructions.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/.github/copilot-instructions.md)
3. [`.github/skills/anti-output-quality-review/SKILL.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/.github/skills/anti-output-quality-review/SKILL.md)
4. [`.github/skills/research-plan-compliance-review/SKILL.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/.github/skills/research-plan-compliance-review/SKILL.md)
5. [`.github/skills/local-colab-experiment-workflow/SKILL.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/.github/skills/local-colab-experiment-workflow/SKILL.md)
6. [`docs/research/PERFORMANCE_OPTIMIZATION_HANDOFF_v1.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/PERFORMANCE_OPTIMIZATION_HANDOFF_v1.md)
7. [`docs/Proposed_Method_SG_VCD.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/Proposed_Method_SG_VCD.md)
8. [`docs/research/REMEDIATION_SPEC_P0_v1.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/REMEDIATION_SPEC_P0_v1.md)
9. [`docs/research/TASK_07_FIX_OPT06_INDEPENDENT_ACCEPTANCE_QA.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/TASK_07_FIX_OPT06_INDEPENDENT_ACCEPTANCE_QA.md)
10. [`docs/research/TASK_08_OPT03_WORKING_MAP_CACHE_IMPLEMENTATION_REPORT.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/TASK_08_OPT03_WORKING_MAP_CACHE_IMPLEMENTATION_REPORT.md)
11. [`docs/research/TASK_08_OPT03_INDEPENDENT_ACCEPTANCE_QA.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/TASK_08_OPT03_INDEPENDENT_ACCEPTANCE_QA.md)
12. [`docs/research/TASK_09_OPT02_MEMORY_CLEANUP_IMPLEMENTATION_REPORT.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/TASK_09_OPT02_MEMORY_CLEANUP_IMPLEMENTATION_REPORT.md)
13. [`docs/research/TASK_10_OPT02_INDEPENDENT_ACCEPTANCE_QA.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/TASK_10_OPT02_INDEPENDENT_ACCEPTANCE_QA.md)
14. [`docs/research/TASK_11_OPT01_BATCHED_DECODING_IMPLEMENTATION_REPORT.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/TASK_11_OPT01_BATCHED_DECODING_IMPLEMENTATION_REPORT.md)
15. [`docs/research/TASK_12_OPT01_INDEPENDENT_ACCEPTANCE_QA.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/TASK_12_OPT01_INDEPENDENT_ACCEPTANCE_QA.md)
16. [`docs/research/TASK_13_OPT01_MASK_SEMANTICS_QA_ADDENDUM.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/TASK_13_OPT01_MASK_SEMANTICS_QA_ADDENDUM.md)
17. [`docs/research/TASK_14_OPT04_GPU_AFFINITY_VECTORIZATION_IMPLEMENTATION_REPORT.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/TASK_14_OPT04_GPU_AFFINITY_VECTORIZATION_IMPLEMENTATION_REPORT.md)

---

## 3. Audit A — Git Status, Diff & Ticket Scope Isolation

Independent inspection of working tree via `git status --short` and `git diff`:

```
 M scripts/benchmark/evaluate_vitextvqa.py
 M scripts/benchmark/sg_vcd/__init__.py
 M scripts/benchmark/sg_vcd/adaptive_propagator.py
 M scripts/benchmark/sg_vcd/attention_extractor.py
 M scripts/benchmark/sg_vcd/feature_cache.py
 M scripts/benchmark/sg_vcd/pipeline.py
?? .github/
?? docs/SG_VCD_Research_Review_Consolidated_Draft.md
?? docs/research/
?? tests/test_opt01_batched_dual_branch_decoding.py
?? tests/test_opt02_memory_cleanup_policy.py
?? tests/test_opt03_working_map_cache.py
?? tests/test_opt04_gpu_affinity_vectorization.py
?? tests/test_opt06_image_cache_identity.py
```

### Scope Isolation Findings:
- **Modified Production Files attributable to OPT-04:** Strictly [`scripts/benchmark/sg_vcd/attention_extractor.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/attention_extractor.py) (`+220 / -23` lines).
- **New Test Files attributable to OPT-04:** Strictly [`tests/test_opt04_gpu_affinity_vectorization.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt04_gpu_affinity_vectorization.py).
- **Documentation attributable to OPT-04:** Strictly [`docs/research/TASK_14_OPT04_GPU_AFFINITY_VECTORIZATION_IMPLEMENTATION_REPORT.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/TASK_14_OPT04_GPU_AFFINITY_VECTORIZATION_IMPLEMENTATION_REPORT.md).
- **Untouched Critical Components:**
  - [`pipeline.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py) and [`dynamic_cd.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/dynamic_cd.py): Zero modifications in Task 14.
  - Stage 1 mathematical formula: Intrinsic affinity formula $\Phi(u) = \max(R(u), M_{\mathrm{aff}}(u))$ is mathematically identical.
  - Stage 1 region selector ([`bbox_selector.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/bbox_selector.py)), ROI cropper ([`roi_cropper.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/roi_cropper.py)), seed propagator ([`adaptive_propagator.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/adaptive_propagator.py)), and metrics: Zero modifications in Task 14.
  - No unrelated refactoring or scope creep was detected.
  - Git working tree remained strictly untouched and uncommitted.

---

## 4. Audit B — CPU NumPy Reference Fidelity

Independent comparison between the original implementation in `HEAD:scripts/benchmark/sg_vcd/attention_extractor.py` and the preserved reference [`_compute_intrinsic_affinity_field_numpy_reference()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/attention_extractor.py#L93-L199):

| Aspect | `HEAD` Original `compute_intrinsic_affinity_field` | Preserved `_compute_intrinsic_affinity_field_numpy_reference` | Fidelity Assessment |
| :--- | :--- | :--- | :---: |
| Input type conversion | `r = np.maximum(np.asarray(r_map, dtype=np.float32), 0.0)` | Adds defensive PyTorch `.detach().cpu().numpy()` conversion if input is a tensor, else `np.asarray` | **Exact** |
| Feature L2 normalization | `norms = np.linalg.norm(v, axis=-1, keepdims=True); v / np.maximum(norms, 1e-8)` | Identical L2 norm along `axis=-1` with `1e-8` clamp | **Exact** |
| Query normalization | `aq / aq_max if aq_max > 1e-12 else r.copy()` | Identical with safe guard `float(aq.max()) if aq.size > 0 else 0.0` | **Exact** |
| Lattice directions | `[(0, 1), (1, 0), (1, 1), (1, -1)]` | Identical 4 lattice directions | **Exact** |
| Bridge radii & slicing | `for k in range(1, max_bridge_radius + 1): ky, kx = k * dy, k * dx ...` | Identical interior slice bounding `y_start:y_end, x_start:x_end` | **Exact** |
| Bilateral Gap Affinity | `geom_mean * cos_pos` with `geom_mean = sqrt(max(0, r_minus * r_plus))` | Identical geometric mean and clamped dot product | **Exact** |
| Stroke continuation | `cont_score = sqrt(max(0, r_neigh * a_center)) * (cos_c ** 2)` on `_pass == 0` | Identical geodesic continuation on Pass 0 only | **Exact** |
| 2-Pass update order | `aff_map = np.maximum(aff_map, pass_aff); current_r = np.maximum(current_r, pass_aff)` | Identical update order across the 2 passes | **Exact** |
| Final normalization | `phi_map = max(r, aff_map); phi_map / max_phi if max_phi > 1e-12` | Identical maximum and normalization | **Exact** |

**Conclusion on Audit B:** The CPU NumPy reference is **100% faithful** to the original baseline algorithm.

---

## 5. Audit C — Vectorized PyTorch Numerical Implementation

Line-by-line inspection of [`compute_intrinsic_affinity_field_gpu()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/attention_extractor.py#L202-L343):

1. **Feature L2 Normalization:**
   ```python
   norms = torch.linalg.vector_norm(v, ord=2, dim=-1, keepdim=True)
   v_norm = v / torch.clamp(norms, min=1e-8)
   ```
   Uses exact L2 vector norm along channel dimension `dim=-1` with minimum clamp `1e-8`, matching NumPy `np.linalg.norm(v, axis=-1)`.
2. **Directional Shifts & Padding:**
   - Bilateral Gap Affinity (Part A): Pads `v_norm` with `(0, 0, radius, radius, radius, radius)` and slices offsets $(k_y, k_x)$ into stacked tensors `v_minus_a` and `v_plus_a` of shape $(4K, H, W, D)$.
   - Geodesic Stroke Continuation (Part B): Pads `v_norm` with `(0, 0, 1, 1, 1, 1)` and slices offsets $(s \cdot d_y, s \cdot d_x)$ into stacked tensor `v_neigh_b` of shape $(8, H, W, D)$.
3. **Interior Coordinate Validity Masks (`valid_mask_a` and `valid_mask_b`):**
   - For each offset $(k_y, k_x)$, `valid_mask_a[idx, y_start:y_end, x_start:x_end] = 1.0` where $y_{\text{start}} = |k_y|, y_{\text{end}} = H - |k_y|, x_{\text{start}} = |k_x|, x_{\text{end}} = W - |k_x|$, and `0.0` everywhere else.
   - For each continuation offset $(d_y, d_x)$, `valid_mask_b[idx, y_s:y_e, x_s:x_e] = 1.0`.
   - This explicitly zeroes out out-of-bounds positions, strictly preventing circular wrap-around or boundary leakage without using `torch.roll`.
4. **Precomputations & Reductions:**
   - Directional feature cosine products `cos_pos_a` and `cos_c_sq_b` are precomputed once outside the 2-pass loop, avoiding redundant matrix multiplications during Pass 1.
   - Reductions across the stacked directional offsets use `torch.amax(..., dim=0)`, matching NumPy `np.maximum.reduce`.
5. **Pass 0 vs Pass 1 Execution Semantics:**
   - Pass 0 executes both Part A and Part B.
   - Pass 1 executes Part A only (`if v_norm is not None and _pass == 0 and cos_c_sq_b is not None:`).
   - In both passes, `aff_map = torch.maximum(aff_map, pass_aff)` and `current_r = torch.maximum(current_r, pass_aff)` in exact lockstep with the NumPy reference.

---

## 6. Audit D — Device, Dtype & Actual GPU Execution Status

1. **Device Resolution:**
   ```python
   if isinstance(feat_grid, torch.Tensor):
       device = feat_grid.device
   elif isinstance(r_map, torch.Tensor):
       device = r_map.device
   elif isinstance(a_query_map, torch.Tensor):
       device = a_query_map.device
   else:
       device = torch.device("cpu")
   ```
   Correctly prioritizes the 3D feature tensor device.
2. **Intermediate Promotion:** Inputs are promoted to `torch.float32` on `device` via `.to(device=device, dtype=torch.float32)`.
3. **No 3D Host Transfer:**
   - In [`tests/test_opt04_gpu_affinity_vectorization.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt04_gpu_affinity_vectorization.py), `_NoCpuTransferTensor` was subclassed to raise `AssertionError` if `.cpu()` or `.numpy()` was invoked on any 3D tensor (`ndim == 3`). `test_b_device_and_dtype_contract_verification` and `test_f_public_api_contract_and_no_feat_grid_cpu_roundtrip` both passed, verifying zero host round-trips for the 3D feature grid.
4. **Actual Runtime Environment Checked Independently:**
   - Python: `3.11.9 (tags/v3.11.9:de54cf5, Apr 2 2024, 10:12:12) [MSC v.1938 64 bit (AMD64)]`
   - PyTorch: `2.14.1+cpu`
   - CUDA available: `False`
   - Transformers: `4.57.6`
   - **CRITICAL TRANSPARENCY NOTICE (INFORMATIONAL FINDING INF-01):**
     In this local developer environment, PyTorch is installed as a CPU build (`2.14.1+cpu`), meaning `torch.cuda.is_available() == False`. Therefore, all local tests executed using PyTorch CPU tensor operations.
     **Actual CUDA GPU speedup and VRAM reduction are formally classified as `NOT_MEASURED` at Gate A.** Vectorized tensor arithmetic, numerical parity, and device independence are verified, but hardware speedup must be validated on an authorized CUDA GPU instance at Gate B.

---

## 7. Audit E — Boundary, Degenerate & Edge Case Verification

Independent test script executed over 15 degenerate configurations:

| # | Case | Shape $(H, W, D)$ | Max $|\Phi_{\mathrm{GPU}} - \Phi_{\mathrm{CPU}}|$ | Max $|M_{\mathrm{aff,GPU}} - M_{\mathrm{aff,CPU}}|$ | Has NaN | Status |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | `1x1` single cell | `(1, 1, 16)` | `0.000e+00` | `0.000e+00` | `False` | **PASS** |
| 2 | `2x2` minimum 2D grid | `(2, 2, 16)` | `0.000e+00` | `0.000e+00` | `False` | **PASS** |
| 3 | `3x3` small grid | `(3, 3, 16)` | `2.980e-08` | `2.980e-08` | `False` | **PASS** |
| 4 | `1x10` horizontal sliver | `(1, 10, 16)` | `0.000e+00` | `2.980e-08` | `False` | **PASS** |
| 5 | `10x1` vertical sliver | `(10, 1, 16)` | `0.000e+00` | `2.980e-08` | `False` | **PASS** |
| 6 | `5x19` odd non-square | `(5, 19, 32)` | `2.980e-08` | `2.980e-08` | `False` | **PASS** |
| 7 | `18x6` even non-square | `(18, 6, 32)` | `5.960e-08` | `5.960e-08` | `False` | **PASS** |
| 8 | `zero_r` (all zeros) | `(16, 16, 16)` | `0.000e+00` | `0.000e+00` | `False` | **PASS** |
| 9 | `const_feat` (ones) | `(16, 16, 16)` | `1.192e-07` | `1.192e-07` | `False` | **PASS** |
| 10 | `zero_norm_feat` (zero vector) | `(16, 16, 16)` | `7.451e-08` | `7.451e-08` | `False` | **PASS** |
| 11 | `negative_inputs` | `(16, 16, 16)` | `7.451e-08` | `7.451e-08` | `False` | **PASS** |
| 12 | `feat_None` (no features) | `(16, 16, 0)` | `1.192e-07` | `1.192e-07` | `False` | **PASS** |
| 13 | `aq_None` (no query map) | `(16, 16, 16)` | `5.960e-08` | `7.451e-08` | `False` | **PASS** |
| 14 | `radius_0` ($K=0$) | `(16, 16, 16)` | `7.451e-08` | `8.941e-08` | `False` | **PASS** |
| 15 | `radius_4` ($K=4$) | `(16, 16, 16)` | `7.451e-08` | `7.451e-08` | `False` | **PASS** |

All degenerate cases handled gracefully with zero division-by-zero, zero NaN/Inf values, and maximum error $\le 1.192 \times 10^{-7}$.

---

## 8. Audit F — Required Ten-Regime Coverage Matrix

The 10 regimes tested in [`tests/test_opt04_gpu_affinity_vectorization.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt04_gpu_affinity_vectorization.py) were mapped against the required regime archetypes from Section 9 of the mandate:

| Required Mandate Archetype | Implemented Regime in Test Suite | Verified Properties | Status |
| :--- | :--- | :--- | :---: |
| 1. Random normalized feature grid | `01_single_compact_text_block_16x16_D64` | Gaussian random noise + prototype modulation | **COVERED** |
| 2. Multi-word horizontal gaps | `02_multi_word_horizontal_gaps_16x32_D64` | 1-patch & 2-patch horizontal gaps across words | **COVERED** |
| 3. Multi-line vertical stack | `03_multi_line_vertical_stack_32x16_D64` | 1-patch & 2-patch row gaps across text lines | **COVERED** |
| 4. Diagonal stroke pattern | `04_diagonal_stroke_tilted_signboard_32x32_D128` | Direction $(1, 1)$ tilted signboard gaps | **COVERED** |
| 5. Anti-diagonal continuation | `05_antidiagonal_stroke_tail_continuation_32x32_D128` | Direction $(1, -1)$ geodesic character tails | **COVERED** |
| 6. Border/corner-touching tokens | `06_border_corner_touching_tokens_16x16_D64` | Activations on outer perimeter & corners | **COVERED** |
| 7. Multi-candidate disjoint regions | `07_multi_candidate_disjoint_regions_32x32_D128` | Separated text islands ($K=3$) | **COVERED** |
| 8. Feature-free mode (`feat_grid=None`) | `08_feature_free_mode_24x24_None` | Pure attention-gap bridging without ViT feats | **COVERED** |
| 9. Degenerate zero / near-zero map | `09_degenerate_zero_map_16x16_D32` | All-zero input maps, zero division safety | **COVERED** |
| 10. Realistic Vintern patch-grid scale | `10_realistic_vintern_grid_32x48_D896_bfloat16` | Realistic $32 \times 48$, $D=896$, `bfloat16` promotion | **COVERED** |

Additional edge archetypes (constant feature grid, zero-norm feature vectors, odd non-square dimensions, small spatial grids $1\times 1, 2\times 2, 3\times 3$) are explicitly verified in `test_g_edge_cases_zero_none_tiny_and_nonsquare_grids`.

---

## 9. Audit G — Stage 1 End-to-End Localization Parity

Independent end-to-end execution of both Stage 1 components across all 10 synthetic regimes:
1. Region Selector: [`KLDivergenceRegionSelector.select_top_k()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/bbox_selector.py#L357-L561)
2. Seed Propagator: [`AdaptiveScalePropagator.propagate()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/adaptive_propagator.py#L388-L510)

| # | Synthetic Regime | Selector BBoxes (GPU vs CPU) | Propagator `propagated_pixel_bbox` (GPU vs CPU) | Stage 1 BBox IoU |
| :-: | :--- | :---: | :---: | :---: |
| 1 | `01_single_compact_text_block_16x16_D64` | `[(95, 135, 265, 225)]` == `[(95, 135, 265, 225)]` | `(67, 102, 253, 222)` == `(67, 102, 253, 222)` | **`100.0%` (`1.0`)** |
| 2 | `02_multi_word_horizontal_gaps_16x32_D64` | `[(115, 95, 325, 205), (315, 95, 505, 205)]` == Identical | `(145, 221, 156, 236)` == `(145, 221, 156, 236)` | **`100.0%` (`1.0`)** |
| 3 | `03_multi_line_vertical_stack_32x16_D64` | `[(55, 55, 265, 245), (55, 235, 265, 365)]` == Identical | `(66, 66, 254, 355)` == `(66, 66, 254, 355)` | **`100.0%` (`1.0`)** |
| 4 | `04_diagonal_stroke_tilted_signboard_32x32_D128` | `[(60, 60, 308, 308), (284, 284, 436, 420)]` == Identical | `(121, 121, 248, 232)` == `(121, 121, 248, 232)` | **`100.0%` (`1.0`)** |
| 5 | `05_antidiagonal_stroke_tail_continuation_32x32_D128` | `[(140, 204, 308, 372), (284, 92, 420, 228)]` == Identical | `(248, 180, 345, 262)` == `(248, 180, 345, 262)` | **`100.0%` (`1.0`)** |
| 6 | `06_border_corner_touching_tokens_16x16_D64` | `[(0, 0, 125, 45), (195, 275, 320, 320)]` == Identical | `(5, 41, 16, 56)` == `(5, 41, 16, 56)` | **`100.0%` (`1.0`)** |
| 7 | `07_multi_candidate_disjoint_regions_32x32_D128` | `[(300, 188, 468, 292), (60, 364, 228, 484)]` == Identical | `(244, 385, 253, 396)` == `(244, 385, 253, 396)` | **`100.0%` (`1.0`)** |
| 8 | `08_feature_free_mode_24x24_None` | `[(215, 135, 365, 265), (95, 135, 225, 285)]` == Identical | `(310, 426, 321, 441)` == `(310, 426, 321, 441)` | **`100.0%` (`1.0`)** |
| 9 | `09_degenerate_zero_map_16x16_D32` | `[(0, 0, 320, 320)]` == `[(0, 0, 320, 320)]` | `(166, 163, 175, 174)` == `(166, 163, 175, 174)` | **`100.0%` (`1.0`)** |
| 10 | `10_realistic_vintern_grid_32x48_D896_bfloat16` | `[(375, 175, 665, 325), (135, 175, 385, 325)]` == Identical | `(243, 517, 260, 540)` == `(243, 517, 260, 540)` | **`100.0%` (`1.0`)** |

**Conclusion on Audit G:** Both the candidate bounding boxes from the region selector and the final refined bounding boxes from the adaptive seed propagator match with **exact coordinate equality and 100.0% IoU**. Full Stage 1 downstream parity is **`VERIFIED`**.

---

## 10. Audit H — Extractor Integration & Cache Lifecycle

1. **Ordering & Reshaping:**
   - In [`PointwiseKLAttentionExtractor.extract()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/attention_extractor.py#L943-L985):
     ```python
     grid_feats = (
         vit_embeds[:num_grid_tiles]
         .detach()
         .float()
         .reshape(gy, gx, 16, 16, -1)
         .permute(0, 2, 1, 3, 4)
         .reshape(H_g, W_g, -1)
         .contiguous()
     )
     ```
     Correct tile-to-patch spatial ordering $(g_y, g_x, 16, 16, D) \to (g_y, 16, g_x, 16, D) \to (16 g_y, 16 g_x, D)$ is preserved.
2. **Tensor Detachment & Graph Safety:** Calling `.detach()` before storing in `self._cached_img_state["grid_feats"]` ensures no computational history or autograd graph is held.
3. **Multi-Question Reuse:** Verified via `test_h_extractor_mock_integration_and_cache_lifecycle` that for questions $Q_1$ and $Q_2$ on image `img_001`, `mock_model.extract_feature` is invoked exactly once (`extract_calls == 1`), and `extractor._cached_img_state["grid_feats"]` is reused by identity (`is`).
4. **Cache Invalidation & Boundary Release:** Calling [`extractor.clear_cache()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/attention_extractor.py#L512-L516) sets both `_cached_img_state` and `_cached_img_key` to `None`, releasing all references to `grid_feats`.

---

## 11. Audit I — Existing Optimization Preservation & Test Execution Audit

### Independent Test Execution Results

All 7 required test suites plus the mandatory Tier-4 anti-hardcoding audit were independently executed using interpreter `..\Document AI\VLM_Reasoning\venv\Scripts\python.exe`:

| Test Suite File | Command | Tests Run | Time | Exit Code | Raw Outcome |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `tests.test_opt04_gpu_affinity_vectorization` | `python -m unittest tests.test_opt04_gpu_affinity_vectorization -v` | **8** | 4.273s | 0 | **`OK (8/8 PASS)`** |
| `tests.test_opt01_batched_dual_branch_decoding` | `python -m unittest tests.test_opt01_batched_dual_branch_decoding -v` | **8** | 5.708s | 0 | **`OK (8/8 PASS)`** |
| `tests.test_opt02_memory_cleanup_policy` | `python -m unittest tests.test_opt02_memory_cleanup_policy -v` | **9** | 9.388s | 0 | **`OK (9/9 PASS)`** |
| `tests.test_opt03_working_map_cache` | `python -m unittest tests.test_opt03_working_map_cache -v` | **8** | 5.585s | 0 | **`OK (8/8 PASS)`** |
| `tests.test_opt06_image_cache_identity` | `python -m unittest tests.test_opt06_image_cache_identity -v` | **11** | 1.832s | 0 | **`OK (11/11 PASS)`** |
| `tests.test_sg_vcd` & `tests.test_adversarial_spatial_processing` | `python -m unittest tests.test_sg_vcd tests.test_adversarial_spatial_processing -v` | **54** | 0.303s | 0 | **`OK (54/54 PASS)`** |
| `tests.test_tier4_workloads...test_05` | `python -m unittest tests.test_tier4_workloads.TestTier4RealWorldWorkloads.test_05_anti_hardcoding_and_heuristics_audit -v` | **1** | 0.176s | 0 | **`OK (1/1 PASS)`** |
| **Complete Aggregate** | **All Required Suites** | **`99`** | **27.265s** | **0** | **`PASS (99/99)`** |

### Test Summary Correction (Non-Blocking Finding NBF-01):
Task 14 reported a total of `98/98 tests PASS`. That summary omitted the mandatory `test_05_anti_hardcoding_and_heuristics_audit` required by the QA mandate. Independent execution of the complete set of required suites yields **`99/99 tests PASS`**.

---

## 12. Complete Findings Inventory

### Blocker Findings:
**None.** All mathematical, boundary, localization, and regression criteria are fully satisfied.

### Non-Blocking Findings:
1. **NBF-01 — Task 14 Report Aggregate Test Count Omission:**
   - *Evidence:* Section 11 of Task 14 report omitted `test_05_anti_hardcoding_and_heuristics_audit`, reporting 98 tests instead of the full 99 tests.
   - *Remediation:* Reconciled above. All 99 required tests are confirmed `PASS`.

### Informational Findings:
1. **INF-01 — Local Environment PyTorch Build & Unmeasured CUDA Speedup:**
   - *Evidence:* Local virtual environment runs `PyTorch 2.14.1+cpu` (`torch.cuda.is_available() == False`).
   - *Assessment:* All synthetic tests executed on CPU tensors. Vectorized tensor correctness is verified, but actual CUDA wall-clock speedup and peak VRAM impact remain `NOT_MEASURED` at Gate A and must be evaluated on CUDA hardware at Gate B.
2. **INF-02 — Downstream Propagator Dtype Casting:**
   - *Evidence:* Downstream `AdaptiveScalePropagator.propagate(..., feat_grid)` expects `np.float32` numpy array or `float32` tensor. If a caller passes raw `bfloat16` directly into `propagate`, PyTorch color transform requires `.float()` promotion.
   - *Assessment:* Not an issue in production pipeline where `propagate` takes `extraction.phi_map` (already `np.float32`), but noteworthy for direct callers.

---

## 13. Final Acceptance Conclusion

- **Gate A Technical Acceptance:** **`PASS WITH NON-BLOCKING FINDINGS`**
- **Gate B Authorization Status:** **`NOT AUTHORIZED`** (Pending explicit Project Owner sign-off).
- All criteria for `TICKET-OPT-04` under Gate A are satisfied. Ready for Project Owner review.

