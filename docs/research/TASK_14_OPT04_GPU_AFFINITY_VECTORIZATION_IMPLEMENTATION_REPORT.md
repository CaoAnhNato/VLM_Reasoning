# TASK 14 — IMPLEMENTATION REPORT: TICKET-OPT-04 (GPU AFFINITY FIELD VECTORIZATION)

- **Ticket:** `TICKET-OPT-04 — GPU Affinity Field Vectorization`
- **Implementer:** `4argon` — Senior ML Systems Engineer
- **Independent QA:** `Antigravity (Anti)` — Task 15
- **Methodology Reviewer:** `ChatGPT`
- **Final Approval Authority:** `Project Owner`
- **Priority:** `P2` — Numerical and Spatial Localization Parity Required
- **Execution Mode:** Local synthetic/mock only — **Gate A**
- **Date:** `2026-11-09`

---

## 1. Executive Summary & Gate A Self-Assessment Verdict

- **Gate A Self-Assessment Verdict:** **`PASS — READY FOR TASK 15 INDEPENDENT QA`**
- **Summary of Implementation:**
  1. Preserved the exact original CPU NumPy implementation of the SG-VCD Stage 1 Intrinsic Affinity Field in [`_compute_intrinsic_affinity_field_numpy_reference()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/attention_extractor.py#L93-L199) inside [`scripts/benchmark/sg_vcd/attention_extractor.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/attention_extractor.py).
  2. Implemented [`compute_intrinsic_affinity_field_gpu()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/attention_extractor.py#L202-L343), a vectorized PyTorch implementation that keeps the $H \times W \times D$ visual feature grid (`feat_grid`) on its target device in `torch.float32`, eliminates all `.cpu().numpy()` round-trips on `feat_grid`, and computes both passes of **Part A (Bilateral Gap Affinity)** and **Part B (Geodesic Stroke-Continuation Affinity)** via constant zero-padded shifted tensor views (`F.pad` + `torch.stack` + `torch.amax`) masked to the exact interior coordinate bounds `[y_start:y_end, x_start:x_end]` of the CPU reference.
  3. Updated [`compute_intrinsic_affinity_field()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/attention_extractor.py#L346-L384) to delegate to [`compute_intrinsic_affinity_field_gpu()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/attention_extractor.py#L202-L343) and convert only the final 2D $(H, W)$ maps (`phi_map`, `aff_map`) to `np.float32` at the [`ExtractionResult`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/attention_extractor.py#L56-L64) boundary.
  4. Updated [`PointwiseKLAttentionExtractor.extract()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/attention_extractor.py#L943-L985) so `grid_feats` remains a detached contiguous `torch.float32` tensor on device inside `self._cached_img_state["grid_feats"]` across same-image multi-question reuse, releasing cleanly on [`clear_cache()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/attention_extractor.py#L709-L712) or image transitions.
  5. Created [`tests/test_opt04_gpu_affinity_vectorization.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt04_gpu_affinity_vectorization.py) covering 10 deterministic synthetic attention/feature-grid regimes and `Tests A–H` (`8/8 PASS`), and verified `98/98 PASS` across all 7 required Gate A test suites.

---

## 2. Governance & Technical Context Loaded

All mandatory governance and technical specifications were loaded and verified prior to implementation:
1. `AGENTS.md` (checked at repository root; absent).
2. [`.github/copilot-instructions.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/.github/copilot-instructions.md)
3. [`.github/skills/anti-output-quality-review/SKILL.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/.github/skills/anti-output-quality-review/SKILL.md)
4. [`.github/skills/research-plan-compliance-review/SKILL.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/.github/skills/research-plan-compliance-review/SKILL.md)
5. [`.github/skills/local-colab-experiment-workflow/SKILL.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/.github/skills/local-colab-experiment-workflow/SKILL.md)
6. [`docs/research/PERFORMANCE_OPTIMIZATION_HANDOFF_v1.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/PERFORMANCE_OPTIMIZATION_HANDOFF_v1.md) (Section 5.4 — `TICKET-OPT-04`)
7. [`docs/Proposed_Method_SG_VCD.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/Proposed_Method_SG_VCD.md)
8. [`docs/research/REMEDIATION_SPEC_P0_v1.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/REMEDIATION_SPEC_P0_v1.md)
9. [`docs/research/TASK_07_FIX_OPT06_INDEPENDENT_ACCEPTANCE_QA.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/TASK_07_FIX_OPT06_INDEPENDENT_ACCEPTANCE_QA.md)
10. [`docs/research/TASK_08_OPT03_INDEPENDENT_ACCEPTANCE_QA.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/TASK_08_OPT03_INDEPENDENT_ACCEPTANCE_QA.md)
11. [`docs/research/TASK_09_OPT02_MEMORY_CLEANUP_IMPLEMENTATION_REPORT.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/TASK_09_OPT02_MEMORY_CLEANUP_IMPLEMENTATION_REPORT.md)
12. [`docs/research/TASK_10_OPT02_INDEPENDENT_ACCEPTANCE_QA.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/TASK_10_OPT02_INDEPENDENT_ACCEPTANCE_QA.md)
13. [`docs/research/TASK_11_OPT01_BATCHED_DECODING_IMPLEMENTATION_REPORT.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/TASK_11_OPT01_BATCHED_DECODING_IMPLEMENTATION_REPORT.md)
14. [`docs/research/TASK_12_OPT01_INDEPENDENT_ACCEPTANCE_QA.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/TASK_12_OPT01_INDEPENDENT_ACCEPTANCE_QA.md)
15. [`docs/research/TASK_13_OPT01_MASK_SEMANTICS_QA_ADDENDUM.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/TASK_13_OPT01_MASK_SEMANTICS_QA_ADDENDUM.md)

---

## 3. Baseline Bottleneck Analysis

Prior to `TICKET-OPT-04`, [`PointwiseKLAttentionExtractor.extract()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/attention_extractor.py#L943-L985) performed:
1. A host transfer `.detach().cpu().numpy()` on `grid_feats` of shape $(H_g, W_g, D)$ (e.g. $32 \times 48 \times 896 \times 4\text{ B} \approx 5.51\text{ MB}$ per image).
2. CPU-side L2 norm and division across the 3D $(H_g, W_g, D)$ array (`np.linalg.norm(v, axis=-1, keepdims=True)`).
3. Nested Python loops over `_pass in range(2)`, `4` lattice directions $\mathcal{D} = \{(0, 1), (1, 0), (1, 1), (1, -1)\}$, bridge radii $k \in \{1, \dots, K\}$, and continuation signs $\{-1, +1\}$, repeatedly slicing and multiplying 3D $(H_g, W_g, D)$ arrays on CPU.

---

## 4. Vectorized PyTorch Architecture & Mathematical Formulation

In [`compute_intrinsic_affinity_field_gpu()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/attention_extractor.py#L202-L343):
1. **Device & Precision Resolution:**
   - Target `device` is resolved from `feat_grid.device` (if `torch.Tensor`), else `r_map.device`, else `a_query_map.device`, else `cpu`.
   - All inputs (`r_map`, `feat_grid`, `a_query_map`) are promoted on `device` to `torch.float32` inside `with torch.no_grad():`.
2. **Feature & Query Normalization:**
   - $R(u) = \max(r_{\text{map}}(u), 0)$.
   - $\hat{v}_u = v_u / \max(\|v_u\|_2, 10^{-8})$ via `torch.linalg.vector_norm(v, ord=2, dim=-1, keepdim=True)` and `torch.clamp(norms, min=1e-8)`.
   - $\tilde{A}(u) = A_Q^+(u) / \max_u A_Q^+(u)$ when $\max_u A_Q^+(u) > 10^{-12}$, else $R(u)$.
3. **Vectorized Directional Stacking with Exact Interior Validity Masks:**
   - **Part A (Bilateral Gap Affinity):**
     - Offsets $\mathcal{O}_A = \{(k \cdot d_y, k \cdot d_x) \mid (d_y, d_x) \in \mathcal{D},\; k \in \{1, \dots, K\}\}$ ($4K = 8$ offsets for $K=2$).
     - For each offset $(k_y, k_x)$, the interior validity mask `valid_mask_a[idx, |k_y| : H - |k_y|, |k_x| : W - |k_x|] = 1.0` matches the exact slice bounds `y_start:y_end, x_start:x_end` of the CPU reference and is `0.0` on out-of-bounds borders.
     - Precomputes `cos_pos_a` of shape $(4K, H, W)$ once before the 2-pass loop from constant-padded `v_pad_a = F.pad(v_norm, (0, 0, K, K, K, K), mode="constant", value=0.0)` and multiplies by `valid_mask_a`.
     - In each pass ($\text{pass} \in \{0, 1\}$), pads `current_r` with `F.pad(current_r, (K, K, K, K), value=0.0)`, stacks `r_minus_a` and `r_plus_a` of shape $(4K, H, W)$, and reduces via `torch.amax(geom_mean_a * cos_pos_a, dim=0)`.
   - **Part B (Geodesic Stroke-Continuation Affinity, Pass 0 only):**
     - Offsets $\mathcal{O}_B = \{(d_y, d_x, s \cdot d_y, s \cdot d_x) \mid (d_y, d_x) \in \mathcal{D},\; s \in \{-1, +1\}\}$ ($8$ directed offsets at $k=1$).
     - Interior mask `valid_mask_b[idx, |d_y| : H - |d_y|, |d_x| : W - |d_x|] = 1.0` enforces the exact `y_s:y_e, x_s:x_e` slice domain.
     - Computes `cont_scores = torch.sqrt(torch.clamp(r_neigh_b * a_tilde.unsqueeze(0), min=0.0)) * cos_c_sq_b` of shape $(8, H, W)$ on `_pass == 0` only, updating `pass_aff = torch.maximum(pass_aff, torch.amax(cont_scores, dim=0))`.
4. **Final Normalization:**
   - $\Phi(u) = \max(R(u), M_{\text{aff}}(u))$, normalized by $\max_u \Phi(u)$ when $\max_u \Phi(u) > 10^{-12}$.

---

## 5. Boundary & No-Wrap-Around Semantics

- Circular shifts (`torch.roll`) are **not** used anywhere.
- Constant zero-padding (`F.pad(..., mode="constant", value=0.0)`) combined with explicit per-direction interior masks `valid_mask_a` and `valid_mask_b` guarantees that border coordinates never bridge across opposite edges of the image lattice.

---

## 6. Two-Pass (`_pass == 0` vs `_pass == 1`) Semantics

- **Pass 0 (`_pass == 0`):** Executes both **Part A (Bilateral Gap Affinity)** and **Part B (Geodesic Stroke-Continuation Affinity)** (when `v_norm is not None`), then updates `aff_map = torch.maximum(aff_map, pass_aff)` and `current_r = torch.maximum(current_r, pass_aff)`.
- **Pass 1 (`_pass == 1`):** Executes **Part A (Bilateral Gap Affinity)** on the updated `current_r` while skipping Part B (`if v_norm is not None and _pass == 0`), matching line `162` of the original CPU implementation.

---

## 7. Integration in `PointwiseKLAttentionExtractor.extract()` & Cache Lifecycle

- In [`PointwiseKLAttentionExtractor.extract()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/attention_extractor.py#L943-L985):
  - `grid_feats` is constructed via `vit_embeds[:num_grid_tiles].detach().float().reshape(gy, gx, 16, 16, -1).permute(0, 2, 1, 3, 4).reshape(H_g, W_g, -1).contiguous()` and kept on `vit_embeds.device` without calling `.cpu().numpy()`.
  - `self._cached_img_state["grid_feats"]` stores this detached `torch.float32` tensor and reuses it across all questions on the same image (`self._cached_img_key == img_key`).
  - Only the 2D $(H_g, W_g)$ output maps `phi_map` and `aff_map` ($\sim 4\text{ KB}$ each) are converted to `np.float32` for [`ExtractionResult`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/attention_extractor.py#L56-L64).
  - Calling [`extractor.clear_cache()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/attention_extractor.py#L709-L712) or transitioning to a new image sets `self._cached_img_state = None` and `self._cached_img_key = None`, releasing `grid_feats` cleanly.

---

## 8. File-by-File Change Summary

| File | Action | Lines Changed | Description |
| :--- | :--- | :--- | :--- |
| [`scripts/benchmark/sg_vcd/attention_extractor.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/attention_extractor.py) | Modified | `+220 / -23` | Added [`_compute_intrinsic_affinity_field_numpy_reference()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/attention_extractor.py#L93-L199) and [`compute_intrinsic_affinity_field_gpu()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/attention_extractor.py#L202-L343); updated [`compute_intrinsic_affinity_field()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/attention_extractor.py#L346-L384) and [`PointwiseKLAttentionExtractor.extract()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/attention_extractor.py#L943-L985). |
| [`tests/test_opt04_gpu_affinity_vectorization.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt04_gpu_affinity_vectorization.py) | Created | `+818` | Gate A test suite covering 10 synthetic attention/feature-grid regimes and `Tests A–H`. |
| [`docs/research/TASK_14_OPT04_GPU_AFFINITY_VECTORIZATION_IMPLEMENTATION_REPORT.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/TASK_14_OPT04_GPU_AFFINITY_VECTORIZATION_IMPLEMENTATION_REPORT.md) | Created | Report | This implementation report. |

---

## 9. Quantitative Numerical & Stage 1 Localization Parity Across 10 Synthetic Regimes

| # | Synthetic Regime | Grid $(H, W, D)$ | $\max_u |\Phi_{\text{GPU}} - \Phi_{\text{CPU}}|$ | $\max_u |M_{\text{aff,GPU}} - M_{\text{aff,CPU}}|$ | Stage 1 Selected `pixel_bboxes` (GPU == CPU) | Stage 1 BBox IoU |
| :-: | :--- | :---: | :---: | :---: | :--- | :---: |
| 1 | `01_single_compact_text_block_16x16_D64` | `16x16x64` (`f32`) | `2.384e-07` | `3.576e-07` | `[(95, 135, 265, 225)]` | `100.0%` (`1.0`) |
| 2 | `02_multi_word_horizontal_gaps_16x32_D64` | `16x32x64` (`f32`) | `4.768e-07` | `4.172e-07` | `[(115, 95, 325, 205), (315, 95, 505, 205)]` | `100.0%` (`1.0`) |
| 3 | `03_multi_line_vertical_stack_32x16_D64` | `32x16x64` (`f32`) | `4.768e-07` | `5.364e-07` | `[(55, 55, 265, 245), (55, 235, 265, 365)]` | `100.0%` (`1.0`) |
| 4 | `04_diagonal_stroke_tilted_signboard_32x32_D128` | `32x32x128` (`f32`) | `5.364e-07` | `4.172e-07` | `[(60, 60, 308, 308), (284, 284, 436, 420)]` | `100.0%` (`1.0`) |
| 5 | `05_antidiagonal_stroke_tail_continuation_32x32_D128` | `32x32x128` (`f32`) | `6.557e-07` | `3.576e-07` | `[(140, 204, 308, 372), (284, 92, 420, 228)]` | `100.0%` (`1.0`) |
| 6 | `06_border_corner_touching_tokens_16x16_D64` | `16x16x64` (`f32`) | `1.192e-07` | `2.980e-07` | `[(0, 0, 125, 45), (195, 275, 320, 320)]` | `100.0%` (`1.0`) |
| 7 | `07_multi_candidate_disjoint_regions_32x32_D128` | `32x32x128` (`f32`) | `4.172e-07` | `2.980e-07` | `[(300, 188, 468, 292), (60, 364, 228, 484)]` | `100.0%` (`1.0`) |
| 8 | `08_feature_free_mode_24x24_None` | `24x24` (`None`) | `1.192e-07` | `5.960e-08` | `[(215, 135, 365, 265), (95, 135, 225, 285)]` | `100.0%` (`1.0`) |
| 9 | `09_degenerate_zero_map_16x16_D32` | `16x16x32` (`f32`) | `0.000e+00` | `0.000e+00` | `[(0, 0, 320, 320)]` | `100.0%` (`1.0`) |
| 10 | `10_realistic_vintern_grid_32x48_D896_bfloat16` | `32x48x896` (`bf16`) | `2.980e-07` | `5.364e-07` | `[(375, 175, 665, 325), (135, 175, 385, 325)]` | `100.0%` (`1.0`) |

- Worst-case $\max_u |\Phi_{\text{GPU}}(u) - \Phi_{\text{CPU}}(u)| = 6.557 \times 10^{-7} \ll 10^{-4}$.
- Worst-case $\max_u |M_{\text{aff,GPU}}(u) - M_{\text{aff,CPU}}(u)| = 5.364 \times 10^{-7} \ll 10^{-4}$.
- Stage 1 bounding-box parity: **`IoU = 1.0` (`100.0%`)** with identical pixel and grid coordinates across all 10 regimes.

---

## 10. Detailed Summary of `tests/test_opt04_gpu_affinity_vectorization.py` (`Tests A–H`)

1. **`test_a_direct_numerical_parity_across_10_synthetic_regimes` (`PASS`):** Verifies $\max_u |\Phi_{\text{GPU}} - \Phi_{\text{CPU}}| \le 10^{-4}$ and $\max_u |M_{\text{aff,GPU}} - M_{\text{aff,CPU}}| \le 10^{-4}$ across all 10 synthetic regimes.
2. **`test_b_device_and_dtype_contract_verification` (`PASS`):** Verifies `float32`, `bfloat16`, and `float16` `feat_grid` tensors stay on device, promote internal computation to `torch.float32`, and never invoke `.cpu()` or `.numpy()` on the 3D `feat_grid` (`_NoCpuTransferTensor` guard).
3. **`test_c_boundary_and_no_wrap_around_verification` (`PASS`):** Verifies opposing border activations (`top/bottom`, `left/right`, opposite corners) produce `0.0` false wrap-around bridging and match CPU reference within `1e-6`.
4. **`test_d_pass0_vs_pass1_two_pass_semantics` (`PASS`):** Verifies Pass-0 `M_cont` activation followed by Pass-1 `M_gap` bridging, and confirms `M_cont` is not re-executed during Pass 1.
5. **`test_e_stage1_bounding_box_iou_parity_across_10_synthetic_regimes` (`PASS`):** Verifies [`KLDivergenceRegionSelector.select_top_k()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/bbox_selector.py#L357-L561) produces identical candidate counts, identical `pixel_bboxes` and `grid_bboxes` (`IoU = 100.0%`), and `kl_scores` within `1e-5` across all 10 regimes.
6. **`test_f_public_api_contract_and_no_feat_grid_cpu_roundtrip` (`PASS`):** Verifies [`compute_intrinsic_affinity_field()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/attention_extractor.py#L346-L384) returns `(np.ndarray, np.ndarray)` of dtype `float32` without calling `.cpu()` or `.numpy()` on the 3D `feat_grid`.
7. **`test_g_edge_cases_zero_none_tiny_and_nonsquare_grids` (`PASS`):** Verifies `(1, 1)`, `(2, 2)`, `(3, 3)`, `(5, 19)`, and `(19, 5)` grids across `max_bridge_radius in (0, 1, 2, 4)`, `feat_grid=None`, `a_query_map=None`, negative values, and zero-norm feature vectors.
8. **`test_h_extractor_mock_integration_and_cache_lifecycle` (`PASS`):** Verifies [`PointwiseKLAttentionExtractor.extract()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/attention_extractor.py#L868-L996) end-to-end with a synthetic mock model/tokenizer, confirming `grid_feats` is cached as a detached `torch.float32` tensor across `Q1`/`Q2` (`extract_calls == 1`), matches the CPU reference within `1e-5`, and is released by [`clear_cache()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/attention_extractor.py#L709-L712).

---

## 11. Full Test Execution Log (`Test I` — All 7 Required Test Suites)

| Suite | Command | Tests Run | Duration | Exit Code | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **OPT-04 Suite** | `python -m unittest -v tests.test_opt04_gpu_affinity_vectorization` | `8` | `5.148s` | `0` | **`PASS (8/8)`** |
| **OPT-01 Suite** | `python -m unittest -v tests.test_opt01_batched_dual_branch_decoding` | `8` | — | `0` | **`PASS (8/8)`** |
| **OPT-02 Suite** | `python -m unittest -v tests.test_opt02_memory_cleanup_policy` | `9` | — | `0` | **`PASS (9/9)`** |
| **OPT-03 Suite** | `python -m unittest -v tests.test_opt03_working_map_cache` | `8` | — | `0` | **`PASS (8/8)`** |
| **OPT-06 Suite** | `python -m unittest -v tests.test_opt06_image_cache_identity` | `11` | — | `0` | **`PASS (11/11)`** |
| **Adversarial Spatial** | `python -m unittest -v tests.test_adversarial_spatial_processing` | `21` | — | `0` | **`PASS (21/21)`** |
| **SG-VCD Core** | `python -m unittest -v tests.test_sg_vcd` | `33` | `19.739s` (combined 90) | `0` | **`PASS (33/33)`** |
| **Total** | **All 7 Gate A Suites** | **`98`** | **`24.887s`** | **`0`** | **`PASS (98/98)`** |

---

## 12. Compatibility with OPT-06, OPT-03, OPT-02, and OPT-01

- **OPT-06 (Image Cache Identity):** Untouched; [`PointwiseKLAttentionExtractor._make_image_key()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/attention_extractor.py#L715-L740) continues to key `self._cached_img_state` by `ImageCacheIdentity`.
- **OPT-03 (Image-Level Working-Map Caching):** Untouched; [`AdaptiveScalePropagator`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/adaptive_propagator.py) working-map cache operates downstream of `ExtractionResult`.
- **OPT-02 (Coordinated Memory Cleanup Policy):** Untouched; no per-question `gc.collect()` or `torch.cuda.empty_cache()` calls were introduced in `attention_extractor.py` (verified by `test_I_ast_verification_no_per_question_cleanup_in_pipeline_or_extractor`).
- **OPT-01 (Batched Positive/Negative Decoding, B=2):** Untouched; `pipeline.py` was not modified.

---

## 13. Scope & Governance Compliance Check

- Modified only [`scripts/benchmark/sg_vcd/attention_extractor.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/attention_extractor.py).
- Created only [`tests/test_opt04_gpu_affinity_vectorization.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt04_gpu_affinity_vectorization.py) and [`docs/research/TASK_14_OPT04_GPU_AFFINITY_VECTORIZATION_IMPLEMENTATION_REPORT.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/TASK_14_OPT04_GPU_AFFINITY_VECTORIZATION_IMPLEMENTATION_REPORT.md).
- Zero real `5CD-AI/Vintern-1B-v3_5` model weights were loaded; all validation executed in local synthetic/mock Gate A mode.
- Working tree left uncommitted per git safety rules.

---

## 14. Risks, Edge Cases & Limitations

- **Floating-Point Reduction Order (`float32` vs `float64`):** PyTorch SIMD reduction along `dim=-1` and `bfloat16 -> float32` promotion exhibit up to `6.56e-7` difference vs CPU NumPy, which is $150\times$ below the $10^{-4}$ tolerance and causes zero bounding-box deviation (`IoU = 1.0`).
- **Gate B Hardware Timing:** Real GPU wall-clock speedup on `5CD-AI/Vintern-1B-v3_5` (`32x48x896` tiles on CUDA) is deferred to Gate B benchmark validation after independent QA acceptance.

---

## 15. Working Tree Status (`git status --short`)

- Modified: `scripts/benchmark/sg_vcd/attention_extractor.py`
- Untracked (new): `tests/test_opt04_gpu_affinity_vectorization.py`, `docs/research/TASK_14_OPT04_GPU_AFFINITY_VECTORIZATION_IMPLEMENTATION_REPORT.md` (alongside prior uncommitted optimization task artifacts).

---

## 16. Handover to Task 15 Independent QA (`Antigravity`)

`TICKET-OPT-04` implementation and Gate A verification are complete. Ready for Task 15 independent read-only acceptance QA.

