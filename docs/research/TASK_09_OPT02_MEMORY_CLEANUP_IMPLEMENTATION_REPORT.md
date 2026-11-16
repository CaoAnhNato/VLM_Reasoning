# TASK 09 — IMPLEMENTATION REPORT: TICKET-OPT-02 (COORDINATED MEMORY CLEANUP POLICY)

**Implementer:** `4argon` — Senior ML Systems Engineer  
**Independent QA Reviewer:** `Antigravity (Anti)`  
**Methodology & Architecture Reviewer:** `ChatGPT`  
**Final Approval Authority:** Project Owner  
**Date:** 2026-11-08  
**Ticket Identity:** `TICKET-OPT-02 — Coordinated Memory Cleanup Policy` (Priority: **P1**)  
**Execution Mode:** Bounded implementation, local synthetic/mock validation only (**Gate A**).

---

## 1. Executive Summary

This report documents the implementation and local synthetic/mock verification of **`TICKET-OPT-02 — Coordinated Memory Cleanup Policy`**, the third approved ticket in the small-first performance optimization sequence (`OPT-06` $\rightarrow$ `OPT-03` $\rightarrow$ `OPT-02` $\rightarrow$ `OPT-01`).

Prior to `TICKET-OPT-02`, global memory cleanup operations (`gc.collect()` and `torch.cuda.empty_cache()`) were invoked unconditionally inside per-question execution paths:
1. Inside [`PointwiseKLAttentionExtractor.extract()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/attention_extractor.py#L782-L783) after Stage 1.1 attention extraction (`torch.cuda.empty_cache()`).
2. Inside [`SGVCDPipeline.run()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L1052-L1055) at the end of every question (`gc.collect()` + `torch.cuda.empty_cache()`).
3. Inside [`run_benchmark()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/evaluate_vitextvqa.py#L1429-L1443) at each image boundary (`gc.collect()` + unconditional `torch.cuda.empty_cache()`).

Across the verified 400-image / 1,222-question ViTextVQA benchmark ([`results/swintextspotter_auto_annot/answer_bboxes_400_verified.json`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/results/swintextspotter_auto_annot/answer_bboxes_400_verified.json), SHA-256 `1428e33d493c0f00256f21c640a1923df989436294ff6190cd349fadddb13c5c`), this legacy policy triggered **`1,622` full-heap Python `gc.collect()` sweeps** (`1,222` per-question + `400` image-boundary) and **`2,844` unconditional `torch.cuda.empty_cache()` host-device synchronizations** (`1,222` in `extractor.extract()` + `1,222` in `pipeline.run()` + `400` at image boundaries).

### Key Implementation Outcomes
1. **Zero Per-Question Global Cleanup:** Removed unconditional `gc.collect()` and `torch.cuda.empty_cache()` calls from [`SGVCDPipeline.run()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L1050-L1056) and [`PointwiseKLAttentionExtractor.extract()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/attention_extractor.py#L782-L785), while preserving all local tensor reference releases (`del suffix_q`, `del pixel_values`, `del pos_vit, neg_vit, ...`) and `try ... finally` attention implementation restoration (`_set_attn_implementation(prev_mode)`).
2. **Coordinated Image-Boundary Cleanup:** Added [`run_image_boundary_cleanup()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/evaluate_vitextvqa.py#L1027-L1059) in [`scripts/benchmark/evaluate_vitextvqa.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/evaluate_vitextvqa.py), invoked inside a `try ... finally` block at the end of each image group in [`run_benchmark()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/evaluate_vitextvqa.py#L1260-L1443) after all questions for that image have finished (or failed and been recorded).
3. **CUDA `0.85` Memory Safety Watchdog:** Added [`check_and_run_cuda_memory_watchdog()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/evaluate_vitextvqa.py#L934-L1024), which calls `torch.cuda.empty_cache()` at an image boundary **only** when:
   $$\frac{\text{torch.cuda.memory\_allocated()}}{\text{total CUDA memory}} > 0.85$$
   while safely handling CPU-only mode (`torch.cuda.is_available() == False` or `device="cpu"`), device index resolution, and unavailable/invalid CUDA memory queries without crashing.
4. **Full Preservation of `OPT-06` and `OPT-03`:** Within an image group, all questions reuse the active image's `OPT-06` visual features and `OPT-03` Stage 1.2 working maps (`(img_work, L, C_rg, C_yb, W_w, H_w, sx, sy)`). At the image boundary, [`SGVCDPipeline.reset_cache()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L360-L382) releases the completed image's cached references before `gc.collect()` runs, without invoking an unconditional `torch.cuda.empty_cache()`.
5. **Verification:** All **9/9** unit and mock integration tests in [`tests/test_opt02_memory_cleanup_policy.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt02_memory_cleanup_policy.py) (`Tests A–I`) and **74/74** tests in the regression suites (`OPT-03`, `OPT-06`, `test_sg_vcd`, `test_adversarial_spatial_processing`, and `test_tier4_workloads`) passed (`0` failures, `0` errors).

---

## 2. Governance and Scope Compliance Evidence

Before inspecting or modifying any code, all mandatory governance and technical specification files were loaded and followed:

| Governance / Specification File | Status | Compliance Verification |
| :--- | :--- | :--- |
| [`.github/copilot-instructions.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/.github/copilot-instructions.md) | Read & Enforced | No scientific logic altered; no unapproved tickets implemented; no git commit/push performed. |
| [`.github/skills/anti-output-quality-review/SKILL.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/.github/skills/anti-output-quality-review/SKILL.md) | Read & Enforced | Exact file/line evidence, exact call counts, and honest `NOT_MEASURED` labels for real-model GPU metrics. |
| [`.github/skills/research-plan-compliance-review/SKILL.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/.github/skills/research-plan-compliance-review/SKILL.md) | Read & Enforced | Stage 1 & Stage 2 mathematical formulations remain 100% invariant (`0` algorithmic drift). |
| [`.github/skills/local-colab-experiment-workflow/SKILL.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/.github/skills/local-colab-experiment-workflow/SKILL.md) | Read & Enforced | Local validation restricted strictly to synthetic/mock CPU execution; `5CD-AI/Vintern-1B-v3_5` never loaded locally. |
| [`docs/research/PERFORMANCE_OPTIMIZATION_HANDOFF_v1.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/PERFORMANCE_OPTIMIZATION_HANDOFF_v1.md) | Read & Enforced | Implemented strictly per Section 4, Ticket 3 (`TICKET-OPT-02`, lines 109–123). |
| [`docs/research/TASK_07_FIX_OPT06_INDEPENDENT_ACCEPTANCE_QA.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/TASK_07_FIX_OPT06_INDEPENDENT_ACCEPTANCE_QA.md) | Read & Enforced | `OPT-06` `ImageCacheIdentity` contract preserved untouched. |
| [`docs/research/TASK_08_OPT03_WORKING_MAP_CACHE_IMPLEMENTATION_REPORT.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/TASK_08_OPT03_WORKING_MAP_CACHE_IMPLEMENTATION_REPORT.md) | Read & Enforced | `OPT-03` working-map caching preserved untouched across questions of the same image. |
| [`docs/research/TASK_08_OPT03_INDEPENDENT_ACCEPTANCE_QA.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/TASK_08_OPT03_INDEPENDENT_ACCEPTANCE_QA.md) | Read & Enforced | Verified dataset statistics (`400` images, `1,222` questions) and exact test/code parity applied. |
| [`docs/research/REMEDIATION_SPEC_P0_v1.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/REMEDIATION_SPEC_P0_v1.md) | Read & Enforced | P0 remediation invariants (`_set_attn_implementation` `finally` block, `bfloat16` lifecycle) preserved. |
| [`docs/Proposed_Method_SG_VCD.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/Proposed_Method_SG_VCD.md) | Read & Enforced | All SG-VCD Stage 1 and Stage 2 equations preserved without modification. |

### Strict Scope Verification
- **Authorized Ticket Implemented:** `TICKET-OPT-02` only.
- **Unapproved Tickets Excluded:** `OPT-01` (Batched Positive/Negative Decoding $B=2$) and `OPT-04` (GPU Affinity Field Vectorization) were **not** implemented.
- **Authorized Target Files Modified / Created:**
  1. [`scripts/benchmark/sg_vcd/pipeline.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py)
  2. [`scripts/benchmark/sg_vcd/attention_extractor.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/attention_extractor.py)
  3. [`scripts/benchmark/evaluate_vitextvqa.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/evaluate_vitextvqa.py)
  4. [`tests/test_opt02_memory_cleanup_policy.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt02_memory_cleanup_policy.py)
  5. [`docs/research/TASK_09_OPT02_MEMORY_CLEANUP_IMPLEMENTATION_REPORT.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/TASK_09_OPT02_MEMORY_CLEANUP_IMPLEMENTATION_REPORT.md)

---

## 3. Ticket Identity Confirmation (`TICKET-OPT-02 — Coordinated Memory Cleanup Policy`)

Per [`docs/research/PERFORMANCE_OPTIMIZATION_HANDOFF_v1.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/PERFORMANCE_OPTIMIZATION_HANDOFF_v1.md#L109-L123) (Section 4, Ticket 3):
- **Ticket ID & Title:** `TICKET-OPT-02 — Coordinated Memory Cleanup Policy`
- **Priority:** `P1`
- **Target Files:** `scripts/benchmark/sg_vcd/pipeline.py`, `scripts/benchmark/sg_vcd/attention_extractor.py`, `scripts/benchmark/evaluate_vitextvqa.py`
- **Authoritative Specification:**
  1. Remove per-question `gc.collect()` and `torch.cuda.empty_cache()` calls from `SGVCDPipeline.run()` and `PointwiseKLAttentionExtractor.extract()`.
  2. Perform `gc.collect()` at the **image boundary** inside `evaluate_vitextvqa.py` (`run_benchmark()`) after all questions for the current image have completed.
  3. Add a memory safety watchdog at the image boundary that calls `torch.cuda.empty_cache()` only if `torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory > 0.85` (generalized safely to the active CUDA device).

---

## 4. Inspection of Existing Memory Cleanup Behavior

An audit of the pre-`OPT-02` codebase identified four locations affecting per-question and image-boundary memory cleanup:

| Location | Pre-`OPT-02` Behavior | Operational Impact on 400-Image / 1,222-QA Benchmark |
| :--- | :--- | :--- |
| [`PointwiseKLAttentionExtractor.extract()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/attention_extractor.py#L782-L784) | Called `if torch.cuda.is_available(): torch.cuda.empty_cache()` unconditionally after every question's Stage 1.1 extraction. | `1,222` unconditional CUDA cache flushes + host-GPU synchronizations mid-pipeline (between Stage 1.1 and Stage 2 of the same question), forcing PyTorch's CUDA caching allocator to re-request `cudaMalloc` blocks during Stage 2 decoding. |
| [`SGVCDPipeline.run()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L1044-L1046) | Called `gc.collect()` and `if torch is not None and torch.cuda.is_available(): torch.cuda.empty_cache()` unconditionally at the end of every question. | `1,222` full-heap Python `gc.collect()` sweeps and `1,222` unconditional `torch.cuda.empty_cache()` calls between questions of the same image. |
| [`evaluate_vitextvqa.py` (`run_benchmark`)](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/evaluate_vitextvqa.py#L1258-L1262) | Called `del image, cached_visual_features`, `gc.collect()`, and unconditional `if torch.cuda.is_available(): torch.cuda.empty_cache()` at the end of each image group, without a `finally` guard and without resetting pipeline image caches before `gc.collect()`. | `400` image-boundary `gc.collect()` calls and `400` unconditional `torch.cuda.empty_cache()` calls; last question's `_cached_working_maps` and `_cached_img_state` remained attached to `pipeline` during the image-boundary `gc.collect()`. |
| [`SGVCDPipeline.reset_cache()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L360-L375) | Delegated to `self.feature_cache.clear()`, which in [`VisualFeatureCache.clear()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/feature_cache.py#L528-L529) calls `torch.cuda.empty_cache()` unconditionally. | Calling `pipeline.reset_cache()` at an image boundary would trigger an unconditional `torch.cuda.empty_cache()` unless `reset_cache()` evicts `self.feature_cache._cache` entries without invoking `empty_cache()`. |

### Summary of Full-Benchmark Cleanup Call Counts (`400` Images, `1,222` Questions)

| Operation | Pre-`OPT-02` Call Count | Post-`OPT-02` Call Count | Reduction |
| :--- | :---: | :---: | :---: |
| Per-question `gc.collect()` inside `pipeline.run()` | `1,222` | **`0`** | `-100.0%` (`-1,222` calls) |
| Image-boundary `gc.collect()` in `evaluate_vitextvqa.py` | `400` | **`400`** | Coordinated once per image (`-75.34%` total `gc.collect()` calls: `1,622` $\rightarrow$ `400`) |
| Per-question `torch.cuda.empty_cache()` (`extractor.extract` + `pipeline.run`) | `2,444` (`1,222 + 1,222`) | **`0`** | `-100.0%` (`-2,444` calls) |
| Image-boundary `torch.cuda.empty_cache()` in `evaluate_vitextvqa.py` | `400` (unconditional) | **`0` to `400`** (conditional on `allocated / total > 0.85`) | `-85.94%` to `-100.0%` total `empty_cache()` calls (`2,844` $\rightarrow$ `0..400`) |

---

## 5. Code Changes by File and Line Range

### 5.1 [`scripts/benchmark/sg_vcd/pipeline.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py)
1. **[`SGVCDPipeline.reset_cache()` (`lines 360–383`)](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L360-L383):**
   - Clears `self._cached_working_maps = None` and `self._cached_working_maps_key = None`.
   - Evicts all tensor references in `self.feature_cache._cache` (`pop` + `del tensor` + `clear()`) without invoking `VisualFeatureCache.clear()`'s unconditional `torch.cuda.empty_cache()`, so image-boundary cache resets never bypass the `0.85` CUDA memory watchdog.
   - Clears `self.extractor.clear_cache()`.
2. **[`SGVCDPipeline.run()` (`lines 1050–1056`)](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L1050-L1056):**
   - Removed the per-question `gc.collect()` and `if torch is not None and torch.cuda.is_available(): torch.cuda.empty_cache()` calls before `return PipelineResult(...)`.
   - Preserved all explicit local tensor cleanup statements inside `_decode_spatial_pmi_contrastive()` (`del (pos_vit, neg_vit, pos_prefix, neg_prefix, pos_kv, neg_kv)` at lines `586–593`) and `_decode_with_tiles()` (`del vit_embeds, prefix_embeds, suffix_q, kv, out` at line `743`).

### 5.2 [`scripts/benchmark/sg_vcd/attention_extractor.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/attention_extractor.py)
1. **[`PointwiseKLAttentionExtractor.extract()` (`lines 782–785`)](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/attention_extractor.py#L782-L785):**
   - Removed `if torch.cuda.is_available(): torch.cuda.empty_cache()` after `del suffix_q`.
   - Preserved `del suffix_q` (`line 782`), `del pixel_values` (`line 726`), `del prefix_embeds, prefix_out` (`line 743`), `del kv_draft, out` (`line 508`), `del att_q_list, out_q, kv_q` (`lines 660–662`), and the `try ... finally: self._set_attn_implementation(prev_mode)` block (`lines 544–563`).

### 5.3 [`scripts/benchmark/evaluate_vitextvqa.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/evaluate_vitextvqa.py)
1. **[`CUDA_MEMORY_WATCHDOG_THRESHOLD` & `_resolve_cuda_device_index()` (`lines 904–931`)](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/evaluate_vitextvqa.py#L904-L931):**
   - Defines `CUDA_MEMORY_WATCHDOG_THRESHOLD: float = 0.85` and resolves the active CUDA device index from `int`, `str` (`"cuda"`, `"cuda:0"`, `"cpu"`), `torch.device`, or `torch.cuda.current_device()`.
2. **[`check_and_run_cuda_memory_watchdog()` (`lines 934–1024`)](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/evaluate_vitextvqa.py#L934-L1024):**
   - Implements the `0.85` CUDA allocated-memory ratio check and conditional `torch.cuda.empty_cache()` execution with structured return diagnostics and logging.
3. **[`run_image_boundary_cleanup()` (`lines 1027–1059`)](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/evaluate_vitextvqa.py#L1027-L1059):**
   - Coordinates `pipeline.reset_cache()`, `gc.collect()`, and `check_and_run_cuda_memory_watchdog()` once per image boundary.
4. **[`run_benchmark()` (`lines 1208–1216` and `1252–1449`)](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/evaluate_vitextvqa.py#L1252-L1449):**
   - Wraps per-image processing (feature extraction + `for item in q_list:` question loop) in a `try ... finally` block that closes `image`, clears `image` and `cached_visual_features`, and invokes `run_image_boundary_cleanup(pipeline=pipeline, device=device, watermark_ratio=CUDA_MEMORY_WATCHDOG_THRESHOLD, log_fn=print)` once after all questions for the active image complete.

---

## 6. Image-Boundary Cleanup Lifecycle Diagram / Flow Description

```
For each image group (_group_key, q_list) in grouped_samples:
  │
  ├─ 1. Open PIL Image `image` once; compute `image_cache_key = build_image_cache_identity(...)` (OPT-06)
  ├─ 2. Extract `cached_visual_features = pipeline.extract_visual_features(image, image_key=image_cache_key)` once
  │
  ├─ 3. try:
  │       For each question `item` in `q_list` (M questions for this image):
  │         ├─ Question 1 (cache miss on OPT-03 working maps):
  │         │    ├─ Stage 1.1: `extractor.extract()` computes & caches `_cached_img_state` under `image_cache_key`
  │         │    ├─ Stage 1.2: `propagator._compute_working_maps(image)` computes `(img_work, L, C_rg, C_yb, ...)`
  │         │    │             and caches in `pipeline._cached_working_maps` under `image_cache_key` (OPT-03)
  │         │    ├─ Stage 1.3–2: Region selection, ROI crop, and contrastive decoding
  │         │    └─ NO `gc.collect()` and NO `torch.cuda.empty_cache()` after Question 1
  │         │
  │         ├─ Questions 2 .. M (cache hits on OPT-06 and OPT-03):
  │         │    ├─ Stage 1.1: Reuses `extractor._cached_img_state` (OPT-06)
  │         │    ├─ Stage 1.2: Reuses `pipeline._cached_working_maps` via `propagate_multi_with_maps` (OPT-03)
  │         │    ├─ Stage 1.3–2: Question-specific region selection, ROI crop, and contrastive decoding
  │         │    └─ NO `gc.collect()` and NO `torch.cuda.empty_cache()` after Questions 2 .. M
  │         │
  │         └─ Record metrics & append to `ckpt_mgr` (per-question exception caught & recorded if any)
  │
  └─ 4. finally (Image Boundary — executes once after all M questions of this image finish):
          ├─ Close `image`; set `image = None`, `cached_visual_features = None`; `del image, cached_visual_features`
          └─ `run_image_boundary_cleanup(pipeline, device, watermark_ratio=0.85)`:
               ├─ a) `pipeline.reset_cache()` -> releases `_cached_working_maps`, `_cached_img_state`, `feature_cache._cache`
               ├─ b) `gc.collect()` -> runs once for the completed image
               └─ c) `check_and_run_cuda_memory_watchdog(device, watermark_ratio=0.85)`:
                      ├─ If CPU-only or CUDA unavailable -> no-op (`status="cpu_or_cuda_unavailable"`)
                      ├─ Query `ratio = memory_allocated(dev) / get_device_properties(dev).total_memory`
                      ├─ If `ratio > 0.85`  -> call `torch.cuda.empty_cache()` (`status="triggered"`)
                      └─ If `ratio <= 0.85` -> skip `torch.cuda.empty_cache()` (`status="below_threshold"`)
```

---

## 7. CUDA 0.85 Memory Watchdog Design and Edge-Case Handling

[`check_and_run_cuda_memory_watchdog()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/evaluate_vitextvqa.py#L934-L1024) handles all runtime and hardware edge cases deterministically:

1. **Strict Threshold Comparison (`ratio > 0.85`):**
   - Computes `ratio = float(allocated_bytes) / float(total_bytes)`.
   - Triggers `torch.cuda.empty_cache()` strictly when `ratio > watermark_ratio` (`0.85`). At the exact boundary `ratio == 0.85`, `empty_cache()` is **not** called (verified in `Test C` and `Test H`).
2. **Active Device Resolution (`_resolve_cuda_device_index`):**
   - Supports `device="cuda:0"`, `device="cuda"`, `device=torch.device("cuda:0")`, `device=0`, or `device=None` (falling back to `torch.cuda.current_device()`).
   - If `device` is explicitly `"cpu"` or `torch.device("cpu")`, returns `None` immediately so no CUDA queries are made even on a machine where `torch.cuda.is_available()` is `True` (verified in `Test E`).
3. **Exception and Invalid-Property Safety:**
   - Wraps `torch.cuda.memory_allocated(dev_idx)` and `torch.cuda.get_device_properties(dev_idx).total_memory` inside `try ... except Exception`.
   - Guards against `total_bytes <= 0` or `allocated_bytes < 0` (`status="invalid_memory_values"`), logging a warning and returning without raising an exception or dividing by zero (verified in `Test E`).
4. **Observability Logging:**
   - Logs `[MEMORY WATCHDOG] cuda:{dev_idx} allocated={alloc_mb:.1f}MB/{total_mb:.1f}MB (ratio={ratio:.4f} > 0.85) -> triggered torch.cuda.empty_cache().` when triggered, and `... (ratio={ratio:.4f} <= 0.85) -> skipped torch.cuda.empty_cache().` when below threshold.

---

## 8. Interaction with `OPT-06` and `OPT-03`

`TICKET-OPT-02` is designed to work in lockstep with `TICKET-OPT-06` and `TICKET-OPT-03`:
- **During the Question Loop (`for item in q_list:`):**
  - `run_benchmark()` computes `image_cache_key = build_image_cache_identity(...)` (`OPT-06`) and `cached_visual_features` once before the loop and passes both to `pipeline.run(image, question, cached_visual_features=cached_visual_features, image_key=image_cache_key)`.
  - Question 1 populates `extractor._cached_img_state` (`OPT-06`) and `pipeline._cached_working_maps` (`OPT-03`).
  - Questions $2 \dots M$ hit both caches with zero per-question `gc.collect()` or `torch.cuda.empty_cache()` interruptions.
- **At the Image Boundary (`finally:` block after `q_list` completes):**
  - All questions for the current image have already been evaluated, so the current image's cached tensors are no longer needed for any future image group.
  - `run_image_boundary_cleanup()` calls `pipeline.reset_cache()` **before** `gc.collect()`, ensuring that the completed image's `_cached_working_maps` (NumPy float32 opponent-color arrays), `_cached_img_state` (Stage 1.1 KV cache and visual embeddings), and `feature_cache._cache` entries have zero live references when `gc.collect()` and the `0.85` CUDA watchdog run.
  - This guarantees both **maximum intra-image cache reuse** (`OPT-06` + `OPT-03`) and **zero cross-image memory retention** (`OPT-02`).

---

## 9. CPU-Only / Mock-Mode Compatibility Analysis

- When `torch.cuda.is_available() == False` or `device == "cpu"` (including `--mock` mode and local CPU unit tests):
  - `check_and_run_cuda_memory_watchdog()` immediately returns `{"cuda_available": False, "triggered_empty_cache": False, "status": "cpu_or_cuda_unavailable", ...}` without calling `torch.cuda.current_device()`, `torch.cuda.memory_allocated()`, `torch.cuda.get_device_properties()`, or `torch.cuda.empty_cache()`.
  - `run_image_boundary_cleanup()` still resets the pipeline's image cache and runs `gc.collect()` once per image boundary on CPU.

---

## 10. Unit and Mock Integration Test Matrix (`Tests A–I`)

All tests are implemented in [`tests/test_opt02_memory_cleanup_policy.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt02_memory_cleanup_policy.py):

| Test ID | Test Method | Scope & Assertions Verified | Status |
| :--- | :--- | :--- | :---: |
| **Test A** | [`test_A_no_global_cleanup_per_question`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt02_memory_cleanup_policy.py#L315-L378) | Runs 4 synthetic questions on 1 image through `PointwiseKLAttentionExtractor.extract()` and `SGVCDPipeline.run()` with `torch.cuda.is_available=True`; asserts `0` calls to `gc.collect()` and `0` calls to `torch.cuda.empty_cache()`. | **PASS** |
| **Test B** | [`test_B_cleanup_at_image_boundary`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt02_memory_cleanup_policy.py#L383-L465) | Runs a 3-question image group through `run_benchmark()`; asserts the execution timeline is `[question_1, question_2, question_3, image_boundary_cleanup]`. | **PASS** |
| **Test C** | [`test_C_watchdog_below_threshold`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt02_memory_cleanup_policy.py#L470-L508) | Mocks CUDA memory ratios `0.50` and exact boundary `0.85`; asserts `torch.cuda.empty_cache()` call count is `0`, `triggered_empty_cache=False`, `status="below_threshold"`, and log message records skipped cleanup. | **PASS** |
| **Test D** | [`test_D_watchdog_above_threshold`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt02_memory_cleanup_policy.py#L513-L547) | Mocks CUDA memory ratios `0.86` and `0.92`; asserts `torch.cuda.empty_cache()` is called once per check, `triggered_empty_cache=True`, `status="triggered"`, and log message records triggered cleanup. | **PASS** |
| **Test E** | [`test_E_cpu_only_compatibility`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt02_memory_cleanup_policy.py#L552-L611) | Tests `torch.cuda.is_available() == False`, `device="cpu"` when CUDA is available, `RuntimeError` during `memory_allocated()`, and `total_memory == 0`; asserts `0` crashes and `0` `empty_cache()` calls. | **PASS** |
| **Test F** | [`test_F_opt03_integration_preservation`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt02_memory_cleanup_policy.py#L616-L669) | Runs 4 questions sharing one image identity; asserts `_compute_working_maps` runs once (`1` miss, `3` hits), working maps remain valid across all 4 questions, and `run_image_boundary_cleanup()` clears `_cached_working_maps` and `feature_cache._cache`. | **PASS** |
| **Test G** | [`test_G_image_transition_and_exception_safety`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt02_memory_cleanup_policy.py#L674-L789) | Runs 2 image groups (`3` QAs + `2` QAs) where Q2 of Image 1 raises `RuntimeError`; asserts Q2 failure is checkpointed, Q3 reuses Image 1's cached working maps, `run_image_boundary_cleanup()` runs twice (`1` per image), and `_compute_working_maps` runs twice. | **PASS** |
| **Test H** | [`test_H_synthetic_20_qa_workload`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt02_memory_cleanup_policy.py#L794-L1006) | Runs a 20-QA synthetic workload across 6 images (`4+3+4+3+3+3`); verifies `0` per-question cleanups, `6` image-boundary `gc.collect()` calls, `6` watchdog checks, `2` conditional `empty_cache()` triggers under ratios `[0.62, 0.78, 0.85, 0.88, 0.74, 0.91]`, `6` `_compute_working_maps` calls (`14` hits), exact cached-vs-uncached output parity across all 20 QAs, and `0` surviving `weakref` references after cleanup. | **PASS** |
| **Test I** | [`test_I_ast_verification_no_per_question_cleanup_in_pipeline_or_extractor`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt02_memory_cleanup_policy.py#L1011-L1032) | AST inspection of `SGVCDPipeline.run` and `PointwiseKLAttentionExtractor.extract` confirming zero `gc.collect` or `empty_cache` calls exist in either method. | **PASS** |

---

## 11. Command Execution Log and Test Results

### Command 1: `TICKET-OPT-02` Unit and Mock Integration Suite (`Tests A–I`)
```powershell
& "..\Document AI\VLM_Reasoning\venv\Scripts\python.exe" -m unittest tests.test_opt02_memory_cleanup_policy -v
```
**Result:**
```text
test_A_no_global_cleanup_per_question ... ok
test_B_cleanup_at_image_boundary ... ok
test_C_watchdog_below_threshold ... ok
test_D_watchdog_above_threshold ... ok
test_E_cpu_only_compatibility ... ok
test_F_opt03_integration_preservation ... ok
test_G_image_transition_and_exception_safety ... ok
test_H_synthetic_20_qa_workload ... ok
test_I_ast_verification_no_per_question_cleanup_in_pipeline_or_extractor ... ok

----------------------------------------------------------------------
Ran 9 tests in 9.809s

OK
```

### Command 2: Regression Test Suites (`OPT-03`, `OPT-06`, `test_sg_vcd`, `test_adversarial_spatial_processing`, `test_tier4_workloads`)
```powershell
& "..\Document AI\VLM_Reasoning\venv\Scripts\python.exe" -m unittest tests.test_opt03_working_map_cache tests.test_opt06_image_cache_identity tests.test_sg_vcd tests.test_adversarial_spatial_processing tests.test_tier4_workloads.TestTier4RealWorldWorkloads.test_05_anti_hardcoding_and_heuristics_audit -v
```
**Result:**
```text
Ran 74 tests in 8.041s

OK
```

---

## 12. 20-QA Synthetic Workload Verification Summary

In [`TestOPT02MemoryCleanupPolicy.test_H_synthetic_20_qa_workload`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt02_memory_cleanup_policy.py#L794-L1006), a 20-QA synthetic scene-text workload distributed across 6 distinct images (`[4, 3, 4, 3, 3, 3]` questions per image) was executed end-to-end through [`run_benchmark()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/evaluate_vitextvqa.py#L1065):

| Metric / Invariant | Expected Value | Observed Value | Status |
| :--- | :---: | :---: | :---: |
| Total synthetic QA samples evaluated | `20` | `20` | **PASS** |
| Unique image groups | `6` | `6` | **PASS** |
| Per-question `gc.collect()` calls inside `pipeline.run()` | `0` | `0` | **PASS** |
| Per-question `torch.cuda.empty_cache()` calls inside `pipeline.run()` / `extractor.extract()` | `0` | `0` | **PASS** |
| Image-boundary `gc.collect()` calls | `6` | `6` | **PASS** |
| Image-boundary CUDA watchdog evaluations | `6` | `6` | **PASS** |
| Mocked CUDA memory ratios across 6 image boundaries | `[0.62, 0.78, 0.85, 0.88, 0.74, 0.91]` | `[0.62, 0.78, 0.85, 0.88, 0.74, 0.91]` | **PASS** |
| Conditional `torch.cuda.empty_cache()` triggers (`ratio > 0.85`) | `2` (`0.88`, `0.91`) | `2` (`[False, False, False, True, False, True]`) | **PASS** |
| `OPT-03` `_compute_working_maps` calls (`6` misses, `14` hits) | `6` | `6` (`70.0%` reduction on 20-QA set) | **PASS** |
| Cached vs. uncached output parity (`pred_bbox`, `pred_answer`, `spatial_confidence`) | `20 / 20` exact match | `20 / 20` exact match | **PASS** |
| Surviving `weakref` references to completed images' working maps after run | `0` | `0` | **PASS** |

---

## 13. Memory Risk and Safety Assessment

### 13.1 Distinction Between Active Allocated Tensors and Cached Allocator Blocks
A critical systems distinction governs PyTorch's CUDA memory management:
1. **Active Allocated Memory (`torch.cuda.memory_allocated()`):**
   - Represents GPU memory currently occupied by live `torch.Tensor` objects reachable from Python variables, model weights, or active caches.
   - **Calling `torch.cuda.empty_cache()` does NOT free active allocated tensors.** Tensors are freed from `memory_allocated()` immediately when their Python reference count drops to zero (via `del` or scope exit, or via `gc.collect()` when part of a Python reference cycle) and their underlying storage is returned to PyTorch's CUDA caching allocator.
2. **Inactive Cached Allocator Blocks (`torch.cuda.memory_reserved() - torch.cuda.memory_allocated()`):**
   - When a CUDA tensor is deleted in Python, PyTorch returns its memory block to PyTorch's internal caching allocator pool (`memory_reserved`) so subsequent tensor allocations of similar shapes can reuse the block in $\mathcal{O}(1)$ time without invoking a blocking OS/driver `cudaFree` / `cudaMalloc` round-trip.
   - Calling `torch.cuda.empty_cache()` forces synchronization and releases those **unallocated cached blocks** back to the CUDA driver, which is only beneficial when allocator fragmentation or high live memory pressure (`allocated / total > 0.85`) risks exhausting contiguous driver memory.

### 13.2 Why Coordinated Image-Boundary Cleanup is Safe on 4 GB VRAM
- **Intra-Question Temporary Tensors:** All large temporary tensors created during a question (`pixel_values`, `prefix_embeds`, `prefix_out`, `kv_draft`, `att_q_list`, `out_q`, `kv_q`, `suffix_q`, `pos_vit`, `neg_vit`, `pos_prefix`, `neg_prefix`, `pos_kv`, `neg_kv`) are local variables or explicitly deleted with `del` as soon as their stage completes. Their reference counts drop to zero immediately without needing `gc.collect()`, and PyTorch's caching allocator immediately reuses their memory blocks on the very next question of the same image.
- **Intra-Image Shared Caches (`OPT-06` and `OPT-03`):** Only a single image's visual embeddings (`~0.46 MB` to `~2.75 MB` in `bfloat16`), Stage 1.1 shared KV cache, and CPU `float32` working maps (`~1.2 MB` for $320 \times 240$) are held across the questions belonging to that image (average `3.055` questions/image).
- **Image-Boundary Release:** At the end of each image group (inside `finally:`), `image` and `cached_visual_features` are deleted, `pipeline.reset_cache()` clears the completed image's cached tensors, `gc.collect()` sweeps any cyclic Python objects once per image, and `check_and_run_cuda_memory_watchdog()` invokes `torch.cuda.empty_cache()` if `allocated / total > 0.85`.

---

## 14. Residual Risks and Limitations

1. **Real-Model GPU Runtime & VRAM Telemetry Not Measured Locally (`NOT_MEASURED`):**
   - Per mandatory local/Colab governance, `5CD-AI/Vintern-1B-v3_5` was not loaded locally. Real-model wall-clock latency savings and real-GPU `memory_allocated` / `memory_reserved` profiles remain **`NOT_MEASURED`** until Gate B execution on the 20-QA real-model validation slice.
2. **Watchdog Metric Scope (`memory_allocated / total_memory`):**
   - As specified by `TICKET-OPT-02` in [`docs/research/PERFORMANCE_OPTIMIZATION_HANDOFF_v1.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/PERFORMANCE_OPTIMIZATION_HANDOFF_v1.md#L118), the watchdog threshold is evaluated at image boundaries after `pipeline.reset_cache()` and `gc.collect()` using `torch.cuda.memory_allocated(dev_idx) / torch.cuda.get_device_properties(dev_idx).total_memory > 0.85`.

---

## 15. Gate A Status and Gate B Readiness Statement

- **Gate A (Local Synthetic / Mock Verification):** **COMPLETE — PASS** (`9/9` tests in `tests.test_opt02_memory_cleanup_policy` + `74/74` regression tests passing).
- **Gate B (20-QA Real-Model Validation Slice):** **NOT_MEASURED / PENDING.**
  - Per the approved small-first execution plan (`OPT-06` $\rightarrow$ `OPT-03` $\rightarrow$ `OPT-02` $\rightarrow$ `OPT-01`), Gate B real-model execution remains gated on independent QA acceptance of `TICKET-OPT-02`, implementation and acceptance of `TICKET-OPT-01` (Batched Positive/Negative Decoding $B=2$), and explicit Project Owner authorization.

---

## 16. Handoff Notes for Independent QA (Anti) and Methodology Review (ChatGPT)

1. **Files Modified / Created for Review:**
   - [`scripts/benchmark/sg_vcd/pipeline.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L360-L383) (`reset_cache` at lines `360–383`, per-question cleanup removal at lines `1050–1056`)
   - [`scripts/benchmark/sg_vcd/attention_extractor.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/attention_extractor.py#L782-L785) (per-question `empty_cache` removal at lines `782–785`)
   - [`scripts/benchmark/evaluate_vitextvqa.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/evaluate_vitextvqa.py#L900-L1060) (watchdog and image-boundary cleanup helpers at lines `900–1060`, `run_benchmark` integration at lines `1208–1216` and `1252–1449`)
   - [`tests/test_opt02_memory_cleanup_policy.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt02_memory_cleanup_policy.py) (`Tests A–I`)
2. **Reproduction Commands for Independent QA:**
   ```powershell
   & "..\Document AI\VLM_Reasoning\venv\Scripts\python.exe" -m unittest tests.test_opt02_memory_cleanup_policy -v
   & "..\Document AI\VLM_Reasoning\venv\Scripts\python.exe" -m unittest tests.test_opt03_working_map_cache tests.test_opt06_image_cache_identity tests.test_sg_vcd tests.test_adversarial_spatial_processing tests.test_tier4_workloads.TestTier4RealWorldWorkloads.test_05_anti_hardcoding_and_heuristics_audit -v
   ```
3. **Git Status:** Uncommitted in working tree per project governance; awaiting independent QA review by `Antigravity (Anti)`.

