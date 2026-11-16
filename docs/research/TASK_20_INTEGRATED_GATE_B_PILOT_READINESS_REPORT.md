# TASK 20 — INTEGRATED GATE B PILOT READINESS REPORT

**Project:** SG-VCD / ViTextVQA  
**Primary Engineer:** 4argon  
**Independent Reviewer:** Antigravity (Anti)  
**Approval Authority:** Project Owner  
**Execution Mode:** Local implementation + synthetic/mock validation only  
**Execution Gate:** Gate A  
**Gate B Authorization Status:** **NOT AUTHORIZED (STRICTLY PROHIBITED)**  
**Date:** 2026-11-11  

---

## 1. Executive Summary & Authorization Boundaries

In accordance with Project Owner decisions formally approved in Task 19R and recorded in [`docs/research/PROJECT_OWNER_DECISIONS_TASK20.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/PROJECT_OWNER_DECISIONS_TASK20.md), Task 20 completes the full engineering implementation and local validation required to prepare the repository for the 3-arm Gate B integration pilot.

### Strict Governance & Execution Boundaries
- **Gate B Authorization Status:** Gate B remains **STRICTLY NOT AUTHORIZED**.
- **Model Loading Restrictions:** No real `5CD-AI/Vintern-1B-v3_5` model weights were downloaded or loaded into memory.
- **Inference Restrictions:** Zero real-model forward passes were executed.
- **Hardware & Cloud Restrictions:** No Google Colab runtime was launched, no remote hardware probes were executed, and no GPU VRAM beyond local synthetic testing was allocated.
- **Scope Restrictions:** No external baselines (CRG, ViCrop, M3ID) were imported or modified. All previous performance optimization changes (OPT-01 through OPT-06) and Task 18B canonical R-03 signed-logit implementations have been preserved without regression.

---

## 2. Summary of Workstream Execution

### Workstream A: Owner Decisions Formalization & Manifest Promotion (OD-01 to OD-05)
1. **Decision Record Creation:**
   - Authored [`docs/research/PROJECT_OWNER_DECISIONS_TASK20.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/PROJECT_OWNER_DECISIONS_TASK20.md) detailing OD-01 (Option 1B: 3-arm pilot), OD-02 (Option 2C: Dual reporting), OD-03 (Option 3A: Freeze 20-QA manifest as-is), OD-04 (Option 4A: Formally adopt P0 Remediation Spec v1.1), and OD-05 (Option 5A: Shared predicted ROI for Stage 2 comparisons; Oracle ROI strictly as separate diagnostic).
2. **Manifest Promotion (OD-03):**
   - Candidate manifest `data/vitextvqa/manifest_integration_20qa.candidate.json` was promoted to `data/vitextvqa/manifest_integration_20qa.json`.
   - Verified exact byte-for-byte identity:
     - Candidate SHA256: `9e9dc05dd87c2fb2de254cc1022dad41d5e745f8070560ddd98c648fab5b269a`
     - Promoted SHA256: `9e9dc05dd87c2fb2de254cc1022dad41d5e745f8070560ddd98c648fab5b269a`
     - Size: 18,179 bytes, 854 lines, 20 QA samples across 6 unique images (`22.jpg`, `30.jpg`, `68.jpg`, `76.jpg`, `111.jpg`, `126.jpg`).
     - Candidate file is retained intact.

### Workstream B: Production Codebase Updates (3 Arms & Oracle ROI)
1. **Pipeline Oracle ROI Support ([`scripts/benchmark/sg_vcd/pipeline.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py)):**
   - Added `oracle_bbox: Optional[Sequence[int]] = None` to `SGVCDPipeline.run()`.
   - When provided, Stage 1 extraction, selection, and propagation are bypassed; `selection.pixel_bbox` and `stage1_bbox` are set directly to `oracle_bbox`, and `spatial_confidence` is forced to `1.0`. Stage 2 calibrated contrastive decoding operates on the ground-truth crop.
2. **Evaluator 3-Arm Support ([`scripts/benchmark/evaluate_vitextvqa.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/evaluate_vitextvqa.py)):**
   - Added `--manifest` argument and native handling for manifests with `"samples"` directly, resolving image paths locally in `data/vitextvqa/st_images/` without network calls.
   - Updated `--method` to support `["ctrl_0", "baseline", "oracle_roi", "oracle_diagnostic", "sg_vcd", "cerv"]`.
   - Cleanly routes the three approved Gate B arms:
     - `CTRL-0`: Uncropped Vintern-1B backbone baseline.
     - `ORACLE_ROI`: Oracle crop (`answer_region_bbox`) followed by Stage 2 CD.
     - `SG-VCD`: Full 2-stage SG-VCD pipeline.
   - Rigorous bounding box validation: if `oracle_bbox` is invalid ($x_2 \le x_1$ or $y_2 \le y_1$), records `loc_status="failed"` and failure object `invalid_oracle_bbox`.

### Workstream C: Dual Reporting & Failure Taxonomy (OD-02 Policy 2C)
1. **Aggregator Enhancements ([`scripts/benchmark/metrics.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/metrics.py)):**
   - Updated `MetricsAggregator` to maintain dual accumulators:
     - Primary Fixed Manifest Denominator: $N_{\mathrm{manifest}} = 20$ (every sample counts; missing/empty/failed responses count as 0.0).
     - Secondary Conditional Diagnostic: $N_{\mathrm{valid}}$ (conditional quality over successfully generated responses; returns `"unavailable"` with `None` when $N_{\mathrm{valid}} = 0$).
   - Ground truth separation: `gt_answers` (primary annotation) and `verified_all_answers` (multi-annotator consensus) are tracked and reported separately.
   - Failure taxonomy counter: tracks categorised failures (`invalid_oracle_bbox`, `empty_generated_answer`, `inference_runtime_exception`, etc.).

### Workstream D: P0 Remediation Spec v1.1 Orthogonal Status Schema
1. **Per-Sample Record Conformance:**
   - Every evaluated question writes an orthogonal status record containing:
     - `run_id`, `sample_key`, `method_id`, `arm_type`, `run_status` (`completed` | `failed`), `is_valid` (boolean).
     - `answer`: `{status: "success" | "invalid" | "not_attempted", raw_text, normalized_text}`.
     - `localization`: `{status: "success" | "failed" | "not_attempted", pred_bbox, spatial_confidence}`.
     - `failure`: `{stage, error_category, error_message}` or `null`.
     - `metrics`: nested Stage 1 and Stage 2 metrics with per-metric `{value, status: "success" | "failed" | "uncomputable"}`.
   - Preserves all flat fields (`pred_answer`, `pred_bbox`, `stage1_metrics`, `stage2_metrics`, `latency_s`) ensuring 100% backward compatibility with all downstream report and checkpoint generators.

### Workstream E: Verification & Parity Test Suite
1. **Test Suite ([`tests/test_task20_integrated_pilot.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_task20_integrated_pilot.py)):**
   - Implemented 14 exhaustive unit and regression tests covering every mandatory validation point.
   - **Result:** **14/14 tests PASS (100% OK)** in 0.048s.
2. **Full Regression Suite:**
   - Ran `test_task20_integrated_pilot.py`, `test_canonical_r03_signed_logits.py`, `test_opt01_batched_dual_branch_decoding.py`, `test_opt02_memory_cleanup_policy.py`, `test_opt03_working_map_cache.py`, `test_opt04_gpu_affinity_vectorization.py`, `test_opt06_image_cache_identity.py`.
   - **Result:** **67/67 tests PASS (100% OK)** with zero regressions.
3. **End-to-End Mock Run:**
   - Ran `evaluate_vitextvqa.py --mock --manifest data/vitextvqa/manifest_integration_20qa.json --method ctrl_0`.
   - Verified that all 20 questions completed cleanly, checkpoints persisted, and `summary_metrics.json` recorded valid dual reporting under `OD-02_2C_DUAL_REPORTING`.

---

## 3. Mandatory 14-Point Validation Matrix

| # | Validation Item | Specification / Requirement | Test / Evidence | Result |
|---|---|---|---|---|
| 1 | Manifest Copy Identity | Byte-for-byte SHA256 match with candidate | `test_01_manifest_copy_identity_and_sha256` | **PASS** |
| 2 | Three-Arm Selection | Identical 20 QIDs evaluated across CTRL-0, ORACLE, SG-VCD | `test_02_three_arm_selection_identical_qids` | **PASS** |
| 3 | Oracle ROI Geometry | $x_2 > x_1$, $y_2 > y_1$, valid image coordinates | `test_03_oracle_roi_geometry_correctness` | **PASS** |
| 4 | Missing/Invalid Oracle ROI | Recorded as failure (`invalid_oracle_bbox`), no silent fallback | `test_04_missing_or_invalid_oracle_roi_failure_handling` | **PASS** |
| 5 | Predicted ROI vs Oracle ROI | Explicit parameter & diagnostic distinction | `test_05_distinction_oracle_roi_vs_predicted_roi` | **PASS** |
| 6 | GT Answers Separation | `gt_answers` vs `verified_all_answers` scored independently | `test_06_gt_answers_and_verified_all_answers_separation` | **PASS** |
| 7 | Fixed Denominator ($N=20$) | Primary reporting divides by 20 regardless of valid count | `test_07_fixed_denominator_n20_primary_reporting` | **PASS** |
| 8 | Conditional Denominator | Diagnostic reporting over $N_{\mathrm{valid}}$, handles $N_{\mathrm{valid}}=0$ | `test_08_conditional_denominator_and_zero_valid` | **PASS** |
| 9 | Failure Taxonomy Counter | Failure categories tracked in summary metadata | `test_09_failure_taxonomy_and_status_schema` | **PASS** |
| 10 | Per-Metric Status & Valid 0.0 | Incorrect valid answer gets 0.0 with status `success` | `test_10_per_metric_status_and_valid_zero_scores` | **PASS** |
| 11 | Checkpoint / Resume Parity | Completed records survive serialization and reload | `test_11_checkpoint_resume_integrity` | **PASS** |
| 12 | No Duplicate Aggregation | Resumed runs skip completed keys cleanly | `test_12_no_duplicate_sample_aggregation` | **PASS** |
| 13 | Canonical R-03 Default | Canonical signed logit formula default active | `test_13_canonical_r03_default_scoring_intact` | **PASS** |
| 14 | OPT-01 Batch Decoding | Batched dual-branch decode intact and functional | `test_14_opt01_batch_decoding_regression_intact` | **PASS** |

---

## 4. Source Files Modified & Created

### Modified Files:
- [`scripts/benchmark/sg_vcd/pipeline.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py): Added `oracle_bbox` support, Stage 1 bypass, and spatial confidence formatting.
- [`scripts/benchmark/metrics.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/metrics.py): Implemented OD-02 Policy 2C dual reporting in `MetricsAggregator`, failure taxonomy, and `normalize_vietnamese_text` alias.
- [`scripts/benchmark/evaluate_vitextvqa.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/evaluate_vitextvqa.py): Added `--manifest` CLI loader, 3-arm method selection (`ctrl_0`, `oracle_roi`, `sg_vcd`), P0 v1.1 orthogonal schema serialization, and checkpoint parity.

### Created Files:
- [`docs/research/PROJECT_OWNER_DECISIONS_TASK20.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/PROJECT_OWNER_DECISIONS_TASK20.md): Formal record of OD-01 through OD-05.
- [`data/vitextvqa/manifest_integration_20qa.json`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/data/vitextvqa/manifest_integration_20qa.json): Promoted frozen 20-QA manifest.
- [`tests/test_task20_integrated_pilot.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_task20_integrated_pilot.py): 14-point integrated validation test suite.
- [`docs/research/TASK_20_INTEGRATED_GATE_B_PILOT_READINESS_REPORT.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/TASK_20_INTEGRATED_GATE_B_PILOT_READINESS_REPORT.md): This report.

---

## 5. Engineering Declaration

Primary ML Systems Engineer (4argon) declares:
1. All Project Owner decisions OD-01 through OD-05 have been faithfully and rigorously implemented.
2. The codebase compiles cleanly, exhibits zero regressions across 70 unit and integration tests, and strictly implements P0 Remediation Specification v1.1.
3. Gate B remains completely unexecuted and unauthorized.
4. The implementation is ready for independent acceptance QA by Antigravity (Anti).

---

## 6. Task 20R Correction Addendum: P0 R-04 Status Schema Conformance

### 6.1 Context & Initial Schema Mismatch Identification
During the post-Task 20 audit review, a schema naming discrepancy was identified between the initial Task 20 evaluator implementation and the strict contract defined in P0 Remediation Specification v1.1, Section R-04:
1. **Status Vocabularies:** `answer.status` initially emitted `"success"`, whereas R-04 strictly requires `"valid"` (or `"invalid"` / `"not_attempted"`).
2. **Per-Metric Statuses:** Metric objects initially emitted `"status": "success"` and `"status": "uncomputable"`, whereas R-04 strictly permits only `{"computed", "not_applicable", "unavailable", "pending_policy"}`.
3. **Localization Enums & Fields:** `localization.applicability` (`applicable` vs `not_applicable`) and `localization.roi_source` (`predicted`, `oracle_gt`, `none`) were missing from the localization block.
4. **Failure Object Schema:** Failure objects used `"error_message"` instead of `"message"`, and omitted `"retryable"`.

### 6.2 Corrected R-04 Schema Implementation
In Task 20R, `scripts/benchmark/evaluate_vitextvqa.py` was updated to restore exact R-04 schema conformance:
- `run_status`: strictly limited to `{"completed", "partial", "failed", "missing_image", "skipped"}`.
- `answer.status`: strictly limited to `{"valid", "invalid", "not_attempted"}`.
- `localization`:
  - `status`: strictly limited to `{"valid", "failed", "not_produced", "not_applicable"}`.
  - `applicability`: strictly limited to `{"applicable", "not_applicable"}`.
  - `roi_source`: strictly limited to `{"predicted", "oracle_gt", "none"}`.
- **Three-Arm Mapping:**
  - `CTRL-0`: `localization.status = "not_applicable"`, `applicability = "not_applicable"`, `roi_source = "none"`, spatial metrics `status = "not_applicable"`.
  - `ORACLE_ROI`: `localization.status = "valid"`, `applicability = "applicable"`, `roi_source = "oracle_gt"`, spatial metrics `status = "computed"`.
  - `SG-VCD`: `localization.status = "valid"`, `applicability = "applicable"`, `roi_source = "predicted"`, spatial metrics `status = "computed"`.
- **Per-Metric Status:** strictly limited to `{"computed", "not_applicable", "unavailable", "pending_policy"}`. A valid answer with an incorrect prediction receives value `0.0` with `status: "computed"`.
- `failure`: object containing `{"stage", "error_category", "message", "retryable"}` or `null` for successful samples.
- **Backward Compatibility:** All existing fields (`is_valid`, `pred_answer`, `pred_bbox`, `stage1_metrics`, `stage2_metrics`, `latency_s`) remain preserved for seamless checkpoint and aggregator compatibility.

### 6.3 Test Verification
Targeted tests (`test_15_r04_exact_vocabulary_and_validation`, `test_16_r04_three_arm_status_mapping`, `test_17_r04_failure_object_schema`) were added to `tests/test_task20_integrated_pilot.py`. All 17 tests passed, and the full 70-test regression suite passed with 100% OK. Mock runs for all 3 arms verified exact serialization.

