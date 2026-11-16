# TASK 19 — GATE B READINESS & PROJECT OWNER DECISION BRIEF

**Document ID:** `TASK_19_GATE_B_READINESS_OWNER_DECISION_BRIEF`  
**Version:** 1.1-corrected (Task 19R Closeout)  
**Primary Author:** 4argon — Senior ML Systems Engineer  
**Independent QA Reviewer:** Antigravity (Anti) — Senior ML Systems Reviewer  
**Final Authority:** Project Owner (User)  
**Governance Scope:** Gate A Evaluation & Decision Preparation  
**Execution Authorization:** Gate B Execution is **STRICTLY PROHIBITED / NOT AUTHORIZED**  
**Date:** 2026-11-09  

---

## 1. EXECUTIVE SUMMARY

This decision brief provides the Project Owner with a consolidated, technically defensible readiness status and an explicit decision framework (OD-01 through OD-05) required prior to considering Gate B authorization (the first real-model integration experiment of SG-VCD on ViTextVQA using `5CD-AI/Vintern-1B-v3_5`).

All Phase 2 optimization and hardening tasks at Gate A have completed successfully:
1. **Core Pipeline Optimizations (OPT-01 through OPT-06):** All five optimization modules (Batched Dual-Branch Decoding OPT-01, Selective Working Map Cache OPT-03, GPU-affinity Vectorization OPT-04, Conservative Memory Cleanup OPT-02, and Image Cache Identity Hardening OPT-06) have been implemented, hardened, and verified with 144 passing unit and regression tests.
2. **Canonical R-03 Signed-Logit Implementation (Task 18B):** The mathematically approved canonical R-03 contrastive scoring formula \(z_{\mathrm{final}} = (1 + \alpha_t) z_{\mathrm{pos}} - \alpha_t z_{\mathrm{neg}}\) has been fully integrated into `scripts/benchmark/sg_vcd/dynamic_cd.py`, with the historical ReLU-clipped formula cleanly separated as an explicitly selectable legacy mode (`--scoring-mode historical_relu`). The CAPC candidate-support filtering has been strictly decoupled from logit scoring.
3. **Colab Resource Preflight Tooling (Tasks 18A / 18A Closeout):** Diagnostic preflight script `scripts/benchmark/colab_gate_b_preflight.py` has been verified offline. Static weight footprints are derived: core architecture parameters total 818M (1.64 GB in BF16), with a conservative untied ceiling of 1.05B (2.10 GB in BF16). Active tensor memory is analytically estimated at \(\approx 3.0\text{--}3.5\text{ GB}\). PyTorch allocator reservation (\(\approx 4.5\text{--}5.5\text{ GB}\)) and peak VRAM during real inference (\(\approx 5.0\text{--}6.0\text{ GB}\)) are strictly classified as `NOT_MEASURED` pending authorized hardware execution.
4. **20-QA Candidate Integration Manifest (Tasks 16 & 17):** Prepared at `data/vitextvqa/manifest_integration_20qa.candidate.json` (SHA256: `9e9dc05dd87c2fb2de254cc1022dad41d5e745f8070560ddd98c648fab5b269a`, 18,179 bytes, 854 lines), covering 20 QA rows across exactly **6 unique images: `22.jpg`, `30.jpg`, `68.jpg`, `76.jpg`, `111.jpg`, `126.jpg`**, with all image files, bounding boxes, and ground-truth annotations independently verified.
5. **Hardware Admission Boundary (Task 18A Restriction):** On Google Colab free-tier NVIDIA Tesla T4 GPUs (Turing architecture, Compute Capability `sm_75`), native hardware BF16 Tensor Cores are not present. Programmatic admission check `native_bf16_support` in `colab_gate_b_preflight.py` evaluates to `FAIL` on T4, producing an overall admission verdict of **`REJECT`**. Task 18A did NOT authorize or implement automatic FP16 fallback. Any transition to FP16 requires a dedicated Project Owner decision and Gate A numerical parity re-validation.

**Current Operational Status:**
- Production code, diagnostic tools, and test suites are **READY** at Gate A.
- Gate B execution remains **BLOCKED** pending the Project Owner's explicit resolution of five open decisions: **OD-01** (Reduced MVBS Baseline Matrix), **OD-02** (Answer Denominator Policy & Metric Protocols), **OD-03** (20-QA Manifest Freezing), **OD-04** (P0 Spec v1.1 Formal Adoption & Status Schema), and **OD-05** (CRG Shared ROI Policy).
- Under no circumstances will real-model weights be downloaded or inference initiated prior to the Project Owner's written determination and authorization.

---

## 2. COMPREHENSIVE GATE B READINESS CHECKLIST

The following matrix audits the readiness of all functional components, governance gates, and experimental prerequisites:

| Component / Subsystem | Status | Primary Artifact / Evidence | Condition / Technical Details |
| :--- | :---: | :--- | :--- |
| **OPT-01 Batched Dual-Branch Decoding** | `READY` | `scripts/benchmark/sg_vcd/pipeline.py`<br>`tests/test_opt01_batched_dual_branch_decoding.py` | Parity verified against sequential decoding; attention mask semantics hardened (`TASK_11`, `TASK_12`, `TASK_13`). |
| **OPT-02 Memory Cleanup Policy** | `READY` | `scripts/benchmark/sg_vcd/pipeline.py`<br>`tests/test_opt02_memory_cleanup_policy.py` | Conservative garbage collection interval (\(k=10\)), OOM threshold trigger (\(>85\%\)), verified (`TASK_09`, `TASK_10`). |
| **OPT-03 Working Map Cache** | `READY` | `scripts/benchmark/sg_vcd/feature_cache.py`<br>`tests/test_opt03_working_map_cache.py` | Avoids redundant cross-attention recomputation; verified (`TASK_08`). |
| **OPT-04 GPU-Affinity Vectorization** | `READY` | `scripts/benchmark/sg_vcd/adaptive_propagator.py`<br>`tests/test_opt04_gpu_affinity_vectorization.py` | Vectorized bounding-box IoU & area calculations on target device; verified (`TASK_14`, `TASK_15`). |
| **OPT-06 Image Cache Hardening** | `READY` | `scripts/benchmark/sg_vcd/feature_cache.py`<br>`tests/test_opt06_image_cache_identity.py` | Robust image identity caching with SHA256 and shape verification (`TASK_07_FIX`, `TASK_07R`). |
| **Canonical R-03 Signed-Logit Scoring** | `READY` | `scripts/benchmark/sg_vcd/dynamic_cd.py`<br>`tests/test_canonical_r03_signed_logits.py` | Implements \((1+\alpha_t)z_{\mathrm{pos}} - \alpha_t z_{\mathrm{neg}}\); historical ReLU mode preserved; CAPC candidate support decoupled (`TASK_18B`). |
| **Regression Test Suite** | `READY` | 144 passing unit & regression tests | Full mock/synthetic test suite passes cleanly with zero failures. |
| **Colab Resource Preflight Tool** | `READY` | `scripts/benchmark/colab_gate_b_preflight.py` | Offline diagnostic script verified; evaluates 6 programmatic checks: `cuda_available`, `total_vram >= 8GB`, `physical_free_vram >= 6GB`, `native_bf16_support`, `host_ram >= 8GB`, `disk_free >= 15GB` (`TASK_18A`). |
| **Hardware Admission on T4** | `REJECT (Native BF16)` | `colab_gate_b_preflight.py#L152-L215` | Tesla T4 lacks native BF16 matrix cores (CC 7.5 < 8.0). Programmatic check fails. FP16 requires Owner decision and Gate A parity re-validation. |
| **Candidate Manifest Traceability** | `READY` | `data/vitextvqa/manifest_integration_20qa.candidate.json` | 20 QA across 6 unique images (`22.jpg`, `30.jpg`, `68.jpg`, `76.jpg`, `111.jpg`, `126.jpg`); verified SHA256 `9e9dc05dd87c2fb2de...`; images verified on local filesystem (`TASK_16`, `TASK_17`). |
| **MVBS Baseline Matrix Selection** | `OWNER DECISION REQUIRED` | Research Plan v2, Sec. 3;<br>Handoff v1, Sec. 6 | **OD-01:** Owner must select Reduced Matrix for 20-QA pilot. |
| **Answer Denominator Policy** | `OWNER DECISION REQUIRED` | `scripts/benchmark/metrics.py`;<br>`evaluate_vitextvqa.py` | **OD-02:** Owner must select Fixed Denominator vs Conditional Quality vs Dual Reporting, and verify dual GT sources. |
| **Manifest Production Freeze** | `OWNER DECISION REQUIRED` | Candidate Manifest | **OD-03:** Owner must authorize freezing candidate to production manifest. |
| **P0 Remediation Spec v1.1 Adoption** | `OWNER DECISION REQUIRED` | `REMEDIATION_SPEC_P0_v1.md` | **OD-04:** Owner must formally ratify Spec v1.1 (Canonical Signed Logits & Orthogonal Status Schema). |
| **CRG Shared ROI Policy** | `OWNER DECISION REQUIRED` | `Proposed_Method_SG_VCD.md`;<br>Handoff v1, Sec. 5 & 6.3 | **OD-05:** Owner must choose Shared Predicted ROI vs Independent ROI for contrastive baseline comparisons. |
| **Gate B Real-Model Execution** | `BLOCKED` | Governance Boundary | Strictly blocked until OD-01 through OD-05 are resolved and Owner grants explicit written authorization. |

---

## 3. PROJECT OWNER DECISION FRAMEWORK (OD-01 THROUGH OD-05)

### OD-01: Reduced MVBS Baseline Matrix for Gate B (20-QA Pilot)

#### 1. Authoritative Background & Context
Research Plan v2 Section 3 and `PERFORMANCE_OPTIMIZATION_HANDOFF_v1.md` Section 5 define the Multi-View Baseline Suite (MVBS) under strict categorical classifications:
- `BACKBONE_CONTROL`: Vanilla Vintern (`CTRL-0`, `5CD-AI/Vintern-1B-v3_5` without guidance).
- `PAPER_METHOD_BASELINE`:
  - `ViCrop` (ICLR 2025): Stage 1 Visual Localization SOTA.
  - `VCD` (CVPR 2024): Stage 2 Visual Contrastive Decoding SOTA (Gaussian noise prior).
  - `CRG` (ECCV 2024): Contrastive Region Guidance (masked region prior).
  - `Localization Heads` (ECCV 2024): Stage 1 baseline (deferred in Handoff Section 6).
  - `M3ID` (EMNLP 2024): Stage 2 multimodal vs text-only contrast (deferred in Handoff Section 6).
- `SG_VCD_METHOD`: Full Two-Stage Saliency-Guided Contrastive Decoding (Stage 1 Saliency Propagation + Stage 2 Canonical Signed Dynamic CD).
- `ABLATION`: Stage 1 only (`ABL-1`), Stage 2 only (`ABL-2`), uncalibrated guidance \(\alpha_t\).
- `DIAGNOSTIC`: Oracle ROI + Stage 2 Contrastive Decoding (using ground-truth bounding box as visual crop).

In `PERFORMANCE_OPTIMIZATION_HANDOFF_v1.md` Section 6, the proposed **Reduced MVBS** comprises: `CTRL-0`, `ViCrop`, `VCD`, `CRG`, `SG-VCD`, and `Oracle ROI Diagnostic`.

However, Gate B is an **infrastructure integration pilot** on 20 QA, whose primary objective is verifying model loading, dual-branch KV-cache mechanics, token generation termination, and metric evaluation without runtime crashes. External paper baselines (`ViCrop`, `VCD`, `CRG`) currently lack verified adapters for `Vintern-1B-v3_5` in this codebase.

#### 2. Candidate Options for Gate B (20-QA Pilot)
- **Option 1A (Minimal Integration — 2 Arms):**
  - Arms:
    1. `CTRL-0` (Vanilla Vintern-1B-v3_5 Backbone Control)
    2. `SG-VCD` (Full proposed pipeline)
  - *Omitted Arms:* All external paper baselines (`ViCrop`, `VCD`, `CRG`) and `Oracle ROI Diagnostic`.
  - *Engineering Scope:* Zero external adapter dependencies. Verifies end-to-end model execution and dual-branch decoding with minimal GPU quota consumption.
- **Option 1B (Balanced Integration — 3 Arms) [Engineer's Recommendation]:**
  - Arms:
    1. `CTRL-0` (Vanilla Vintern-1B-v3_5 Backbone Control)
    2. `Oracle ROI Diagnostic` (`DIAGNOSTIC` upper-bound reference using GT bounding box)
    3. `SG-VCD` (Full proposed pipeline)
  - *Omitted Arms:* External paper baselines (`ViCrop`, `VCD`, `CRG`).
  - *Engineering Scope:* Implements the Oracle ROI diagnostic condition natively using existing manifest `answer_region_bbox` annotations. Isolates Stage 1 visual localization error from Stage 2 contrastive decoding dynamics without requiring unverified external baseline ports.
- **Option 1C (Full Reduced MVBS — 6 Arms):**
  - Arms: `CTRL-0`, `ViCrop`, `VCD`, `CRG`, `SG-VCD`, and `Oracle ROI Diagnostic`.
  - *Engineering Scope:* Requires implementing, porting, and validating code for `ViCrop`, `VCD`, and `CRG` on Vintern-1B-v3_5 prior to launching Gate B.

#### 3. Trade-offs & Scientific Rationale
- *Option 1A* guarantees the fastest plumbing verification, but if SG-VCD yields lower accuracy than expected, it provides no diagnostic data to distinguish whether the failure was caused by Stage 1 spatial localization or Stage 2 logit contrast.
- *Option 1B* introduces the essential `Oracle ROI Diagnostic` ceiling. If SG-VCD underperforms but Oracle ROI succeeds, the bottleneck is conclusively isolated to Stage 1 localization. If both fail, Stage 2 contrastive decoding is implicated. This diagnostic insight is achieved without the schedule delay of porting three external baselines.
- *Option 1C* prematurely front-loads external paper baseline porting before basic model loading and decoding plumbing are validated. Full MVBS evaluation is formally scheduled for Phase 3 (100-image screening pilot).

*Note on Execution Times:* Prior estimates of execution duration (e.g. 8–15 minutes) are unverified hypothetical planning assumptions. Actual runtime is `NOT_MEASURED` until an authorized run occurs.

#### 4. Engineer's Recommendation
**Adopt Option 1B (Balanced Integration: `CTRL-0`, `Oracle ROI Diagnostic`, `SG-VCD`).**

---

### OD-02: Answer Denominator Policy & Metric Protocols

#### 1. Authoritative Metric Definitions & Answer Sources
Inspection of `scripts/benchmark/metrics.py` and `scripts/benchmark/evaluate_vitextvqa.py` establishes the exact metrics implemented for ViTextVQA Stage 2 VQA evaluation:
- **Token-F1** (`compute_token_f1`): Precision/recall harmonic mean on normalized token sets.
- **ANLS** (`compute_anls`): Average Normalized Levenshtein Similarity with threshold \(\tau = 0.5\).
- **Substring Match** (`compute_substring_match`): Binary containment of ground truth in prediction.
- **Exact Match** (`compute_exact_match`): Strict string equality after answer normalization.
- **VQA Consensus Accuracy** (`compute_vqa_accuracy`): \(\min(1.0, \frac{\text{count}}{3})\).

*(Loose F1 and Average Precision are not part of the established SG-VCD evaluation protocol and must not be used).*

Furthermore, the benchmark explicitly supports **two separate annotation answer sources**:
1. `gt_answers`: Original ViTextVQA dataset ground-truth strings.
2. `verified_all_answers`: Human-verified multi-reference ground-truth strings.

Following the Task 17 QA audit recommendation, **all metrics must be reported separately for both answer sources** (`em_gt` vs `em_verified`, `f1_gt` vs `f1_verified`, `anls_gt` vs `anls_verified`, `substring_gt` vs `substring_verified`). They must never be merged into a single unlabeled score.

#### 2. Candidate Denominator Options
During real-model inference, language models occasionally output empty strings, fail syntax parsers, exceed max length, or encounter runtime exceptions. The denominator policy determines how these cases are scored:

- **Option 2A (Conditional Quality — Effective Subset Denominator):**
  - *Unit of Evaluation:* Valid generation responses.
  - *Denominator:* \(N_{\mathrm{valid}} \le 20\).
  - *Treatment of Failures:* Empty strings, unparseable outputs, and runtime exceptions are excluded from the denominator.
  - *Formula:* \(\text{Score}_{\mathrm{cond}} = \frac{1}{N_{\mathrm{valid}}} \sum_{i \in \mathrm{Valid}} s_i\).
  - *Risk:* Severe positive selection bias. A method that crashes on difficult samples can artificially appear more accurate than a method that processes all samples.
- **Option 2B (Strict Fixed Manifest Denominator — Unconditional):**
  - *Unit of Evaluation:* Manifest items.
  - *Denominator:* Fixed to \(N=20\).
  - *Treatment of Failures:* Any missing prediction, empty string, malformed output, or runtime exception receives a score of \(s_i = 0.0\).
  - *Formula:* \(\text{Score}_{\mathrm{uncond}} = \frac{1}{20} \sum_{i=1}^{20} s_i\).
  - *Benefit:* Strict comparability and penalty for model fragility; eliminates selection bias.
  - *Risk:* A minor formatting bug could reduce overall scores, obscuring the textual quality of non-buggy outputs.
- **Option 2C (Dual Reporting: Primary Fixed Denominator + Secondary Diagnostic) [Engineer's Recommendation]:**
  - *Primary Headline Score:* Computed strictly under **Option 2B** (Fixed Manifest Denominator \(N=20\)).
  - *Secondary Diagnostic Score:* Reported under **Option 2A** (\(N_{\mathrm{valid}}\)) alongside an explicit **Failure Taxonomy Table** categorizing every non-valid sample (e.g. `runtime_exception`, `empty_string`, `repetition_collapse`, `timeout`).
  - *Answer Sources:* Both Option 2B and Option 2A scores are reported across both `gt_answers` and `verified_all_answers`.

#### 3. Engineer's Recommendation
**Adopt Option 2C (Dual Reporting with Strict Fixed Denominator as Primary).**

---

### OD-03: Candidate 20-QA Manifest Freezing

#### 1. Manifest Identity & Source Evidence
Inspection of the actual candidate file `data/vitextvqa/manifest_integration_20qa.candidate.json` reveals:
- **File Path:** `data/vitextvqa/manifest_integration_20qa.candidate.json`
- **SHA256:** `9E9DC05DD87C2FB2DE254CC1022DAD41D5E745F8070560DDD98C648FAB5B269A`
- **Byte Size:** 18,179 bytes
- **Line Count:** 854 lines
- **Total Samples:** 20 rows
- **Actual Unique Images (6 Images):**
  1. `22.jpg` (4 questions: QIDs 74, 75, 76, 77)
  2. `30.jpg` (3 questions: QIDs 107, 108, 109)
  3. `68.jpg` (4 questions: QIDs 245, 246, 247, 248)
  4. `76.jpg` (3 questions: QIDs 267, 268, 269)
  5. `111.jpg` (2 questions: QIDs 348, 349)
  6. `126.jpg` (4 questions: QIDs 374, 375, 376, 377)
  *(Correction: `180.jpg` does not exist in this manifest; the image list is strictly `22.jpg`, `30.jpg`, `68.jpg`, `76.jpg`, `111.jpg`, `126.jpg`, confirming Task 16/17 historical evidence).*
- **Prompt Diversity & Clustering Analysis:**
  - Total QIDs: 20 unique QIDs.
  - Unique `(image, question)` text pairs: **16 distinct pairs**.
  - On `30.jpg`, QIDs 107, 108, and 109 share the exact question text `"cửa tiệm trong ảnh bán gì ?"` (each corresponding to distinct ground truth items: `bếp gas`, `bếp từ`, `hút mùi`).
  - On `126.jpg`, QIDs 375, 376, and 377 share the exact question text `"ở đây có những cửa hàng gì ?"` (each corresponding to distinct ground truth store names: `giầy đồ hiệu`, `anta sports`, `in thêu may cờ`).
  - All 6 image files exist locally with valid JPEG headers; all 20 annotations trace directly to verified annotations in `results/swintextspotter_auto_annot/answer_bboxes_400_verified.json`.

#### 2. Candidate Options
- **Option 3A (Freeze Candidate Manifest As-Is) [Engineer's Recommendation]:**
  - Promote `manifest_integration_20qa.candidate.json` to production `manifest_integration_20qa.json`.
  - Accept intra-image clustering on `30.jpg` and `126.jpg` as acceptable for Gate B. Because Gate B is a technical integration pilot, repeated queries on the same image validate image-caching efficiency and state isolation (OPT-06).
- **Option 3B (Re-sample for 20 Unique Images):**
  - Re-sample questions from the dataset to enforce 20 distinct images (1 QA per image), eliminating prompt clustering.
  - Generates Candidate v2, requiring re-running Task 16 preflight and Task 17 QA audits.

#### 3. Engineer's Recommendation
**Adopt Option 3A (Freeze Candidate Manifest As-Is to Production upon Owner Approval).**  
*(Note: Candidate remains strictly unfrozen and unmodified during Task 19).*

---

### OD-04: Formal Adoption of Remediation Specification P0 v1.1 & Orthogonal Status Schema

#### 1. Mathematical Formulation Reconciliation
In initial research drafts (P0 v1.0), dynamic contrastive decoding used a ReLU-clipped formulation:
\[
z_{\mathrm{final}} = z_{\mathrm{pos}} + \alpha_t \cdot \mathrm{ReLU}(z_{\mathrm{pos}} - z_{\mathrm{neg}})
\]
Methodology review proved this formula mathematically truncates contrastive penalties whenever \(z_{\mathrm{neg}} > z_{\mathrm{pos}}\), preventing the contrastive branch from penalizing visual hallucination tokens that have higher baseline logits in the negative branch.

The approved canonical R-03 formulation restores signed logit scaling:
\[
z_{\mathrm{final}} = (1 + \alpha_t) z_{\mathrm{pos}} - \alpha_t z_{\mathrm{neg}}
\]
Task 18B implemented this canonical formula in `scripts/benchmark/sg_vcd/dynamic_cd.py`, verified it with 9 dedicated unit tests, preserved the legacy ReLU behavior under an explicit `--scoring-mode historical_relu` flag, and decoupled CAPC candidate support.

#### 2. Orthogonal Status Schema Specification (P0 v1.1 R-04)
`REMEDIATION_SPEC_P0_v1.md` Section R-04 specifies an orthogonal, multidimensional result status contract to prevent conflating execution failure, invalid output, and missing data:

| Status Dimension | Allowed Values | Semantic Meaning |
| :--- | :--- | :--- |
| `run_status` | `completed`, `partial`, `failed`, `missing_image`, `skipped` | `completed`: all configured stages finished without execution exception.<br>`partial`: one stage completed, another failed.<br>`failed`: execution aborted prior to producing stage outputs.<br>`missing_image` / `skipped`: input missing or intentionally bypassed. |
| `answer.status` | `valid`, `invalid`, `not_attempted` | `valid`: generated text passed validator.<br>`invalid`: output produced but failed validation (e.g. empty or corrupt).<br>`not_attempted`: no answer produced due to pipeline failure. |
| `localization.status`| `valid`, `failed`, `not_produced`, `not_applicable` | `valid`: valid ROI produced.<br>`failed`: localizer exception.<br>`not_produced`: zero boxes returned.<br>`not_applicable`: method does not produce localization. |
| `localization.applicability` | `applicable`, `not_applicable` | Whether the method variant is configured to output bounding boxes. |
| `localization.roi_source` | `predicted`, `oracle_gt`, `none` | Identifies whether the ROI was model-predicted or an oracle diagnostic. |
| Per-metric `status` | `computed`, `not_applicable`, `unavailable`, `pending_policy` | `computed`: valid metric value (including genuine 0.0).<br>`unavailable`: required inputs missing.<br>`pending_policy`: aggregation policy unratified. |
| `failure` | Object or `null` | Records `{stage, error_category, message, retryable}`. |

#### 3. Specification-to-Code Discrepancy Analysis
Inspection of the active benchmark runner `scripts/benchmark/evaluate_vitextvqa.py` reveals a clear **specification-to-code mismatch**:
- Current `evaluate_vitextvqa.py` records a flat dictionary (`sample_key`, `pred_answer`, `pred_bbox`, `stage1_metrics`, `stage2_metrics`, `latency_s`).
- When an exception occurs during inference (`except Exception as e:`), the runner sets `pred_answer = ""` and `pred_bbox = [0, 0, W, H]`, evaluates metrics on the empty string (yielding 0.0), and updates the aggregator count unconditionally. It does not populate nested status fields (`run_status`, `answer.status`, `failure`).
- **Impact:** While current code naturally implements an unconditional denominator (Option 2B), it lacks granular diagnostic status classification. Implementing the full P0 R-04 nested schema requires updating `CheckpointManager` and record serialization.

#### 4. Candidate Options
- **Option 4A (Formally Ratify Remediation Spec v1.1) [Engineer's Recommendation]:**
  - Officially approve P0 Remediation Spec v1.1. Ratifies Canonical R-03 Signed Logits as production default. Establishes the orthogonal status schema as the governing specification, scheduling the record schema update for Phase 3.
- **Option 4B (Retain Spec v1.0 and Treat Signed Logits as Experimental):**
  - Retains ReLU-clipped scoring as primary and leaves signed logits as an experimental branch.

#### 5. Engineer's Recommendation
**Adopt Option 4A (Formally Ratify Remediation Spec v1.1).**

---

### OD-05: Contrastive Region Guidance (CRG) Shared ROI Policy & Conceptual Disambiguation

#### 1. Delineation of Spatial Localization Concepts
To eliminate methodological ambiguity, four distinct concepts must be rigorously distinguished:
1. **`Oracle ROI Diagnostic` (`DIAGNOSTIC`):**
   - Derived directly from ground-truth annotations (`answer_region_bbox`).
   - *Purpose:* Measures theoretical Stage 2 performance ceiling assuming 100% perfect visual localization. It is an independent diagnostic condition and **never** a model-predicted ROI.
2. **`SG-VCD Predicted ROI` (`SG_VCD_METHOD`):**
   - Dynamically generated by SG-VCD Stage 1 (Adaptive Visual Saliency Propagation) using attention heatmaps from the vision backbone.
   - *Purpose:* Real-world autonomous operation without ground-truth boxes.
3. **`Shared Predicted ROI Protocol` (Controlled Stage 2 Algorithmic Comparison):**
   - The bounding box predicted by SG-VCD Stage 1 is frozen and fed identically into comparison Stage 2 contrastive decoders (such as CRG).
   - *Purpose:* Isolates Stage 2 decoding algorithms by holding visual localization constant across methods.
   - *Authoritative Reference:* `PERFORMANCE_OPTIMIZATION_HANDOFF_v1.md`, Section 5, Table row "CRG":
     > *"In ViTextVQA, CRG must use the exact same frozen shared ROI as SG-VCD for fair comparison."*
   - *Authoritative Reference:* Handoff Section 6.3:
     > *"To isolate Stage 2 decoding mechanisms from Stage 1 localization differences, CRG should be evaluated using the exact same bounding boxes predicted by SG-VCD Stage 1."*
4. **`Independent ROI Protocol` (End-to-End System Evaluation):**
   - Each comparison baseline uses its own native localization mechanism (e.g. CRG with text-heuristic crop, or external systems).
   - *Purpose:* Measures composite end-to-end performance in real-world deployment.

#### 2. Candidate Options for Baseline Evaluation
- **Option 5A (Shared Predicted ROI Protocol) [Engineer's Recommendation for Gate B / Stage 2 Ablations]:**
  - Contrastive baseline arms (such as CRG) evaluate Stage 2 using the exact bounding box predicted by SG-VCD Stage 1. Isolates contrastive decoding mathematics from spatial localization errors.
- **Option 5B (Independent ROI Protocol):**
  - Each method generates its own bounding box independently.
- **Option 5C (Dual ROI Protocol) [Engineer's Recommendation for Phase 3 Screening]:**
  - Shared Predicted ROI protocol is used for Stage 2 algorithmic ablations; Independent ROI protocol is used for end-to-end benchmarking against external literature.

#### 3. Engineer's Recommendation
**Adopt Option 5A for Gate B integration; plan Option 5C for full Phase 3 screening.**

---

## 4. CONSOLIDATED DECISION MATRIX FOR PROJECT OWNER

| Decision ID | Topic | Candidate Options | Engineer Recommendation | Risk / Trade-off Summary |
| :---: | :--- | :--- | :---: | :--- |
| **OD-01** | Reduced MVBS Baseline Matrix | **1A:** Minimal (2 arms: Vanilla, SG-VCD)<br>**1B:** Balanced (3 arms: Vanilla, Oracle Diagnostic, SG-VCD)<br>**1C:** Full (6 arms: all reduced baselines) | **Option 1B (Balanced: 3 Arms)** | *1A* lacks diagnostic anchor; *1C* requires unported external baseline adapters before Gate B; *1B* provides Oracle diagnostic ceiling while avoiding external dependencies. *(Durations unmeasured).* |
| **OD-02** | Answer Denominator Policy | **2A:** Conditional Quality (\(N_{\mathrm{valid}} \le 20\))<br>**2B:** Strict Fixed Denominator (\(N=20\))<br>**2C:** Dual Reporting (Fixed primary + Conditional diagnostic) | **Option 2C (Dual Reporting)** | *2A* introduces positive selection bias (dropping crashes inflates accuracy); *2B* can penalize minor format bugs; *2C* upholds scientific publication rigor while maintaining diagnostic visibility across both GT sources. |
| **OD-03** | Manifest Production Freeze | **3A:** Freeze candidate as-is (\(20\text{ QA} / 6\text{ images}\))<br>**3B:** Re-sample for 20 distinct images | **Option 3A (Freeze As-Is)** | Candidate SHA256 verified (`9E9DC05D...`); images (`22, 30, 68, 76, 111, 126`) and GT verified. *3A* unblocks Gate B; clustering tests feature caching (OPT-06); *3B* delays schedule with zero plumbing benefit. |
| **OD-04** | P0 Remediation Spec v1.1 Adoption | **4A:** Ratify v1.1 (Canonical Signed Logits & Status Schema)<br>**4B:** Retain v1.0 (Signed Logits as ablation) | **Option 4A (Ratify Spec v1.1)** | *4A* repairs mathematical truncation of hallucinated tokens; establishes orthogonal status schema contract; notes evaluator flat schema mismatch. *4B* retains theoretically flawed ReLU clipping. |
| **OD-05** | CRG Shared ROI Policy | **5A:** Shared Predicted ROI for Gate B<br>**5B:** Independent ROI<br>**5C:** Dual Protocol (Shared for ablation, Indep. for E2E) | **Option 5A (Gate B)**<br>*(Option 5C for full screening)* | *5A* strictly isolates Stage 2 decoding mechanics from Stage 1 localization variance; *5B* confounds bounding box jitter with logit scoring errors. Oracle ROI remains a separate diagnostic condition. |

---

## 5. PROPOSED POST-DECISION ACTION SEQUENCE

Upon the Project Owner's recording of decisions for OD-01 through OD-05, the following strict sequential protocol is proposed:

```
[Project Owner Resolves OD-01 .. OD-05]
                 │
                 ▼
[Action 1: Manifest Freeze (if 3A approved)]
Copy candidate manifest to:
data/vitextvqa/manifest_integration_20qa.json
Verify SHA256 checksum match.
                 │
                 ▼
[Action 2: Remote Colab Diagnostic Preflight Probe]
Run scripts/benchmark/colab_gate_b_preflight.py on target Colab instance.
Evaluate programmatic checks: CUDA available, Total VRAM >= 8GB,
Free VRAM >= 6GB, Native BF16 support, Host RAM >= 8GB, Disk Free >= 15GB.
If T4 instance assigned (native BF16 fails), report REJECT to Owner.
NO WEIGHT DOWNLOADS OR MODEL INFERENCE.
                 │
                 ▼
[Action 3: Submit Gate B Execution Authorization Request]
Submit preflight diagnostic JSON report and final configuration to Project Owner.
AWAIT FORMAL WRITTEN PROJECT OWNER GATE B AUTHORIZATION.
                 │
                 ▼
[Action 4: Gate B Execution (Only Upon Owner Authorization)]
Execute 20-QA integration run on target Colab instance using:
python scripts/benchmark/evaluate_vitextvqa.py --dataset <manifest> ...
Generate output predictions, execution trace, and dual-denominator metrics.
```

---

## 6. DECLARATION OF COMPLIANCE

The Primary Engineer (4argon) certifies:
- No real-model weights were downloaded or instantiated.
- No model inference was executed locally or remotely.
- No Colab benchmark was initiated.
- The candidate manifest remains unfrozen pending Project Owner decision OD-03.
- Production code and tests remain completely untouched during this documentation task.
- Gate B execution remains strictly **NOT AUTHORIZED**.
