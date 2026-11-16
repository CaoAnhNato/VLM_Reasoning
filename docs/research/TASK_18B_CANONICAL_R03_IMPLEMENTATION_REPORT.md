# TASK 18B — CANONICAL R-03 SIGNED LOGIT IMPLEMENTATION REPORT

**Project:** SG-VCD / ViTextVQA  
**Implementer:** 4argon — Senior ML Systems Engineer  
**Independent Reviewer:** Antigravity (Anti) — separate Task 18B QA  
**Approval Authority:** Project Owner  
**Priority:** HIGH — Schedule Acceleration  
**Mode:** Local implementation & synthetic/mock validation only (Gate A)  
**Real-model / Colab Execution Status:** **STRICTLY NOT AUTHORIZED**  
**Date:** `2026-11-09`

---

## 1. Executive Summary & Objective

This report documents the implementation and local synthetic validation of the approved **Canonical R-03 Signed-Logit Contrastive Decoding Formula** in the production SG-VCD Stage 2 path.

### 1.1 The Canonical R-03 Formula
Under the approved canonical R-03 specification, the Stage 2 contrastive score is computed as:

$$\boxed{z_{\mathrm{final}} = (1 + \alpha_t) z_{\mathrm{pos}} - \alpha_t z_{\mathrm{neg}}}$$

Algebraically equivalent to:
$$z_{\mathrm{final}} = z_{\mathrm{pos}} + \alpha_t (z_{\mathrm{pos}} - z_{\mathrm{neg}})$$

### 1.2 Separation from Historical ReLU-Clipped Dynamic-CD
The historical exploratory dynamic-CD implementation applied a non-linear $\mathrm{ReLU}$ clipping to the contrastive logit difference:
$$z_{\mathrm{legacy}} = z_{\mathrm{pos}} + \alpha_t \cdot \mathrm{ReLU}(z_{\mathrm{pos}} - z_{\mathrm{neg}})$$

When negative branch logits exceed positive branch logits ($z_{\mathrm{pos}} < z_{\mathrm{neg}}$), historical ReLU clipping clamped the delta to zero ($z_{\mathrm{legacy}} = z_{\mathrm{pos}}$), neutralizing the model's ability to penalize visual hallucinations or distractor priors. Under canonical R-03, negative differences are preserved, actively penalizing tokens where the negative (hallucinatory/distractor) view scores higher than the grounded evidence view.

The implementation establishes `scoring_mode="canonical_signed"` as the production default while preserving `scoring_mode="historical_relu"` as an explicitly selectable mode for exact historical reproducibility.

---

## 2. Mandatory Governance & Technical Context Loaded

Before code modification, the following binding repository governance and technical documents were consulted:
1. `AGENTS.md` (verified absent).
2. [`.github/copilot-instructions.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/.github/copilot-instructions.md)
3. [`.github/skills/anti-output-quality-review/SKILL.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/.github/skills/anti-output-quality-review/SKILL.md)
4. [`.github/skills/research-plan-compliance-review/SKILL.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/.github/skills/research-plan-compliance-review/SKILL.md)
5. [`.github/skills/local-colab-experiment-workflow/SKILL.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/.github/skills/local-colab-experiment-workflow/SKILL.md)
6. [`docs/research/PERFORMANCE_OPTIMIZATION_HANDOFF_v1.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/PERFORMANCE_OPTIMIZATION_HANDOFF_v1.md)
7. [`docs/Proposed_Method_SG_VCD.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/Proposed_Method_SG_VCD.md)
8. Current Research Plan v2 and applicable P0 remediation specification (`REMEDIATION_SPEC_P0_v1.md`).
9. `TASK_18A_RESOURCE_PREFLIGHT_CORRECTION_REPORT.md` and `TASK_18A_INDEPENDENT_ACCEPTANCE_QA.md`.

---

## 3. Production Source Code Changes

All changes were strictly confined to the Stage 2 decoding interface, with zero modifications to Stage 1 attention extraction, affinity field vectorization, or caching policies.

### 3.1 [`scripts/benchmark/sg_vcd/dynamic_cd.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/dynamic_cd.py)
- **Added `scoring_mode` Configuration:**
  - Added parameter `scoring_mode: str = "canonical_signed"` to `DynamicContrastiveDecoder.__init__`.
  - Implemented strict validation accepting `"canonical_signed"` (aliases: `"canonical_r03"`, `"signed"`) and `"historical_relu"` (aliases: `"legacy_relu"`, `"relu"`). Any unsupported string raises a clear `ValueError`.
- **Decoupled Scoring from Candidate-Support Masking:**
  - Renamed `delta_z` in CAPC masking to `delta_z_candidate = torch.relu(pos_logits - neg_logits)`.
  - Preserved CAPC candidate-support filtering ($V_{\mathrm{head}}$) and DCEB peak contrast tracking ($m_t$) on `delta_z_candidate`.
  - Implemented the contrastive scoring branch:
    ```python
    if self.scoring_mode == "canonical_signed":
        contrastive_score = (1.0 + alpha_t) * pos_logits - alpha_t * neg_logits
    else:
        contrastive_score = pos_logits + alpha_t * delta_z_candidate

    t_entropy = max(0.05, 1.0 - self.eta_temp * math.tanh(entropy_val / self.tau_h))
    calibrated_logits = contrastive_score / t_entropy
    ```
  - Preserved candidate-support masking `calibrated_logits.masked_fill(~valid_mask, float("-inf"))`.

### 3.2 [`scripts/benchmark/sg_vcd/pipeline.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py)
- **Pipeline Wiring:**
  - Added `scoring_mode: str = "canonical_signed"` to `SGVCDPipeline.__init__` and forwarded it to `DynamicContrastiveDecoder`.
- **Parity Trace Recording ([`_record_decode_step_trace`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L685-L732)):**
  - Updated `pre_mask_logits` calculation to evaluate `(1.0 + alpha_t) * pos_logits - alpha_t * neg_logits` when `scoring_mode == "canonical_signed"`.
  - Preserved `signed_logits` tracking for OPT-01 batched vs. sequential parity validation.

### 3.3 [`scripts/benchmark/evaluate_vitextvqa.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/evaluate_vitextvqa.py)
- **CLI Flag Integration:**
  - Added CLI argument `--scoring-mode` / `--scoring_mode` with choices `["canonical_signed", "historical_relu"]` (default: `"canonical_signed"`).
  - Forwarded `scoring_mode=getattr(args, "scoring_mode", "canonical_signed")` to `SGVCDPipeline` instantiation.

---

## 4. Architectural Analysis & Key Invariants Preserved

### 4.1 Preservation of Separate Candidate-Support Logic (Section 3.3)
A vital requirement of the task specification is ensuring that candidate-support logic is NOT removed or conflated with the logit scoring formula:
1. **Plausibility Truncation (APC / CAPC):**
   - $V_{\mathrm{head}}(t) = \{ w \in V \mid P_{\mathrm{pos}}(w) \ge \beta_t \cdot \max_v P_{\mathrm{pos}}(v) \lor \Delta z_t(w) \ge \theta_c \cdot \max_v \Delta z_t(v) \}$.
   - The candidate bypass relies on positive visual evidence: $\Delta z_t(w) = \mathrm{ReLU}(z_{\mathrm{pos}}(w) - z_{\mathrm{neg}}(w))$.
   - This prevents tokens with negative contrast from erroneously triggering the contrastive plausibility bypass.
2. **Logit Scoring:**
   - Once $V_{\mathrm{head}}(t)$ is identified via `valid_mask`, candidate tokens receive the signed score: $(1 + \alpha_t) z_{\mathrm{pos}} - \alpha_t z_{\mathrm{neg}}$.
   - Tokens outside $V_{\mathrm{head}}(t)$ are assigned $-\infty$ via `masked_fill(~valid_mask, float("-inf"))`.
   - $\mathrm{ReLU}$ is **never** applied to the final score tensor.

### 4.2 Per-Token Synchronization (Section 3.1)
- At each autoregressive decoding step $t \in [0, T-1]$, positive and negative logits $z_{\mathrm{pos}}^{(t)}$ and $z_{\mathrm{neg}}^{(t)}$ are computed for the identical step position.
- In batched decoding ($B=2$), `logits[0:1]` and `logits[1:2]` originate from the same parallel forward pass with synchronized sequence lengths and identical KV-cache history lengths.
- `alpha_t` is calculated per step from the positive and negative output distributions at step $t$ ($D_{\mathrm{JS}}(P_{\mathrm{pos}}^{(t)} \parallel P_{\mathrm{neg}}^{(t)})$ and $H(P_{\mathrm{pos}}^{(t)})$), ensuring token-level synchronization without latency drift.

---

## 5. Numerical Reference Cases: Canonical R-03 vs. Historical ReLU

To demonstrate that the two modes are functionally distinct and behave correctly on negative contrast deltas, consider a concrete deterministic example with $\alpha_t = 0.50$:

| Token ID | Description | $z_{\mathrm{pos}}$ | $z_{\mathrm{neg}}$ | $\Delta z = z_{\mathrm{pos}} - z_{\mathrm{neg}}$ | Canonical R-03: $(1+\alpha)z_{\mathrm{pos}} - \alpha z_{\mathrm{neg}}$ | Historical ReLU: $z_{\mathrm{pos}} + \alpha \cdot \mathrm{ReLU}(\Delta z)$ | Difference |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Token 0** | Visual Evidence Token | $4.0$ | $1.0$ | $+3.0$ | $1.5 \times 4.0 - 0.5 \times 1.0 = \mathbf{5.50}$ | $4.0 + 0.5 \times 3.0 = \mathbf{5.50}$ | Identical ($0.00$) |
| **Token 1** | Language Prior Distractor | $4.2$ | $8.0$ | $-3.8$ | $1.5 \times 4.2 - 0.5 \times 8.0 = \mathbf{2.30}$ | $4.2 + 0.5 \times 0.0 = \mathbf{4.20}$ | **$-1.90$ (Penalized)** |
| **Token 2** | Low-probability Neutral | $1.0$ | $0.5$ | $+0.5$ | $1.5 \times 1.0 - 0.5 \times 0.5 = \mathbf{1.25}$ | $1.0 + 0.5 \times 0.5 = \mathbf{1.25}$ | Identical ($0.00$) |

### Outcome Analysis:
- **Under Historical ReLU:**
  - Token 0 score = $5.50$
  - Token 1 score = $4.20$
  - If $\alpha_t$ had been small (e.g. $\alpha_t = 0.05$), Token 0 would score $4.15$ while Token 1 would score $4.20$, allowing the distractor to win because its negative difference was clipped to zero!
- **Under Canonical R-03:**
  - Token 1 is actively penalized ($2.30$ vs. $4.20$), suppressing the hallucination even under small $\alpha_t$.
  - When $\alpha_t = 0$: both modes yield $z_{\mathrm{pos}}$ identically.

---

## 6. Gate A Validation Evidence

### 6.1 Dedicated Canonical R-03 Unit Test Suite
Created [`tests/test_canonical_r03_signed_logits.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_canonical_r03_signed_logits.py), covering all 9 required verification areas:

```powershell
& "..\Document AI\VLM_Reasoning\venv\Scripts\python.exe" -m unittest tests/test_canonical_r03_signed_logits.py
```
**Output:**
```
.........
----------------------------------------------------------------------
Ran 9 tests in 0.048s

OK
```
**Test Breakdown:**
1. `test_01_canonical_formula_exact_reference`: PASS. Confirms output matches exact mathematical formula `(1 + alpha) * z_pos - alpha * z_neg`.
2. `test_02_alpha_zero_parity`: PASS. Confirms that when $\alpha_t = 0$, output strictly equals positive branch argmax.
3. `test_03_divergence_from_legacy_relu_on_negative_difference`: PASS. Confirms canonical signed formula penalizes negative contrast tokens where historical ReLU clips to zero.
4. `test_04_negative_differences_not_clipped`: PASS. Confirms negative differences are not clipped by ReLU.
5. `test_05_candidate_support_separation`: PASS. Confirms CAPC plausibility bypass continues using $\mathrm{ReLU}(z_{\mathrm{pos}} - z_{\mathrm{neg}})$.
6. `test_06_mode_separation_and_aliasing`: PASS. Confirms valid aliases (`canonical_r03`, `signed`, `historical_relu`, `legacy_relu`) and validates that unknown mode strings raise `ValueError`.
7. `test_07_per_token_synchronization`: PASS. Confirms shape mismatch detection between positive and negative branches.
8. `test_08_pipeline_default_and_explicit_mode_propagation`: PASS. Confirms `SGVCDPipeline` defaults to `canonical_signed` and correctly forwards explicit overrides.
9. `test_09_numerical_precision_bfloat16_and_float32`: PASS. Confirms numerical consistency across `torch.bfloat16` and `torch.float32`.

### 6.2 OPT-01 Batched Dual-Branch Decoding Regression Suite
Ran the full OPT-01 parity test suite under canonical R-03:
```powershell
& "..\Document AI\VLM_Reasoning\venv\Scripts\python.exe" -m unittest tests/test_opt01_batched_dual_branch_decoding.py
```
**Output:**
```
........
----------------------------------------------------------------------
Ran 8 tests in 5.617s

OK
```
All 8 tests passed, confirming that batched dual-branch decoding ($B=2$) and sequential reference decoding achieve strict numerical parity under canonical R-03 signed scoring.

### 6.3 Stage 2 & Full SG-VCD Regression Suites
Ran the broader Stage 2 test suite:
```powershell
& "..\Document AI\VLM_Reasoning\venv\Scripts\python.exe" -m unittest tests/test_sg_vcd.py tests/test_dceb_adversarial.py tests/test_adversarial_m3_2.py tests/test_adversarial_r7_1.py
```
**Output:**
```
Ran 86 tests in 2.89s
OK
```
Ran all optimization test suites:
```powershell
& "..\Document AI\VLM_Reasoning\venv\Scripts\python.exe" -m unittest tests/test_opt02_memory_cleanup_policy.py tests/test_opt03_working_map_cache.py tests/test_opt04_gpu_affinity_vectorization.py tests/test_opt06_image_cache_identity.py
```
**Output:**
```
Ran 41 tests in 4.349s
OK
```
**Total Passing Regression Tests:** $9 + 8 + 86 + 41 = \mathbf{144\text{ tests passed with 0 failures, 0 errors, and 0 skipped tests}}$.

---

## 7. Working Tree Verification & Diff Summary

Execution of `git status --short` confirms strict scope preservation:
```powershell
 M scripts/benchmark/evaluate_vitextvqa.py
 M scripts/benchmark/sg_vcd/__init__.py
 M scripts/benchmark/sg_vcd/adaptive_propagator.py
 M scripts/benchmark/sg_vcd/attention_extractor.py
 M scripts/benchmark/sg_vcd/dynamic_cd.py
 M scripts/benchmark/sg_vcd/feature_cache.py
 M scripts/benchmark/sg_vcd/pipeline.py
?? .github/
?? docs/SG_VCD_Research_Review_Consolidated_Draft.md
?? docs/research/
?? scripts/benchmark/colab_gate_b_preflight.py
?? tests/test_canonical_r03_signed_logits.py
?? tests/test_opt01_batched_dual_branch_decoding.py
?? tests/test_opt02_memory_cleanup_policy.py
?? tests/test_opt03_working_map_cache.py
?? tests/test_opt04_gpu_affinity_vectorization.py
?? tests/test_opt06_image_cache_identity.py
```

### Specific Task 18B Files:
- **Modified:**
  - `scripts/benchmark/sg_vcd/dynamic_cd.py`
  - `scripts/benchmark/sg_vcd/pipeline.py`
  - `scripts/benchmark/evaluate_vitextvqa.py`
- **Created:**
  - `tests/test_canonical_r03_signed_logits.py`
  - `docs/research/TASK_18B_CANONICAL_R03_IMPLEMENTATION_REPORT.md` (this report)
- **Preserved:** All pre-existing uncommitted OPT-01 through OPT-06 implementation and test files remain completely intact.
- **Zero changes made to model weights, datasets, or Git history.**

---

## 8. Explicit Limitations & Gate B Notice

### Limitations
1. **Synthetic/Mock Validation Only:** All testing was conducted using synthetic mock models, deterministic tensors, and CPU/CUDA mocks. No real `5CD-AI/Vintern-1B-v3_5` weights were downloaded or instantiated.
2. **Unmeasured Metrics:** GPU latency, real peak VRAM, and token generation speed on physical Tensor Core hardware cannot be measured without authorized Gate B execution.

---

## 9. Completion Handoff

```
+--------------------------------------------------------------------------------+
|                         TASK 18B COMPLETION STATUS                             |
|                                                                                |
|             IMPLEMENTATION COMPLETE — READY FOR INDEPENDENT QA                 |
+--------------------------------------------------------------------------------+
```

### Explicit Gate B Notice
**Gate B execution and real-model inference remain STRICTLY NOT AUTHORIZED.**  
Task 18B completion does not grant permission for Colab runs, real-model forward passes, manifest freeze, or benchmark execution. All artifacts are prepared for independent acceptance review by `Antigravity (Anti)`.

