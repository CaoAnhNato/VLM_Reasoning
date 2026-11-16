# TASK 21 — T4 PRECISION COMPATIBILITY & CONDITIONAL GATE B PILOT REPORT

**Project:** SG-VCD / ViTextVQA  
**Primary Engineer:** 4argon  
**Independent Reviewer:** Antigravity (Anti) — consolidated review deferred until Colab T4 pilot completion  
**Approval Authority:** Project Owner  
**Date of Owner Authorization:** `2026-11-12`  
**Current Execution Stage:** **Phase A (Gate A) Complete | Phase B/C Colab Tooling Ready & Mock-Verified | Real Colab T4 Pilot Pending Owner Runtime Execution**  

---

## 1. Local Gate A Precision Findings (Phase A)

The local environment runs `Python 3.11.9` with `PyTorch 2.14.1+cpu` (`torch.cuda.is_available() == False`), supporting CPU arithmetic for `torch.float32`, `torch.bfloat16`, and `torch.float16` across `nn.Linear`, `nn.LayerNorm`, `nn.Conv2d`, `F.scaled_dot_product_attention`, and `F.softmax`.

### Full Dtype Path Audit
1. **Model Loader ([`evaluate_vitextvqa.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/evaluate_vitextvqa.py)):**
   - Previously hardcoded `torch_dtype=torch.bfloat16` in `load_vintern_model()` and `pv.to(torch.bfloat16)` in `CTRL-0`.
   - Updated to accept `--precision bf16|fp16` (default: `bf16`), routed through [`resolve_precision_and_dtype`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/feature_cache.py) with **zero implicit fallback** (unsupported precision strings or dtypes raise `ValueError`).
   - Added [`inspect_model_dtypes`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/evaluate_vitextvqa.py) to measure actual parameter and buffer dtypes per submodule (`vision_model`, `mlp1`, `language_model`), recognizing that custom modules (e.g., rotary `inv_freq` buffers or explicit FP32 norm accumulators) can retain FP32 buffers even when `from_pretrained(..., torch_dtype=torch.float16)` is requested.
2. **Vision Encoder & Feature Cache ([`feature_cache.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/feature_cache.py)):**
   - Updated `VisualFeatureCache.get_or_compute` and `get_or_compute_visual_features` to accept `dtype` / `precision`, cast `pixel_values` to `target_dtype`, and reject cache hits whose stored `vit_embeds.dtype != target_dtype`.
3. **Stage 1 Attention Extraction ([`attention_extractor.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/attention_extractor.py)):**
   - Updated `PointwiseKLAttentionExtractor.__init__` and `extract()` to use `self.torch_dtype` for `pixel_values` and enforce dtype equality on image cache hits.
   - Confirmed sensitive Stage 1 aggregations (`_compute_paired_visual_attentions`, KL/JS head consensus weighting, `grid_feats`, `compute_pointwise_kl_map`, `compute_intrinsic_affinity_field`) already promote tensors to `float32` / `float64`. Added `logits.float()` and `NaN` guard in `_rollout_reading_trajectory`.
4. **Stage 2 Dual-Branch Decoding & KV Cache ([`pipeline.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py)):**
   - Updated `SGVCDPipeline.__init__`, `extract_visual_features`, `_decode_with_tiles`, and `_prepare_dual_branch_stage2_inputs` to propagate `self.precision` / `self.torch_dtype`.
   - `_validate_dual_branch_inputs` continues to enforce `pos_inputs.dtype == neg_inputs.dtype` (no mixed-branch conversion).
   - `_validate_kv_cache` additionally asserts `k.dtype == v.dtype` across every layer.
   - Prefill and step logits remain promoted via `.float()` before entering Stage 2 decoding.
5. **Dynamic Contrastive Decoder & Critical FP16 Underflow/Overflow Guard ([`dynamic_cd.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/dynamic_cd.py)):**
   - **Critical Software Hazard Identified:** In `torch.float16`, the minimum normal value is $2^{-14} \approx 6.10 \times 10^{-5}$ and the minimum subnormal is $2^{-24} \approx 5.96 \times 10^{-8}$. Directly evaluating `.clamp(min=1e-12)` or `.clamp(min=1e-9)` on a `float16` probability tensor flushes to `0.0` (`float(torch.tensor(1e-9, dtype=torch.float16)) == 0.0`), which produces `log(0.0) = -inf` and `0.0 * (-inf) = NaN` if `decode_step` or `compute_js_divergence` is invoked directly on FP16 tensors.
   - **Fix Implemented:** Both `compute_js_divergence` and `decode_step` now enforce matching branch shapes/dtypes, reject `NaN` / `+inf` (or all-`-inf` rows) with `ValueError`, and promote inputs to `float32` (`pos_f32`, `neg_f32`) before computing softmax, entropy, KL/JS divergence, `alpha_t`, canonical R-03 signed scoring `(1 + alpha_t) * z_pos - alpha_t * z_neg` (or separately selectable `historical_relu`), TESG temperature scaling, DCEB EOS boost, and CAPC masking.

---

## 2. Precision Implementation & Changed Files

| File | Summary of Changes |
|---|---|
| [`scripts/benchmark/sg_vcd/feature_cache.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/feature_cache.py) | Added `resolve_precision_and_dtype()`, `dtype`/`precision` arguments, and dtype-matching cache hit check. |
| [`scripts/benchmark/sg_vcd/__init__.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/__init__.py) | Exported `resolve_precision_and_dtype`. |
| [`scripts/benchmark/sg_vcd/attention_extractor.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/attention_extractor.py) | Added `precision`/`torch_dtype` wiring, replaced hardcoded `torch.bfloat16`, verified cache dtype match, added `NaN` rollout guard. |
| [`scripts/benchmark/sg_vcd/dynamic_cd.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/dynamic_cd.py) | Enforced branch dtype parity, non-finite rejection, and internal FP32 promotion in `compute_js_divergence` and `decode_step`. |
| [`scripts/benchmark/sg_vcd/pipeline.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py) | Propagated `precision`/`torch_dtype` across Stage 1/2/feature-cache and added `k.dtype == v.dtype` check in `_validate_kv_cache`. |
| [`scripts/benchmark/evaluate_vitextvqa.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/evaluate_vitextvqa.py) | Added `--precision bf16\|fp16`, `inspect_model_dtypes`, explicit loader dtype, `run_metadata.json`, and stage timing capture. |
| [`scripts/benchmark/colab_gate_b_preflight.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/colab_gate_b_preflight.py) | Added `--precision bf16\|fp16`, live CUDA FP16 kernel probe (`probe_fp16_runtime_support`), and `--allow-bf16-exception` preserving native BF16 `REJECT` verdict. |
| [`scripts/benchmark/run_colab_t4_gate_b_pilot.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/run_colab_t4_gate_b_pilot.py) | Created single-session Colab pilot runner (Phase B preflight -> Phase C.1 1-QA smoke check -> Phase C.2 20-QA 3-arm pilot). |
| [`tests/test_task21_precision_compatibility.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_task21_precision_compatibility.py) | Created 8-test synthetic comparison & preflight validation suite. |

---

## 3. FP32 / BF16 / FP16 Synthetic Numerical Comparison (CPU Mock Evidence Only)

All measurements below were captured on deterministic CPU synthetic tensors and a 2-layer synthetic transformer (`hidden_dim=64, vocab_size=128, num_layers=2, num_heads=4, num_kv_heads=2`) inside [`tests/test_task21_precision_compatibility.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_task21_precision_compatibility.py). **These are synthetic software-level measurements, not real `5CD-AI/Vintern-1B-v3_5` weights or CUDA kernel measurements.**

### 3.1 Stage 2 Logit, Alpha, Entropy, JS, CAPC Mask & Ranking Parity (50 Steps vs FP32)
*(Relative score error denominator: `clamp(|z_fp32|, min=1e-6)`)*

| Precision | Max `|Δα_t|` | Max `|ΔH|` | Max `|ΔD_JS|` | Max Score Abs Error | Max Score Rel Error | Argmax Agreement | Top-5 Exact Match | CAPC Mask Exact | EOS Agreement |
|---|---|---|---|---|---|---|---|---|---|
| **BF16** | `1.658792e-02` | `2.255785e-02` | `6.291673e-03` | `1.163645e-01` | `7.494819e+00` | `49/50 (98%)` | `43/50 (86%)` | `32/50 (64%)` | `50/50 (100%)` |
| **FP16** | `2.166274e-03` | `3.222585e-03` | `5.796403e-04` | `1.075268e-02` | `8.853808e-01` | `50/50 (100%)` | `50/50 (100%)` | `48/50 (96%)` | `50/50 (100%)` |

*Note on recommended tolerance (not a pre-approved scientific criterion):* For pure FP16 vs FP32 Stage 2 logit scoring under FP32 promotion, observed errors stay bounded well within `<= 2e-2` score abs error and `<= 5e-3` alpha abs error (`~7.7x–10.9x` tighter than BF16 due to FP16's 10-bit vs BF16's 7-bit mantissa).

### 3.2 Two-Layer Synthetic Causal Transformer Multi-Step Parity (10 Steps vs FP32)

| Precision | Pos Logits Abs Err | Neg Logits Abs Err | Contrastive Abs Err | Contrastive Rel Err | Per-Step Token Agree | Top-5 Agree | OPT-01 Batched vs Sequential Parity Diff |
|---|---|---|---|---|---|---|---|
| **BF16** | `1.098120e-02` | `1.188129e-02` | `1.338047e-02` | `6.351254e+00` | `10/10 (100%)` | `6/10 (60%)` | `0.000000e+00` |
| **FP16** | `1.213908e-03` | `1.288474e-03` | `1.479099e-03` | `4.463980e-01` | `10/10 (100%)` | `9/10 (90%)` | `0.000000e+00` |

### 3.3 Stage 1 Attention Extraction & ROI Comparison vs FP32 (Synthetic Signboard Image)

| Precision | `Phi` Max Abs Error | `R` Max Abs Error | `A_query` Max Abs Error | Selected ROI `pixel_bbox` | `kl_score` Abs Error vs FP32 |
|---|---|---|---|---|---|
| **FP32 (Ref)** | `0.0` | `0.0` | `0.0` | `(105, 0, 222, 129)` | `0.0` |
| **BF16** | `8.230972e-02` | `5.613637e-02` | `2.703164e-07` | `(98, 0, 222, 129)` | `7.496927e-03` |
| **FP16** | `1.366615e-02` | `7.884681e-03` | `3.376044e-08` | `(98, 0, 222, 129)` | `1.390316e-03` |

### 3.4 Overflow / Underflow & Kernel Findings
- **PyTorch CPU FP16 Operators:** `nn.Linear`, `nn.LayerNorm`, `nn.Conv2d`, `F.scaled_dot_product_attention`, and `F.softmax` all execute on `torch 2.14.1+cpu` in `torch.float16` without kernel errors.
- **FP16 Dynamic Range Risk:** Values `> 65504.0` overflow to `+inf` in `torch.float16` (whereas `torch.bfloat16` accommodates up to `3.39e38`). Guarded in `DynamicContrastiveDecoder.decode_step` via explicit non-finite `ValueError`.

---

## 4. Phase A Decision & Rationale

**Phase A Decision:** **`GATE_A_FP16_COMPATIBLE`**

**Rationale:**
1. All SG-VCD Stage 1 and Stage 2 decoding components support explicit `--precision fp16` without implicit fallback or cross-branch dtype mixing.
2. Sensitive decoding math (softmax, entropy, KL/JS divergence, `alpha_t`, canonical R-03 logits, CAPC masking) runs in FP32, preventing FP16 subnormal flush-to-zero `NaN`s.
3. On CPU synthetic benchmarks, FP16 demonstrates strictly smaller error relative to FP32 than BF16, 100% token/argmax agreement, and `0.0` OPT-01 batched vs. sequential parity drift.

**Explicit Limitation:** CPU synthetic validation does **not** establish real T4 CUDA kernel behavior or guarantee that real `5CD-AI/Vintern-1B-v3_5` activations remain within FP16's `±65504` range across real ViTextVQA images. Therefore, Phase B live T4 runtime FP16 probing and Phase C.1 first-sample real inference smoke checking remain mandatory prior to the full 20-QA 3-arm pilot.

---

## 5. Actual T4 Preflight, Inference Execution & Results (Phases B & C)

**STATUS:** **`NOT_EXECUTED_IN_LOCAL_SESSION — PENDING PROJECT OWNER COLAB T4 EXECUTION`**

Because this environment is local Windows (`torch 2.14.1+cpu`, `cuda: False`) without direct remote access to Google Colab GPU runtimes, Phases B and C real-model execution have been prepared, tested in local `--mock` mode (`pilot_consolidated_summary.json` -> `PILOT_COMPLETED`), and staged for one-command execution on the Project Owner's Colab Tesla T4 runtime:

### Exact Colab T4 Execution Commands (Owner Runtime)
1. **Primary Phase A-Selected Path (FP16):**
   ```bash
   python scripts/benchmark/run_colab_t4_gate_b_pilot.py \
       --precision fp16 \
       --output-dir results/task21_t4_gate_b_pilot
   ```
2. **Conditional BF16 Exception Path (ONLY if FP16 triggers a Section 6 runtime stop condition):**
   ```bash
   python scripts/benchmark/run_colab_t4_gate_b_pilot.py \
       --precision bf16 \
       --allow-bf16-exception \
       --output-dir results/task21_t4_gate_b_pilot_bf16_exc
   ```

*(Actual T4 preflight JSON, per-arm R-04 results, VRAM/RAM/latency telemetry, and raw predictions will populate Sections 5–9 once executed on the Colab T4 runtime.)*

---

## 6. Local Test Commands & Exact Verified Outputs (Gate A / Mock Tooling)

1. **Task 21 + Task 20 + Task 18B + OPT Regression Suite:**
   ```powershell
   & "..\Document AI\VLM_Reasoning\venv\Scripts\python.exe" -m unittest `
       tests/test_task21_precision_compatibility.py `
       tests/test_task20_integrated_pilot.py `
       tests/test_canonical_r03_signed_logits.py `
       tests/test_opt01_batched_dual_branch_decoding.py `
       tests/test_opt02_memory_cleanup_policy.py `
       tests/test_opt03_working_map_cache.py `
       tests/test_opt04_gpu_affinity_vectorization.py `
       tests/test_opt06_image_cache_identity.py
   ```
   **Result:** `Ran 78 tests in 28.207s` -> **`OK`**

2. **Stage 2 Adversarial Numerical Suite:**
   ```powershell
   & "..\Document AI\VLM_Reasoning\venv\Scripts\python.exe" -m unittest tests/test_adversarial_m3_2.py
   ```
   **Result:** `Ran 17 tests in 0.591s` -> **`OK`**

3. **Preflight Dry Run (`--precision fp16 --json` on local CPU):**
   **Result:** Exit code `1` (`REJECT`, `cuda_available: false`, `host_ram: 7.36 GB < 8.0 GB`, `disk_free: 123.93 GB PASS`).

4. **End-to-End Pilot Orchestrator Local Mock Smoke Test (`run_colab_t4_gate_b_pilot.py --mock --precision fp16`):**
   **Result:** Exit code `0`, `status: PILOT_COMPLETED`, smoke check (`1/1 completed, valid, valid`), `ctrl_0` (`20/20 completed, not_applicable`), `oracle_roi` (`20/20 completed, valid`), `sg_vcd` (`20/20 completed, valid`).

---

## 7. Git Diff Summary & Working-Tree Status

`git diff --stat` (excluding new untracked files):
```text
 scripts/benchmark/evaluate_vitextvqa.py         | 1023 ++++++++++++++---
 scripts/benchmark/metrics.py                    |  174 ++-
 scripts/benchmark/sg_vcd/__init__.py            |    5 +
 scripts/benchmark/sg_vcd/adaptive_propagator.py |   73 +-
 scripts/benchmark/sg_vcd/attention_extractor.py |  339 +++++-
 scripts/benchmark/sg_vcd/dynamic_cd.py          |   79 +-
 scripts/benchmark/sg_vcd/feature_cache.py       |  524 ++++++++-
 scripts/benchmark/sg_vcd/pipeline.py            | 1390 +++++++++++++++++++----
 8 files changed, 3074 insertions(+), 533 deletions(-)
```
Untracked Task 21 deliverables added:
- `scripts/benchmark/run_colab_t4_gate_b_pilot.py`
- `tests/test_task21_precision_compatibility.py`
- `docs/research/TASK_21_T4_PRECISION_AND_GATE_B_PILOT_REPORT.md`

