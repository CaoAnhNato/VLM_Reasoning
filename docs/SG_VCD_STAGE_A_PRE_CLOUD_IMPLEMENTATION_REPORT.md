# SG-VCD: Stage A Cloud Feasibility & Preflight Implementation Report

**Document ID:** `SG-VCD-STAGE-A-PRE-CLOUD-REPORT`  
**Target Repository:** `VLM_Reasoning` (`docs/SG_VCD_STAGE_A_PRE_CLOUD_IMPLEMENTATION_REPORT.md`)  
**Backbone Model:** `5CD-AI/Vintern-1B-v3_5` (`InternViT-300M` + `Qwen2.5-0.5B-Instruct`)  
**Pinned Hugging Face Commit:** `b98f263eab246eb5269ade64edbdca8a887dc44d`  
**Target Platform:** Linux Ubuntu 22.04 LTS, Ampere/Ada Cloud GPU (NVIDIA RTX 3090 / 4090 / A5000, 24 GB VRAM, Compute Capability $\ge 8.0$), PyTorch 2.4.1+cu121  
**Audit Status:** READY WITH CONDITIONS (Static Code Audit & Algorithmic Unit Tests Verified)  

---

## 1. Project Overview

### 1.1 Project Title and Identification
**SG-VCD: Saliency-Guided Visual Contrastive Decoding for Vietnamese Scene-Text Visual Question Answering (ViTextVQA).**

### 1.2 ViTextVQA Problem Context
Vietnamese Visual Question Answering on scene-text images poses unique challenges for Vision-Language Models (VLMs):
1. **Diacritic and Orthographic Sensitivity:** Vietnamese utilizes complex tonal diacritics and composite characters (under Unicode NFC normalization), where minor character-level misalignments or subtle visual distortions can fundamentally invert semantic meaning.
2. **Dense, Distracting Background Scene Text:** Scene-text environments typically feature numerous incidental text instances (store signs, banners, phone numbers, addresses). Standard autoregressive decoding often hallucinates or attends to prominent, high-contrast background text rather than the specific text region conditioned on the question.
3. **Visual Token Dilution:** Modern high-resolution multimodal models compress image patches into large token sequences. Without question-guided spatial focusing, the model's language decoder experiences attention dilution across irrelevant image patches.

### 1.3 Research Objective
SG-VCD investigates whether question-conditioned text region saliency can guide visual contrastive decoding to reduce hallucination and enhance answer grounding in Vietnamese scene-text VQA, operating without task-specific parameter fine-tuning.

### 1.4 Backbone Model Selection and Provenance
The project standardizes on **Vintern-1B-v3_5** (`5CD-AI/Vintern-1B-v3_5`):
- **Visual Encoder:** `InternViT-300M` (300 million parameters, native 448×448 tile preprocessing).
- **Language Decoder:** `Qwen2.5-0.5B-Instruct` (24 decoder layers, hidden dimension 896, 14 query heads, 2 key-value heads under Grouped-Query Attention, head dimension 64).
- **Pinned Commit Hash:** `b98f263eab246eb5269ade64edbdca8a887dc44d` (guarantees immutability of model configuration and remote execution code).

---

## 2. Research Methodology and Overall Architecture

### 2.1 Two-Stage Research Architecture
The overall SG-VCD methodology is planned across two distinct research stages:

```
+-----------------------------------------------------------------------------------+
|                            OVERALL SG-VCD PIPELINE                                |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  [Input Image] + [Vietnamese Question]                                            |
|         │                                                                         |
|         ▼                                                                         |
|  +─────────────────────────────────────────────────────────────────────────────+  |
|  | RESEARCH STAGE 1: Question-Conditioned Text ROI Localization & Selection    |  |
|  | - Extract text proposals / OCR tokens (e.g. SwinTextSpotter / ViTextVQA GT) |  |
|  | - Compute question-conditioned attention saliency across candidate regions |  |
|  | - Rank and select target text Regions of Interest (ROIs)                    |  |
|  +─────────────────────────────────────────────────────────────────────────────+  |
|         │ Selected Salient ROI Mask / Bounding Box                                |
|         ▼                                                                         |
|  +─────────────────────────────────────────────────────────────────────────────+  |
|  | RESEARCH STAGE 2: Saliency-Guided Visual Contrastive Decoding (SG-VCD)      |  |
|  | - Generate unguided logits: P_orig = Logits(V_full, Question)                |  |
|  | - Generate distractor/masked logits: P_mask = Logits(V_masked, Question)    |  |
|  | - Contrastive formulation: Logits_VCD = (1 + α) * P_orig - α * P_mask       |  |
|  | - Dynamic truncation / adaptive filtering (e.g. DOLA / VCD criteria)        |  |
|  +─────────────────────────────────────────────────────────────────────────────+  |
|         │                                                                         |
|         ▼                                                                         |
|  [Grounded Vietnamese Answer]                                                     |
+-----------------------------------------------------------------------------------+
```

- **Research Stage 1 (Planned):** Question-conditioned text ROI localization and selection. Investigates methods to localize the specific spatial text region referenced by the question using attention-map saliency, bounding box proposals, and candidate ranking.
- **Research Stage 2 (Planned):** Saliency-guided visual contrastive decoding. Applies contrastive logit amplification between the full image representation and a saliency-perturbed visual representation, guiding the decoding trajectory away from background hallucination.

> **Research Integrity Notice:** Neither Research Stage 1 nor Research Stage 2 has been implemented or experimentally validated at this juncture. They represent the long-term scientific objective of SG-VCD.

### 2.2 Distinction from Stage A Technical Preflight Labels
The technical modules implemented in `experiments/stage1_vintern_pilot/` (**A0, A1, A2**) must not be conflated with the scientific research stages:
- **A0, A1, A2** are **technical preflight feasibility gates** designed strictly to test whether the cloud environment, hardware precision, tokenizer boundaries, and attention instrumentation can execute reliably without architectural divergence.
- Passing Stage A authorizes proceeding to Research Stage 1 development; it does not constitute research completion.

---

## 3. Stage A Feasibility Package Overview

The Stage A feasibility package (`experiments/stage1_vintern_pilot/`) isolates three sequential diagnostic gates:

### 3.1 Stage A0 — Baseline Model Preflight (`run_stage_a0.py`)
- **Primary Objective:** Verifies baseline model loading and end-to-end multimodal forward inference in native `bfloat16` on an Ampere/Ada cloud GPU.
- **Scope & Checks:**
  - Enforces single-tile input constraints (`num_patches == 1`, 448×448).
  - Explicitly initializes and validates `<IMG_CONTEXT>` token IDs.
  - Performs synchronized GPU latency profiling and peak VRAM tracking.
  - Executes real tensor-level sanity checks ensuring `visual_features` and `forward_logits` are strictly finite, containing zero `NaN` or `Inf` values.
  - Prohibits silent fallback from the official multimodal forward path.

### 3.2 Stage A1 — Token Alignment & Vietnamese NFC Audit (`run_stage_a1.py`)
- **Primary Objective:** Audits tokenization and boundary mapping directly from the rendered conversation template to ensure reliable question span identification.
- **Scope & Checks:**
  - Verifies exact token span extraction from the full prompt without re-tokenizing isolated string fragments.
  - Tests contiguous visual token placement ($1 \times 256 = 256$ tokens).
  - Detects Byte-level BPE boundary straddling and partial multi-byte UTF-8 replacement characters (`\ufffd`).
  - Audits diverse Vietnamese interrogative patterns ("gì", "ở đâu", "bao nhiêu", "màu gì") under Unicode NFC normalization.
  - Gated to return `INCONCLUSIVE` on ambiguous token boundaries rather than fabricating indices.
- **Execution Mode & Non-CUDA Scope:** Stage A1 transparently labels its execution as `CPU_TOKEN_AUDIT` or `GPU_EXECUTION`. Because its primary function is tokenizer boundary mapping and prompt construction, A1 does not require GPU tensor inference. Loading a tokenizer/model on CUDA during A1 is an environment convenience, not evidence of CUDA tensor inference.

### 3.3 Stage A2 — Restricted Diagnostic Attention Proof (`run_stage_a2.py`)
- **Primary Objective:** Validates the non-invasive observer mechanism required to extract decoder attention weights for downstream spatial saliency mapping.
- **Scope & Checks:**
  - Non-invasive context manager wraps the target decoder layer's `self_attn` without altering weights or return contracts.
  - Intercepts keyword `position_embeddings` matching Transformers 4.45.2 `Qwen2DecoderLayer` execution.
  - Re-derives exact post-RoPE Q/K tensors using projection weights and rotary embeddings, strictly rejecting raw tensor fallbacks.
  - Reconstructs query-to-visual attention probabilities via causal Log-Sum-Exp normalization incorporating the actual 4D attention mask.
  - Partitions validation into three separate diagnostic gates:
    1. *Mathematical Equivalence Test:* Side-channel vs independent FP32 full reference ($\text{tol} \le 10^{-5}$).
    2. *Model Agreement Test:* Side-channel vs actual eager model attention weights ($\text{tol} \le 5 \times 10^{-3}$, provisional operational threshold).
    3. *Forward Invariance Test:* Independent Pass 1 baseline vs Pass 2 instrumented observer on logits ($\text{tol} \le 10^{-4}$).

---

## 4. Technical Implementation Details

### 4.1 Model Loading and Precision Enforcement
- `load_vintern_model()` in `vintern_utils.py` loads `5CD-AI/Vintern-1B-v3_5` under revision `b98f263eab246eb5269ade64edbdca8a887dc44d`.
- Explicitly enforces `torch_dtype = torch.bfloat16`. If CUDA hardware does not support native `bfloat16`, execution halts immediately; silent fallback to `float16`, `float32`, or CPU is strictly prohibited.
- Sets `_attn_implementation = "eager"` across `model.config`, `model.language_model.config`, and every decoder layer's `self_attn.config` to prevent kernel divergence.

### 4.2 Dynamic Image Preprocessing and Token Alignment
- `load_image_tensor()` supports dynamic patch tiling with ImageNet normalization ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
- For Stage A preflight, single-tile configuration (`max_num=1`, `use_thumbnail=False`) produces exactly 1 patch of dimension `[1, 3, 448, 448]`.
- The visual encoder projects this tile into 256 visual feature embeddings (`embeds_flat: [256, 896]`).
- `<IMG_CONTEXT>` placeholder tokens are validated: `num_placeholders == embeds_flat.shape[0]`.

### 4.3 Qwen2 RoPE and GQA Extraction Mechanism
In Transformers 4.45.2, `Qwen2Attention` employs Grouped-Query Attention (GQA):
- **Query Heads ($H_Q$):** 14 heads, dimension $d = 64$.
- **Key/Value Heads ($H_{KV}$):** 2 heads, dimension $d = 64$.
- **GQA Ratio:** $14 / 2 = 7$ query heads per KV head group.
- **RoPE Execution Sequence in Transformers 4.45.2:**
  1. `Qwen2Model.forward()` computes shared `(cos, sin)` positional embeddings through its rotary embedding module (`self.rotary_emb(value_states, position_ids)`).
  2. The shared `position_embeddings` tuple is passed down to each `Qwen2DecoderLayer`.
  3. `Qwen2DecoderLayer.forward()` forwards them to `self_attn.forward()` using the `position_embeddings` keyword argument.
  4. `Qwen2Attention` also contains `self.rotary_emb` and can calculate RoPE internally when `position_embeddings` is omitted.
  5. `DiagnosticAttentionCollector` extracts `position_embeddings` directly from `kwargs`. If absent, it queries `parent.self_attn.rotary_emb` (instantiated directly on `Qwen2Attention`).
  6. Post-RoPE tensors `q_post, k_post` are computed via `apply_rotary_pos_emb(q_raw, k_raw, cos, sin)`.
  7. If `(cos, sin)` cannot be obtained or validated, a `RuntimeError` is raised immediately; raw unrotated Q/K tensors are never accepted.

### 4.4 Attention Masking and Causal Log-Sum-Exp Normalization
For query position $p$ and key sequence $0 \le j \le p$:
$$\text{raw\_logits}_{h, p, j} = \frac{q_{h, p} \cdot k_{\text{group}(h), j}^\top}{\sqrt{d}} + \text{mask}_{h, p, j}$$
$$\text{LSE\_denom}_{h, p} = \text{LogSumExp}_{j \le p}\left(\text{raw\_logits}_{h, p, j}\right)$$
$$P_{\text{vis}}(h, p, v) = \exp\left(\text{raw\_logits}_{h, p, v} - \text{LSE\_denom}_{h, p}\right), \quad \forall v \in \text{VisualIndices}, v \le p$$

This guarantees:
1. Normalization is performed over all causally valid keys ($j \le p$), not merely over visual tokens.
2. Total visual attention mass is strictly bounded: $\sum_{v} P_{\text{vis}}(h, p, v) \le 1.0$.
3. Any 4D external attention mask slice is incorporated before exponentiation.

### 4.5 Robust Error Handling & Provenance Manifest
- `run_stage_a2.py` wraps model forward passes in a top-level `try...except` block, ensuring that if an exception occurs:
  - `DiagnosticAttentionCollector.__exit__` executes and restores the original attention forward method.
  - Exception string and full traceback are recorded.
  - The diagnostic JSON report is written to disk with status `FAIL` or `INCONCLUSIVE`.
  - No exception is silently swallowed or converted to `PASS`.
- `reproducibility_manifest_template.json` records complete hardware metadata, software package locks, Git commit hashes, and gate execution timestamps.

---

## 5. Technical Audit Findings and Remediation Summary

During static code inspection and architectural audits, nine technical issues were identified and remediated. To preserve strict scientific integrity, verification statuses distinguish static resolution (`RESOLVED_STATIC`), live runtime verification (`PASS_RUNTIME`), skipped/unexecuted components (`NOT_RUN`), and inconclusive states (`INCONCLUSIVE`):

| Defect / Risk Identified | Root Cause Analysis | Remediation Action Implemented | Verification Status |
| :--- | :--- | :--- | :---: |
| **1. Fake NaN/Inf Sanity Checks** | Helper functions returned hardcoded `is_finite=True` without evaluating tensor elements. | Implemented `inspect_tensor_finite()` performing real `torch.isnan().any()` and `torch.isinf().any()` checks. | **RESOLVED_STATIC** |
| **2. Dummy Attention Proof** | Side-channel computation was bypassed; `max_diff` was hardcoded to `0.0`. | Implemented authentic `compute_side_channel_attention()` with causal LSE and GQA head expansion. | **RESOLVED_STATIC** |
| **3. Logits Self-Subtraction** | Forward invariance test subtracted `vanilla_logits` from itself. | Separated execution into Pass 1 (baseline) and Pass 2 (instrumented observer), comparing distinct output tensors. | **RESOLVED_STATIC** |
| **4. Raw Q/K RoPE Fallback** | Observer fell back to `q_post, k_post = q_raw, k_raw` if RoPE resolution failed. | Removed fallback entirely. Raises `RuntimeError`; never accepts unrotated Q/K as post-RoPE tensors. | **RESOLVED_STATIC** |
| **5. Ungated Model Attention Diff** | `model_max_err` was logged but excluded from the pass gate. | Added explicit `model_agreement_test` gate ($\text{tol} \le 5 \times 10^{-3}$). Reports `INCONCLUSIVE` if model reference missing. | **RESOLVED_STATIC** |
| **6. Token Alignment Ambiguity** | Byte-level BPE could split multi-byte UTF-8 bytes into `\ufffd`, causing silent index drift. | Enhanced `find_token_span_from_full_prompt()` to flag `\ufffd` and boundary straddling as `AMBIGUOUS`. | **RESOLVED_STATIC** |
| **7. RoPE API Misunderstanding** | Initial report claimed `Qwen2Attention` does not own `rotary_emb`. | Corrected documentation and code: `Qwen2Attention` owns `self.rotary_emb`; keyword extraction matches `Qwen2DecoderLayer`. | **RESOLVED_STATIC** |
| **8. Multimodal Forward Fallback** | Silent `try-except` caught errors in `model(...)` and switched to `language_model` embedding injection. | Removed fallback completely. If top-level `model(...)` throws, Stage A0 records `FAIL` and halts immediately. | **RESOLVED_STATIC** |
| **9. Coupled Dependency Flags** | Single `try/except` caused `HAS_TORCH` to become `False` when `transformers` was missing. | Decoupled into `HAS_TORCH`, `HAS_TORCHVISION`, `HAS_PIL`, and `HAS_TRANSFORMERS`. Tensor checks require PyTorch only. | **RESOLVED_STATIC** |

### Remaining Runtime Uncertainties (Cloud Execution Gated):
Static code inspection does NOT constitute runtime verification on the live Vintern model. The following items require empirical execution on the cloud GPU instance:
1. **Live Remote Code Instantiation (`NOT_RUN`):** Actual instantiation and weight loading of Vintern remote code (`modeling_internvl_chat.py`) from Hugging Face Hub under commit `b98f263eab246eb5269ade64edbdca8a887dc44d`.
2. **Live Hardware BF16 Model Agreement (`NOT_RUN`):** Empirical confirmation that eager model attention weights match side-channel LSE within the provisional threshold of $5 \times 10^{-3}$ on native Ampere/Ada hardware.
3. **Live Peak VRAM Headroom (`NOT_RUN`):** Memory profiling confirming that peak allocation stays safely below the 20 GiB envelope during single-tile inference.
4. **Live Multimodal Forward Path (`NOT_RUN`):** Verification that the official `model(pixel_values, input_ids, image_flags)` forward path executes without exceptions on the real host.

---

## 6. Unit Testing and Verification Scorecard

### 6.1 Test Execution Environment
- **Platform:** Windows 64-bit AMD64 (Local Developer Workstation)
- **Python Version:** 3.11.9
- **PyTorch:** NOT INSTALLED locally (tensor tests evaluated via standard-library pure Python proofs)
- **torchvision / transformers / PIL:** NOT INSTALLED locally
- **Execution Mode:** Local CPU (No CUDA)

### 6.2 Preflight Test Results (`experiments/stage1_vintern_pilot/tests/run_unit_tests.py`)

The test suite explicitly separates **Pure Python algorithmic tests** (which execute locally on CPU) from **PyTorch tensor tests** (which require PyTorch and run on CPU/PyTorch, without requiring CUDA unless explicitly configured):

| Test Identifier | Tested Subsystem | Evaluation Category | Result | Description |
| :--- | :--- | :--- | :---: | :--- |
| `test_causal_lse_and_visual_mass_bounded` | Causal LSE Normalization | Pure Python | **PASS_RUNTIME** | LSE denominator calculated over all causal keys; visual mass bounded $\le 1.0$. |
| `test_attention_masking_integration` | Attention Mask Slicing | Pure Python | **PASS_RUNTIME** | 4D attention mask rows ($-10^9$) properly integrated into LSE denominator. |
| `test_gqa_group_mapping` | GQA Head Expansion | Pure Python | **PASS_RUNTIME** | 14 Query heads mapped to 2 KV heads (7:1 group repetition). |
| `test_rope_position_embeddings_math` | RoPE Norm Invariance | Pure Python | **PASS_RUNTIME** | Norm preservation under 2D Givens rotation vectors $(cos, sin)$. |
| `test_clean_finite_values` | Numerical Sanity | Pure Python | **PASS_RUNTIME** | Finite float sequence correctly identified without false NaN/Inf flags. |
| `test_dependency_flags_decoupling` | Dependency Handling | Pure Python | **PASS_RUNTIME** | Verifies `HAS_TORCH` decoupling from `HAS_TRANSFORMERS`. |
| `test_values_containing_nan` | Numerical Sanity | Pure Python | **PASS_RUNTIME** | NaN values immediately trigger non-finite flag. |
| `test_values_containing_positive_and_negative_inf` | Numerical Sanity | Pure Python | **PASS_RUNTIME** | Positive and negative Infinity flagged accurately. |
| `test_pure_python_backend_consistency` | Backend Invariance | Pure Python | **PASS_RUNTIME** | Eager attention configuration verified across observer contexts. |
| `test_pure_python_observer_invariance` | Observer Invariance | Pure Python | **PASS_RUNTIME** | Zero output deviation between Pass 1 and Pass 2 on mock layers. |
| `test_alignment_with_leading_newline_and_vietnamese` | Token Alignment | Pure Python | **PASS_RUNTIME** | Prompt-derived character mapping across newlines and Vietnamese NFC text. |
| `test_detection_of_ambiguous_partial_bytes` | BPE Ambiguity Detection | Pure Python | **PASS_RUNTIME** | Multi-byte UTF-8 boundary straddling (`\ufffd`) flagged as `AMBIGUOUS`. |
| `test_interrogative_variants` | Question Span Extraction | Pure Python | **PASS_RUNTIME** | Exact span match for "gì", "ở đâu", "bao nhiêu", "màu gì". |
| `test_pytorch_side_channel_vs_fp32_ref` | Side-Channel vs FP32 Ref | PyTorch Tensor (CPU) | **NOT_RUN** | Skipped locally; requires PyTorch installation (gated for Cloud Gate 0). |
| `test_pytorch_tensor_finite_checks` | Tensor Finite Checks | PyTorch Tensor (CPU) | **NOT_RUN** | Skipped locally; requires PyTorch installation (gated for Cloud Gate 0). |
| `test_pytorch_observer_invariance` | Observer Invariance | PyTorch Tensor (CPU) | **NOT_RUN** | Skipped locally; requires PyTorch installation (gated for Cloud Gate 0). |

**Summary Statistics:**
- **Total Tests Evaluated:** 16
- **Passed (Local Pure Python):** 13 (`PASS_RUNTIME`)
- **Skipped (PyTorch Absent Locally):** 3 (`NOT_RUN` — reserved for cloud virtual environment)
- **Failures / Errors:** 0
- **Status:** **ALL EXECUTABLE PREFLIGHT TESTS PASS**

---

## 7. Current Project Limitations

To maintain scientific integrity and transparent documentation, the following limitations are explicitly recorded:
1. **No Cloud Model Downloads Yet:** The model weights (`model.safetensors`, 3.75 GB) have not yet been downloaded into the cloud environment.
2. **No Live GPU Inference Performed:** No GPU execution has occurred. All local validation was performed via static code analysis, compilation checks, and pure-Python mathematical proofs.
3. **No ViTextVQA 400-Image Evaluation:** Neither Stage A0 nor Stage A2 represents a full dataset evaluation. They are single-image / single-tile preflight probes.
4. **No Localization Accuracy Established:** No text localization metric (IoU, precision, recall) has been evaluated on ViTextVQA.
5. **Research Stage 1 Not Implemented:** Saliency-guided ROI selection is in design phase and has not been integrated.
6. **Research Stage 2 Not Implemented:** Saliency-guided visual contrastive decoding has not been coded or benchmarked.
7. **Full SG-VCD Integration Pending:** End-to-end integration of Stage 1 + Stage 2 will occur only after cloud preflight gates (A0, A1, A2) confirm technical feasibility.

---

## 8. Cloud Pilot Execution Plan

Execution on the rented cloud GPU instance must follow this strict sequential order:

```
+─────────────────────────────────────────────────────────────────────────────+
|                          CLOUD EXECUTION GATES                              |
+─────────────────────────────────────────────────────────────────────────────+
|                                                                             |
|  [GATE 0: Environment & Hardware Verification]                              |
|  $ bash experiments/stage1_vintern_pilot/setup_cloud_env.sh                 |
|  $ source experiments/stage1_vintern_pilot/cloud_env.sh                     |
|  $ python experiments/stage1_vintern_pilot/verify_cloud_env.py              |
|  $ python experiments/stage1_vintern_pilot/tests/run_unit_tests.py          |
|  -> Verification Structure & Subsystem Roles:                               |
|     1. Gate 0 (verify_cloud_env.py) verifies CUDA driver, GPU CC >= 8.0,    |
|        VRAM capacity >= 22 GiB, and hardware bfloat16 capability.            |
|     2. Unit test runner (run_unit_tests.py) runs with PyTorch installed:    |
|        all 16 tests must report PASS (13 Pure Python CPU tests + 3 PyTorch   |
|        CPU tensor tests). PyTorch tensor tests are not CUDA tests.           |
|     3. Stage A0 executes genuine multimodal forward GPU inference and        |
|        verifies that features and logits are finite and free of NaN/Inf.     |
|     4. Stage A1 executes tokenizer and prompt-token alignment; it does NOT   |
|        perform heavy CUDA tensor inference and operates cleanly in           |
|        CPU_TOKEN_AUDIT mode. Loading on CUDA during A1 is for environment    |
|        convenience and is not presented as proof of CUDA tensor inference.   |
|     5. Stage A2 executes GPU attention diagnostics and dual-pass forward     |
|        invariance validation.                                                |
|         │                                                                   |
|         ▼ PASS                                                              |
|  [GATE 1: Stage A0 — Baseline Multimodal Preflight]                         |
|  Synthetic plumbing test (isolated 448x448 tile):                          |
|  $ python experiments/stage1_vintern_pilot/run_stage_a0.py                  |
|  Real-image pilot test (supports positional or flagged arguments):          |
|  $ python experiments/stage1_vintern_pilot/run_stage_a0.py \                |
|           /path/to/image.jpg "Biển hiệu trong ảnh có chữ gì?"               |
|  $ python experiments/stage1_vintern_pilot/run_stage_a0.py \                |
|           --image /path/to/image.jpg --question "..."                       |
|  -> Requirement: Single tile verified, zero NaN/Inf, peak VRAM < 20 GB      |
|         │                                                                   |
|         ▼ PASS                                                              |
|  [GATE 2: Stage A1 — Token Alignment & NFC Audit]                           |
|  $ python experiments/stage1_vintern_pilot/run_stage_a1.py                  |
|  -> Requirement: 100% exact NFC token matches on live Qwen tokenizer       |
|         │                                                                   |
|         ▼ PASS                                                              |
|  [GATE 3: Stage A2 — Restricted Attention Proof]                            |
|  Synthetic test (Layer 12, synthetic default):                              |
|  $ python experiments/stage1_vintern_pilot/run_stage_a2.py --layer 12       |
|  Real-image test (Layer 12, positional or flagged):                         |
|  $ python experiments/stage1_vintern_pilot/run_stage_a2.py \                |
|           --layer 12 --image /path/to/image.jpg --question "..."            |
|  -> Requirement: Math diff <= 1e-5, Model diff <= 5e-3, Invariance <= 1e-4  |
+─────────────────────────────────────────────────────────────────────────────+
```

### Purpose and CLI Scope of Input Modes:
- **Synthetic Preflight:** Validates model loading, CUDA memory allocation, tensor shapes, and observer mechanics in a strictly controlled, reproducible environment without dataset dependencies.
- **Real-Image Pilot:** Validates that the model forward path, prompt templating, and attention collector execute cleanly on an actual ViTextVQA scene-text sample. Both `run_stage_a0.py` and `run_stage_a2.py` fully parse and forward optional image path and question arguments (positional or `--image` / `--question`). If an explicit image path does not exist, both scripts terminate immediately with `FAIL` (no silent synthetic replacement).
- **Scope Notice:** Neither run mode replaces or approximates a full 400-image ViTextVQA evaluation. Stage A1 intentionally uses synthetic/template sequences because its goal is prompt-level tokenizer boundary mapping across question variants, independent of image content.

---

## 9. Reproducibility, Security, and Data Safety

### 9.1 Provenance and Dependency Pinning
- **Model Revision:** Pinned to commit `b98f263eab246eb5269ade64edbdca8a887dc44d`.
- **Direct Dependencies:** Locked in `requirements-stage-a.lock`.
- **Transitive Dependencies:** Automatically frozen into `requirements-stage-a.lock.resolved` upon setup.
- **Cache Persistence:** Hugging Face cache redirected to `/workspace/hf_cache` via persistent environment variable `export HF_HOME=/workspace/hf_cache`.

### 9.2 Data Safety & Git Exclusion Rules
To protect reviewed human annotations and dataset integrity:
- Annotation workspaces (`data/xanylabeling_workspace/Cam_Tu_201_300/`, `Thanh_Tam_301_400/`) are strictly excluded from git tracking.
- The following `.gitignore` rules are enforced:
  ```gitignore
  # Datasets & annotations
  data/raw/
  data/interim/
  data/processed/
  data/xanylabeling_workspace/
  ViTextVQA_images/
  *.jpg
  *.png
  *.jsonl

  # Model weights & caches
  *.safetensors
  *.bin
  *.pt
  *.pth
  hf_cache/
  .cache/

  # Environments & secrets
  .venv*
  venv/
  .env*
  *.token

  # Logs
  logs/*.log
  __pycache__/
  ```

---

## 10. Next Research Steps

Upon successful cloud completion of Gates 0, A0, A1, and A2, the research program will proceed in the following structured sequence:

1. **Complete Research Stage 1 (Text ROI Localization):**
   - Implement candidate text region extraction (evaluating SwinTextSpotter bounding boxes vs ground-truth annotations).
   - Develop cross-modal question-conditioned saliency scoring across decoder layers.
2. **Evaluate Stage 1 Localization Accuracy:**
   - Benchmark localization accuracy against published baselines (LocalizationHeads, GroundLMM, ST-VQA_Loc).
3. **Complete Research Stage 2 (Contrastive Decoding):**
   - Implement saliency-guided visual masking and contrastive logit formulation.
   - Calibrate contrastive guidance strength $\alpha$ and adaptive truncation parameters.
4. **End-to-End SG-VCD Integration:**
   - Integrate Stage 1 ROI selection with Stage 2 contrastive decoding.
5. **Full ViTextVQA Evaluation & Ablation Study:**
   - Evaluate against published paper baselines:
     - Vanilla Vintern-1B-v3_5 baseline (greedy / beam search).
     - Standard Contrastive Decoding (CD / VCD).
     - Contrastive Region Guidance (CRG).
     - LazyMCoT adaptive routing.
   - Conduct ablation studies:
     - Effect of saliency layer selection (Layer 12 vs multi-layer aggregation).
     - Effect of contrastive guidance parameter $\alpha$.
     - Impact of ground-truth ROIs vs predicted ROIs.

