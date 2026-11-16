# TASK 18A — QA FINDINGS RECONCILIATION & NUMERICAL CONSISTENCY CLOSEOUT

**Project:** SG-VCD / ViTextVQA  
**Role:** 4argon — Senior ML Systems Engineer (Documentation Closeout)  
**Independent Reviewer:** Antigravity (Anti) — Task 18A Independent QA  
**Authority:** Project Owner  
**Scope:** Documentation-only reconciliation. No production implementation changes.  
**Reference Deliverables:**
- Audit Target: [`docs/research/TASK_18A_RESOURCE_PREFLIGHT_CORRECTION_REPORT.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/TASK_18A_RESOURCE_PREFLIGHT_CORRECTION_REPORT.md)
- Independent QA Report: [`docs/research/TASK_18A_INDEPENDENT_ACCEPTANCE_QA.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/TASK_18A_INDEPENDENT_ACCEPTANCE_QA.md)
**Gate:** Gate B preparation only — **Execution strictly NOT AUTHORIZED**

---

## 1. Executive Summary & Disposition Matrix

This reconciliation record documents the final disposition of all numerical issues, architectural derivations, and classification labels identified during the Task 18A QA and closeout cycle. All updates have been fully incorporated into `TASK_18A_RESOURCE_PREFLIGHT_CORRECTION_REPORT.md`.

No changes to production Python code, benchmark datasets, evaluator interfaces, or model weights were performed.

### Final Disposition Matrix

| Issue ID | Area | Severity | Disposition | Sections Updated in Report | Summary of Final Resolution |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **NUM-01** | Parameter Count & Embedding Accounting | Non-blocking | **RECONCILED** | Sections 1, 5.1, 5.2 | Established that Qwen2.5-0.5B's $\approx 494\text{M}$ parameter count already includes the $151,936 \times 896 \approx 136.1\text{M}$ token embedding matrix. Core architectural sum is $\approx \mathbf{818\text{M}}$ parameters ($1.64\text{ GB}$ in BF16). Removed double-counting of embeddings. Labeled $1.05\text{B}$ ($2.10\text{ GB}$) strictly as an `ANALYTICAL_UPPER_BOUND_ASSUMPTION` based on external model card specifications. |
| **NUM-02** | Attention Memory & 175 MB Correction | Non-blocking | **RECONCILED** | Sections 1, 5.1, 5.2 | Recalculated base ($Q=30, K=1,822$) and draft ($Q=40, K=1,832$) scenarios independently for single vs. paired cold branches. Corrected the loose $175\text{ MB}$ draft estimate to exactly $\mathbf{171.95\text{ MB}} \approx \mathbf{172.0\text{ MB}}$ (or $\mathbf{113.2\text{ MB}}$ if generic prompt is 6 tokens). Distinquished raw tensor payload from allocator reserved memory ($\approx 200\text{--}250\text{ MB}$). |
| **NUM-03** | Key Sequence Length Trace & Classification | Non-blocking | **RECONCILED** | Sections 5.1, 5.2 | Traced token construction in `attention_extractor.py`: visual tokens ($1,792$), system prefix ($\approx 20\text{--}25$), question suffix ($T_q \approx 30$), draft trajectory ($T_{\mathrm{draft}} \le 10$). Reconciled derivation of $1,822$ (base) vs. $1,832$ (draft) without prefix. Explicitly classified full runtime range $K \in [1,822, 1,860]$ as an `ANALYTICAL_ESTIMATE`. |
| **NUM-04** | Checkpoint Serialization vs. Loaded Dtype | Non-blocking | **RECONCILED** | Sections 1, 5.1, 5.2 | Separated disk checkpoint size ($3.75\text{ GB}$ `MEASURED`), checkpoint tensor dtypes (unverified without loading safetensors headers; removed unverified assertion of FP32 vision serialization), and loaded model parameter dtype (`torch.bfloat16` verified via `evaluate_vitextvqa.py#L216`), which guarantees in-memory device allocation $\le 2.10\text{ GB}$. |
| **NUM-05** | Budget Table Consistency & Classifications | Non-blocking | **RECONCILED** | Section 5.2 | Validated that every number in the Stage 1 and Stage 2 text strictly agrees with the budget table, equations, and classifications (`MEASURED`, `DERIVED_FROM_VERIFIED_CONFIG`, `ANALYTICAL_UPPER_BOUND_ASSUMPTION`, `ANALYTICAL_ESTIMATE`, `NOT_MEASURED`). Real peak VRAM and allocator reserves remain explicitly unmeasured. |

---

## 2. Detailed Technical Reconciliation by Issue

### 1. Parameter Count & Qwen Embedding Accounting (NUM-01)

#### The Verification & Error Correction:
- **Prior Draft Issue:** Prior text described the model count as $814\text{M}$ core parameters plus $136\text{M}$ embeddings yielding $1.05\text{B}$ parameters. This was mathematically inconsistent because Qwen2.5-0.5B's published parameter count of $\approx 490\text{M}\text{--}494\text{M}$ **already accounts for the token embedding matrix**.
- **Detailed Architectural Derivation of Qwen2.5-0.5B:**
  - Hidden dimension $D = 896$, intermediate dimension $D_{\mathrm{mlp}} = 4,864$, layers $L = 24$, query heads $H_q = 14$, KV heads $H_{\mathrm{kv}} = 2$, vocabulary $V = 151,936$.
  - 24 Transformer Layers:
    - Self-Attention per layer: $W_q (896 \times 896) + W_k (896 \times 128) + W_v (896 \times 128) + W_o (896 \times 896) = 1,835,008$ params.
    - SwiGLU MLP per layer: $W_{\mathrm{gate}} (896 \times 4,864) + W_{\mathrm{up}} (896 \times 4,864) + W_{\mathrm{down}} (4,864 \times 896) = 13,074,432$ params.
    - RMSNorms per layer: $2 \times 896 = 1,792$ params.
    - Total per layer: $14,911,232$ params $\times 24\text{ layers} = \mathbf{357,869,568\text{ params}}$ ($\approx 357.9\text{M}$).
  - Word Embedding Matrix: $V \times D = 151,936 \times 896 = \mathbf{136,134,656\text{ params}}$ ($\approx 136.1\text{M}$).
  - Final RMSNorm: $896$ params.
  - **Total Qwen LM with tied embeddings:** $357.9\text{M} + 136.1\text{M} \approx \mathbf{494.0\text{M\ parameters}}$.
- **Core Architectural Sum:**
  $$\text{InternViT-300M } (304\text{M}) + \text{Projector } (20\text{M}) + \text{Qwen-0.5B } (494\text{M}) \approx \mathbf{818\text{M\ parameters}}$$
  *(If lm_head is untied from embed_tokens: $818\text{M} + 136\text{M} \approx \mathbf{954\text{M\ parameters}}$).*
- **Status of 1.05B Figure:**
  The $1.05\text{B}$ figure is NOT obtained by adding 136M a second time. It is an **`ANALYTICAL_UPPER_BOUND_ASSUMPTION`** inherited from the model's published card, providing a safe upper bound for memory provisioning.
- **Static Device Footprints in BF16:**
  - Core ($818\text{M}$ params): $818 \times 10^6 \times 2\text{ B} \approx \mathbf{1.64\text{ GB}}$.
  - Untied ($954\text{M}$ params): $954 \times 10^6 \times 2\text{ B} \approx \mathbf{1.91\text{ GB}}$.
  - Ceiling ($1.05\text{B}$ params): $1.05 \times 10^9 \times 2\text{ B} \approx \mathbf{2.10\text{ GB}}$.

---

### 2. Attention Memory Arithmetic & Correction of the 175 MB Estimate (NUM-02)

#### The Verification & Mathematical Correction:
- **Prior Draft Issue:** Prior text reported a paired cold draft scenario as "up to $175\text{ MB}$" without a step-by-step tensor derivation.
- **Source Inspection of Branch Architecture:**
  - In `attention_extractor.py#L735`, `full_suffix_q` is generated for the query branch using `_rollout_reading_trajectory` ($Q_q = T_q + T_{\mathrm{draft}} \le 30 + 10 = 40$ tokens).
  - In `attention_extractor.py#L753`, the generic baseline `out_g` evaluates `suffix_gen` (`"<|im_end|>\n<|im_start|>assistant\n"`). **It is never draft-augmented** ($Q_{\mathrm{gen}} \le 30$, actually $6\text{--}8$ tokens).
- **Exact Independent Calculations:**
  - **Base Scenario ($Q_q = 30, K = 1,822$):**
    - Per layer: $1 \times 14 \times 30 \times 1,822 \times 4\text{ B} = 3,060,960\text{ bytes} \approx \mathbf{3.06\text{ MB}}$ ($2.92\text{ MiB}$).
    - Single branch (24 layers in `out_q.attentions`): $24 \times 3.06\text{ MB} = \mathbf{73.46\text{ MB}}$ ($70.06\text{ MiB}$).
    - Paired cold extraction ($Q_q = 30, Q_{\mathrm{gen}} \le 30$):
      $$\text{Paired Base Payload} = 73.46\text{ MB} + 73.46\text{ MB} = \mathbf{146.92\text{ MB}}\ (\approx \mathbf{147.0\text{ MB}})$$
  - **Draft Scenario ($Q_q = 40, K = 1,832$):**
    - Per layer: $1 \times 14 \times 40 \times 1,832 \times 4\text{ B} = 4,103,680\text{ bytes} \approx \mathbf{4.10\text{ MB}}$ ($3.91\text{ MiB}$).
    - Single branch (24 layers in `out_q.attentions`): $24 \times 4.10\text{ MB} = \mathbf{98.49\text{ MB}}$ ($93.93\text{ MiB}$).
    - Paired cold extraction ($Q_q = 40, Q_{\mathrm{gen}} \le 30$):
      $$\text{Paired Draft Payload} = 98.49\text{ MB} + 73.46\text{ MB} = \mathbf{171.95\text{ MB}}\ (\approx \mathbf{172.0\text{ MB}})$$
      *(The prior $175\text{ MB}$ estimate was an imprecise rounded number; the exact upper bound is $\mathbf{172.0\text{ MB}}$. If generic query uses its true token length $Q_{\mathrm{gen}} \approx 6$, paired payload drops to $98.5 + 14.7 = \mathbf{113.2\text{ MB}}$).*
  - **Warm Single-Branch Extraction:**
    When `p_g_heads` is cached (subsequent questions for the same image), `out_g` is bypassed:
    - Base warm: $\mathbf{73.5\text{ MB}}$.
    - Draft warm: $\mathbf{98.5\text{ MB}}$.
- **Payload vs. Allocator-Reserved Memory:**
  - Raw returned payload: $\mathbf{73.5\text{--}98.5\text{ MB}}$ (single) or $\mathbf{147.0\text{--}172.0\text{ MB}}$ (paired cold).
  - PyTorch Allocator Reserved: Because CUDA allocations are pooled in rounded $20\text{ MB}$ / $50\text{ MB}$ slabs, memory reserved on device during eager attention forward may reach $\mathbf{\approx 200\text{--}250\text{ MB}}$ (`ANALYTICAL_ESTIMATE` / `NOT_MEASURED`).

---

### 3. Key Sequence Length Reconciliation (NUM-03)

- **Source Code Trace ([`attention_extractor.py#L604-L638`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/attention_extractor.py#L604-L638)):**
  - Prefix prompt structure:
    `"<|im_start|>system\nBạn là một trợ lý AI hữu ích.<|im_end|>\n<|im_start|>user\n<img>"` + $(1,792\text{ visual tokens})$ + `</img>\n`.
  - Visual tokens: $7\text{ tiles} \times 256 = \mathbf{1,792\text{ tokens}}$ (`MEASURED`).
  - System prompt + user wrapper: $\approx 20\text{--}25$ tokens (`ANALYTICAL_ESTIMATE`).
  - Suffix prompt structure:
    `f"{q_text}<|im_end|>\n<|im_start|>assistant\n"`.
  - Question suffix: $T_q \approx 30$ tokens.
  - Draft reading trajectory: $T_{\mathrm{draft}} \le 10$ tokens.
- **Reconciliation of 1,822 vs. 1,832:**
  - $1,822$: Approximates $T_{\mathrm{vis}} (1,792) + T_q (30)$, abstracting away system prefix and draft.
  - $1,832$: Approximates $T_{\mathrm{vis}} (1,792) + T_q (30) + T_{\mathrm{draft}} (10)$, abstracting away system prefix.
  - Full sequence with system prefix: $K$ spans analytically between $1,822$ (minimum base question without prefix/draft) and $\approx 1,850\text{--}1,860$ (with system prefix and full 10 draft tokens).
- **Classification:** Sequence length $K$ is classified as an **`ANALYTICAL_ESTIMATE`**, as runtime tokenization varies per prompt string.

---

### 4. Checkpoint Serialization vs. Loaded Model Parameter Dtype (NUM-04)

- **Prior Draft Issue:** Prior text asserted that the $3.75\text{ GB}$ file size was caused by "vision encoder weights saved in FP32 format".
- **Correction:** Without downloading or opening `model.safetensors` headers locally, the internal tensor dtypes on disk cannot be confirmed. Inferring FP32 storage of specific layers from total file size alone is technically unsupported.
- **Defensible Statement:**
  1. Checkpoint file size on disk is $\approx \mathbf{3.75\text{ GB}}$ (`MEASURED`).
  2. The internal storage dtype of individual tensors in the safetensors archive is unverified.
  3. In [`evaluate_vitextvqa.py#L215-L220`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/evaluate_vitextvqa.py#L215-L220), `model` is instantiated via `AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16)`. Hugging Face casts all model weights to `bfloat16` upon instantiation on device, bounding in-memory parameter footprint to $\le \mathbf{2.10\text{ GB}}$.

---

### 5. Budget Table Consistency & Classification Integrity (NUM-05)

All quantities in Section 5.2 of `TASK_18A_RESOURCE_PREFLIGHT_CORRECTION_REPORT.md` now agree with the equations and text:

| Resource Item | Value / Range | Classification | Derivation / Verification Basis |
| :--- | :---: | :---: | :--- |
| **Model Core Architecture** | $\approx 818\text{M\ params}$ | `DERIVED_FROM_VERIFIED_CONFIG` | InternViT ($304\text{M}$) + Projector ($20\text{M}$) + Qwen-0.5B with tied embeds ($494\text{M}$). |
| **Model Parameter Ceiling** | $1.05\text{B\ params}$ | `ANALYTICAL_UPPER_BOUND_ASSUMPTION` | External packaging ceiling; not verified via weight instantiation. |
| **Static Weight Footprint (Core)** | $1.64\text{ GB}$ | `DERIVED_FROM_VERIFIED_CONFIG` | $818\text{M params} \times 2\text{ bytes (bfloat16)}$. |
| **Static Weight Footprint (Ceiling)**| $\le 2.10\text{ GB}$ | `ANALYTICAL_UPPER_BOUND_ASSUMPTION` | $1.05\text{B params} \times 2\text{ bytes (bfloat16)}$. |
| **Safetensors Storage Size** | $3.75\text{ GB}$ | `MEASURED` | File size on Hugging Face; layer-level serialization dtypes unverified. |
| **Candidate Manifest Max Tiles** | $7\text{ tiles}$ | `MEASURED` | Verified on `68.jpg`, `76.jpg`, `126.jpg` via `dynamic_preprocess_grid()` ($6\text{ grid} + 1\text{ thumb}$). |
| **Max Visual Sequence Length** | $1,792\text{ tokens}$ | `MEASURED` | $7\text{ tiles} \times 256\text{ patch tokens/tile}$. |
| **Stage 1 Phase 1 Peak (ViT)** | $\approx 0.35 - 0.45\text{ GB}$ | `ANALYTICAL_ESTIMATE` | InternViT forward activations; freed before language model suffix forward. |
| **Stage 1 Suffix Attn Payload (Base)**| $73.5\text{ MB}$ / $147.0\text{ MB}$ | `DERIVED_FROM_VERIFIED_CONFIG` | Single branch ($73.5\text{ MB}$); paired cold extraction ($147.0\text{ MB}$). |
| **Stage 1 Suffix Attn Payload (Draft)**| $98.5\text{ MB}$ / $172.0\text{ MB}$ | `DERIVED_FROM_VERIFIED_CONFIG` | Single branch ($98.5\text{ MB}$); paired cold extraction ($172.0\text{ MB}$). |
| **Stage 1 Phase 3 Peak (Suffix Total)**| $\approx 0.20 - 0.22\text{ GB}$ | `ANALYTICAL_ESTIMATE` | Attentions ($0.17\text{ GB}$) + shared KV ($0.045\text{ GB}$) + grid_feats ($0.005\text{ GB}$). |
| **Stage 1 Conservative Total Budget** | $\mathbf{\approx 0.65 - 0.70\text{ GB}}$ | `ANALYTICAL_ESTIMATE` | Conservative bound absorbing Phase 1 & Phase 3 without immediate GC. |
| **Stage 2 Calculated KV Payload** | $\mathbf{45.40\text{ MB}}$ | `DERIVED_FROM_VERIFIED_CONFIG` | $2 \times 2 \times 24 \times 1,847 \times 2 \times 64 \times 2\text{ bytes}$ ($B=2, L=24, T=1,847, H=2, D=64$). |
| **Stage 2 Logits Buffer** | $\mathbf{1.22\text{ MB/step}}$ | `DERIVED_FROM_VERIFIED_CONFIG` | $2 \times 1 \times 151,936 \times 4\text{ bytes}$. |
| **Stage 2 Working & Allocator Budget** | $\mathbf{\approx 0.20 - 0.35\text{ GB}}$ | `ANALYTICAL_ESTIMATE` | Base KV ($0.045\text{ GB}$) + logits ($0.001\text{ GB}$) + allocator block pooling ($\approx 0.25\text{ GB}$). |
| **Total Peak Active Tensor Memory** | $\mathbf{\approx 3.0 - 3.5\text{ GB}}$ | `ANALYTICAL_ESTIMATE` | Static weights ($2.1\text{ GB}$) + Stage 1 peak ($0.67\text{ GB}$) + Stage 2 active ($0.35\text{ GB}$) + logits. |
| **PyTorch Allocator Reserved Memory** | $\mathbf{\approx 4.5 - 5.5\text{ GB}}$ | `NOT_MEASURED` | Dynamic allocator chunk pooling on real CUDA hardware (requires empirical profiling). |
| **Peak VRAM During Real Inference** | $\mathbf{\approx 5.0 - 6.0\text{ GB}}$ | `NOT_MEASURED` | Analytical bound only; unmeasured pending authorized Gate B run. |
| **Minimum Safe Admission Threshold** | $\mathbf{\ge 8.0\text{ GB}}$ | `ADEQUATELY JUSTIFIED` | Provides $\ge 2.0\text{ GB}$ ($>40\%$) margin above peak allocator reserve. |
| **Host System RAM Requirement** | $\mathbf{\ge 8.0\text{ GB}}$ | `ADEQUATELY JUSTIFIED` | Checkpoint deserialization, memory mapping, and Python heap overhead. |
| **Working Volume Free Disk Space** | $\mathbf{\ge 15.0\text{ GB}}$ | `ADEQUATELY JUSTIFIED` | Model weights cache ($2.5\text{ GB}$) + dataset images ($1\text{ GB}$) + wheels/scratch. |

---

## 3. Working Tree Verification

Execution of `git status --short` confirms strict scope preservation:
- Documentation files modified or created during Task 18A:
  - `docs/research/TASK_18A_RESOURCE_PREFLIGHT_CORRECTION_REPORT.md` (updated)
  - `docs/research/TASK_18A_QA_FINDINGS_RECONCILIATION.md` (updated)
  - `docs/research/TASK_18A_INDEPENDENT_ACCEPTANCE_QA.md` (preserved intact)
  - `scripts/benchmark/colab_gate_b_preflight.py` (preserved intact)
- All pre-existing uncommitted OPT-01 through OPT-06 implementation and test files remain completely preserved and untouched.
- No `git reset`, `git checkout`, `git clean`, commit, or push was performed.

---

## 4. Final Closeout Status & Gate B Notice

```
+--------------------------------------------------------------------------------+
|                         TASK 18A CLOSEOUT STATUS                               |
|                                                                                |
|                    DOCUMENTATION CLOSEOUT COMPLETE                             |
+--------------------------------------------------------------------------------+
```

### Explicit Gate B Notice
**Gate B execution and real-model inference remain STRICTLY NOT AUTHORIZED.**  
No model weights were downloaded, no forward passes were run, and no Colab instances were launched. Execution of Gate B is solely reserved for Project Owner formal sign-off.
