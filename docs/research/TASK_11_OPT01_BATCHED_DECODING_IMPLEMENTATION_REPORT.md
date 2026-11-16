# TASK 11 — IMPLEMENTATION REPORT: TICKET-OPT-01 (BATCHED POSITIVE/NEGATIVE DECODING, B=2)

- **Ticket:** `TICKET-OPT-01 — Batched Positive/Negative Decoding (B=2)`
- **Priority:** `P2` (Strict Numerical and Token Parity Required)
- **Implementer:** `4argon` — Senior ML Systems Engineer
- **Independent QA:** `Antigravity (Anti)` — separate Task 12
- **Methodology Reviewer:** `ChatGPT`
- **Final Approval:** `Project Owner`
- **Execution Mode:** Local synthetic/mock only — **Gate A** (zero real Vintern model loading, zero dataset-wide execution)
- **Timestamp:** `2026-11-08 18:43 +07:00`

---

## 1. Executive Summary

`TICKET-OPT-01` has been implemented in [`scripts/benchmark/sg_vcd/pipeline.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py) and verified locally under **Gate A** via a dedicated 8-test unit and mock-integration suite in [`tests/test_opt01_batched_dual_branch_decoding.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt01_batched_dual_branch_decoding.py) plus all regression suites (`90/90` total tests passing).

Key accomplishments:
1. **Batched $B=2$ Prefill and Autoregressive Decoding:** [`SGVCDPipeline._decode_spatial_pmi_contrastive`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L1298-L1343) (delegating to [`SGVCDPipeline._run_batched_dual_branch_decode`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L733-L972)) stacks the positive evidence branch (`batch index 0`) and negative prior branch (`batch index 1`) along `dim=0` (`batched_inputs = torch.cat([pos_inputs, neg_inputs], dim=0)`) for a single $B=2$ prefill pass and feeds `next_embed_batched = torch.cat([next_embed, next_embed], dim=0)` with a single $B=2$ KV cache `batch_kv` at each decoding step.
2. **Exact $2\times$ Reduction in Transformer Forward Launches:** For $M$ generated tokens, Stage 2 transformer forward calls drop from $2 + 2M$ ($2$ prefill + $2M$ step calls at $B=1$) to $1 + M$ ($1$ prefill + $M$ step calls at $B=2$).
3. **Strict Numerical and Token Parity (`0.000e+00` Max Logit Diff, `100%` Token Identity):** Across both a deterministic 2-layer causal self-attention transformer with sinusoidal positional phase and KV cache and an actual 2-layer Hugging Face `Qwen2ForCausalLM` (`DynamicCache`), maximum absolute differences in `pos_logits`, `neg_logits`, `contrastive_logits_pre_mask`, `contrastive_logits_signed`, and `contrastive_logits_final` are `0.000e+00` ($\le 10^{-5}$) in both `torch.float32` and `torch.bfloat16`, with **100% identical token IDs** at every step.
4. **Reference Path Preserved Without Silent Fallback:** The pre-change sequential $B=1$ implementation is preserved in [`SGVCDPipeline._run_sequential_dual_branch_decode_reference`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L974-L1191) and [`SGVCDPipeline._decode_spatial_pmi_contrastive_sequential_reference`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L1345-L1389) strictly for parity testing. Production execution never silently falls back to $B=1$.

---

## 2. Governance and Scope Compliance

| Governance / Constraint Item | Status | Evidence |
|---|---|---|
| Read `.github/copilot-instructions.md` and all 3 mandatory project skills before work | **COMPLIANT** | Loaded and applied hardware/Gate A rules, research plan compliance rules, and QA evidence standards. |
| Authorized scope restricted to `TICKET-OPT-01` only | **COMPLIANT** | Only Stage 2 dual-branch batching ($B=2$) and its parity/regression tests were implemented. `OPT-04` and other tickets were not touched. |
| No real Vintern model execution (`5CD-AI/Vintern-1B-v3_5`) locally | **COMPLIANT** | Zero real model weights loaded; all tests use lightweight synthetic/mock 2-layer causal transformers on CPU. |
| No Gate B or Colab smoke/benchmark execution | **COMPLIANT** | Gate A local synthetic/mock verification only. |
| Preserve `OPT-06`, `OPT-03`, and `OPT-02` | **COMPLIANT** | Verified by `tests.test_opt06_image_cache_identity` (11/11), `tests.test_opt03_working_map_cache` (8/8), `tests.test_opt02_memory_cleanup_policy` (9/9), and AST audit in Test H. |
| Leave working tree uncommitted (`no git commit/push/reset`) | **COMPLIANT** | Verified via `git status --short`. |

---

## 3. Pre-Change Dual-Branch Decoding Audit

Before `OPT-01`, [`SGVCDPipeline._decode_spatial_pmi_contrastive`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py) executed Stage 2 contrastive decoding sequentially at batch size $B=1$:
1. **Input Preparation:**
   - Built positive visual tokens `pos_vit = torch.cat([thumb_vit] + pos_sub_vits, dim=0)` and negative visual tokens `neg_vit = torch.cat([thumb_vit] + neg_sub_vits, dim=0)` of identical shape `(N_patches, num_image_token, D)` (`D = 896` in Vintern-1B, `num_image_token = 256`).
   - Built `pos_prefix, suffix_q, _, _ = self.extractor._build_prefix_and_suffix_embeds(pos_vit, decoding_question)` and `neg_prefix, _, _, _ = self.extractor._build_prefix_and_suffix_embeds(neg_vit, decoding_question)`.
   - Concatenated `pos_inputs = torch.cat([pos_prefix, suffix_q], dim=1)` and `neg_inputs = torch.cat([neg_prefix, suffix_q], dim=1)`, both of shape `(1, L_total, D)` with identical sequence length `L_total = L_prefix + L_suffix` and zero padding.
2. **Sequential Prefill ($2\times$ Forward Calls at $B=1$):**
   - Called `pos_out = self.model.language_model.model(inputs_embeds=pos_inputs, use_cache=True, return_dict=True)` followed by `neg_out = self.model.language_model.model(inputs_embeds=neg_inputs, use_cache=True, return_dict=True)` (or `self.model.language_model(...)` when `has_backbone` is `False`).
   - Stored two separate $B=1$ KV caches (`pos_kv`, `neg_kv`) and computed `pos_logits` and `neg_logits` separately.
3. **Sequential Autoregressive Loop ($2\times$ Forward Calls per Step at $B=1$):**
   - At each step $t \in \{0, \dots, M-1\}$, [`DynamicContrastiveDecoder.decode_step`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/dynamic_cd.py#L142-L287) selected a single `next_token_id` from `(pos_logits, neg_logits)`.
   - Embedded `next_embed = embed_layer([[next_token_id]])` of shape `(1, 1, D)` and launched two separate $B=1$ transformer forward passes: one with `past_key_values=pos_kv` and one with `past_key_values=neg_kv`.

---

## 4. Tensor, Mask, Position, and KV-Cache Contract Analysis

1. **Input Tensor Contract (`pos_inputs`, `neg_inputs`):**
   - Both branches contain the exact same number of visual tiles $N_{\text{patches}} = 1 + N_{\text{roi}} + \mathbb{I}[\text{context\_tile}]$ and identical tokenized prompt template (`"Hỏi: {q}\nTrả lời ngắn gọn: "`).
   - Consequently, `pos_inputs` and `neg_inputs` always have shape `(1, L, D)` with equal sequence length `L`, equal hidden dimension `D`, identical `dtype`, and identical `device`.
   - [`SGVCDPipeline._validate_dual_branch_inputs`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L440-L485) enforces 3D rank, `B == 1`, `L > 0`, `D > 0`, `pos_inputs.shape == neg_inputs.shape`, `pos_inputs.dtype == neg_inputs.dtype`, and `pos_inputs.device == neg_inputs.device`. Any mismatch raises `ValueError` immediately (no silent padding or truncation).
2. **Attention Mask & Position IDs Contract:**
   - Because `pos_inputs` and `neg_inputs` have identical unpadded length `L`, when `attention_mask` and `position_ids` are omitted (`None`), Qwen2 / HF causal decoders construct identical causal masks and contiguous positions `0 .. L-1` for prefill and position `L + step` for decode step `step` across both batch rows.
   - When explicit `attention_mask` or `position_ids` are supplied to [`_run_batched_dual_branch_decode`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L733-L972) or [`_run_sequential_dual_branch_decode_reference`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L974-L1191), [`SGVCDPipeline._prepare_optional_mask_and_positions`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L607-L683) validates 2D rank and sequence length `L`, expands `(1, L)` to `(2, L)` or verifies that a `(2, L)` input has identical rows (`torch.equal(row[0], row[1])`), and during each decode step appends `1` to `attention_mask` (`(2, L + step + 1)`) and advances `position_ids` by `+1` (`bat_pos[:, -1:] + 1`).
3. **KV-Cache Contract (`transformers==4.57.6` and Legacy Support):**
   - Runtime inspection of `transformers==4.57.6` confirmed that `Qwen2Model.forward` uses `DynamicCache(config=self.config)` whose per-layer tensors live in `past_key_values.layers[i].keys` and `past_key_values.layers[i].values` with 4D shape `(B, H_kv, S, D_h)`.
   - [`SGVCDPipeline._extract_kv_cache_tensors`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L488-L553) and [`SGVCDPipeline._validate_kv_cache`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L556-L604) validate all three cache formats:
     1. HF `DynamicCache` / `Cache` with `.layers` (`layer.keys`, `layer.values`),
     2. Cache objects with `.key_cache` and `.value_cache` lists, and
     3. Legacy tuple/list of `(key, value)` tensor pairs.
   - At prefill, `_validate_kv_cache` asserts `expected_batch_size=2` and `expected_seq_len=prefill_seq_len`. At each decode step `step`, it asserts `expected_seq_len=prefill_seq_len + step` before the forward pass and `expected_seq_len=prefill_seq_len + step + 1` after the forward pass.

---

## 5. Batched Decoding Design ($B=2$)

### 5.1 Architectural Flow

```
Stage 2 Input Preparation (_prepare_dual_branch_stage2_inputs)
  pos_inputs: (1, L, D)  [global_thumb + Laplacian-sharpened ROI tiles + prompt]
  neg_inputs: (1, L, D)  [global_thumb + Gaussian-blurred ROI tiles + prompt]
         │
         ▼
Prefill Stacking (B=2)
  batched_inputs = torch.cat([pos_inputs, neg_inputs], dim=0)  # (2, L, D)
  prefill_out = llm(inputs_embeds=batched_inputs, use_cache=True, return_dict=True)
  batch_kv: B=2 KV cache with seq_len = L
  logits = lm_head(prefill_out.last_hidden_state[:, -1:, :])[:, -1, :]  # (2, V)
  pos_logits = logits[0:1].float()  # (1, V), batch index 0
  neg_logits = logits[1:2].float()  # (1, V), batch index 1
         │
         ▼
Autoregressive Step Loop (step = 0 .. max_new_tokens - 1)
  next_token_id, alpha_t, entropy_t, d_js_val = decoder.decode_step(pos_logits, neg_logits, ...)
  if next_token_id in (eos_id, tokenizer.eos_token_id): break
  next_embed = embed_layer([[next_token_id]])                  # (1, 1, D)
  next_embed_batched = torch.cat([next_embed, next_embed], 0)  # (2, 1, D)
  step_out = llm(inputs_embeds=next_embed_batched, past_key_values=batch_kv, use_cache=True)
  batch_kv: B=2 KV cache with seq_len = L + step + 1
  logits = lm_head(step_out.last_hidden_state[:, -1:, :])[:, -1, :]     # (2, V)
  pos_logits = logits[0:1].float()  # (1, V)
  neg_logits = logits[1:2].float()  # (1, V)
```

### 5.2 Exact Code Locations in [`scripts/benchmark/sg_vcd/pipeline.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py)

| Method | Lines | Purpose |
|---|---|---|
| [`SGVCDPipeline._validate_dual_branch_inputs`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L440-L485) | `440–485` | Validates 3D `(1, L, D)` shape, dtype, and device equality between `pos_inputs` and `neg_inputs`. |
| [`SGVCDPipeline._extract_kv_cache_tensors`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L488-L553) | `488–553` | Extracts per-layer `(K, V)` tensor pairs from `.layers`, `.key_cache`/`.value_cache`, or legacy tuple caches. |
| [`SGVCDPipeline._validate_kv_cache`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L556-L604) | `556–604` | Verifies 4D `(B, H, S, D_h)` shapes, batch size (`B=2` or `B=1`), and monotonic sequence length `S`. |
| [`SGVCDPipeline._prepare_optional_mask_and_positions`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L607-L683) | `607–683` | Validates and prepares optional `attention_mask` and `position_ids` for $B=2$ and $B=1$ paths. |
| [`SGVCDPipeline._record_decode_step_trace`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L685-L731) | `685–731` | Records per-step float32 `pos_logits`, `neg_logits`, pre-mask/signed/final contrastive logits, token IDs, and cache length. |
| [`SGVCDPipeline._run_batched_dual_branch_decode`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L733-L972) | `733–972` | Primary $B=2$ batched prefill and autoregressive contrastive decoding implementation. |
| [`SGVCDPipeline._run_sequential_dual_branch_decode_reference`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L974-L1191) | `974–1191` | Pre-change $B=1$ sequential dual-branch reference implementation for parity verification. |
| [`SGVCDPipeline._prepare_dual_branch_stage2_inputs`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L1193-L1296) | `1193–1296` | Shared Stage 2 visual/text embedding construction returning `(pos_inputs, neg_inputs, spatial_confidence)`. |
| [`SGVCDPipeline._decode_spatial_pmi_contrastive`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L1298-L1343) | `1298–1343` | Public Stage 2 entry point calling `_prepare_dual_branch_stage2_inputs` + `_run_batched_dual_branch_decode`. |
| [`SGVCDPipeline._decode_spatial_pmi_contrastive_sequential_reference`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L1345-L1389) | `1345–1389` | Reference Stage 2 entry point calling `_prepare_dual_branch_stage2_inputs` + `_run_sequential_dual_branch_decode_reference`. |

---

## 6. Numerical and Token Parity Evidence

Parity between [`_run_batched_dual_branch_decode`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L733-L972) ($B=2$) and [`_run_sequential_dual_branch_decode_reference`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L974-L1191) ($B=1$) was measured across every decoding step in `float32`:

| Model Backbone | Dtype | Steps | Max `|pos_logits|` Diff | Max `|neg_logits|` Diff | Max Pre-Mask Contrastive Diff | Max Signed Contrastive Diff | Max Final Masked Contrastive Diff | Token ID Parity |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `_SyntheticTwoLayerCausalTransformer` (2 layers, 4 heads, sinusoidal RoPE, SwiGLU) | `torch.float32` | 8 | `0.000e+00` | `0.000e+00` | `0.000e+00` | `0.000e+00` | `0.000e+00` | **100% (8/8)** |
| `_SyntheticTwoLayerCausalTransformer` (2 layers, 4 heads, sinusoidal RoPE, SwiGLU) | `torch.bfloat16` | 8 | `0.000e+00` | `0.000e+00` | `0.000e+00` | `0.000e+00` | `0.000e+00` | **100% (8/8)** |
| Hugging Face `Qwen2ForCausalLM` (2 layers, 4 Q heads, 2 KV heads, `DynamicCache`) | `torch.float32` | 6 | `0.000e+00` | `0.000e+00` | `0.000e+00` | `0.000e+00` | `0.000e+00` | **100% (6/6)** |
| Hugging Face `Qwen2ForCausalLM` (2 layers, 4 Q heads, 2 KV heads, `DynamicCache`) | `torch.bfloat16` | 6 | `0.000e+00` | `0.000e+00` | `0.000e+00` | `0.000e+00` | `0.000e+00` | **100% (6/6)** |
| `_SyntheticTwoLayerCausalTransformer` with `.key_cache`/`.value_cache` + explicit `attention_mask` & `position_ids` | `torch.float32` | 5 | `0.000e+00` | `0.000e+00` | `0.000e+00` | `0.000e+00` | `0.000e+00` | **100% (5/5)** |
| Direct `_DirectCausalLMWrapper` (`has_backbone=False`) | `torch.float32` | 5 | `0.000e+00` | `0.000e+00` | `0.000e+00` | `0.000e+00` | `0.000e+00` | **100% (5/5)** |

All configurations satisfy the mandatory tolerance $\le 10^{-5}$ with zero token deviations.

---

## 7. Forward-Call Reduction Evidence

In [`TestOPT01BatchedDualBranchDecoding.test_D_forward_call_reduction`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt01_batched_dual_branch_decoding.py#L958-L1021), the 2-layer causal transformer backbone records every `forward` call (`batch_size`, `q_len`, `past_len`, `is_prefill`) for `max_new_tokens = 6` (with input sequence length `L = 22`):

| Path | Prefill Forward Calls | Decode Step Forward Calls | Total Forward Calls ($M=6$) | General Formula ($M$ tokens) | Batch Size per Call |
|---|---:|---:|---:|---|---:|
| **Pre-change Sequential Reference ($B=1$)** | `2` (`q_len=22, past_len=0`) | `12` (`2` calls/step, `q_len=1, past_len=22..27`) | `14` | $2 + 2M$ | `1` |
| **OPT-01 Batched Dual-Branch ($B=2$)** | `1` (`q_len=22, past_len=0`) | `6` (`1` call/step, `q_len=1, past_len=22..27`) | `7` | $1 + M$ | `2` |
| **Reduction Factor** | **$2.0\times$ fewer (`-50%`)** | **$2.0\times$ fewer (`-50%`)** | **$2.0\times$ fewer (`-50%`)** | **$\frac{1+M}{2+2M} = 0.5\times$** | — |

Furthermore, when EOS is emitted at step 0 (`test_F_eos_and_stopping_behavior`, Case 1), `_run_batched_dual_branch_decode` executes **1** prefill forward pass and **0** decode-step forward passes.

---

## 8. Branch Separation and KV-Cache Validation

In [`TestOPT01BatchedDualBranchDecoding.test_C_branch_separation_and_cache_correctness`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt01_batched_dual_branch_decoding.py#L871-L954):
1. **Batch Index Assignment:** Batch index `0` is strictly the positive branch and matches a standalone $B=1$ positive run (`max_abs_diff = 0.0 <= 1e-5`); batch index `1` is strictly the negative branch and matches a standalone $B=1$ negative run (`max_abs_diff = 0.0 <= 1e-5`).
2. **Zero Cross-Branch Leakage:**
   - Perturbing the negative prefix (`neg_B -> neg_C`) while keeping `pos_A` fixed produces **`0.0` (`0.000e+00`)** difference in batch index `0` keys `k[0:1]` and values `v[0:1]` across all layers and all 4 autoregressive decode steps.
   - Perturbing the positive prefix (`pos_A -> pos_D`) while keeping `neg_B` fixed produces **`0.0` (`0.000e+00`)** difference in batch index `1` keys `k[1:2]` and values `v[1:2]` across all layers and all 4 autoregressive decode steps.
3. **Cache Sequence Length Advancement:** Starting from prefill length `15`, `_validate_kv_cache` confirms `expected_seq_len` advances strictly as `15 -> 16 -> 17 -> 18 -> 19` (`+1` per generated token).

---

## 9. Interaction with OPT-06, OPT-03, and OPT-02

- **OPT-06 (Stable Image Identity and Cache Key):** Untouched and fully preserved. `SGVCDPipeline.run` still resolves `ImageCacheIdentity` once and shares `cached_thumb_vit` with Stage 2.
- **OPT-03 (Image-Level Working-Map Caching):** Untouched and fully preserved. Stage 1 working-map cache reuse across same-image questions operates before Stage 2 decoding.
- **OPT-02 (Coordinated Memory Cleanup Policy):** Fully preserved. [`_run_batched_dual_branch_decode`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L733-L972) uses only lightweight `del` of local tensor references (`prefill_out`, `step_out`, `batch_kv`, `pos_logits`, `neg_logits`) and never calls `gc.collect()` or `torch.cuda.empty_cache()` per question or per token step (verified via AST inspection in `test_H_existing_pipeline_regression_and_opt02_03_06_preservation` and `test_opt02_memory_cleanup_policy.py::test_I`).
- Note on existing untracked synthetic test mocks in [`tests/test_opt06_image_cache_identity.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt06_image_cache_identity.py), [`tests/test_opt03_working_map_cache.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt03_working_map_cache.py), and [`tests/test_opt02_memory_cleanup_policy.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt02_memory_cleanup_policy.py): Their minimal `_DummyAttentionOutput` helper previously hardcoded `batch_size = 1` (`torch.ones((1, seq_len, hidden_dim))`) regardless of `inputs_embeds.shape[0]`. Those three untracked test helpers were updated to propagate `batch_size = int(inputs_embeds.shape[0])` and `total_len = past_len + q_len` so that `_run_batched_dual_branch_decode` can enforce strict $B=2$ output and KV-cache validation without any fallback.

---

## 10. Files Modified and Created

| File | Status | Lines Changed / Size | Description |
|---|---|---|---|
| [`scripts/benchmark/sg_vcd/pipeline.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py) | Modified | `+1002 / -169` (`lines 439–1390`) | Implemented $B=2$ batched dual-branch prefill & decoding, KV-cache & input validators, step-trace recorder, and $B=1$ sequential reference methods. |
| [`tests/test_opt01_batched_dual_branch_decoding.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt01_batched_dual_branch_decoding.py) | Created | `1,347 lines` | Dedicated unit & mock-integration test suite covering mandatory `Tests A–H`. |
| [`tests/test_opt02_memory_cleanup_policy.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt02_memory_cleanup_policy.py) | Updated (untracked) | `lines 76–128` | Updated `_DummyAttentionOutput` / `_DummyInnerLLM.forward` to respect `inputs_embeds.shape[0]` (`B=2`). |
| [`tests/test_opt03_working_map_cache.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt03_working_map_cache.py) | Updated (untracked) | `lines 68–120` | Updated `_DummyAttentionOutput` / `_DummyInnerLLM.forward` to respect `inputs_embeds.shape[0]` (`B=2`). |
| [`tests/test_opt06_image_cache_identity.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt06_image_cache_identity.py) | Updated (untracked) | `lines 58–110` | Updated `_DummyAttentionOutput` / `_DummyInnerLLM.forward` to respect `inputs_embeds.shape[0]` (`B=2`). |
| [`docs/research/TASK_11_OPT01_BATCHED_DECODING_IMPLEMENTATION_REPORT.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/TASK_11_OPT01_BATCHED_DECODING_IMPLEMENTATION_REPORT.md) | Created | This report | Full implementation and Gate A verification report for `TICKET-OPT-01`. |

---

## 11. Test Suite Design (`test_opt01_batched_dual_branch_decoding.py`)

[`tests/test_opt01_batched_dual_branch_decoding.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt01_batched_dual_branch_decoding.py) implements a real 2-layer causal self-attention transformer (`_SyntheticTwoLayerCausalTransformer`) with multi-head QKV projections, sinusoidal position-dependent phase rotation, causal masking, SwiGLU MLP, and KV caching, plus tests against Hugging Face `Qwen2ForCausalLM`:

- **Test A (`test_A_two_layer_transformer_numerical_parity`):** Verifies `pos_logits`, `neg_logits`, `contrastive_logits_pre_mask`, `contrastive_logits_signed`, and `contrastive_logits_final` match within $\le 10^{-5}$ (observed `0.0`) across 8 steps on `_SyntheticTwoLayerCausalTransformer` and 6 steps on 2-layer `Qwen2ForCausalLM` in both `float32` and `bfloat16`.
- **Test B (`test_B_token_sequence_identity_across_diverse_workloads`):** Verifies 100% token sequence identity across multi-tile ROI crops, context tiles, low-confidence fallback (`stage1_kl_score=0.0`), ESSC cooldown, and two-phase colon (`:`) delimiter reasoning transition.
- **Test C (`test_C_branch_separation_and_cache_correctness`):** Verifies batch index `0` vs `1` separation, zero cross-branch leakage (`0.0` cache/logit change when perturbing the opposite branch), and `+1` cache sequence length advancement per step.
- **Test D (`test_D_forward_call_reduction`):** Verifies batched execution uses `1` prefill + `M` decode forward calls (`B=2`) vs `2` prefill + `2M` decode forward calls (`B=1`) in the reference path.
- **Test E (`test_E_masks_positions_and_cache_apis`):** Verifies HF `DynamicCache` (`.layers`), `.key_cache`/`.value_cache` list caches, legacy `(K, V)` tuple caches, direct `language_model(...)` CausalLM path (`has_backbone=False`), and explicit `attention_mask` and `position_ids` propagation.
- **Test F (`test_F_eos_and_stopping_behavior`):** Verifies step-0 immediate EOS (`1` prefill call, `0` step calls), mid-sequence EOS at step 3, and `max_new_tokens` cap parity.
- **Test G (`test_G_error_and_shape_handling`):** Verifies clear `ValueError` exceptions on mismatched branch sequence lengths, hidden dims, batch sizes, dtypes, asymmetric `attention_mask`, divergent `position_ids`, and invalid/non-advancing KV caches.
- **Test H (`test_H_existing_pipeline_regression_and_opt02_03_06_preservation`):** Verifies full `SGVCDPipeline.run` Stage 1 (`stage1_bbox`, `stage1_bboxes`, `kl_score`, `isotropic_box`, `magnification`) and Stage 2 (`pred_text`, `vcd_alpha_mean`, `entropy_mean`) parity plus AST verification of zero per-question `gc.collect()` / `empty_cache()`.

---

## 12. Executed Test Commands and Results

All required commands from Section 7 (Test I) were executed using `..\Document AI\VLM_Reasoning\venv\Scripts\python.exe`:

| # | Command | Tests Run | Result | Duration | Exit Code |
|---|---|---:|---|---:|---:|
| 1 | `& "..\Document AI\VLM_Reasoning\venv\Scripts\python.exe" -m unittest tests.test_opt01_batched_dual_branch_decoding -v` | `8` | `OK (8/8 passed)` | `5.225s` | `0` |
| 2 | `& "..\Document AI\VLM_Reasoning\venv\Scripts\python.exe" -m unittest tests.test_opt02_memory_cleanup_policy -v` | `9` | `OK (9/9 passed)` | `10.219s` | `0` |
| 3 | `& "..\Document AI\VLM_Reasoning\venv\Scripts\python.exe" -m unittest tests.test_opt03_working_map_cache -v` | `8` | `OK (8/8 passed)` | `5.448s` | `0` |
| 4 | `& "..\Document AI\VLM_Reasoning\venv\Scripts\python.exe" -m unittest tests.test_opt06_image_cache_identity -v` | `11` | `OK (11/11 passed)` | `2.005s` | `0` |
| 5 | `& "..\Document AI\VLM_Reasoning\venv\Scripts\python.exe" -m unittest tests.test_sg_vcd tests.test_adversarial_spatial_processing -v` | `54` | `OK (54/54 passed)` | `0.350s` | `0` |
| 6 | `& "..\Document AI\VLM_Reasoning\venv\Scripts\python.exe" -m unittest tests.test_tier4_workloads.TestTier4RealWorldWorkloads.test_05_anti_hardcoding_and_heuristics_audit -v` | `1` | `OK (1/1 passed)` | `0.324s` | `0` |
| **Total** | **All 6 Gate A + Regression Suites** | **`91`** | **`91/91 PASS`** | **`23.571s`** | **`0`** |

---

## 13. Gate A Assessment

| Gate A Criterion | Requirement | Observed Evidence | Verdict |
|---|---|---|---|
| Prefill batching ($B=2$) | Single forward pass with `[pos_inputs, neg_inputs]` | `1` prefill call with `batch_size=2` (`Test D`) | **PASS** |
| Per-step decoding batching ($B=2$) | `1` forward pass per token step with `batch_kv` | `M` step calls with `batch_size=2` vs `2M` in reference (`Test D`) | **PASS** |
| Positive / Negative / Contrastive logit parity | Max abs diff $\le 10^{-5}$ in `float32` | `0.000e+00` across all steps (`Test A`, `Test E`) | **PASS** |
| Token ID sequence parity | 100% identical token IDs under greedy decoding | 100% match across all scenarios (`Test A`, `Test B`, `Test E`, `Test F`, `Test H`) | **PASS** |
| Branch separation & KV cache integrity | Zero cross-branch leakage; `+1` seq_len per step | `0.0` cross-branch perturbation diff; strict `_validate_kv_cache` (`Test C`, `Test G`) | **PASS** |
| Regression & OPT-06 / OPT-03 / OPT-02 preservation | Zero regressions across all existing suites | `91/91` unit and regression tests pass | **PASS** |

**Overall Gate A Verdict:** **PASS**

---

## 14. Risks, Limitations, and Gate B Verification Plan

1. **Local Gate A Scope Limitation:** All verification in Task 11 was performed on CPU using deterministic 2-layer causal transformers (`_SyntheticTwoLayerCausalTransformer` and 2-layer `Qwen2ForCausalLM`) in `float32` and `bfloat16`. Real `5CD-AI/Vintern-1B-v3_5` CUDA kernels (FlashAttention-2 / cuDNN SDPA on T4/L4/A100 in Colab) were not executed locally per hardware governance rules.
2. **CUDA Kernel Numerical Reduction Order in Gate B:** While CPU `sdpa` and `eager` kernels exhibit `0.000e+00` batching difference, GPU tensor-core GEMM tile selection can occasionally vary by $\sim 10^{-6}$ in `float32` accumulator order between $B=1$ and $B=2$. Gate B on Colab will verify that on real `Vintern-1B-v3_5` weights over the 20-image ViTextVQA smoke slice:
   - Stepwise logit differences remain $\le 10^{-5}$ (or produce 100% identical greedy token trajectories and identical EM/F1/ANLS scores), and
   - Wall-clock Stage 2 decoding latency decreases as expected while peak VRAM remains well within budget.

---

## 15. Working Tree Status

`git status --short` output at task completion:

```text
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
?? tests/test_opt06_image_cache_identity.py
```

All changes remain uncommitted in the working tree as required.

---

## 16. Handoff for Task 12 Independent QA (Anti)

`TICKET-OPT-01` implementation and Gate A verification are complete and ready for **Task 12 Independent Read-Only Acceptance QA** by `Antigravity (Anti)`:
- Primary implementation: [`scripts/benchmark/sg_vcd/pipeline.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L439-L1390)
- Primary test suite: [`tests/test_opt01_batched_dual_branch_decoding.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt01_batched_dual_branch_decoding.py)
- Implementation report: [`docs/research/TASK_11_OPT01_BATCHED_DECODING_IMPLEMENTATION_REPORT.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/TASK_11_OPT01_BATCHED_DECODING_IMPLEMENTATION_REPORT.md)

