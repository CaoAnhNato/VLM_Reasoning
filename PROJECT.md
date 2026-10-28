# Project: SG-VCD >80% Accuracy Upgrade Campaign

## Architecture
- **Target Model**: `5CD-AI/Vintern-1B-v3_5` (`InternViT-300M` visual backbone + `Qwen2.5-0.5B-Instruct` language model).
- **Stage 1 (Spatial Grounding)**:
  - Attention extraction via KL-divergence against uniform prior: $w_t = D_{KL}(p_t \parallel U(\Omega))$.
  - Text-density high-frequency gradient prior filtering to reject non-text visual sinks.
  - Tighter bounding box fusion to eliminate magnification collapse ($M = 448 / s \ge 1.0\times$).
  - Adaptive seed propagation across OCR tokens / text regions.
- **Stage 2 (Answer Generation & Contrastive Decoding)**:
  - Contrastive decoding: $z_{calibrated} = (1 + \alpha_t) z_{pos} - \alpha_t z_{neg}$.
  - Dynamic contrast factor $\alpha_t$: Single $\tanh(S_{spatial})$ scaling with entropy modulation.
  - Relaxed candidate pruning: $\beta_0 \approx 0.03 - 0.04$ preserving subordinate entity tokens.
- **Evaluation & Benchmarking**:
  - Full 400 human-annotated images (1,222 questions) from 4 workspaces.
  - Stage 1 metrics: `Mean GT Inclusion Recall` ($\ge 80\%$), `Multi-Box Text Coverage` ($\ge 80\%$), `Sub-box Coverage Recall` ($\ge 80\%$).
  - Stage 2 metrics: `Token F1 (Verified GT)` ($> 80\%$), `Substring Match / Substring Inclusion` ($> 80\%$), `Relaxed Accuracy (ANLS)` ($> 80\%$).
- **Hardware & Environment**:
  - Local: RTX 3050 (4GB VRAM) for local-first smoke tests, unit tests, and E2E verification.
  - Remote: NVIDIA L4 GPU (22 GB VRAM) on Google Colab (`improve_SG-VCD.ipynb`) via `colab-mcp` for 400-image benchmark runs.

---

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Substring Match & ANLS Metrics Engine | Implement `compute_substring_match`, `compute_anls` ($\tau=0.5$), and Stage 2 aggregators in `metrics.py` | M1 | Survey (Explorer 2) |
| 2 | Benchmark Harness Metrics Export | Update `evaluate_vitextvqa.py` to stream and export `substring_verified` and `anls_verified` to CSV/JSON | M1 | Survey (Explorer 2) |
| 3 | Master Test Suite Metric Tests | Add Tier 1 tests for Substring Match and ANLS in `tests/test_tier1_features.py` | M1 | Survey (Explorer 2) |
| 4 | Tighter Bounding Box Fusion | Replace unconstrained outer union in `pipeline.py:534-546` to prevent magnification collapse ($M \ge 1.0\times$) | M2 | Survey (Explorer 1) |
| 5 | Non-Text Visual Sink Mitigation | Apply high-frequency gradient attention filtering so attention does not sink into smooth scene objects | M2 | Survey (Explorer 1) |
| 6 | Contrast Factor $\alpha_t$ Restoration | Eliminate redundant double $\tanh$ scaling across `pipeline.py` and `dynamic_cd.py`, restoring true contrast range | M2 | Survey (Explorer 1) |
| 7 | Plausibility Truncation Tuning | Loosen $\beta_0$ to 0.03–0.04 to prevent premature truncation of true subordinate entity tokens | M2 | Survey (Explorer 1) |
| 8 | Multi-Tile High-Resolution Evidence | Configure multi-tile visual evidence on Colab L4 GPU (22GB VRAM) to combine context and high-res focal patch | M2 | Survey (Explorer 1 & 3) |
| 9 | Local Verification, Review & Forensic Audit | Run 100% tests locally (`tests/run_e2e_tests.py`), AST anti-cheating & zero-leakage verification | M3 | Survey (Explorer 2) |
| 10 | Git Staging, Commit & Push | Track `scripts/benchmark/sg_vcd/`, `metrics.py`, `tests/` and push clean commits to `CaoAnhNato/VLM_Reasoning` | M4 | Survey (Explorer 3) |
| 11 | Colab L4 400-Image Benchmark Execution | Execute `evaluate_vitextvqa.py` on Colab L4 GPU, stream & download artifacts to `results/sg_vcd_400_benchmark/` | M5 | Survey (Explorer 3) |
| 12 | Metric Verification & Documentation Update | Verify all Stage 2 metrics >80% and Stage 1 >=80%; update `docs/Proposed_Method_SG_VCD.md` | M6 | ORIGINAL_REQUEST |

---

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Metrics Engine & Benchmark Runner Infrastructure | Features 1, 2, 3 (`metrics.py`, `evaluate_vitextvqa.py`, `tests/test_tier1_features.py`) | none | **DONE** (Passed Gate: 71/71 tests, 2x APPROVE, 2x Challenger APPROVE, Auditor CLEAN) |
| M2 | Algorithmic Upgrades (Spatial & Contrastive) | Features 4, 5, 6, 7, 8 (`pipeline.py`, `dynamic_cd.py`, `bbox_selector.py`, `roi_cropper.py`) | M1 | **DONE** (Implemented by Worker M2; 71/71 tests pass, CUDA smoke test pass) |
| M3 | Local Testing, Review, Challenge & Forensic Audit | Feature 9 (`tests/run_e2e_tests.py`, Reviewers, Challengers, Auditor) | M2 | **DONE** (Passed Gate: 71/71 tests, 14/14 spatial adversarial, 17/17 logit adversarial, Auditor CLEAN) |
| M4 | Git Staging, Commit & Remote Push | Feature 10 (`CaoAnhNato/VLM_Reasoning:main`) | M3 | **DONE** (Commit `c6a1ba3` pushed to `origin main`) |
| M5 | Colab L4 Full 400-Image Benchmark Execution | Feature 11 (`improve_SG-VCD.ipynb`, artifact download) | M4 | **DONE** (1,222/1,222 Qs evaluated, all 7 artifacts retrieved) |
| M6 | Closed-Loop Verification & Documentation (Round 1) | Feature 12 (Target evaluation >80%, `docs/Proposed_Method_SG_VCD.md`) | M5 | **DONE** (Round 1 Benchmark complete: Substring 76.94%, F1 71.77%, ANLS 67.37%) |

### Round 2 Milestones (Multi-Worker Sharding & >80% Metric Convergence)
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| R2-1 | Multi-Worker Data Sharding Engine & Stage 1/2 Upgrades | Sharding CLI (`--num-workers`, `--worker-id`, `--multi-proc`, atomic merge); Stage 1 horizontal line preservation; Stage 2 filler stripping | M6 | **DONE** (Implemented by Worker R2-1) |
| R2-2 | Local Verification, 20-Sample Pre-Test & Forensic Audit | Unit tests, 20-sample multi-worker CUDA test, Reviewer & Auditor | R2-1 | **DONE** (Passed Gate: 71/71 master tests, 5/5 unit tests, 11/11 sharding adversarial, 12/12 preamble adversarial, Auditor CLEAN) |
| R2-3 | Git Staging, Commit & Remote Push | Commit and push to `CaoAnhNato/VLM_Reasoning:main` | R2-2 | **IN_PROGRESS** |
| R2-4 | Colab L4 Sharded Benchmark Execution | 20-sample pre-test on L4 GPU -> Full 400-image sharded run (4-6 workers) -> Artifact download | R2-3 | PLANNED |
| R2-5 | Final Metric Verification (>80%) & SOTA Documentation | Verify all Stage 2 metrics >80.0% and Stage 1 >=80.0%; Update `docs/Proposed_Method_SG_VCD.md` | R2-4 | PLANNED |

---

## Interface Contracts

### `scripts/benchmark/metrics.py` ↔ `scripts/benchmark/evaluate_vitextvqa.py`
- `compute_substring_match(prediction: str, ground_truths: Sequence[str]) -> float`: Returns float in $[0.0, 1.0]$.
- `compute_anls(prediction: str, ground_truths: Sequence[str], tau: float = 0.5) -> float`: Returns float in $[0.0, 1.0]$.
- `compute_stage2_metrics(...) -> Dict[str, float]`:
  - Contains `"substring_gt"`, `"substring_verified"`, `"anls_gt"`, `"anls_verified"`.
- `MetricsAggregator.summary()["stage2"]`:
  - Contains `"substring_match_gt_pct"`, `"substring_match_verified_pct"`, `"anls_gt_pct"`, `"anls_verified_pct"`.

### `scripts/benchmark/sg_vcd/` ↔ `scripts/benchmark/evaluate_vitextvqa.py`
- `SGVCDPipeline.forward(image: Image.Image, question: str, ...) -> SGVCDOutput`:
  - `pred_text: str`: Generated answer text.
  - `pred_bbox: List[int]`: `[x1, y1, x2, y2]` in pixel coordinates.
  - `stage1_metrics`: Spatial bounding box dictionary.

---

## Code Layout
- `scripts/benchmark/sg_vcd/`:
  - `__init__.py`: Package entry.
  - `pipeline.py`: SG-VCD main pipeline.
  - `dynamic_cd.py`: Contrastive decoding logic & $\alpha_t$ / $\beta_t$ formulation.
  - `attention_extractor.py`: KL-divergence attention map extraction.
  - `bbox_selector.py`: Spatial bounding box proposal & filtering.
  - `adaptive_propagator.py`: Text seed propagation.
  - `roi_cropper.py`: Visual evidence cropping & multi-tile assembly.
- `scripts/benchmark/`:
  - `metrics.py`: Stage 1 & Stage 2 metrics engine.
  - `evaluate_vitextvqa.py`: 400-image benchmark harness.
- `tests/`:
  - `run_e2e_tests.py`: Master runner (Tiers 1–4).
  - `test_tier1_features.py`: Feature coverage unit tests.
  - `test_tier2_boundaries.py`: Boundary and edge case tests.
  - `test_tier3_combinations.py`: Cross-feature integration tests.
  - `test_tier4_workloads.py`: Hardware safety & anti-cheating AST audit.

---

## Multi-Worker Data Sharding Protocol on Single L4 GPU (Mandatory User Directive)

### 1. Hardware Context & Capacity Audit
- **Remote Environment**: Google Colab L4 GPU.
- **Hardware Capacity**: 22.03 GB VRAM, 53.0 GB System RAM, 235.7 GB Disk.
- **Measured Single-Worker Footprint**:
  - GPU VRAM: 2.31 GB allocated (only ~10.5% of 22.03 GB capacity).
  - System RAM: 4.60 GB allocated (only ~8.7% of 53.0 GB capacity).
  - Throughput: 33.81 questions/min (40.29 minutes for 1,222 questions).
- **Optimization Potential**: Over 19.5 GB VRAM and 48 GB System RAM remain unused. Multi-worker data sharding on the same GPU will saturate the L4 architecture and accelerate execution by 4x–5x.

### 2. Multi-Worker Sharding Architecture
- **Worker Configuration**: Parallel Python worker processes (`num_workers` = 4 or 6) sharing `cuda:0` via separate CUDA contexts or `torch.multiprocessing`.
- **Memory Budget**:
  - 4 workers $\times$ 2.31 GB = **~9.24 GB VRAM** (Safe headroom: > 12.5 GB free).
  - 6 workers $\times$ 2.31 GB = **~13.86 GB VRAM** (Safe headroom: > 8.0 GB free).
- **Data Partitioning**: The 1,222 questions (or 400 images) are deterministically sharded across workers:
  Worker $k$ processes indices $i \equiv k \pmod K$.
- **Result Aggregation**: Each worker writes its partial predictions to `predictions_shard_{k}.jsonl`, and an aggregator combines and sorts them by question ID to produce identical global evaluation metrics.

### 3. Mandatory Pre-Test Protocol
- **MANDATORY**: Before launching any full 400-image benchmark with Multi-Worker Sharding, subagents MUST run a stability pre-test on **~20 samples** across all worker processes.
- **Pass Criteria**:
  1. Zero CUDA out-of-memory errors or context initialization crashes.
  2. Complete result aggregation without duplicate or dropped question IDs (exact match with sequential run).
  3. Peak total VRAM $\le 16.0$ GB.

### 4. Local Execution Safety Constraint (Mandatory User Directive)
- **MANDATORY**: When executing locally, workers, reviewers, and challengers MUST ONLY use synthetic/mock data (`--mock` flag, mock multi-process mode, synthetic unittests like `tests/test_adversarial_*.py`).
- **PROHIBITION**: Strictly avoid loading real 1B VLM model weights or running full benchmark datasets locally to prevent RAM/CPU exhaustion and crashes on the local host machine. Real model inference across 400 images is reserved exclusively for the remote Google Colab L4 environment.

