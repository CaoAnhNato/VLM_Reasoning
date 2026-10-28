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
| M4 | Git Staging, Commit & Remote Push | Feature 10 (`CaoAnhNato/VLM_Reasoning:main`) | M3 | **IN_PROGRESS** (Ready for commit & push) |
| M5 | Colab L4 Full 400-Image Benchmark Execution | Feature 11 (`improve_SG-VCD.ipynb`, artifact download) | M4 | PLANNED |
| M6 | Closed-Loop Verification & Documentation | Feature 12 (Target evaluation >80%, `docs/Proposed_Method_SG_VCD.md`) | M5 | PLANNED |

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
