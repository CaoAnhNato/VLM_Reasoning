# Training-Free VLM Track Research Protocol

This document defines the invariant scientific, operational, and platform methodology for iterative training-free VLM decoding research executed by `/run-workflow` (`scripts/workflows/vlm_track_research/main.py`). All numeric caps, branch targets, mutable path lists, and track toggles are resolved at runtime from `configs/tracks/workflow_defaults.json` and `configs/tracks/track_{1,2}.json` via `scripts/workflows/track_ops.py` and are never duplicated here.

## 1. Domain Glossary & Baseline Contract

- **Frozen Base (`base/frsa-011-frozen`)**: Pairs immutable pure Layer-0 lexical + Poisson-Bregman ROI foveation (`src/nevfe/core/frozen_foveation.py`) with locked `IG-VFED (1, 1, 1)` (`lambda_param = 0.50`, `beta_param = 0.05`) inside `NEVFEEngine.decode_step()`, routed uniformly through `VinternNEVFEAdapter.generate()`.
- **Track Scope Separation**:
  - **Track 1 (`Team 1 - TF`)**: Post-`lm_head` output-trajectory calibration and dynamic Bellman/Lookahead decoding targeting `BEIT-D Issue #1` under zoomed ROI tiles. Dataset-artifact `BEIT-D Issue #2` (adjacent signboard overflow from ROI border padding) is explicitly excluded.
  - **Track 2 (`Team 2 - TF`)**: Read-only intra-decoder self-verification across upper decoder layers (`[0.5, 1.0)` relative depth). Strictly locked to single-step decoding (`decoder_forward_calls_per_step_max <= 1` per stream) and zero KV-cache perturbation (`max_kv_diff == 0.0`).
- **Evaluation Regimes**:
  - **Calibration Bank v2 (`data/calibration_bank_v2/`)**: Local CPU-only (`CUDA_VISIBLE_DEVICES=""`) sanity & stability diagnostics across 30 `is_calibration: true` QIDs without loading `Vintern-1B` locally.
  - **Stage A Probe**: 100-QID stratified probe (`30` calibration + `70` held-out QIDs summarized by `comp_70`) with Leave-One-Out (LOO) net-contribution gating (`FULL_PASS | SUBMASK_HINT | FAIL`).
  - **Stage B Benchmark**: Full 1,222-QID benchmark evaluated on Colab L4 strictly after Stage A `FULL_PASS`.
  - **Blindfold Rule**: Sample-level predictions and text diffs may only be inspected for the 30 `is_calibration: true` QIDs. Probe `CLASS_A..CLASS_E` aggregates summarize the 100-QID probe manifest; the remaining 1,122 held-out QIDs report aggregate metrics only.

## 2. Theoretical Guardrails

1. **Zero Training & Zero Heuristics**: Training, LoRA, learned gating weights, external OCR engines, regexes, stopword lists, string-manipulation heuristics (`.strip()`, `.replace()`, etc.), and ground-truth key access are forbidden across `src/nevfe/core/` and all track diffs in `src/nevfe/adapter/` and `src/nevfe/pipeline.py`.
2. **Immutable Files**: `scripts/benchmark/metrics.py`, `tests/test_metrics_relaxation.py`, and `src/nevfe/core/frozen_foveation.py` must maintain zero diff against `base/frsa-011-frozen` and match pinned LF-normalized SHA-256 digests.
3. **Parameter Discipline**: `lambda_param = 0.50` and `beta_param = 0.05` remain locked inside `AlgorithmConfig`. Track iterations may add at most the cumulative scalar budget in `workflow_defaults.json` (`max_new_scalars`), preferring a single adaptive expression, and must expose named components via `COMPONENT_NAMES` and binary `ablation_mask`.
4. **Core Engine Purity**: `src/nevfe/core/engine.py` remains backbone-agnostic (no hardcoded backbone vocabulary/hidden/layer integers), stays within the AST line budget, and binds line-2 `# SPEC_REF` to the exact LF-normalized SHA-256 of the committed iteration spec (`docs/specs/track_<id>/spec_iter_<k>.md`). Implementers must never modify `docs/specs/` after `spec_sha` is committed.
5. **Backbone-Agnostic Layer Indexing**: Any adapter upper-layer hook indexing must use `resolve_relative_decoder_layer_indices(num_layers, fraction=(0.5, 1.0))` with `output_attentions=False`.

## 3. Native Multi-Turn Literature & Provenance Workflow

- Each iteration opens one fresh multi-turn `Researcher` handle (`model="inherit"`, discarded at iteration end to prevent quadratic context growth).
- **Turn 1**: Reads the `< 3 KB` compacted epistemic preamble (`track_ops.py state render-epistemic-preamble`) and latest `forensic_delta.md`, queries the track's NotebookLM notebook and rate-limited `tools/s2_folks_cli/fetch_papers.py` (`~/.gemini/s2_api.lock`), and selects `1–3` core arXiv IDs or DOIs plus target equations.
- **Provenance Verification**: A `flash` extractor caches raw arXiv `.tex` method sections or original PDF page excerpts (`<= 25 KB`/paper under `.workflow_state/paper_cache/`) and writes `core_paper_verification.json`. `track_ops.py papers verify` deterministically verifies source SHA-256 and normalized equation/section anchors (`VERIFIED_TEX | VERIFIED_PDF`); any `UNAVAILABLE_NEEDS_WAIVER` without an author-signed waiver pauses in `PAUSED_INFRA`.
- **Turn 2**: Authors `docs/specs/track_<id>/spec_iter_<k>.md` strictly from verified primitives and commits `spec_sha`.

## 4. 3-Tier Failure Taxonomy & Circuit Breakers

1. **Tier 1 — Fixable Mechanical Defects (`IMPLEMENT_AUDIT`)**: Audit/AST/parameter/KV/unit-test/Bank-v2 failures trigger bounded self-repair (`flash` retries followed by one `inherit` escalation) at `0` CU cost. Exhausting retries routes to symmetric rollback and increments `consecutive_no_gain`.
2. **Tier 2 — Infrastructure / Auth / Stall Faults (`PAUSED_INFRA`)**: MCP/NotebookLM/Drive auth errors, `ToolError.denied`, `WAVE_STALLED_MARKER`, missing paper waivers, or S2 `HTTP 429` pause cleanly after confirming L4 teardown with **zero** increment to `iteration` or `consecutive_no_gain`.
3. **Tier 3 — Scientific Non-Improvement (`RATCHET_DECIDE` -> `MINI_BOOST` / `HALTED_FOR_BOOST`)**:
   - Stage A `FAIL` / `SUBMASK_HINT` or Stage B ratchet rejection triggers `track_ops.py git rollback-clean` back to `<best_commit>` over `src/nevfe/` and `tests/` (preserving `docs/specs/`, `reports/`, `.workflow_state/`, and `tests/test_metrics_relaxation.py`), logs a falsified claim into `reports/track_<id>/epistemic_ledger.json`, and increments `consecutive_no_gain`.
   - At `consecutive_no_gain == 1`: runs a single stateless TRIZ contradiction diagnosis (`MINI_BOOST`, `model="inherit"`).
   - At `consecutive_no_gain >= no_gain_halt_threshold`: halts cleanly in `HALTED_FOR_BOOST` and writes `reports/track_<id>/iter_<k>/BOOST_HANDOFF_iter_<k>.md`.

## 5. Manual `/boost` Handshake (`mode="post_boost"`)

- Running `mode="run"` while `fsm_state == "HALTED_FOR_BOOST"` refuses to clear `consecutive_no_gain` and prints the exact `mode="post_boost"` command.
- After running manual `/boost`, record `## Diagnosis` and `## Next Direction` in `reports/track_<id>/iter_<k>/BOOST_RESOLUTION_iter_<k>.md` (or commit manual code/spec guidance on the track branch) and invoke `/run-workflow` with `mode="post_boost"` and `boost_notes_path`.
- `track_ops.py` checks SHA-256 uniqueness against `consumed_boost_digests` (preventing replay), records the directive into `epistemic_ledger.json`, and resets `consecutive_no_gain = 0` once within the same track iteration cap.

## 6. Windows UTF-8 LF & PowerShell 5.1 Hygiene

- All Python file writers must open text files with `encoding="utf-8", newline="\n"` and strip control bytes (`\x00..`) from extracted PDF text; Check #8 verifies UTF-8 decoding and zero `\x00` bytes on all emitted reports.
- Host commands run in PowerShell 5.1 with `CUDA_VISIBLE_DEVICES=""`, `KMP_DUPLICATE_LIB_OK="TRUE"`, and `PYTHONIOENCODING="utf-8"`. Never use `&&` chaining or `>`/`>>`/`Out-File` redirection for text files (which produce UTF-16LE).
- Host Google Drive CLI runs read-only (`gdrive readonly ...`); remote code transport uses per-track disposable remote staging refs (`refs/heads/colab-staging/track-<id>`), pruned automatically upon track `DONE`.
