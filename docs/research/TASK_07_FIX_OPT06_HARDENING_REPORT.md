# TASK 07-FIX: HARDEN OPT-06 — RESOLVE M-01, M-02, M-03

**Date:** 2026-11-08  
**Task ID:** `TASK 07-FIX` (`TICKET-OPT-06` Hardening: Resolve Minor Findings `M-01`, `M-02`, `M-03`)  
**Implementer:** `4argon`  
**Independent QA Reviewer:** `Antigravity (Anti)`  
**Methodology & Architecture Reviewer:** `ChatGPT`  
**Final Approval Authority:** `Project Owner`  
**Status:** **READY FOR ANTI QA REVIEW** *(not yet fully accepted)*

---

## 1. Summary of M-01, M-02, and M-03 Fixes

This task implements the three authorized minor hardening items (`M-01`, `M-02`, `M-03`) identified during the independent read-only QA review in `docs/research/TASK_07R_OPT06_INDEPENDENT_QA.md`:

| Finding ID | Target Symbol & File | Root Cause in Initial `OPT-06` | Hardened Resolution |
| :--- | :--- | :--- | :--- |
| **`M-01`** | `_normalize_relative_image_path()` in `scripts/benchmark/sg_vcd/feature_cache.py` | After computing `rel = os.path.relpath(abs_path, abs_root)` for in-root paths, execution fell through to the `"data"` / `"results"` anchor search and `parts[-3:]` tail truncation. Consequently, two in-root paths deeper than 3 levels with distinct upper directories (e.g., `splitA/a/b/c/22.jpg` vs `splitB/a/b/c/22.jpg`) were both truncated to `b/c/22.jpg` and collided when `image_id` was omitted. | When `dataset_root` is provided and `os.path.relpath` succeeds without escaping `dataset_root` (`rel != ".."` and not starting with `"../"`), `_normalize_relative_image_path()` now returns the normalized relative POSIX path directly at **full depth** without running the `"data"` / `"results"` anchor search or `parts[-3:]` truncation. Furthermore, the `parts[-3:]` truncation in the out-of-root / no-`dataset_root` fallback was removed so deep paths outside `dataset_root` also retain their full normalized relative/non-drive structure. |
| **`M-02`** | `build_image_cache_identity()` in `scripts/benchmark/sg_vcd/feature_cache.py` | When `isinstance(image_key, ImageCacheIdentity)` was true, `build_image_cache_identity()` checked only `(image_key.width, image_key.height) == image.size` and returned `image_key` immediately without validating explicit caller-supplied metadata (`dataset_version`, `dataset_namespace`, `image_id`, `image_path`, `content_fingerprint`). On a dimension mismatch, it silently discarded `image_key` and rebuilt a new identity instead of failing fast. | The $O(1)$ fast-path is preserved when `image_key` is consistent, and now strictly validates `(image_key.width, image_key.height) == (w, h)` plus any explicitly supplied non-`None` `dataset_version`, `dataset_namespace`, `image_id`, `image_path` (normalized with `dataset_root`), and `content_fingerprint` (as well as conflicts between `dataset_version` and `dataset_namespace`). Any contradiction raises a descriptive `ValueError` immediately rather than silently reusing or rebuilding a mismatched identity. |
| **`M-03`** | `_sample_image_group_key()` in `scripts/benchmark/evaluate_vitextvqa.py` | `group_samples_by_image()` grouped by `(sample["image_id"], sample["image_path"] or sample["image_filename"])`, whereas `load_benchmark_dataset()` unique-image logging, `--test`, `--test-images`, and multi-worker sharding (`img_to_idx`) keyed on `sample["image_filename"]` alone. In multi-directory or composite datasets where two distinct images share the same basename (e.g., `dir_a/22.jpg` and `dir_b/22.jpg`), `--test`, `--test-images`, and sharding collapsed them into a single image bucket. | Introduced a unified helper `_sample_image_group_key(sample: Dict[str, Any]) -> Tuple[str, str]` in `scripts/benchmark/evaluate_vitextvqa.py` and applied it consistently across `load_benchmark_dataset()` unique-image counting, `group_samples_by_image()`, `--test`, `--test-images`, and multi-worker sharding (`img_to_idx`). |

---

## 2. Files Modified and Added

### Modified Tracked Files (within `TASK 07-FIX` scope)
1. `scripts/benchmark/sg_vcd/feature_cache.py`
   - Updated `_normalize_relative_image_path()` (lines `84–177`) for `M-01`.
   - Updated `build_image_cache_identity()` (lines `322–385`) for `M-02`.
2. `scripts/benchmark/evaluate_vitextvqa.py`
   - Added `_sample_image_group_key(sample)` (lines `400–410`) and updated `load_benchmark_dataset()` (line `396`), `group_samples_by_image()` (lines `413–423`), `--test` (lines `953–961`), `--test-images` (lines `962–974`), and multi-worker sharding `img_to_idx` (lines `984–993`) for `M-03`.

### Modified Test File (originally created in `TASK 07`, extended in `TASK 07-FIX`)
3. `tests/test_opt06_image_cache_identity.py`
   - Added `test_M01_relative_path_normalization_deep_paths_in_root_and_fallback` (lines `626–676`).
   - Added `test_M02_precomputed_identity_consistency_validation` (lines `678–742`).
   - Added `test_M03_unified_image_grouping_sampling_and_sharding` (lines `744–865`).

### Added Documentation Deliverable
4. `docs/research/TASK_07_FIX_OPT06_HARDENING_REPORT.md` (this report).

---

## 3. Exact Behavior Before vs. After

### 3.1 `M-01` — Relative Path Normalization (`_normalize_relative_image_path`)

| Scenario | Behavior Before (`TASK 07`) | Behavior After (`TASK 07-FIX`) |
| :--- | :--- | :--- |
| **In-root path at depth > 3** (`dataset_root="/workspace/ds"`, `image_path="/workspace/ds/splitA/sub1/sub2/sub3/22.jpg"` vs `"/workspace/ds/splitB/sub1/sub2/sub3/22.jpg"`) | `os.path.relpath` produced `"splitA/sub1/sub2/sub3/22.jpg"` and `"splitB/sub1/sub2/sub3/22.jpg"`, which then fell through to `parts[-3:]` truncation, collapsing both to `"sub2/sub3/22.jpg"`. | `os.path.relpath` succeeds inside `dataset_root` (`not rel_posix.startswith("../")` and `rel_posix != ".."`), returning `"splitA/sub1/sub2/sub3/22.jpg"` and `"splitB/sub1/sub2/sub3/22.jpg"` directly at full depth. Zero collision. |
| **In-root path containing `"data"` or `"results"` directory segment** (`dataset_root="/workspace/ds"`, `image_path="/workspace/ds/tenant_a/data/sub/22.jpg"` vs `"/workspace/ds/tenant_b/data/sub/22.jpg"`) | Fell through to `"data"` anchor search, stripping `"tenant_a"` and `"tenant_b"` so both collapsed to `"data/sub/22.jpg"`. | Returns `"tenant_a/data/sub/22.jpg"` and `"tenant_b/data/sub/22.jpg"` directly relative to `dataset_root`. Zero collision. |
| **Out-of-root / omitted `dataset_root` fallback with deep paths** (`image_path="/mnt/storage/a/b/c/d/22.jpg"` vs `"/mnt/storage/x/b/c/d/22.jpg"`, `dataset_root=None`) | Truncated via `parts[-3:]` to `"c/d/22.jpg"` for both paths. | Preserves full normalized non-drive relative path (`"mnt/storage/a/b/c/d/22.jpg"` vs `"mnt/storage/x/b/c/d/22.jpg"`), or full subpath from `"data"` / `"results"` anchor when present. |
| **Bare basename or Windows/POSIX separators** (`"sub\\dir\\22.jpg"`, `"22.jpg"`) | Normalized backslashes to `"/"` and stripped leading `"./"`. | Unchanged; Windows backslashes and POSIX forward slashes normalize identically across platforms. |

### 3.2 `M-02` — Precomputed Identity Consistency Validation (`build_image_cache_identity`)

| Scenario | Behavior Before (`TASK 07`) | Behavior After (`TASK 07-FIX`) |
| :--- | :--- | :--- |
| **`image_key` is a valid `ImageCacheIdentity` and no conflicting metadata is passed** | Returned `image_key` in $O(1)$ without hashing pixels. | Returned `image_key` in $O(1)$ without hashing pixels. |
| **`image_key` is a valid `ImageCacheIdentity` and matching explicit metadata (`image_id`, `dataset_version`, `image_path`, `content_fingerprint`) is passed** | Returned `image_key` in $O(1)$ without checking metadata fields. | Validates all explicitly passed fields against `image_key` in $O(1)` and returns `image_key`. |
| **`image_key` dimensions `(width, height)` do not match `image.size`** | Silently ignored `image_key` and rebuilt a new `ImageCacheIdentity`. | Raises `ValueError` immediately with message indicating `image_key dimensions (...) != image.size (...)`. |
| **Explicit `image_id`, `dataset_version`, `dataset_namespace`, `image_path`, or `content_fingerprint` contradicts `image_key`** | Silently ignored the contradiction and returned the stale `image_key`. | Raises `ValueError` immediately identifying the exact mismatched attribute (`image_id`, `dataset_version`, `image_path`, or `content_fingerprint`). |
| **Both `dataset_version` and `dataset_namespace` are passed with conflicting values** | `dataset_version` silently overwrote `dataset_namespace`. | Raises `ValueError` on conflicting `dataset_version` vs `dataset_namespace` (both in the precomputed `ImageCacheIdentity` fast-path and when constructing a new `ImageCacheIdentity`). |

### 3.3 `M-03` — Unified Image Grouping Key (`_sample_image_group_key` in `evaluate_vitextvqa.py`)

| Code Path in `scripts/benchmark/evaluate_vitextvqa.py` | Key Used Before (`TASK 07`) | Key Used After (`TASK 07-FIX`) |
| :--- | :--- | :--- |
| `load_benchmark_dataset()` unique image count log | `s["image_filename"]` | `_sample_image_group_key(s)` |
| `group_samples_by_image()` | `(str(s.get("image_id", "")).strip(), str(s.get("image_path") or s.get("image_filename", "")).strip().replace("\\", "/"))` | `_sample_image_group_key(s)` |
| `--test` smoke-test image deduplication | `s["image_filename"]` | `_sample_image_group_key(s)` |
| `--test-images N` unique image subset selection | `s["image_filename"]` | `_sample_image_group_key(s)` |
| Multi-worker sharding `img_to_idx` (`--num-workers`, `--worker-id`) | `s["image_filename"]` | `_sample_image_group_key(s)` |

---

## 4. Tests Added or Modified

In `tests/test_opt06_image_cache_identity.py`:

1. **`test_M01_relative_path_normalization_deep_paths_in_root_and_fallback`**
   - Verifies that two deep in-root paths (`splitA/level1/level2/level3/22.jpg` and `splitB/level1/level2/level3/22.jpg`) under the same `dataset_root` preserve their full relative paths (`"splitA/level1/level2/level3/22.jpg"` vs `"splitB/level1/level2/level3/22.jpg"`) and produce distinct `ImageCacheIdentity.cache_key` values even when `image_id=None`.
   - Verifies that two in-root paths containing `"data"` in an interior segment (`tenant_a/data/sub1/sub2/22.jpg` vs `tenant_b/data/sub1/sub2/22.jpg`) preserve their leading tenant directory relative to `dataset_root` and do not collide.
   - Verifies that out-of-root / `dataset_root=None` deep paths without `"data"`/`"results"` anchors (`archive_a/x/y/z/22.jpg` vs `archive_b/x/y/z/22.jpg`) do not collide via 3-segment tail truncation.
   - Verifies bare basename fallback (`"22.jpg"` -> `"22.jpg"`) and Windows backslash normalization (`splitA\level1\level2\level3\22.jpg` == `splitA/level1/level2/level3/22.jpg`).

2. **`test_M02_precomputed_identity_consistency_validation`**
   - Verifies that passing a consistent precomputed `ImageCacheIdentity` (with `None` optional parameters or matching `image_id`, `dataset_version`, `image_path`, `dataset_root`, `content_fingerprint`) returns the exact same `ImageCacheIdentity` instance in $O(1)$ without triggering `Image.resize`.
   - Verifies that passing an image with mismatched dimensions (`(64, 64)` vs `(48, 32)`) raises `ValueError`.
   - Verifies that passing a contradictory `image_id`, `dataset_version`, `dataset_namespace`, `image_path`, `content_fingerprint`, or conflicting `dataset_version` vs `dataset_namespace` raises `ValueError`.

3. **`test_M03_unified_image_grouping_sampling_and_sharding`**
   - Constructs a synthetic multi-directory dataset of 8 QA pairs across 4 distinct images where 3 images share the same filename `"22.jpg"` in different subdirectories/IDs (`dir_a/22.jpg` [`id=101`], `dir_b/22.jpg` [`id=102`], `dir_c/22.jpg` [`id=103`]) and 1 image has `"30.jpg"` (`dir_d/30.jpg` [`id=104`]), with 2 QA pairs per image.
   - Verifies `group_samples_by_image()` produces 4 distinct image groups of 2 questions each.
   - Verifies `run_benchmark(..., test_images=2, mock=True)` selects the first 2 unique images (`id=101` and `id=102`, 4 questions total) rather than collapsing `id=101`, `102`, and `103` into one `"22.jpg"` bucket.
   - Verifies `run_benchmark(..., test_mode=True, mock=True)` selects 1 question from each of the first 3 distinct images (`QID 1`, `QID 3`, `QID 5`) even though all 3 have `image_filename == "22.jpg"`.
   - Verifies 2-worker sharding (`num_workers=2`, `worker_id=0` and `worker_id=1`) partitions the 4 distinct images evenly (2 images / 4 questions per worker) with zero overlap and full coverage of all 8 questions (`QID 1..8`).

---

## 5. Exact Test Execution Output

All five mandatory test suites from Section 7 of the task specification were executed using the project virtual environment (`..\Document AI\VLM_Reasoning\venv\Scripts\python.exe`).

### Summary Table

| # | Test Suite / Command | Passed | Failed | Skipped | Time | Exit Code |
| :-: | :--- | :-: | :-: | :-: | :-: | :-: |
| 1 | `python -m unittest -v tests.test_opt06_image_cache_identity` | **11** | 0 | 0 | `3.139s` | `0` |
| 2 | `python -m unittest -v tests.test_sg_vcd` | **33** | 0 | 0 | `0.246s` | `0` |
| 3 | `python -m unittest -v tests.test_multi_worker_sharding` | **5** | 0 | 0 | `0.083s` | `0` |
| 4 | `python -m unittest -v tests.test_adversarial_sharding` | **11** | 0 | 0 | `46.039s` | `0` |
| 5 | `python -m unittest -v tests.test_tier4_workloads.TestTier4RealWorldWorkloads.test_05_anti_hardcoding_and_heuristics_audit` | **1** | 0 | 0 | `0.231s` | `0` |
| **Total** | **All 5 Required Verification Suites** | **61** | **0** | **0** | **`49.738s`** | **`0`** |

### Exact Console Output per Suite

#### Suite 1: `tests.test_opt06_image_cache_identity`
```text
Command: & "..\Document AI\VLM_Reasoning\venv\Scripts\python.exe" -m unittest -v tests.test_opt06_image_cache_identity

test_A_same_image_identity_reuses_cache_and_avoids_redundant_thumbnail_resize (tests.test_opt06_image_cache_identity.TestOPT06ImageCacheIdentity.test_A_same_image_identity_reuses_cache_and_avoids_redundant_thumbnail_resize)
Test A — Same image identity: ... ok
test_B_different_image_identities_same_dimensions_and_pixels_do_not_collide (tests.test_opt06_image_cache_identity.TestOPT06ImageCacheIdentity.test_B_different_image_identities_same_dimensions_and_pixels_do_not_collide)
Test B — Different image identities: ... ok
test_C_same_basename_different_directories_do_not_collide (tests.test_opt06_image_cache_identity.TestOPT06ImageCacheIdentity.test_C_same_basename_different_directories_do_not_collide)
Test C — Same basename, different directories: ... ok
test_D_same_image_id_different_dataset_namespaces_do_not_collide (tests.test_opt06_image_cache_identity.TestOPT06ImageCacheIdentity.test_D_same_image_id_different_dataset_namespaces_do_not_collide)
Test D — Same image ID, different dataset namespace: ... ok
test_E_changed_image_content_or_fingerprint_invalidates_cached_state (tests.test_opt06_image_cache_identity.TestOPT06ImageCacheIdentity.test_E_changed_image_content_or_fingerprint_invalidates_cached_state)
Test E — Image changes: ... ok
test_F_missing_stable_identity_falls_back_to_content_hash_with_fallback_flag (tests.test_opt06_image_cache_identity.TestOPT06ImageCacheIdentity.test_F_missing_stable_identity_falls_back_to_content_hash_with_fallback_flag)
Test F — Missing stable identity: ... ok
test_G_backward_compatibility_positional_callers (tests.test_opt06_image_cache_identity.TestOPT06ImageCacheIdentity.test_G_backward_compatibility_positional_callers)
Test G — Backward compatibility: ... ok
test_H_cache_lifecycle_invalidation_and_fresh_run_isolation (tests.test_opt06_image_cache_identity.TestOPT06ImageCacheIdentity.test_H_cache_lifecycle_invalidation_and_fresh_run_isolation)
Test H — Cache lifecycle: ... ok
test_M01_relative_path_normalization_deep_paths_in_root_and_fallback (tests.test_opt06_image_cache_identity.TestOPT06ImageCacheIdentity.test_M01_relative_path_normalization_deep_paths_in_root_and_fallback)
M-01 — Relative path normalization hardening: ... ok
test_M02_precomputed_identity_consistency_validation (tests.test_opt06_image_cache_identity.TestOPT06ImageCacheIdentity.test_M02_precomputed_identity_consistency_validation)
M-02 — Validate precomputed ImageCacheIdentity consistency: ... ok
test_M03_unified_image_grouping_sampling_and_sharding (tests.test_opt06_image_cache_identity.TestOPT06ImageCacheIdentity.test_M03_unified_image_grouping_sampling_and_sharding)
M-03 — Unified image grouping key across group_samples_by_image, --test, --test-images, ... ok

----------------------------------------------------------------------
Ran 11 tests in 3.139s

OK
EXIT_1=0
```

#### Suite 2: `tests.test_sg_vcd`
```text
Command: & "..\Document AI\VLM_Reasoning\venv\Scripts\python.exe" -m unittest -v tests.test_sg_vcd

test_ahra_head_recency_anchoring (tests.test_sg_vcd.TestDynamicContrastiveDecoderCalibrated.test_ahra_head_recency_anchoring) ... ok
test_capc_preserves_high_contrast_diacritic_token (tests.test_sg_vcd.TestDynamicContrastiveDecoderCalibrated.test_capc_preserves_high_contrast_diacritic_token) ... ok
test_essc_cooldown_delay (tests.test_sg_vcd.TestDynamicContrastiveDecoderCalibrated.test_essc_cooldown_delay) ... ok
test_qiacf_intent_adaptive_contrast_factor (tests.test_sg_vcd.TestDynamicContrastiveDecoderCalibrated.test_qiacf_intent_adaptive_contrast_factor) ... ok
test_tesg_entropy_temperature_cooling (tests.test_sg_vcd.TestDynamicContrastiveDecoderCalibrated.test_tesg_entropy_temperature_cooling) ... ok
test_dceb_active_reading_margin (tests.test_sg_vcd.TestDynamicContrastiveEOSBoosting.test_dceb_active_reading_margin) ... ok
test_dceb_respects_apc_plausibility_constraint (tests.test_sg_vcd.TestDynamicContrastiveEOSBoosting.test_dceb_respects_apc_plausibility_constraint) ... ok
test_dceb_step0_protection (tests.test_sg_vcd.TestDynamicContrastiveEOSBoosting.test_dceb_step0_protection) ... ok
test_dceb_supports_list_of_eos_ids (tests.test_sg_vcd.TestDynamicContrastiveEOSBoosting.test_dceb_supports_list_of_eos_ids) ... ok
test_dceb_termination_boost (tests.test_sg_vcd.TestDynamicContrastiveEOSBoosting.test_dceb_termination_boost) ... ok
test_backward_compatibility_crop_result (tests.test_sg_vcd.TestDynamicMultiPatchROICropper.test_backward_compatibility_crop_result) ... ok
test_exact_isotropic_magnification_invariant (tests.test_sg_vcd.TestDynamicMultiPatchROICropper.test_exact_isotropic_magnification_invariant) ... ok
test_grid_selection_moderate_horizontal (tests.test_sg_vcd.TestDynamicMultiPatchROICropper.test_grid_selection_moderate_horizontal) ... ok
test_grid_selection_square (tests.test_sg_vcd.TestDynamicMultiPatchROICropper.test_grid_selection_square) ... ok
test_grid_selection_vertical_banner (tests.test_sg_vcd.TestDynamicMultiPatchROICropper.test_grid_selection_vertical_banner) ... ok
test_grid_selection_wide_banner (tests.test_sg_vcd.TestDynamicMultiPatchROICropper.test_grid_selection_wide_banner) ... ok
test_isotropic_cropper_subclass_equivalence (tests.test_sg_vcd.TestDynamicMultiPatchROICropper.test_isotropic_cropper_subclass_equivalence) ... ok
test_physical_pixel_span_capping (tests.test_sg_vcd.TestDynamicMultiPatchROICropper.test_physical_pixel_span_capping) ... ok
test_crop_multi_patch_sequence_invariants (tests.test_sg_vcd.TestPipelineMockIntegration.test_crop_multi_patch_sequence_invariants) ... ok
test_concise_prompt_format (tests.test_sg_vcd.TestPromptAndPreambleDisentanglement.test_concise_prompt_format) ... ok
test_strip_conversational_intro_clauses (tests.test_sg_vcd.TestPromptAndPreambleDisentanglement.test_strip_conversational_intro_clauses) ... ok
test_strip_cum_tu_and_tra_loi (tests.test_sg_vcd.TestPromptAndPreambleDisentanglement.test_strip_cum_tu_and_tra_loi) ... ok
test_strip_prompt_echo_with_quotes (tests.test_sg_vcd.TestPromptAndPreambleDisentanglement.test_strip_prompt_echo_with_quotes) ... ok
test_strip_prompt_echo_yeu_cau (tests.test_sg_vcd.TestPromptAndPreambleDisentanglement.test_strip_prompt_echo_yeu_cau) ... ok
test_strip_unicode_curved_quotes (tests.test_sg_vcd.TestPromptAndPreambleDisentanglement.test_strip_unicode_curved_quotes) ... ok
test_intra_word_apostrophe_preservation (tests.test_sg_vcd.TestSyntacticSpanRefiner.test_intra_word_apostrophe_preservation) ... ok
test_multiline_concluding_payload (tests.test_sg_vcd.TestSyntacticSpanRefiner.test_multiline_concluding_payload) ... ok
test_qamif_non_collapsing_invariant (tests.test_sg_vcd.TestSyntacticSpanRefiner.test_qamif_non_collapsing_invariant) ... ok
test_qamif_question_echo_peeling (tests.test_sg_vcd.TestSyntacticSpanRefiner.test_qamif_question_echo_peeling) ... ok
test_quotation_and_terminal_punctuation_sanitizer (tests.test_sg_vcd.TestSyntacticSpanRefiner.test_quotation_and_terminal_punctuation_sanitizer) ... ok
test_scpdo_colonic_delimiter (tests.test_sg_vcd.TestSyntacticSpanRefiner.test_scpdo_colonic_delimiter) ... ok
test_scpdo_numeric_time_and_ratio_preservation (tests.test_sg_vcd.TestSyntacticSpanRefiner.test_scpdo_numeric_time_and_ratio_preservation) ... ok
test_spaced_dash_preservation (tests.test_sg_vcd.TestSyntacticSpanRefiner.test_spaced_dash_preservation) ... ok

----------------------------------------------------------------------
Ran 33 tests in 0.246s

OK
EXIT_2=0
```

#### Suite 3: `tests.test_multi_worker_sharding`
```text
Command: & "..\Document AI\VLM_Reasoning\venv\Scripts\python.exe" -m unittest -v tests.test_multi_worker_sharding

test_01_deterministic_sharding_partitioning (tests.test_multi_worker_sharding.TestMultiWorkerShardingEngine.test_01_deterministic_sharding_partitioning) ... ok
test_02_per_shard_checkpoint_manager_isolation (tests.test_multi_worker_sharding.TestMultiWorkerShardingEngine.test_02_per_shard_checkpoint_manager_isolation) ... ok
test_03_atomic_aggregation_merges_and_sorts_all_shards (tests.test_multi_worker_sharding.TestMultiWorkerShardingEngine.test_03_atomic_aggregation_merges_and_sorts_all_shards) ... ok
test_01_stage1_diacritic_vertical_padding (tests.test_multi_worker_sharding.TestStage1AndStage2Upgrades.test_01_stage1_diacritic_vertical_padding) ... ok
test_02_stage2_conversational_preamble_stripping (tests.test_multi_worker_sharding.TestStage1AndStage2Upgrades.test_02_stage2_conversational_preamble_stripping) ... ok

----------------------------------------------------------------------
Ran 5 tests in 0.083s

OK
EXIT_3=0
```

#### Suite 4: `tests.test_adversarial_sharding`
```text
Command: & "..\Document AI\VLM_Reasoning\venv\Scripts\python.exe" -m unittest -v tests.test_adversarial_sharding

test_adv_10_deterministic_equivalence_1w_vs_2w_vs_4w (tests.test_adversarial_sharding.TestAdversarialDeterministicEquivalenceCLI.test_adv_10_deterministic_equivalence_1w_vs_2w_vs_4w) ... ok
test_adv_11_prime_number_shards_uneven_partition (tests.test_adversarial_sharding.TestAdversarialDeterministicEquivalenceCLI.test_adv_11_prime_number_shards_uneven_partition) ... ok
test_adv_01_all_shards_missing (tests.test_adversarial_sharding.TestAdversarialEmptyAndMissingShards.test_adv_01_all_shards_missing) ... ok
test_adv_02_sparse_missing_shards (tests.test_adversarial_sharding.TestAdversarialEmptyAndMissingShards.test_adv_02_sparse_missing_shards) ... ok
test_adv_03_zero_byte_and_whitespace_shards (tests.test_adversarial_sharding.TestAdversarialEmptyAndMissingShards.test_adv_03_zero_byte_and_whitespace_shards) ... ok
test_adv_04_corrupted_and_truncated_jsonl (tests.test_adversarial_sharding.TestAdversarialEmptyAndMissingShards.test_adv_04_corrupted_and_truncated_jsonl) ... ok
test_adv_05_fallback_to_checkpoint_json_when_jsonl_missing (tests.test_adversarial_sharding.TestAdversarialEmptyAndMissingShards.test_adv_05_fallback_to_checkpoint_json_when_jsonl_missing) ... ok
test_adv_06_reverse_order_completion (tests.test_adversarial_sharding.TestAdversarialOutOfOrderAndDeduplication.test_adv_06_reverse_order_completion) ... ok
test_adv_07_shuffled_intra_shard_ordering (tests.test_adversarial_sharding.TestAdversarialOutOfOrderAndDeduplication.test_adv_07_shuffled_intra_shard_ordering) ... ok
test_adv_08_cross_shard_duplicate_keys (tests.test_adversarial_sharding.TestAdversarialOutOfOrderAndDeduplication.test_adv_08_cross_shard_duplicate_keys) ... ok
test_adv_09_robustness_when_global_index_missing_or_none (tests.test_adversarial_sharding.TestAdversarialOutOfOrderAndDeduplication.test_adv_09_robustness_when_global_index_missing_or_none) ... ok

----------------------------------------------------------------------
Ran 11 tests in 46.039s

OK
EXIT_4=0
```

#### Suite 5: `tests.test_tier4_workloads.TestTier4RealWorldWorkloads.test_05_anti_hardcoding_and_heuristics_audit`
```text
Command: & "..\Document AI\VLM_Reasoning\venv\Scripts\python.exe" -m unittest -v tests.test_tier4_workloads.TestTier4RealWorldWorkloads.test_05_anti_hardcoding_and_heuristics_audit

test_05_anti_hardcoding_and_heuristics_audit (tests.test_tier4_workloads.TestTier4RealWorldWorkloads.test_05_anti_hardcoding_and_heuristics_audit)
Anti-cheating AST audit: ... ok

----------------------------------------------------------------------
Ran 1 test in 0.231s

OK
EXIT_5=0
```

---

## 6. `git diff --stat`

```text
 scripts/benchmark/evaluate_vitextvqa.py         | 119 ++++--
 scripts/benchmark/sg_vcd/__init__.py            |   3 +
 scripts/benchmark/sg_vcd/attention_extractor.py |  86 ++++-
 scripts/benchmark/sg_vcd/feature_cache.py       | 458 ++++++++++++++++++++++--
 scripts/benchmark/sg_vcd/pipeline.py            |  82 ++++-
 5 files changed, 683 insertions(+), 65 deletions(-)
```
*(Note: The cumulative `git diff --stat` against `HEAD` includes the base `TICKET-OPT-06` implementation from `TASK 07` plus the `M-01`, `M-02`, and `M-03` hardening edits in `scripts/benchmark/sg_vcd/feature_cache.py` and `scripts/benchmark/evaluate_vitextvqa.py`.)*

---

## 7. `git status --short`

```text
 M scripts/benchmark/evaluate_vitextvqa.py
 M scripts/benchmark/sg_vcd/__init__.py
 M scripts/benchmark/sg_vcd/attention_extractor.py
 M scripts/benchmark/sg_vcd/feature_cache.py
 M scripts/benchmark/sg_vcd/pipeline.py
?? .github/
?? docs/SG_VCD_Research_Review_Consolidated_Draft.md
?? docs/research/
?? tests/test_opt06_image_cache_identity.py
```

---

## 8. Confirmation That No Unrelated Optimization Was Implemented

- **Confirmed:** Only `M-01`, `M-02`, and `M-03` within `TICKET-OPT-06` were implemented.
- **Confirmed:** `OPT-03` (Stage 1 Attention Extraction Passes), `OPT-02` (Batched/Single-Pass Positive + Negative Decoding), `OPT-01` (Model-Call Audit & Per-Sample Instrumentation), and `OPT-04` (Multi-Tile Vision Encoding Reuse) have **NOT** been implemented.
- **Confirmed:** No changes were made to SG-VCD Stage 1 spatial localization formulas, Stage 2 contrastive decoding equations, prompts, dataset annotations, or metric calculations.
- **Confirmed:** No real Vintern model weights were loaded, no real-model inference was executed, no datasets/models were downloaded, and no git commits or pushes were performed.

---

## 9. Unresolved Limitations

- **None within `TICKET-OPT-06` (`M-01`, `M-02`, `M-03`).** All three minor findings from `docs/research/TASK_07R_OPT06_INDEPENDENT_QA.md` are resolved and covered by automated unit and mock-integration tests.
- Real-model execution on the 20-QA small integration sample (`--test-images` / 20-QA manifest on Google Colab) remains gated behind independent QA verification and subsequent `P0` remediation / optimization tickets as specified in `docs/research/PERFORMANCE_OPTIMIZATION_HANDOFF_v1.md`.

---

## 10. QA Readiness Statement

This implementation of `TASK 07-FIX` (`M-01`, `M-02`, `M-03` hardening for `TICKET-OPT-06`) is **READY FOR ANTI QA REVIEW**, and is **not yet fully accepted** until verified by `Antigravity (Anti)`, reviewed by `ChatGPT`, and approved by the `Project Owner`.

