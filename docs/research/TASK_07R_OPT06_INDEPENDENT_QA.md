# TASK 07-R: INDEPENDENT READ-ONLY QA REPORT — TICKET-OPT-06 (STABLE IMAGE IDENTITY AND CACHE KEY)

**Document ID:** `TASK_07R_OPT06_INDEPENDENT_QA`  
**Date:** 2026-11-07  
**QA Reviewer:** Antigravity (Anti) — Independent Technical QA  
**Implementer Reviewed:** 4argon  
**Methodology & Architecture Review Authority:** ChatGPT  
**Final Approval Authority:** Project Owner  
**Governance Applied:**
- [`.github/copilot-instructions.md`](../../.github/copilot-instructions.md)
- [`.github/skills/anti-output-quality-review/SKILL.md`](../../.github/skills/anti-output-quality-review/SKILL.md)
- [`.github/skills/research-plan-compliance-review/SKILL.md`](../../.github/skills/research-plan-compliance-review/SKILL.md)
- [`.github/skills/local-colab-experiment-workflow/SKILL.md`](../../.github/skills/local-colab-experiment-workflow/SKILL.md)

---

## 1. Executive Verdict

**Verdict: PASS WITH CONDITIONS (Conditionally Ready for ChatGPT & Project Owner Review)**

### Summary Justification
1. **Core OPT-06 Contract Met:** Independent inspection and synthetic/static probes confirm that [`ImageCacheIdentity`](../../scripts/benchmark/sg_vcd/feature_cache.py#L33-L81) and [`build_image_cache_identity()`](../../scripts/benchmark/sg_vcd/feature_cache.py#L213-L290) construct a deterministic 5-component identity `(dataset_version, image_identity, width, height, content_fingerprint)` that eliminates per-question `(16 x 16)` thumbnail resizing in Stable Identity Mode (`thumb16_resizes = 0` across multi-question runs) and prevents redundant fingerprint recomputation when `image_key` is pre-built once per image in [`run_benchmark()`](../../scripts/benchmark/evaluate_vitextvqa.py#L1084-L1158) (`fingerprint_calls = 0` across 3 questions on the same image).
2. **Real 400-Image Dataset Audit Passed (400/400 Unique Identities):** Static metadata and local file audit on [`results/swintextspotter_auto_annot/answer_bboxes_400_verified.json`](../../results/swintextspotter_auto_annot/answer_bboxes_400_verified.json) (400 images, 1,222 QA; all 400 image files verified present on disk) confirmed **400 distinct `image_id` values**, **400 distinct normalized paths**, **400 distinct composite cache keys**, **0 dimension mismatches**, and **100% identical QA-to-image partition** compared to the legacy grouping.
3. **Zero Critical or Major Defects; 3 Minor & 2 Informational Findings:** No defect invalidates behavior on the current ViTextVQA 400-image dataset or violates research/algorithmic invariants. However, three **Minor** edge-case weaknesses (path depth truncation to `parts[-3:]` when `"data"`/`"results"` anchors are absent, trusting a pre-built `ImageCacheIdentity` solely by `(width, height)`, and basename-keyed `--test-images` / multi-worker sharding in `evaluate_vitextvqa.py`) are documented with minimal recommended fixes.
4. **Decision Gate Reminder:** Per project governance, this independent technical QA report does **not** constitute final acceptance; `OPT-06` remains subject to **ChatGPT** methodological review and **Project Owner** approval, and local unit tests/static audits do not substitute for authorized Colab Gate B (20-QA) runtime verification.

---

## 2. Findings Classification (Critical / Major / Minor / Informational)

| ID | Severity | Category | Location | Summary |
|---|---|---|---|---|
| **M-01** | **Minor** | R1 — Path Normalization | [`scripts/benchmark/sg_vcd/feature_cache.py#L84-L132`](../../scripts/benchmark/sg_vcd/feature_cache.py#L84-L132) | `_normalize_relative_image_path()` truncates paths without `"data"`/`"results"` segments to `parts[-3:]` even after `os.path.relpath(..., dataset_root)` succeeds, and matches the *first* `"data"`/`"results"` segment via `parts.index(anchor)`. |
| **M-02** | **Minor** | R1 / R3 — Identity Fast-Path | [`scripts/benchmark/sg_vcd/feature_cache.py#L244-L246`](../../scripts/benchmark/sg_vcd/feature_cache.py#L244-L246) | `build_image_cache_identity()` returns a precomputed `ImageCacheIdentity` whenever `(image_key.width, image_key.height) == image.size`, without checking whether caller-supplied `image_id` or `dataset_version` conflicts with `image_key`. |
| **M-03** | **Minor** | R3 / R7 — Runner Grouping | [`scripts/benchmark/evaluate_vitextvqa.py#L946-L986`](../../scripts/benchmark/evaluate_vitextvqa.py#L946-L986) | While `group_samples_by_image()` uses the new composite key, `--test`, `--test-images`, and multi-worker sharding (`img_to_idx`) still bucket by bare `s["image_name"]`. Safe on the 400-image dataset (400 unique `image_name`s), but inconsistent for multi-directory datasets with duplicate basenames. |
| **I-01** | **Informational** | R5 — Type Contract | [`scripts/benchmark/sg_vcd/feature_cache.py#L216-L269`](../../scripts/benchmark/sg_vcd/feature_cache.py#L216-L269) | `image_key` type hint allows `Tuple[Any, ...]`, but `build_image_cache_identity()` only handles `ImageCacheIdentity` and `str`; a raw `tuple` without `image_id`/`image_path` silently falls through to `FALLBACK_IMAGE_IDENTITY` (`is_fallback=True`). |
| **I-02** | **Informational** | R2 — Fingerprint Semantics | [`scripts/benchmark/sg_vcd/feature_cache.py#L138-L210`](../../scripts/benchmark/sg_vcd/feature_cache.py#L138-L210) | Neither Stable Mode (`os.stat` + 9-point probe) nor Fallback Mode (`16x16` `NEAREST` thumbnail + 9-point probe) is a cryptographic full-image digest; small interior pixel edits off the probe/thumbnail sampling lattice are not detected unless file metadata or `content_fingerprint` changes. |

---

## 3. Detailed Review Evidence (Checklist R1 – R5)

### R1 — Identity Correctness
- **Inspected Functions:** [`ImageCacheIdentity`](../../scripts/benchmark/sg_vcd/feature_cache.py#L33-L81), [`_normalize_relative_image_path`](../../scripts/benchmark/sg_vcd/feature_cache.py#L84-L132), and [`build_image_cache_identity`](../../scripts/benchmark/sg_vcd/feature_cache.py#L213-L290).
- **Verified Properties:**
  1. **Dataset namespace/version:** Resolved at [`feature_cache.py#L248-L250`](../../scripts/benchmark/sg_vcd/feature_cache.py#L248-L250) from `dataset_version` or alias `dataset_namespace`, defaulting to `UNSCOPED_DATASET_NAMESPACE` (`"__unscoped_dataset__"`).
  2. **Image ID and relative path:** Normalized deterministically at [`feature_cache.py#L252-L269`](../../scripts/benchmark/sg_vcd/feature_cache.py#L252-L269). Both `int(22)` and `str("22")` normalize to `"id=22"`. When both `image_id` and `image_path` are supplied, both are embedded (`"id=22|path=data/vitextvqa/st_images/22.jpg"`).
  3. **Same basename in different directories:** Produces distinct identities (`"path=data/A/imgs/22.jpg"` vs `"path=data/B/imgs/22.jpg"`).
  4. **Same image ID in different dataset namespaces:** Produces distinct keys (`ds=vitextvqa_20qa_gate_b|...` vs `ds=vitextvqa_100img_gate_c|...`).
  5. **Width and height inclusion:** Stored on `ImageCacheIdentity.width` and `.height` and serialized into `to_cache_key()` (`dim={width}x{height}`).
  6. **Explicit manifest fingerprints:** Handled at [`feature_cache.py#L180-L181`](../../scripts/benchmark/sg_vcd/feature_cache.py#L180-L181) (`return f"explicit:{str(content_fingerprint).strip()}"`), bypassing all file stat and pixel probing.
  7. **Missing stable identity & bare basename rejection:** At [`feature_cache.py#L252-L269`](../../scripts/benchmark/sg_vcd/feature_cache.py#L252-L269), `has_stable_id = has_explicit_id or has_dir_scope`. Independent probe confirmed:
     - `build_image_cache_identity(img, image_path="22.jpg").is_fallback == True` (bare basename is **not** trusted as a globally unique identity).
     - `build_image_cache_identity(img, image_id=0).is_fallback == False` (`0` is properly treated as a valid integer ID via `None is not raw_id and bool(str(raw_id).strip())`).
     - `build_image_cache_identity(img, image_id="").is_fallback == True`.
- **Relative-Path Normalization Heuristic Analysis (Finding M-01):**
  - In [`_normalize_relative_image_path()`](../../scripts/benchmark/sg_vcd/feature_cache.py#L99-L126), even after `os.path.relpath(raw, dataset_root)` computes a clean relative path, lines 117–125 still run the anchor loop (`for anchor in ("data", "results"): ... else: if len(parts) > 3: parts = parts[-3:]`).
  - Independent probe results:
    - `N("/mnt/ds1/a/b/c/22.jpg", None)` vs `N("/mnt/ds2/a/b/c/22.jpg", None)` $\rightarrow$ both return `'b/c/22.jpg'` (**COLLIDE** on path component).
    - `N("root/x/p/q/r/22.jpg", dataset_root="root")` vs `N("root/y/p/q/r/22.jpg", dataset_root="root")` $\rightarrow$ both return `'q/r/22.jpg'` (**COLLIDE** on path component because neither contains `"data"` or `"results"` and depth $>3$).
    - `N("D:\\proj\\split1\\imgs\\22.jpg", None)` vs `N("/proj/split1/imgs/22.jpg", None)` $\rightarrow$ both return `'split1/imgs/22.jpg'` (portable across Windows/POSIX).

---

### R2 — Fingerprint Limitations (Independent Verification Matrix)
[`compute_image_content_fingerprint()`](../../scripts/benchmark/sg_vcd/feature_cache.py#L163-L210) is **not** a cryptographic full-image digest. In Stable Identity Mode (`use_thumbnail_bytes=False`), it hashes `f"{w}x{h}|{mode}|{stat_part}|{nine_pt}"` where `stat_part` is `stat:{st_size}:{st_mtime_ns}` (when `image_path` is an existing file) and `nine_pt` samples 9 coordinates `{(0,0), (w//2,0), (w-1,0), (0,h//2), (w//2,h//2), (w-1,h//2), (0,h-1), (w//2,h-1), (w-1,h-1)}`.

Independent verification across all required test cases:

| Case | Scenario | Detected in Stable Mode (`is_fallback=False`)? | Detected in Fallback Mode (`is_fallback=True`)? | Mechanism / Evidence |
|---|---|---|---|---|
| **1** | Same image, unchanged metadata & content | **Yes (Cache Hit)** | **Yes (Cache Hit)** | Produces identical `probe:3e18a3c4cd8dac88e72871e8` key. |
| **2** | Different content with changed file size (`st_size` $636 \rightarrow 37787$) | **Yes (Invalidated)** | **Yes (Invalidated)** | `stat:{st_size}:{st_mtime_ns}` changes in SHA-256 input. |
| **3** | Different content with changed modification time (`st_mtime_ns` changed, same BMP file size, interior pixel `(60,100)` altered, 9 probes unchanged) | **Yes (Invalidated)** | **Yes (Invalidated)** | `st_mtime_ns` changes in `stat_part` even when `st_size` and all 9 probe pixels are unchanged. |
| **4** | In-memory mutation outside all 9 probe points (`img.putpixel((33, 77), (1, 2, 3))` on $300\times 200$ image; same path/stat) | **No (Not Detected)** | **No if off $16\times 16$ `NEAREST` lattice; Yes if on lattice** | Stable mode probes only the 9 coordinates (`detected: False`). Fallback mode uses `image.resize((16, 16), NEAREST)`, which samples a $16\times 16$ point grid (stride $20\text{ px}$ on a $320\times 320$ image) and also misses a $3\times 3$ block at `(1..3, 1..3)` (`detected: False`). |
| **5** | On-disk content mutation preserving exact `st_size` and restored `st_mtime_ns` (`os.utime`), with 9 probe pixels unchanged | **No (Not Detected)** | **No if off $16\times 16$ `NEAREST` lattice** | Both `stat_part` and `nine_pt` remain identical (`detected: False`). |

**Assessment of Implementer's Documentation:** Section 9 of 4argon's report accurately disclosed Case 4 for Stable Identity Mode without claiming cryptographic collision resistance. It slightly overstated Fallback Mode by implying `16x16` thumbnail hashing catches arbitrary interior pixel mutations, whereas `Image.Resampling.NEAREST` only subsamples 256 point coordinates rather than box-averaging all pixels.

---

### R3 — Identity Propagation and Repeated Work
Tracing [`run_benchmark()`](../../scripts/benchmark/evaluate_vitextvqa.py#L1066-L1159) $\rightarrow$ [`SGVCDPipeline`](../../scripts/benchmark/sg_vcd/pipeline.py#L371-L826) $\rightarrow$ [`VisualFeatureCache`](../../scripts/benchmark/sg_vcd/feature_cache.py#L308-L425) $\rightarrow$ [`PointwiseKLAttentionExtractor`](../../scripts/benchmark/sg_vcd/attention_extractor.py#L320-L775):

1. **Per-Image Outer Loop in `run_benchmark()` ([`evaluate_vitextvqa.py#L1066-L1104`](../../scripts/benchmark/evaluate_vitextvqa.py#L1066-L1104)):**
   - Loads `image = Image.open(img_path).convert("RGB")` once per image group.
   - Calls `image_cache_key = build_image_cache_identity(...)` **once per image** (line 1084).
   - Calls `cached_visual_features = pipeline.extract_visual_features(image, image_key=image_cache_key)` **once per image** (line 1099). Inside `extract_visual_features` $\rightarrow$ `feature_cache.get_or_compute` $\rightarrow$ `compute_image_key` $\rightarrow$ `build_image_cache_identity`, the `isinstance(image_key, ImageCacheIdentity)` fast-path at [`feature_cache.py#L244-L246`](../../scripts/benchmark/sg_vcd/feature_cache.py#L244-L246) returns `image_cache_key` in $O(1)$ without re-running `compute_image_content_fingerprint`.
2. **Per-Question Inner Loop in `run_benchmark()` ([`evaluate_vitextvqa.py#L1106-L1159`](../../scripts/benchmark/evaluate_vitextvqa.py#L1106-L1159)):**
   - Passes `cached_visual_features=cached_visual_features` and `image_key=image_cache_key` into `pipeline.run(image, question, **run_kwargs)` (lines 1154–1159).
   - Inside `pipeline.run()` ([`pipeline.py#L799-L824`](../../scripts/benchmark/sg_vcd/pipeline.py#L799-L824)):
     - `resolved_image_key = build_image_cache_identity(..., image_key=image_key)` hits the `isinstance(image_key, ImageCacheIdentity)` fast-path (0 fingerprint calls).
     - Skips `extract_visual_features` because `cached_visual_features is not None`.
     - Calls `self.extractor.extract(image, question, image_key=resolved_image_key)`, which calls `_make_image_key(..., image_key=image_key)` $\rightarrow$ hits the same fast-path (0 fingerprint calls) and matches `self._cached_img_key == img_key` on questions $2 \dots Q$.
3. **Measured Call Counts Across 3 Questions on 1 Image (Independent Probe):**
   - **Scenario A (`run_benchmark` path, `image_key` built once per image outside question loop):**
     - `compute_image_content_fingerprint` calls inside loop: **`0`** (1 call total when building `image_cache_key` once for the image).
     - `(16 x 16)` thumbnail `Image.resize` calls: **`0`**.
     - Extractor ViT + LLM prefix prefill calls across 3 questions: **`1`** (questions 2 and 3 reuse `_cached_img_state`).
   - **Scenario B (Legacy positional caller `pipeline.run(im, q)` with no identity arguments and no pre-extracted `cached_visual_features`):**
     - On each `pipeline.run(im, q)` call, `build_image_cache_identity(im)` runs once at the top of `pipeline.run` (computing 1 fallback fingerprint and 1 `16x16` thumbnail resize per question) and then passes `resolved_image_key` to both `extract_visual_features` and `extractor.extract`, which both hit the `isinstance(image_key, ImageCacheIdentity)` fast-path. Thus, across 3 questions, `compute_image_content_fingerprint` is called `3` times (once per `run()`, down from twice per `run()` if `extract_visual_features` and `extractor.extract` computed keys independently).

---

### R4 — Cache Lifecycle and Isolation
- [`PointwiseKLAttentionExtractor.clear_cache()`](../../scripts/benchmark/sg_vcd/attention_extractor.py#L314-L317) and cache-miss branch ([`attention_extractor.py#L708-L709`](../../scripts/benchmark/sg_vcd/attention_extractor.py#L708-L709)) reset `_cached_img_state = None` and `_cached_img_key = None`.
- [`SGVCDPipeline.__init__()`](../../scripts/benchmark/sg_vcd/pipeline.py#L353-L355) constructs `VisualFeatureCache(max_entries=16)` per pipeline instance when `feature_cache is None`, while preserving shared cache behavior when `feature_cache` is explicitly injected.
- [`SGVCDPipeline.reset_cache()`](../../scripts/benchmark/sg_vcd/pipeline.py#L357-L369) clears both `self.feature_cache` and `self.extractor`.
- [`run_benchmark()`](../../scripts/benchmark/evaluate_vitextvqa.py#L1039-L1051) calls `default_feature_cache.clear()` and `pipeline.reset_cache()` at run initialization.
- Multi-worker sharding in [`evaluate_vitextvqa.py#L1200-L1245`](../../scripts/benchmark/evaluate_vitextvqa.py#L1200-L1245) spawns separate OS subprocesses per shard (`subprocess.Popen`), so each worker has an isolated address space and cache lifecycle.

---

### R5 — Backward Compatibility
- All new parameters on [`VisualFeatureCache.compute_image_key`](../../scripts/benchmark/sg_vcd/feature_cache.py#L308-L320), [`VisualFeatureCache.get_or_compute`](../../scripts/benchmark/sg_vcd/feature_cache.py#L355-L369), [`get_or_compute_visual_features`](../../scripts/benchmark/sg_vcd/feature_cache.py#L441-L456), [`PointwiseKLAttentionExtractor._make_image_key`](../../scripts/benchmark/sg_vcd/attention_extractor.py#L320-L330), [`PointwiseKLAttentionExtractor.extract`](../../scripts/benchmark/sg_vcd/attention_extractor.py#L665-L676), [`SGVCDPipeline.extract_visual_features`](../../scripts/benchmark/sg_vcd/pipeline.py#L371-L382), and [`SGVCDPipeline.run`](../../scripts/benchmark/sg_vcd/pipeline.py#L763-L778) are optional keyword-only arguments with `None` defaults.
- Other callers in the repository (e.g., [`scripts/benchmark/run_colab_l4_all_models.py#L743`](../../scripts/benchmark/run_colab_l4_all_models.py#L743) calling `self.sg_vcd_pipeline.extract_visual_features(image)` positionally) remain 100% compatible.
- In [`SGVCDPipeline.run()`](../../scripts/benchmark/sg_vcd/pipeline.py#L816-L824), signature inspection (`inspect.signature(self.extractor.extract).parameters`) checks for `"image_key"` or `VAR_KEYWORD` (`**kwargs`) *before* calling `self.extractor.extract`, rather than wrapping the call in a broad `try ... except TypeError:` that could mask internal `TypeError` bugs inside `extract()`.
- [`ImageCacheIdentity`](../../scripts/benchmark/sg_vcd/feature_cache.py#L63-L70) implements `__iter__`, `__len__`, and `__getitem__`, allowing tuple unpacking and indexing by legacy callers of `_make_image_key(image)`.

---

## 4. Test Execution Evidence and Exact Failures (Checklist R6)

### 4.1 Audit of `tests/test_opt06_image_cache_identity.py` (Tests A–H)
Independent inspection of [`tests/test_opt06_image_cache_identity.py`](../../tests/test_opt06_image_cache_identity.py) confirms that all 8 tests exercise the real `VisualFeatureCache`, `PointwiseKLAttentionExtractor`, `SGVCDPipeline`, and `group_samples_by_image` classes against a stateful synthetic `_DummyVinternModel` whose `extract_feature()` scales output tensors by `float(self.extract_feature_calls)` ([`test_opt06_image_cache_identity.py#L143-L153`](../../tests/test_opt06_image_cache_identity.py#L143-L153)). Consequently, any unintended cache miss produces a numerically different tensor (`torch.equal` fails) and increments `extract_feature_calls` / `prefill_calls`.
- **Minor test-coverage note on Test E:** In [`test_E_changed_image_content_or_fingerprint_invalidates_cached_state`](../../tests/test_opt06_image_cache_identity.py#L396-L448), sub-case 1 changes uniform image color `(10, 20, 30)` $\rightarrow$ `(200, 210, 220)` (which changes all 9 probe points), sub-case 2 changes explicit `content_fingerprint`, and sub-case 3 appends 64 bytes to the on-disk file (changing `st_size` and `st_mtime_ns`). Test E does not include a negative assertion demonstrating that an interior-only in-memory edit outside the 9 probe points is *not* detected in Stable Identity Mode (verified separately by our R2 probe above).

### 4.2 Summary of Executed Test Suites
| Suite / Command | Total | Passed | Failed | Skipped | Status |
|---|---:|---:|---:|---:|---|
| `tests.test_opt06_image_cache_identity` | 8 | 8 | 0 | 0 | **PASS** (`1.432s`) |
| `tests.test_sg_vcd` | 33 | 33 | 0 | 0 | **PASS** (`0.321s`) |
| `tests.test_tier2_boundaries` | 25 | 25 | 0 | 0 | **PASS** |
| `tests.test_tier3_combinations` | 13 | 13 | 0 | 0 | **PASS** |
| `tests.test_tier4_workloads` (incl. AST audit) | 5 | 4 | 0 | 1 | **PASS** (1 GPU skip) |
| `tests.test_multi_worker_sharding` | 5 | 5 | 0 | 0 | **PASS** |
| `tests.test_adversarial_sharding` (CLI `--mock`) | 11 | 11 | 0 | 0 | **PASS** |
| `tests.test_sg_vcd_adversarial` | 15 | 14 | 1 | 0 | 1 pre-existing fail |
| `tests/run_e2e_tests.py` (`test_tier1_features`) | — | — | 1 (import) | — | Pre-existing import error |

### 4.3 Forensic Verification of Every Failed or Skipped Test
1. **Skipped Test:** `tests.test_tier4_workloads.TestTier4RealWorldWorkloads.test_02_rtx_3050_vram_budget_ceiling`
   - **Exact Skip Reason:** `skipped 'CUDA is not available on this environment; skipping GPU VRAM test.'`
   - **Causality Assessment:** The local virtual environment (`..\Document AI\VLM_Reasoning\venv`) has a CPU-only build of PyTorch (`torch.cuda.is_available() == False`), and local real-model GPU inference is prohibited by [`.github/skills/local-colab-experiment-workflow/SKILL.md`](../../.github/skills/local-colab-experiment-workflow/SKILL.md). Unrelated to `OPT-06`.
2. **Failed Import (`tests/run_e2e_tests.py` $\rightarrow$ `tests/test_tier1_features.py`):**
   - **Exact Error:** `ImportError: cannot import name 'canonical_normalize_vietnamese_vqa' from 'scripts.benchmark.metrics'` at [`tests/test_tier1_features.py#L39`](../../tests/test_tier1_features.py#L39).
   - **Independent Git Proof:** Running `git show HEAD:scripts/benchmark/metrics.py` and `git show HEAD:tests/test_tier1_features.py` proved that at `HEAD` (`574f36d`), `canonical_normalize_vietnamese_vqa` is **already absent** from `scripts/benchmark/metrics.py` (`False`) while still imported in `tests/test_tier1_features.py` (`True`). Furthermore, `scripts/benchmark/metrics.py` has zero diff in `git status` and is out of scope for `OPT-06` (governed by P0 remediation `R-02`).
3. **Failed Test:** `tests.test_sg_vcd_adversarial.TestSyntacticSpanRefinerAdversarial.test_spaced_dash_brand_in_question`
   - **Exact Failure:** Assertion failure in `SyntacticSpanRefiner` when testing spaced-dash brand segmentation.
   - **Independent Git Proof:** Comparing `SyntacticSpanRefiner` in `HEAD:scripts/benchmark/sg_vcd/pipeline.py` against the working tree (`s_head == s_wt`) returned **`True` (byte-for-byte identical to `HEAD`)**. `SyntacticSpanRefiner` is a pure text post-processor that does not interact with `feature_cache`, `attention_extractor`, or `ImageCacheIdentity`. Demonstrably pre-existing and unrelated to `OPT-06`.

---

## 5. Dataset-Level Identity Audit (Checklist R7)

A static, read-only metadata and local-file audit was executed on [`results/swintextspotter_auto_annot/answer_bboxes_400_verified.json`](../../results/swintextspotter_auto_annot/answer_bboxes_400_verified.json) (SHA-256 prefix: `1428e33d493c0f00`):

- **Image & Question Counts:**
  - `len(images)` = **400** image records (`metadata.num_images = 400`).
  - Total questions across all 400 images = **1,222** (`metadata.total_quests = 1222`, **1,222** distinct `question_id`s).
  - Images with missing or zero `width`/`height` in manifest = **0**.
- **Uniqueness of Identifiers in Manifest:**
  - Distinct `image_id` values = **400 / 400** (all `int`, and for all 400 records `str(image_id) == os.path.splitext(image_name)[0]`).
  - Distinct `image_name` values = **400 / 400** (0 basename duplicates in the manifest).
  - Manifest contains `content_fingerprint` or `sha256` field = **`False`** (0/400 records).
- **On-Disk Image Resolution & Duplicate Copies Across Candidate Directories:**
  - All **400 / 400** images exist locally in `data/vitextvqa/st_images/` (first candidate directory in `load_benchmark_dataset`) **and** each of the 400 images also has a second copy in one of the four `data/xanylabeling_workspace/*` directories (`Cao_Anh_001_100`, `Trieu_101_200`, `Cam_Tu_201_300`, `Thanh_Tam_301_400`).
  - Full SHA-256 comparison across both copies of all 400 images confirmed **0 byte-wise differences** (`images whose copies differ byte-wise: 0`).
- **Composite Identity & Grouping Verification Across All 400 Real Images:**
  - Calling `build_image_cache_identity(im, image_id=i["image_id"], image_path=p, dataset_root=PROJECT_ROOT, dataset_version="answer_bboxes_400_verified:dev")` on all 400 on-disk images produced:
    - **400 / 400 distinct `to_cache_key(1)` strings**.
    - **400 / 400 distinct normalized relative paths** (`'data/vitextvqa/st_images/<id>.jpg'`, 0 duplicates).
    - **0 mismatches** between manifest `(width, height)` and actual `PIL.Image.size`.
  - Grouping all 1,222 QA via `group_samples_by_image()` produced **400 groups** whose ordered question-ID partitions are **100% identical** (`identical partition of QA: True`) to the legacy grouping on this manifest, while preventing cross-dataset or cross-directory collisions on future manifests.

---

## 6. Scope and Repository-State Assessment (Checklist R8)

Independent inspection of `git status --short`, `git diff --stat`, and `git log -1`:

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

```text
 scripts/benchmark/evaluate_vitextvqa.py         |  82 +++++-
 scripts/benchmark/sg_vcd/__init__.py            |   3 +
 scripts/benchmark/sg_vcd/attention_extractor.py |  86 +++++-
 scripts/benchmark/sg_vcd/feature_cache.py       | 363 ++++++++++++++++++++++--
 scripts/benchmark/sg_vcd/pipeline.py            |  82 +++++-
 5 files changed, 566 insertions(+), 50 deletions(-)
```

- **Scope Verification:**
  - Only the 5 intended `OPT-06` source files were modified, and `tests/test_opt06_image_cache_identity.py` was added.
  - **No** `OPT-03` (working-map cache), `OPT-02` (memory cleanup removal — `gc.collect()` and `torch.cuda.empty_cache()` remain untouched at [`pipeline.py#L963-L965`](../../scripts/benchmark/sg_vcd/pipeline.py#L963-L965) and [`attention_extractor.py#L778-L779`](../../scripts/benchmark/sg_vcd/attention_extractor.py#L778-L779)), `OPT-01` (batched decoding), or `OPT-04` (GPU affinity field) code was introduced.
  - **No** git commits or pushes were performed (`HEAD` remains at `574f36d`).

---

## 7. Recommended Minor Fixes (Optional Before or During Next Ticket)

None of the findings below block `OPT-06` on the ViTextVQA 400-image dataset, but applying these minimal refinements will harden the implementation for arbitrary directory layouts and callers:

1. **Fix for M-01 ([`scripts/benchmark/sg_vcd/feature_cache.py#L104-L126`](../../scripts/benchmark/sg_vcd/feature_cache.py#L104-L126)):**
   - When `dataset_root` is supplied and `os.path.relpath(raw, dataset_root)` succeeds without a leading `".."`, return that normalized relative POSIX path directly (with `has_dir_scope = "/" in rel`) instead of running the `"data"`/`"results"` anchor search or `parts[-3:]` truncation.
2. **Fix for M-02 ([`scripts/benchmark/sg_vcd/feature_cache.py#L244-L246`](../../scripts/benchmark/sg_vcd/feature_cache.py#L244-L246)):**
   - Before returning `image_key` on the `isinstance(image_key, ImageCacheIdentity)` fast-path, verify that if `image_id` or `dataset_version` was also explicitly passed by the caller, it matches the corresponding field in `image_key`.
3. **Fix for M-03 ([`scripts/benchmark/evaluate_vitextvqa.py#L946-L986`](../../scripts/benchmark/evaluate_vitextvqa.py#L946-L986)):**
   - Extract a small helper `_sample_image_group_key(s)` used by `group_samples_by_image()`, `--test`, `--test-images`, and `img_to_idx` multi-worker sharding so all four places partition images by the exact same composite identity.

---

## 8. Final Recommendation

**Recommendation: CONDITIONALLY ACCEPT (`PASS WITH CONDITIONS`)** and forward `TICKET-OPT-06` along with this independent QA report to **ChatGPT** (for methodological/architecture review) and the **Project Owner** (for gate approval and decision on whether to apply the 3 optional minor hardening fixes M-01..M-03 before proceeding to the next ticket).

