# BÁO CÁO KIỂM ĐỊNH CHẤT LƯỢNG ĐỘC LẬP (INDEPENDENT ACCEPTANCE QA) — `TICKET-OPT-03`: IMAGE-LEVEL WORKING-MAP CACHING

- **Mã tài liệu:** `TASK_08_OPT03_INDEPENDENT_ACCEPTANCE_QA`
- **Ngày thực hiện:** 2026-11-08
- **Tác nhân kiểm định (Agent):** Antigravity (`Anti`) — Independent Senior ML Systems Reviewer
- **Tác nhân triển khai được kiểm định:** `4argon` (`TASK 08`)
- **Cấp thẩm định phương pháp luận:** ChatGPT
- **Cấp phê duyệt cuối cùng:** Project Owner
- **Chế độ thực thi:** `STRICT READ-ONLY` (Chỉ đọc mã nguồn và chạy kiểm thử synthetic/mock độc lập; không chỉnh sửa mã nguồn, kiểm thử, dữ liệu hay báo cáo hiện có)
- **Phạm vi kiểm định:** `TICKET-OPT-03 — Image-Level Working-Map Caching`

---

## 1. Tóm tắt Điều hành & Định danh Chính xác của Ticket (Executive Summary & Exact Ticket Identity)

### 1.1. Định danh chính xác của Ticket
- **Ticket được kiểm định:** **`TICKET-OPT-03 — Image-Level Working-Map Caching`**
- **Nguồn đặc tả gốc:** Mục 4, Ticket 2 của [`docs/research/PERFORMANCE_OPTIMIZATION_HANDOFF_v1.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/PERFORMANCE_OPTIMIZATION_HANDOFF_v1.md#L94-L108).
- **Điều kiện tiên quyết đã nghiệm thu:** `TICKET-OPT-06 — Stable Image Identity and Cache Key` (đã được nghiệm thu kỹ thuật tại [`docs/research/TASK_07_FIX_OPT06_INDEPENDENT_ACCEPTANCE_QA.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/TASK_07_FIX_OPT06_INDEPENDENT_ACCEPTANCE_QA.md)).

### 1.2. Tóm tắt kết quả kiểm định độc lập
Qua kiểm tra trực tiếp cây mã nguồn hiện hành (`working tree`), đối chiếu `git diff` độc lập, kiểm toán tập dữ liệu chuẩn [`answer_bboxes_400_verified.json`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/results/swintextspotter_auto_annot/answer_bboxes_400_verified.json), và thực thi độc lập 6 bộ kiểm thử cùng các kịch bản thăm dò (probe):

1. **Phần mã nguồn triển khai (`adaptive_propagator.py`, `pipeline.py`) và bộ kiểm thử (`tests/test_opt03_working_map_cache.py`) đạt yêu cầu kỹ thuật cốt lõi (Zero Blocking Defects):**
   - Tái sử dụng chính xác bộ 6 phần tử `working_maps = (opp_t, bg_smooth_t, stroke_energy_t, horiz_mod_t, scale_x, scale_y)` giữa các câu hỏi thuộc cùng một bức ảnh dưới khóa `resolved_image_key: ImageCacheIdentity` của `OPT-06`.
   - Tự động vô hiệu hóa và giải phóng tham chiếu `working_maps` cũ **trước khi** tính toán `working_maps` cho ảnh mới, đảm bảo dung lượng bộ nhớ đệm tối đa 1 ảnh (`single-image capacity bound`) và không rò rỉ dữ liệu cũ khi xảy ra ngoại lệ chuyển ảnh (`exception safety`).
   - Xóa sạch cả `_cached_working_maps` và `_cached_working_maps_key` khi gọi [`SGVCDPipeline.reset_cache()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L360-L375).
   - Không lưu đệm bất kỳ đại lượng nào phụ thuộc câu hỏi (bản đồ chú ý $\Phi, R, M_{\text{aff}}, A_Q$, vùng chọn Top-$K$, hạt giống `seed_patches`, hay kết quả lan truyền hộp giới hạn).
   - Đạt tương đương số học 100% (`Synthetic Stage 1 parity` và `Mock end-to-end pipeline parity`) giữa chế độ có lưu đệm và không lưu đệm; các tensor lưu đệm hoàn toàn chỉ đọc (`read-only`), không bị biến đổi tại chỗ (`in-place mutation`) và không phát sinh phép sao chép (`clone`) phòng thủ dư thừa.
2. **Các phát hiện không chặn nghiệm thu (`Non-Blocking Findings`) cần ghi nhận và đính chính trong hồ sơ:**
   - **`NBF-01` (MEDIUM — Sai lệch số liệu câu hỏi và phép tính tỷ lệ tiết kiệm cache trong báo cáo của `4argon`):** Báo cáo [`TASK_08_OPT03_WORKING_MAP_CACHE_IMPLEMENTATION_REPORT.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/TASK_08_OPT03_WORKING_MAP_CACHE_IMPLEMENTATION_REPORT.md#L20) ghi `1,324` câu hỏi ($\sim 3.31$ QA/ảnh), `924` cache hits và giảm `69.8%`. Kiểm toán trực tiếp trên tệp dữ liệu chuẩn [`answer_bboxes_400_verified.json`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/results/swintextspotter_auto_annot/answer_bboxes_400_verified.json) chứng minh con số chuẩn xác là **400 ảnh / 1.222 câu hỏi** (`3.055` QA/ảnh), tương ứng giảm số lần gọi `_compute_working_maps` từ **1.222 xuống 400**, đạt chính xác **822 cache hits** và mức giảm **`67.27%` (`67.2668%`)**.
   - **`NBF-02` (MEDIUM — Sai lệch mô tả tên trường và câu lệnh `assert` của `test_E` / `test_F` trong văn bản báo cáo của `4argon`):** Mục 4 và Mục 8 của báo cáo `4argon` mô tả một số tên thuộc tính và câu lệnh kiểm tra không khớp với mã nguồn thực tế của [`tests/test_opt03_working_map_cache.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt03_working_map_cache.py) (chi tiết tại Mục 6 và Mục 11).
   - **`NBF-03` (LOW — Kiểm tra biên của `_is_valid_working_maps`):** Hàm [`_is_valid_working_maps`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/adaptive_propagator.py#L150-L183) chưa kiểm tra tường minh `opp_t.shape[2] == 3`, `not isinstance(scale_x, bool)`, `math.isfinite`, và sự đồng nhất `device` giữa 4 tensor nếu người dùng bên ngoài truyền tuple tự tạo.
   - **`NBF-04` (LOW — Hàm phụ trợ chẩn đoán `run_stage1_only`):** [`SGVCDPipeline.run_stage1_only()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L675-L712) gọi trực tiếp `self.propagator.propagate(...)` không qua `_cached_working_maps` (nằm ngoài phạm vi ticket `OPT-03` và vẫn hoạt động tương thích ngược 100%).

### 1.3. Phán quyết tổng thể
- **Phán quyết QA Kỹ thuật:** **`PASS WITH NON-BLOCKING FINDINGS`**
- **Trạng thái Gate B (Colab Real-Model Integration):** **`NOT AUTHORIZED`** (Chưa đủ điều kiện mở cổng cho đến khi hoàn tất các ticket phụ thuộc và có phê duyệt tường minh của Project Owner).

---

## 2. Danh mục Tài liệu Quản trị và Kỹ thuật Đã Đọc (Governance Files Read)

Toàn bộ các tài liệu quản trị và đặc tả kỹ thuật bắt buộc tại Mục 0 đã được đọc và đối chiếu trước khi kiểm định:

1. `AGENTS.md`: Đã kiểm tra tại thư mục gốc kho mã nguồn — không tồn tại tệp `AGENTS.md` riêng; quyền hạn và quy tắc vận hành tuân theo [`.github/copilot-instructions.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/.github/copilot-instructions.md) và [`PROJECT.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/PROJECT.md).
2. [`.github/copilot-instructions.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/.github/copilot-instructions.md)
3. [`.github/skills/anti-output-quality-review/SKILL.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/.github/skills/anti-output-quality-review/SKILL.md)
4. [`.github/skills/research-plan-compliance-review/SKILL.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/.github/skills/research-plan-compliance-review/SKILL.md)
5. [`.github/skills/local-colab-experiment-workflow/SKILL.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/.github/skills/local-colab-experiment-workflow/SKILL.md)
6. [`docs/research/PERFORMANCE_OPTIMIZATION_HANDOFF_v1.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/PERFORMANCE_OPTIMIZATION_HANDOFF_v1.md)
7. [`docs/research/TASK_07_FIX_OPT06_INDEPENDENT_ACCEPTANCE_QA.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/TASK_07_FIX_OPT06_INDEPENDENT_ACCEPTANCE_QA.md)
8. [`docs/research/TASK_08_OPT03_WORKING_MAP_CACHE_IMPLEMENTATION_REPORT.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/TASK_08_OPT03_WORKING_MAP_CACHE_IMPLEMENTATION_REPORT.md)
9. [`docs/research/REMEDIATION_SPEC_P0_v1.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/REMEDIATION_SPEC_P0_v1.md)
10. [`docs/Proposed_Method_SG_VCD.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/Proposed_Method_SG_VCD.md)

---

## 3. Các File và Hàm Mã nguồn Đã Kiểm tra Độc lập (Source Files and Functions Independently Inspected)

| Tệp tin | Các lớp / hàm / vùng dòng thực tế đã kiểm tra | Nội dung kiểm định độc lập |
| :--- | :--- | :--- |
| [`scripts/benchmark/sg_vcd/adaptive_propagator.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/adaptive_propagator.py) | • [`_is_valid_working_maps`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/adaptive_propagator.py#L150-L183)<br>• [`BaseSeedPropagator.propagate`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/adaptive_propagator.py#L190-L200)<br>• [`BaseSeedPropagator.propagate_multi`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/adaptive_propagator.py#L202-L226)<br>• [`AdaptiveScalePropagator._compute_working_maps`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/adaptive_propagator.py#L316-L389)<br>• [`AdaptiveScalePropagator.propagate`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/adaptive_propagator.py#L568-L1311)<br>• [`AdaptiveScalePropagator.propagate_multi`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/adaptive_propagator.py#L1314-L1344) | Kiểm tra logic xác thực tuple 6 phần tử; kiểm tra toàn bộ các dòng `598–1277` trong `propagate()` để xác nhận `opp_t`, `bg_smooth_t`, `stroke_energy_t`, `horiz_mod_t` chỉ được đọc (out-of-place operations) và lệnh `del` chỉ giải phóng biến cục bộ. |
| [`scripts/benchmark/sg_vcd/pipeline.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py) | • [`PipelineResult`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L77-L100)<br>• [`SGVCDPipeline.__init__`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L311-L358)<br>• [`SGVCDPipeline.reset_cache`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L360-L375)<br>• [`SGVCDPipeline.run_stage1_only`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L675-L712)<br>• [`SGVCDPipeline.run`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L768-L1065) | Kiểm tra việc tạo `resolved_image_key` (`lines 804–813`), kiểm tra nhánh cache hit/miss/exception tại Stage 1.3 (`lines 875–930`), và kiểm tra tất cả các điểm gọi `propagate` / `propagate_multi`. |
| [`scripts/benchmark/sg_vcd/feature_cache.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/feature_cache.py) | • [`ImageCacheIdentity`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/feature_cache.py#L38-L55)<br>• [`build_image_cache_identity`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/feature_cache.py#L254-L385) | Kiểm tra tính tương thích giữa `resolved_image_key` của `OPT-06` và khóa `_cached_working_maps_key` của `OPT-03`. |
| [`scripts/benchmark/evaluate_vitextvqa.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/evaluate_vitextvqa.py) | • [`load_benchmark_dataset`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/evaluate_vitextvqa.py#L316-L398)<br>• [`_sample_image_group_key`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/evaluate_vitextvqa.py#L400-L410)<br>• [`group_samples_by_image`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/evaluate_vitextvqa.py#L413-L423)<br>• [`run_benchmark`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/evaluate_vitextvqa.py#L1045-L1200) | Kiểm tra thứ tự gom nhóm câu hỏi theo ảnh và cách truyền `image_cache_key` vào `pipeline.run(...)`. |
| [`scripts/benchmark/sg_vcd/roi_cropper.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/roi_cropper.py) & [`tests/test_adversarial_m3_spatial.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_adversarial_m3_spatial.py) | • [`DynamicMultiTileROICropper.crop`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/roi_cropper.py#L118-L233)<br>• [`TestAdversarialBoundingBoxes.test_04_extreme_aspect_ratios`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_adversarial_m3_spatial.py#L119-L135) | Kiểm tra lịch sử commit Git và nguyên nhân gốc rễ của 2 subtest lỗi `1200 != 400` / `400 != 1200`. |
| [`tests/test_opt03_working_map_cache.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt03_working_map_cache.py) | Toàn bộ tệp (`lines 1–677`), bao gồm `_QuestionConditionedMockExtractor`, `_DummyVinternModel`, và `test_A` $\rightarrow$ `test_H` | Kiểm tra từng câu lệnh `assert`, đối chiếu giữa thực tế kiểm thử và lời tuyên bố trong báo cáo của `4argon`. |

---

## 4. Các Lệnh Kiểm Thử Độc lập và Kết quả Thực tế (Test Commands and Actual Results)

Tất cả các bộ kiểm thử yêu cầu tại Mục 6 đã được chạy độc lập bằng Python trong môi trường ảo của dự án (`..\Document AI\VLM_Reasoning\venv\Scripts\python.exe`).

### 4.1. Bảng tổng hợp kết quả thực thi kiểm thử độc lập

| STT | Lệnh thực thi | Tổng số test | Đạt (Pass) | Lỗi (Fail) | Bỏ qua (Skip) | Thời gian | Mã thoát (Exit Code) |
| :-: | :--- | :-: | :-: | :-: | :-: | :-: | :-: |
| 1 | `& $py -m unittest -v tests.test_opt03_working_map_cache` | **8** | **8** | 0 | 0 | `8.301s` | `0` |
| 2 | `& $py -m unittest -v tests.test_opt06_image_cache_identity` | **11** | **11** | 0 | 0 | `2.506s` | `0` |
| 3 | `& $py -m unittest -v tests.test_sg_vcd` | **33** | **33** | 0 | 0 | `0.257s` | `0` |
| 4 | `& $py -m unittest -v tests.test_adversarial_spatial_processing` | **21** | **21** | 0 | 0 | `0.046s` | `0` |
| 5A | `& $py -m unittest -v tests.test_adversarial_m3_spatial.TestAdversarialMultiTileSlicing tests.test_adversarial_m3_spatial.TestAdversarialSobelGradientPrior` | **8** | **8** | 0 | 0 | `1.064s` | `0` |
| 5B | `& $py -m unittest -v tests.test_adversarial_m3_spatial.TestAdversarialBoundingBoxes` | **6** | **5** | **1** *(2 subtests)* | 0 | `0.979s` | `1` |
| 6 | `& $py -m unittest -v tests.test_tier4_workloads.TestTier4RealWorldWorkloads.test_05_anti_hardcoding_and_heuristics_audit` | **1** | **1** | 0 | 0 | `0.184s` | `0` |

### 4.2. Đầu ra thực tế của bộ kiểm thử `tests.test_opt03_working_map_cache`
```text
test_A_first_question_cache_miss_stores_under_resolved_image_key (tests.test_opt03_working_map_cache.TestOPT03WorkingMapCache.test_A_first_question_cache_miss_stores_under_resolved_image_key)
Test A — First question/cache miss: ... ok
test_B_same_image_multiple_questions_reuses_cache_and_runs_question_specific_steps (tests.test_opt03_working_map_cache.TestOPT03WorkingMapCache.test_B_same_image_multiple_questions_reuses_cache_and_runs_question_specific_steps)
Test B — Same image/multiple questions: ... ok
test_C_image_transition_invalidates_previous_maps_and_retains_single_entry (tests.test_opt03_working_map_cache.TestOPT03WorkingMapCache.test_C_image_transition_invalidates_previous_maps_and_retains_single_entry)
Test C — Image transition: ... ok
test_D_reset_cache_clears_working_maps_and_key (tests.test_opt03_working_map_cache.TestOPT03WorkingMapCache.test_D_reset_cache_clears_working_maps_and_key)
Test D — Reset lifecycle: ... ok
test_E_cached_versus_uncached_exact_parity (tests.test_opt03_working_map_cache.TestOPT03WorkingMapCache.test_E_cached_versus_uncached_exact_parity)
Test E — Cached versus uncached parity: ... ok
test_F_repeated_use_and_immutability_zero_mutation_zero_clone (tests.test_opt03_working_map_cache.TestOPT03WorkingMapCache.test_F_repeated_use_and_immutability_zero_mutation_zero_clone)
Test F — Repeated use and immutability: ... ok
test_G_failure_safety_on_image_transition_exception (tests.test_opt03_working_map_cache.TestOPT03WorkingMapCache.test_G_failure_safety_on_image_transition_exception)
Test G — Failure safety: ... ok
test_H_regression_backward_compatibility_and_non_accumulation_across_images (tests.test_opt03_working_map_cache.TestOPT03WorkingMapCache.test_H_regression_backward_compatibility_and_non_accumulation_across_images)
Test H — Regression & Memory Non-Accumulation: ... ok

----------------------------------------------------------------------
Ran 8 tests in 8.301s

OK
EXIT_1=0
```

---

## 5. Đánh giá Vòng đời Bộ nhớ đệm và Định danh Khóa (Cache Lifecycle and Identity Assessment)

Kiểm tra trực tiếp mã nguồn tại [`scripts/benchmark/sg_vcd/pipeline.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py) và [`scripts/benchmark/sg_vcd/adaptive_propagator.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/adaptive_propagator.py) xác nhận:

1. **Tính đồng nhất của khóa định danh (`ImageCacheIdentity`):**
   - Tại đầu hàm [`SGVCDPipeline.run()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L804-L813), `resolved_image_key` được xây dựng đúng 1 lần qua `build_image_cache_identity(...)`.
   - Cùng một thực thể `resolved_image_key` này được truyền cho `self.extract_visual_features(image, image_key=resolved_image_key)` (Stage 2 global thumbnail cache), `self.extractor.extract(image, question, image_key=resolved_image_key)` (Stage 1.1 visual token cache), và dùng làm khóa kiểm tra `self._cached_working_maps_key == resolved_image_key` tại Stage 1.3 (`line 878`).
   - Giới hạn vốn có của fingerprint 9 điểm ảnh trong `OPT-06` (`probe:...`) được giữ nguyên đúng như đã ghi nhận tại [`TASK_07_FIX_OPT06_INDEPENDENT_ACCEPTANCE_QA.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/TASK_07_FIX_OPT06_INDEPENDENT_ACCEPTANCE_QA.md#L189-L197) (cơ chế phát hiện thay đổi dựa trên nỗ lực tốt nhất trong phạm vi một lần chạy, không phải hàm băm mật mã toàn ảnh).
2. **Thứ tự vô hiệu hóa và an toàn ngoại lệ (Invalidation Before Computation & Exception Safety):**
   - Tại [`pipeline.py#L877-L891`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L877-L891):
     ```python
     if (
         self._cached_working_maps_key == resolved_image_key
         and _is_valid_working_maps(self._cached_working_maps)
     ):
         working_maps = self._cached_working_maps
     else:
         # Release previous cached working maps and key before computing new maps
         self._cached_working_maps = None
         self._cached_working_maps_key = None
         if hasattr(self.propagator, "_compute_working_maps"):
             computed_maps = self.propagator._compute_working_maps(image)
             if _is_valid_working_maps(computed_maps):
                 self._cached_working_maps = computed_maps
                 self._cached_working_maps_key = resolved_image_key
             working_maps = computed_maps
     ```
   - Việc gán `self._cached_working_maps = None` và `self._cached_working_maps_key = None` diễn ra **trước** lời gọi `self.propagator._compute_working_maps(image)`. Nếu `_compute_working_maps(image)` ném ngoại lệ, khối `except Exception:` tại `line 964` bắt lỗi và rơi về hộp dự phòng `selection.pixel_bbox`, trong khi cả `_cached_working_maps` lẫn `_cached_working_maps_key` đều đã ở trạng thái `None` — tuyệt đối không thể xảy ra hiện tượng dùng nhầm `working_maps` của ảnh cũ cho ảnh mới hay gắn khóa mới vào tensor cũ.
3. **Làm sạch qua `reset_cache()`:**
   - [`SGVCDPipeline.reset_cache()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L360-L375) xóa đồng bộ `self._cached_working_maps = None`, `self._cached_working_maps_key = None`, `self.feature_cache.clear()`, và `self.extractor.clear_cache()`.
4. **Đánh giá hàm `_is_valid_working_maps` (`NBF-03` — Non-blocking):**
   - Trong [`adaptive_propagator.py#L150-L183`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/adaptive_propagator.py#L150-L183), hàm kiểm tra đúng kiểu `tuple` độ dài 6, kiểu `torch.Tensor` cho 4 phần tử đầu, số chiều `ndim` (`3, 3, 2, 2`), khớp kích thước 2 chiều đầu `opp_t.shape[:2] == stroke_energy_t.shape`, và `float(scale_x) > 0.0`, `float(scale_y) > 0.0`.
   - Tuy nhiên, có sự khác biệt nhỏ giữa mô tả trong báo cáo của `4argon` ([`TASK_08_OPT03_WORKING_MAP_CACHE_IMPLEMENTATION_REPORT.md#L66-L68`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/TASK_08_OPT03_WORKING_MAP_CACHE_IMPLEMENTATION_REPORT.md#L66-L68)) và mã nguồn thực tế:
     1. Báo cáo ghi tensor `opp_t` và `bg_smooth_t` có kích thước `(3, H_w, W_w)`, nhưng thực tế [`_compute_working_maps`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/adaptive_propagator.py#L378-L379) trả về dạng kênh cuối `(H_w, W_w, 3)` (`HWC`).
     2. Báo cáo ghi `_is_valid_working_maps` loại trừ kiểu `bool` cho `scale_x, scale_y` và kiểm tra kích thước dương, nhưng mã thực tế dùng `float(scale_x) <= 0.0` (trong Python `float(True) == 1.0` nên `True` vẫn vượt qua) và chưa kiểm tra `opp_t.shape[2] == 3`, `stroke_energy_t.numel() > 0`, `math.isfinite`, hay cùng `device`. Dù vậy, đối với mọi tuple sinh bởi `_compute_working_maps(image)` (`H_w, W_w >= 64`, `shape[2] == 3`, cùng `self.device`), hàm hoạt động hoàn toàn chính xác.

---

## 6. Kiểm toán Trọng tâm B — Đánh giá Tính Tương đương Giữa Có Cache và Không Cache (Critical Audit B — Cached/Uncached Parity & Mock Boundaries)

### 6.1. Phân định rõ ranh giới Thật / Mock trong `test_E_cached_versus_uncached_exact_parity`

Kiểm tra trực tiếp [`tests/test_opt03_working_map_cache.py#L462-L531`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt03_working_map_cache.py#L462-L531):

| Thành phần trong `test_E` | Mã nguồn thực tế hay Mock/Stub? | Chi tiết thực thi |
| :--- | :---: | :--- |
| **Ảnh đầu vào (`img`)** | **Synthetic PIL Image** | Tạo bởi [`_make_synthetic_signboard_image()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt03_working_map_cache.py#L66-L95) (`320x240` RGB có 2 biển hiệu ngang phía trên và phía dưới). |
| **Stage 1.1 (`extractor`)** | **Mock (`_QuestionConditionedMockExtractor`)** | Trả về bản đồ $\Phi$ tổng hợp có đỉnh chú ý ở biển hiệu trên (`Q1`, `Q3`) hoặc biển hiệu dưới (`Q2`, `Q4`). |
| **Stage 1.2 (`selector`)** | **Mã nguồn thật (`KLDivergenceRegionSelector`)** | Chạy thuật toán chọn vùng Top-$K$ thật (`select_top_k`) kèm điều biến gradient Sobel trên ảnh synthetic. |
| **Stage 1.3 (`propagator`)** | **Mã nguồn thật (`AdaptiveScalePropagator`)** | Chạy toàn bộ thuật toán thật: `_compute_working_maps`, tách mẫu màu tiền cảnh/hậu cảnh, tự hiệu chuẩn chiều cao chữ $H_{\text{char}}$ bằng tự tương quan 1D, quét dải 1D và mở vành đai dấu tiếng Việt $\pm 0.5 H_{\text{char}}$. |
| **Stage 1.4 (`cropper`)** | **Mã nguồn thật (`DynamicMultiTileROICropper`)** | Chạy thuật toán cắt ROI đa mảnh đẳng hướng thật (`crop`). |
| **Stage 2 (`model`, `tokenizer`, `decoder`)** | **Mock LLM (`_DummyVinternModel`, `_DummyTokenizer`) + Decoder thật** | Dùng mô hình giả lập trọng số cố định ưu tiên token kết thúc câu và hàm giải mã giả lập trả về chuỗi cố định `"nhà thuốc bảo tâm an"`. |

### 6.2. Đối chiếu giữa các phép kiểm tra thực tế trong `test_E` / `test_F` và lời tuyên bố trong báo cáo của `4argon` (`NBF-02`)

Kiểm tra đối chiếu từng dòng cho thấy:
1. **Những gì `test_E` thực sự chứng minh trong mã kiểm thử ([`lines 473–531`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt03_working_map_cache.py#L473-L531)):**
   - **Phần 1 (Trực tiếp trên `AdaptiveScalePropagator`, `lines 473–492`):** So sánh `propagate_multi(..., working_maps=shared_maps)` với `propagate(..., working_maps=None)` trên 4 hạt giống `[(12, 7), (16, 22), (8, 7), (20, 23)]` thông qua hàm [`_assert_seed_propagation_result_equal`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt03_working_map_cache.py#L277-L299). Hàm này kiểm tra **từng trường một** của `SeedPropagationResult`: `seed_pixel`, `seed_patch`, `propagated_patch_bbox`, `propagated_pixel_bbox`, `core_pixel_bbox`, toàn bộ 9 trường của `prototype` (**bao gồm `char_height_px` và `char_height_patches`** cùng `mu_fg`, `mu_bg`, `delta_mu`), `np.array_equal(patch_mask)`, và `np.array_equal(compatibility_map)`.
   - **Phần 2 (Tích hợp qua `SGVCDPipeline.run()`, `lines 494–531`):** Chạy 4 câu hỏi (`Q1..Q4`) với cache bật so với gọi `reset_cache()` trước mỗi câu hỏi, và kiểm tra `stage1_bbox`, `stage1_bboxes`, `selection.(pixel_bbox, grid_bbox, kl_score)`, `multi_selection.(pixel_bboxes, grid_bboxes, kl_scores)`, `crop.isotropic_box`, `crop.magnification`, `pred_text`, và `pred_sg_vcd`.
2. **Điểm bất일치 trong văn bản báo cáo của `4argon` (`NBF-02`):**
   - Trong [`TASK_08_OPT03_WORKING_MAP_CACHE_IMPLEMENTATION_REPORT.md#L209-L223`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/TASK_08_OPT03_WORKING_MAP_CACHE_IMPLEMENTATION_REPORT.md#L209-L223), `4argon` viết rằng `test_E` chạy 3 câu hỏi (`Q1, Q2, Q3`) và kiểm tra các thuộc tính `seed_boxes`, `propagated_boxes`, `grid_dims`, `num_patches`, `selected_k`, `answer`, `raw_answer`, `num_steps`, `avg_js_divergence`, cũng như `np.array_equal(np.asarray(tile_c), np.asarray(tile_u))` trên `crop_result.roi_tiles`. Thực tế, các tên thuộc tính đó không tồn tại trên `PipelineResult`, và `test_E` trong file test chưa gọi `np.array_equal` trên các mảng điểm ảnh `crop.roi_tiles`.
   - Tương tự, tại dòng `221–222` của báo cáo, `4argon` viết rằng `test_F` kiểm tra `t_after_Q3.data_ptr() == ptr_Q1` và đếm số lần gọi `torch.Tensor.clone == 0`. Thực tế, [`test_F`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt03_working_map_cache.py#L532-L568) kiểm tra `self.assertIs(wm_after, wm)` và `self.assertTrue(torch.equal(wm_after[i], snap_*))`.
   - **Kiểm chứng bổ sung bằng Probe độc lập của Anti QA:** Để xác minh thực chất các mảnh ảnh cắt `crop.roi_tiles` và `crop.neg_tiles` có khớp bit-for-bit hay không, Anti QA đã chạy probe độc lập đối chiếu trực tiếp `np.array_equal(np.asarray(tc), np.asarray(tu))` trên toàn bộ `roi_tiles` và `neg_tiles` của cả 4 câu hỏi `Q1..Q4`:
     ```text
     Q1: stage1_bboxes=True, grid=True, roi_tiles_bit_eq=True, neg_tiles_bit_eq=True
     Q2: stage1_bboxes=True, grid=True, roi_tiles_bit_eq=True, neg_tiles_bit_eq=True
     Q3: stage1_bboxes=True, grid=True, roi_tiles_bit_eq=True, neg_tiles_bit_eq=True
     Q4: stage1_bboxes=True, grid=True, roi_tiles_bit_eq=True, neg_tiles_bit_eq=True
     ```

### 6.3. Phân loại chính xác mức độ tương đương (Mandatory Terminology Classification)
- **`Synthetic Stage 1 parity`:** **VERIFIED (100% Exact Bit-for-Bit Parity)** — Toàn bộ hộp giới hạn `SeedPropagationResult`, chiều cao chữ `char_height_px`, nguyên mẫu màu, mặt nạ patch, bản đồ tương thích, và các mảnh ảnh cắt ROI (`roi_tiles`, `neg_tiles`) đều giống hệt 100% giữa chế độ có lưu đệm và không lưu đệm.
- **`Mock end-to-end pipeline parity`:** **VERIFIED (100% Exact Parity under Mock Extractor & Mock LLM)** — Luồng `SGVCDPipeline.run()` hoàn chỉnh với mock extractor và mock LLM cho kết quả giống hệt 100%.
- **`Real-model inference parity`:** **`NOT TESTED`** — Theo đúng quy định cấm nạp trọng số thật `5CD-AI/Vintern-1B-v3_5` ở máy cục bộ, tính tương đương trên mô hình Vintern thật chưa được kiểm thử và phải thực hiện tại Gate B trên Google Colab khi được phép.

---

## 7. Kiểm toán Trọng tâm A — Đối soát Số lượng Mẫu Tập dữ liệu và Phép tính Tiết kiệm Cache (Critical Audit A — Dataset-Count and Savings Arithmetic Reconciliation)

Kiểm toán độc lập đã được thực hiện trực tiếp bằng script Python trên tệp dữ liệu gốc và hàm nạp dữ liệu thực tế của dự án:

1. **Bản chụp tập dữ liệu chuẩn có thẩm quyền (Authoritative Dataset Snapshot):**
   - Đường dẫn chuẩn: [`results/swintextspotter_auto_annot/answer_bboxes_400_verified.json`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/results/swintextspotter_auto_annot/answer_bboxes_400_verified.json) (được tham chiếu bởi [`evaluate_vitextvqa.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/evaluate_vitextvqa.py), [`PROJECT.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/PROJECT.md), [`REMEDIATION_SPEC_P0_v1.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/REMEDIATION_SPEC_P0_v1.md#L112-L126), và [`Research Plan v2`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/SG_VCD_Research_Review_Consolidated_Draft_v2.md)).
   - Toàn bộ kho mã nguồn hoàn toàn **không có** bất kỳ tệp dữ liệu JSON nào có `1,324` câu hỏi (cả `answer_bboxes_400.json`, `answer_bboxes_400_verified.json`, và `summary_dev_400.json` đều có đúng `400` ảnh và `1,222` câu hỏi). Con số `1,324` trong báo cáo của `4argon` là một lỗi tính nhẩm từ con số trung bình xấp xỉ ($400 \times 3.31 = 1,324$).
2. **Số lượng ảnh chính xác (Exact Image Count):** **`400` ảnh** (`len(data["images"]) == 400`, `len(grouped_samples) == 400`).
3. **Số lượng câu hỏi chính xác (Exact QA Count):** **`1,222` câu hỏi** (`sum(len(im["quests"]) for im in data["images"]) == 1222`, `len(load_benchmark_dataset(...)) == 1222`, trung bình chính xác **`3.055` QA/ảnh**).
4. **Bằng chứng băm và siêu dữ liệu phiên bản (Manifest / Hash / Version Evidence):**
   - **SHA-256:** `1428e33d493c0f00256f21c640a1923df989436294ff6190cd349fadddb13c5c`
   - **Metadata nội tại:** `{"model": "SwinTextSpotter (SWINTS-swin-vintext)", "split": "dev", "num_images": 400, "total_quests": 1222, "matched_quests": 1153, "quest_match_rate_pct": 100.0, "total_answer_region_bboxes": 1037, "verified_quests": 1222, "still_missing_quests": 0}`
5. **Kiểm chứng gom nhóm liên tiếp trong trình chạy thực tế (`evaluate_vitextvqa.py`):**
   - Trong [`run_benchmark()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/evaluate_vitextvqa.py#L1061-L1167), `grouped_samples = group_samples_by_image(remaining_samples)` gom toàn bộ câu hỏi của từng ảnh theo khóa `_sample_image_group_key(s)` thành một danh sách liên tiếp `q_list`. Vòng lặp ngoài duyệt qua từng ảnh (`400` nhóm ảnh), và vòng lặp trong (`for item in q_list:`) gọi `pipeline.run(image, question, ..., image_key=image_cache_key)` liên tiếp cho tất cả các câu hỏi của ảnh đó trước khi chuyển sang ảnh tiếp theo. Ngay cả khi chạy phân mảnh đa tiến trình (`--num-workers > 1`), phép chia mảnh tại `lines 984–993` phân hoạch theo `_sample_image_group_key(s)`, nên mọi câu hỏi của cùng một ảnh luôn nằm trọn trong cùng một worker và được xử lý liên tiếp nhau.
6. **Bảng đối chiếu số học chính xác Trước và Sau `OPT-03`:**

| Chỉ số | Giá trị sai trong Báo cáo `4argon` | Giá trị Chính xác Đã Kiểm toán (`SOURCE-VERIFIED`) | Giải thích / Công thức |
| :--- | :---: | :---: | :--- |
| Tổng số ảnh duy nhất ($M$) | `400` | **`400`** | `len(grouped_samples) == 400` |
| Tổng số câu hỏi ($N$) | `1,324` *(Sai)* | **`1,222`** | `len(samples) == 1222` |
| Số câu hỏi trung bình / ảnh ($N / M$) | $\sim 3.31$ *(Sai)* | **`3.055`** | $1,222 / 400 = 3.055$ |
| Số lần gọi `_compute_working_maps` **TRƯỚC `OPT-03`** | `1,324` *(Sai)* | **`1,222`** | 1 lần gọi cho mỗi câu hỏi (do `propagate_multi` tính 1 lần/câu hỏi) |
| Số lần gọi `_compute_working_maps` **SAU `OPT-03`** | `400` | **`400`** | 1 lần gọi ở câu hỏi đầu tiên của mỗi nhóm ảnh (`400` Cache Misses) |
| **Số lần Cache Hit chính xác ($N - M$)** | `924` *(Sai)* | **`822`** | $1,222 - 400 = 822$ lần tái sử dụng từ bộ đệm |
| **Tỷ lệ giảm số lần gọi chính xác ($(N - M) / N$)** | `69.8%` *(Sai)* | **`67.27%` (`67.2668%`)** | $(822 / 1,222) \times 100\% = 67.266776\%$ |

---

## 8. Kiểm toán Trọng tâm C — Phân tích Hai Subtest Lỗi trong `tests.test_adversarial_m3_spatial` (Critical Audit C — Two Pre-Existing Subtest Failures and Evidence of Origin)

### 8.1. Tên test chính xác và Traceback thực tế
Trong bộ kiểm thử [`tests/test_adversarial_m3_spatial.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_adversarial_m3_spatial.py#L119-L135), phương thức `TestAdversarialBoundingBoxes.test_04_extreme_aspect_ratios` có 2 subtest bị `FAIL`:
1. `test_04_extreme_aspect_ratios (name='slit_100_to_1', bbox=(50, 500, 1050, 510))` $\rightarrow$ `AssertionError: 1200 != 400` tại dòng `131` (`self.assertEqual(sx2 - sx1, sy2 - sy1)`).
2. `test_04_extreme_aspect_ratios (name='column_1_to_100', bbox=(500, 50, 510, 1050))` $\rightarrow$ `AssertionError: 400 != 1200` tại dòng `131` (`self.assertEqual(sx2 - sx1, sy2 - sy1)`).

### 8.2. Bằng chứng xác thực nguồn gốc từ lịch sử Git (Git Commit Provenance — Mức độ bằng chứng: `HIGH / SOURCE-VERIFIED`)
Không dựa vào danh sách file thay đổi đơn thuần, Anti QA đã truy vết trực tiếp lịch sử `git log` của cả hai tệp [`scripts/benchmark/sg_vcd/roi_cropper.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/roi_cropper.py) và [`tests/test_adversarial_m3_spatial.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_adversarial_m3_spatial.py):
1. **Commit `c6a1ba3973ba734e950147cd286df8b07cd7ccb5` (ngày `2026-10-28 14:57:29 +0700`):** Tạo mới đồng thời `roi_cropper.py` (khi đó `IsotropicROICropper` luôn mở rộng mọi hộp thành một hình vuông duy nhất $1:1$ có cạnh $s = \max(\Delta_x, \Delta_y)$) và `tests/test_adversarial_m3_spatial.py` (dòng `131` kiểm tra hình vuông $1:1$: `self.assertEqual(sx2 - sx1, sy2 - sy1)`).
2. **Commit `9c43e05a867891a5f6b1dfaf83480fa7c7cbbbb4` (ngày `2026-10-29 16:01:52 +0700`, *"feat(sg-vcd): implement Round 5 Stage 2 convergence architecture (multi-tile RoI, CAPC, AHRA, QIACF, TESG)"*):** Nâng cấp `roi_cropper.py` lên `DynamicMultiTileROICropper` (và gán bí danh `IsotropicROICropper = DynamicMultiTileROICropper` tại [`roi_cropper.py#L340`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/roi_cropper.py#L340)). Với hộp ngang dẹt `slit_100_to_1` (`1000 x 10`), `DynamicMultiTileROICropper.select_grid()` chọn lưới $(3, 1)$ gồm 3 mảnh vuông $400 \times 400$, nên khung bao `isotropic_box` có kích thước $1200 \times 400$ (tỷ lệ $3:1$); với hộp dọc `column_1_to_100` (`10 x 1000`), thuật toán chọn lưới $(1, 3)$ nên `isotropic_box` có kích thước $400 \times 1200$ (tỷ lệ $1:3$). Commit `9c43e05` đã bổ sung bài test mới `TestDynamicMultiPatchROICropper` trong `tests/test_sg_vcd.py` (kiểm tra đúng lưới $(3, 1)$ và $(1, 3)$), nhưng **quên cập nhật câu lệnh `assertEqual(sx2 - sx1, sy2 - sy1)` cũ từ thời Round 1 trong `tests/test_adversarial_m3_spatial.py`**.
3. **Tính độc lập với `OPT-03`:** `TestAdversarialBoundingBoxes` khởi tạo trực tiếp `self.cropper = IsotropicROICropper(target_size=448)` và gọi `self.cropper.crop(canvas, bbox)` — hoàn toàn không đi qua `AdaptiveScalePropagator` hay `SGVCDPipeline`.
4. **Phân loại:** Đây là **nợ kiểm thử hồi quy tồn đọng từ trước (`Pre-existing test-contract debt from commit 9c43e05`)**, hoàn toàn không phải lỗi hồi quy do `OPT-03` hay `OPT-06` gây ra.

---

## 9. Đánh giá Tính Tương thích Ngược và Vòng đời Bộ nhớ (Backward Compatibility and Memory Lifecycle Assessment)

1. **Tính tương thích ngược cho các lớp con `BaseSeedPropagator` và nơi gọi cũ:**
   - Trong [`BaseSeedPropagator.propagate`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/adaptive_propagator.py#L190-L200), [`BaseSeedPropagator.propagate_multi`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/adaptive_propagator.py#L202-L226), [`AdaptiveScalePropagator.propagate`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/adaptive_propagator.py#L568-L576), và [`AdaptiveScalePropagator.propagate_multi`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/adaptive_propagator.py#L1314-L1321), tham số `working_maps` đều là tùy chọn với giá trị mặc định `None`.
   - Cả `BaseSeedPropagator.propagate_multi` (`lines 212–225`) và `SGVCDPipeline.run` (`lines 893–930`) đều dùng `inspect.signature` để kiểm tra xem phương thức `propagate_multi` / `propagate` của đối tượng propagator hiện tại có nhận `working_maps` (hoặc `**kwargs`) hay không trước khi truyền đối số từ khóa `working_maps=working_maps`. Do đó, mọi lớp con tùy chỉnh không có `_compute_working_maps` hoặc không nhận tham số `working_maps` đều hoạt động bình thường không lỗi.
   - Kiểm tra tất cả các nơi gọi `.propagate(` và `.propagate_multi(` trong kho mã nguồn xác nhận:
     - [`SGVCDPipeline.run()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L893-L930): Truyền `working_maps=working_maps` từ bộ đệm cấp ảnh.
     - [`SGVCDPipeline.run_stage1_only()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L694-L698): Gọi `self.propagator.propagate(image=image, seed_patch=seed_patch, grid_ratio=extraction.grid_ratio)` với `working_maps=None` mặc định (tương thích ngược hoàn toàn; ghi nhận tại `NBF-04`).
2. **Tính bất biến của Tensor được lưu đệm và giải phóng tham chiếu cục bộ:**
   - Kiểm tra toàn bộ thân hàm [`AdaptiveScalePropagator.propagate()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/adaptive_propagator.py#L598-L1277) xác nhận `opp_t`, `bg_smooth_t`, `stroke_energy_t`, và `horiz_mod_t` chỉ tham gia các phép cắt lát (`slicing`) và phép toán tạo tensor mới ngoài chỗ (`out-of-place`). Không có bất kỳ phép gán chỉ mục (`tensor[...] = ...`) hay phương thức biến đổi tại chỗ (`add_`, `mul_`, `clamp_`, `zero_`) nào được gọi trên 4 tensor này.
   - Lệnh `del (opp_t, bg_smooth_t, stroke_energy_t, ...)` tại dòng `772` và `if not has_valid_working_maps: del horiz_mod_t` tại dòng `1274–1275` chỉ xóa tên biến cục bộ trong phạm vi hàm `propagate()`, không ảnh hưởng đến tuple lưu trong `self._cached_working_maps`.
3. **Không tích tụ bộ nhớ qua nhiều ảnh (Non-Accumulation of Tensors):**
   - `SGVCDPipeline` chỉ giữ đúng 1 thuộc tính `_cached_working_maps` (tối đa 1 ảnh tại mọi thời điểm). Kiểm thử `weakref.ref` trong `test_C` và `test_H` (qua chuỗi 6 ảnh liên tiếp) xác nhận ngay khi chuyển sang ảnh mới và chạy `gc.collect()`, toàn bộ các tensor `working_maps` của các ảnh trước đó đều được giải phóng hoàn toàn (`alive_flags == [False, False, False, False, False, True]`).

---

## 10. Trạng thái Git và Tóm tắt Diff Cô lập cho `OPT-03` (Git Status and Isolated Diff Summary)

### 10.1. Trạng thái `git status --short` hiện tại
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
?? tests/test_opt03_working_map_cache.py
?? tests/test_opt06_image_cache_identity.py
```

### 10.2. Phân tách cô lập giữa thay đổi của `OPT-03`, điều kiện tiên quyết `OPT-06`, và các tệp có sẵn

| Nhóm thay đổi | Tệp tin | Số dòng thay đổi (`git diff`) | Bản chất thay đổi |
| :--- | :--- | :---: | :--- |
| **1. Thay đổi thuộc riêng `TICKET-OPT-03` (Task 08)** | [`scripts/benchmark/sg_vcd/adaptive_propagator.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/adaptive_propagator.py) | `+62 / -7` dòng | Bổ sung `_is_valid_working_maps` và hỗ trợ `working_maps` trong `BaseSeedPropagator` & `AdaptiveScalePropagator`. |
| **1. Thay đổi thuộc riêng `TICKET-OPT-03` (Task 08)** | [`scripts/benchmark/sg_vcd/pipeline.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py) | `+53 / -11` dòng *(trên tổng `+138 / -16` so với `HEAD`)* | Bổ sung `_is_valid_working_maps` import, `self._cached_working_maps`, `self._cached_working_maps_key`, xóa trong `reset_cache()`, và lưu đệm working-map tại Stage 1.3 (`lines 875–930`). |
| **1. Thay đổi thuộc riêng `TICKET-OPT-03` (Task 08)** | [`tests/test_opt03_working_map_cache.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt03_working_map_cache.py) | Tệp mới (`677` dòng) | Bộ kiểm thử đơn vị và tích hợp mock Tests A–H cho `OPT-03`. |
| **1. Thay đổi thuộc riêng `TICKET-OPT-03` (Task 08)** | [`docs/research/TASK_08_OPT03_WORKING_MAP_CACHE_IMPLEMENTATION_REPORT.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/TASK_08_OPT03_WORKING_MAP_CACHE_IMPLEMENTATION_REPORT.md) | Tệp mới (`304` dòng) | Báo cáo triển khai Task 08 của `4argon`. |
| **2. Thay đổi từ `TICKET-OPT-06` (Task 07 & 07-FIX đã nghiệm thu)** | `feature_cache.py`, `evaluate_vitextvqa.py`, `attention_extractor.py`, `__init__.py`, phần `OPT-06` trong `pipeline.py`, `tests/test_opt06_image_cache_identity.py` | `+683 / -65` dòng (từ Task 07-FIX) | Không bị sửa đổi thêm trong Task 08 (ngoại trừ phần `OPT-03` nối tiếp vào `pipeline.py`). |
| **3. Tệp quản trị / tài liệu nghiên cứu không liên quan** | `.github/`, `docs/SG_VCD_Research_Review_Consolidated_Draft.md`, các báo cáo Task 01–07 trong `docs/research/` | Untracked (`??`) | Giữ nguyên trạng tuyệt đối. |

- **Xác nhận tuân thủ giới hạn phạm vi:** `OPT-02` (Coordinated Memory Cleanup Policy), `OPT-01` (Batched Positive/Negative Decoding), và `OPT-04` (GPU Affinity Field Vectorization) hoàn toàn **chưa được triển khai** trong Task 08 (lệnh `gc.collect()` và `torch.cuda.empty_cache()` cuối `pipeline.run` tại `lines 1044–1046` vẫn giữ nguyên cho đến khi `OPT-02` được cấp phép).

---

## 11. Danh sách Phát hiện Chặn và Không Chặn (Blocking and Non-Blocking Findings)

### 11.1. Phát hiện chặn nghiệm thu (Blocking Findings)
- **Không có (`0` Blocking Findings).**

### 11.2. Phát hiện không chặn nghiệm thu (Non-Blocking Findings)

| Mã phát hiện | Mức độ (Severity) | Vị trí tệp & dòng | Bằng chứng cụ thể (Evidence) | Tác động (Impact) | Khuyến nghị xử lý (Recommended Action) |
| :-: | :-: | :--- | :--- | :--- | :--- |
| **`NBF-01`** | **`MEDIUM`** | [`docs/research/TASK_08_OPT03_WORKING_MAP_CACHE_IMPLEMENTATION_REPORT.md#L20`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/TASK_08_OPT03_WORKING_MAP_CACHE_IMPLEMENTATION_REPORT.md#L20), [`#L237-L239`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/TASK_08_OPT03_WORKING_MAP_CACHE_IMPLEMENTATION_REPORT.md#L237-L239) | Báo cáo ghi tập dữ liệu ViTextVQA có `1,324` câu hỏi ($\sim 3.31$ QA/ảnh), `924` cache hits và giảm `69.8%` số lần gọi `_compute_working_maps`. Kiểm toán trực tiếp [`answer_bboxes_400_verified.json`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/results/swintextspotter_auto_annot/answer_bboxes_400_verified.json) xác nhận tập chuẩn có **400 ảnh / 1.222 câu hỏi** (`3.055` QA/ảnh), tương ứng giảm từ **1.222 xuống 400 lần gọi**, đạt **822 cache hits** và giảm **`67.27%` (`67.2668%`)**. | Không ảnh hưởng mã nguồn, nhưng gây sai lệch số liệu thống kê nếu trích dẫn trực tiếp báo cáo triển khai của `4argon` mà không kèm báo cáo QA này. | Sử dụng số liệu đã đính chính trong Mục 7 của báo cáo QA này (`400` ảnh, `1,222` QA, `822` hits, `67.27%`) làm số liệu chính thức của dự án. |
| **`NBF-02`** | **`MEDIUM`** | [`docs/research/TASK_08_OPT03_WORKING_MAP_CACHE_IMPLEMENTATION_REPORT.md#L64-L88`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/TASK_08_OPT03_WORKING_MAP_CACHE_IMPLEMENTATION_REPORT.md#L64-L88), [`#L209-L223`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/TASK_08_OPT03_WORKING_MAP_CACHE_IMPLEMENTATION_REPORT.md#L209-L223), [`#L266`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/TASK_08_OPT03_WORKING_MAP_CACHE_IMPLEMENTATION_REPORT.md#L266) | Văn bản báo cáo của `4argon` mô tả sai một số chi tiết so với mã thực tế đã viết:<br>1. Ghi `test_E` kiểm tra các trường không tồn tại trên `PipelineResult` (`seed_boxes`, `propagated_boxes`, `grid_dims`, `num_patches`, `selected_k`, `answer`, `raw_answer`, `num_steps`, `avg_js_divergence`) và ghi `test_E` có `np.array_equal` trên `roi_tiles` (thực tế `test_E` kiểm tra toàn bộ các trường thật của `SeedPropagationResult` và `PipelineResult` trên 4 câu hỏi `Q1..Q4`, còn `np.array_equal` trên `roi_tiles`/`neg_tiles` được Anti QA kiểm chứng bổ sung bằng probe).<br>2. Ghi `test_F` kiểm tra `data_ptr()` và đếm `clone == 0` (thực tế `test_F` kiểm tra `assertIs` và `torch.equal`).<br>3. Ghi `opp_t` có shape `(3, H_w, W_w)` (thực tế là `(H_w, W_w, 3)`), ghi `propagate_multi` nhận `boxes_and_masks, m_aff_map, a_q_map` (thực tế nhận `seed_patches, grid_ratio, working_maps`), và ghi file test dài `495` dòng (thực tế dài `677` dòng). | Chỉ là lỗi thuyết minh trong báo cáo; mã thực tế trong `adaptive_propagator.py`, `pipeline.py` và `tests/test_opt03_working_map_cache.py` hoạt động đúng. | Ghi nhận biên bản đối chiếu tại Mục 6 của báo cáo QA này; tùy chọn bổ sung thêm 2 dòng `assertTrue(np.array_equal(...))` cho `roi_tiles` vào `test_E` nếu Project Owner yêu cầu chỉnh sửa sau này. |
| **`NBF-03`** | **`LOW`** | [`scripts/benchmark/sg_vcd/adaptive_propagator.py#L150-L183`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/adaptive_propagator.py#L150-L183) | `_is_valid_working_maps` chưa kiểm tra tường minh `opp_t.shape[2] == 3`, `bg_smooth_t.shape[2] == 3`, kích thước không gian `> 0`, `not isinstance(scale_x, bool)`, `math.isfinite`, và cùng `device` giữa 4 tensor. | Không xảy ra lỗi khi dùng `_compute_working_maps` nội bộ, nhưng có thể lọt qua nếu người dùng bên ngoài truyền tuple `working_maps` lỗi cấu trúc kênh/thiết bị. | Tùy chọn làm chặt thêm trong đợt bảo trì sau nếu cần mở rộng giao diện `working_maps` cho bên thứ ba. |
| **`NBF-04`** | **`LOW`** | [`scripts/benchmark/sg_vcd/pipeline.py#L675-L712`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L675-L712) | Phương thức phụ trợ `SGVCDPipeline.run_stage1_only` gọi trực tiếp `self.propagator.propagate(...)` không dùng `_cached_working_maps`. | Không ảnh hưởng `SGVCDPipeline.run()` hay `evaluate_vitextvqa.py` (vốn luôn gọi `pipeline.run()`). | Không bắt buộc sửa trong phạm vi `OPT-03`. |

---

## 12. Phán quyết Cuối cùng (Final Verdict)

> ### **PHÁN QUYẾT CHẤP THUẬN KỸ THUẬT: `PASS WITH NON-BLOCKING FINDINGS`**

- **Đánh giá chấp thuận kỹ thuật (Technical Acceptability Decision):**
  - Bản triển khai mã nguồn của **`TICKET-OPT-03 — Image-Level Working-Map Caching`** trên [`scripts/benchmark/sg_vcd/adaptive_propagator.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/adaptive_propagator.py), [`scripts/benchmark/sg_vcd/pipeline.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py), và [`tests/test_opt03_working_map_cache.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt03_working_map_cache.py) **ĐẠT YÊU CẦU KỸ THUẬT (`TECHNICALLY ACCEPTABLE`)** và không cần sửa lại mã nguồn trước khi chuyển sang ticket tiếp theo.
  - Các sai lệch về số liệu tập dữ liệu (`1,222` QA vs `1,324` QA; `822` hits / `67.27%` vs `924` hits / `69.8%`) và mô tả văn bản trong báo cáo của `4argon` đã được đính chính đầy đủ và có bằng chứng xác thực tại báo cáo QA độc lập này.
  - Cải thiện thời gian chạy thực tế (runtime latency) và bộ nhớ GPU (VRAM) trên mô hình thật `5CD-AI/Vintern-1B-v3_5` được ghi nhận chính thức ở trạng thái **`NOT_MEASURED`** cho đến khi thực hiện phép đo chuẩn hóa trên Google Colab.

---

## 13. Trạng thái Sẵn sàng cho Gate B (Gate B Readiness)

- **Trạng thái Gate B (Google Colab 20-QA Real-Model Integration):** **`NOT AUTHORIZED`**
- **Các điều kiện phụ thuộc bắt buộc chưa hoàn tất trước khi được phép chạy Gate B:**
  1. **Quyết định phê duyệt của Project Owner** đối với kết quả nghiệm thu `TICKET-OPT-03` và cấp phép triển khai các ticket tối ưu hóa còn lại theo lộ trình (`TICKET-OPT-02`, `TICKET-OPT-01`, `TICKET-OPT-04`).
  2. **Đóng băng danh sách mẫu tích hợp `20 QA` (`Frozen 20-QA Manifest`)** và xác nhận quy trình/notebook chạy trên Google Colab theo đúng kỹ năng `local-colab-experiment-workflow`.
  3. **Phê duyệt mở cổng thực thi (Explicit Execution Gate Approval)** từ Project Owner trước khi khởi chạy bất kỳ tác vụ suy luận mô hình thật nào trên Google Colab.

