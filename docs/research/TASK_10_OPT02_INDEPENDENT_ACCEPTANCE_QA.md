# BÁO CÁO NGHIỆM THU ĐỘC LẬP (INDEPENDENT ACCEPTANCE QA) — `TICKET-OPT-02`: COORDINATED MEMORY CLEANUP POLICY (`TASK 10`)

- **Mã tài liệu:** `TASK_10_OPT02_INDEPENDENT_ACCEPTANCE_QA`
- **Ngày thực hiện:** 2026-11-08
- **Người kiểm định độc lập (Independent QA Reviewer):** Antigravity (`Anti`)
- **Tác nhân triển khai được kiểm định:** `4argon` (`TASK 09`)
- **Cấp thẩm định phương pháp luận & kiến trúc:** ChatGPT
- **Quyền quyết định cuối cùng:** Project Owner
- **Chế độ thực thi:** `STRICT READ-ONLY` (Chỉ kiểm tra mã nguồn, diff, chạy kiểm thử độc lập và lập báo cáo QA; không chỉnh sửa mã nguồn, kiểm thử, cấu hình hay báo cáo triển khai của `4argon`)
- **Đầu vào được kiểm định:** [`docs/research/TASK_09_OPT02_MEMORY_CLEANUP_IMPLEMENTATION_REPORT.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/TASK_09_OPT02_MEMORY_CLEANUP_IMPLEMENTATION_REPORT.md)

---

## Tóm tắt Điều hành & Phán quyết Nghiệm thu (Executive Verdict)

- **Phán quyết QA Kỹ thuật:** **`PASS WITH NON-BLOCKING FINDINGS`**
- **Trạng thái Gate A (Local Synthetic / Mock Verification):** **`PASS`** (`9/9` bài kiểm thử trong `tests.test_opt02_memory_cleanup_policy` đạt `OK` trong `10.862s`; `74/74` bài kiểm thử hồi quy đạt `OK` trong `8.617s`; probe độc lập về `weakref` xác nhận giải phóng 100% các tensor lưu đệm theo ảnh).
- **Trạng thái Gate B (20-QA Real-Model Validation Slice):** **`NOT AUTHORIZED / NOT EXECUTED`** (Chưa được thực hiện và chưa được Project Owner cấp phép).

### Tóm tắt kết luận kỹ thuật
1. **Triển khai mã nguồn (`pipeline.py`, `attention_extractor.py`, `evaluate_vitextvqa.py`) và bộ kiểm thử (`tests/test_opt02_memory_cleanup_policy.py`) đáp ứng đầy đủ các yêu cầu cốt lõi của `TICKET-OPT-02` (`0` lỗi BLOCKER):**
   - Đã loại bỏ hoàn toàn các lời gọi `gc.collect()` và `torch.cuda.empty_cache()` theo từng câu hỏi (per-question) khỏi [`SGVCDPipeline.run()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L1050-L1056) và [`PointwiseKLAttentionExtractor.extract()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/attention_extractor.py#L782-L785), không còn helper ẩn nào gọi global cleanup theo từng QA.
   - Đã chuyển việc dọn bộ nhớ sang ranh giới ảnh (image boundary) trong [`run_benchmark()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/evaluate_vitextvqa.py#L1260-L1443) thông qua khối `try ... finally` và hàm [`run_image_boundary_cleanup()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/evaluate_vitextvqa.py#L1027-L1060).
   - Đã cài đặt đúng công thức CUDA memory watchdog tại [`check_and_run_cuda_memory_watchdog()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/evaluate_vitextvqa.py#L933-L1024): chỉ gọi `torch.cuda.empty_cache()` khi `torch.cuda.memory_allocated(dev_idx) / total_bytes > 0.85`, an toàn với chế độ CPU-only (`torch.cuda.is_available() == False` hoặc `device="cpu"`) và các lỗi truy vấn CUDA.
   - Đã cập nhật [`SGVCDPipeline.reset_cache()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L360-L383) để giải phóng `self.feature_cache._cache` mà không kích hoạt `torch.cuda.empty_cache()` vô điều kiện nằm trong [`VisualFeatureCache.clear()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/feature_cache.py#L522-L530).
   - Giữ nguyên 100% công thức toán học Stage 1 / Stage 2 của SG-VCD, bảo toàn tái sử dụng cache `OPT-06` và `OPT-03` xuyên suốt mọi câu hỏi của cùng một ảnh, và không chạm sang `OPT-01` hay `OPT-04`.
2. **Các phát hiện không chặn nghiệm thu (`Non-Blocking Findings`) cần lưu ý:**
   - **`NBF-01` (NON-BLOCKING — Sai lệch mô tả cấu trúc tuple `_compute_working_maps` và tên hàm trong báo cáo của `4argon` và chú thích `test_H`):** Báo cáo [`TASK_09_OPT02_MEMORY_CLEANUP_IMPLEMENTATION_REPORT.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/TASK_09_OPT02_MEMORY_CLEANUP_IMPLEMENTATION_REPORT.md) (Mục 1, 6, 8, 13.2) và chú thích tại [`tests/test_opt02_memory_cleanup_policy.py#L882`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt02_memory_cleanup_policy.py#L882) mô tả `_compute_working_maps` trả về `(img_work, L, C_rg, C_yb, W_w, H_w, sx, sy)` (mảng NumPy `L`) và gọi tên hàm là `propagate_multi_with_maps`. Trong mã nguồn thực tế ([`adaptive_propagator.py#L388`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/adaptive_propagator.py#L388) và [`pipeline.py#L906`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L906)), `_compute_working_maps` trả về bộ 6 phần tử `(opp, bg_smooth, stroke_energy, horiz_mod, scale_x, scale_y)` (trong đó `maps[1]` là `bg_smooth: torch.Tensor`), và hàm được gọi là `propagate_multi(..., working_maps=working_maps)`.
   - **`NBF-02` (NON-BLOCKING — Đồng nhất thuật ngữ "call count" của `torch.cuda.empty_cache()` với "host-device synchronization" trong báo cáo):** Mục 1 (`dòng 22`) và Mục 4 (`dòng 86`) của báo cáo triển khai gọi `2,844` / `1,222` lần gọi `torch.cuda.empty_cache()` là `"host-device synchronizations"`. Số lần gọi hàm (`2,844` calls) được xác minh từ mã nguồn và luồng điều khiển, nhưng số lần đồng bộ hóa phần cứng thực tế phụ thuộc vào trạng thái các khối bộ nhớ đệm trong CUDA caching allocator trên GPU thật (chưa đo đạc ở Gate A).
   - **`NBF-03` (NON-BLOCKING — Edge case nhỏ trong `_resolve_cuda_device_index` và `run_image_boundary_cleanup`):** Trong [`_resolve_cuda_device_index()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/evaluate_vitextvqa.py#L910-L922), `isinstance(device, int)` chưa loại trừ `bool` (`isinstance(True, int) == True`), và chuỗi `"cuda:invalid"` rơi xuống `torch.cuda.current_device()` thay vì báo lỗi định dạng. Trong [`run_image_boundary_cleanup()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/evaluate_vitextvqa.py#L1050-L1055), `gc.collect()` không bọc `try/except` và `log_fn` bên trong watchdog không bọc `try/except` nếu hàm ghi log tùy chỉnh ném ngoại lệ. Không ảnh hưởng đến đường chạy chuẩn của `evaluate_vitextvqa.py`.

---

## 1. Các Tài liệu Governance Đã Đọc (Governance Files Read)

Trước khi kiểm tra kho mã nguồn và chạy kiểm thử, toàn bộ các tài liệu quản trị và kỹ thuật bắt buộc tại Mục 2 đã được kiểm tra và đọc đầy đủ:

| STT | Tài liệu | Trạng thái | Ghi chú kiểm chứng |
| :-: | :--- | :-: | :--- |
| 1 | `AGENTS.md` | Không tồn tại (`Test-Path AGENTS.md == False`) | Đã kiểm tra tại thư mục gốc; quy tắc quản trị nằm tại `.github/copilot-instructions.md` và `PROJECT.md`. |
| 2 | [`.github/copilot-instructions.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/.github/copilot-instructions.md) | Đã đọc (`lines 1–43`) | Tuân thủ bất biến nghiên cứu, không chạy mô hình thật ở local, kiểm định độc lập không sửa mã nguồn. |
| 3 | [`.github/skills/anti-output-quality-review/SKILL.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/.github/skills/anti-output-quality-review/SKILL.md) | Đã đọc (`lines 1–40`) | Quy trình QA độc lập: đối chiếu diff, truy vết call path, tự chạy kiểm thử và phân loại mức độ phát hiện. |
| 4 | [`.github/skills/research-plan-compliance-review/SKILL.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/.github/skills/research-plan-compliance-review/SKILL.md) | Đã đọc (`lines 1–36`) | Đảm bảo không thay đổi thuật toán SG-VCD, CERV, bộ chỉ số Strict hay tập dữ liệu 400 ảnh / 1.222 QA. |
| 5 | [`.github/skills/local-colab-experiment-workflow/SKILL.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/.github/skills/local-colab-experiment-workflow/SKILL.md) | Đã đọc (`lines 1–36`) | Tuân thủ nghiêm ngặt giới hạn local synthetic/mock; Gate B yêu cầu phê duyệt riêng của Project Owner. |
| 6 | [`docs/research/PERFORMANCE_OPTIMIZATION_HANDOFF_v1.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/PERFORMANCE_OPTIMIZATION_HANDOFF_v1.md#L109-L123) | Đã đọc (`lines 80–170`) | Đối chiếu trực tiếp với đặc tả gốc của Ticket 3: `TICKET-OPT-02 — Coordinated Memory Cleanup Policy`. |
| 7 | [`docs/research/TASK_07_FIX_OPT06_INDEPENDENT_ACCEPTANCE_QA.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/TASK_07_FIX_OPT06_INDEPENDENT_ACCEPTANCE_QA.md) | Đã đọc (`lines 1–90`) | Xác nhận các bất biến của `OPT-06` (`ImageCacheIdentity`, `_sample_image_group_key`). |
| 8 | [`docs/research/TASK_08_OPT03_INDEPENDENT_ACCEPTANCE_QA.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/TASK_08_OPT03_INDEPENDENT_ACCEPTANCE_QA.md) | Đã đọc (`lines 1–90`) | Xác nhận các bất biến của `OPT-03` (`_cached_working_maps`, `_cached_working_maps_key`, số liệu 400 ảnh / 1.222 QA). |
| 9 | [`docs/research/REMEDIATION_SPEC_P0_v1.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/REMEDIATION_SPEC_P0_v1.md) | Đã đọc (`lines 1–90`) | Kiểm tra sự bảo toàn các quy tắc P0. |
| 10 | [`docs/Proposed_Method_SG_VCD.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/Proposed_Method_SG_VCD.md) | Đã đọc (`lines 1–90`) | Kiểm tra sự bảo toàn kiến trúc 2 tầng của SG-VCD. |

---

## 2. Git Status / Diff và Phạm vi File Được Audit (Git Status / Diff & Audited Scope)

### 2.1. Kết quả `git status --short` và `git diff --stat`
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
?? tests/test_opt02_memory_cleanup_policy.py
?? tests/test_opt03_working_map_cache.py
?? tests/test_opt06_image_cache_identity.py
```

### 2.2. Phân tách chính xác phạm vi thay đổi của `TICKET-OPT-02`
Do working tree đang giữ các thay đổi chưa commit của `OPT-06` và `OPT-03` theo quy định dự án, kiểm định viên đã đối chiếu từng vùng dòng thay đổi để xác nhận phần thuộc `TICKET-OPT-02`:

| Tệp tin | Vùng dòng thuộc `OPT-02` | Nội dung thay đổi thực tế | Đánh giá phạm vi |
| :--- | :---: | :--- | :---: |
| [`scripts/benchmark/sg_vcd/pipeline.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py) | `lines 360–383`<br>`lines 1050–1056` | • Cập nhật `SGVCDPipeline.reset_cache()` để xóa `self.feature_cache._cache` mà không gọi `empty_cache()` vô điều kiện.<br>• Xóa `gc.collect()` và `torch.cuda.empty_cache()` ở cuối `SGVCDPipeline.run()`. | **Đúng phạm vi `OPT-02`** |
| [`scripts/benchmark/sg_vcd/attention_extractor.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/attention_extractor.py) | `lines 782–785` | • Xóa `if torch.cuda.is_available(): torch.cuda.empty_cache()` sau `del suffix_q` trong `PointwiseKLAttentionExtractor.extract()`. | **Đúng phạm vi `OPT-02`** |
| [`scripts/benchmark/evaluate_vitextvqa.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/evaluate_vitextvqa.py) | `lines 900–1061`<br>`lines 1208–1215`<br>`lines 1260–1443` | • Thêm `CUDA_MEMORY_WATCHDOG_THRESHOLD = 0.85`, `_resolve_cuda_device_index()`, `check_and_run_cuda_memory_watchdog()`, `run_image_boundary_cleanup()`.<br>• Xóa `default_feature_cache._cache` khi khởi tạo không gọi `empty_cache()` vô điều kiện.<br>• Đặt vòng lặp câu hỏi của từng ảnh vào khối `try ... finally` và gọi `run_image_boundary_cleanup()` tại ranh giới ảnh. | **Đúng phạm vi `OPT-02`** |
| [`tests/test_opt02_memory_cleanup_policy.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt02_memory_cleanup_policy.py) | `lines 1–1036` | • Bộ kiểm thử đơn vị và tích hợp mock mới cho `OPT-02` (`Tests A–I`). | **Đúng phạm vi `OPT-02`** |
| [`docs/research/TASK_09_OPT02_MEMORY_CLEANUP_IMPLEMENTATION_REPORT.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/TASK_09_OPT02_MEMORY_CLEANUP_IMPLEMENTATION_REPORT.md) | `lines 1–312` | • Báo cáo triển khai `TASK 09` của `4argon`. | **Đúng phạm vi `OPT-02`** |

- **Xác nhận không xâm phạm ngoài phạm vi:** Không có bất kỳ thay đổi nào đối với công thức toán học Stage 1 / Stage 2, giải mã tương phản (`dynamic_cd.py`), cắt ROI (`roi_cropper.py`), chọn hộp (`bbox_selector.py`), chỉ số đánh giá (`metrics.py`), hay dữ liệu annotation. Không có mã lệnh nào của `OPT-01` (ghép batch $B=2$) hoặc `OPT-04` (vector hóa trường ái lực trên GPU) được đưa vào.

---

## 3. Kết Quả Kiểm Định Mã Nguồn Chi Tiết Theo Hạng Mục A–G (Source-Level Findings A–G)

### Hạng mục A — Diff và Scope
- **Trạng thái:** **ĐẠT (`PASS`)**.
- Toàn bộ thay đổi thuộc đúng phạm vi `TICKET-OPT-02`. Không có thao tác Git phá hủy hoặc commit/push trái phép.

### Hạng mục B — Loại bỏ Per-Question Cleanup
- **Trạng thái:** **ĐẠT (`PASS`) — Verified by Source & AST/Mock Tests**.
- Kiểm tra trực tiếp mã nguồn và quét toàn bộ thư mục `scripts/benchmark/`:
  1. Trong [`SGVCDPipeline.run()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L776-L1072): Đã loại bỏ hoàn toàn `gc.collect()` và `torch.cuda.empty_cache()` tại dòng `1050–1056`. Toàn bộ các hàm phụ được gọi trong `run()` (`build_image_cache_identity`, `extract_visual_features`, `extractor.extract`, `selector.select_top_k`, `propagator._compute_working_maps`, `propagator.propagate_multi`, `_fuse_seed_propagation`, `cropper.crop`, `_decode_spatial_pmi_contrastive`, `_decode_with_tiles`, `strip_conversational_preamble`) đều không chứa bất kỳ lời gọi `gc.collect()` hay `torch.cuda.empty_cache()` nào.
  2. Trong [`PointwiseKLAttentionExtractor.extract()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/attention_extractor.py#L665-L799): Đã loại bỏ hoàn toàn `if torch.cuda.is_available(): torch.cuda.empty_cache()` tại dòng `782–785`. Các hàm phụ được gọi trong `extract()` (`_make_image_key`, `dynamic_preprocess_grid`, `_build_prefix_and_suffix_embeds`, `_compute_paired_visual_attentions`, `_rollout_reading_trajectory`, `_clone_kv_cache`, `compute_pointwise_kl_map`, `compute_intrinsic_affinity_field`) đều không chứa `gc.collect()` hay `empty_cache()`.
  3. **Bảo toàn giải phóng biến cục bộ và khôi phục trạng thái:**
     - Trong [`attention_extractor.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/attention_extractor.py): Giữ nguyên `del kv_draft, out` (`dòng 508`), khối `try ... finally: self._set_attn_implementation(prev_mode)` (`dòng 543–563`), `del att_q_list, out_q, kv_q` và `del att_g_list, out_g, kv_g` (`dòng 660–662`), `del pixel_values` (`dòng 726`), `del prefix_embeds, prefix_out` (`dòng 743`), và `del suffix_q` (`dòng 782`).
     - Trong [`pipeline.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py): Giữ nguyên `del thumb_pv` (`dòng 502`), `del all_roi_pv` (`dòng 511`), `del ctx_pv` (`dòng 521`), `del all_roi_vit` (`dòng 530`), `del pos_out, pos_inputs` (`dòng 558`), `del neg_out, neg_inputs` (`dòng 567`), `del pos_step, neg_step` (`dòng 653`), `del (pos_vit, neg_vit, pos_prefix, neg_prefix, pos_kv, neg_kv)` (`dòng 677`), và `if cached_thumb_vit is None: del thumb_vit` (`dòng 678–679`).
     - Thứ tự tính toán và các tham chiếu giữa Stage 1 và Stage 2 được giữ nguyên tuyệt đối.

### Hạng mục C — `SGVCDPipeline.reset_cache()` và Lifecycle của Cache
- **Trạng thái:** **ĐẠT (`PASS`) — Verified by Source & Independent Weakref Probe**.
- Chi tiết kiểm định được trình bày tại **Mục 6** bên dưới.

### Hạng mục D — Image-Boundary Cleanup và Exception Safety
- **Trạng thái:** **ĐẠT (`PASS`) — Verified by Source & Control-Flow Analysis**.
- Chi tiết kiểm định được trình bày tại **Mục 7** bên dưới.

### Hạng mục E — CUDA Watchdog (`0.85` Threshold)
- **Trạng thái:** **ĐẠT (`PASS`) — Verified by Source & Mock Tests**.
- Chi tiết kiểm định được trình bày tại **Mục 8** bên dưới.

### Hạng mục F — Tái Chạy Độc lập Test Suite
- **Trạng thái:** **ĐẠT (`PASS`) — Verified by Independent Execution**.
- Chi tiết lệnh chạy và đầu ra thực tế được trình bày tại **Mục 4** và **Mục 5** bên dưới.

### Hạng mục G — Kiểm tra Tính Chính xác của Báo cáo Triển khai (`TASK 09`)
- **Trạng thái:** **ĐẠT VỚI LƯU Ý KHÔNG CHẶN (`PASS WITH NON-BLOCKING FINDINGS`: `NBF-01`, `NBF-02`)**.
- Chi tiết đối chiếu số liệu và các điểm sai lệch/overclaim trong báo cáo `TASK 09` được trình bày tại **Mục 9** bên dưới.

---

## 4. Các Lệnh Đã Thực Thi Độc Lập và Kết Quả Thực Tế (Executed Commands & Actual Outcomes)

Toàn bộ các lệnh kiểm định được thực thi độc lập bởi `Antigravity` bằng môi trường ảo Python của kho mã nguồn:
- **Interpreter thực tế:** `D:\Trường Đại Học Công Thương\Năm 4_HUIT\Các vấn đề hiện đại trong AI\Document AI\VLM_Reasoning\venv\Scripts\python.exe`

| STT | Lệnh thực thi độc lập | Kết quả | Thời gian | Exit Code |
| :-: | :--- | :---: | :---: | :-: |
| 1 | `& "..\Document AI\VLM_Reasoning\venv\Scripts\python.exe" -m unittest tests.test_opt02_memory_cleanup_policy -v` | **9 / 9 PASS (`OK`)** | `10.862s` | `0` |
| 2 | `& "..\Document AI\VLM_Reasoning\venv\Scripts\python.exe" -m unittest tests.test_opt03_working_map_cache tests.test_opt06_image_cache_identity tests.test_sg_vcd tests.test_adversarial_spatial_processing tests.test_tier4_workloads.TestTier4RealWorldWorkloads.test_05_anti_hardcoding_and_heuristics_audit -v` | **74 / 74 PASS (`OK`)** | `8.617s` | `0` |
| 3 | Independent Weakref & Cache Ownership Probe (`python -c "..."`) | **PASS (`OK`)** | `< 1.0s` | `0` |

### Đầu ra thực tế của Lệnh 1 (`tests.test_opt02_memory_cleanup_policy`)
```text
test_A_no_global_cleanup_per_question (tests.test_opt02_memory_cleanup_policy.TestOPT02MemoryCleanupPolicy.test_A_no_global_cleanup_per_question)
Test A — No global cleanup per question: ... ok
test_B_cleanup_at_image_boundary (tests.test_opt02_memory_cleanup_policy.TestOPT02MemoryCleanupPolicy.test_B_cleanup_at_image_boundary)
Test B — Cleanup at image boundary: ... ok
test_C_watchdog_below_threshold (tests.test_opt02_memory_cleanup_policy.TestOPT02MemoryCleanupPolicy.test_C_watchdog_below_threshold)
Test C — Watchdog below threshold: ... ok
test_D_watchdog_above_threshold (tests.test_opt02_memory_cleanup_policy.TestOPT02MemoryCleanupPolicy.test_D_watchdog_above_threshold)
Test D — Watchdog above threshold: ... ok
test_E_cpu_only_compatibility (tests.test_opt02_memory_cleanup_policy.TestOPT02MemoryCleanupPolicy.test_E_cpu_only_compatibility)
Test E — CPU-only compatibility: ... ok
test_F_opt03_integration_preservation (tests.test_opt02_memory_cleanup_policy.TestOPT02MemoryCleanupPolicy.test_F_opt03_integration_preservation)
Test F — OPT-03 integration preservation: ... ok
test_G_image_transition_and_exception_safety (tests.test_opt02_memory_cleanup_policy.TestOPT02MemoryCleanupPolicy.test_G_image_transition_and_exception_safety)
Test G — Image transition and exception safety: ... ok
test_H_synthetic_20_qa_workload (tests.test_opt02_memory_cleanup_policy.TestOPT02MemoryCleanupPolicy.test_H_synthetic_20_qa_workload)
Test H — Synthetic 20-QA workload (Gate A): ... ok
test_I_ast_verification_no_per_question_cleanup_in_pipeline_or_extractor (tests.test_opt02_memory_cleanup_policy.TestOPT02MemoryCleanupPolicy.test_I_ast_verification_no_per_question_cleanup_in_pipeline_or_extractor)
Test I — AST verification: ... ok

----------------------------------------------------------------------
Ran 9 tests in 10.862s

OK
```

### Đầu ra thực tế của Lệnh 3 (Independent Weakref & Cache Ownership Probe)
```text
Interpreter: D:\Trường Đại Học Công Thương\Năm 4_HUIT\Các vấn đề hiện đại trong AI\Document AI\VLM_Reasoning\venv\Scripts\python.exe
After cleanup weakrefs alive: {'feature_cache_tensor': False, 'working_map_opp': False, 'working_map_bg': False, 'extractor_vit': False, 'extractor_kv': False, 'model_weight_preserved': True}
```

---

## 5. Bảng Kiểm Chứng Chi Tiết Các Bài Kiểm Thử A–I (Test Matrix Verification A–I)

| Mã Test | Hàm kiểm thử trong [`tests/test_opt02_memory_cleanup_policy.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt02_memory_cleanup_policy.py) | Kỳ vọng (Expected) | Quan sát thực tế từ mã test & kết quả chạy (Observed) | Trạng thái |
| :---: | :--- | :--- | :--- | :---: |
| **Test A** | [`test_A_no_global_cleanup_per_question`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt02_memory_cleanup_policy.py#L315-L378) | `0` lần gọi `gc.collect()` và `0` lần gọi `torch.cuda.empty_cache()` khi chạy nhiều câu hỏi qua `extractor.extract()` và `pipeline.run()`. | Mock `gc.collect`, `torch.cuda.is_available=True`, `torch.cuda.empty_cache`; chạy 4 QA trực tiếp qua `PointwiseKLAttentionExtractor.extract()` và 4 QA qua `SGVCDPipeline.run()`. Cả hai bộ đếm đều bằng `0`. | **PASS** |
| **Test B** | [`test_B_cleanup_at_image_boundary`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt02_memory_cleanup_policy.py#L383-L469) | Cleanup chỉ chạy 1 lần sau khi toàn bộ câu hỏi của ảnh hoàn tất. | Chạy 3 QA của 1 ảnh qua `run_benchmark()`; `timeline` ghi nhận đúng thứ tự: `question_completed` (Q1) $\rightarrow$ `question_completed` (Q2) $\rightarrow$ `question_completed` (Q3) $\rightarrow$ `image_boundary_cleanup`. | **PASS** |
| **Test C** | [`test_C_watchdog_below_threshold`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt02_memory_cleanup_policy.py#L474-L512) | Khi `allocated / total <= 0.85` (bao gồm `0.50` và biên `0.85`), không gọi `empty_cache()`. | Với `total_bytes = 4000 * 1024 * 1024`, kiểm tra tại `ratio = 0.50` và đúng biên `ratio = 0.85` (`3400.0MB/4000.0MB`): `mock_empty_cache.call_count == 0`, `triggered_empty_cache == False`, `status == "below_threshold"`. | **PASS** |
| **Test D** | [`test_D_watchdog_above_threshold`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt02_memory_cleanup_policy.py#L517-L551) | Khi `allocated / total > 0.85`, gọi `empty_cache()` đúng 1 lần mỗi lần kiểm tra. | Kiểm tra tại `ratio = 0.86` và `ratio = 0.92`: `mock_empty_cache.call_count == 1`, `triggered_empty_cache == True`, `status == "triggered"`. | **PASS** |
| **Test E** | [`test_E_cpu_only_compatibility`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt02_memory_cleanup_policy.py#L556-L615) | An toàn trên CPU-only, `device="cpu"`, lỗi `RuntimeError` khi truy vấn CUDA, và `total_memory == 0`. | Kiểm tra cả 4 nhánh: `is_available() == False`, `device="cpu"` khi `is_available() == True`, `memory_allocated` ném `RuntimeError`, và `total_memory == 0`. Không gọi `empty_cache()`, không crash. | **PASS** |
| **Test F** | [`test_F_opt03_integration_preservation`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt02_memory_cleanup_policy.py#L620-L673) | Giữ nguyên cache `OPT-03` và `OPT-06` xuyên suốt các câu hỏi cùng ảnh; xóa sạch khi gọi `run_image_boundary_cleanup()`. | Chạy 4 câu hỏi trên cùng `img_key`: `spy_compute_maps.call_count == 1` (1 miss, 3 hits); sau khi gọi `run_image_boundary_cleanup()`, `_cached_working_maps` là `None`, `_cached_working_maps_key` là `None`, `len(feature_cache._cache) == 0`. | **PASS** |
| **Test G** | [`test_G_image_transition_and_exception_safety`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt02_memory_cleanup_policy.py#L678-L795) | Khi 1 QA lỗi giữa chừng, ghi nhận bản ghi lỗi, các QA còn lại của ảnh đó vẫn dùng cache, cleanup chạy 1 lần mỗi ảnh, và ảnh kế tiếp chạy bình thường. | Chạy 2 nhóm ảnh (`3` QA + `2` QA) qua `run_benchmark()`, trong đó Q2 ném `RuntimeError`: Q2 được ghi vào `checkpoint.json` với `pred_answer=""`, Q3 vẫn chạy và tái sử dụng working maps của ảnh 1 (`spy_compute_maps.call_count == 2`), `spy_cleanup.call_count == 2`, tổng cộng `5` bản ghi trong `checkpoint.json`. | **PASS** |
| **Test H** | [`test_H_synthetic_20_qa_workload`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt02_memory_cleanup_policy.py#L800-L1014) | Chạy thực tế qua `run_benchmark()` với 20 QA trên 6 ảnh; kiểm tra số lần cleanup, watchdog triggers, cache hits, output parity và weakref. | Xem phân tích chuyên sâu tại mục ngay bên dưới (**PASS**). | **PASS** |
| **Test I** | [`test_I_ast_verification_no_per_question_cleanup_in_pipeline_or_extractor`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt02_memory_cleanup_policy.py#L1019-L1040) | Phân tích cây cú pháp AST của `SGVCDPipeline.run` và `PointwiseKLAttentionExtractor.extract` không có `collect` hoặc `empty_cache`. | `ast.walk()` xác nhận `forbidden_calls == []` trên cả hai phương thức. | **PASS** |

### Phân tích chuyên sâu về `Test H` (`test_H_synthetic_20_qa_workload`)
1. **Xác nhận đường chạy thực tế:** `Test H` không mô phỏng vòng lặp giả lập bên ngoài mà gọi trực tiếp [`run_benchmark(args)`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt02_memory_cleanup_policy.py#L976) với `args.method = "sg_vcd"`, `args.mock = False`, đi qua toàn bộ luồng thật của `evaluate_vitextvqa.py`: `group_samples_by_image()` $\rightarrow$ mở ảnh PIL từ đĩa $\rightarrow$ `build_image_cache_identity()` $\rightarrow$ `pipeline.extract_visual_features()` $\rightarrow$ `pipeline.run()` (bao gồm `AdaptiveScalePropagator._compute_working_maps` / `propagate_multi`, `KLDivergenceRegionSelector.select_top_k`, `IsotropicROICropper.crop`, `_decode_spatial_pmi_contrastive`) $\rightarrow$ `compute_stage1_metrics` / `compute_stage2_metrics` $\rightarrow$ `ckpt_mgr.append_record()` $\rightarrow$ khối `finally` gọi `run_image_boundary_cleanup()`.
2. **Làm rõ bản chất kiểm chứng `weakref` trong `Test H`:**
   - Tại dòng [`881–885`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt02_memory_cleanup_policy.py#L881-L885), `weak_refs.append(weakref.ref(maps[1]))` gắn tham chiếu yếu (`weakref`) vào đối tượng `maps[1]` (tức tensor `bg_smooth: torch.Tensor` có kích thước `(H_w, W_w, 3)` trên CPU thuộc tuple `working_maps` do `AdaptiveScalePropagator._compute_working_maps()` tạo ra cho mỗi ảnh trong 6 ảnh).
   - Kết quả `len(alive_refs) == 0` tại dòng `1012` chứng minh chính xác rằng **các đối tượng `torch.Tensor` trong `_cached_working_maps` của cả 6 bức ảnh đã được xóa hết mọi tham chiếu Python (`refcount == 0`) và được thu hồi bởi Python garbage collector sau khi kết thúc các nhóm ảnh**.
   - **Giới hạn cần phân biệt rõ:** Kiểm chứng `weakref` trên môi trường giả lập CPU chứng minh không bị giữ tham chiếu đối tượng Python (`Python object reference retention`), **không phải** là phép đo thực nghiệm VRAM phần cứng (`GPU VRAM leak measurement`) trên mô hình `Vintern-1B-v3_5` thật.

---

## 6. Kiểm Tra Cache Lifecycle `OPT-03` / `OPT-06` và `SGVCDPipeline.reset_cache()`

Kiểm định trực tiếp cấu trúc của [`VisualFeatureCache`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/feature_cache.py#L388-L530), [`SGVCDPipeline.reset_cache()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L360-L383), và [`PointwiseKLAttentionExtractor.clear_cache()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/attention_extractor.py#L314-L318):

1. **Cấu trúc và contract của `VisualFeatureCache`:**
   - `VisualFeatureCache.__init__` (`feature_cache.py#L399-L400`) chỉ có duy nhất một cấu trúc lưu trữ là `self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()`.
   - Trong `VisualFeatureCache.clear()` (`feature_cache.py#L522-L530`), hàm gốc thực hiện:
     ```python
     for k in list(self._cache.keys()):
         tensor = self._cache.pop(k)
         del tensor
     self._cache.clear()
     if torch is not None and torch.cuda.is_available():
         torch.cuda.empty_cache()
     ```
   - Vì `OrderedDict` là lớp con của `dict` (`isinstance(OrderedDict(), dict) == True`), đoạn mã trong [`SGVCDPipeline.reset_cache()` (`pipeline.py#L370-L377`)](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L370-L377) thực hiện chính xác cùng thao tác `pop(k, None)` + `del tensor` + `cache_dict.clear()` trên `self.feature_cache._cache` mà **không** kích hoạt lệnh `torch.cuda.empty_cache()` vô điều kiện ở dòng `529` của `feature_cache.py`. Nếu người dùng truyền một đối tượng `feature_cache` tùy chỉnh không có thuộc tính `_cache` kiểu `dict`, nhánh `elif hasattr(self.feature_cache, "clear"): self.feature_cache.clear()` vẫn đảm bảo tương thích ngược.
2. **Xác nhận giải phóng toàn diện cả 3 tầng cache ảnh (Verified by Independent Probe):**
   - **Tầng 1 (`OPT-06` VisualFeatureCache):** `self.feature_cache._cache` rỗng (`len == 0`), `weakref` tới tensor `vit_embeds` trong `feature_cache` trả về `None`.
   - **Tầng 2 (`OPT-06` Extractor Stage 1.1 Cache):** `self.extractor.clear_cache()` gán `self._cached_img_state = None` và `self._cached_img_key = None`, giải phóng toàn bộ dictionary chứa `"vit_embeds"`, `"shared_kv"`, `"grid_feats"`, `"suffix_gen"`, `"p_g_heads"`. Probe độc lập xác nhận `weakref` tớiทั้ง `vit_embeds` và tensor `shared_kv[0][0]` đều trả về `None`.
   - **Tầng 3 (`OPT-03` Stage 1.2 Working-Map Cache):** `self._cached_working_maps = None` và `self._cached_working_maps_key = None`. Probe độc lập xác nhận `weakref` tới các tensor `opp` và `bg_smooth` đều trả về `None`.
3. **Bảo toàn trọng số mô hình và dữ liệu ngoài cache:** Probe độc lập xác nhận trọng số mô hình (`pipe.model.language_model.lm_head.weight`) hoàn toàn không bị ảnh hưởng (`model_weight_preserved == True`).
4. **Bảo toàn cache giữa các câu hỏi trong cùng một ảnh:** `reset_cache()` chỉ được gọi:
   - Một lần khi khởi tạo pipeline trước vòng lặp ảnh (`evaluate_vitextvqa.py#L1227`), và
   - Một lần trong khối `finally` tại ranh giới ảnh (`evaluate_vitextvqa.py#L1437` $\rightarrow$ `run_image_boundary_cleanup()`), tức **sau khi toàn bộ câu hỏi trong `q_list` của ảnh đó đã chạy xong**. Do đó, `OPT-03` và `OPT-06` đạt 100% cache hit cho các câu hỏi thứ $2 \dots M$ của cùng một ảnh.
5. **Kiểm tra trạng thái rỗng / gọi lặp lại:** Khi cache đang rỗng hoặc `reset_cache()` / `run_image_boundary_cleanup()` được gọi liên tiếp nhiều lần, vòng lặp `for k in list(cache_dict.keys()):` duyệt qua danh sách rỗng và hoàn tất an toàn ở $\mathcal{O}(1)$ mà không phát sinh lỗi.

---

## 7. Kiểm Tra Image-Boundary Cleanup và Exception Safety (Control-Flow Analysis)

Kiểm tra toàn bộ luồng điều khiển của [`run_benchmark()` (`evaluate_vitextvqa.py#L1237-L1443`)](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/evaluate_vitextvqa.py#L1237-L1443) qua từng giai đoạn:

| Giai đoạn trong `for _group_key, q_list in grouped_samples.items():` | Dòng mã nguồn | Chuyện gì xảy ra nếu phát sinh lỗi / ngoại lệ? | Image-boundary cleanup có chạy đúng không? |
| :--- | :---: | :--- | :--- |
| **1. Kiểm tra đường dẫn ảnh / Mở ảnh PIL (`Image.open(img_path).convert("RGB")`)** | `1248–1256` | Nếu thiếu file (`1248–1250`) hoặc `Image.open` lỗi (`1252–1256`), in thông báo `[SKIP]` / `[ERROR]` và `continue` sang ảnh tiếp theo. Chưa có tensor hay cache nào của ảnh này được tạo ra. | **An toàn** (Cache của ảnh trước đó đã được dọn sạch ở cuối vòng lặp ảnh trước). |
| **2. Tạo khóa định danh `build_image_cache_identity(...)`** | `1261–1268` | Nằm **bên trong** khối `try:` (bắt đầu từ dòng `1260`). Nếu ném ngoại lệ, luồng nhảy vào `finally:` (`dòng 1427–1442`) để đóng `image`, xóa `image` / `cached_visual_features` và gọi `run_image_boundary_cleanup()`. | **Có (`finally` luôn chạy)**. |
| **3. Trích xuất `cached_visual_features` (`pipeline.extract_visual_features`)** | `1271–1282` | Được bọc trong `try ... except Exception: cached_visual_features = None` (bên trong `try:` lớn của ảnh). Nếu lỗi, vòng lặp câu hỏi vẫn tiếp tục chạy và cuối ảnh vẫn đi qua `finally:`. | **Có (`finally` luôn chạy)**. |
| **4. Chạy suy luận từng câu hỏi (`for item in q_list:`)** | `1284–1426` | Khối `try ... except Exception as e:` nội bộ (`1302–1374`) bắt mọi ngoại lệ của từng QA, gán `pred_answer = ""`, `pred_bbox = [0, 0, W, H]`, `spatial_conf = 0.0`, ghi nhận bản ghi lỗi vào `ckpt_mgr` (`dòng 1405`), và tiếp tục chạy các QA còn lại của cùng ảnh đó. Không gọi cleanup sau từng QA. | **Có** (`finally` chỉ chạy đúng 1 lần sau khi toàn bộ `q_list` kết thúc). |
| **5. Khối `finally` tại ranh giới ảnh** | `1427–1442` | 1) Đóng `image.close()` trong `try ... except Exception: pass` (`1429–1433`).<br>2) Gán `image = None; cached_visual_features = None; del image, cached_visual_features` (`1434–1436`).<br>3) Gọi `run_image_boundary_cleanup(...)` (`1437–1442`). | **Chạy đúng 1 lần mỗi image group**. |
| **6. Lỗi bên trong `run_image_boundary_cleanup()`** | `1042–1059` | • `pipeline.reset_cache()` được bọc `try ... except Exception: pipeline_cache_reset = False` (`1044–1048`), nên lỗi trong `reset_cache()` không làm bỏ dở `gc.collect()` hay watchdog.<br>• Các lệnh truy vấn CUDA trong `check_and_run_cuda_memory_watchdog()` đều được bọc `try ... except Exception` (`960–966`, `974–985`). | **An toàn** *(Xem lưu ý nhỏ `NBF-03` về `gc.collect()` và `log_fn`)*. |

---

## 8. Kiểm Tra CUDA `0.85` Memory Watchdog: Biên, Thiết Bị và Xử Lý Lỗi

Kiểm định độc lập hàm [`_resolve_cuda_device_index()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/evaluate_vitextvqa.py#L907-L930) và [`check_and_run_cuda_memory_watchdog()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/evaluate_vitextvqa.py#L933-L1024):

| Kịch bản kiểm tra | Hành vi trong mã nguồn | Kết quả kiểm chứng |
| :--- | :--- | :---: |
| **1. CPU-only (`torch.cuda.is_available() == False`)** | Dòng `957–958`: trả về ngay `status="cpu_or_cuda_unavailable"` trước khi gọi bất kỳ hàm `torch.cuda.*` nào. | **ĐẠT (`PASS`)** |
| **2. `device="cpu"` hoặc `torch.device("cpu")` khi CUDA khả dụng** | `_resolve_cuda_device_index("cpu")` trả về `None` (`dòng 914–915`, `925–926`); dòng `968–969` trả về ngay mà không gọi `memory_allocated` hay `get_device_properties`. | **ĐẠT (`PASS`)** |
| **3. Phân giải active CUDA device (`None`, `"cuda"`, `"cuda:N"`, `torch.device("cuda", N)`, `int`)** | • `None` / `"cuda"` / `torch.device("cuda")`: gọi `int(torch.cuda.current_device())`.<br>• `"cuda:N"` / `torch.device("cuda", N)` / `int >= 0`: trả về đúng chỉ số thiết bị `N`, truyền chính xác `dev_idx` vào cả `torch.cuda.memory_allocated(dev_idx)` (`dòng 975`) và `torch.cuda.get_device_properties(dev_idx)` (`dòng 976`). | **ĐẠT (`PASS`)** |
| **4. Dưới ngưỡng (`ratio < 0.85`, ví dụ `0.50`, `0.62`, `0.74`, `0.78`)** | Dòng `1006`: `if ratio > threshold:` là `False` $\rightarrow$ nhánh `else` (`1015–1022`), không gọi `torch.cuda.empty_cache()`, `status="below_threshold"`. | **ĐẠT (`PASS`)** |
| **5. Đúng bằng ngưỡng (`ratio == 0.85`)** | Do dùng toán tử so sánh nghiêm ngặt `>` (`ratio > 0.85`), tại đúng `0.8500` hàm **không** gọi `torch.cuda.empty_cache()`, `triggered_empty_cache=False`, `status="below_threshold"`. | **ĐẠT (`PASS`)** |
| **6. Trên ngưỡng (`ratio > 0.85`, ví dụ `0.86`, `0.88`, `0.91`, `0.92`)** | Gọi `torch.cuda.empty_cache()` đúng 1 lần (`dòng 1007`), `triggered_empty_cache=True`, `status="triggered"`. | **ĐẠT (`PASS`)** |
| **7. Ngoại lệ khi truy vấn CUDA hoặc giá trị bộ nhớ không hợp lệ (`total_memory <= 0`, `allocated < 0`)** | • Dòng `974–985`: bắt `Exception`, trả về `status="memory_query_unavailable:..."`.<br>• Dòng `987–996`: kiểm tra `if total_bytes <= 0 or allocated_bytes < 0:`, trả về `status="invalid_memory_values"` trước khi thực hiện phép chia `float(allocated_bytes) / float(total_bytes)`, ngăn chặn hoàn toàn lỗi `ZeroDivisionError`. | **ĐẠT (`PASS`)** |
| **8. Phân biệt bản chất bộ nhớ (`memory_allocated` vs `memory_reserved`)** | Công thức triển khai tuân thủ đúng yêu cầu của Handoff (`memory_allocated(dev_idx) / total_memory > 0.85`). Như đã phân tích tại Mục 13.1 của báo cáo `TASK 09`, `memory_allocated()` đo các tensor đang còn sống, còn `empty_cache()` chỉ giải phóng các khối bộ nhớ đệm không còn sử dụng (`unused cached allocator blocks`) trong `memory_reserved()`, không giải phóng các tensor đang active và không phải là cơ chế bảo đảm tuyệt đối chống OOM nếu tổng kích thước các tensor đang sống vượt quá dung lượng VRAM vật lý. | **ĐẠT (`PASS`)** |

---

## 9. Kiểm Tra Tính Chính Xác và Các Điểm Sai Lệch / Overclaim Trong Báo Cáo Triển Khai (`TASK 09`)

### 9.1. Bảng phân loại mức độ xác thực của các số liệu trong báo cáo `TASK 09`

| Khẳng định / Số liệu trong báo cáo `TASK 09` | Cấp độ bằng chứng (Verification Class) | Kết quả đối chiếu độc lập của `Anti` |
| :--- | :---: | :--- |
| Tập dữ liệu chuẩn có **400 ảnh / 1.222 câu hỏi** (`SHA-256: 1428e33d...`) | **Verified by Source / Artifact** | Chính xác 100% (đã kiểm toán trên `results/swintextspotter_auto_annot/answer_bboxes_400_verified.json`). |
| Trước `OPT-02`: `1,222` per-question `gc.collect()` + `400` image-boundary `gc.collect()` = **`1,622` lần gọi `gc.collect()`** | **Verified by Source & Control Flow** | Chính xác (1 lần trong `pipeline.run` cho mỗi QA trong 1.222 QA + 1 lần cuối mỗi nhóm ảnh cho 400 nhóm ảnh). |
| Trước `OPT-02`: `1,222` (`extractor.extract`) + `1,222` (`pipeline.run`) + `400` (image boundary) = **`2,844` lần gọi `torch.cuda.empty_cache()`** | **Verified by Source & Control Flow** | Chính xác về **số lần gọi hàm (`call count`)** trên đường chạy chuẩn 400 ảnh / 1.222 QA khi `torch.cuda.is_available() == True`. *(Xem `NBF-02` về thuật ngữ "synchronizations")*. |
| Sau `OPT-02`: **`0` per-question cleanup**; **`400` image-boundary `gc.collect()`**; **`0–400` lần gọi `empty_cache()` có điều kiện** | **Verified by Source, AST & Local Mock Tests** | Chính xác (`Test A`, `Test H`, `Test I` xác nhận `0` lần gọi per-question và đúng `1` lần cleanup mỗi ranh giới ảnh). |
| Cải thiện thời gian chạy thực tế (wall-clock latency) và dung lượng VRAM trên mô hình `5CD-AI/Vintern-1B-v3_5` | **Not Measured on Real GPU/Model (`NOT_MEASURED`)** | Báo cáo `TASK 09` đã ghi nhận trung thực là `NOT_MEASURED` tại Mục 14.1 và Mục 15. |

### 9.2. Các điểm sai lệch nhỏ trong văn bản báo cáo `TASK 09` và chú thích kiểm thử
1. **Sai lệch mô tả kiểu dữ liệu tuple `_compute_working_maps` và tên phương thức (`NBF-01`):**
   - Trong báo cáo `TASK 09` tại Mục 1 (`dòng 35`), Mục 6 (`dòng 150, 156`), Mục 8 (`dòng 197`), Mục 13.2 (`dòng 268`) và trong chú thích tại [`tests/test_opt02_memory_cleanup_policy.py#L882`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt02_memory_cleanup_policy.py#L882), `4argon` viết rằng `_compute_working_maps` trả về `(img_work, L, C_rg, C_yb, W_w, H_w, sx, sy)` (mảng NumPy opponent-color) và gọi hàm `propagate_multi_with_maps`.
   - Thực tế trong [`scripts/benchmark/sg_vcd/adaptive_propagator.py#L316-L388`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/adaptive_propagator.py#L316-L388) và [`scripts/benchmark/sg_vcd/pipeline.py#L906-L911`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L906-L911), `_compute_working_maps` trả về bộ 6 phần tử `(opp, bg_smooth, stroke_energy, horiz_mod, scale_x, scale_y)` gồm 4 `torch.Tensor` (`dtype=torch.float32`, trên `self.device`) và 2 số thực `float`, đồng thời phương thức được gọi là `self.propagator.propagate_multi(..., working_maps=working_maps)`.
   - Lưu ý: Tại dòng [`883`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt02_memory_cleanup_policy.py#L883), lệnh `weakref.ref(maps[1])` trong `Test H` vẫn hoạt động hoàn toàn đúng và hợp lệ vì `maps[1]` là `bg_smooth` (`torch.Tensor`, hỗ trợ `weakref`), chỉ có dòng comment ở dòng `882` và lời mô tả trong báo cáo là ghi theo tên biến cũ.
2. **Sử dụng cụm từ `"host-device synchronizations"` cho số lần gọi `torch.cuda.empty_cache()` (`NBF-02`):**
   - Tại Mục 1 (`dòng 22`) và Mục 4 (`dòng 86`) của báo cáo `TASK 09`, `4argon` gọi `2,844` và `1,222` lần gọi `torch.cuda.empty_cache()` là `"unconditional torch.cuda.empty_cache() host-device synchronizations"`.
   - Cần hiệu đính về mặt thuật ngữ: con số `2,844` / `1,222` là **số lần gọi hàm `torch.cuda.empty_cache()`** (inferred from control flow on the 400-image / 1,222-QA manifest), không phải số lần đồng bộ hóa phần cứng đã được đo bằng profiler trên GPU thật.

---

## 10. Danh Sách Các Phát Hiện (Findings), Mức Độ Nghiêm Trọng và Yêu Cầu Khắc Phục

| Mã Finding | Mức độ (Severity) | Tệp tin & Dòng tham chiếu | Bằng chứng & Tác động | Yêu cầu khắc phục (Remediation Request) |
| :---: | :---: | :--- | :--- | :--- |
| **`NBF-01`** | **NON-BLOCKING** *(Documentation / Comment Accuracy)* | • [`docs/research/TASK_09_OPT02_MEMORY_CLEANUP_IMPLEMENTATION_REPORT.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/TASK_09_OPT02_MEMORY_CLEANUP_IMPLEMENTATION_REPORT.md#L35) (`lines 35, 150, 156, 197, 268`)<br>• [`tests/test_opt02_memory_cleanup_policy.py#L882`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt02_memory_cleanup_policy.py#L882) | Mô tả tuple `_compute_working_maps` là `(img_work, L, C_rg, C_yb, W_w, H_w, sx, sy)` (NumPy arrays) và tên hàm là `propagate_multi_with_maps`, trong khi mã nguồn thực tế (`adaptive_propagator.py#L388`, `pipeline.py#L906`) trả về `(opp, bg_smooth, stroke_energy, horiz_mod, scale_x, scale_y)` (`torch.Tensor`) và gọi `propagate_multi(..., working_maps=working_maps)`. Không ảnh hưởng đến tính đúng đắn của mã nguồn hay kết quả chạy `Test H`. | Không chặn nghiệm thu Gate A. Ghi nhận đính chính chính thức trong báo cáo QA này; có thể cập nhật lại chú thích dòng `882` nếu có đợt chỉnh sửa tệp test sau này. |
| **`NBF-02`** | **NON-BLOCKING** *(Terminology Precision)* | • [`docs/research/TASK_09_OPT02_MEMORY_CLEANUP_IMPLEMENTATION_REPORT.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/TASK_09_OPT02_MEMORY_CLEANUP_IMPLEMENTATION_REPORT.md#L22) (`lines 22, 86`) | Đồng nhất số lần gọi hàm `torch.cuda.empty_cache()` (`2,844` / `1,222` calls) với số lần `"host-device synchronizations"` khi chưa chạy đo đạc thực tế trên GPU. | Không chặn nghiệm thu Gate A. Khi trích dẫn số liệu này trong các báo cáo tổng hợp tiếp theo, ghi rõ là **"số lần gọi `torch.cuda.empty_cache()` (`call count`)"**. |
| **`NBF-03`** | **NON-BLOCKING** *(Defensive Edge Cases)* | • [`scripts/benchmark/evaluate_vitextvqa.py#L910-L922`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/evaluate_vitextvqa.py#L910-L922)<br>• [`scripts/benchmark/evaluate_vitextvqa.py#L1050-L1055`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/evaluate_vitextvqa.py#L1050-L1055) | 1) Trong `_resolve_cuda_device_index()`, `isinstance(device, int)` chưa loại trừ `isinstance(device, bool)` và chuỗi sai định dạng `"cuda:abc"` rơi về `current_device()`.<br>2) Trong `run_image_boundary_cleanup()`, `gc.collect()` không bọc `try/except`, nên nếu `gc.collect()` bị monkeypatch ném lỗi thì watchdog phía sau sẽ không chạy. Không ảnh hưởng đến đường chạy chuẩn của `evaluate_vitextvqa.py`. | Không chặn nghiệm thu Gate A (tùy chọn củng cố nếu chỉnh sửa `evaluate_vitextvqa.py` trong tương lai). |
| **`INFO-01`** | **INFORMATIONAL** | • [`docs/research/TASK_09_OPT02_MEMORY_CLEANUP_IMPLEMENTATION_REPORT.md#L276-L282`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/TASK_09_OPT02_MEMORY_CLEANUP_IMPLEMENTATION_REPORT.md#L276-L282) | Hiệu năng thời gian thực tế và hành vi VRAM trên mô hình thật `5CD-AI/Vintern-1B-v3_5` chưa được đo đạc ở local theo đúng quy tắc quản trị (`NOT_MEASURED`). | Giữ nguyên trạng thái `NOT_MEASURED` cho đến khi Project Owner cấp phép chạy Gate B. |

---

## 11. Phán Quyết Cuối Cùng và Trạng Thái Gate A (Final Verdict & Gate A Status)

- **Phán quyết nghiệm thu độc lập cho `TICKET-OPT-02`:** **`PASS WITH NON-BLOCKING FINDINGS`**
- **Trạng thái Gate A (Local Synthetic / Mock Verification):** **`PASS` (Đủ điều kiện nghiệm thu kỹ thuật Gate A cho `TICKET-OPT-02`)**.
  - Mã nguồn triển khai đạt 100% các yêu cầu chức năng và bất biến thuật toán của `TICKET-OPT-02`, không có lỗi `BLOCKER`, không cần mở vòng sửa lỗi bắt buộc (`OPT-02-FIX`) trước khi chuyển sang bước tiếp theo trong kế hoạch đã duyệt.

---

## 12. Xác Nhận Về Gate B (Gate B Authorization Status)

- **Gate B (20-QA Real-Model Validation Slice trên GPU):** **`NOT EXECUTED` và `NOT AUTHORIZED`**.
- Theo đúng quy trình quản trị tại [`.github/copilot-instructions.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/.github/copilot-instructions.md) và [`.github/skills/local-colab-experiment-workflow/SKILL.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/.github/skills/local-colab-experiment-workflow/SKILL.md), không có bất kỳ trọng số mô hình thật (`5CD-AI/Vintern-1B-v3_5`) hay suy luận GPU thực tế nào được chạy trong lượt kiểm định này, và Gate B chỉ được phép thực hiện khi có phê duyệt tường minh từ **Project Owner**.

