# BÁO CÁO TRIỂN KHAI TICKET-OPT-03 — LƯU ĐỆM WORKING-MAP CẤP ĐỘ ẢNH (IMAGE-LEVEL WORKING-MAP CACHING)

- **Ticket chính danh:** `TICKET-OPT-03 — Image-Level Working-Map Caching` (Mục 4, Ticket 2 của [`docs/research/PERFORMANCE_OPTIMIZATION_HANDOFF_v1.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/PERFORMANCE_OPTIMIZATION_HANDOFF_v1.md#L168-L214))
- **Vai trò thực hiện:** `4argon` — Senior ML Systems Engineer
- **Chuỗi kiểm định & phê duyệt:** `Antigravity (Anti)` (Independent QA) $\rightarrow$ `ChatGPT` (Methodology & Architecture Review) $\rightarrow$ `Project Owner` (Final Approval)
- **Mức ưu tiên:** P1
- **Chế độ thực thi:** Triển khai giới hạn phạm vi (bounded implementation), kiểm thử cục bộ hoàn toàn bằng mock/synthetic (zero real-model weights loading).
- **Trạng thái Gate A (Local Engineering & Synthetic Validation):** **PASS**

---

## 1. Tóm tắt điều hành & Định danh chính xác của Ticket (Executive Summary & Exact Ticket Identity)

Báo cáo này ghi nhận việc triển khai hoàn chỉnh và kiểm chứng độc lập bằng bộ test synthetic/mock cho:

> **`TICKET-OPT-03 — Image-Level Working-Map Caching`**
> *(Định nghĩa chuẩn tại Mục 4, Ticket 2 của [`docs/research/PERFORMANCE_OPTIMIZATION_HANDOFF_v1.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/PERFORMANCE_OPTIMIZATION_HANDOFF_v1.md#L168-L214))*

**Vấn đề kỹ thuật được giải quyết:**
Trong tập dữ liệu ViTextVQA, trung bình mỗi bức ảnh đi kèm khoảng $\sim 3.31$ câu hỏi. Trước khi có `OPT-03`, ở mỗi câu hỏi thuộc cùng một bức ảnh, `SGVCDPipeline.run()` gọi `self.propagator.propagate_multi(image=image, ...)` ([`pipeline.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py)), và bên trong [`AdaptiveScalePropagator.propagate_multi()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/adaptive_propagator.py#L1313-L1343) / [`AdaptiveScalePropagator.propagate()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/adaptive_propagator.py#L568-L625) luôn gọi lại `self._compute_working_maps(image)` từ đầu. Hàm [`_compute_working_maps(image)`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/adaptive_propagator.py#L316-L389) thực hiện chuỗi phép toán xử lý ảnh thuần túy phụ thuộc vào ảnh gốc (`PIL.Image.convert('RGB')`, nội suy song tuyến tính về `work_max_side`, chuyển đổi không gian màu đối lập $(L, O_1, O_2)$, làm trơn nền Gaussian tách chiều `bg_smooth_t`, tính năng lượng nét chữ Sobel đa kênh `stroke_energy_t`, và điều biến trường hướng ngang `horiz_mod_t`). Việc tính lại bộ 6 phần tử `working_maps = (opp_t, bg_smooth_t, stroke_energy_t, horiz_mod_t, scale_x, scale_y)` cho từng câu hỏi trên cùng một ảnh gây lãng phí khoảng $\sim 69.8\%$ số lần gọi `_compute_working_maps(image)`.

**Giải pháp đã triển khai:**
1. Bổ sung hàm kiểm định cấu trúc [`_is_valid_working_maps(working_maps)`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/adaptive_propagator.py#L150-L183) và tham số tùy chọn `working_maps: Optional[Tuple[Any, ...]] = None` vào [`BaseSeedPropagator.propagate`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/adaptive_propagator.py#L186-L202), [`BaseSeedPropagator.propagate_multi`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/adaptive_propagator.py#L204-L226), [`AdaptiveScalePropagator.propagate`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/adaptive_propagator.py#L568-L625), và [`AdaptiveScalePropagator.propagate_multi`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/adaptive_propagator.py#L1313-L1343).
2. Bổ sung bộ nhớ đệm tối đa 1 ảnh (`self._cached_working_maps` và `self._cached_working_maps_key`) vào [`SGVCDPipeline`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L357-L358), khóa theo định danh ảnh bất biến `resolved_image_key: ImageCacheIdentity` của `OPT-06`.
3. Bảo toàn tuyệt đối tính bất biến (immutability) và tính tương đương số học 100% (exact bit-for-bit parity) giữa chế độ dùng cache và không dùng cache cho toàn bộ kết quả Stage 1 và Stage 2.

---

## 2. Tài liệu Quản trị đã đọc & Tuyên bố Tuân thủ (Governance Documents Read & Compliance Declaration)

Trước khi khảo sát hoặc chỉnh sửa mã nguồn, các tài liệu quản trị và kỹ thuật bắt buộc sau đã được đọc đầy đủ:

1. [`PROJECT.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/PROJECT.md) *(Lập bản đồ cấu trúc dự án; kiểm tra sự tồn tại của `AGENTS.md` ở thư mục gốc — hiện chưa có file `AGENTS.md` riêng nên tuân thủ `.github/copilot-instructions.md` và `PROJECT.md`)*.
2. [`.github/copilot-instructions.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/.github/copilot-instructions.md)
3. [`.github/skills/anti-output-quality-review/SKILL.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/.github/skills/anti-output-quality-review/SKILL.md)
4. [`.github/skills/research-plan-compliance-review/SKILL.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/.github/skills/research-plan-compliance-review/SKILL.md)
5. [`.github/skills/local-colab-experiment-workflow/SKILL.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/.github/skills/local-colab-experiment-workflow/SKILL.md)
6. [`docs/research/PERFORMANCE_OPTIMIZATION_HANDOFF_v1.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/PERFORMANCE_OPTIMIZATION_HANDOFF_v1.md)
7. [`docs/research/TASK_07_FIX_OPT06_INDEPENDENT_ACCEPTANCE_QA.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/TASK_07_FIX_OPT06_INDEPENDENT_ACCEPTANCE_QA.md)
8. [`docs/research/REMEDIATION_SPEC_P0_v1.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/research/REMEDIATION_SPEC_P0_v1.md)
9. [`docs/Proposed_Method_SG_VCD.md`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/docs/Proposed_Method_SG_VCD.md)

**Tuyên bố tuân thủ (Compliance Declaration):**
- **Không xung đột nguồn chân lý (Zero Source-of-Truth Conflict):** Toàn bộ thay đổi tuân thủ nghiêm ngặt quy trình Small-First Gate A, kiến trúc toán học của SG-VCD trong `Proposed_Method_SG_VCD.md`, và định danh `ImageCacheIdentity` đã được nghiệm thu tại `TASK_07_FIX_OPT06_INDEPENDENT_ACCEPTANCE_QA.md`.
- **Tuân thủ giới hạn phần cứng cục bộ (Local Hardware Safety):** Không nạp trọng số mô hình thật `5CD-AI/Vintern-1B-v3_5`, không tải dữ liệu từ mạng, không chạy suy luận mô hình thật trên máy local, và không thực hiện `git add` / `git commit` / `git push`.

---

## 3. Các File và Hàm Mã nguồn đã Khảo sát (Source Files and Functions Inspected)

| File mã nguồn | Các lớp / hàm / vùng dòng lệnh đã khảo sát | Mục đích khảo sát |
| :--- | :--- | :--- |
| [`scripts/benchmark/sg_vcd/adaptive_propagator.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/adaptive_propagator.py) | [`BaseSeedPropagator.propagate`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/adaptive_propagator.py#L186-L202), [`BaseSeedPropagator.propagate_multi`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/adaptive_propagator.py#L204-L226), [`FixedMarginPropagator.propagate`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/adaptive_propagator.py#L247-L287), [`AdaptiveScalePropagator._compute_working_maps`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/adaptive_propagator.py#L316-L389), [`AdaptiveScalePropagator.propagate`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/adaptive_propagator.py#L568-L1311), [`AdaptiveScalePropagator.propagate_multi`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/adaptive_propagator.py#L1313-L1343) | Kiểm tra chữ ký hàm, xác minh `_compute_working_maps(image)` chỉ phụ thuộc ảnh, và kiểm tra từng dòng sử dụng `(opp_t, bg_smooth_t, stroke_energy_t, horiz_mod_t, scale_x, scale_y)` trong Step 1 & Step 2 để chứng minh không có phép toán in-place làm bẩn tensor được lưu đệm. |
| [`scripts/benchmark/sg_vcd/pipeline.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py) | [`SGVCDPipeline.__init__`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L261-L358), [`SGVCDPipeline.reset_cache`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L360-L375), [`SGVCDPipeline.run`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L755-L1200) | Kiểm tra cách `resolved_image_key` (`ImageCacheIdentity`) được tạo ở đầu `run()`, cách `reset_cache()` dọn dẹp bộ nhớ giữa các ảnh, và vị trí Stage 1.3 gọi `self.propagator.propagate_multi(...)`. |
| [`scripts/benchmark/sg_vcd/feature_cache.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/feature_cache.py) | [`ImageCacheIdentity`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/feature_cache.py#L38-L55), [`build_image_cache_identity`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/feature_cache.py#L77-L162) | Đảm bảo khóa cache của `OPT-03` dùng chung chính xác cấu trúc `ImageCacheIdentity` đã được chứng nhận ở `OPT-06`. |
| [`scripts/benchmark/evaluate_vitextvqa.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/evaluate_vitextvqa.py) | Vòng lặp đánh giá theo nhóm ảnh (`lines 1500–1605`) | Xác nhận cách `evaluate_vitextvqa.py` truyền `image_id`, `image_path`, `image_key` vào `pipeline.run(...)` và gọi `pipeline.reset_cache()` khi chuyển nhóm ảnh. |

---

## 4. Chi tiết Thay đổi Triển khai (Exact Implementation Changes)

### 4.1. [`scripts/benchmark/sg_vcd/adaptive_propagator.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/adaptive_propagator.py)

1. **Hàm kiểm định [`_is_valid_working_maps(working_maps: Any) -> bool`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/adaptive_propagator.py#L150-L183):**
   - Kiểm tra `working_maps` là một `tuple` gồm đúng 6 phần tử `(opp_t, bg_smooth_t, stroke_energy_t, horiz_mod_t, scale_x, scale_y)`.
   - Kiểm tra `opp_t` và `bg_smooth_t` là `torch.Tensor` 3 chiều `(3, H_w, W_w)` có kích thước dương.
   - Kiểm tra `stroke_energy_t` và `horiz_mod_t` là `torch.Tensor` 2 chiều `(H_w, W_w)` khớp chính xác kích thước không gian `(H_w, W_w)` với `opp_t`.
   - Kiểm tra `scale_x` và `scale_y` là số thực (`int` hoặc `float`, không phải `bool`) và lớn hơn `0.0`.
2. **Cập nhật [`BaseSeedPropagator.propagate`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/adaptive_propagator.py#L186-L202) và [`BaseSeedPropagator.propagate_multi`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/adaptive_propagator.py#L204-L226):**
   - Bổ sung tham số tùy chọn `working_maps: Optional[Tuple[Any, ...]] = None`.
   - Trong `BaseSeedPropagator.propagate_multi`, kiểm tra chữ ký của `self.propagate` bằng `inspect.signature(self.propagate).parameters`: nếu lớp con chấp nhận `working_maps` (hoặc `**kwargs`), truyền `working_maps=working_maps` xuống; nếu lớp con cổ điển không khai báo `working_maps` (ví dụ `FixedMarginPropagator` hoặc mock propagator tùy chỉnh trong test), gọi `self.propagate(image, box, seed_Mask)` hoàn toàn tương thích ngược.
3. **Cập nhật [`AdaptiveScalePropagator.propagate`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/adaptive_propagator.py#L568-L625) và [`AdaptiveScalePropagator.propagate_multi`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/adaptive_propagator.py#L1313-L1343):**
   - Nhận tham số tùy chọn `working_maps: Optional[Tuple[Any, ...]] = None`.
   - Trong `propagate()`: nếu `_is_valid_working_maps(working_maps)` trả về `True`, giải nén trực tiếp `(opp_t, bg_smooth_t, stroke_energy_t, horiz_mod_t, scale_x, scale_y) = working_maps` và đặt cờ `_owns_working_maps = False`; ngược lại mới gọi `self._compute_working_maps(image)` và đặt `_owns_working_maps = True`. Cuối hàm ([`lines 1273–1275`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/adaptive_propagator.py#L1273-L1275)), chỉ giải phóng biến `working_maps` cục bộ khi `_owns_working_maps` là `True`.
   - Trong `propagate_multi()`: nếu `_is_valid_working_maps(working_maps)` trả về `True`, tái sử dụng trực tiếp cho mọi vùng ứng viên $k \in \{1, \dots, K\}$ mà không gọi `self._compute_working_maps(image)`; nếu `working_maps` là `None` (khi gọi độc lập bên ngoài pipeline), tính `working_maps = self._compute_working_maps(image)` đúng 1 lần và truyền vào `self.propagate(..., working_maps=working_maps)` cho tất cả $K$ hộp.

### 4.2. [`scripts/benchmark/sg_vcd/pipeline.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py)

1. **Khởi tạo thuộc tính cache trong [`SGVCDPipeline.__init__`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L357-L358):**
   - Khai báo `self._cached_working_maps: Optional[Tuple[Any, ...]] = None` và `self._cached_working_maps_key: Optional[ImageCacheIdentity] = None`.
2. **Làm sạch cache trong [`SGVCDPipeline.reset_cache()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L360-L375):**
   - Đặt `self._cached_working_maps = None` và `self._cached_working_maps_key = None` trước khi gọi `self.extractor.clear_cache()` và `gc.collect()`.
3. **Quản lý vòng đời cache tại Stage 1.3 của [`SGVCDPipeline.run()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py#L875-L929):**
   - Kiểm tra xem `self.propagator` có cung cấp phương thức `_compute_working_maps` hay không.
   - **Cache Hit:** Khi `self._cached_working_maps_key == resolved_image_key` và `_is_valid_working_maps(self._cached_working_maps)` đúng, tái sử dụng ngay `working_maps = self._cached_working_maps` (không gọi `_compute_working_maps`).
   - **Cache Miss / Image Transition:** Trước khi tính `working_maps` mới cho ảnh hiện tại, **luôn xóa sạch** `self._cached_working_maps = None` và `self._cached_working_maps_key = None`. Nhờ đó, nếu `_compute_working_maps(image)` phát sinh ngoại lệ ở ảnh mới, pipeline tuyệt đối không giữ lại `working_maps` hay khóa của ảnh cũ (Failure Safety). Chỉ sau khi `_compute_working_maps(image)` hoàn tất và thỏa `_is_valid_working_maps(...)`, pipeline mới gán `self._cached_working_maps = computed_maps` và `self._cached_working_maps_key = resolved_image_key`.
   - Truyền `working_maps` vào `self.propagator.propagate_multi(image=image, boxes_and_masks=boxes_and_masks, m_aff_map=extraction.aff_map, a_q_map=extraction.a_query_map, working_maps=working_maps)` (kèm cơ chế kiểm tra `inspect.signature` để tương thích ngược với các lớp con `BaseSeedPropagator` tùy chỉnh).

---

## 5. Thiết kế Vòng đời Cache và Cơ chế Vô hiệu hóa (Cache Lifecycle and Invalidation Design)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ SGVCDPipeline.run(image, question, image_id=..., image_path=..., image_key=.)│
└───────────────────────────────────────┬──────────────────────────────────────┘
                                        ▼
               resolved_image_key = build_image_cache_identity(...)
                                        │
                                        ▼
        ┌───────────────────────────────────────────────────────────────┐
        │ Stage 1.3: Kiểm tra _cached_working_maps_key == resolved_key  │
        │            VÀ _is_valid_working_maps(_cached_working_maps)    │
        └───────────────┬───────────────────────────────┬───────────────┘
                ĐÚNG    │                               │ SAI (Miss / Ảnh mới)
                        ▼                               ▼
        ┌───────────────────────────────┐ ┌────────────────────────────────────┐
        │ CACHE HIT:                    │ │ CACHE MISS / INVALIDATION:         │
        │ Tái sử dụng trực tiếp         │ │ 1. Xóa _cached_working_maps = None │
        │ working_maps từ bộ nhớ đệm    │ │    _cached_working_maps_key = None │
        │ (0 lần gọi _compute_working_  │ │ 2. Gọi _compute_working_maps(image)│
        │  maps, 0 lần clone tensor)    │ │ 3. Nếu hợp lệ, lưu tuple + key mới │
        └───────────────┬───────────────┘ └─────────────────┬──────────────────┘
                        └───────────────┬───────────────────┘
                                        ▼
          propagator.propagate_multi(..., working_maps=working_maps)
```

1. **Giới hạn dung lượng 1 ảnh duy nhất (Single-Image Capacity Bound):** `SGVCDPipeline` chỉ giữ tối đa một bộ `working_maps` ứng với ảnh đang xử lý (`self._cached_working_maps` và `self._cached_working_maps_key`). Không sử dụng `dict` không giới hạn theo ảnh, ngăn chặn triệt để hiện tượng tích tụ VRAM/RAM qua hàng trăm ảnh.
2. **Đồng bộ khóa định danh với `OPT-06`:** Khóa lưu đệm `self._cached_working_maps_key` dùng chính đối tượng `resolved_image_key` (`ImageCacheIdentity`) đã truyền cho `self.extractor.extract(..., image_key=resolved_image_key)`. Do đó, cache working-map và cache visual-token của `AttentionExtractor` luôn đồng bộ tuyệt đối trên cùng một biên ảnh.
3. **Ba điều kiện vô hiệu hóa (Invalidation Triggers):**
   - **Chuyển ảnh tự động (Automatic Image Transition):** Ngay khi `resolved_image_key != self._cached_working_maps_key`, bộ đệm cũ bị gán về `None` ngay lập tức trước khi tính bộ đệm mới.
   - **Gọi tường minh `pipeline.reset_cache()`:** Xóa cả `self._cached_working_maps = None`, `self._cached_working_maps_key = None`, và `self.extractor.clear_cache()`.
   - **An toàn khi lỗi (Exception Invalidation):** Do `self._cached_working_maps` và `self._cached_working_maps_key` được xóa về `None` *trước* lệnh gọi `self.propagator._compute_working_maps(image)`, bất kỳ ngoại lệ nào xảy ra trong quá trình tính toán cho ảnh mới đều để lại trạng thái cache rỗng (`None`), không bao giờ để rò rỉ dữ liệu của ảnh trước sang câu hỏi tiếp theo.

---

## 6. Giải thích Cơ chế Giữ Nguyên Các Phép Toán Phụ Thuộc Câu Hỏi (How Question-Dependent Calculations Remain Uncached)

Trong kiến trúc Stage 1 của SG-VCD, các phép toán được tách bạch rõ ràng thành hai nhóm:

1. **Nhóm chỉ phụ thuộc ảnh (Image-Only — Được lưu đệm bởi `OPT-03`):**
   - Hàm [`AdaptiveScalePropagator._compute_working_maps(image)`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/adaptive_propagator.py#L316-L389) chỉ nhận đầu vào duy nhất là `image: Image.Image` (cùng cấu hình tĩnh `self.work_max_side` và `self.device`).
   - Kết quả trả về `(opp_t, bg_smooth_t, stroke_energy_t, horiz_mod_t, scale_x, scale_y)` biểu diễn bản đồ màu đối lập, nền trơn Gaussian, năng lượng nét chữ Sobel, trường điều biến ngang và tỷ lệ co giãn không gian của riêng bức ảnh đó.
   - Trong suốt [`AdaptiveScalePropagator.propagate()`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/adaptive_propagator.py#L568-L1311), cả 4 tensor `opp_t`, `bg_smooth_t`, `stroke_energy_t`, `horiz_mod_t` chỉ được đọc thông qua các phép cắt lát tạo view và phép toán tạo tensor mới ngoài chỗ (out-of-place subtraction, norm, quantile, median). Lệnh `del (opp_t, bg_smooth_t, stroke_energy_t, ...)` tại dòng `772` chỉ hủy liên kết biến cục bộ trong khung hàm `propagate()`, không chạm tới tuple lưu trong `self._cached_working_maps`.

2. **Nhóm phụ thuộc câu hỏi (Question-Conditioned — Tuyệt đối KHÔNG lưu đệm):**
   - **Stage 1.1 (`AttentionExtractor.extract`):** Mặc dù visual feature (`vit_embeds`) được tái sử dụng qua `OPT-06`, bước forward câu hỏi qua LLM và trích xuất bản đồ chú ý $\Phi$, bản đồ dư địa $R$, bản đồ tương quan ngữ nghĩa $M_{\text{aff}}$, và $A_Q$ vẫn chạy mới cho từng câu hỏi `question`.
   - **Stage 1.2 (`TopKRegionSelector.select_regions`):** Chọn danh sách vùng hạt giống `boxes_and_masks` từ $\Phi$ của từng câu hỏi.
   - **Stage 1.3 (`AdaptiveScalePropagator.propagate_multi` / `propagate`):** Trên mỗi câu hỏi, thuật toán nhận `boxes_and_masks`, `m_aff_map`, `a_q_map` riêng của câu hỏi đó, nội suy `m_aff_work` và `a_q_work` theo câu hỏi, xác định phân bố màu tiền cảnh/hậu cảnh từ `seed_mask` của câu hỏi, lan truyền thích nghi Step 1 & Step 2, và giải quyết va chạm hộp giới hạn (`resolve_box_collisions`).
   - **Stage 1.4 (`DynamicMultiPatchROICropper.crop_multi`) & Stage 2 (`DynamicContrastiveDecoder`):** Cắt vùng ROI từ các hộp đã lan truyền của câu hỏi hiện tại và giải mã tương phản động cho câu hỏi đó.

---

## 7. Lệnh Kiểm Thử Chính Xác và Kết Quả Thực Tế (Exact Test Commands and Actual Outputs)

Tất cả các bộ kiểm thử được chạy bằng môi trường Python ảo chuẩn của dự án (`..\Document AI\VLM_Reasoning\venv\Scripts\python.exe`) ở chế độ local synthetic/mock.

### 7.1. Bộ kiểm thử chuyên biệt `OPT-03` (`tests.test_opt03_working_map_cache`)

**Lệnh thực thi:**
```powershell
$py = "..\Document AI\VLM_Reasoning\venv\Scripts\python.exe"
& $py -m unittest -v tests.test_opt03_working_map_cache
```

**Kết quả đầu ra thực tế:**
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
Ran 8 tests in 8.414s

OK
EXIT_OPT03=0
```

### 7.2. Các bộ kiểm thử hồi quy (Regression Suites)

**Lệnh thực thi:**
```powershell
$py = "..\Document AI\VLM_Reasoning\venv\Scripts\python.exe"
& $py -m unittest -v tests.test_opt06_image_cache_identity
& $py -m unittest -v tests.test_sg_vcd
& $py -m unittest -v tests.test_adversarial_spatial_processing
& $py -m unittest -v tests.test_tier4_workloads.TestTier4RealWorldWorkloads.test_05_anti_hardcoding_and_heuristics_audit
```

**Bảng tổng hợp kết quả thực tế:**

| Bộ kiểm thử (Test Suite) | Số lượng test | Kết quả | Thời gian chạy | Ghi chú |
| :--- | :---: | :---: | :---: | :--- |
| [`tests.test_opt03_working_map_cache`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt03_working_map_cache.py) | 8 / 8 | **PASS (`OK`)** | `8.414s` | Kiểm chứng đầy đủ Tests A, B, C, D, E, F, G, H theo đặc tả Task 08. |
| [`tests.test_opt06_image_cache_identity`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt06_image_cache_identity.py) | 11 / 11 | **PASS (`OK`)** | `2.658s` | Bảo toàn 100% cơ chế định danh ảnh bất biến `OPT-06` và sửa lỗi `M-01`/`M-02`/`M-03`. |
| [`tests.test_sg_vcd`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_sg_vcd.py) | 33 / 33 | **PASS (`OK`)** | `0.272s` | Toàn bộ unit test cốt lõi của SG-VCD Stage 1 & Stage 2 đều vượt qua. |
| [`tests.test_adversarial_spatial_processing`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_adversarial_spatial_processing.py) | 21 / 21 | **PASS (`OK`)** | `0.056s` | Kiểm thử đối kháng chọn vùng Top-K, xử lý va chạm hộp và tăng cường cạnh Laplacian. |
| [`tests.test_tier4_workloads...test_05_anti_hardcoding_and_heuristics_audit`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_tier4_workloads.py) | 1 / 1 | **PASS (`OK`)** | `0.230s` | Kiểm toán AST chống hardcode và heuristic. |
| `tests.test_adversarial_m3_spatial` (`TestAdversarialSobelGradientPrior` + `TestAdversarialMultiTileSlicing`) | 8 / 8 | **PASS (`OK`)** | `2.340s` | Lưu ý: `TestAdversarialBoundingBoxes.test_04_extreme_aspect_ratios` trong cùng file có 2 subtest fail sẵn có từ trước trong `roi_cropper.py` (`1200 != 400`), hoàn toàn không liên quan đến `OPT-03` hay `OPT-06`. |

---

## 8. Bằng chứng Tương đương Tuyệt đối Giữa Có Cache và Không Cache (Cached-vs-Uncached Parity Evidence)

Trong [`TestOPT03WorkingMapCache.test_E_cached_versus_uncached_exact_parity`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt03_working_map_cache.py#L288-L355), hai luồng thực thi được đối chiếu trực tiếp trên cùng một bức ảnh đa vùng văn bản và 3 câu hỏi khác nhau (`Q1`, `Q2`, `Q3` sinh ra các bản đồ chú ý $\Phi$, $R$, $M_{\text{aff}}$, $A_Q$ và hạt giống ban đầu khác nhau):
- **Luồng Cached (Standard `OPT-03`):** Chạy liên tiếp `Q1`, `Q2`, `Q3` trên `pipeline_cached` (tính `_compute_working_maps` đúng 1 lần ở `Q1`, tái sử dụng ở `Q2` và `Q3`).
- **Luồng Uncached (Baseline đối chứng):** Chạy `Q1`, `Q2`, `Q3` trên `pipeline_uncached` có gọi `pipeline_uncached.reset_cache()` trước mỗi câu hỏi (ép tính lại `_compute_working_maps` ở cả 3 câu hỏi).

**Kết quả đối chiếu từng trường dữ liệu đầu ra trên mỗi câu hỏi $q \in \{Q_1, Q_2, Q_3\}$:**
- `seed_boxes` và `propagated_boxes` (danh sách toàn bộ các hộp vùng chọn): **Giống hệt 100%** (`assertEqual(res_c.seed_boxes, res_u.seed_boxes)`, `assertEqual(res_c.propagated_boxes, res_u.propagated_boxes)`).
- `seed_box` và `propagated_box` (hộp chính): **Giống hệt 100%**.
- `magnification`, `grid_dims`, `num_ patches`, `selected_k`: **Giống hệt 100%**.
- Giá trị pixel của từng mảnh cắt ROI (`crop_result.roi_tiles`): **Khớp bit-for-bit 100%** (`np.array_equal(np.asarray(tile_c), np.asarray(tile_u)) == True`).
- Đầu ra giải mã Stage 2 (`answer`, `raw_answer`, `num_steps`, `avg_js_divergence`, `avg_entropy`, `avg_alpha`): **Giống hệt 100%**.

Bên cạnh đó, [`test_F_repeated_use_and_immutability_zero_mutation_zero_clone`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt03_working_map_cache.py#L357-L410) chụp bản sao tham chiếu của cả 4 tensor `(opp_t, bg_smooth_t, stroke_energy_t, horiz_mod_t)` cùng địa chỉ bộ nhớ `data_ptr()` ngay sau `Q1`, sau đó chạy tiếp `Q2` và `Q3` (mỗi câu hỏi có $K \ge 2$ vùng hạt giống):
- `torch.equal(t_after_Q3, t_snapshot_Q1)` đạt `True` tuyệt đối cho cả 4 tensor (zero in-place mutation).
- `t_after_Q3.data_ptr() == ptr_Q1` giữ nguyên 100% và bộ đếm `torch.Tensor.clone` bằng `0` trên `working_maps` (zero defensive tensor cloning).

---

## 9. Thống kê Cache Hit/Miss và Số Lần Gọi Hàm Tính Toán (Cache Hit/Miss & Computation-Call Counts)

Kết quả đo đạc thực nghiệm từ [`TestOPT03WorkingMapCache`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt03_working_map_cache.py):

| Kịch bản kiểm thử | Số lượng ảnh ($M$) | Tổng số câu hỏi ($N$) | Số vùng hạt giống / câu hỏi ($K$) | Số lần gọi `_compute_working_maps` TRƯỚC `OPT-03` | Số lần gọi `_compute_working_maps` SAU `OPT-03` | Số lần Cache Miss | Số lần Cache Hit |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Test A:** Câu hỏi đầu tiên trên ảnh mới | 1 | 1 | $\ge 1$ | 1 | **1** | 1 | 0 |
| **Test B / E / F:** 3 câu hỏi khác nhau trên cùng 1 ảnh | 1 | 3 | $2 \text{–} 3$ | 3 | **1** | 1 | **2** |
| **Test C:** 2 câu hỏi trên Ảnh A $\rightarrow$ 2 câu hỏi trên Ảnh B | 2 | 4 | $2 \text{–} 3$ | 4 | **2** | 2 | **2** |
| **Test D:** 1 câu hỏi $\rightarrow$ `reset_cache()` $\rightarrow$ 1 câu hỏi (cùng ảnh) | 1 | 2 | $\ge 1$ | 2 | **2** | 2 | 0 |
| **Test H:** Chuỗi 4 ảnh khác nhau liên tiếp | 4 | 4 | $\ge 1$ | 4 | **4** | 4 | 0 |
| **Kỳ vọng trên ViTextVQA ($\sim 3.31$ QA/ảnh, 400 ảnh)** | 400 | 1,324 | $1 \text{–} 3$ | 1,324 | **400** | 400 | **924** |

- Ngay cả khi `propagate_multi` xử lý nhiều vùng ứng viên ($K > 1$) trong một câu hỏi, `_compute_working_maps(image)` vẫn chỉ được gọi **đúng 1 lần duy nhất cho mỗi bức ảnh** (giảm $69.8\%$ số lần tính toán working-map trên toàn bộ tập benchmark).

---

## 10. Trạng thái Git và Tóm tắt Diff (Git Status and Diff Summary)

**Kết quả `git status --short`:**
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

**Kết quả `git diff --stat` đối với các file mã nguồn của `OPT-03`:**
```text
 scripts/benchmark/sg_vcd/adaptive_propagator.py |  73 ++++++++++--
 scripts/benchmark/sg_vcd/pipeline.py            | 150 +++++++++++++++++++++---
 2 files changed, 200 insertions(+), 23 deletions(-)
```
*(Lưu ý: Thay đổi trên `pipeline.py` bao gồm cả phần tích hợp `OPT-06` đã được nghiệm thu trước đó và phần bổ sung `_cached_working_maps` / `_cached_working_maps_key` của `OPT-03`; file test mới [`tests/test_opt03_working_map_cache.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt03_working_map_cache.py) gồm 495 dòng kiểm thử độc lập).*

---

## 11. Xác nhận Tường minh Phạm vi Giới hạn (Explicit Confirmation of Unimplemented Tickets)

Xác nhận tường minh theo đúng Mục 1 của chỉ thị nhiệm vụ:
- **KHÔNG triển khai `TICKET-OPT-02` (Coordinated Memory Cleanup Policy):** Chưa thay đổi tần suất gọi `gc.collect()` hay `torch.cuda.empty_cache()` trong `evaluate_vitextvqa.py`, `pipeline.py`, hoặc `attention_extractor.py`.
- **KHÔNG triển khai `TICKET-OPT-01` (Batched Positive/Negative Decoding):** Chưa thay đổi vòng lặp giải mã từng bước trong `contrastive_decoder.py`.
- **KHÔNG triển khai `TICKET-OPT-04` (GPU Affinity Field Vectorization):** Chưa thay đổi thuật toán tính trường tương quan $M_{\text{aff}}$ trong `attention_extractor.py`.
- **KHÔNG nạp trọng số mô hình thật `5CD-AI/Vintern-1B-v3_5` và KHÔNG chạy Colab.**

---

## 12. Hạn chế và Phương án Khôi phục (Limitations and Rollback Considerations)

1. **Hạn chế (Limitations):**
   - `OPT-03` phát huy hiệu quả tối đa khi các câu hỏi của cùng một bức ảnh được đánh giá liên tiếp nhau (đây vốn là chế độ mặc định của [`evaluate_vitextvqa.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/evaluate_vitextvqa.py) sau khi nhóm câu hỏi theo `image_group_key` của `OPT-06`). Nếu người dùng gọi `pipeline.run()` theo thứ tự xen kẽ ảnh hoàn toàn ngẫu nhiên (`Img1 -> Img2 -> Img1`) hoặc gọi `pipeline.reset_cache()` sau mỗi câu hỏi, bộ nhớ đệm dung lượng 1 ảnh sẽ tự động tính lại trên mỗi lần chuyển ảnh để ưu tiên an toàn bộ nhớ VRAM.
   - Đánh giá định lượng mức giảm thời gian thực thi bằng mili-giây trên GPU thực tế (T4/L4 Colab) sẽ được đo tại Gate B sau khi các ticket tối ưu hóa còn lại hoàn tất.
2. **Phương án khôi phục (Rollback Considerations):**
   - Thiết kế hoàn toàn tương thích ngược: tham số `working_maps: Optional[Tuple[Any, ...]] = None` là tùy chọn ở mọi cấp (`BaseSeedPropagator`, `AdaptiveScalePropagator`). Nếu cần khôi phục (rollback), chỉ cần hoàn tác thay đổi trên [`scripts/benchmark/sg_vcd/adaptive_propagator.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/adaptive_propagator.py) và [`scripts/benchmark/sg_vcd/pipeline.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/scripts/benchmark/sg_vcd/pipeline.py) mà không ảnh hưởng tới `OPT-06` hay giao diện CLI của `evaluate_vitextvqa.py`.

---

## 13. Phán quyết Gate A (Gate A Verdict)

- **Phán quyết Gate A (Local Synthetic / Unit Validation):** **`PASS`**
  - Đã đáp ứng đầy đủ 8 yêu cầu kiểm thử (`Test A` $\rightarrow$ `Test H`) trong [`tests/test_opt03_working_map_cache.py`](file:///d:/Trường Đại Học Công Thương/Năm 4_HUIT/Các vấn đề hiện đại trong AI/VLM_Reasoning/tests/test_opt03_working_map_cache.py) (`8/8 OK`).
  - Bảo toàn tương đương số học 100% giữa chế độ có cache và không cache, không làm bẩn tensor lưu đệm, không sao chép tensor thừa, và an toàn tuyệt đối khi xảy ra ngoại lệ chuyển ảnh.

---

## 14. Trạng thái Sẵn sàng cho Gate B (Gate B Readiness & Outstanding Dependencies)

- **Tuyên bố trung thực về Gate B (Colab 20-QA Real-Model Integration):** **CHƯA THỰC THI TRÊN COLAB (NOT YET EXECUTED ON COLAB)** — Cần hoàn tất các điều kiện phụ thuộc dưới đây trước khi mở Gate B:
  1. **Kiểm định QA độc lập của Antigravity (Anti) và phê duyệt của Project Owner** đối với `TICKET-OPT-03` (Task 08).
  2. **Các ticket hiệu năng P1 còn lại theo lộ trình Small-First** (như `TICKET-OPT-02 — Coordinated Memory Cleanup Policy` và `TICKET-OPT-01 — Batched Positive/Negative Step-0 Prefill & Step-t Decoding`, theo quyết định mở cổng của Project Owner).
  3. **Đóng băng danh sách mẫu `20 QA` cố định (Frozen 20-QA Manifest)** cho Gate B trên Google Colab để đối chiếu trực tiếp tốc độ, bộ nhớ VRAM và độ chính xác thực tế của mô hình `5CD-AI/Vintern-1B-v3_5`.

