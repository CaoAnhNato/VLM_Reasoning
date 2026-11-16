# BÁO CÁO KIỂM ĐỊNH CHẤT LƯỢNG ĐỘC LẬP (ACCEPTANCE QA) — BẢN CỦNG CỐ `TICKET-OPT-06` (`TASK 07-FIX-R`)

**Mã tài liệu:** `TASK_07_FIX_OPT06_INDEPENDENT_ACCEPTANCE_QA`  
**Ngày thực hiện:** 2026-11-08  
**Người kiểm định độc lập (Independent QA):** Antigravity (Gemini 3.8 Medium)  
**Tác nhân triển khai được kiểm định:** 4argon (`TASK 07` + `TASK 07-FIX`)  
**Cấp thẩm định phương pháp & kiến trúc:** ChatGPT  
**Cấp phê duyệt nghiệm thu cuối cùng:** Project Owner  
**Quy tắc & Kỹ năng bắt buộc đã áp dụng:**
- [`.github/copilot-instructions.md`](../../.github/copilot-instructions.md)
- [`.github/skills/anti-output-quality-review/SKILL.md`](../../.github/skills/anti-output-quality-review/SKILL.md)
- [`.github/skills/research-plan-compliance-review/SKILL.md`](../../.github/skills/research-plan-compliance-review/SKILL.md)
- [`.github/skills/local-colab-experiment-workflow/SKILL.md`](../../.github/skills/local-colab-experiment-workflow/SKILL.md)

---

## 1. Kết Luận Điều Hành (Executive Verdict)

### **Phán quyết QA Kỹ thuật: `PASS` (ĐẠT — Đủ điều kiện nghiệm thu kỹ thuật chính thức cho `TICKET-OPT-06`)**

### Tóm tắt căn cứ phán quyết
1. **Xác nhận khắc phục triệt để 3 lỗi nhỏ (`M-01`, `M-02`, `M-03`):** Kiểm tra độc lập trên mã nguồn thực tế và diff (không phụ thuộc vào báo cáo của 4argon) xác nhận cả ba điểm yếu biên nêu tại [`docs/research/TASK_07R_OPT06_INDEPENDENT_QA.md`](TASK_07R_OPT06_INDEPENDENT_QA.md) đã được xử lý đúng yêu cầu kỹ thuật:
   - **`M-01` ([`_normalize_relative_image_path()`](../../scripts/benchmark/sg_vcd/feature_cache.py#L84-L177)):** Đường dẫn nằm trong `dataset_root` được trả về nguyên vẹn ở **toàn bộ độ sâu thư mục** ngay sau khi `relpath` thành công, không còn rơi xuống bước tìm neo `"data"` / `"results"` hay cắt cụt `parts[-3:]`. Nhánh dự phòng ngoài `dataset_root` cũng đã loại bỏ phép cắt `parts[-3:]`, ngăn chặn triệt để va chạm giữa các cây thư mục sâu có cùng tên tệp cơ sở (`basename`).
   - **`M-02` ([`build_image_cache_identity()`](../../scripts/benchmark/sg_vcd/feature_cache.py#L254-L385)):** Nhánh nhanh $O(1)$ khi `isinstance(image_key, ImageCacheIdentity)` vẫn giữ nguyên hiệu năng (`0` lần gọi `Image.resize`, `0` lần gọi `compute_image_content_fingerprint`), đồng thời kiểm tra nghiêm ngặt kích thước `(width, height)` và mọi siêu dữ liệu định danh tường minh do nơi gọi truyền vào (`dataset_version`, `dataset_namespace`, `image_id`, `image_path`, `content_fingerprint`). Mọi mâu thuẫn đều kích hoạt ngoại lệ `ValueError` tức thời thay vì âm thầm tạo lại khóa hoặc dùng khóa sai lệch.
   - **`M-03` ([`_sample_image_group_key()`](../../scripts/benchmark/evaluate_vitextvqa.py#L400-L410)):** Toàn bộ 5 vị trí gom nhóm/lấy mẫu/phân mảnh theo ảnh trong [`scripts/benchmark/evaluate_vitextvqa.py`](../../scripts/benchmark/evaluate_vitextvqa.py) (`load_benchmark_dataset()`, `group_samples_by_image()`, `--test`, `--test-images`, và phân mảnh đa tiến trình `img_to_idx`) đều sử dụng chung hàm `_sample_image_group_key(s)`.
2. **Bằng chứng thực thi kiểm thử tập trung:** Cả 3 bộ kiểm thử trọng tâm (`tests.test_opt06_image_cache_identity`: **11/11 PASS**, `tests.test_multi_worker_sharding`: **5/5 PASS**, `tests.test_adversarial_sharding`: **11/11 PASS**) cùng các kịch bản thăm dò độc lập (probe) đều đạt 100% với mã thoát `0`.
3. **Tuân thủ bất biến nghiên cứu và giới hạn phạm vi:** Không có bất kỳ thay đổi nào xâm phạm phương pháp SG-VCD 2 giai đoạn, CERV, công thức điểm số, tập dữ liệu 400 ảnh / 1.222 câu hỏi, hay các vé tối ưu hóa chưa được cấp phép (`OPT-03`, `OPT-02`, `OPT-01`, `OPT-04`). Không tải trọng số Vintern thật và không chạy suy luận mô hình thật trên máy cục bộ.

---

## 2. Bằng Chứng Kiểm Định Độc Lập Cho `M-01`, `M-02`, `M-03`

### 2.1 Kiểm định `M-01` — Chuẩn hóa đường dẫn tương đối (`_normalize_relative_image_path`)

- **Vị trí mã nguồn kiểm tra:** [`scripts/benchmark/sg_vcd/feature_cache.py#L84-L177`](../../scripts/benchmark/sg_vcd/feature_cache.py#L84-L177).
- **Phân tích luồng thực thi thực tế:**
  1. **Đường dẫn nằm trong `dataset_root` (dòng `113–151`):**
     - Hàm trích xuất ký tự ổ đĩa (`raw_drive`, `root_drive`) và kiểm tra tính tương thích tương đối/tuyệt đối (`raw_drive == root_drive and raw_is_abs == root_is_abs`).
     - Tính `posixpath.relpath` (hoặc dự phòng `os.path.relpath(raw, str(dataset_root).strip()).replace("\\", "/")`).
     - Nếu `in_root_rel` hợp lệ (`!= "."` và không bắt đầu bằng `".."`), hàm tách `rel_parts` và **trả về trực tiếp** `(normalized_in_root, len(rel_parts) > 1)` tại dòng `151`.
     - **Hệ quả kỹ thuật:** Các đường dẫn trong `dataset_root` giữ nguyên 100% độ sâu thư mục; không bị cắt bởi neo `"data"` / `"results"` nội bộ và không bị cắt cụt `parts[-3:]`.
  2. **Đường dẫn nằm ngoài `dataset_root` hoặc không truyền `dataset_root` (dòng `153–176`):**
     - Loại bỏ tiền tố ổ đĩa Windows (`norm[2:]` nếu `norm[1] == ":"`) và dấu `/` đầu chuỗi.
     - Nếu tồn tại phân đoạn `"data"` hoặc `"results"`, neo từ phân đoạn đó đến hết đường dẫn (`parts = parts[idx:]`).
     - Nếu không có `"data"` hoặc `"results"`, **giữ nguyên toàn bộ `parts`** (`"/".join(parts)`), hoàn toàn không còn lệnh cắt `parts[-3:]`.
  3. **Xử lý ký tự phân cách Windows (`\`) và POSIX (`/`):**
     - Mọi dấu `\` được chuẩn hóa về `/` ngay từ dòng `111` (`norm_raw = raw.replace("\\", "/")`) và dòng `114` (`norm_root = ...replace("\\", "/")`).
  4. **Xử lý ký tự ổ đĩa (Drive Letters):**
     - Khi truyền `dataset_root="D:\\proj"` và `image_path="C:\\proj\\a\\b\\22.jpg"`, do `raw_drive ("c:") != root_drive ("d:")` và `os.path.relpath` ném `ValueError` giữa hai ổ đĩa khác nhau trên Windows, đường dẫn trên ổ `C:` không bị nhận nhầm là nằm trong `D:\proj` mà chuyển sang nhánh ngoài `dataset_root` (`('proj/a/b/22.jpg', True)` so với `('a/b/22.jpg', True)` của `D:\proj\a\b\22.jpg`).
     - *Lưu ý kỹ thuật nhỏ (Informational `I-01`)*: Trong trường hợp đặc biệt khi `dataset_root=None` (hoặc cả hai đường dẫn đều nằm ngoài `dataset_root`) **và** hai đường dẫn nằm trên hai ổ đĩa Windows khác nhau nhưng có cấu trúc thư mục giống hệt nhau và không chứa `"data"`/`"results"` (ví dụ `C:\mount\a\b\22.jpg` và `D:\mount\a\b\22.jpg`), bước lược bỏ ký tự ổ đĩa ở dòng `155–156` (nhằm đảm bảo tính khả chuyển giữa Windows `D:\...` và POSIX `/...`) sẽ chuẩn hóa cả hai thành `'mount/a/b/22.jpg'`. Tuy nhiên, trong thực tế vận hành có `image_id`, `dataset_version`, và `os.stat` riêng cho từng tệp trên đĩa nên không gây va chạm trên tập dữ liệu dự án (xem Mục 4).

- **Kết quả chạy Probe độc lập cho `M-01`:**
  ```text
  M01 in-root deep: ('a/b/c/d/e/22.jpg', True) ('x/b/c/d/e/22.jpg', True)
  M01 win vs posix in-root: ('a/b/c/d/22.jpg', True) ('a/b/c/d/22.jpg', True)
  M01 diff drive vs root: ('proj/a/b/22.jpg', True)
  M01 diff drive no root: ('mount/a/b/22.jpg', True) ('mount/a/b/22.jpg', True)
  ```
- **Kiểm tra unit test `test_M01` thực tế:** [`tests/test_opt06_image_cache_identity.py#L593-L662`](../../tests/test_opt06_image_cache_identity.py#L593-L662) kiểm tra đầy đủ 6 kịch bản: đường dẫn sâu trong `dataset_root` (`root/x/p/q/r/22.jpg` vs `root/y/p/q/r/22.jpg`), đường dẫn trong `dataset_root` có chứa thư mục con tên `data` (`exp_a/data/imgs/22.jpg` vs `exp_b/data/imgs/22.jpg`), đường dẫn sâu khi `dataset_root=None` (`/mnt/ds1/a/b/c/22.jpg` vs `/mnt/ds2/a/b/c/22.jpg`), đường dẫn ngoài `dataset_root`, tương thích neo `data/` và `results/`, đồng nhất Windows/POSIX, và tên tệp đơn lẻ (`("22.jpg", False)`).

---

### 2.2 Kiểm định `M-02` — Xác thực tính nhất quán của `ImageCacheIdentity` tính sẵn (`build_image_cache_identity`)

- **Vị trí mã nguồn kiểm tra:** [`scripts/benchmark/sg_vcd/feature_cache.py#L254-L385`](../../scripts/benchmark/sg_vcd/feature_cache.py#L254-L385).
- **Phân tích luồng thực thi thực tế:**
  1. **Kiểm tra xung đột giữa `dataset_version` và `dataset_namespace` (dòng `284–290`):** Nếu cả hai cùng được truyền vào với giá trị khác nhau (sau khi `.strip()`), hàm lập tức ném `ValueError` ngay cả trước khi vào nhánh `isinstance(image_key, ImageCacheIdentity)`.
  2. **Xác thực trên nhánh nhanh `isinstance(image_key, ImageCacheIdentity)` (dòng `296–344`):**
     - **Kích thước ảnh (dòng `297–301`):** Nếu `image_key.width != w or image_key.height != h`, ném `ValueError` ngay lập tức (không còn âm thầm tạo lại khóa mới như bản `TASK 07`).
     - **`dataset_version` / `dataset_namespace` (dòng `302–308`):** Nếu nơi gọi truyền tường minh `ds_raw` và `image_key.dataset_version != expected_ds`, ném `ValueError`.
     - **`image_id` (dòng `310–317`):** Tách `key_tokens = image_key.image_identity.split("|")`. Nếu nơi gọi truyền `image_id` tường minh mà `image_key` không có token `id=...` hoặc `id_tokens[0] != f"id={str(raw_id).strip()}"`, ném `ValueError`.
     - **`image_path` (dòng `318–330`):** Nếu nơi gọi truyền `image_path` tường minh, chuẩn hóa qua `_normalize_relative_image_path(image_path, dataset_root=dataset_root)` và đối chiếu với token `path=...` trong `image_key.image_identity`; nếu thiếu hoặc khác biệt, ném `ValueError`.
     - **`content_fingerprint` (dòng `331–343`):** Nếu nơi gọi truyền `content_fingerprint` tường minh, chuẩn hóa tiền tố (`explicit:`, `probe:`, `thumb:`) và kiểm tra `image_key.content_fingerprint not in (expected_fp, raw_fp)`; nếu mâu thuẫn, ném `ValueError`.
     - **Trả về $O(1)$ (dòng `344`):** Khi mọi kiểm tra đều hợp lệ (hoặc khi nơi gọi chỉ truyền `image_key=pre_key` và để trống các tham số tùy chọn), trả về chính đối tượng `image_key` (`return image_key`) mà **không** gọi `compute_image_content_fingerprint()` và **không** gọi `Image.resize()`.
- **Kết quả chạy Probe độc lập cho `M-02`:**
  ```text
  M02 fast-path resize_calls= 0 fp_calls= 0 same_obj= True
  ```
- **Kiểm tra unit test `test_M02` thực tế:** [`tests/test_opt06_image_cache_identity.py#L664-L730`](../../tests/test_opt06_image_cache_identity.py#L664-L730) kiểm tra đầy đủ: trả về cùng thực thể (`assertIs`) khi lược bỏ hoặc truyền khớp tham số tùy chọn, và ném `ValueError` trong cả 6 trường hợp mâu thuẫn (`image_id`, `dataset_version`, `dataset_namespace`, `dataset_version` xung đột `dataset_namespace`, `image_path`, `content_fingerprint`, và lệch kích thước `(width, height)`).

---

### 2.3 Kiểm định `M-03` — Khóa gom nhóm ảnh thống nhất (`_sample_image_group_key`)

- **Vị trí mã nguồn kiểm tra:** [`scripts/benchmark/evaluate_vitextvqa.py#L396-L423`](../../scripts/benchmark/evaluate_vitextvqa.py#L396-L423) và [`#L953-L993`](../../scripts/benchmark/evaluate_vitextvqa.py#L953-L993).
- **Đối chiếu tất cả các vị trí gom nhóm/lấy mẫu/phân mảnh trong `evaluate_vitextvqa.py`:**

| STT | Chức năng / Đường dẫn mã nguồn | Dòng thực tế | Biểu thức khóa sử dụng | Trạng thái |
| :-: | :--- | :-: | :--- | :-: |
| 1 | Định nghĩa hàm [`_sample_image_group_key(sample)`](../../scripts/benchmark/evaluate_vitextvqa.py#L400-L410) | `400–410` | `f"ds={ds_ver or ''}\|id={img_id if img_id is not None else ''}\|path={path_or_name}"` (hoặc `str(path_or_name)` nếu thiếu cả `ds_ver` và `img_id`) | **Thống nhất** |
| 2 | Log đếm số ảnh duy nhất trong [`load_benchmark_dataset()`](../../scripts/benchmark/evaluate_vitextvqa.py#L396) | `396` | `len({_sample_image_group_key(s) for s in samples})` | **Thống nhất** |
| 3 | Gom nhóm câu hỏi theo ảnh trong [`group_samples_by_image()`](../../scripts/benchmark/evaluate_vitextvqa.py#L413-L423) | `421` | `group_key = _sample_image_group_key(s)` | **Thống nhất** |
| 4 | Chế độ smoke test [`--test`](../../scripts/benchmark/evaluate_vitextvqa.py#L953-L961) | `957` | `img_key = _sample_image_group_key(s)` | **Thống nhất** |
| 5 | Chế độ giới hạn số ảnh [`--test-images`](../../scripts/benchmark/evaluate_vitextvqa.py#L962-L974) | `966`, `973` | `img_key = _sample_image_group_key(s)` | **Thống nhất** |
| 6 | Phân mảnh đa tiến trình [`img_to_idx`](../../scripts/benchmark/evaluate_vitextvqa.py#L984-L993) (`--num-workers`, `--worker-id`) | `988`, `991`, `992` | `img_to_idx[_sample_image_group_key(s)] % args.num_workers == args.worker_id` | **Thống nhất** |

- **Kết quả kiểm thử tổng hợp độc lập (6 ảnh logic cùng mang tên `22.jpg` trên 3 thư mục `dir_1..3` $\times$ 2 namespace `ns_alpha`, `ns_beta`, mỗi ảnh 3 câu hỏi = 18 QA):**
  ```text
  Total logical image groups: 6 Total QAs: 18
  num_workers=1: group counts=[6], QA counts=[18] -> DISJOINT & COMPLETE
  num_workers=2: group counts=[3, 3], QA counts=[9, 9] -> DISJOINT & COMPLETE
  num_workers=3: group counts=[2, 2, 2], QA counts=[6, 6, 6] -> DISJOINT & COMPLETE
  num_workers=4: group counts=[2, 2, 1, 1], QA counts=[6, 6, 3, 3] -> DISJOINT & COMPLETE
  ```
  Mọi cấu hình `num_workers` đều phân hoạch các nhóm ảnh hoàn toàn rời nhau (`isdisjoint == True`, không có bất kỳ nhóm ảnh nào bị chia cắt hoặc trùng lặp giữa các worker) và bao phủ đúng 18/18 câu hỏi.

- **Ghi chú đối chiếu độc lập với báo cáo của 4argon (`I-02` — Informational):**
  - Trong [`docs/research/TASK_07_FIX_OPT06_HARDENING_REPORT.md`](TASK_07_FIX_OPT06_HARDENING_REPORT.md) (Mục 1, dòng 21), 4argon mô tả chữ ký của hàm là `_sample_image_group_key(sample: Dict[str, Any]) -> Tuple[str, str]`, nhưng trong mã nguồn thực tế ([`evaluate_vitextvqa.py#L400`](../../scripts/benchmark/evaluate_vitextvqa.py#L400)), hàm có kiểu trả về là `str` (`f"ds=...|id=...|path=..."`). Kiểu `str` thực tế trong mã nguồn bao gồm cả 3 thành phần `(dataset_version, image_id, path_or_name)`, chặt chẽ hơn mô tả `Tuple[str, str]` trong văn bản báo cáo.
  - Ngoài ra, số dòng ghi trong Mục 2 của báo cáo 4argon lệch khoảng 25–35 dòng so với tệp thực tế (ví dụ `feature_cache.py` dòng `281–345` thực tế so với `322–385` trong báo cáo; `tests/test_opt06_image_cache_identity.py` dòng `593–877` thực tế so với `626–865` trong báo cáo). Sự khác biệt nhỏ trong tài liệu thuyết minh này không ảnh hưởng đến mã nguồn hay kiểm thử.

---

## 3. Kết Quả Thực Thi Kiểm Thử Thực Tế (Focused Test Execution)

Tuân thủ Mục 7 của yêu cầu nhiệm vụ, quá trình QA chỉ chạy các bộ kiểm thử trọng tâm liên quan trực tiếp đến định danh bộ nhớ đệm và phân mảnh đa tiến trình bằng môi trường ảo của dự án (`..\Document AI\VLM_Reasoning\venv\Scripts\python.exe`). Không chạy lại không cần thiết `tests.test_sg_vcd` vì qua kiểm tra mã nguồn và `git diff`, không có bất kỳ thay đổi nào trong `TASK 07-FIX` tác động tới `dynamic_cd.py`, `bbox_selector.py`, `roi_cropper.py`, `adaptive_propagator.py`, hay `SyntacticSpanRefiner`.

### Bảng tổng hợp kết quả thực thi

| STT | Lệnh thực thi | Tổng số test | Đạt (Passed) | Lỗi (Failed) | Bỏ qua (Skipped) | Thời gian | Mã thoát (Exit Code) |
| :-: | :--- | :-: | :-: | :-: | :-: | :-: | :-: |
| 1 | `& "..\Document AI\VLM_Reasoning\venv\Scripts\python.exe" -m unittest -v tests.test_opt06_image_cache_identity` | **11** | **11** | 0 | 0 | `3.523s` | `0` |
| 2 | `& "..\Document AI\VLM_Reasoning\venv\Scripts\python.exe" -m unittest -v tests.test_multi_worker_sharding` | **5** | **5** | 0 | 0 | `0.049s` | `0` |
| 3 | `& "..\Document AI\VLM_Reasoning\venv\Scripts\python.exe" -m unittest -v tests.test_adversarial_sharding` | **11** | **11** | 0 | 0 | `41.959s` | `0` |
| **Tổng** | **3 bộ kiểm thử trọng tâm được thực thi** | **27** | **27** | **0** | **0** | **`45.531s`** | **`0`** |

### Chi tiết đầu ra thực thi từng bộ kiểm thử

#### 1. `tests.test_opt06_image_cache_identity`
```text
test_A_same_image_identity_reuses_cache_and_avoids_redundant_thumbnail_resize ... ok
test_B_different_image_identities_same_dimensions_and_pixels_do_not_collide ... ok
test_C_same_basename_different_directories_do_not_collide ... ok
test_D_same_image_id_different_dataset_namespaces_do_not_collide ... ok
test_E_changed_image_content_or_fingerprint_invalidates_cached_state ... ok
test_F_missing_stable_identity_falls_back_to_content_hash_with_fallback_flag ... ok
test_G_backward_compatibility_positional_callers ... ok
test_H_cache_lifecycle_invalidation_and_fresh_run_isolation ... ok
test_M01_relative_path_normalization_deep_paths_in_root_and_fallback ... ok
test_M02_precomputed_identity_consistency_validation ... ok
test_M03_unified_image_grouping_sampling_and_sharding ... ok

----------------------------------------------------------------------
Ran 11 tests in 3.523s

OK
EXIT_OPT06=0
```

#### 2. `tests.test_multi_worker_sharding`
```text
test_01_deterministic_sharding_partitioning ... ok
test_02_per_shard_checkpoint_manager_isolation ... ok
test_03_atomic_aggregation_merges_and_sorts_all_shards ... ok
test_01_stage1_diacritic_vertical_padding ... ok
test_02_stage2_conversational_preamble_stripping ... ok

----------------------------------------------------------------------
Ran 5 tests in 0.049s

OK
EXIT_MW=0
```

#### 3. `tests.test_adversarial_sharding`
```text
test_adv_10_deterministic_equivalence_1w_vs_2w_vs_4w ... ok
test_adv_11_prime_number_shards_uneven_partition ... ok
test_adv_01_all_shards_missing ... ok
test_adv_02_sparse_missing_shards ... ok
test_adv_03_zero_byte_and_whitespace_shards ... ok
test_adv_04_corrupted_and_truncated_jsonl ... ok
test_adv_05_fallback_to_checkpoint_json_when_jsonl_missing ... ok
test_adv_06_reverse_order_completion ... ok
test_adv_07_shuffled_intra_shard_ordering ... ok
test_adv_08_cross_shard_duplicate_keys ... ok
test_adv_09_robustness_when_global_index_missing_or_none ... ok

----------------------------------------------------------------------
Ran 11 tests in 41.959s

OK
EXIT_ADV=0
```

---

## 4. Các Giới Hạn Còn Tồn Tại (Remaining Limitations)

1. **Giới hạn bản chất của cơ chế Fingerprint trong Stable Identity Mode (Bảo lưu tường minh theo Mục 8):**
   - Trong chế độ định danh ổn định (`is_fallback=False`), [`compute_image_content_fingerprint()`](../../scripts/benchmark/sg_vcd/feature_cache.py#L204-L251) kết hợp kích thước `(w, h)`, `image.mode`, siêu dữ liệu tệp trên đĩa `os.stat` (`st_size`, `st_mtime_ns`) và mẫu 9 điểm ảnh cố định (`_sample_nine_point_pixels`: 4 góc, 4 trung điểm cạnh, 1 tâm ảnh) khi không có `content_fingerprint` tường minh từ manifest.
   - Cơ chế này là **cơ chế phát hiện thay đổi dựa trên nỗ lực tốt nhất (best-effort invalidation mechanism)** nhằm triệt tiêu chi phí CPU resize ảnh `(16 x 16)` ở mỗi câu hỏi, **KHÔNG phải là hàm băm mật mã toàn bộ điểm ảnh (cryptographic full-image digest)**.
   - Nếu một đối tượng `PIL.Image` bị chỉnh sửa trực tiếp trong bộ nhớ RAM tại các tọa độ nằm ngoài 9 điểm lấy mẫu (hoặc tệp trên đĩa bị sửa nội dung nhưng giữ nguyên kích thước byte `st_size`, khôi phục nguyên trạng `st_mtime_ns` bằng `os.utime`, và không chạm tới 9 điểm lấy mẫu) mà người gọi không truyền `content_fingerprint` mới, thì `probe:...` sẽ không thay đổi. Tương tự, chế độ dự phòng (`use_thumbnail_bytes=True`) lấy mẫu lưới điểm `16 x 16` bằng `NEAREST` nên cũng không phải là hàm băm mật mã toàn ảnh.
   - **Đánh giá mức độ chấp nhận được:** Giới hạn này **hoàn toàn chấp nhận được** đối với vòng đời bộ nhớ đệm theo từng lần chạy (`per-run cache lifecycle`) và tập dữ liệu 400 ảnh đã kiểm toán vì:
     1. Bộ nhớ đệm (`default_feature_cache`, `pipeline.feature_cache`, `extractor._cached_img_state`) được xóa sạch khi bắt đầu mỗi lần chạy [`run_benchmark()`](../../scripts/benchmark/evaluate_vitextvqa.py#L1046-L1058) và mỗi worker chạy trong một tiến trình hệ điều hành độc lập.
     2. Trong suốt một lần chạy đánh giá, các tệp ảnh ViTextVQA là tĩnh (chỉ đọc) và không bị sửa đổi điểm ảnh giữa chừng trong RAM.
     3. Kiểm toán toàn bộ 400 ảnh / 1.222 câu hỏi của [`answer_bboxes_400_verified.json`](../../results/swintextspotter_auto_annot/answer_bboxes_400_verified.json) đã chứng minh 400/400 ảnh có `image_id` phân biệt, đường dẫn chuẩn hóa phân biệt và khóa `ImageCacheIdentity` phân biệt 100%.

2. **Lược bỏ ký tự ổ đĩa Windows ở nhánh dự phòng ngoài `dataset_root` (`I-01` — Informational):**
   - Tại [`_normalize_relative_image_path()`](../../scripts/benchmark/sg_vcd/feature_cache.py#L155-L156), khi `dataset_root=None` (hoặc khi đường dẫn nằm ngoài `dataset_root`) và không có neo `"data"` / `"results"`, tiền tố ổ đĩa `C:` / `D:` được lược bỏ để giữ tính tương thích đa nền tảng Windows/POSIX. Nếu có hai tệp trùng tên và trùng toàn bộ đường dẫn thư mục trên hai ổ đĩa khác nhau (`C:\mount\a\b\22.jpg` và `D:\mount\a\b\22.jpg`), cùng được gọi mà không có `dataset_root`, không có `image_id`, cùng kích thước và cùng fingerprint, phần `path=` sẽ trùng nhau. Trường hợp này không xảy ra trong quy trình thực nghiệm của dự án (vốn luôn truyền `dataset_root=PROJECT_ROOT`, `image_id`, và `dataset_version`).

---

## 5. Xác Minh Phạm Vi Git và Trạng Thái Kho Mã Nguồn (Git and Scope Verification)

### 5.1 Kết quả `git status --short`
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

### 5.2 Kết quả `git diff --stat` (Tích lũy so với `HEAD`)
```text
 scripts/benchmark/evaluate_vitextvqa.py         | 119 ++++--
 scripts/benchmark/sg_vcd/__init__.py            |   3 +
 scripts/benchmark/sg_vcd/attention_extractor.py |  86 ++++-
 scripts/benchmark/sg_vcd/feature_cache.py       | 458 ++++++++++++++++++++++--
 scripts/benchmark/sg_vcd/pipeline.py            |  82 ++++-
 5 files changed, 683 insertions(+), 65 deletions(-)
```

### 5.3 Phân tách rõ phần triển khai `OPT-06` gốc (`TASK 07`) và phần củng cố `M-01`/`M-02`/`M-03` (`TASK 07-FIX`)

| Tệp tin | Thay đổi tích lũy từ `TASK 07` (Triển khai `OPT-06` gốc) | Thay đổi tăng thêm trong `TASK 07-FIX` (`M-01`, `M-02`, `M-03`) |
| :--- | :--- | :--- |
| [`scripts/benchmark/sg_vcd/feature_cache.py`](../../scripts/benchmark/sg_vcd/feature_cache.py) | (`+363 / -2` dòng ở `TASK 07`): Thêm `ImageCacheIdentity`, `_normalize_relative_image_path`, `_sample_nine_point_pixels`, `compute_image_content_fingerprint`, `build_image_cache_identity`, mở rộng `VisualFeatureCache.compute_image_key` và `get_or_compute`. | (`+95` dòng ròng ở `TASK 07-FIX`):<br>• **`M-01` (dòng `84–177`):** Trả về ngay đường dẫn tương đối đầy đủ khi nằm trong `dataset_root`; bỏ `parts[-3:]` ở nhánh dự phòng.<br>• **`M-02` (dòng `281–345`):** Kiểm tra xung đột `dataset_version` vs `dataset_namespace` và xác thực kích thước, `dataset_version`, `image_id`, `image_path`, `content_fingerprint` khi `isinstance(image_key, ImageCacheIdentity)`, ném `ValueError` khi mâu thuẫn. |
| [`scripts/benchmark/evaluate_vitextvqa.py`](../../scripts/benchmark/evaluate_vitextvqa.py) | (`+82 / -11` dòng ở `TASK 07`): Trích xuất `dataset_version`, `image_id`, `content_fingerprint` khi nạp dataset; tạo `image_cache_key` một lần cho mỗi nhóm ảnh và truyền vào `extract_visual_features()` / `pipeline.run()`; xóa cache đầu mỗi lần chạy. | (`+37 / -15` dòng ở `TASK 07-FIX`):<br>• **`M-03` (dòng `396–423`, `953–993`):** Bổ sung hàm `_sample_image_group_key(sample)` và áp dụng đồng bộ tại `load_benchmark_dataset()`, `group_samples_by_image()`, `--test`, `--test-images`, và phân mảnh đa tiến trình `img_to_idx`. |
| [`scripts/benchmark/sg_vcd/attention_extractor.py`](../../scripts/benchmark/sg_vcd/attention_extractor.py) | (`+86 / -12` dòng ở `TASK 07`): Thêm `clear_cache()`, nâng cấp `_make_image_key()` và `extract()` nhận `image_key` cùng các tham số định danh tùy chọn. | **Không thay đổi trong `TASK 07-FIX`** (`0` dòng thay đổi). |
| [`scripts/benchmark/sg_vcd/pipeline.py`](../../scripts/benchmark/sg_vcd/pipeline.py) | (`+82 / -10` dòng ở `TASK 07`): Khởi tạo `VisualFeatureCache` riêng cho mỗi thực thể pipeline (nếu không truyền vào), thêm `reset_cache()`, truyền `resolved_image_key` qua `extract_visual_features()` và `extractor.extract()`. | **Không thay đổi trong `TASK 07-FIX`** (`0` dòng thay đổi). |
| [`scripts/benchmark/sg_vcd/__init__.py`](../../scripts/benchmark/sg_vcd/__init__.py) | (`+3` dòng ở `TASK 07`): Xuất `ImageCacheIdentity`, `build_image_cache_identity`, `compute_image_content_fingerprint`. | **Không thay đổi trong `TASK 07-FIX`** (`0` dòng thay đổi). |
| [`tests/test_opt06_image_cache_identity.py`](../../tests/test_opt06_image_cache_identity.py) | Tạo mới tệp kiểm thử với 8 bài test `test_A` .. `test_H` (`591` dòng). | Bổ sung 3 bài test mới `test_M01_*`, `test_M02_*`, `test_M03_*` (dòng `593–877`, tổng cộng `881` dòng). |

- **Xác nhận không xâm phạm phạm vi:** Không có tệp nào ngoài phạm vi `OPT-06` bị sửa đổi. Không có thay đổi nào được stage (`git add`), commit hay push.

---

## 6. Đánh Giá Điều Kiện Nghiệm Thu Chính Thức Cho `OPT-06`

- **Trạng thái kỹ thuật:** **ĐỦ ĐIỀU KIỆN NGHIỆM THU KỸ THUẬT CHÍNH THỨC (`PASS`)**.
- Mọi yêu cầu kỹ thuật của `TICKET-OPT-06` trong [`docs/research/PERFORMANCE_OPTIMIZATION_HANDOFF_v1.md`](PERFORMANCE_OPTIMIZATION_HANDOFF_v1.md) và cả 3 yêu cầu củng cố `M-01`, `M-02`, `M-03` trong [`docs/research/TASK_07R_OPT06_INDEPENDENT_QA.md`](TASK_07R_OPT06_INDEPENDENT_QA.md) đều đã được hoàn thành và kiểm chứng độc lập bằng mã nguồn và kiểm thử thực tế.
- Theo quy trình quản trị dự án ([`.github/copilot-instructions.md`](../../.github/copilot-instructions.md)), báo cáo QA độc lập này xác nhận phần triển khai đạt chuẩn kỹ thuật để **ChatGPT** thông qua về mặt phương pháp/kiến trúc và **Project Owner** ra quyết định nghiệm thu chính thức (`Final Acceptance`).

---

## 7. Đánh Giá Điều Kiện Bắt Đầu An Toàn Cho `OPT-03`

- **Trạng thái sẵn sàng kỹ thuật cho `TICKET-OPT-03` (Image-Level Working-Map Caching):** **SẴN SÀNG VÀ AN TOÀN VỀ MẶT KỸ THUẬT ĐỂ BẮT ĐẦU**.
- **Căn cứ kỹ thuật:**
  1. Theo sơ đồ phụ thuộc tại [`docs/research/PERFORMANCE_OPTIMIZATION_HANDOFF_v1.md#L70-L76`](PERFORMANCE_OPTIMIZATION_HANDOFF_v1.md#L70-L76), `TICKET-OPT-06` là điều kiện tiên quyết duy nhất của `TICKET-OPT-03`.
  2. Trong [`SGVCDPipeline.run()`](../../scripts/benchmark/sg_vcd/pipeline.py#L799-L808), khóa định danh ảnh chuẩn hóa `resolved_image_key: ImageCacheIdentity` đã được tính sẵn một lần duy nhất ở đầu hàm `run()` và hàm [`SGVCDPipeline.reset_cache()`](../../scripts/benchmark/sg_vcd/pipeline.py#L357-L369) đã được thiết lập để giải phóng bộ nhớ đệm cấp ảnh giữa các lần chạy.
  3. Do đó, `TICKET-OPT-03` có thể tái sử dụng trực tiếp `resolved_image_key` để kiểm tra tính hợp lệ của `_cached_working_maps` (`self._cached_working_maps_key == resolved_image_key`) và xóa `_cached_working_maps` trong `reset_cache()` cũng như khi chuyển sang ảnh mới mà không có nguy cơ rò rỉ trạng thái hay va chạm khóa giữa các ảnh trùng tên.
- **Lưu ý quản trị:** Việc bắt đầu triển khai mã nguồn cho `TICKET-OPT-03` sẽ được thực hiện ngay khi có lệnh giao việc/phê duyệt từ **Project Owner**.

