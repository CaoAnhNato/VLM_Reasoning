# BÁO CÁO THỰC NGHIỆM PROFILING BENCHMARK VLM TRÊN TẬP DỮ LIỆU VITEXTVQA

Tài liệu này ghi nhận đặc tả chi tiết cấu hình phần cứng, hạ tầng tính toán đám mây và kết quả đo đạc sơ bộ (profiling) trên tập con 100 mẫu thử nghiệm từ bộ dữ liệu **ViTextVQA**, phục vụ việc đánh giá năng lực suy luận và lập dự toán tài nguyên cho bài toán Zero-shot Visual Question Answering tiếng Việt.

---

## 1. ĐẶC TẢ CẤU HÌNH PHẦN CỨNG MÔI TRƯỜNG THỰC NGHIỆM BASELINE

Các phiên đo đạc hiệu năng (profiling runs) cho 7 mô hình Vision-Language Models (VLMs) được thực hiện độc lập trên cùng một phiên máy chủ ảo GPU chuyên dụng (Cloud Instance) tại nền tảng **Vast.ai** nhằm bảo đảm tính nhất quán và kiểm soát sai số thực nghiệm.

### 1.1. Hạ tầng Tính toán GPU
- **Card đồ họa (GPU):** 1x NVIDIA GeForce RTX 5060 Ti
- **Dung lượng bộ nhớ video (VRAM):** 16 GB GDDR7 (16,384 MB)
- **Băng thông bộ nhớ (Memory Bandwidth):** 367.3 GB/s
- **Năng lực tính toán dấu phẩy động (Compute Performance):** 22.8 TFLOPS (FP16/BF16 Tensor Cores)
- **Chỉ số hiệu năng học sâu thực tế (DLPerf Benchmark):** 38.3
- **Giao tiếp Bus:** PCI-e Gen 4.0 x16

### 1.2. Hạ tầng Hệ thống Máy chủ (Host System)
- **Bộ vi xử lý (CPU):** Intel® Core™ i5-13500 (Kiến trúc Raptor Lake, tiến trình Intel 7, phân bổ 10 cores / 20 threads, xung nhịp cơ sở 2.5 GHz, Turbo Boost đơn nhân đạt 4.8 GHz)
- **Bộ nhớ trong (Host RAM):** 32.1 GB DDR5
- **Ổ lưu trữ (Storage):** 1TB NVMe SSD Western Digital WD_BLACK SN7100
  - *Tốc độ đọc tuần tự kiểm tra (Disk Read Bandwidth):* 3,288 MB/s
  - *Dung lượng đĩa khả dụng:* ~35 GB phân vùng làm việc `/workspace`
- **Mã định danh máy chủ (Cloud Metadata):**
  - *Instance ID:* `#55412268`
  - *Host ID:* `#516546`
  - *Đơn giá thuê:* **$0.142 / giờ** ($0.100 GPU compute + $0.042 disk storage)

### 1.3. Môi trường Phần mềm & Frameworks
- **Hệ điều hành:** Linux x86_64 (Ubuntu 22.04 LTS Kernel 6.8.0)
- **Python Runtime:** Python 3.12.3 (Virtual Environment `/venv/main`)
- **CUDA Runtime / Driver:** CUDA 12.4 / Driver Version 565.57.01
- **Deep Learning Libraries:**
  - `torch`: 2.5.1+cu124 (kích hoạt PyTorch SDPA & Flash-Attention)
  - `transformers`: 4.45.0
  - `tokenizers`: 0.22.2
  - `accelerate`: 0.34.2
  - `unsloth`: 2024.10.3 (hỗ trợ 4-bit dynamic quant via bitsandbytes)

---

## 2. TỔNG HỢP KẾT QUẢ ĐO ĐẠC PROFILING (100 MẪU TEST VITEXTVQA)

Dữ liệu được trích xuất trực tiếp từ tệp [profiling_summary.csv](file:///c:/Users/Admin/HUIT - Học Tập/Năm 4/Computer_Vision/results/profiling_summary.csv) sau khi hoàn tất chuỗi thực nghiệm:

| Tên Mô Hình | Backend Suy Luận | Thời Gian Nạp (s) | Độ Trễ TB (s/mẫu) | Thông Lượng (mẫu/phút) | Đỉnh VRAM Cấp Phát (GB) | Đỉnh VRAM Dự Trữ (GB) | Đỉnh RAM Host (GB) | Sơ Bộ F1 (%) | Sơ Bộ EM (%) | Ước Lượng Full 10,028 mẫu (Giờ) | Chi Phí Full Ước Tính (USD) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **unsloth/Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit** | Unsloth (4-bit) | 65.55 | 1.64 | 36.50 | 5.54 | 5.78 | 3.00 | 38.26 | 14.00 | 4.60 | $0.97 |
| **unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit** | Unsloth (4-bit) | 98.61 | 2.03 | 29.52 | 9.57 | 9.92 | 2.90 | 37.66 | 9.00 | 5.69 | $1.19 |
| **5CD-AI/Vintern-1B-v3_5** | Vintern Native (BF16) | 7.13 | 0.45 | 132.09 | 2.30 | 2.83 | 2.72 | **42.21** | 12.00 | **1.27** | **$0.27** |
| **5CD-AI/Vintern-3B-beta** | Vintern Native (BF16) | 73.73 | 0.55 | 109.01 | 7.41 | 7.70 | 2.09 | 37.35 | 6.00 | 1.55 | $0.33 |
| **erax-ai/EraX-VL-2B-V1.5** | HF Transformers (BF16) | 57.35 | 1.28 | 46.71 | 9.28 | 13.98 | 2.86 | 32.42 | 1.00 | 3.59 | $0.75 |
| **erax-ai/EraX-VL-7B-V1.5** | HF Transformers (BF16) | 154.21 | 4.73 | 12.69 | 14.19 | 14.85 | 8.27 | **44.22** | 13.00 | 13.22 | $2.78 |
| **unsloth/aya-vision-8b-unsloth-bnb-4bit** | Unsloth (4-bit) | 108.00 | 2.02 | 29.70 | 7.27 | 7.34 | 2.64 | 19.68 | 3.00 | 5.66 | $1.19 |

---

## 3. NHẬN XÉT SƠ BỘ VỀ MẶT HIỆU NĂNG VÀ ĐỘ CHÍNH XÁC

1. **Vintern-1B-v3_5 (5CD-AI)** đạt sự cân bằng tối ưu vượt trội giữa tốc độ và độ chính xác:
   - Thông lượng đạt mức cao nhất toàn bộ dải thử nghiệm (**132.09 mẫu/phút**, độ trễ chỉ **0.45s/mẫu**).
   - Điểm số F1 sơ bộ đạt **42.21%**, chỉ xếp sau duy nhất mô hình EraX-VL-7B-V1.5 (44.22%) có kích thước tham số lớn gấp 7 lần.
   - Mức chiếm dụng VRAM cực kỳ tiết kiệm (**2.30 GB** cấp phát), mở ra không gian rất lớn để thực hiện xử lý theo lô (batching) trên các dòng GPU cao cấp hơn.

2. **Vintern-3B-beta (5CD-AI)** sau khi khắc phục triệt để cấu hình `InternVLChatConfig` và `trust_remote_code=True`:
   - Hoạt động ổn định với thông lượng **109.01 mẫu/phút** (độ trễ **0.55s/mẫu**).
   - Điểm F1 sơ bộ đạt **37.35%**, xác nhận mô hình hoàn toàn bình thường khi chạy ở định dạng `bfloat16`.
   - VRAM chiếm dụng ở mức vừa phải (**7.41 GB**), an toàn trên các GPU 16GB trở lên.

3. **Tổng thời gian và ngân sách baseline (Toàn bộ 7 mô hình, Batch Size = 1 trên RTX 5060 Ti):**
   - Tổng thời gian tích lũy để chạy trọn vẹn 10,028 mẫu cho cả 7 mô hình: **35.58 giờ** (~1.48 ngày liên tục).
   - Tổng chi phí hạ tầng ước tính: **$5.05 USD**.
