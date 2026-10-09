# HƯỚNG DẪN CHI TIẾT THỰC THI ZERO-SHOT BENCHMARK TRÊN MÁY ẢO VAST.AI
**Dự án:** VLM Reasoning trên ViTextVQA  
**Mục tiêu:** Tải dataset chính chủ từ Hugging Face và đánh giá tuần tự 7 mô hình VLM với cơ chế tự động dọn dẹp bộ nhớ đệm (Sequential Cache Eviction).

---

## 1. TỔNG QUAN QUY TRÌNH THỰC HIỆN

Toàn bộ quy trình từ lúc thuê máy ảo đến khi có kết quả bảng điểm cuối cùng được tinh gọn qua 8 bước:

```
[1. Thuê GPU Vast.ai] ➔ [2. SSH vào máy ảo] ➔ [3. Clone GitHub Repo]
        │
        ▼
[4. Chạy setup_vastai.sh] ➔ [5. Tải Dataset qua HF Token] ➔ [6. Smoke Test (10 mẫu)]
        │
        ▼
[7. Chạy Full Benchmark ngầm bằng tmux] ➔ [8. Đẩy kết quả lên GitHub]
```

---

## 2. HƯỚNG DẪN TỪNG BƯỚC CHI TIẾT

### BƯỚC 1: THUÊ MÁY ẢO TRÊN VAST.AI

1. Truy cập vào [Vast.ai Console](https://cloud.vast.ai/create/) và đăng nhập tài khoản.
2. Tại thanh tìm kiếm bộ lọc (Filter):
   * **GPU Model:** Chọn `RTX 4090` (24GB) hoặc `RTX 5080 / RTX 5070` (16GB VRAM) hoặc `RTX 3090`.
   * **Số lượng GPU:** `1 GPU`.
   * **Disk Space (Ổ đĩa):** Chỉ cần đặt **`40 GB` đến `50 GB`** (Nhờ cơ chế xóa cache sau mỗi model của script, bạn không cần thuê 100GB giúp tiết kiệm tối đa chi phí).
3. **Cấu hình Docker Image (Edit Image):**
   * Chọn template chuẩn: `PyTorch` (hoặc image: `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel`).
   * Chọn kiểu khởi động: **`Run interactive shell server, SSH`**.
4. Bấm **RENT** (Giá dao động khoảng \$0.25 – \$0.38 / giờ).

---

### BƯỚC 2: KẾT NỐI SSH VÀO MÁY ẢO

1. Chờ máy ảo chuyển từ trạng thái `Creating` sang `Running` (khoảng 1–2 phút).
2. Bấm vào nút **Open** hoặc sao chép lệnh SSH Proxy được cung cấp (biểu tượng chiếc chìa khóa), có dạng:
   ```bash
   ssh -p <PORT> root@<IP_ADDRESS> -L 8080:localhost:8080
   ```
3. Mở Terminal (hoặc PowerShell) trên máy tính của bạn và dán lệnh trên để đăng nhập vào máy ảo.

---

### BƯỚC 3: TẢI MÃ NGUỒN DỰ ÁN TỪ GITHUB

Trên terminal của máy ảo Vast.ai, chạy lệnh clone repository của dự án:

```bash
# Di chuyển vào thư mục làm việc
cd /workspace || cd /root

# Clone dự án về máy
git clone https://github.com/CaoAnhNato/VLM_Reasoning.git
cd VLM_Reasoning

# Cấp quyền thực thi cho các script
chmod +x scripts/benchmark/*.sh
```

---

### BƯỚC 4: THIẾT LẬP MÔI TRƯỜNG TỰ ĐỘNG

Chạy script cài đặt toàn bộ PyTorch, Hugging Face CLI, Unsloth, và các tiện ích cần thiết chỉ với một câu lệnh:

```bash
./scripts/benchmark/setup_vastai.sh
```
*Thời gian chạy: khoảng 2–3 phút.*

---

### BƯỚC 5: TẢI BỘ DỮ LIỆU ViTextVQA TỪ HUGGING FACE

Vì bạn đã được cấp quyền truy cập vào repo chính thức `minhquan6203/ViTextVQA`, hãy lấy **Hugging Face User Access Token** của bạn:
1. Vào [Hugging Face Tokens Settings](https://huggingface.co/settings/tokens).
2. Tạo hoặc sao chép một Token (quyền `read`), có dạng `hf_xxxxxxxxxxxxxxxxxxxxxxxx`.
3. Chạy lệnh tải dữ liệu:

```bash
# Cách 1: Truyền token trực tiếp
./scripts/benchmark/download_dataset.sh hf_your_token_here

# Hoặc Cách 2: Export biến môi trường
export HF_TOKEN="hf_your_token_here"
./scripts/benchmark/download_dataset.sh
```

> **Ghi chú tốc độ:** Nhờ mạng CDN của Vast.ai, tệp dữ liệu 5.8GB sẽ được tải về trọn vẹn chỉ trong khoảng **30–45 giây**!

---

### BƯỚC 6: CHẠY THỬ NGHIỆM NHANH (SMOKE TEST 10 MẪU)

Trước khi chạy toàn bộ tập dữ liệu, hãy chạy kiểm thử nhanh 10 mẫu để xác nhận ảnh nạp tốt, mô hình sinh câu trả lời mượt mà và hàm tính F1/Exact Match hoạt động đúng:

```bash
./scripts/benchmark/run_all_benchmarks.sh 10
```

Kiểm tra kết quả nhanh sau 1–2 phút:
```bash
cat results/vitextvqa_benchmark_summary.csv
```
Nếu thấy các hàng số liệu hiển thị đầy đủ, hệ thống đã sẵn sàng 100% cho full benchmark!

---

### BƯỚC 7: CHẠY FULL BENCHMARK TRONG NỀN BẰNG `TMUX`

Toàn bộ tập Test chính thức (~3.300 ảnh / ~10.000 QA) cho 7 mô hình sẽ mất từ **2 đến 4 giờ** tùy theo tốc độ GPU. Để máy tính cá nhân của bạn có thể tắt hoặc không lo rớt mạng Wi-Fi làm gián đoạn, **bắt buộc chạy trong phiên `tmux`**:

```bash
# 1. Tạo một phiên làm việc ngầm tên là "benchmark"
tmux new -s benchmark

# 2. Khởi chạy toàn bộ 7 mô hình
./scripts/benchmark/run_all_benchmarks.sh
```

* **Để thoát ra ngoài (Detach tmux)** mà máy ảo vẫn tiếp tục chạy ngầm: Nhấn tổ hợp phím `Ctrl + B`, sau đó thả tay ra và nhấn phím `D`.
* **Để quay lại xem tiến độ (Attach tmux):**
  ```bash
  tmux attach -t benchmark
  ```
* **Để theo dõi log thời gian thực mà không cần vào tmux:**
  ```bash
  tail -f results/logs/*.log
  ```

---

### BƯỚC 8: THU THẬP KẾT QUẢ VÀ ĐỒNG BỘ LÊN GITHUB

Khi script hoàn tất, toàn bộ kết quả sẽ được ghi vào thư mục `results/`:
* `results/vitextvqa_benchmark_summary.csv`: Bảng tổng hợp điểm số F1 và Exact Match của cả 7 mô hình.
* `results/predictions/*.jsonl`: Toàn bộ câu trả lời chi tiết của từng câu hỏi phục vụ phân tích định tính.
* `results/logs/*.log`: Nhật ký chạy lệnh để kiểm chứng tính minh bạch khoa học.

Đẩy kết quả trực tiếp lên GitHub:
```bash
git config --global user.name "Cao Anh"
git config --global user.email "your_email@example.com"

git add results/
git commit -m "feat(benchmark): add zero-shot results for 7 VLMs on ViTextVQA"
git push origin main
```

*(Hoặc tải file `vitextvqa_benchmark_summary.csv` về máy cá nhân bằng lệnh `scp` hoặc SFTP).*

---

## 3. BẢNG MẪU ĐỐI CHIẾU KẾT QUẢ VỚI PAPER GỐC

Sau khi chạy xong, mở file `results/vitextvqa_benchmark_summary.csv` để đối chiếu với mốc chuẩn của paper gốc:

| Nhóm Mô Hình | Tên Mô Hình | Quy mô | F1 (Full Test) | EM (Full Test) | F1 (100 mẫu) | EM (100 mẫu) | Ghi Chú Đối Chiếu |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Mốc Paper gốc** | **GPT-4o** | - | - | - | **55.51%** | **25.00%** | *Số liệu công bố tại Bảng 5.6 paper gốc* |
| **Mốc Paper gốc** | **Gemini-1.5-Flash** | - | - | - | **57.75%** | **26.00%** | *Số liệu công bố tại Bảng 5.6 paper gốc* |
| **Mốc Paper gốc** | **QwenVL-7B** | 7B | - | - | **53.80%** | **28.00%** | *Số liệu công bố tại Bảng 5.6 paper gốc* |
| **General VLM** | Qwen2.5-VL-7B | 7B | ... | ... | ... | ... | *Model general multilingual chính* |
| **General VLM** | Qwen2.5-VL-3B | 3B | ... | ... | ... | ... | *Đánh giá trade-off kích thước 3B vs 7B* |
| **Vietnamese VLM** | Vintern-3B-beta | 3B | ... | ... | ... | ... | *Chuyên biệt tiếng Việt quy mô 3B* |
| **Vietnamese VLM** | Vintern-1B-v3.5 | 1B | ... | ... | ... | ... | *Chuyên biệt tiếng Việt quy mô siêu nhỏ 1B* |
| **Vietnamese VLM** | EraX-VL-7B-V1.5 | 7B | ... | ... | ... | ... | *Model tiếng Việt quy mô 7B* |
| **Vietnamese VLM** | EraX-VL-2B-V1.5 | 2B | ... | ... | ... | ... | *Model tiếng Việt quy mô 2B* |
| **Family Độc lập** | Aya Vision 8B | 8B | ... | ... | ... | ... | *Multilingual family khác từ Cohere* |

---

## 4. XỬ LÝ SỰ CỐ THƯỜNG GẶP (TROUBLESHOOTING)

1. **Lỗi `CUDA Out of Memory (OOM)`:**
   * Mặc định script sử dụng `bfloat16` và giới hạn `max_new_tokens=64`. Nếu vẫn gặp OOM trên GPU 16GB, thêm cờ `--load_in_4bit` khi chạy.
2. **Lỗi không tìm thấy file ảnh khi chạy:**
   * Script đã có cơ chế tự động dò tìm thư mục ảnh đệ quy. Nếu cần chỉ định chính xác, bạn có thể truyền tham số `--image_dir data/vitextvqa/test_images`.
3. **Lỗi mất kết nối SSH giữa chừng:**
   * Yên tâm! Nếu bạn đã chạy trong `tmux` như hướng dẫn ở Bước 7, script vẫn đang âm thầm chạy trên máy ảo. Chỉ cần SSH lại và gõ `tmux attach -t benchmark`.
