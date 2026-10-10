# HƯỚNG DẪN CHI TIẾT THỰC THI ZERO-SHOT BENCHMARK TRÊN MÁY ẢO VAST.AI
**Dự án:** VLM Reasoning trên ViTextVQA  
**Mục tiêu:** Tải dataset chính chủ từ Hugging Face và đánh giá tuần tự 7 mô hình VLM với cơ chế tự động dọn dẹp bộ nhớ đệm (Sequential Cache Eviction) và **tiếp tục chạy nối tiếp (Resume) khi tắt/mở máy ảo**.

---

## 1. TỔNG QUAN QUY TRÌNH THỰC HIỆN

```
[1. Thuê GPU Vast.ai] ➔ [2. SSH vào máy ảo] ➔ [3. Clone GitHub Repo]
        │
        ▼
[4. Chạy setup_vastai.sh] ➔ [5. Tải Dataset qua HF Token] ➔ [6. Smoke Test (10 mẫu)]
        │
        ▼
[7. Chạy Full Benchmark ngầm bằng tmux] ➔ [8. Đẩy kết quả lên GitHub & Tắt máy]
```

---

## 2. HƯỚNG DẪN TỪNG BƯỚC CHI TIẾT

### BƯỚC 1: THUÊ MÁY ẢO TRÊN VAST.AI

1. Truy cập vào [Vast.ai Console](https://cloud.vast.ai/create/) và đăng nhập tài khoản.
2. Dựa trên danh sách máy thực tế hiện tại (như trong ảnh bạn chụp):
   * **LỰA CHỌN KHUYẾN NGHỊ SỐ 1:** Máy **1x RTX 5070 Ti (16GB VRAM) tại Vietnam** (Giá chỉ **\$0.210/giờ**, băng thông mạng lên tới **1812 / 2028 Mbps**). Lựa chọn này cho ping cực thấp và tải dữ liệu chỉ trong vài chục giây.
   * Hoặc máy **1x RTX 5080 (16GB VRAM)** (Giá **\$0.304/giờ**).
3. **Cấu hình Template & Ổ đĩa:**
   * **Template:** Giữ mặc định **`PyTorch (Vast)`** (Image `vastai/pytorch...`).  
     *(Container này **đã cài sẵn PyTorch và CUDA**, script cài đặt của chúng ta sẽ tự động nhận diện và bỏ qua bước cài đặt lại PyTorch, tiết kiệm tối đa thời gian).*
   * **Disk Space (Ổ đĩa):** Chỉ cần đặt **`40 GB` đến `50 GB`** (Nhờ cơ chế xóa cache sau mỗi model của script, bạn không cần thuê 100GB giúp tiết kiệm tiền thuê).
4. Bấm **RENT**.

---

### BƯỚC 2: KẾT NỐI SSH VÀO MÁY ẢO

1. Chờ máy ảo chuyển từ trạng thái `Creating` sang `Running` (khoảng 1–2 phút).
2. Sao chép lệnh SSH Proxy được cung cấp (biểu tượng chiếc chìa khóa trên Vast.ai), có dạng:
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

Chạy script cài đặt các thư viện bổ trợ (Hugging Face CLI, Unsloth, Transformers, accelerate...):

```bash
./scripts/benchmark/setup_vastai.sh
```
*Lưu ý: Script sẽ tự phát hiện PyTorch CUDA có sẵn trong container và chỉ mất khoảng **30 – 45 giây** để hoàn tất toàn bộ.*

---

### BƯỚC 5: TẢI BỘ DỮ LIỆU ViTextVQA TỪ HUGGING FACE

Vì bạn đã được cấp quyền truy cập vào repo chính thức `minhquan6203/ViTextVQA`:
1. Lấy **Hugging Face Token** (quyền `read`) tại [Hugging Face Settings -> Tokens](https://huggingface.co/settings/tokens).
2. Chạy lệnh tải dữ liệu:

```bash
# Truyền token trực tiếp vào script
./scripts/benchmark/download_dataset.sh hf_your_token_here
```
*Tốc độ: Nhờ mạng 2000 Mbps của máy chủ Vast.ai, tệp dữ liệu 5.8GB sẽ tải xong trong khoảng **20 – 35 giây**!*

---

### BƯỚC 6: CHẠY THỬ NGHIỆM NHANH (SMOKE TEST 10 MẪU)

Trước khi chạy toàn bộ tập dữ liệu, hãy chạy kiểm thử nhanh 10 mẫu để xác nhận ảnh nạp tốt, mô hình sinh câu trả lời mượt mà:

```bash
./scripts/benchmark/run_all_benchmarks.sh 10
```

Kiểm tra kết quả nhanh sau 1–2 phút:
```bash
cat results/vitextvqa_benchmark_summary.csv
```

---

### BƯỚC 7: CHẠY FULL BENCHMARK TRONG NỀN BẰNG `TMUX`

Chạy toàn bộ 7 mô hình tuần tự trong phiên `tmux` để có thể yên tâm tắt máy tính cá nhân mà tiến trình không bị dừng:

```bash
# 1. Tạo phiên làm việc ngầm
tmux new -s benchmark

# 2. Khởi chạy toàn bộ mô hình (Mặc định tự động bỏ qua các model đã hoàn tất)
./scripts/benchmark/run_all_benchmarks.sh
```

* **Thoát ra ngoài màn hình chính (Detach tmux):** Nhấn tổ hợp phím `Ctrl + B`, thả tay ra rồi nhấn phím `D`.
* **Quay lại xem tiến độ (Attach tmux):**
  ```bash
  tmux attach -t benchmark
  ```
* **Theo dõi log thời gian thực:**
  ```bash
  tail -f results/logs/*.log
  ```

---

## 3. CƠ CHẾ NỐI TIẾP KHI CẦN TẮT MÁY ẢO HOẶC BỊ NGẮT GIỮA CHỪNG

> **Hệ thống hỗ trợ 2 CẤP ĐỘ RESUME TUYỆT ĐỐI:**
> 1. **Cấp độ Mô hình (Model-level Resume):** Chạy xong Model #1, #2 $\to$ Tự động nhảy sang Model #3.
> 2. **Cấp độ Mẫu dữ liệu (Sample-level Resume):** Giả sử đang chạy dở Model #1 đến mẫu số **133** thì bị tắt máy $\to$ Lần chạy sau sẽ **đọc file kết quả cũ và chạy tiếp ngay từ mẫu 133**, không phải chạy lại từ mẫu số 0!

---

### TÌNH HUỐNG 1: ĐANG CHẠY DỞ MẪU (VÍ DỤ ĐANG CHẠY MẪU 133 CỦA MODEL #1)

* **Cơ chế ghi tức thời (Real-time Streaming Flush):** Mỗi khi mô hình sinh xong câu trả lời cho 1 câu hỏi, script gọi ngay hàm `flush()` để ghi đĩa trực tiếp vào `results/predictions/<tên_model>_preds.jsonl`. Dù máy bị tắt đột ngột, toàn bộ các mẫu trước đó vẫn an toàn 100%.
* **Khi bật máy chạy lại:**
  ```bash
  ./scripts/benchmark/run_all_benchmarks.sh
  ```
  Script sẽ tự động:
  1. Quét file `results/predictions/<model>_preds.jsonl`, phát hiện đã có 133 mẫu hoàn thành.
  2. Bỏ qua 133 mẫu đã có, **tiếp tục đánh giá từ mẫu thứ 134 (index 133, id ...)**.
  3. Ghép nối kết quả cũ và mới, tính toán điểm số tổng hợp toàn diện (Accuracy/EM, F1) và cập nhật vào `summary.csv`.

---

### TÌNH HUỐNG 2: ĐÃ CHẠY XONG MODEL #1 VÀ #2, CẦN TẮT MÁY ĐỂ HÔM SAU CHẠY MODEL #3

#### Bước A: Trước khi tắt máy ảo (Lưu kết quả lên GitHub)
Ngay trên máy ảo hiện tại, chạy lệnh commit và đẩy thư mục `results/` lên GitHub:
```bash
git config --global user.name "Cao Anh"
git config --global user.email "your_email@example.com"

git add results/
git commit -m "feat(benchmark): save results for completed models and samples"
git push origin main
```
*Sau khi push thành công, bạn bấm **Destroy / Stop Instance** trên web Vast.ai để ngừng tính tiền.*

---

#### Bước B: Khi thuê máy ảo mới (Hôm sau)
Khi thuê máy ảo mới, bạn tiến hành setup bình thường:
```bash
git clone https://github.com/CaoAnhNato/VLM_Reasoning.git
cd VLM_Reasoning
chmod +x scripts/benchmark/*.sh

./scripts/benchmark/setup_vastai.sh
./scripts/benchmark/download_dataset.sh hf_your_token_here
```
*Lưu ý: Vì bạn đã push `results/` lên GitHub ở Bước A, nên khi `git clone`, thư mục `results/` trên máy mới đã có sẵn kết quả cũ!*

---

#### Bước C: Kích hoạt chạy tiếp nối
Bạn có 2 cách cực kỳ linh hoạt:

* **Cách 1: Chạy tự động (Khuyên dùng - Auto Resume):**
  ```bash
  tmux new -s benchmark "./scripts/benchmark/run_all_benchmarks.sh"
  ```
  👉 **Script thông minh sẽ tự động rà soát:**
  - Model #1 (`Qwen2.5-VL-3B`) đã đủ 100% mẫu $\to$ **Tự động BỎ QUA trong 1 giây**.
  - Model #2 (`Qwen2.5-VL-7B`) đã đủ 100% mẫu $\to$ **Tự động BỎ QUA trong 1 giây**.
  - Model #3 (`Vintern-1B-v3.5`) chưa có hoặc chạy dở $\to$ **Tự động CHẠY NỐI TIẾP NGAY LẬP TỨC!**

* **Cách 2: Chỉ định rõ ràng mốc bắt đầu bằng cờ lệnh:**
  ```bash
  # Bắt đầu chạy từ model số 3 trở đi
  ./scripts/benchmark/run_all_benchmarks.sh --start 3

  # Hoặc chỉ chạy DUY NHẤT một model số 3
  ./scripts/benchmark/run_all_benchmarks.sh --only 3

  # Nếu muốn ép chạy lại từ đầu (bỏ qua kết quả cũ)
  ./scripts/benchmark/run_all_benchmarks.sh --force
  ```

---

## 4. DANH SÁCH 7 MÔ HÌNH VÀ THỨ TỰ ĐÁNH GIÁ

| Thứ tự (`ID`) | Tên Mô Hình | Quy mô | Backend | Ghi Chú Đánh Giá |
| :---: | :--- | :---: | :---: | :--- |
| **#1** | `Qwen/Qwen2.5-VL-3B-Instruct` | 3B | Unsloth / Native | General VLM nhỏ (đối sánh 3B vs 7B) |
| **#2** | `Qwen/Qwen2.5-VL-7B-Instruct` | 7B | Unsloth / Native | General VLM chính |
| **#3** | `5CD-AI/Vintern-1B-v3.5` | 1B | Native (bfloat16) | Vietnamese VLM siêu nhỏ (chuyên OCR) |
| **#4** | `5CD-AI/Vintern-3B-beta` | 3B | Native (bfloat16) | Vietnamese VLM 3B (so sánh trực tiếp Qwen 3B) |
| **#5** | `EraX-AI/EraX-VL-2B-V1.5` | 2B | Native (bfloat16) | Vietnamese VLM 2B |
| **#6** | `EraX-AI/EraX-VL-7B-V1.5` | 7B | Native (bfloat16) | Vietnamese VLM 7B (so sánh Qwen 7B) |
| **#7** | `CohereForAI/aya-vision-8b` | 8B | Native (bfloat16) | Multilingual VLM family độc lập |

---

## 5. BẢNG MẪU ĐỐI CHIẾU KẾT QUẢ VỚI PAPER GỐC

Sau khi hoàn thành, file `results/vitextvqa_benchmark_summary.csv` sẽ cho ra kết quả đối chiếu trực tiếp với mốc tham chiếu:

| Mô Hình | Quy mô | F1 (Full Test) | EM (Full Test) | F1 (100 mẫu) | EM (100 mẫu) | Ghi Chú Đối Chiếu |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **GPT-4o (Paper gốc)** | - | - | - | **55.51%** | **25.00%** | *Bảng 5.6 paper ViTextVQA gốc* |
| **Gemini-1.5-Flash (Paper)** | - | - | - | **57.75%** | **26.00%** | *Bảng 5.6 paper ViTextVQA gốc* |
| **QwenVL-7B (Paper)** | 7B | - | - | **53.80%** | **28.00%** | *Bảng 5.6 paper ViTextVQA gốc* |
| **Qwen2.5-VL-7B** | 7B | ... | ... | ... | ... | *Thực nghiệm mới của nhóm* |
| **Qwen2.5-VL-3B** | 3B | ... | ... | ... | ... | *Thực nghiệm mới của nhóm* |
| **Vintern-1B-v3.5** | 1B | ... | ... | ... | ... | *Thực nghiệm mới của nhóm* |
| **Vintern-3B-beta** | 3B | ... | ... | ... | ... | *Thực nghiệm mới của nhóm* |
| **EraX-VL-2B-V1.5** | 2B | ... | ... | ... | ... | *Thực nghiệm mới của nhóm* |
| **EraX-VL-7B-V1.5** | 7B | ... | ... | ... | ... | *Thực nghiệm mới của nhóm* |
| **Aya Vision 8B** | 8B | ... | ... | ... | ... | *Thực nghiệm mới của nhóm* |
