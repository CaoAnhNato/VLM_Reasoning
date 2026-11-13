# MANDATORY COLAB-MCP EXECUTION & SINGLETON CONNECTION RULES

> [!IMPORTANT]
> **QUY TẮC BẮT BUỘC KHI TƯƠNG TÁC VỚI GOOGLE COLAB QUA MCP (COLAB-MCP)**

---

## 1. Mandatory Singleton Notebook Connection Rule (ProxyPort Invariant)

- **Chỉ mở kết nối 1 lần duy nhất**: Agent chỉ được phép gọi `open_colab_browser_connection` một lần đầu tiên khi bắt đầu làm việc nếu chưa có phiên active.
- **TUYỆT ĐỐI CẤM** gọi lại lệnh `open_colab_browser_connection` khi kết nối đã tồn tại. Việc gọi lại sẽ sinh tab/notebook mới trên trình duyệt và làm gián đoạn `ProxyPort`, khiến người dùng phải thao tác cấu hình thủ công lại rất mất thời gian.
- **Kế thừa phiên giữa các Subagent**: Mọi subagent được điều phối (`invoke_subagent`) bắt buộc phải **kế thừa và dùng chung phiên kết nối hiện tại** thông qua các tool tương tác cell (`run_code_cell`, `get_cells`, `update_cell`...). Tuyệt đối không mở lại kết nối mới.
- **Quyền đóng notebook thuộc về Người Dùng**: Agent không tự ý đóng hoặc reset session notebook; người dùng sẽ tự đóng sau khi hoàn thành phiên làm việc.

---

## 2. Mandatory Asynchronous Background Execution for Long Tasks (> 2 phút)

- **Cấm chạy đồng bộ các tác vụ nặng**: Đối với các tác vụ huấn luyện (training), đánh giá (evaluation), trích xuất đặc trưng hoặc tải dữ liệu nặng có khả năng kéo dài quá 2 phút, Agent **BẮT BUỘC** phải khởi chạy bất đồng bộ trong nền (Background Process) qua `subprocess.Popen` hoặc `nohup`.
- **Mẫu kích hoạt tác vụ nền**:
  ```python
  import subprocess
  process = subprocess.Popen(
      ["python", "train.py"],
      stdout=open("train.log", "w"),
      stderr=subprocess.STDOUT
  )
  print(f"Task started in background! PID = {process.pid}")
  ```
- **Không để Agent bị đơ (Zero Blocking)**: Cell kích hoạt nền phải hoàn thành ngay lập tức (1–2 giây) kèm `PID`. Agent tuyệt đối không để một synchronous tool call chạy vượt ngưỡng timeout 3 phút của Antigravity harness (`context deadline exceeded`).

---

## 3. Scheduled Health-Check & Monitoring via `/schedule`

- Sau khi khởi chạy tác vụ nền, Agent kích hoạt tính năng kiểm tra định kỳ (sử dụng công cụ `schedule` với khoảng thời gian phù hợp, ví dụ 120s – 300s).
- Tại mỗi lần kiểm tra, Agent chạy cell giám sát:
  ```python
  # Xem tiến trình còn chạy không và đọc 20 dòng log mới nhất
  !ps aux | grep python
  !tail -n 20 train.log
  !nvidia-smi
  ```
- Nếu tiến trình hoàn tất thành công hoặc gặp lỗi kết thúc (dead/zombie), Agent dừng lịch kiểm tra và tổng hợp báo cáo kết quả cho người dùng.

---

## 4. Fallback Handling for Synchronous Timeouts

- Nếu vô tình một cell đồng bộ vượt quá 3 phút và gặp lỗi `context deadline exceeded` từ Antigravity Harness:
  - Agent **KHÔNG ĐƯỢC HOẢNG LOẠN** hay cố gắng khởi động lại notebook: Kernel Colab vẫn đang tiếp tục thực thi ngầm trên máy chủ Google.
  - Sử dụng ngay `get_cells` để đọc lại `outputs` của cell và xác nhận tiến độ thực tế trước khi thực hiện bước tiếp theo.
