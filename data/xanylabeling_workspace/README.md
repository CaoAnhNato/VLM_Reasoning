# Hướng Dẫn Gán Nhãn X-AnyLabeling Theo Thành Viên

## 1. Cài đặt thư viện
```powershell
python -m venv venv
.\venv\Scripts\pip.exe install x-anylabeling-cvhub==4.2.0 pyqt6==6.7.1 pyqt6-qt6==6.7.3 pyqt6-sip==13.8.0 pyqt6-webengine==6.7.0 pyqt6-webengine-qt6==6.7.3 Pillow
```

## 2. Mở App theo tên thành viên
- **Cao_Anh (Ảnh 1 - 100)**:
  ```powershell
  .\venv\Scripts\python.exe scripts/benchmark/launch_xanylabeling.py --member Cao_Anh
  ```
- **Trieu (Ảnh 101 - 200)**:
  ```powershell
  .\venv\Scripts\python.exe scripts/benchmark/launch_xanylabeling.py --member Trieu
  ```
- **Cam_Tu (Ảnh 201 - 300)**:
  ```powershell
  .\venv\Scripts\python.exe scripts/benchmark/launch_xanylabeling.py --member Cam_Tu
  ```
- **Thanh_Tam (Ảnh 301 - 400)**:
  ```powershell
  .\venv\Scripts\python.exe scripts/benchmark/launch_xanylabeling.py --member Thanh_Tam
  ```

## 3. Cách bổ sung thêm đáp án bị thiếu trong GT gốc (Ví dụ ảnh `126.jpg`)
1. Nhấn **`R`**, kẻ khung quanh biển hiệu/vùng chữ bị thiếu.
2. Chọn đúng nhãn **`Q{k}_gt`** của câu hỏi tương ứng (ví dụ `Q2_gt`).
3. Nhập đáp án mới vào ô **Description** (góc dưới bên phải) của box vừa vẽ.

![Ví dụ 126.jpg](guide_previews/xany_preview_126.jpg)

## 4. Nộp kết quả sau khi hoàn thành
- Sau khi hoàn thành 100 ảnh, mỗi thành viên upload nguyên thư mục của mình (`Cao_Anh_001_100`, `Trieu_101_200`, `Cam_Tu_201_300`, `Thanh_Tam_301_400`) lên Google Drive:
- 🔗 **Link Google Drive nộp bài:** [https://drive.google.com/drive/folders/1WR7wmf57ljGHHpia6R3b4xs-15bl0xH3?usp=drive_link](https://drive.google.com/drive/folders/1WR7wmf57ljGHHpia6R3b4xs-15bl0xH3?usp=drive_link)
