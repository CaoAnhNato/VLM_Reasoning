# Danh sách Gán nhãn X-AnyLabeling — `00_sample_for_leader`

- **Tổng số ảnh:** `8`
- **Tổng số câu hỏi (Quests):** `25`
- **Số câu hỏi cần kéo box từ góc trái trên (`[CẦN KÉO BOX]`):** `1` (trên `1` ảnh)
- **Các ảnh ưu tiên kiểm tra trước (có box chờ kéo):** `22.jpg`

## Hướng dẫn bổ sung thêm BBox & Đáp án bị thiếu trong bộ dữ liệu gốc

Khi kiểm tra ảnh có câu hỏi liệt kê (ví dụ `126.jpg`: *"ở đây có những cửa hàng gì ?"* có 6 cửa hàng trên phố nhưng GT gốc chỉ ghi 3):
1. **Vẽ thêm BBox cho các đáp án bị thiếu**: Nhấn phím **`R`**, khoanh vào biển hiệu bị thiếu (ví dụ `Like Kid's`, `SMART SHOES`, `Yến Kids`), chọn đúng nhãn **`Q{k}_gt`** của câu hỏi đó (ví dụ `Q2_gt`).
2. **Ghi bổ sung đáp án**: Ở ô **Description** (góc dưới bên phải X-AnyLabeling) của box vừa vẽ, nhập tên đáp án mới (ví dụ: `like kid's` hoặc `Hỏi: ở đây có những cửa hàng gì ? || ĐA: like kid's`). Hoặc sửa trực tiếp phần sau `|| ĐA:` trên box có sẵn (ngăn cách nhiều đáp án bằng dấu `/`).
3. Khi chạy `--merge`, hệ thống sẽ tự động gom toàn bộ các BBox cùng nhãn `Q{k}_gt` và cập nhật danh sách đáp án mới vào dữ liệu chuẩn.

## Bảng tra cứu nhanh Câu hỏi & Đáp án theo từng ảnh

| STT | Tên ảnh | Số Quest | Trạng thái Auto-Detect | Chi tiết `Qk_gt` (Câu hỏi $\rightarrow$ Đáp án GT) |
| :---: | :--- | :---: | :--- | :--- |
| 1 | `22.jpg` | 4 | ⚠️ Cần kéo 1 box | **`Q1_gt` (`qid=74`)**: cửa tiệm màu xanh là gì ? $\rightarrow$ `nhà thuốc` ✅ <br> **`Q2_gt` (`qid=75`)**: nhà thuốc trong ảnh tên gì ? $\rightarrow$ `bảo tâm an` ✅ <br> **`Q3_gt` (`qid=76`)**: cửa tiệm bên phải nhà thuốc chuyên gì ? $\rightarrow$ `bán sỉ và lẻ` ✅ <br> **`Q4_gt` (`qid=77`)**: ảnh này lấy nguồn từ đâu ? $\rightarrow$ `wutacam` ⚠️ **[CẦN KÉO BOX]** |
| 2 | `30.jpg` | 1 | ✅ Đã có sẵn | **`Q1_gt` (`qid=107`)**: cửa tiệm trong ảnh bán gì ? $\rightarrow$ `bếp gas / bếp từ / hút mùi` ✅ |
| 3 | `38.jpg` | 6 | ✅ Đã có sẵn | **`Q1_gt` (`qid=138`)**: nơi này tên là gì ? $\rightarrow$ `cbam` ✅ <br> **`Q2_gt` (`qid=139`)**: trường này làm về gì ? $\rightarrow$ `đào tạo quản lý doanh nghiệp` ✅ <br> **`Q3_gt` (`qid=140`)**: nơi này thuộc về phòng nào ? $\rightarrow$ `thương mại và công nghiệp việt nam` ✅ <br> **`Q4_gt` (`qid=141`)**: số điện thoại nơi này là gì ? $\rightarrow$ `( 08 ) 39 142 168` ✅ <br> **`Q5_gt` (`qid=142`)**: số fax nơi này là gì ? $\rightarrow$ `08 ) 39 142 116` ✅ <br> **`Q6_gt` (`qid=143`)**: địa chỉ website nơi này là gì ? $\rightarrow$ `www . cbam . edu . v1` ✅ |
| 4 | `68.jpg` | 4 | ✅ Đã có sẵn | **`Q1_gt` (`qid=245`)**: đổi máy mới trong vòng bao nhiêu ngày ? $\rightarrow$ `30 ngày` ✅ <br> **`Q2_gt` (`qid=246`)**: thoải mái đổi trả cái gì ? $\rightarrow$ `điện thoại máy tính bảng` ✅ <br> **`Q3_gt` (`qid=247`)**: ở đây có bán điện thoại nào ? $\rightarrow$ `galaxy note9` ✅ <br> **`Q4_gt` (`qid=248`)**: ở đây có bán điện thoại của hãng nào ? $\rightarrow$ `apple` ✅ |
| 5 | `76.jpg` | 3 | ✅ Đã có sẵn | **`Q1_gt` (`qid=267`)**: tên của sản phẩm này là gì ? $\rightarrow$ `nước cốt dừa tươi` ✅ <br> **`Q2_gt` (`qid=268`)**: sản phẩm thuộc của hãng nào ? $\rightarrow$ `cocoxim` ✅ <br> **`Q3_gt` (`qid=269`)**: thể tích thực của sản phẩm là bao nhiêu ? $\rightarrow$ `400ml` ✅ |
| 6 | `77.jpg` | 3 | ✅ Đã có sẵn | **`Q1_gt` (`qid=270`)**: tên của quán coffee ? $\rightarrow$ `như nguyễn` ✅ <br> **`Q2_gt` (`qid=271`)**: số điện thoại của quán là gì ? $\rightarrow$ `0945166799 / 0945166899` ✅ <br> **`Q3_gt` (`qid=273`)**: địa chỉ của quán ở đâu ? $\rightarrow$ `24 đường 52 , p . an phú , q2` ✅ |
| 7 | `111.jpg` | 2 | ✅ Đã có sẵn | **`Q1_gt` (`qid=348`)**: bản tin thời sự này diễn ra vào ngày nào ? $\rightarrow$ `19 - 12 - 2021` ✅ <br> **`Q2_gt` (`qid=349`)**: bản tin thời sự này được phát sóng ở kênh nào ? $\rightarrow$ `ttv11` ✅ |
| 8 | `126.jpg` | 2 | ✅ Đã có sẵn | **`Q1_gt` (`qid=374`)**: bức hình được lấy từ báo nào ? $\rightarrow$ `kiến thức` ✅ <br> **`Q2_gt` (`qid=375`) *(3 box rời)***: ở đây có những cửa hàng gì ? $\rightarrow$ `giầy đồ hiệu / anta sports / in thêu may cờ` ✅ |
