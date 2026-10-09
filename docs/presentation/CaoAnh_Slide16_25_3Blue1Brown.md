# KỊCH BẢN THUYẾT TRÌNH PHONG CÁCH 3BLUE1BROWN (SLIDE 16 – 25)
**Đề tài:** Hiểu tài liệu bằng AI: Từ OCR đến Vision Language Model  
**Người trình bày:** Cao Anh (Phần 4: Kỷ nguyên OCR-Free & Phần 5: Nền tảng VLM)  
**Thời lượng dự kiến:** 5 – 6 phút (~30–35 giây / slide)  
**Triết lý thiết kế (3Blue1Brown Spirit):**  
- **Visual-First (Ưu tiên trực quan):** Thay thế các gạch đầu dòng dày đặc chữ bằng sơ đồ hình học, hình vẽ đối xứng, và phép biến đổi trực quan.
- **Minimal Text on Slide (Tối giản chữ):** Mỗi slide chỉ giữ lại 1 câu hỏi dẫn dắt hoặc 2–3 từ khóa neo tư duy (Anchor Keywords) trên nền trắng tinh tế.
- **Narrative-Driven Script (Lời dẫn giàu trực giác):** Người thuyết trình đóng vai trò là người dẫn dắt ánh nhìn và tư duy của khán giả, dùng các phép so sánh trực giác để biến khái niệm kỹ thuật phức tạp trở nên hiển nhiên.

---

## TỔNG QUAN PHÂN BỔ THỜI GIAN & MẠCH DẪN

```
[Slide 16] Vấn đề: Hiệu ứng Domino của OCR truyền thống (35s)
   │
   ▼
[Slide 17 - 19] Giải pháp OCR-Free: Donut ➔ Nougat ➔ Pix2Struct (105s)
   │
   ▼
[Slide 20] Bước ngoặt: Thế lưỡng nan & Giới hạn của OCR-Free (35s)
   │
   ▼
[Slide 21 - 22] Bước nhảy vọt: Bản chất VLM & Kiến trúc Bộ Ba Vũ Trụ (70s)
   │
   ▼
[Slide 23 - 24] Đại diện tiêu biểu: Qwen-VL (Độ phân giải động) & Vintern (Bản địa hóa VN) (70s)
   │
   ▼
[Slide 25] Hợp nhất: Tài liệu, Bảng biểu & Biểu đồ trong một không gian biểu diễn (35s)
```

---

## CHI TIẾT TỪNG SLIDE (SLIDE 16 ➔ 25)

---

### SLIDE 16: TẠI SAO CẦN OCR-FREE? (THE CASCADE FAILURE PROBLEM)

#### 1. Mục tiêu nhận thức (Core Takeaway)
Khán giả nhận ra rằng việc cố gắng sửa sai cho từng module trong pipeline phân tầng (Detection -> Crop -> Recognition -> Parsing) là một "cuộc chiến vô vọng" vì **lỗi tích lũy dây chuyền (Cascading Errors)**.

#### 2. Thiết kế giao diện Slide (Minimal Layout - Nền Trắng)
* **Tiêu đề slide:** *Tại sao cần OCR-Free?*
* **Chữ hiển thị trên slide (Rất ít):**
  * Từ khóa trung tâm: `Hiệu ứng Domino (Cascading Failure)`
  * Công thức trực quan: $\text{P(Thành công)} = P_1 \times P_2 \times P_3 \times P_4 \to 0$
* **Hình ảnh trung tâm:**
  * **Bên trái (Pipeline cũ):** 4 khối domino đang đổ vào nhau:  
    `[Ảnh mờ/lệch góc] ➔ [Cắt sai Bounding Box] ➔ [OCR nhận diện sai chữ] ➔ [Parser sập hoàn toàn]`
  * **Bên phải (Tầm nhìn mới):** Một thấu kính hội tụ duy nhất biến toàn bộ bức ảnh trực tiếp thành chuỗi văn bản cấu trúc: `[Ảnh Pixels] ➔ [ End-to-End Lens ] ➔ [Văn bản Cấu trúc]`.

#### 3. Yêu cầu hình ảnh & Nguồn tài liệu
* **Nguồn ảnh minh họa:** Lấy cảm hứng từ *Figure 1 của Donut paper (Geewook Kim et al., ECCV 2022)* hoặc vẽ lại sơ đồ dạng vector phẳng (Flat Design) với tông màu đối lập: nhánh pipeline màu xám đứt gãy, nhánh End-to-End màu xanh ngọc (cyan/blue).

#### 4. Lời thoại thuyết trình (Speaker Script - Cao Anh)
> *"Chào thầy và các bạn. Như bạn Phú Triệu vừa chỉ ra, các mô hình LayoutLM đã làm rất tốt việc kết hợp tọa độ và chữ. Nhưng hãy dừng lại một nhịp và tự hỏi: **Nếu ngay từ đầu, bộ máy OCR đọc sai chữ hoặc cắt lệch một ô bảng, thì liệu mô hình ngôn ngữ phía sau có thông minh đến mấy có cứu vãn được không?**  
> Trong toán học, xác suất thành công của một chuỗi mắt xích bằng tích xác suất của từng mắt xích độc lập. Khi ảnh thực tế bị nghiêng, mờ hay dính vết mực, pipeline truyền thống sẽ chịu một **hiệu ứng domino**: một lỗi nhỏ ở khâu phát hiện vùng chữ sẽ bị khuếch đại thành thảm họa ở khâu trích xuất cuối cùng.  
> Câu hỏi đặt ra là: **Tại sao chúng ta không vứt bỏ hoàn toàn bước OCR trung gian, để mô hình 'nhìn' bức ảnh và đọc hiểu trực tiếp như đôi mắt con người?** Đó chính là khởi đầu của kỷ nguyên **OCR-Free**."*

---

### SLIDE 17: DONUT (DOCUMENT UNDERSTANDING TRANSFORMER)

#### 1. Mục tiêu nhận thức (Core Takeaway)
Hiểu được bản chất của Donut: Đổi bài toán nhận dạng tài liệu từ *"cắt ghép nhiều bước"* thành bài toán dịch máy **"Image-to-Sequence"** thuần túy.

#### 2. Thiết kế giao diện Slide
* **Tiêu đề slide:** *Donut: Nhìn ảnh, Đọc cấu trúc*
* **Chữ hiển thị trên slide:**
  * Từ khóa: `Image-to-Sequence` • `No OCR Off-the-shelf`
  * Cấu trúc trực quan: $\text{Visual Patches} \xrightarrow{\text{Swin}} \text{Latent Space} \xrightarrow{\text{BART}} \text{\{JSON Tree\}}$
* **Hình ảnh trung tâm:**
  * Hình ảnh bức ảnh hóa đơn được chia thành các lưới ô vuông (patches) phân tầng qua Swin Transformer, truyền thẳng qua bộ giải mã ngôn ngữ BART, và xuất ra cây dữ liệu JSON phát sáng ở đầu ra.

#### 3. Yêu cầu hình ảnh & Nguồn tài liệu
* **Nguồn ảnh:** Trích xuất trực tiếp *Figure 2: Architecture of Donut* trong bài báo *“OCR-free Document Understanding Transformer” (ECCV 2022)*.
* **Ghi chú thiết kế:** Làm nổi bật sự tối giản của kiến trúc Encoder-Decoder (Swin-BART) trên nền trắng, không có bất kỳ engine OCR (như Tesseract hay PaddleOCR) chen ngang.

#### 4. Lời thoại thuyết trình (Speaker Script - Cao Anh)
> *"Và giải pháp tiên phong đầu tiên cho ý tưởng táo bạo này mang tên **Donut (ECCV 2022)**.  
> Hãy nhìn vào luồng dữ liệu trên slide: Donut không tìm kiếm từng con chữ, cũng không vẽ bất kỳ bounding box nào. Thay vào đó, toàn bộ trang tài liệu được xem như một bức tranh phong cảnh. Bộ mã hóa thị giác Swin Transformer quét qua bức tranh theo từng cụm điểm ảnh đa tỷ lệ, biến ánh sáng và bố cục thành các vector ngữ nghĩa.  
> Sau đó, bộ giải mã ngôn ngữ BART chỉ đơn giản là 'kể lại' những gì nó thấy dưới dạng một cây cấu trúc JSON chuẩn xác. **Ảnh vào, JSON ra**. Đơn giản, thanh lịch, và triệt tiêu hoàn toàn sự phụ thuộc vào các công cụ OCR cồng kềnh."*

---

### SLIDE 18: NOUGAT (NEURAL OPTICAL UNDERSTANDING FOR ACADEMIC DOCUMENTS)

#### 1. Mục tiêu nhận thức (Core Takeaway)
Nhận thấy sức mạnh đặc thù của kiến trúc OCR-Free khi chinh phục bài toán khó nhất của tài liệu học thuật: **Công thức toán học phức tạp và bảng biểu lồng nhau**.

#### 2. Thiết kế giao diện Slide
* **Tiêu đề slide:** *Nougat: Chuyển hóa Tri thức Học thuật*
* **Chữ hiển thị trên slide:**
  * Từ khóa: `ViT + mBART` • `PDF scan ➔ LaTeX / Markdown`
* **Hình ảnh trung tâm:**
  * Đối chiếu 1-1 cực kỳ ấn tượng:
    * Bên trái: Một đoạn scan bài báo khoa học chứa công thức tích phân nhiều tầng, ma trận phức tạp và chú thích chân trang.
    * Mũi tên biến đổi ở giữa mang logo Nougat.
    * Bên phải: Đoạn mã nguồn LaTeX tương ứng hiển thị sắc nét: `\int_{0}^{\infty} \frac{x^n}{e^x - 1} dx`.

#### 3. Yêu cầu hình ảnh & Nguồn tài liệu
* **Nguồn ảnh:** Trích xuất từ *Figure 1 của Meta AI Nougat Paper (Lukas Blecher et al., arXiv:2308.13418)*.
* **Ghi chú thiết kế:** Đặt ảnh chụp PDF scan cũ và code LaTeX được render toán học đối xứng nhau, tạo cảm giác chuyển đổi mượt mà và trực quan.

#### 4. Lời thoại thuyết trình (Speaker Script - Cao Anh)
> *"Nếu Donut chứng minh OCR-Free hiệu quả trên hóa đơn ngắn, thì Meta AI đã đưa bài toán lên một đỉnh cao mới với **Nougat**.  
> Bất kỳ ai từng làm việc với các bài báo khoa học PDF cũ đều biết: OCR truyền thống hoàn toàn bất lực trước các công thức toán học lồng nhau, ma trận hay bảng biểu nhiều cột. Ký tự Hy Lạp bị đọc thành rác, dấu căn bậc hai bị cắt vụn.  
> Nougat giải quyết vấn đề này bằng một cơ chế giải mã tự hồi quy thông minh: mô hình đọc toàn trang PDF như một chuỗi raster và trực tiếp 'viết lại' trang giấy đó thành mã nguồn **LaTeX và Markdown**. Nougat không chỉ đọc chữ, nó thực sự hiểu **cú pháp của toán học**."*

---

### SLIDE 19: PIX2STRUCT (SCREENSHOT PARSING WITH VISUAL PRETRAINING)

#### 1. Mục tiêu nhận thức (Core Takeaway)
Thấu hiểu chiến lược tiền huấn luyện độc đáo: Thay vì dạy mô hình đọc chữ, Google đã dạy nó **tái cấu trúc hình học của thế giới web (Screenshot-to-DOM)**.

#### 2. Thiết kế giao diện Slide
* **Tiêu đề slide:** *Pix2Struct: Tái Cấu Trúc Không Gian Thị Giác*
* **Chữ hiển thị trên slide:**
  * Từ khóa: `Screenshot-to-DOM Pretraining` • `Variable Aspect Ratio`
* **Hình ảnh trung tâm:**
  * Mô phỏng phép biến đổi 3 chiều:
    * Lớp trên: Ảnh chụp màn hình một trang web với các nút bấm, biểu đồ và văn bản.
    * Lớp dưới (trong suốt kéo xuống): Khung lưới HTML DOM dạng cây phân cấp hiển thị các thẻ `<div>`, `<table>`, `<button>`.

#### 3. Yêu cầu hình ảnh & Nguồn tài liệu
* **Nguồn ảnh:** Sơ đồ tiền huấn luyện từ bài báo *“Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding” (Kenton Lee et al., ACL 2023)*.
* **Ghi chú thiết kế:** Nhấn mạnh vào cơ chế bảo toàn tỷ lệ khung hình (Variable aspect ratio patch extraction) giúp mô hình không bị méo hình ảnh khi độ phân giải thay đổi.

#### 4. Lời thoại thuyết trình (Speaker Script - Cao Anh)
> *"Ở một hướng tiếp cận khác, các nhà nghiên cứu tại Google đã đặt ra câu hỏi: Đâu là bài toán tổng quát nhất để dạy cho máy tính hiểu bố cục thị giác? Câu trả lời của họ là **Pix2Struct**.  
> Họ thu thập hàng triệu ảnh chụp màn hình trang web, và bắt mô hình thực hiện một nhiệm vụ tưởng chừng bất khả thi: **Chỉ nhìn vào bức ảnh chụp, hãy tái tạo lại toàn bộ cây mã nguồn HTML DOM bên dưới**.  
> Nhờ nhiệm vụ tiền huấn luyện này, Pix2Struct học được mối liên hệ bản năng giữa hình ảnh thị giác và cấu trúc phân cấp logic: đâu là tiêu đề, đâu là bảng dữ liệu, và đâu là nội dung phụ trợ."*

---

### SLIDE 20: ƯU VÀ NHƯỢC ĐIỂM OCR-FREE (THE ARCHITECTURAL DILEMMA)

#### 1. Mục tiêu nhận thức (Core Takeaway)
Khán giả nhìn thấy rõ sự đánh đổi (Trade-off): OCR-Free giải quyết triệt để lỗi lan truyền, nhưng lại sinh ra căn bệnh nan y mới: **Ảo giác số liệu (Hallucination)** và **Trôi dạt lặp từ (Repetition Drift)** trên văn bản dài.

#### 2. Thiết kế giao diện Slide
* **Tiêu đề slide:** *Thế Lưỡng Nan của Kỷ Nguyên OCR-Free*
* **Chữ hiển thị trên slide:**
  * Cực bên trái (Xanh lá): `Ưu Điểm: Không lan truyền lỗi • Bố cục tự do`
  * Cực bên phải (Đỏ san hô): `Nhược Điểm: Ảo giác con số • Lặp từ vô hạn • Thiếu tọa độ`
* **Hình ảnh trung tâm:**
  * Một chiếc cân hình học đối xứng tinh tế:
    * Đĩa cân 1: Biểu tượng mắt xích liền mạch (End-to-End Elegance).
    * Đĩa cân 2: Một con số bị biến đổi ngẫu nhiên: `$1,000,000` ➔ sinh nhầm thành `$100,000` (mất số 0) và một vòng lặp vô tận của token `...table table table`.

#### 3. Yêu cầu hình ảnh & Nguồn tài liệu
* **Nguồn ảnh:** Thiết kế biểu đồ dạng so sánh trực quan đối xứng (Comparison Matrix) tối giản hoặc đồ thị radar biểu diễn trade-off giữa Accuracy vs Hallucination Risk.

#### 4. Lời thoại thuyết trình (Speaker Script - Cao Anh)
> *"Nhưng thưa các bạn, trong kỹ thuật máy tính, không có bữa trưa nào là miễn phí (No Free Lunch).  
> OCR-Free mang lại sự thanh lịch tuyệt đối: không còn mắt xích lỗi lan truyền, xử lý bố cục tự do. Nhưng đổi lại, nó đẩy chúng ta vào một **thế lưỡng nan mới**:  
> Bởi vì bộ giải mã là một mô hình ngôn ngữ tự hồi quy, khi gặp những trang tài liệu dày đặc con số tài chính hay văn bản quá dài, mô hình bắt đầu 'đoán mò' theo xác suất từ ngữ. Nó có thể tự ý làm rơi một số 0 trong bảng cân đối kế toán, hoặc rơi vào vòng lặp vô tận sinh lại cùng một đoạn văn. Hơn nữa, nó chỉ sinh ra văn bản mà **hoàn toàn mất đi tọa độ pixel** để chúng ta kiểm chứng.  
> Để khắc phục căn bệnh ảo giác này và mở rộng năng lực suy luận, nhân loại đã tiến tới một bước nhảy vọt vĩ đại hơn: **Vision Language Models (VLM)**."*

---

### SLIDE 21: KỶ NGUYÊN VISION LANGUAGE MODEL (VLM LÀ GÌ?)

#### 1. Mục tiêu nhận thức (Core Takeaway)
Định vị VLM không chỉ là một công cụ đọc tài liệu, mà là **sự giao thoa hoàn hảo giữa Khối Thị Giác (Eyes) và Khối Đại Não Ngôn Ngữ Khổng Lồ (Brain)**.

#### 2. Thiết kế giao diện Slide
* **Tiêu đề slide:** *Kỷ Nguyên VLM: Giao Thoa Giữa Thị Giác & Ngôn Ngữ*
* **Chữ hiển thị trên slide:**
  * Từ khóa: `Visual Perception (Mắt)` + `Reasoning Backbone (Não)` = `Multimodal Understanding`
* **Hình ảnh trung tâm:**
  * Sơ đồ Venn giao thoa rực rỡ nhưng trang nhã:
    * Vòng tròn 1 (Xanh dương): Không gian biểu diễn thế giới thực qua Điểm ảnh (Pixel Space / Vision Encoder).
    * Vòng tròn 2 (Tím ấm): Không gian biểu diễn tri thức và suy luận logic qua Ngôn ngữ (Semantic Space / LLM).
    * Vùng giao thoa ở giữa: Logo đại diện của kỷ nguyên VLM đa phương thức.

#### 3. Yêu cầu hình ảnh & Nguồn tài liệu
* **Nguồn ảnh:** Ý tưởng đồ họa vector 3Blue1Brown: Hai luồng sóng ánh sáng từ hai không gian khác nhau hội tụ tại một điểm tiêu cự.

#### 4. Lời thoại thuyết trình (Speaker Script - Cao Anh)
> *"Vậy chính xác thì **Vision Language Model (VLM)** là gì?  
> Nếu các mô hình OCR-Free trước đây như Donut giống như một người thư ký chỉ biết chép lại những gì mình thấy, thì VLM là sự kết hợp giữa **một đôi mắt sắc bén** và **một bộ não uyên bác**.  
> VLM không đơn thuần là đọc chữ. Nó đưa thế giới thị giác phức tạp của bức ảnh vào chung một không gian tư duy với mô hình ngôn ngữ lớn hàng tỷ tham số. Lần đầu tiên, cỗ máy không chỉ nhìn thấy bảng số liệu, mà nó có thể suy luận: 'Tại sao doanh thu quý này lại giảm so với quý trước?' ngay trên trang tài liệu."*

---

### SLIDE 22: KIẾN TRÚC TỔNG QUÁT VLM (THE UNIVERSAL TRIAD)

#### 1. Mục tiêu nhận thức (Core Takeaway)
Nắm vững **"Bộ Ba Vũ Trụ"** tạo nên mọi VLM hiện đại: Vision Encoder ➔ Projector (Cầu nối) ➔ LLM Backbone.

#### 2. Thiết kế giao diện Slide
* **Tiêu đề slide:** *Bộ Ba Kiến Trúc Của VLM*
* **Chữ hiển thị trên slide:**
  * Khối 1: `1. Vision Encoder (Thu nhận điểm ảnh)`
  * Khối 2: `2. Multimodal Projector (Phiên dịch ngôn ngữ)`
  * Khối 3: `3. LLM Backbone (Suy luận & Sinh phản hồi)`
* **Hình ảnh trung tâm:**
  * Sơ đồ dòng chảy 3 giai đoạn tinh giản tuyệt đối:
    1. Hình ảnh ➔ Đi qua lưới nơ-ron ViT ➔ Tạo ra hàng nghìn Visual Tokens.
    2. Đi qua **Khối Projector (MLP / Cross-Attention)**: Thu nhỏ và 'phiên dịch' vector ảnh về cùng số chiều với vector chữ.
    3. Trộn lẫn với Text Prompt ➔ Đi thẳng vào lõi LLM (Qwen / Llama) để suy luận.

#### 3. Yêu cầu hình ảnh & Nguồn tài liệu
* **Nguồn ảnh:** Lấy cảm hứng từ sơ đồ kiến trúc chuẩn của LLaVA (Haotian Liu et al., NeurIPS 2023) hoặc Qwen-VL Technical Report, vẽ lại theo phong cách tối giản phẳng.

#### 4. Lời thoại thuyết trình (Speaker Script - Cao Anh)
> *"Dù trên thị trường có hàng chục mô hình VLM khác nhau, tất cả chúng đều chia sẻ chung một **bộ ba kiến trúc vũ trụ**:  
> Đầu tiên là **Vision Encoder**—đóng vai trò võng mạc, biến hàng triệu điểm ảnh thành các gói token thị giác.  
> Ở giữa là thành phần kỳ diệu nhất: **Multimodal Projector**. Nó đóng vai trò như một người phiên dịch viên, chiếu các vector hình ảnh sang cùng một ngôn ngữ đại số mà mô hình ngôn ngữ có thể hiểu được.  
> Và cuối cùng là **LLM Backbone**—đại não tiếp nhận cả vector ảnh lẫn câu hỏi của bạn để tiến hành phân tích và đưa ra lời giải đáp. Một cấu trúc module hóa tuyệt đẹp và cực kỳ linh hoạt."*

---

### SLIDE 23: QWEN-VL (DYNAMIC RESOLUTION & 2D-ROPE)

#### 1. Mục tiêu nhận thức (Core Takeaway)
Hiểu được bước đột phá kỹ thuật then chốt của Qwen-VL: **Giải phóng độ phân giải cố định (Dynamic Resolution)** và **Định vị không gian 2 chiều (2D-RoPE)**.

#### 2. Thiết kế giao diện Slide
* **Tiêu đề slide:** *Qwen-VL: Đột Phá Độ Phân Giải Động*
* **Chữ hiển thị trên slide:**
  * Từ khóa: `Dynamic Resolution (Không méo ảnh)` • `2D-RoPE (Tọa độ 2 chiều)`
* **Hình ảnh trung tâm:**
  * So sánh đối chiếu 2 cơ chế xử lý ảnh:
    * **Cách cũ (Static Resizing):** Ảnh hóa đơn dài ngoằng bị bóp méo thành hình vuông 224x224 nhòe nhoẹt.
    * **Qwen-VL (Dynamic Patching):** Ảnh giữ nguyên tỷ lệ gốc, tự động phân rã thành số lượng patch linh hoạt theo đúng hình dáng thực tế, kèm ma trận lưới tọa độ 2D-RoPE bảo toàn vị trí dòng và cột.

#### 3. Yêu cầu hình ảnh & Nguồn tài liệu
* **Nguồn ảnh:** Trích xuất từ *Figure: Naive Dynamic Resolution and 2D-RoPE* trong bài báo *Qwen2-VL / Qwen2.5-VL Technical Report (Alibaba, 2024)*.

#### 4. Lời thoại thuyết trình (Speaker Script - Cao Anh)
> *"Một trong những đại diện xuất sắc nhất của kiến trúc này hiện nay chính là **Qwen-VL**, đặc biệt là thế hệ Qwen2.5-VL.  
> Điểm nghẽn lớn nhất của thị giác máy tính truyền thống là ép mọi bức ảnh về một hình vuông cố định, khiến các trang tài liệu khổ A4 dài bị bóp nghẹt và mờ chữ. Qwen-VL đã đập tan rào cản này bằng cơ chế **Naive Dynamic Resolution**: Ảnh dài hay ngắn, độ phân giải bao nhiêu, mô hình sẽ sinh ra số lượng token thị giác linh hoạt tương ứng mà không làm biến dạng tỷ lệ gốc.  
> Đồng thời, với mã hóa vị trí **2D-RoPE**, mô hình nắm bắt chính xác tọa độ hàng và cột trong không gian 2 chiều, biến nó thành một chuyên gia đọc hiểu tài liệu vượt trội."*

---

### SLIDE 24: VINTERN (MLLM CHUYÊN BIỆT CHO TÀI LIỆU VIỆT NAM)

#### 1. Mục tiêu nhận thức (Core Takeaway)
Tự hào giới thiệu giải pháp mã nguồn mở bản địa hóa cho Việt Nam: **Vintern-1B**, mô hình nhỏ gọn nhưng đánh bại các gã khổng lồ quốc tế trên tác vụ OCR và tài liệu tiếng Việt.

#### 2. Thiết kế giao diện Slide
* **Tiêu đề slide:** *Vintern-1B: Tối Ưu Hóa Bản Địa Cho Tiếng Việt*
* **Chữ hiển thị trên slide:**
  * Từ khóa: `1B Parameters (Nhẹ & Tinh gọn)` • `SOTA OCR Tiếng Việt & Dấu Thanh Điệu`
* **Hình ảnh trung tâm:**
  * Biểu đồ cột hiệu năng trực quan:
    * Thể hiện điểm số nhận dạng OCR tiếng Việt và hóa đơn: Vintern-1B (cột cao nổi bật màu xanh) vượt qua cả các mô hình 7B–14B quốc tế trên tập benchmark tiếng Việt.
    * Minh họa phóng to một cụm chữ tiếng Việt có dấu phức tạp (như *'Biên lai thu thuế'*) được mô hình nhận diện chính xác từng dấu mũ và dấu thanh.

#### 3. Yêu cầu hình ảnh & Nguồn tài liệu
* **Nguồn ảnh:** Logo và kết quả benchmark từ bài báo kỹ thuật *Vintern-1B Technical Report (5CD-AI / ngxson, 2024)* trên Hugging Face.

#### 4. Lời thoại thuyết trình (Speaker Script - Cao Anh)
> *"Nhưng đối với chúng ta, bài toán đặt ra là: Liệu một mô hình quốc tế có thể đọc tốt tiếng Việt với hệ thống dấu thanh điệu phức tạp và các mẫu hóa đơn đặc thù tại nước ta không?  
> Câu trả lời đến từ cộng đồng nghiên cứu Việt Nam với **Vintern-1B**.  
> Được xây dựng trên nền tảng InternVL, Vintern được tinh chỉnh chuyên sâu trên hàng triệu mẫu tài liệu và văn bản đời thực tại Việt Nam. Điều kỳ diệu là: Dù chỉ sở hữu **1 tỷ tham số**—hoàn toàn có thể chạy mượt mà trên một chiếc laptop cá nhân—Vintern-1B lại đạt độ chính xác OCR và đọc hiểu tài liệu tiếng Việt vượt trội hơn cả những mô hình khổng lồ hàng chục tỷ tham số của phương Tây. Một minh chứng tuyệt vời cho sức mạnh của sự bản địa hóa dữ liệu!"*

---

### SLIDE 25: TÀI LIỆU, BẢNG BIỂU VÀ BIỂU ĐỒ (UNIFIED MULTIMODAL REPRESENTATION)

#### 1. Mục tiêu nhận thức (Core Takeaway)
Tổng kết lại sự hợp nhất toàn vẹn: VLM không phân biệt giữa chữ, bảng, hay biểu đồ—chúng cùng hòa vào một **dòng chảy biểu diễn thống nhất**, mở đường cho Tầng Suy luận (Reasoning) tiếp theo.

#### 2. Thiết kế giao diện Slide
* **Tiêu đề slide:** *Sự Hợp Nhất Đa Phương Thức Toàn Diện*
* **Chữ hiển thị trên slide:**
  * Từ khóa: `Văn Bản • Bảng Biểu • Biểu Đồ` ➔ `Cấu Trúc Markdown Thống Nhất`
* **Hình ảnh trung tâm:**
  * Một trang Báo cáo tài chính phức tạp phân rã thành 3 luồng ánh sáng hội tụ:
    * Luồng 1 (Đoạn văn thuyết minh)
    * Luồng 2 (Bảng cân đối kế toán không viền)
    * Luồng 3 (Biểu đồ cột tăng trưởng)
    * Cả 3 cùng đi qua lăng kính VLM và xuất ra một trang tài liệu Markdown cấu trúc hoàn hảo, các con số được liên kết ngữ nghĩa chặt chẽ với nhau.

#### 3. Yêu cầu hình ảnh & Nguồn tài liệu
* **Nguồn ảnh:** Thiết kế đồ họa tổng hợp (Infographic) thể hiện khả năng "End-to-End Page Parsing" của VLM thế hệ mới.

#### 4. Lời thoại thuyết trình (Speaker Script - Cao Anh & Bàn giao)
> *"Thưa thầy và các bạn, nhìn lại cả một hành trình:  
> Chúng ta bắt đầu từ kỷ nguyên OCR phân mảnh với những mắt xích dễ vỡ; đi qua OCR-Free với khát vọng tinh gọn; và giờ đây dừng chân ở đỉnh cao **Vision Language Model**.  
> Trên slide 25 này, các bạn có thể thấy bức tranh toàn cảnh: Cho dù trang tài liệu có chứa văn bản dài, những bảng biểu không viền phức tạp, hay thậm chí là biểu đồ phân tích, VLM đều nhìn nhận chúng dưới một không gian biểu diễn thống nhất. Nó biến toàn bộ trang giấy thành dữ liệu số có cấu trúc và sẵn sàng cho các bài toán phân tích chuyên sâu.  
> Nhưng liệu cỗ máy chỉ dừng lại ở việc đọc và chép số liệu? **Làm sao để nó có thể tư duy, tự kiểm tra tính cân đối kế toán và thực hiện reasoning logic đỉnh cao trên những con số đó?**  
> Xin mời phần thuyết trình tiếp theo của bạn **Thanh Tâm** sẽ mở ra cho chúng ta kỷ nguyên kỳ diệu của **Reasoning trên VLM**!"*

---

## BẢNG TỔNG KẾT TÀI NGUYÊN HÌNH ẢNH CẦN THIẾT CHO DESIGNER

| Slide | Nội dung chính | Loại hình ảnh cần tìm / thiết kế | Nguồn tham khảo ưu tiên |
| :---: | :--- | :--- | :--- |
| **16** | Tại sao cần OCR-Free? | Sơ đồ so sánh Domino Failure vs. End-to-End Lens | Donut Paper (ECCV 2022) Fig 1 |
| **17** | Donut | Sơ đồ luồng: Swin Encoder -> BART Decoder -> JSON Tree | Donut Paper Fig 2 |
| **18** | Nougat | Cặp ảnh đối xứng: PDF scan công thức -> LaTeX render | Nougat Paper (Meta AI 2023) Fig 1 |
| **19** | Pix2Struct | Sơ đồ 3D: Screenshot web -> Cây HTML DOM | Pix2Struct Paper (ACL 2023) Fig 1 |
| **20** | Ưu/Nhược OCR-Free | Biểu tượng chiếc cân đối xứng giữa Flexibility vs Hallucination | Thiết kế vector phẳng tối giản |
| **21** | VLM là gì? | Sơ đồ Venn giao thoa giữa Pixel Space và Language Space | Minh họa phong cách 3Blue1Brown |
| **22** | Kiến trúc tổng quát | Sơ đồ 3 khối: Vision Encoder ➔ Projector ➔ LLM Backbone | LLaVA / Qwen-VL Architecture |
| **23** | Qwen-VL | So sánh biến dạng ảnh cố định vs. Chia patch động 2D-RoPE | Qwen2-VL Technical Report (2024) |
| **24** | Vintern | Biểu đồ cột so sánh điểm OCR tiếng Việt + Zoom chi tiết dấu | Vintern-1B Tech Report / Hugging Face |
| **25** | Hợp nhất đa phương thức | Infographic hội tụ 3 yếu tố (Text, Table, Chart) thành Markdown | Minh họa tổng hợp End-to-End VLM |
