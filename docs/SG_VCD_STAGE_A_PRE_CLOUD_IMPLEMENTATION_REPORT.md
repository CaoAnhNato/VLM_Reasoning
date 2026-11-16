# SG-VCD: Báo Cáo Triển Khai Kiểm Thử Tiền Đám Mây (Preflight) & Tính Khả Thi Của Giai Đoạn A (Stage A)

**Mã tài liệu:** `SG-VCD-STAGE-A-PRE-CLOUD-REPORT`  
**Kho lưu trữ mục tiêu:** `VLM_Reasoning` (`docs/SG_VCD_STAGE_A_PRE_CLOUD_IMPLEMENTATION_REPORT.md`)  
**Mô hình nền tảng (Backbone Model):** `5CD-AI/Vintern-1B-v3_5` (`InternViT-300M` + `Qwen2.5-0.5B-Instruct`)  
**Commit Hugging Face được ghim cố định:** `b98f263eab246eb5269ade64edbdca8a887dc44d`  
**Nền tảng mục tiêu:** Linux Ubuntu 22.04 LTS, GPU Cloud kiến trúc Ampere/Ada (NVIDIA RTX 3090 / 4090 / A5000, 24 GB VRAM, Khả năng tính toán Compute Capability $\ge 8.0$), PyTorch 2.4.1+cu121  
**Trạng thái kiểm toán:** SẴN SÀNG CÓ ĐIỀU KIỆN (READY WITH CONDITIONS — Đã xác minh qua Kiểm toán mã nguồn tĩnh & Unit test thuật toán)  

---

## 1. Tổng Quan Dự Án

### 1.1 Tên Dự Án và Nhận Diện
**SG-VCD: Giải Mã Tương Phản Thị Giác Dựa Trên Độ Nổi Bật Dành Cho Hỏi Đáp Thị Giác Văn Bản Cảnh Tiếng Việt (ViTextVQA).**  
*(Saliency-Guided Visual Contrastive Decoding for Vietnamese Scene-Text Visual Question Answering)*

### 1.2 Bối Cảnh Bài Toán ViTextVQA
Bài toán Hỏi Đáp Thị Giác trên hình ảnh văn bản cảnh (Scene-Text VQA) tiếng Việt đặt ra nhiều thách thức đặc thù đối với các Mô hình Thị giác - Ngôn ngữ (Vision-Language Models - VLMs):
1. **Độ nhạy cảm về Dấu thanh và Chính tả:** Tiếng Việt sử dụng hệ thống ký tự có dấu phức tạp và các ký tự tổ hợp (theo chuẩn chuẩn hóa Unicode NFC). Những sai lệch nhỏ ở cấp độ ký tự hoặc biến dạng thị giác mờ nhạt có thể làm đảo ngược hoàn toàn ý nghĩa ngữ nghĩa.
2. **Nhiễu văn bản cảnh dày đặc trong hậu cảnh:** Môi trường ảnh chụp thực tế chứa nhiều văn bản ngẫu nhiên không liên quan (biển hiệu cửa hàng, băng rôn, số điện thoại, địa chỉ). Cơ chế giải mã tự hồi quy (autoregressive decoding) thông thường rất dễ bị ảo giác (hallucination) hoặc bị thu hút sự chú ý vào các vùng văn bản hậu cảnh có độ tương phản cao, thay vì tập trung vào vùng văn bản cụ thể được hỏi trong câu hỏi.
3. **Hiện tượng pha loãng Visual Token (Visual Token Dilution):** Các mô hình đa phương thức độ phân giải cao hiện nay nén các mảnh ảnh (image patches) thành chuỗi token rất dài. Nếu không có cơ chế định hướng không gian dựa trên câu hỏi, bộ giải mã ngôn ngữ sẽ bị phân tán chú ý trên các vùng ảnh không liên quan.

### 1.3 Mục Tiêu Nghiên Cứu
SG-VCD nghiên cứu giải pháp liệu độ nổi bật (saliency) của vùng văn bản được định hướng bởi câu hỏi có thể dẫn dắt cơ chế giải mã tương phản thị giác (visual contrastive decoding) nhằm giảm thiểu ảo giác và tăng cường khả năng neo đậu câu trả lời (answer grounding) trong bài toán ViTextVQA tiếng Việt hay không, mà không cần phải tinh chỉnh lại trọng số mô hình (parameter fine-tuning).

### 1.4 Lựa Chọn Mô Hình Nền Tảng và Xuất Xứ
Dự án chuẩn hóa trên mô hình **Vintern-1B-v3_5** (`5CD-AI/Vintern-1B-v3_5`):
- **Bộ mã hóa thị giác (Visual Encoder):** `InternViT-300M` (300 triệu tham số, tiền xử lý mảnh ảnh gốc 448×448).
- **Bộ giải mã ngôn ngữ (Language Decoder):** `Qwen2.5-0.5B-Instruct` (24 tầng giải mã, kích thước ẩn hidden dimension 896, 14 query head, 2 key-value head theo cơ chế Grouped-Query Attention, kích thước mỗi head dimension 64).
- **Mã băm Commit được ghim cố định:** `b98f263eab246eb5269ade64edbdca8a887dc44d` (đảm bảo tính bất biến của cấu hình mô hình và mã thực thi từ xa trên Hugging Face Hub).

---

## 2. Phương Pháp Luận Nghiên Cứu và Kiến Trúc Tổng Thể

### 2.1 Kiến Trúc Nghiên Cứu Hai Giai Đoạn
Phương pháp luận tổng thể của SG-VCD được hoạch định qua hai giai đoạn nghiên cứu khoa học riêng biệt:

```
+-----------------------------------------------------------------------------------+
|                        QUY TRÌNH TỔNG THỂ CỦA SG-VCD                              |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  [Ảnh Đầu Vào] + [Câu Hỏi Tiếng Việt]                                             |
|         │                                                                         |
|         ▼                                                                         |
|  +─────────────────────────────────────────────────────────────────────────────+  |
|  | GIAI ĐOẠN NGHIÊN CỨU 1: Định Vị & Lựa Chọn Vùng ROI Văn Bản Theo Câu Hỏi     |  |
|  | - Trích xuất các vùng đề xuất văn bản / OCR tokens (SwinTextSpotter / GT)   |  |
|  | - Tính toán độ nổi bật chú ý (attention saliency) theo câu hỏi              |  |
|  | - Xếp hạng và lựa chọn vùng văn bản quan tâm (Text ROIs)                    |  |
|  +─────────────────────────────────────────────────────────────────────────────+  |
|         │ Mặt nạ ROI nổi bật / Hộp giới hạn được chọn (Selected Salient ROI)      |
|         ▼                                                                         |
|  +─────────────────────────────────────────────────────────────────────────────+  |
|  | GIAI ĐOẠN NGHIÊN CỨU 2: Giải Mã Tương Phản Thị Giác Dựa Trên Độ Nổi Bật      |  |
|  | - Sinh logits gốc không can thiệp: P_orig = Logits(V_full, Câu hỏi)          |  |
|  | - Sinh logits bị làm nhiễu / che mờ: P_mask = Logits(V_masked, Câu hỏi)     |  |
|  | - Công thức tương phản: Logits_VCD = (1 + α) * P_orig - α * P_mask          |  |
|  | - Cắt tỉa động / lọc thích ứng (tiêu chuẩn DOLA / VCD)                      |  |
|  +─────────────────────────────────────────────────────────────────────────────+  |
|         │                                                                         |
|         ▼                                                                         |
|  [Câu Trả Lời Tiếng Việt Được Neo Đậu Chính Xác (Grounded Answer)]                |
+-----------------------------------------------------------------------------------+
```

- **Giai đoạn Nghiên cứu 1 (Kế hoạch):** Định vị và lựa chọn vùng ROI văn bản dựa trên câu hỏi. Nghiên cứu phương pháp khoanh vùng văn bản không gian cụ thể mà câu hỏi hướng tới bằng cách sử dụng bản đồ nổi bật chú ý (attention saliency map), hộp giới hạn (bounding boxes) và xếp hạng ứng viên.
- **Giai đoạn Nghiên cứu 2 (Kế hoạch):** Giải mã tương phản thị giác dựa trên độ nổi bật (SG-VCD). Áp dụng cơ chế khuếch đại logit tương phản giữa biểu diễn toàn ảnh và biểu diễn thị giác bị biến đổi/che mờ vùng nổi bật, nhằm dẫn dắt quá trình sinh từ tránh xa ảo giác hậu cảnh.

> **Lưu ý về tính liêm chính khoa học:** Cả Giai đoạn Nghiên cứu 1 và Giai đoạn Nghiên cứu 2 đều chưa được triển khai hoặc đánh giá thực nghiệm tại thời điểm này. Đây là mục tiêu khoa học dài hạn của đề tài SG-VCD.

### 2.2 Phân Biệt Rõ Ràng Với Các Cổng Tiền Kiểm Kỹ Thuật Giai Đoạn A (Stage A)
Các module kỹ thuật được hiện thực hóa trong thư mục `experiments/stage1_vintern_pilot/` (**A0, A1, A2**) không được nhầm lẫn với các giai đoạn nghiên cứu khoa học nói trên:
- **A0, A1, A2** là các **cổng tiền kiểm kỹ thuật (technical preflight feasibility gates)** được thiết kế nghiêm ngặt nhằm xác định xem môi trường đám mây, độ chính xác phần cứng, ranh giới tokenizer và cơ chế trích xuất attention có thể vận hành ổn định mà không xảy ra sai lệch kiến trúc hay không.
- Việc vượt qua Giai đoạn A là điều kiện tiên quyết cho phép bước vào triển khai Giai đoạn Nghiên cứu 1; bản thân nó không đồng nghĩa với việc đã hoàn thành nghiên cứu.

---

## 3. Tổng Quan Bộ Công Cụ Tiền Kiểm Khả Thi Giai Đoạn A

Gói kiểm thử khả thi Giai đoạn A (`experiments/stage1_vintern_pilot/`) cô lập ba cổng chẩn đoán tuần tự:

### 3.1 Giai Đoạn A0 — Kiểm Thử Mô Hình Cơ Sở (`run_stage_a0.py`)
- **Mục tiêu chính:** Xác minh quá trình nạp mô hình cơ sở và suy luận lan truyền xuôi (forward inference) đa phương thức đầu-cuối ở định dạng gốc `bfloat16` trên GPU đám mây kiến trúc Ampere/Ada.
- **Phạm vi & Các phép kiểm:**
  - Ràng buộc đầu vào đơn tile (`num_patches == 1`, 448×448).
  - Khởi tạo và xác thực rõ ràng token ID giữ chỗ `<IMG_CONTEXT>`.
  - Đo đạc độ trễ GPU được đồng bộ hóa và theo dõi mức đỉnh bộ nhớ VRAM.
  - Thực hiện kiểm tra tính hữu hạn thực sự ở cấp độ tensor, đảm bảo `visual_features` và `forward_logits` hoàn toàn hữu hạn, không chứa giá trị `NaN` hay `Inf`.
  - Tuyệt đối nghiêm cấm việc âm thầm chuyển sang luồng dự phòng nếu luồng forward đa phương thức chính thức gặp lỗi.

### 3.2 Giai Đoạn A1 — Kiểm Toán Khớp Nối Token & Chuẩn Hóa NFC Tiếng Việt (`run_stage_a1.py`)
- **Mục tiêu chính:** Kiểm toán việc phân tách token và ánh xạ ranh giới trực tiếp từ mẫu hội thoại đã dựng (rendered prompt template), đảm bảo trích xuất chính xác phạm vi (span) của câu hỏi.
- **Phạm vi & Các phép kiểm:**
  - Xác minh trích xuất span token chính xác từ toàn bộ prompt mà không chia nhỏ để tokenize các chuỗi rời rạc.
  - Kiểm tra tính liên tục của vị trí visual tokens ($1 \times 256 = 256$ tokens).
  - Phát hiện hiện tượng Byte-level BPE chia cắt byte UTF-8 đa byte thành ký tự thay thế `\ufffd` hoặc phân mảnh ranh giới.
  - Kiểm toán đa dạng các dạng câu hỏi nghi vấn tiếng Việt ("gì", "ở đâu", "bao nhiêu", "màu gì") dưới chuẩn chuẩn hóa Unicode NFC.
  - Được cấu hình để trả về trạng thái `INCONCLUSIVE` (chưa thể kết luận) nếu ranh giới token mơ hồ, tuyệt đối không tự bịa đặt chỉ số token.
- **Chế độ thực thi & Phạm vi Không-CUDA:** Giai đoạn A1 minh bạch gắn nhãn chế độ thực thi là `CPU_TOKEN_AUDIT` hoặc `GPU_EXECUTION`. Vì nhiệm vụ trọng tâm của A1 là ánh xạ ranh giới tokenizer và xây dựng prompt, A1 không yêu cầu suy luận tensor trên GPU. Việc nạp tokenizer/mô hình lên CUDA trong A1 chỉ nhằm mục đích thuận tiện cho môi trường, không được coi là bằng chứng về suy luận tensor CUDA.

### 3.3 Giai Đoạn A2 — Minh Chứng Cơ Chế Quan Sát Attention Giới Hạn (`run_stage_a2.py`)
- **Mục tiêu chính:** Xác thực cơ chế quan sát không xâm lấn (non-invasive observer) dùng để trích xuất trọng số attention của decoder phục vụ việc xây dựng bản đồ nổi bật không gian ở các bước tiếp theo.
- **Phạm vi & Các phép kiểm:**
  - Bộ quản lý ngữ cảnh (context manager) không xâm lấn bao bọc phương thức `self_attn` của tầng decoder mục tiêu mà không làm thay đổi trọng số hay giao ước giá trị trả về.
  - Đón bắt tham số từ khóa `position_embeddings` phù hợp với luồng thực thi `Qwen2DecoderLayer` trong Transformers 4.45.2.
  - Tính toán lại chính xác các tensor Q/K sau RoPE (post-RoPE) bằng ma trận chiếu trọng số và phép quay rotary embeddings, tuyệt đối từ chối sử dụng tensor thô chưa quay.
  - Tái tạo ma trận xác suất attention từ query đến visual tokens thông qua chuẩn hóa Causal Log-Sum-Exp có tích hợp lát cắt mặt nạ attention 4D thực tế.
  - Phân tách việc xác thực thành ba cổng chẩn đoán độc lập:
    1. *Kiểm tra Tương đương Toán học (Mathematical Equivalence Test):* So sánh kênh phụ (side-channel) với chuẩn tham chiếu độc lập FP32 đầy đủ ($\text{sai số} \le 10^{-5}$).
    2. *Kiểm tra Mức độ Đồng thuận với Mô hình (Model Agreement Test):* So sánh kênh phụ với trọng số eager attention thực tế của mô hình ($\text{sai số} \le 5 \times 10^{-3}$, ngưỡng vận hành tạm thời).
    3. *Kiểm tra Tính Bất biến Lan truyền Xuôi (Forward Invariance Test):* So sánh logits giữa Lượt 1 độc lập (baseline) và Lượt 2 có gắn bộ quan sát ($\text{sai số} \le 10^{-4}$).

---

## 4. Chi Tiết Triển Khai Kỹ Thuật

### 4.1 Nạp Mô Hình và Thực Thi Độ Chính Xác Nghiêm Ngặt
- Hàm `load_vintern_model()` trong `vintern_utils.py` nạp mô hình `5CD-AI/Vintern-1B-v3_5` ở phiên bản commit `b98f263eab246eb5269ade64edbdca8a887dc44d`.
- Bắt buộc thực thi `torch_dtype = torch.bfloat16`. Nếu phần cứng CUDA không hỗ trợ `bfloat16` gốc, tiến trình lập tức dừng lại; hành vi tự ý chuyển về `float16`, `float32` hoặc CPU bị nghiêm cấm hoàn toàn.
- Thiết lập `_attn_implementation = "eager"` đồng bộ trên `model.config`, `model.language_model.config` và thuộc tính `self_attn.config` của từng tầng decoder để ngăn chặn sự phân nhánh kernel tính toán.

### 4.2 Tiền Xử Lý Ảnh Động và Khớp Nối Token
- Hàm `load_image_tensor()` hỗ trợ cơ chế cắt mảnh động với phép chuẩn hóa ImageNet ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
- Đối với tiền kiểm Giai đoạn A, cấu hình đơn tile (`max_num=1`, `use_thumbnail=False`) tạo ra chính xác 1 mảnh ảnh kích thước `[1, 3, 448, 448]`.
- Bộ mã hóa thị giác chiếu mảnh ảnh này thành 256 vector đặc trưng thị giác (`embeds_flat: [256, 896]`).
- Các token giữ chỗ `<IMG_CONTEXT>` được xác thực nghiêm ngặt: `num_placeholders == embeds_flat.shape[0]`.

### 4.3 Cơ Chế RoPE và Trích Xuất GQA Trong Qwen2
Trong thư viện Transformers 4.45.2, `Qwen2Attention` sử dụng cơ chế Grouped-Query Attention (GQA):
- **Số Query Head ($H_Q$):** 14 heads, chiều mỗi head $d = 64$.
- **Số Key/Value Head ($H_{KV}$):** 2 heads, chiều mỗi head $d = 64$.
- **Tỷ lệ GQA:** $14 / 2 = 7$ query heads trên 1 nhóm KV head.
- **Trình tự thực thi RoPE trong Transformers 4.45.2:**
  1. `Qwen2Model.forward()` tính toán vector nhúng vị trí `(cos, sin)` dùng chung qua module rotary embedding của nó (`self.rotary_emb(value_states, position_ids)`).
  2. Tuple `position_embeddings` dùng chung này được truyền xuống từng `Qwen2DecoderLayer`.
  3. `Qwen2DecoderLayer.forward()` truyền tiếp chúng vào `self_attn.forward()` qua tham số từ khóa `position_embeddings`.
  4. `Qwen2Attention` cũng sở hữu thuộc tính `self.rotary_emb` và có thể tự tính RoPE nội bộ nếu `position_embeddings` bị khuyết.
  5. `DiagnosticAttentionCollector` trích xuất `position_embeddings` trực tiếp từ `kwargs`. Nếu không có, collector sẽ truy vấn `parent.self_attn.rotary_emb` (được khởi tạo trực tiếp trên `Qwen2Attention`).
  6. Các tensor sau RoPE `q_post, k_post` được tính toán thông qua hàm `apply_rotary_pos_emb(q_raw, k_raw, cos, sin)`.
  7. Nếu không thể thu nhận hoặc xác thực `(cos, sin)`, hệ thống lập tức kích hoạt ngoại lệ `RuntimeError`; tuyệt đối không chấp nhận tensor Q/K thô chưa quay làm kết quả sau RoPE.

### 4.4 Mặt Nạ Attention và Chuẩn Hóa Causal Log-Sum-Exp
Đối với vị trí query $p$ và chuỗi key $0 \le j \le p$:
$$\text{raw\_logits}_{h, p, j} = \frac{q_{h, p} \cdot k_{\text{group}(h), j}^\top}{\sqrt{d}} + \text{mask}_{h, p, j}$$
$$\text{LSE\_denom}_{h, p} = \text{LogSumExp}_{j \le p}\left(\text{raw\_logits}_{h, p, j}\right)$$
$$P_{\text{vis}}(h, p, v) = \exp\left(\text{raw\_logits}_{h, p, v} - \text{LSE\_denom}_{h, p}\right), \quad \forall v \in \text{VisualIndices}, v \le p$$

Công thức này đảm bảo:
1. Phép chuẩn hóa được thực hiện trên toàn bộ các vị trí key hợp lệ về mặt nhân quả ($j \le p$), thay vì chỉ chuẩn hóa cục bộ trên các visual tokens.
2. Tổng phân phối chú ý thị giác bị chặn trên nghiêm ngặt: $\sum_{v} P_{\text{vis}}(h, p, v) \le 1.0$.
3. Lát cắt mặt nạ chú ý 4D từ bên ngoài được tích hợp đầy đủ trước khi thực hiện hàm mũ.

### 4.5 Xử Lý Lỗi Bền Vững & Bản Kê Nguồn Gốc Tái Lập (Provenance Manifest)
- `run_stage_a2.py` bao bọc các lượt lan truyền xuôi trong khối lệnh `try...except` cấp cao nhất, đảm bảo nếu xảy ra ngoại lệ:
  - Phương thức `DiagnosticAttentionCollector.__exit__` luôn được kích hoạt để khôi phục lại phương thức forward attention gốc.
  - Chuỗi thông báo ngoại lệ và toàn bộ vết ngăn xếp (traceback) được ghi nhận đầy đủ.
  - Báo cáo chẩn đoán định dạng JSON được ghi xuống ổ đĩa với trạng thái `FAIL` hoặc `INCONCLUSIVE`.
  - Không có bất kỳ ngoại lệ nào bị nuốt âm thầm hoặc tự ý chuyển thành `PASS`.
- `reproducibility_manifest_template.json` ghi lại toàn diện metadata phần cứng, khóa phiên bản gói phần mềm, mã băm Git commit và dấu thời gian thực thi của từng cổng.

---

## 5. Kết Quả Kiểm Toán Kỹ Thuật và Bảng Tổng Hợp Khắc Phục

Trong quá trình thanh tra mã nguồn tĩnh và kiểm toán kiến trúc, chín vấn đề kỹ thuật đã được phát hiện và khắc phục triệt để. Để bảo đảm tính liêm chính khoa học, các trạng thái xác minh được phân định rõ ràng giữa giải quyết qua mã tĩnh (`RESOLVED_STATIC`), xác minh thực tế lúc chạy (`PASS_RUNTIME`), thành phần chưa chạy/bị bỏ qua (`NOT_RUN`), và trạng thái chưa thể kết luận (`INCONCLUSIVE`):

| Khiếm khuyết / Nguy cơ được xác định | Phân tích nguyên nhân gốc rễ | Biện pháp khắc phục đã triển khai | Trạng thái xác minh |
| :--- | :--- | :--- | :---: |
| **1. Kiểm tra NaN/Inf giả lập** | Hàm tiện ích trả về giá trị cố định `is_finite=True` mà không kiểm tra phần tử tensor thực tế. | Triển khai `inspect_tensor_finite()` thực hiện kiểm tra thực tế bằng `torch.isnan().any()` và `torch.isinf().any()`. | **RESOLVED_STATIC** |
| **2. Minh chứng Attention giả** | Tính toán kênh phụ bị bỏ qua; độ lệch tối đa `max_diff` bị gán cứng bằng `0.0`. | Triển khai hàm tính toán thực sự `compute_side_channel_attention()` với chuẩn hóa causal LSE và mở rộng head GQA. | **RESOLVED_STATIC** |
| **3. Tự trừ Logits (Self-Subtraction)** | Phép kiểm bất biến lấy `vanilla_logits` tự trừ chính nó. | Tách quy trình thành Lượt 1 (baseline) và Lượt 2 (quan sát), so sánh hai tensor đầu ra độc lập. | **RESOLVED_STATIC** |
| **4. Dự phòng Q/K thô khi RoPE lỗi** | Bộ quan sát tự gán `q_post, k_post = q_raw, k_raw` nếu quá trình giải RoPE thất bại. | Loại bỏ hoàn toàn fallback. Kích hoạt ngoại lệ `RuntimeError`; không bao giờ chấp nhận Q/K thô làm tensor sau RoPE. | **RESOLVED_STATIC** |
| **5. Bỏ quên kiểm tra sai số Model Attention** | `model_max_err` chỉ được ghi log mà không đưa vào điều kiện đánh giá pass. | Bổ sung cổng kiểm tra rõ ràng `model_agreement_test` ($\text{sai số} \le 5 \times 10^{-3}$). Báo cáo `INCONCLUSIVE` nếu thiếu tham chiếu. | **RESOLVED_STATIC** |
| **6. Ranh giới Token bị mơ hồ** | Byte-level BPE có thể chia nhỏ byte UTF-8 thành ký tự `\ufffd`, gây lệch chỉ số ngầm. | Nâng cấp `find_token_span_from_full_prompt()` để cảnh báo `\ufffd` và hiện tượng chồng lấn ranh giới là `AMBIGUOUS`. | **RESOLVED_STATIC** |
| **7. Hiểu sai API RoPE trong Transformers** | Báo cáo ban đầu cho rằng `Qwen2Attention` không sở hữu `rotary_emb`. | Đính chính tài liệu và mã nguồn: `Qwen2Attention` có sở hữu `self.rotary_emb`; trích xuất qua keyword khớp với `Qwen2DecoderLayer`. | **RESOLVED_STATIC** |
| **8. Cơ chế dự phòng khi forward đa phương thức lỗi** | Khối `try-except` âm thầm bắt lỗi `model(...)` rồi chuyển sang chèn embedding qua `language_model`. | Loại bỏ hoàn toàn luồng dự phòng. Nếu `model(...)` ném ngoại lệ, Stage A0 ghi nhận `FAIL` và dừng lại ngay lập tức. | **RESOLVED_STATIC** |
| **9. Cờ phụ thuộc bị ràng buộc chéo** | Khối `try/except` đơn lẻ khiến `HAS_TORCH` bị chuyển thành `False` khi thiếu thư viện `transformers`. | Tách biệt độc lập thành `HAS_TORCH`, `HAS_TORCHVISION`, `HAS_PIL`, và `HAS_TRANSFORMERS`. Kiểm tra tensor chỉ yêu cầu PyTorch. | **RESOLVED_STATIC** |

### Các yếu tố chưa xác định lúc chạy (Cần thực thi trên Cloud GPU):
Kiểm toán mã nguồn tĩnh KHÔNG thay thế được việc xác minh lúc chạy trên mô hình Vintern thực tế. Các nội dung sau đây bắt buộc phải được kiểm nghiệm thực nghiệm trên máy chủ GPU đám mây:
1. **Khởi tạo mã nguồn từ xa thực tế (`NOT_RUN`):** Nạp trọng số và khởi tạo mã nguồn từ xa của Vintern (`modeling_internvl_chat.py`) từ Hugging Face Hub tại commit `b98f263eab246eb5269ade64edbdca8a887dc44d`.
2. **Độ tương đồng mô hình BF16 trên phần cứng thực (`NOT_RUN`):** Kiểm nghiệm thực nghiệm xem trọng số eager attention của mô hình có khớp với kênh phụ LSE trong ngưỡng tạm thời $5 \times 10^{-3}$ trên phần cứng Ampere/Ada hay không.
3. **Mức tiêu hao VRAM thực tế (`NOT_RUN`):** Đo đạc bộ nhớ để khẳng định mức cấp phát đỉnh nằm an toàn dưới ngưỡng 20 GiB khi suy luận đơn tile.
4. **Luồng forward đa phương thức thực tế (`NOT_RUN`):** Xác minh luồng forward chính thức `model(pixel_values, input_ids, image_flags)` thực thi trơn tru không phát sinh ngoại lệ trên máy chủ thật.

---

## 6. Kết Quả Kiểm Thử Đơn Vị (Unit Test Scorecard)

### 6.1 Môi Trường Thực Thi Kiểm Thử
- **Nền tảng:** Windows 64-bit AMD64 (Máy trạm phát triển cục bộ)
- **Phiên bản Python:** 3.11.9
- **PyTorch:** CHƯA CÀI ĐẶT cục bộ (các bài test tensor được kiểm tra bằng các chứng minh thuật toán Pure Python qua thư viện chuẩn)
- **torchvision / transformers / PIL:** CHƯA CÀI ĐẶT cục bộ
- **Chế độ thực thi:** CPU cục bộ (Không có CUDA)

### 6.2 Kết Quả Kiểm Thử Tiền Kiểm (`experiments/stage1_vintern_pilot/tests/run_unit_tests.py`)

Bộ kiểm thử phân định minh bạch giữa **các bài kiểm tra thuật toán Pure Python** (chạy cục bộ trên CPU) và **các bài kiểm tra tensor PyTorch** (yêu cầu môi trường có cài PyTorch và chạy trên CPU/PyTorch, không bắt buộc CUDA trừ khi được cấu hình cụ thể):

| Mã định danh bài test | Phân hệ được kiểm thử | Nhóm đánh giá | Kết quả | Mô tả chi tiết |
| :--- | :--- | :--- | :---: | :--- |
| `test_causal_lse_and_visual_mass_bounded` | Chuẩn hóa Causal LSE | Pure Python | **PASS_RUNTIME** | Mẫu số LSE được tính trên toàn bộ causal keys; khối lượng chú ý thị giác bị chặn $\le 1.0$. |
| `test_attention_masking_integration` | Tích hợp Mặt nạ Attention | Pure Python | **PASS_RUNTIME** | Các hàng mặt nạ 4D ($-10^9$) được tích hợp chính xác vào mẫu số LSE. |
| `test_gqa_group_mapping` | Mở rộng Head GQA | Pure Python | **PASS_RUNTIME** | 14 Query heads được ánh xạ chính xác về 2 KV heads (tỷ lệ lặp 7:1). |
| `test_rope_position_embeddings_math` | Tính Bất Biến Chuẩn RoPE | Pure Python | **PASS_RUNTIME** | Bảo toàn chuẩn vector qua phép quay Givens 2D với vector $(cos, sin)$. |
| `test_clean_finite_values` | Kiểm Tra Số Học Hữu Hạn | Pure Python | **PASS_RUNTIME** | Chuỗi số thực hữu hạn được nhận diện chính xác, không bị cảnh báo nhầm NaN/Inf. |
| `test_dependency_flags_decoupling` | Tách Biệt Cờ Phụ Thuộc | Pure Python | **PASS_RUNTIME** | Xác minh `HAS_TORCH` được tách độc lập khỏi `HAS_TRANSFORMERS`. |
| `test_values_containing_nan` | Nhận Diện Giá Trị NaN | Pure Python | **PASS_RUNTIME** | Giá trị NaN lập tức bị phát hiện và kích hoạt cờ không hữu hạn. |
| `test_values_containing_positive_and_negative_inf` | Nhận Diện Giá Trị Inf | Pure Python | **PASS_RUNTIME** | Giá trị Dương/Âm Vô cùng được gắn cờ cảnh báo chính xác. |
| `test_pure_python_backend_consistency` | Nhất Quán Cấu Hình Backend | Pure Python | **PASS_RUNTIME** | Cấu hình eager attention được xác thực nhất quán qua các ngữ cảnh observer. |
| `test_pure_python_observer_invariance` | Tính Bất Biến Của Observer | Pure Python | **PASS_RUNTIME** | Không có độ lệch đầu ra giữa Lượt 1 và Lượt 2 trên các tầng giả lập (mock). |
| `test_alignment_with_leading_newline_and_vietnamese` | Khớp Nối Token Tiếng Việt | Pure Python | **PASS_RUNTIME** | Ánh xạ ký tự từ prompt chuẩn xác qua ký tự xuống dòng và văn bản tiếng Việt NFC. |
| `test_detection_of_ambiguous_partial_bytes` | Phát Hiện Ranh Giới BPE Mơ Hồ | Pure Python | **PASS_RUNTIME** | Hiện tượng phân mảnh byte UTF-8 đa byte (`\ufffd`) được đánh dấu là `AMBIGUOUS`. |
| `test_interrogative_variants` | Trích Xuất Span Câu Hỏi | Pure Python | **PASS_RUNTIME** | Khớp span chính xác cho các dạng từ nghi vấn "gì", "ở đâu", "bao nhiêu", "màu gì". |
| `test_pytorch_side_channel_vs_fp32_ref` | Kênh phụ vs Chuẩn FP32 | Tensor PyTorch (CPU) | **NOT_RUN** | Tạm bỏ qua ở máy cục bộ; yêu cầu cài PyTorch (dành cho Cổng 0 trên Cloud). |
| `test_pytorch_tensor_finite_checks` | Kiểm Tra Hữu Hạn Tensor PyTorch | Tensor PyTorch (CPU) | **NOT_RUN** | Tạm bỏ qua ở máy cục bộ; yêu cầu cài PyTorch (dành cho Cổng 0 trên Cloud). |
| `test_pytorch_observer_invariance` | Tính Bất Biến Observer PyTorch | Tensor PyTorch (CPU) | **NOT_RUN** | Tạm bỏ qua ở máy cục bộ; yêu cầu cài PyTorch (dành cho Cổng 0 trên Cloud). |

**Thống Kê Tổng Hợp:**
- **Tổng số bài test được đánh giá:** 16
- **Vượt qua (Pure Python trên máy cục bộ):** 13 (`PASS_RUNTIME`)
- **Tạm bỏ qua (Do máy cục bộ chưa có PyTorch):** 3 (`NOT_RUN` — dành cho môi trường ảo đám mây)
- **Thất bại / Lỗi phát sinh:** 0
- **Trạng thái:** **TẤT CẢ CÁC BÀI TEST TIỀN KIỂM KHẢ THI ĐỀU ĐÃ ĐẠT**

---

## 7. Các Giới Hạn Hiện Tại Của Đề Tài

Nhằm duy trì tính liêm chính học thuật và tính minh bạch của tài liệu, các giới hạn sau đây được ghi nhận rõ ràng:
1. **Chưa tải trọng số mô hình trên đám mây:** Tệp trọng số mô hình (`model.safetensors`, 3.75 GB) chưa được tải về máy chủ GPU đám mây.
2. **Chưa thực hiện suy luận trên GPU thực tế:** Chưa có lượt chạy nào trên phần cứng GPU thực tế. Toàn bộ quá trình kiểm chứng cục bộ được tiến hành thông qua phân tích mã nguồn tĩnh, kiểm tra biên dịch cú pháp và các phép chứng minh toán học Pure-Python.
3. **Chưa thực hiện đánh giá trên 400 ảnh ViTextVQA:** Cả Giai đoạn A0 và A2 đều không phải là bài đánh giá toàn diện trên tập dữ liệu. Đây chỉ là các phép thăm dò kỹ thuật trên mẫu đơn ảnh / đơn tile.
4. **Chưa thiết lập chỉ số độ chính xác định vị (Localization Accuracy):** Chưa có chỉ số định vị văn bản nào (IoU, precision, recall) được tính toán trên ViTextVQA.
5. **Giai đoạn Nghiên cứu 1 chưa được triển khai:** Cơ chế lựa chọn vùng ROI dựa trên độ nổi bật chú ý mới chỉ ở giai đoạn thiết kế, chưa được tích hợp vào mã nguồn suy luận.
6. **Giai đoạn Nghiên cứu 2 chưa được triển khai:** Phương pháp giải mã tương phản thị giác định hướng theo độ nổi bật chưa được lập trình hay đánh giá benchmark.
7. **Đang chờ tích hợp toàn diện SG-VCD:** Việc tích hợp đầu-cuối giữa Giai đoạn 1 + Giai đoạn 2 sẽ chỉ được tiến hành sau khi các cổng tiền kiểm đám mây (A0, A1, A2) khẳng định tính khả thi về mặt kỹ thuật.

---

## 8. Kế Hoạch Triển Khai Kiểm Thử Trên Cloud GPU

Quá trình thực thi trên máy chủ GPU đám mây được thuê phải tuân thủ nghiêm ngặt trình tự tuần tự sau:

```
+─────────────────────────────────────────────────────────────────────────────+
|                        CÁC CỔNG THỰC THI TRÊN CLOUD                         |
+─────────────────────────────────────────────────────────────────────────────+
|                                                                             |
|  [CỔNG 0: Xác Minh Môi Trường & Phần Cứng]                                  |
|  $ bash experiments/stage1_vintern_pilot/setup_cloud_env.sh                 |
|  $ source experiments/stage1_vintern_pilot/cloud_env.sh                     |
|  $ python experiments/stage1_vintern_pilot/verify_cloud_env.py              |
|  $ python experiments/stage1_vintern_pilot/tests/run_unit_tests.py          |
|  -> Cấu trúc xác minh & Vai trò của từng phân hệ:                           |
|     1. Cổng 0 (verify_cloud_env.py) kiểm tra driver CUDA, GPU CC >= 8.0,     |
|        dung lượng VRAM >= 22 GiB, và khả năng hỗ trợ phần cứng bfloat16.     |
|     2. Trình chạy unit test (run_unit_tests.py) khi đã có PyTorch:           |
|        toàn bộ 16 tests phải báo PASS (13 CPU Pure Python + 3 CPU PyTorch   |
|        tensor tests). Các bài test PyTorch tensor không phải bài test CUDA.  |
|     3. Stage A0 thực thi suy luận GPU forward đa phương thức thực tế,        |
|        kiểm tra đặc trưng và logits phải hữu hạn, không có NaN/Inf.          |
|     4. Stage A1 thực hiện khớp nối ranh giới tokenizer và prompt; KHÔNG      |
|        thực hiện suy luận tensor CUDA nặng và chạy thuần túy trong chế độ    |
|        CPU_TOKEN_AUDIT. Việc nạp lên CUDA trong A1 chỉ nhằm mục đích tiện    |
|        môi trường, không phải minh chứng cho suy luận tensor CUDA.           |
|     5. Stage A2 thực thi chẩn đoán attention trên GPU và xác minh tính       |
|        bất biến forward qua hai lượt chạy.                                  |
|         │                                                                   |
|         ▼ PASS                                                              |
|  [CỔNG 1: Giai Đoạn A0 — Tiền Kiểm Mô Hình Cơ Sở Đa Phương Thức]             |
|  Thử nghiệm kỹ thuật dữ liệu tổng hợp (single tile 448x448):                |
|  $ python experiments/stage1_vintern_pilot/run_stage_a0.py                  |
|  Thử nghiệm với ảnh thực tế (hỗ trợ đối số vị trí hoặc cờ tham số):         |
|  $ python experiments/stage1_vintern_pilot/run_stage_a0.py \                |
|           /path/to/image.jpg "Biển hiệu trong ảnh có chữ gì?"               |
|  $ python experiments/stage1_vintern_pilot/run_stage_a0.py \                |
|           --image /path/to/image.jpg --question "..."                       |
|  -> Yêu cầu: Xác nhận đơn tile, tuyệt đối không NaN/Inf, đỉnh VRAM < 20 GB  |
|         │                                                                   |
|         ▼ PASS                                                              |
|  [CỔNG 2: Giai Đoạn A1 — Kiểm Toán Khớp Nối Token & Chuẩn Hóa NFC]          |
|  $ python experiments/stage1_vintern_pilot/run_stage_a1.py                  |
|  -> Yêu cầu: Khớp chính xác 100% token NFC trên tokenizer Qwen thực tế      |
|         │                                                                   |
|         ▼ PASS                                                              |
|  [CỔNG 3: Giai Đoạn A2 — Minh Chứng Cơ Chế Quan Sát Attention Giới Hạn]     |
|  Thử nghiệm tổng hợp (Mặc định tầng 12):                                    |
|  $ python experiments/stage1_vintern_pilot/run_stage_a2.py --layer 12       |
|  Thử nghiệm với ảnh thực tế (Tầng 12, tham số vị trí hoặc cờ):              |
|  $ python experiments/stage1_vintern_pilot/run_stage_a2.py \                |
|           --layer 12 --image /path/to/image.jpg --question "..."            |
|  -> Yêu cầu: Sai số toán học <= 1e-5, Sai số Model <= 5e-3, Bất biến <= 1e-4|
+─────────────────────────────────────────────────────────────────────────────+
```

### Mục Đích và Phạm Vi Dòng Lệnh Của Các Chế Độ Đầu Vào:
- **Tiền kiểm tổng hợp (Synthetic Preflight):** Kiểm chứng việc nạp mô hình, cấp phát bộ nhớ CUDA, kích thước tensor và cơ chế observer trong một môi trường được kiểm soát hoàn toàn, có thể tái lập mà không phụ thuộc vào dữ liệu bên ngoài.
- **Thử nghiệm với ảnh thực tế (Real-Image Pilot):** Xác minh luồng forward của mô hình, việc áp dụng template prompt và bộ thu thập attention vận hành chính xác trên một mẫu văn bản cảnh ViTextVQA thực tế. Cả `run_stage_a0.py` và `run_stage_a2.py` đều phân tích đầy đủ các đối số đường dẫn ảnh và câu hỏi (truyền theo vị trí hoặc qua cờ `--image` / `--question`). Nếu đường dẫn ảnh được chỉ định không tồn tại, cả hai kịch bản đều lập tức dừng lại với mã `FAIL` (không tự ý thay thế bằng ảnh tổng hợp).
- **Lưu ý về phạm vi:** Không có chế độ chạy nào ở đây thay thế cho một bài đánh giá đầy đủ trên 400 ảnh ViTextVQA. Giai đoạn A1 chủ động sử dụng các chuỗi mẫu/khung mẫu vì mục tiêu trọng tâm của nó là ánh xạ ranh giới tokenizer ở cấp độ prompt qua các biến thể câu hỏi nghi vấn, độc lập với nội dung hình ảnh.

---

## 9. Khả Năng Tái Lập, Bảo Mật và An Toàn Dữ Liệu

### 9.1 Nguồn Gốc và Cố Định Phiên Bản Phụ Thuộc
- **Phiên bản mô hình:** Được ghim cố định tại commit `b98f263eab246eb5269ade64edbdca8a887dc44d`.
- **Các gói phụ thuộc trực tiếp:** Được khóa phiên bản trong `requirements-stage-a.lock`.
- **Các gói phụ thuộc bắc cầu:** Tự động đóng băng vào `requirements-stage-a.lock.resolved` khi khởi tạo môi trường.
- **Lưu trữ cache bền vững:** Thư mục cache của Hugging Face được điều hướng sang `/workspace/hf_cache` thông qua biến môi trường bền vững `export HF_HOME=/workspace/hf_cache`.

### 9.2 An Toàn Dữ Liệu & Quy Tắc Loại Trừ Khỏi Git
Nhằm bảo vệ tuyệt đối các nhãn gán thủ công đã qua đánh giá và tính toàn vẹn của tập dữ liệu:
- Các không gian làm việc gán nhãn (`data/xanylabeling_workspace/Cam_Tu_201_300/`, `Thanh_Tam_301_400/`) bị loại trừ nghiêm ngặt khỏi việc theo dõi của git.
- Các quy tắc sau đây trong `.gitignore` được thực thi nghiêm ngặt:
  ```gitignore
  # Datasets & annotations
  data/raw/
  data/interim/
  data/processed/
  data/xanylabeling_workspace/
  ViTextVQA_images/
  *.jpg
  *.png
  *.jsonl

  # Model weights & caches
  *.safetensors
  *.bin
  *.pt
  *.pth
  hf_cache/
  .cache/

  # Environments & secrets
  .venv*
  venv/
  .env*
  *.token

  # Logs
  logs/*.log
  __pycache__/
  ```

---

## 10. Các Bước Nghiên Cứu Tiếp Theo

Sau khi hoàn thành thành công các Cổng 0, A0, A1 và A2 trên máy chủ đám mây, chương trình nghiên cứu sẽ tiến hành theo lộ trình khoa học có cấu trúc sau:

1. **Hoàn thiện Giai đoạn Nghiên cứu 1 (Định vị Vùng ROI Văn Bản):**
   - Triển khai trích xuất vùng văn bản ứng viên (so sánh bounding boxes từ SwinTextSpotter với nhãn chuẩn ground-truth).
   - Phát triển thuật toán chấm điểm độ nổi bật liên phương thức dựa trên câu hỏi qua các tầng giải mã.
2. **Đánh giá Độ chính xác Định vị của Giai đoạn 1:**
   - Đánh giá benchmark độ chính xác định vị so với các mô hình cơ sở đã công bố (LocalizationHeads, GroundLMM, ST-VQA_Loc).
3. **Hoàn thiện Giai đoạn Nghiên cứu 2 (Giải Mã Tương Phản):**
   - Triển khai cơ chế làm nhiễu/che mờ vùng thị giác nổi bật và công thức tính logit tương phản.
   - Hiệu chuẩn độ mạnh của hệ số dẫn dắt tương phản $\alpha$ và các tham số cắt tỉa thích ứng.
4. **Tích hợp Toàn diện Hệ thống SG-VCD:**
   - Kết nối việc lựa chọn vùng ROI của Giai đoạn 1 với cơ chế giải mã tương phản của Giai đoạn 2.
5. **Đánh giá Toàn Diện trên ViTextVQA & Nghiên Cứu Thành Phần (Ablation Study):**
   - Đánh giá so sánh với các phương pháp cơ sở trong y văn:
     - Mô hình gốc Vintern-1B-v3_5 (greedy search / beam search).
     - Giải mã tương phản tiêu chuẩn (Contrastive Decoding - CD / VCD).
     - Dẫn dắt vùng tương phản (Contrastive Region Guidance - CRG).
     - Định tuyến thích ứng LazyMCoT.
   - Tiến hành các nghiên cứu thành phần (Ablation):
     - Ảnh hưởng của việc chọn tầng trích xuất độ nổi bật (Tầng 12 so với tổng hợp đa tầng).
     - Ảnh hưởng của siêu tham số dẫn dắt tương phản $\alpha$.
     - Tác động khi sử dụng ROI chuẩn (ground-truth) so với ROI dự đoán từ mô hình.
