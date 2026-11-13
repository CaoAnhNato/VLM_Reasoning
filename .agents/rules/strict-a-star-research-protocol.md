---
trigger: manual
description: Hiến pháp Nghiên cứu & Quy tắc Kiểm toán Đối kháng (Strict A/A* Research Rules) với mục tiêu Token F1 > 70.0% và Substring Match > 70.0% trên ViTextVQA 400 ảnh.
---

# HIẾN PHÁP NGHIÊN CỨU & QUY TẮC KIỂM TOÁN ĐỐI KHÁNG (STRICT A/A* RESEARCH RULES)

## RULE 1: CẤM NGỤY BIỆN "TRẦN NĂNG LỰC TOÁN HỌC" ĐỂ DỪNG SỚM (NO SURRENDER / NO IMPOSSIBILITY EXCUSES)
1. Cấm tuyệt đối mọi Agent viết báo cáo kết luận rằng mục tiêu `Token F1 > 70.0%` và `Substring Match > 70.0%` trên `5CD-AI/Vintern-1B-v3_5` (tập 400 ảnh / 1.222 câu hỏi ViTextVQA) là "bất khả thi về mặt toán học" (như cách CERV đã ngụy biện bằng Bất đẳng thức Xử lý Dữ liệu và đổ lỗi cho nhiễu nhãn).
2. Mọi lập luận đổ lỗi cho độ phân giải của `InternViT-300M` khi mới chỉ thử nghiệm cắt 1 khung hình đơn (single crop) hoặc nhìn toàn cảnh (global view) đều bị `Critic` và `Auditor` bác bỏ ngay lập tức. Nếu một khung nhìn đơn bị mất chi tiết hoặc đứt gãy ký tự, thuật toán phải giải quyết bằng cơ chế lấy mẫu thị giác đa tiêu cự, tổng hợp bằng chứng không gian-thời gian, hoặc căn chỉnh trường chú ý liên tầng.
3. Không được kết thúc dự án (Phase 2) khi chưa chạm ngưỡng `Token F1 > 70.0%` VÀ `Substring Match > 70.0%` dưới giao thức chấm điểm nghiêm ngặt `Strict A*`.

## RULE 2: TIÊU CHUẨN ĐÓNG GÓP KHOA HỌC TẦM CỠ A/A* (ANTI-FRANKENSTEIN & UNIFIED PRINCIPLE)
1. **Quyền Phá Bỏ Prototype Cũ:** Agent hoàn toàn không bị ràng buộc phải giữ lại kiến trúc CERV. Nếu phân tích cho thấy CERV đã chạm ngưỡng bão hòa hoặc cồng kềnh, hãy mạnh dạn loại bỏ CERV để thiết kế một thuật toán hoàn toàn mới.
2. **Cấm Chắp Vá Cơ Học (No Frankenstein Pipelines):** Nghiêm cấm việc lấy từng mảnh nhỏ từ các bài báo khác nhau (ví dụ: lấy 1 cổng routing + 1 phép cắt bounding box + 1 công thức trừ logit VCD + 1 bộ lọc chuỗi hậu xử lý) rồi ghép nối tuần tự. Một công trình tầm cỡ CVPR / ICCV / NeurIPS / ICML / ICLR / ACL phải được dẫn dắt bởi **một nguyên lý toán học cốt lõi, xuyên suốt và thanh lịch (Unified First-Principles Formulation)**.
3. **Định Hướng Nguyên Lý Gợi Mở (Không Giới Hạn):**
   - *Lý thuyết Vận chuyển Tối ưu (Optimal Transport / Wasserstein Geodesics)* giữa phân phối token truy vấn và trường đặc trưng thị giác đa tỷ lệ.
   - *Nguyên lý Năng lượng Tự do Biến phân (Variational Free Energy / Active Inference)*: Xem quá trình đọc văn bản cảnh của VLM như một bài toán suy luận chủ động giảm thiểu bất định nhận thức qua các phép chiếu không gian con.
   - *Lý thuyết Phổ Đồ thị & Trường Chú ý (Spectral Graph Theory / Cross-Modal Diffusion)*: Khử nhiễu cấu trúc chú ý bằng phân rã phổ hoặc khuếch tán nhiệt trên đồ thị patch-token để tách biệt thực thể ký tự khỏi nhiễu nền và quán tính ngôn ngữ (preamble inertia).
   - *Nút thắt Thông tin Nhân quả (Causal Information Bottleneck)*: Can thiệp trực tiếp vào không gian trạng thái ẩn (hidden states / KV representations) hoặc điều phối luồng thông tin đa chế độ để mô hình tự sinh trực tiếp thực thể chính xác mà không cần cắt chuỗi bằng regex.

## RULE 3: TÍNH THÍCH NGHI TỰ ĐỘNG & NGÂN SÁCH SIÊU THAM SỐ TỐI THIỂU (SELF-CALIBRATING ADAPTIVITY)
1. **Cấm Rừng Tham Số (Hyperparameter Forest Ban):** Thuật toán đề xuất được phép có **tối đa 2 siêu tham số vô hướng toàn cục** (hoặc hoàn toàn **Parameter-Free / Self-Calibrating**). Mọi ngưỡng cắt (thresholds), trọng số pha trộn, hay kích thước vùng nhìn phải được suy ra tự động dưới dạng hàm đóng (closed-form) từ thống kê nội tại của ma trận chú ý / phân phối logit (ví dụ: dựa trên phổ giá trị kỳ dị SVD, ngưỡng Marchenko-Pastur trong lý thuyết ma trận ngẫu nhiên, hoặc độ cong thông tin Fisher).
2. **Tính Thích Nghi Đa Mô Hình & Đa Tác Vụ (Universal VLM Adaptivity):** Thuật toán chỉ được sử dụng các thành phần phổ quát có ở mọi kiến trúc VLM (Visual Patch Embeddings, Cross/Self-Attention Matrices, Hidden States, Logits, KV-Cache). Không được thiết kế cơ chế chỉ chạy được trên cấu trúc riêng của `Vintern` mà phải đóng gói dưới dạng một lớp bọc tổng quát (Universal Training-Free Wrapper) tương thích với LLaVA, Qwen-VL, InternVL, PaliGemma.

## RULE 4: CẤM TUYỆT ĐỐI GIAN LẬN, REGEX & MẸO XỬ LÝ HẬU KỲ (ZERO HEURISTICS & ANTI-CHEATING)
1. **Cấm Can Thiệp Hậu Xử Lý Chuỗi (No String-Level Post-Processing Tricks):**
   - Cấm sử dụng danh sách từ dừng, từ nối tiếng Việt hay bất kỳ ngôn ngữ nào (như tập `{"của", "ở", "trong", "này", "là"}` trong CERV).
   - Cấm sử dụng biểu thức chính quy (Regex) để xóa tiền tố dẫn đề, chuẩn hóa số điện thoại, ngày tháng, địa chỉ, hay sửa lỗi chính tả sau khi mô hình đã sinh chuỗi. Nếu muốn mô hình không sinh câu dẫn đề dư thừa hoặc không bị đứt gãy token số, thuật toán phải giải quyết từ **bên trong cơ chế giải mã hoặc không gian biểu diễn (Decoding / Latent Space)** dựa trên nguyên lý toán học chung.
2. **Bảo Vệ Nguyên Trạng Giao Thức Chấm Điểm:**
   - Tuyệt đối không được chỉnh sửa, ghi đè hoặc monkey-patch `scripts/benchmark/metrics.py` và `tests/test_metrics_relaxation.py`.
   - Trước và sau mỗi cột mốc (Milestone), `Auditor` bắt buộc phải chạy kiểm tra chữ ký `git diff` trên thư mục `scripts/benchmark/` và chạy `unittest tests/test_metrics_relaxation.py`.
3. **Cô Lập Dữ Liệu Đánh Giá (Zero Data Leakage):**
   - Thuật toán suy luận tuyệt đối không được đọc trường `ground_truth`, `answers`, hoặc `bboxes` của tập test trong quá trình chạy.
   - Không được lưu cache kết quả hay chia sẻ thông tin thống kê xuyên mẫu (cross-sample leakage) trong lúc đánh giá.
   - Không được gọi bất kỳ thư viện OCR ngoại vi nào (như PaddleOCR, EasyOCR, Tesseract) hay mô hình ngôn ngữ/thị giác phụ trợ bên ngoài `5CD-AI/Vintern-1B-v3_5`.

## RULE 5: BẮT BUỘC KHAI PHÁ TRI THỨC MỞ LIÊN TỤC TRÊN INTERNET/ARXIV (MANDATORY LITERATURE LOOP)
1. Các Agent `Explorers` không được phép chỉ dựa vào kiến thức huấn luyện sẵn. Trong mỗi vòng lặp nghiên cứu, `Explorers` bắt buộc phải sử dụng công cụ tìm kiếm Internet / ArXiv để khảo sát ít nhất **10–15 bài báo khoa học mới nhất (2024–2026)** từ CVPR, ICCV, ECCV, NeurIPS, ICML, ICLR, ACL về các chủ đề:
   - *Training-free visual grounding & high-resolution perception in MLLMs*
   - *Attention sink elimination, spectral attention filtering & cross-modal information flow*
   - *Information-theoretic decoding, representation steering & activation patching in VLMs*
   - *Tokenization boundary alignment & autoregressive trajectory control without heuristics*
2. Mọi ý tưởng lấy cảm hứng từ tài liệu tham khảo phải được trừu tượng hóa và nâng cấp thành một cơ chế toán học mới, chứng minh được sự vượt trội về mặt lý thuyết lẫn thực nghiệm so với các bài báo gốc.
