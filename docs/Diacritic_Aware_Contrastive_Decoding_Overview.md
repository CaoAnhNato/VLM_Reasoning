# 1. TỔNG QUAN ĐỀ TÀI VÀ ĐẶT VẤN ĐỀ NGHIÊN CỨU

## 1.1. Bối Cảnh Thực Tiễn và Thách Thức Cốt Lõi

Trong những năm gần đây, các mô hình thị giác - ngôn ngữ quy mô lớn (Vision-Language Models - VLMs) như GPT-4V, LLaVA, InternVL và Qwen-VL đã đạt được những bước tiến vượt bậc trong việc hiểu nội dung hình ảnh và trả lời câu hỏi tự nhiên. Tuy nhiên, khi triển khai vào các bài toán trả lời câu hỏi dựa trên văn bản trong ảnh (Text-based Visual Question Answering - TextVQA) thuộc các ngôn ngữ có dấu thanh điệu phức tạp như tiếng Việt, các mô hình này bộc lộ những khiếm khuyết mang tính hệ thống.

### 1.1.1. Hiện Tượng "Ảo Giác Dấu Thanh Điệu" (Diacritic Hallucination) trong VLM

Ngôn ngữ tiếng Việt sở hữu hệ thống ký tự Latinh mở rộng với 5 dấu thanh (sắc, huyền, hỏi, ngã, nặng) cùng các dấu phụ nguyên âm (ă, â, ê, ô, ơ, ư, đ). Sự thay đổi của một dấu thanh điệu nhỏ sẽ làm biến đổi hoàn toàn ngữ nghĩa của từ ngữ (ví dụ: *cơm* - *cớm*, *giá* - *gia*, *bún* - *bùn*). Trong bối cảnh ảnh đời thực (biển hiệu đường phố, thực đơn quán ăn, bao bì nhãn hàng), các ký tự này thường chịu ảnh hưởng bởi phông chữ cách điệu, biến dạng góc nghiêng, khoảng cách xa hoặc độ phân giải thấp.

Hầu hết các VLM mã nguồn mở hiện nay được tiền huấn luyện trên các tập dữ liệu đa ngữ khổng lồ nhưng áp đảo bởi tiếng Anh và tiếng Trung. Hậu quả là phần lõi mô hình ngôn ngữ lớn (LLM backbone) sở hữu một "thiên kiến ngôn ngữ tiên nghiệm" (Language Prior) cực kỳ mạnh mẽ. Khi tiếp nhận các vùng văn bản có dấu thanh điệu nhỏ hoặc hơi mờ, LLM backbone có xu hướng lấn át tín hiệu thị giác thu nhận từ bộ mã hóa hình ảnh (Visual Encoder), dẫn đến việc sinh ra các từ không dấu hoặc nhầm lẫn dấu theo thói quen ngôn ngữ tiếng Anh/tiếng Pháp không dấu. Hiện tượng này được định nghĩa là **Ảo giác dấu thanh điệu (Diacritic Hallucination)**.

### 1.1.2. Hạn Chế của Các Phương Pháp Giải Mã Hiện Tại

Để khắc phục hiện tượng ảo giác thị giác trong VLM, các nghiên cứu gần đây thường tiếp cận theo hai hướng chính:
- **Hướng tiếp cận huấn luyện bổ sung (Supervised Fine-Tuning / LoRA):** Đòi hỏi tài nguyên tính toán phần cứng lớn, thời gian huấn luyện kéo dài và đối mặt với rủi ro quên tri thức gốc (Catastrophic Forgetting) khi thích ứng với ngôn ngữ tài nguyên thấp.
- **Hướng tiếp cận can thiệp lúc giải mã (Contrastive Decoding - CD):** Các phương pháp tiêu biểu như Visual Contrastive Decoding (VCD) giải quyết ảo giác bằng cách lấy hiệu số phân phối xác suất giữa ảnh gốc và ảnh bị làm mờ (blurred image). Tuy nhiên, việc làm mờ đồng nhất toàn bộ bức ảnh sẽ phá hủy cấu trúc của mọi ký tự văn bản, làm suy giảm nghiêm trọng khả năng đọc chữ (OCR fidelity) và khiến mô hình mất đi ngữ cảnh không gian tổng thể. Hơn nữa, việc áp dụng CD liên tục trên mọi token gây tăng gấp đôi độ trễ tính toán không cần thiết.

## 1.2. Tập Dữ Liệu Thực Nghiệm ViTextVQA

Nghiên cứu lựa chọn bộ dữ liệu **ViTextVQA** làm nền tảng kiểm chứng và đánh giá thực nghiệm chính cho toàn bộ giải pháp đề xuất.

### 1.2.1. Nguồn Gốc và Quy Mô Bộ Dữ Liệu ViTextVQA

ViTextVQA là bộ dữ liệu chuẩn mực đầu tiên dành riêng cho bài toán Text-based Visual Question Answering tại Việt Nam, được công bố bởi nhóm nghiên cứu UIT-NLP tại hội nghị quốc tế ACL 2024 (*arXiv:2404.10652*). Bộ dữ liệu bao gồm:
- **Quy mô mẫu:** Hơn **16.000 hình ảnh thực tế** được thu thập trực tiếp tại nhiều đô thị của Việt Nam.
- **Số lượng câu hỏi:** Hơn **50.000 cặp câu hỏi - câu trả lời** tự nhiên bằng tiếng Việt do con người gán nhãn và kiểm định chất lượng nghiêm ngặt.
- **Định dạng dữ liệu:** Đi kèm tọa độ bounding box của các khối văn bản (text regions) và nội dung văn bản OCR tương ứng.

### 1.2.2. Điểm Nghẽn Thách Thức trên Văn Bản Ngữ Cảnh Đời Sống Việt Nam

Khác với các tập dữ liệu tài liệu scan phẳng (DocVQA), hình ảnh trong ViTextVQA phản ánh độ phức tạp cao của môi trường đời sống: ánh sáng chói, phông chữ viết tay, bảng hiệu kim loại phản chiếu, và đặc biệt là tỷ lệ các ký tự có dấu thanh điệu chiếm mật độ rất dày đặc. Đây là môi trường thử thách khắc nghiệt nhất nhằm bóc trần sự phụ thuộc thái quá vào language prior của các mô hình VLM hiện đại, tạo cơ sở thực nghiệm vững chắc để đánh giá năng lực của thuật toán giải mã mới.

---

# 2. PHƯƠNG PHÁP ĐỀ XUẤT: DIACRITIC-AWARE CONTRASTIVE DECODING (DACD)

Đề tài đề xuất phương pháp **Diacritic-Aware Contrastive Decoding (DACD)** – một giải thuật can thiệp quá trình sinh văn bản tại thời điểm suy luận (Inference-time / Training-free) nhằm triệt tiêu ảo giác dấu thanh điệu mà không cần thay đổi bất kỳ trọng số nào của mô hình nền tảng.

## 2.1. Cơ Sở Lý Thuyết và Kiến Trúc Giải Pháp

Phương pháp DACD hoạt động trực tiếp trong vòng lặp sinh tự hồi quy (autoregressive decoding loop) của mô hình VLM, can thiệp vào lớp logits thông qua cơ chế `LogitsProcessor` trong thư viện chuẩn Hugging Face Transformers.

### 2.1.1. Nguyên Lý Can Thiệp Decoding-Time

Tại mỗi bước giải mã $t$, mô hình ngôn ngữ dự đoán phân phối xác suất cho token tiếp theo $y_t$ dựa trên ngữ cảnh lịch sử $y_{<t}$, câu hỏi truy vấn $Q$ và đặc trưng hình ảnh $I$. Thay vì áp dụng giải mã tham lam (Greedy Search) chọn token có xác suất cao nhất $\arg\max P(y_t \mid I, Q, y_{<t})$, DACD thiết lập một bộ lọc động dựa trên độ bất định (uncertainty) và mức độ liên quan thị giác của các ký tự có dấu.

### 2.1.2. Định Dạng Toán Học và Hàm Phạt Tần Số Cao

Cơ chế của DACD được cấu thành bởi hai khối xử lý liên hoàn:

#### a) Bóc tách đặc trưng tần số của dấu thanh điệu

Các dấu thanh điệu tiếng Việt (sắc, huyền, hỏi, ngã, nặng) có kích thước vật lý rất nhỏ và phân bố tập trung ở dải tần số không gian cao (High-Frequency Spatial Components) nằm ở nửa trên hoặc nửa dưới của ký tự nguyên âm cơ sở. Khi tạo phân phối đối chứng (negative contrastive view), thay vì làm mờ ngẫu nhiên toàn bộ ảnh như VCD, DACD áp dụng bộ lọc thông thấp cục bộ (Local Low-Pass Filter) hoặc mặt nạ tần số cao (High-Frequency Suppression Mask) chỉ riêng trên các vùng chứa văn bản nghi vấn $I_{\text{crop}}$:
$$I_{\text{distorted}} = \mathcal{F}^{-1} \left( \mathcal{F}(I_{\text{crop}}) \odot \mathcal{M}_{\text{low}} \right)$$
Trong đó $\mathcal{F}$ biểu diễn phép biến đổi Fourier/Wavelet 2 chiều và $\mathcal{M}_{\text{low}}$ là mặt nạ triệt tiêu các thành phần tần số cao đại diện cho nét dấu thanh điệu.

#### b) Cơ chế tự động bù trừ Logits (Diacritic-Gated Re-weighting)

Phân phối logits của token tiếp theo được hiệu chỉnh theo công thức đối sánh có điều kiện:
$$\mathcal{L}_{\text{calibrated}}(y_t) = (1 + \alpha) \cdot \mathcal{L}_{\text{original}}(y_t) - \alpha \cdot \mathcal{L}_{\text{distorted}}(y_t) + \beta \cdot \mathbb{I}_{\text{diacritic}}(y_t)$$
Trong đó:
- $\mathcal{L}_{\text{original}}(y_t)$ là phân phối logits xuất phát từ hình ảnh sắc nét ban đầu.
- $\mathcal{L}_{\text{distorted}}(y_t)$ là phân phối logits xuất phát từ hình ảnh đã bị triệt tiêu chi tiết dấu thanh điệu (đại diện cho thiên kiến thuần túy của language prior khi mất tín hiệu dấu).
- $\alpha > 0$ là hệ số tương phản nhằm trừ khử ảnh hưởng của language prior.
- $\mathbb{I}_{\text{diacritic}}(y_t)$ là hàm chỉ thị kích hoạt thưởng điểm (bonus reward) nếu token ứng viên $y_t$ chứa dấu thanh điệu trùng khớp với mật độ năng lượng tần số cao đo được trên vùng ảnh cục bộ.
- Cơ chế cổng Entropy Gating: Bước tương phản trên chỉ được kích hoạt khi entropy của phân phối gốc vượt qua ngưỡng bất định $\tau$ ($H(P_{\text{original}}) > \tau$), giúp duy trì 100% tốc độ sinh ở các từ vựng thông thường.

## 2.2. Tính Mới (Novelty) và Đóng Góp Học Thuật Chuẩn Q3–Q4

Đề tài đáp ứng đầy đủ các tiêu chuẩn khắt khe của một bài báo khoa học thuộc phân khúc **Q3–Q4 Scopus / Hội nghị Quốc tế (IEEE RIVF, KSE, MAPR)** nhờ các đóng góp sau:
1. **Phát hiện và hình thức hóa vấn đề đặc thù:** Là nghiên cứu tiên phong định nghĩa và phân tích bản chất cơ chế của hiện tượng "Ảo giác dấu thanh điệu" (Diacritic Hallucination) trong các VLM hiện đại khi xử lý ngôn ngữ Latinh có dấu.
2. **Đóng góp thuật toán nguyên bản (Algorithmic Contribution):** Đề xuất giải thuật giải mã tương phản có chọn lọc theo tần số không gian (Frequency-Selective Contrastive Decoding), khắc phục triệt để nhược điểm làm suy giảm độ nét OCR của phương pháp VCD kinh điển.
3. **Hiệu quả tài nguyên vượt trội:** Thuật toán hoàn toàn là **Training-Free**, không yêu cầu bất kỳ chi phí phần cứng nào để huấn luyện lại mô hình, có thể tích hợp dưới dạng một thư viện mở rộng nhẹ (plug-and-play) cho mọi VLM mã nguồn mở hiện hành.

---

# 3. THIẾT KẾ THỰC NGHIỆM VÀ ĐÁNH GIÁ (EXPERIMENTAL SETUP)

## 3.1. Các Mô Hình So Sánh Cơ Sở (Baselines)

Nghiên cứu sẽ triển khai thực nghiệm so sánh đa diện trên các dòng mô hình VLM tiêu biểu:
- **Mô hình nền tảng tiếng Việt chuyên biệt:** **Vintern-1B-v2** (`5CD-AI/Vintern-1B-v2`), mô hình kích thước nhỏ (1 tỷ tham số) tối ưu cho tác vụ OCR tiếng Việt.
- **Mô hình đa ngữ tiên tiến thế hệ mới:** **Qwen2.5-VL-3B-Instruct** và **Qwen2.5-VL-7B-Instruct** (Alibaba), đại diện cho dòng VLM mã nguồn mở mạnh nhất hiện nay về định vị và đọc chữ.
- **Các phương pháp giải mã đối chứng:**
  - Greedy Search (chuẩn mặc định của Hugging Face).
  - Beam Search ($K = 3$).
  - Vanilla Visual Contrastive Decoding (VCD - CVPR 2024).
  - Phương pháp đề xuất: DACD (Diacritic-Aware Contrastive Decoding).

## 3.2. Tiêu Chí Đo Lường và Đánh Đổi (Metrics & Trade-offs)

Nghiên cứu sử dụng hệ thống chỉ số đo lường toàn diện nhằm phản ánh chính xác cả chất lượng văn bản lẫn hiệu năng tính toán:
- **Độ chính xác nội dung câu trả lời:**
  - Accuracy chuẩn VQA (Exact Match / Top-1 Accuracy).
  - Average Normalized Levenshtein Similarity (ANLS) – thước đo chuẩn mực quốc tế cho bài toán TextVQA, đo lường khoảng cách chỉnh sửa giữa chuỗi dự đoán và nhãn thực tế.
  - Diacritic Precision & Recall: Thước đo chuyên biệt tự thiết kế nhằm đo lường tỷ lệ đoán đúng dấu thanh điệu trên các từ khóa quan trọng.
- **Đo lường chi phí tính toán (Computational Trade-offs):**
  - Thời gian suy luận trung bình trên mỗi mẫu (Latency per sample - ms).
  - Tốc độ sinh token (Tokens per second).
  - Bộ nhớ GPU tiêu thụ đỉnh (Peak VRAM consumption - GB).

# 4. QUẢN TRỊ DỰ ÁN VÀ KẾ HOẠCH TRIỂN KHAI 6 TUẦN

Nhằm đảm bảo tiến độ thực nghiệm diễn ra đúng hạn, tránh nguy cơ chậm trễ khi giảng viên hoặc hội đồng kiểm tra tiến độ đột xuất (Timeline Audit), nhóm nghiên cứu gồm 4 thành viên được phân định trách nhiệm rõ ràng theo mô hình 2 thành viên chủ lực và 2 thành viên hỗ trợ chuyên trách.

## 4.1. Ma Trận Phân Vai Trách Nhiệm (RACI Matrix)

Việc phân bổ vai trò dựa trên thế mạnh chuyên môn của từng thành viên, đồng thời khoán đứt quyền sở hữu độc lập (Ownership) để mỗi cá nhân đều có sản phẩm đầu ra hữu hình:

| Thành viên | Vai trò dự án | Trách nhiệm cốt lõi (Core Ownership) | Sản phẩm nghiệm thu cá nhân |
| :--- | :--- | :--- | :--- |
| **Thanh Tâm**<br>*(Chủ lực - Mô hình)* | **Kỹ sư Mô hình & Giải thuật** | - Thiết lập inference pipeline cho VLM (Vintern-1B, Qwen2.5-VL).<br>- Lập trình thuật toán DACD kế thừa `LogitsProcessor`.<br>- Tinh chỉnh siêu tham số và tối ưu vòng lặp giải mã trên GPU. | - File mã nguồn thuật toán `dacd_processor.py`.<br>- Script can thiệp logit giải mã chạy mượt trên PyTorch. |
| **Cẩm Tú**<br>*(Chủ lực - Phân tích)* | **Chuyên gia Phân tích & Đánh giá** | - Xây dựng hệ thống phân loại lỗi dấu thanh điệu (Error Taxonomy).<br>- Phân tích định tính (Qualitative) & định lượng (Quantitative).<br>- Phụ trách viết phần Methodology, Discussion & Error Analysis. | - Bảng phân loại lỗi (Excel/Markdown) có dẫn chứng trực quan.<br>- Bản thảo các chương học thuật chính của bài báo. |
| **Phú Triệu**<br>*(Hỗ trợ - Dữ liệu & Assets)* | **Kỹ sư Dữ liệu & Trích xuất Thị giác** | - Tải, chia tách (split) và tiền xử lý bộ dữ liệu ViTextVQA.<br>- Viết công cụ tự động cắt (crop) các vùng bounding box nghi vấn.<br>- Thu thập và lọc các mẫu thử nghiệm trực quan (Case study images). | - Bộ dữ liệu train/val/test sạch kèm nhãn JSON chuẩn.<br>- Kho ảnh trích xuất lỗi dấu kèm ảnh nhiệt (Attention map). |
| **Cao Anh**<br>*(Hỗ trợ - Hạ tầng & Benchmark)* | **Kỹ sư Thực nghiệm & Đo lường** | - Cài đặt môi trường `venv`/CUDA, cố định seed thực nghiệm.<br>- Viết script tự động chạy benchmark hàng loạt và ghi log (WandB/CSV).<br>- Đo đạc chi phí phần cứng (Latency, Throughput, VRAM peak). | - Pipeline tự động chạy benchmark chỉ bằng 1 câu lệnh CLI.<br>- Bảng số liệu đo đạc tài nguyên (FLOPs/FPS/VRAM). |

## 4.2. Lộ Trình 6 Tuần Chi Tiết và Sản Phẩm Nghiệm Thu Từng Giai Đoạn

Lộ trình được thiết kế theo nguyên tắc "Đóng băng mã nguồn ở Tuần 4.5" và dành trọn vẹn Tuần 6 làm "Vùng đệm an toàn dự phòng" (Contingency Buffer Week), giúp nhóm luôn làm chủ tiến độ trước mọi đợt kiểm tra:

### 4.2.1. Tuần 1: Thiết Lập Môi Trường, Đo Baseline & Phân Tích Lỗi Ban Đầu
- **Nhiệm vụ cụ thể:**
  - Cao Anh: Thiết lập môi trường ảo chuẩn (`requirements.txt`, Python `venv`, kiểm tra tương thích CUDA/GPU), tải mã nguồn mô hình Vintern-1B và Qwen2.5-VL-3B về máy trạm.
  - Phú Triệu: Tải bộ dữ liệu ViTextVQA từ Hugging Face, lọc và chuẩn hóa tập 500 mẫu test nhanh đầu tiên (Mini-Test Set) để phục vụ debug nhanh cho toàn nhóm.
  - Thanh Tâm: Viết script chạy suy luận Zero-shot chuẩn của Hugging Face trên tập Mini-Test.
  - Cẩm Tú: Tiếp nhận kết quả suy luận từ Thanh Tâm, rà soát chi tiết 100 mẫu trả lời sai đầu tiên để xây dựng bảng phân loại lỗi (Error Taxonomy): phân tách rõ tỷ lệ sai do mất dấu thanh điệu, sai do nhầm ký tự Latinh cơ sở, và sai do không hiểu câu hỏi.
- **Sản phẩm nghiệm thu hữu hình:**
  - Tài liệu hướng dẫn cài đặt môi trường (`environment_setup.md`) và mã nguồn chạy thử nghiệm thành công.
  - Bảng số liệu Zero-shot Baseline (Accuracy & ANLS) trên 500 mẫu đầu tiên.
  - Báo cáo phân loại sơ bộ: "Thống kê 5 dạng ảo giác dấu thanh điệu phổ biến nhất trên ViTextVQA".

### 4.2.2. Tuần 2: Xây Dựng Module Thuật Toán DACD & Công Cụ Trích Xuất Đặc Trưng
- **Nhiệm vụ cụ thể:**
  - Phú Triệu: Phát triển hàm trích xuất vùng ảnh chữ (crop text region) từ bounding box có sẵn trong ViTextVQA; cài đặt bộ lọc thông thấp (Gaussian/Fourier) để tạo ảnh đối chứng mất chi tiết nét dấu.
  - Thanh Tâm: Xây dựng class `DiacriticContrastiveLogitsProcessor` kế thừa từ `transformers.LogitsProcessor`. Lập trình công thức đối sánh logits: $\mathcal{L}_{\text{calibrated}} = (1 + \alpha) \cdot \mathcal{L}_{\text{original}} - \alpha \cdot \mathcal{L}_{\text{distorted}}$.
  - Cẩm Tú: Thiết kế bộ tiêu chuẩn đo lường chuyên biệt "Diacritic Precision & Recall" nhằm định lượng số lượng từ ngữ sai dấu được hiệu chỉnh thành công.
  - Cao Anh: Thiết lập hệ thống ghi log tự động (Weights & Biases hoặc xuất file CSV chuẩn) để chuẩn bị cho giai đoạn benchmark quy mô lớn.
- **Sản phẩm nghiệm thu hữu hình:**
  - Tệp Jupyter Notebook demo chạy thành công trên 20 mẫu thử nghiệm: So sánh trực quan câu trả lời trước khi sửa (sai dấu) và sau khi áp dụng DACD (đúng dấu).
  - Module mã nguồn `dacd_processor.py` đã kiểm thử đơn vị (Unit Test) và đưa lên Git repository.

### 4.2.3. Tuần 3: Thực Nghiệm Toàn Diện & Tinh Chỉnh Siêu Tham Số (Mid-term Checkpoint)
- **Nhiệm vụ cụ thể:**
  - Cao Anh: Vận hành script tự động chạy toàn bộ tập Test chính thức của ViTextVQA trên các cấu hình: Greedy Search, Beam Search, Vanilla VCD và DACD đề xuất.
  - Thanh Tâm: Thực hiện quét siêu tham số (Hyperparameter Sweep): khảo sát hệ số tương phản $\alpha \in \{0.2, 0.5, 0.8, 1.0\}$ và ngưỡng entropy gating $\tau$.
  - Phú Triệu: Giám sát toàn bộ tiến trình chạy GPU, kịp thời lọc các mẫu ảnh bị lỗi kích thước hoặc xử lý ngoại lệ Out-of-Memory.
  - Cẩm Tú: Thu thập tệp log số liệu từ Cao Anh, xử lý dữ liệu và vẽ biểu đồ thể hiện sự phụ thuộc của độ chính xác ANLS theo các giá trị $\alpha$ và $\tau$.
- **Sản phẩm nghiệm thu hữu hình:**
  - Bảng số liệu thực nghiệm đầy đủ (Full Benchmark Table) so sánh DACD với các baseline chuẩn.
  - Biểu đồ đường (Line Chart) thể hiện tính hội tụ và tương quan của siêu tham số.

### 4.2.4. Tuần 4: Nghiên Cứu Đánh Đổi (Ablation Studies) & Đóng Băng Code (Code Freeze)
- **Nhiệm vụ cụ thể:**
  - Thanh Tâm & Cao Anh: Triển khai các bài thực nghiệm bóc tách thành phần (Ablation Experiments):
    - Cấu hình có Entropy Gating so với không có Entropy Gating.
    - Cấu hình triệt tiêu tần số cục bộ (DACD) so với làm mờ toàn ảnh (Vanilla VCD).
    - Đo đạc chính xác các thông số phần cứng: Tốc độ sinh token (tokens/s), độ trễ suy luận (latency/sample), bộ nhớ đỉnh (Peak VRAM).
  - Cẩm Tú: Tổng hợp bảng kết quả Ablation Study, phân tích số liệu chứng minh hiệu quả giảm thiểu độ trễ của cơ chế Entropy Gating.
  - Phú Triệu: Trích xuất 10 cặp ảnh Case Study tiêu biểu nhất (5 ca sửa đúng thành công và 5 ca ngoại lệ còn hạn chế kèm phân tích lý do).
  - Toàn bộ nhóm (Cuối Tuần 4): **ĐÓNG BĂNG MÃ NGUỒN (CODE FREEZE)**. Tuyệt đối không bổ sung thêm tính năng thuật toán mới.
- **Sản phẩm nghiệm thu hữu hình:**
  - Bảng kết quả Ablation Study hoàn chỉnh về độ chính xác và chi phí tính toán.
  - Bộ tài liệu minh họa 10 Case Study trực quan có chú thích chi tiết.
  - Mã nguồn dự án được gắn thẻ phát hành chính thức (Git Release v1.0).

### 4.2.5. Tuần 5: Phân Tích Định Tính Chuyên Sâu & Hoàn Thiện Bản Thảo Báo Cáo
- **Nhiệm vụ cụ thể:**
  - Cẩm Tú (Chủ trì): Hoàn thiện các phần trọng tâm của bài báo: Tóm tắt (Abstract), Mở đầu (Introduction), Phương pháp (Methodology), Thảo luận kết quả (Discussion) và Phân tích lỗi (Error Analysis).
  - Thanh Tâm: Hoàn thiện chương cài đặt thực nghiệm (Experimental Setup) và vẽ sơ đồ khối kiến trúc giải thuật DACD chuẩn phong cách hội nghị.
  - Cao Anh: Sử dụng công cụ `/formatting-docx` để tự động hóa căn chỉnh bố cục toàn bộ tài liệu theo quy chuẩn Times New Roman 13pt, lề 3-2-3-3cm.
  - Phú Triệu: Thiết kế bộ slide thuyết trình báo cáo đề tài (Presentation Deck) từ 15 đến 20 slide chuẩn mực học thuật.
- **Sản phẩm nghiệm thu hữu hình:**
  - Bản thảo hoàn chỉnh định dạng Word (`.docx`) của bài báo khoa học / đề tài tốt nghiệp (Draft v1.0).
  - Bộ slide báo cáo nghiệm thu sơ bộ.

### 4.2.6. Tuần 6: Vùng Đệm An Toàn (Buffer Week), Review Chéo & Diễn Tập Báo Cáo
- **Mục tiêu chiến lược:** Vùng đệm thời gian dự phòng 100% để khắc phục mọi sự cố phát sinh từ các tuần trước mà không làm ảnh hưởng đến thời hạn nộp cuối cùng.
- **Nhiệm vụ cụ thể:**
  - Trong trường hợp tiến độ đúng hạn:
    - Cẩm Tú và Thanh Tâm: Thực hiện đọc soát chéo (Peer-review), kiểm tra tính nhất quán giữa các công thức toán và số liệu trong bảng biểu.
    - Cao Anh: Chạy công cụ kiểm tra tự động `docx_styler.py` để xác nhận 100% tiêu chí thể thức đạt chuẩn.
    - Phú Triệu: Hoàn thiện tài liệu hướng dẫn tái lập thực nghiệm (`README.md`) trên GitHub repository.
    - Cả 4 thành viên: Tổ chức buổi diễn tập thuyết trình thử (Mock Defense), chuẩn bị sẵn câu trả lời cho các câu hỏi chất vấn từ hội đồng.
- **Sản phẩm nghiệm thu cuối cùng:**
  - Tệp báo cáo hoàn thiện chính thức (`.docx` và `.pdf`) đã thông qua kiểm duyệt nội bộ.
  - Bộ slide thuyết minh chính thức (`.pptx`).
  - Kho mã nguồn hoàn chỉnh sẵn sàng công bố mở.

## 4.3. Cơ Chế Kiểm Soát Tiến Độ và Quản Trị Rủi Ro Dự Án

Để đảm bảo dự án vận hành trơn tru và không bị động trước các đợt kiểm tra tiến độ, nhóm áp dụng 3 nguyên tắc quản trị cốt lõi:

#### a) Cơ chế nghiệm thu sản phẩm hữu hình (Tangible Deliverables First)
Tại mỗi mốc thời gian của từng tuần, nhóm bắt buộc phải có sản phẩm vật lý cụ thể (file notebook, bảng số liệu CSV, tài liệu markdown cập nhật, hoặc commit code trên GitHub). Tuyệt đối không báo cáo tiến độ bằng lời nói chung chung khi làm việc với giảng viên hướng dẫn.

#### b) Quy tắc đóng băng mã nguồn (Code Freeze) và vùng đệm thời gian
Việc áp dụng Code Freeze ở Tuần 4.5 và dành trọn vẹn Tuần 6 làm Buffer Week giúp giảm thiểu tối đa rủi ro phát sinh lỗi kỹ thuật sát ngày nộp bài, đảm bảo nhóm có đủ thời gian để trau chuốt phần lập luận học thuật và hình thức báo cáo.

#### c) Chiến lược xử lý sự cố và hỗ trợ chéo (Cross-functional Support)
Nếu thành viên chủ lực kỹ thuật (Thanh Tâm) gặp lỗi phức tạp kéo dài quá 48 giờ, Cao Anh sẽ tham gia hỗ trợ tra cứu giải pháp trên các diễn đàn khoa học (Stack Overflow, GitHub Issues), trong khi Cẩm Tú và Phú Triệu tiếp tục đẩy mạnh công tác chuẩn bị dữ liệu và khung dàn ý bài báo, ngăn ngừa tình trạng đình trệ toàn bộ chuỗi công việc.

