# PHƯƠNG PHÁP ĐỀ XUẤT: CERV (COUNTERFACTUAL EVIDENCE ROUTING & VERIFICATION)

## 1. Tổng Quan Về Phương Pháp Hoạt Động Của CERV

Khi trả lời câu hỏi dựa trên hình ảnh (Visual Question Answering - VQA), các mô hình Thị giác - Ngôn ngữ lớn (MLLMs) thường gặp ba rào cản cốt lõi trên các chi tiết nhỏ như biển hiệu, địa chỉ hay con số: **(1) Suy giảm độ phân giải** khi ảnh gốc bị thu nhỏ lúc đưa vào mô hình, **(2) Ảo giác ngôn ngữ** khi mô hình tự đoán mò câu trả lời theo thói quen từ vựng thay vì đọc từ ảnh, và **(3) Hiện tượng dẫn đề tự tin giả** khi mô hình mở đầu câu trả lời bằng việc lặp lại câu hỏi (ví dụ: *"Số điện thoại trên bảng hiệu là:"*) với xác suất rất cao trước khi thực sự đọc đến con số nhỏ trên ảnh. Ngược lại, nếu lúc nào cũng cắt nhỏ ảnh và ép giải mã tương phản ngay từ từ đầu tiên trên mọi câu hỏi, hệ thống vừa làm mất ngữ cảnh toàn cảnh của các câu hỏi dễ (chiếm ~65%–70% dữ liệu), vừa làm biến dạng cấu trúc ngữ pháp của các câu dẫn đề.

Để giải quyết triệt để bài toán trên mà không cần huấn luyện lại, không dùng ngưỡng cắt cứng thủ công và áp dụng tổng quát cho mọi kiến trúc MLLM, phương pháp **CERV (Counterfactual Evidence Routing & Verification)** thiết lập một chu trình suy luận nhân quả gồm **3 giai đoạn gắn kết chặt chẽ quanh Điểm rẽ Thực thể**:

![Sơ đồ tổng quan phương pháp hoạt động của thuật toán CERV](C:/Users/Admin/.gemini/antigravity/brain/befd0801-8f78-44cc-a3f9-2bcbaef5fa3e/cerv_pipeline.png)
*Hình 1: Kiến trúc hoạt động tổng thể 3 giai đoạn của CERV ([Bản vẽ Draw.io](file:///c:/Users/Admin/HUIT%20-%20H%E1%BB%8Dc%20T%E1%BA%ADp/N%C4%83m%204/Computer_Vision/docs/figures/cerv_pipeline.drawio) | [Ảnh PNG](file:///c:/Users/Admin/HUIT%20-%20H%E1%BB%8Dc%20T%E1%BA%ADp/N%C4%83m%204/Computer_Vision/docs/figures/cerv_pipeline.png)).*

1. **Giai đoạn 1 — Cổng Định Tuyến Hậu Nghiệm Bayesian Tại Điểm Rẽ Thực Thể:**
   Thay vì chỉ nhìn vào từ đầu tiên và dùng một ngưỡng cố định, CERV theo dõi quỹ đạo sinh từ của mô hình để phân biệt giữa các từ **dẫn đề lặp lại câu hỏi** (chú ý vào văn bản câu hỏi) và **Điểm rẽ Thực thể** (thời điểm mô hình chuyển sự chú ý sang ảnh để đọc thông tin trả lời). Tại mỗi bước, CERV ước lượng *Giá trị Kỳ vọng của Thông tin Mẫu Thị giác* theo lý thuyết quyết định Bayesian:
   - **Luồng Nhanh (Fast-Path):** Nếu ngay cả tại điểm rẽ thực thể, mô hình đã nhìn rõ chi tiết trên ảnh (rủi ro dự đoán thấp và vùng nhìn hội tụ), lợi ích của việc cắt phóng đại ảnh nhỏ hơn rủi ro nhiễu nền. CERV xuất kết quả ngay lập tức với **0% chi phí tính toán phụ**.
   - **Luồng Chậm (Slow-Path):** Khi đến điểm rẽ thực thể (sau khi đã đi qua câu dẫn đề), nếu mô hình vừa phụ thuộc mạnh vào ảnh vừa rơi vào trạng thái phân vân giữa nhiều phương án do chữ quá nhỏ, tỷ số hậu nghiệm Bayesian tự động vượt mức cân bằng và kích hoạt Giai đoạn 2 và 3 ngay tại vị trí thực thể đó.

2. **Giai đoạn 2 — Định Vị Tương Phản Có Điều Kiện Tiền Tố Tại Điểm Rẽ:**
   Khác với các phương pháp định vị tĩnh ở từ đầu tiên (nơi sự chú ý còn bị phân tán bởi toàn bộ câu hỏi), CERV trích xuất bản đồ chú ý **ngay tại điểm rẽ thực thể khi đã có sẵn cụm từ dẫn đề**. Cụm từ dẫn đề đóng vai trò mỏ neo ngữ nghĩa giúp mô hình hội tụ thẳng vào thực thể cần tìm. Tiếp đó, CERV lấy bản đồ chú ý **khi có câu hỏi** trừ đi bản đồ chú ý **khi thay bằng truy vấn rỗng** tại tầng hội tụ mạnh nhất để xóa sạch nhiễu nền (cây cối, xe cộ, góc ảnh), rồi tạo ra hai góc nhìn đối lập: **Ảnh phóng đại bằng chứng** (làm rõ nét chữ nhỏ) và **Ảnh che bằng chứng** (che đi đúng vùng thực thể vừa tìm được).

3. **Giai đoạn 3 — Giải Mã Kiểm Chứng Phản Sự Thật Neo Tại Điểm Rẽ:**
   CERV **giữ nguyên cụm từ dẫn đề** đã sinh ở Giai đoạn 1 và chỉ thực hiện dự đoán song song trên hai góc nhìn bắt đầu từ **vị trí thực thể trở đi**. Những từ vẫn có xác suất cao trên **Ảnh che bằng chứng** chính là ảo giác ngôn ngữ hoặc thông tin đọc nhầm từ vùng lân cận. Bằng cách lấy điểm số dự đoán từ **Ảnh phóng đại bằng chứng** trừ đi điểm số từ **Ảnh che bằng chứng** với **trọng số khuếch đại tự thích nghi theo độ phân kỳ thông tin** (tự động đẩy lên cao ở các từ thực thể bị che và tự động giảm về $0$ ở các từ ngữ pháp đóng câu khi hai luồng đồng thuận), CERV bảo toàn sự trôi chảy tự nhiên của toàn bộ câu trả lời đồng thời đảm bảo thực thể được chứng thực 100% bởi điểm ảnh thực tế.

---

## 2. Chi Tiết Về Cách Hoạt Động Của Thuật Toán

Ba giai đoạn của CERV vận hành như một hệ thống thống nhất xoay quanh **Điểm rẽ Thực thể $t^*$** và **tiền tố dẫn đề $y_{<t^*}$**: Giai đoạn 1 xác định thời điểm $t^*$ bằng quy tắc quyết định Bayesian không tham số ngưỡng; Giai đoạn 2 sử dụng chính tiền tố $y_{<t^*}$ tại bước $t^*$ để định vị vùng bằng chứng sạch và tạo cặp ảnh đối lập $(I^+, I^-)$; và Giai đoạn 3 neo cố định $y_{<t^*}$ để kiểm chứng phản sự thật cho các từ thực thể $t \ge t^*$.

### 2.1. Giai đoạn 1: Cổng Định Tuyến Hậu Nghiệm Bayesian Tại Điểm Rẽ Thực Thể (Bayesian Semantic-Fork Gate)

Cho ảnh gốc $I$ (được mã hóa thành tập các vùng ảnh $\mathcal{I}$, $|\mathcal{I}| = P$) và câu hỏi $Q$ (gồm tập các từ truy vấn $\mathcal{Q}$). Khi mô hình sinh chuỗi từ trả lời $y_1, y_2, \dots, y_t$ trên ảnh toàn cục $I$, mỗi bước sinh từ $t$ chịu sự chi phối của hai giả thuyết nguồn thông tin đối lập trong không gian Bayesian:
- **Giả thuyết Sao chép Cú pháp ($\mathcal{H}_{\text{echo}}$):** Từ $y_t$ là từ dẫn đề lặp lại cấu trúc câu hỏi $Q$ (ví dụ: *"Số"*, *"điện"*, *"thoại"*, *"trên"*, *"bảng"*, *"là:"*) hoặc từ nối ngữ pháp.
- **Giả thuyết Thực thể Thị giác ($\mathcal{H}_{\text{vis}}$):** Từ $y_t$ là thực thể thông tin đòi hỏi phải trích xuất trực tiếp từ điểm ảnh của $I$ (ví dụ: đầu số *"028"*).

![Minh họa trực quan Giai đoạn 1: Cổng Định Tuyến Hậu Nghiệm Bayesian Tại Điểm Rẽ Thực Thể](C:/Users/Admin/.gemini/antigravity/brain/befd0801-8f78-44cc-a3f9-2bcbaef5fa3e/cerv_stage1_uncertainty_gate.png)
*Hình 2: Minh họa cơ chế theo dõi quỹ đạo sinh từ để tách câu dẫn đề ($t < t^*$) và tự động định tuyến tại Điểm rẽ Thực thể $t^*$ bằng tỷ số hậu nghiệm Bayesian $\Lambda_t$ ([Bản vẽ Draw.io](file:///c:/Users/Admin/HUIT%20-%20H%E1%BB%8Dc%20T%E1%BA%ADp/N%C4%83m%204/Computer_Vision/docs/figures/cerv_stage1_uncertainty_gate.drawio) | [Ảnh PNG](file:///c:/Users/Admin/HUIT%20-%20H%E1%BB%8Dc%20T%E1%BA%ADp/N%C4%83m%204/Computer_Vision/docs/figures/cerv_stage1_uncertainty_gate.png)).*

Gọi $A_{t, i}$ là trọng số chú ý trung bình tại bước $t$ từ từ đang sinh đến thành phần thứ $i$. Xác suất hậu nghiệm để bước $t$ thuộc chế độ đọc thực thể thị giác $\mathcal{H}_{\text{vis}}$ được xác định trực tiếp từ tỷ trọng chú ý giữa vùng ảnh $\mathcal{I}$ và vùng văn bản câu hỏi $\mathcal{Q}$:

$$\pi_t = P\left(\mathcal{H}_{\text{vis}} \mid I, Q, y_{<t}\right) = \frac{\sum_{i \in \mathcal{I}} A_{t,i}}{\sum_{i \in \mathcal{I}} A_{t,i} + \sum_{j \in \mathcal{Q}} A_{t,j}}$$

Tại bước $t$, với phân phối xác suất dự đoán từ vựng $p_t(y \mid I, Q, y_{<t})$, **Rủi ro Bayes hiện tại** khi quyết định chọn ngay từ có xác suất cao nhất là $R_{\text{token}}(t) = 1 - \max_{y \in V} p_t(y \mid I, Q, y_{<t})$. Đồng thời, mức độ mất tập trung không gian trên ảnh được đo bằng entropy chuẩn hóa của phân phối chú ý thị giác $\tilde{H}(A_t^{\text{img}}) = -\frac{1}{\log P} \sum_{i \in \mathcal{I}} \hat{A}_{t,i} \log \hat{A}_{t,i} \in [0, 1]$ (với $\hat{A}_{t,i} = A_{t,i} / \sum_{k \in \mathcal{I}} A_{t,k}$).

Theo lý thuyết quyết định Bayesian, **Giá trị Kỳ vọng của Thông tin Mẫu Thị giác (Expected Value of Visual Sample Information — $\mathrm{EVSI}_t$)** khi thực hiện phóng đại và kiểm chứng ảnh tại bước $t$ là tích của xác suất hậu nghiệm cần ảnh, rủi ro dự đoán từ vựng và độ phân tán không gian:

$$\mathrm{EVSI}_t = \underbrace{\pi_t}_{\text{Xác suất hậu nghiệm cần ảnh}} \times \underbrace{\left(1 - \max_{y \in V} p_t(y)\right)}_{\text{Rủi ro Bayes mức từ vựng}} \times \underbrace{\tilde{H}\left(A_t^{\text{img}}\right)}_{\text{Độ phân tán chú ý trên ảnh}}$$

Ngược lại, rủi ro nền (baseline risk) thuộc về thành phần ngôn ngữ không thể giải quyết bằng việc cắt ảnh cộng với chi phí suy luận cơ sở $c_0 = 1/\log|V|$ là:

$$\mathcal{R}_0(t) = (1 - \pi_t)\left(1 - \max_{y \in V} p_t(y)\right) + c_0$$

Quy tắc định tuyến động hoàn toàn không cần ngưỡng cứng $\tau$ được xác định bởi **Tỷ số Hậu nghiệm Bayesian (Bayesian Posterior Odds Ratio)**:

$$\Lambda_t = \frac{\mathrm{EVSI}_t}{\mathcal{R}_0(t)} = \frac{\pi_t \left(1 - \max_{y} p_t(y)\right) \tilde{H}(A_t^{\text{img}})}{(1 - \pi_t)\left(1 - \max_{y} p_t(y)\right) + c_0}$$

Như minh họa tại **Hình 2**, tỷ số $\Lambda_t$ giải quyết đồng thời bài toán câu dẫn đề và bài toán phân luồng tự động:
- **Giai đoạn dẫn đề lặp lại câu hỏi ($t < t^*$):** Khi mô hình sinh các từ *"SĐT"*, *"trên"*, *"bảng"*, *"là:"*, mô hình chủ yếu sao chép từ câu hỏi $Q$ ($\pi_t \ll 0.5$) và có xác suất dự đoán rất nhọn ($1 - \max_y p_t(y) \approx 0.02$). Do đó $\mathrm{EVSI}_t \approx 0 \implies \Lambda_t \ll 1$, hệ thống đi qua chuỗi dẫn đề một cách tự nhiên mà không bị đánh lừa.
- **Trường hợp $\Lambda_t \le 1, \; \forall t$ (Fast-Path — Nhánh trên Hình 2):** Ngay cả khi đến **Điểm rẽ Thực thể $t^*$** (từ `'Minh'` trong *"Tên nhà thuốc là: Minh An"*), dù $\pi_{t^*}$ tăng cao nhưng do chữ trên bảng hiệu lớn và rõ nét nên $\max_y p_{t^*}(y) = 0.92$ và sự chú ý hội tụ sắc nét ($\tilde{H}(A_{t^*}^{\text{img}})$ thấp), dẫn tới $\Lambda_{t^*} = 0.28 \le 1$. Hệ thống hoàn tất câu trả lời trên luồng nhanh với **0% chi phí phụ**.
- **Trường hợp xuất hiện bước đầu tiên $t^*$ thỏa $\Lambda_{t^*} > 1$ (Slow-Path — Nhánh dưới Hình 2):** Ngay tại bước $t^*$ khi kết thúc cụm dẫn đề $y_{<t^*}$ (*"SĐT trên bảng là:"*) và bắt đầu đọc đầu số nhỏ `'028'`, mô hình chuyển sự chú ý sang ảnh ($\pi_{t^*}$ tăng vọt) nhưng đồng thời rơi vào trạng thái phân vân giữa các đầu số (`'028'`, `'090'`, `'128'`) và bản đồ chú ý bị tán xạ. Khi đó $\mathrm{EVSI}_{t^*}$ vượt ngưỡng rủi ro nền ($\Lambda_{t^*} = 2.65 > 1$), CERV lập tức đánh dấu $t^*$ là **Điểm rẽ Thực thể**, khóa tiền tố dẫn đề $y_{<t^*}$ và kích hoạt Giai đoạn 2.

---

### 2.2. Giai đoạn 2: Định Vị Tương Phản Có Điều Kiện Tiền Tố Tại Điểm Rẽ $t^*$ (Prefix-Conditioned VAQ Localization)

Khi Slow-Path được kích hoạt tại Điểm rẽ Thực thể $t^*$, CERV tận dụng chính trạng thái ẩn tại bước $t^*$ (đã bao gồm tiền tố dẫn đề $y_{<t^*}$) để định vị tọa độ thực thể trên ảnh $I$. Việc định vị tại bước $t^*$ thay vì bước $t=1$ mang lại lợi thế quyết định: tiền tố $y_{<t^*}$ (ví dụ: *"SĐT trên bảng là:"*) đóng vai trò bộ lọc ngữ cảnh sắc bén, ép mô hình chỉ tìm kiếm vùng chứa con số tiếp theo thay vì phân tán sự chú ý ra toàn bộ các danh từ trong câu hỏi. Quy trình gồm 3 bước tuần tự tại **Hình 3**:

![Minh họa trực quan Giai đoạn 2: Định Vị Tương Phản Có Điều Kiện Tiền Tố](C:/Users/Admin/.gemini/antigravity/brain/befd0801-8f78-44cc-a3f9-2bcbaef5fa3e/cerv_stage2_vaq_localization.png)
*Hình 3: Minh họa 3 bước xử lý của Giai đoạn 2 tại Điểm rẽ Thực thể $t^*$: Khử nhiễu nền có điều kiện tiền tố $y_{<t^*}$ (2.2.1), chọn tầng hội tụ đỉnh $\ell^*$ (2.2.2) và kiến tạo cặp góc nhìn đối lập $I^+, I^-$ (2.2.3) ([Bản vẽ Draw.io](file:///c:/Users/Admin/HUIT%20-%20H%E1%BB%8Dc%20T%E1%BA%ADp/N%C4%83m%204/Computer_Vision/docs/figures/cerv_stage2_vaq_localization.drawio) | [Ảnh PNG](file:///c:/Users/Admin/HUIT%20-%20H%E1%BB%8Dc%20T%E1%BA%ADp/N%C4%83m%204/Computer_Vision/docs/figures/cerv_stage2_vaq_localization.png)).*

#### Bước 2.2.1. Khử Nhiễu Nền Có Điều Kiện Tiền Tố (Prefix-Conditioned Query Contrast)
Tại bước $t^*$, bản đồ liên kết giữa từ đang sinh với $P$ vùng ảnh tại tầng $\ell \in \{1, \dots, L\}$, ký hiệu là $s_\ell(I, Q, y_{<t^*}) \in \mathbb{R}^P$, bao gồm hai thành phần: (1) kích hoạt thị giác hướng thẳng vào thực thể cần điền tiếp theo sau tiền tố $y_{<t^*}$, và (2) nhiễu cấu trúc nền bất biến (*Attention Sink* tập trung ở tán cây, xe máy, biển hiệu phụ và các góc ảnh).

Để tách bỏ hoàn toàn nhiễu nền, CERV đối chiếu với bản đồ liên kết thị giác của **truy vấn rỗng có cùng tiền tố** $s_\ell(I, \varnothing, y_{<t^*}) \in \mathbb{R}^P$ (thay câu hỏi $Q$ bằng chuỗi rỗng $\varnothing$). Bản đồ bằng chứng thị giác thuần túy tại bước $t^*$ và tầng $\ell$ được thu nhận qua phép hiệu chỉnh lưu dương:

$$d_\ell\left(I, Q, y_{<t^*}\right) = \left[ s_\ell\left(I, Q, y_{<t^*}\right) - s_\ell\left(I, \varnothing, y_{<t^*}\right) \right]_+ = \max\left(0, \; s_\ell\left(I, Q, y_{<t^*}\right) - s_\ell\left(I, \varnothing, y_{<t^*}\right)\right)$$

#### Bước 2.2.2. Tự Động Lựa Chọn Tầng Hội Tụ Đỉnh Tại Điểm Rẽ ($\ell^*$)
Vì độ sâu hội tụ ngữ nghĩa thay đổi tùy theo dòng mô hình và độ phức tạp của thực thể, CERV định lượng cường độ **Kích hoạt Thị giác tại Điểm rẽ (Fork-VAQ)** ở mỗi tầng $\ell$ bằng chuẩn bậc hai:

$$\mathrm{VAQ}_\ell(t^*) = \left\| d_\ell\left(I, Q, y_{<t^*}\right) \right\|_2, \quad \ell^* = \arg\max_{\ell \in [L/4, \, 3L/4]} \mathrm{VAQ}_\ell(t^*)$$

Tầng $\ell^*$ đạt cực đại $\mathrm{VAQ}_\ell(t^*)$ chính là tầng mà mô hình định vị rõ nhất thực thể tiếp nối sau tiền tố $y_{<t^*}$. Bản đồ $d_{\ell^*}(I, Q, y_{<t^*})$ tại tầng này được chuẩn hóa thành phân phối xác suất không gian trên $P$ vùng ảnh:

$$\hat{d}_{\ell^*}(p) = \frac{d_{\ell^*}\left(p \mid I, Q, y_{<t^*}\right)}{\sum_{j=1}^{P} d_{\ell^*}\left(j \mid I, Q, y_{<t^*}\right) + \epsilon}, \quad \forall p \in \{1, \dots, P\}$$

#### Bước 2.2.3. Kiến Tạo Cặp Góc Nhìn Đối Lập ($I^+$ và $I^-$)
Sắp xếp các vùng ảnh giảm dần theo trọng số $\hat{d}_{\ell^*}(p)$, CERV trích xuất tập con nhỏ nhất các vùng ảnh đứng đầu $\mathcal{T}$ tích lũy đủ tỷ lệ năng lượng bằng chứng $\rho$ (mặc định $\rho = 0.85$):

$$\mathcal{T} = \mathrm{TopPatches}\left(\hat{d}_{\ell^*}, \, \rho\right)$$

Từ khung hình chữ nhật bao đóng $\mathcal{B}_{\text{crop}}$ quanh tập $\mathcal{T}$ (được đảm bảo kích thước tối thiểu bằng khung cảm thụ chuẩn của bộ mã hóa thị giác), CERV tạo ra hai góc nhìn đối lập cho Giai đoạn 3:
1. **Luồng Dương — Ảnh Phóng Đại Bằng Chứng ($I^+$):** Cắt vùng $\mathcal{B}_{\text{crop}}$ từ ảnh gốc $I$ và đưa về độ phân giải chuẩn của bộ mã hóa ảnh, giúp khôi phục chi tiết tần số cao của chuỗi số `"028 3925 1842"`:
   $$I^+ = \mathrm{Crop}\left(I, \, \mathcal{B}_{\text{crop}}\right)$$
2. **Luồng Phản Sự Thật — Ảnh Che Bằng Chứng ($I^-$):** Trên cùng khung cắt $\mathcal{B}_{\text{crop}}$, che khuất các vùng ảnh thực thể có trọng số cao nhất $\mathcal{T}_{\text{peak}}$ bằng giá trị màu trung bình cục bộ:
   $$I^- = \mathrm{Crop}\left(\mathrm{Mask}\left(I, \, \mathcal{T}_{\text{peak}}\right), \, \mathcal{B}_{\text{crop}}\right)$$

---

### 2.3. Giai đoạn 3: Giải Mã Kiểm Chứng Phản Sự Thật Neo Tại Điểm Rẽ ($t \ge t^*$)

Điểm khác biệt cốt lõi của Giai đoạn 3 trong CERV so với các phương pháp giải mã tương phản truyền thống là **Cơ chế Neo Tiền Tố tại Điểm rẽ $t^*$**: CERV **giữ nguyên hoàn toàn chuỗi từ dẫn đề $y_{<t^*}$** đã được sinh tự nhiên ở Giai đoạn 1, và chỉ thực hiện giải mã đối chiếu song song trên hai góc nhìn $(I^+, Q, y_{<t^*})$ và $(I^-, Q, y_{<t^*})$ **bắt đầu từ bước thực thể $t = t^*, t^*+1, \dots$ cho đến khi kết thúc câu**. Cơ chế này được minh họa tại **Hình 4**:

![Minh họa trực quan Giai đoạn 3: Giải Mã Kiểm Chứng Phản Sự Thật Neo Tại Điểm Rẽ](C:/Users/Admin/.gemini/antigravity/brain/befd0801-8f78-44cc-a3f9-2bcbaef5fa3e/cerv_stage3_counterfactual_verification.png)
*Hình 4: Minh họa cơ chế neo tiền tố dẫn đề $y_{<t^*}$ và đối chiếu điểm số dự đoán giữa Luồng Dương $I^+$ và Luồng Âm $I^-$ từ bước $t \ge t^*$, giúp bảo toàn câu dẫn đề và loại bỏ triệt để các từ đoán mò (`'3825'`, `'128'`) ([Bản vẽ Draw.io](file:///c:/Users/Admin/HUIT%20-%20H%E1%BB%8Dc%20T%E1%BA%ADp/N%C4%83m%204/Computer_Vision/docs/figures/cerv_stage3_counterfactual_verification.drawio) | [Ảnh PNG](file:///c:/Users/Admin/HUIT%20-%20H%E1%BB%8Dc%20T%E1%BA%ADp/N%C4%83m%204/Computer_Vision/docs/figures/cerv_stage3_counterfactual_verification.png)).*

Tại mỗi bước giải mã thực thể $t \ge t^*$, mô hình sinh hai vector điểm số dự đoán có cùng điều kiện tiền tố $y_{<t} = (y_{<t^*}, y_{t^* : t-1})$:
- $z^+_t = z_t\left(I^+, Q, y_{<t}\right)$: Điểm số dự đoán khi **nhìn rõ thực thể trên ảnh phóng đại** (bao gồm cả tín hiệu điểm ảnh thực tế lẫn thiên kiến ngôn ngữ từ tiền tố $y_{<t}$).
- $z^-_t = z_t\left(I^-, Q, y_{<t}\right)$: Điểm số dự đoán khi **thực thể mục tiêu đã bị che khuất** (chỉ còn lại thiên kiến đoán mò theo thói quen ngôn ngữ nối tiếp tiền tố $y_{<t}$ và các chi tiết phụ xung quanh).

Độ lợi bằng chứng thực tế ở mức từ vựng tại bước $t \ge t^*$ (**Fork-Anchored Visual Activation of Tokens — $\mathrm{VAT}_t$**) là hiệu số trực tiếp giữa hai luồng:

$$\mathrm{VAT}_t = z^+_t - z^-_t, \quad \forall t \ge t^*$$

Như thể hiện trên biểu đồ giữa **Hình 4**:
- Vì cả hai luồng đều chia sẻ chung tiền tố dẫn đề $y_{<t^*}$ (*"SĐT trên bảng là:"*), mọi thói quen đoán mò số điện thoại phổ biến (`'3825'`) hoặc đọc nhầm số địa chỉ không bị che xung quanh (`'128'`, `'Quận 3'`) đều sinh ra mức điểm cao tương đương trên cả $I^+$ lẫn $I^-$ $\implies z^+_t(y) \approx z^-_t(y) \implies \mathrm{VAT}_t(y) \approx 0$.
- Chỉ duy nhất từ đúng `'3925'` (xuất hiện rõ trong vùng bằng chứng của $I^+$ nhưng đã bị che mất trên $I^-$) mới tạo ra sự sụt giảm xác suất đột ngột giữa hai luồng ($0.85$ xuống $0.12$) $\implies \mathrm{VAT}_t(\text{'3925'}) = +0.73 \gg 0$.

Thay vì áp đặt một hệ số khuếch đại cố định $\alpha = 1.0$ cho mọi bước $t \ge t^*$ (vốn có thể làm giảm độ trôi chảy ngữ pháp khi mô hình sinh tiếp các từ đóng câu sau thực thể nơi hai luồng $I^+$ và $I^-$ đã đồng thuận), CERV xác định **Hệ số Khuếch đại Thông tin Thích nghi $\alpha_t \in [0, 1]$** tại từng bước thời gian dựa trên tỷ số giữa độ phân kỳ Jensen-Shannon $\mathrm{JSD}(p_t^+ \parallel p_t^-)$ và cận trên entropy của phân phối hỗn hợp trung bình $m_t = \frac{1}{2}(p_t^+ + p_t^-)$:

$$\alpha_t = \frac{\mathrm{JSD}\left(p_t^+ \parallel p_t^-\right)}{H\left(\frac{p_t^+ + p_t^-}{2}\right)} = 1 - \frac{H\left(p_t^+\right) + H\left(p_t^-\right)}{2 H\left(m_t\right)} \in [0, 1]$$

Theo tính lõm của hàm entropy Shannon ($0 \le \mathrm{JSD}(p_t^+ \parallel p_t^-) \le H(m_t)$), trọng số $\alpha_t$ luôn bị chặn chặt trong đoạn $[0, 1]$ mà không cần bất kỳ hằng số làm trơn $\epsilon$ hay ngưỡng thủ công nào:
- **Tại các bước từ thực thể bị che khuất trên $I^-$ (như `'3925'`):** Phân phối $p_t^+$ (nhìn rõ chữ) và $p_t^-$ (đã che chữ) khác biệt mạnh ($\mathrm{JSD}(p_t^+ \parallel p_t^-) \to H(m_t)$), kéo $\alpha_t \to 1$ để khuếch đại tối đa tín hiệu điểm ảnh thực và triệt tiêu ảo giác ngôn ngữ.
- **Tại các từ nối hoặc từ đóng câu ngữ pháp sau thực thể:** Khi cả hai luồng $I^+$ và $I^-$ dự đoán tương đồng ($p_t^+ \approx p_t^- \implies \mathrm{JSD}(p_t^+ \parallel p_t^-) \to 0$), trọng số $\alpha_t$ tự động thu về $0$, trả công thức giải mã về đúng phân phối tự nhiên $z^+_t(y)$.

Điểm số giải mã cuối cùng tại mỗi bước $t \ge t^*$ kết hợp trọng số thích nghi $\alpha_t$ với điều kiện lọc độ hợp lý xác suất $\mathcal{V}_{\text{head}}(z^+_t)$ (với $\beta = 0.1$) để ngăn khuếch đại nhiễu ở đuôi phân phối:

$$z^{\text{CERV}}_t(y) = \begin{cases} z^+_t(y) + \alpha_t \left( z^+_t(y) - z^-_t(y) \right) = (1 + \alpha_t) z^+_t(y) - \alpha_t z^-_t(y), & \text{nếu } y \in \mathcal{V}_{\text{head}}(z^+_t) \\ -\infty, & \text{ngược lại} \end{cases}$$

trong đó:

$$\mathcal{V}_{\text{head}}(z^+_t) = \left\{ y \in V \;:\; p_t\left(y \mid I^+, Q, y_{<t}\right) \ge \beta \max_{w \in V} p_t\left(w \mid I^+, Q, y_{<t}\right) \right\}$$

Từ thực thể tại mỗi bước $t \ge t^*$ được xác định bởi $y^*_t = \arg\max_{y \in V} z^{\text{CERV}}_t(y)$ và nối tiếp vào sau tiền tố dẫn đề $y_{<t^*}$ cho đến khi gặp ký hiệu kết thúc chuỗi `<eos>`. Nhờ sự kết hợp chặt chẽ giữa **Phát hiện Điểm rẽ Thực thể bằng Tỷ số Hậu nghiệm Bayesian (Giai đoạn 1)**, **Định vị Tương phản Có điều kiện Tiền tố tại $t^*$ (Giai đoạn 2)** và **Kiểm chứng Phản sự thật Thích nghi $\alpha_t$ Neo tại $t^*$ (Giai đoạn 3)**, CERV giải quyết trọn vẹn bài toán câu dẫn đề, loại bỏ hoàn toàn các ngưỡng/trọng số cứng thủ công và chứng thực nhân quả từng thực thể thị giác trên mọi dòng mô hình MLLM.
