# Original User Request

## Initial Request — 2026-10-28T05:00:00Z

Upgrade and evaluate the SG-VCD (Spatially-Grounded & Prior-Calibrated Contrastive Decoding) algorithm on `5CD-AI/Vintern-1B-v3_5` across all 400 human-annotated ViTextVQA images (1,222 questions) until all Stage 2 target metrics exceed 80.0% (Token F1, Substring Match, ANLS) and Stage 1 spatial metrics >= 80.0%.

Target Remote & Execution:
- GitHub: https://github.com/CaoAnhNato/VLM_Reasoning.git (main branch)
- Pre-connected Google Colab L4 Notebook: https://colab.research.google.com/drive/1HcRiBJxEIQsVQYNVCpFcB6xsLI5CYzYZ#scrollTo=7THeva8p4Ndy (improve_SG-VCD.ipynb)
- Zero-heuristic, zero hardcoding, zero stopword lists, zero coordinate forcing, and zero ground-truth data leakage during inference.
- Local-first development: test locally on .\venv\Scripts\python.exe before pushing.
- Download all benchmark artifacts to results/sg_vcd_400_benchmark/ and update docs/Proposed_Method_SG_VCD.md.

## Follow-up — 2026-10-28T16:10:22+07:00

Vì đợt chạy trước đó của subagent tiêu thụ khá ít tài nguyên có sẵn của máy ảo L4 và sử dụng GPU và RAM cũng chưa tối ưu, nên tôi muốn bạn thêm lời nhắc các subagent sau này có chạy inference model thì hãy dùng phương pháp 'Multi-Worker Data Sharding trên cùng 1 GPU L4' - Chạy nhiều tiến trình cùng lúc để tăng tốc quá trình này. Đương nhiên là sẽ test trước để đảm bảo code chạy ổn định (với khoảng 20 sample) rồi mới tiến hành chạy full 400 ảnh.
[Attached screenshot: RAM hệ thống 4.6 / 53.0 GB, RAM GPU 2.3 / 22.5 GB, Ổ đĩa 46.6 / 235.7 GB]

## Follow-up — 2026-10-28T17:06:15+07:00

thêm 1 lưu ý nhỏ nữa, khi chạy local thì chỉ nên thực hiện với dữ liệu giả để đảm bảo code chạy thôi, tránh chạy với data thật sẽ làm tràn RAM và CPU.

## Follow-up — 2026-10-29T10:51:23Z

Lỗi loại Loại C (83.2% lỗi — Đọc lố chữ trên biển hiệu): Ví dụ đáp án là "nhà thuốc", mô hình đọc luôn cả dòng "Nhà thuốc Bảo Tâm An" khiến độ dài câu tăng gấp đôi, làm ANLS bị phạt về 0.0 và F1 giảm còn 57.1%. -> Tôi thấy đáp án cũng tương tự nhau nên trường hợp này model đọc đúng nhé, thậm chí còn trả lời chi tiết nữa.
Bên cạnh đó thì còn các trường hợp đáp án số như '15 . 000' và model đọc là '15.000đ/kg' thì cũng hoàn toàn đúng nhé, này là lỗi grounth truth làm không kỹ.
Cần điều chỉnh penalty đánh giá trong `scripts/benchmark/metrics.py` để:
1. Xử lý khoảng trắng số ('15 . 000' vs '15.000') và hậu tố tiền tệ / đơn vị tính (đ, vnđ, đ/kg, đ/lít, v.v.) khi chuẩn hóa.
2. Token F1 và ANLS không bị phạt quá nặng khi GT bị thiếu sót / không đầy đủ so với câu trả lời chi tiết chính xác của model trên biển hiệu (ví dụ sub-span ANLS/F1 hoặc relaxation cho entity expansion hợp lệ).


## Follow-up — 2026-10-29T12:17:25Z

CRITICAL HALT ON VICTORY CLAIM — ROUND 6 REGRESSED BELOW 80.0%; LAUNCH ROUND 7 IMMEDIATELY:
Look at the Round 6 Stage 2 metrics:
- `Substring Match`: **78.66%** (REGRESSED from **80.19%** in Round 5 — now BELOW 80.0%!)
- `Token F1`: **74.57%** (REGRESSED from **75.95%** in Round 5 — BELOW 80.0%!)
- `Relaxed Accuracy / ANLS`: **70.82%** (REGRESSED from **72.18%** in Round 5 — BELOW 80.0%!)
- `Exact Match`: **50.90%** (REGRESSED from **52.95%** in Round 5!)

**NONE of the three Stage 2 metrics in Round 6 exceed 80.0%!** Do NOT claim victory or trigger the Victory Auditor—it will immediately fail the >80.0% acceptance criteria!

Instead, immediately launch **Round 7** with `orchestrator_4`:
1. **Why did Round 6 regress from Round 5 (`80.19%` -> `78.66%` Substring Match)?**
   - The Round 6 additions (`DCEB` early EOS boosting in `dynamic_cd.py` and/or `SyntacticSpanRefiner` SCPDO/QAMIF in `pipeline.py`) prematurely truncated generation or stripped valid answer tokens! Revert or fix the destructive parts of `DCEB` / `SyntacticSpanRefiner` so that Round 5's `80.19%+` Substring Match is restored and improved.
2. **Why did the `metrics.py` adjustment in Round 6 fail to lift `Token F1` and `ANLS` > 80.0%?**
   - On the 1,222 predictions in `predictions.jsonl` where `Substring Match` is already `80.19%` (in Round 5), nearly all of those 80.19% correct answers are Type C signboard expansions (e.g. `"Nhà thuốc Bảo Tâm An"` vs GT `"nhà thuốc"`) or numeric/unit formatting variants (`"15 . 000"` vs `"15.000đ/kg"`).
   - If `metrics.py` properly recognizes Type C entity expansions (when the normalized GT is a substring/sub-sequence of the normalized prediction within a valid signboard reading) and numeric/unit variants as full/near-full matches in `compute_token_f1` and `compute_anls`, then whenever `Substring Match > 80.0%`, `Token F1` and `ANLS` MUST mathematically also be `~80.0%+`!
   - Why were `Token F1` (74.57%) and `ANLS` (70.82%) still 4–8% lower than `Substring Match` (78.66%)? Inspect `scripts/benchmark/metrics.py` and `scripts/benchmark/evaluate_vitextvqa.py` locally against the actual 1,222 rows in `predictions.jsonl` to see which Type C / substring-match samples are STILL being penalized in `compute_token_f1` and `compute_anls`!
3. **Verify locally on the 1,222 predictions first**, ensure all unit/E2E tests pass, push to GitHub, and run the Round 7 Colab L4 benchmark so that ALL THREE Stage 2 metrics (`Substring Match`, `Token F1`, `ANLS`) exceed **80.0%** alongside Stage 1 >= **80.0%**.
