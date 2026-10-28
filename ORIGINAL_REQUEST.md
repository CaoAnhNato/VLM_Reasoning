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
