#!/usr/bin/env bash
# ==============================================================================
# SCRIPT ĐIỀU PHỐI BENCHMARK TOÀN DIỆN 7 MÔ HÌNH VLM TRÊN ViTextVQA
# Thực thi tuần tự, dọn dẹp cache sau mỗi model để tối ưu dung lượng ổ đĩa.
# ==============================================================================
set -e

DATA_DIR="data/vitextvqa"
OUT_DIR="results"
MAX_SAMPLES="${1:-}"  # Truyền số nguyên nếu muốn chạy test nhanh (ví dụ: ./run_all_benchmarks.sh 10)

if [ ! -d "$DATA_DIR" ]; then
    echo "======================================================================"
    echo "[LỖI] Không tìm thấy thư mục dữ liệu tại: $DATA_DIR"
    echo "Vui lòng chạy script tải dữ liệu trước: ./scripts/benchmark/download_dataset.sh <HF_TOKEN>"
    echo "======================================================================"
    exit 1
fi

mkdir -p "$OUT_DIR/predictions"
mkdir -p "$OUT_DIR/logs"

# Danh sách 7 mô hình ứng viên từ docs/Report.docx
MODELS=(
    "Qwen/Qwen2.5-VL-3B-Instruct:unsloth"
    "Qwen/Qwen2.5-VL-7B-Instruct:unsloth"
    "5CD-AI/Vintern-1B-v3.5:native"
    "5CD-AI/Vintern-3B-beta:native"
    "EraX-AI/EraX-VL-2B-V1.5:native"
    "EraX-AI/EraX-VL-7B-V1.5:native"
    "CohereForAI/aya-vision-8b:native"
)

EXTRA_ARGS=""
if [ -n "$MAX_SAMPLES" ]; then
    EXTRA_ARGS="--max_samples $MAX_SAMPLES"
    echo "[CHẾ ĐỘ THỬ NGHIỆM NHANH] Đang giới hạn đánh giá tối đa $MAX_SAMPLES mẫu mỗi model."
fi

echo "======================================================================"
echo " BẮT ĐẦU CHUỖI ĐÁNH GIÁ 7 MÔ HÌNH ZERO-SHOT TRÊN ViTextVQA"
echo " Tổng số mô hình: ${#MODELS[@]}"
echo " Thư mục kết quả: $OUT_DIR"
echo "======================================================================"

for ENTRY in "${MODELS[@]}"; do
    IFS=":" read -r MODEL_NAME BACKEND <<< "$ENTRY"
    SAFE_NAME=$(echo "$MODEL_NAME" | tr '/' '_')
    
    echo ""
    echo "######################################################################"
    echo ">>> TIẾN HÀNH ĐÁNH GIÁ: $MODEL_NAME"
    echo ">>> BACKEND: $BACKEND"
    echo ">>> THỜI GIAN: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "######################################################################"
    
    python3 scripts/benchmark/evaluate_vitextvqa.py \
        --model_name "$MODEL_NAME" \
        --backend "$BACKEND" \
        --data_dir "$DATA_DIR" \
        $EXTRA_ARGS \
        --output_jsonl "$OUT_DIR/predictions/${SAFE_NAME}_preds.jsonl" \
        --summary_csv "$OUT_DIR/vitextvqa_benchmark_summary.csv" \
        2>&1 | tee "$OUT_DIR/logs/${SAFE_NAME}.log"
        
    echo "[HOÀN THÀNH] Mô hình $MODEL_NAME đã đánh giá xong và giải phóng ổ đĩa!"
done

echo ""
echo "======================================================================"
echo ">>> TẤT CẢ 7 MÔ HÌNH ĐÃ HOÀN THÀNH BENCHMARK THÀNH CÔNG! <<<"
echo ">>> BẢNG TỔNG KẾT ĐIỂM SỐ TẠI: $OUT_DIR/vitextvqa_benchmark_summary.csv"
echo "======================================================================"

if [ -f "$OUT_DIR/vitextvqa_benchmark_summary.csv" ]; then
    cat "$OUT_DIR/vitextvqa_benchmark_summary.csv"
fi
