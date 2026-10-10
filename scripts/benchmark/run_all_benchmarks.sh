#!/usr/bin/env bash
# ==============================================================================
# SCRIPT ĐIỀU PHỐI BENCHMARK TOÀN DIỆN 7 MÔ HÌNH VLM TRÊN ViTextVQA
# Hỗ trợ tự động RESUME (bỏ qua mô hình đã chạy), chạy từ model chỉ định,
# và dọn dẹp cache tuần tự sau mỗi model để tối ưu dung lượng ổ đĩa.
# ==============================================================================
set -e

DATA_DIR="data/vitextvqa"
OUT_DIR="results"
START_INDEX=1
ONLY_INDEX=""
MAX_SAMPLES=""
FORCE_RERUN=false

# Phân tích tham số dòng lệnh
while [[ $# -gt 0 ]]; do
    case "$1" in
        --start|--start-from)
            START_INDEX="$2"
            shift 2
            ;;
        --only)
            ONLY_INDEX="$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --force)
            FORCE_RERUN=true
            shift
            ;;
        [0-9]*)
            # Nếu người dùng truyền trực tiếp số, hiểu là max_samples (ví dụ: ./run_all_benchmarks.sh 10)
            MAX_SAMPLES="$1"
            shift
            ;;
        *)
            echo "Tham số không hợp lệ: $1"
            echo "Cách dùng:"
            echo "  ./run_all_benchmarks.sh                  # Chạy tự động resume các model còn lại"
            echo "  ./run_all_benchmarks.sh --start 3        # Bắt đầu chạy từ model số 3"
            echo "  ./run_all_benchmarks.sh --only 3         # Chỉ chạy riêng model số 3"
            echo "  ./run_all_benchmarks.sh 10               # Chạy thử 10 mẫu (smoke test)"
            echo "  ./run_all_benchmarks.sh --force          # Chạy lại tất cả, ghi đè kết quả cũ"
            exit 1
            ;;
    esac
done

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

TOTAL_MODELS=${#MODELS[@]}

echo "======================================================================"
echo " CHƯƠNG TRÌNH BENCHMARK 7 MÔ HÌNH VLM TRÊN ViTextVQA"
echo " Tổng số mô hình: $TOTAL_MODELS"
echo " Thư mục kết quả: $OUT_DIR"
if [ -n "$MAX_SAMPLES" ]; then echo " Giới hạn số mẫu: $MAX_SAMPLES mẫu/model (Test nhanh)"; fi
if [ -n "$ONLY_INDEX" ]; then echo " Chỉ định chạy: Model số $ONLY_INDEX"; fi
if [ "$START_INDEX" -gt 1 ]; then echo " Bắt đầu từ: Model số $START_INDEX"; fi
echo "======================================================================"

for i in "${!MODELS[@]}"; do
    MODEL_NUM=$((i + 1))
    ENTRY="${MODELS[$i]}"
    IFS=":" read -r MODEL_NAME BACKEND <<< "$ENTRY"
    SAFE_NAME=$(echo "$MODEL_NAME" | tr '/' '_')
    PRED_FILE="$OUT_DIR/predictions/${SAFE_NAME}_preds.jsonl"

    # 1. Kiểm tra nếu chỉ định chạy 1 model cụ thể
    if [ -n "$ONLY_INDEX" ] && [ "$MODEL_NUM" -ne "$ONLY_INDEX" ]; then
        continue
    fi

    # 2. Kiểm tra nếu chỉ định bắt đầu từ model số X
    if [ -z "$ONLY_INDEX" ] && [ "$MODEL_NUM" -lt "$START_INDEX" ]; then
        echo "[BỎ QUA] Model #$MODEL_NUM ($MODEL_NAME) nhỏ hơn mốc bắt đầu #$START_INDEX."
        continue
    fi

    # 3. Thông báo trạng thái checkpoint / resume nếu đã có dữ liệu trước đó
    if [ -f "$PRED_FILE" ] && [ -s "$PRED_FILE" ]; then
        COUNT_PREDS=$(wc -l < "$PRED_FILE" | tr -d ' ')
        echo ""
        echo "----------------------------------------------------------------------"
        echo "[RESUME CHECKPOINT] Model #$MODEL_NUM ($MODEL_NAME)"
        echo "Đã ghi nhận $COUNT_PREDS mẫu dự đoán tại: $PRED_FILE"
        if [ "$FORCE_RERUN" = true ]; then
            echo "Chế độ --force: Sẽ ghi đè và đánh giá lại toàn bộ từ đầu!"
        else
            echo "Tiến trình sẽ tự động tiếp tục từ mẫu tiếp theo (hoặc bỏ qua nếu đã đủ 100%)."
        fi
        echo "----------------------------------------------------------------------"
    fi

    echo ""
    echo "######################################################################"
    echo ">>> ĐANG CHẠY MODEL #$MODEL_NUM / $TOTAL_MODELS: $MODEL_NAME"
    echo ">>> BACKEND: $BACKEND"
    echo ">>> BẮT ĐẦU: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "######################################################################"

    EXTRA_ARGS=""
    if [ -n "$MAX_SAMPLES" ]; then
        EXTRA_ARGS="$EXTRA_ARGS --max_samples $MAX_SAMPLES"
    fi
    if [ "$FORCE_RERUN" = true ]; then
        EXTRA_ARGS="$EXTRA_ARGS --overwrite"
    fi

    python3 scripts/benchmark/evaluate_vitextvqa.py \
        --model_name "$MODEL_NAME" \
        --backend "$BACKEND" \
        --data_dir "$DATA_DIR" \
        $EXTRA_ARGS \
        --output_jsonl "$PRED_FILE" \
        --summary_csv "$OUT_DIR/vitextvqa_benchmark_summary.csv" \
        2>&1 | tee "$OUT_DIR/logs/${SAFE_NAME}.log"

    echo "[HOÀN THÀNH MODEL #$MODEL_NUM] Đã đánh giá xong $MODEL_NAME và giải phóng ổ đĩa!"
done

echo ""
echo "======================================================================"
echo ">>> TIẾN TRÌNH BENCHMARK ĐÃ KẾT THÚC! <<<"
echo ">>> BẢNG TỔNG HỢP ĐIỂM SỐ HIỆN TẠI TẠI: $OUT_DIR/vitextvqa_benchmark_summary.csv"
echo "======================================================================"

if [ -f "$OUT_DIR/vitextvqa_benchmark_summary.csv" ]; then
    column -s, -t < "$OUT_DIR/vitextvqa_benchmark_summary.csv" 2>/dev/null || cat "$OUT_DIR/vitextvqa_benchmark_summary.csv"
fi
