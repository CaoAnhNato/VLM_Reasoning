#!/usr/bin/env bash
# ==============================================================================
# SCRIPT ĐIỀU PHỐI BENCHMARK TOÀN DIỆN 7 MÔ HÌNH VLM TRÊN ViTextVQA
# Hỗ trợ tự động RESUME (bỏ qua mô hình đã chạy), chạy từ model chỉ định,
# và dọn dẹp cache tuần tự sau mỗi model để tối ưu dung lượng ổ đĩa.
# ==============================================================================
set -eo pipefail

DATA_DIR="data/vitextvqa"
OUT_DIR="results"
START_INDEX=1
ONLY_INDEX=""
FORCE_RERUN=false
PROFILE_MODE=false
PROFILE_SAMPLES=100
HF_TOKEN="${HF_TOKEN:-}"
BATCH_SIZE=4
HOURLY_RATE=""

# Phân tích tham số dòng lệnh
while [[ $# -gt 0 ]]; do
    case "$1" in
        --token|--hf_token|--hf-token)
            HF_TOKEN="$2"
            shift 2
            ;;
        --profile)
            PROFILE_MODE=true
            if [[ $# -ge 2 && "$2" =~ ^[0-9]+$ ]]; then
                PROFILE_SAMPLES="$2"
                shift 2
            else
                shift 1
            fi
            ;;
        --start|--start-from)
            START_INDEX="$2"
            shift 2
            ;;
        --only)
            ONLY_INDEX="$2"
            shift 2
            ;;
        --force)
            FORCE_RERUN=true
            shift
            ;;
        --batch_size|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --hourly_rate|--hourly-rate)
            HOURLY_RATE="$2"
            shift 2
            ;;
        *)
            echo "Tham số không hợp lệ: $1"
            echo "Cách dùng:"
            echo "  ./run_all_benchmarks.sh                  # Chạy Full-testcase 100% toàn bộ 7 mô hình (tự động resume)"
            echo "  ./run_all_benchmarks.sh --hf_token <TOK> # Chạy Full-testcase kèm Hugging Face Token"
            echo "  ./run_all_benchmarks.sh --batch_size 4   # Chỉ định kích thước lô tối đa"
            echo "  ./run_all_benchmarks.sh --profile        # Chạy Profiling 100 mẫu đo tốc độ, VRAM, RAM và ngoại suy chi phí Vast.ai"
            echo "  ./run_all_benchmarks.sh --profile 50     # Chạy Profiling với 50 mẫu"
            echo "  ./run_all_benchmarks.sh --profile 100 --hf_token <TOK> --hourly_rate 0.469 # Profiling kèm giá thuê RTX 5090"
            echo "  ./run_all_benchmarks.sh --start 3        # Bắt đầu chạy từ model số 3"
            echo "  ./run_all_benchmarks.sh --only 3         # Chỉ chạy riêng model số 3"
            echo "  ./run_all_benchmarks.sh --force          # Chạy lại tất cả từ đầu, ghi đè kết quả cũ"
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
mkdir -p "$OUT_DIR/profiling"
mkdir -p "$OUT_DIR/logs"
mkdir -p "$OUT_DIR/logs/profile"

# Danh sách 7 mô hình ứng viên từ docs/Report.docx
MODELS=(
    "unsloth/Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit:unsloth"
    "unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit:unsloth"
    "5CD-AI/Vintern-1B-v3_5:native"
    "5CD-AI/Vintern-3B-beta:native"
    "erax-ai/EraX-VL-2B-V1.5:native"
    "erax-ai/EraX-VL-7B-V1.5:native"
    "unsloth/aya-vision-8b-unsloth-bnb-4bit:auto"
)

TOTAL_MODELS=${#MODELS[@]}

echo "======================================================================"
echo " CHƯƠNG TRÌNH BENCHMARK 7 MÔ HÌNH VLM TRÊN ViTextVQA"
if [ "$PROFILE_MODE" = true ]; then
    echo " CHẾ ĐỘ THỰC THI: PROFILING HIỆU NĂNG ($PROFILE_SAMPLES mẫu đầu tiên)"
else
    echo " CHẾ ĐỘ THỰC THI: FULL-TESTCASE (100% Tập Test Set)"
fi
# Kích hoạt chế độ tải dữ liệu siêu tốc bằng Xet High Performance của Hugging Face
export HF_XET_HIGH_PERFORMANCE=1

if [ -n "$HF_TOKEN" ]; then
    export HF_TOKEN="$HF_TOKEN"
    echo " Hugging Face Token: [ĐÃ CẤU HÌNH (Authenticated)]"
    if command -v hf &> /dev/null; then
        hf auth login --token "$HF_TOKEN" &>/dev/null || true
    fi
else
    echo " Hugging Face Token: [CHƯA CẤU HÌNH (Unauthenticated - Khuyến nghị dùng --hf_token)]"
fi
echo " Chế độ tăng tốc Hub: Xet High Performance (HF_XET_HIGH_PERFORMANCE=1)"
echo " Tổng số mô hình: $TOTAL_MODELS"
echo " Thư mục kết quả: $OUT_DIR"
if [ -n "$ONLY_INDEX" ]; then echo " Chỉ định chạy: Model số $ONLY_INDEX"; fi
if [ "$START_INDEX" -gt 1 ]; then echo " Bắt đầu từ: Model số $START_INDEX"; fi
echo "======================================================================"

cleanup_hf_models() {
    echo ""
    echo "----------------------------------------------------------------------"
    echo "[DỌN DẸP Ổ ĐĨA] Đang xóa sạch model weights cache để giải phóng tối đa ổ đĩa..."
    local CACHE_DIRS=(
        "${HF_HOME:-/workspace/.hf_home}/hub"
        "${HF_HOME:-/workspace/.hf_home}"
        "/workspace/.hf_home/hub"
        "/workspace/.hf_home"
        "${HUGGINGFACE_HUB_CACHE:-}"
        "$HOME/.cache/huggingface/hub"
        "$HOME/.cache/huggingface"
        "/root/.cache/huggingface/hub"
        "/root/.cache/huggingface"
    )

    for cdir in "${CACHE_DIRS[@]}"; do
        if [ -n "$cdir" ] && [ -d "$cdir" ]; then
            find "$cdir" -maxdepth 2 -type d -name "models--*" -prune -exec rm -rf {} + 2>/dev/null || true
            find "$cdir" -maxdepth 2 -type d -name ".tmp*" -prune -exec rm -rf {} + 2>/dev/null || true
            find "$cdir" -maxdepth 2 -type d -name "tmp*" -prune -exec rm -rf {} + 2>/dev/null || true
        fi
    done

    local MODULE_DIRS=(
        "/workspace/.hf_home/modules"
        "$HOME/.cache/huggingface/modules"
        "/root/.cache/huggingface/modules"
    )
    for mdir in "${MODULE_DIRS[@]}"; do
        if [ -d "$mdir" ]; then
            rm -rf "$mdir"/* 2>/dev/null || true
        fi
    done

    rm -rf /tmp/huggingface* /tmp/xet* /tmp/torch* /tmp/*models* 2>/dev/null || true
    echo "-> Dung lượng ổ đĩa khả dụng hiện tại:"
    df -h / | awk 'NR==1 || NR==2'
    echo "----------------------------------------------------------------------"
}

# Dọn dẹp sơ bộ trước khi bắt đầu chuỗi thực nghiệm
cleanup_hf_models

for i in "${!MODELS[@]}"; do
    MODEL_NUM=$((i + 1))
    ENTRY="${MODELS[$i]}"
    IFS=":" read -r MODEL_NAME BACKEND <<< "$ENTRY"
    SAFE_NAME=$(echo "$MODEL_NAME" | tr '/' '_')

    # 1. Kiểm tra nếu chỉ định chạy 1 model cụ thể
    if [ -n "$ONLY_INDEX" ] && [ "$MODEL_NUM" -ne "$ONLY_INDEX" ]; then
        continue
    fi

    # 2. Kiểm tra nếu chỉ định bắt đầu từ model số X
    if [ -z "$ONLY_INDEX" ] && [ "$MODEL_NUM" -lt "$START_INDEX" ]; then
        echo "[BỎ QUA] Model #$MODEL_NUM ($MODEL_NAME) nhỏ hơn mốc bắt đầu #$START_INDEX."
        continue
    fi

    # Đảm bảo dọn dẹp sạch sẽ ổ đĩa trước khi tải model mới
    cleanup_hf_models

    # Thiết lập đường dẫn file và cờ chạy theo chế độ
    if [ "$PROFILE_MODE" = true ]; then
        PRED_FILE="$OUT_DIR/profiling/${SAFE_NAME}_profile_${PROFILE_SAMPLES}_preds.jsonl"
        LOG_FILE="$OUT_DIR/logs/profile/${SAFE_NAME}.log"
        EXTRA_ARGS="--profile $PROFILE_SAMPLES --profile_csv $OUT_DIR/profiling_summary.csv"
    else
        PRED_FILE="$OUT_DIR/predictions/${SAFE_NAME}_preds.jsonl"
        LOG_FILE="$OUT_DIR/logs/${SAFE_NAME}.log"
        EXTRA_ARGS=""
        if [ "$FORCE_RERUN" = true ]; then
            EXTRA_ARGS="--overwrite"
        fi
    fi

    # Bổ sung HF Token vào tham số chạy nếu có
    if [ -n "$HF_TOKEN" ]; then
        EXTRA_ARGS="$EXTRA_ARGS --hf_token $HF_TOKEN"
    fi

    # Bổ sung batch_size và hourly_rate nếu người dùng cấu hình
    if [ "$BATCH_SIZE" -gt 0 ] 2>/dev/null; then
        EXTRA_ARGS="$EXTRA_ARGS --batch_size $BATCH_SIZE"
    fi
    if [ -n "$HOURLY_RATE" ]; then
        EXTRA_ARGS="$EXTRA_ARGS --hourly_rate $HOURLY_RATE"
    fi

    # Tự động phát hiện python binary
    PYTHON_CMD="python3"
    if [ -f "/venv/main/bin/python" ]; then
        PYTHON_CMD="/venv/main/bin/python"
    elif [ -f "./venv/bin/python" ]; then
        PYTHON_CMD="./venv/bin/python"
    elif command -v python3 &>/dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &>/dev/null; then
        PYTHON_CMD="python"
    fi

    # 3. Thông báo trạng thái checkpoint / resume nếu đã có dữ liệu trước đó (chỉ áp dụng ở Full mode)
    if [ "$PROFILE_MODE" = false ] && [ -f "$PRED_FILE" ] && [ -s "$PRED_FILE" ]; then
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

    $PYTHON_CMD scripts/benchmark/evaluate_vitextvqa.py \
        --model_name "$MODEL_NAME" \
        --backend "$BACKEND" \
        --data_dir "$DATA_DIR" \
        $EXTRA_ARGS \
        --output_jsonl "$PRED_FILE" \
        --summary_csv "$OUT_DIR/vitextvqa_benchmark_summary.csv" \
        2>&1 | tee "$LOG_FILE"

    EXIT_CODE=${PIPESTATUS[0]}

    # Luôn dọn dẹp model weights ngay sau khi hoàn thành lượt chạy (hoặc khi gặp lỗi)
    cleanup_hf_models

    if [ $EXIT_CODE -ne 0 ]; then
        echo "======================================================================"
        echo "[LỖI NGHIÊM TRỌNG] Model #$MODEL_NUM ($MODEL_NAME) gặp lỗi thoát (Exit code: $EXIT_CODE)!"
        echo "Vui lòng kiểm tra chi tiết tại file log: $LOG_FILE"
        echo "======================================================================"
        exit $EXIT_CODE
    fi

    echo "[HOÀN THÀNH MODEL #$MODEL_NUM] Đã xử lý xong $MODEL_NAME và giải phóng ổ đĩa!"
done

echo ""
if [ "$PROFILE_MODE" = true ]; then
    echo "======================================================================"
    echo ">>> TIẾN TRÌNH PROFILING HIỆU NĂNG ĐÃ KẾT THÚC! <<<"
    echo ">>> BẢNG TỔNG HỢP PROFILING HIỆU NĂNG TẠI: $OUT_DIR/profiling_summary.csv"
    echo "======================================================================"
    if [ -f "$OUT_DIR/profiling_summary.csv" ]; then
        column -s, -t < "$OUT_DIR/profiling_summary.csv" 2>/dev/null || cat "$OUT_DIR/profiling_summary.csv"
    fi
else
    echo "======================================================================"
    echo ">>> TIẾN TRÌNH BENCHMARK ĐÃ KẾT THÚC! <<<"
    echo ">>> BẢNG TỔNG HỢP ĐIỂM SỐ HIỆN TẠI TẠI: $OUT_DIR/vitextvqa_benchmark_summary.csv"
    echo "======================================================================"
    if [ -f "$OUT_DIR/vitextvqa_benchmark_summary.csv" ]; then
        column -s, -t < "$OUT_DIR/vitextvqa_benchmark_summary.csv" 2>/dev/null || cat "$OUT_DIR/vitextvqa_benchmark_summary.csv"
    fi
fi
