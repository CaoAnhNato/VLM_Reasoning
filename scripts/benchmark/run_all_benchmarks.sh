#!/usr/bin/env bash
# ==============================================================================
# SCRIPT ĐIỀU PHỐI BENCHMARK 7 MÔ HÌNH VLM GỐC TRÊN ViTextVQA (400 VERIFIED IMAGES)
# Tối ưu cho Google Colab NVIDIA L4 GPU (24GB VRAM):
#   - Tải model qua Hugging Face API Key (--hf-token / $HF_TOKEN) + Xet High Performance
#   - Cache toàn bộ 400 ảnh vào RAM (/dev/shm + 5-thread RGB In-Memory Cache)
#   - Chạy song song 5 process (--num-workers 5) kèm hiển thị thanh tiến trình tqdm
#   - Xuất đầy đủ logs, predictions (.jsonl/.csv), summary (.csv/.json/.md) và file nén .zip
# ==============================================================================
set -eo pipefail

HF_TOKEN="${HF_TOKEN:-}"
NUM_WORKERS=5
WORKER_MODE="auto"
MODELS="all"
OUTPUT_DIR="results/colab_l4_benchmark"
CLEANUP_CACHE="--cleanup-cache"
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --token|--hf_token|--hf-token)
            HF_TOKEN="$2"
            shift 2
            ;;
        --num-workers|--num_workers|--workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --worker-mode|--worker_mode)
            WORKER_MODE="$2"
            shift 2
            ;;
        --models|--model-name|--only)
            MODELS="$2"
            shift 2
            ;;
        --output-dir|--output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --no-cleanup)
            CLEANUP_CACHE=""
            shift 1
            ;;
        --force|--overwrite)
            EXTRA_ARGS="$EXTRA_ARGS --overwrite"
            shift 1
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift 1
            ;;
    esac
done

export HF_XET_HIGH_PERFORMANCE=1
if [ -n "$HF_TOKEN" ]; then
    export HF_TOKEN="$HF_TOKEN"
fi

PYTHON_CMD="python3"
if [ -f "./venv/Scripts/python.exe" ]; then
    PYTHON_CMD="./venv/Scripts/python.exe"
elif [ -f "./venv/bin/python" ]; then
    PYTHON_CMD="./venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

exec "$PYTHON_CMD" scripts/benchmark/run_colab_l4_all_models.py \
    --models "$MODELS" \
    --num-workers "$NUM_WORKERS" \
    --worker-mode "$WORKER_MODE" \
    --output-dir "$OUTPUT_DIR" \
    $CLEANUP_CACHE \
    ${HF_TOKEN:+--hf-token "$HF_TOKEN"} \
    $EXTRA_ARGS
