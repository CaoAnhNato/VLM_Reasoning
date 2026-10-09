#!/usr/bin/env bash
# ==============================================================================
# SCRIPT TẢI BỘ DỮ LIỆU ViTextVQA TỪ HUGGING FACE HUB
# Repo: minhquan6203/ViTextVQA (Gated Dataset)
# ==============================================================================
set -e

HF_TOKEN="${1:-$HF_TOKEN}"
REPO_ID="minhquan6203/ViTextVQA"
OUTPUT_DIR="data/vitextvqa"

if [ -z "$HF_TOKEN" ]; then
    echo "======================================================================"
    echo "[LỖI] Chưa có Hugging Face Token!"
    echo "Cách dùng 1: export HF_TOKEN=\"hf_your_token_here\" && ./download_dataset.sh"
    echo "Cách dùng 2: ./download_dataset.sh hf_your_token_here"
    echo "======================================================================"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "======================================================================"
echo ">>> ĐANG TẢI BỘ DỮ LIỆU: $REPO_ID"
echo ">>> ĐÍCH LƯU TRỮ: $OUTPUT_DIR"
echo "======================================================================"

huggingface-cli download "$REPO_ID" \
    --repo-type dataset \
    --local-dir "$OUTPUT_DIR" \
    --token "$HF_TOKEN" \
    --quiet=False

echo "======================================================================"
echo ">>> TẢI DỮ LIỆU HOÀN TẤT THÀNH CÔNG!"
echo ">>> DANH SÁCH TỆP TRONG $OUTPUT_DIR:"
echo "======================================================================"
ls -lh "$OUTPUT_DIR"
