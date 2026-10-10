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

export HF_TOKEN="$HF_TOKEN"
download_success=false

# Phương thức 1: Sử dụng Hugging Face Hub CLI hiện đại (hf download)
if command -v hf &> /dev/null; then
    echo "[PHƯƠNG THỨC 1] Sử dụng Hugging Face CLI mới nhất: 'hf download'..."
    if hf download "$REPO_ID" \
        --repo-type dataset \
        --local-dir "$OUTPUT_DIR" \
        --token "$HF_TOKEN"; then
        download_success=true
    else
        echo "[CẢNH BÁO] 'hf download' gặp sự cố, tự động chuyển sang phương thức dự phòng..."
    fi
fi

# Phương thức 2: Dự phòng qua Python API huggingface_hub snapshot_download (luôn hoạt động 100%)
if [ "$download_success" = false ]; then
    echo "[PHƯƠNG THỨC 2] Sử dụng Python API (huggingface_hub.snapshot_download)..."
    python3 -c "
import os, sys
from huggingface_hub import snapshot_download

repo_id = '$REPO_ID'
local_dir = '$OUTPUT_DIR'
token = '$HF_TOKEN'

print(f'>>> Đang tải dataset {repo_id} về thư mục {local_dir}...')
snapshot_download(
    repo_id=repo_id,
    repo_type='dataset',
    local_dir=local_dir,
    token=token
)
print('>>> Tải dữ liệu qua Python API thành công!')
"
fi

# ==============================================================================
# TỰ ĐỘNG GIẢI NÉN FILE ẢNH (.ZIP) VÀ GIẢI PHÓNG BỘ NHỚ ĐỆM Ổ CỨNG
# ==============================================================================
for zip_file in "$OUTPUT_DIR"/*.zip; do
    if [ -f "$zip_file" ]; then
        echo ""
        echo "======================================================================"
        echo ">>> PHÁT HIỆN TỆP NÉN ẢNH: $(basename "$zip_file")"
        echo ">>> ĐANG TIẾN HÀNH GIẢI NÉN VÀO: $OUTPUT_DIR..."
        echo "======================================================================"
        if command -v unzip &> /dev/null; then
            unzip -q -o "$zip_file" -d "$OUTPUT_DIR"
        else
            python3 -c "import zipfile; zipfile.ZipFile('$zip_file').extractall('$OUTPUT_DIR')"
        fi
        echo "[THÀNH CÔNG] Đã giải nén hoàn tất $(basename "$zip_file")!"

        echo ">>> Đang xóa tệp nén $(basename "$zip_file") để giải phóng 5.3GB bộ nhớ SSD..."
        rm -f "$zip_file"
        echo "[DỌN DẸP] Đã giải phóng dung lượng ổ đĩa thành công!"
    fi
done

echo "======================================================================"
echo ">>> DỮ LIỆU ĐÃ SẴN SÀNG ĐỂ BENCHMARK!"
echo ">>> DANH SÁCH TỆP TRONG $OUTPUT_DIR:"
echo "======================================================================"
ls -lh "$OUTPUT_DIR"
