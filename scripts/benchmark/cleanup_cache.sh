#!/usr/bin/env bash
# ==============================================================================
# SCRIPT DỌN DẸP TOÀN BỘ CACHE MÔ HÌNH HUGGING FACE ĐỂ GIẢI PHÓNG Ổ ĐĨA
# Tuyệt đối giữ nguyên Dataset (data/vitextvqa) và Kết quả (results/)
# ==============================================================================
set -e

echo "======================================================================"
echo " BẮT ĐẦU DỌN DẸP MODEL CACHE TRÊN HỆ THỐNG"
echo " Dung lượng ổ đĩa TRƯỚC khi dọn dẹp:"
df -h / | awk 'NR==1 || NR==2'
echo "======================================================================"

# 1. Danh sách các thư mục cache Hugging Face trên Vast.ai & Linux
CACHE_DIRS=(
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

DELETED_COUNT=0
for cdir in "${CACHE_DIRS[@]}"; do
    if [ -n "$cdir" ] && [ -d "$cdir" ]; then
        echo "[KIỂM TRA] Thư mục cache: $cdir"
        # Xóa các thư mục weights mô hình models--*
        find "$cdir" -maxdepth 2 -type d -name "models--*" -exec echo " -> Đang xóa:" {} \; -exec rm -rf {} + 2>/dev/null || true
        # Xóa các file tải tạm thời
        find "$cdir" -maxdepth 2 -type d -name ".tmp*" -exec rm -rf {} + 2>/dev/null || true
        find "$cdir" -maxdepth 2 -type d -name "tmp*" -exec rm -rf {} + 2>/dev/null || true
    fi
done

# 2. Xóa các module Python custom code đã tải về
MODULE_DIRS=(
    "/workspace/.hf_home/modules"
    "$HOME/.cache/huggingface/modules"
    "/root/.cache/huggingface/modules"
)
for mdir in "${MODULE_DIRS[@]}"; do
    if [ -d "$mdir" ]; then
        echo "[XÓA MODULES] Đang dọn dẹp thư mục: $mdir"
        rm -rf "$mdir"/* 2>/dev/null || true
    fi
done

# 3. Dọn dẹp file tạm trong /tmp
rm -rf /tmp/huggingface* /tmp/xet* /tmp/torch* /tmp/*models* 2>/dev/null || true

echo ""
echo "======================================================================"
echo " ĐÃ HOÀN TẤT DỌN DẸP Ổ ĐĨA!"
echo " Dung lượng ổ đĩa SAU khi dọn dẹp:"
df -h / | awk 'NR==1 || NR==2'
echo "======================================================================"
