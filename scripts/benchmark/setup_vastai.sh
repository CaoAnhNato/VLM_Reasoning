#!/usr/bin/env bash
# ==============================================================================
# SCRIPT THIẾT LẬP MÔI TRƯỜNG THỰC NGHIỆM TRÊN MÁY ẢO VAST.AI
# Tối ưu cho GPU 16GB (RTX 4090 / RTX 50-series)
# ==============================================================================
set -e

echo "=== [BƯỚC 1/5] Cập nhật hệ thống & Cài đặt công cụ nền tảng ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    unzip \
    p7zip-full \
    aria2 \
    git-lfs \
    tmux \
    htop \
    nvtop \
    build-essential

git lfs install

echo "=== [BƯỚC 2/5] Nâng cấp PIP & Cài đặt tiện ích Hugging Face ==="
python3 -m pip install --upgrade pip
pip install "huggingface_hub[cli]" python-dotenv tqdm pandas openpyxl tabulate

echo "=== [BƯỚC 3/5] Cài đặt PyTorch, Torchvision & Accelerator ==="
# Kiểm tra nếu PyTorch chưa có sẵn với CUDA
python3 -c "import torch; print('PyTorch CUDA status:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')" || {
    echo "Cài đặt PyTorch CUDA 12.4..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
}

echo "=== [BƯỚC 4/5] Cài đặt Transformers & thư viện VLM cần thiết ==="
pip install --no-cache-dir \
    transformers>=4.45.0 \
    accelerate>=0.34.0 \
    bitsandbytes>=0.43.0 \
    sentencepiece \
    timm \
    einops \
    pillow \
    qwen-vl-utils

echo "=== [BƯỚC 5/5] Cài đặt Unsloth cho VLM Fast Inference ==="
# Cài đặt Unsloth tối ưu cho GPU thế hệ mới
pip install --no-deps "unsloth[cu124-torch240] @ git+https://github.com/unslothai/unsloth.git" || {
    echo "[CẢNH BÁO] Cài đặt Unsloth thất bại hoặc không tương thích kiến trúc GPU. Sẽ sử dụng Hugging Face Transformers Native Fallback."
}

echo "=================================================================="
echo ">>> THIẾT LẬP MÔI TRƯỜNG TRÊN VAST.AI HOÀN TẤT THÀNH CÔNG! <<<"
echo "=================================================================="
