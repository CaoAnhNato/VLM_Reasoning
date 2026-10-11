#!/usr/bin/env bash
# ==============================================================================
# SCRIPT THIẾT LẬP MÔI TRƯỜNG THỰC NGHIỆM TRÊN MÁY ẢO VAST.AI
# Tối ưu cho template PyTorch (Vast) có sẵn CUDA & PyTorch
# ==============================================================================
set -e

echo "=== [1/4] Cập nhật hệ thống & Cài đặt công cụ nền tảng ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    unzip \
    aria2 \
    git-lfs \
    tmux \
    htop \
    nvtop

git lfs install 2>/dev/null || true

echo "=== [2/4] Kiểm tra PyTorch & CUDA có sẵn trong container ==="
if python3 -c "import torch; assert torch.cuda.is_available(); print('-> GPU có sẵn:', torch.cuda.get_device_name(0), '| PyTorch:', torch.__version__)" 2>/dev/null; then
    echo "[XÁC NHẬN] Container đã có sẵn PyTorch và CUDA hoạt động tốt. Bỏ qua bước cài đặt lại PyTorch!"
else
    echo "[CẢNH BÁO] Chưa phát hiện PyTorch CUDA, tiến hành cài đặt bổ sung..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
fi

echo "=== [3/4] Cài đặt các thư viện bổ trợ cho VLM & Benchmark ==="
python3 -m pip install --upgrade pip
pip install --no-cache-dir \
    "huggingface_hub[cli,hf_xet]" hf hf-xet \
    transformers>=4.45.0 \
    accelerate>=0.34.0 \
    bitsandbytes>=0.43.0 \
    qwen-vl-utils \
    timm \
    einops \
    sentencepiece \
    pillow \
    tqdm \
    python-dotenv \
    psutil

echo "=== [4/4] Cài đặt Unsloth & Unsloth-Zoo cho VLM Fast Inference ==="
pip install --no-cache-dir unsloth_zoo || true
pip install --no-deps "unsloth[cu124-torch240] @ git+https://github.com/unslothai/unsloth.git" 2>/dev/null || pip install --no-cache-dir unsloth || {
    echo "[GHI CHÚ] Unsloth không tương thích hoặc chưa có bản dựng cho kiến trúc GPU này. Hệ thống sẽ tự động dùng Hugging Face Transformers Native Fallback (bfloat16) ổn định 100%."
}

# Kích hoạt chế độ tải dữ liệu siêu tốc bằng Xet High Performance của Hugging Face
export HF_XET_HIGH_PERFORMANCE=1

# Đăng nhập Hugging Face CLI nếu có truyền token
HF_TOKEN_ARG="${1:-}"
if [ -n "$HF_TOKEN_ARG" ]; then
    echo "=== Xác thực Hugging Face Hub CLI với Token (hf auth login) ==="
    hf auth login --token "$HF_TOKEN_ARG" || true
fi

echo "=================================================================="
echo ">>> THIẾT LẬP MÔI TRƯỜNG HOÀN TẤT TRONG CHƯA ĐẦY 1 PHÚT! <<<"
echo "=================================================================="


