#!/bin/bash
# GPU Environment Setup Script for French Kokoro TTS Fine-Tuning

set -e

echo "=== 🚀 ALL-IN-ONE GPU ENVIRONMENT SETUP ==="

# 1. System Packages
echo "[1/4] Installing system dependencies (espeak-ng, libsndfile1, ffmpeg, git)..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update && sudo apt-get install -y espeak-ng libsndfile1 ffmpeg git python3-pip python3-venv
fi

# 2. Virtual Environment
echo "[2/4] Creating & activating Python virtual environment (.venv)..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip setuptools wheel

# 3. PyTorch CUDA & Fine-Tuning Dependencies
echo "[3/4] Installing all GPU fine-tuning requirements..."
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_gpu.txt

# 4. NLTK Data Pre-download
echo "[4/4] Pre-downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')" 2>/dev/null || true

echo "\n=== ✅ GPU ENVIRONMENT SETUP COMPLETE! ==="
echo "You can now run:"
echo "1. python scripts/prepare_kokoro_weights.py"
echo "2. ./scripts/run_french_gpu_training.sh"
