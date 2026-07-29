#!/bin/bash
# GPU Environment Setup Script for French Kokoro TTS Fine-Tuning

set -e

echo "=== 🚀 SETTING UP GPU ENVIRONMENT FOR KOKORO FINE-TUNING ==="

# 1. System Packages
echo "[1/4] Installing system dependencies (espeak-ng, libsndfile1, ffmpeg)..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update && sudo apt-get install -y espeak-ng libsndfile1 ffmpeg git
fi

# 2. Virtual Environment
echo "[2/4] Creating Python virtual environment (.venv)..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip setuptools wheel

# 3. PyTorch CUDA Installation
echo "[3/4] Installing PyTorch & Torchaudio with CUDA support..."
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Fine-Tuning & Kokoro Dependencies
echo "[4/4] Installing fine-tuning packages (accelerate, misaki, whisper, StyleTTS2 requirements)..."
pip install accelerate transformers librosa soundfile pyyaml tensorboard munch phonemizer misaki openai-whisper pydub scipy static_ffmpeg Cython einops numba

echo "\n=== ✅ GPU ENVIRONMENT SETUP COMPLETE! ==="
echo "Next steps:"
echo "1. Convert base weights: python scripts/prepare_kokoro_weights.py"
echo "2. Launch GPU fine-tuning: ./scripts/run_french_gpu_training.sh"
