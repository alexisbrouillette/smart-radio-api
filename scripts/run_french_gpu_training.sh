#!/bin/bash
# French Kokoro TTS Fine-Tuning Execution Script
# Runs Stage 1, Stage 2, and Voicepack Extraction using kikiri-tts framework

set -e

echo "=== 🎙️ FRENCH KOKORO TTS FINE-TUNING PIPELINE ==="

# 0. Auto-activate .venv if present
if [ -f ".venv/bin/activate" ]; then
    echo "[ENV] Activating virtual environment (.venv)..."
    source .venv/bin/activate
fi

# 1. Environment & Submodule Check
echo "[1/4] Checking submodules & dependencies..."
if [ ! -d "kikiri-tts/StyleTTS2" ] || [ ! -f "kikiri-tts/StyleTTS2/train_first.py" ]; then
    echo "[SUBMODULE] Fast cloning kikiri-tts submodules (--depth 1)..."
    git submodule update --init --recursive --depth 1
fi

if ! command -v espeak-ng &> /dev/null; then
    echo "⚠️ WARNING: espeak-ng is not installed. Run 'sudo apt-get install espeak-ng libsndfile1' or 'conda install -c conda-forge espeak-ng'"
fi

# 2. Build & Install Monotonic Alignment Extension
echo "[2/4] Building and installing monotonic alignment module..."
python -c "import monotonic_align" 2>/dev/null || pip install git+https://github.com/resemble-ai/monotonic_align.git

# Copy config into kikiri-tts/configs/ if needed
mkdir -p kikiri-tts/configs
cp configs/config_french_stanley.yml kikiri-tts/configs/config_french_stanley.yml

cd kikiri-tts/StyleTTS2

# 3. Stage 1 Training: Acoustic Model Fine-Tuning
echo "\n[3/4] Launching Stage 1 Fine-Tuning (Acoustic Model)..."
accelerate launch train_first.py --config_path ../configs/config_french_stanley.yml

# 4. Stage 2 Training: Prosody & Duration Fine-Tuning
echo "\n[4/4] Launching Stage 2 Fine-Tuning (Prosody & Voicepack)..."
accelerate launch train_second.py --config_path ../configs/config_french_stanley.yml

echo "\n=== 🎉 TRAINING COMPLETE! ==="
echo "To extract the Stanley French voicepack (.pt), run:"
echo "python scripts/extract_voicepack.py --model kikiri-tts/StyleTTS2/logs/kokoro-french-stanley/epoch_2nd_00009.pth --audio-dir dataset_french/wavs --output voices/stanley_french.pt"
