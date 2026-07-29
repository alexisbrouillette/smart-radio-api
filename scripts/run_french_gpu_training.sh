#!/bin/bash
# French Kokoro TTS Fine-Tuning Execution Script
# Runs Stage 1, Stage 2, and Voicepack Extraction using kikiri-tts framework

set -e

echo "=== 🎙️ FRENCH KOKORO TTS FINE-TUNING PIPELINE ==="

# 1. Environment & Dependencies Check
echo "[1/4] Checking dependencies..."
if ! command -v espeak-ng &> /dev/null; then
    echo "⚠️ WARNING: espeak-ng is not installed. Run 'sudo apt-get install espeak-ng libsndfile1'"
fi

# 2. Build Monotonic Alignment Extension inside kikiri-tts/StyleTTS2
echo "[2/4] Building monotonic alignment module..."
cd kikiri-tts/StyleTTS2
if [ ! -d "monotonic_align/build" ]; then
    if [ ! -d "monotonic_align" ]; then
        git clone https://github.com/resemble-ai/monotonic_align.git
    fi
    cd monotonic_align
    python setup.py build_ext --inplace
    cd ..
fi

# 3. Stage 1 Training: Acoustic Model Fine-Tuning
echo "\n[3/4] Launching Stage 1 Fine-Tuning (Acoustic Model)..."
accelerate launch train_first.py --config_path ../configs/config_french_stanley.yml

# 4. Stage 2 Training: Prosody & Duration Fine-Tuning
echo "\n[4/4] Launching Stage 2 Fine-Tuning (Prosody & Voicepack)..."
accelerate launch train_second.py --config_path ../configs/config_french_stanley.yml

echo "\n=== 🎉 TRAINING COMPLETE! ==="
echo "To extract the Stanley French voicepack (.pt), run:"
echo "python scripts/extract_voicepack.py --model kikiri-tts/StyleTTS2/logs/kokoro-french-stanley/epoch_2nd_00009.pth --audio-dir dataset_french/wavs --output voices/stanley_french.pt"
