#!/bin/bash
# Stage 2 Training Script: Prosody, Duration & Sentence Cadence Fine-Tuning

set -e

echo "=== 🎙️ LAUNCHING STAGE 2 FINE-TUNING (PROSODY & RHYTHM) ==="

# 0. Auto-activate .venv if present
if [ -f ".venv/bin/activate" ]; then
    echo "[ENV] Activating virtual environment (.venv)..."
    source .venv/bin/activate
fi

# 1. Ensure config file is copied to kikiri-tts/configs/
mkdir -p kikiri-tts/configs
cp configs/config_french_stanley.yml kikiri-tts/configs/config_french_stanley.yml

# 2. Check first_stage.pth checkpoint
CHECKPOINT="kikiri-tts/StyleTTS2/logs/kokoro-french-stanley/first_stage.pth"
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Missing Stage 1 checkpoint at '$CHECKPOINT'. Make sure Stage 1 completed successfully!"
    exit 1
fi

echo "✅ Found Stage 1 checkpoint: $CHECKPOINT"

# 3. Enter StyleTTS2 directory and launch Stage 2
cd kikiri-tts/StyleTTS2

echo "[STAGE 2] Running train_second.py with Accelerate (10 Epochs)..."
accelerate launch train_second.py --config_path ../configs/config_french_stanley.yml

echo "\n=== 🎉 STAGE 2 TRAINING COMPLETE! ==="
echo "Stage 2 checkpoints saved to: kikiri-tts/StyleTTS2/logs/kokoro-french-stanley/"
