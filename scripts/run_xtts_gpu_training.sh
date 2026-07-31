#!/bin/bash
# Shell script to run XTTS-v2 dataset preparation and GPU fine-tuning for Stanley

set -e

echo "=== 🎙️ STANLEY XTTS-v2 FINE-TUNING PIPELINE ==="

# Auto-detect Coqui environment or python
if [ -f "/home/alexis/Documents/projets/coqui-ai-TTS/venv/bin/python" ]; then
    PYTHON_BIN="/home/alexis/Documents/projets/coqui-ai-TTS/venv/bin/python"
elif [ -f ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
else
    PYTHON_BIN="python"
fi

echo "[1/2] Preparing XTTS CSV metadata..."
$PYTHON_BIN scripts/prepare_xtts_dataset.py --audio-dir dataset_french/wavs --output-dir dataset_french

echo "[2/2] Launching XTTS-v2 Fine-Tuning..."
$PYTHON_BIN scripts/train_xtts_stanley.py \
  --train-csv dataset_french/metadata_train.csv \
  --eval-csv dataset_french/metadata_val.csv \
  --output-dir functions/xtts_fine-tuned \
  --epochs 10 \
  --batch-size 2 \
  --language fr

echo "=== 🎉 TRAINING PIPELINE READY! Checkpoints will be in functions/xtts_fine-tuned/ ==="
