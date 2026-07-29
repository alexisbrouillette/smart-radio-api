#!/bin/bash
# Disk space cleanup script for Kokoro Fine-Tuning

echo "=== 🧹 CLEANING INTERMEDIATE CHECKPOINTS & TEMP FILES ==="

# 1. Remove intermediate epoch checkpoints (keeping first_stage.pth)
if [ -d "kikiri-tts/StyleTTS2/logs/kokoro-french-stanley" ]; then
    echo "Removing intermediate epoch files..."
    rm -f kikiri-tts/StyleTTS2/logs/kokoro-french-stanley/epoch_*.pth
fi

# 2. Remove HuggingFace Hub cache
if [ -d "$HOME/.cache/huggingface" ]; then
    echo "Removing HuggingFace download cache..."
    rm -rf "$HOME/.cache/huggingface"
fi

# 3. Remove temp PyTorch zip files
rm -rf /tmp/torch_* 2>/dev/null || true

echo "=== ✅ CLEANUP COMPLETE! ==="
df -h .
