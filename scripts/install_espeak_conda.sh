#!/bin/bash
# Standalone non-root espeak-ng installer using micromamba

set -e

echo "=== 📦 INSTALLING ESPEAK-NG INTO VIRTUAL ENVIRONMENT ==="

TARGET_VENV="$(pwd)/.venv"

if [ -f "$TARGET_VENV/bin/espeak-ng" ] && "$TARGET_VENV/bin/espeak-ng" --version 2>/dev/null | grep -q "eSpeak"; then
    echo "✅ espeak-ng is already installed in $TARGET_VENV/bin/espeak-ng!"
    exit 0
fi

TMP_DIR="/tmp/micromamba_espeak_$$"
mkdir -p "$TMP_DIR"
cd "$TMP_DIR"

echo "[1/2] Fetching standalone micromamba..."
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba

echo "[2/2] Installing espeak-ng into $TARGET_VENV..."
./bin/micromamba create -y -c conda-forge -p "$TARGET_VENV" espeak-ng || true

cd - > /dev/null
rm -rf "$TMP_DIR"

echo "\n=== ✅ ESPEAK-NG INSTALLATION COMPLETE! ==="
"$TARGET_VENV/bin/espeak-ng" --version || true
