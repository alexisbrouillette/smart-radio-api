#!/bin/bash
# Standalone non-root espeak-ng installer using micromamba

set -e

echo "=== 📦 INSTALLING ESPEAK-NG INTO VIRTUAL ENVIRONMENT ==="

if [ -f ".venv/bin/espeak-ng" ] && .venv/bin/espeak-ng --version 2>/dev/null | grep -q "eSpeak"; then
    echo "✅ espeak-ng is already installed in .venv/bin/espeak-ng!"
    exit 0
fi

mkdir -p /tmp/micromamba_setup
cd /tmp/micromamba_setup

echo "[1/2] Fetching standalone micromamba..."
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba

echo "[2/2] Installing espeak-ng into project .venv..."
./bin/micromamba install -y -c conda-forge espeak-ng -p "$PWD/../../.venv" || true

cd - > /dev/null
rm -rf /tmp/micromamba_setup

echo "\n=== ✅ ESPEAK-NG INSTALLATION COMPLETE! ==="
.venv/bin/espeak-ng --version || true
