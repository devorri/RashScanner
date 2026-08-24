#!/bin/bash
# ==============================================================================
# setup_env.sh - Complete Environment Setup for Rashilience Raspberry Pi Kiosk
# ==============================================================================

set -e

echo "======================================================="
echo "   Setting up Rashilience AI Environment for Pi 4      "
echo "======================================================="

# Navigate to project directory
cd "$(dirname "$0")"

# 1. Update and install required system packages
echo ""
echo "[1/4] Installing system dependencies via apt..."
sudo apt update
sudo apt install -y \
  python3-pip \
  python3-venv \
  python3-dev \
  python3-picamera2 \
  python3-opencv \
  libatlas-base-dev \
  libopenblas-dev \
  libglib2.0-0 \
  chromium \
  curl \
  pax-utils

# 2. Create Python virtual environment with system site-packages
# (--system-site-packages allows the venv to access native picamera2 camera drivers)
echo ""
echo "[2/4] Creating virtual environment (venv)..."
if [ ! -d "venv" ]; then
  python3 -m venv --system-site-packages venv
  echo "Virtual environment created."
else
  echo "Existing virtual environment found."
fi

# 3. Activate venv & upgrade pip
echo ""
echo "[3/4] Installing Python packages in venv..."
source venv/bin/activate
pip install --upgrade pip setuptools wheel

# Install required Python packages
pip install Flask>=2.3.0 "numpy>=1.24.0,<2.0.0" opencv-python-headless Pillow

# Install TFLite Runtime (tries ai-edge-litert first, then tflite-runtime, then tensorflow)
echo "Installing TFLite runtime backend..."
pip install ai-edge-litert || pip install tflite-runtime || pip install tensorflow || true

# 4. Verification Check
echo ""
echo "[4/4] Verifying installation and AI model loading..."
python3 -c "
import flask
import cv2
import numpy as np
import app
print('[SUCCESS] All Python libraries and TFLite model loaded successfully!')
"

# Ensure execution permissions
chmod +x start_kiosk.sh setup_env.sh 2>/dev/null || true

echo ""
echo "======================================================="
echo "   Setup Complete! To launch the kiosk run:           "
echo "   ./start_kiosk.sh                                    "
echo "======================================================="
