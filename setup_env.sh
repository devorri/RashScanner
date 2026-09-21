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
  v4l-utils \
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

# Install ALL required Python packages from requirements.txt
pip install -r requirements.txt

# Also install pyOpenSSL for optional adhoc HTTPS support
pip install pyOpenSSL || true

# Verify OpenCV import; only install PyPI opencv-python if system python3-opencv was missing
python3 -c "import cv2; print('OpenCV OK:', cv2.__version__)" 2>/dev/null || pip install opencv-python-headless

# Install TFLite Runtime as additional fallback backend
echo "Installing TFLite runtime backend..."
pip install ai-edge-litert || pip install tflite-runtime || true

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

# 5. Install Desktop Launcher Icon & Permissions
echo ""
echo "[5/5] Installing Rashilience Desktop Launcher Icon..."
python3 create_shortcuts.py || true

# Ensure execution permissions
chmod +x start_kiosk.sh setup_env.sh *.desktop 2>/dev/null || true

echo ""
echo "======================================================="
echo "   Setup Complete! Single Desktop Icon Installed!     "
echo "   Double-click 'Rashilience Scanner' on Desktop       "
echo "   Select Model Mode directly on the Landing Screen   "
echo "======================================================="
