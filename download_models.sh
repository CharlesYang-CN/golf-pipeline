#!/bin/bash
# Download model weights during Docker build / CI
# Publicly available models are auto-downloaded.
# For SMPL (requires account), place files in romp_models/ or set SMPL_URL secret.
set -euo pipefail

MODEL_DIR="${1:-/tmp/models}"
mkdir -p "$MODEL_DIR/romp"

echo "=== Downloading ROMP checkpoint ==="
wget -q -O "$MODEL_DIR/romp/ROMP.pkl" \
  "https://github.com/Arthur151/ROMP/releases/download/V2.0/ROMP.pkl"

echo "=== Downloading MediaPipe Pose Landmarker ==="
wget -q -O "$MODEL_DIR/pose_landmarker.task" \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker/float16/latest/pose_landmarker.task"

echo "=== Downloading MobileNetV2 pretrained ==="
wget -q -O "$MODEL_DIR/mobilenet_v2.pth.tar" \
  "https://github.com/tonylins/pytorch-mobilenet-v2/releases/download/v1.0/mobilenet_v2.pth.tar"

echo "=== Downloading YOLO11n ==="
pip install -q ultralytics 2>/dev/null || true
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')" 2>/dev/null || \
  wget -q -O "$MODEL_DIR/yolo11n.pt" \
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"

echo "=== Models downloaded to $MODEL_DIR ==="
ls -lh "$MODEL_DIR/"
