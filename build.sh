#!/bin/bash
# Prepare Docker build context for golf-pipeline.
#
# Usage:
#   bash build.sh           # Prepare for local docker build
#   bash build.sh --github  # Prepare for GitHub repo (slimmed)
set -euo pipefail

BUILD_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="/workspace"
MODE="${1:-local}"

echo "=== Preparing build context (mode: $MODE) ==="

need_repos() {
    # Copy source repos into build context (for local builds from /workspace)
    rm -rf "$BUILD_DIR/angle" "$BUILD_DIR/romp" "$BUILD_DIR/swing" "$BUILD_DIR/romp_models"

    echo "[1/5] Copying angle repo..."
    cp -r "$WORKSPACE/angle" "$BUILD_DIR/angle"
    find "$BUILD_DIR/angle" -name "*.mp4" -delete

    echo "[2/5] Copying ROMP inference code..."
    cp -r "$WORKSPACE/mesh/simple_romp" "$BUILD_DIR/romp"
    find "$BUILD_DIR/romp" -name "*.mp4" -delete 2>/dev/null || true
    find "$BUILD_DIR/romp" -name "*.ipynb_checkpoints" -type d -exec rm -rf {} + 2>/dev/null || true

    echo "[3/5] Copying Swing-Phase-Detection..."
    mkdir -p "$BUILD_DIR/swing/models"
    cp "$WORKSPACE/Swing-Phase-Detection/model.py" "$BUILD_DIR/swing/"
    cp "$WORKSPACE/Swing-Phase-Detection/MobileNetV2.py" "$BUILD_DIR/swing/"
    cp "$WORKSPACE/Swing-Phase-Detection/mobilenet_v2.pth.tar" "$BUILD_DIR/swing/"
    cp "$WORKSPACE/Swing-Phase-Detection/yolo11n.pt" "$BUILD_DIR/swing/"
    cp "$WORKSPACE/Swing-Phase-Detection/models/swingnet_2000.pth.tar" "$BUILD_DIR/swing/models/"

    echo "[4/5] Copying ROMP model weights..."
    mkdir -p "$BUILD_DIR/romp_models"
    for f in ROMP.pkl SMPL_NEUTRAL.pth SMPL_NEUTRAL.pkl \
             J_regressor_extra.npy J_regressor_h36m.npy smpl_kid_template.npy; do
        if [ -f "/root/.romp/$f" ]; then
            cp -v "/root/.romp/$f" "$BUILD_DIR/romp_models/"
        else
            echo "  WARNING: /root/.romp/$f not found"
        fi
    done

    echo "[5/5] Cleanup..."
    echo "Build context ready."
    du -sh "$BUILD_DIR"
}

if [ "$MODE" = "local" ]; then
    need_repos
elif [ "$MODE" = "--github" ]; then
    need_repos
    echo ""
    echo "=== GitHub-specific notes ==="
    echo "Make sure Git LFS is tracking model files:"
    echo "  git lfs track '*.pkl' '*.pth' '*.pth.tar' '*.pt' '*.npy' '*.task'"
else
    echo "Unknown mode: $MODE (use 'local' or '--github')"
    exit 1
fi

echo ""
echo "To build Docker image:"
echo "  docker build -t golf-pipeline:latest ."
echo ""
echo "For RunPod deployment:"
echo "  docker tag golf-pipeline:latest <registry>/golf-pipeline:latest"
echo "  docker push <registry>/golf-pipeline:latest"
