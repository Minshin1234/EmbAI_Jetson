#!/bin/bash

# Configuration
CONTAINER_IMAGE="vex-jetson-app"
PROJECT_DIR="/home/minchan/vex_jetson"

echo "Launching YOLO Camera Stream from local image..."
echo "Image: $CONTAINER_IMAGE"

# Run the container
# --privileged: Full device access (USB webcam, serial ports, etc.)
# --runtime nvidia: Enables GPU access for YOLO inference
# --network host: Shares host network for easy Flask access
sudo docker run -it --rm \
    --privileged \
    --runtime nvidia \
    --network host \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e OPENBLAS_CORETYPE=ARMV8 \
    -e OMP_NUM_THREADS=1 \
    -v "$PROJECT_DIR":/app \
    -v "$HOME/.cache/ultralytics":/root/.cache/Ultralytics \
    -v /tmp/argus_socket:/tmp/argus_socket \
    -v /usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra \
    -w /app \
    "$CONTAINER_IMAGE" \
    python3 stream.py
