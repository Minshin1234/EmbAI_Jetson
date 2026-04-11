#!/bin/bash

# Configuration
IMAGE_NAME="vex-jetson-app"

echo "Building the Jetson App Docker image (this will bake the dependencies like Flask/YOLO)..."
echo "Image Name: $IMAGE_NAME"

# Build the custom image using the Dockerfile
sudo docker build -t "$IMAGE_NAME" .

echo "Build complete! You can now run the app instantly using ./run_docker.sh"
