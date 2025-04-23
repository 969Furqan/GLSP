#!/bin/bash

# Create necessary directories if they don't exist
mkdir -p config checkpoints images output data logs post

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "docker-compose is not installed. Please install it first."
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "Docker is not running. Please start Docker first."
    exit 1
fi

# Check if NVIDIA Docker runtime is installed
if ! docker info | grep -q "Runtimes.*nvidia"; then
    echo "NVIDIA Docker runtime is not installed. Please install it first."
    echo "Visit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    exit 1
fi

# Check if NVIDIA drivers are installed
if ! command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA drivers are not installed or not found."
    echo "Installing without GPU support..."
    sed -i 's/runtime: nvidia/# runtime: nvidia/' docker-compose.yml
fi

# Build and run the container
docker-compose up --build 