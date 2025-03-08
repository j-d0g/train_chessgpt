#!/bin/bash

# Make script executable
# chmod +x run_docker.sh

# Build the Docker image
build_image() {
  echo "Building Docker image..."
  docker build -t chessgpt .
}

# Run an interactive shell in the container
shell() {
  echo "Starting interactive shell..."
  docker run --gpus all -it --rm \
    -v $(pwd):/app \
    chessgpt
}

# Run the training script with GPU configuration
train() {
  echo "Starting ChessGPT training on GPU..."
  docker run --gpus all -it --rm \
    -v $(pwd):/app \
    chessgpt python train.py config/train_shakespeare_char.py
}

# Run the training script with Mac MPS configuration
train_mac() {
  echo "Starting ChessGPT training for Mac..."
  docker run -it --rm \
    -v $(pwd):/app \
    chessgpt python train.py config/train_shakespeare_char_mac.py
}

# Run the sampling script
sample() {
  echo "Sampling from ChessGPT model..."
  docker run --gpus all -it --rm \
    -v $(pwd):/app \
    chessgpt python sample.py --out_dir=out-chess-gpu
}

# Prepare the dataset
prepare_data() {
  echo "Preparing the chess dataset..."
  docker run --gpus all -it --rm \
    -v $(pwd):/app \
    chessgpt python data/lichess_hf_dataset/prepare.py
}

# Process command line arguments
case "$1" in
  build)
    build_image
    ;;
  shell)
    shell
    ;;
  train)
    train
    ;;
  train_mac)
    train_mac
    ;;
  sample)
    sample
    ;;
  prepare)
    prepare_data
    ;;
  *)
    echo "Usage: $0 {build|shell|train|train_mac|sample|prepare}"
    exit 1
    ;;
esac

exit 0 