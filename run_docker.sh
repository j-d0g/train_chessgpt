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
train_lichess() {
  echo "Starting ChessGPT training on GPU with lichess dataset..."
  
  # Check if WANDB_API_KEY is set
  WANDB_KEY=${WANDB_API_KEY:-}
  
  if [ -z "$WANDB_KEY" ] && [ -f "../.env" ]; then
    # Try to source the parent .env file
    echo "Looking for wandb key in ../.env file..."
    source "../.env"
    WANDB_KEY=${wandb:-}
  fi
  
  if [ -z "$WANDB_KEY" ]; then
    echo "No WANDB_API_KEY found in environment or .env file."
    read -p "Enter your Weights & Biases API key: " WANDB_KEY
    
    if [ -z "$WANDB_KEY" ]; then
      echo "Error: wandb API key is required for logging."
      echo "You can find your API key at https://wandb.ai/settings"
      echo "Run with: export WANDB_API_KEY=your_key"
      exit 1
    fi
  fi
  
  # Run in interactive mode
  docker run --gpus all -it --rm \
    -e WANDB_API_KEY="$WANDB_KEY" \
    -v $(pwd):/app \
    chessgpt python train.py config/train_lichess.py "$@"
}

train_stockfish() {
  echo "Starting ChessGPT training on GPU with stockfish dataset..."
  
  # Check if WANDB_API_KEY is set
  WANDB_KEY=${WANDB_API_KEY:-}
  
  if [ -z "$WANDB_KEY" ] && [ -f "../.env" ]; then
    # Try to source the parent .env file
    echo "Looking for wandb key in ../.env file..."
    source "../.env"
    WANDB_KEY=${wandb:-}
  fi
  
  if [ -z "$WANDB_KEY" ]; then
    echo "No WANDB_API_KEY found in environment or .env file."
    read -p "Enter your Weights & Biases API key: " WANDB_KEY
    
    if [ -z "$WANDB_KEY" ]; then
      echo "Error: wandb API key is required for logging."
      echo "You can find your API key at https://wandb.ai/settings"
      echo "Run with: export WANDB_API_KEY=your_key"
      exit 1
    fi
  fi
  
  # Run in interactive mode
  docker run --gpus all -it --rm \
    -e WANDB_API_KEY="$WANDB_KEY" \
    -v $(pwd):/app \
    chessgpt python train.py config/train_stockfish.py "$@"
}

# Run the training script with Mac MPS configuration
train_mac() {
  echo "Starting ChessGPT training for Mac..."
  docker run -it --rm \
    -v $(pwd):/app \
    chessgpt python train.py config/train_chess_mac.py "$@"
}

# View logs of a running container
logs() {
  if [ -z "$1" ]; then
    echo "Error: Container name is required"
    echo "Usage: $0 logs CONTAINER_NAME"
    exit 1
  fi
  
  echo "Following logs for container $1..."
  docker logs -f "$1"
}

# Attach to a running container
attach() {
  if [ -z "$1" ]; then
    echo "Error: Container name is required"
    echo "Usage: $0 attach CONTAINER_NAME"
    exit 1
  fi
  
  echo "Attaching to container $1..."
  echo "Note: To detach without stopping the container, use Ctrl+P followed by Ctrl+Q"
  docker attach "$1"
}

# List all running training containers
list_training() {
  echo "Listing all running ChessGPT training containers:"
  docker ps --filter "name=chessgpt_"
}

# Run the sampling script
sample() {
  echo "Sampling from ChessGPT model..."
  docker run --gpus all -it --rm \
    -v $(pwd):/app \
    chessgpt python sample.py --start=";1." "$@"
}

sample_move() {
  echo "Sampling a move from ChessGPT model..."
  docker run --gpus all -it --rm \
    -v $(pwd):/app \
    chessgpt python sample.py --start=";1." --max_new_tokens=6 "$@"
}

# Prepare the dataset
prepare_lichess() {
  echo "Preparing the chess dataset..."
  docker run --gpus all -it --rm \
    -v $(pwd):/app \
    chessgpt python data/hf_dataset_lichess/prepare.py "$@"
}

prepare_stockfish() {
  echo "Preparing the stockfish dataset..."
  docker run --gpus all -it --rm \
    -v $(pwd):/app \
    chessgpt python data/hf_dataset_stockfish/prepare.py "$@"
}

train_distributed() {
  echo "Starting distributed ChessGPT training on multiple GPUs..."
  
  # Check if WANDB_API_KEY is set
  WANDB_KEY=${WANDB_API_KEY:-}
  
  if [ -z "$WANDB_KEY" ] && [ -f "../.env" ]; then
    # Try to source the parent .env file
    echo "Looking for wandb key in ../.env file..."
    source "../.env"
    WANDB_KEY=${wandb:-}
  fi
  
  if [ -z "$WANDB_KEY" ]; then
    echo "No WANDB_API_KEY found in environment or .env file."
    read -p "Enter your Weights & Biases API key: " WANDB_KEY
  fi
  
  # Run with torchrun
  docker run --gpus all -it --rm \
    -e WANDB_API_KEY="$WANDB_KEY" \
    -v $(pwd):/app \
    chessgpt torchrun --standalone --nproc_per_node=2 train.py config/train_stockfish.py "$@"
}

# Upload checkpoint to Hugging Face Hub
upload_checkpoint() {
  echo "Starting checkpoint upload..."
  
  # Check for HF_TOKEN and HF_USERNAME
  HF_TOKEN_VAL=${HF_TOKEN:-}
  HF_USERNAME_VAL=${HF_USERNAME:-}
  
  # Check .env file if variables are not in environment
  if [ -z "$HF_TOKEN_VAL" ] || [ -z "$HF_USERNAME_VAL" ]; then
    if [ -f "../.env" ]; then
      echo "Looking for Hugging Face credentials in ../.env file..."
      source "../.env"
      HF_TOKEN_VAL=${HF_TOKEN:-$HF_TOKEN_VAL} # Keep env var if set
      HF_USERNAME_VAL=${HF_USERNAME:-$HF_USERNAME_VAL} # Keep env var if set
    fi
  fi

  # Check if HF_TOKEN is found
  if [ -z "$HF_TOKEN_VAL" ]; then
    echo "Error: HF_TOKEN environment variable not found in environment or ../.env file."
    echo "Please set it. You can generate a token at https://huggingface.co/settings/tokens"
    exit 1
  fi

  # Prompt for username if not found (optional, but good for repo creation)
  if [ -z "$HF_USERNAME_VAL" ]; then
     echo "HF_USERNAME not found in environment or ../.env file."
     read -p "Enter your Hugging Face username (optional, press Enter to skip): " HF_USERNAME_VAL
  fi
  
  # Expecting checkpoint_path and model_name as arguments $1 and $2
  # Additional arguments like --repo and --message can be passed via "$@"
  if [ "$#" -lt 2 ]; then
      echo "Error: Missing required arguments."
      echo "Usage: $0 upload CHECKPOINT_PATH MODEL_NAME [--repo REPO_NAME] [--message COMMIT_MESSAGE]"
      exit 1
  fi
  
  # Extract mandatory args
  CHECKPOINT_PATH_ARG=$1
  MODEL_NAME_ARG=$2
  shift 2 # Remove the first two arguments

  echo "Uploading $CHECKPOINT_PATH_ARG for model $MODEL_NAME_ARG..."

  docker run --gpus all -it --rm \
    -e HF_TOKEN="$HF_TOKEN_VAL" \
    -e HF_USERNAME="$HF_USERNAME_VAL" \
    -v $(pwd):/app \
    chessgpt python upload_checkpoint.py "$CHECKPOINT_PATH_ARG" "$MODEL_NAME_ARG" "$@"
}

# Process command line arguments
# First argument is the command, all other arguments are passed to the function
cmd=$1
shift 1 # Remove the first argument (the command)

case "$cmd" in
  build)
    build_image
    ;;
  shell)
    shell
    ;;
  train_lichess)
    train_lichess "$@"
    ;;
  train_stockfish)
    train_stockfish "$@"
    ;;
  train_mac)
    train_mac "$@"
    ;;
  sample)
    sample "$@"
    ;;
  sample_move)
    sample_move "$@"
    ;;
  prepare_lichess)
    prepare_lichess "$@"
    ;;
  prepare_stockfish)
    prepare_stockfish "$@"
    ;;
  logs)
    logs "$@"
    ;;
  attach)
    attach "$@"
    ;;
  list)
    list_training
    ;;
  train_distributed)
    train_distributed "$@"
    ;;
  upload)
    upload_checkpoint "$@"
    ;;
  *)
    echo "Usage: $0 {build|shell|train_lichess|train_stockfish|train_mac|sample|sample_move|prepare_lichess|prepare_stockfish|logs|attach|list|train_distributed|upload} [additional arguments]"
    echo "Examples:"
    echo "  $0 sample --temperature=0.9 --max_new_tokens=100"
    echo "  $0 train_lichess --batch_size=64"
    echo "  $0 logs CONTAINER_NAME"
    echo "  $0 attach CONTAINER_NAME"
    echo "  $0 list"
    echo "  $0 upload out/my_model/ckpt.pt my_model --repo my-chess-repo --message 'Upload iteration 500'"
    exit 1
    ;;
esac

exit 0 