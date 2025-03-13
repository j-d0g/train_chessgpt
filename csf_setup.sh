#!/bin/bash

# CSF Setup Script for ChessGPT Training
# This script automates the setup process for training ChessGPT on CSF

echo "========== ChessGPT CSF Setup =========="
echo "Setting up environment for ChessGPT training on CSF..."

# Check if we're on a GPU node
if [ -z "$(nvidia-smi 2>/dev/null)" ]; then
    echo "ERROR: No GPU detected. Please run this script on a GPU node."
    echo "Request a GPU node with: qrsh -l v100=1"
    exit 1
fi

echo "✓ GPU detected"

# Load required modules
echo "Loading CUDA and NCCL modules..."
module load libs/cuda/12.2.2
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to load CUDA module. Please check module availability."
    exit 1
fi

module load libs/nccl/2.20.3
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to load NCCL module. Please check module availability."
    exit 1
fi

echo "✓ Modules loaded successfully"

# Install required Python packages
echo "Installing required Python packages..."
pip install sympy
if [ $? -ne 0 ]; then
    echo "WARNING: Failed to install sympy. Continuing anyway..."
fi

# Install compatible versions of pydantic and wandb
echo "Installing compatible versions of pydantic and wandb..."
pip uninstall -y pydantic wandb
pip install pydantic==1.10.8 wandb==0.15.5
if [ $? -ne 0 ]; then
    echo "WARNING: Failed to install pydantic and wandb. This may cause issues with logging."
fi

echo "✓ Python packages installed"

# Verify PyTorch CUDA access
echo "Verifying PyTorch CUDA access..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
if [ $? -ne 0 ]; then
    echo "WARNING: Failed to verify PyTorch CUDA access. Training may not use GPU."
fi

# Performance optimization options
echo ""
echo "========== Performance Optimization =========="
echo "Would you like to optimize for performance? (y/n)"
read -p "> " optimize

if [[ $optimize == "y" || $optimize == "Y" ]]; then
    # Create optimized config
    echo "Creating optimized configuration..."
    cat > config/train_stockfish_small_8_optimized.py << EOL
# ChessGPT-2 25M - Optimized for V100 Performance
out_dir = "out-stockfish_small_8_optimized"
eval_interval = 4000
eval_iters = 100
log_interval = 10

always_save_checkpoint = True

wandb_log = True
wandb_project = "chess-gpt"
dataset = "hf_dataset_stockfish"
wandb_run_name = "stockfish-small-8-optimized"

# Performance optimizations
gradient_accumulation_steps = 1
batch_size = 100
block_size = 1023  # context of up to 1023 tokens (because dataset block size is 1024)

# baby GPT model :)
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.0

learning_rate = 3e-4
max_iters = 600000
lr_decay_iters = max_iters  # make equal to max_iters usually
min_lr = 3e-5  # learning_rate / 10 usually
beta2 = 0.95  # make a bit bigger because number of tokens per iter is small

warmup_iters = 500  # not super necessary potentially
compile = True  # Enable PyTorch compilation for better performance
EOL

    echo "✓ Created optimized configuration: config/train_stockfish_small_8_optimized.py"
    
    # Ask about data optimization
    echo "Would you like to optimize data access by copying to local storage? (y/n)"
    echo "Note: This may take some time depending on data size."
    read -p "> " optimize_data
    
    if [[ $optimize_data == "y" || $optimize_data == "Y" ]]; then
        echo "Optimizing data access..."
        mkdir -p $TMPDIR/chess_data
        cp -r ./data/* $TMPDIR/chess_data/
        mv ./data ./data_original
        ln -s $TMPDIR/chess_data ./data
        echo "✓ Data optimization complete"
        echo "NOTE: When finished, run: mv ./data_original ./data"
    fi
fi

echo ""
echo "========== Setup Complete =========="
echo "You can now run training with:"
if [[ $optimize == "y" || $optimize == "Y" ]]; then
    echo "python train.py config/train_stockfish_small_8_optimized.py"
else
    echo "python train.py config/train_stockfish_small_8.py"
fi
echo ""
echo "Happy training!" 