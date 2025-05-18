# ChessGPT Docker Setup for RTX4090

This README explains how to set up and train ChessGPT using Docker on an RTX4090 GPU.

## Prerequisites

- SSH access to the workstation with RTX4090
- Docker installed on the workstation
- NVIDIA Container Toolkit (nvidia-docker2) installed

## Step-by-Step Instructions

### 1. Clone the repository (and sub-modules) (if needed)

```bash
git clone --recursive-submodules https://github.com/your-username/ChessGPT.git
cd ChessGPT/train_ChessGPT
```

### 2. Make the helper script executable

```bash
chmod +x run_docker.sh
```

### 3. Build the Docker image

```bash
./run_docker.sh build
```

### 4. Prepare the dataset

For lichess dataset:
```bash
./run_docker.sh prepare_lichess
```

For stockfish dataset:
```bash
./run_docker.sh prepare_stockfish
```

### 5. Start training

For lichess dataset:
```bash
./run_docker.sh train_lichess
```

For stockfish dataset:
```bash
./run_docker.sh train_stockfish
```

For Mac users (using MPS):
```bash
./run_docker.sh train_mac
```

You can also add additional training parameters:
```bash
./run_docker.sh train_stockfish --batch_size=64
```

### 6. Sample from the trained model

```bash
./run_docker.sh sample
```

For sampling just a single move:
```bash
./run_docker.sh sample_move
```

You can also add parameters:
```bash
./run_docker.sh sample --temperature=0.9 --max_new_tokens=100
```

### 7. Open an interactive shell (for debugging or custom commands)

```bash
./run_docker.sh shell
```

### 8. Monitoring and logging

List all running ChessGPT containers:
```bash
./run_docker.sh list
```

View logs of a specific container:
```bash
./run_docker.sh logs CONTAINER_NAME
```

Attach to a running container:
```bash
./run_docker.sh attach CONTAINER_NAME
```

### 9. Upload Trained Checkpoints to Hugging Face Hub

Upload a trained checkpoint to Hugging Face:
```bash
./run_docker.sh upload [checkpoint_path] [model_name] [options]
```

Example:
```bash
./run_docker.sh upload out-stockfish-small-24/ckpt.pt small-24 --repo my-chessgpt
```

Additional options:
```bash
./run_docker.sh upload out/my_model/ckpt.pt my_model --repo my-chess-repo --message "Upload iteration 500"
```

#### Hugging Face Credentials

The script requires Hugging Face credentials for uploading:

1. Create a Hugging Face access token at https://huggingface.co/settings/tokens
2. Add your credentials to the parent directory's `.env` file:
   ```
   HF_TOKEN=your_huggingface_access_token
   HF_USERNAME=your_huggingface_username
   ```
3. Alternatively, you can set these as environment variables before running the script

## Configuration

The configuration files are located in the `config/` directory:
- `train_lichess.py` - Configuration for training on the lichess dataset
- `train_stockfish.py` - Configuration for training on the stockfish dataset
- `train_chess_mac.py` - Configuration for training on Mac with MPS

You can modify:
- Model size parameters (`n_layer`, `n_head`, `n_embd`)
- Batch size and learning rate
- Number of training iterations

## Wandb Integration

By default, training logs to Weights & Biases. The script will:
1. Check for WANDB_API_KEY in the environment variables
2. Look for a wandb key in the parent directory's `.env` file
3. Prompt you to enter your API key if not found

To disable Wandb logging, set `wandb_log = False` in the config file.

## Troubleshooting

- **GPU not available**: Make sure the NVIDIA Container Toolkit is installed on the host
- **Out of memory errors**: Reduce batch size or model size in the config file
- **Permission errors**: Make sure your user has permissions to run Docker
- **Wandb authentication errors**: Check that your API key is correct
- **Docker build failures**: Check for network connectivity or disk space issues 