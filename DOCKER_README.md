# ChessGPT Docker Setup for RTX4090

This README explains how to set up and train ChessGPT using Docker on an RTX4090 GPU.

## Prerequisites

- SSH access to the workstation with RTX4090
- Docker installed on the workstation
- NVIDIA Container Toolkit (nvidia-docker2) installed

## Step-by-Step Instructions

### 1. Clone the repository (if needed)

```bash
git clone https://github.com/your-username/ChessGPT.git
cd ChessGPT
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

```bash
./run_docker.sh prepare
```

### 5. Start training

```bash
./run_docker.sh train
```

### 6. Sample from the trained model

```bash
./run_docker.sh sample
```

### 7. Open an interactive shell (for debugging or custom commands)

```bash
./run_docker.sh shell
```

## Configuration

The RTX4090 configuration is in `config/train_chess_gpu.py`. You can modify:

- Model size parameters (`n_layer`, `n_head`, `n_embd`)
- Batch size and learning rate
- Number of training iterations

## Wandb Integration

By default, training logs to Weights & Biases. To enable this:

1. Inside the Docker container shell, run:
   ```bash
   wandb login
   ```
2. Enter your API key when prompted

To disable Wandb logging, set `wandb_log = False` in the config file.

## Troubleshooting

- **GPU not available**: Make sure the NVIDIA Container Toolkit is installed on the host
- **Out of memory errors**: Reduce batch size or model size in the config file
- **Permission errors**: Make sure your user has permissions to run Docker 