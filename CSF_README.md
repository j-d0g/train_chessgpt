# ChessGPT Training on CSF (Computational Shared Facility)

This document provides a step-by-step guide for setting up and running ChessGPT training on the University's Computational Shared Facility (CSF) with NVIDIA V100 GPUs.

## Quick Start

```bash
# 1. Request a GPU node
qrsh -l v100=1

# 2. Run the setup script
./csf_setup.sh

# 3. Run training
python train.py config/train_stockfish_small_8.py
```

## Detailed Setup Process

### 1. Requesting GPU Resources

```bash
# Request a single V100 GPU
qrsh -l v100=1

# Alternative: Request a V100 GPU with specific CPU cores
qrsh -l v100=1 -pe smp.pe 8 -cwd bash
```

**Note**: Requesting fewer CPU cores may result in better GPU performance in some cases.

### 2. Loading Required Modules

```bash
# Load CUDA and NCCL modules
module load libs/cuda/12.2.2
module load libs/nccl/2.20.3
```

### 3. Installing Required Python Packages

```bash
# Install required packages
pip install sympy
pip install pydantic==1.10.8 wandb==0.15.5
```

**Important**: The specific versions of pydantic and wandb are critical for compatibility.

### 4. Running the Training Script

```bash
# Navigate to the training directory
cd /path/to/train_ChessGPT

# Run the training script
python train.py config/train_stockfish_small_8.py
```

## Performance Optimization

For better performance, consider these optimizations:

1. **Adjust batch size and gradient accumulation**:
   - Increase batch size (e.g., 64-100)
   - Reduce gradient accumulation steps (e.g., 1-2)

2. **Move data to local storage**:
   ```bash
   # Run the data optimization script
   ./optimize_data_access.sh
   ```

3. **Enable PyTorch compilation**:
   - Set `compile = True` in your config file

## Troubleshooting

### NCCL Library Issues

If you encounter `ImportError: libnccl.so.2: cannot open shared object file`:
- Ensure you've loaded the correct CUDA and NCCL modules
- Check that PyTorch CUDA version matches the loaded CUDA module

### wandb Compatibility Issues

If you encounter `ImportError: cannot import name 'IncEx' from 'pydantic.main'`:
- Install compatible versions: `pip install pydantic==1.10.8 wandb==0.15.5`

### Performance Issues

If training is slow (high ms/iter or low MFU):
- Try requesting fewer CPU cores
- Increase batch size
- Move data to local storage
- Enable PyTorch compilation

## Environment Information

- CUDA: 12.2.2
- NCCL: 2.20.3
- PyTorch: 2.2.x with CUDA support
- GPU: NVIDIA Tesla V100-SXM2-16GB 