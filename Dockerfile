FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install required packages
RUN pip install numpy transformers datasets tiktoken wandb tqdm

# Copy the ChessGPT repository into the container
COPY . /app/

# Default command (can be overridden)
CMD ["bash"] 