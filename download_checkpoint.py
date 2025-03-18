#!/usr/bin/env python3
"""
Download ChessGPT checkpoints from Hugging Face Hub.
Usage: python download_checkpoint.py model_name output_path [repo_name]
Example: python download_checkpoint.py large-24-184K_iters downloaded_ckpt.pt jd0g/chess-gpt
"""

import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

def download_checkpoint(model_name, output_path, repo_name="jd0g/chess-gpt"):
    """
    Download a checkpoint from Hugging Face Hub
    Args:
        model_name: Name of the model to download (e.g., small-24, large-24-184K_iters)
        output_path: Path where to save the downloaded checkpoint
        repo_name: Name of the Hugging Face repository
    """
    # Get token from environment variable
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN environment variable not found. Please set it in your .env file.")
        return False
    
    # Path in repo for the checkpoint
    path_in_repo = f"checkpoints/{model_name}/ckpt.pt"
    
    print(f"Downloading checkpoint {model_name} from {repo_name}...")
    
    try:
        # Download the file from the Hugging Face Hub
        output_file = hf_hub_download(
            repo_id=repo_name,
            filename=path_in_repo,
            token=token,
            local_dir=os.path.dirname(output_path),
            local_dir_use_symlinks=False,
            force_download=True
        )
        
        # Move the file to the desired output path if needed
        if output_file != output_path:
            import shutil
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.move(output_file, output_path)
            print(f"Moved file from {output_file} to {output_path}")
        
        print(f"Checkpoint downloaded successfully to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error downloading checkpoint: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ChessGPT checkpoints from Hugging Face Hub")
    parser.add_argument("model_name", help="Name of the model to download (e.g., small-24, large-24-184K_iters)")
    parser.add_argument("output_path", help="Path where to save the downloaded checkpoint")
    parser.add_argument("--repo", default="jd0g/chess-gpt", help="Hugging Face repository name")
    
    args = parser.parse_args()
    
    # Call the download function
    success = download_checkpoint(
        args.model_name,
        args.output_path,
        args.repo
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1) 