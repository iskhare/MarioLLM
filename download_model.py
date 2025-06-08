#!/usr/bin/env python3

import os
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from huggingface_hub import snapshot_download
import torch

def download_model_with_retry(model_name, max_retries=3, cache_dir=None):
    """Download model with retry logic"""
    
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries}: Downloading {model_name}")
            
            # Download with explicit cache directory
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            
            # Download model files
            print("Downloading model files...")
            snapshot_download(
                repo_id=model_name,
                cache_dir=cache_dir,
                local_files_only=False,
                resume_download=True,  # Resume if interrupted
                force_download=False   # Don't re-download if already exists
            )
            
            # Test loading tokenizer
            print("Testing tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                cache_dir=cache_dir
            )
            
            # Test loading processor
            print("Testing processor...")
            try:
                processor = AutoProcessor.from_pretrained(
                    model_name,
                    cache_dir=cache_dir
                )
                print("✓ Processor loaded successfully")
            except Exception as e:
                print(f"⚠ No processor available: {e}")
            
            # Test loading model (without putting on GPU yet)
            print("Testing model loading...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map='cpu',  # Load to CPU first
                cache_dir=cache_dir
            )
            
            print("✓ Model downloaded and tested successfully!")
            del model  # Free memory
            return True
            
        except Exception as e:
            print(f"✗ Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print("Waiting 30 seconds before retry...")
                time.sleep(30)
            else:
                print("All download attempts failed!")
                return False
    
    return False

def main():
    # Models to try (in order of preference)
    models = [
        'microsoft/Phi-3.5-vision-instruct',  # Smaller, more reliable
        'meta-llama/Llama-3.2-11B-Vision-Instruct'  # Original choice
    ]
    
    cache_dir = os.path.expanduser("~/.cache/huggingface/transformers")
    
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Trying to download: {model_name}")
        print(f"{'='*60}")
        
        success = download_model_with_retry(model_name, max_retries=3, cache_dir=cache_dir)
        
        if success:
            print(f"\n✓ Successfully downloaded {model_name}")
            print(f"You can now update config.py to use: MODEL_PATH = '{model_name}'")
            break
        else:
            print(f"\n✗ Failed to download {model_name}, trying next model...")
    
    print("\nDownload process completed!")

if __name__ == "__main__":
    main() 