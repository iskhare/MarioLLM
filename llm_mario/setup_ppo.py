#!/usr/bin/env python3
"""
Setup script for PPO Mario LLM training
"""

import os
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def check_cuda():
    """Check CUDA availability"""
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    else:
        print("⚠️  CUDA not available, using CPU (training will be slow)")
        return False


def download_model(model_name="microsoft/DialoGPT-medium"):
    """Download and cache a model suitable for RL training"""
    print(f"Downloading model: {model_name}")
    
    try:
        # Download tokenizer
        print("  Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Download model
        print("  Downloading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        print(f"✅ Model downloaded successfully!")
        print(f"   Model size: ~{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False


def create_directories():
    """Create necessary directories"""
    dirs = ['models', 'logs', 'screenshots']
    
    for dir_name in dirs:
        dir_path = Path(dir_name)
        dir_path.mkdir(exist_ok=True)
        print(f"✅ Created directory: {dir_name}")


def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        'torch',
        'transformers', 
        'peft',
        'gym_super_mario_bros',
        'nes_py',
        'opencv-python',
        'pillow',
        'numpy',
        'tensorboardX',
        'wandb'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'gym_super_mario_bros':
                import gym_super_mario_bros
            elif package == 'nes_py':
                import nes_py
            elif package == 'pillow':
                from PIL import Image
            elif package == 'tensorboardX':
                from tensorboardX import SummaryWriter
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    
    return True


def main():
    print("Setting up PPO Mario LLM Training Environment")
    print("=" * 50)
    
    # Check dependencies
    print("\n1. Checking dependencies...")
    if not check_dependencies():
        print("\n❌ Setup failed: missing dependencies")
        sys.exit(1)
    
    # Check CUDA
    print("\n2. Checking CUDA...")
    has_cuda = check_cuda()
    
    # Create directories
    print("\n3. Creating directories...")
    create_directories()
    
    # Download model
    print("\n4. Setting up model...")
    model_choice = input("Choose model to download:\n"
                        "1. microsoft/DialoGPT-medium (355M params, recommended)\n"
                        "2. microsoft/DialoGPT-large (774M params, better but slower)\n"
                        "3. Skip model download\n"
                        "Choice (1-3): ").strip()
    
    if model_choice == "1":
        success = download_model("microsoft/DialoGPT-medium")
    elif model_choice == "2":
        success = download_model("microsoft/DialoGPT-large")
    else:
        print("Skipping model download")
        success = True
    
    if not success:
        print("\n❌ Setup failed: could not download model")
        sys.exit(1)
    
    # Final setup
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Start training:")
    print("   python train_ppo.py --max-episodes 100 --display")
    print("\n2. Or run inference with a trained model:")
    print("   python main.py --episodes 5 --display --load-checkpoint models/ppo_mario_final")
    print("\n3. Monitor training with tensorboard:")
    print("   tensorboard --logdir logs")
    
    if has_cuda:
        print("\n💡 Pro tip: Use --use-wandb for cloud logging and experiment tracking")
    else:
        print("\n⚠️  Note: CPU training will be very slow. Consider using Google Colab or a GPU instance.")


if __name__ == "__main__":
    main() 