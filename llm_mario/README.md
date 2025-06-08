# LLM Mario - PPO with LoRA Finetuning

Train a Large Language Model to play Super Mario Bros using Proximal Policy Optimization (PPO) with LoRA (Low-Rank Adaptation) finetuning.

## Overview

This project combines:
- **Large Language Models** for understanding game state and making decisions
- **Computer Vision** for processing game screenshots
- **PPO (Proximal Policy Optimization)** for on-policy reinforcement learning
- **LoRA Finetuning** for efficient model adaptation
- **Value Function Estimation** for improved learning

## Features

- 🎮 **Mario Environment**: Super Mario Bros 1-1 level
- 🧠 **LLM + Vision**: Multimodal input processing
- 🎯 **PPO Training**: On-policy RL with value head
- 🔧 **LoRA Finetuning**: Memory-efficient adaptation
- 📊 **Logging**: TensorBoard and Weights & Biases support
- 💾 **Checkpointing**: Save and resume training
- 🖥️ **GPU Support**: CUDA acceleration

## Quick Start

### 1. Setup Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Run setup script
python setup_ppo.py
```

### 2. Start Training
```bash
# Train with display
python train_ppo.py --max-episodes 100 --display

# Train with logging
python train_ppo.py --max-episodes 1000 --use-wandb

# Train with custom model
python train_ppo.py --model-path "gpt2-medium" --max-episodes 500
```

### 3. Run Inference
```bash
# Run with trained model
python main.py --episodes 5 --display --load-checkpoint models/ppo_mario_final

# Evaluate performance
python main.py --episodes 20 --load-checkpoint models/ppo_mario_ep_500
```

## Configuration

Key parameters in `config.py`:

### Model Settings
- `LOCAL_MODEL_PATH`: Base model to use (default: "microsoft/DialoGPT-medium")
- `DEVICE`: Training device (auto-detected)
- `MAX_LENGTH`: Maximum sequence length

### LoRA Settings
- `LORA_R`: Rank of adaptation (default: 16)
- `LORA_ALPHA`: Scaling parameter (default: 32)
- `LORA_DROPOUT`: Dropout rate (default: 0.1)

### PPO Settings
- `PPO_LEARNING_RATE`: Learning rate (default: 1e-5)
- `PPO_BATCH_SIZE`: Batch size (default: 64)
- `PPO_CLIP_COEF`: Clipping coefficient (default: 0.2)
- `PPO_GAMMA`: Discount factor (default: 0.99)

## Architecture

### PPO Agent
```
LLM Backbone (with LoRA)
    ↓
Hidden States
    ↓
Vision Encoder ← Game Screenshot
    ↓
Feature Fusion
    ↓
Policy Head → Action Probabilities
Value Head → State Value
```

### Training Loop
1. **Collect Experience**: Agent plays Mario, stores (state, action, reward, value)
2. **Calculate Advantages**: Use GAE (Generalized Advantage Estimation)
3. **PPO Update**: Update policy and value networks
4. **Repeat**: Continue until convergence

## Action Space

The agent can choose from 7 actions:
- `0`: NOOP (do nothing)
- `1`: RIGHT (move right)
- `2`: RIGHT + JUMP (move right and jump)
- `3`: RIGHT + RUN (move right and run)
- `4`: RIGHT + RUN + JUMP (move right, run and jump)
- `5`: JUMP (jump in place)
- `6`: LEFT (move left)

## Monitoring

### TensorBoard
```bash
tensorboard --logdir logs
```

### Weights & Biases
```bash
python train_ppo.py --use-wandb
```

## Model Checkpoints

Models are saved in `models/` directory:
- `ppo_mario_ep_N/`: Checkpoint after N episodes
- `ppo_mario_final/`: Final trained model

## Hardware Requirements

### Minimum
- CPU: Multi-core processor
- RAM: 8GB
- Storage: 5GB free space

### Recommended
- GPU: NVIDIA RTX 3060 or better
- RAM: 16GB
- Storage: 10GB free space

## Training Tips

1. **Start Small**: Begin with shorter episodes to debug
2. **Monitor Progress**: Use TensorBoard to track learning
3. **Adjust Hyperparameters**: Tune learning rate and batch size
4. **Use GPU**: Training on CPU is very slow
5. **Save Frequently**: Use checkpointing to resume training

## Troubleshooting

### Common Issues

**Out of Memory Error**
```bash
# Reduce batch size
PPO_BATCH_SIZE = 32  # in config.py
```

**Slow Training**
```bash
# Check GPU usage
nvidia-smi
```

**Model Not Learning**
```bash
# Reduce learning rate
PPO_LEARNING_RATE = 5e-6  # in config.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Acknowledgments

- OpenAI for the Gym environment
- Hugging Face for the Transformers library
- Microsoft for the LoRA implementation
- The Mario AI community for inspiration 