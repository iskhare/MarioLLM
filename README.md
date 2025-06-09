# LLM-Based Super Mario Bros Reinforcement Learning

<div align="center">
  <img src="./image.jpg" alt="Super Mario RL" width="30%"/>
</div>

This project explores using vision-language models (VLMs) as decision-making agents for playing Super Mario Bros through reinforcement learning. We implement two distinct approaches: a PPO-trained vision-language model and a traditional Double Deep Q-Network (DDQN) for comparison.

## Project Overview

Our project implements two complementary approaches:

### 1. Vision-Language Model with PPO (`llm_mario/`)

- **Model**: Qwen2.5-VL-3B-Instruct vision-language model
- **Fine-tuning**: LoRA (Low-Rank Adaptation) with 4-bit quantization
- **Training**: Proximal Policy Optimization (PPO) with custom policy/value heads
- **Input**: Game screenshots (base64 encoded) + game state information  
- **Output**: Action selection with value estimation

The VLM agent processes both visual game screenshots and textual game state information to make informed decisions. The model is fine-tuned using LoRA adapters and trained with PPO to learn optimal Mario gameplay policies.

### 2. Traditional DDQN Baseline (`ddqn/`)

- **Architecture**: Convolutional neural network with experience replay
- **Algorithm**: Double Deep Q-Network (DDQN) 
- **Input**: Preprocessed game frames
- **Purpose**: Baseline comparison for the VLM approach

## Setup

### Prerequisites

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure CUDA is available for GPU acceleration (required for the VLM approach).

### Running the Code

#### Vision-Language Model (PPO)
```bash
cd llm_mario
python main.py --train --display --wandb  # Training mode with visualization and logging
python main.py --display  # Evaluation mode
```

#### DDQN Baseline
```bash
cd ddqn
python main.py  # Run DDQN training
```

## Implementation Details

### Vision-Language Model Features
- **Multi-modal input**: Combines visual screenshots with structured game state
- **Memory efficient**: 4-bit quantization with gradient checkpointing
- **Advanced training**: PPO with Generalized Advantage Estimation (GAE)
- **Monitoring**: Weights & Biases integration for experiment tracking

### Technical Architecture
- **Base Model**: Qwen2.5-VL-3B-Instruct (3 billion parameters)
- **Adaptation**: LoRA fine-tuning (rank 16, alpha 32)
- **Policy Network**: Custom policy/value heads on top of VLM features
- **Environment**: Super Mario Bros 1-1 with action space simplification

## Team

- Ishan Khare
- Gabriel Seir  
- Anthony Zhan

## Related Work

- [Playing Super Mario with LLMs as a Benchmark](https://www.reddit.com/r/singularity/comments/1j1pxru/playing_super_mario_with_llms_as_a_benchmark_by)
- [Claude Plays Pokemon](https://www.twitch.tv/claudeplayspokemon)
