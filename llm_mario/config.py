import torch
from transformers import BitsAndBytesConfig

# Model Configuration
MODEL_PATH = 'Qwen/Qwen2.5-VL-3B-Instruct'
if not torch.cuda.is_available():
    raise ValueError("CUDA not detected, exiting...")
DEVICE = 'cuda'
MAX_LENGTH = 8192

# LoRA Configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# PPO Configuration
PPO_LEARNING_RATE = 3e-5
PPO_EPOCHS = 4
PPO_BATCH_SIZE = 128
PPO_MINIBATCH_SIZE = 32
PPO_CLIP_COEF = 0.2
PPO_GAE_LAMBDA = 0.95
PPO_GAMMA = 0.99
PPO_VALUE_COEF = 0.1
PPO_ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 1.0

# Environment Configuration
ENV_NAME = 'SuperMarioBros-1-1-v3'
RENDER_MODE = 'human'

# Game Settings
EPISODES = 10000
MAX_STEPS_PER_EPISODE = 1200
FRAME_SKIP = 4
IMAGE_SIZE = (84, 84)

# Agent Configuration
MAX_MEMORY_ITEMS = 512
SCREENSHOT_HISTORY = 10
INCLUDE_GAME_STATE = True
REPLAY_BUFFER_SIZE = 10000

# Action space mapping for SIMPLE_MOVEMENT
ACTION_MAPPING = {
    0: 'NOOP',
    1: 'right', 
    2: 'right + A (jump)',
    3: 'right + B (run)',
    4: 'right + A + B (run and jump)',
    5: 'A (jump)',
    6: 'left'
}

# Training Configuration
TRAIN_EVERY = 3
SAVE_MODEL_EVERY = 50

# Logging
VERBOSE = True
SCREENSHOT_DIR = 'screenshots'
SCREENSHOT_EVERY = 100
SAVE_DIR = 'checkpoints'
LOG_DIR = 'logs'

WANDB_PROJECT = 'cs224r-project'
WANDB_RUN_NAME = 'llama-ppo'

REPORT_EVERY = 50

QUANT_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_storage=torch.bfloat16,
)