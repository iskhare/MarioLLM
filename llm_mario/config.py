import os
import torch

# Model Configuration
LOCAL_MODEL_PATH = os.getenv('LOCAL_MODEL_PATH', 'microsoft/DialoGPT-medium')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_LENGTH = 512

# LoRA Configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# PPO Configuration
PPO_LEARNING_RATE = 1e-5
PPO_EPOCHS = 4
PPO_BATCH_SIZE = 64
PPO_MINIBATCH_SIZE = 16
PPO_CLIP_COEF = 0.2
PPO_GAE_LAMBDA = 0.95
PPO_GAMMA = 0.99
PPO_VALUE_COEF = 0.5
PPO_ENTROPY_COEF = 0.01
PPO_MAX_GRAD_NORM = 0.5

# Environment Configuration
ENV_NAME = 'SuperMarioBros-1-1-v0'
RENDER_MODE = 'human'  # Use 'human' for visible display

# Game Settings
MAX_STEPS_PER_EPISODE = 1000
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
SAVE_MODEL_EVERY = 100
LOG_EVERY = 10
EVALUATION_EPISODES = 5

# Logging
VERBOSE = True
SAVE_SCREENSHOTS = False
SCREENSHOT_DIR = 'screenshots'
SCREENSHOT_EVERY = 30
MODEL_SAVE_DIR = 'models'
LOG_DIR = 'logs'

REPORT_EVERY = 10