import os

# API Configuration
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
DEFAULT_MODEL = 'claude-3-haiku-20240307'

# Environment Configuration
ENV_NAME = 'SuperMarioBros-1-1-v3'
RENDER_MODE = 'rgb_array'  # Use 'human' for visible display

# Game Settings
MAX_STEPS_PER_EPISODE = 1000
FRAME_SKIP = 4
IMAGE_SIZE = (84, 84)

# Agent Configuration
MAX_MEMORY_ITEMS = 10
SCREENSHOT_HISTORY = 9
INCLUDE_GAME_STATE = True

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

# Logging
VERBOSE = True
SAVE_SCREENSHOTS = False
SCREENSHOT_DIR = 'screenshots'
SCREENSHOT_EVERY = 10