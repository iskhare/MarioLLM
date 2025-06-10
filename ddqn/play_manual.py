import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import pygame
import sys

ENV_NAME = 'SuperMarioBros-1-1-v0'

# Initialize pygame for keyboard input
pygame.init()
pygame.display.set_mode((1, 1))  # Minimal display for pygame events

env = gym_super_mario_bros.make(ENV_NAME, render_mode='human', apply_api_compatibility=True)
env = JoypadSpace(env, SIMPLE_MOVEMENT)

print("Controls:")
print("Arrow Left: Move left")
print("Arrow Right: Move right")
print("Z: Jump (A button)")
print("X: Run/Fire (B button)")
print("Combine keys for complex moves (e.g., Right + Z to jump while moving)")
print("ESC or Q: Quit")
print("Press any key to start...")

def get_action_from_keys():
    """Convert current key states to Mario action"""
    keys = pygame.key.get_pressed()
    
    # Check for quit
    if keys[pygame.K_ESCAPE] or keys[pygame.K_q]:
        return None
    
    # Map key combinations to SIMPLE_MOVEMENT actions
    # SIMPLE_MOVEMENT actions: [['NOOP'], ['right'], ['right', 'A'], ['right', 'B'], ['right', 'A', 'B'], ['A'], ['left']]
    
    left = keys[pygame.K_LEFT]
    right = keys[pygame.K_RIGHT]
    a_button = keys[pygame.K_z]  # Jump
    b_button = keys[pygame.K_x]  # Run/Fire
    
    # Handle combinations with right movement
    if right and a_button and b_button:
        return 4  # ['right', 'A', 'B']
    elif right and b_button:
        return 3  # ['right', 'B']
    elif right and a_button:
        return 2  # ['right', 'A']
    elif right:
        return 1  # ['right']
    # Standalone jump
    elif a_button:
        return 5  # ['A']
    # Left movement
    elif left:
        return 6  # ['left']
    # No input
    else:
        return 0  # ['NOOP']

done = False
state, _ = env.reset()

clock = pygame.time.Clock()

try:
    total_reward = 0
    while not done:
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                break
        
        # Get action from keyboard
        action = get_action_from_keys()
        if action is None:  # Quit requested
            break
            
        # Take action in environment
        state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        # Control frame rate (60 FPS)
        clock.tick(60)
    print('Total Reward: ', total_reward)
        
except KeyboardInterrupt:
    print("\nGame interrupted by user")

env.close()
pygame.quit()
sys.exit()