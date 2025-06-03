#!/usr/bin/env python3

import pygame
import sys
from agent import MarioEmulator
import config


def main():
    """Manual play demo for comparison with LLM agent"""
    
    # Initialize pygame for keyboard input
    pygame.init()
    pygame.display.set_mode((1, 1))  # Minimal display for pygame events
    
    # Initialize emulator with display
    emulator = MarioEmulator(config.ENV_NAME, render_mode='human')
    
    print("Manual Mario Demo")
    print("=" * 50)
    print("Controls:")
    print("  Arrow Left: Move left")
    print("  Arrow Right: Move right")
    print("  Z: Jump (A button)")
    print("  X: Run/Fire (B button)")
    print("  Combine keys for complex moves")
    print("  ESC or Q: Quit")
    print("=" * 50)
    print("Starting game...")
    
    def get_action_from_keys():
        """Convert current key states to Mario action"""
        keys = pygame.key.get_pressed()
        
        # Check for quit
        if keys[pygame.K_ESCAPE] or keys[pygame.K_q]:
            return None
        
        # Map key combinations to actions
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
    
    try:
        emulator.reset()
        done = False
        clock = pygame.time.Clock()
        
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
            _, reward, done, truncated, info = emulator.step(action)
            
            # Show game state periodically
            if emulator.step_count % 100 == 0:
                game_state = emulator.get_game_state()
                print(f"Step {emulator.step_count}: "
                      f"X={game_state['x_pos']}, "
                      f"Score={game_state['score']}, "
                      f"Lives={game_state['life']}")
            
            # Control frame rate
            clock.tick(60)
            
            if done or truncated:
                break
        
        # Final statistics
        final_state = emulator.get_game_state()
        print("\nGame Over!")
        print(f"Final Position: {final_state['x_pos']}")
        print(f"Final Score: {final_state['score']}")
        print(f"Total Steps: {final_state['step_count']}")
        print(f"Total Reward: {final_state['episode_reward']}")
        
    except KeyboardInterrupt:
        print("\nGame interrupted by user")
    
    finally:
        emulator.close()
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    main() 