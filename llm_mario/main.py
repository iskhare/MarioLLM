#!/usr/bin/env python3

import argparse
import os
import sys
import time
from agent import LLMAgent, MarioEmulator
import config


def main():
    parser = argparse.ArgumentParser(description='LLM Agent playing Super Mario Bros')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to run')
    parser.add_argument('--max-steps', type=int, default=config.MAX_STEPS_PER_EPISODE, help='Max steps per episode')
    parser.add_argument('--display', action='store_true', help='Show the game display')
    parser.add_argument('--save-screenshots', action='store_true', help='Save screenshots')
    parser.add_argument('--model', type=str, default=config.DEFAULT_MODEL, help='Claude model to use')
    parser.add_argument('--api-key', type=str, help='Anthropic API key (overrides env var)')
    
    args = parser.parse_args()
    
    # Check API key
    api_key = args.api_key or config.ANTHROPIC_API_KEY
    if not api_key:
        print("Error: ANTHROPIC_API_KEY must be set as environment variable or passed with --api-key")
        sys.exit(1)
    
    # Set up screenshot directory if needed
    if args.save_screenshots:
        os.makedirs(config.SCREENSHOT_DIR, exist_ok=True)
    
    # Initialize emulator and agent
    render_mode = 'human' if args.display else 'rgb_array'
    emulator = MarioEmulator(config.ENV_NAME, render_mode=render_mode)
    agent = LLMAgent(api_key=api_key, model=args.model)
    
    print(f"Starting LLM Mario Agent with {args.model}")
    print(f"Environment: {config.ENV_NAME}")
    print(f"Episodes: {args.episodes}, Max steps: {args.max_steps}")
    print("-" * 50)
    
    total_rewards = []
    
    try:
        for episode in range(args.episodes):
            print(f"\nEpisode {episode + 1}/{args.episodes}")
            
            # Reset environment and agent
            emulator.reset()
            agent.reset_episode()
            
            episode_reward = 0
            step_count = 0
            done = False
            
            while not done and step_count < args.max_steps:
                # Get screenshot and game state
                screenshot = emulator.get_screenshot(format='base64')
                game_state = emulator.get_game_state()
                
                if screenshot is None:
                    print("Warning: Could not get screenshot, skipping step")
                    break
                
                # Agent chooses action
                action = agent.choose_action(screenshot, game_state)
                
                # Take step in environment
                _, reward, done, truncated, info = emulator.step(action)
                episode_reward += reward
                step_count += 1
                
                # Update agent memory
                agent.update_memory(reward, done or truncated)
                
                # Save screenshot if requested
                if args.save_screenshots and step_count % config.SCREENSHOT_EVERY == 0:
                    screenshot_pil = emulator.get_screenshot(format='pil')
                    if screenshot_pil:
                        screenshot_path = os.path.join(
                            config.SCREENSHOT_DIR, 
                            f"episode_{episode+1}_step_{step_count}.png"
                        )
                        screenshot_pil.save(screenshot_path)
                
                # Progress reporting
                if step_count % 50 == 0:
                    print(f"Step {step_count}: X={game_state['x_pos']}, Score={game_state['score']}, Reward={episode_reward:.1f}")
                
                if done or truncated:
                    break
                    
                # Small delay to avoid rate limiting
                time.sleep(0.1)
            
            total_rewards.append(episode_reward)
            final_state = emulator.get_game_state()
            
            print(f"Episode {episode + 1} completed:")
            print(f"  Steps: {step_count}")
            print(f"  Total reward: {episode_reward:.2f}")
            print(f"  Final position: {final_state['x_pos']}")
            print(f"  Final score: {final_state['score']}")
            print(f"  Status: {'Completed' if done else 'Time limit reached'}")
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    finally:
        emulator.close()
    
    # Summary statistics
    if total_rewards:
        print("\n" + "="*50)
        print("SUMMARY")
        print("="*50)
        print(f"Episodes completed: {len(total_rewards)}")
        print(f"Average reward: {sum(total_rewards)/len(total_rewards):.2f}")
        print(f"Best reward: {max(total_rewards):.2f}")
        print(f"Worst reward: {min(total_rewards):.2f}")


if __name__ == "__main__":
    main() 