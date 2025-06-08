#!/usr/bin/env python3

import argparse
import os
import sys
import time
from agent import PPOAgent, MarioEmulator
import config


def main():
    parser = argparse.ArgumentParser(description='PPO Agent playing Super Mario Bros')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to run')
    parser.add_argument('--max-steps', type=int, default=config.MAX_STEPS_PER_EPISODE, help='Max steps per episode')
    parser.add_argument('--display', action='store_true', help='Show the game display')
    parser.add_argument('--save-screenshots', action='store_true', help='Save screenshots')
    parser.add_argument('--model-path', type=str, default=config.LOCAL_MODEL_PATH, help='Path to local model')
    parser.add_argument('--load-checkpoint', type=str, help='Path to load trained model checkpoint')
    
    args = parser.parse_args()
    
    # Set up screenshot directory if needed
    if args.save_screenshots:
        os.makedirs(config.SCREENSHOT_DIR, exist_ok=True)
    
    # Initialize emulator and agent
    render_mode = 'human' if args.display else 'rgb_array'
    emulator = MarioEmulator(config.ENV_NAME, render_mode=render_mode)
    agent = PPOAgent(model_path=args.model_path)
    
    # Load checkpoint if specified
    if args.load_checkpoint:
        if os.path.exists(args.load_checkpoint):
            agent.load_model(args.load_checkpoint)
            print(f"Loaded checkpoint from: {args.load_checkpoint}")
        else:
            print(f"Warning: Checkpoint not found at {args.load_checkpoint}")
    
    print(f"Starting PPO Mario Agent with {args.model_path}")
    print(f"Device: {config.DEVICE}")
    print(f"Environment: {config.ENV_NAME}")
    print(f"Episodes: {args.episodes}, Max steps: {args.max_steps}")
    print("-" * 50)
    
    total_rewards = []
    total_distances = []
    
    try:
        for episode in range(args.episodes):
            print(f"\nEpisode {episode + 1}/{args.episodes}")
            
            # Reset environment and agent
            emulator.reset()
            agent.reset_episode()
            
            # Set to evaluation mode
            agent.model.eval()
            agent.policy_value_head.eval()
            
            episode_reward = 0
            step_count = 0
            done = False
            max_x_pos = 0
            
            while not done and step_count < args.max_steps:
                # Get screenshot and game state
                screenshot = emulator.get_screenshot(format='base64')
                game_state = emulator.get_game_state()
                
                if screenshot is None:
                    print("Warning: Could not get screenshot, skipping step")
                    break
                
                # Agent chooses action
                action, log_prob, value = agent.choose_action(screenshot, game_state)
                
                # Take step in environment
                _, reward, done, truncated, info = emulator.step(action)
                episode_reward += reward
                max_x_pos = max(max_x_pos, game_state['x_pos'])
                step_count += 1
                
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
                if step_count % config.REPORT_EVERY == 0:
                    action_name = config.ACTION_MAPPING.get(action, 'Unknown')
                    print(f"Step {step_count}: Action={action}({action_name}), X={game_state['x_pos']}, "
                          f"Score={game_state['score']}, Reward={episode_reward:.1f}, Value={value:.3f}")
                
                if done or truncated:
                    break
                    
                # Small delay
                time.sleep(0.05)
            
            total_rewards.append(episode_reward)
            total_distances.append(max_x_pos)
            final_state = emulator.get_game_state()
            
            print(f"Episode {episode + 1} completed:")
            print(f"  Steps: {step_count}")
            print(f"  Total reward: {episode_reward:.2f}")
            print(f"  Max distance: {max_x_pos}")
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
        print(f"Average distance: {sum(total_distances)/len(total_distances):.2f}")
        print(f"Best distance: {max(total_distances):.2f}")


if __name__ == "__main__":
    main() 