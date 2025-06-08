#!/usr/bin/env python3

import argparse
import os
import sys
import time
import numpy as np
import torch
from pathlib import Path
import wandb
from tensorboardX import SummaryWriter

from agent import PPOAgent, MarioEmulator
import config


def setup_logging(args):
    """Setup logging with tensorboard and optionally wandb"""
    log_dir = Path(config.LOG_DIR) / f"ppo_mario_{int(time.time())}"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    writer = SummaryWriter(log_dir)
    
    if args.use_wandb:
        wandb.init(
            project="mario-llm-ppo",
            config={
                "learning_rate": config.PPO_LEARNING_RATE,
                "batch_size": config.PPO_BATCH_SIZE,
                "model_path": config.LOCAL_MODEL_PATH,
                "max_episodes": args.max_episodes,
                "lora_r": config.LORA_R,
                "lora_alpha": config.LORA_ALPHA,
            }
        )
    
    return writer, log_dir


def evaluate_agent(agent, emulator, num_episodes=5):
    """Evaluate the agent's performance"""
    total_rewards = []
    total_distances = []
    
    for episode in range(num_episodes):
        emulator.reset()
        agent.model.eval()
        agent.policy_value_head.eval()
        
        episode_reward = 0
        step_count = 0
        done = False
        max_x_pos = 0
        
        while not done and step_count < config.MAX_STEPS_PER_EPISODE:
            screenshot = emulator.get_screenshot(format='base64')
            game_state = emulator.get_game_state()
            
            if screenshot is None:
                break
                
            action, _, _ = agent.choose_action(screenshot, game_state)
            _, reward, done, truncated, info = emulator.step(action)
            
            episode_reward += reward
            max_x_pos = max(max_x_pos, game_state['x_pos'])
            step_count += 1
            
            if done or truncated:
                break
        
        total_rewards.append(episode_reward)
        total_distances.append(max_x_pos)
    
    return {
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'mean_distance': np.mean(total_distances),
        'std_distance': np.std(total_distances)
    }


def main():
    parser = argparse.ArgumentParser(description='Train PPO Agent on Super Mario Bros')
    parser.add_argument('--max-episodes', type=int, default=1000, help='Maximum training episodes')
    parser.add_argument('--max-steps', type=int, default=config.MAX_STEPS_PER_EPISODE, help='Max steps per episode')
    parser.add_argument('--model-path', type=str, default=config.LOCAL_MODEL_PATH, help='Path to local model')
    parser.add_argument('--display', action='store_true', help='Show the game display')
    parser.add_argument('--save-screenshots', action='store_true', help='Save screenshots')
    parser.add_argument('--use-wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--eval-every', type=int, default=50, help='Evaluate every N episodes')
    parser.add_argument('--save-every', type=int, default=config.SAVE_MODEL_EVERY, help='Save model every N episodes')
    
    args = parser.parse_args()
    
    # Setup logging
    writer, log_dir = setup_logging(args)
    print(f"Logging to: {log_dir}")
    
    # Create directories
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    if args.save_screenshots:
        os.makedirs(config.SCREENSHOT_DIR, exist_ok=True)
    
    # Initialize environment and agent
    render_mode = 'human' if args.display else 'rgb_array'
    emulator = MarioEmulator(config.ENV_NAME, render_mode=render_mode)
    agent = PPOAgent(model_path=args.model_path)
    
    print(f"Starting PPO Training with {args.model_path}")
    print(f"Device: {config.DEVICE}")
    print(f"Max episodes: {args.max_episodes}")
    print(f"LoRA Config: r={config.LORA_R}, alpha={config.LORA_ALPHA}")
    print("-" * 60)
    
    episode_rewards = []
    episode_distances = []
    global_step = 0
    
    try:
        for episode in range(args.max_episodes):
            print(f"\nEpisode {episode + 1}/{args.max_episodes}")
            
            # Reset environment
            emulator.reset()
            agent.reset_episode()
            
            episode_reward = 0
            step_count = 0
            done = False
            max_x_pos = 0
            episode_losses = []
            
            while not done and step_count < args.max_steps:
                # Get current state
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
                global_step += 1
                
                # Store experience
                state_data = {
                    'screenshot': screenshot,
                    'game_state': game_state.copy()
                }
                agent.store_experience(state_data, action, log_prob, value, reward, done or truncated)
                
                # Train PPO periodically
                if global_step % config.PPO_BATCH_SIZE == 0:
                    loss = agent.train_ppo()
                    if loss is not None:
                        episode_losses.append(loss)
                        writer.add_scalar('train/ppo_loss', loss, global_step)
                        if args.use_wandb:
                            wandb.log({'train/ppo_loss': loss}, step=global_step)
                
                # Save screenshot if requested
                if (args.save_screenshots and 
                    step_count % config.SCREENSHOT_EVERY == 0):
                    screenshot_pil = emulator.get_screenshot(format='pil')
                    if screenshot_pil:
                        screenshot_path = os.path.join(
                            config.SCREENSHOT_DIR, 
                            f"ep_{episode+1}_step_{step_count}.png"
                        )
                        screenshot_pil.save(screenshot_path)
                
                # Progress reporting
                if step_count % config.REPORT_EVERY == 0:
                    print(f"  Step {step_count}: X={game_state['x_pos']}, "
                          f"Score={game_state['score']}, Reward={episode_reward:.1f}")
                
                if done or truncated:
                    break
                    
                # Small delay to prevent overwhelming
                time.sleep(0.01)
            
            # Episode summary
            episode_rewards.append(episode_reward)
            episode_distances.append(max_x_pos)
            final_state = emulator.get_game_state()
            avg_loss = np.mean(episode_losses) if episode_losses else 0
            
            print(f"Episode {episode + 1} completed:")
            print(f"  Steps: {step_count}")
            print(f"  Total reward: {episode_reward:.2f}")
            print(f"  Max distance: {max_x_pos}")
            print(f"  Final score: {final_state['score']}")
            print(f"  Average loss: {avg_loss:.4f}")
            print(f"  Status: {'Completed' if done else 'Time limit reached'}")
            
            # Log episode metrics
            writer.add_scalar('episode/reward', episode_reward, episode)
            writer.add_scalar('episode/distance', max_x_pos, episode)
            writer.add_scalar('episode/steps', step_count, episode)
            writer.add_scalar('episode/avg_loss', avg_loss, episode)
            
            if args.use_wandb:
                wandb.log({
                    'episode/reward': episode_reward,
                    'episode/distance': max_x_pos,
                    'episode/steps': step_count,
                    'episode/avg_loss': avg_loss
                }, step=episode)
            
            # Evaluation
            if (episode + 1) % args.eval_every == 0:
                print(f"\nEvaluating after episode {episode + 1}...")
                eval_metrics = evaluate_agent(agent, emulator, config.EVALUATION_EPISODES)
                
                print(f"Evaluation Results:")
                print(f"  Mean reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
                print(f"  Mean distance: {eval_metrics['mean_distance']:.2f} ± {eval_metrics['std_distance']:.2f}")
                
                # Log evaluation metrics
                writer.add_scalar('eval/mean_reward', eval_metrics['mean_reward'], episode)
                writer.add_scalar('eval/mean_distance', eval_metrics['mean_distance'], episode)
                
                if args.use_wandb:
                    wandb.log({
                        'eval/mean_reward': eval_metrics['mean_reward'],
                        'eval/mean_distance': eval_metrics['mean_distance']
                    }, step=episode)
            
            # Save model
            if (episode + 1) % args.save_every == 0:
                model_path = os.path.join(config.MODEL_SAVE_DIR, f"ppo_mario_ep_{episode+1}")
                agent.save_model(model_path)
                print(f"Model saved to: {model_path}")
    
    except KeyboardInterrupt:
        print("\nTraining stopped by user")
    
    finally:
        emulator.close()
        writer.close()
        if args.use_wandb:
            wandb.finish()
    
    # Final summary
    if episode_rewards:
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Episodes completed: {len(episode_rewards)}")
        print(f"Average reward: {np.mean(episode_rewards):.2f}")
        print(f"Best reward: {max(episode_rewards):.2f}")
        print(f"Average distance: {np.mean(episode_distances):.2f}")
        print(f"Best distance: {max(episode_distances):.2f}")
        
        # Save final model
        final_model_path = os.path.join(config.MODEL_SAVE_DIR, "ppo_mario_final")
        agent.save_model(final_model_path)
        print(f"Final model saved to: {final_model_path}")


if __name__ == "__main__":
    main() 