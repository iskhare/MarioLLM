#!/usr/bin/env python3

import argparse
import os
import time
import wandb

from agent import PPOAgent, MarioEmulator
import config


def main():
    parser = argparse.ArgumentParser(description='PPO Agent playing Super Mario Bros')
    parser.add_argument('--display', action='store_true', help='Show the game display')
    parser.add_argument('--save-screenshots', action='store_true', help='Save screenshots')
    parser.add_argument('--load-checkpoint', type=str, help='Path to load trained model checkpoint')
    parser.add_argument('--train', action='store_true', help='Enable training mode')
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    
    args = parser.parse_args()
    
    # Initialize wandb if requested and available
    use_wandb = args.wandb
    if use_wandb:
        wandb_config = {
            'episodes': config.EPISODES,
            'max_steps': config.MAX_STEPS_PER_EPISODE,
            'model_path': config.MODEL_PATH,
            'training_mode': args.train,
            'train_every': config.TRAIN_EVERY,
            'save_every': config.SAVE_MODEL_EVERY,
            'env_name': config.ENV_NAME,
            'device': config.DEVICE,
            'max_steps_per_episode': config.MAX_STEPS_PER_EPISODE,
            'report_every': config.REPORT_EVERY,
            'screenshot_every': config.SCREENSHOT_EVERY
        }
        
        wandb.init(
            project=config.WANDB_PROJECT,
            name=config.WANDB_RUN_NAME,
            config=wandb_config
        )
        print(f"Initialized wandb logging to project: {config.WANDB_PROJECT}")
    
    # Set up directories
    if args.save_screenshots:
        os.makedirs(config.SCREENSHOT_DIR, exist_ok=True)
    
    if args.train:
        os.makedirs(config.SAVE_DIR, exist_ok=True)
    
    # Initialize emulator and agent
    render_mode = 'human' if args.display else 'rgb_array'
    emulator = MarioEmulator(config.ENV_NAME, render_mode=render_mode)
    agent = PPOAgent(model_path=config.MODEL_PATH)
    
    # Load checkpoint if specified
    if args.load_checkpoint:
        if os.path.exists(args.load_checkpoint):
            agent.load_model(args.load_checkpoint)
            print(f"Loaded checkpoint from: {args.load_checkpoint}")
        else:
            print(f"Warning: Checkpoint not found at {args.load_checkpoint}")
    
    mode = "Training" if args.train else "Evaluation"
    print(f"Starting PPO Mario Agent in {mode} mode with {config.MODEL_PATH}")
    print(f"Device: {config.DEVICE}")
    print(f"Environment: {config.ENV_NAME}")
    print(f"Episodes: {config.EPISODES}, Max steps: {config.MAX_STEPS_PER_EPISODE}")
    if args.train:
        print(f"Training every: {config.TRAIN_EVERY} episodes")
        print(f"Saving every: {config.SAVE_MODEL_EVERY} episodes")
    print("-" * 50)
    
    total_rewards = []
    total_distances = []
    training_losses = []
    
    try:
        for episode in range(config.EPISODES):
            print(f"\nEpisode {episode + 1}/{config.EPISODES}")
            
            # Reset environment and agent
            emulator.reset()
            agent.reset_episode()
            
            # Set model mode
            if args.train:
                agent.model.train()
                agent.policy_value_head.train()
            else:
                agent.model.eval()
                agent.policy_value_head.eval()
            
            episode_reward = 0
            step_count = 0
            done = False
            max_x_pos = 0
            episode_values = []
            
            while not done and step_count < config.MAX_STEPS_PER_EPISODE:
                # Get screenshot and game state
                screenshot = emulator.get_screenshot(format='base64')
                game_state = emulator.get_game_state()
                
                if screenshot is None:
                    print("Warning: Could not get screenshot, skipping step")
                    break
                
                # Agent chooses action
                action, log_prob, value = agent.choose_action(screenshot, game_state)
                episode_values.append(value)
                
                # Take step in environment
                _, reward, done, truncated, info = emulator.step(action)
                episode_reward += reward
                max_x_pos = max(max_x_pos, game_state['x_pos'])
                step_count += 1
                
                # Store experience for training
                if args.train:
                    state_data = {
                        'screenshot': screenshot,
                        'game_state': game_state
                    }
                    agent.store_experience(
                        state_data=state_data,
                        action=action,
                        log_prob=log_prob,
                        value=value,
                        reward=reward,
                        done=done or truncated
                    )
                
                # Log step-level metrics to wandb
                if use_wandb and step_count % (config.REPORT_EVERY * 2) == 0:
                    wandb.log({
                        'step/reward': reward,
                        'step/x_position': game_state['x_pos'],
                        'step/score': game_state['score'],
                        'step/value_estimate': value,
                        'step/episode': episode + 1,
                        'step/step_in_episode': step_count
                    })
                
                # Save screenshot if requested
                if args.save_screenshots and step_count % config.SCREENSHOT_EVERY == 0:
                    screenshot_pil = emulator.get_screenshot(format='pil')
                    if screenshot_pil:
                        screenshot_path = os.path.join(
                            config.SCREENSHOT_DIR, 
                            f"episode_{episode+1}_step_{step_count}.png"
                        )
                        screenshot_pil.save(screenshot_path)
                        
                        # Log screenshot to wandb
                        if use_wandb:
                            wandb.log({
                                f"screenshots/episode_{episode+1}_step_{step_count}": wandb.Image(screenshot_pil)
                            })
                
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
            
            # Log episode-level metrics to wandb
            if use_wandb:
                episode_metrics = {
                    'episode/reward': episode_reward,
                    'episode/max_distance': max_x_pos,
                    'episode/final_position': final_state['x_pos'],
                    'episode/final_score': final_state['score'],
                    'episode/steps': step_count,
                    'episode/completed': done,
                    'episode/avg_value_estimate': sum(episode_values) / len(episode_values) if episode_values else 0,
                    'episode/number': episode + 1
                }
                wandb.log(episode_metrics)
            
            print(f"Episode {episode + 1} completed:")
            print(f"  Steps: {step_count}")
            print(f"  Total reward: {episode_reward:.2f}")
            print(f"  Max distance: {max_x_pos}")
            print(f"  Final position: {final_state['x_pos']}")
            print(f"  Final score: {final_state['score']}")
            print(f"  Status: {'Completed' if done else 'Time limit reached'}")
            
            # Training
            if args.train and (episode + 1) % config.TRAIN_EVERY == 0:
                print(f"\nTraining PPO after episode {episode + 1}...")
                train_loss = agent.train_ppo()
                if train_loss is not None:
                    training_losses.append(train_loss)
                    print(f"Training loss: {train_loss:.4f}")
                    
                    if use_wandb:
                        wandb.log({
                            'training/loss': train_loss,
                            'training/episode': episode + 1,
                            'training/buffer_size': len(agent.experience_buffer)
                        })
                else:
                    print("Not enough experience for training yet")
            
            # Save model checkpoint
            if args.train and (episode + 1) % config.SAVE_MODEL_EVERY == 0:
                checkpoint_path = os.path.join(config.SAVE_DIR, f"checkpoint_episode_{episode + 1}")
                print(f"Saving checkpoint to {checkpoint_path}")
                agent.save_model(checkpoint_path)
    
    except KeyboardInterrupt:
        print("\nStopped by user")
        
        # Save model if training was interrupted
        if args.train:
            checkpoint_path = os.path.join(config.SAVE_DIR, "checkpoint_interrupted")
            print(f"Saving interrupted training checkpoint to {checkpoint_path}")
            agent.save_model(checkpoint_path)
    
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
        
        if args.train and training_losses:
            print(f"Training sessions: {len(training_losses)}")
            print(f"Average training loss: {sum(training_losses)/len(training_losses):.4f}")
            print(f"Final training loss: {training_losses[-1]:.4f}")
        
        # Log summary statistics to wandb
        if use_wandb:
            summary_metrics = {
                'summary/episodes_completed': len(total_rewards),
                'summary/average_reward': sum(total_rewards) / len(total_rewards),
                'summary/best_reward': max(total_rewards),
                'summary/worst_reward': min(total_rewards),
                'summary/average_distance': sum(total_distances) / len(total_distances),
                'summary/best_distance': max(total_distances)
            }
            
            if training_losses:
                summary_metrics.update({
                    'summary/training_sessions': len(training_losses),
                    'summary/average_training_loss': sum(training_losses) / len(training_losses),
                    'summary/final_training_loss': training_losses[-1]
                })
            
            wandb.log(summary_metrics)
            wandb.finish()


if __name__ == "__main__":
    main() 