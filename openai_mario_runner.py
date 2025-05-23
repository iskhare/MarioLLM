import torch
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
import argparse
import time
import os

from environment import create_env
from openai_mario_agent import OpenAIMarioAgent


def run_openai_mario(args):
    """Run Mario with OpenAI LLM agent"""
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("ERROR: Please set your OPENAI_API_KEY environment variable")
        print("You can do this by:")
        print("1. Creating a .env file with: OPENAI_API_KEY=your_key_here")
        print("2. Or running: export OPENAI_API_KEY=your_key_here")
        return
    
    print("🎮 Starting OpenAI-powered Mario Agent...")
    print(f"Stage: {args.stage_id}")
    print(f"Action Space: {args.action_space}")
    print("=" * 50)
    
    # Set up action space
    action_space_map = {
        'RIGHT_ONLY': RIGHT_ONLY,
        'SIMPLE_MOVEMENT': SIMPLE_MOVEMENT, 
        'COMPLEX_MOVEMENT': COMPLEX_MOVEMENT
    }
    action_space = action_space_map[args.action_space]

    # Create environment
    env = create_env(
        args.stage_id, 
        action_space, 
        use_wrappers=False,  # Keep original frames for better state description
        skip_frames=4,       # Skip some frames for performance
        obs_size=84, 
        stack_size=1
    )

    # Create OpenAI agent
    mario_agent = OpenAIMarioAgent(action_space)

    # Run episodes
    for episode in range(args.episodes):
        print(f"\n🚀 Episode {episode + 1}/{args.episodes}")
        print("-" * 30)
        
        state = env.reset()
        total_reward = 0
        steps = 0
        start_time = time.time()
        
        while True:
            # Get environment info 
            info = {}
            if hasattr(env.unwrapped, '_get_info'):
                info = env.unwrapped._get_info()
            
            # Agent chooses action
            action = mario_agent.act(state, info)
            
            # Execute action
            next_state, reward, done, trunc, info = env.step(action)
            
            total_reward += reward
            steps += 1
            
            state = next_state
            
            # Add delay if requested
            if args.delay:
                time.sleep(args.delay)
            
            # Check termination conditions
            if done or info.get("flag_get", False):
                break
                
            # Safety: stop if taking too long
            if steps > args.max_steps:
                print(f"⏰ Stopped after {args.max_steps} steps")
                break
        
        # Episode summary
        elapsed_time = time.time() - start_time
        final_x = info.get('x_pos', 0)
        success = info.get("flag_get", False)
        
        print("\n" + "=" * 50)
        print(f"📊 EPISODE {episode + 1} SUMMARY:")
        print(f"Success: {'🎉 YES!' if success else '❌ No'}")
        print(f"Final X Position: {final_x}")
        print(f"Total Reward: {total_reward:.1f}")
        print(f"Steps Taken: {steps}")
        print(f"Time: {elapsed_time:.1f}s")
        print(f"Progress: {min(100, int(final_x/32))}/100%")
        
        if success:
            print("🏆 LEVEL COMPLETE! Mario reached the flag!")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Super Mario Bros with OpenAI Agent")
    
    parser.add_argument('--stage_id', type=str, default='SuperMarioBros-1-1-v0',
                        help='Mario stage to play')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to run')
    parser.add_argument('--action_space', type=str, default='SIMPLE_MOVEMENT',
                        choices=['RIGHT_ONLY', 'SIMPLE_MOVEMENT', 'COMPLEX_MOVEMENT'],
                        help='Action space for Mario')
    parser.add_argument('--delay', type=float, default=0.0,
                        help='Delay between actions in seconds (for observation)')
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='Maximum steps per episode')
    
    args = parser.parse_args()
    run_openai_mario(args) 