import torch
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
import argparse

from environment import create_env
from agent import Mario
from logging import MetricLogger


def main(args):
    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")
    print()
    action_space = COMPLEX_MOVEMENT
    if args.action_space == 'RIGHT_ONLY':
        action_space = RIGHT_ONLY
    elif args.action_space == 'SIMPLE_MOVEMENT':
        action_space = SIMPLE_MOVEMENT

    env = create_env(args.stage_id, action_space, args.use_wrappers, args.skip_frames, args.obs_size, args.stack_size)

    save_dir = args.save_dir

    if args.use_wrappers:
        mario = Mario(state_dim=(args.stack_size, args.obs_size, args.obs_size), action_dim=env.action_space.n, save_dir=save_dir)
    else:
        mario = Mario(state_dim=(3, 240, 256), action_dim=env.action_space.n, save_dir=save_dir)

    logger = MetricLogger(save_dir)

    episodes = args.episodes
    for e in range(episodes):

        state = env.reset()

        # Play the game!
        while True:
            # Example training loop that we'll have to adjust
            # Will also have to adjust what we're logging in logging.py

            # Run agent on the state
            action = mario.act(state)

            # Agent performs action
            next_state, reward, done, trunc, info = env.step(action)

            # Remember
            mario.cache(state, next_state, action, reward, done)

            # Learn
            q, loss = mario.learn()

            # Logging
            logger.log_step(reward, loss, q)

            # Update state
            state = next_state

            # Check if end of game
            if done or info["flag_get"]:
                break

        logger.log_episode()

        if (e % 20 == 0) or (e == episodes - 1):
            logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Super Mario RL Runner")

    parser.add_argument('--stage_id', type=str, default='SuperMarioBros-1-1-v0',
                            help='Gym environment name')
    # Later we can incorporate random stage selection of multiple stages using gym.make('SuperMarioBrosRandomStages-v0', stages=['1-4', '2-4', '3-4', '4-4'])


    # 40,000 episodes recommend for training but we can adjust
    parser.add_argument('--episodes', type=int, default=40000,
                            help='Total number of episodes to run')
    parser.add_argument('--save_dir', type=str, default='./models/',
                            help='Directory to save trained models')
    parser.add_argument('--render', action='store_true',
                            help='Render the environment during training')
    parser.add_argument('--action_space', type=str, default='COMPLEX_MOVEMENT',
                        help='Action space')
    parser.add_argument('--use_wrappers', type=bool, default=True,
                        help='Use wrappers')
    parser.add_argument('--skip_frames', type=int, default=4,
                        help='Number of skip frames')
    parser.add_argument('--obs_size', type=int, default=84,
                        help='Observation frame size')
    parser.add_argument('--stack_size', type=int, default=4,
                        help='Number of stack frames')

    args = parser.parse_args()

    main(args)