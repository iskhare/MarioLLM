import torch

import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY

from agents import Random_Agent, Greedy_Agent, Lookahead_Agent

from nes_py.wrappers import JoypadSpace
from wrappers import apply_wrappers

import matplotlib.pyplot as plt
import numpy as np


if torch.cuda.is_available():
    print("Using CUDA device:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available")

ENV_NAME = 'SuperMarioBros-1-2-v0'
DISPLAY = True
NUM_OF_EPISODES = 1000

live_env = gym_super_mario_bros.make(
    ENV_NAME,
    render_mode='human' if DISPLAY else None,
    apply_api_compatibility=True
)
live_env = JoypadSpace(live_env, RIGHT_ONLY)
live_env = apply_wrappers(live_env)

def make_sim_env():
    sim = gym_super_mario_bros.make(
        ENV_NAME,
        render_mode=None,
        apply_api_compatibility=True
    )
    sim = JoypadSpace(sim, RIGHT_ONLY)
    sim = apply_wrappers(sim)
    return sim

# Create one “live” env for training:
env = live_env

random_agent = Random_Agent(num_actions=env.action_space.n)
greedy_agent = Greedy_Agent(make_sim_env_fn=make_sim_env, live_env=live_env, num_actions=live_env.action_space.n)
lookahead_agent = Lookahead_Agent(
    make_sim_env_fn=make_sim_env,
    live_env=live_env,
    num_actions=live_env.action_space.n
)
agent = 'RAN'

env.reset()
next_state, reward, done, trunc, info = env.step(action=0)

total_rewards = []
for i in range(NUM_OF_EPISODES):
    print("Episode:", i)
    done = False

    state, _ = env.reset()
    if agent in ['GREEDY', 'LOOK']:
        # clear history for simulation-based agents
        if agent == 'GREEDY':
            greedy_agent.action_history.clear()
        else:
            lookahead_agent.action_history.clear()

    total_reward = 0
    while not done:
        if agent == 'RAN':
            a = random_agent.choose_action()
            new_state, reward, done, truncated, info = env.step(a)
        elif agent == 'GREEDY':
            new_state, reward, done, truncated, info = greedy_agent.choose_and_update(state)
        else:
            new_state, reward, done, truncated, info = lookahead_agent.choose_and_update(state)

        total_reward += reward
        state = new_state


        total_reward += reward
        state = new_state

    total_rewards.append(total_reward)

    print("Total reward:", total_reward)


env.close()

# Plot total reward per episode
plt.figure(figsize=(8, 4))
plt.plot(range(1, NUM_OF_EPISODES + 1), total_rewards, marker='o', linestyle='-')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Random Agent: Total Reward per Episode')
plt.grid(True)
plt.tight_layout()
plt.show()

# Print average reward
avg_reward = np.mean(total_rewards)
print(f'Average reward over {NUM_OF_EPISODES} episodes: {avg_reward:.2f}')