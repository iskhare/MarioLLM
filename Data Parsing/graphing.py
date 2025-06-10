import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('episode_reward_epsilon_full.csv')

# Plotting Episode vs. Total Reward and Epsilon on one graph
episodes = df.iloc[:, 0]   # assuming first column is Episode
rewards = df.iloc[:, 1]    # assuming second column is Reward
epsilons = df.iloc[:, 2]   # assuming third column is Epsilon

print(np.mean(rewards))

fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.set_xlabel('Episode')
ax1.set_ylabel('Total Reward', color='tab:blue')
ax1.plot(episodes, rewards, color='tab:blue', label='Reward')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Epsilon', color='tab:red')
ax2.plot(episodes, epsilons, color='tab:red', linestyle='--', label='Epsilon')
ax2.tick_params(axis='y', labelcolor='tab:red')

plt.title('Episode vs. Total Reward and Epsilon for DDQN Training')
fig.tight_layout()
plt.show()
