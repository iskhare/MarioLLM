# LLM-Based Super Mario Bros Reinforcement Learning

<div align="center">
  <img src="./image.jpg" alt="Super Mario RL" width="30%"/>
</div>

This project explores the use of open-source large language models (LLMs) as decision-making agents in a reinforcement learning setting, specifically for playing Super Mario Bros. We integrate LLMs into the Model Context Protocol (MCP) to create a closed-loop control system where the model observes game state, chooses actions, and receives feedback.

## Setup

### Prerequisites
1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up your Fireworks API key:
   - Sign up at [Fireworks.ai](https://fireworks.ai)
   - Generate an API key from the [API Keys page](https://fireworks.ai/account/api-keys)
   - Set the environment variable:
     ```bash
     export FIREWORKS_API_KEY="your_api_key_here"
     ```

## Project Overview

Our approach involves:
1. Using Fireworks.ai's open-source LLMs (LLaMA 3.3 70B Instruct) to play Super Mario Bros  
2. Using the Model Context Protocol (MCP) for environment interaction  
3. Implementing a closed-loop control system for continuous feedback  
4. Training through behavioral cloning and reinforcement learning

## Team

- Ishan Khare
- Gabriel Seir
- Anthony Zhan

## Related Work

- [Playing Super Mario with LLMs as a Benchmark](https://www.reddit.com/r/singularity/comments/1j1pxru/playing_super_mario_with_llms_as_a_benchmark_by)
- [Claude Plays Pokemon](https://www.twitch.tv/claudeplayspokemon)
