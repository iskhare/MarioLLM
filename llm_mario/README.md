# LLM Mario Agent

An AI agent that uses Large Language Models (Claude 3.5 Haiku) to play Super Mario Bros through computer vision and natural language reasoning.

## Overview

This project implements an LLM-based agent that:
- Observes the game state through screenshots
- Analyzes the visual information using Claude's vision capabilities
- Makes strategic decisions about Mario's actions
- Learns from recent performance to improve gameplay

## Features

- **Vision-based gameplay**: Agent analyzes screenshots to understand game state
- **Natural language reasoning**: Uses Claude's reasoning capabilities for strategic decisions
- **Memory system**: Tracks recent actions and outcomes for context
- **Stuck detection**: Identifies when Mario isn't progressing and tries different strategies
- **Game state tracking**: Monitors position, score, coins, lives, and other metrics
- **Flexible configuration**: Easy to adjust models, settings, and parameters

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your Anthropic API key:
   ```bash
   export ANTHROPIC_API_KEY=your_api_key_here
   ```

## Usage

### Basic Usage
```bash
python main.py
```

### Advanced Options
```bash
python main.py --episodes 10 --max-steps 2000 --display --save-screenshots
```

### Command Line Arguments
- `--episodes`: Number of episodes to run (default: 5)
- `--max-steps`: Maximum steps per episode (default: 1000)
- `--display`: Show the game window while playing
- `--save-screenshots`: Save periodic screenshots to disk
- `--model`: Claude model to use (default: claude-3-5-haiku-20241022)
- `--api-key`: Anthropic API key (overrides environment variable)

## Configuration

Edit `config.py` to customize:
- API settings and model selection
- Game parameters and action mappings
- Memory and screenshot settings
- Logging and debugging options

## Project Structure

```
llm_mario/
├── agent/
│   ├── __init__.py          # Package exports
│   ├── llm_agent.py         # Main LLM agent implementation
│   └── emulator.py          # Mario environment wrapper
├── config.py                # Configuration settings
├── main.py                  # Main execution script
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## How It Works

1. **Environment Setup**: Creates Super Mario Bros environment using `gym_super_mario_bros`
2. **Screenshot Capture**: Takes RGB screenshots of the current game state
3. **LLM Analysis**: Sends screenshot and game data to Claude for analysis
4. **Action Selection**: Claude responds with chosen action and reasoning
5. **Game Execution**: Action is executed in the Mario environment
6. **Memory Update**: Results are stored for future context

## Action Space

The agent can choose from these actions:
- `0`: NOOP (do nothing)
- `1`: Move right
- `2`: Move right + jump
- `3`: Move right + run
- `4`: Move right + run + jump
- `5`: Jump in place
- `6`: Move left

## Performance Tracking

The agent tracks:
- Episode rewards and scores
- Mario's position progression
- Completion rates and survival time
- Action patterns and effectiveness

## Comparison to Similar Projects

### vs. Claude Plays Pokemon
- **Similarities**: Both use Claude's vision capabilities for gameplay
- **Differences**: 
  - Mario has different action space and objectives
  - Mario requires real-time positioning awareness
  - Different memory reading approaches (Pokemon uses ROM memory, Mario uses gym info)

### vs. DDQN Implementation
- **Similarities**: Same environment and action space
- **Differences**:
  - LLM uses reasoning vs. learned neural network policies
  - No training required, immediate deployment
  - Interpretable decision making process

## Extending the Project

### Custom Strategies
- Modify the system prompt in `llm_agent.py`
- Add specialized game state analysis
- Implement level-specific strategies

### Different Models
- Change `DEFAULT_MODEL` in config.py
- Experiment with different Claude models
- Compare performance across model versions

### Enhanced Memory
- Extend memory system for longer context
- Add screenshot history analysis
- Implement performance-based learning

## Troubleshooting

### Common Issues
1. **API Key Error**: Ensure `ANTHROPIC_API_KEY` is set correctly
2. **Environment Issues**: Check that gym_super_mario_bros is installed properly
3. **Rate Limiting**: Increase delay between API calls if needed
4. **Display Issues**: Use `--display` flag for debugging visual problems

### Performance Tips
- Use `claude-3-5-haiku` for faster, cheaper gameplay
- Adjust `MAX_STEPS_PER_EPISODE` based on your goals
- Enable verbose logging for debugging

## Future Improvements

- **Multi-level support**: Extend to different Mario levels
- **Advanced memory**: Implement more sophisticated context management
- **Performance metrics**: Add detailed analytics and visualization
- **Real-time optimization**: Adaptive strategy based on performance
- **Model comparison**: Framework for testing different LLMs 