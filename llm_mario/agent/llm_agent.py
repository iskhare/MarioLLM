from fireworks.client import Fireworks
import json
import time
from typing import List, Dict, Any, Optional
from collections import deque
import base64

from .emulator import MarioEmulator
import config


class LLMAgent:
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or config.FIREWORKS_API_KEY
        self.model = model or config.DEFAULT_MODEL
        
        if not self.api_key:
            raise ValueError("FIREWORKS_API_KEY must be set in environment variables or passed to constructor")
        
        self.client = Fireworks(api_key=self.api_key)
        self.memory = deque(maxlen=config.MAX_MEMORY_ITEMS)
        self.screenshot_history = deque(maxlen=config.SCREENSHOT_HISTORY)
        
        # Game state tracking
        self.last_x_pos = 0
        self.stuck_counter = 0
        self.last_action = 0
        
    def get_system_prompt(self) -> str:
        """System prompt that defines the agent's role and capabilities"""
        return """You are an AI agent playing Super Mario Bros. Your goal is to help Mario complete the level by moving right, avoiding enemies, collecting coins, and reaching the flag.

Available actions:
0: NOOP (do nothing)
1: right (move right)
2: right + A (move right and jump)
3: right + B (move right and run)
4: right + A + B (move right, run, and jump)
5: A (jump in place)
6: left (move left)

Game information you'll receive:
- Screenshot of the current game state
- Mario's position (x_pos, y_pos)
- Score, coins, lives, time remaining
- Recent performance metrics

Strategy tips:
- Always try to move right to progress through the level
- Jump over gaps and enemies
- Use running (B button) to move faster and jump farther
- Collect coins when safe to do so
- Avoid enemies or jump on them to defeat them
- If you're stuck in the same position, try different actions

Respond with a JSON object containing:
{
    "action": <action_number>,
    "reasoning": "<brief explanation of why you chose this action>",
    "observation": "<what you see in the current game state>"
}"""

    def format_game_state(self, game_state: Dict[str, Any]) -> str:
        """Format game state information for the prompt"""
        return f"""Current Game State:
- Position: ({game_state['x_pos']}, {game_state['y_pos']})
- Score: {game_state['score']}
- Coins: {game_state['coins']}
- Lives: {game_state['life']}
- Time: {game_state['time']}
- Status: {game_state['status']}
- World-Stage: {game_state['world']}-{game_state['stage']}
- Steps taken: {game_state['step_count']}
- Episode reward: {game_state['episode_reward']}"""

    def format_recent_actions(self) -> str:
        """Format recent actions and observations for context"""
        if not self.memory:
            return "No recent actions."
        
        recent = "Recent actions:\n"
        for i, item in enumerate(list(self.memory)[-3:]):
            recent += f"Step -{len(self.memory)-i}: Action {item['action']} ({config.ACTION_MAPPING.get(item['action'], 'Unknown')}) -> Reward: {item['reward']:.2f}\n"
        return recent

    def detect_stuck_state(self, current_x_pos: int) -> bool:
        """Detect if Mario is stuck in the same position"""
        if current_x_pos <= self.last_x_pos:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        
        self.last_x_pos = current_x_pos
        return self.stuck_counter > 5

    def choose_action(self, screenshot: str, game_state: Dict[str, Any]) -> int:
        """Choose an action based on the current game state"""
        try:
            # Build the messages for OpenAI-compatible format
            messages = [
                {"role": "system", "content": self.get_system_prompt()},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{screenshot}"
                            }
                        },
                        {
                            "type": "text",
                            "text": f"""{self.format_game_state(game_state)}

{self.format_recent_actions()}

Stuck detection: {'Mario seems stuck! Try a different approach.' if self.detect_stuck_state(game_state['x_pos']) else 'Mario is making progress.'}

Based on the screenshot and game state, choose your next action. Focus on moving right and progressing through the level."""
                        }
                    ]
                }
            ]

            # Make API call using Fireworks client
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=300,
                messages=messages
            )

            # Parse response from Fireworks API
            response_text = response.choices[0].message.content
            
            # Try to extract JSON from response
            try:
                # Look for JSON in the response
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    response_data = json.loads(json_match.group())
                    action = response_data.get('action', 1)  # Default to moving right
                    reasoning = response_data.get('reasoning', 'No reasoning provided')
                    observation = response_data.get('observation', 'No observation provided')
                else:
                    # Fallback parsing
                    action = 1
                    reasoning = "Failed to parse JSON, defaulting to move right"
                    observation = response_text
            except json.JSONDecodeError:
                # Fallback action
                action = 1
                reasoning = "JSON parsing failed, defaulting to move right"
                observation = response_text

            # Validate action
            if not isinstance(action, int) or action < 0 or action >= len(config.ACTION_MAPPING):
                action = 1

            # Store in memory
            self.memory.append({
                'action': action,
                'reasoning': reasoning,
                'observation': observation,
                'game_state': game_state.copy(),
                'timestamp': time.time()
            })

            if config.VERBOSE:
                print(f"Action: {action} ({config.ACTION_MAPPING.get(action, 'Unknown')})")
                print(f"Reasoning: {reasoning}")

            self.last_action = action
            return action

        except Exception as e:
            print(f"Error in choose_action: {e}")
            # Fallback to moving right
            return 1

    def update_memory(self, reward: float, done: bool):
        """Update the last memory item with reward information"""
        if self.memory:
            self.memory[-1]['reward'] = reward
            self.memory[-1]['done'] = done

    def reset_episode(self):
        """Reset agent state for a new episode"""
        self.last_x_pos = 0
        self.stuck_counter = 0
        self.last_action = 0 