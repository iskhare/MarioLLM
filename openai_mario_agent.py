import openai
import json
import time
from typing import Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class OpenAIMarioAgent:
    def __init__(self, action_space, api_key=None):
        """
        Mario agent that uses OpenAI's API to make decisions
        """
        self.client = openai.OpenAI(
            api_key=api_key or os.getenv('OPENAI_API_KEY')
        )
        self.action_space = action_space
        self.action_map = self._create_action_map()
        self.system_prompt = self._create_system_prompt()
        
        # Track game state for context
        self.previous_x_pos = 0
        self.stuck_counter = 0
        
    def _create_action_map(self):
        """Map action names to action indices based on action space size"""
        if len(self.action_space) == 2:  # RIGHT_ONLY
            return {
                "NOOP": 0,
                "RIGHT": 1
            }
        elif len(self.action_space) == 7:  # SIMPLE_MOVEMENT  
            return {
                "NOOP": 0,
                "RIGHT": 1,
                "A": 2,          # Jump
                "B": 3,          # Run/Fire
                "A_RIGHT": 4,    # Jump Right
                "B_RIGHT": 5,    # Run Right  
                "A_B": 6         # Jump + Run
            }
        else:  # COMPLEX_MOVEMENT (12 actions)
            return {
                "NOOP": 0,
                "RIGHT": 1,
                "A": 2,          # Jump
                "B": 3,          # Run/Fire
                "A_RIGHT": 4,    # Jump Right
                "B_RIGHT": 5,    # Run Right
                "A_B": 6,        # Jump + Run
                "A_B_RIGHT": 7,  # Jump + Run Right
                "LEFT": 8,       # Move Left
                "A_LEFT": 9,     # Jump Left
                "B_LEFT": 10,    # Run Left
                "A_B_LEFT": 11   # Jump + Run Left
            }
    
    def _create_system_prompt(self):
        """Create the system prompt for the OpenAI model"""
        available_actions = list(self.action_map.keys())
        
        return f"""You are an AI playing Super Mario Bros. Your goal is to control Mario to reach the flag at the end of level 1-1.

AVAILABLE ACTIONS: {', '.join(available_actions)}

GAME MECHANICS:
- Move RIGHT to progress through the level
- Use A to JUMP over obstacles and enemies
- Use B to RUN (makes Mario faster)
- Combine actions like A_RIGHT (jump while moving right) for efficiency
- Avoid falling into pits (deadly)
- Jump on enemies to defeat them
- Collect coins and power-ups when safe

STRATEGY TIPS:
- Always bias toward moving RIGHT since that's how you progress
- Jump early when you see obstacles or enemies
- Use B_RIGHT to move faster when the path is clear
- If you get stuck in the same position, try jumping (A) or jumping right (A_RIGHT)

You will receive the current game state and must respond with EXACTLY ONE action name from the available actions. 
Respond with only the action name, nothing else."""

    def describe_game_state(self, observation, info):
        """Convert game state to text description for the LLM"""
        x_pos = info.get('x_pos', 0)
        y_pos = info.get('y_pos', 79)  # Default ground level
        coins = info.get('coins', 0)
        score = info.get('score', 0)
        time_left = info.get('time', 400)
        life = info.get('life', 2)
        
        # Check if Mario is stuck
        if x_pos == self.previous_x_pos:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        self.previous_x_pos = x_pos
        
        # Determine Mario's situation
        situation = "normal"
        if self.stuck_counter > 3:
            situation = "stuck - need to jump or change direction"
        elif y_pos > 79:  # Mario is in the air
            situation = "jumping/falling"
        elif x_pos < 50:
            situation = "early in level"
        elif x_pos > 2000:
            situation = "near end of level"
        
        description = f"""MARIO GAME STATE:
Position: X={x_pos}, Y={y_pos}
Score: {score}, Coins: {coins}, Time: {time_left}, Lives: {life}
Situation: {situation}
Progress: {min(100, int(x_pos/32))}/100% through level

What action should Mario take?"""
        
        return description

    def get_action_from_openai(self, game_state_description):
        """Send game state to OpenAI and get action response"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Use the fast, cheap model for gaming
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": game_state_description}
                ],
                max_tokens=10,  # We only need a short action name
                temperature=0.3  # Some randomness but mostly deterministic
            )
            
            action_text = response.choices[0].message.content.strip().upper()
            return self.parse_action(action_text)
            
        except Exception as e:
            print(f"OpenAI API error: {e}")
            # Fallback to moving right if API fails
            return self.action_map.get("RIGHT", 1)
    
    def parse_action(self, llm_response):
        """Parse LLM response to get valid action index"""
        action_text = llm_response.strip().upper()
        
        # Try exact match first
        if action_text in self.action_map:
            return self.action_map[action_text]
        
        # Try partial matches
        for action_name, action_idx in self.action_map.items():
            if action_name in action_text:
                return action_idx
        
        # If stuck, try jumping right
        if self.stuck_counter > 3:
            return self.action_map.get("A_RIGHT", self.action_map.get("A", 2))
        
        # Default fallback
        return self.action_map.get("RIGHT", 1)
    
    def act(self, observation, info):
        """Main method called by game loop"""
        description = self.describe_game_state(observation, info)
        action_idx = self.get_action_from_openai(description)
        
        # Print what the agent is doing
        action_name = next(name for name, idx in self.action_map.items() if idx == action_idx)
        x_pos = info.get('x_pos', 0)
        print(f"X: {x_pos:4d} | Action: {action_name}")
        
        return action_idx 