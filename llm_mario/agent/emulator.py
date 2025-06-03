import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import numpy as np
from PIL import Image
import io
import base64
from wrappers import apply_wrappers


class MarioEmulator:
    def __init__(self, env_name='SuperMarioBros-1-1-v3', render_mode='human'):
        self.env_name = env_name
        self.render_mode = render_mode
        self.env = gym_super_mario_bros.make(
            env_name, 
            render_mode=render_mode,
            apply_api_compatibility=True
        )
        
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)
        self.env = apply_wrappers(self.env)
        
        self.current_state = None
        self.last_info = {}
        self.step_count = 0
        self.episode_reward = 0
        
    def reset(self):
        """Reset the environment and return initial state"""
        self.current_state, self.last_info = self.env.reset()
        self.step_count = 0
        self.episode_reward = 0
        return self.current_state
    
    def step(self, action):
        """Take a step in the environment"""
        if not isinstance(action, int) or action < 0 or action >= len(SIMPLE_MOVEMENT):
            action = 0  # Default to NOOP for invalid actions
            
        self.current_state, reward, done, truncated, info = self.env.step(action)
        self.last_info = info
        self.step_count += 1
        self.episode_reward += reward
        
        return self.current_state, reward, done, truncated, info
    
    def get_screenshot(self, format='base64'):
        """Get current screenshot in specified format"""
        try:
            if self.render_mode == 'rgb_array':
                # Get the current frame directly from the environment
                frame = self.env.render()
            else:
                # For other modes, try to get the screen from the state
                frame = self.current_state
                
            if frame is None:
                return None
                
            # Convert numpy array to PIL Image
            if isinstance(frame, np.ndarray):
                if len(frame.shape) == 3:
                    image = Image.fromarray(frame)
                else:
                    # Handle grayscale or other formats
                    image = Image.fromarray(frame, mode='L')
            else:
                return None
                
            if format == 'base64':
                buffer = io.BytesIO()
                image.save(buffer, format='PNG')
                img_str = base64.b64encode(buffer.getvalue()).decode()
                return img_str
            elif format == 'pil':
                return image
            elif format == 'numpy':
                return np.array(image)
            else:
                return image
        except Exception as e:
            print(f"Error getting screenshot: {e}")
            return None
    
    def get_game_state(self):
        """Extract game state information"""
        return {
            'x_pos': self.last_info.get('x_pos', 0),
            'y_pos': self.last_info.get('y_pos', 0), 
            'score': self.last_info.get('score', 0),
            'coins': self.last_info.get('coins', 0),
            'life': self.last_info.get('life', 0),
            'stage': self.last_info.get('stage', 0),
            'world': self.last_info.get('world', 0),
            'time': self.last_info.get('time', 0),
            'step_count': self.step_count,
            'episode_reward': self.episode_reward,
            'status': self.last_info.get('status', 'small')
        }
    
    def close(self):
        """Close the environment"""
        self.env.close()
    
    def get_action_space_size(self):
        """Get the size of the action space"""
        return self.env.action_space.n
    
    def get_action_meanings(self):
        """Get the meaning of each action"""
        return [str(actions) for actions in SIMPLE_MOVEMENT] 