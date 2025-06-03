import time
import datetime
import json
import os
from typing import Dict, List, Any
import numpy as np


def get_current_date_time_string():
    """Get current datetime as string for filenames"""
    return datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")


class Timer:
    """Simple timer utility for performance tracking"""
    def __init__(self):
        self.times = []
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def end(self, message=''):
        if self.start_time is None:
            return
        elapsed = time.time() - self.start_time
        self.times.append(elapsed)
        if message:
            print(f"Time taken {message}: {elapsed:.3f}s")
        return elapsed

    def average(self):
        return sum(self.times) / len(self.times) if self.times else 0

    def total(self):
        return sum(self.times)


class PerformanceLogger:
    """Track and analyze agent performance"""
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.episode_data = []
        self.step_data = []
        
    def log_episode(self, episode: int, reward: float, steps: int, 
                   final_x_pos: int, final_score: int, completed: bool):
        """Log episode-level metrics"""
        data = {
            'episode': episode,
            'timestamp': time.time(),
            'reward': reward,
            'steps': steps,
            'final_x_pos': final_x_pos,
            'final_score': final_score,
            'completed': completed,
            'x_progress_rate': final_x_pos / steps if steps > 0 else 0
        }
        self.episode_data.append(data)
        
    def log_step(self, episode: int, step: int, action: int, reward: float,
                x_pos: int, score: int, reasoning: str = ''):
        """Log step-level metrics"""
        data = {
            'episode': episode,
            'step': step,
            'timestamp': time.time(),
            'action': action,
            'reward': reward,
            'x_pos': x_pos,
            'score': score,
            'reasoning': reasoning
        }
        self.step_data.append(data)
        
    def save_logs(self, filename_prefix='mario_logs'):
        """Save logs to JSON files"""
        timestamp = get_current_date_time_string()
        
        episode_file = os.path.join(self.log_dir, f"{filename_prefix}_episodes_{timestamp}.json")
        with open(episode_file, 'w') as f:
            json.dump(self.episode_data, f, indent=2)
        
        if self.step_data:
            step_file = os.path.join(self.log_dir, f"{filename_prefix}_steps_{timestamp}.json")
            with open(step_file, 'w') as f:
                json.dump(self.step_data, f, indent=2)
        
        print(f"Logs saved to {episode_file}")
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.episode_data:
            return {}
        
        rewards = [ep['reward'] for ep in self.episode_data]
        steps = [ep['steps'] for ep in self.episode_data]
        x_positions = [ep['final_x_pos'] for ep in self.episode_data]
        completion_rate = sum(ep['completed'] for ep in self.episode_data) / len(self.episode_data)
        
        return {
            'total_episodes': len(self.episode_data),
            'avg_reward': np.mean(rewards),
            'max_reward': max(rewards),
            'min_reward': min(rewards),
            'avg_steps': np.mean(steps),
            'avg_x_position': np.mean(x_positions),
            'max_x_position': max(x_positions),
            'completion_rate': completion_rate,
            'total_runtime': time.time() - self.episode_data[0]['timestamp'] if self.episode_data else 0
        }


def analyze_action_distribution(memory: List[Dict]) -> Dict[str, Any]:
    """Analyze distribution of actions taken"""
    if not memory:
        return {}
    
    actions = [item['action'] for item in memory if 'action' in item]
    action_counts = {}
    for action in actions:
        action_counts[action] = action_counts.get(action, 0) + 1
    
    total_actions = len(actions)
    action_percentages = {action: count/total_actions * 100 
                         for action, count in action_counts.items()}
    
    return {
        'total_actions': total_actions,
        'action_counts': action_counts,
        'action_percentages': action_percentages,
        'most_common_action': max(action_counts, key=action_counts.get) if action_counts else None
    }


def print_progress_bar(current: int, total: int, prefix: str = '', suffix: str = '', 
                      decimals: int = 1, length: int = 50, fill: str = '█'):
    """Print a progress bar"""
    percent = ("{0:." + str(decimals) + "f}").format(100 * (current / float(total)))
    filled_length = int(length * current // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if current == total:
        print()


def save_screenshot_with_metadata(screenshot_pil, game_state: Dict, 
                                 action: int, reasoning: str, 
                                 save_dir: str, filename: str):
    """Save screenshot with game state metadata"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save image
    img_path = os.path.join(save_dir, f"{filename}.png")
    screenshot_pil.save(img_path)
    
    # Save metadata
    metadata = {
        'filename': f"{filename}.png",
        'timestamp': time.time(),
        'game_state': game_state,
        'action': action,
        'reasoning': reasoning
    }
    
    meta_path = os.path.join(save_dir, f"{filename}_meta.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2) 