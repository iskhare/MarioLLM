import torch
from torchvision import transforms as T
from PIL import Image
import numpy as np
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunk, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape, antialias=True), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation

def create_env(stage_id, action_space, use_wrappers=True, skip_frames=4, obs_size=84, stack_size=4):
    # Initialize Super Mario environment (in v0.26 change render mode to 'human' to see results on the screen)
    if gym.__version__ < '0.26':
        env = gym_super_mario_bros.make(stage_id, new_step_api=True)
    else:
        env = gym_super_mario_bros.make(stage_id, render_mode='rgb', apply_api_compatibility=True)

    # Set up the action space
    env = JoypadSpace(env, action_space)

    if use_wrappers:
        # Apply Wrappers to environment
        env = SkipFrame(env, skip=skip_frames)
        env = GrayScaleObservation(env)
        env = ResizeObservation(env, shape=obs_size)
        if gym.__version__ < '0.26':
            env = FrameStack(env, num_stack=stack_size, new_step_api=True)
        else:
            env = FrameStack(env, num_stack=stack_size)


    return env