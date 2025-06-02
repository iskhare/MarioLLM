import numpy as np
from gym import Wrapper
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack


class SkipFrame(Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self.skip):
            # Because we’ve already applied StepAPICompatibilityWrapper,
            # self.env.step(action) will return exactly 5 values:
            # (next_state, reward, terminated, truncated, info)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            done = terminated or truncated
            if done:
                break
        # Return five values again, so the next wrapper still sees a 5-tuple:
        return next_state, total_reward, done, False, info
        # Note: we collapse “terminated/truncated” into one `done` boolean (for simplicity).
        # The fifth slot is still `info` so that every wrapper sees 5 outputs.


class ResetWrapper(Wrapper):
    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}
        return obs, info


class StepAPICompatibilityWrapper(Wrapper):
    """
    Wraps an environment whose .step() returns a 4-tuple into one that always returns
    a 5-tuple (obs, reward, terminated, truncated, info).  This must come before
    any other wrapper that expects a 5-tuple.
    """
    def step(self, action):
        result = self.env.step(action)
        if len(result) == 4:
            obs, reward, done, info = result
            terminated = done
            truncated = False
            return obs, reward, terminated, truncated, info
        else:
            # Already a 5-tuple
            return result


def apply_wrappers(env):
    """
    1) StepAPICompatibilityWrapper first → forces every .step() to be 5-tuple
    2) ResetWrapper next → ensures reset() always returns (obs, info)
    3) SkipFrame next  → now skip-frames will also return 5 values
    4) Then GrayScaleObservation → returns a single-channel frame, passes along 5-tuple
    5) Then FrameStack → stacks 4 gray frames into shape (4, H, W), still returns 5-tuple
    """
    # 1) Guarantee step() → 5-tuple
    env = StepAPICompatibilityWrapper(env)

    # 2) Guarantee reset() → obs, info
    env = ResetWrapper(env)

    # 3) Skip 4 frames per action; SkipFrame.step must output 5 values
    env = SkipFrame(env, skip=4)

    env = ResizeObservation(env, shape=84)  # Resize frame from 240x256 to 84x84

    # 4) Convert each frame to grayscale (still a 5-tuple at every step)
    env = GrayScaleObservation(env)

    # 5) Stack 4 consecutive gray frames → shape (4, H, W), still 5-tuple
    env = FrameStack(env, num_stack=4, lz4_compress=True)

    return env
