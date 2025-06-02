# agents.py

import numpy as np

# Under RIGHT_ONLY:
#   0 = NOOP
#   1 = RIGHT
#   2 = RIGHT + JUMP
#   3 = RIGHT + RUN
#   4 = RIGHT + RUN + JUMP
RIGHT_ACTION_INDEX = 1
RIGHT_JUMP_INDEX = 2

# Weight for x-position delta to encourage forward progress
X_BONUS_WEIGHT = 0.01


class Random_Agent:
    def __init__(self, num_actions: int):
        self.num_actions = num_actions

    def choose_action(self, observation=None):
        return np.random.randint(self.num_actions)



class Lookahead_Agent:
    def __init__(self, make_sim_env_fn, live_env, num_actions: int):
        """
        A depth-2 lookahead agent that uses immediate reward and a small x‐position bonus
        to avoid obstacles. Internally, we “deep unwrap” to access the NES core’s _x_position.
        """
        self.make_sim_env_fn = make_sim_env_fn
        self.live_env = live_env
        self.num_actions = num_actions
        self.action_history = []

    def record_action(self, action: int):
        self.action_history.append(action)

    def choose_and_update(self, observation=None):
        best_score = -np.inf
        tie_candidates = []

        # First-ply loop: evaluate each a0
        for a0 in range(self.num_actions):
            sim0 = self._make_sim_and_replay_history()
            nes0 = self._get_nes_core(sim0)
            x0_before = nes0._x_position

            _, r0, done0, trunc0, _ = sim0.step(a0)
            nes0_after = self._get_nes_core(sim0)
            x0_after = nes0_after._x_position

            if done0 or trunc0 or r0 < 0:
                score0 = -np.inf
            else:
                base_score0 = float(r0) + X_BONUS_WEIGHT * (x0_after - x0_before)

                # Second-ply: evaluate each a1
                best1 = -np.inf
                for a1 in range(self.num_actions):
                    sim1 = self._make_sim_and_replay_history_with_extra(a0)
                    nes1 = self._get_nes_core(sim1)
                    x1_before = nes1._x_position

                    _, r1, done1, trunc1, _ = sim1.step(a1)
                    nes1_after = self._get_nes_core(sim1)
                    x1_after = nes1_after._x_position

                    if done1 or trunc1 or r1 < 0:
                        child_score = -np.inf
                    else:
                        child_score = float(r1) + X_BONUS_WEIGHT * (x1_after - x1_before)

                    if child_score > best1:
                        best1 = child_score

                score0 = base_score0 + (best1 if best1 > -np.inf else 0.0)

            if score0 > best_score:
                best_score = score0
                tie_candidates = [a0]
            elif score0 == best_score:
                tie_candidates.append(a0)

        # Tie-breaking
        if best_score == -np.inf:
            chosen = 0  # NOOP
        else:
            simR = self._make_sim_and_replay_history()
            nesR = self._get_nes_core(simR)
            xR_before = nesR._x_position

            _, rR, doneR, truncR, _ = simR.step(RIGHT_ACTION_INDEX)
            nesR_after = self._get_nes_core(simR)
            xR_after = nesR_after._x_position

            right_safe_and_advances = not(doneR or truncR or rR < 0) and (xR_after - xR_before) > 0

            if right_safe_and_advances and (RIGHT_ACTION_INDEX in tie_candidates):
                chosen = RIGHT_ACTION_INDEX
            else:
                if (not right_safe_and_advances) and (RIGHT_JUMP_INDEX in tie_candidates):
                    chosen = RIGHT_JUMP_INDEX
                else:
                    chosen = tie_candidates[0]

        new_state, reward, done, truncated, info = self.live_env.step(chosen)
        self.record_action(chosen)
        return new_state, reward, done, truncated, info

    def _make_sim_and_replay_history(self):
        sim = self.make_sim_env_fn()
        state, info = sim.reset()
        for a in self.action_history:
            state, reward, done, truncated, info = sim.step(a)
            if done or truncated:
                break
        return sim

    def _make_sim_and_replay_history_with_extra(self, extra_action):
        sim = self.make_sim_env_fn()
        state, info = sim.reset()
        done_flag = False
        for a in self.action_history:
            state, reward, done, truncated, info = sim.step(a)
            if done or truncated:
                done_flag = True
                break
        if done_flag:
            return sim
        sim.step(extra_action)
        return sim

    def _get_nes_core(self, env):
        """
        Recursively unwrap env.env or env.unwrapped until we find
        the NES core that has attribute _x_position.
        """
        obj = env
        while True:
            if hasattr(obj, "_x_position"):
                return obj
            if hasattr(obj, "env"):
                obj = obj.env
                continue
            if hasattr(obj, "unwrapped"):
                inner = obj.unwrapped
                if inner is obj:
                    break
                obj = inner
                continue
            break
        raise AttributeError("Could not find NES core with _x_position")
