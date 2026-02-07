from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np

@dataclass
class QLearningConfig:
    learning_rate: float = 0.1
    discount_factor: float = 0.99
    epsilon: float = 1.0
    epsilon_decay: float = 0.99
    min_epsilon: float = 0.01
    max_distance: int = 1000  # discretisation (héritée du notebook)

class QLearningAgent:
    """Agent Q-learning (table) basé sur l'état 'distance_to_stop'.

    Limites connues (héritées du notebook):
    - état très compressé (une distance)
    - le max_distance est arbitraire
    """

    def __init__(self, env, config: QLearningConfig = QLearningConfig()):
        self.env = env
        self.cfg = config
        num_actions = env.action_space.n
        self.q_table = np.zeros((config.max_distance + 1, num_actions), dtype=np.float32)

    def _state_to_index(self, state: Dict[str, Any]) -> int:
        d = float(state["distance_to_stop"])
        if not np.isfinite(d):
            return self.cfg.max_distance
        return int(np.clip(int(d), 0, self.cfg.max_distance))

    def act(self, state: Dict[str, Any]) -> int:
        if np.random.rand() < self.cfg.epsilon:
            return int(self.env.action_space.sample())
        idx = self._state_to_index(state)
        return int(np.argmax(self.q_table[idx]))

    def train(self, num_episodes: int) -> List[float]:
        rewards: List[float] = []

        for ep in range(num_episodes):
            out = self.env.reset()
            state = out[0] if isinstance(out, tuple) else out  # gymnasium vs gym
            total = 0.0

            done = False
            while not done:
                action = self.act(state)

                step_out = self.env.step(action)
                if len(step_out) == 5:  # gymnasium
                    next_state, reward, terminated, truncated, _info = step_out
                    done = bool(terminated or truncated)
                else:  # gym
                    next_state, reward, done, _info = step_out

                total += float(reward)

                s = self._state_to_index(state)
                ns = self._state_to_index(next_state)

                q_value = self.q_table[s][action]
                next_q_value = float(np.max(self.q_table[ns]))
                td_error = float(reward) + self.cfg.discount_factor * next_q_value - float(q_value)
                self.q_table[s][action] += self.cfg.learning_rate * td_error

                state = next_state

            self.cfg.epsilon = max(self.cfg.min_epsilon, self.cfg.epsilon * self.cfg.epsilon_decay)
            rewards.append(total)

        return rewards
