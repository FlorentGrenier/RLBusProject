from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class QLearningConfig:
    learning_rate: float = 0.1
    discount_factor: float = 0.99
    epsilon: float = 1.0
    epsilon_decay: float = 0.99
    min_epsilon: float = 0.01
    max_distance: int = 1000


class QLearningAgent:
    """Tabular Q-learning agent with a compact discrete state."""

    def __init__(self, env, config: QLearningConfig | None = None):
        self.env = env
        self.cfg = config or QLearningConfig()
        self.num_actions = env.action_space.n
        self.q_table: Dict[Tuple[int, int, int, int], np.ndarray] = {}

    def _state_to_key(self, state: Dict[str, Any]) -> Tuple[int, int, int, int]:
        distance = float(state["distance_to_target"])
        if not np.isfinite(distance):
            distance_idx = self.cfg.max_distance
        else:
            distance_idx = int(np.clip(int(distance), 0, self.cfg.max_distance))

        return (
            int(state["passenger_on"]),
            int(state["passenger_off"]),
            int(state.get("current_node_is_stop", 0)),
            distance_idx,
        )

    def _q_values(self, state: Dict[str, Any]) -> np.ndarray:
        key = self._state_to_key(state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.num_actions, dtype=np.float32)
        return self.q_table[key]

    def _valid_actions(self, state: Dict[str, Any]) -> List[int]:
        action_mask = state.get("action_mask")
        if action_mask is None:
            return list(range(self.num_actions))

        valid_actions = np.flatnonzero(np.asarray(action_mask)).tolist()
        return valid_actions or [0]

    def act(self, state: Dict[str, Any]) -> int:
        valid_actions = self._valid_actions(state)
        if np.random.rand() < self.cfg.epsilon:
            return int(np.random.choice(valid_actions))

        q_values = self._q_values(state).copy()
        invalid_actions = [idx for idx in range(self.num_actions) if idx not in valid_actions]
        if invalid_actions:
            q_values[invalid_actions] = -np.inf
        return int(np.argmax(q_values))

    def train(self, num_episodes: int) -> List[float]:
        rewards: List[float] = []

        for _episode in range(num_episodes):
            out = self.env.reset()
            state = out[0] if isinstance(out, tuple) else out
            total = 0.0
            done = False

            while not done:
                action = self.act(state)
                step_out = self.env.step(action)
                if len(step_out) == 5:
                    next_state, reward, terminated, truncated, _info = step_out
                    done = bool(terminated or truncated)
                else:
                    next_state, reward, done, _info = step_out

                total += float(reward)

                state_q = self._q_values(state)
                next_state_q = self._q_values(next_state)
                next_valid_actions = self._valid_actions(next_state)
                next_q_value = float(np.max(next_state_q[next_valid_actions])) if next_valid_actions else 0.0
                td_error = float(reward) + self.cfg.discount_factor * next_q_value - float(state_q[action])
                state_q[action] += self.cfg.learning_rate * td_error

                state = next_state

            self.cfg.epsilon = max(self.cfg.min_epsilon, self.cfg.epsilon * self.cfg.epsilon_decay)
            rewards.append(total)

        return rewards
