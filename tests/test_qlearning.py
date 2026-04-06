from __future__ import annotations

from bus_graph_rl.agents.qlearning import QLearningAgent, QLearningConfig


class TinyEnv:
    def __init__(self):
        self.action_space = type("ActionSpace", (), {"n": 2})()
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1
        state = {
            "passenger_on": 0,
            "passenger_off": 0,
            "current_node_is_stop": 1,
            "distance_to_target": 1.0,
            "action_mask": [1, 1],
        }
        return state, {}

    def step(self, action: int):
        reward = 5.0 if action == 1 else -1.0
        next_state = {
            "passenger_on": 1,
            "passenger_off": 0,
            "current_node_is_stop": 1,
            "distance_to_target": 0.0,
            "action_mask": [1, 0],
        }
        return next_state, reward, True, False, {}


def test_qlearning_updates_q_values():
    env = TinyEnv()
    agent = QLearningAgent(
        env,
        QLearningConfig(learning_rate=1.0, discount_factor=0.0, epsilon=0.0, epsilon_decay=1.0),
    )

    rewards = agent.train(1)
    initial_state = {
        "passenger_on": 0,
        "passenger_off": 0,
        "current_node_is_stop": 1,
        "distance_to_target": 1.0,
        "action_mask": [1, 1],
    }

    assert rewards == [-1.0]
    assert agent._q_values(initial_state)[0] == -1.0
    assert env.reset_calls == 1


def test_qlearning_respects_action_mask():
    env = TinyEnv()
    agent = QLearningAgent(env, QLearningConfig(epsilon=0.0))
    state = {
        "passenger_on": 0,
        "passenger_off": 0,
        "current_node_is_stop": 1,
        "distance_to_target": 1.0,
        "action_mask": [1, 0],
    }

    assert agent.act(state) == 0
