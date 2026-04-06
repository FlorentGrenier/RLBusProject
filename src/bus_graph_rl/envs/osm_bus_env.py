from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import random
import numpy as np
import networkx as nx

try:
    import gymnasium as gym
    from gymnasium import spaces

    _GYMNASIUM = True
except Exception:
    import gym
    from gym import spaces

    _GYMNASIUM = False

from ..graph.loader import OSMGraphSpec, load_osm_graph


@dataclass
class BusEnvConfig:
    """Configuration for the OSM bus environment."""

    area_name: str = "Toulouse"
    area_point: Optional[Tuple[float, float]] = None
    area_distance: Optional[int] = None
    n_bus_stops: int = 100
    reward_pickup_success: float = 10.0
    reward_dropoff_success: float = 25.0
    reward_move: float = -1.0
    reward_invalid_stop: float = -2.0
    reward_invalid_move: float = -3.0
    reward_revisit: float = -0.5
    max_steps: int = 500


class OSMBusEnv(gym.Env):
    """Simple pickup/dropoff environment over an OSM graph."""

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[BusEnvConfig] = None):
        super().__init__()
        self.config = config or BusEnvConfig()

        spec = OSMGraphSpec(
            area_name=self.config.area_name,
            area_point=self.config.area_point,
            area_distance=self.config.area_distance,
        )
        self.graph = load_osm_graph(spec)

        nodes = list(self.graph.nodes())
        k = min(self.config.n_bus_stops, len(nodes))
        for node in random.sample(nodes, k=k):
            self.graph.nodes[node]["bus_stop"] = True

        self.max_neighbors = max((self.graph.out_degree(node) for node in self.graph.nodes()), default=0)
        self.current_node: int | None = None
        self.pickup_node: int | None = None
        self.dropoff_node: int | None = None
        self.passenger_on = False
        self.passenger_off = False
        self.steps = 0
        self.visited_nodes: set[int] = set()

        self.action_space = spaces.Discrete(self.max_neighbors + 1)
        self.observation_space = spaces.Dict(
            {
                "passenger_on": spaces.Discrete(2),
                "passenger_off": spaces.Discrete(2),
                "current_node_is_stop": spaces.Discrete(2),
                "distance_to_target": spaces.Box(
                    low=0.0,
                    high=np.finfo(np.float32).max,
                    shape=(),
                    dtype=np.float32,
                ),
                "action_mask": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.action_space.n,),
                    dtype=np.int8,
                ),
            }
        )

    def _target_node(self) -> int | None:
        if not self.passenger_on:
            return self.pickup_node
        if not self.passenger_off:
            return self.dropoff_node
        return None

    def _distance_to_target(self) -> float:
        assert self.current_node is not None
        target = self._target_node()
        if target is None:
            return 0.0
        try:
            distance = nx.shortest_path_length(
                self.graph,
                source=self.current_node,
                target=target,
                weight="travel_time",
            )
            return float(distance)
        except Exception:
            return float("inf")

    def _neighbors(self) -> list[int]:
        assert self.current_node is not None
        return sorted(set(self.graph.successors(self.current_node)))

    def _action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        mask[0] = 1
        for idx, _neighbor in enumerate(self._neighbors(), start=1):
            if idx >= self.action_space.n:
                break
            mask[idx] = 1
        return mask

    def _get_obs(self) -> Dict[str, Any]:
        assert self.current_node is not None
        return {
            "passenger_on": int(self.passenger_on),
            "passenger_off": int(self.passenger_off),
            "current_node_is_stop": int(self.graph.nodes[self.current_node].get("bus_stop", False)),
            "distance_to_target": np.float32(self._distance_to_target()),
            "action_mask": self._action_mask(),
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if _GYMNASIUM:
            super().reset(seed=seed)

        self.steps = 0
        self.passenger_on = False
        self.passenger_off = False
        self.visited_nodes = set()

        stop_nodes = [node for node, data in self.graph.nodes(data=True) if data.get("bus_stop", False)]
        if len(stop_nodes) < 2:
            raise RuntimeError("Not enough bus_stop nodes; increase area or lower n_bus_stops.")

        self.pickup_node, self.dropoff_node = np.random.choice(stop_nodes, size=2, replace=False).tolist()
        start_candidates = [node for node in self.graph.nodes() if node != self.dropoff_node]
        self.current_node = random.choice(start_candidates)
        self.visited_nodes.add(self.current_node)

        obs = self._get_obs()
        info = {
            "pickup_node": self.pickup_node,
            "dropoff_node": self.dropoff_node,
            "current_node": self.current_node,
        }
        return (obs, info) if _GYMNASIUM else obs

    def step(self, action: int):
        assert self.current_node is not None
        assert self.pickup_node is not None
        assert self.dropoff_node is not None

        self.steps += 1
        reward = 0.0
        terminated = False

        if action == 0:
            if (not self.passenger_on) and self.current_node == self.pickup_node:
                self.passenger_on = True
                reward = self.config.reward_pickup_success
            elif self.passenger_on and (not self.passenger_off) and self.current_node == self.dropoff_node:
                self.passenger_off = True
                reward = self.config.reward_dropoff_success
                terminated = True
            else:
                reward = self.config.reward_invalid_stop
        else:
            neighbors = self._neighbors()
            neighbor_idx = action - 1
            if 0 <= neighbor_idx < len(neighbors):
                next_node = neighbors[neighbor_idx]
                revisit_penalty = self.config.reward_revisit if next_node in self.visited_nodes else 0.0
                self.current_node = next_node
                self.visited_nodes.add(self.current_node)
                reward = self.config.reward_move + revisit_penalty
            else:
                reward = self.config.reward_invalid_move

        truncated = self.steps >= self.config.max_steps
        obs = self._get_obs()
        info = {
            "pickup_node": self.pickup_node,
            "dropoff_node": self.dropoff_node,
            "current_node": self.current_node,
        }

        if _GYMNASIUM:
            return obs, float(reward), bool(terminated), bool(truncated), info

        done = bool(terminated or truncated)
        return obs, float(reward), done, info

    def num_bus_stops(self) -> int:
        return sum(1 for _, data in self.graph.nodes(data=True) if data.get("bus_stop", False))
