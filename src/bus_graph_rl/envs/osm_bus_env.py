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
    """Config de l'env OSM (issu du notebook, puis nettoyé)."""
    area_name: str = "Toulouse"
    area_point: Optional[Tuple[float, float]] = None  # (lat, lon)
    area_distance: Optional[int] = None               # meters
    n_bus_stops: int = 100
    reward_stop_success: float = 10.0
    reward_default: float = -1.0
    max_steps: int = 500

class OSMBusEnv(gym.Env):
    """Env minimal : le bus se déplace sur un graphe OSM et décide 'stop ou pas'.

    Attention : c'est une base de refactor. L'agent Q-learning du notebook utilise
    `distance_to_stop` comme état discret (à améliorer).
    """

    metadata = {"render_modes": []}

    def __init__(self, config: BusEnvConfig = BusEnvConfig()):
        super().__init__()
        self.config = config

        spec = OSMGraphSpec(
            area_name=config.area_name,
            area_point=config.area_point,
            area_distance=config.area_distance,
        )
        self.graph = load_osm_graph(spec)

        # Tag random bus stops
        nodes = list(self.graph.nodes())
        k = min(config.n_bus_stops, len(nodes))
        for node in random.sample(nodes, k=k):
            self.graph.nodes[node]["bus_stop"] = True

        self.start_node: int | None = None
        self.end_node: int | None = None
        self.passenger_on = False
        self.passenger_off = False
        self.steps = 0

        # Action: 0 = do nothing, 1 = stop if current node is a bus_stop
        self.action_space = spaces.Discrete(2)

        self.observation_space = spaces.Dict({
            "passenger_on": spaces.Discrete(2),
            "passenger_off": spaces.Discrete(2),
            # note: le notebook utilisait un scalaire (pas un array)
            "distance_to_stop": spaces.Box(low=0.0, high=np.finfo(np.float32).max, shape=(), dtype=np.float32),
        })

    def _distance_to_stop(self) -> float:
        assert self.start_node is not None and self.end_node is not None
        try:
            d = nx.shortest_path_length(
                self.graph,
                source=self.start_node,
                target=self.end_node,
                weight="travel_time",
            )
            return float(d)
        except Exception:
            return float("inf")

    def _get_obs(self) -> Dict[str, Any]:
        return {
            "passenger_on": int(self.passenger_on),
            "passenger_off": int(self.passenger_off),
            "distance_to_stop": np.float32(self._distance_to_stop()),
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if _GYMNASIUM:
            super().reset(seed=seed)

        self.steps = 0
        self.passenger_on = False
        self.passenger_off = False

        # pick 2 distinct bus stop nodes
        stop_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("bus_stop", False)]
        if len(stop_nodes) < 2:
            raise RuntimeError("Not enough bus_stop nodes; increase area or lower n_bus_stops.")
        self.start_node, self.end_node = np.random.choice(stop_nodes, size=2, replace=False).tolist()

        obs = self._get_obs()
        info = {}
        return (obs, info) if _GYMNASIUM else obs

    def step(self, action: int):
        assert self.start_node is not None and self.end_node is not None
        self.steps += 1

        # action effect
        if action == 1 and self.graph.nodes[self.start_node].get("bus_stop", False):
            self.passenger_on = True
            self.passenger_off = True

        # reward
        reward = self.config.reward_stop_success if (self.passenger_on and self.passenger_off) else self.config.reward_default

        terminated = (self.start_node == self.end_node)
        truncated = (self.steps >= self.config.max_steps)

        # move to one neighbor (very naive, copied from notebook spirit)
        neighbors = list(nx.neighbors(self.graph, self.start_node))
        if neighbors:
            self.start_node = neighbors[0]

        obs = self._get_obs()
        info = {}

        if _GYMNASIUM:
            return obs, float(reward), bool(terminated), bool(truncated), info
        else:  # gym legacy
            done = bool(terminated or truncated)
            return obs, float(reward), done, info

    # Convenience helpers (from notebook)
    def num_bus_stops(self) -> int:
        return sum(1 for _, d in self.graph.nodes(data=True) if d.get("bus_stop", False))
