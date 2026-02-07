from __future__ import annotations

"""Env "grille" (adjacency matrix) - extrait du notebook, encore incomplet.

Le notebook contenait un squelette de BusEnv basé sur:
- une matrice d'adjacence chargée via np.load
- networkx.from_numpy_matrix
- torch_geometric Data

TODO:
- implémenter build_data()
- implémenter reset/step + rewards
"""

import itertools
import numpy as np
import networkx as nx

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover
    import gym  # type: ignore
    from gym import spaces  # type: ignore

try:
    import torch
    from torch_geometric.data import Data
except Exception:  # pragma: no cover
    torch = None
    Data = None

class GridBusEnv(gym.Env):
    def __init__(self, path_to_map: str, fixed_start_node: bool = False):
        state = np.load(path_to_map, allow_pickle=True)
        self.adj, self.obstacles = state[0], state[1].nonzero()[0]

        self.G = nx.from_numpy_matrix(self.adj)
        self.n_nodes = self.adj.shape[0]
        self.valid_nodes = [n for n in range(self.n_nodes) if n not in self.obstacles]

        self.edge_index = None
        self.pos = None
        self.data = None

        if torch is not None:
            self.edge_index = torch.tensor(self.adj.nonzero(), dtype=torch.long)
            self.grid_size = int(np.sqrt(self.n_nodes))
            self.pos = torch.tensor(list(itertools.product(range(self.grid_size), range(self.grid_size))))
            # self.data = self.build_data()  # TODO

        self.fixed_start_node = fixed_start_node
        self.observation_space = spaces.Discrete(self.n_nodes)
        self.action_space = spaces.Discrete(4)  # Left, Up, Right, Down

    # def build_data(self) -> Data:
    #     ...
