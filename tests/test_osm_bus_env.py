from __future__ import annotations

import networkx as nx
import numpy as np

from bus_graph_rl.envs.osm_bus_env import BusEnvConfig, OSMBusEnv


def build_test_graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    graph.add_nodes_from([1, 2, 3, 4])

    edges = [
        (1, 2),
        (2, 1),
        (2, 3),
        (3, 2),
        (3, 4),
        (4, 3),
    ]
    for source, target in edges:
        graph.add_edge(source, target, travel_time=1.0)

    return graph


def make_env(monkeypatch) -> OSMBusEnv:
    monkeypatch.setattr(
        "bus_graph_rl.envs.osm_bus_env.load_osm_graph",
        lambda _spec: build_test_graph(),
    )
    return OSMBusEnv(BusEnvConfig(area_name="Test", n_bus_stops=4, max_steps=10))


def test_reset_returns_masked_observation(monkeypatch):
    env = make_env(monkeypatch)

    obs, info = env.reset(seed=123)

    assert obs["action_mask"].shape == (env.action_space.n,)
    assert obs["action_mask"][0] == 1
    assert info["pickup_node"] != info["dropoff_node"]
    assert np.isfinite(obs["distance_to_target"])


def test_service_pickup_and_dropoff(monkeypatch):
    env = make_env(monkeypatch)
    env.reset(seed=123)

    env.current_node = env.pickup_node
    obs, reward, terminated, truncated, _info = env.step(0)

    assert reward == env.config.reward_pickup_success
    assert obs["passenger_on"] == 1
    assert not terminated
    assert not truncated

    env.current_node = env.dropoff_node
    env.passenger_on = True
    env.passenger_off = False
    obs, reward, terminated, truncated, _info = env.step(0)

    assert reward == env.config.reward_dropoff_success
    assert obs["passenger_off"] == 1
    assert terminated
    assert not truncated


def test_invalid_move_is_penalized(monkeypatch):
    env = make_env(monkeypatch)
    env.reset(seed=123)

    _obs, reward, terminated, truncated, _info = env.step(env.action_space.n + 2)

    assert reward == env.config.reward_invalid_move
    assert not terminated
    assert not truncated
