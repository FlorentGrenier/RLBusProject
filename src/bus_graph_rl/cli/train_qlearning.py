from __future__ import annotations

import argparse

from ..envs.osm_bus_env import OSMBusEnv, BusEnvConfig
from ..agents.qlearning import QLearningAgent, QLearningConfig
from ..utils.seeding import seed_everything

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--area", type=str, default="Toulouse")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)

    env = OSMBusEnv(BusEnvConfig(area_name=args.area))
    agent = QLearningAgent(env, QLearningConfig())
    rewards = agent.train(args.episodes)

    print(f"Done. episodes={args.episodes} last_reward={rewards[-1] if rewards else None} epsilon={agent.cfg.epsilon}")

if __name__ == "__main__":
    main()
