"""Two submission agents, random and template, played under gymnasium as the scorer plays them.

uv run python examples/02_play_agents.py
uv run python examples/02_play_agents.py --task=small
"""

import fire
import gymnasium as gym
from shockbench_flow_gym import play_episode  # also registers the ShockBench/* environments

from sbf_starter import env_id
from sbf_starter.agents import load


def main(task: str = "tiny", episodes: int = 3) -> None:
    """Print both agents' costs on dev episodes 0 .. episodes - 1."""
    env = gym.make(env_id(task))
    agents = {name: load(name) for name in ("random", "mine")}
    for n in range(episodes):
        costs = {name: play_episode(env, cls, n) for name, cls in agents.items()}
        print(f"dev episode {n}: " + ", ".join(f"{k} {v:,.0f} USD" for k, v in costs.items()))
    print("costs only (lower is better): `uv run sbf compare template random` puts them on the score's scale")


if __name__ == "__main__":
    fire.Fire(main)
