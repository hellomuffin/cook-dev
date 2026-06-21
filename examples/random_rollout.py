"""Run a random PettingZoo rollout and print the team return."""
from cooksim.envs import CookSimParallelEnv
import numpy as np


def main():
    env = CookSimParallelEnv(layout="open_kitchen", seed=0)
    obs, info = env.reset(seed=0)
    rng = np.random.default_rng(0)
    total = 0.0
    while env.agents:
        actions = {a: int(rng.integers(0, 6)) for a in env.agents}
        obs, rewards, term, trunc, info = env.step(actions)
        total += sum(rewards.values()) / max(1, len(rewards))
    print(f"episode team return (random): {total:.1f}")
    print("stats:", info)


if __name__ == "__main__":
    main()
