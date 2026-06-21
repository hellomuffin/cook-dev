"""RL environment wrapper tests."""
import numpy as np

from cooksim.envs import CookSimParallelEnv, CookSimGymEnv, RandomPartner


def test_parallel_api():
    from pettingzoo.test import parallel_api_test
    env = CookSimParallelEnv("bistro", seed=0)
    parallel_api_test(env, num_cycles=100)


def test_parallel_obs_shapes_and_bounds():
    env = CookSimParallelEnv("open_kitchen", seed=0)
    obs, _ = env.reset(seed=0)
    for a in env.agents:
        sp = env.observation_space(a)
        assert sp.contains(obs[a]), f"obs out of space for {a}"
        assert obs[a]["grid"].dtype == np.float32


def test_shared_vs_individual_reward():
    shared = CookSimParallelEnv("cramped_room", reward_mode="shared", seed=0)
    shared.reset(seed=0)
    o, r, t, tr, i = shared.step({a: 5 for a in shared.agents})
    assert len(set(r.values())) <= 1  # all equal under shared

    indiv = CookSimParallelEnv("cramped_room", reward_mode="individual", seed=0)
    indiv.reset(seed=0)
    indiv.step({a: 5 for a in indiv.agents})  # just runs


def test_gym_env_episode():
    env = CookSimGymEnv("cramped_room", partner_policy=RandomPartner(0), seed=1)
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)
    done = False
    steps = 0
    while not done and steps < 50:
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        done = term or trunc
        steps += 1
    assert "score" in info


def test_determinism():
    def rollout():
        env = CookSimParallelEnv("cramped_room", seed=42)
        env.reset(seed=42)
        rng = np.random.default_rng(0)
        total = 0.0
        for _ in range(100):
            acts = {a: int(rng.integers(0, 6)) for a in env.possible_agents}
            _, r, _, _, _ = env.step(acts)
            total += sum(r.values())
        return total
    assert rollout() == rollout()
