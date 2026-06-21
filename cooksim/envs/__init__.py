"""RL environment wrappers for CookSim."""
from .gym_env import CookSimGymEnv, RandomPartner, noop_partner
from .pettingzoo_env import CookSimParallelEnv, make_aec_env

__all__ = [
    "CookSimGymEnv",
    "CookSimParallelEnv",
    "make_aec_env",
    "RandomPartner",
    "noop_partner",
]

# Optional Gymnasium registration (no-op if gymnasium changes its API).
try:
    from gymnasium.envs.registration import register

    register(id="CookSim-v0", entry_point="cooksim.envs.gym_env:CookSimGymEnv")
except Exception:  # pragma: no cover
    pass
