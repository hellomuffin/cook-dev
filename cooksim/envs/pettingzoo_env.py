"""PettingZoo ParallelEnv wrapper for CookSim.

This is the canonical multi-agent interface: every cook is an agent acting
simultaneously each step. Overcooked is fully cooperative, so by default all
agents receive the shared team reward.
"""
from __future__ import annotations

import functools
from typing import Dict, List, Optional

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

from ..core.config import GameConfig
from ..core.game import KitchenGame
from ..core.layout import Layout, generate_layout
from ..core.recipes import RecipeBook
from ..layouts import load_layout
from . import observation as obs_enc


def _resolve_layout(layout) -> Layout:
    if isinstance(layout, Layout):
        return layout
    if isinstance(layout, str):
        return load_layout(layout)
    if isinstance(layout, dict):
        return Layout.from_dict(layout)
    raise TypeError(f"Unsupported layout spec: {type(layout)}")


class CookSimParallelEnv(ParallelEnv):
    metadata = {"name": "cooksim_v0", "render_modes": ["state", "ansi"]}

    def __init__(
        self,
        layout="cramped_room",
        config: Optional[GameConfig] = None,
        recipe_book: Optional[RecipeBook] = None,
        n_players: Optional[int] = None,
        reward_mode: str = "shared",   # "shared" | "individual"
        seed: Optional[int] = None,
        procedural: Optional[dict] = None,
    ):
        super().__init__()
        self._layout_spec = layout
        self._procedural = procedural
        self.config = config or GameConfig()
        self.recipe_book = recipe_book or RecipeBook()
        self._n_players_override = n_players
        self.reward_mode = reward_mode
        self._seed = seed

        self.game = self._new_game(seed)
        self.possible_agents = [f"cook_{i}" for i in range(self.game.n_players)]
        self.agents = list(self.possible_agents)
        self._terrain_cache = obs_enc._static_terrain(self.game)
        self._build_spaces()

    # ------------------------------------------------------------- helpers
    def _new_game(self, seed) -> KitchenGame:
        if self._procedural is not None:
            layout = generate_layout(seed=seed, **self._procedural)
        else:
            layout = _resolve_layout(self._layout_spec)
        return KitchenGame(
            layout,
            config=self.config,
            recipe_book=self.recipe_book,
            n_players=self._n_players_override,
            seed=seed,
        )

    def _build_spaces(self):
        g = self.game
        gshape = obs_enc.grid_shape(g)
        oshape = obs_enc.orders_shape(g)
        self._obs_space = spaces.Dict(
            {
                "grid": spaces.Box(0.0, 1.0, shape=gshape, dtype=np.float32),
                "orders": spaces.Box(0.0, 1.0, shape=oshape, dtype=np.float32),
                "time": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            }
        )
        self._act_space = spaces.Discrete(6)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._obs_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._act_space

    def _agent_index(self, agent: str) -> int:
        return int(agent.split("_")[1])

    def _obs(self, agent: str) -> dict:
        idx = self._agent_index(agent)
        return {
            "grid": obs_enc.encode_grid(self.game, idx, self._terrain_cache),
            "orders": obs_enc.encode_orders(self.game),
            "time": np.array(
                [self.game.tick / max(1, self.config.horizon)], dtype=np.float32
            ),
        }

    def _all_obs(self) -> Dict[str, dict]:
        return {a: self._obs(a) for a in self.agents}

    # ------------------------------------------------------------- API
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is None:
            seed = self._seed
        self.game = self._new_game(seed)
        self._terrain_cache = obs_enc._static_terrain(self.game)
        self.agents = list(self.possible_agents)
        infos = {a: {} for a in self.agents}
        return self._all_obs(), infos

    def step(self, actions: Dict[str, int]):
        ordered = [int(actions.get(a, 5)) for a in self.possible_agents]
        self.game.step(ordered)

        if self.reward_mode == "individual":
            rewards = {
                a: float(self.game.agent_rewards[self._agent_index(a)])
                for a in self.agents
            }
        else:
            team = float(sum(self.game.agent_rewards))
            rewards = {a: team for a in self.agents}

        done = self.game.done
        terminations = {a: False for a in self.agents}
        truncations = {a: done for a in self.agents}
        infos = {a: {"score": self.game.score, "stats": dict(self.game.stats)} for a in self.agents}

        obs = self._all_obs()
        if done:
            self.agents = []
        return obs, rewards, terminations, truncations, infos

    def render(self):
        return self.game.render_state()

    def state(self):
        return self.game.render_state()

    def close(self):
        pass


def make_aec_env(**kwargs):
    """Return an AEC-wrapped, sanity-checked version of the parallel env."""
    env = CookSimParallelEnv(**kwargs)
    env = parallel_to_aec(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env
