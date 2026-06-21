"""Single-agent Gymnasium wrapper.

Controls cook 0; all other cooks are driven by a ``partner_policy`` callable
(defaults to a no-op partner). Useful for ad-hoc / single-agent training and
for the classic "play against a fixed partner" Overcooked setting.
"""
from __future__ import annotations

import random
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..core.config import GameConfig
from ..core.game import KitchenGame
from ..core.layout import Layout, generate_layout
from ..core.recipes import RecipeBook
from ..layouts import load_layout
from . import observation as obs_enc
from .pettingzoo_env import _resolve_layout

PartnerPolicy = Callable[[KitchenGame, int], int]


def noop_partner(game: KitchenGame, cook_id: int) -> int:
    return 5  # STAY


class RandomPartner:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def __call__(self, game: KitchenGame, cook_id: int) -> int:
        return self.rng.randint(0, 5)


class CookSimGymEnv(gym.Env):
    metadata = {"render_modes": ["state", "ansi"]}

    def __init__(
        self,
        layout="cramped_room",
        config: Optional[GameConfig] = None,
        recipe_book: Optional[RecipeBook] = None,
        n_players: Optional[int] = None,
        partner_policy: Optional[PartnerPolicy] = None,
        seed: Optional[int] = None,
        procedural: Optional[dict] = None,
    ):
        super().__init__()
        self._layout_spec = layout
        self._procedural = procedural
        self.config = config or GameConfig()
        self.recipe_book = recipe_book or RecipeBook()
        self._n_players_override = n_players
        self.partner_policy = partner_policy or noop_partner
        self._seed = seed

        self.game = self._new_game(seed)
        self._terrain_cache = obs_enc._static_terrain(self.game)
        g = self.game
        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(0.0, 1.0, shape=obs_enc.grid_shape(g), dtype=np.float32),
                "orders": spaces.Box(0.0, 1.0, shape=obs_enc.orders_shape(g), dtype=np.float32),
                "time": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Discrete(6)

    def _new_game(self, seed) -> KitchenGame:
        if self._procedural is not None:
            layout = generate_layout(seed=seed, **self._procedural)
        else:
            layout = _resolve_layout(self._layout_spec)
        return KitchenGame(
            layout, config=self.config, recipe_book=self.recipe_book,
            n_players=self._n_players_override, seed=seed,
        )

    def _obs(self):
        return {
            "grid": obs_enc.encode_grid(self.game, 0, self._terrain_cache),
            "orders": obs_enc.encode_orders(self.game),
            "time": np.array([self.game.tick / max(1, self.config.horizon)], dtype=np.float32),
        }

    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is None:
            seed = self._seed
        super().reset(seed=seed)
        self.game = self._new_game(seed)
        self._terrain_cache = obs_enc._static_terrain(self.game)
        return self._obs(), {}

    def step(self, action: int):
        acts = [5] * self.game.n_players
        acts[0] = int(action)
        for cid in range(1, self.game.n_players):
            acts[cid] = int(self.partner_policy(self.game, cid))
        reward = self.game.step(acts)
        terminated = False
        truncated = self.game.done
        info = {"score": self.game.score, "stats": dict(self.game.stats)}
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        return self.game.render_state()
