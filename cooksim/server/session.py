"""A single live game session driven by the WebSocket server."""
from __future__ import annotations

from typing import Dict, List, Optional

from ..agents import GreedyChef, RandomAgent
from ..core.config import GameConfig
from ..core.enums import Action
from ..core.game import KitchenGame
from ..core.layout import Layout, generate_layout
from ..layouts import load_layout


class GameSession:
    def __init__(self):
        self.game: Optional[KitchenGame] = None
        self.pending: Dict[int, int] = {}          # player -> action (persistent)
        self.bots: Dict[int, object] = {}          # player -> agent callable
        self.tps: float = 7.0
        self.layout_name: str = "cramped_room"
        self.reset(layout="cramped_room")

    def reset(
        self,
        layout=None,
        layout_data: Optional[dict] = None,
        procedural: Optional[dict] = None,
        n_players: Optional[int] = None,
        config: Optional[dict] = None,
        seed: Optional[int] = 0,
        keep_bots: bool = True,
        order_recipes: Optional[list] = None,
    ):
        if layout_data is not None:
            lay = Layout.from_dict(layout_data)
            self.layout_name = lay.name
        elif procedural is not None:
            lay = generate_layout(seed=seed, **procedural)
            self.layout_name = lay.name
        else:
            name = layout or self.layout_name
            lay = load_layout(name)
            self.layout_name = name

        cfg = GameConfig(**config) if config else GameConfig()
        self.game = KitchenGame(lay, config=cfg, n_players=n_players, seed=seed,
                                order_recipe_ids=order_recipes)
        self.pending = {i: int(Action.STAY) for i in range(self.game.n_players)}
        if not keep_bots:
            self.bots = {}
        # drop bots that reference players who no longer exist
        self.bots = {p: b for p, b in self.bots.items() if p < self.game.n_players}

    # -- control ----------------------------------------------------------
    def set_action(self, player: int, action: int):
        if 0 <= player < self.game.n_players:
            self.pending[player] = int(action)

    def add_bot(self, player: int, kind: str = "greedy", role: str = "any"):
        if 0 <= player < self.game.n_players:
            self.bots[player] = (
                GreedyChef(seed=player + 1, role=role) if kind == "greedy"
                else RandomAgent(player)
            )

    def remove_bot(self, player: int):
        self.bots.pop(player, None)

    def tick(self):
        if self.game is None:
            return
        actions: List[int] = []
        for i in range(self.game.n_players):
            if i in self.bots:
                actions.append(int(self.bots[i](self.game, i)))
            else:
                actions.append(self.pending.get(i, int(Action.STAY)))
        self.game.step(actions)

    def state(self) -> dict:
        st = self.game.render_state()
        st["layout_name"] = self.layout_name
        st["bots"] = sorted(self.bots.keys())
        st["tps"] = self.tps
        st["n_players"] = self.game.n_players
        return st
