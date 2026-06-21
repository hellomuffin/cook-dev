"""The CookSim simulation core.

``KitchenGame`` owns the full mutable world state and implements a single
deterministic ``step`` over all cooks' actions. It is renderer- and
framework-agnostic; the Gym/PettingZoo wrappers and the WebSocket server are
thin shells around it.
"""
from __future__ import annotations

import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

from .config import GameConfig
from .enums import (
    DIRECTION_VECTORS,
    MOVE_ACTIONS,
    SOURCE_ITEMS,
    Action,
    Direction,
    Terrain,
)
from .items import Ingredient, Plate
from .layout import Layout
from .orders import OrderManager
from .player import Cook
from .recipes import RecipeBook, feasible_recipes
from .stations import CookStation, CuttingBoard, Sink

Pos = Tuple[int, int]


class KitchenGame:
    def __init__(
        self,
        layout: Layout,
        config: Optional[GameConfig] = None,
        recipe_book: Optional[RecipeBook] = None,
        n_players: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        self.layout = layout
        self.config = config or GameConfig()
        self.book = recipe_book or RecipeBook()
        self.n_players = n_players or max(1, len(layout.start_positions))
        self.rng = random.Random(seed)
        self.tick = 0
        self.score = 0.0
        self.stats: Dict[str, int] = Counter()
        self.agent_rewards: List[float] = [0.0] * self.n_players

        self.cooks: List[Cook] = []
        self.stations: Dict[Pos, object] = {}
        self.counter_items: Dict[Pos, object] = {}
        self.feasible = feasible_recipes(self.book.recipes, layout)
        self.orders = OrderManager(
            self.book,
            self.rng,
            max_orders=self.config.max_orders,
            base_time=self.config.order_base_time,
            time_per_difficulty=self.config.order_time_per_difficulty,
            expire_penalty=self.config.expire_penalty,
            enabled=self.config.orders_enabled,
            recipes=self.feasible,
        )
        self._build()

    # ------------------------------------------------------------------ setup
    def _build(self) -> None:
        cfg = self.config
        for y, row in enumerate(self.layout.grid):
            for x, terrain in enumerate(row):
                if terrain == Terrain.POT:
                    self.stations[(x, y)] = CookStation(
                        cfg.pot_capacity, cfg.pot_cook_time, cfg.pot_burn_time
                    )
                elif terrain == Terrain.PAN:
                    self.stations[(x, y)] = CookStation(
                        cfg.pan_capacity, cfg.pan_cook_time, cfg.pan_burn_time
                    )
                elif terrain == Terrain.OVEN:
                    self.stations[(x, y)] = CookStation(
                        cfg.oven_capacity, cfg.oven_cook_time, cfg.oven_burn_time
                    )
                elif terrain == Terrain.CUTTING_BOARD:
                    self.stations[(x, y)] = CuttingBoard(cfg.chop_time)
                elif terrain == Terrain.SINK:
                    self.stations[(x, y)] = Sink(cfg.wash_time)

        starts = list(self.layout.start_positions)
        # Fallback spawns if the layout under-specifies them.
        if len(starts) < self.n_players:
            for y, row in enumerate(self.layout.grid):
                for x, t in enumerate(row):
                    if t == Terrain.FLOOR and (x, y) not in starts:
                        starts.append((x, y))
                        if len(starts) >= self.n_players:
                            break
                if len(starts) >= self.n_players:
                    break
        for i in range(self.n_players):
            x, y = starts[i]
            self.cooks.append(Cook(i, x, y, Direction.SOUTH))

    def _first_sink(self) -> Optional[Sink]:
        for s in self.stations.values():
            if isinstance(s, Sink):
                return s
        return None

    # ------------------------------------------------------------------ step
    def step(self, actions: List[int]) -> float:
        """Advance one tick. ``actions`` is one Action per cook. Returns the
        total (sparse + shaped) reward produced this tick."""
        assert len(actions) == len(self.cooks), "one action per cook required"
        actions = [Action(a) for a in actions]
        # Per-cook reward attribution (interaction rewards go to the actor;
        # team-level order penalties are split evenly).
        self.agent_rewards = [0.0] * len(self.cooks)

        # 1) Movement / facing -------------------------------------------
        self._resolve_movement(actions)

        # 2) Interactions ------------------------------------------------
        for i, (cook, action) in enumerate(zip(self.cooks, actions)):
            if action == Action.INTERACT:
                self.agent_rewards[i] += self._interact(cook)
            elif action in MOVE_ACTIONS:
                cook.last_action = "walk"
            elif action == Action.STAY:
                cook.last_action = "idle"

        # 3) Station + order timers --------------------------------------
        for station in self.stations.values():
            if isinstance(station, CookStation):
                station.tick()
        team_penalty = self.orders.tick()
        if team_penalty and self.cooks:
            share = team_penalty / len(self.cooks)
            self.agent_rewards = [r + share for r in self.agent_rewards]

        reward = sum(self.agent_rewards)
        self.tick += 1
        self.score += reward
        return reward

    def _resolve_movement(self, actions: List[Action]) -> None:
        pos = {c.id: c.pos for c in self.cooks}
        target: Dict[int, Pos] = dict(pos)
        for cook, action in zip(self.cooks, actions):
            if action in MOVE_ACTIONS:
                cook.direction = MOVE_ACTIONS[action]
                dx, dy = DIRECTION_VECTORS[cook.direction]
                nx, ny = cook.x + dx, cook.y + dy
                if self._walkable(nx, ny):
                    target[cook.id] = (nx, ny)

        # Cancel direct swaps (two cooks exchanging cells).
        ids = [c.id for c in self.cooks]
        for i in ids:
            for j in ids:
                if i < j and target[i] == pos[j] and target[j] == pos[i] and target[i] != pos[i]:
                    target[i], target[j] = pos[i], pos[j]

        # Iteratively cancel contested moves until stable.
        for _ in range(len(self.cooks) + 1):
            counts = Counter(target.values())
            changed = False
            for i in ids:
                if target[i] == pos[i]:
                    continue
                contested = counts[target[i]] > 1
                blocked = any(
                    j != i and target[j] == pos[j] == target[i] for j in ids
                )
                if contested or blocked:
                    target[i] = pos[i]
                    changed = True
            if not changed:
                break

        by_id = {c.id: c for c in self.cooks}
        for cid, (tx, ty) in target.items():
            by_id[cid].x, by_id[cid].y = tx, ty

    def _walkable(self, x: int, y: int) -> bool:
        return (
            self.layout.in_bounds(x, y)
            and self.layout.grid[y][x] == Terrain.FLOOR
        )

    # ------------------------------------------------------------- interaction
    def _interact(self, cook: Cook) -> float:
        fx, fy = cook.facing_pos()
        if not self.layout.in_bounds(fx, fy):
            cook.last_action = "idle"
            return 0.0
        terrain = self.layout.grid[fy][fx]
        cook.last_action = "interact"
        pos = (fx, fy)

        if terrain in SOURCE_ITEMS:
            return self._interact_source(cook, SOURCE_ITEMS[terrain])
        if terrain == Terrain.PLATE_SOURCE:
            return self._interact_plate_source(cook)
        if terrain == Terrain.COUNTER:
            return self._interact_counter(cook, pos)
        if terrain in (Terrain.POT, Terrain.PAN, Terrain.OVEN):
            return self._interact_cook_station(cook, pos)
        if terrain == Terrain.CUTTING_BOARD:
            return self._interact_cutting_board(cook, pos)
        if terrain == Terrain.SINK:
            return self._interact_sink(cook, pos)
        if terrain == Terrain.SERVING:
            return self._interact_serving(cook)
        if terrain == Terrain.TRASH:
            return self._interact_trash(cook)
        return 0.0

    def _interact_source(self, cook: Cook, name: str) -> float:
        if cook.holding is None:
            cook.holding = Ingredient(name)
            self.stats["pickup_ingredient"] += 1
            return 0.0
        if isinstance(cook.holding, Plate) and not cook.holding.dirty:
            if len(cook.holding.contents) < self.config.plate_capacity:
                cook.holding.contents.append(Ingredient(name))
                return 0.0
        return 0.0

    def _interact_plate_source(self, cook: Cook) -> float:
        if cook.holding is None:
            cook.holding = Plate()
            self.stats["pickup_plate"] += 1
            return self.config.shaped("plate_pickup")
        return 0.0

    def _interact_counter(self, cook: Cook, pos: Pos) -> float:
        on_counter = self.counter_items.get(pos)
        held = cook.holding
        # Place held item on an empty counter.
        if held is not None and on_counter is None:
            self.counter_items[pos] = held
            cook.holding = None
            return 0.0
        # Pick up from counter with empty hands.
        if held is None and on_counter is not None:
            cook.holding = on_counter
            del self.counter_items[pos]
            return 0.0
        # Combine ingredient + plate in either direction.
        if held is not None and on_counter is not None:
            plate, ing = self._as_plate_and_ingredient(held, on_counter)
            if plate is not None and not plate.dirty and len(plate.contents) < self.config.plate_capacity:
                plate.contents.append(ing)
                # Keep whichever object is the plate; remove the ingredient.
                if held is plate:
                    del self.counter_items[pos]
                else:
                    cook.holding = None
                return 0.0
        return 0.0

    @staticmethod
    def _as_plate_and_ingredient(a, b):
        if isinstance(a, Plate) and isinstance(b, Ingredient):
            return a, b
        if isinstance(b, Plate) and isinstance(a, Ingredient):
            return b, a
        return None, None

    def _interact_cook_station(self, cook: Cook, pos: Pos) -> float:
        station: CookStation = self.stations[pos]
        held = cook.holding
        if isinstance(held, Ingredient) and held.state in (held.state.RAW, held.state.CHOPPED):
            if station.add(held):
                cook.holding = None
                self.stats["place_in_pot"] += 1
                return self.config.shaped("place_in_pot")
            return 0.0
        if isinstance(held, Plate) and not held.dirty:
            if station.take_into_plate(held, self.config.plate_capacity):
                self.stats["soup_pickup"] += 1
                return self.config.shaped("soup_pickup")
            return 0.0
        if held is None:
            if station.status == "raw":
                if station.start():
                    self.stats["start_cooking"] += 1
                    return self.config.shaped("start_cooking")
            elif station.status == "burnt":
                station.clear()
                self.stats["burnt_cleared"] += 1
        return 0.0

    def _interact_cutting_board(self, cook: Cook, pos: Pos) -> float:
        board: CuttingBoard = self.stations[pos]
        held = cook.holding
        if isinstance(held, Ingredient) and held.state == held.state.RAW and board.item is None:
            if board.place(held):
                cook.holding = None
            return 0.0
        # Scoop a finished (chopped) item straight onto a carried plate.
        if (
            isinstance(held, Plate) and not held.dirty
            and board.item is not None and board.item.state == board.item.state.CHOPPED
            and len(held.contents) < self.config.plate_capacity
        ):
            held.contents.append(board.take())
            return 0.0
        if held is None and board.item is not None:
            if board.item.state == board.item.state.RAW:
                board.chop()
                if board.item.state == board.item.state.CHOPPED:
                    self.stats["useful_chop"] += 1
                    return self.config.shaped("useful_chop")
                return 0.0
            else:  # chopped -> pick up
                cook.holding = board.take()
        return 0.0

    def _interact_sink(self, cook: Cook, pos: Pos) -> float:
        sink: Sink = self.stations[pos]
        held = cook.holding
        if isinstance(held, Plate) and held.dirty:
            sink.deposit_dirty()
            cook.holding = None
            return 0.0
        if held is None:
            if sink.clean_ready > 0:
                if sink.take_clean():
                    cook.holding = Plate()
            elif sink.dirty > 0:
                sink.wash()
        return 0.0

    def _interact_serving(self, cook: Cook) -> float:
        held = cook.holding
        if not isinstance(held, Plate) or held.dirty or not held.contents:
            return 0.0
        recipe = self.book.match_plate(held)
        if recipe is not None:
            order = self.orders.fulfill(recipe)
            if order is not None:
                reward = recipe.base_reward
                self.stats["deliveries"] += 1
                cook.holding = None
                self._return_plate()
                return reward
        # Invalid / unwanted delivery: discard food, dirty the plate, penalize.
        self.stats["failed_deliveries"] += 1
        held.contents = []
        held.dirty = True
        return self.config.invalid_serve_penalty

    def _return_plate(self) -> None:
        if not self.config.return_dirty_plates:
            return
        sink = self._first_sink()
        if sink is not None:
            sink.dirty += 1

    def _interact_trash(self, cook: Cook) -> float:
        if cook.holding is None:
            return 0.0
        if isinstance(cook.holding, Plate):
            # Keep the (now empty, clean) plate; just bin its contents.
            if cook.holding.contents or cook.holding.dirty:
                cook.holding.contents = []
                cook.holding.dirty = False
                self.stats["trashed"] += 1
                return self.config.trash_penalty
            return 0.0
        cook.holding = None
        self.stats["trashed"] += 1
        return self.config.trash_penalty

    # --------------------------------------------------------------- queries
    @property
    def done(self) -> bool:
        return self.tick >= self.config.horizon

    def render_state(self) -> dict:
        stations = []
        for (x, y), st in self.stations.items():
            terrain = self.layout.grid[y][x]
            stations.append(
                {"x": x, "y": y, "type": terrain.value, **st.render_state()}
            )
        objects = [
            {"x": x, "y": y, "item": item.to_dict()}
            for (x, y), item in self.counter_items.items()
        ]
        return {
            "tick": self.tick,
            "width": self.layout.width,
            "height": self.layout.height,
            "terrain": [[t.value for t in row] for row in self.layout.grid],
            "stations": stations,
            "objects": objects,
            "players": [c.render_state() for c in self.cooks],
            "orders": self.orders.render_state(),
            "score": round(self.score, 2),
            "stats": dict(self.stats),
            "done": self.done,
        }
