"""Multi-channel grid observation encoding for RL agents.

Each agent receives an egocentric-aware observation built from the global
state: a stack of ``H x W`` feature planes plus a compact orders vector and a
scalar time feature. The encoding is fully observable (Overcooked is a
fully-observable Dec-POMDP) but distinguishes *self* from *teammates*.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..core.enums import SOURCE_ITEMS, Terrain
from ..core.game import KitchenGame
from ..core.items import Ingredient, Plate

INGREDIENT_VOCAB = [
    "onion", "tomato", "lettuce", "mushroom", "meat",
    "fish", "bun", "cheese", "rice", "dough", "egg", "potato",
]
ING_INDEX = {n: i for i, n in enumerate(INGREDIENT_VOCAB)}
STATE_INDEX = {"raw": 0, "chopped": 1, "cooked": 2, "burnt": 3}

# Static terrain plane indices.
TERRAIN_PLANES = {
    Terrain.FLOOR: 0,
    Terrain.COUNTER: 1,
    Terrain.PLATE_SOURCE: 3,
    Terrain.POT: 4,
    Terrain.PAN: 5,
    Terrain.CUTTING_BOARD: 6,
    Terrain.SINK: 7,
    Terrain.SERVING: 8,
    Terrain.TRASH: 9,
    Terrain.OVEN: 10,
}
# every ingredient source collapses to plane 2
N_TERRAIN_PLANES = 11

# Dynamic plane layout (offsets after terrain planes).
DYN = {
    "self_pos": 0,
    "self_dir": 1,
    "other_pos": 2,
    "held_ingredient": 3,
    "held_ingredient_state": 4,
    "held_plate": 5,
    "held_plate_contents": 6,
    "counter_ingredient": 7,
    "counter_ingredient_state": 8,
    "counter_plate": 9,
    "counter_plate_contents": 10,
    "pot_count": 11,
    "pot_progress": 12,
    "pot_ready": 13,
    "pot_burnt": 14,
    "board_progress": 15,
    "board_has_item": 16,
    "sink_dirty": 17,
    "sink_clean": 18,
}
N_DYN_PLANES = len(DYN)
N_PLANES = N_TERRAIN_PLANES + N_DYN_PLANES

ORDER_FEATURES = 4  # present, recipe_id_norm, time_left_norm, reward_norm


def _ing_norm(name: str) -> float:
    return (ING_INDEX.get(name, -1) + 1) / (len(INGREDIENT_VOCAB) + 1)


def _state_norm(state: str) -> float:
    return (STATE_INDEX.get(state, 0) + 1) / (len(STATE_INDEX) + 1)


def grid_shape(game: KitchenGame) -> tuple:
    return (N_PLANES, game.layout.height, game.layout.width)


def orders_shape(game: KitchenGame) -> tuple:
    return (game.config.max_orders * ORDER_FEATURES,)


def _static_terrain(game: KitchenGame) -> np.ndarray:
    h, w = game.layout.height, game.layout.width
    planes = np.zeros((N_TERRAIN_PLANES, h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            t = game.layout.grid[y][x]
            if t in SOURCE_ITEMS:
                planes[2, y, x] = 1.0
            else:
                planes[TERRAIN_PLANES.get(t, 0), y, x] = 1.0
    return planes


def encode_grid(game: KitchenGame, agent_id: int, terrain_cache: np.ndarray) -> np.ndarray:
    h, w = game.layout.height, game.layout.width
    dyn = np.zeros((N_DYN_PLANES, h, w), dtype=np.float32)

    def put(key, x, y, val):
        dyn[DYN[key], y, x] = val

    for cook in game.cooks:
        if cook.id == agent_id:
            put("self_pos", cook.x, cook.y, 1.0)
            put("self_dir", cook.x, cook.y, (int(cook.direction) + 1) / 4.0)
            held = cook.holding
            if isinstance(held, Ingredient):
                put("held_ingredient", cook.x, cook.y, _ing_norm(held.name))
                put("held_ingredient_state", cook.x, cook.y, _state_norm(held.state.value))
            elif isinstance(held, Plate):
                put("held_plate", cook.x, cook.y, 1.0)
                put("held_plate_contents", cook.x, cook.y,
                    len(held.contents) / max(1, game.config.plate_capacity))
        else:
            dyn[DYN["other_pos"], cook.y, cook.x] += 1.0

    for (x, y), item in game.counter_items.items():
        if isinstance(item, Ingredient):
            put("counter_ingredient", x, y, _ing_norm(item.name))
            put("counter_ingredient_state", x, y, _state_norm(item.state.value))
        elif isinstance(item, Plate):
            put("counter_plate", x, y, 1.0)
            put("counter_plate_contents", x, y,
                len(item.contents) / max(1, game.config.plate_capacity))

    from ..core.stations import CookStation, CuttingBoard, Sink
    for (x, y), st in game.stations.items():
        if isinstance(st, CookStation):
            put("pot_count", x, y, len(st.contents) / max(1, st.capacity))
            put("pot_progress", x, y, st.render_state()["progress"])
            put("pot_ready", x, y, 1.0 if st.status == "cooked" else 0.0)
            put("pot_burnt", x, y, 1.0 if st.status == "burnt" else 0.0)
        elif isinstance(st, CuttingBoard):
            put("board_progress", x, y, st.render_state()["progress"])
            put("board_has_item", x, y, 1.0 if st.item is not None else 0.0)
        elif isinstance(st, Sink):
            put("sink_dirty", x, y, min(1.0, st.dirty / 4.0))
            put("sink_clean", x, y, min(1.0, st.clean_ready / 4.0))

    return np.concatenate([terrain_cache, dyn], axis=0)


def encode_orders(game: KitchenGame) -> np.ndarray:
    vec = np.zeros((game.config.max_orders * ORDER_FEATURES,), dtype=np.float32)
    recipe_ids = {r.id: i for i, r in enumerate(game.book.recipes)}
    nrec = max(1, len(recipe_ids))
    max_reward = max((r.base_reward for r in game.book.recipes), default=1.0)
    for i, order in enumerate(game.orders.orders[: game.config.max_orders]):
        base = i * ORDER_FEATURES
        vec[base + 0] = 1.0
        vec[base + 1] = (recipe_ids.get(order.recipe.id, 0) + 1) / (nrec + 1)
        vec[base + 2] = order.time_left / max(1, order.time_total)
        vec[base + 3] = order.recipe.base_reward / max(1.0, max_reward)
    return vec
