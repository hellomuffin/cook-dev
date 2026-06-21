"""Scripted agents: a random baseline and a greedy planning chef.

``GreedyChef`` is a goal-directed heuristic that navigates with BFS and runs a
small state machine (gather -> cook -> plate -> serve). It is intentionally
simple — good enough to make solo play fun and to give autonomous demos — and
is NOT meant to be an optimal coordinated policy.
"""
from __future__ import annotations

import random
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from ..core.enums import DIRECTION_VECTORS, Action, Direction, SOURCE_ITEMS, Terrain
from ..core.game import KitchenGame
from ..core.items import Ingredient, Plate
from ..core.stations import CookStation, CuttingBoard

Pos = Tuple[int, int]


class RandomAgent:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def __call__(self, game: KitchenGame, cook_id: int) -> int:
        return self.rng.randint(0, 5)


def _adjacent_floor(game: KitchenGame, x: int, y: int) -> List[Pos]:
    out = []
    for d in Direction:
        dx, dy = DIRECTION_VECTORS[d]
        nx, ny = x + dx, y + dy
        if game._walkable(nx, ny):
            out.append((nx, ny))
    return out


def _dir_between(frm: Pos, to: Pos) -> Optional[Direction]:
    dx, dy = to[0] - frm[0], to[1] - frm[1]
    for d, (vx, vy) in DIRECTION_VECTORS.items():
        if (vx, vy) == (dx, dy):
            return d
    return None


def _bfs_first_step(game: KitchenGame, start: Pos, goals: Set[Pos]) -> Optional[Action]:
    """Action for the first step of a shortest floor path to any goal tile."""
    if start in goals:
        return None
    prev: Dict[Pos, Pos] = {start: start}
    q = deque([start])
    found = None
    while q:
        cur = q.popleft()
        if cur in goals:
            found = cur
            break
        for d in Direction:
            dx, dy = DIRECTION_VECTORS[d]
            nxt = (cur[0] + dx, cur[1] + dy)
            if nxt not in prev and game._walkable(*nxt):
                prev[nxt] = cur
                q.append(nxt)
    if found is None:
        return None
    # walk back to find first step
    node = found
    while prev[node] != start:
        node = prev[node]
    d = _dir_between(start, node)
    return {
        Direction.NORTH: Action.NORTH,
        Direction.SOUTH: Action.SOUTH,
        Direction.EAST: Action.EAST,
        Direction.WEST: Action.WEST,
    }[d] if d is not None else None


_MOVES = [Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST]


class GreedyChef:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self._last_pos: Optional[Pos] = None
        self._stuck = 0
        self._target: Optional[str] = None   # committed recipe id

    def _unstick(self, game: KitchenGame, cook, action: int) -> int:
        """Break deadlocks: if we keep trying to move but never advance, take a
        random walkable step to break the symmetry with another cook."""
        moving = action in (int(Action.NORTH), int(Action.SOUTH),
                            int(Action.EAST), int(Action.WEST))
        if cook.pos == self._last_pos and moving:
            self._stuck += 1
        else:
            self._stuck = 0
        self._last_pos = cook.pos
        if self._stuck >= 3:
            self._stuck = 0
            options = []
            for d in _MOVES:
                dx, dy = DIRECTION_VECTORS[Direction(int(d))]
                if game._walkable(cook.x + dx, cook.y + dy):
                    options.append(int(d))
            if options:
                return self.rng.choice(options)
        return action

    # -- low level: move to a station cell and interact ------------------
    def _go_interact(self, game: KitchenGame, cook, cell: Pos) -> int:
        stand = _adjacent_floor(game, *cell)
        if not stand:
            return Action.STAY
        if cook.pos in stand:
            want = _dir_between(cook.pos, cell)
            if want is not None and cook.direction != want:
                return {
                    Direction.NORTH: Action.NORTH, Direction.SOUTH: Action.SOUTH,
                    Direction.EAST: Action.EAST, Direction.WEST: Action.WEST,
                }[want]
            return Action.INTERACT
        step = _bfs_first_step(game, cook.pos, set(stand))
        if step is None:
            return Action.STAY
        return step

    # -- world queries ---------------------------------------------------
    def _cells(self, game: KitchenGame, terrain: Terrain) -> List[Pos]:
        return [
            (x, y)
            for y, row in enumerate(game.layout.grid)
            for x, t in enumerate(row)
            if t == terrain
        ]

    def _nearest(self, game: KitchenGame, cook, cells: List[Pos]) -> Optional[Pos]:
        if not cells:
            return None
        return min(cells, key=lambda c: abs(c[0] - cook.x) + abs(c[1] - cook.y))

    def _source_for(self, game: KitchenGame, name: str) -> Optional[Terrain]:
        for terrain, n in SOURCE_ITEMS.items():
            if n == name:
                return terrain
        return None

    def _target_order(self, game: KitchenGame, cook_id: int = 0):
        feasible_ids = {r.id for r in game.feasible}
        orders = game.orders.orders
        active = {o.recipe.id for o in orders}
        # Keep the committed target while it is still on the board.
        if self._target and self._target in active and self._target in feasible_ids:
            cand = [o for o in orders if o.recipe.id == self._target]
            return min(cand, key=lambda o: o.time_left)
        # Otherwise pick a fresh target. Order feasible orders by time left
        # (most likely to finish first) and offset by cook id so multiple
        # cooks naturally split up onto different dishes instead of colliding.
        makeable = sorted(
            [o for o in orders if o.recipe.id in feasible_ids],
            key=lambda o: -o.time_left,
        )
        if not makeable:
            self._target = None
            return None
        # de-duplicate by recipe so cooks pick distinct dishes when possible
        seen, distinct = set(), []
        for o in makeable:
            if o.recipe.id not in seen:
                seen.add(o.recipe.id)
                distinct.append(o)
        choice = distinct[cook_id % len(distinct)]
        self._target = choice.recipe.id
        return choice

    # -- main policy -----------------------------------------------------
    def __call__(self, game: KitchenGame, cook_id: int) -> int:
        cook = game.cooks[cook_id]
        action = self._plan(game, cook_id)
        return self._unstick(game, cook, action)

    def _plan(self, game: KitchenGame, cook_id: int) -> int:
        from collections import Counter as _Counter
        cook = game.cooks[cook_id]
        held = cook.holding
        order = self._target_order(game, cook_id)
        recipe = order.recipe if order else (game.feasible[0] if game.feasible else None)
        if recipe is None:
            return int(Action.STAY)

        # --- recipe component needs --------------------------------------
        cooked_need = _Counter(n for n, st in recipe.contents if st == "cooked")
        chop_need = _Counter(n for n, st in recipe.contents if st == "chopped")
        raw_need = _Counter(n for n, st in recipe.contents if st == "raw")

        # --- cook station state ------------------------------------------
        pots = [(c, s) for c, s in game.stations.items() if isinstance(s, CookStation)]
        cooked_ready = [c for c, s in pots if s.status == "cooked"]
        cooking = [c for c, s in pots if s.status == "cooking"]
        burnt = [c for c, s in pots if s.status == "burnt"]
        raw_started = [c for c, s in pots if s.status == "raw" and s.contents]
        # prefer filling a pot that already has ingredients (keep a batch together)
        fillable = sorted([(c, s) for c, s in pots if s.can_add()],
                          key=lambda cs: -len(cs[1].contents))
        fillable_cells = [c for c, s in fillable]
        in_stations = _Counter(ing.name for _, s in pots for ing in s.contents)
        cooked_short = _Counter()
        for name, k in cooked_need.items():
            short = k - in_stations.get(name, 0)
            if short > 0:
                cooked_short[name] = short

        # --- cutting board state -----------------------------------------
        boards = [(c, s) for c, s in game.stations.items() if isinstance(s, CuttingBoard)]
        board_chopped = [(c, s) for c, s in boards if s.item and s.item.state.value == "chopped"]
        board_raw = [(c, s) for c, s in boards if s.item and s.item.state.value == "raw"]
        board_empty = [c for c, s in boards if s.item is None]
        chopped_ready = _Counter(s.item.name for _, s in board_chopped)
        chop_pipeline = chopped_ready + _Counter(s.item.name for _, s in board_raw)
        chop_short = _Counter()
        for name, k in chop_need.items():
            short = k - chop_pipeline.get(name, 0)
            if short > 0:
                chop_short[name] = short

        def go(cell):
            return int(self._go_interact(game, cook, cell)) if cell else int(Action.STAY)

        def source_cell(name):
            t = self._source_for(game, name)
            return self._nearest(game, cook, self._cells(game, t)) if t else None

        def empty_counter():
            return self._nearest(game, cook, [
                c for c in self._cells(game, Terrain.COUNTER) if c not in game.counter_items
            ])

        # ================= holding a plate (assembly workspace) ===========
        if isinstance(held, Plate):
            if held.dirty:
                sink = self._nearest(game, cook, self._cells(game, Terrain.SINK))
                return go(sink or self._nearest(game, cook, self._cells(game, Terrain.TRASH)))
            have = _Counter((i.name, i.state.value) for i in held.contents)
            need = _Counter((n, st) for n, st in recipe.contents)
            matched = game.book.match_plate(held)
            if held.contents and matched is not None:
                active_ids = {o.recipe.id for o in game.orders.orders}
                if (not game.orders.enabled) or matched.id in active_ids:
                    return go(self._nearest(game, cook, self._cells(game, Terrain.SERVING)))
                # finished a dish nobody wants any more -> cut losses, bin it
                self._target = None
                return go(self._nearest(game, cook, self._cells(game, Terrain.TRASH)))
            if held.contents and (have - need):            # wrong items -> bin
                return go(self._nearest(game, cook, self._cells(game, Terrain.TRASH)))
            missing = need - have
            # add a raw component straight from its source
            for (name, st) in missing:
                if st == "raw":
                    return go(source_cell(name))
            # add a chopped component from a ready board
            for (name, st) in missing:
                if st == "chopped":
                    cell = next((c for c, s in board_chopped if s.item.name == name), None)
                    if cell:
                        return go(cell)
            # scoop a cooked component from a ready station
            if any(st == "cooked" for _, st in missing) and cooked_ready:
                return go(self._nearest(game, cook, cooked_ready))
            # nothing addable yet: wait if things are cooking/chopping, else
            # stash the plate and go do production work
            if cooking or board_raw or raw_started:
                if cooking:
                    return go(self._nearest(game, cook, cooking))
                return go(empty_counter())
            return go(empty_counter())

        # ================= holding an ingredient ==========================
        if isinstance(held, Ingredient):
            nm, state = held.name, held.state.value
            # route raw items that must be cooked to a cook station
            if state in ("raw", "chopped") and cooked_short.get(nm, 0) > 0 and fillable_cells:
                return go(fillable_cells[0])
            # route raw items that must be chopped to a free board
            if state == "raw" and chop_short.get(nm, 0) > 0 and board_empty:
                return go(self._nearest(game, cook, board_empty))
            return go(empty_counter())                     # otherwise set it down

        # ================= empty handed ===================================
        if burnt:
            return go(self._nearest(game, cook, burnt))
        # finish any chopping already in progress (needs active interaction)
        useful_raw_board = [c for c, s in board_raw if s.item.name in chop_need]
        if useful_raw_board:
            return go(self._nearest(game, cook, useful_raw_board))
        # load cook stations
        if cooked_short and fillable_cells:
            name = next(iter(cooked_short))
            return go(source_cell(name))
        # start a sufficiently-filled station
        if not cooked_short and raw_started:
            return go(self._nearest(game, cook, raw_started))
        # start a chop: grab a raw ingredient and carry it to a board
        if chop_short and board_empty:
            name = next(iter(chop_short))
            return go(source_cell(name))
        # production is underway / done -> fetch a plate and assemble
        if cooked_ready or cooking or chopped_ready or raw_need or chop_need or cooked_need:
            return go(self._nearest(game, cook, self._cells(game, Terrain.PLATE_SOURCE)))
        return int(Action.STAY)
