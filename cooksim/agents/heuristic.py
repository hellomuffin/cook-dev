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
    def __init__(self, seed: int = 0, role: str = "any"):
        self.rng = random.Random(seed)
        self._last_pos: Optional[Pos] = None
        self._stuck = 0
        self._target: Optional[str] = None   # committed recipe id
        self._stash: Optional[Pos] = None     # counter holding our in-progress plate
        self._commit_tick = 0                 # game tick we committed to _target
        self._commit_deliv = 0                # deliveries at commit time
        self._avoid: dict = {}                # recipe_id -> tick until which to skip it
        # "any" = do everything; "cook" = only produce components (man the
        # stations); "server" = only plate & deliver. Assigning the two roles
        # to a 2-cook team gives clean Overcooked-style division of labour.
        self.role = role

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

    # If we've been committed this long with no delivery, the recipe is likely
    # unmakeable in this kitchen (e.g. needs more boards than exist) — give up.
    ABANDON_TICKS = 220
    AVOID_TICKS = 500

    def _target_order(self, game: KitchenGame, cook_id: int = 0):
        feasible_ids = {r.id for r in game.feasible}
        orders = game.orders.orders
        active = {o.recipe.id for o in orders}
        deliveries = game.stats.get("deliveries", 0)

        # Abandon a target we can't seem to finish; blacklist it for a while.
        if self._target is not None:
            if deliveries > self._commit_deliv:
                self._commit_deliv = deliveries
                self._commit_tick = game.tick
            elif game.tick - self._commit_tick > self.ABANDON_TICKS:
                self._avoid[self._target] = game.tick + self.AVOID_TICKS
                self._target = None
                self._stash = None

        # Keep the committed target while it is still on the board.
        if self._target and self._target in active and self._target in feasible_ids:
            cand = [o for o in orders if o.recipe.id == self._target]
            return min(cand, key=lambda o: o.time_left)

        # Pick a fresh target, skipping recently-blacklisted recipes. Order by
        # time left and offset by cook id so cooks split onto distinct dishes.
        makeable = sorted(
            [o for o in orders if o.recipe.id in feasible_ids
             and self._avoid.get(o.recipe.id, 0) <= game.tick],
            key=lambda o: -o.time_left,
        )
        if not makeable:  # everything blacklisted? fall back to any feasible order
            makeable = sorted([o for o in orders if o.recipe.id in feasible_ids],
                              key=lambda o: -o.time_left)
        if not makeable:
            self._target = None
            return None
        seen, distinct = set(), []
        for o in makeable:
            if o.recipe.id not in seen:
                seen.add(o.recipe.id)
                distinct.append(o)
        choice = distinct[cook_id % len(distinct)]
        if choice.recipe.id != self._target:
            self._commit_tick = game.tick
            self._commit_deliv = deliveries
            self._stash = None
        self._target = choice.recipe.id
        return choice

    def _park_target(self, game: KitchenGame) -> Optional[Pos]:
        """A neutral interior floor tile (near the kitchen centre, touching no
        station) where an idle cook can wait without blocking access tiles."""
        if getattr(self, "_park_sig", None) != id(game.layout):
            cx, cy = game.layout.width / 2, game.layout.height / 2
            floors = [
                (x, y)
                for y in range(game.layout.height)
                for x in range(game.layout.width)
                if game._walkable(x, y)
            ]
            def touches_station(x, y):
                for d in Direction:
                    dx, dy = DIRECTION_VECTORS[d]
                    nx, ny = x + dx, y + dy
                    if game.layout.in_bounds(nx, ny) and game.layout.grid[ny][nx] != Terrain.FLOOR:
                        return True
                return False
            clear = [p for p in floors if not touches_station(*p)]
            pool = clear or floors
            self._park = min(pool, key=lambda p: abs(p[0] - cx) + abs(p[1] - cy)) if pool else None
            self._park_sig = id(game.layout)
        return self._park

    # -- main policy -----------------------------------------------------
    def __call__(self, game: KitchenGame, cook_id: int) -> int:
        cook = game.cooks[cook_id]
        action = self._plan(game, cook_id)
        # If idle, step off any station-access tile toward a neutral spot so we
        # don't block a teammate who needs to reach that station.
        if action == int(Action.STAY):
            park = self._park_target(game)
            if park is not None and cook.pos != park:
                step = _bfs_first_step(game, cook.pos, {park})
                if step is not None:
                    action = int(step)
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
        need = _Counter((n, st) for n, st in recipe.contents)

        # Components already secured on our plate (in hand or stashed) so we
        # don't re-produce them — the key to multi-chop recipes on one board.
        secured = _Counter()
        if isinstance(held, Plate):
            for i in held.contents:
                secured[(i.name, i.state.value)] += 1
        elif self._stash is not None:
            st_it = game.counter_items.get(self._stash)
            if isinstance(st_it, Plate):
                for i in st_it.contents:
                    secured[(i.name, i.state.value)] += 1
        secured_cooked, secured_chopped = _Counter(), _Counter()
        for (nm, stt), c in secured.items():
            if stt == "cooked":
                secured_cooked[nm] += c
            elif stt == "chopped":
                secured_chopped[nm] += c

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
            short = k - in_stations.get(name, 0) - secured_cooked.get(name, 0)
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
            short = k - chop_pipeline.get(name, 0) - secured_chopped.get(name, 0)
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

        def stash_held():
            """Set the plate down on a counter, remembering where so we can come
            back to the SAME partial plate after freeing our hands to chop."""
            cell = empty_counter()
            if cell is not None and isinstance(held, Plate) and held.contents:
                self._stash = cell
            return go(cell)

        def get_plate():
            """Prefer reclaiming our stashed in-progress plate over a fresh one."""
            if self._stash is not None:
                it = game.counter_items.get(self._stash)
                if isinstance(it, Plate) and not it.dirty:
                    have = _Counter((i.name, i.state.value) for i in it.contents)
                    if not (have - need):     # everything on it is still wanted
                        return self._stash
                else:
                    self._stash = None         # stash gone / taken
            return self._nearest(game, cook, self._cells(game, Terrain.PLATE_SOURCE))

        # ================= holding a plate (assembly workspace) ===========
        if isinstance(held, Plate):
            # we are now carrying our (possibly reclaimed) plate
            if self._stash is not None and not isinstance(game.counter_items.get(self._stash), Plate):
                self._stash = None
            # a dedicated producer never carries a plate — set it down
            if self.role == "cook" and not held.dirty:
                return stash_held()
            if held.dirty:
                sink = self._nearest(game, cook, self._cells(game, Terrain.SINK))
                return go(sink or self._nearest(game, cook, self._cells(game, Terrain.TRASH)))
            have = _Counter((i.name, i.state.value) for i in held.contents)
            seq = recipe.contents
            cur = tuple((i.name, i.state.value) for i in held.contents)
            active_ids = {o.recipe.id for o in game.orders.orders}
            # Decisions are made against our COMMITTED target, not "any recipe a
            # plate happens to match" — an in-progress ordered dish can pass
            # through states that ARE another complete recipe (e.g. bun+meat is a
            # Burger en route to a Cheeseburger); we must keep building, not bin.
            on_track = (cur == seq[:len(cur)]) if recipe.ordered else (not (have - need))
            complete = (cur == seq) if recipe.ordered else (not (have - need) and not (need - have))
            if complete and held.contents:                 # our target dish is done
                if (not game.orders.enabled) or recipe.id in active_ids:
                    return go(self._nearest(game, cook, self._cells(game, Terrain.SERVING)))
                self._target = None                        # nobody wants it any more
                return go(self._nearest(game, cook, self._cells(game, Terrain.TRASH)))
            if held.contents and not on_track:             # wrong items / wrong order -> bin
                self._stash = None
                return go(self._nearest(game, cook, self._cells(game, Terrain.TRASH)))
            # ----- ordered (stacked) dishes: add the next layer, in sequence ----
            if recipe.ordered:
                nm, st = seq[len(cur)]              # the one next layer we may add
                if st == "raw":
                    return go(source_cell(nm))
                if st == "chopped":
                    cell = next((c for c, s in board_chopped if s.item.name == nm), None)
                    return go(cell) if cell else stash_held()   # else free hands & chop it
                # cooked: scoop only from a station holding exactly this item
                cell = next((c for c, s in pots if s.status == "cooked"
                             and any(ing.name == nm for ing in s.contents)), None)
                return go(cell) if cell else stash_held()       # else free hands & cook it
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
            # nothing addable yet. A server keeps its plate and waits by the
            # stove (the producer will fill it); others free their hands.
            if self.role == "server":
                if cooking:
                    return go(self._nearest(game, cook, cooking))
                if cooked_short or raw_started or board_raw or chop_short:
                    return int(Action.STAY)   # hold the plate, let the cook work
                return int(Action.STAY)
            # A generalist needs free hands to chop the remaining components, so
            # it stashes this partial plate (remembering it) and goes to produce.
            if cooking and not (chop_short or chopped_ready):
                return go(self._nearest(game, cook, cooking))
            return stash_held()

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
        # A dedicated server goes straight for a plate and lets the cook produce
        # (with a fallback to help produce only if nothing is underway at all).
        if self.role == "server":
            production_underway = bool(cooked_ready or cooking or chopped_ready
                                       or raw_started or board_raw)
            needs_anything = bool(cooked_need or chop_need or raw_need)
            if production_underway or needs_anything:
                return go(get_plate())
            return int(Action.STAY)

        # producer / generalist
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
        # a generalist also fetches a plate to assemble; a dedicated cook does not
        if self.role != "cook" and (cooked_ready or cooking or chopped_ready
                                     or raw_need or chop_need or cooked_need):
            return go(get_plate())
        return int(Action.STAY)
