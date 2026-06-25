"""Manipulation-level natural-language control for CookSim.

A streaming VLM drives the cook with small *physical* commands, never whole
sub-tasks. Two kinds of command:

  LOCOMOTION (the only mover, and the only command that names a target):
      "go to <object>"        — pathfind adjacent to an object and face it.
      Targets may carry qualifiers to disambiguate when several exist:
        content  : "go to the pot cooking rice"
        state    : "go to the done pot" / "the empty board" / "the dirty plate"
        spatial  : "the left pot" / "the nearest pot" / "the top-left pan"
        identity : "pot 2"           (stations numbered row-major within a type)
        exact    : "the pot at (2,0)"

  MANIPULATION (acts on the tile the cook FACES; never moves; one object op):
      pick up · put down · chop · turn on · scoop · serve · wash · discard · wait

No command bundles task functionality: there is no "cook the soup" or "chop the
potato" that fetches+places+dices. The VLM must compose a chain and time it
against what it sees. A raw low-level action (N/S/E/W/INTERACT/STAY, by name or
int) is also accepted for fine-grained control.

The controller is closed-loop and DEFENSIVE: every parse / grounding / execution
problem becomes a clean ``failed`` status with a human-readable reason — it never
raises and never leaves the engine in a bad state (commands only ever emit valid
low-level actions).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from ..core.enums import SOURCE_ITEMS, Action, Direction, Terrain
from ..core.items import Ingredient, Plate
from ..core.stations import CookStation, CuttingBoard, Sink
from .heuristic import _adjacent_floor, _bfs_first_step, _dir_between

Pos = Tuple[int, int]

_D2A = {Direction.NORTH: Action.NORTH, Direction.SOUTH: Action.SOUTH,
        Direction.EAST: Action.EAST, Direction.WEST: Action.WEST}
_SOURCE_BY_NAME = {n: t for t, n in SOURCE_ITEMS.items()}
INGREDIENTS = sorted(_SOURCE_BY_NAME)
_ALIAS = {"patty": "meat", "beef": "meat", "bread": "bun", "salmon": "fish",
          "shroom": "mushroom", "spud": "potato", "egg": "egg"}
_STATION_TERRAIN = {
    "pot": (Terrain.POT,), "pan": (Terrain.PAN,), "oven": (Terrain.OVEN,),
    "board": (Terrain.CUTTING_BOARD,), "sink": (Terrain.SINK,),
    "serving": (Terrain.SERVING,), "trash": (Terrain.TRASH,),
    "counter": (Terrain.COUNTER,), "plate": (Terrain.PLATE_SOURCE,),
}
_STATION_SYN = {
    "pot": "pot", "pan": "pan", "skillet": "pan", "oven": "oven",
    "cutting board": "board", "chopping board": "board", "board": "board",
    "sink": "sink", "serving": "serving", "pass": "serving", "window": "serving",
    "hatch": "serving", "delivery": "serving", "counter": "counter",
    "trash": "trash", "bin": "trash", "garbage": "trash",
    "plate stack": "plate", "plate dispenser": "plate", "plates": "plate", "plate": "plate",
}
_STATES = {"done": "cooked", "ready": "cooked", "cooked": "cooked", "finished": "cooked",
           "empty": "empty", "cooking": "cooking", "burning": "burnt", "burnt": "burnt",
           "raw": "raw", "dirty": "dirty", "clean": "clean", "full": "full", "chopped": "chopped"}
_SPATIAL = ["top-left", "top left", "top-right", "top right", "bottom-left", "bottom left",
            "bottom-right", "bottom right", "leftmost", "rightmost", "nearest", "closest",
            "left", "right", "top", "bottom", "upper", "lower", "far"]

_VERBS = {
    "goto":   ["go to", "walk to", "move to", "head to", "navigate to", "go back to", "approach", "go towards"],
    "wait":   ["wait", "idle", "do nothing"],
    "chop":   ["chop", "cut", "slice", "dice", "mince"],
    "start":  ["turn on", "start", "begin cooking", "switch on", "ignite"],
    "scoop":  ["scoop", "ladle", "plate up", "dish out", "spoon", "add to plate", "scoop onto"],
    "serve":  ["serve", "deliver", "hand in", "hand over", "submit"],
    "wash":   ["wash", "clean", "rinse"],
    "trash":  ["trash", "discard", "throw away", "throw out", "dump", "bin it"],
    "putdown":["put down", "set down", "put", "place", "drop", "add", "load", "insert"],
    "pickup": ["pick up", "grab", "take", "get", "collect", "lift", "fetch"],
}
_VERB_ORDER = ["goto", "wait", "chop", "start", "scoop", "serve", "wash", "trash", "putdown", "pickup"]
_LOW = {"north": Action.NORTH, "up": Action.NORTH, "south": Action.SOUTH, "down": Action.SOUTH,
        "east": Action.EAST, "right": Action.EAST, "west": Action.WEST, "left": Action.WEST,
        "interact": Action.INTERACT, "stay": Action.STAY, "wait_tick": Action.STAY}

VERB_HELP = {
    "goto": "go to <object>  — the only move; names a target (+qualifiers)",
    "pickup": "pick up  — take the item you face (into an empty hand)",
    "putdown": "put down  — drop the held item into/onto the tile you face",
    "chop": "chop  — dice the one raw item on the board you face",
    "start": "turn on  — start the filled pot/pan/oven you face",
    "scoop": "scoop  — add the faced station's food onto the plate you hold",
    "serve": "serve  — hand the finished plate to the serving pass you face",
    "wash": "wash  — run a dirty plate through the sink you face",
    "trash": "discard  — bin the item you hold at the trash you face",
    "wait": "wait [until <object> is <state>]  — idle / watch",
}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
@dataclass
class Intent:
    verb: str
    obj: str = ""
    ingredient: str = ""
    quals: Dict[str, object] = field(default_factory=dict)
    until: Optional["Intent"] = None
    raw: str = ""


def _canon_ingredient(t: str) -> str:
    for n in INGREDIENTS:
        if re.search(rf"\b{n}\b", t):
            return n
    for a, n in _ALIAS.items():
        if re.search(rf"\b{a}\b", t):
            return n
    return ""


def _parse_quals(t: str) -> Dict[str, object]:
    q: Dict[str, object] = {}
    m = re.search(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", t)
    if m:
        q["coord"] = (int(m.group(1)), int(m.group(2)))
    m = (re.search(r"(?:#|number\s*)(\d+)", t)
         or re.search(r"\b(?:pot|pan|oven|board|sink|station)\s+(\d+)\b", t))
    if m:
        q["id"] = int(m.group(1))
    for s in _SPATIAL:
        if f" {s} " in f" {t} ":
            q["spatial"] = s.replace(" ", "-")
            break
    for w, st in _STATES.items():
        if re.search(rf"\b{w}\b", t):
            q.setdefault("state", st)
    mc = re.search(r"(?:with|containing|holding|cooking|that has|that contains|of)\s+(?:the\s+|some\s+|a\s+)?(\w+)", t)
    if mc:
        ci = _canon_ingredient(" " + mc.group(1) + " ")
        if ci:
            q["content"] = ci
    return q


def _parse_target(t: str) -> Tuple[str, str, Dict[str, object]]:
    """Return (obj_token, ingredient, quals) for a 'go to' / 'wait until' phrase."""
    quals = _parse_quals(t)
    ing = _canon_ingredient(t)
    # pick the EARLIEST-mentioned station word (the head noun usually comes first,
    # e.g. "the PLATES on the left counter" -> plate, not counter); tie-break longer.
    best = None
    for syn, canon in _STATION_SYN.items():
        m = re.search(rf"\b{re.escape(syn)}\b", t)
        if m:
            key = (m.start(), -len(syn))
            if best is None or key < best[0]:
                best = (key, canon)
    obj = best[1] if best else ("source" if ing else "")
    return obj, ing, quals


def parse(text: str) -> Optional[Intent]:
    if not isinstance(text, str):
        return None
    t = " " + text.strip().lower() + " "
    verb = None
    for v in _VERB_ORDER:
        for ph in sorted(_VERBS[v], key=len, reverse=True):
            if f" {ph} " in t or t.strip().startswith(ph + " ") or t.strip() == ph:
                verb = v
                break
        if verb:
            break
    if verb is None:
        return None
    it = Intent(verb=verb, raw=text.strip())
    if verb == "goto":
        it.obj, it.ingredient, it.quals = _parse_target(t)
    elif verb == "wait":
        m = re.search(r"\buntil\b(.*)", t)
        if m:
            o, ing, q = _parse_target(" " + m.group(1) + " ")
            sub = Intent("goto", o, ing, q)
            it.until = sub
        nm = re.search(r"\b(\d+)\b", t)
        if nm and "id" not in it.quals:
            it.quals["ticks"] = int(nm.group(1))
    return it


# ---------------------------------------------------------------------------
# Grounding (for 'go to' / 'wait until')
# ---------------------------------------------------------------------------
@dataclass
class Target:
    tile: Optional[Pos]
    label: str
    ambiguous: bool = False


def _tiles_of(game, terrains) -> List[Pos]:
    return [(x, y) for y, row in enumerate(game.layout.grid)
            for x, t in enumerate(row) if t in terrains]


def _state_match(obj, state) -> bool:
    if isinstance(obj, CookStation):
        return {"empty": obj.status == "empty" or not obj.contents,
                "cooked": obj.status == "cooked", "cooking": obj.status == "cooking",
                "burnt": obj.status == "burnt", "raw": obj.status == "raw",
                "full": obj.is_full}.get(state, False)
    if isinstance(obj, CuttingBoard):
        if state == "empty":
            return obj.item is None
        if state in ("chopped", "cooked"):
            return obj.item is not None and obj.item.state.value == "chopped"
        if state == "raw":
            return obj.item is not None and obj.item.state.value == "raw"
    if isinstance(obj, Sink):
        if state == "dirty":
            return obj.dirty > 0
        if state == "clean":
            return obj.clean_ready > 0
    return False


def _has_content(obj, ing) -> bool:
    if isinstance(obj, CookStation):
        return any(i.name == ing for i in obj.contents)
    if isinstance(obj, CuttingBoard):
        return obj.item is not None and obj.item.name == ing
    return False


def _pick_spatial(tiles, spatial, cook) -> Pos:
    if spatial in ("nearest", "closest"):
        return min(tiles, key=lambda p: abs(p[0] - cook.x) + abs(p[1] - cook.y))
    if spatial == "far":
        return max(tiles, key=lambda p: abs(p[0] - cook.x) + abs(p[1] - cook.y))
    keys = {"left": lambda p: (p[0], p[1]), "leftmost": lambda p: (p[0], p[1]),
            "right": lambda p: (-p[0], p[1]), "rightmost": lambda p: (-p[0], p[1]),
            "top": lambda p: (p[1], p[0]), "upper": lambda p: (p[1], p[0]),
            "bottom": lambda p: (-p[1], p[0]), "lower": lambda p: (-p[1], p[0]),
            "top-left": lambda p: (p[0] + p[1], p[1]), "top-right": lambda p: (-p[0] + p[1], p[1]),
            "bottom-left": lambda p: (p[0] - p[1], -p[1]), "bottom-right": lambda p: (-p[0] - p[1], -p[1])}
    return min(tiles, key=keys.get(spatial, lambda p: (p[1], p[0])))


def ground(it: Intent, game, cook) -> Optional[Target]:
    obj = it.obj
    # bare coordinate, e.g. "go to (1,3)" — face that exact non-floor tile
    if not obj and not it.ingredient and "coord" in it.quals:
        c = it.quals["coord"]
        if game.layout.in_bounds(*c) and game.layout.grid[c[1]][c[0]] != Terrain.FLOOR:
            return Target(c, f"tile {c}")
        return None
    if obj == "source" or (not obj and it.ingredient):
        if it.ingredient in _SOURCE_BY_NAME:
            tiles = _tiles_of(game, (_SOURCE_BY_NAME[it.ingredient],))
            if tiles:
                return Target(_pick_spatial(tiles, it.quals.get("spatial", "nearest"), cook),
                              f"{it.ingredient} source")
        return None
    terrains = _STATION_TERRAIN.get(obj)
    if not terrains:
        return None
    cands = _tiles_of(game, terrains)
    if not cands:
        return None
    if "coord" in it.quals and it.quals["coord"] in cands:
        return Target(it.quals["coord"], f"{obj} at {it.quals['coord']}")
    if "content" in it.quals:
        cands = [c for c in cands if _has_content(game.stations.get(c), it.quals["content"])] or cands
    if "state" in it.quals:
        cands = [c for c in cands if _state_match(game.stations.get(c), it.quals["state"])] or cands
    if "id" in it.quals:
        ordered = sorted(cands, key=lambda p: (p[1], p[0]))
        i = it.quals["id"] - 1
        if 0 <= i < len(ordered):
            return Target(ordered[i], f"{obj} #{it.quals['id']}")
    amb = len(cands) > 1 and "spatial" not in it.quals
    return Target(_pick_spatial(cands, it.quals.get("spatial", "nearest"), cook), obj, ambiguous=amb)


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------
def _faced(game, cook):
    fx, fy = cook.facing_pos()
    if not game.layout.in_bounds(fx, fy):
        return None, None, None, None
    terrain = game.layout.grid[fy][fx]
    return (fx, fy), terrain, game.stations.get((fx, fy)), game.counter_items.get((fx, fy))


class _Skill:
    label = "skill"

    def step(self, game, cook):
        raise NotImplementedError


class GoTo(_Skill):
    def __init__(self, tile, label, max_ticks=300):
        self.tile, self.label, self.max_ticks, self.t = tile, f"go to {label}", max_ticks, 0
        self.error = None

    def step(self, game, cook):
        self.t += 1
        if self.t > self.max_ticks:
            self.error = "timeout reaching target"
            return int(Action.STAY), "failed"
        stand = _adjacent_floor(game, *self.tile)
        if not stand:
            self.error = "no floor tile next to the target"
            return int(Action.STAY), "failed"
        if cook.pos not in stand:
            a = _bfs_first_step(game, cook.pos, set(stand))
            if a is None:
                self.error = "target is unreachable"
                return int(Action.STAY), "failed"
            return int(a), "running"
        want = _dir_between(cook.pos, self.tile)
        if want is not None and cook.direction != want:
            return int(_D2A[want]), "running"
        return int(Action.STAY), "done"


# manipulation precondition + done predicates, keyed by verb -----------------
def _validate(verb, game, cook):
    """Return (ok: bool, reason: str). Manipulations act on the FACED tile."""
    tile, terrain, st, ci = _faced(game, cook)
    held = cook.holding
    if verb == "pickup":
        if held is not None:
            return False, "your hands are full"
        if terrain in SOURCE_ITEMS or terrain == Terrain.PLATE_SOURCE:
            return True, ""
        if isinstance(st, CuttingBoard) and st.item is not None and st.item.state.value != "raw":
            return True, ""
        if ci is not None:
            return True, ""
        return False, "nothing here to pick up — go to a source/board/counter first"
    if verb == "putdown":
        if held is None:
            return False, "your hands are empty"
        if isinstance(st, CookStation) and isinstance(held, Ingredient) and st.can_add():
            return True, ""
        if isinstance(st, CuttingBoard) and st.item is None and isinstance(held, Ingredient) and held.state.value == "raw":
            return True, ""
        if terrain == Terrain.COUNTER and ci is None:
            return True, ""
        return False, "can't put that down here — face an empty board/counter or an open pot/pan/oven"
    if verb == "chop":
        if held is not None:
            return False, "free your hands to chop (put the item on the board first)"
        if isinstance(st, CuttingBoard) and st.item is not None and st.item.state.value == "raw":
            return True, ""
        return False, "you're not facing a board with a raw item — go to a board and put something on it"
    if verb == "start":
        if not isinstance(st, CookStation):
            return False, "you're not facing a pot/pan/oven"
        if st.status in ("cooking", "cooked"):
            return True, ""           # already on — treated as a no-op success
        if st.status == "raw" and st.contents:
            return True, ""
        return False, "nothing to start — the station is empty"
    if verb == "scoop":
        if not isinstance(held, Plate) or held.dirty:
            return False, "hold a clean plate to scoop onto"
        if isinstance(st, CookStation) and st.status == "cooked":
            return True, ""
        if isinstance(st, CuttingBoard) and st.item is not None and st.item.state.value == "chopped":
            return True, ""
        if terrain in SOURCE_ITEMS:
            return True, ""
        return False, "nothing ready to scoop here — face a done station, a chopped board, or a source"
    if verb == "serve":
        if not isinstance(held, Plate) or held.dirty or not held.contents:
            return False, "hold a finished (non-empty, clean) plate to serve"
        if terrain != Terrain.SERVING:
            return False, "you're not facing the serving pass"
        return True, ""
    if verb == "wash":
        if not isinstance(st, Sink):
            return False, "you're not facing a sink"
        if (isinstance(held, Plate) and held.dirty) or st.dirty > 0 or st.clean_ready > 0:
            return True, ""
        return False, "nothing to wash"
    if verb == "trash":
        if held is None:
            return False, "your hands are empty"
        if terrain != Terrain.TRASH:
            return False, "you're not facing the trash"
        return True, ""
    return False, f"unknown manipulation '{verb}'"


def _done(verb, game, cook, snap):
    tile, terrain, st, ci = _faced(game, cook)
    held = cook.holding
    if verb == "pickup":
        return held is not None
    if verb == "putdown":
        return held is None
    if verb == "chop":
        return isinstance(st, CuttingBoard) and st.item is not None and st.item.state.value == "chopped"
    if verb == "start":
        return isinstance(st, CookStation) and st.status in ("cooking", "cooked")
    if verb == "scoop":
        return isinstance(held, Plate) and len(held.contents) > snap.get("plate", 0)
    if verb == "serve":
        return held is None or (isinstance(held, Plate) and (held.dirty or not held.contents))
    if verb == "wash":
        return isinstance(held, Plate) and not held.dirty
    if verb == "trash":
        return held is None or (isinstance(held, Plate) and not held.contents and not held.dirty)
    return True


class Manipulate(_Skill):
    def __init__(self, verb, max_ticks=40):
        self.verb, self.label, self.max_ticks, self.t = verb, verb, max_ticks, 0
        self.error = None
        self._validated = False
        self._snap = {}

    def step(self, game, cook):
        self.t += 1
        if not self._validated:
            ok, reason = _validate(self.verb, game, cook)
            if not ok:
                self.error = reason
                return int(Action.STAY), "failed"
            self._validated = True
            held = cook.holding
            self._snap = {"plate": len(held.contents) if isinstance(held, Plate) else 0}
        if _done(self.verb, game, cook, self._snap):
            return int(Action.STAY), "done"
        if self.t > self.max_ticks:
            self.error = f"'{self.verb}' did not complete (stuck)"
            return int(Action.STAY), "failed"
        return int(Action.INTERACT), "running"


class Wait(_Skill):
    def __init__(self, ticks=5, until=None, max_ticks=600):
        self.label = "wait"
        self.n, self.until, self.max_ticks, self.t = ticks, until, max_ticks, 0
        self.error = None

    def step(self, game, cook):
        self.t += 1
        if self.until is not None:
            tgt = ground(self.until, game, cook)
            state = self.until.quals.get("state", "cooked")
            if tgt and tgt.tile is not None and _state_match(game.stations.get(tgt.tile), state):
                return int(Action.STAY), "done"
            if self.t > self.max_ticks:
                self.error = "condition not met before timeout"
                return int(Action.STAY), "failed"
            return int(Action.STAY), "running"
        if self.t > self.n:
            return int(Action.STAY), "done"
        return int(Action.STAY), "running"


class Raw(_Skill):
    def __init__(self, action):
        self.action, self.label, self._fired = int(action), f"low:{Action(int(action)).name}", False

    def step(self, game, cook):
        if self._fired:
            return int(Action.STAY), "done"
        self._fired = True
        return self.action, "running"


# ---------------------------------------------------------------------------
# Controller / driver
# ---------------------------------------------------------------------------
def compile_action(nl, game, cook):
    """(skill, None) on success or (None, reason). Never raises."""
    try:
        if isinstance(nl, Action):
            nl = int(nl)
        if isinstance(nl, int):
            if 0 <= nl <= 5:
                return Raw(nl), None
            return None, f"invalid low-level action int: {nl}"
        if not isinstance(nl, str) or not nl.strip():
            return None, "empty command"
        low = _LOW.get(nl.strip().lower())
        if low is not None:
            return Raw(int(low)), None
        it = parse(nl)
        if it is None:
            return None, f"no recognized verb in {nl!r}"
        if it.verb == "goto":
            t = ground(it, game, cook)
            if t is None or t.tile is None:
                return None, f"could not ground target in {nl!r}"
            return GoTo(t.tile, t.label), None
        if it.verb == "wait":
            return Wait(ticks=int(it.quals.get("ticks", 5)), until=it.until), None
        return Manipulate(it.verb), None
    except Exception as e:                       # defensive: never crash the loop
        return None, f"internal error compiling {nl!r}: {e!r}"


class HighLevelController:
    """Holds the active command for one cook; turns it into low-level actions."""

    def __init__(self, game, cook_id: int = 0):
        self.game, self.cook_id = game, cook_id
        self.skill: Optional[_Skill] = None
        self.status, self.error = "idle", None

    @property
    def cook(self):
        return self.game.cooks[self.cook_id]

    def issue(self, action) -> bool:
        self.skill, self.error = compile_action(action, self.game, self.cook)
        self.status = "running" if self.skill is not None else "failed"
        return self.skill is not None

    def step(self) -> int:
        if self.skill is None:
            return int(Action.STAY)
        try:
            act, self.status = self.skill.step(self.game, self.cook)
        except Exception as e:                   # defensive
            self.status, self.error = "failed", f"internal error: {e!r}"
            self.skill = None
            return int(Action.STAY)
        if self.status in ("done", "failed"):
            self.error = getattr(self.skill, "error", None)
            self.skill = None
        return int(act)


def run_one(game, command, cook_id=0, others=None, max_ticks=400):
    """Execute a single high-level command, stepping the sim. Returns a dict."""
    ctrl = HighLevelController(game, cook_id)
    if not ctrl.issue(command):
        return {"command": command, "status": "failed", "ticks": 0, "error": ctrl.error}
    t = 0
    while ctrl.status == "running" and t < max_ticks:
        a = ctrl.step()
        acts = [int(Action.STAY)] * len(game.cooks)
        acts[cook_id] = a
        if others:
            for cid in range(len(game.cooks)):
                if cid != cook_id:
                    acts[cid] = int(others(game, cid))
        game.step(acts)
        t += 1
    status = ctrl.status if t < max_ticks else "timeout"
    return {"command": command, "status": status, "ticks": t, "error": ctrl.error}


def run_plan(game, plan, cook_id=0, others=None, max_ticks_per=400):
    """Execute a list of commands in order; returns a per-command log."""
    return [run_one(game, c, cook_id, others, max_ticks_per) for c in plan]
