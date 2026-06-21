"""Core enumerations for CookSim."""
from __future__ import annotations

from enum import Enum, IntEnum


class Direction(IntEnum):
    """Facing / movement directions. Values index into DIRECTION_VECTORS."""

    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3


# (dx, dy) in grid coordinates, y grows downward (screen convention).
DIRECTION_VECTORS = {
    Direction.NORTH: (0, -1),
    Direction.SOUTH: (0, 1),
    Direction.EAST: (1, 0),
    Direction.WEST: (-1, 0),
}


class Action(IntEnum):
    """Discrete actions available to each cook every tick."""

    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3
    INTERACT = 4
    STAY = 5


# Actions that correspond to a movement/facing direction.
MOVE_ACTIONS = {
    Action.NORTH: Direction.NORTH,
    Action.SOUTH: Direction.SOUTH,
    Action.EAST: Direction.EAST,
    Action.WEST: Direction.WEST,
}


class Terrain(str, Enum):
    """A grid cell's terrain / fixed feature.

    ``FLOOR`` is the only walkable cell. Everything else is an impassable
    feature that a cook can interact with from an adjacent floor tile.
    """

    FLOOR = "floor"
    COUNTER = "counter"        # holds at most one loose item on top
    WALL = "wall"             # purely decorative / blocking, no interaction
    # Infinite sources -------------------------------------------------
    ONION_SOURCE = "onion_source"
    TOMATO_SOURCE = "tomato_source"
    LETTUCE_SOURCE = "lettuce_source"
    MUSHROOM_SOURCE = "mushroom_source"
    MEAT_SOURCE = "meat_source"
    FISH_SOURCE = "fish_source"
    BUN_SOURCE = "bun_source"
    CHEESE_SOURCE = "cheese_source"
    RICE_SOURCE = "rice_source"
    DOUGH_SOURCE = "dough_source"
    EGG_SOURCE = "egg_source"
    POTATO_SOURCE = "potato_source"
    PLATE_SOURCE = "plate_source"
    # Processing stations ---------------------------------------------
    POT = "pot"               # boils / cooks ingredients into soup
    PAN = "pan"               # fries ingredients (patties, fish, eggs)
    OVEN = "oven"             # bakes (pizza, bread, baked dishes)
    CUTTING_BOARD = "cutting_board"  # chop raw ingredients
    SINK = "sink"             # wash dirty plates
    # Terminals --------------------------------------------------------
    SERVING = "serving"       # deliver finished dishes
    TRASH = "trash"           # discard anything


# Which terrain types are infinite ingredient/plate dispensers.
SOURCE_ITEMS = {
    Terrain.ONION_SOURCE: "onion",
    Terrain.TOMATO_SOURCE: "tomato",
    Terrain.LETTUCE_SOURCE: "lettuce",
    Terrain.MUSHROOM_SOURCE: "mushroom",
    Terrain.MEAT_SOURCE: "meat",
    Terrain.FISH_SOURCE: "fish",
    Terrain.BUN_SOURCE: "bun",
    Terrain.CHEESE_SOURCE: "cheese",
    Terrain.RICE_SOURCE: "rice",
    Terrain.DOUGH_SOURCE: "dough",
    Terrain.EGG_SOURCE: "egg",
    Terrain.POTATO_SOURCE: "potato",
}

# Stations that apply heat (transform RAW/CHOPPED -> COOKED over time).
COOK_STATIONS = {Terrain.POT, Terrain.PAN, Terrain.OVEN}

WALKABLE = {Terrain.FLOOR}

# Stations that keep their own mutable, position-keyed state.
STATEFUL_STATIONS = {Terrain.POT, Terrain.PAN, Terrain.OVEN, Terrain.CUTTING_BOARD, Terrain.SINK}


class PrepState(str, Enum):
    """Preparation state of an ingredient."""

    RAW = "raw"
    CHOPPED = "chopped"
    COOKED = "cooked"
    BURNT = "burnt"


# Single-character codes used in textual layout files. Mirrors (and extends)
# the overcooked_ai layout grammar so existing maps are easy to port.
TERRAIN_TO_CHAR = {
    Terrain.FLOOR: " ",
    Terrain.COUNTER: "X",
    Terrain.WALL: "#",
    Terrain.ONION_SOURCE: "O",
    Terrain.TOMATO_SOURCE: "T",
    Terrain.LETTUCE_SOURCE: "L",
    Terrain.MUSHROOM_SOURCE: "M",
    Terrain.MEAT_SOURCE: "E",
    Terrain.FISH_SOURCE: "F",
    Terrain.BUN_SOURCE: "B",
    Terrain.CHEESE_SOURCE: "C",
    Terrain.RICE_SOURCE: "R",
    Terrain.DOUGH_SOURCE: "G",
    Terrain.EGG_SOURCE: "Z",
    Terrain.POTATO_SOURCE: "Y",
    Terrain.PLATE_SOURCE: "D",
    Terrain.POT: "P",
    Terrain.PAN: "A",
    Terrain.OVEN: "N",
    Terrain.CUTTING_BOARD: "/",
    Terrain.SINK: "W",
    Terrain.SERVING: "S",
    Terrain.TRASH: "U",
}
CHAR_TO_TERRAIN = {v: k for k, v in TERRAIN_TO_CHAR.items()}
# Players are marked 1-9 on floor cells in layout files.
