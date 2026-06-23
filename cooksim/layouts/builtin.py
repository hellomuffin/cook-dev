"""Built-in kitchen layouts.

Grammar (one char per cell)::

    ' ' floor      X counter     # wall
    O onion  T tomato  L lettuce  M mushroom  E meat
    F fish   B bun     C cheese   R rice       D plates
    P pot    A pan     / board    W sink       S serving   U trash
    1-9      cook spawn points (on floor)
"""
from __future__ import annotations

from typing import Dict, List

from ..core.layout import Layout

_LAYOUTS: Dict[str, List[str]] = {
    # Classic tiny room — onion soup only, two cooks.
    "cramped_room": [
        "XXPXX",
        "O   O",
        "X1 2X",
        "XDXSX",
    ],
    # Forced-coordination: a counter wall splits the two cooks; one side has
    # the ingredients, the other has pots + serving.
    "forced_coordination": [
        "XXXPX",
        "O X1S",
        "O2X D",
        "XXXPX",
    ],
    # A roomy single kitchen with the full station set and many ingredients.
    "open_kitchen": [
        "XXXXPPXXXX",
        "O        D",
        "T   12   /",
        "L        W",
        "M   34   A",
        "E        S",
        "XXXXUUXXXX",
    ],
    # Two pots, salad + soup capable, ring topology.
    "bistro": [
        "XXPPXXX",
        "O     D",
        "T  1  /",
        "L  2  S",
        "C     W",
        "XXXUXXX",
    ],
    # Burger joint: pans, buns, meat, cheese.
    "burger_bar": [
        "XXAAXX",
        "B    D",
        "E 12 S",
        "C    /",
        "XXWUXX",
    ],
    # Pizzeria: ovens for baking; dough, tomato, cheese + a board for caprese.
    "pizzeria": [
        "XXNNXX",
        "G    D",
        "T 12 /",
        "C    S",
        "XXUWXX",
    ],
    # Full diner: pans, pots, board, sink and a wide pantry — every prep type.
    "diner": [
        "XXAAPPXXX",
        "E       D",
        "B  12   /",
        "L  34   W",
        "Y       S",
        "XXXTCZUXX",
    ],
    # Grand kitchen: every ingredient source and every station — any of the
    # 19 recipes can be made here. Roomy enough for a full brigade of cooks.
    "grand_kitchen": [
        "XOTLMEFBCX",
        "R        P",
        "G  1 2   P",
        "Z        N",
        "Y  3 4   A",
        "D        S",
        "XXX//WUXXX",
    ],
}


def list_layouts() -> List[str]:
    return sorted(_LAYOUTS.keys())


def load_layout(name: str) -> Layout:
    if name not in _LAYOUTS:
        raise KeyError(f"Unknown layout '{name}'. Available: {list_layouts()}")
    return Layout.from_lines(name, _LAYOUTS[name])


def all_layouts() -> Dict[str, Layout]:
    return {name: Layout.from_lines(name, lines) for name, lines in _LAYOUTS.items()}
