"""Item model: ingredients and plates that cooks carry and combine."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .enums import PrepState


@dataclass
class Ingredient:
    """A single food ingredient with a preparation state."""

    name: str
    state: PrepState = PrepState.RAW

    def to_dict(self) -> dict:
        return {"kind": "ingredient", "name": self.name, "state": self.state.value}

    def copy(self) -> "Ingredient":
        return Ingredient(self.name, self.state)

    @property
    def key(self) -> tuple:
        return (self.name, self.state.value)


@dataclass
class Plate:
    """A plate. May be clean or dirty, and may hold a set of ingredients.

    A plate becomes a finished dish when its contents match a recipe; the
    ``cooked`` flag tracks whether soup/fried contents were transferred hot.
    """

    dirty: bool = False
    contents: List[Ingredient] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "kind": "plate",
            "dirty": self.dirty,
            "contents": [c.to_dict() for c in self.contents],
        }

    def copy(self) -> "Plate":
        return Plate(self.dirty, [c.copy() for c in self.contents])

    @property
    def content_keys(self) -> tuple:
        """Sorted multiset of (name, state) — used to match unordered recipes."""
        return tuple(sorted(c.key for c in self.contents))

    @property
    def content_seq(self) -> tuple:
        """(name, state) in the order ingredients were ADDED — used to match
        order-sensitive recipes (e.g. layered/stacked dishes)."""
        return tuple(c.key for c in self.contents)

    def is_empty(self) -> bool:
        return not self.contents


# An "item" carried by a cook or resting on a counter is either an
# Ingredient or a Plate.
Item = object  # type alias for documentation purposes


def item_to_dict(item: Optional[object]) -> Optional[dict]:
    if item is None:
        return None
    return item.to_dict()


def copy_item(item: Optional[object]) -> Optional[object]:
    if item is None:
        return None
    return item.copy()
