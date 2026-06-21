"""The cook (agent) model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .enums import Direction
from .items import copy_item, item_to_dict


@dataclass
class Cook:
    """A controllable cook on the grid."""

    id: int
    x: int
    y: int
    direction: Direction = Direction.SOUTH
    holding: Optional[object] = None  # Ingredient | Plate | None

    # transient animation hint for the renderer (set each step)
    last_action: str = "idle"

    @property
    def pos(self) -> tuple:
        return (self.x, self.y)

    def facing_pos(self) -> tuple:
        from .enums import DIRECTION_VECTORS
        dx, dy = DIRECTION_VECTORS[self.direction]
        return (self.x + dx, self.y + dy)

    def render_state(self) -> dict:
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "dir": int(self.direction),
            "dir_name": self.direction.name.lower(),
            "holding": item_to_dict(self.holding),
            "action": self.last_action,
        }
