"""Stateful processing stations: pots, pans, cutting boards and sinks."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .enums import PrepState
from .items import Ingredient, Plate


@dataclass
class CookStation:
    """A pot or pan. Holds ingredients and transforms them over time.

    Status lifecycle::

        empty -> filling (status="raw") -> cooking -> cooked -> burnt

    The station is recipe-agnostic: it simply transitions every ingredient
    from RAW/CHOPPED to COOKED, then to BURNT if left on the heat too long.
    """

    capacity: int = 3
    cook_time: int = 20
    burn_time: int = 25          # extra ticks after "cooked" before burning
    contents: List[Ingredient] = field(default_factory=list)
    cooking: bool = False
    timer: int = 0
    status: str = "empty"        # empty|raw|cooking|cooked|burnt
    auto_start: bool = True      # begin cooking automatically when full

    # -- queries -----------------------------------------------------------
    @property
    def is_full(self) -> bool:
        return len(self.contents) >= self.capacity

    def can_add(self) -> bool:
        return self.status in ("empty", "raw") and not self.is_full

    # -- mutations ---------------------------------------------------------
    def add(self, ing: Ingredient) -> bool:
        if not self.can_add():
            return False
        self.contents.append(ing)
        self.status = "raw"
        if self.auto_start and self.is_full:
            self.start()
        return True

    def start(self) -> bool:
        if self.status == "raw" and self.contents:
            self.cooking = True
            self.status = "cooking"
            self.timer = 0
            return True
        return False

    def take_into_plate(self, plate: Plate, plate_capacity: int = 3) -> bool:
        """Ladle cooked contents onto a clean plate (merging with whatever is
        already plated, e.g. rice + soup), provided there is room."""
        if self.status != "cooked" or plate.dirty:
            return False
        if len(plate.contents) + len(self.contents) > plate_capacity:
            return False
        plate.contents.extend(c.copy() for c in self.contents)
        self.reset()
        return True

    def clear(self) -> Optional[str]:
        """Empty a burnt station (contents are discarded). Returns reason."""
        if self.status == "burnt":
            self.reset()
            return "burnt_cleared"
        return None

    def reset(self) -> None:
        self.contents = []
        self.cooking = False
        self.timer = 0
        self.status = "empty"

    def tick(self) -> None:
        if not self.cooking:
            return
        self.timer += 1
        if self.timer == self.cook_time:
            for c in self.contents:
                c.state = PrepState.COOKED
            self.status = "cooked"
        elif self.timer >= self.cook_time + self.burn_time:
            for c in self.contents:
                c.state = PrepState.BURNT
            self.status = "burnt"
            self.cooking = False

    def render_state(self) -> dict:
        progress = 0.0
        if self.status == "cooking":
            progress = min(1.0, self.timer / max(1, self.cook_time))
        elif self.status in ("cooked", "burnt"):
            progress = 1.0
        # how close to burning (0..1) once cooked
        burn_progress = 0.0
        if self.status == "cooked":
            burn_progress = min(1.0, (self.timer - self.cook_time) / max(1, self.burn_time))
        return {
            "contents": [c.to_dict() for c in self.contents],
            "status": self.status,
            "progress": progress,
            "burn_progress": burn_progress,
            "capacity": self.capacity,
        }


@dataclass
class CuttingBoard:
    """Chops a single raw ingredient. Progress advances per interaction."""

    chop_time: int = 4
    item: Optional[Ingredient] = None
    progress: int = 0

    def place(self, ing: Ingredient) -> bool:
        if self.item is not None or ing.state != PrepState.RAW:
            return False
        self.item = ing
        self.progress = 0
        return True

    def chop(self) -> bool:
        """One unit of chopping work. Returns True if it did something."""
        if self.item is None or self.item.state != PrepState.RAW:
            return False
        self.progress += 1
        if self.progress >= self.chop_time:
            self.item.state = PrepState.CHOPPED
        return True

    def take(self) -> Optional[Ingredient]:
        item, self.item, self.progress = self.item, None, 0
        return item

    def render_state(self) -> dict:
        return {
            "item": self.item.to_dict() if self.item else None,
            "progress": min(1.0, self.progress / max(1, self.chop_time)),
            "done": self.item is not None and self.item.state == PrepState.CHOPPED,
        }


@dataclass
class Sink:
    """Holds dirty plates and washes them clean over repeated interactions."""

    wash_time: int = 5
    dirty: int = 0
    clean_ready: int = 0
    progress: int = 0

    def deposit_dirty(self) -> bool:
        self.dirty += 1
        return True

    def wash(self) -> bool:
        if self.dirty <= 0:
            return False
        self.progress += 1
        if self.progress >= self.wash_time:
            self.progress = 0
            self.dirty -= 1
            self.clean_ready += 1
        return True

    def take_clean(self) -> bool:
        if self.clean_ready > 0:
            self.clean_ready -= 1
            return True
        return False

    def render_state(self) -> dict:
        return {
            "dirty": self.dirty,
            "clean_ready": self.clean_ready,
            "progress": min(1.0, self.progress / max(1, self.wash_time)),
        }
