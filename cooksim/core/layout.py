"""Kitchen layouts: textual parsing, JSON (de)serialization and procedural
generation."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .enums import (
    CHAR_TO_TERRAIN,
    SOURCE_ITEMS,
    TERRAIN_TO_CHAR,
    Terrain,
)


@dataclass
class Layout:
    name: str
    grid: List[List[Terrain]]            # grid[y][x]
    start_positions: List[Tuple[int, int]]

    @property
    def height(self) -> int:
        return len(self.grid)

    @property
    def width(self) -> int:
        return len(self.grid[0]) if self.grid else 0

    def terrain_at(self, x: int, y: int) -> Terrain:
        return self.grid[y][x]

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    # -- (de)serialization -------------------------------------------------
    @classmethod
    def from_lines(cls, name: str, lines: List[str]) -> "Layout":
        """Parse a textual layout. Digits 1-9 mark cook spawn points (on floor)."""
        width = max(len(line) for line in lines)
        grid: List[List[Terrain]] = []
        starts: Dict[int, Tuple[int, int]] = {}
        for y, line in enumerate(lines):
            row: List[Terrain] = []
            for x in range(width):
                ch = line[x] if x < len(line) else " "
                if ch.isdigit() and ch != "0":
                    starts[int(ch)] = (x, y)
                    row.append(Terrain.FLOOR)
                else:
                    row.append(CHAR_TO_TERRAIN.get(ch, Terrain.FLOOR))
            grid.append(row)
        start_positions = [starts[k] for k in sorted(starts)]
        return cls(name, grid, start_positions)

    def to_lines(self) -> List[str]:
        start_lookup = {pos: i + 1 for i, pos in enumerate(self.start_positions)}
        lines = []
        for y, row in enumerate(self.grid):
            chars = []
            for x, t in enumerate(row):
                if (x, y) in start_lookup:
                    chars.append(str(start_lookup[(x, y)]))
                else:
                    chars.append(TERRAIN_TO_CHAR[t])
            lines.append("".join(chars))
        return lines

    @classmethod
    def from_dict(cls, d: dict) -> "Layout":
        grid = [[Terrain(t) for t in row] for row in d["grid"]]
        starts = [tuple(p) for p in d.get("start_positions", [])]
        return cls(d.get("name", "custom"), grid, starts)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "grid": [[t.value for t in row] for row in self.grid],
            "start_positions": [list(p) for p in self.start_positions],
        }

    def copy(self) -> "Layout":
        return Layout(self.name, [row[:] for row in self.grid], list(self.start_positions))

    # -- validation --------------------------------------------------------
    def validate(self) -> List[str]:
        """Return a list of human-readable problems (empty == valid)."""
        issues = []
        flat = [t for row in self.grid for t in row]
        if Terrain.SERVING not in flat:
            issues.append("No serving window (S).")
        if Terrain.PLATE_SOURCE not in flat:
            issues.append("No plate dispenser (D).")
        if not any(t in SOURCE_ITEMS for t in flat):
            issues.append("No ingredient sources.")
        if not self.start_positions:
            issues.append("No cook spawn points.")
        for x, y in self.start_positions:
            if not self.in_bounds(x, y) or self.grid[y][x] != Terrain.FLOOR:
                issues.append(f"Spawn point {(x, y)} is not on a floor tile.")
        return issues


# ----------------------------------------------------------------------------
# Procedural generation
# ----------------------------------------------------------------------------
def generate_layout(
    width: int = 9,
    height: int = 7,
    n_players: int = 2,
    seed: Optional[int] = None,
    n_pots: int = 2,
    n_cutting_boards: int = 1,
    include_sink: bool = True,
    n_pans: int = 0,
    n_ovens: int = 0,
    ingredient_sources: Optional[List[Terrain]] = None,
    style: str = "ring",
) -> Layout:
    """Generate a solvable kitchen layout.

    ``style`` is one of ``"ring"`` (counters line the walls with an open
    interior) or ``"divided"`` (a central counter wall splits two halves —
    the classic "forced coordination" topology).
    """
    rng = random.Random(seed)
    if ingredient_sources is None:
        ingredient_sources = [Terrain.ONION_SOURCE, Terrain.TOMATO_SOURCE]

    width = max(6, width)
    height = max(5, height)
    grid = [[Terrain.FLOOR for _ in range(width)] for _ in range(height)]

    # Border ring of counters.
    border: List[Tuple[int, int]] = []
    for x in range(width):
        for y in (0, height - 1):
            grid[y][x] = Terrain.COUNTER
            border.append((x, y))
    for y in range(height):
        for x in (0, width - 1):
            grid[y][x] = Terrain.COUNTER
            if (x, y) not in border:
                border.append((x, y))

    if style == "divided" and width >= 7:
        # A vertical counter wall through the middle with no gap forces the
        # two cooks (placed on opposite sides) to pass items across it.
        mid = width // 2
        for y in range(1, height - 1):
            grid[y][mid] = Terrain.COUNTER

    # Required stations to place on the border (impassable cells).
    required: List[Terrain] = [Terrain.SERVING, Terrain.PLATE_SOURCE]
    required += [Terrain.POT] * n_pots
    required += [Terrain.PAN] * n_pans
    required += [Terrain.OVEN] * n_ovens
    required += [Terrain.CUTTING_BOARD] * n_cutting_boards
    if include_sink:
        required += [Terrain.SINK]
    required += list(ingredient_sources)

    # Shuffle border positions but keep corners as plain counters (corners are
    # awkward to reach). Place stations on edge midpoints.
    rng.shuffle(border)
    corners = {(0, 0), (0, height - 1), (width - 1, 0), (width - 1, height - 1)}
    slots = [p for p in border if p not in corners]
    rng.shuffle(slots)

    for terrain in required:
        if not slots:
            break
        x, y = slots.pop()
        grid[y][x] = terrain

    # Cook spawn points on interior floor.
    interior = [
        (x, y)
        for y in range(1, height - 1)
        for x in range(1, width - 1)
        if grid[y][x] == Terrain.FLOOR
    ]
    rng.shuffle(interior)
    if style == "divided" and width >= 7:
        mid = width // 2
        left = [p for p in interior if p[0] < mid]
        right = [p for p in interior if p[0] > mid]
        starts = []
        for i in range(n_players):
            pool = left if i % 2 == 0 else right
            if pool:
                starts.append(pool.pop())
        start_positions = starts
    else:
        start_positions = interior[:n_players]

    name = f"procedural_{style}_{seed}"
    layout = Layout(name, grid, start_positions)
    return layout
