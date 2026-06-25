"""Render a CookSim game state to a legible top-down image for a VLM to watch.

This is the "visual stream" the high-level (VLM) controller observes. Stations
carry an ID badge (numbered row-major within each type) and a coordinate grid is
drawn faintly, so the model can ground commands as "pot 2" or "(2,0)" when plain
language is ambiguous. Uses plain text + colored tiles (no emoji fonts needed).
"""
from __future__ import annotations

from typing import Dict, Tuple

from PIL import Image, ImageDraw, ImageFont

from ..core.enums import SOURCE_ITEMS, Direction, Terrain
from ..core.items import Ingredient, Plate
from ..core.stations import CookStation, CuttingBoard, Sink

Pos = Tuple[int, int]

_BG = (24, 21, 31)
_FLOOR = (40, 36, 51)
_COL = {
    Terrain.WALL: (28, 25, 36), Terrain.COUNTER: (74, 64, 52),
    Terrain.POT: (70, 96, 120), Terrain.PAN: (90, 80, 110), Terrain.OVEN: (120, 70, 50),
    Terrain.CUTTING_BOARD: (95, 80, 60), Terrain.SINK: (50, 95, 120),
    Terrain.SERVING: (150, 120, 50), Terrain.TRASH: (110, 60, 60),
    Terrain.PLATE_SOURCE: (110, 110, 120),
}
_LABEL = {Terrain.POT: "POT", Terrain.PAN: "PAN", Terrain.OVEN: "OVEN",
          Terrain.CUTTING_BOARD: "BOARD", Terrain.SINK: "SINK", Terrain.SERVING: "PASS",
          Terrain.TRASH: "TRASH", Terrain.PLATE_SOURCE: "PLATES", Terrain.COUNTER: ""}
_IDTYPES = {Terrain.POT, Terrain.PAN, Terrain.OVEN, Terrain.CUTTING_BOARD, Terrain.SINK}
_DIRDELTA = {Direction.NORTH: (0, -1), Direction.SOUTH: (0, 1), Direction.EAST: (1, 0), Direction.WEST: (-1, 0)}


def _font(sz):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def station_ids(game) -> Dict[Pos, str]:
    """Stable per-type ID label for each station tile (row-major, 1-based)."""
    out: Dict[Pos, str] = {}
    for terr in _IDTYPES:
        tiles = sorted((x, y) for y, row in enumerate(game.layout.grid)
                       for x, t in enumerate(row) if t == terr)
        for i, p in enumerate(tiles, 1):
            out[p] = f"{_LABEL.get(terr, '?')} {i}"
    return out


def _ing_short(i: Ingredient) -> str:
    s = {"raw": "", "chopped": "chop", "cooked": "ckd", "burnt": "BURNT"}.get(i.state.value, "")
    return i.name[:5] + ("·" + s if s else "")


def _station_status(game, pos) -> str:
    s = game.stations.get(pos)
    if isinstance(s, CookStation):
        if not s.contents:
            return "empty"
        names = "+".join(i.name[:4] for i in s.contents)
        if s.status == "cooking":
            return f"{names} cooking {int(s.timer / max(1, s.cook_time) * 100)}%"
        if s.status == "cooked":
            return f"{names} DONE"
        if s.status == "burnt":
            return f"{names} BURNT"
        return f"{names} raw"
    if isinstance(s, CuttingBoard):
        if s.item is None:
            return "empty"
        return _ing_short(s.item) + ("" if s.item.state.value == "chopped" else f" {int(s.progress/max(1,s.chop_time)*100)}%")
    if isinstance(s, Sink):
        return f"dirty{s.dirty} clean{s.clean_ready}"
    return ""


def render_observation(game, cell: int = 78, pad: int = 26) -> Image.Image:
    W, H = game.layout.width, game.layout.height
    img = Image.new("RGB", (W * cell + 2 * pad, H * cell + 2 * pad + 18), _BG)
    d = ImageDraw.Draw(img)
    f_lab, f_sm, f_id, f_hud = _font(15), _font(12), _font(12), _font(14)
    ids = station_ids(game)

    for y in range(H):
        for x in range(W):
            terr = game.layout.grid[y][x]
            x0, y0 = pad + x * cell, pad + y * cell
            x1, y1 = x0 + cell - 3, y0 + cell - 3
            col = _COL.get(terr, _FLOOR)
            d.rounded_rectangle([x0, y0, x1, y1], radius=8, fill=col)
            cx = x0 + cell // 2
            if terr in SOURCE_ITEMS:
                d.text((cx, y0 + cell * 0.30), SOURCE_ITEMS[terr], fill=(180, 230, 170), font=f_lab, anchor="mm")
                d.text((cx, y0 + cell * 0.62), "source", fill=(140, 170, 135), font=f_sm, anchor="mm")
            elif terr in _LABEL and _LABEL[terr]:
                d.text((cx, y0 + 16), _LABEL[terr], fill=(235, 230, 240), font=f_lab, anchor="mm")
                st = _station_status(game, (x, y))
                if st:
                    cc = (250, 210, 120) if ("DONE" in st or "chop" in st and "%" not in st) else \
                         (250, 140, 120) if "BURNT" in st else (200, 195, 210)
                    d.text((cx, y0 + cell * 0.62), st, fill=cc, font=f_sm, anchor="mm")
                if (x, y) in ids:
                    n = ids[(x, y)].split()[-1]
                    d.ellipse([x1 - 20, y0 + 2, x1 - 2, y0 + 20], fill=(90, 169, 230))
                    d.text((x1 - 11, y0 + 11), n, fill=(8, 30, 50), font=f_id, anchor="mm")
            # loose item on a counter
            if (x, y) in game.counter_items:
                it = game.counter_items[(x, y)]
                lab = _ing_short(it) if isinstance(it, Ingredient) else "plate"
                d.text((cx, y0 + cell * 0.62), lab, fill=(255, 220, 160), font=f_sm, anchor="mm")
            # faint coordinate label
            d.text((x0 + 4, y0 + 2), f"{x},{y}", fill=(95, 88, 110), font=f_sm, anchor="lt")

    # cooks
    for c in game.cooks:
        cx, cy = pad + c.x * cell + cell // 2, pad + c.y * cell + cell // 2
        d.ellipse([cx - 17, cy - 17, cx + 17, cy + 17], fill=(240, 200, 90), outline=(30, 25, 20), width=2)
        d.text((cx, cy), str(c.id + 1), fill=(40, 30, 10), font=f_lab, anchor="mm")
        dx, dy = _DIRDELTA[c.direction]
        d.polygon([(cx + dx * 22, cy + dy * 22), (cx + dx * 12 - dy * 7, cy + dy * 12 - dx * 7),
                   (cx + dx * 12 + dy * 7, cy + dy * 12 + dx * 7)], fill=(255, 235, 150))
        held = c.holding
        if held is not None:
            if isinstance(held, Plate):
                txt = "plate" + (("[" + ",".join(_ing_short(i) for i in held.contents) + "]") if held.contents else "")
                if held.dirty:
                    txt = "dirty plate"
            else:
                txt = _ing_short(held)
            d.rounded_rectangle([cx - 46, cy - 38, cx + 46, cy - 22], radius=5, fill=(20, 18, 26))
            d.text((cx, cy - 30), "hold:" + txt, fill=(255, 225, 160), font=f_sm, anchor="mm")

    # HUD: active orders
    orders = ", ".join(o.recipe.name for o in game.orders.orders[:4]) or "(free cook)"
    d.text((pad, H * cell + pad + 4), f"orders: {orders}   score:{int(game.score)}",
           fill=(200, 195, 210), font=f_hud, anchor="lt")
    return img
