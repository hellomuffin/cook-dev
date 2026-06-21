"""Recipe definitions and matching.

A recipe is identified by the exact multiset of (ingredient name, prep state)
that must sit on a delivered plate. Cooking stations do not need to know about
recipes — they simply transform ingredient states over time — so recipe
validation happens only at the serving window. This keeps the simulation
flexible: any combination a player can physically assemble can be scored.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .enums import PrepState
from .items import Plate


@dataclass(frozen=True)
class Recipe:
    id: str
    name: str
    # multiset of (name, state) required on the final plate
    contents: Tuple[Tuple[str, str], ...]
    base_reward: float
    color: str = "#ffcc66"        # used by the renderer for the order ticket
    difficulty: int = 1           # influences order timer / spawn weighting
    steps: Tuple[str, ...] = ()   # human-readable prep steps (for the UI)

    @property
    def content_keys(self) -> Tuple[Tuple[str, str], ...]:
        return tuple(sorted(self.contents))


def _c(name: str, state: PrepState) -> Tuple[str, str]:
    return (name, state.value)


# ----------------------------------------------------------------------------
# Built-in recipe catalogue. Far richer than the single onion-soup of the
# original overcooked_ai. Recipes are grouped by the kind of prep they need.
# ----------------------------------------------------------------------------
RAW = PrepState.RAW
CHOP = PrepState.CHOPPED
COOK = PrepState.COOKED

DEFAULT_RECIPES: List[Recipe] = [
    # --- Soups (boil in a pot) ----------------------------------------------
    Recipe("onion_soup", "Onion Soup", (_c("onion", COOK),) * 3, 20, "#e8c14a", 1,
           ("Drop 3 onions in a pot", "Let it cook", "Plate & serve at the pass")),
    Recipe("tomato_soup", "Tomato Soup", (_c("tomato", COOK),) * 3, 20, "#d9482f", 1,
           ("Drop 3 tomatoes in a pot", "Cook", "Plate & serve")),
    Recipe("mushroom_soup", "Mushroom Soup", (_c("mushroom", COOK),) * 3, 26, "#9c8466", 2,
           ("Drop 3 mushrooms in a pot", "Cook", "Plate & serve")),
    Recipe("mixed_soup", "Mixed Soup",
           (_c("onion", COOK), _c("onion", COOK), _c("tomato", COOK)), 30, "#cf7a3a", 2,
           ("2 onions + 1 tomato in a pot", "Cook", "Plate & serve")),
    # --- Salads (chop & assemble, no heat) ----------------------------------
    Recipe("garden_salad", "Garden Salad", (_c("lettuce", CHOP), _c("tomato", CHOP)), 18, "#5fb04a", 1,
           ("Chop lettuce", "Chop tomato", "Plate both & serve")),
    Recipe("caprese", "Caprese", (_c("tomato", CHOP), _c("cheese", RAW)), 20, "#df6d52", 1,
           ("Chop tomato", "Add cheese", "Plate & serve")),
    Recipe("greek_salad", "Greek Salad",
           (_c("lettuce", CHOP), _c("tomato", CHOP), _c("cheese", RAW)), 30, "#7fc35a", 2,
           ("Chop lettuce", "Chop tomato", "Add cheese, plate & serve")),
    # --- Burgers (fry + chop + assemble) ------------------------------------
    Recipe("burger", "Burger", (_c("bun", RAW), _c("meat", COOK)), 26, "#b5651d", 2,
           ("Fry a patty", "Add a bun", "Plate & serve")),
    Recipe("cheeseburger", "Cheeseburger",
           (_c("bun", RAW), _c("meat", COOK), _c("cheese", RAW)), 34, "#c87a2a", 3,
           ("Fry a patty", "Bun + cheese", "Plate & serve")),
    Recipe("deluxe_burger", "Deluxe Burger",
           (_c("bun", RAW), _c("meat", COOK), _c("lettuce", CHOP), _c("tomato", CHOP)), 46, "#a85a1c", 4,
           ("Fry a patty", "Chop lettuce + tomato", "Stack on a bun, plate & serve")),
    # --- Pan / fried --------------------------------------------------------
    Recipe("fried_fish", "Fried Fish", (_c("fish", COOK),), 22, "#7fa8c9", 2,
           ("Fry the fish", "Plate & serve")),
    Recipe("fried_egg", "Fried Egg", (_c("egg", COOK),), 12, "#f4d35e", 1,
           ("Crack an egg in the pan", "Fry", "Plate & serve")),
    Recipe("loaded_fries", "Loaded Fries", (_c("potato", COOK), _c("cheese", RAW)), 26, "#e6b24a", 2,
           ("Fry the potato", "Top with cheese", "Plate & serve")),
    Recipe("fish_and_chips", "Fish & Chips", (_c("fish", COOK), _c("potato", COOK)), 38, "#c8a36a", 3,
           ("Fry the fish", "Fry the potato", "Plate both & serve")),
    # --- Rice ---------------------------------------------------------------
    Recipe("rice_bowl", "Rice Bowl", (_c("rice", COOK),), 12, "#efe7d2", 1,
           ("Cook the rice", "Plate & serve")),
    Recipe("sushi", "Sushi", (_c("rice", COOK), _c("fish", CHOP)), 36, "#a9c4d8", 3,
           ("Cook rice", "Chop raw fish", "Plate both & serve")),
    Recipe("fried_rice", "Fried Rice",
           (_c("rice", COOK), _c("egg", COOK), _c("onion", CHOP)), 44, "#dcb86a", 4,
           ("Cook rice", "Fry egg", "Chop onion, plate all & serve")),
    # --- Oven / baked -------------------------------------------------------
    Recipe("pizza", "Pizza",
           (_c("dough", COOK), _c("tomato", COOK), _c("cheese", COOK)), 42, "#e08a3c", 4,
           ("Dough + tomato + cheese into the oven", "Bake", "Plate & serve")),
    Recipe("veggie_bake", "Veggie Bake",
           (_c("potato", COOK), _c("mushroom", COOK), _c("cheese", COOK)), 40, "#caa15a", 4,
           ("Potato + mushroom + cheese in the oven", "Bake", "Plate & serve")),
]


def feasible_recipes(recipes: List[Recipe], layout) -> List[Recipe]:
    """Subset of ``recipes`` that the given layout can physically produce.

    Requires a source for every ingredient, a cook station for any cooked
    component and a cutting board for any chopped component.
    """
    from .enums import COOK_STATIONS, SOURCE_ITEMS, Terrain

    flat = {t for row in layout.grid for t in row}
    available = {SOURCE_ITEMS[t] for t in flat if t in SOURCE_ITEMS}
    has_cook = bool(flat & COOK_STATIONS)
    has_board = Terrain.CUTTING_BOARD in flat

    out = []
    for r in recipes:
        names = {n for n, _ in r.contents}
        states = {s for _, s in r.contents}
        if not names <= available:
            continue
        if PrepState.COOKED.value in states and not has_cook:
            continue
        if PrepState.CHOPPED.value in states and not has_board:
            continue
        out.append(r)
    return out


class RecipeBook:
    """Holds the active recipes and answers matching queries."""

    def __init__(self, recipes: Optional[List[Recipe]] = None):
        self.recipes: List[Recipe] = list(recipes if recipes is not None else DEFAULT_RECIPES)
        self._by_keys: Dict[Tuple, Recipe] = {r.content_keys: r for r in self.recipes}
        self._by_id: Dict[str, Recipe] = {r.id: r for r in self.recipes}

    def match_plate(self, plate: Plate) -> Optional[Recipe]:
        """Return the recipe a plate exactly satisfies, or None."""
        if plate.dirty or not plate.contents:
            return None
        # A burnt ingredient can never satisfy a recipe.
        if any(c.state == PrepState.BURNT for c in plate.contents):
            return None
        return self._by_keys.get(plate.content_keys)

    def get(self, recipe_id: str) -> Optional[Recipe]:
        return self._by_id.get(recipe_id)

    def __iter__(self):
        return iter(self.recipes)

    def to_dict(self) -> List[dict]:
        return [
            {
                "id": r.id,
                "name": r.name,
                "contents": [{"name": n, "state": s} for n, s in r.contents],
                "reward": r.base_reward,
                "color": r.color,
                "difficulty": r.difficulty,
                "steps": list(r.steps),
            }
            for r in self.recipes
        ]
