"""Customer order queue with timers, rewards and penalties."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional

from .recipes import Recipe, RecipeBook


@dataclass
class Order:
    uid: int
    recipe: Recipe
    time_left: int
    time_total: int

    @property
    def urgency(self) -> float:
        return 1.0 - (self.time_left / max(1, self.time_total))

    def render_state(self) -> dict:
        return {
            "id": self.uid,
            "recipe": self.recipe.id,
            "name": self.recipe.name,
            "contents": [{"name": n, "state": s} for n, s in self.recipe.contents],
            "steps": list(self.recipe.steps),
            "ordered": self.recipe.ordered,
            "color": self.recipe.color,
            "reward": self.recipe.base_reward,
            "time_left": self.time_left,
            "time_total": self.time_total,
        }


class OrderManager:
    """Maintains a fixed-size queue of pending orders.

    Orders are drawn from the recipe book (difficulty-weighted), given a
    countdown proportional to their difficulty, and removed when fulfilled or
    when they expire (incurring a penalty).
    """

    def __init__(
        self,
        recipe_book: RecipeBook,
        rng: random.Random,
        max_orders: int = 4,
        base_time: int = 80,
        time_per_difficulty: int = 40,
        expire_penalty: float = -8.0,
        enabled: bool = True,
        recipes: Optional[List[Recipe]] = None,
    ):
        self.book = recipe_book
        self.rng = rng
        self.max_orders = max_orders
        self.base_time = base_time
        self.time_per_difficulty = time_per_difficulty
        self.expire_penalty = expire_penalty
        self.enabled = enabled
        # The pool orders are drawn from (defaults to every recipe in the book).
        self.pool: List[Recipe] = list(recipes) if recipes else list(self.book.recipes)
        if not self.pool:
            self.pool = list(self.book.recipes)
        self.orders: List[Order] = []
        self._next_uid = 0
        self._weights = [max(1, 4 - r.difficulty) for r in self.pool]
        if enabled:
            self.refill()

    def _spawn(self) -> Order:
        recipe = self.rng.choices(self.pool, weights=self._weights, k=1)[0]
        total = self.base_time + recipe.difficulty * self.time_per_difficulty
        order = Order(self._next_uid, recipe, total, total)
        self._next_uid += 1
        return order

    def refill(self) -> None:
        while len(self.orders) < self.max_orders:
            self.orders.append(self._spawn())

    def tick(self) -> float:
        """Advance timers; return penalty reward from any expirations."""
        if not self.enabled:
            return 0.0
        penalty = 0.0
        survivors: List[Order] = []
        for o in self.orders:
            o.time_left -= 1
            if o.time_left <= 0:
                penalty += self.expire_penalty
            else:
                survivors.append(o)
        self.orders = survivors
        self.refill()
        return penalty

    def fulfill(self, recipe: Recipe) -> Optional[Order]:
        """Match a delivered recipe to the most urgent matching order."""
        if not self.enabled:
            # In free-cook mode, any valid recipe scores.
            return Order(-1, recipe, 0, 1)
        matches = [o for o in self.orders if o.recipe.id == recipe.id]
        if not matches:
            return None
        order = min(matches, key=lambda o: o.time_left)
        self.orders.remove(order)
        self.refill()
        return order

    def render_state(self) -> List[dict]:
        return [o.render_state() for o in self.orders]
