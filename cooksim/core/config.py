"""Tunable simulation parameters."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class GameConfig:
    # Capacities ----------------------------------------------------------
    plate_capacity: int = 4
    pot_capacity: int = 3
    pan_capacity: int = 2
    oven_capacity: int = 3
    # Timings (in ticks) --------------------------------------------------
    pot_cook_time: int = 20
    pot_burn_time: int = 25
    pan_cook_time: int = 15
    pan_burn_time: int = 18
    oven_cook_time: int = 28
    oven_burn_time: int = 30
    chop_time: int = 4
    wash_time: int = 5
    # Orders --------------------------------------------------------------
    orders_enabled: bool = True
    max_orders: int = 4
    order_base_time: int = 110
    order_time_per_difficulty: int = 55
    expire_penalty: float = -6.0
    # Plates / serving ----------------------------------------------------
    return_dirty_plates: bool = True
    invalid_serve_penalty: float = -3.0
    trash_penalty: float = 0.0
    # Episode -------------------------------------------------------------
    horizon: int = 800
    # Reward shaping ------------------------------------------------------
    use_shaped_rewards: bool = True
    shaping_factor: float = 1.0
    shaped_rewards: Dict[str, float] = field(
        default_factory=lambda: {
            "place_in_pot": 0.3,
            "start_cooking": 0.2,
            "useful_chop": 0.3,
            "soup_pickup": 1.0,
            "plate_pickup": 0.1,
        }
    )

    def shaped(self, event: str) -> float:
        if not self.use_shaped_rewards:
            return 0.0
        return self.shaped_rewards.get(event, 0.0) * self.shaping_factor
