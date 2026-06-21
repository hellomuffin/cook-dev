"""Run the built-in GreedyChef heuristic on each layout and report deliveries.

This is also a quick sanity benchmark for the simulator.
"""
from cooksim import KitchenGame, GameConfig, load_layout, list_layouts
from cooksim.agents import GreedyChef


def run(layout_name, horizon=1500, seed=0):
    g = KitchenGame(load_layout(layout_name), config=GameConfig(horizon=horizon), seed=seed)
    bots = [GreedyChef(seed=i + 1) for i in range(g.n_players)]
    while not g.done:
        g.step([bots[i](g, i) for i in range(g.n_players)])
    return g


if __name__ == "__main__":
    for name in list_layouts():
        g = run(name)
        print(
            f"{name:22s} cooks={g.n_players} score={g.score:8.1f} "
            f"served={g.stats.get('deliveries', 0):3d} failed={g.stats.get('failed_deliveries', 0)}"
        )
