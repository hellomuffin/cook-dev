"""Core simulation tests."""
import pytest

from cooksim import KitchenGame, GameConfig, load_layout, list_layouts, generate_layout
from cooksim.core.enums import Action
from cooksim.core.items import Ingredient, Plate
from cooksim.core.enums import PrepState


def test_all_builtin_layouts_valid():
    for name in list_layouts():
        lay = load_layout(name)
        assert lay.validate() == [], f"{name} invalid: {lay.validate()}"


def test_layout_roundtrip():
    lay = load_layout("bistro")
    lines = lay.to_lines()
    from cooksim.core.layout import Layout
    lay2 = Layout.from_lines("bistro2", lines)
    assert [[t.value for t in r] for r in lay.grid] == [[t.value for t in r] for r in lay2.grid]
    # dict roundtrip
    lay3 = Layout.from_dict(lay.to_dict())
    assert lay3.width == lay.width and lay3.height == lay.height


@pytest.mark.parametrize("style", ["ring", "divided"])
def test_procedural_generation_valid(style):
    for seed in range(5):
        lay = generate_layout(width=11, height=7, n_players=2, style=style, seed=seed)
        assert lay.validate() == []


def test_movement_and_collision():
    from cooksim.core.enums import Direction
    g = KitchenGame(load_layout("cramped_room"), n_players=2, seed=0)
    # two cooks cannot occupy the same cell; swapping is blocked
    c0, c1 = g.cooks
    c0.x, c0.y = 1, 2
    c1.x, c1.y = 2, 2
    # already facing the contested direction (orient-first: a turn step would
    # otherwise just rotate them), so this step is a real move attempt
    c0.direction, c1.direction = Direction.EAST, Direction.WEST
    # c0 tries to move east into c1, c1 tries west into c0 -> both blocked
    g.step([int(Action.EAST), int(Action.WEST)])
    assert (c0.x, c0.y) != (c1.x, c1.y)
    assert (c0.x, c0.y) == (1, 2) and (c1.x, c1.y) == (2, 2)  # neither swapped


def test_orient_first_movement():
    """A cook must face a direction before it can advance in it: pressing a new
    direction only turns (no step) this tick; it walks once already facing."""
    from cooksim.core.enums import Direction
    g = KitchenGame(load_layout("open_kitchen"), n_players=1, seed=0)
    c = g.cooks[0]
    # place on open floor facing south, with east clear
    c.x, c.y, c.direction = 4, 3, Direction.SOUTH
    assert g._walkable(c.x + 1, c.y)
    start = c.pos
    g.step([int(Action.EAST)])           # not facing east yet -> pure turn
    assert c.direction == Direction.EAST and c.pos == start
    assert c.last_action == "turn"
    g.step([int(Action.EAST)])           # now facing east -> actually steps
    assert c.pos == (start[0] + 1, start[1])
    assert c.last_action == "walk"


def test_full_soup_delivery_scores():
    from cooksim.core.enums import DIRECTION_VECTORS, Direction
    g = KitchenGame(load_layout("cramped_room"), n_players=1, seed=0)
    cook = g.cooks[0]
    N, S, E, W, I, St = 0, 1, 2, 3, 4, 5
    seq = ([N, W, I, E, N, I, W, W, I, E, N, I, W, W, I, E, N, I,
            S, W, S, I] + [St] * 25 + [E, N, N, I, S, E, S, I])
    move_acts = {N, S, E, W}
    for a in seq:
        before = cook.pos
        g.step([a])
        # Orient-first: a move into an OPEN tile only turns this tick, so take
        # the follow-up step to reproduce the original one-tile-per-move path.
        if a in move_acts and cook.pos == before:
            dx, dy = DIRECTION_VECTORS[Direction(a)]
            if g._walkable(cook.x + dx, cook.y + dy):
                g.step([a])
    assert g.stats.get("deliveries", 0) == 1
    assert g.score > 15


def test_pot_cooks_and_burns():
    cfg = GameConfig(pot_cook_time=3, pot_burn_time=2)
    g = KitchenGame(load_layout("cramped_room"), config=cfg, seed=0)
    pot = g.stations[(2, 0)]
    for _ in range(3):
        pot.add(Ingredient("onion"))
    assert pot.status == "cooking"
    for _ in range(3):
        pot.tick()
    assert pot.status == "cooked"
    assert all(c.state == PrepState.COOKED for c in pot.contents)
    for _ in range(2):
        pot.tick()
    assert pot.status == "burnt"


def test_recipe_matching():
    from cooksim.core.recipes import RecipeBook
    book = RecipeBook()
    plate = Plate(contents=[Ingredient("onion", PrepState.COOKED)] * 3)
    assert book.match_plate(plate).id == "onion_soup"
    # burnt never matches
    plate.contents[0].state = PrepState.BURNT
    assert book.match_plate(plate) is None


def test_feasible_recipes_filtering():
    from cooksim.core.recipes import RecipeBook, feasible_recipes
    book = RecipeBook()
    feas = feasible_recipes(book.recipes, load_layout("cramped_room"))
    ids = {r.id for r in feas}
    assert "onion_soup" in ids
    assert "burger" not in ids  # no bun/meat sources or pan in cramped_room


def test_recipe_catalogue_size_and_steps():
    from cooksim.core.recipes import RecipeBook
    book = RecipeBook()
    assert len(book.recipes) >= 18
    # multi-step recipes carry human-readable steps and mix prep types
    deluxe = book.get("deluxe_burger")
    assert deluxe is not None and len(deluxe.steps) >= 2
    states = {st for _, st in deluxe.contents}
    assert "cooked" in states and "chopped" in states  # genuinely multi-step


def test_board_to_plate_assembly():
    """A chopped item can be scooped directly onto a carried plate."""
    from cooksim.core.stations import CuttingBoard
    from cooksim.core.enums import PrepState
    g = KitchenGame(load_layout("diner"), seed=0)
    # find a cutting board and put a chopped lettuce on it
    board_cell = next(c for c, s in g.stations.items() if isinstance(s, CuttingBoard))
    board = g.stations[board_cell]
    board.item = Ingredient("lettuce", PrepState.CHOPPED)
    cook = g.cooks[0]
    cook.holding = Plate()
    # stand next to the board facing it and interact
    bx, by = board_cell
    cook.x, cook.y = bx, by + 1 if g._walkable(bx, by + 1) else by - 1
    from cooksim.core.enums import Direction
    cook.direction = Direction.NORTH if cook.y > by else Direction.SOUTH
    g._interact(cook)
    assert any(i.name == "lettuce" and i.state == PrepState.CHOPPED for i in cook.holding.contents)
    assert board.item is None


@pytest.mark.parametrize("layout", ["pizzeria", "diner", "burger_bar"])
def test_greedy_completes_multistep(layout):
    from cooksim.agents import GreedyChef
    g = KitchenGame(load_layout(layout), config=GameConfig(horizon=2000), n_players=1, seed=0)
    bot = GreedyChef(0)
    while not g.done:
        g.step([bot(g, 0)])
    assert g.stats.get("deliveries", 0) >= 3
    assert g.stats.get("failed_deliveries", 0) == 0


def test_render_state_schema():
    g = KitchenGame(load_layout("open_kitchen"), seed=0)
    g.step([5] * g.n_players)
    s = g.render_state()
    for key in ("tick", "width", "height", "terrain", "stations", "objects",
                "players", "orders", "score", "stats", "done"):
        assert key in s
    assert len(s["players"]) == g.n_players
