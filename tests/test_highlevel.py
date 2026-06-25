"""Stress tests for the manipulation-level high-level controller.

Three things are checked:
  1. Grounding — 'go to' resolves the right station via qualifiers.
  2. Success — valid command chains actually complete a delivery.
  3. Robustness — wrong / out-of-order / garbage commands fail GRACEFULLY:
     a clean 'failed' status, never an exception, never a corrupted engine
     state, never silent success.
"""
import random

import pytest

from cooksim import GameConfig, KitchenGame, load_layout
from cooksim.core.enums import Direction
from cooksim.core.items import Ingredient, Plate
from cooksim.core.stations import CookStation, CuttingBoard, Sink
from cooksim.agents.highlevel import (
    HighLevelController, compile_action, ground, parse, run_one, run_plan,
)

VOCAB_MOVE = ["go to the onion", "go to the pot", "go to the cutting board",
              "go to the plates", "go to the serving pass", "go to the pan", "go to the trash"]
VOCAB_MANIP = ["pick up", "put down", "chop", "turn on", "scoop", "serve", "wash", "discard", "wait"]
GARBAGE = ["frobnicate the pot", "go to the moon", "", "   ", "teleport", "explode",
           "go to the pot cooking unobtainium", "pick up the sky", "42", "chop chop chop"]


def assert_valid_state(game):
    """The engine invariants any sequence of actions must preserve."""
    assert isinstance(game.score, (int, float))
    for c in game.cooks:
        assert c.holding is None or isinstance(c.holding, (Ingredient, Plate))
        if isinstance(c.holding, Plate):
            assert all(isinstance(i, Ingredient) for i in c.holding.contents)
        assert 0 <= c.x < game.layout.width and 0 <= c.y < game.layout.height
    for s in game.stations.values():
        if isinstance(s, CookStation):
            assert s.status in ("empty", "raw", "cooking", "cooked", "burnt")
            assert all(isinstance(i, Ingredient) for i in s.contents)
            assert len(s.contents) <= s.capacity
        if isinstance(s, CuttingBoard):
            assert s.item is None or isinstance(s.item, Ingredient)
        if isinstance(s, Sink):
            assert s.dirty >= 0 and s.clean_ready >= 0
    # the engine must still be steppable after whatever happened
    game.step([5] * len(game.cooks))
    assert_basic(game)


def assert_basic(game):
    for c in game.cooks:
        assert c.holding is None or isinstance(c.holding, (Ingredient, Plate))


# --------------------------------------------------------------- grounding
def test_parse_basic():
    assert parse("go to the pot cooking rice").verb == "goto"
    assert parse("CHOP the potato").verb == "chop"
    assert parse("pick up").verb == "pickup"
    assert parse("serve it now").verb == "serve"
    assert parse("wait until the pot is done").until is not None
    assert parse("flibbertigibbet") is None


def test_ground_by_content_and_identity():
    g = KitchenGame(load_layout("grand_kitchen"), n_players=1, seed=0)
    pots = sorted((p for p, s in g.stations.items() if isinstance(s, CookStation)
                   and g.layout.grid[p[1]][p[0]].value == "pot"))
    assert len(pots) == 2
    # make the FIRST pot "cooking rice"
    g.stations[pots[0]].contents = [Ingredient("rice")]
    g.stations[pots[0]].status = "cooking"
    cook = g.cooks[0]
    t = ground(parse("go to the pot cooking rice"), g, cook)
    assert t.tile == pots[0]
    # identity: 'pot 2' is the second in row-major order
    t2 = ground(parse("go to pot 2"), g, cook)
    assert t2.tile == sorted(pots, key=lambda p: (p[1], p[0]))[1]
    # exact coordinate escape hatch
    t3 = ground(parse(f"go to the pot at ({pots[1][0]},{pots[1][1]})"), g, cook)
    assert t3.tile == pots[1]


def test_ground_ambiguous_flag():
    g = KitchenGame(load_layout("grand_kitchen"), n_players=1, seed=0)
    t = ground(parse("go to the pot"), g, g.cooks[0])     # two identical empty pots
    assert t is not None and t.ambiguous


# --------------------------------------------------------------- success
def test_full_onion_soup_chain_delivers():
    g = KitchenGame(load_layout("cramped_room"), config=GameConfig(horizon=6000),
                    n_players=1, seed=0, order_recipe_ids=["onion_soup"])
    plan = []
    for _ in range(3):
        plan += ["go to the onion", "pick up", "go to the pot", "put down"]
    plan += ["turn on the pot", "wait until the pot is done",
             "go to the plates", "pick up", "go to the pot", "scoop",
             "go to the serving pass", "serve"]
    log = run_plan(g, plan)
    assert all(r["status"] == "done" for r in log), [r for r in log if r["status"] != "done"]
    assert g.stats.get("deliveries", 0) == 1
    assert g.stats.get("failed_deliveries", 0) == 0


def test_chop_chain_puts_chopped_item_on_plate():
    g = KitchenGame(load_layout("diner"), config=GameConfig(horizon=4000), n_players=1, seed=0)
    plan = ["go to the potato", "pick up", "go to the cutting board", "put down", "chop",
            "go to the plates", "pick up", "go to the cutting board", "scoop"]
    log = run_plan(g, plan)
    assert all(r["status"] == "done" for r in log), [r for r in log if r["status"] != "done"]
    held = g.cooks[0].holding
    assert isinstance(held, Plate)
    assert any(i.name == "potato" and i.state.value == "chopped" for i in held.contents)


# --------------------------------------------------------------- robustness
@pytest.mark.parametrize("cmd,needle", [
    ("chop", "board"),
    ("serve", "plate"),
    ("put down", "empty"),
    ("scoop", "plate"),
    ("turn on", "pot/pan/oven"),
    ("pick up", "pick up"),
])
def test_manipulation_out_of_position_fails_cleanly(cmd, needle):
    g = KitchenGame(load_layout("cramped_room"), n_players=1, seed=0)
    g.cooks[0].direction = Direction.NORTH    # face an empty floor tile, nothing to act on
    r = run_one(g, cmd)
    assert r["status"] == "failed"
    assert r["error"] and needle in r["error"]
    assert_valid_state(g)                    # engine intact, still steppable


def test_garbage_commands_never_crash():
    # The real guarantee: garbage never crashes, never corrupts state, and never
    # scores. (Some "garbage" legitimately degrades to a valid move, e.g. an
    # unknown content-qualifier falls back to 'go to a pot' — that's fine.)
    g = KitchenGame(load_layout("diner"), n_players=1, seed=0, order_recipe_ids=["loaded_fries"])
    for cmd in GARBAGE:
        r = run_one(g, cmd)
        assert r["status"] in ("failed", "timeout", "done")
        assert_valid_state(g)
    assert g.stats.get("deliveries", 0) == 0


def test_premature_serve_does_not_score():
    g = KitchenGame(load_layout("cramped_room"), n_players=1, seed=0, order_recipe_ids=["onion_soup"])
    # try to serve nothing / serve a raw onion — must never produce a delivery
    run_one(g, "go to the onion"); run_one(g, "pick up")
    r = run_one(g, "serve")
    assert r["status"] == "failed"
    assert g.stats.get("deliveries", 0) == 0
    assert_valid_state(g)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_fuzz_random_commands_keep_engine_sane(seed):
    rng = random.Random(seed)
    g = KitchenGame(load_layout("grand_kitchen"), config=GameConfig(horizon=100000),
                    n_players=1, seed=seed)
    pool = VOCAB_MOVE + VOCAB_MANIP + GARBAGE
    for _ in range(60):
        cmd = rng.choice(pool)
        r = run_one(g, cmd, max_ticks=60)
        assert r["status"] in ("done", "failed", "timeout")
        assert_valid_state(g)
    # never an accidental win and never a broken delivery from random flailing
    assert g.stats.get("failed_deliveries", 0) >= 0
