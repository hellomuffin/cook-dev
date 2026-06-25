# VLM control: the manipulation-level action interface

A streaming VLM drives the CookSim cook with small **physical** commands and must
compose a chain to cook anything — no command bundles task functionality.
This doc covers the interface, the converter, the VLM observation, and the
real Gemini-in-the-loop test results.

See also the visual explainer: <https://hellomuffin.github.io/cook-dev/actions.html>

## Model choices (one online, one offline)

| Slot | Pick | Why |
|---|---|---|
| **online / streaming** | **StreamingVLM** (Qwen2.5-VL-7B fine-tune, [arxiv 2510.09608](https://arxiv.org/abs/2510.09608)) | causal, infinite-stream, stable latency/memory. Alt: Qwen2.5-Omni-7B, MiniCPM-o 2.6 |
| **offline / hi-accuracy** | **Qwen2.5-VL-72B** (or InternVL3-78B, MIT) | SOTA open video understanding + temporal grounding |

For the live tests here we used Gemini (closed) on the key already in the repo,
because it exposes *both* surfaces: `generate_content` (offline) and the Live API
(streaming). The interface is model-agnostic.

## The command set

**Locomotion — the only mover, the only command that names a target:**

- `go to <object>` — pathfind adjacent to an object and face it. Disambiguate with qualifiers:
  - content : `go to the pot cooking rice`
  - state   : `go to the done pot` / `the empty board` / `the dirty plate`
  - spatial : `the left pot` / `the nearest pot` / `the top-left pan`
  - identity: `pot 2`  (stations numbered row-major within a type)
  - exact   : `the pot at (2,0)`  — or a bare `go to (2,0)`

**Manipulation — acts on the FACED tile only, one object operation, never moves:**

`pick up` · `put down` · `chop` · `turn on` · `scoop` · `serve` · `wash` · `discard` · `wait [until <object> is <state>]`

A manipulation **fails** (with a reason) if the cook isn't standing at and facing
the right tile — so the VLM must `go to` first. Nothing fetches, sequences, or
plates for it. A raw low-level action (`north/…/interact/stay` or int 0–5) is also
accepted for fine-grained control.

## Code

- `cooksim/agents/highlevel.py` — `parse` → `ground` → skills (`GoTo`, `Manipulate`,
  `Wait`, `Raw`) → `HighLevelController`; `run_one` / `run_plan` drive a `KitchenGame`.
  Defensive: every parse/ground/exec problem becomes a clean `failed` status — it
  never raises and never corrupts engine state.
- `cooksim/agents/observe.py` — `render_observation(game)` → a legible top-down PIL
  image (station **ID badges** + a coordinate grid, cook facing arrow, held item,
  station status). This is the "video stream" the VLM watches; the IDs/coords are
  what make ambiguous language groundable.
- `tools/gemini_cook.py` — closed perception→command loop, `--mode offline|streaming`.
- `tests/test_highlevel.py` — grounding, success chains, graceful-failure & fuzz.

## Test results (real runs)

**Unit stress tests** (`pytest tests/test_highlevel.py`, 18 cases, all green):
- grounding by content / identity / coordinate / ambiguity flag;
- full onion-soup and chop→plate chains complete a delivery;
- every out-of-position manipulation fails cleanly with a helpful reason;
- garbage + fuzz (random commands × seeds) **never crash the engine, never corrupt
  state, never score by accident** — the key robustness guarantee.

**Gemini in the loop:**

| Task | Mode / model | Result |
|---|---|---|
| Onion soup (cramped_room) | offline `gemini-2.5-flash` | ✅ delivered, 20 commands |
| Onion soup (cramped_room) | streaming Live `gemini-3.1-flash-live-preview` (audio→transcript) | ✅ delivered, 20 commands |
| Garden salad (bistro) | offline `gemini-2.5-flash` | ❌ task-failed: stranded a chopped item, looped — engine stayed sane |
| Garden salad (bistro) | offline `gemini-2.5-pro` | ✅ delivered with the full double-stash (chop→plate→stash on counter→chop→retrieve→serve) |

### Findings

1. **The engine is robust.** Across pathological runs (20+ consecutive failed
   commands, negative score from order expiry, random fuzz) it never crashed,
   never entered an unknown state, and never produced a false delivery. Bad
   commands are exactly "task fails, engine fine."
2. **Difficulty scales with required planning.** A linear task (gather→cook→plate→
   serve) is easy for a small model; a single-board salad needs a *double-stash*
   (plate a chopped item, set the plate on a counter to free hands, prep the next,
   retrieve) that flash fails and pro solves. This is the intended consequence of
   withholding task functionality.
3. **The harness needs a stuck-breaker.** A VLM can repeat a failing command; the
   loop aborts after a window of mostly-failures, and the prompt teaches recovery
   (don't repeat failures; use `scoop` with a held plate, not `pick up`, to move
   chopped food; stash a partial plate on a counter).

### Reproduce

```
pytest tests/test_highlevel.py -q
GEMINI_API_KEY=... python tools/gemini_cook.py --mode offline  --recipe onion_soup
GEMINI_API_KEY=... python tools/gemini_cook.py --mode streaming --recipe onion_soup
GEMINI_API_KEY=... python tools/gemini_cook.py --mode offline --layout bistro \
    --recipe garden_salad --model gemini-2.5-pro
```
(needs `pillow` + `google-genai`; a GIF of what Gemini saw + did is written to `tmp/`.)
