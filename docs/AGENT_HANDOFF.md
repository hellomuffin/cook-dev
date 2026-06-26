# CookSim — Agent Handoff & Engineering Guide

This is the single source of truth for continuing work on **CookSim** (an Overcooked-style
cooking simulator) and its **streaming-VLM control** research. It documents the codebase,
the environment, every script, the exact commands, the findings, and the gotchas. Read it
top-to-bottom once; after that the cheat-sheet (§12) is usually enough.

Last updated: 2026-06-25.

---

## 0. TL;DR / current state

- **What CookSim is:** a from-scratch Overcooked-style game — Python engine (`cooksim/core`),
  Gym/PettingZoo wrappers (`cooksim/envs`), a real-time 3D WebGL renderer (`web/js/render3d.js`),
  a FastAPI+WebSocket server (`cooksim/server`), a level editor, and a teaching website (`site/`).
- **Where it lives now:** GitHub **`github.com/hellomuffin/cook-bench`** (PUBLIC). The old
  `hellomuffin/cooksim` repo was made **private** (reversible takedown) — *do not use it*. Local
  working copy: **`~/cooksim`** (git `origin` already points to cook-bench).
- **Live website:** **https://hellomuffin.github.io/cook-bench/** (served from the `gh-pages` branch).
  Subpages: `/taxonomy.html` (recipe/event space) and `/actions.html` (VLM control + streaming videos).
- **Active research:** controlling the cook with a **streaming VLM**. Two working approaches:
  (a) **offline sliding-window** (best), (b) **Gemini Live** (real streaming API). See §8–9.
- **Env:** micromamba env **`cooksim`**. Headless GPU rendering needs `LD_LIBRARY_PATH`. Gemini API
  key is the user's personal AIza key (see §10). H200 GPUs available.

---

## 1. Repository layout

```
~/cooksim/
├── cooksim/                     # the Python package
│   ├── core/                    # the engine (renderer/framework-agnostic)
│   │   ├── game.py              # KitchenGame: the authoritative sim. step(), _interact*, render_state()
│   │   ├── enums.py             # Action (N/S/E/W/INTERACT/STAY), Direction, Terrain, SOURCE_ITEMS
│   │   ├── stations.py          # CookStation (pot/pan/oven), CuttingBoard, Sink
│   │   ├── items.py             # Ingredient, Plate (content_keys = sorted multiset, content_seq = order)
│   │   ├── recipes.py           # Recipe, RecipeBook, DEFAULT_RECIPES (19), feasible_recipes()
│   │   ├── orders.py            # OrderManager (order queue, timers, rewards)
│   │   ├── layout.py            # Layout, generate_layout (procedural)
│   │   ├── player.py            # Cook (x, y, direction, holding, facing_pos())
│   │   └── config.py            # GameConfig (capacities, cook/burn times, shaped rewards)
│   ├── envs/                    # Gymnasium CookSim-v0 + PettingZoo CookSimParallelEnv
│   ├── agents/
│   │   ├── heuristic.py         # GreedyChef (BFS planner bot) + helpers (_bfs_first_step, _adjacent_floor, _dir_between)
│   │   ├── highlevel.py         # ★ manipulation-level NL controller (parse→ground→skills→HighLevelController, run_plan)
│   │   └── observe.py           # top-down PIL renderer — DEPRECATED for VLM input (leaks state as text); see §8.4
│   ├── server/app.py            # FastAPI; serves web/ at /static; WebSocket /ws; `cooksim-server` entrypoint
│   └── layouts/builtin.py       # 8 built-in layouts (cramped_room, bistro, diner, grand_kitchen, …)
├── web/                         # browser front-end (served by the server)
│   ├── index.html               # the live game client (import map → three.js CDN)
│   ├── headless_render.html     # ★ headless single-frame 3D renderer harness (for VLM RGB capture)
│   ├── js/render3d.js           # ★ the 3D renderer (THREE.js). setState(state)/update(dt). window.KitchenRenderer
│   ├── js/game.js               # live client (WebSocket, HUD, URL auto-start)
│   └── js/editor.js, editor.html
├── site/                        # the teaching website (deployed to gh-pages)
│   ├── index.html               # main walkthrough
│   ├── taxonomy.html            # interactive recipe/event-space taxonomy
│   ├── actions.html             # ★ VLM-control explainer + streaming videos
│   └── assets/clips/*.mp4        # 13 gameplay clips ; assets/stream/*.mp4 = VLM streaming videos
├── tools/                       # scripts (see §11)
├── tests/                       # pytest: test_core.py, test_envs.py, test_highlevel.py
├── docs/                        # highlevel_actions.md, AGENT_HANDOFF.md (this file), screenshots
├── tmp/                         # scratch (gitignored) — render frames, logs, recordings
└── pyproject.toml               # deps; extras: [server], [vlm]=pillow+google-genai, [dev]=pytest
```

---

## 2. Environment & setup (read this before running anything)

- **micromamba** (NOT system conda). Binary: `~/.local/bin/micromamba`; root `~/micromamba`.
  Env name: **`cooksim`**. Run things as: `micromamba run -n cooksim <cmd>`.
- **Headless Chromium / WebGL** (for rendering & recording) needs the env's libs on the path:
  ```bash
  export LD_LIBRARY_PATH="/fsx/home/chenhao.zheng/micromamba/envs/cooksim/lib:$LD_LIBRARY_PATH"
  ```
  Omitting this → `libatk-1.0.so.0: cannot open shared object file` when launching Playwright.
- **GPU**: 8× NVIDIA H200. Headless WebGL renders on the GPU via these Chromium args (in
  `tools/record_clips.py` and the streaming agents as `GPU`):
  `--no-sandbox --use-gl=angle --use-angle=vulkan --enable-features=Vulkan --ignore-gpu-blocklist --enable-gpu`
- **ANGLE/Vulkan shadow bug**: shadow maps black out the scene headlessly → disabled by setting
  `window.__noshadow=true` before the renderer loads. Real browsers keep shadows.
- **WebGL screenshot**: `render3d.js` enables `preserveDrawingBuffer` only when `window.__capture`
  is set (so `canvas.toDataURL()` works for headless capture). No effect on normal play.
- **ffmpeg** lives in the env (not on PATH): call `micromamba run -n cooksim ffmpeg …`.
- **Pillow + google-genai** are the `[vlm]` extra: `pip install -e ".[vlm]"` or they're already installed.
- **Scratch**: use `~/cooksim/tmp/` (gitignored). **Do NOT use `/tmp`** — it gets cleaned and will
  delete scripts/results mid-run. (This is a standing user rule.)

### Running the server
```bash
export LD_LIBRARY_PATH="/fsx/home/chenhao.zheng/micromamba/envs/cooksim/lib:$LD_LIBRARY_PATH"
micromamba run -n cooksim python -m cooksim.server.app --port 8085   # serves web/ at /static, ws at /ws
```
**Gotchas:** (a) launching the server with a nested `( … & )` subshell often fails to actually start it
— launch it as its own background task/command. (b) If a previous server is hung it may hold the port;
`pkill -9 -f cooksim.server.app`, wait 2–3s, then start on a **fresh port** (e.g. 8087/8088) if bind fails.
(c) The headless render page is at `http://127.0.0.1:<port>/static/headless_render.html`.

---

## 3. The engine (cooksim/core) — what an agent must know

- **Actions (per cook, per tick):** `0 N, 1 S, 2 E, 3 W, 4 INTERACT, 5 STAY`. `game.step([a0, a1, …])`
  takes one action per cook and advances one tick. There is **no "cook"/"deliver" command** — every
  interaction is the same `INTERACT` button applied to the faced tile.
- **Orient-first movement** (`_resolve_movement`): a MOVE only *turns* the cook if it isn't already
  facing that way (no step that tick); it walks only once already facing. So turning and acting/moving
  are never combined. `last_action` is `"turn"` when a move didn't change the cell, else `"walk"`.
- **INTERACT** acts on `cook.facing_pos()` — the single adjacent tile the cook faces. Only `FLOOR` is
  walkable; stations/sources/counters are non-walkable and interacted with from an adjacent floor tile.
- **Stations** (`stations.py`): pot(cap 3, cook 20/burn 25), pan(cap 2, 15/18), oven(cap 3, 28/30) all
  output `cooked`; cutting board chops (4 interactions); sink washes. Cooking stations **auto-start when
  full** (`auto_start`), else INTERACT with empty hands when `status=="raw"` to start. After `cook_time`
  → `cooked`; after `+burn_time` → `burnt` (burnt never matches a recipe).
- **Recipes** (`recipes.py`): a recipe = a multiset of `(ingredient, state)` matched **set-wise** at the
  serving pass (`RecipeBook.match_plate`). 12 ingredients × {raw, chopped, cooked}. 19 authored recipes.
- **Order-sensitive recipes (NEW):** `Recipe.ordered: bool`. When true, `contents` is the required
  **addition sequence** and `accepts(plate)` checks `plate.content_seq == contents`. The book indexes a
  multiset → **list** of recipes (so order disambiguates same-ingredient dishes). 6 ordered: burger,
  cheeseburger, deluxe_burger, loaded_fries, sushi, fried_rice. *Subtlety:* an in-progress ordered dish
  can BE another complete recipe (bun+patty = Burger en route to Cheeseburger) — the GreedyChef bot
  reasons about its committed target, not "any recipe a plate matches".
- **render_state()** returns the full JSON the renderer consumes: terrain grid, stations (with
  `type`, `contents`, `status`, `progress`, board `item`/`done`, sink counts), `players`
  (`x,y,dir,dir_name,holding,action`), `orders`, `score`, `stats`.
- **Cook facing in the renderer:** front of the model is +z (yaw 0 = south). `DIR_YAW` in render3d.js:
  `north:π, south:0, east:π/2, west:-π/2` (east/west were once swapped — fixed; if a cook looks
  backwards, that map is the suspect).

---

## 4. The website & deployment

Three pages in `site/` (self-contained HTML, shared dark CSS, no build step):
`index.html` (walkthrough), `taxonomy.html` (recipe/event-space, click-to-expand), `actions.html`
(VLM control + streaming videos). All use **relative** links so they work at any base path.

**GitHub Pages serves the `gh-pages` branch ROOT** (i.e. the *contents* of `site/` live at the branch
root, not under `/site/`). The site lives at `https://hellomuffin.github.io/cook-bench/`.

### Deploy procedure (copy `site/` → gh-pages root, via a worktree)
```bash
cd ~/cooksim
git add site/ && git commit -m "…"           # commit source on main first
git push origin main
git fetch -q origin gh-pages
rm -rf tmp/ghp_wt && git worktree add -q tmp/ghp_wt gh-pages
rsync -a --delete --exclude='.git' --exclude='.nojekyll' site/ tmp/ghp_wt/
touch tmp/ghp_wt/.nojekyll
git -C tmp/ghp_wt add -A && git -C tmp/ghp_wt commit -m "Deploy: …"
git -C tmp/ghp_wt push origin gh-pages
git worktree remove tmp/ghp_wt --force
```
Then poll the Pages build (token is embedded in the remote URL):
```bash
tok=$(git remote get-url origin | grep -oP 'ghp_[A-Za-z0-9]+')
until [ "$(curl -s -H "Authorization: token $tok" \
  https://api.github.com/repos/hellomuffin/cook-bench/pages/builds/latest | grep -oP '"status":\s*"\K[^"]+')" = "built" ]; do sleep 6; done
curl -s -o /dev/null -w "%{http_code}\n" "https://hellomuffin.github.io/cook-bench/actions.html?cb=$RANDOM"
```
**Verify pages render with no JS errors** before deploying: `tools/check_page.py` (Playwright) loads a
page, collects console/page errors, screenshots. (Note it expects a `#expandAll` button → only fully
works on taxonomy.html; for others write a 10-line inline Playwright check.) Always eyeball a screenshot
with the Read tool.

### Recording the 13 gameplay clips (site/assets/clips)
```bash
# 1) start the server (port 8085). 2) record (GPU, __noshadow):
export LD_LIBRARY_PATH=…/cooksim/lib:$LD_LIBRARY_PATH
micromamba run -n cooksim python tools/record_clips.py        # → tmp/vids/*.webm
bash tools/encode_clips.sh tmp/vids                           # → site/assets/clips/*.mp4 + .jpg
```

---

## 5. ★ The VLM control system (cooksim/agents/highlevel.py)

A streaming VLM drives the cook with **manipulation-level** natural-language commands. The design
principle (decided with the user): **abstract the motor control, never the task functionality.** A
command is a single physical act; it is never a sub-goal. The VLM must compose a chain.

### Command taxonomy
- **Locomotion (the only mover, the only command that names a target):**
  `go to <object>` — BFS-navigate adjacent and face it. Carries **grounding qualifiers**:
  - content: `the pot cooking rice`
  - state: `the done pot`, `the empty board`, `the dirty plate`
  - spatial: `the left pot`, `the nearest pot`, `the top-left pan`
  - identity: `pot 2` (stations numbered row-major within a type)
  - exact: `the pot at (2,0)` or bare `go to (2,0)`
- **Manipulations (act on the FACED tile only; one object op; never move; FAIL if out of position):**
  `pick up`, `put down`, `chop`, `turn on`, `scoop`, `serve`, `wash`, `discard`, `wait [until <obj> is <state>]`.
  A raw low-level action (`north/…/interact/stay` or int 0–5) is also accepted.

### Implementation (key symbols)
- `parse(text) -> Intent`  (verb + obj + ingredient + quals). Object detection picks the **earliest-
  mentioned** station word (handles verbose model replies like "plates on the left counter" → plate).
- `ground(intent, game, cook) -> Target(tile, label, ambiguous)`  — resolves qualifiers vs live state.
- Skills: `GoTo`, `Manipulate`, `Wait`, `Raw`; each `step(game, cook) -> (low_action, status)`.
- `HighLevelController(game, cook_id)` — `.issue(action)` then `.step()` each tick; `.status` ∈
  {running, done, failed, idle}, `.error` has a human reason. **Defensive**: never raises, never
  corrupts engine state; bad commands → clean `failed`.
- `_validate(verb, game, cook) -> (ok, reason)` and `_MANIP` set — used by streaming agents to decide
  if a manipulation is executable *now* (for the premature-manipulation / queuing logic).
- `run_one(game, cmd)` / `run_plan(game, [cmds])` — execute commands, stepping the sim; return per-cmd
  `{command, status, ticks, error}`.

### Tests: `tests/test_highlevel.py` (18 cases, all green)
Grounding (content/identity/coord/ambiguity), success chains deliver, every out-of-position
manipulation fails with a reason, and **garbage + fuzz never crash the engine / corrupt state / score
by accident** (the key robustness invariant). Run: `micromamba run -n cooksim python -m pytest tests/ -q`
(40 total across core/envs/highlevel).

---

## 6. ★ Headless RGB rendering harness (the VLM's "eyes")

`web/headless_render.html` drives the REAL `render3d.js` headlessly and returns a PNG data URL:
- `window.__renderFrame(stateJson)`: `setState(state)` → `update(1.0)` (snaps cook to its tile) →
  `update(0.05)` → overrides the follow-camera with a **FIXED whole-kitchen camera** (so a viewer can
  judge absolute position & arrival frame-to-frame; the follow-cam re-centers and hides motion) →
  `composer.render()` → `canvas.toDataURL("image/png")`.
- Sets `window.__capture=true` (preserveDrawingBuffer) and `window.__noshadow=true`.

**Python side** (in the streaming agents): authoritative `KitchenGame` in Python; per tick:
`png = await page.evaluate("(s)=>window.__renderFrame(s)", json.dumps(game.render_state()))` → decode
base64 → that PNG is the VLM observation (pure 3D RGB, **no text/state**). This is the honest input.
The earlier `observe.py` top-down PIL renderer drew the state *as text labels* (station status,
"holding", coords) → **do not use it as VLM input** (it leaks the answer). Keep it only as a quick
debug viz.

---

## 7. ★ Streaming-VLM agents (tools/) — four generations, newest is best

All agents: pure-RGB input (except the two earliest), per-frame/decision loop, compose a time-aligned
video (`left = the exact frame the model sees | right = a HUMAN-ONLY HUD with the action, button,
command log, outcome`), encode mp4 with ffmpeg.

| Script | Method | Input | Status |
|---|---|---|---|
| `gemini_cook.py` | one call per *decision point* (skill completes) | state-leaking image + state text | early / superseded |
| `gemini_stream_agent.py` | per-frame, auto-continue while a skill runs | state-leaking image | superseded (leaks state) |
| `gemini_stream_rgb.py` | **per-frame**, pure RGB, single `continue` no-op | real 3D RGB | honest, works |
| `gemini_stream_sw.py` | **★ sliding window** (last 5 frames + last 5 responses, ~1/s, empty-or-action) | real 3D RGB | **best offline method** |
| `gemini_live_agent.py` | **★ real Gemini Live** (bidi session, per-frame ping, audio→transcript) | real 3D RGB | works (audio models weak) |

`gemini_stream_rgb.py` is the **shared library** — `gemini_stream_sw.py` and `gemini_live_agent.py`
import `make_game, GPU, PAGE, render_frame, composite, encode, hold_str, _MANIP` from it. `PAGE`
honors `COOKSIM_PAGE` env (point it at whatever port the server is on).

### Tasks (in `make_game(task)` in gemini_stream_rgb.py)
- `loaded_fries` (diner): fry potato → plate → add cheese → serve (ordered).
- `soup_pick` (bistro): **instance-grounding test** — two pots cooking, **left = tomato (red)**,
  **right = onion (yellow)**; order is Onion Soup; the VLM must *visually* pick the onion pot. Pots
  cook fast and **never burn** (`pot_cook_time=10, pot_burn_time=1e9`) so the test is the instance
  choice, not a race. (Earlier bug: with cook_time 34/default burn, the soup burned before a slow
  streaming loop could scoop.)

### Controller semantics used by the streaming agents (important)
- **Premature manipulation = harmless no-op** that does NOT abort navigation (a `scoop` issued mid-walk
  is ignored, the cook keeps walking) — prevents catastrophic abort-spirals.
- **Atomic navigation** (live agent): a new `go to` does not abort an in-progress walk.
- **Intent-queued manipulations** (live agent): a manipulation said too early is remembered and
  **auto-fires when the cook reaches position** — so frame-perfect timing isn't required. (If you want
  to measure frame-perfect timing instead, make this a flag.)

---

## 8. ★ Findings — streaming-VLM methods & models (the research result)

### Model recommendations (from research + colleagues)
- **Online/streaming, open-source:** **StreamingVLM** (Qwen2.5-VL-7B fine-tune, arXiv:2510.09608) is the
  most on-point. Alts: Qwen2.5-Omni-7B, MiniCPM-o 2.6, VITA-1.5.
- **Offline/high-accuracy, open-source:** **Qwen2.5-VL-72B** (or InternVL3-78B, MIT-licensed).
- **Colleagues' guidance (Honglu, Kanchana):** offline > online for video (arXiv:2507.09313 Tab 3);
  the strong inference-time policy is a **sliding window** (arXiv:2604.07634 §4.1).

### Two correct ways to do "streaming" with Gemini
1. **Offline → streaming (sliding window) — BEST.** About once per second, send the model the **last 5
   rendered frames** (oldest→newest, for motion) + its **last 5 responses**, and it answers with an
   **empty action (no-op) or one command**. Implemented in `gemini_stream_sw.py`. On the latest models
   (`gemini-3.1-pro-preview`, `gemini-3.5-flash`) **both succeed** on the instance task (+21), pure RGB,
   no state leak. The multi-frame motion window is what lets the model judge "has my command finished".
2. **Gemini Live (real streaming API).** Audio-native models; you get text via
   `output_audio_transcription`. **Gemini Live is NOT proactive** — it only responds when prompted, so
   you **ping it every frame** ("next action?"), with all context in the `system_instruction` up front.
   The **session keeps its own internal memory** (you don't resend history). Implemented in
   `gemini_live_agent.py`. `gemini-3.1-flash-live-preview` **completes** the instance task (+21) with the
   per-frame ping + an anti-idle prompt nudge + atomic-nav + intent-queuing. The audio Live models are
   **weak & chatty** (often mistime/stall) — offline still wins.

### Offline vs Live — internal memory (a question that came up)
- **Offline `generate_content`** is **stateless** per call → memory is *external* (we resend frames +
  past responses). The sliding window is this.
- **Gemini Live session** is **stateful** → internal KV-cache memory across the session; you don't
  resend. But on the consumer key the Live models are **audio-output** (transcribe to get text). Some
  Live models elsewhere (e.g. gemini-2.0-flash-live-001) output text directly; not on this key.

### Gemini API gotchas (learned the hard way)
- **Offline**, to get clean JSON commands: set `response_mime_type="application/json"`. For **flash**,
  disable thinking (`thinking_config=ThinkingConfig(thinking_budget=0)`) and a small `max_output_tokens`.
  For **pro** (which *requires* thinking), **cap** thinking (`thinking_budget≈512`) and give a large
  `max_output_tokens` (≥2048) — otherwise thinking eats the whole budget and `resp.text` is empty.
- The model wraps JSON in ```json fences → strip them before `json.loads`.
- **Live**: config `response_modalities=["AUDIO"]`, `output_audio_transcription=AudioTranscriptionConfig()`,
  `system_instruction`, `context_window_compression=ContextWindowCompressionConfig(sliding_window=SlidingWindow())`.
  Send a frame with `session.send_realtime_input(video=PIL.Image)`, ping with
  `session.send_client_content(turns=Content(role="user", parts=[Part(text="Next action?")]), turn_complete=True)`,
  read `resp.server_content.output_transcription.text` until `turn_complete`. Instruct the model to SAY
  one short command (or "continue"); parse the transcript.
- **Latest model ids on this key:** offline `gemini-3.5-flash`, `gemini-3.1-pro-preview`,
  `gemini-3-pro-preview`, `gemini-2.5-flash/pro`. Live (bidi): `gemini-2.5-flash-native-audio-latest`,
  `gemini-3.1-flash-live-preview`. List them with the `/v1beta/models` REST call (see cheat-sheet).
- Honglu's reference Gemini-Live demo (raw websockets, Vertex) is at **`/fsx/home/chenhao.zheng/demo/demo.py`**
  — it uses the salesforce GCP project via `gcloud` auth (not available from this sandbox), but shows the
  1-FPS / ping-each-second / proactivity pattern. Our `gemini_live_agent.py` reproduces it with the
  genai SDK + the consumer key.

---

## 9. Older feedback loop (still present)

`tools/gemini_feedback.py` — sends rendered clips to Gemini (gemini-2.5-pro, Files API) and collects
structured legibility feedback; writes `feedback/round_<N>/summary.json`. Used during the renderer
"legibility iteration" (10 rounds, plateaued ~4.4/10; trajectory in the project memory). Mostly historical.

---

## 10. Secrets / accounts

- **Gemini API key (user's personal AIza key):** `<REDACTED — never commit the key; set it via the GEMINI_API_KEY env var (ask the user)>`
  Pass as `GEMINI_API_KEY=…`. (Earlier a Salesforce LLM-gateway `sk-…` key was tried but the gateway
  host is firewalled from this sandbox — use the AIza key.)
- **GitHub:** the `git remote` URL embeds a PAT (`ghp_…`) with `repo` + `delete_repo` scope. Extract it
  with `git remote get-url origin | grep -oP 'ghp_[A-Za-z0-9]+'` for API calls. Owner: `hellomuffin`.
- **Doubao / BytePlus ModelArk** (not yet wired): OpenAI-compatible, base url
  `https://ark.ap-southeast.bytepluses.com/api/v3`, 500k free tokens for new users. The streaming agents
  are model-agnostic enough to point a thin OpenAI client there for an offline comparison.

---

## 11. Scripts reference (tools/)

| Script | What it does | Run |
|---|---|---|
| `record_clips.py` | record the 13 gameplay clips on the GPU | needs server@8085 + LD_LIBRARY_PATH → `tmp/vids/*.webm` |
| `encode_clips.sh` | trim/encode webm → `site/assets/clips/*.mp4 + .jpg` | `bash tools/encode_clips.sh tmp/vids` |
| `gemini_feedback.py` | clip → Gemini legibility feedback JSON | `GEMINI_API_KEY=… python tools/gemini_feedback.py --round N` |
| `check_page.py` | headless render-check a site page (console errors + screenshot) | `python tools/check_page.py site/taxonomy.html out.png` |
| `gemini_stream_rgb.py` | per-frame pure-RGB streaming agent (+ shared lib for the others) | see cheat-sheet |
| `gemini_stream_sw.py` | **sliding-window** streaming agent (best offline) | see cheat-sheet |
| `gemini_live_agent.py` | **Gemini Live** streaming agent | see cheat-sheet |
| `gemini_cook.py`, `gemini_stream_agent.py` | earlier agents (state-leaking) — superseded | — |

---

## 12. Cheat-sheet (copy/paste)

```bash
cd ~/cooksim
export LD=/fsx/home/chenhao.zheng/micromamba/envs/cooksim/lib
export KEY="$GEMINI_API_KEY"   # set GEMINI_API_KEY in your shell first; NEVER hardcode/commit the key
RUN() { micromamba run -n cooksim env LD_LIBRARY_PATH="$LD:$LD_LIBRARY_PATH" "$@"; }

# --- tests ---
micromamba run -n cooksim python -m pytest tests/ -q                 # 40 tests

# --- run the engine + a high-level plan (no VLM, no browser) ---
micromamba run -n cooksim python -c "
from cooksim.core.game import KitchenGame; from cooksim.core.config import GameConfig
from cooksim.layouts import load_layout; from cooksim.agents.highlevel import run_plan
g=KitchenGame(load_layout('cramped_room'),config=GameConfig(horizon=4000),n_players=1,seed=0,order_recipe_ids=['onion_soup'])
plan=[c for _ in range(3) for c in ['go to the onion','pick up','go to the pot','put down']]+['turn on the pot','wait until the pot is done','go to the plates','pick up','go to the pot','scoop','go to the serving pass','serve']
print(run_plan(g,plan)[-1], 'deliveries=',g.stats.get('deliveries'))"

# --- start the render server (use a fresh port if 8085 is stuck) ---
RUN python -m cooksim.server.app --port 8088 > tmp/srv.log 2>&1 &     # launch as its own bg task
until curl -s -o /dev/null http://127.0.0.1:8088/static/headless_render.html; do sleep 1; done

# --- streaming agents (server must be up; set COOKSIM_PAGE to its port) ---
export COOKSIM_PAGE=http://127.0.0.1:8088/static/headless_render.html
GEMINI_API_KEY=$KEY RUN env COOKSIM_PAGE=$COOKSIM_PAGE GEMINI_API_KEY=$KEY \
  python tools/gemini_stream_sw.py  --task soup_pick --model gemini-3.1-pro-preview   # → tmp/sw_*.mp4
GEMINI_API_KEY=$KEY RUN env COOKSIM_PAGE=$COOKSIM_PAGE GEMINI_API_KEY=$KEY \
  python tools/gemini_live_agent.py --task soup_pick --model gemini-3.1-flash-live-preview  # → tmp/live_*.mp4

# --- list available Gemini models on the key ---
curl -s "https://generativelanguage.googleapis.com/v1beta/models?key=$KEY&pageSize=300" | \
  micromamba run -n cooksim python -c "import json,sys; d=json.load(sys.stdin); \
  print('LIVE:',[m['name'].split('/')[-1] for m in d['models'] if 'bidiGenerateContent' in m.get('supportedGenerationMethods',[])])"

# --- extract a frame from a result video to eyeball ---
micromamba run -n cooksim ffmpeg -y -i tmp/sw_gemini31propreview_soup_pick.mp4 -vf "select=eq(n\,15)" -frames:v 1 tmp/f.png
```

---

## 13. Known gotchas (will bite you)

- **cwd resets to `~` between tool calls** — always `cd ~/cooksim` (or use absolute paths). Logs/outputs
  written to `tmp/…` go missing if you ran from `~`.
- **Server backgrounding**: nested `( … & )` often doesn't start it; launch as a dedicated background
  command. If the port is stuck after a kill, switch ports.
- **`pkill … && next`**: `pkill` returns non-zero when nothing matched → breaks `&&` chains. Run it
  standalone or append `; true`.
- **LD_LIBRARY_PATH** must be exported for any Playwright/Chromium use (rendering, recording, page checks).
- **Don't write to `/tmp`** (cleaned). Use `~/cooksim/tmp/`.
- **soup_pick burning**: keep `pot_burn_time` huge for that task or the soup burns before the (slow)
  streaming loop scoops.
- **Cooked colors converge** (everything browns) → for a *visual* instance test, distinguish pots while
  they're still **cooking** (raw onion = yellow, raw tomato = red); they're indistinguishable once cooked.
- **Audio Live models are weak/chatty**: expect mistimes/stalls; the per-frame ping + anti-idle nudge +
  intent-queue + earliest-object grounding are what make them complete.

---

## 14. Open threads / suggested next steps

1. **Compositional recipe/event generator** (designed, not yet coded): sample an archetype grammar →
   fill slots from (ingredient × plausible-state) → validate with `feasible_recipes` + capacity →
   place sites on a layout to a target route-length/difficulty → set deadline. The taxonomy page
   describes the full space (≈55 plausible dishes; ~10⁷ placements/recipe).
2. **Doubao / BytePlus ModelArk** offline comparison (OpenAI-compatible; free tokens) — point a thin
   client at the sliding-window loop and compare to Gemini.
3. **Frame-perfect-timing flag** for the streaming agents (disable intent-queuing) if you want to
   *measure* the model's completion judgement rather than help it.
4. **Batch eval harness**: run sliding-window vs Live across many tasks/seeds, log success rate &
   #commands, to produce an offline-vs-online table like arXiv:2507.09313.
5. **HUD/ID overlay in-game** (the renderer already numbers cooks; numbering stations on-screen would
   help grounding for human play and for VLMs that key off "pot 2").
