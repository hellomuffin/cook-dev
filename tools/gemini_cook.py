#!/usr/bin/env python3
"""Drive the CookSim cook with Gemini through the manipulation interface.

Gemini receives the recipe (text) + the current top-down frame (image) and must
emit ONE manipulation command at a time; we compile it to low-level actions, step
the sim, and feed the next frame back — a closed perception->command loop. Two
modes exercise both Gemini surfaces:

    offline   : a fresh generate_content() call per decision (whole-frame, hi-acc)
    streaming : one persistent Live-API session, frames streamed in as it watches

    GEMINI_API_KEY=... python tools/gemini_cook.py --mode offline  --recipe onion_soup
    GEMINI_API_KEY=... python tools/gemini_cook.py --mode streaming --recipe onion_soup

Writes a GIF of what Gemini saw + did to tmp/gemini_<mode>_<recipe>.gif.
"""
import argparse
import asyncio
import io
import json
import os
import re
import sys

from google import genai
from google.genai import types

from cooksim.core.config import GameConfig
from cooksim.core.game import KitchenGame
from cooksim.layouts import load_layout
from cooksim.core.recipes import RecipeBook
from cooksim.agents.highlevel import HighLevelController
from cooksim.agents.observe import render_observation

VOCAB = """COMMANDS you may output (exactly one per turn):
- "go to <object>"  — the ONLY way to move. Targets: an ingredient source (e.g. "go to the onion"),
  a station ("go to the pot" / "the cutting board" / "the pan" / "the oven" / "the sink"),
  "go to the plates", "go to the serving pass", "go to the trash".
  Disambiguate when several exist with a qualifier: "the pot cooking rice", "the empty pot",
  "the left pot", "pot 2", or "the pot at (x,y)" using the small (x,y) labels in the image.
- "pick up"   — take the item on the tile you FACE (a source gives one item; a board gives its chopped
  item; the plate stack gives a clean plate). Only with empty hands.
- "put down"  — drop the held item into/onto the tile you FACE (into a pot/pan/oven, onto an empty board,
  onto an empty counter).
- "chop"      — dice the raw item on the board you FACE (hands must be free).
- "turn on"   — start the filled pot/pan/oven you FACE.
- "scoop"     — add the food from the station you FACE onto the plate you HOLD (a DONE pot/pan/oven, a
  chopped board, or a raw source). You must already hold a clean plate.
- "serve"     — hand the finished plate to the serving pass you FACE.
- "wait until the pot is done"  — idle and watch something finish cooking.

RULES:
- A manipulation ONLY works while you stand at and FACE the correct tile; otherwise it FAILS and you must
  "go to" that tile first. The yellow disc is the cook; the triangle is the way it faces.
- Build the dish step by step. Think about what is already done from the image before acting.

RECOVERY (read carefully):
- If your LAST command FAILED, do NOT repeat it — change your approach.
- You can hold only ONE item. To move chopped/cooked food onto a plate, HOLD a clean plate and "scoop"
  at that board/station — do NOT "pick up" the food (picking up a chopped item leaves you stuck, since a
  board only accepts RAW items). Leave chopped food on the board and come back with a plate.
- A board holds one item at a time; finish/clear it before chopping the next thing.
- To free your hands while carrying a partial plate, "go to the counter" then "put down" to STASH the
  plate on a counter; do other prep; then "go to the counter", "pick up" the plate, and continue.
Respond with STRICT JSON only: {"reason": "<one short clause>", "command": "<one command>"}"""


def recipe_text(rid):
    r = RecipeBook().get(rid)
    if r is None:
        return rid
    comps = ", ".join(f"{n} ({s})" for n, s in r.contents)
    steps = " ; ".join(r.steps)
    order = " IMPORTANT: ingredients must be added to the plate in this exact order." if r.ordered else ""
    return f"RECIPE — {r.name}: a plate holding [{comps}]. Hints: {steps}.{order}"


def parse_command(text):
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()   # strip code fences
    m = re.search(r"\{.*\}", t, re.S)
    if m:
        try:
            return str(json.loads(m.group(0)).get("command", "")).strip() or None
        except Exception:
            pass
    m = re.search(r'"command"\s*:\s*"([^"]+)"', t)
    if m:
        return m.group(1).strip()
    for ln in t.splitlines():
        if ln.strip() and not ln.strip().startswith(("`", "{", "}")):
            return ln.strip().strip('"')[:80]
    return None


def png_bytes(game):
    buf = io.BytesIO()
    render_observation(game).save(buf, format="PNG")
    return buf.getvalue()


def build_turn(game, recipe, history):
    hist = "\n".join(f"  {i+1}. {h}" for i, h in enumerate(history[-8:])) or "  (none yet)"
    return (f"{recipe}\n\nYour recent commands and outcomes:\n{hist}\n\n"
            f"Here is the current top-down view. Output the single next command as JSON.")


def step_with(game, ctrl, command, max_ticks=400):
    ctrl.issue(command)
    if ctrl.skill is None:
        return "FAILED-to-parse: " + str(ctrl.error)
    t = 0
    while ctrl.status == "running" and t < max_ticks:
        a = ctrl.step()
        acts = [5] * len(game.cooks)
        acts[0] = a
        game.step(acts)
        t += 1
    st = ctrl.status if t < max_ticks else "timeout"
    return f"{st}" + (f" ({ctrl.error})" if ctrl.error else "")


# --------------------------------------------------------------------------- offline
def _stuck(history, window=10, thresh=7):
    """True if most of the recent window failed (break a flailing loop)."""
    recent = history[-window:]
    fails = sum(1 for h in recent if "-> failed" in h or "-> FAILED" in h or "-> timeout" in h)
    return len(history) >= window and fails >= thresh


def run_offline(client, model, game, recipe, max_decisions, frames):
    ctrl = HighLevelController(game, 0)
    history = []
    start = game.stats.get("deliveries", 0)
    for k in range(max_decisions):
        if _stuck(history):
            print("[stuck] 5 consecutive failures — aborting"); break
        frames.append(render_observation(game))
        turn = build_turn(game, recipe, history)
        try:
            ck = dict(temperature=0.2, response_mime_type="application/json")
            if "flash" in model:                 # flash: no thinking, tight budget
                ck.update(max_output_tokens=400, thinking_config=types.ThinkingConfig(thinking_budget=0))
            else:                                # pro: cap thinking, reserve room for the JSON
                ck.update(max_output_tokens=4096, thinking_config=types.ThinkingConfig(thinking_budget=512))
            resp = client.models.generate_content(
                model=model,
                contents=[VOCAB, turn, types.Part.from_bytes(data=png_bytes(game), mime_type="image/png")],
                config=types.GenerateContentConfig(**ck),
            )
            cmd = parse_command(resp.text)
        except Exception as e:
            print(f"[offline] API error: {e!r}"); break
        if not cmd:
            print(f"[{k:2}] (no command parsed) -> retry")
            history.append("(no command parsed) -> failed")
            continue
        outcome = step_with(game, ctrl, cmd)
        history.append(f'"{cmd}" -> {outcome}')
        print(f"[{k:2}] {cmd:42} -> {outcome}")
        if game.stats.get("deliveries", 0) > start:
            frames.append(render_observation(game))
            return True
    return game.stats.get("deliveries", 0) > start


# --------------------------------------------------------------------------- streaming
async def run_streaming(client, model, game, recipe, max_decisions, frames):
    ctrl = HighLevelController(game, 0)
    history = []
    start = game.stats.get("deliveries", 0)
    # The Live models on the consumer API are native-audio: they emit AUDIO, so we
    # enable output transcription and ask Gemini to SPEAK one plain command, then
    # parse the transcript. (Persistent session = true streaming context.)
    sys_inst = VOCAB.replace(
        'Respond with STRICT JSON only: {"reason": "<one short clause>", "command": "<one command>"}',
        "Reply by SAYING exactly one command in plain words (for example, say: go to the onion). "
        "Say only the command, nothing else.")
    try:
        trans = types.AudioTranscriptionConfig()
    except Exception:
        trans = {}
    cfg = types.LiveConnectConfig(response_modalities=["AUDIO"], system_instruction=sys_inst,
                                  output_audio_transcription=trans)
    won = False
    async with client.aio.live.connect(model=model, config=cfg) as session:
        for k in range(max_decisions):
            if _stuck(history):
                print("[stuck] 5 consecutive failures — aborting"); break
            frames.append(render_observation(game))
            turn = build_turn(game, recipe, history)
            await session.send_client_content(turns=types.Content(role="user", parts=[
                types.Part(text=turn),
                types.Part(inline_data=types.Blob(mime_type="image/png", data=png_bytes(game))),
            ]))
            text = ""
            async for resp in session.receive():
                if resp.text:
                    text += resp.text
                sc = getattr(resp, "server_content", None)
                if sc and getattr(sc, "output_transcription", None) and sc.output_transcription.text:
                    text += sc.output_transcription.text
                if sc and getattr(sc, "turn_complete", False):
                    break
            cmd = parse_command(text)
            if not cmd:
                print(f"[{k}] no command parsed; stop"); break
            outcome = step_with(game, ctrl, cmd)
            history.append(f'"{cmd}" -> {outcome}')
            print(f"[{k:2}] {cmd:42} -> {outcome}")
            if game.stats.get("deliveries", 0) > start:
                frames.append(render_observation(game)); won = True; break
    return won


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["offline", "streaming"], default="offline")
    ap.add_argument("--layout", default="cramped_room")
    ap.add_argument("--recipe", default="onion_soup")
    ap.add_argument("--max-decisions", type=int, default=30)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("set GEMINI_API_KEY")
    client = genai.Client(api_key=key)
    model = args.model or ("gemini-2.5-flash" if args.mode == "offline" else "gemini-3.1-flash-live-preview")
    game = KitchenGame(load_layout(args.layout), config=GameConfig(horizon=100000),
                       n_players=1, seed=0, order_recipe_ids=[args.recipe])
    recipe = recipe_text(args.recipe)
    print(f"=== Gemini cook | mode={args.mode} model={model} layout={args.layout} recipe={args.recipe} ===")
    print(recipe)
    frames = []
    if args.mode == "offline":
        won = run_offline(client, model, game, recipe, args.max_decisions, frames)
    else:
        won = asyncio.run(run_streaming(client, model, game, recipe, args.max_decisions, frames))
    os.makedirs("tmp", exist_ok=True)
    out = f"tmp/gemini_{args.mode}_{args.recipe}.gif"
    if frames:
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=700, loop=0)
    print(f"\nRESULT: {'SUCCESS — delivered!' if won else 'did not complete'}  "
          f"(deliveries={game.stats.get('deliveries',0)}, failed={game.stats.get('failed_deliveries',0)}, score={int(game.score)})")
    print(f"frames -> {out}")


if __name__ == "__main__":
    main()
