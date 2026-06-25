#!/usr/bin/env python3
"""Streaming VLM control via the SLIDING-WINDOW inference-time policy.

This implements the strong offline->streaming adaptation recommended by Kanchana
(Sec 4.1 of arXiv:2604.07634): about once per second, send the model the LAST 5
rendered frames (oldest->newest, for motion) plus its LAST 5 responses, and ask it
to reply with an EMPTY action (no-op) or one command. Input is PURE rendered RGB
(the real 3D scene, no labels/state). The model judges from the frames whether its
last command finished. Latest models: gemini-3.5-flash / gemini-3.1-pro-preview.

    GEMINI_API_KEY=... python tools/gemini_stream_sw.py --task soup_pick --model gemini-3.1-pro-preview
"""
import argparse, asyncio, io, json, os, re, sys, time

from PIL import Image
from playwright.async_api import async_playwright
from google import genai
from google.genai import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cooksim.core.enums import Action
from cooksim.agents.highlevel import HighLevelController, _validate, parse as hl_parse
from gemini_stream_rgb import (make_game, GPU, PAGE, render_frame, composite, encode, hold_str, _MANIP)

WINDOW = 5          # last N frames sent each decision (motion context)
HIST = 5            # last N model responses sent each decision

SYS = """You are a STREAMING controller for one cook in a 3D top-down cooking game. About once per second you
receive the LAST FEW video frames (oldest -> newest) so you can see MOTION, plus the recent actions you issued.
You output ONE action for now.

Read everything from the frames: where the cook is (blue chef), which way it faces, what it holds (a white plate,
with food on it once filled), what is inside each pot/pan/board, and whether food is still cooking vs done.
Use the sequence of frames to judge whether your last command has FINISHED (e.g. the cook has stopped moving and
is now standing next to the target).

OUTPUT strict JSON: {"action": "<command or empty string>", "reason": "<short, state what you SEE>"}
- action = "" (empty)  -> NO-OP: keep doing the current thing. Use this while the cook is still walking to its
  target, or while food is still cooking.
- action = a command   -> change what the cook is doing.

INSTANCE / SPATIAL GROUNDING — IMPORTANT: when several of an object exist (e.g. two pots) they may hold DIFFERENT
food. LOOK at the frames to see which one has what you need (yellow onions vs red tomatoes) and name THAT one:
"the pot with the onions", "the left pot", "the right pot", or "pot 2". Wrong instance ruins the dish.

COMMANDS: "go to <object>" (the only move; + the instance qualifiers above; objects: an ingredient source, a
pot/pan/oven/cutting board/sink, "the plates", "the serving pass", "the trash") · "pick up" · "put down" ·
"chop" · "turn on" · "scoop" (hold a plate, add the faced station's food onto it) · "serve".
RULES: a manipulation only works while standing at and FACING the right tile, else it does nothing — go to it
first. To plate cooked/chopped food, HOLD a plate and "scoop" (don't "pick up"). The cook moves one tile per
frame and takes several frames to arrive — reply "" (empty) until the frames show it has clearly arrived."""

NOOP = {"", "continue", "none", "no-op", "noop", "wait", "watch", "stay", "keep going", "null", "(empty)"}


def parse_action(text):
    if not text:
        return ""
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", t, re.S)
    if m:
        try:
            return str(json.loads(m.group(0)).get("action", "")).strip()
        except Exception:
            pass
    m = re.search(r'"action"\s*:\s*"([^"]*)"', t)
    return m.group(1).strip() if m else ""


def ask(client, model, recipe, frames, responses):
    hist = "\n".join(f"  {i+1}. {r}" for i, r in enumerate(responses[-HIST:])) or "  (none yet)"
    txt = (f"{recipe}\n\nYour last {HIST} actions (most recent last):\n{hist}\n\n"
           f"Below are your last {len(frames)} video frames, oldest first. Output one action JSON for NOW "
           f'(use "" to keep doing the current thing).')
    parts = [SYS, txt] + [types.Part.from_bytes(data=f, mime_type="image/png") for f in frames]
    ck = dict(temperature=0.1, response_mime_type="application/json")
    if "flash" in model:
        ck.update(max_output_tokens=400, thinking_config=types.ThinkingConfig(thinking_budget=0))
    else:
        ck.update(max_output_tokens=4096, thinking_config=types.ThinkingConfig(thinking_budget=640))
    last = None
    for k in range(4):
        try:
            r = client.models.generate_content(model=model, contents=parts,
                                                config=types.GenerateContentConfig(**ck))
            return parse_action(r.text)
        except Exception as e:
            last = e
            if any(s in str(e) for s in ("503", "UNAVAILABLE", "500", "INTERNAL", "429", "RESOURCE")):
                time.sleep(2 + 2 * k); continue
            raise
    raise last


async def run(args):
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    game, recipe = make_game(args.task)
    ctrl = HighLevelController(game, 0)
    print(f"=== SLIDING-WINDOW RGB | model={args.model} task={args.task} poll={args.poll}s win={WINDOW} ===\n{recipe}")
    comp_frames, fbuf, responses, log = [], [], [], []
    fails = 0
    start = game.stats.get("deliveries", 0)
    won = False
    async with async_playwright() as p:
        b = await p.chromium.launch(args=GPU)
        page = await b.new_page(viewport={"width": 800, "height": 580})
        await page.goto(PAGE, wait_until="load")
        await page.wait_for_function("window.__ready===true", timeout=20000)
        await page.wait_for_timeout(600)
        decided = "(none)"
        for fi in range(args.max_frames):
            png = await render_frame(page, game.render_state())
            fbuf.append(png)
            rgb = Image.open(io.BytesIO(png)).convert("RGB")
            prev = ctrl.skill.label if ctrl.skill else None
            is_cmd = False; vlm = "(running)"; event = ""
            if fi % args.poll == 0:                       # ~once per second: a decision
                window = fbuf[-WINDOW:]
                try:
                    act = await asyncio.to_thread(ask, client, args.model, recipe, window, responses)
                except Exception as e:
                    print(f"[f{fi}] API error: {e!r}"); break
                if act and act.strip().lower() not in NOOP:
                    vlm = act.strip(); is_cmd = True
                    it = hl_parse(vlm); verb = it.verb if it else None
                    if verb in _MANIP:
                        okv, why = _validate(verb, game, game.cooks[0])
                        if okv:
                            ctrl.issue(vlm)
                        else:
                            event = f"(too early: {why})" if ctrl.skill is None else ""
                    else:
                        if not ctrl.issue(vlm):
                            event = f"✗ rejected: {ctrl.error}"; fails += 1
                    responses.append(vlm); log.append((fi, vlm)); decided = vlm
                else:
                    responses.append("(empty / no-op)"); vlm = "empty · no-op"
            low = ctrl.step() if ctrl.skill is not None else int(Action.STAY)
            acts = [int(Action.STAY)] * len(game.cooks); acts[0] = low
            game.step(acts)
            if not event and prev and ctrl.skill is None:
                event = f"✓ {prev} done" if ctrl.status == "done" else f"✗ {prev} failed: {ctrl.error}"
                fails += (ctrl.status == "failed")
            if game.stats.get("deliveries", 0) > start:
                event = f"✓✓ DELIVERED  +{int(game.score)}"
            info = dict(frame=fi, model=args.model, vlm=vlm, is_cmd=is_cmd, reason="",
                        low=Action(low).name, log=log, event=event, task=args.task)
            comp_frames.append(composite(rgb, info))
            print(f"[f{fi:3}] {(('CMD ' + vlm) if is_cmd else vlm):32} btn={Action(low).name:8}"
                  f" hold={hold_str(game.cooks[0].holding):16} {('| ' + event) if event else ''}")
            if game.stats.get("deliveries", 0) > start:
                won = True
                png2 = await render_frame(page, game.render_state())
                comp_frames.append(composite(Image.open(io.BytesIO(png2)).convert("RGB"),
                                   {**info, "frame": fi + 1, "vlm": "— task complete —", "is_cmd": True, "low": "STAY"}))
                break
            if fails >= 8 and ctrl.skill is None:
                print("[stuck]"); break
        await b.close()
    out = f"tmp/sw_{args.model.replace('.','').replace('-','')}_{args.task}.mp4"
    encode(comp_frames, out, fps=args.fps)
    print(f"\nRESULT: {'SUCCESS' if won else 'did not complete'}  "
          f"(deliveries={game.stats.get('deliveries',0)}, frames={len(comp_frames)}, score={int(game.score)})\nvideo -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["loaded_fries", "soup_pick"], default="soup_pick")
    ap.add_argument("--model", default="gemini-3.1-pro-preview")
    ap.add_argument("--poll", type=int, default=3, help="ticks between decisions (~1s)")
    ap.add_argument("--max-frames", type=int, default=140)
    ap.add_argument("--fps", type=int, default=4)
    args = ap.parse_args()
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("set GEMINI_API_KEY")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
