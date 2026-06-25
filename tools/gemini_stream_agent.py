#!/usr/bin/env python3
"""Streaming-VLM control of the CookSim cook (true frame-by-frame).

This is the inference-time adaptation of an OFFLINE Gemini model into a STREAMING
controller (no audio): the game runs continuously; at each frame the VLM sees the
current frame and must emit a TIME-ALIGNED action that is either

    {"action": "continue"}          # no-interrupt: let the current skill keep running
    {"action": "<a manipulation command>"}   # interrupt & switch to a new command

A high-level command (e.g. "go to the pan") then auto-drives several low-level
ticks; the VLM mostly says "continue" while it runs and speaks only at transitions.

It records every frame (observation, VLM output, the agent's low-level button, the
running skill, and outcome events) and renders a final TIME-ALIGNED composite video:
   [ visual observation | VLM action | agent button | skill | outcome ].

    GEMINI_API_KEY=... python tools/gemini_stream_agent.py \
        --layout diner --recipe loaded_fries --model gemini-2.5-flash --poll 1
"""
import argparse
import io
import json
import os
import re
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont
from google import genai
from google.genai import types

from cooksim.core.config import GameConfig
from cooksim.core.game import KitchenGame
from cooksim.core.enums import Action
from cooksim.core.items import Ingredient, Plate
from cooksim.layouts import load_layout
from cooksim.core.recipes import RecipeBook
from cooksim.agents.highlevel import HighLevelController
from cooksim.agents.observe import render_observation

SYS = """You are a STREAMING controller for one cook in a top-down cooking game. You see ONE frame at a
time and must output ONE time-aligned action per frame.

OUTPUT (strict JSON): {"action": "<command|continue>", "reason": "<short>"}
- Use "continue" when the cook is already doing the right thing or nothing needs to change — this is the
  NO-INTERRUPT / no-op choice; the current command keeps running. Prefer "continue" while a command is in
  progress (e.g. the cook is still walking, or the pan is still cooking). Only issue a new command at a
  transition (a step finished, or something is now ready).
- You are queried at DECISION POINTS (the previous command just finished, or something is cooking).
  Navigation and manipulations finish on their own — do not babysit them. If the cook is IDLE (no command
  running), you MUST issue the next command — never say "continue" while idle.

COMMANDS (one per turn when you do act):
- "go to <object>": the ONLY move. Targets: an ingredient source ("go to the potato"), a station
  ("go to the pan"/"the pot"/"the cutting board"/"the oven"/"the sink"), "go to the plates",
  "go to the serving pass", "go to the trash". Disambiguate with: "the pan with the potato", "the done
  pot", "the left pot", "pan 2", or "the counter at (x,y)" using the (x,y) labels in the image.
- "pick up"  : take the item on the tile you FACE (empty hands).
- "put down" : drop the held item into/onto the tile you FACE (a pot/pan/oven, empty board, empty counter).
- "chop"     : dice the raw item on the board you FACE (free hands).
- "turn on"  : start the filled pot/pan/oven you FACE.
- "scoop"    : add the food from the station you FACE onto the plate you HOLD (a DONE station, a chopped
  board, or a raw source). Hold a clean plate first.
- "serve"    : hand the finished plate to the serving pass you FACE.
- "wait until the <station> is done": let cooking finish (you may also just keep saying continue).
RULES: a manipulation only works while standing at and FACING the right tile, else it FAILS — "go to" first.
To move chopped/cooked food onto a plate, HOLD a plate and "scoop" (do NOT "pick up" the food). To free your
hands while carrying a partial plate, "go to the counter" then "put down" to stash it, then pick it up later.
The yellow disc is the cook; the triangle shows its facing. Build the recipe step by step."""

NOOP = {"continue", "none", "no-op", "noop", "no interrupt", "no-interrupt", "wait", "watch", "keep going", "stay"}


def recipe_text(rid):
    r = RecipeBook().get(rid)
    if not r:
        return rid
    comps = ", ".join(f"{n} ({s})" for n, s in r.contents)
    o = "  Ingredients MUST be added to the plate in THIS order." if r.ordered else ""
    return f"RECIPE — {r.name}: plate = [{comps}]. Steps: {' ; '.join(r.steps)}.{o}"


def png(game):
    b = io.BytesIO(); render_observation(game).save(b, format="PNG"); return b.getvalue()


def parse_action(text):
    if not text:
        return None
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", t, re.S)
    if m:
        try:
            return str(json.loads(m.group(0)).get("action", "")).strip()
        except Exception:
            pass
    m = re.search(r'"action"\s*:\s*"([^"]+)"', t)
    return m.group(1).strip() if m else None


def hold_str(h):
    if h is None:
        return "empty"
    if isinstance(h, Plate):
        return "dirty plate" if h.dirty else ("plate[" + ",".join(i.name[:4] for i in h.contents) + "]" if h.contents else "plate")
    return f"{h.name}/{h.state.value}"


def ask(client, model, recipe, ctrl, history, frame_png):
    skill = ctrl.skill.label if ctrl.skill else "idle"
    status = ctrl.status
    hist = "\n".join(f"  f{f}: {c}" for f, c in history[-8:]) or "  (none)"
    turn = (f"{recipe}\nCurrently executing: {skill} [{status}].  Holding: {hold_str(ctrl.cook.holding)}.\n"
            f"Recent commands you issued:\n{hist}\n\nCurrent frame below. Output the time-aligned action JSON.")
    ck = dict(temperature=0.1, response_mime_type="application/json")
    if "flash" in model:
        ck.update(max_output_tokens=350, thinking_config=types.ThinkingConfig(thinking_budget=0))
    else:
        ck.update(max_output_tokens=4096, thinking_config=types.ThinkingConfig(thinking_budget=512))
    import time
    last = None
    for attempt in range(4):                     # retry transient 5xx/network blips
        try:
            resp = client.models.generate_content(
                model=model, contents=[SYS, turn, types.Part.from_bytes(data=frame_png, mime_type="image/png")],
                config=types.GenerateContentConfig(**ck))
            return parse_action(resp.text)
        except Exception as e:
            last = e
            if any(s in str(e) for s in ("503", "UNAVAILABLE", "500", "INTERNAL", "429", "RESOURCE_EXHAUSTED")):
                time.sleep(2 + 2 * attempt); continue
            raise
    raise last


# --------------------------------------------------------------------------- composite video
def _font(sz, bold=True):
    for p in (f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf",):
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def _wrap(d, text, font, w):
    out, line = [], ""
    for word in text.split():
        if d.textlength(line + " " + word, font=font) > w and line:
            out.append(line); line = word
        else:
            line = (line + " " + word).strip()
    if line:
        out.append(line)
    return out


def composite(obs, info, hud_w=400):
    H = obs.height
    img = Image.new("RGB", (obs.width + hud_w, H), (18, 16, 24))
    img.paste(obs, (0, 0))
    d = ImageDraw.Draw(img)
    x = obs.width + 16
    fb, fm, fs = _font(17), _font(14), _font(12, False)
    d.text((x, 14), f"STREAMING  ·  frame {info['frame']}  t={info['tick']}", fill=(150, 200, 255), font=fm)
    d.text((x, 34), f"model: {info['model']}", fill=(150, 145, 165), font=fs)
    # VLM action
    y = 64
    if info["is_cmd"]:
        d.text((x, y), "VLM ▸ COMMAND:", fill=(120, 230, 150), font=fm)
        for i, ln in enumerate(_wrap(d, info["vlm"], fb, hud_w - 32)):
            d.text((x, y + 22 + i * 22), ln, fill=(150, 255, 180), font=fb)
        y += 22 + 22 * max(1, len(_wrap(d, info["vlm"], fb, hud_w - 32))) + 8
    else:
        d.text((x, y), "VLM ▸ continue (watching)", fill=(150, 150, 165), font=fm)
        y += 34
    # agent + skill
    d.text((x, y), f"agent button: {info['low']}", fill=(255, 210, 140), font=fm); y += 24
    sk = _wrap(d, f"doing: {info['skill']} [{info['sstatus']}]", fm, hud_w - 32)
    for ln in sk:
        d.text((x, y), ln, fill=(200, 195, 215), font=fm); y += 22
    d.text((x, y), f"holding: {info['hold']}", fill=(200, 195, 215), font=fs); y += 22
    # divider + command log
    d.line([(x, y + 4), (x + hud_w - 32, y + 4)], fill=(60, 54, 74)); y += 14
    d.text((x, y), "commands issued:", fill=(150, 145, 165), font=fs); y += 18
    for f, c in info["log"][-9:]:
        d.text((x, y), f"f{f}: {c[:40]}", fill=(180, 200, 230), font=fs); y += 16
    # outcome / events at bottom
    if info["event"]:
        col = (255, 215, 120) if "DELIVER" in info["event"] else (255, 150, 130) if "✗" in info["event"] else (150, 230, 170)
        d.rounded_rectangle([x - 4, H - 58, x + hud_w - 28, H - 26], radius=6, fill=(30, 26, 38))
        d.text((x, H - 50), info["event"], fill=col, font=fm)
    d.text((x, H - 22), f"score: {info['score']}   target: {info['target']}", fill=(190, 185, 205), font=fs)
    return img


def encode(frames, path, fps=4):
    os.makedirs("tmp/_sf", exist_ok=True)
    for i, fr in enumerate(frames):
        fr.save(f"tmp/_sf/{i:05d}.png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", "tmp/_sf/%05d.png",
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p", "-c:v", "libx264", path], check=True)
    for f in os.listdir("tmp/_sf"):
        os.remove("tmp/_sf/" + f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="diner")
    ap.add_argument("--recipe", default="loaded_fries")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--poll", type=int, default=1, help="query the VLM every N frames; auto-continue between")
    ap.add_argument("--max-frames", type=int, default=220)
    args = ap.parse_args()
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("set GEMINI_API_KEY")
    client = genai.Client(api_key=key)
    game = KitchenGame(load_layout(args.layout), config=GameConfig(horizon=10 ** 9),
                       n_players=1, seed=0, order_recipe_ids=[args.recipe])
    ctrl = HighLevelController(game, 0)
    recipe = recipe_text(args.recipe)
    print(f"=== STREAMING agent | model={args.model} layout={args.layout} recipe={args.recipe} poll={args.poll} ===")
    print(recipe)
    frames, log, history = [], [], []
    fails = idle_cont = 0
    start_deliv = game.stats.get("deliveries", 0)
    won = False
    for fi in range(args.max_frames):
        obs = render_observation(game)
        sk = ctrl.skill
        sktype = type(sk).__name__ if sk else None
        prev_label = sk.label if sk else None
        # Navigation & manipulation are ATOMIC: while one runs we auto-"continue"
        # (no interrupt), so a half-finished 'go to' is never aborted. We query the
        # VLM only at decision points: when idle, or periodically during a long wait.
        ask_now = (sk is None) or (sktype == "Wait" and fi % args.poll == 0)
        is_cmd = False
        vlm = "continue"
        reject = None
        if ask_now:
            try:
                a = ask(client, args.model, recipe, ctrl, history, png(game))
            except Exception as e:
                print(f"[f{fi}] API error: {e!r}"); break
            if a and a.strip().lower() not in NOOP:
                vlm = a.strip()
                ok = ctrl.issue(vlm)
                is_cmd = True
                log.append((fi, vlm))
                history.append((fi, f'"{vlm}" -> {"ok" if ok else "REJECTED:" + str(ctrl.error)}'))
                if not ok:
                    reject = ctrl.error
        idle_cont = idle_cont + 1 if (sk is None and not is_cmd) else 0
        # step one tick
        low = ctrl.step() if ctrl.skill is not None else int(Action.STAY)
        acts = [int(Action.STAY)] * len(game.cooks); acts[0] = low
        game.step(acts)
        # events from skill-completion transitions
        event = ""
        if reject:
            event = f"✗ rejected: {reject}"; fails += 1
        elif prev_label and ctrl.skill is None:
            if ctrl.status == "done":
                event = f"✓ {prev_label} done"
            elif ctrl.status == "failed":
                event = f"✗ {prev_label} failed: {ctrl.error}"; fails += 1
        if game.stats.get("deliveries", 0) > start_deliv:
            event = f"✓✓ DELIVERED  +{int(game.score)}"
        vlm_disp = vlm if is_cmd else ("continue · watching" if ask_now else "continue · auto")
        info = dict(frame=fi, tick=game.tick, model=args.model, vlm=vlm_disp, is_cmd=is_cmd,
                    low=Action(low).name, skill=(ctrl.skill.label if ctrl.skill else (prev_label or "idle")),
                    sstatus=ctrl.status, hold=hold_str(game.cooks[0].holding), log=log, event=event,
                    score=int(game.score), target=args.recipe)
        frames.append(composite(obs, info))
        tag = f"CMD {vlm}" if is_cmd else vlm_disp
        print(f"[f{fi:3} t{game.tick:3}] {tag:34} btn={Action(low).name:8} {('| ' + event) if event else ''}")
        if game.stats.get("deliveries", 0) > start_deliv:
            frames.append(composite(render_observation(game), {**info, "frame": fi + 1, "tick": game.tick,
                          "vlm": "— task complete —", "is_cmd": True, "low": "STAY",
                          "event": f"✓✓ DELIVERED  +{int(game.score)}"}))
            won = True; break
        if fails >= 6 and ctrl.skill is None:
            print("[stuck] too many failures — aborting"); break
        if idle_cont >= 8:
            print("[idle-stall] VLM kept saying continue while idle — aborting"); break
    os.makedirs("tmp", exist_ok=True)
    out = f"tmp/stream_{args.model.replace('.','').replace('-','')}_{args.recipe}.mp4"
    encode(frames, out, fps=4)
    print(f"\nRESULT: {'SUCCESS — delivered!' if won else 'did not complete'}  "
          f"(deliveries={game.stats.get('deliveries',0)}, frames={len(frames)}, score={int(game.score)})")
    print(f"time-aligned video -> {out}")


if __name__ == "__main__":
    main()
