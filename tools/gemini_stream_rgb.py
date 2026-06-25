#!/usr/bin/env python3
"""Streaming VLM control from PURE RENDERED RGB — no game state ever leaked.

The authoritative game runs in Python; each tick we render the REAL 3D scene
(web/headless_render.html) and screenshot it. That image is the ONLY thing the
VLM sees — no station labels, no status text, no coordinates, no "holding". Every
frame the VLM outputs ONE time-aligned action: a command, or "continue" (the
single no-op / no-interrupt). It must judge from the pixels whether its last
command has finished. Its only memory is the list of commands it has issued
(external; the offline model is stateless per call).

Tasks:
  loaded_fries : fry a potato -> plate -> add cheese -> serve (ordered).
  soup_pick    : TWO pots cooking (one onions=yellow, one tomatoes=red); the order
                 is Onion Soup. The VLM must visually pick the ONION pot — wrong
                 instance delivers tomato soup and fails. (instance grounding)

    GEMINI_API_KEY=... python tools/gemini_stream_rgb.py --task soup_pick --model gemini-2.5-pro
"""
import argparse, asyncio, base64, io, json, os, re, subprocess, sys, time

from PIL import Image, ImageDraw, ImageFont
from playwright.async_api import async_playwright
from google import genai
from google.genai import types

from cooksim.core.config import GameConfig
from cooksim.core.game import KitchenGame
from cooksim.core.enums import Action, PrepState
from cooksim.core.items import Ingredient, Plate
from cooksim.core.stations import CookStation
from cooksim.layouts import load_layout
from cooksim.core.recipes import RecipeBook
from cooksim.agents.highlevel import HighLevelController, _validate, parse as hl_parse

_MANIP = {"pickup", "putdown", "chop", "start", "scoop", "serve", "wash", "trash"}

GPU = ["--no-sandbox", "--use-gl=angle", "--use-angle=vulkan", "--enable-features=Vulkan",
       "--ignore-gpu-blocklist", "--enable-gpu", "--autoplay-policy=no-user-gesture-required"]
PAGE = "http://127.0.0.1:8085/static/headless_render.html"

SYS = """You are a STREAMING controller for one cook in a 3D top-down cooking game. You receive ONE
video frame at a time and output exactly ONE action for THIS frame.

You see ONLY the image — read everything from it: where the cook is (the blue chef with a number),
which way it faces (it leans/points toward the tile it will use), what it holds (in its hands, or a
white plate with food on it), what is inside each pot/pan/board, and whether something is still cooking
vs done (browned food, steam, a bright ring).

OUTPUT strict JSON: {"action": "<command|continue>", "reason": "<short, state what you SEE>"}
- "continue" is the no-op / NO-INTERRUPT: it keeps the current command running. Use it while the cook is
  still walking to its target, or while food is still cooking. YOU decide from the frame whether the last
  command has finished yet.
- Otherwise output one command.

PACING — IMPORTANT: the cook moves ONE tile per frame and takes SEVERAL frames to reach a target. After you
issue a "go to", reply "continue" on EVERY following frame until you clearly SEE the cook standing directly
next to (and facing) that target. Only then issue the next command. If you issue a new "go to" too early,
the cook abandons the trip and never arrives — so be patient and watch it walk.

INSTANCE / SPATIAL GROUNDING — IMPORTANT: when several of an object exist (e.g. two pots) they may hold
DIFFERENT food. LOOK at the frame to see which one has what you need (e.g. yellow onions vs red tomatoes)
and name THAT specific one: "the pot with the onions", "the left pot", "the right pot", or "pot 2".
Choosing the wrong instance ruins the dish.

COMMANDS:
- "go to <object>" — the ONLY move (+ the instance qualifiers above). Objects: an ingredient source, a
  pot/pan/oven/cutting board/sink, "the plates", "the serving pass", "the trash".
- "pick up"  — take the item on the tile you FACE (empty hands).
- "put down" — drop the held item into/onto the tile you FACE.
- "chop"     — dice the raw item on the board you FACE (free hands).
- "turn on"  — start the filled pot/pan/oven you FACE.
- "scoop"    — add the food from the station you FACE onto the plate you HOLD (hold a clean plate first).
- "serve"    — hand the finished plate to the serving pass you FACE.
- "wait until the pot is done" — let cooking finish (or just keep saying continue).
RULES: a manipulation only works while standing at and FACING the right tile, else it FAILS — "go to"
first. To plate cooked/chopped food, HOLD a plate and "scoop" (do NOT "pick up" it). Build step by step."""

NOOP = {"continue", "none", "no-op", "noop", "no interrupt", "wait", "watch", "keep going", "stay", "hold"}


def make_game(task):
    if task == "soup_pick":
        g = KitchenGame(load_layout("bistro"), config=GameConfig(horizon=10**9, pot_cook_time=34),
                        n_players=1, seed=0, order_recipe_ids=["onion_soup"])
        pots = sorted(p for p, s in g.stations.items() if isinstance(s, CookStation))
        for ing, pos in (("tomato", pots[0]), ("onion", pots[1])):     # left red, right yellow
            g.stations[pos].contents = [Ingredient(ing) for _ in range(3)]
            g.stations[pos].status = "cooking"; g.stations[pos].cooking = True; g.stations[pos].timer = 1
        recipe = ("RECIPE — Onion Soup: a plate with 3 cooked ONIONS. Two pots are cooking; only ONE has "
                  "onions (yellow) — the other has tomatoes (red). Plate the ONION soup and serve.")
        return g, recipe
    g = KitchenGame(load_layout("diner"), config=GameConfig(horizon=10**9),
                    n_players=1, seed=0, order_recipe_ids=["loaded_fries"])
    return g, ("RECIPE — Loaded Fries: a plate with [potato (cooked), cheese (raw)], potato added first. "
               "Fry a potato in a pan, plate it, add cheese, serve.")


def parse_action(text):
    if not text:
        return None
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", t, re.S)
    if m:
        try:
            return str(json.loads(m.group(0)).get("action", "")).strip() or None
        except Exception:
            pass
    m = re.search(r'"action"\s*:\s*"([^"]+)"', t)
    return m.group(1).strip() if m else None


def ask(client, model, recipe, past, img_png):
    mem = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(past[-10:])) or "  (none yet)"
    turn = (f"{recipe}\n\nThe commands you have issued so far (your only memory):\n{mem}\n\n"
            f"This is the current video frame. Output one action JSON for this frame.")
    ck = dict(temperature=0.1, response_mime_type="application/json")
    if "flash" in model:
        ck.update(max_output_tokens=350, thinking_config=types.ThinkingConfig(thinking_budget=0))
    else:
        ck.update(max_output_tokens=4096, thinking_config=types.ThinkingConfig(thinking_budget=512))
    last = None
    for k in range(4):
        try:
            r = client.models.generate_content(
                model=model, contents=[SYS, turn, types.Part.from_bytes(data=img_png, mime_type="image/png")],
                config=types.GenerateContentConfig(**ck))
            return parse_action(r.text)
        except Exception as e:
            last = e
            if any(s in str(e) for s in ("503", "UNAVAILABLE", "500", "INTERNAL", "429", "RESOURCE_EXHAUSTED")):
                time.sleep(2 + 2 * k); continue
            raise
    raise last


# ---- composite (human-facing; the model never sees this panel) ----
def _font(sz, bold=True):
    try:
        return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf", sz)
    except Exception:
        return ImageFont.load_default()


def _wrap(d, text, font, w):
    out, line = [], ""
    for word in str(text).split():
        if d.textlength(line + " " + word, font=font) > w and line:
            out.append(line); line = word
        else:
            line = (line + " " + word).strip()
    if line:
        out.append(line)
    return out


def composite(rgb, info, hud_w=420):
    H = rgb.height
    img = Image.new("RGB", (rgb.width + hud_w, H), (16, 14, 22))
    img.paste(rgb, (0, 0))
    d = ImageDraw.Draw(img)
    x = rgb.width + 16
    fb, fm, fs = _font(18), _font(14), _font(12, False)
    d.text((x, 12), f"STREAMING (pure RGB)  ·  frame {info['frame']}", fill=(150, 200, 255), font=fm)
    d.text((x, 31), f"model: {info['model']}   ·   model sees only the left image →", fill=(140, 135, 155), font=fs)
    y = 60
    if info["is_cmd"]:
        d.text((x, y), "VLM ▸ COMMAND:", fill=(120, 230, 150), font=fm); y += 22
        for ln in _wrap(d, info["vlm"], fb, hud_w - 28):
            d.text((x, y), ln, fill=(150, 255, 180), font=fb); y += 23
    else:
        d.text((x, y), "VLM ▸ continue (no-interrupt)", fill=(150, 150, 165), font=fm); y += 26
    if info.get("reason"):
        for ln in _wrap(d, "“" + info["reason"] + "”", fs, hud_w - 28)[:3]:
            d.text((x, y), ln, fill=(150, 175, 205), font=fs); y += 16
    y += 6
    d.text((x, y), f"agent button: {info['low']}", fill=(255, 210, 140), font=fm); y += 24
    d.line([(x, y), (x + hud_w - 28, y)], fill=(60, 54, 74)); y += 10
    d.text((x, y), "commands issued (memory):", fill=(150, 145, 165), font=fs); y += 18
    for f, c in info["log"][-9:]:
        d.text((x, y), f"f{f}: {c[:42]}", fill=(180, 200, 230), font=fs); y += 16
    if info["event"]:
        col = (255, 215, 120) if "DELIVER" in info["event"] else (255, 150, 130) if "✗" in info["event"] else (150, 230, 170)
        d.rounded_rectangle([x - 4, H - 54, x + hud_w - 24, H - 24], radius=6, fill=(30, 26, 38))
        d.text((x, H - 47), info["event"], fill=col, font=fm)
    d.text((x, H - 20), f"task: {info['task']}", fill=(170, 165, 185), font=fs)
    return img


def encode(frames, path, fps=4):
    os.makedirs("tmp/_rsf", exist_ok=True)
    for i, fr in enumerate(frames):
        fr.save(f"tmp/_rsf/{i:05d}.png")
    run = ["/fsx/home/chenhao.zheng/.local/bin/micromamba", "run", "-n", "cooksim", "ffmpeg", "-y",
           "-loglevel", "error", "-framerate", str(fps), "-i", "tmp/_rsf/%05d.png", "-movflags", "+faststart",
           "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "26", path]
    subprocess.run(run, check=True)
    for f in os.listdir("tmp/_rsf"):
        os.remove("tmp/_rsf/" + f)


def hold_str(h):
    if h is None:
        return "empty"
    if isinstance(h, Plate):
        return "dirty plate" if h.dirty else ("plate[" + ",".join(i.name[:4] for i in h.contents) + "]" if h.contents else "plate")
    return f"{h.name}/{h.state.value}"


async def render_frame(page, state):
    url = await page.evaluate("(s)=>window.__renderFrame(s)", json.dumps(state))
    return base64.b64decode(url.split(",", 1)[1])


async def run(args):
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    game, recipe = make_game(args.task)
    ctrl = HighLevelController(game, 0)
    print(f"=== STREAMING RGB | model={args.model} task={args.task} ===\n{recipe}")
    frames, past, log = [], [], []
    fails = idle = 0
    start = game.stats.get("deliveries", 0)
    won = False
    async with async_playwright() as p:
        b = await p.chromium.launch(args=GPU)
        page = await b.new_page(viewport={"width": 800, "height": 580})
        await page.goto(PAGE, wait_until="load")
        await page.wait_for_function("window.__ready===true", timeout=20000)
        await page.wait_for_timeout(600)
        for fi in range(args.max_frames):
            png = await render_frame(page, game.render_state())
            rgb = Image.open(io.BytesIO(png)).convert("RGB")
            prev = ctrl.skill.label if ctrl.skill else None
            # EVERY frame: the VLM decides command vs continue from the image alone
            try:
                resp = await asyncio.to_thread(ask, client, args.model, recipe, past, png)
            except Exception as e:
                print(f"[f{fi}] API error: {e!r}"); break
            is_cmd = False; vlm = "continue"; reject = None; too_early = None
            if resp and resp.strip().lower() not in NOOP:
                vlm = resp.strip(); is_cmd = True; past.append(vlm); log.append((fi, vlm))
                it = hl_parse(vlm)
                verb = it.verb if it else None
                if verb in _MANIP:
                    okv, why = _validate(verb, game, game.cooks[0])
                    if okv:
                        ctrl.issue(vlm)          # in position -> execute the manipulation
                    else:
                        too_early = why          # not in position: harmless no-op; keep walking
                else:
                    ok = ctrl.issue(vlm)         # go to / wait / low-level
                    if not ok:
                        reject = ctrl.error
            idle = idle + 1 if (ctrl.skill is None and not is_cmd) else 0
            low = ctrl.step() if ctrl.skill is not None else int(Action.STAY)
            acts = [int(Action.STAY)] * len(game.cooks); acts[0] = low
            game.step(acts)
            event = ""
            if reject:
                event = f"✗ rejected: {reject}"; fails += 1
            elif too_early and ctrl.skill is None:   # tried to manipulate with nothing underway
                event = f"✗ not ready: {too_early}"; fails += 1
            elif prev and ctrl.skill is None:
                event = f"✓ {prev} done" if ctrl.status == "done" else f"✗ {prev} failed: {ctrl.error}"
                fails += (ctrl.status == "failed")
            if game.stats.get("deliveries", 0) > start:
                event = f"✓✓ DELIVERED  +{int(game.score)}"
            info = dict(frame=fi, model=args.model, vlm=vlm, is_cmd=is_cmd, reason="",
                        low=Action(low).name, log=log, event=event, task=args.task)
            frames.append(composite(rgb, info))
            print(f"[f{fi:3}] {('CMD ' + vlm) if is_cmd else 'continue · no-interrupt':38} btn={Action(low).name:8}"
                  f" hold={hold_str(game.cooks[0].holding):16} {('| ' + event) if event else ''}")
            if game.stats.get("deliveries", 0) > start:
                won = True
                png2 = await render_frame(page, game.render_state())
                frames.append(composite(Image.open(io.BytesIO(png2)).convert("RGB"),
                              {**info, "frame": fi + 1, "vlm": "— task complete —", "is_cmd": True, "low": "STAY"}))
                break
            if fails >= 7 and ctrl.skill is None:
                print("[stuck] too many failures"); break
            if idle >= 8:
                print("[idle-stall]"); break
        await b.close()
    out = f"tmp/rgb_{args.model.replace('.','').replace('-','')}_{args.task}.mp4"
    encode(frames, out, fps=args.fps)
    print(f"\nRESULT: {'SUCCESS' if won else 'did not complete'}  "
          f"(deliveries={game.stats.get('deliveries',0)}, failed_serves={game.stats.get('failed_deliveries',0)}, "
          f"frames={len(frames)}, score={int(game.score)})\nvideo -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["loaded_fries", "soup_pick"], default="soup_pick")
    ap.add_argument("--model", default="gemini-2.5-pro")
    ap.add_argument("--max-frames", type=int, default=150)
    ap.add_argument("--fps", type=int, default=4)
    args = ap.parse_args()
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("set GEMINI_API_KEY")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
