#!/usr/bin/env python3
"""Gemini LIVE (real streaming, stateful session) control of the CookSim cook.

Uses the consumer Gemini API key. The Live models are audio-native, so we read
text via output_audio_transcription. Gemini Live is NOT proactive, so — as the
user suggested — we PROMPT IT EVERY FRAME: all the context goes in the system
instruction up front, then each frame we stream the rendered RGB frame and send a
light "next action?" ping, and read back its spoken (transcribed) command. The
Live SESSION keeps its own internal memory of the frames + responses (the true
streaming context), so we don't resend history.

    GEMINI_API_KEY=... python tools/gemini_live_agent.py --task soup_pick \
        --model gemini-2.5-flash-native-audio-latest
"""
import argparse, asyncio, io, os, re, sys

from PIL import Image
from playwright.async_api import async_playwright
from google import genai
from google.genai import types

from cooksim.core.enums import Action
from cooksim.agents.highlevel import HighLevelController, _validate, parse as hl_parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gemini_stream_rgb import make_game, GPU, PAGE, render_frame, composite, encode, hold_str, _MANIP

SYS = """You are a STREAMING controller for one cook in a 3D top-down cooking game. You watch a live video and
I will ask you, frame by frame, for the next action. Read everything from the picture: where the cook is (the
blue chef), which way it faces, what it holds (a white plate, with food on it once filled), what is inside each
pot/pan, and whether food is still cooking vs done.

EACH TIME I ask, reply by SAYING exactly ONE short thing and nothing else:
- say "continue" = NO-OP / no interrupt: keep doing the current thing (use this while the cook is still walking,
  or while food is still cooking; you judge from the video whether the last command has finished).
- OR say one command.

INSTANCE / SPATIAL GROUNDING — IMPORTANT: when several of an object exist (two pots) they may hold DIFFERENT
food. LOOK to see which has what you need (yellow onions vs red tomatoes) and name THAT one: "the pot with the
onions", "the left pot", "the right pot". Choosing the wrong instance ruins the dish.

COMMANDS: "go to <object>" (the only move; + the instance qualifiers; objects: an ingredient source, a pot /
pan / cutting board / sink, "the plates", "the serving pass") · "pick up" · "put down" · "chop" · "turn on" ·
"scoop" (hold a plate, add the faced station's food onto it) · "serve".
RULES: a manipulation only works while standing at and FACING the right tile, else nothing happens — go to it
first. To plate cooked food, HOLD a plate and "scoop" (don't "pick up"). The cook moves one tile per frame and
takes several frames to arrive — say "continue" until you SEE it has clearly arrived. Keep replies to a few words."""

NOOP = {"continue", "none", "no-op", "noop", "wait", "watch", "stay", "keep going", "nothing", "no action", "hold", "same"}


def parse_cmd(transcript):
    if not transcript:
        return None
    t = transcript.strip().lower().strip('."!,')
    for w in NOOP:                       # explicit no-op words
        if t == w or t.startswith(w + " ") or t.startswith(w + ","):
            return ""
    it = hl_parse(transcript)
    if it is None or it.verb == "wait":
        return ""
    return transcript.strip()


async def turn(session, img, ping="Next action?"):
    """Stream one frame + a light ping; return the model's transcribed reply."""
    await session.send_realtime_input(video=img)
    await session.send_client_content(turns=types.Content(role="user", parts=[types.Part(text=ping)]),
                                      turn_complete=True)
    text = ""
    try:
        async with asyncio.timeout(20):
            async for resp in session.receive():
                sc = getattr(resp, "server_content", None)
                if sc and getattr(sc, "output_transcription", None) and sc.output_transcription.text:
                    text += sc.output_transcription.text
                if resp.text:
                    text += resp.text
                if sc and getattr(sc, "turn_complete", False):
                    break
    except (asyncio.TimeoutError, TimeoutError):
        pass
    return text.strip()


async def run(args):
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    game, recipe = make_game(args.task)
    ctrl = HighLevelController(game, 0)
    print(f"=== GEMINI LIVE | model={args.model} task={args.task} ===\n{recipe}")
    sys_inst = SYS + "\n\n" + recipe
    cfg = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        output_audio_transcription=types.AudioTranscriptionConfig(),
        system_instruction=types.Content(parts=[types.Part(text=sys_inst)]),
        context_window_compression=types.ContextWindowCompressionConfig(
            sliding_window=types.SlidingWindow()),
    )
    comp, log = [], []
    fails = idle = 0
    pending = None
    start = game.stats.get("deliveries", 0)
    won = False
    async with async_playwright() as p:
        b = await p.chromium.launch(args=GPU)
        page = await b.new_page(viewport={"width": 800, "height": 580})
        await page.goto(PAGE, wait_until="load")
        await page.wait_for_function("window.__ready===true", timeout=20000)
        await page.wait_for_timeout(500)
        async with client.aio.live.connect(model=args.model, config=cfg) as session:
            for fi in range(args.max_frames):
                png = await render_frame(page, game.render_state())
                rgb = Image.open(io.BytesIO(png)).convert("RGB")
                prev = ctrl.skill.label if ctrl.skill else None
                event = ""
                # fire a queued manipulation the moment the cook is in position
                if pending and _validate(hl_parse(pending).verb, game, game.cooks[0])[0]:
                    ctrl.issue(pending); event = f"(queued '{pending}' fired)"; pending = None
                reply = await turn(session, rgb)
                cmd = parse_cmd(reply)
                is_cmd = bool(cmd)
                vlm = cmd if is_cmd else "continue · no-op"
                if is_cmd and event.startswith("(queued"):
                    pass                   # already acted this frame; let it run
                elif is_cmd:
                    it = hl_parse(cmd); verb = it.verb if it else None
                    running_goto = (ctrl.skill is not None and type(ctrl.skill).__name__ == "GoTo"
                                    and ctrl.status == "running")
                    if verb in _MANIP:
                        okv, why = _validate(verb, game, game.cooks[0])
                        if okv:
                            ctrl.issue(cmd); pending = None
                        else:
                            pending = cmd  # intent: do it when the cook reaches position
                            event = f"(queued: {why})"
                    elif verb == "goto" and running_goto:
                        pass               # atomic navigation: don't abort an in-progress walk
                    elif not ctrl.issue(cmd):
                        event = f"✗ rejected: {ctrl.error}"; fails += 1
                    log.append((fi, cmd))
                low = ctrl.step() if ctrl.skill is not None else int(Action.STAY)
                acts = [int(Action.STAY)] * len(game.cooks); acts[0] = low
                game.step(acts)
                if not event and prev and ctrl.skill is None:
                    event = f"✓ {prev} done" if ctrl.status == "done" else f"✗ {prev} failed: {ctrl.error}"
                    fails += (ctrl.status == "failed")
                if game.stats.get("deliveries", 0) > start:
                    event = f"✓✓ DELIVERED  +{int(game.score)}"
                info = dict(frame=fi, model=args.model, vlm=vlm, is_cmd=is_cmd, reason=(reply[:60] if reply else ""),
                            low=Action(low).name, log=log, event=event, task=args.task)
                comp.append(composite(rgb, info))
                print(f"[f{fi:3}] say={reply[:34]!r:36} -> {(cmd or 'no-op'):26} btn={Action(low).name:8}"
                      f" hold={hold_str(game.cooks[0].holding):16} {('| ' + event) if event else ''}")
                if game.stats.get("deliveries", 0) > start:
                    won = True
                    png2 = await render_frame(page, game.render_state())
                    comp.append(composite(Image.open(io.BytesIO(png2)).convert("RGB"),
                                {**info, "frame": fi + 1, "vlm": "— task complete —", "is_cmd": True, "low": "STAY"}))
                    break
                idle = idle + 1 if (ctrl.skill is None and not is_cmd) else 0
                if fails >= 9 and ctrl.skill is None:
                    print("[stuck]"); break
                if idle >= 12:
                    print("[idle-stall] kept saying continue while idle"); break
        await b.close()
    out = f"tmp/live_{args.model.replace('.','').replace('-','')}_{args.task}.mp4"
    encode(comp, out, fps=args.fps)
    print(f"\nRESULT: {'SUCCESS' if won else 'did not complete'}  "
          f"(deliveries={game.stats.get('deliveries',0)}, frames={len(comp)}, score={int(game.score)})\nvideo -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["loaded_fries", "soup_pick"], default="soup_pick")
    ap.add_argument("--model", default="gemini-2.5-flash-native-audio-latest")
    ap.add_argument("--max-frames", type=int, default=80)
    ap.add_argument("--fps", type=int, default=4)
    args = ap.parse_args()
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("set GEMINI_API_KEY")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
