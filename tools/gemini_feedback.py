#!/usr/bin/env python3
"""Send CookSim gameplay clips to Gemini and collect structured realism feedback.

    export GEMINI_API_KEY=AIza...
    python tools/gemini_feedback.py --round 1 --model gemini-2.5-pro

Writes feedback/round_<N>/<clip>.json per clip and feedback/round_<N>/summary.json
(a severity-sorted, deduped issue list + avg rating + all_good_enough flag).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from google import genai
from google.genai import types
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
CLIPS = ROOT / "site" / "assets" / "clips"

CLIP_DESC = {
    "onion_soup": "Onion Soup — gather 3 onions, boil them in a pot, plate, serve.",
    "greek_salad": "Greek Salad — chop lettuce AND tomato on a single cutting board, add cheese, plate, serve.",
    "deluxe_burger": "Deluxe Burger — fry a patty, chop lettuce + tomato, stack on a bun (4 components), serve.",
    "cheeseburger": "Cheeseburger — fry a patty, add a bun and cheese, plate, serve.",
    "pizza": "Pizza — put dough + tomato + cheese in the oven, bake, plate, serve.",
    "sushi": "Sushi — cook rice, chop raw fish, plate both, serve.",
    "fish_and_chips": "Fish & Chips — fry the fish, fry the potato, plate both, serve.",
    "fried_rice": "Fried Rice — cook rice, fry an egg, chop onion, plate all, serve.",
    "loaded_fries": "Loaded Fries — fry the potato, top with cheese, serve.",
    "veggie_bake": "Veggie Bake — bake potato + mushroom + cheese in the oven, serve.",
    "ep_bistro": "FULL EPISODE in the Bistro — a whole timed round of soups & salads; several dishes completed, some missed.",
    "ep_diner": "FULL EPISODE in the Diner — a whole timed round across the full menu; many completed, some missed.",
    "ep_grand": "FULL EPISODE in the Grand Kitchen — a whole timed round using every station and ingredient.",
}

SYSTEM = (
    "You are a game-UX reviewer who specializes in ACTION LEGIBILITY — whether a viewer can "
    "clearly see and understand what is happening on screen. You are NOT a graphics critic. "
    "You must completely IGNORE visual fidelity: lighting, shadows, materials, art style, polish, "
    "whether the scene looks '2D' or 'flat', whether objects 'float', and framerate are all "
    "OUT OF SCOPE and must never be raised. Judge only readability of the gameplay."
)

PROMPT = """\
I am building **CookSim**, an Overcooked-style cooking simulator (a research benchmark and a game).
I care about ONE thing only: **action legibility** — can a viewer clearly SEE and UNDERSTAND every
action the cook performs and every change of state? I do NOT care about visual realism, lighting,
shadows, materials, art quality, how "2D/flat" it looks, or polish. Please IGNORE all of that
entirely and never mention it.

This is an UNEDITED screen recording of one AI cook playing CookSim.
Intended task in this clip: {desc}
(The left panel, if visible, lists the customer orders and the score — but judge the 3D scene, not
the panel: ideally the action is readable WITHOUT the UI.)

Go through the clip and, for each of the following, decide whether it is CLEARLY and unambiguously
readable to a first-time viewer. If not, say exactly what was confusing:
1. PICKUP  — when the cook takes an ingredient or a plate, is it clear THAT it picked something up,
   WHAT the item is, and WHERE it came from (which dispenser/counter/station)?
2. PLACE/LOAD — when it puts an item down or into a pot / pan / oven / cutting board, is that clear?
3. PROCESS — while chopping / boiling / frying / baking, is it clear an action is happening, AT WHICH
   station, and how far along it is?
4. STATE CHANGE — does each ingredient visibly look different between raw -> chopped -> cooked, and can
   you tell what is inside a pot/pan/oven and whether it's done/burnt?
5. CARRY — at any moment, is it clear what the cook is currently holding?
6. TARGETING — is it clear which station/tile the cook is interacting with (does it approach/face it
   in a way that makes the target obvious)?
7. DELIVERY — when a finished dish is served at the pass, is it CLEAR that a delivery happened and that
   it SUCCEEDED (or failed)? Is the "you completed an order" moment obvious?
8. SENSE — does the whole sequence read as sensible cooking, with no steps that happen "magically",
   instantly, or in a way a viewer can't follow?

For EVERY action that is unclear, return an issue with:
- step: one of pickup | place | process | state | carry | target | deliver | sense
- problem: what you observed and why it was hard to read
- fix: a concrete change that would make THAT action legible (e.g. a clearer pickup motion, a visibly
  different chopped model, an explicit "served!" cue). Suggest gameplay-readability changes only.
- severity: low | med | high

Do NOT raise any issue about graphics, lighting, shadows, realism, framerate, or art style — those are
explicitly not my concern.

Give legibility_rating 1-10 (10 = a first-time viewer can instantly read every action and state change
with the UI hidden). good_enough = true ONLY if legibility_rating >= 8 with no high-severity gaps."""


class Issue(BaseModel):
    step: str
    problem: str
    fix: str
    severity: str


class Feedback(BaseModel):
    issues: list[Issue]
    legibility_rating: int
    good_enough: bool
    summary: str


def review_clip(client, model, mp4: Path):
    desc = CLIP_DESC.get(mp4.stem, f"{mp4.stem} being cooked and served.")
    f = client.files.upload(file=str(mp4))
    for _ in range(60):
        if f.state.name == "ACTIVE":
            break
        if f.state.name == "FAILED":
            raise RuntimeError("file processing failed")
        time.sleep(2)
        f = client.files.get(name=f.name)
    resp = client.models.generate_content(
        model=model,
        contents=[f, PROMPT.format(desc=desc)],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,
            temperature=0.3,
            response_mime_type="application/json",
            response_schema=Feedback,
        ),
    )
    fb = json.loads(resp.text)
    fb["clip"] = mp4.stem
    try:
        client.files.delete(name=f.name)
    except Exception:
        pass
    return fb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-2.5-pro")
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--only", default="", help="comma-separated clip stems to review")
    args = ap.parse_args()

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("set GEMINI_API_KEY")
    client = genai.Client(api_key=key)

    clips = sorted(CLIPS.glob("*.mp4"))
    if args.only:
        want = set(args.only.split(","))
        clips = [c for c in clips if c.stem in want]
    outdir = ROOT / "feedback" / f"round_{args.round}"
    outdir.mkdir(parents=True, exist_ok=True)

    all_fb = []
    for mp4 in clips:
        print(f"=== {mp4.stem} ({mp4.stat().st_size//1024} KB) ===", flush=True)
        try:
            fb = review_clip(client, args.model, mp4)
        except Exception as e:  # noqa: BLE001
            print("  ERROR:", e, flush=True)
            fb = {"clip": mp4.stem, "error": str(e), "issues": [], "legibility_rating": 0, "good_enough": False}
        (outdir / f"{mp4.stem}.json").write_text(json.dumps(fb, indent=2))
        all_fb.append(fb)
        print(f"  rating={fb.get('legibility_rating')} good_enough={fb.get('good_enough')} "
              f"issues={len(fb.get('issues', []))}", flush=True)

    issues = []
    for fb in all_fb:
        for it in fb.get("issues", []):
            issues.append({**it, "clip": fb.get("clip")})
    rank = {"high": 0, "med": 1, "medium": 1, "low": 2}
    issues.sort(key=lambda i: rank.get(str(i.get("severity", "low")).lower(), 3))
    ratings = [fb["legibility_rating"] for fb in all_fb if isinstance(fb.get("legibility_rating"), (int, float)) and fb.get("legibility_rating")]
    summary = {
        "round": args.round, "model": args.model, "n_clips": len(all_fb),
        "avg_rating": round(sum(ratings) / len(ratings), 2) if ratings else None,
        "ratings": {fb["clip"]: fb.get("legibility_rating") for fb in all_fb},
        "all_good_enough": all(fb.get("good_enough") for fb in all_fb) if all_fb else False,
        "n_high": sum(1 for i in issues if str(i.get("severity")).lower() == "high"),
        "issues_by_severity": issues,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\navg_rating={summary['avg_rating']} all_good_enough={summary['all_good_enough']} "
          f"high_severity={summary['n_high']} total_issues={len(issues)}")
    print(f"-> {outdir}/summary.json")


if __name__ == "__main__":
    main()
