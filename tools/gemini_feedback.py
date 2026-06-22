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

SYSTEM = ("You are a senior game designer and real-time graphics engineer doing a critical, "
          "specific, honest design review of a cooking game.")

PROMPT = """\
I am building **CookSim**, an open-source Overcooked-style cooking simulator that renders in
real-time 3D (Three.js/WebGL) and is also a reinforcement-learning benchmark. My goal is for it
to look and feel as close to **Overcooked 2** as possible while staying readable.

This is an UNEDITED screen recording of a single AI cook playing CookSim.
What this clip should show: {desc}
The left panel shows the customer order tickets and the score; the 3D scene is the kitchen.

Critique it, focusing on:
1. ACTION LEGIBILITY — can you clearly SEE the cook pick up, chop, cook/fry/bake, plate, and
   DELIVER? Does the cook visibly face & reach the station it uses? Do ingredients visibly change
   state (raw->chopped->cooked)? Is a successful delivery obvious?
2. RENDERING — models, materials, lighting, shadows, scale, the look of each station (pot, pan,
   oven, cutting board, sink, the SERVING PASS) and the food/plated dishes.
3. ANIMATION & GAME FEEL — movement, timing, success/failure feedback.
4. Anything that looks unrealistic or breaks the kitchen illusion vs Overcooked 2.

For each issue give area, problem (what you observe), fix (a concrete change), and severity
(low/med/high). Give realism_rating 1-10 and good_enough=true ONLY if rating>=9 with no high-severity
issues. Be strict; this is for iterative improvement."""


class Issue(BaseModel):
    area: str
    problem: str
    fix: str
    severity: str


class Feedback(BaseModel):
    issues: list[Issue]
    realism_rating: int
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
            fb = {"clip": mp4.stem, "error": str(e), "issues": [], "realism_rating": 0, "good_enough": False}
        (outdir / f"{mp4.stem}.json").write_text(json.dumps(fb, indent=2))
        all_fb.append(fb)
        print(f"  rating={fb.get('realism_rating')} good_enough={fb.get('good_enough')} "
              f"issues={len(fb.get('issues', []))}", flush=True)

    issues = []
    for fb in all_fb:
        for it in fb.get("issues", []):
            issues.append({**it, "clip": fb.get("clip")})
    rank = {"high": 0, "med": 1, "medium": 1, "low": 2}
    issues.sort(key=lambda i: rank.get(str(i.get("severity", "low")).lower(), 3))
    ratings = [fb["realism_rating"] for fb in all_fb if isinstance(fb.get("realism_rating"), (int, float)) and fb.get("realism_rating")]
    summary = {
        "round": args.round, "model": args.model, "n_clips": len(all_fb),
        "avg_rating": round(sum(ratings) / len(ratings), 2) if ratings else None,
        "ratings": {fb["clip"]: fb.get("realism_rating") for fb in all_fb},
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
