#!/usr/bin/env python3
"""Send CookSim gameplay clips to Gemini (via the Salesforce Eng AI Model
Gateway) and collect structured realism/quality feedback.

The sandbox these clips were generated in cannot reach the internal gateway, so
run this on a machine that can (VPN / Salesforce network):

    export ENG_AI_MODEL_GW_URL=https://eng-ai-model-gateway.sfproxy.devx-preprod.aws-esvc1-useast2.aws.sfdc.cl
    export ENG_AI_MODEL_GW_KEY=sk-...

    # 1) sanity-check connectivity + model name (text only):
    python tools/gemini_feedback.py --ping --model gemini-2.5-pro

    # 2) run a feedback round over every clip:
    python tools/gemini_feedback.py --model gemini-2.5-pro --round 1

Writes one JSON per clip to feedback/round_<N>/<clip>.json plus a combined
feedback/round_<N>/summary.json with a severity-sorted issue list. Hand that
summary back to the assistant to drive the next round of fixes.
"""
import argparse
import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLIPS = ROOT / "site" / "assets" / "clips"

# What each clip is meant to show — woven into the prompt so Gemini can judge
# whether the intended multi-step task actually reads on screen.
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
    "ep_bistro": "FULL EPISODE in the Bistro — a whole timed round of soups & salads; the cook completes several dishes and misses some.",
    "ep_diner": "FULL EPISODE in the Diner — a whole timed round across the full menu; many dishes completed, some missed.",
    "ep_grand": "FULL EPISODE in the Grand Kitchen — a whole timed round using every station and ingredient.",
}

SYSTEM = (
    "You are a senior game designer and real-time graphics engineer doing a critical "
    "design review of a cooking game. Be specific, honest, and concrete."
)

PROMPT_TMPL = """\
I am building **CookSim**, an open-source Overcooked-style cooking simulator. It renders in
real-time 3D (Three.js / WebGL) and is used both for human play and as a reinforcement-learning
benchmark. My goal is for it to look and feel as close to **Overcooked 2** as possible while
staying readable.

This is an UNEDITED screen recording of a single AI agent (one cook) playing CookSim.
**What this clip should show:** {desc}
The left panel shows the customer order tickets and the score; the 3D scene is the kitchen.

Watch the whole clip closely and critique it. Focus especially on:
1. ACTION LEGIBILITY — can you clearly SEE the cook pick up, chop, cook/fry/bake, plate, and
   DELIVER? Does the cook visibly face and reach the station it uses? Do ingredients visibly
   change state (raw -> chopped -> cooked)? Is it obvious when a dish is successfully served?
2. RENDERING — models, materials, lighting, shadows, scale, and the look of each station
   (pot, pan, oven, cutting board, sink, the SERVING PASS) and the food/plated dishes.
3. ANIMATION & GAME FEEL — movement, timing, and feedback on success/failure.
4. ANYTHING that looks unrealistic or breaks the illusion of a real kitchen, vs. Overcooked 2.

For every issue, give: area, problem (what you see), fix (a concrete change), severity (low|med|high).
Then give an overall realism_rating from 1-10 and good_enough = true ONLY if rating >= 9 with no
high-severity issues.

Respond with ONLY valid JSON, no prose, in exactly this shape:
{{"clip":"{clip}","issues":[{{"area":"","problem":"","fix":"","severity":"low|med|high"}}],
"realism_rating":0,"good_enough":false,"summary":""}}"""


def gw():
    url = os.environ.get("ENG_AI_MODEL_GW_URL", "").rstrip("/")
    key = os.environ.get("ENG_AI_MODEL_GW_KEY", "")
    if not url or not key:
        sys.exit("Set ENG_AI_MODEL_GW_URL and ENG_AI_MODEL_GW_KEY in the environment.")
    return url, key


def post(path, payload, timeout=300):
    url, key = gw()
    req = urllib.request.Request(
        url + path,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def chat(model, content, timeout=300):
    """OpenAI-compatible chat completion. `content` is the user message content
    (string or list of parts)."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": content},
        ],
        "temperature": 0.3,
        "max_tokens": 2000,
    }
    resp = post("/v1/chat/completions", payload, timeout=timeout)
    return resp["choices"][0]["message"]["content"]


def video_part(mp4: Path):
    """A video content part as an inline base64 data URL (LiteLLM/Gemini route
    video data URLs to Gemini inline_data). If your gateway wants a different
    shape, tweak this one function."""
    b64 = base64.b64encode(mp4.read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:video/mp4;base64,{b64}"}}


def parse_json(text: str):
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1]
        if t.startswith("json"):
            t = t[4:]
        t = t.rsplit("```", 1)[0] if "```" in t else t
    return json.loads(t)


def ping(model):
    print(f"Pinging gateway with model={model} ...")
    out = chat(model, "Reply with exactly: OK", timeout=60)
    print("Response:", out)


def run_round(model, rnd):
    outdir = ROOT / "feedback" / f"round_{rnd}"
    outdir.mkdir(parents=True, exist_ok=True)
    clips = sorted(CLIPS.glob("*.mp4"))
    if not clips:
        sys.exit(f"No clips found in {CLIPS}")
    all_fb = []
    for mp4 in clips:
        name = mp4.stem
        desc = CLIP_DESC.get(name, f"{name} (a CookSim dish being cooked and served).")
        prompt = PROMPT_TMPL.format(desc=desc, clip=name)
        print(f"\n=== {name} ({mp4.stat().st_size//1024} KB) ===")
        try:
            raw = chat(model, [{"type": "text", "text": prompt}, video_part(mp4)])
            fb = parse_json(raw)
        except Exception as e:  # noqa: BLE001
            print("  ERROR:", e)
            fb = {"clip": name, "error": str(e)}
        (outdir / f"{name}.json").write_text(json.dumps(fb, indent=2))
        all_fb.append(fb)
        r = fb.get("realism_rating", "?")
        ge = fb.get("good_enough", "?")
        print(f"  rating={r} good_enough={ge}")

    # combined, severity-sorted issue list
    issues = []
    for fb in all_fb:
        for it in fb.get("issues", []):
            it = dict(it)
            it["clip"] = fb.get("clip")
            issues.append(it)
    order = {"high": 0, "med": 1, "low": 2}
    issues.sort(key=lambda i: order.get(i.get("severity", "low"), 3))
    ratings = [fb["realism_rating"] for fb in all_fb if isinstance(fb.get("realism_rating"), (int, float))]
    summary = {
        "round": rnd,
        "model": model,
        "n_clips": len(all_fb),
        "avg_rating": round(sum(ratings) / len(ratings), 2) if ratings else None,
        "all_good_enough": all(fb.get("good_enough") for fb in all_fb) if all_fb else False,
        "issues_by_severity": issues,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {outdir}/summary.json  avg_rating={summary['avg_rating']} "
          f"all_good_enough={summary['all_good_enough']} issues={len(issues)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-2.5-pro")
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--ping", action="store_true", help="just verify connectivity + model")
    args = ap.parse_args()
    if args.ping:
        ping(args.model)
    else:
        run_round(args.model, args.round)


if __name__ == "__main__":
    main()
