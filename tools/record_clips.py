#!/usr/bin/env python3
"""Record the 13 CookSim demo clips on the GPU (NVIDIA via ANGLE/Vulkan).

Recipe clips are canvas-focused (ui=min), the 3 episode clips keep the HUD.
Shadow maps are disabled for headless GPU recording (ANGLE/Vulkan shadow bug);
real browsers on real GPUs still get shadows. Needs a running server (port 8085)
and, in the env, LD_LIBRARY_PATH including the conda lib dir.

    python tools/record_clips.py            # -> tmp/vids/*.webm
"""
import asyncio, os, shutil, sys
from playwright.async_api import async_playwright

ARGS = ["--no-sandbox", "--use-gl=angle", "--use-angle=vulkan", "--enable-features=Vulkan",
        "--ignore-gpu-blocklist", "--enable-gpu", "--autoplay-policy=no-user-gesture-required"]
BASE = os.environ.get("COOKSIM_BASE", "http://127.0.0.1:8085/")
OUT = os.environ.get("COOKSIM_VIDS", "tmp/vids")

RECIPE = [
    ("onion_soup", "cramped_room", "onion_soup", 26),
    ("greek_salad", "bistro", "greek_salad", 30),
    ("deluxe_burger", "diner", "deluxe_burger", 32),
    ("cheeseburger", "burger_bar", "cheeseburger", 26),
    ("pizza", "pizzeria", "pizza", 28),
    ("sushi", "grand_kitchen", "sushi", 30),
    ("fish_and_chips", "grand_kitchen", "fish_and_chips", 30),
    ("fried_rice", "grand_kitchen", "fried_rice", 32),
    ("loaded_fries", "diner", "loaded_fries", 26),
    ("veggie_bake", "grand_kitchen", "veggie_bake", 30),
]
EPISODE = [
    ("ep_bistro", "bistro", 40),
    ("ep_diner", "diner", 42),
    ("ep_grand", "grand_kitchen", 42),
]


async def rec(p, out, url, secs, W, H):
    b = await p.chromium.launch(args=ARGS)
    ctx = await b.new_context(viewport={"width": W, "height": H},
        record_video_dir=OUT, record_video_size={"width": W, "height": H})
    pg = await ctx.new_page()
    await pg.add_init_script("window.__noshadow=true;")
    await pg.goto(url, wait_until="load")
    await pg.wait_for_timeout(secs * 1000 + 1500)
    v = pg.video
    await ctx.close()
    shutil.move(await v.path(), f"{OUT}/{out}.webm")
    await b.close()
    print("rec", out, os.path.getsize(f"{OUT}/{out}.webm") // 1024, "KB", flush=True)


async def main():
    os.makedirs(OUT, exist_ok=True)
    only = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else None
    async with async_playwright() as p:
        for out, lay, rec_, secs in RECIPE:
            if only and out not in only:
                continue
            await rec(p, out, f"{BASE}?play=1&layout={lay}&recipe={rec_}&players=1&bots=1&speed=4&ui=min", secs, 1120, 690)
        for out, lay, secs in EPISODE:
            if only and out not in only:
                continue
            await rec(p, out, f"{BASE}?play=1&layout={lay}&players=1&bots=1&speed=7", secs, 1280, 720)
asyncio.run(main())
print("ALL RECORDED")
