#!/usr/bin/env python3
"""Headless render check for a static site page: load it, collect any console
errors / page errors, expand all <details>, and screenshot. Usage:

    python tools/check_page.py site/taxonomy.html [out.png]
"""
import asyncio, os, sys
from playwright.async_api import async_playwright

PAGE = sys.argv[1] if len(sys.argv) > 1 else "site/taxonomy.html"
OUT = sys.argv[2] if len(sys.argv) > 2 else "tmp/taxonomy_check.png"


async def main():
    url = "file://" + os.path.abspath(PAGE)
    errs = []
    async with async_playwright() as p:
        b = await p.chromium.launch(args=["--no-sandbox"])
        pg = await b.new_page(viewport={"width": 1100, "height": 1400})
        pg.on("console", lambda m: errs.append(f"console.{m.type}: {m.text}") if m.type in ("error", "warning") else None)
        pg.on("pageerror", lambda e: errs.append(f"pageerror: {e}"))
        await pg.goto(url, wait_until="load")
        await pg.wait_for_timeout(400)
        n_cats = await pg.eval_on_selector_all("details.cat", "els => els.length")
        n_stats = await pg.eval_on_selector_all(".stat", "els => els.length")
        await pg.click("#expandAll")
        await pg.wait_for_timeout(300)
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        await pg.screenshot(path=OUT, full_page=True)
        await b.close()
    print(f"categories={n_cats} stat_cards={n_stats}")
    print("errors:", errs if errs else "none")
    print("screenshot:", OUT)


asyncio.run(main())
