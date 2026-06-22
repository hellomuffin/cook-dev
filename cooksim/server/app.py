"""FastAPI + WebSocket server.

Serves the WebGL front-end and exposes:

* ``GET  /``                  the live game / player UI
* ``GET  /editor``            the visual level editor
* ``GET  /api/layouts``       list of built-in layouts
* ``GET  /api/layout/{name}`` one layout as JSON
* ``GET  /api/recipes``       the recipe book
* ``POST /api/validate``      validate a layout dict
* ``POST /api/generate``      procedurally generate a layout
* ``WS   /ws``                a live, controllable game session

WebSocket protocol — client -> server JSON messages::

    {"type":"reset", "layout":"open_kitchen", "n_players":2, "seed":0}
    {"type":"reset", "layout_data":{...}}            # custom layout
    {"type":"reset", "procedural":{"width":11,...}}  # random layout
    {"type":"action", "player":0, "action":4}        # persistent action
    {"type":"set_speed", "tps":7}                    # 0 pauses
    {"type":"add_bot", "player":1, "kind":"greedy"}
    {"type":"remove_bot", "player":1}
    {"type":"step"}                                  # single manual tick

server -> client::

    {"type":"init", "recipes":[...], "layouts":[...], "state":{...}}
    {"type":"state", "state":{...}}
"""
from __future__ import annotations

import asyncio
import os
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..core.layout import Layout, generate_layout
from ..core.recipes import RecipeBook
from ..layouts import all_layouts, list_layouts, load_layout
from .session import GameSession

WEB_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "web"))

app = FastAPI(title="CookSim", version="0.1.0")
_RECIPES = RecipeBook()


@app.get("/")
def index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


@app.get("/editor")
def editor():
    return FileResponse(os.path.join(WEB_DIR, "editor.html"))


@app.get("/api/layouts")
def api_layouts():
    return {"layouts": list_layouts()}


@app.get("/api/layout/{name}")
def api_layout(name: str):
    try:
        return load_layout(name).to_dict()
    except KeyError:
        return JSONResponse({"error": f"unknown layout {name}"}, status_code=404)


@app.get("/api/recipes")
def api_recipes():
    return {"recipes": _RECIPES.to_dict()}


@app.post("/api/validate")
async def api_validate(payload: dict):
    try:
        lay = Layout.from_dict(payload)
    except Exception as e:  # noqa: BLE001
        return {"valid": False, "issues": [f"parse error: {e}"]}
    issues = lay.validate()
    return {"valid": not issues, "issues": issues}


@app.post("/api/generate")
async def api_generate(payload: dict):
    lay = generate_layout(**payload)
    return lay.to_dict()


@app.websocket("/ws")
async def ws(ws: WebSocket):
    await ws.accept()
    session = GameSession()
    await ws.send_json(
        {
            "type": "init",
            "recipes": _RECIPES.to_dict(),
            "layouts": list_layouts(),
            "state": session.state(),
        }
    )

    stop = asyncio.Event()

    async def game_loop():
        while not stop.is_set():
            if session.tps > 0:
                session.tick()
                try:
                    await ws.send_json({"type": "state", "state": session.state()})
                except Exception:
                    stop.set()
                    return
                await asyncio.sleep(1.0 / session.tps)
            else:
                await asyncio.sleep(0.05)

    loop_task = asyncio.create_task(game_loop())

    try:
        while True:
            msg = await ws.receive_json()
            mtype = msg.get("type")
            if mtype == "action":
                session.set_action(int(msg["player"]), int(msg["action"]))
            elif mtype == "reset":
                session.reset(
                    layout=msg.get("layout"),
                    layout_data=msg.get("layout_data"),
                    procedural=msg.get("procedural"),
                    n_players=msg.get("n_players"),
                    config=msg.get("config"),
                    seed=msg.get("seed", 0),
                    order_recipes=msg.get("order_recipes"),
                )
                await ws.send_json({"type": "state", "state": session.state()})
            elif mtype == "set_speed":
                session.tps = float(msg.get("tps", 7))
            elif mtype == "add_bot":
                session.add_bot(int(msg["player"]), msg.get("kind", "greedy"), msg.get("role", "any"))
            elif mtype == "remove_bot":
                session.remove_bot(int(msg["player"]))
            elif mtype == "step":
                session.tick()
                await ws.send_json({"type": "state", "state": session.state()})
    except WebSocketDisconnect:
        pass
    finally:
        stop.set()
        loop_task.cancel()


# Static assets (js, css). Mounted last so it doesn't shadow API routes.
if os.path.isdir(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


def main():
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Run the CookSim server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
