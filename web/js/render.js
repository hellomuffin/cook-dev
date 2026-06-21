/* CookSim isometric 2.5D renderer (PixiJS v7).
 *
 * Renders the kitchen in a 2:1 isometric projection with real depth: counters
 * and walls are extruded blocks with directional per-face shading, floors are
 * tiled diamonds, stations are modelled fixtures, and chefs are billboarded
 * characters with contact shadows. Cooking emits steam/fire, sinks shimmer,
 * deliveries burst sparkles. Cook motion is interpolated between server ticks.
 *
 * View API (unchanged from the old renderer): new KitchenRenderer(view),
 * setState(state), update(dt), resize(W,H).
 */
(function (global) {
  "use strict";

  const TW = 96, TH = 48;          // tile diamond width / height
  const CH = 26;                   // counter block height
  const WH = 46;                   // wall block height

  // ---- palette ------------------------------------------------------------
  const COL = { bg: 0x232029 };
  const FLOOR_A = 0xece4d4, FLOOR_B = 0xe2d8c2, GROUT = 0xc7b89c;
  const COUNTER = 0xc79a62, WALL = 0x726058;
  const ING = {
    onion:    { c: 0xe8c14a, c2: 0xcda23a },
    tomato:   { c: 0xe14b34, c2: 0xb5392a },
    lettuce:  { c: 0x6cc24a, c2: 0x4e9a36 },
    mushroom: { c: 0xb79a82, c2: 0x8a6f59 },
    meat:     { c: 0xb35a45, c2: 0x843d2e },
    fish:     { c: 0x9cbcd6, c2: 0x6f93b0 },
    bun:      { c: 0xd9a566, c2: 0xb9844a },
    cheese:   { c: 0xf2c94c, c2: 0xd0a728 },
    rice:     { c: 0xf3efe3, c2: 0xd9d2bd },
    dough:    { c: 0xeed9a8, c2: 0xd5bd86 },
    egg:      { c: 0xf6e6c5, c2: 0xe7cf9c },
    potato:   { c: 0xc89b5e, c2: 0xa67d44 },
  };

  function lerp(a, b, t) { return a + (b - a) * t; }
  function shade(hex, f) {
    const r = Math.min(255, ((hex >> 16) & 255) * f);
    const g = Math.min(255, ((hex >> 8) & 255) * f);
    const b = Math.min(255, (hex & 255) * f);
    return (Math.round(r) << 16) | (Math.round(g) << 8) | Math.round(b);
  }

  // =========================================================================
  class KitchenRenderer {
    constructor(view) {
      this.app = new PIXI.Application({
        view, antialias: true, backgroundColor: COL.bg,
        resolution: window.devicePixelRatio || 1, autoDensity: true,
      });
      this.world = new PIXI.Container();
      this.app.stage.addChild(this.world);
      this.base = new PIXI.Graphics();        // static geometry (built on layout change)
      this.dyn = new PIXI.Graphics();          // redrawn every frame
      this.labels = new PIXI.Container();      // pooled text
      this.world.addChild(this.base, this.dyn, this.labels);

      this.state = null;
      this.cookView = {};
      this.particles = [];
      this.sparkles = [];
      this._lastScore = 0;
      this._sig = "";
      this._t = 0;
    }

    iso(x, y) {
      return [(x - y) * (TW / 2), (x + y) * (TH / 2)];
    }

    fit(W, H) {
      if (!this.state) return;
      const s = this.state;
      let minX = 1e9, maxX = -1e9, minY = 1e9, maxY = -1e9;
      for (let y = 0; y <= s.height; y++) for (let x = 0; x <= s.width; x++) {
        const [sx, sy] = this.iso(x, y);
        minX = Math.min(minX, sx); maxX = Math.max(maxX, sx);
        minY = Math.min(minY, sy); maxY = Math.max(maxY, sy);
      }
      minY -= WH; maxY += TH;
      const bw = maxX - minX, bh = maxY - minY, pad = 36;
      const sc = Math.min((W - pad * 2) / bw, (H - pad * 2) / bh);
      this.world.scale.set(sc);
      this.world.position.set(W / 2 - ((minX + maxX) / 2) * sc, H / 2 - ((minY + maxY) / 2) * sc);
    }

    setState(state) {
      this.state = state;
      const sig = state.width + "x" + state.height + ":" + JSON.stringify(state.terrain);
      if (sig !== this._sig) { this._sig = sig; this._buildBase(); this.cookView = {}; }
      for (const p of state.players)
        if (!this.cookView[p.id]) this.cookView[p.id] = { x: p.x, y: p.y, bob: 0, phase: Math.random() * 6 };
      if (this._lastScore && state.score > this._lastScore + 0.5) this._emitDelivery();
      this._lastScore = state.score;
      this.fit(this.app.renderer.width / this.app.renderer.resolution,
               this.app.renderer.height / this.app.renderer.resolution);
    }

    // ---- iso primitives --------------------------------------------------
    _diamond(g, cx, cy, col, line) {
      g.beginFill(col);
      g.drawPolygon([cx, cy - TH / 2, cx + TW / 2, cy, cx, cy + TH / 2, cx - TW / 2, cy]);
      g.endFill();
      if (line !== undefined) {
        g.lineStyle(1, line, 0.35);
        g.drawPolygon([cx, cy - TH / 2, cx + TW / 2, cy, cx, cy + TH / 2, cx - TW / 2, cy]);
        g.lineStyle(0);
      }
    }

    _block(g, cx, cy, h, base, topInset) {
      // left face
      g.beginFill(shade(base, 0.62));
      g.drawPolygon([cx - TW / 2, cy, cx, cy + TH / 2, cx, cy + TH / 2 - h, cx - TW / 2, cy - h]);
      g.endFill();
      // right face
      g.beginFill(shade(base, 0.82));
      g.drawPolygon([cx + TW / 2, cy, cx, cy + TH / 2, cx, cy + TH / 2 - h, cx + TW / 2, cy - h]);
      g.endFill();
      // top face
      const ty = cy - h;
      this._diamond(g, cx, ty, shade(base, 1.0));
      g.lineStyle(1, shade(base, 1.18), 0.5);
      g.drawPolygon([cx, ty - TH / 2, cx + TW / 2, ty, cx, ty + TH / 2, cx - TW / 2, ty]);
      g.lineStyle(0);
      return ty; // top-centre y
    }

    // ---- static base -----------------------------------------------------
    _buildBase() {
      this.base.clear();
      const g = this.base, st = this.state;
      // floor first
      for (let y = 0; y < st.height; y++) for (let x = 0; x < st.width; x++) {
        const [cx, cy] = this.iso(x, y);
        if (st.terrain[y][x] === "floor")
          this._diamond(g, cx, cy, ((x + y) % 2 ? FLOOR_B : FLOOR_A), GROUT);
      }
      // raised blocks back-to-front
      const cells = [];
      for (let y = 0; y < st.height; y++) for (let x = 0; x < st.width; x++)
        if (st.terrain[y][x] !== "floor") cells.push([x, y, st.terrain[y][x]]);
      cells.sort((a, b) => (a[0] + a[1]) - (b[0] + b[1]));
      for (const [x, y, t] of cells) {
        const [cx, cy] = this.iso(x, y);
        if (t === "wall") { this._block(g, cx, cy, WH, WALL); continue; }
        const ty = this._block(g, cx, cy, CH, COUNTER);
        this._fixture(g, t, cx, ty);
      }
    }

    // static part of each station (no dynamic contents)
    _fixture(g, t, cx, ty) {
      if (t === "counter") return;
      if (t === "pot") { this._pot(g, cx, ty); }
      else if (t === "pan") { this._pan(g, cx, ty); }
      else if (t === "oven") { this._oven(g, cx, ty); }
      else if (t === "cutting_board") { this._board(g, cx, ty); }
      else if (t === "sink") { this._sink(g, cx, ty); }
      else if (t === "serving") { this._serving(g, cx, ty); }
      else if (t === "trash") { this._trash(g, cx, ty); }
      else if (t === "plate_source") { this._plateStack(g, cx, ty); }
      else if (t.endsWith("_source")) { this._crate(g, cx, ty, t.replace("_source", "")); }
    }

    _pot(g, cx, ty) {
      g.beginFill(0x000000, 0.18); g.drawEllipse(cx, ty + 2, 26, 11); g.endFill();
      g.beginFill(0x8b9298); g.drawRoundedRect(cx - 24, ty - 16, 48, 22, 8); g.endFill();
      g.beginFill(0xbfc5cb); g.drawRoundedRect(cx - 22, ty - 16, 44, 14, 7); g.endFill();
      g.beginFill(0xeef1f3, 0.7); g.drawRoundedRect(cx - 17, ty - 14, 30, 4, 2); g.endFill();
      g.lineStyle(3, 0x6f757b); g.moveTo(cx - 24, ty - 8); g.lineTo(cx - 31, ty - 7);
      g.moveTo(cx + 24, ty - 8); g.lineTo(cx + 31, ty - 7); g.lineStyle(0);
    }
    _pan(g, cx, ty) {
      g.beginFill(0x000000, 0.18); g.drawEllipse(cx, ty + 1, 22, 9); g.endFill();
      g.beginFill(0x33333a); g.drawEllipse(cx, ty - 5, 22, 11); g.endFill();
      g.beginFill(0x4a4a52); g.drawEllipse(cx, ty - 6, 18, 8); g.endFill();
      g.lineStyle(5, 0x2a2a30); g.moveTo(cx + 18, ty - 5); g.lineTo(cx + 36, ty - 11); g.lineStyle(0);
    }
    _oven(g, cx, ty) {
      g.beginFill(0x4a4d55); g.drawRoundedRect(cx - 26, ty - 30, 52, 34, 6); g.endFill();
      g.beginFill(0x33353c); g.drawRoundedRect(cx - 22, ty - 26, 44, 22, 4); g.endFill();
      g.beginFill(0xffae54, 0.9); g.drawRoundedRect(cx - 18, ty - 23, 36, 15, 3); g.endFill();
      g.beginFill(0x2a2c32); g.drawCircle(cx - 14, ty - 1, 2.4); g.drawCircle(cx, ty - 1, 2.4); g.drawCircle(cx + 14, ty - 1, 2.4); g.endFill();
    }
    _board(g, cx, ty) {
      g.beginFill(0xb98e4c); g.drawEllipse(cx, ty - 3, 26, 13); g.endFill();
      g.beginFill(0xd8b06a); g.drawEllipse(cx, ty - 5, 24, 11); g.endFill();
      g.lineStyle(1, 0xb98e4c, 0.6);
      for (let i = -1; i <= 1; i++) { g.moveTo(cx - 14, ty - 5 + i * 5); g.lineTo(cx + 12, ty - 5 + i * 5); }
      g.lineStyle(0);
      g.beginFill(0xd0d4d8); g.drawPolygon([cx + 10, ty - 14, cx + 22, ty - 18, cx + 14, ty - 8]); g.endFill();
    }
    _sink(g, cx, ty) {
      g.beginFill(0x8b9298); g.drawRoundedRect(cx - 26, ty - 14, 52, 20, 5); g.endFill();
      g.beginFill(0x5a6066); g.drawRoundedRect(cx - 21, ty - 11, 42, 14, 3); g.endFill();
      g.lineStyle(3, 0xaab0b6); g.moveTo(cx + 12, ty - 11); g.lineTo(cx + 12, ty - 20); g.lineTo(cx + 2, ty - 20); g.lineStyle(0);
    }
    _serving(g, cx, ty) {
      g.beginFill(0x2f7d57); g.drawRoundedRect(cx - 28, ty - 30, 56, 34, 6); g.endFill();
      g.beginFill(0x49b07d); g.drawRoundedRect(cx - 24, ty - 16, 48, 20, 4); g.endFill();
      g.beginFill(0xffffff, 0.92); g.drawPolygon([cx - 9, ty - 8, cx + 9, ty - 8, cx, ty + 2]); g.endFill();
      g.beginFill(0xffd98a); g.drawRoundedRect(cx - 26, ty - 30, 52, 6, 2); g.endFill();
    }
    _trash(g, cx, ty) {
      g.beginFill(0x33363c); g.drawRoundedRect(cx - 16, ty - 22, 32, 28, 4); g.endFill();
      g.beginFill(0x44474e); g.drawRoundedRect(cx - 14, ty - 20, 28, 24, 3); g.endFill();
      g.beginFill(0x26282d); g.drawRoundedRect(cx - 19, ty - 26, 38, 6, 3); g.endFill();
    }
    _plateStack(g, cx, ty) {
      for (let i = 4; i >= 0; i--) {
        g.beginFill(i === 0 ? 0xffffff : 0xe8ebee);
        g.drawEllipse(cx, ty - 2 - i * 3, 22, 8); g.endFill();
        g.lineStyle(1, 0xbfc6cc, 0.7); g.drawEllipse(cx, ty - 2 - i * 3, 22, 8); g.lineStyle(0);
      }
    }
    _crate(g, cx, ty, ing) {
      g.beginFill(0x7a5230); g.drawRoundedRect(cx - 24, ty - 20, 48, 26, 4); g.endFill();
      g.beginFill(0x8a5f38); g.drawRoundedRect(cx - 21, ty - 17, 42, 20, 3); g.endFill();
      g.lineStyle(1.5, 0x6a4526, 0.7); g.moveTo(cx - 21, ty - 7); g.lineTo(cx + 21, ty - 7); g.lineStyle(0);
      this._ingredient(g, ing, cx, ty - 11, 0.92, "raw");
    }

    // ---- dynamic per-frame ----------------------------------------------
    update(dt) {
      this._t += dt;
      if (!this.state) return;
      const g = this.dyn; g.clear();
      this.labels.removeChildren();
      const st = this.state;

      // collect depth-sorted dynamic drawables
      const draw = [];
      const topY = (x, y) => this.iso(x, y)[1] - CH;

      for (const s of st.stations) {
        const [cx] = this.iso(s.x, s.y);
        const ty = topY(s.x, s.y);
        draw.push({ d: s.x + s.y + 0.1, fn: () => this._stationDyn(g, s, cx, ty) });
      }
      for (const o of st.objects) {
        const [cx] = this.iso(o.x, o.y);
        const ty = topY(o.x, o.y);
        draw.push({ d: o.x + o.y + 0.15, fn: () => this._item(g, cx, ty - 6, o.item, 0.9) });
      }
      const interp = Math.min(1, dt * 12);
      for (const p of st.players) {
        const v = this.cookView[p.id];
        v.x = lerp(v.x, p.x, interp); v.y = lerp(v.y, p.y, interp);
        const moving = Math.abs(v.x - p.x) > 0.02 || Math.abs(v.y - p.y) > 0.02;
        v.phase += dt * (moving ? 12 : 3);
        v.bob = moving ? Math.sin(v.phase) * 2.6 : Math.sin(v.phase) * 0.7;
        draw.push({ d: v.x + v.y + 0.5, fn: () => this._cook(g, p, v) });
      }
      draw.sort((a, b) => a.d - b.d);
      for (const dd of draw) dd.fn();

      this._fx(g, dt);
    }

    _stationDyn(g, s, cx, ty) {
      if (s.type === "pot" || s.type === "oven") {
        if (s.contents && s.contents.length) {
          const col = this._mix(s.contents, s.status);
          g.beginFill(col); g.drawEllipse(cx, ty - 12, 19, 8); g.endFill();
          g.beginFill(0xffffff, 0.16); g.drawEllipse(cx - 5, ty - 14, 6, 2.4); g.endFill();
          if (s.status === "cooking" && Math.random() < 0.3)
            this.particles.push({ x: cx + (Math.random() - .5) * 18, y: ty - 14, vy: -10, life: 1, kind: "steam" });
        }
        if (s.status === "cooking") this._ring(g, cx, ty - 34, 11, s.progress, 0x66d98a);
        if (s.status === "cooked") {
          this._ring(g, cx, ty - 34, 11, 1, 0x66d98a);
          if (s.burn_progress > 0.55 && Math.floor(this._t * 4) % 2 === 0) this._ring(g, cx, ty - 34, 11, s.burn_progress, 0xe8a23a);
          if (Math.random() < 0.22) this.particles.push({ x: cx + (Math.random() - .5) * 16, y: ty - 16, vy: -12, life: 1, kind: "steam" });
        }
        if (s.status === "burnt") {
          this._badge(g, cx, ty - 34, 0x2a2a2a, "!");
          if (Math.random() < 0.5) this.particles.push({ x: cx + (Math.random() - .5) * 16, y: ty - 16, vy: -16, life: 1, kind: "fire" });
        }
      } else if (s.type === "pan") {
        if (s.contents && s.contents.length) {
          const col = this._mix(s.contents, s.status);
          g.beginFill(col); g.drawEllipse(cx, ty - 8, 14, 6); g.endFill();
          if (s.status === "cooking" && Math.random() < 0.25)
            this.particles.push({ x: cx + (Math.random() - .5) * 14, y: ty - 10, vy: -9, life: 1, kind: "steam" });
        }
        if (s.status === "cooking") this._ring(g, cx, ty - 30, 10, s.progress, 0x66d98a);
        if (s.status === "cooked") this._ring(g, cx, ty - 30, 10, 1, 0x66d98a);
        if (s.status === "burnt") this._badge(g, cx, ty - 30, 0x2a2a2a, "!");
      } else if (s.type === "cutting_board") {
        if (s.item) {
          this._item(g, cx, ty - 9, s.item, 0.8);
          if (!s.done && s.progress > 0) this._ring(g, cx, ty - 30, 9, s.progress, 0xe8c14a);
          if (s.done) this._badge(g, cx + 17, ty - 22, 0x4cd07a, "✓");
        }
      } else if (s.type === "sink") {
        if (s.dirty > 0 || s.clean_ready > 0) {
          g.beginFill(0x6fc3e8, 0.7 + 0.2 * Math.sin(this._t * 4)); g.drawEllipse(cx, ty - 6, 19, 7); g.endFill();
        }
        if (s.progress > 0) this._ring(g, cx, ty - 28, 9, s.progress, 0x6fc3e8);
        if (s.dirty > 0) this._badge(g, cx - 17, ty - 22, 0x9a7b50, String(s.dirty));
        if (s.clean_ready > 0) this._badge(g, cx + 17, ty - 22, 0xffffff, String(s.clean_ready), 0x556);
      }
    }

    _mix(contents, status) {
      let r = 0, gg = 0, b = 0;
      for (const c of contents) { const k = ING[c.name] || { c: 0xcccccc }; r += (k.c >> 16) & 255; gg += (k.c >> 8) & 255; b += k.c & 255; }
      const n = contents.length || 1; r /= n; gg /= n; b /= n;
      if (status === "burnt") { r *= .3; gg *= .28; b *= .28; }
      return (Math.round(r) << 16) | (Math.round(gg) << 8) | Math.round(b);
    }

    // ---- items -----------------------------------------------------------
    _item(g, cx, cy, item, sc) {
      sc = sc || 1; if (!item) return;
      g.beginFill(0x000000, 0.14); g.drawEllipse(cx, cy + 14 * sc, 15 * sc, 5 * sc); g.endFill();
      if (item.kind === "plate") this._plate(g, cx, cy, item, sc);
      else this._ingredient(g, item.name, cx, cy, sc, item.state);
    }
    _plate(g, cx, cy, plate, sc) {
      const rim = plate.dirty ? 0xb7a98f : 0xe2e7eb, face = plate.dirty ? 0x9c8e74 : 0xffffff;
      g.beginFill(rim); g.drawEllipse(cx, cy, 22 * sc, 12 * sc); g.endFill();
      g.beginFill(face); g.drawEllipse(cx, cy, 17 * sc, 9 * sc); g.endFill();
      if (plate.dirty) { g.beginFill(0x6f5d3f, .6); g.drawCircle(cx - 5, cy, 2.5); g.drawCircle(cx + 4, cy + 2, 2); g.endFill(); return; }
      const cont = plate.contents || [];
      if (!cont.length) return;
      const allCooked = cont.every(c => c.state === "cooked"), same = cont.every(c => c.name === cont[0].name);
      if (allCooked && same && cont.length >= 2) {
        const col = this._mix(cont, "cooked");
        g.beginFill(col); g.drawEllipse(cx, cy - 1, 15 * sc, 7 * sc); g.endFill();
        g.beginFill(0xffffff, .2); g.drawEllipse(cx - 4, cy - 3, 5 * sc, 2 * sc); g.endFill();
      } else {
        const n = cont.length;
        cont.forEach((c, i) => this._ingredient(g, c.name, cx + (i - (n - 1) / 2) * 10 * sc, cy - 4 * sc, sc * 0.62, c.state));
      }
    }
    _ingredient(g, name, cx, cy, s, state) {
      const k = ING[name] || { c: 0xcccccc, c2: 0x999999 };
      let c = k.c, c2 = k.c2;
      if (state === "burnt") { c = 0x3a342f; c2 = 0x241f1c; }
      switch (name) {
        case "onion":
          g.beginFill(c); g.drawCircle(cx, cy, 12 * s); g.endFill();
          g.lineStyle(1, c2, .7); g.moveTo(cx - 7 * s, cy); g.quadraticCurveTo(cx, cy - 13 * s, cx + 7 * s, cy); g.lineStyle(0);
          g.beginFill(0x6cae3a); g.drawRect(cx - s, cy - 17 * s, 2 * s, 5 * s); g.endFill(); break;
        case "tomato":
          g.beginFill(c); g.drawCircle(cx, cy, 11 * s); g.endFill();
          g.beginFill(0x3f8f3a); g.drawCircle(cx, cy - 9 * s, 3 * s); g.endFill(); break;
        case "lettuce":
          g.beginFill(c2); g.drawCircle(cx, cy, 11 * s); g.endFill(); g.beginFill(c);
          for (let i = 0; i < 5; i++) { const a = i / 5 * Math.PI * 2; g.drawCircle(cx + Math.cos(a) * 5 * s, cy + Math.sin(a) * 5 * s, 5 * s); } g.endFill(); break;
        case "mushroom":
          g.beginFill(0xefe6d8); g.drawRoundedRect(cx - 4 * s, cy, 8 * s, 11 * s, 3 * s); g.endFill();
          g.beginFill(c); g.drawEllipse(cx, cy, 12 * s, 8 * s); g.endFill(); break;
        case "meat":
          g.beginFill(c); g.drawRoundedRect(cx - 12 * s, cy - 7 * s, 24 * s, 14 * s, 6 * s); g.endFill();
          if (state === "cooked") { g.lineStyle(1.4 * s, 0x3a241c, .6); g.moveTo(cx - 8 * s, cy - 3 * s); g.lineTo(cx + 8 * s, cy - 3 * s); g.moveTo(cx - 8 * s, cy + 3 * s); g.lineTo(cx + 8 * s, cy + 3 * s); g.lineStyle(0); } break;
        case "fish":
          g.beginFill(c); g.drawEllipse(cx - 2 * s, cy, 12 * s, 7 * s); g.endFill();
          g.beginFill(c); g.drawPolygon([cx + 9 * s, cy, cx + 16 * s, cy - 6 * s, cx + 16 * s, cy + 6 * s]); g.endFill();
          g.beginFill(0x2c3a44); g.drawCircle(cx - 8 * s, cy - s, 1.4 * s); g.endFill(); break;
        case "bun":
          g.beginFill(c); g.drawEllipse(cx, cy, 13 * s, 9 * s); g.endFill();
          g.beginFill(0xfff2d0); g.drawCircle(cx - 4 * s, cy - 2 * s, 1.1 * s); g.drawCircle(cx + 3 * s, cy - s, 1.1 * s); g.endFill(); break;
        case "cheese":
          g.beginFill(c); g.drawPolygon([cx - 11 * s, cy + 7 * s, cx + 11 * s, cy + 7 * s, cx, cy - 9 * s]); g.endFill();
          g.beginFill(c2); g.drawCircle(cx - 2 * s, cy + 2 * s, 1.6 * s); g.endFill(); break;
        case "rice":
          g.beginFill(c); g.drawEllipse(cx, cy + 1, 12 * s, 8 * s); g.endFill();
          g.beginFill(0xffffff); for (let i = 0; i < 6; i++) g.drawEllipse(cx + (i - 3) * 3 * s, cy + ((i % 2) - .5) * 5 * s, 2 * s, 1 * s); g.endFill(); break;
        case "dough":
          g.beginFill(c); g.drawEllipse(cx, cy, 12 * s, 9 * s); g.endFill();
          g.beginFill(c2, .5); g.drawCircle(cx - 3 * s, cy, 2 * s); g.drawCircle(cx + 4 * s, cy + 2 * s, 1.6 * s); g.endFill(); break;
        case "egg":
          g.beginFill(0xffffff); g.drawEllipse(cx, cy, 12 * s, 9 * s); g.endFill();
          g.beginFill(0xf6b73c); g.drawCircle(cx, cy, 4 * s); g.endFill(); break;
        case "potato":
          g.beginFill(c); g.drawEllipse(cx, cy, 12 * s, 9 * s); g.endFill();
          g.beginFill(c2, .6); g.drawCircle(cx - 4 * s, cy - 2 * s, 1.2 * s); g.drawCircle(cx + 3 * s, cy + 2 * s, 1 * s); g.endFill(); break;
        default: g.beginFill(c); g.drawCircle(cx, cy, 10 * s); g.endFill();
      }
      if (state === "chopped") { g.lineStyle(1, 0xffffff, .55); g.moveTo(cx - 8 * s, cy - 8 * s); g.lineTo(cx + 8 * s, cy + 8 * s); g.lineStyle(0); }
      if (state === "cooked") { g.beginFill(0xffd27f, .16); g.drawCircle(cx, cy, 13 * s); g.endFill(); }
    }

    // ---- cooks -----------------------------------------------------------
    _cook(g, p, v) {
      const [bx, by] = this.iso(v.x, v.y);
      const cy = by - CH * 0; // cooks stand on floor (counters they face are raised)
      const colors = [0x4a90d9, 0xe0533d, 0x52b35a, 0xd9a02b, 0x9b59b6, 0x16a59a];
      const body = colors[p.id % colors.length];
      const dir = p.dir_name;
      const fx = dir === "east" ? 1 : dir === "west" ? -1 : 0;
      const fy = dir === "south" ? 1 : dir === "north" ? -1 : 0;
      const oy = -2 + v.bob;
      g.beginFill(0x000000, 0.24); g.drawEllipse(bx, by + 6, 20, 8); g.endFill();
      // body
      g.beginFill(body); g.drawRoundedRect(bx - 14, cy - 24 + oy, 28, 30, 11); g.endFill();
      g.beginFill(shade(body, 1.18), .35); g.drawRoundedRect(bx - 11, cy - 22 + oy, 9, 26, 6); g.endFill();
      g.beginFill(0xffffff, .9); g.drawRoundedRect(bx - 9, cy - 16 + oy, 18, 22, 7); g.endFill();
      // arms
      g.beginFill(body);
      g.drawCircle(bx - 12 + fx * 6, cy - 6 + fy * 3 + oy, 5);
      g.drawCircle(bx + 12 + fx * 6, cy - 6 + fy * 3 + oy, 5); g.endFill();
      // head
      const hx = bx + fx * 3, hy = cy - 32 + oy;
      g.beginFill(0xf2c9a0); g.drawCircle(hx, hy, 11); g.endFill();
      g.beginFill(0xffffff);
      g.drawRoundedRect(hx - 11, hy - 9, 22, 7, 3);
      g.drawCircle(hx - 7, hy - 12, 6); g.drawCircle(hx, hy - 14, 7); g.drawCircle(hx + 7, hy - 12, 6); g.endFill();
      if (dir !== "north") { g.beginFill(0x2a2a2a); g.drawCircle(hx - 4 + fx * 3, hy + 1 + fy * 2, 1.7); g.drawCircle(hx + 4 + fx * 3, hy + 1 + fy * 2, 1.7); g.endFill(); }
      if (p.holding) this._item(g, bx + fx * 16, cy - 4 + fy * 12 + oy, p.holding, 0.92);
      const t = this._text(String(p.id + 1), 12, 0xffffff, true);
      t.position.set(bx, cy - 9 + oy); this.labels.addChild(t);
    }

    // ---- helpers / fx ----------------------------------------------------
    _ring(g, cx, cy, r, frac, color) {
      g.lineStyle(4, 0x000000, .28); g.arc(cx, cy, r, 0, Math.PI * 2); g.lineStyle(0);
      g.lineStyle(4, color, 1); g.arc(cx, cy, r, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * Math.max(.001, frac)); g.lineStyle(0);
    }
    _badge(g, cx, cy, color, txt, tc) {
      g.beginFill(0x000000, .3); g.drawCircle(cx + 1, cy + 1, 9); g.endFill();
      g.beginFill(color); g.drawCircle(cx, cy, 8.5); g.endFill();
      const t = this._text(txt, 11, tc || 0xffffff, true); t.position.set(cx, cy); this.labels.addChild(t);
    }
    _text(str, size, fill, bold) {
      const t = new PIXI.Text(str, { fontFamily: "Arial", fontSize: size, fontWeight: bold ? "800" : "400", fill });
      t.anchor.set(0.5); return t;
    }
    _emitDelivery() {
      for (const s of this.state.stations) if (s.type === "serving") {
        const [cx, cyy] = this.iso(s.x, s.y); const ty = cyy - CH - 14;
        for (let i = 0; i < 18; i++) { const a = Math.random() * Math.PI * 2, sp = 30 + Math.random() * 70; this.sparkles.push({ x: cx, y: ty, vx: Math.cos(a) * sp, vy: Math.sin(a) * sp - 20, life: 1 }); }
      }
    }
    _fx(g, dt) {
      for (const p of this.particles) { p.x += (Math.random() - .5) * 6 * dt; p.y += p.vy * dt; p.life -= dt * 1.1; }
      this.particles = this.particles.filter(p => p.life > 0);
      for (const p of this.particles) {
        let col = 0xffffff, a = p.life * 0.32;
        if (p.kind === "fire") { col = p.life > .6 ? 0xffd24a : 0xe2562a; a = p.life * 0.7; }
        g.beginFill(col, a); g.drawCircle(p.x, p.y, (1 - p.life) * 6 + 3); g.endFill();
      }
      for (const s of this.sparkles) { s.x += s.vx * dt; s.y += s.vy * dt; s.vy += 120 * dt; s.life -= dt * 1.6; }
      this.sparkles = this.sparkles.filter(s => s.life > 0);
      for (const s of this.sparkles) { g.beginFill(0xffe14a, Math.max(0, s.life)); g.drawStar ? g.drawStar(s.x, s.y, 4, 4, 2) : g.drawCircle(s.x, s.y, 3); g.endFill(); }
    }
    resize(W, H) { this.app.renderer.resize(W, H); this.fit(W, H); }
  }

  global.KitchenRenderer = KitchenRenderer;
})(window);
