/* CookSim Level Editor — vanilla JS, no build step. */
(function () {
  "use strict";

  // ---- Terrain definitions -------------------------------------------------
  // type, label, emoji, group, and whether it renders as a counter background.
  const TERRAINS = {
    floor:           { label: "Floor",        icon: "",    group: "basics",    counter: false },
    counter:         { label: "Counter",      icon: "🪵",  group: "basics",    counter: true  },
    wall:            { label: "Wall",         icon: "🧱",  group: "basics",    counter: false },
    onion_source:    { label: "Onion",        icon: "🧅",  group: "sources",   counter: true  },
    tomato_source:   { label: "Tomato",       icon: "🍅",  group: "sources",   counter: true  },
    lettuce_source:  { label: "Lettuce",      icon: "🥬",  group: "sources",   counter: true  },
    mushroom_source: { label: "Mushroom",     icon: "🍄",  group: "sources",   counter: true  },
    meat_source:     { label: "Meat",         icon: "🥩",  group: "sources",   counter: true  },
    fish_source:     { label: "Fish",         icon: "🐟",  group: "sources",   counter: true  },
    bun_source:      { label: "Bun",          icon: "🍞",  group: "sources",   counter: true  },
    cheese_source:   { label: "Cheese",       icon: "🧀",  group: "sources",   counter: true  },
    rice_source:     { label: "Rice",         icon: "🍚",  group: "sources",   counter: true  },
    dough_source:    { label: "Dough",        icon: "🥟",  group: "sources",   counter: true  },
    egg_source:      { label: "Egg",          icon: "🥚",  group: "sources",   counter: true  },
    potato_source:   { label: "Potato",       icon: "🥔",  group: "sources",   counter: true  },
    plate_source:    { label: "Plates",       icon: "🍽️",  group: "sources",   counter: true  },
    pot:             { label: "Pot",          icon: "🍲",  group: "stations",  counter: true  },
    pan:             { label: "Pan",          icon: "🍳",  group: "stations",  counter: true  },
    oven:            { label: "Oven",         icon: "🔥",  group: "stations",  counter: true  },
    cutting_board:   { label: "Cutting board",icon: "🔪",  group: "stations",  counter: true  },
    sink:            { label: "Sink",         icon: "🚰",  group: "stations",  counter: true  },
    serving:         { label: "Serving",      icon: "🛎️",  group: "terminals", counter: true  },
    trash:           { label: "Trash",        icon: "🗑️",  group: "terminals", counter: true  }
  };
  const ORDER = Object.keys(TERRAINS);

  // ---- State ---------------------------------------------------------------
  const state = {
    name: "untitled",
    width: 9,
    height: 7,
    grid: [],                 // grid[y][x] = terrain string
    spawns: [],               // [[x,y], ...]
    brush: "counter",         // active terrain
    tool: "paint",            // "paint" | "spawn" | "fill"
    painting: false,
    paintErase: false
  };

  // ---- DOM refs ------------------------------------------------------------
  const el = {};
  function $(id) { return document.getElementById(id); }

  // ---- Grid helpers --------------------------------------------------------
  function makeGrid(w, h, fill) {
    const g = [];
    for (let y = 0; y < h; y++) {
      const row = [];
      for (let x = 0; x < w; x++) row.push(fill || "floor");
      g.push(row);
    }
    return g;
  }

  function inBounds(x, y) {
    return x >= 0 && y >= 0 && x < state.width && y < state.height;
  }

  function spawnIndexAt(x, y) {
    for (let i = 0; i < state.spawns.length; i++) {
      if (state.spawns[i][0] === x && state.spawns[i][1] === y) return i;
    }
    return -1;
  }

  // ---- Rendering -----------------------------------------------------------
  function render() {
    const grid = el.grid;
    grid.style.gridTemplateColumns = "repeat(" + state.width + ", var(--cell))";
    grid.innerHTML = "";
    for (let y = 0; y < state.height; y++) {
      for (let x = 0; x < state.width; x++) {
        const t = state.grid[y][x];
        const def = TERRAINS[t] || TERRAINS.floor;
        const cell = document.createElement("div");
        cell.className = "cell t-" + t + (def.counter ? " is-counter" : "");
        cell.dataset.x = x;
        cell.dataset.y = y;
        if (def.icon) {
          const ic = document.createElement("span");
          ic.className = "icon";
          ic.textContent = def.icon;
          cell.appendChild(ic);
        }
        const si = spawnIndexAt(x, y);
        if (si >= 0) {
          const sp = document.createElement("div");
          sp.className = "spawn";
          sp.textContent = String(si + 1);
          cell.appendChild(sp);
        }
        grid.appendChild(cell);
      }
    }
    el.name.value = state.name;
    el.inpW.value = state.width;
    el.inpH.value = state.height;
  }

  // ---- Cell editing --------------------------------------------------------
  function applyBrush(x, y, erase) {
    if (!inBounds(x, y)) return;
    if (state.tool === "spawn") {
      toggleSpawn(x, y);
      return;
    }
    if (state.tool === "fill") {
      floodFill(x, y, erase ? "floor" : state.brush);
      return;
    }
    // paint
    const target = erase ? "floor" : state.brush;
    if (state.grid[y][x] === target) return;
    state.grid[y][x] = target;
    // if a non-floor terrain lands on a spawn, drop that spawn
    if (target !== "floor") {
      const si = spawnIndexAt(x, y);
      if (si >= 0) state.spawns.splice(si, 1);
    }
    render();
  }

  function toggleSpawn(x, y) {
    if (state.grid[y][x] !== "floor") {
      flashBanner(false, ["Spawn points can only be placed on floor tiles."]);
      return;
    }
    const si = spawnIndexAt(x, y);
    if (si >= 0) state.spawns.splice(si, 1);
    else state.spawns.push([x, y]);
    render();
  }

  function floodFill(x, y, replacement) {
    const startT = state.grid[y][x];
    if (startT === replacement) return;
    const stack = [[x, y]];
    while (stack.length) {
      const [cx, cy] = stack.pop();
      if (!inBounds(cx, cy)) continue;
      if (state.grid[cy][cx] !== startT) continue;
      state.grid[cy][cx] = replacement;
      if (replacement !== "floor") {
        const si = spawnIndexAt(cx, cy);
        if (si >= 0) state.spawns.splice(si, 1);
      }
      stack.push([cx + 1, cy], [cx - 1, cy], [cx, cy + 1], [cx, cy - 1]);
    }
    render();
  }

  // ---- Resize --------------------------------------------------------------
  function resize(w, h) {
    w = clamp(w, 5, 20); h = clamp(h, 5, 20);
    const ng = makeGrid(w, h, "floor");
    for (let y = 0; y < Math.min(h, state.height); y++)
      for (let x = 0; x < Math.min(w, state.width); x++)
        ng[y][x] = state.grid[y][x];
    state.width = w; state.height = h; state.grid = ng;
    state.spawns = state.spawns.filter(function (s) {
      return s[0] >= 0 && s[1] >= 0 && s[0] < w && s[1] < h;
    });
    render();
  }

  function clamp(v, lo, hi) { v = parseInt(v, 10); if (isNaN(v)) v = lo; return Math.max(lo, Math.min(hi, v)); }

  // ---- Layout (de)serialization -------------------------------------------
  function toLayout() {
    return {
      name: el.name.value || "untitled",
      grid: state.grid.map(function (r) { return r.slice(); }),
      start_positions: state.spawns.map(function (s) { return [s[0], s[1]]; })
    };
  }

  function loadLayout(layout) {
    if (!layout || !Array.isArray(layout.grid) || !layout.grid.length) {
      flashBanner(false, ["Invalid layout: missing grid."]);
      return;
    }
    const h = layout.grid.length;
    const w = layout.grid[0].length;
    const g = makeGrid(w, h, "floor");
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < (layout.grid[y] ? layout.grid[y].length : 0); x++) {
        const t = layout.grid[y][x];
        g[y][x] = TERRAINS[t] ? t : "floor";
      }
    }
    state.width = w;
    state.height = h;
    state.grid = g;
    state.name = layout.name || "untitled";
    state.spawns = (layout.start_positions || [])
      .filter(function (s) { return Array.isArray(s) && s.length >= 2 && s[0] < w && s[1] < h && s[0] >= 0 && s[1] >= 0; })
      .map(function (s) { return [s[0], s[1]]; });
    render();
    clearBanner();
  }

  // ---- Banner --------------------------------------------------------------
  function flashBanner(ok, issues) {
    const b = el.banner;
    b.className = "banner show " + (ok ? "ok" : "err");
    if (ok) {
      b.textContent = "Valid ✓ — this layout is ready to play.";
    } else {
      b.innerHTML = "<b>" + (issues.length ? "Issues found:" : "Invalid layout.") + "</b>" +
        (issues.length ? "<ul>" + issues.map(function (i) { return "<li>" + escapeHtml(i) + "</li>"; }).join("") + "</ul>" : "");
    }
  }
  function clearBanner() { el.banner.className = "banner"; el.banner.textContent = ""; }
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  // ---- Local (client-side) validation fallback -----------------------------
  function localValidate(layout) {
    const issues = [];
    if (!layout.grid || !layout.grid.length) issues.push("Grid is empty.");
    const w = layout.grid[0].length;
    layout.grid.forEach(function (row, y) {
      if (row.length !== w) issues.push("Row " + y + " has inconsistent width.");
      row.forEach(function (t) { if (!TERRAINS[t]) issues.push("Unknown terrain '" + t + "'."); });
    });
    if (!layout.start_positions || layout.start_positions.length < 1)
      issues.push("Need at least one start position.");
    (layout.start_positions || []).forEach(function (s, i) {
      const x = s[0], y = s[1];
      if (!layout.grid[y] || layout.grid[y][x] === undefined) issues.push("Spawn " + (i + 1) + " is out of bounds.");
      else if (layout.grid[y][x] !== "floor") issues.push("Spawn " + (i + 1) + " is not on a floor tile.");
    });
    const flat = layout.grid.flat();
    if (flat.indexOf("serving") < 0) issues.push("No serving station — dishes cannot be delivered.");
    return { valid: issues.length === 0, issues: issues };
  }

  // ---- API calls -----------------------------------------------------------
  function api(method, url, body) {
    const opts = { method: method, headers: {} };
    if (body !== undefined) {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    }
    return fetch(url, opts).then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    });
  }

  function loadLayoutList() {
    api("GET", "/api/layouts").then(function (data) {
      const sel = el.layoutSelect;
      sel.innerHTML = '<option value="">— choose —</option>';
      (data.layouts || []).forEach(function (n) {
        const o = document.createElement("option");
        o.value = n; o.textContent = n;
        sel.appendChild(o);
      });
    }).catch(function () {
      // server not running — leave list empty quietly
    });
  }

  function validate() {
    const layout = toLayout();
    api("POST", "/api/validate", layout).then(function (res) {
      flashBanner(!!res.valid, res.issues || []);
    }).catch(function () {
      const res = localValidate(layout);
      flashBanner(res.valid, res.issues);
    });
  }

  function generate() {
    const params = {
      width: clamp(el.genW.value, 5, 20),
      height: clamp(el.genH.value, 5, 20),
      n_players: clamp(el.genPlayers.value, 1, 4),
      n_pots: Math.max(0, parseInt(el.genPots.value, 10) || 0),
      n_cutting_boards: Math.max(0, parseInt(el.genBoards.value, 10) || 0),
      include_sink: el.genSink.checked,
      style: el.genStyle.value,
      seed: Math.max(0, parseInt(el.genSeed.value, 10) || 0)
    };
    api("POST", "/api/generate", params).then(function (layout) {
      loadLayout(layout);
    }).catch(function () {
      flashBanner(false, ["Could not reach /api/generate (is the server running?)."]);
    });
  }

  // ---- Export / Import -----------------------------------------------------
  function exportJSON() {
    const layout = toLayout();
    const text = JSON.stringify(layout, null, 2);
    el.ioArea.value = text;
    const blob = new Blob([text], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = (layout.name || "layout") + ".json";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  function copyJSON() {
    const text = JSON.stringify(toLayout(), null, 2);
    el.ioArea.value = text;
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(function () {
        flashTransient(el.btnCopy, "Copied!");
      }, function () {});
    }
  }

  function importJSON() {
    let layout;
    try { layout = JSON.parse(el.ioArea.value); }
    catch (e) { flashBanner(false, ["Could not parse JSON: " + e.message]); return; }
    loadLayout(layout);
  }

  function flashTransient(btn, txt) {
    const old = btn.textContent;
    btn.textContent = txt;
    setTimeout(function () { btn.textContent = old; }, 1200);
  }

  // ---- Play handoff --------------------------------------------------------
  // Validates, then stows the layout in localStorage["cooksim_play_layout"]
  // and navigates to "/". The game page reads that key on load.
  const PLAY_KEY = "cooksim_play_layout";
  function play() {
    const layout = toLayout();
    function go() {
      try { localStorage.setItem(PLAY_KEY, JSON.stringify(layout)); } catch (e) {}
      window.location.href = "/";
    }
    api("POST", "/api/validate", layout).then(function (res) {
      if (res.valid) { go(); }
      else { flashBanner(false, (res.issues || []).concat(["Fix the issues above before playing."])); }
    }).catch(function () {
      const res = localValidate(layout);
      if (res.valid) { go(); }
      else { flashBanner(false, res.issues.concat(["Fix the issues above before playing."])); }
    });
  }

  // ---- Palette UI ----------------------------------------------------------
  function chipStyle(t) {
    const def = TERRAINS[t];
    if (t === "floor") return "background:var(--floor);";
    if (t === "wall") return "background:var(--wall);color:#fff;";
    return "background:linear-gradient(180deg,#c0975c,var(--counter));";
  }

  function makeSwatch(t) {
    const def = TERRAINS[t];
    const s = document.createElement("div");
    s.className = "swatch";
    s.dataset.terrain = t;
    s.innerHTML =
      '<span class="chip" style="' + chipStyle(t) + '">' + (def.icon || "") + "</span>" +
      '<span class="lbl">' + def.label + "</span>";
    s.addEventListener("click", function () {
      state.tool = "paint";
      state.brush = t;
      refreshSelection();
    });
    return s;
  }

  function makeToolSwatch(tool, icon, label) {
    const s = document.createElement("div");
    s.className = "swatch";
    s.dataset.tool = tool;
    s.innerHTML = '<span class="chip" style="background:#fff;">' + icon + '</span><span class="lbl">' + label + "</span>";
    s.addEventListener("click", function () { state.tool = tool; refreshSelection(); });
    return s;
  }

  function buildPalette() {
    el.palBasics.innerHTML = "";
    el.palSources.innerHTML = "";
    el.palStations.innerHTML = "";
    el.palTerminals.innerHTML = "";
    const map = { basics: el.palBasics, sources: el.palSources, stations: el.palStations, terminals: el.palTerminals };
    ORDER.forEach(function (t) {
      map[TERRAINS[t].group].appendChild(makeSwatch(t));
    });
    el.toolRow.innerHTML = "";
    el.toolRow.appendChild(makeToolSwatch("spawn", "🧑‍🍳", "Spawn point"));
    el.toolRow.appendChild(makeToolSwatch("fill", "🪣", "Fill"));
    refreshSelection();
  }

  function refreshSelection() {
    document.querySelectorAll(".swatch").forEach(function (s) {
      let on = false;
      if (s.dataset.tool) on = (state.tool === s.dataset.tool);
      else if (s.dataset.terrain) on = (state.tool === "paint" || state.tool === "fill") && s.dataset.terrain === state.brush;
      s.classList.toggle("active", on);
    });
  }

  // ---- Pointer painting ----------------------------------------------------
  function cellFromEvent(e) {
    let node = e.target;
    while (node && !node.classList.contains("cell")) node = node.parentElement;
    if (!node) return null;
    return { x: parseInt(node.dataset.x, 10), y: parseInt(node.dataset.y, 10) };
  }

  function bindGrid() {
    const g = el.grid;
    g.addEventListener("contextmenu", function (e) { e.preventDefault(); });
    g.addEventListener("pointerdown", function (e) {
      const c = cellFromEvent(e);
      if (!c) return;
      e.preventDefault();
      state.painting = true;
      state.paintErase = (e.button === 2);
      applyBrush(c.x, c.y, state.paintErase);
    });
    g.addEventListener("pointermove", function (e) {
      if (!state.painting) return;
      if (state.tool === "fill" || state.tool === "spawn") return; // drag only for paint
      const c = cellFromEvent(e);
      if (!c) return;
      applyBrush(c.x, c.y, state.paintErase);
    });
    window.addEventListener("pointerup", function () { state.painting = false; });
  }

  // ---- Wiring --------------------------------------------------------------
  function init() {
    el.grid = $("grid");
    el.name = $("layoutName");
    el.banner = $("banner");
    el.layoutSelect = $("layoutSelect");
    el.inpW = $("inpW"); el.inpH = $("inpH");
    el.palBasics = $("palBasics"); el.palSources = $("palSources");
    el.palStations = $("palStations"); el.palTerminals = $("palTerminals");
    el.toolRow = $("toolRow");
    el.genW = $("genW"); el.genH = $("genH"); el.genPlayers = $("genPlayers");
    el.genPots = $("genPots"); el.genBoards = $("genBoards"); el.genStyle = $("genStyle");
    el.genSeed = $("genSeed"); el.genSink = $("genSink");
    el.ioArea = $("ioArea"); el.btnCopy = $("btnCopy");

    state.grid = makeGrid(state.width, state.height, "floor");
    // default starter spawn
    state.spawns = [[1, 1]];

    buildPalette();
    bindGrid();
    render();

    $("btnValidate").addEventListener("click", validate);
    $("btnPlay").addEventListener("click", play);
    $("btnApplySize").addEventListener("click", function () {
      resize(el.inpW.value, el.inpH.value);
    });
    $("btnClear").addEventListener("click", function () {
      state.grid = makeGrid(state.width, state.height, "floor");
      state.spawns = [];
      render();
    });
    $("btnGenerate").addEventListener("click", generate);
    $("btnExport").addEventListener("click", exportJSON);
    $("btnCopy").addEventListener("click", copyJSON);
    $("btnImport").addEventListener("click", importJSON);
    $("btnReloadLayouts").addEventListener("click", loadLayoutList);
    el.name.addEventListener("input", function () { state.name = el.name.value; });
    el.layoutSelect.addEventListener("change", function () {
      const n = el.layoutSelect.value;
      if (!n) return;
      api("GET", "/api/layout/" + encodeURIComponent(n)).then(loadLayout).catch(function () {
        flashBanner(false, ["Could not load layout '" + n + "'."]);
      });
    });

    loadLayoutList();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
