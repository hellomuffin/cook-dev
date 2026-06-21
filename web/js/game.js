/* CookSim live client: WebSocket networking, keyboard control, HUD + UI. */
(function () {
  "use strict";

  const ACT = { N: 0, S: 1, E: 2, W: 3, INTERACT: 4, STAY: 5 };

  // Two local control schemes so two people can co-op on one keyboard.
  const SCHEME_A = { ArrowUp: ACT.N, ArrowDown: ACT.S, ArrowRight: ACT.E, ArrowLeft: ACT.W, " ": ACT.INTERACT, Enter: ACT.INTERACT };
  const SCHEME_B = { w: ACT.N, s: ACT.S, d: ACT.E, a: ACT.W, f: ACT.INTERACT, Shift: ACT.INTERACT };

  const canvas = document.getElementById("stage");
  const renderer = new window.KitchenRenderer(canvas);

  let ws = null;
  let recipes = [];
  let layouts = [];
  let lastState = null;
  let controlled = { 0: SCHEME_A, 1: SCHEME_B }; // player -> scheme
  const held = { 0: new Set(), 1: new Set() };
  const curAction = { 0: ACT.STAY, 1: ACT.STAY };
  let resultShown = false;
  let curLayout = "cramped_room";
  let sandbox = false;

  // ---- networking --------------------------------------------------------
  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.onopen = () => setStatus("connected", true);
    ws.onclose = () => { setStatus("disconnected — retrying", false); setTimeout(connect, 1200); };
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === "init") {
        recipes = msg.recipes;
        layouts = msg.layouts;
        populateLayouts(msg.layouts);
        renderRecipeBook();
        buildLevelGrid();
        onState(msg.state);
        if (!maybePlayCustom()) showMenu();
      } else if (msg.type === "state") {
        onState(msg.state);
      }
    };
  }

  function send(obj) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj)); }

  function maybePlayCustom() {
    const raw = localStorage.getItem("cooksim_play_layout");
    if (raw) {
      localStorage.removeItem("cooksim_play_layout");
      try {
        const data = JSON.parse(raw);
        curLayout = data.name || "custom";
        resultShown = false; sandbox = false;
        controlled = { 0: SCHEME_A, 1: SCHEME_B };
        send({ type: "reset", layout_data: data, seed: 0 });
        return true;
      } catch (e) {}
    }
    return false;
  }

  // ---- level select + results overlay ----------------------------------
  const overlay = document.getElementById("overlay");
  function showMenu() {
    send({ type: "set_speed", tps: 0 });
    document.getElementById("menuView").classList.remove("hidden");
    document.getElementById("resultView").classList.add("hidden");
    buildLevelGrid();
    overlay.classList.remove("hidden");
  }
  function hideOverlay() { overlay.classList.add("hidden"); }

  function bestKey(l) { return "cooksim_best_" + l; }
  function bestScore(l) { return parseInt(localStorage.getItem(bestKey(l)) || "0", 10); }

  function buildLevelGrid() {
    const grid = document.getElementById("levelGrid");
    if (!grid) return;
    grid.innerHTML = "";
    for (const l of layouts) {
      const card = document.createElement("div");
      card.className = "level-card";
      const best = bestScore(l);
      card.innerHTML = `<div class="lc-name">${l.replace(/_/g, " ")}</div>
        <div class="lc-sub">kitchen</div>
        <div class="lc-best">${best ? "Best " + best : "&nbsp;"}</div>`;
      card.onclick = () => startLevel(l);
      grid.appendChild(card);
    }
  }

  function startLevel(layout, custom) {
    curLayout = layout;
    resultShown = false;
    const sel = document.getElementById("layoutSel");
    if (sel && [...sel.options].some(o => o.value === layout)) sel.value = layout;
    document.getElementById("nplayers").value = document.getElementById("menuPlayers").value;
    sandbox = document.getElementById("menuSandbox").checked;
    const n = parseInt(document.getElementById("menuPlayers").value, 10);
    const addBots = document.getElementById("menuBots").checked;
    controlled = { 0: SCHEME_A, 1: SCHEME_B };
    const cfg = sandbox ? { horizon: 1000000, orders_enabled: true } : {};
    send({ type: "reset", layout, n_players: n, seed: 0, config: cfg });
    if (addBots) {
      // bot every seat except the human-controlled cook 0 (and cook 1 if 2+ humans not desired)
      for (let i = (n >= 2 ? 1 : 0); i < n; i++) { send({ type: "add_bot", player: i, kind: "greedy" }); delete controlled[i]; }
    }
    send({ type: "set_speed", tps: parseFloat(document.getElementById("speed").value) || 7 });
    hideOverlay();
  }

  function showResults(state) {
    send({ type: "set_speed", tps: 0 });
    const score = Math.round(state.score);
    const served = (state.stats && state.stats.deliveries) || 0;
    const prevBest = bestScore(curLayout);
    if (score > prevBest) localStorage.setItem(bestKey(curLayout), String(score));
    const stars = score >= 250 ? 3 : score >= 120 ? 2 : score >= 30 ? 1 : 0;
    document.getElementById("stars").innerHTML =
      [0, 1, 2].map(i => `<span class="${i < stars ? "on" : ""}">★</span>`).join("");
    document.getElementById("rScore").textContent = score;
    document.getElementById("rServed").textContent = served;
    document.getElementById("rBest").textContent = Math.max(prevBest, score);
    document.getElementById("resultTitle").textContent =
      stars === 3 ? "Michelin star service! 🌟" : stars >= 1 ? "Service complete!" : "Kitchen's closed…";
    document.getElementById("menuView").classList.add("hidden");
    document.getElementById("resultView").classList.remove("hidden");
    overlay.classList.remove("hidden");
  }

  document.getElementById("menuBtn").onclick = showMenu;
  document.getElementById("menuRandom").onclick = () => {
    resultShown = false; sandbox = document.getElementById("menuSandbox").checked;
    const n = parseInt(document.getElementById("menuPlayers").value, 10);
    curLayout = "random";
    send({ type: "reset", procedural: { width: 11, height: 7, n_players: n, n_pots: 2, n_pans: 1, n_ovens: 1, n_cutting_boards: 1, include_sink: true, style: "ring" }, seed: Math.floor(Math.random() * 99999) });
    if (document.getElementById("menuBots").checked) for (let i = (n >= 2 ? 1 : 0); i < n; i++) { send({ type: "add_bot", player: i, kind: "greedy" }); delete controlled[i]; }
    send({ type: "set_speed", tps: parseFloat(document.getElementById("speed").value) || 7 });
    hideOverlay();
  };
  document.getElementById("rAgain").onclick = () => startLevel(curLayout);
  document.getElementById("rMenu").onclick = showMenu;

  // ---- state -> HUD ------------------------------------------------------
  function onState(state) {
    lastState = state;
    renderer.setState(state);
    document.getElementById("score").textContent = Math.round(state.score);
    document.getElementById("tick").textContent = state.tick;
    document.getElementById("delivered").textContent = (state.stats && state.stats.deliveries) || 0;
    renderOrders(state.orders);
    renderSeats(state);
    document.getElementById("layoutTitle").textContent = state.layout_name;
    if (state.done && !sandbox && !resultShown) { resultShown = true; showResults(state); }
  }

  function renderOrders(orders) {
    const box = document.getElementById("orders");
    box.innerHTML = "";
    for (const o of orders) {
      const frac = o.time_left / o.time_total;
      const card = document.createElement("div");
      card.className = "order" + (frac < 0.25 ? " urgent" : "");
      card.style.borderColor = o.color;
      const items = o.contents.map(c => `<span class="chip" title="${c.state}">${iconFor(c.name)}${c.state === "chopped" ? "▰" : c.state === "cooked" ? "♨" : ""}</span>`).join("");
      const steps = (o.steps && o.steps.length) ? `<div class="order-steps">${o.steps.map((s, i) => `<span>${i + 1}. ${s}</span>`).join("")}</div>` : "";
      card.innerHTML = `
        <div class="order-name" style="color:${o.color}">${o.name} <b>+${o.reward}</b></div>
        <div class="order-items">${items}</div>${steps}
        <div class="timer"><div class="bar" style="width:${(frac * 100).toFixed(0)}%;background:${frac < 0.25 ? "#e0533d" : o.color}"></div></div>`;
      box.appendChild(card);
    }
  }

  const ICON = { onion: "🧅", tomato: "🍅", lettuce: "🥬", mushroom: "🍄", meat: "🥩", fish: "🐟", bun: "🍞", cheese: "🧀", rice: "🍚", dough: "🥟", egg: "🥚", potato: "🥔", plate: "🍽️" };
  function iconFor(n) { return ICON[n] || "•"; }

  function renderRecipeBook() {
    const box = document.getElementById("recipeBook");
    if (!box) return;
    box.innerHTML = recipes.map(r => {
      const items = r.contents.map(c => `${iconFor(c.name)}${c.state === "chopped" ? "▰" : c.state === "cooked" ? "♨" : ""}`).join(" ");
      return `<div class="recipe"><span style="color:${r.color}">${r.name}</span><span class="ritems">${items}</span><b>+${r.reward}</b></div>`;
    }).join("");
  }

  function renderSeats(state) {
    const box = document.getElementById("seats");
    box.innerHTML = "";
    const bots = new Set(state.bots || []);
    for (let i = 0; i < state.n_players; i++) {
      const row = document.createElement("div");
      row.className = "seat";
      const isBot = bots.has(i);
      const ctrl = controlled[i] ? (i === 0 ? "Arrows + Space" : "WASD + F") : (isBot ? "GreedyChef 🤖" : "—");
      row.innerHTML = `<span class="pname p${i % 6}">Cook ${i + 1}</span>
        <span class="pctrl">${ctrl}</span>
        <button data-p="${i}" class="botbtn">${isBot ? "Remove bot" : "Add bot"}</button>`;
      row.querySelector("button").onclick = () => {
        if (isBot) { send({ type: "remove_bot", player: i }); }
        else { send({ type: "add_bot", player: i, kind: "greedy" }); delete controlled[i]; }
      };
      box.appendChild(row);
    }
  }

  function setStatus(text, ok) {
    const el = document.getElementById("status");
    el.textContent = text; el.className = ok ? "ok" : "bad";
  }

  // ---- controls ----------------------------------------------------------
  function populateLayouts(layouts) {
    const sel = document.getElementById("layoutSel");
    sel.innerHTML = layouts.map(n => `<option value="${n}">${n}</option>`).join("");
  }

  function doReset() {
    const layout = document.getElementById("layoutSel").value;
    const n = parseInt(document.getElementById("nplayers").value, 10);
    const seed = parseInt(document.getElementById("seed").value, 10) || 0;
    controlled = { 0: SCHEME_A, 1: SCHEME_B };
    curLayout = layout; resultShown = false;
    send({ type: "reset", layout, n_players: n, seed });
  }

  document.getElementById("resetBtn").onclick = doReset;
  document.getElementById("layoutSel").onchange = doReset;
  document.getElementById("genBtn").onclick = () => {
    const style = document.getElementById("genStyle").value;
    const seed = Math.floor(Math.random() * 100000);
    const n = parseInt(document.getElementById("nplayers").value, 10);
    send({ type: "reset", procedural: { width: 11, height: 7, n_players: n, n_pots: 2, n_cutting_boards: 1, include_sink: true, style }, seed });
  };
  const speed = document.getElementById("speed");
  speed.oninput = () => {
    document.getElementById("speedVal").textContent = speed.value;
    send({ type: "set_speed", tps: parseFloat(speed.value) });
  };
  document.getElementById("addAllBots").onclick = () => {
    if (!lastState) return;
    for (let i = 0; i < lastState.n_players; i++) { send({ type: "add_bot", player: i, kind: "greedy" }); delete controlled[i]; }
  };

  // ---- keyboard ----------------------------------------------------------
  function recompute(player) {
    const scheme = controlled[player];
    if (!scheme) return;
    const down = held[player];
    let act = ACT.STAY;
    // interact takes priority, then most recent movement
    for (const k of down) if (scheme[k] === ACT.INTERACT) act = ACT.INTERACT;
    if (act !== ACT.INTERACT) {
      for (const k of down) if (scheme[k] !== undefined && scheme[k] !== ACT.INTERACT) act = scheme[k];
    }
    if (act !== curAction[player]) { curAction[player] = act; send({ type: "action", player, action: act }); }
  }

  function keyToPlayer(key) {
    for (const p of Object.keys(controlled)) {
      if (controlled[p][key] !== undefined) return parseInt(p, 10);
    }
    return null;
  }

  window.addEventListener("keydown", (e) => {
    const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
    const p = keyToPlayer(key);
    if (p === null) return;
    e.preventDefault();
    held[p].add(key); recompute(p);
  });
  window.addEventListener("keyup", (e) => {
    const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
    const p = keyToPlayer(key);
    if (p === null) return;
    held[p].delete(key); recompute(p);
  });

  // ---- animation loop ----------------------------------------------------
  let last = performance.now();
  function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000); last = now;
    renderer.update(dt);
    requestAnimationFrame(frame);
  }
  function resize() {
    const wrap = document.getElementById("stageWrap");
    renderer.resize(wrap.clientWidth, wrap.clientHeight);
  }
  window.addEventListener("resize", resize);
  setTimeout(resize, 30);
  requestAnimationFrame(frame);
  connect();
})();
