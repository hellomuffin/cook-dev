/* CookSim true-3D renderer (Three.js, WebGL2).
 *
 * Renders the kitchen as a real 3D scene: extruded counters, modelled
 * stations, capsule chefs and food meshes under physically-based materials,
 * a key directional light with soft shadow maps, a hemisphere fill, ACES tone
 * mapping and Unreal bloom for glow (oven, sparks, progress rings). Cook motion
 * is interpolated between server ticks.
 *
 * Exposes the same view API as the old renderer (so game.js is unchanged):
 *   new KitchenRenderer(canvas) · setState(state) · update(dt) · resize(W,H)
 */
import * as THREE from "three";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

const CH = 0.62;           // counter height (world units, tile = 1)
const WH = 1.05;           // wall height

const ING = {
  onion:    { c: 0xe8c14a, c2: 0x9ec64a },
  tomato:   { c: 0xe1452f, c2: 0x3f8f3a },
  lettuce:  { c: 0x66c244, c2: 0x4e9a36 },
  mushroom: { c: 0xb79a82, c2: 0xefe6d8 },
  meat:     { c: 0xb0563f, c2: 0x6e3526 },
  fish:     { c: 0x9cbcd6, c2: 0x6f93b0 },
  bun:      { c: 0xd9a566, c2: 0xfff2d0 },
  cheese:   { c: 0xf2c94c, c2: 0xd0a728 },
  rice:     { c: 0xf6f1e6, c2: 0xe0d8c4 },
  dough:    { c: 0xeed9a8, c2: 0xd5bd86 },
  egg:      { c: 0xfaf3e2, c2: 0xf6b73c },
  potato:   { c: 0xcb9e60, c2: 0xa67d44 },
};
const COOK_COLORS = [0x4a90d9, 0xe0533d, 0x52b35a, 0xe0a52b, 0x9b59b6, 0x16a59a];
const DIR_YAW = { north: Math.PI, south: 0, east: Math.PI / 2, west: -Math.PI / 2 };
// facing offset in world (x, z): grid north = -z, south = +z, east = +x, west = -x
const DIR_VEC = { north: [0, -1], south: [0, 1], east: [1, 0], west: [-1, 0] };

function lerp(a, b, t) { return a + (b - a) * t; }
function mat(color, opts = {}) {
  return new THREE.MeshStandardMaterial(Object.assign({ color, roughness: 0.72, metalness: 0.0 }, opts));
}
function tintFor(state, base) {
  if (state === "burnt") return 0x140f0c;                 // clearly charred black
  if (state === "cooked") {                                // strongly browned/seared
    const c = new THREE.Color(base); c.lerp(new THREE.Color(0x5a2606), 0.8); return c.getHex();
  }
  return base;
}

class KitchenRenderer3D {
  constructor(view) {
    const r = new THREE.WebGLRenderer({ canvas: view, antialias: true, powerPreference: "high-performance", preserveDrawingBuffer: !!window.__capture });
    r.shadowMap.enabled = !window.__noshadow;   // disabled for GPU (ANGLE/Vulkan) headless recording
    r.shadowMap.type = THREE.PCFSoftShadowMap;
    r.outputColorSpace = THREE.SRGBColorSpace;
    r.toneMapping = THREE.ACESFilmicToneMapping;
    r.toneMappingExposure = 0.86;
    this.renderer = r;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x20222b);
    scene.fog = new THREE.Fog(0x20222b, 18, 38);
    this.scene = scene;

    // image-based lighting so metal/rough materials get real reflections
    const pmrem = new THREE.PMREMGenerator(r);
    scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

    this.camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);

    // lighting (IBL provides ambient, so direct lights stay modest)
    const hemi = new THREE.HemisphereLight(0xfff4e2, 0x2a2230, 0.28);
    scene.add(hemi);
    const key = new THREE.DirectionalLight(0xfff0d6, 1.7);
    key.position.set(6, 12, 5);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.bias = -0.0006;
    key.shadow.normalBias = 0.02;
    const sc = key.shadow.camera;
    sc.near = 1; sc.far = 50; sc.left = -14; sc.right = 14; sc.top = 14; sc.bottom = -14;
    this.key = key; scene.add(key); scene.add(key.target);
    const fill = new THREE.DirectionalLight(0x9fb6e0, 0.22);
    fill.position.set(-7, 6, -4); scene.add(fill);

    // postprocessing — bloom only on genuinely bright (emissive) things
    this.composer = new EffectComposer(r);
    this.composer.addPass(new RenderPass(scene, this.camera));
    this.bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.14, 0.5, 0.96);  // gentle — don't wash items into "magic light"
    this.composer.addPass(this.bloom);

    this.staticGroup = new THREE.Group(); scene.add(this.staticGroup);
    this.dynGroup = new THREE.Group(); scene.add(this.dynGroup);

    this.state = null; this._sig = ""; this.center = new THREE.Vector3();
    this.cooks = {}; this.stationDyn = {}; this.items = {}; this.particles = []; this.sprites = [];
    this._pulse = {};   // per-cook interact pulse, triggered once per state with action=interact
    this._flyers = [];  // transient meshes flying hand<->station to show transfers
    this._lastScore = 0; this._t = 0;
    this._steamTex = this._softTex(0xffffff);
    this._fireTex = this._softTex(0xffd27f);
    this._sparkTex = this._softTex(0xffe14a);
    this._ringTex = this._ringTexture();
    this._geo = {};   // shared geometry cache
  }

  // ---- shared geometry helpers ----------------------------------------
  g(key, make) { return this._geo[key] || (this._geo[key] = make()); }
  box(w, h, d) { return this.g(`b${w}_${h}_${d}`, () => new THREE.BoxGeometry(w, h, d)); }
  cyl(rt, rb, h, s = 20) { return this.g(`c${rt}_${rb}_${h}_${s}`, () => new THREE.CylinderGeometry(rt, rb, h, s)); }
  sph(rad, s = 16) { return this.g(`s${rad}_${s}`, () => new THREE.SphereGeometry(rad, s, s)); }

  iso() {}                       // (compat no-op)
  toWorld(x, y) { return new THREE.Vector3(x - this.center.x, 0, y - this.center.z); }

  // ---- camera ----------------------------------------------------------
  // A follow-camera keeps the cook large & centred so every action (hands,
  // station, held item, state change) is readable — far more legible than a
  // fit-the-whole-kitchen shot where the cook is a few pixels tall.
  fit(W, H) {
    if (!this.state) return;
    const w = this.state.width, h = this.state.height;
    this.center.set((w - 1) / 2, 0, (h - 1) / 2);
    const cam = this.camera;
    cam.aspect = W / H;
    // view radius: cook stays large & readable, but enough kitchen shows that
    // the stations it's working (and their progress rings) stay on screen
    this._viewR = 3.7;
    this._camDist = this._viewR / Math.tan((cam.fov * Math.PI / 180) / 2);
    cam.updateProjectionMatrix();
    if (!this._camFocus) this._camFocus = new THREE.Vector3(0, 0, 0);
  }

  _updateCamera(dt) {
    if (!this._camDist) return;
    let fx = 0, fz = 0, n = 0;
    for (const id in this.cooks) { const c = this.cooks[id]; const wp = this.toWorld(c.x, c.y); fx += wp.x; fz += wp.z; n++; }
    if (n) {
      const k = Math.min(1, dt * 3.5);
      this._camFocus.x += (fx / n - this._camFocus.x) * k;
      this._camFocus.z += (fz / n - this._camFocus.z) * k;
    }
    // clamp focus so the view never drifts off the kitchen into empty space
    const halfW = (this.state.width - 1) / 2, halfH = (this.state.height - 1) / 2;
    const mx = Math.max(0, halfW - (this._viewR - 2)), mz = Math.max(0, halfH - (this._viewR - 2));
    const f = this._camFocus;
    f.x = Math.max(-mx, Math.min(mx, f.x));
    f.z = Math.max(-mz, Math.min(mz, f.z));
    const d = this._camDist;
    this.camera.position.set(f.x, d * 0.82, f.z + d * 0.66);
    this.camera.lookAt(f.x, 0.3, f.z);
    this.key.target.position.set(f.x, 0, f.z);
  }

  setState(state) {
    this.state = state;
    const sig = state.width + "x" + state.height + ":" + JSON.stringify(state.terrain);
    if (sig !== this._sig) { this._sig = sig; this._buildStatic(); this._resetDyn(); }
    const dScore = state.score - (this._lastScore || 0);
    if (this._lastScore !== undefined && dScore > 0.5) this._delivery(Math.round(dScore));
    this._lastScore = state.score;
    const fails = (state.stats && state.stats.failed_deliveries) || 0;
    if (this._lastFails !== undefined && fails > this._lastFails) this._failFeedback();
    this._lastFails = fails;
    // each state where a cook is interacting fires one fresh action pulse
    for (const p of state.players) if (p.action === "interact") this._pulse[p.id] = true;
    this.fit(this.renderer.domElement.width / this.renderer.getPixelRatio(),
             this.renderer.domElement.height / this.renderer.getPixelRatio());
  }

  _resetDyn() {
    this.dynGroup.clear();
    this.cooks = {}; this.stationDyn = {}; this.items = {};
    for (const s of this.sprites) this.scene.remove(s.spr);
    this.sprites = []; this.particles = [];
  }

  // ---- static geometry -------------------------------------------------
  _buildStatic() {
    this.staticGroup.clear();
    const st = this.state, w = st.width, h = st.height;
    this.center.set((w - 1) / 2, 0, (h - 1) / 2);

    // floor (single textured plane)
    const tex = this._floorTex(w, h);
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(w, h),
      new THREE.MeshStandardMaterial({ map: tex, roughness: 0.92 }));
    floor.rotation.x = -Math.PI / 2;
    floor.position.set((w - 1) / 2 - this.center.x, 0, (h - 1) / 2 - this.center.z);
    floor.receiveShadow = true;
    this.staticGroup.add(floor);
    // skirting / base platform under everything
    const baseM = mat(0x2c2733, { roughness: 0.95 });
    const base = new THREE.Mesh(new THREE.BoxGeometry(w + 0.4, 0.3, h + 0.4), baseM);
    base.position.set((w - 1) / 2 - this.center.x, -0.16, (h - 1) / 2 - this.center.z);
    base.receiveShadow = true; this.staticGroup.add(base);

    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const t = st.terrain[y][x];
      if (t === "floor") continue;
      const p = this.toWorld(x, y);
      if (t === "wall") { this.staticGroup.add(this._wall(p)); continue; }
      this.staticGroup.add(this._counter(p));
      const fx = this._fixture(t, p);
      if (fx) this.staticGroup.add(fx);
    }
  }

  _counter(p) {
    const grp = new THREE.Group();
    if (!this._woodBody) {
      this._woodBody = new THREE.MeshStandardMaterial({ map: this._woodTex(0xa8794a, 0x8a6038), roughness: 0.62, metalness: 0.04 });
      this._woodTop = new THREE.MeshStandardMaterial({ map: this._woodTex(0xd9b683, 0xc09a64), roughness: 0.4, metalness: 0.04 });
    }
    const body = new THREE.Mesh(this.box(0.94, CH, 0.94), this._woodBody);
    body.position.set(p.x, CH / 2, p.z); body.castShadow = true; body.receiveShadow = true;
    const top = new THREE.Mesh(this.box(0.98, 0.1, 0.98), this._woodTop);
    top.position.set(p.x, CH + 0.02, p.z); top.castShadow = true; top.receiveShadow = true;
    grp.add(body, top); return grp;
  }
  _wall(p) {
    const m = new THREE.Mesh(this.box(0.98, WH, 0.98), mat(0x6f5c52, { roughness: 0.85 }));
    m.position.set(p.x, WH / 2, p.z); m.castShadow = true; m.receiveShadow = true; return m;
  }

  // station fixtures (static parts) -------------------------------------
  _fixture(t, p) {
    const grp = new THREE.Group(); grp.position.set(p.x, CH + 0.07, p.z);
    const top = 0;
    if (t === "counter") return null;
    if (t === "pot") {
      const pot = new THREE.Mesh(this.cyl(0.34, 0.3, 0.34, 24), mat(0xc2c7cc, { metalness: 0.85, roughness: 0.3 }));
      pot.position.y = 0.17; pot.castShadow = true; grp.add(pot);
      const rim = new THREE.Mesh(this.g("potrim", () => new THREE.TorusGeometry(0.34, 0.03, 10, 24)), mat(0xe6eaee, { metalness: 0.9, roughness: 0.25 }));
      rim.rotation.x = Math.PI / 2; rim.position.y = 0.34; grp.add(rim);
    } else if (t === "pan") {
      const pan = new THREE.Mesh(this.cyl(0.32, 0.3, 0.08, 24), mat(0x33333a, { metalness: 0.6, roughness: 0.4 }));
      pan.position.y = 0.06; pan.castShadow = true; grp.add(pan);
      const handle = new THREE.Mesh(this.box(0.45, 0.05, 0.08), mat(0x222228));
      handle.position.set(0.42, 0.06, 0); grp.add(handle);
    } else if (t === "oven") {
      const o = new THREE.Mesh(this.box(0.8, 0.6, 0.7), mat(0x44474e, { metalness: 0.5, roughness: 0.4 }));
      o.position.y = 0.3; o.castShadow = true; grp.add(o);
      const door = new THREE.Mesh(this.box(0.62, 0.4, 0.04), new THREE.MeshStandardMaterial({ color: 0x1a1a1f, emissive: 0xff7a1e, emissiveIntensity: 1.6, roughness: 0.4 }));
      door.position.set(0, 0.3, 0.36); grp.add(door);
    } else if (t === "cutting_board") {
      const b = new THREE.Mesh(this.box(0.6, 0.06, 0.42), mat(0xceA45e, { roughness: 0.7 }));
      b.position.y = 0.03; b.castShadow = true; grp.add(b);
      const knife = new THREE.Mesh(this.box(0.04, 0.02, 0.26), mat(0xd6dade, { metalness: 0.8, roughness: 0.3 }));
      knife.position.set(0.16, 0.08, 0.06); knife.rotation.y = 0.3; grp.add(knife);
    } else if (t === "sink") {
      const b = new THREE.Mesh(this.box(0.74, 0.18, 0.6), mat(0xb8bdc2, { metalness: 0.5, roughness: 0.35 }));
      b.position.y = 0.09; b.castShadow = true; grp.add(b);
      const basin = new THREE.Mesh(this.box(0.58, 0.1, 0.44), mat(0x4f555b, { roughness: 0.5 }));
      basin.position.y = 0.14; grp.add(basin);
      const fa = new THREE.Mesh(this.cyl(0.03, 0.03, 0.3, 8), mat(0xcfd4d8, { metalness: 0.8, roughness: 0.3 }));
      fa.position.set(0, 0.28, -0.2); grp.add(fa);
    } else if (t === "serving") {
      // a clear "PASS" hatch: a framed window above a glowing hand-off shelf
      const frameMat = mat(0x3a3f47, { roughness: 0.5, metalness: 0.3 });
      const left = new THREE.Mesh(this.box(0.12, 0.78, 0.5), frameMat); left.position.set(-0.42, 0.39, -0.1); left.castShadow = true; grp.add(left);
      const right = new THREE.Mesh(this.box(0.12, 0.78, 0.5), frameMat); right.position.set(0.42, 0.39, -0.1); right.castShadow = true; grp.add(right);
      const lintel = new THREE.Mesh(this.box(0.96, 0.16, 0.5), frameMat); lintel.position.set(0, 0.86, -0.1); lintel.castShadow = true; grp.add(lintel);
      // dark "kitchen behind" opening
      const opening = new THREE.Mesh(this.box(0.74, 0.6, 0.05), mat(0x14161b, { roughness: 0.9 }));
      opening.position.set(0, 0.5, -0.28); grp.add(opening);
      // glowing hand-off shelf at the front edge — where the plate goes
      const shelf = new THREE.Mesh(this.box(0.84, 0.06, 0.34),
        new THREE.MeshStandardMaterial({ color: 0x8be0b0, emissive: 0x36c07e, emissiveIntensity: 0.9, roughness: 0.4 }));
      shelf.position.set(0, 0.12, 0.16); shelf.castShadow = true; grp.add(shelf);
      // service bell on the shelf
      const bellBase = new THREE.Mesh(this.cyl(0.1, 0.12, 0.03, 16), mat(0xddd6c4)); bellBase.position.set(0.26, 0.17, 0.16); grp.add(bellBase);
      const bell = new THREE.Mesh(this.sph(0.08), mat(0xd9c24a, { metalness: 0.7, roughness: 0.3 })); bell.scale.y = 0.8; bell.position.set(0.26, 0.21, 0.16); grp.add(bell);
      // a downward chevron over the shelf marking "drop here"
      const chev = new THREE.Mesh(this.g("chev", () => new THREE.ConeGeometry(0.12, 0.16, 4)),
        new THREE.MeshStandardMaterial({ color: 0xffe07a, emissive: 0xffc34a, emissiveIntensity: 1.3 }));
      chev.rotation.x = Math.PI; chev.position.set(-0.12, 0.42, 0.12); grp.add(chev);
    } else if (t === "trash") {
      const bin = new THREE.Mesh(this.cyl(0.26, 0.22, 0.5, 16), mat(0x3a3d44, { roughness: 0.6 }));
      bin.position.y = 0.25; bin.castShadow = true; grp.add(bin);
    } else if (t === "plate_source") {
      for (let i = 0; i < 5; i++) {
        const pl = new THREE.Mesh(this.cyl(0.28, 0.28, 0.03, 20), mat(0xf2f4f6, { roughness: 0.3 }));
        pl.position.y = 0.02 + i * 0.035; grp.add(pl);
      }
    } else if (t.endsWith("_source")) {
      const crate = new THREE.Mesh(this.box(0.74, 0.4, 0.74), mat(0x7a5230, { roughness: 0.8 }));
      crate.position.y = 0.2; crate.castShadow = true; grp.add(crate);
      const ing = this._ingredient(t.replace("_source", ""), "raw"); ing.position.y = 0.5; ing.scale.setScalar(0.9); grp.add(ing);
    }
    grp.traverse(o => { if (o.isMesh) o.castShadow = true; });
    return grp;
  }

  // ---- ingredient / plate meshes --------------------------------------
  _ingredient(name, state) {
    const k = ING[name] || { c: 0xcccccc, c2: 0x999999 };
    const col = tintFor(state, k.c);
    const grp = new THREE.Group();
    const add = (geo, m, x = 0, y = 0, z = 0, sx, sy, sz) => {
      const mesh = new THREE.Mesh(geo, m); mesh.position.set(x, y, z);
      if (sx !== undefined) mesh.scale.set(sx, sy, sz); mesh.castShadow = true; grp.add(mesh); return mesh;
    };
    const M = mat(col, { roughness: 0.55 });
    switch (name) {
      case "onion": add(this.sph(0.16), M, 0, 0.16); add(this.cyl(0.0, 0.03, 0.1, 6), mat(0x6cae3a), 0, 0.32); break;
      case "tomato": add(this.sph(0.15), M, 0, 0.15); add(this.sph(0.05), mat(k.c2), 0, 0.29); break;
      case "lettuce": for (let i = 0; i < 5; i++) { const a = i / 5 * 6.28; add(this.sph(0.1), mat(i % 2 ? k.c : k.c2), Math.cos(a) * 0.08, 0.13, Math.sin(a) * 0.08); } break;
      case "mushroom": add(this.cyl(0.06, 0.07, 0.16, 10), mat(0xefe6d8), 0, 0.08); add(this.sph(0.15), M, 0, 0.18, 0, 1, 0.6, 1); break;
      case "meat": add(this.cyl(0.18, 0.18, 0.09, 16), M, 0, 0.09); break;
      case "fish": { const b = add(this.sph(0.15), M, 0, 0.12, 0, 1.5, 0.7, 0.8); add(this.cyl(0, 0.12, 0.02, 3), M, -0.26, 0.12, 0).rotation.z = Math.PI / 2; break; }
      case "bun": add(this.sph(0.17), M, 0, 0.12, 0, 1, 0.62, 1); break;
      case "cheese": { const m = add(this.cyl(0.0, 0.2, 0.16, 3), M, 0, 0.1); m.rotation.y = 0.5; break; }
      case "rice": add(this.cyl(0.16, 0.12, 0.12, 16), mat(tintFor(state, 0xf6f1e6), { roughness: 0.8 }), 0, 0.08); break;
      case "dough": add(this.sph(0.17), M, 0, 0.1, 0, 1, 0.55, 1); break;
      case "egg": add(this.sph(0.16), mat(tintFor(state, 0xfaf3e2)), 0, 0.07, 0, 1, 0.5, 1); add(this.sph(0.06), mat(state === "cooked" ? 0xf2a51e : 0xf6b73c), 0, 0.12); break;
      case "potato": add(this.sph(0.16), M, 0, 0.13, 0, 1.2, 0.85, 0.95); break;
      default: add(this.sph(0.15), M, 0, 0.15);
    }
    if (state === "chopped") {
      // chopped = a little pile of diced cubes, clearly different from raw
      grp.clear();
      const dice = mat(col, { roughness: 0.5 });
      const cube = this.g("dice", () => new THREE.BoxGeometry(0.09, 0.09, 0.09));
      const spots = [[-0.1, 0.05, -0.05], [0.08, 0.05, 0.04], [0, 0.05, 0.11], [-0.04, 0.05, 0.05], [0.11, 0.05, -0.06], [-0.12, 0.14, 0.03]];
      for (const [x, y, z] of spots) {
        const d = new THREE.Mesh(cube, dice); d.position.set(x, y, z);
        d.rotation.set(Math.sin(x * 9), x * 4 + z, Math.cos(z * 7)); d.castShadow = true; grp.add(d);
      }
    }
    if (state === "cooked") {
      // warm sear glow + glossier surface so even pale foods (rice/egg/potato)
      // read as unmistakably COOKED, not raw
      grp.traverse(o => {
        if (o.isMesh && o.material.emissive) {
          o.material = o.material.clone();
          o.material.emissive = new THREE.Color(0x5a2400); o.material.emissiveIntensity = 0.45;
          o.material.roughness = Math.max(0.2, (o.material.roughness ?? 0.5) - 0.25);
        }
      });
      // a darker browned "crust" cap sitting on top makes the state change obvious
      const crust = new THREE.Mesh(this.sph(0.13), new THREE.MeshStandardMaterial({ color: 0x3d1c08, roughness: 0.4, emissive: 0x2a1000, emissiveIntensity: 0.3 }));
      crust.scale.set(1, 0.32, 1); crust.position.y = 0.24; grp.add(crust);
    }
    return grp;
  }

  _isHotSoup(plate) {
    const c = plate.contents || [];
    return c.length >= 2 && c.every(i => i.state === "cooked") && c.every(i => i.name === c[0].name);
  }

  _plate(plate) {
    const grp = new THREE.Group();
    const dirty = plate.dirty;
    // a proper bowl: rim + concave interior, so plated food clearly sits in it
    const bowl = new THREE.Mesh(this.cyl(0.34, 0.22, 0.1, 28), mat(dirty ? 0x9c8e74 : 0xf3f5f7, { roughness: 0.28 }));
    bowl.position.y = 0.05; bowl.castShadow = true; grp.add(bowl);
    const rim = new THREE.Mesh(this.g("bowlrim", () => new THREE.TorusGeometry(0.32, 0.03, 10, 28)), mat(dirty ? 0x8a7c64 : 0xe6e9ec, { roughness: 0.3 }));
    rim.rotation.x = Math.PI / 2; rim.position.y = 0.1; grp.add(rim);
    if (!dirty && plate.contents && plate.contents.length) {
      // a shaded inner well so light foods (rice, egg, bun) clearly stand out
      const well = new THREE.Mesh(this.cyl(0.27, 0.27, 0.02, 24), mat(0xc4ccd6, { roughness: 0.5 }));
      well.position.y = 0.11; grp.add(well);
    }
    if (!dirty && plate.contents && plate.contents.length) {
      const cont = plate.contents;
      if (this._isHotSoup({ contents: cont })) {
        const col = ING[cont[0].name] ? tintFor("cooked", ING[cont[0].name].c) : 0xcc8844;
        // a generous domed serving of soup that fills the bowl
        const soup = new THREE.Mesh(this.cyl(0.3, 0.28, 0.1, 28),
          new THREE.MeshStandardMaterial({ color: col, roughness: 0.32, emissive: new THREE.Color(col).multiplyScalar(0.16) }));
        soup.position.y = 0.12; grp.add(soup);
        const dome = new THREE.Mesh(this.sph(0.27), new THREE.MeshStandardMaterial({ color: col, roughness: 0.32 }));
        dome.scale.y = 0.35; dome.position.y = 0.15; grp.add(dome);
      } else {
        // a clear, generous mound of the components sitting up in the bowl so a
        // finished dish is never mistaken for an empty plate
        const n = cont.length;
        cont.forEach((c, i) => {
          const m = this._ingredient(c.name, c.state); m.scale.setScalar(0.85);
          const ang = n > 1 ? (i / n) * 6.28 : 0, rr = n > 1 ? 0.13 : 0;
          m.position.set(Math.cos(ang) * rr, 0.2, Math.sin(ang) * rr); grp.add(m);
        });
      }
    }
    return grp;
  }
  _itemMesh(item) { return item.kind === "plate" ? this._plate(item) : this._ingredient(item.name, item.state); }
  _itemSig(item) { return item ? JSON.stringify(item) : "0"; }

  // ---- chef -----------------------------------------------------------
  _buildCook(p) {
    const grp = new THREE.Group();
    const col = COOK_COLORS[p.id % COOK_COLORS.length];
    const body = new THREE.Mesh(this.g("cookbody", () => new THREE.CapsuleGeometry(0.26, 0.34, 6, 16)), mat(col, { roughness: 0.6 }));
    body.position.y = 0.5; body.castShadow = true; grp.add(body);
    const apron = new THREE.Mesh(this.box(0.34, 0.4, 0.06), mat(0xf3f1ea, { roughness: 0.6 }));
    apron.position.set(0, 0.46, 0.22); grp.add(apron);
    const head = new THREE.Mesh(this.sph(0.18), mat(0xf2c9a0, { roughness: 0.6 }));
    head.position.y = 0.92; head.castShadow = true; grp.add(head);
    const hatB = new THREE.Mesh(this.cyl(0.19, 0.19, 0.12, 16), mat(0xffffff, { roughness: 0.5 }));
    hatB.position.y = 1.04; grp.add(hatB);
    const puff = new THREE.Mesh(this.sph(0.16), mat(0xffffff, { roughness: 0.5 }));
    puff.position.y = 1.16; grp.add(puff);
    // arms
    const arm = (sx) => { const a = new THREE.Mesh(this.g("arm", () => new THREE.CapsuleGeometry(0.07, 0.22, 4, 8)), mat(col, { roughness: 0.6 })); a.position.set(sx * 0.3, 0.5, 0.16); a.rotation.x = -0.7; grp.add(a); return a; };
    const aL = arm(-1), aR = arm(1);
    // legs
    const leg = (sx) => { const l = new THREE.Mesh(this.g("leg", () => new THREE.CapsuleGeometry(0.09, 0.18, 4, 8)), mat(0x33343c, { roughness: 0.7 })); l.position.set(sx * 0.12, 0.18, 0); grp.add(l); return l; };
    const lL = leg(-1), lR = leg(1);
    // eyes
    const eye = (sx) => { const e = new THREE.Mesh(this.sph(0.025, 8), mat(0x222228)); e.position.set(sx * 0.07, 0.93, 0.16); grp.add(e); };
    eye(-1); eye(1);
    // number label sprite
    const spr = this._textSprite(String(p.id + 1), col);
    spr.position.y = 1.45; spr.scale.set(0.4, 0.4, 1); grp.add(spr);
    // an upper-body pivot so the cook can visibly lean in to interact
    grp.userData = { legs: [lL, lR], arms: [aL, aR], torso: body, head, heldSig: "0", held: null, reach: 0, pop: 0 };
    this.dynGroup.add(grp);
    return grp;
  }

  // ---- dynamic update --------------------------------------------------
  update(dt) {
    this._t += dt;
    if (!this.state) { return; }
    const st = this.state;
    const seenItems = new Set();
    this._updateCamera(dt);

    // cooks
    const seenCooks = new Set();
    const interp = Math.min(1, dt * 12);
    for (const p of st.players) {
      seenCooks.add(p.id);
      let c = this.cooks[p.id];
      if (!c) { c = this.cooks[p.id] = { grp: this._buildCook(p), x: p.x, y: p.y, phase: Math.random() * 6 }; }
      c.x = lerp(c.x, p.x, interp); c.y = lerp(c.y, p.y, interp);
      const moving = Math.abs(c.x - p.x) > 0.02 || Math.abs(c.y - p.y) > 0.02;
      c.phase += dt * (moving ? 11 : 2.5);
      const bob = moving ? Math.abs(Math.sin(c.phase)) * 0.06 : Math.sin(c.phase) * 0.015;
      const wp = this.toWorld(c.x, c.y);
      c.grp.position.set(wp.x, bob, wp.z);
      c.grp.rotation.y = THREE.MathUtils.lerp(c.grp.rotation.y, DIR_YAW[p.dir_name] || 0, 0.3);
      const sw = moving ? Math.sin(c.phase) * 0.5 : 0;
      const ud = c.grp.userData;
      ud.legs[0].rotation.x = sw; ud.legs[1].rotation.x = -sw;

      // --- interaction animation: a strong, timed "work" pulse -------
      // Triggered once per state where the cook interacts (so even a single
      // pick-up/place/serve plays a full, visible motion), and re-triggered
      // every tick while chopping/cooking so it reads as repeated action.
      const facing = DIR_VEC[p.dir_name] || [0, 0];
      if (this._pulse[p.id]) {
        ud.actT = 0.5;                       // (re)start a half-second work pulse
        this._useBurst(wp.x + facing[0] * 0.7, wp.z + facing[1] * 0.7);
        delete this._pulse[p.id];
      }
      ud.actT = Math.max(0, (ud.actT || 0) - dt);
      // ramp up fast, ease down — two pumps over the pulse for a chop/grab feel
      const a = ud.actT / 0.5;                                   // 1 -> 0
      const env = Math.sin(Math.min(1, a) * Math.PI);            // 0..1..0 envelope
      const pump = Math.abs(Math.sin(a * Math.PI * 2));          // two strokes
      const act = env * (0.55 + 0.45 * pump);                    // strong overall

      ud.torso.rotation.x = act * 0.6;                           // big lean toward station
      ud.head.position.z = 0.02 + act * 0.14;
      // arms swing out and down toward the station (clear reaching/chopping)
      const armAng = -0.7 - act * 1.5;
      ud.arms[0].rotation.x = armAng; ud.arms[1].rotation.x = armAng;
      // the whole cook lunges toward the tile it faces, then settles back
      c.grp.position.set(wp.x + facing[0] * act * 0.22, bob, wp.z + facing[1] * act * 0.22);

      // held item
      const sig = this._itemSig(p.holding);
      if (sig !== ud.heldSig) {
        if (ud.held) c.grp.remove(ud.held);
        ud.held = null;
        if (p.holding) {
          const m = this._itemMesh(p.holding);
          c.grp.add(m); ud.held = m;
          ud.pop = 1;                                  // pop-in when just acquired
          // a quick puff at the faced tile so the grab reads as a deliberate
          // TAKE/transfer, not a teleport (round-14: pickups felt "magical")
          this._useBurst(wp.x + facing[0] * 0.5, wp.z + facing[1] * 0.5);
        }
        ud.heldSig = sig;
      }
      if (ud.held) {
        ud.pop = lerp(ud.pop, 0, Math.min(1, dt * 5));   // ~0.2s grab arc
        // hand position (with the work-thrust toward the station)
        const hy = 0.5 + act * 0.12, hz = 0.4 + act * 0.28;
        // when just acquired (pop~1) the item starts out at the faced station
        // and travels into the hands, so a viewer sees it being TAKEN
        ud.held.position.set(0, lerp(hy, 0.62, ud.pop), lerp(hz, 0.85, ud.pop));
        ud.held.scale.setScalar(0.9);
        // a hot, finished dish in hand visibly steams
        if (p.holding.kind === "plate" && (p.holding.contents || []).some(it => it.state === "cooked") && Math.random() < 0.2) {
          this._spawn(wp.x + facing[0] * 0.42, 0.95, wp.z + facing[1] * 0.42, "steam");
        }
      }
    }
    for (const id in this.cooks) if (!seenCooks.has(+id)) { this.dynGroup.remove(this.cooks[id].grp); delete this.cooks[id]; }

    // station dynamic content
    for (const s of st.stations) {
      const key = s.x + "," + s.y;
      const sig = JSON.stringify(s);
      let d = this.stationDyn[key];
      if (!d) d = this.stationDyn[key] = { grp: new THREE.Group(), sig: "" };
      if (d.sig !== sig) {
        d.grp.clear();
        this._stationContent(d.grp, s);
        const wp = this.toWorld(s.x, s.y); d.grp.position.set(wp.x, CH + 0.07, wp.z);
        if (!d.added) { this.dynGroup.add(d.grp); d.added = true; }
        d.sig = sig;
      }
      this._stationFx(s);
    }

    // loose counter items
    const present = new Set();
    for (const o of st.objects) {
      const key = o.x + "," + o.y; present.add(key);
      const sig = this._itemSig(o.item);
      let it = this.items[key];
      if (!it || it.sig !== sig) {
        if (it) this.dynGroup.remove(it.grp);
        const m = this._itemMesh(o.item); const wp = this.toWorld(o.x, o.y);
        m.position.set(wp.x, CH + 0.12, wp.z); this.dynGroup.add(m);
        this.items[key] = { grp: m, sig };
      }
    }
    for (const key in this.items) if (!present.has(key)) { this.dynGroup.remove(this.items[key].grp); delete this.items[key]; }

    this._updateParticles(dt);
    this._updateFlyers(dt);
    this.composer.render();
  }

  _stationContent(grp, s) {
    if ((s.type === "pot" || s.type === "oven") && s.contents && s.contents.length) {
      // show the actual ingredient pieces (what & how many is legible). The
      // oven is enclosed, so its contents sit on a visible tray on top of it.
      const baseY = s.type === "oven" ? 0.66 : 0.27;
      const n = s.contents.length;
      s.contents.forEach((ing, i) => {
        const m = this._ingredient(ing.name, ing.state); m.scale.setScalar(0.62);
        const ang = n > 1 ? i / n * 6.28 : 0, rr = n > 1 ? 0.11 : 0;
        m.position.set(Math.cos(ang) * rr, baseY + 0.04, Math.sin(ang) * rr); grp.add(m);
      });
      if (s.status === "cooking" || s.status === "cooked" || s.status === "burnt") {
        const col = this._mix(s.contents, s.status);
        const liq = new THREE.Mesh(this.cyl(0.27, 0.25, 0.04, 20),
          new THREE.MeshStandardMaterial({ color: col, roughness: 0.4, transparent: true, opacity: 0.5,
            emissive: new THREE.Color(col).multiplyScalar(s.status === "burnt" ? 0 : 0.12) }));
        liq.position.y = baseY - 0.04; grp.add(liq);
      }
    } else if (s.type === "pan" && s.contents && s.contents.length) {
      s.contents.forEach((ing, i) => { const m = this._ingredient(ing.name, ing.state); m.position.set((i - (s.contents.length - 1) / 2) * 0.18, 0.11, 0); m.scale.setScalar(0.92); grp.add(m); });
    } else if (s.type === "cutting_board" && s.item) {
      const m = this._ingredient(s.item.name, s.item.state); m.position.set(0, 0.1, 0); m.scale.setScalar(1.0); grp.add(m);
    } else if (s.type === "sink" && (s.dirty > 0 || s.clean_ready > 0)) {
      const water = new THREE.Mesh(this.box(0.56, 0.04, 0.42), new THREE.MeshStandardMaterial({ color: 0x6fc3e8, transparent: true, opacity: 0.85, roughness: 0.15, metalness: 0.3, emissive: 0x1d6f9c, emissiveIntensity: 0.3 }));
      water.position.y = 0.16; grp.add(water);
    }
    // progress ring — a dim full-circle track + a bright filled arc, on EVERY
    // station that is actively processing, so "how far along" is always clear
    let frac = -1, color = 0x66d98a;
    if (s.status === "cooking") frac = s.progress;
    else if (s.status === "cooked") frac = 1;
    else if (s.type === "cutting_board" && s.item && !s.done) { frac = s.progress; color = 0xe8c14a; }
    else if (s.type === "sink" && s.progress > 0) { frac = s.progress; color = 0x6fc3e8; }
    if (frac >= 0) {
      const yR = 1.05;   // well above the station & its contents
      const track = new THREE.Mesh(this.g("ringtrack", () => new THREE.TorusGeometry(0.36, 0.07, 10, 36)),
        new THREE.MeshStandardMaterial({ color: 0x1a1a1a, emissive: 0x0a0a0a, transparent: true, opacity: 0.7 }));
      track.rotation.x = -Math.PI / 2; track.position.y = yR; grp.add(track);
      // a bold UPRIGHT progress bar standing above the station — unmistakable
      // "how far along" from the angled follow-camera, on EVERY active station
      const BW = 0.62, BH = 0.18, byBase = 1.32;
      const btrack = new THREE.Mesh(this.box(BW + 0.06, BH + 0.06, 0.05), new THREE.MeshStandardMaterial({ color: 0x101216, emissive: 0x050608, transparent: true, opacity: 0.92 }));
      btrack.position.y = byBase; grp.add(btrack);
      const fillW = Math.max(0.001, BW * Math.min(1, frac));
      const fill = new THREE.Mesh(this.box(fillW, BH, 0.07), new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 2.0 }));
      fill.position.set(-BW / 2 + fillW / 2, byBase, 0.01); grp.add(fill);
      if (frac > 0.001) {
        const ring = new THREE.Mesh(new THREE.TorusGeometry(0.36, 0.1, 10, 36, Math.PI * 2 * frac),
          new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 1.8 }));
        ring.rotation.x = -Math.PI / 2; ring.position.y = yR; grp.add(ring);
        if (frac >= 1) {  // a clear "done" tick floats up when ready
          const done = new THREE.Mesh(this.sph(0.07), new THREE.MeshStandardMaterial({ color: 0x9af0bf, emissive: 0x4cd07a, emissiveIntensity: 1.5 }));
          done.position.y = yR; grp.add(done);
          const big = new THREE.Mesh(this.box(BW + 0.06, BH + 0.06, 0.08), new THREE.MeshStandardMaterial({ color: 0x9af0bf, emissive: 0x4cd07a, emissiveIntensity: 2.2 }));
          big.position.y = byBase; grp.add(big);  // bar flashes solid green when READY
        }
      }
    }
  }

  _stationFx(s) {
    const wp = this.toWorld(s.x, s.y);
    const top = CH + 0.5;
    if ((s.type === "pot" || s.type === "oven" || s.type === "pan") && s.status === "cooking" && Math.random() < 0.3)
      this._spawn(wp.x, top, wp.z, "steam");
    if (s.status === "cooked" && Math.random() < 0.4) this._spawn(wp.x, top, wp.z, "steam");  // cooked = clearly steaming
    if (s.status === "burnt" && Math.random() < 0.5) this._spawn(wp.x, top, wp.z, "fire");
  }

  _mix(contents, status) {
    // average the per-ingredient colours AT THEIR CURRENT STATE, so a cooked
    // soup looks browned (not raw-coloured) and a burnt one looks charred.
    let r = 0, g = 0, b = 0;
    for (const c of contents) {
      const base = (ING[c.name] || { c: 0xcccccc }).c;
      const col = tintFor(status === "burnt" ? "burnt" : c.state, base);
      r += (col >> 16) & 255; g += (col >> 8) & 255; b += col & 255;
    }
    const n = contents.length || 1;
    return (Math.round(r / n) << 16) | (Math.round(g / n) << 8) | Math.round(b / n);
  }

  // ---- particles (sprites) --------------------------------------------
  _spawn(x, y, z, kind) {
    const tex = kind === "fire" ? this._fireTex : this._steamTex;
    const spr = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false, blending: kind === "steam" ? THREE.NormalBlending : THREE.AdditiveBlending }));
    spr.position.set(x + (Math.random() - .5) * 0.3, y, z + (Math.random() - .5) * 0.3);
    spr.material.opacity = kind === "fire" ? 0.8 : 0.4;
    this.scene.add(spr);
    this.particles.push({ spr, life: 1, vy: kind === "fire" ? 1.2 : 0.8, kind });
  }
  _delivery(delta) {
    // sparkle burst at the serving pass...
    for (const s of this.state.stations) if (s.type === "serving") {
      const wp = this.toWorld(s.x, s.y);
      for (let i = 0; i < 18; i++) {
        const spr = new THREE.Sprite(new THREE.SpriteMaterial({ map: this._sparkTex, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }));
        spr.position.set(wp.x, CH + 0.7, wp.z); spr.scale.setScalar(0.2); this.scene.add(spr);
        const a = Math.random() * 6.28, sp = 1 + Math.random() * 2.2;
        this.particles.push({ spr, life: 1, vy: Math.sin(a) * sp * 0.4 + 1.6, vx: Math.cos(a) * sp * 0.6, vz: Math.sin(a) * sp * 0.6, kind: "spark", grav: -4 });
      }
    }
    // ...and a brief "ORDER DONE +N" banner anchored at the camera focus, so it
    // is in-frame regardless of where the pass is. Short-lived so it doesn't
    // linger over the cook (round-13: a stale banner obscured the character).
    const f = this._camFocus || new THREE.Vector3();
    this._popText(f.x, 1.8, f.z, "✓ +" + (delta || ""), "#5fe39a", 1.5, 2.4);
  }
  _failFeedback() {
    const f = this._camFocus || new THREE.Vector3();
    this._popText(f.x, 1.7, f.z, "✗ WRONG", "#ff6a5a", 2.0, 2.2);
  }
  _popText(x, y, z, text, color, life, sc) {
    sc = sc || 1.9;
    const C = document.createElement("canvas"); C.width = 512; C.height = 160;
    const ctx = C.getContext("2d");
    ctx.font = "bold 110px Arial"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.lineWidth = 14; ctx.strokeStyle = "rgba(0,0,0,.78)"; ctx.strokeText(text, 256, 80);
    ctx.fillStyle = color; ctx.fillText(text, 256, 80);
    const tex = new THREE.CanvasTexture(C); tex.colorSpace = THREE.SRGBColorSpace;
    const spr = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false }));
    spr.position.set(x, y, z); spr.scale.set(sc, sc * 0.31, 1); this.scene.add(spr);
    this.particles.push({ spr, life: life || 2.0, vy: 1.3, kind: "pop" });  // rises up & out quickly
  }
  // A small, brief puff at the worked tile — a subtle contact cue. The real
  // signal is the cook's lunge + the visibly changing item, NOT a glowing flash
  // (round-6: the old bright ring read as a "magic light" that hid the action).
  _useBurst(x, z) {
    const top = CH + 0.32;
    for (let i = 0; i < 3; i++) {
      const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: this._steamTex, transparent: true, depthWrite: false, blending: THREE.NormalBlending, color: 0xfff0d0 }));
      s.position.set(x + (Math.random() - .5) * 0.2, top, z + (Math.random() - .5) * 0.2);
      s.scale.setScalar(0.14); s.material.opacity = 0.45; this.scene.add(s);
      const a = Math.random() * 6.28;
      this.particles.push({ spr: s, life: 0.5, vx: Math.cos(a) * 0.4, vz: Math.sin(a) * 0.4, vy: 0.6, kind: "puff" });
    }
  }
  _flyItem(item, from, to) {
    if (!item) return;
    const m = item.kind === "plate" ? this._plate(item) : this._ingredient(item.name, item.state);
    m.scale.setScalar(0.8); m.position.set(from.x, from.y, from.z);
    this.scene.add(m);
    this._flyers.push({ m, from, to, t: 0, dur: 0.32 });
  }
  _updateFlyers(dt) {
    const alive = [];
    for (const f of this._flyers) {
      f.t += dt / f.dur;
      if (f.t >= 1) { this.scene.remove(f.m); continue; }
      const e = f.t;
      f.m.position.set(
        f.from.x + (f.to.x - f.from.x) * e,
        f.from.y + (f.to.y - f.from.y) * e + Math.sin(e * Math.PI) * 0.3,  // little arc
        f.from.z + (f.to.z - f.from.z) * e);
      f.m.scale.setScalar(0.8);
      alive.push(f);
    }
    this._flyers = alive;
  }
  _updateParticles(dt) {
    const alive = [];
    for (const p of this.particles) {
      p.life -= dt * (p.kind === "spark" ? 1.5 : p.kind === "ring" ? 2.6 : p.kind === "pop" ? 0.5 : 1.0);
      p.spr.position.y += (p.vy || 0) * dt;
      if (p.vx) p.spr.position.x += p.vx * dt;
      if (p.vz) p.spr.position.z += p.vz * dt;
      if (p.grav) p.vy += p.grav * dt;
      if (p.kind === "pop") {
        p.spr.material.opacity = Math.min(1, p.life * 1.6);    // float up & fade
      } else if (p.kind === "ring") {
        p.spr.scale.setScalar(0.3 + (1 - p.life) * 1.1);
        p.spr.material.opacity = Math.max(0, p.life) * 0.9;
      } else {
        const baseScale = p.kind === "spark" ? 0.16 : 0.3 + (1 - p.life) * 0.5;
        p.spr.scale.setScalar(baseScale);
        p.spr.material.opacity = Math.max(0, p.life) * (p.kind === "fire" ? 0.8 : p.kind === "spark" ? 1 : 0.4);
      }
      if (p.life > 0) alive.push(p); else this.scene.remove(p.spr);
    }
    this.particles = alive;
  }
  _ringTexture() {
    const C = document.createElement("canvas"); C.width = C.height = 64; const ctx = C.getContext("2d");
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 7; ctx.beginPath(); ctx.arc(32, 32, 24, 0, 6.28); ctx.stroke();
    const t = new THREE.CanvasTexture(C); t.colorSpace = THREE.SRGBColorSpace; return t;
  }

  // ---- textures --------------------------------------------------------
  _floorTex(w, h) {
    const C = document.createElement("canvas"); const px = 64; C.width = w * px; C.height = h * px;
    const ctx = C.getContext("2d");
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      ctx.fillStyle = ((x + y) % 2) ? "#b8ac90" : "#c4b89c";
      ctx.fillRect(x * px, y * px, px, px);
      ctx.strokeStyle = "rgba(120,105,82,.55)"; ctx.lineWidth = 2; ctx.strokeRect(x * px, y * px, px, px);
      // subtle per-tile speckle for texture
      ctx.fillStyle = "rgba(90,78,60,.06)";
      for (let k = 0; k < 18; k++) ctx.fillRect(x * px + Math.random() * px, y * px + Math.random() * px, 2, 2);
    }
    const t = new THREE.CanvasTexture(C); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 4; return t;
  }
  _woodTex(base, dark) {
    const C = document.createElement("canvas"); C.width = C.height = 128; const ctx = C.getContext("2d");
    const hex = (h) => "#" + h.toString(16).padStart(6, "0");
    ctx.fillStyle = hex(base); ctx.fillRect(0, 0, 128, 128);
    ctx.strokeStyle = hex(dark); ctx.globalAlpha = 0.35;
    for (let i = 0; i < 22; i++) {
      ctx.lineWidth = 1 + (i % 3); ctx.beginPath();
      const yy = (i / 22) * 128 + Math.sin(i) * 3;
      ctx.moveTo(0, yy);
      for (let x = 0; x <= 128; x += 16) ctx.lineTo(x, yy + Math.sin((x + i) * 0.12) * 4);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    const t = new THREE.CanvasTexture(C); t.colorSpace = THREE.SRGBColorSpace; return t;
  }
  _softTex(hex) {
    const C = document.createElement("canvas"); C.width = C.height = 64; const ctx = C.getContext("2d");
    const col = "#" + hex.toString(16).padStart(6, "0");
    const g = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
    g.addColorStop(0, col); g.addColorStop(0.4, col); g.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(32, 32, 32, 0, 6.28); ctx.fill();
    const t = new THREE.CanvasTexture(C); t.colorSpace = THREE.SRGBColorSpace; return t;
  }
  _textSprite(str, color) {
    const C = document.createElement("canvas"); C.width = C.height = 64; const ctx = C.getContext("2d");
    ctx.fillStyle = "#" + color.toString(16).padStart(6, "0"); ctx.beginPath(); ctx.arc(32, 32, 26, 0, 6.28); ctx.fill();
    ctx.strokeStyle = "rgba(255,255,255,.9)"; ctx.lineWidth = 4; ctx.stroke();
    ctx.fillStyle = "#fff"; ctx.font = "bold 38px Arial"; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(str, 32, 34);
    const t = new THREE.CanvasTexture(C); t.colorSpace = THREE.SRGBColorSpace;
    return new THREE.Sprite(new THREE.SpriteMaterial({ map: t, transparent: true, depthWrite: false }));
  }

  resize(W, H) {
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    this.renderer.setSize(W, H, false);
    this.composer.setSize(W, H);
    this.bloom.setSize(W, H);
    this.fit(W, H);
  }
}

window.KitchenRenderer = KitchenRenderer3D;
window.dispatchEvent(new Event("cooksim-renderer-ready"));
