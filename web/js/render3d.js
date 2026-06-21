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
const DIR_YAW = { north: Math.PI, south: 0, east: -Math.PI / 2, west: Math.PI / 2 };

function lerp(a, b, t) { return a + (b - a) * t; }
function mat(color, opts = {}) {
  return new THREE.MeshStandardMaterial(Object.assign({ color, roughness: 0.72, metalness: 0.0 }, opts));
}
function tintFor(state, base) {
  if (state === "burnt") return 0x33302c;
  if (state === "cooked") {
    const c = new THREE.Color(base); c.offsetHSL(0, 0.05, -0.06); return c.getHex();
  }
  return base;
}

class KitchenRenderer3D {
  constructor(view) {
    const r = new THREE.WebGLRenderer({ canvas: view, antialias: true, powerPreference: "high-performance" });
    r.shadowMap.enabled = true;
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
    this.bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.42, 0.6, 0.95);
    this.composer.addPass(this.bloom);

    this.staticGroup = new THREE.Group(); scene.add(this.staticGroup);
    this.dynGroup = new THREE.Group(); scene.add(this.dynGroup);

    this.state = null; this._sig = ""; this.center = new THREE.Vector3();
    this.cooks = {}; this.stationDyn = {}; this.items = {}; this.particles = []; this.sprites = [];
    this._lastScore = 0; this._t = 0;
    this._steamTex = this._softTex(0xffffff);
    this._fireTex = this._softTex(0xffd27f);
    this._sparkTex = this._softTex(0xffe14a);
    this._geo = {};   // shared geometry cache
  }

  // ---- shared geometry helpers ----------------------------------------
  g(key, make) { return this._geo[key] || (this._geo[key] = make()); }
  box(w, h, d) { return this.g(`b${w}_${h}_${d}`, () => new THREE.BoxGeometry(w, h, d)); }
  cyl(rt, rb, h, s = 20) { return this.g(`c${rt}_${rb}_${h}_${s}`, () => new THREE.CylinderGeometry(rt, rb, h, s)); }
  sph(rad, s = 16) { return this.g(`s${rad}_${s}`, () => new THREE.SphereGeometry(rad, s, s)); }

  iso() {}                       // (compat no-op)
  toWorld(x, y) { return new THREE.Vector3(x - this.center.x, 0, y - this.center.z); }

  // ---- camera fit ------------------------------------------------------
  fit(W, H) {
    if (!this.state) return;
    const w = this.state.width, h = this.state.height;
    this.center.set((w - 1) / 2, 0, (h - 1) / 2);
    const radius = Math.max(w, h) * 0.62 + 1.4;
    const cam = this.camera;
    cam.aspect = W / H;
    const dist = radius / Math.tan((cam.fov * Math.PI / 180) / 2) * (W > H ? 0.58 : 0.86);
    cam.position.set(0, dist * 0.8, dist * 0.64);
    cam.lookAt(0, -0.5, 0.3);
    cam.updateProjectionMatrix();
    this.key.target.position.set(0, 0, 0);
  }

  setState(state) {
    this.state = state;
    const sig = state.width + "x" + state.height + ":" + JSON.stringify(state.terrain);
    if (sig !== this._sig) { this._sig = sig; this._buildStatic(); this._resetDyn(); }
    if (this._lastScore && state.score > this._lastScore + 0.5) this._delivery();
    this._lastScore = state.score;
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
      const back = new THREE.Mesh(this.box(0.84, 0.7, 0.16), mat(0x2f7d57, { roughness: 0.55 }));
      back.position.set(0, 0.35, -0.34); back.castShadow = true; grp.add(back);
      const strip = new THREE.Mesh(this.box(0.84, 0.1, 0.06), new THREE.MeshStandardMaterial({ color: 0x49b07d, emissive: 0x49b07d, emissiveIntensity: 1.1 }));
      strip.position.set(0, 0.66, -0.27); grp.add(strip);
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
      case "rice": add(this.cyl(0.16, 0.12, 0.12, 16), mat(0xf6f1e6, { roughness: 0.8 }), 0, 0.08); break;
      case "dough": add(this.sph(0.17), M, 0, 0.1, 0, 1, 0.55, 1); break;
      case "egg": add(this.sph(0.16), mat(0xfaf3e2), 0, 0.07, 0, 1, 0.5, 1); add(this.sph(0.06), mat(0xf6b73c), 0, 0.12); break;
      case "potato": add(this.sph(0.16), M, 0, 0.13, 0, 1.2, 0.85, 0.95); break;
      default: add(this.sph(0.15), M, 0, 0.15);
    }
    if (state === "chopped") { grp.scale.setScalar(0.82); grp.children.forEach((c, i) => { c.position.x += (i - 1) * 0.05; }); }
    if (state === "cooked") grp.traverse(o => { if (o.isMesh && o.material.emissive) { o.material = o.material.clone(); o.material.emissive = new THREE.Color(0x3a1d00); o.material.emissiveIntensity = 0.25; } });
    return grp;
  }

  _plate(plate) {
    const grp = new THREE.Group();
    const dirty = plate.dirty;
    const disc = new THREE.Mesh(this.cyl(0.32, 0.3, 0.04, 24), mat(dirty ? 0x9c8e74 : 0xf3f5f7, { roughness: 0.3 }));
    disc.position.y = 0.02; disc.castShadow = true; grp.add(disc);
    if (!dirty && plate.contents && plate.contents.length) {
      const cont = plate.contents, allCooked = cont.every(c => c.state === "cooked"), same = cont.every(c => c.name === cont[0].name);
      if (allCooked && same && cont.length >= 2) {
        const col = ING[cont[0].name] ? tintFor("cooked", ING[cont[0].name].c) : 0xcc8844;
        const soup = new THREE.Mesh(this.cyl(0.26, 0.26, 0.06, 24), new THREE.MeshStandardMaterial({ color: col, roughness: 0.35, emissive: new THREE.Color(col).multiplyScalar(0.12) }));
        soup.position.y = 0.06; grp.add(soup);
      } else {
        const n = cont.length;
        cont.forEach((c, i) => { const m = this._ingredient(c.name, c.state); m.scale.setScalar(0.6); m.position.set((i - (n - 1) / 2) * 0.22, 0.04, 0); grp.add(m); });
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
    // contact shadow blob
    grp.userData = { legs: [lL, lR], arms: [aL, aR], heldSig: "0", held: null };
    this.dynGroup.add(grp);
    return grp;
  }

  // ---- dynamic update --------------------------------------------------
  update(dt) {
    this._t += dt;
    if (!this.state) { return; }
    const st = this.state;
    const seenItems = new Set();

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
      c.grp.userData.legs[0].rotation.x = sw; c.grp.userData.legs[1].rotation.x = -sw;
      // held item
      const sig = this._itemSig(p.holding);
      if (sig !== c.grp.userData.heldSig) {
        if (c.grp.userData.held) c.grp.remove(c.grp.userData.held);
        c.grp.userData.held = null;
        if (p.holding) { const m = this._itemMesh(p.holding); m.position.set(0, 0.5, 0.4); m.scale.setScalar(0.9); c.grp.add(m); c.grp.userData.held = m; }
        c.grp.userData.heldSig = sig;
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
    this.composer.render();
  }

  _stationContent(grp, s) {
    if ((s.type === "pot" || s.type === "oven") && s.contents && s.contents.length) {
      const col = this._mix(s.contents, s.status);
      const liq = new THREE.Mesh(this.cyl(0.28, 0.26, 0.1, 20), new THREE.MeshStandardMaterial({ color: col, roughness: 0.4, emissive: new THREE.Color(col).multiplyScalar(s.status === "burnt" ? 0 : 0.15) }));
      liq.position.y = s.type === "oven" ? 0.16 : 0.28; grp.add(liq);
    } else if (s.type === "pan" && s.contents && s.contents.length) {
      const m = this._ingredient(s.contents[0].name, s.contents[0].state); m.position.y = 0.08; m.scale.setScalar(0.85); grp.add(m);
    } else if (s.type === "cutting_board" && s.item) {
      const m = this._ingredient(s.item.name, s.item.state); m.position.set(0, 0.06, 0); m.scale.setScalar(0.8); grp.add(m);
    } else if (s.type === "sink" && (s.dirty > 0 || s.clean_ready > 0)) {
      const water = new THREE.Mesh(this.box(0.56, 0.04, 0.42), new THREE.MeshStandardMaterial({ color: 0x6fc3e8, transparent: true, opacity: 0.85, roughness: 0.15, metalness: 0.3, emissive: 0x1d6f9c, emissiveIntensity: 0.3 }));
      water.position.y = 0.16; grp.add(water);
    }
    // progress ring (emissive torus arc)
    let frac = 0, color = 0x66d98a;
    if (s.status === "cooking") frac = s.progress;
    else if (s.status === "cooked") { frac = 1; }
    else if (s.type === "cutting_board" && s.item && !s.done) { frac = s.progress; color = 0xe8c14a; }
    else if (s.type === "sink" && s.progress > 0) { frac = s.progress; color = 0x6fc3e8; }
    if (frac > 0.001) {
      const ring = new THREE.Mesh(new THREE.TorusGeometry(0.26, 0.04, 8, 28, Math.PI * 2 * frac),
        new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 1.4 }));
      ring.rotation.x = -Math.PI / 2; ring.position.y = 0.7; grp.add(ring);
    }
  }

  _stationFx(s) {
    const wp = this.toWorld(s.x, s.y);
    const top = CH + 0.5;
    if ((s.type === "pot" || s.type === "oven") && s.status === "cooking" && Math.random() < 0.25)
      this._spawn(wp.x, top, wp.z, "steam");
    if (s.status === "cooked" && Math.random() < 0.16) this._spawn(wp.x, top, wp.z, "steam");
    if (s.status === "burnt" && Math.random() < 0.4) this._spawn(wp.x, top, wp.z, "fire");
  }

  _mix(contents, status) {
    let r = 0, g = 0, b = 0; for (const c of contents) { const k = ING[c.name] || { c: 0xcccccc }; r += (k.c >> 16) & 255; g += (k.c >> 8) & 255; b += k.c & 255; }
    const n = contents.length || 1; r /= n; g /= n; b /= n; if (status === "burnt") { r *= .3; g *= .28; b *= .28; }
    return (Math.round(r) << 16) | (Math.round(g) << 8) | Math.round(b);
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
  _delivery() {
    for (const s of this.state.stations) if (s.type === "serving") {
      const wp = this.toWorld(s.x, s.y);
      for (let i = 0; i < 14; i++) {
        const spr = new THREE.Sprite(new THREE.SpriteMaterial({ map: this._sparkTex, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }));
        spr.position.set(wp.x, CH + 0.8, wp.z); spr.scale.setScalar(0.18); this.scene.add(spr);
        const a = Math.random() * 6.28, sp = 1 + Math.random() * 2;
        this.particles.push({ spr, life: 1, vy: Math.sin(a) * sp * 0.4 + 1.5, vx: Math.cos(a) * sp * 0.5, vz: Math.sin(a) * sp * 0.5, kind: "spark", grav: -4 });
      }
    }
  }
  _updateParticles(dt) {
    const alive = [];
    for (const p of this.particles) {
      p.life -= dt * (p.kind === "spark" ? 1.5 : 1.0);
      p.spr.position.y += p.vy * dt;
      if (p.vx) p.spr.position.x += p.vx * dt;
      if (p.vz) p.spr.position.z += p.vz * dt;
      if (p.grav) p.vy += p.grav * dt;
      const baseScale = p.kind === "spark" ? 0.18 : 0.3 + (1 - p.life) * 0.5;
      p.spr.scale.setScalar(baseScale);
      p.spr.material.opacity = Math.max(0, p.life) * (p.kind === "fire" ? 0.8 : p.kind === "spark" ? 1 : 0.4);
      if (p.life > 0) alive.push(p); else this.scene.remove(p.spr);
    }
    this.particles = alive;
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
