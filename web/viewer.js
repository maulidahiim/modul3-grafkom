/* Module 3 -- real-time textured model in Three.js (Lab 3 deliverable).
 *
 * Everything that must MATCH raster3d.py comes from assets/scene.json, which
 * build_web.py writes from scene.py: camera, light, and the 4x4 model matrix.
 * The OBJ in assets/ is the mesh exactly as Python sees it (CCW-front).
 *
 * Matching notes (each one was a real mismatch before):
 *  - Light units: since r155 Three.js lights are physically based; Lambert is
 *    albedo * irradiance / PI. raster3d.py uses albedo * (ambient + (1-ambient)
 *    * max(0, n.l)), so both intensities are multiplied by PI here.
 *  - DataTexture defaults to NEAREST filtering; raster3d.py samples BILINEAR
 *    (no mipmap) -> LinearFilter, RepeatWrapping, no mips (toggle with a button).
 *  - UV origin: OBJ v=0 is the BOTTOM row; a PPM stores the TOP row first, so
 *    the rows are flipped once while parsing (flipY stays false).
 *  - FrontSide: CCW-front is the pinned convention; DoubleSide would hide a
 *    winding bug (inside-out model) instead of showing it.
 *
 * URL options:  ?aa=0  (no MSAA, for a closer pixel match)   ?assets=other/
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";

const params = new URLSearchParams(location.search);
const BASE = params.get("assets") || "assets/";
const AA = params.get("aa") !== "0";
const statusEl = document.getElementById("status");
const say = (msg, bad = false) => {
  statusEl.textContent = msg;
  statusEl.style.color = bad ? "#f66" : "";
};

// --- parse a binary P6 PPM into a DataTexture (Three.js has no PPM loader) ---
async function loadPPMTexture(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const buf = new Uint8Array(await res.arrayBuffer());
  let p = 0;
  const isSpace = (c) => c === 32 || c === 9 || c === 10 || c === 13;
  const tok = () => {
    for (;;) {
      while (isSpace(buf[p])) p++;
      if (buf[p] === 35) { while (buf[p] !== 10) p++; continue; }   // # comment
      break;
    }
    let s = "";
    while (!isSpace(buf[p])) s += String.fromCharCode(buf[p++]);
    return s;
  };
  if (tok() !== "P6") throw new Error("not a binary P6 PPM");
  const w = +tok(), h = +tok();
  if (+tok() !== 255) throw new Error("only maxval 255 supported");
  p++;                                           // exactly one whitespace byte
  const rgba = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y++) {
    const src = p + (h - 1 - y) * w * 3;         // flip rows: GL row 0 = bottom
    for (let x = 0; x < w; x++) {
      const s = src + x * 3, d = (y * w + x) * 4;
      rgba[d] = buf[s]; rgba[d + 1] = buf[s + 1]; rgba[d + 2] = buf[s + 2]; rgba[d + 3] = 255;
    }
  }
  const tex = new THREE.DataTexture(rgba, w, h, THREE.RGBAFormat);
  tex.flipY = false;
  return tex;
}

async function loadTexture(url) {
  const tex = url.toLowerCase().endsWith(".ppm")
    ? await loadPPMTexture(url)
    : await new THREE.TextureLoader().loadAsync(url);   // png/jpg: flipY=true
  tex.colorSpace = THREE.SRGBColorSpace;                // decode to linear for lighting
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearFilter;                   // = raster3d.py default
  tex.generateMipmaps = false;
  tex.needsUpdate = true;
  return tex;
}

const cfgRes = await fetch(BASE + "scene.json");
if (!cfgRes.ok) {
  say(`Could not load ${BASE}scene.json -- run "python build_web.py" first, then ` +
      `"npx serve web".`, true);
  throw new Error("scene.json missing");
}
const cfg = await cfgRes.json();
const W = cfg.width, H = cfg.height;
const cam = cfg.camera;

const canvas = document.getElementById("c");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: AA, preserveDrawingBuffer: true });
renderer.setPixelRatio(1);                    // otherwise a HiDPI screen exports 2x
renderer.setSize(W, H, false);
canvas.width = W; canvas.height = H;
renderer.outputColorSpace = THREE.SRGBColorSpace;   // = srgb.py linear_to_srgb

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);

const camera = new THREE.PerspectiveCamera(cam.fovy, W / H, cam.near, cam.far);
const controls = new OrbitControls(camera, canvas);
function resetCamera() {
  camera.position.set(...cam.eye);
  controls.target.set(...cam.center);
  controls.update();
}
resetCamera();

const amb = cfg.light.ambient;
scene.add(new THREE.AmbientLight(0xffffff, amb * Math.PI));
const sun = new THREE.DirectionalLight(0xffffff, (1 - amb) * Math.PI);
sun.position.set(...cfg.light.direction);     // shines from here toward the origin
scene.add(sun);

let model = null;
let material = null;
try {
  const [obj, tex] = await Promise.all([
    new OBJLoader().loadAsync(BASE + cfg.model),
    loadTexture(BASE + cfg.texture),
  ]);
  material = new THREE.MeshLambertMaterial({ map: tex, side: THREE.FrontSide });
  obj.traverse((c) => { if (c.isMesh) c.material = material; });
  obj.matrixAutoUpdate = false;               // use OUR matrix, not position/rotation/scale
  obj.matrix.fromArray(cfg.model_matrix_colmajor);
  obj.matrixWorldNeedsUpdate = true;
  scene.add(obj);
  model = obj;
  say(`${cfg.name}: ${cfg.triangles} triangles | three r${THREE.REVISION} | ` +
      `MSAA ${AA ? "on" : "off"} | ${navigator.userAgent}`);
} catch (e) {
  say(`Could not load model/texture: ${e}. Serve the web/ folder over HTTP ` +
      `(npx serve web), not file://`, true);
}

// --- frame-time / fps readout (Lab 3: "interactive frame rate") ---
const fpsEl = document.getElementById("fps");
let frames = 0, t0 = performance.now();
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
  frames++;
  const t = performance.now();
  if (t - t0 > 500) {
    fpsEl.textContent = `${(frames * 1000 / (t - t0)).toFixed(0)} fps`;
    frames = 0; t0 = t;
  }
}
animate();

document.getElementById("reset").addEventListener("click", resetCamera);

document.getElementById("wire").addEventListener("click", () => {
  if (material) material.wireframe = !material.wireframe;
});

const mipBtn = document.getElementById("mip");
mipBtn.addEventListener("click", () => {
  if (!material) return;
  const on = material.map.minFilter === THREE.LinearFilter;
  material.map.minFilter = on ? THREE.LinearMipmapLinearFilter : THREE.LinearFilter;
  material.map.generateMipmaps = on;
  material.map.needsUpdate = true;
  mipBtn.textContent = `Mipmap: ${on ? "on (trilinear)" : "off (bilinear)"}`;
});

document.getElementById("export").addEventListener("click", () => {
  renderer.render(scene, camera);
  const gl = renderer.getContext();
  const px = new Uint8Array(W * H * 4);
  gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, px);
  const body = new Uint8Array(W * H * 3);
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const s = ((H - 1 - y) * W + x) * 4, d = (y * W + x) * 3;   // GL rows are bottom-up
      body[d] = px[s]; body[d + 1] = px[s + 1]; body[d + 2] = px[s + 2];
    }
  const header = new TextEncoder().encode(`P6\n${W} ${H}\n255\n`);
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([header, body]));
  a.download = `${cfg.name}_three.ppm`;
  a.click();
});
