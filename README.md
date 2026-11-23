# Module 3 -- 3D Transforms, Projection, Texture & Real-Time

Praktikum Grafika Komputer, S1 Informatika UM Surabaya -- Modul 3 (dasar UTS).

**Rule:** the transform stack, the z-buffer rasteriser and the ray tracer are
**NumPy only**. Pillow is optional (reference cross-checks + `convert_texture.py`).
The real-time renderer (`web/`) is Three.js and is the GitHub Pages deliverable.

**Conventions** (pinned at the top of `transforms.py`, never mixed): column
vectors, pre-multiply (`p' = M @ p`), chains right-to-left; right-handed;
camera looks down `-z`; NDC `[-1,1]^3`, near at `z=-1`; viewport flips `y`;
**CCW = front face** (outward normals, positive signed volume); UV origin
bottom-left (OBJ), flipped once in `objio.load_obj`; texel centres at
`(i+0.5)/W` (same as GL_LINEAR + GL_REPEAT). One scene (camera, light, model
matrix) for every renderer: `scene.py`.

## Quick start (clean clone)

```bash
python -m pip install -r requirements.txt
python make_model.py                 # data/pusaka_model.obj (180 tris) + data/relief.ppm
python tests.py                      # -> ALL CHECKS PASSED (13 ok), each with its numbers
python worked_example.py             # the running point (2,3,4) through every space
python build_web.py                  # web/assets/ (model + texture + scene.json)
npx serve web                        # open the printed URL (or: python -m http.server -d web 8000)
```

## One command per task

| Goal | Command | Lab / report |
|---|---|---|
| model + texture (fallback) | `python make_model.py` | Case Study |
| every from-scratch check | `python tests.py` | A2, A6 |
| running point through every space | `python worked_example.py` | Step 1 |
| trace a vertex of **your** model + read the columns of M, V, pose | `python worked_example.py --model data/mymodel.obj --vertex 0` | Lab 1.3-1.4, A1 |
| rasterise from scratch (+ depth image, frame time) | `python raster3d.py --model M --w 512 --h 512` | Lab 2.1 |
| ray-trace one frame | `python raytrace.py --model M --w 512 --h 512` | Lab 2.1, A2 |
| compare two frames (PSNR/SSIM/diff image) | `python compare.py A.ppm B.ppm --diff out/diff.ppm` | A2, Lab 3.2 |
| three broken scenes + evidence + fix proof | `python broken_scene.py --all` | Lab 2.2, A3 |
| perspective-correct vs naive UV (0.1 vs 0.5) + swimming render | `python lab2_texture.py --e4` | Lab 2.3, A5 |
| mipmap on/off from far away (PSNR vs SSAA truth, flicker) | `python lab2_texture.py --mip` | Lab 2.4, A5 |
| pitfall fixes, measured (back-face cull; float/reverse-Z depth) | `python lab2_pitfall.py` | Lab 2.5 |
| raster vs ray-trace frame time + hard shadow | `python bench.py [--model M] [--light x y z]` | Lab 3.3, A4 |
| package the model for WebGL / Pages | `python build_web.py [--model M --tex T]` | Lab 3.1, 3.4 |
| PNG/JPG texture -> PPM | `python convert_texture.py in.png data/mymodel.ppm --size 512` | Lab 1.2 |

`M` defaults to `data/pusaka_model.obj`. Your own model: put `data/mymodel.obj`
and `data/mymodel.ppm` (same stem) and every script picks the texture up;
`build_web.py` uses `mymodel.obj` automatically when it exists.

## Unit table

| File | Job |
|---|---|
| `transforms.py` | mat4 T/S/R, `perspective`, `ortho`, `look_at`, `project`, `viewport`, `normal_matrix`, bare `perspective_divide_2d` (E1-E3) |
| `scene.py` | the one camera / light / model matrix (`Ry(35) . S(1.4) . FIT`) used everywhere |
| `objio.py` | OBJ loader (`v`/`vt`/`f`, fan triangulation, `v` flip, **winding check via signed volume**) + `save_obj` |
| `make_model.py` | deterministic 15-box temple OBJ (CCW-front, per-face planar UVs) + `relief.ppm` |
| `raster3d.py` | MVP -> near clip -> w-divide -> viewport -> barycentric z-buffer -> perspective-correct UV -> bilinear / trilinear-mip texture -> Lambert (+ shadow map, back-face cull, float32 / reverse-Z depth) |
| `raytrace.py` | pinhole rays + Moller-Trumbore, nearest hit, same shading (+ shadow rays) |
| `broken_scene.py` | inside-out / zfight / stretched: render, depth image, matrices, SSIM before/after vs reference, depth histogram |
| `lab2_texture.py` | E4 measured from the rasteriser + checker floor; mipmap experiment |
| `lab2_pitfall.py` | back-face culling and depth-precision fixes, measured |
| `bench.py` | frame time + shadow comparison, writes `out/bench.csv` |
| `metrics.py` / `compare.py` | PSNR + global SSIM; amplified difference image |
| `ppm_io.py` / `srgb.py` | P6 PPM r/w and sRGB<->linear (Module 1) |
| `build_web.py`, `web/` | Three.js viewer reading `web/assets/scene.json`; OrbitControls, reset camera, export PPM, mipmap toggle, fps |
| `.github/workflows/pages.yml` | clean build (`make_model` + `tests` must pass) and deploy `web/` to GitHub Pages |

## Fixes applied to the starter (what changed and why)

1. **Winding was CW, not CCW.** `make_model.box()` listed every face clockwise
   seen from outside (normal of the "+y top" face pointed down). With
   back-face culling the *correct* render showed the temple inside-out and the
   `inside-out` bug demo was backwards; the Three.js viewer hid it with
   `DoubleSide`. Faces re-wound; `objio.load_obj` now checks the signed volume
   and re-winds a CW mesh; cull vs no-cull is now byte-identical (-19.7 % fragments).
2. **Degenerate UVs.** 120 of 180 triangles had zero UV area (the x-y planar
   projection gives constant `u` on the +-x sides and constant `v` on top/bottom),
   so the texture was smeared into stripes. Now: per-face planar UVs in world
   units (one tile per 2 units), 0 degenerate triangles.
3. **Texel convention.** Sampler used `u*(W-1)` and clamped; now `u*W-0.5` with
   REPEAT wrap (= WebGL `LINEAR` + `REPEAT`), so NumPy and WebGL sample the same texel.
4. **WebGL viewer would not match:** r155+ physically based lights (Lambert divides
   by pi) -> intensities x pi; `DataTexture` defaulted to NEAREST -> LINEAR;
   explicit row flip instead of `flipY` on a data texture; model matrix taken
   from `scene.json` instead of retyped; assets now live **inside** `web/` so
   `npx serve web` and GitHub Pages work (`../data/` is unreachable there).
5. `raster3d.py --out x.ppm` overwrote the colour image with the depth image
   when the name had no "raster" in it; fragments beyond the far plane were
   not rejected; colour is now rounded (not truncated) in both renderers ->
   raster vs raytrace is byte-identical (PSNR inf).
6. `tests.py` runs from any directory, prints its numbers, counts checks, and
   adds view-matrix, near-clip, mipmap, normal-matrix and raster-vs-raytrace checks.
7. Missing Lab 2 pieces added: SSIM before/after vs reference and a depth
   histogram (`broken_scene.py`), E4 render (`lab2_texture.py --e4`), mipmap
   (`lab2_texture.py --mip`), measured pitfall fixes (`lab2_pitfall.py`),
   shadows + frame time (`bench.py`). Model auto-fit (`scene.FIT`) so any
   exported model lands in frame.

## Verified

| Item | Status | Date | Toolchain |
|---|---|---|---|
| clean clone -> `make_model` -> `tests.py` | **ALL CHECKS PASSED (13 ok)** | 2026-11-19 | Linux, Python 3.13 (+3.11 syntax), NumPy 2.5.3, Pillow 12.3.0 |
| raster vs raytrace 256x256 | **PSNR inf (byte-identical), SSIM 1.000** | 2026-11-19 | same |
| `broken_scene.py --all` | inside-out SSIM 0.804 -> 1.000; zfight 2852 -> 184 px, ties 100 % -> 3.2 %; stretched 0.601 -> 1.000 | 2026-11-19 | same |
| `lab2_texture.py` | u measured 0.503 naive / 0.102 correct; mip PSNR vs SSAA 28.9 -> 37.6 dB, flicker 29.7 -> 0.7 codes/frame | 2026-11-19 | same |
| `web/` viewer | JS syntax + PPM parser unit-tested in Node 22; **not yet opened in a browser** | 2026-11-19 | -- |
| **YOUR toolchain** | _fill in: OS, Python, NumPy, Node, browser + version, three r160_ | | |

Before submitting: open the viewer (`npx serve web`), click **Reset kamera**,
**Export PPM** at 512x512, run
`python raster3d.py --model data/mymodel.obj --w 512 --h 512` and
`python compare.py out/temple_raster.ppm <name>_three.ppm --diff out/diff_web.ppm`
-- expect ~30 dB with the residual on anti-aliased silhouette edges (`?aa=0`
removes MSAA). Record browser + version in the row above.
