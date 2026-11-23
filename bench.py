"""Module 3 -- Lab 3 item 3 / A4: rasterisation vs ray tracing, MEASURED.

WRITE YOUR PREDICTION FIRST (frame-time ratio raytrace/raster; which one gives
better shadows/reflections and why), then run:

  python bench.py                                   # temple, 256x256, 3 runs
  python bench.py --model data/mymodel.obj --w 512 --h 512 --runs 5
  python bench.py --light 0.8 0.7 -0.3              # a light that casts VISIBLE
                                                    # shadows from this camera
  python bench.py --light 0.8 0.7 -0.3 --shadow-bias 0.004   # shadow acne

Reports median frame time for raster3d.py and raytrace.py, with and without a
hard shadow (raster: shadow map; ray tracer: one shadow ray per hit), their
ratio, how many pixels each marks as shadowed, and where the two shadow
answers disagree. Writes out/bench.csv and out/bench_shadow_*.ppm.
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np

import raster3d
import raytrace
import scene
from metrics import psnr
from objio import load_obj
from ppm_io import read_ppm, write_ppm


def timed(fn, runs):
    ts, res = [], None
    for _ in range(runs):
        t0 = time.perf_counter(); res = fn(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=scene.DEFAULT_MODEL)
    ap.add_argument("--tex")
    ap.add_argument("--w", type=int, default=256)
    ap.add_argument("--h", type=int, default=256)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--light", type=float, nargs=3, default=scene.LIGHT["light_dir"])
    ap.add_argument("--shadow-bias", type=float, default=0.01,
                    help="shadow-map depth bias (NDC). Too small -> acne; too big -> peter-panning")
    ap.add_argument("--shadow-res", type=int, default=512)
    a = ap.parse_args()

    mesh = load_obj(a.model)
    tp = scene.texture_for(a.model, a.tex)
    tex = read_ppm(tp) if tp else None
    M = scene.model_matrix(mesh)
    kw = dict(w=a.w, h=a.h, tex_u8=tex, model=M, light_dir=tuple(a.light))

    rows = []
    print(f"model {a.model}: {mesh.n_tris} triangles, {a.w}x{a.h}, "
          f"light {tuple(a.light)}, median of {a.runs} runs")
    imgs = {}
    for shadow in (False, True):
        tr, (ri, _, _) = timed(lambda: raster3d.render(
            mesh, shadow=shadow, shadow_bias=a.shadow_bias, shadow_res=a.shadow_res, **kw), a.runs)
        tt, (ti, _) = timed(lambda: raytrace.render(mesh, shadow=shadow, **kw), a.runs)
        imgs[shadow] = (ri, ti)
        rows.append(dict(shadow=shadow, raster_ms=tr * 1e3, raytrace_ms=tt * 1e3,
                         ratio=tt / tr))
        tag = "with shadow   " if shadow else "no shadow     "
        print(f"  {tag} raster3d {tr * 1e3:8.0f} ms   raytrace {tt * 1e3:8.0f} ms   "
              f"ratio rt/raster {tt / tr:5.1f}x")

    (r0, t0), (r1, t1) = imgs[False], imgs[True]
    sr = np.abs(r0.astype(int) - r1.astype(int)).max(2) > 0
    st = np.abs(t0.astype(int) - t1.astype(int)).max(2) > 0
    p = psnr(r1, t1)
    print(f"  shadowed pixels: raster (shadow map) {int(sr.sum())}   "
          f"raytrace (shadow rays) {int(st.sum())}   agree on {int((sr & st).sum())}")
    print(f"  shadow frames raster vs raytrace: PSNR "
          f"{'inf' if np.isinf(p) else f'{p:.1f}'} dB  "
          f"(raster-only {int((sr & ~st).sum())} px = shadow-map acne/aliasing, "
          f"raytrace-only {int((st & ~sr).sum())} px = missed by the map)")
    print("  note: frame time grows ~ pixels x triangles for the ray tracer "
          "(no BVH here) but ~ triangles + covered pixels for the rasteriser.")
    os.makedirs("out", exist_ok=True)
    write_ppm("out/bench_shadow_raster.ppm", r1)
    write_ppm("out/bench_shadow_rt.ppm", t1)
    with open("out/bench.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]) + ["triangles", "w", "h"])
        wr.writeheader()
        for r in rows:
            wr.writerow({**r, "triangles": mesh.n_tris, "w": a.w, "h": a.h})
    print("  wrote out/bench.csv, out/bench_shadow_raster.ppm, out/bench_shadow_rt.ppm")


if __name__ == "__main__":
    main()
