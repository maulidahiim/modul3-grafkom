"""Module 3 -- Lab 2 item 5: pitfall-driven fixes, MEASURED.

  python lab2_pitfall.py --cull     # back-face culling: fragment drop, same image
  python lab2_pitfall.py --depth    # z-fight scene re-run: 16-bit / float32 /
                                    # reverse-Z float32 / float64, near = 1e-3
  python lab2_pitfall.py            # both
"""
from __future__ import annotations

import argparse
import time

import numpy as np

import raster3d
import scene
from broken_scene import _zfight_scene, TRI_COLS
from metrics import psnr
from objio import load_obj
from ppm_io import read_ppm


def cull(model_path=None, w=256, h=256):
    path = model_path or scene.DEFAULT_MODEL
    mesh = load_obj(path)
    tp = scene.texture_for(path)
    tex = read_ppm(tp) if tp else None
    M = scene.model_matrix(mesh)
    out = {}
    for on in (False, True):
        st = {}
        t0 = time.perf_counter()
        img, _, _ = raster3d.render(mesh, w=w, h=h, tex_u8=tex, model=M,
                                    cull_back=on, stats=st)
        out[on] = (img, st, time.perf_counter() - t0)
    (a, sa, ta), (b, sb, tb) = out[False], out[True]
    p = psnr(a, b)
    drop = 100 * (1 - sb["depth_passed"] / sa["depth_passed"])
    print(f"BACK-FACE CULLING  ({path}, {mesh.n_tris} tris, {w}x{h})")
    print(f"    triangles culled          : {sb['culled_tris']} / {mesh.n_tris}")
    print(f"    fragments written (z-pass): {sa['depth_passed']} -> {sb['depth_passed']}"
          f"   (-{drop:.1f}%)")
    print(f"    coverage tests            : {sa['coverage_tested']} -> {sb['coverage_tested']}")
    print(f"    frame time                : {ta*1000:.0f} ms -> {tb*1000:.0f} ms")
    print(f"    image with vs without cull: PSNR {'inf' if np.isinf(p) else f'{p:.1f}'} dB"
          f"  ({'identical -> winding is consistent CCW-front' if np.isinf(p) else 'DIFFERENT -> some faces are wound CW: fix the mesh'})")


def depth(w=192, h=192):
    m = _zfight_scene()
    cam = dict(eye=(0, 1, 6), center=(0, 1, -1), cull_back=False, far=200,
               tri_colors=TRI_COLS, light_dir=(0, 0.2, 1))
    ref, _, _ = raster3d.render(m, w=w, h=h, near=1.0, **cam)
    print("DEPTH PRECISION  (z-fight scene, wrong pixels vs float64 / near=1.0 truth)")
    rows = [("16-bit int, near=1e-3", dict(near=1e-3, depth_bits=16)),
            ("16-bit int, near=1.0 ", dict(near=1.0, depth_bits=16)),
            ("float32,    near=1e-3", dict(near=1e-3, depth_mode="f32")),
            ("reverse-Z f32, n=1e-3", dict(near=1e-3, depth_mode="reverse-z-f32")),
            ("float64,    near=1e-3", dict(near=1e-3))]
    for name, kw in rows:
        img, _, _ = raster3d.render(m, w=w, h=h, **kw, **cam)
        bad = int(np.sum(np.abs(img.astype(int) - ref.astype(int)).max(2) > 60))
        print(f"    {name} : {bad:5d} wrong pixels")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cull", action="store_true")
    ap.add_argument("--depth", action="store_true")
    ap.add_argument("--model")
    a = ap.parse_args()
    both = not (a.cull or a.depth)
    if a.cull or both:
        cull(a.model)
    if a.depth or both:
        depth()


if __name__ == "__main__":
    main()
