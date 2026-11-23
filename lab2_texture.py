"""Module 3 -- Lab 2 items 3 and 4: texture & sampling evidence (A5).

  python lab2_texture.py --e4      # perspective-correct vs naive UV (0.1 vs 0.5)
  python lab2_texture.py --mip     # mipmap on/off, far view: flicker + aliasing
  python lab2_texture.py           # both

--e4   (a) MEASURES u from the real rasteriser: a 'u-ramp' texture whose red
           channel IS u (linear light), on the E4 edge L(w=1) .. R(w=9);
           reads back the pixel at the screen midpoint for both modes.
       (b) renders a checkered floor both ways -> out/e4_floor_naive_vs_pc.ppm
           (naive: the two triangles of the quad disagree along the diagonal
           -- the 'swimming' kink; correct: straight checker lines).
--mip  renders the temple from far away with plain bilinear and with a
       4-level trilinear mipmap, against a 4x4 supersampled ground truth;
       reports PSNR vs ground truth, high-frequency energy, and frame-to-frame
       flicker over a slow orbit (the Module 1 aliasing, now on a texture).
"""
from __future__ import annotations

import argparse
import os

import numpy as np

import raster3d
import scene
from metrics import psnr
from objio import Mesh, load_obj
from ppm_io import read_ppm, write_ppm
from srgb import srgb_to_linear, linear_to_srgb, to_u8

OUT = "out"


# ----------------------------------------------------------------------- E4
def e4_measure():
    W = H = 200
    # camera at origin looking down -z, fovy 90 -> f = 1, ndc_x = x / -z
    L = [-0.8, -0.5, -1.0]           # ndc (-0.8, -0.5), eye depth w_L = 1
    R = [0.8 * 9, -0.5 * 9, -9.0]    # ndc (+0.8, -0.5), eye depth w_R = 9
    C = [0.0, 0.5 * 5, -5.0]         # ndc ( 0.0, +0.5)
    uv = np.array([[0.0, 0.5], [1.0, 0.5], [0.5, 0.5]])
    m = Mesh(np.array([L, R, C]), uv, [[0, 1, 2]], [[0, 1, 2]])
    # u-ramp texture: red channel = u in LINEAR light (sRGB-encoded in the file)
    ramp = np.zeros((4, 256, 3)); ramp[..., 0] = (np.arange(256) + 0.5) / 256
    ramp_u8 = to_u8(linear_to_srgb(ramp))
    kw = dict(w=W, h=H, tex_u8=ramp_u8, eye=(0, 0, 0), center=(0, 0, -1),
              fovy=90.0, near=0.5, far=20.0, ambient=1.0, light_dir=(0, 0, 1))
    row = int((1 - (-0.5 * 0.5 + 0.5)) * H) - 1          # just above the edge
    res = {}
    for name, pc in (("naive", False), ("perspective-correct", True)):
        img, _, _ = raster3d.render(m, perspective_correct=pc, **kw)
        res[name] = float(srgb_to_linear(img[row, W // 2, 0] / 255.0))
    print("E4  edge L (w=1, u=0) -> R (w=9, u=1); u read back from the rasteriser "
          f"at the screen midpoint (row {row}, col {W // 2}):")
    for k, v in res.items():
        print(f"    {k:20s} u = {v:.3f}")
    print("    analytic: naive 0.500, perspective-correct 0.100 "
          "(tiny offset: the sampled row is 1 px inside the triangle)")
    return res


def checker(n=256, k=8):
    y, x = np.mgrid[0:n, 0:n]
    c = ((x // (n // k) + y // (n // k)) % 2).astype(float)
    lin = np.dstack([0.15 + 0.8 * c, 0.15 + 0.6 * c, 0.1 + 0.3 * c])
    return to_u8(linear_to_srgb(lin))


def e4_floor():
    W, H = 256, 192
    s = 4.0
    pos = np.array([[-s, 0, s], [s, 0, s], [s, 0, -s], [-s, 0, -s]], float)
    uv = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], float)
    m = Mesh(pos, uv, [[0, 1, 2], [0, 2, 3]], [[0, 1, 2], [0, 2, 3]])
    kw = dict(w=W, h=H, tex_u8=checker(), eye=(0.0, 1.2, 4.6),
              center=(0.0, 0.0, -2.0), fovy=60.0, near=0.5, far=30.0,
              ambient=1.0, light_dir=(0, 1, 0))
    naive, _, _ = raster3d.render(m, perspective_correct=False, **kw)
    good, _, _ = raster3d.render(m, perspective_correct=True, **kw)
    gap = np.full((H, 4, 3), 255, np.uint8)
    os.makedirs(OUT, exist_ok=True)
    write_ppm(f"{OUT}/e4_floor_naive_vs_pc.ppm", np.concatenate([naive, gap, good], 1))
    cov = good.max(2) > 0
    d = np.abs(naive.astype(int) - good.astype(int)).max(2)
    print(f"E4  floor quad, naive vs perspective-correct: {100 * np.mean(d[cov] > 40):.1f}% "
          f"of floor pixels differ by > 40 codes; PSNR {psnr(naive, good):.1f} dB")
    print(f"    wrote {OUT}/e4_floor_naive_vs_pc.ppm  (left naive: bent checker, kink "
          f"along the diagonal; right: correct)")


# ------------------------------------------------------------------- mipmap
def _far_render(mesh, tex, model, w, h, mip, yaw_off=0.0, dist=3.2):
    c = np.asarray(scene.CAMERA["center"], float)
    e = np.asarray(scene.CAMERA["eye"], float)
    a = np.radians(yaw_off)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    eye = c + R @ ((e - c) * dist)
    col, dep, _ = raster3d.render(mesh, w=w, h=h, tex_u8=tex, model=model,
                                  eye=tuple(eye), far=40.0, mip_levels=mip)
    return col, dep


def _ssaa(mesh, tex, model, w, h, k=4):
    big, dep = _far_render(mesh, tex, model, w * k, h * k, None)
    lin = srgb_to_linear(big / 255.0).reshape(h, k, w, k, 3).mean((1, 3))
    return to_u8(linear_to_srgb(lin))


def _hf_energy(img, mask):
    g = srgb_to_linear(img / 255.0).mean(2)
    lap = np.abs(4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:])
    return float(lap[mask[1:-1, 1:-1]].mean())


def mipmap_lab(levels=4, frames=8, step_deg=0.4):
    mesh = load_obj(scene.DEFAULT_MODEL)
    tex = read_ppm(scene.DEFAULT_TEXTURE)
    M = scene.model_matrix(mesh)
    W = H = 192
    off, dep = _far_render(mesh, tex, M, W, H, None)
    on, _ = _far_render(mesh, tex, M, W, H, levels)
    ref = _ssaa(mesh, tex, M, W, H)
    fin = np.isfinite(dep)
    inner = fin.copy()                           # erode: ignore silhouette pixels
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        inner &= np.roll(fin, (dy, dx), (0, 1))
    os.makedirs(OUT, exist_ok=True)
    gap = np.full((H, 4, 3), 255, np.uint8)
    write_ppm(f"{OUT}/mip_off_on_ssaa.ppm", np.concatenate([off, gap, on, gap, ref], 1))

    def flicker(mip):
        prev = None; diffs = []
        for i in range(frames):
            img, d = _far_render(mesh, tex, M, W, H, mip, yaw_off=i * step_deg)
            msk = np.isfinite(d)
            if prev is not None:
                both = msk & prev[1]
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    both &= np.roll(both, (dy, dx), (0, 1))
                diffs.append(np.abs(img.astype(int) - prev[0].astype(int)).mean(2)[both].mean())
            prev = (img, msk)
        return float(np.mean(diffs))

    fl_off, fl_on = flicker(None), flicker(levels)
    print(f"MIP temple from 3.2x farther, {W}x{H}, ground truth = 4x4 supersampled (linear light)")
    print(f"    {'':22s} {'no mipmap':>12s} {f'mipmap ({levels} lv)':>14s} {'SSAA truth':>12s}")
    print(f"    {'PSNR vs truth (dB)':22s} {psnr(off, ref):12.1f} {psnr(on, ref):14.1f} {'-':>12s}")
    print(f"    {'HF energy (|Lap|)':22s} {_hf_energy(off, inner):12.4f} "
          f"{_hf_energy(on, inner):14.4f} {_hf_energy(ref, inner):12.4f}")
    print(f"    {'flicker (codes/frame)':22s} {fl_off:12.2f} {fl_on:14.2f} {'-':>12s}"
          f"   ({frames} frames, orbit {step_deg} deg/frame)")
    print(f"    wrote {OUT}/mip_off_on_ssaa.ppm  (left: no mip, middle: mip, right: truth)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e4", action="store_true")
    ap.add_argument("--mip", action="store_true")
    ap.add_argument("--levels", type=int, default=4)
    args = ap.parse_args()
    both = not (args.e4 or args.mip)
    if args.e4 or both:
        e4_measure(); e4_floor()
    if args.mip or both:
        mipmap_lab(args.levels)


if __name__ == "__main__":
    main()
