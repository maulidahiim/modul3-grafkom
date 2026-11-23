"""Module 3 -- Lab 1 input: a low-poly Tugu Pahlawan (Surabaya), built
procedurally (made with AI assistance -- say so in the report).

Writes:
  data/mymodel.obj   stepped square plinth + fluted, tapering shaft
                     ("paku terbalik": 10 flutes = 10 November, 11 segments
                     = month 11) + pointed tip. CCW-front, real UVs.
  data/mymodel.ppm   512x512 texture ATLAS (one texture for the whole model):
                     top half  = white concrete (shaft, v in [0.5, 1])
                     bottom half = grey stone blocks (plinth, v in [0, 0.5])

Units are metres (the real monument is about 41 m tall); scene.py's FIT
scales it into frame, so no manual rescaling is needed.

Run:  python make_tugu.py            (then: python raster3d.py --model data/mymodel.obj)
Open data/mymodel.obj in Blender (File > Import > Wavefront) to inspect/edit it.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from ppm_io import write_ppm
from srgb import linear_to_srgb, to_u8

N_FLUTES = 10          # lengkungan  -> 2 * N_FLUTES vertices around
N_SEGMENTS = 11        # ruas
PLINTH = [(14.0, 0.0, 1.2), (11.0, 1.2, 2.2), (8.0, 2.2, 3.0)]   # (width, y0, y1)
SHAFT_Y0, SHAFT_Y1, TIP_Y = 3.0, 41.0, 44.0
R_BOTTOM, R_TOP = 3.0, 0.9
GROOVE = 0.86          # groove radius / ridge radius
BELT = 1.07            # 'sabuk' ring bulge at every segment boundary


class Builder:
    def __init__(self):
        self.v, self.vt, self.f = [], [], []          # f: ((vi, ti), (vi, ti), (vi, ti))

    def vert(self, p):
        self.v.append(tuple(float(x) for x in p)); return len(self.v) - 1

    def uv(self, u, v):
        self.vt.append((float(u), float(v))); return len(self.vt) - 1

    def tri(self, a, b, c):                            # each = (vi, ti), CCW from outside
        self.f.append((a, b, c))


def plinth_box(B, w, y0, y1):
    h = w / 2
    P = [(-h, y0, -h), (h, y0, -h), (h, y0, h), (-h, y0, h),
         (-h, y1, -h), (h, y1, -h), (h, y1, h), (-h, y1, h)]
    vi = [B.vert(p) for p in P]
    # quads CCW seen from outside (verified by signed volume below)
    quads = [(4, 7, 6, 5), (0, 1, 2, 3),               # top, bottom
             (0, 4, 5, 1), (2, 6, 7, 3),               # -z, +z
             (3, 7, 4, 0), (1, 5, 6, 2)]               # -x, +x
    for q in quads:
        # per-face UV into the STONE half of the atlas (v in [0.02, 0.48])
        uvs = [(0.0, 0.02), (0.0, 0.48), (1.0, 0.48), (1.0, 0.02)]
        if q in [(4, 7, 6, 5), (0, 1, 2, 3)]:
            uvs = [(0.0, 0.02), (0.0, 0.48), (1.0, 0.48), (1.0, 0.02)]
        t = [B.uv(*uv) for uv in uvs]
        a, b, c, d = [(vi[q[k]], t[k]) for k in range(4)]
        B.tri(a, b, c); B.tri(a, c, d)


def shaft(B):
    n = 2 * N_FLUTES
    ys = np.linspace(SHAFT_Y0, SHAFT_Y1, N_SEGMENTS + 1)
    rings = []
    for i, y in enumerate(ys):
        t = (y - SHAFT_Y0) / (SHAFT_Y1 - SHAFT_Y0)
        r = R_TOP + (R_BOTTOM - R_TOP) * (1 - t) ** 1.6      # flared foot, slim top
        if 0 < i < N_SEGMENTS:
            r *= BELT                                         # sabuk at each boundary
        ring = []
        for j in range(n):
            a = 2 * np.pi * j / n
            rr = r * (1.0 if j % 2 == 0 else GROOVE)          # ridge / groove
            ring.append(B.vert((rr * np.sin(a), y, rr * np.cos(a))))
        rings.append(ring)
    v_of = lambda y: 0.52 + 0.46 * (y - SHAFT_Y0) / (TIP_Y - SHAFT_Y0)
    for i in range(N_SEGMENTS):
        for j in range(n):
            j2 = (j + 1) % n
            u0, u1 = j / n, (j + 1) / n                       # seam: u runs 0..1
            t00 = B.uv(u0, v_of(ys[i])); t10 = B.uv(u1, v_of(ys[i]))
            t01 = B.uv(u0, v_of(ys[i + 1])); t11 = B.uv(u1, v_of(ys[i + 1]))
            a = (rings[i][j], t00); b = (rings[i][j2], t10)
            c = (rings[i + 1][j2], t11); d = (rings[i + 1][j], t01)
            B.tri(a, b, c); B.tri(a, c, d)
    # pointed tip
    apex = B.vert((0.0, TIP_Y, 0.0))
    for j in range(n):
        j2 = (j + 1) % n
        ta = B.uv((j + 0.5) / n, 0.98)
        B.tri((rings[-1][j], B.uv(j / n, v_of(SHAFT_Y1))),
              (rings[-1][j2], B.uv((j + 1) / n, v_of(SHAFT_Y1))), (apex, ta))
    # bottom cap (sits on the plinth; closes the mesh)
    centre = B.vert((0.0, SHAFT_Y0, 0.0)); tc = B.uv(0.5, 0.52)
    for j in range(n):
        j2 = (j + 1) % n
        B.tri((rings[0][j2], tc), (rings[0][j], tc), (centre, tc))


def texture(n=512, seed=10111945):
    rng = np.random.RandomState(seed)                 # fixed seed -> deterministic
    y, x = np.mgrid[0:n, 0:n].astype(float)
    img = np.zeros((n, n, 3))
    half = n // 2
    # --- top half: white concrete, fine noise, faint vertical pour lines
    noise = rng.rand(half, n)
    noise = (noise + np.roll(noise, 1, 0) + np.roll(noise, 1, 1) + np.roll(noise, -1, 1)) / 4
    conc = 0.72 + 0.10 * (noise - 0.5) + 0.03 * np.sin(2 * np.pi * x[:half] / 24.0)
    img[:half] = conc[..., None] * np.array([1.0, 0.98, 0.93])
    # --- bottom half: grey stone blocks with mortar lines
    yy, xx = y[half:] - half, x[half:]
    rows = 4
    bh = half / rows
    row = np.floor(yy / bh)
    bw = n / 6
    xo = xx + (row % 2) * bw / 2
    mortar = (np.mod(yy, bh) < 3) | (np.mod(xo, bw) < 3)
    block_id = (row * 13 + np.floor(xo / bw)).astype(int)
    shade = 0.38 + 0.10 * rng.rand(int(block_id.max()) + 1)[block_id]
    sn = rng.rand(half, n) * 0.05
    stone = np.where(mortar, 0.18, shade + sn)
    img[half:] = stone[..., None] * np.array([0.95, 0.93, 0.90])
    return to_u8(linear_to_srgb(np.clip(img, 0, 1)))


def signed_volume(B):
    P = np.array(B.v)
    tris = np.array([[a[0], b[0], c[0]] for a, b, c in B.f])
    p = P[tris]
    return float(np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])).sum() / 6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    B = Builder()
    for w, y0, y1 in PLINTH:
        plinth_box(B, w, y0, y1)
    shaft(B)
    vol = signed_volume(B)
    assert vol > 0, f"winding check failed (signed volume {vol})"
    obj = os.path.join(a.out, "mymodel.obj")
    with open(obj, "w") as fh:
        fh.write("# Tugu Pahlawan (Surabaya), low-poly -- make_tugu.py "
                 "(procedural, made with AI assistance)\n")
        fh.write("# units: metres, y up, CCW front, UV origin bottom-left\n")
        for p in B.v:
            fh.write(f"v {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
        for u, v in B.vt:
            fh.write(f"vt {u:.5f} {v:.5f}\n")
        for tri in B.f:
            fh.write("f " + " ".join(f"{vi + 1}/{ti + 1}" for vi, ti in tri) + "\n")
    tex = texture()
    write_ppm(os.path.join(a.out, "mymodel.ppm"), tex)
    print(f"wrote {obj}  ({len(B.v)} verts, {len(B.f)} triangles, signed volume "
          f"{vol:.0f} m^3 > 0 -> CCW-front)")
    print(f"wrote {a.out}/mymodel.ppm  ({tex.shape[1]}x{tex.shape[0]} atlas: "
          f"top = concrete shaft, bottom = stone plinth)")


if __name__ == "__main__":
    main()