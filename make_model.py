"""Module 3 -- deterministic Pusaka3D temple: a stacked-tier candi.

No RNG, no binary assets in Git. Writes:
  data/pusaka_model.obj   180 triangles / 120 vertices (stacked boxes,
                          per-face planar UVs in world units)
  data/relief.ppm         256x256 procedural sandstone texture (Module 1 style)

Your graded input is a low-poly model of your OWN building; this is the
documented fallback.

Run:  python make_model.py [--out data]
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from objio import save_obj
from ppm_io import write_ppm
from srgb import linear_to_srgb, to_u8


TEX_WORLD = 2.0   # one texture tile per 2.0 model units (REPEAT wrap) = the
                  # density the original base box had horizontally


def face_uv(p, axis_sign):
    """Planar UV for ONE face, projected along its own normal and scaled in
    world units, so every face gets a real (non-degenerate) UV with the same
    texel density. The u axis points to the viewer's RIGHT when looking at
    the face from outside, so carvings are never mirrored."""
    x, y, z = p
    ax, sg = axis_sign
    if ax == "z":   u, v = (x if sg > 0 else -x), y          # front/back
    elif ax == "x": u, v = (-z if sg > 0 else z), y          # right/left
    else:           u, v = x, (-z if sg > 0 else z)          # top/bottom
    return (u / TEX_WORLD, v / TEX_WORLD)


def box(cx, cz, half, y0, y1):
    """Axis-aligned box, centre (cx,*,cz), horizontal half-size `half`,
    vertical span y0..y1. Returns (8 positions, 12 triangles as local idx,
    per-triangle-corner UVs (12,3,2)). CCW-front (outward-facing)."""
    x0, x1 = cx - half, cx + half
    z0, z1 = cz - half, cz + half
    P = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1],   # bottom 0..3
        [x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1],   # top    4..7
    ])
    # each face: 2 triangles, wound CCW seen from outside, + its normal axis
    quads = [
        # (fixed: the original lists were CW seen from outside -> normals
        #  pointed INTO the box and back-face culling removed the front)
        (((4, 6, 5), (4, 7, 6)), ("y", +1)),   # +y top
        (((0, 1, 2), (0, 2, 3)), ("y", -1)),   # -y bottom
        (((0, 5, 1), (0, 4, 5)), ("z", -1)),   # -z
        (((2, 7, 3), (2, 6, 7)), ("z", +1)),   # +z
        (((3, 4, 0), (3, 7, 4)), ("x", -1)),   # -x
        (((1, 6, 2), (1, 5, 6)), ("x", +1)),   # +x
    ]
    faces, uvs = [], []
    for tris, axis in quads:
        for tri in tris:
            faces.append(tri)
            uvs.append([face_uv(P[i], axis) for i in tri])
    return P, faces, np.array(uvs)


def build_temple():
    positions, faces, uvs, face_uvs = [], [], [], []
    uv_index = {}
    voff = 0
    # 6 stacked tiers, shrinking; a finial box on top
    tiers = [(0.0, 0.0, 1.00, 0.00, 0.35),
             (0.0, 0.0, 0.82, 0.35, 0.62),
             (0.0, 0.0, 0.66, 0.62, 0.85),
             (0.0, 0.0, 0.50, 0.85, 1.05),
             (0.0, 0.0, 0.34, 1.05, 1.22),
             (0.0, 0.0, 0.20, 1.22, 1.36),
             (0.0, 0.0, 0.07, 1.36, 1.52)]
    # four corner shrines (small boxes)
    for (sx, sz) in [(-0.8, -0.8), (0.8, -0.8), (-0.8, 0.8), (0.8, 0.8)]:
        tiers.append((sx, sz, 0.16, 0.0, 0.30))
        tiers.append((sx, sz, 0.10, 0.30, 0.44))
    for (cx, cz, half, y0, y1) in tiers:
        P, F, UV = box(cx, cz, half, y0, y1)
        for (a, b, c), tri_uv in zip(F, UV):
            faces.append((a + voff, b + voff, c + voff))
            ids = []
            for uv in tri_uv:                       # de-duplicate vt lines
                key = (round(float(uv[0]), 5), round(float(uv[1]), 5))
                if key not in uv_index:
                    uv_index[key] = len(uvs); uvs.append(key)
                ids.append(uv_index[key])
            face_uvs.append(tuple(ids))
        positions.extend(P.tolist())
        voff += len(P)
    return np.array(positions), np.array(uvs), faces, face_uvs


def relief_texture(n=256):
    """Procedural sandstone relief -- same idea as Module 1's make_data."""
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    fx = 0.22
    raw = np.sin(2 * np.pi * fx * (x + 0.5 * y))
    ridges = 0.5 + 0.40 * np.tanh(2.5 * raw)
    slow = 0.10 * np.sin(2 * np.pi * 0.02 * x) * np.cos(2 * np.pi * 0.015 * y)
    lum = np.clip(ridges + slow, 0.05, 0.98)
    lum[int(0.33 * n):int(0.37 * n), :] *= 0.6      # a carved band
    tint = (1.00, 0.82, 0.60)
    lin = np.dstack([lum * tint[0], lum * tint[1], lum * tint[2]])
    return to_u8(linear_to_srgb(np.clip(lin, 0, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    positions, uvs, faces, face_uvs = build_temple()
    save_obj(os.path.join(args.out, "pusaka_model.obj"), positions, uvs, faces, face_uvs)
    tex = relief_texture()
    nbytes = write_ppm(os.path.join(args.out, "relief.ppm"), tex)
    print(f"wrote {args.out}/pusaka_model.obj  ({len(positions)} verts, "
          f"{len(faces)} triangles, {len(uvs)} uvs)")
    print(f"wrote {args.out}/relief.ppm  ({tex.shape[1]}x{tex.shape[0]}, {nbytes} bytes)")


if __name__ == "__main__":
    main()
