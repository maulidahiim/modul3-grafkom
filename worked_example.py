"""Module 3 -- Fully Worked Example, Step 1: the running point (2,3,4) through
every coordinate space.

Model order matches Example E1: translate (1,0,-1) FIRST, then yaw 90 deg,
i.e.  M = Ry(90) @ T(1,0,-1).

Run:  python worked_example.py                                   # the E1 point
      python worked_example.py --model data/mymodel.obj --vertex 0   # Lab 1.3 / 1.4

With --model, one vertex of YOUR model is traced through every space using
the exact scene the renderers use (scene.py), and the columns of M, V and
the camera pose are read out (Lab 1 step 4, A1).
"""
from __future__ import annotations

import numpy as np

from transforms import (translate, rot_y, look_at, perspective, project,
                        viewport, perspective_divide_2d)

import argparse

np.set_printoptions(precision=3, suppress=True)

_ap = argparse.ArgumentParser()
_ap.add_argument("--model")
_ap.add_argument("--vertex", type=int, default=0)
_ap.add_argument("--w", type=int, default=512)
_ap.add_argument("--h", type=int, default=512)
_args = _ap.parse_args()


def _trace_model(path, vi, W, H):
    import scene
    from objio import load_obj
    np.set_printoptions(precision=4, suppress=True)
    mesh = load_obj(path, verbose=True)
    if not 0 <= vi < len(mesh.positions):
        raise SystemExit(f"--vertex must be 0..{len(mesh.positions) - 1}")
    cam = scene.CAMERA
    M = scene.model_matrix(mesh)
    V = look_at(cam["eye"], cam["center"])
    P = perspective(cam["fovy"], W / H, cam["near"], cam["far"])
    p = np.r_[mesh.positions[vi], 1.0]
    world = M @ p; eye = V @ world; clip = P @ eye; ndc = clip[:3] / clip[3]
    scr = viewport(ndc[None, :], W, H)[0]
    print(f"model {path}: {mesh.n_tris} triangles, {len(mesh.positions)} vertices; "
          f"tracing vertex #{vi}\n")
    print(f"object  p            = {p}")
    print(f"world   M p          = {world}")
    print(f"view    V (M p)      = {eye}   (z={eye[2]:.3f}: "
          f"{'in front' if eye[2] < 0 else 'BEHIND the camera'})")
    print(f"clip    P V M p      = {clip}   (w = -z_eye = {clip[3]:.4f})")
    print(f"ndc     clip / w     = {ndc}   "
          f"({'inside' if np.all(np.abs(ndc) <= 1) else 'OUTSIDE'} the [-1,1]^3 cube)")
    print(f"screen  ({W}x{H})    = x {scr[0]:.2f}  y {scr[1]:.2f}   depth01 "
          f"{(ndc[2] + 1) / 2:.4f}\n")
    print("M (model matrix) =\n" + np.array2string(M))
    names = ["+x", "+y", "+z"]
    print("column reading of M (where the MODEL's axes / origin land in WORLD):")
    for k in range(3):
        c = M[:3, k]
        print(f"  col {k + 1}: model {names[k]} -> {c}  (length {np.linalg.norm(c):.4f})")
    print(f"  col 4: model origin -> {M[:3, 3]}")
    print("\nV (view matrix) =\n" + np.array2string(V))
    pose = np.linalg.inv(V)
    print("column reading of the camera POSE = V^-1 (the camera placed in WORLD):")
    print(f"  col 1: camera right (+x) -> {pose[:3, 0]}")
    print(f"  col 2: camera up    (+y) -> {pose[:3, 1]}")
    print(f"  col 3: camera back  (+z) -> {pose[:3, 2]}   (it LOOKS down -z = {-pose[:3, 2]})")
    print(f"  col 4: camera position   -> {pose[:3, 3]}   (= eye {np.array(cam['eye'])})")
    print("check: rows 1-3 of V's 3x3 are those same axes (V = R^T T(-c)); "
          f"R^T R = I: {np.allclose(V[:3, :3] @ V[:3, :3].T, np.eye(3))}")


if _args.model:
    _trace_model(_args.model, _args.vertex, _args.w, _args.h)
    raise SystemExit(0)

p = np.array([2.0, 3.0, 4.0])
print(f"object p        = {np.r_[p, 1.0]}")

M = rot_y(90) @ translate((1, 0, -1))
world = (M @ np.r_[p, 1.0])
print(f"model  M = Ry(90) . T(1,0,-1)        ; world  = M p       = {world}")

V = look_at(eye=(0, 0, 8), center=(0, 0, 0))
eye = V @ world
print(f"view   V = lookAt(eye=[0,0,8], ctr=0); eye    = V world   = {eye}   "
      f"(z={eye[2]:.0f}: {'in front' if eye[2] < 0 else 'BEHIND'}, dist {abs(eye[2]):.0f})")

P = perspective(60, 1.0, 1.0, 50.0)
clip = P @ eye
print(f"proj   P = perspective(60deg, 1, 1, 50)")
print(f"clip            = P eye              = {clip}")
ndc = clip[:3] / clip[3]
print(f"ndc  = clip/clip.w                   = {ndc}")

W, H = 800, 600
scr = viewport(ndc[None, :], W, H)[0]
print(f"screen ({W}x{H}) = [{scr[0]:.0f} {scr[1]:.0f}]  depth {(ndc[2] + 1) * 0.5:.3f}")

print()
print("--- Example E1 cross-check (bare similar-triangles projection) ---")
print(f"world point (pre-view) = {world[:3]}   -> after Ry(90) of (3,3,3): (3,3,-3)")
proj2d = perspective_divide_2d(world[:3], d=2)
print(f"perspective divide d=2 of (3,3,-3)   = {proj2d}   (expected [2. 2.])")
