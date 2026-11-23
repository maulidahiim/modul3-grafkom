"""Module 3 -- DIAGNOSE-AND-FIX: three broken renders, each fixed by one change.

  python broken_scene.py --bug inside-out   # winding / backface convention
  python broken_scene.py --bug zfight       # near plane 1e-3 -> depth crush
  python broken_scene.py --bug stretched    # matrix multiply order swapped
  python broken_scene.py --all              # evidence table for all three

Your Lab 2 task: from the EVIDENCE (the render, the depth image, the printed
matrices/values) -- write the diagnosis BEFORE reading the fix -- identify the
cause, apply the one change, and prove it (SSIM to the reference; a depth
histogram for z-fighting).

Outputs per bug (in out/):  bug_<name>_bad.ppm, bug_<name>_fixed.ppm,
bug_<name>_ref.ppm, bug_<name>_bad_depth.ppm, bug_<name>_fixed_depth.ppm,
and for zfight also zfight_depth_hist.csv (+ a text histogram on stdout).
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from transforms import rot_y, scale, translate
from objio import load_obj, Mesh
from ppm_io import read_ppm, write_ppm
from metrics import psnr, ssim
import raster3d
import scene


def _load():
    mesh = load_obj(scene.DEFAULT_MODEL)
    tex = read_ppm(scene.DEFAULT_TEXTURE)
    return mesh, tex


GOOD_MODEL = scene.model_matrix()          # Ry(35) . S(1.4)


def bug_inside_out(mesh, tex, w=192, h=192):
    # BUG: every triangle's winding flipped, rendered WITH back-face culling on
    # -> the outward faces are now "back-facing" and get culled; you see through
    # the temple to its far interior. FIX: re-wind the faces (or, if you cannot
    # touch the mesh, disable culling).
    flipped = Mesh(mesh.positions, mesh.uvs,
                   mesh.faces[:, ::-1].copy(),
                   None if mesh.face_uvs is None else mesh.face_uvs[:, ::-1].copy())
    bad, db, fbad = raster3d.render(flipped, w=w, h=h, tex_u8=tex, model=GOOD_MODEL,
                                    cull_back=True)
    good, dg, fgood = raster3d.render(mesh, w=w, h=h, tex_u8=tex, model=GOOD_MODEL,
                                      cull_back=True)
    fin = np.isfinite(db) & np.isfinite(dg)
    dd = db[fin] - dg[fin]
    farther = 100.0 * np.mean(dd > 1e-6)
    ev = (f"flipped winding + back-face cull: the outward faces are now "
          f"'back-facing' and culled, so the render shows the temple's "
          f"interior (far) surfaces instead. Visible depth is {dd.mean():+.3f} "
          f"NDC on average and FARTHER on {farther:.0f}% of the shared pixels "
          f"(depth image: bad is darker); {fbad} vs {fgood} fragments. "
          f"Check: objio.signed_volume() is negative for the bad mesh. "
          f"FIX: re-wind faces CCW (faces[:, ::-1]) -- objio.load_obj does it "
          f"automatically when the volume is negative.")
    return dict(bad=bad, fixed=good, ref=good, bad_depth=db, fixed_depth=dg,
                evidence=ev, pitfall="12.6 winding / backface drift")


def _zfight_scene():
    """Two quads that CROSS: A is flat at z=-9; B is A tilted ~1.2 deg about its
    own centre, so B's top is a few cm in front of A and B's bottom a few cm
    behind. They intersect along a near-horizontal line. With a float depth
    buffer the crossing is a crisp edge; with a low-precision buffer and a tiny
    `near`, the two depths collapse to one code across a wide band and draw
    order (A first) wins it -- z-fighting."""
    cy, cz = 1.0, -9.0
    quadA = np.array([[-3.0, -1, 0], [3, -1, 0], [3, 3, 0], [-3, 3, 0]]) + [0, 0, cz]
    a = np.radians(1.2)
    R = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    quadB = (quadA - [0, cy, cz]) @ R.T + [0, cy, cz]
    pos = np.vstack([quadA, quadB])
    faces = [[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6]]
    return Mesh(pos, np.zeros((0, 2)), faces, [])


TRI_COLS = [(230, 130, 40), (230, 130, 40), (60, 120, 210), (60, 120, 210)]  # A orange, B blue


def bug_zfight(mesh, tex, w=192, h=192):
    m = _zfight_scene()
    cam = dict(eye=(0, 1, 6), center=(0, 1, -1), cull_back=False, far=200,
               tri_colors=TRI_COLS, light_dir=(0, 0.2, 1))
    # low-precision (16-bit) depth buffer; the float-depth render is the truth
    bad, db, _ = raster3d.render(m, w=w, h=h, near=1e-3, depth_bits=16, **cam)
    good, dg, _ = raster3d.render(m, w=w, h=h, near=1.0, depth_bits=16, **cam)
    ref, _, _ = raster3d.render(m, w=w, h=h, near=1.0, depth_bits=None, **cam)
    fight_bad = int(np.sum(np.abs(bad.astype(int) - ref.astype(int)).max(2) > 60))
    fight_good = int(np.sum(np.abs(good.astype(int) - ref.astype(int)).max(2) > 60))
    hist = {nm: _code_gap_hist(m, w, h, nr, cam) for nm, nr in
            (("near=1e-3", 1e-3), ("near=1.0", 1.0))}
    ev = (f"near=1e-3 + 16-bit depth: {fight_bad} pixels wrong vs the float-depth "
          f"reference (the two quads collapse to one depth code and draw order "
          f"decides).  near=1.0 + 16-bit depth: only {fight_good} pixels differ "
          f"-> raising `near` is the fix.")
    return dict(bad=bad, fixed=good, ref=ref, bad_depth=db, fixed_depth=dg,
                evidence=ev, hist=hist, pitfall="12.2 near plane too small")


HIST_BINS = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 64),
             (65, 256), (257, 10 ** 9)]


def _code_gap_hist(m, w, h, near, cam):
    """Depth HISTOGRAM evidence: render quad A alone and quad B alone into a
    16-bit buffer, then for every pixel both cover, histogram |codeA - codeB|.
    A gap of 0 codes = the buffer cannot tell them apart = a fighting pixel."""
    cam = {k: v for k, v in cam.items() if k != "tri_colors"}
    qa = Mesh(m.positions, m.uvs, m.faces[:2], [])
    qb = Mesh(m.positions, m.uvs, m.faces[2:], [])
    _, da, _ = raster3d.render(qa, w=w, h=h, near=near, depth_bits=16, **cam)
    _, dbb, _ = raster3d.render(qb, w=w, h=h, near=near, depth_bits=16, **cam)
    both = np.isfinite(da) & np.isfinite(dbb)
    code = lambda z: np.round((z * 0.5 + 0.5) * 2 ** 16).astype(np.int64)
    gap = np.abs(code(da[both]) - code(dbb[both]))
    return [int(np.sum((gap >= a) & (gap <= b))) for a, b in HIST_BINS], int(both.sum())


def bug_stretched(mesh, tex, w=192, h=192):
    # WRONG: apply scale AFTER rotate-in-world by writing the product backwards
    wrong_model = scale((1.4, 1.4, 1.4)) @ rot_y(35) @ translate((0, 0, 0))
    wrong_model = wrong_model @ np.diag([2.2, 0.5, 1.0, 1.0])  # non-uniform, wrong place
    bad, db, _ = raster3d.render(mesh, w=w, h=h, tex_u8=tex, model=wrong_model)
    good, dg, _ = raster3d.render(mesh, w=w, h=h, tex_u8=tex, model=GOOD_MODEL)
    col_len = lambda M: np.linalg.norm(M[:3, :3], axis=0)
    ev = ("model matrix rows (bad):\n" + np.array2string(wrong_model[:3], precision=2) +
          "\nvs good:\n" + np.array2string(GOOD_MODEL[:3], precision=2) +
          f"\ncolumn lengths (= scale along model +x,+y,+z): bad "
          f"{np.round(col_len(wrong_model), 2)}  good {np.round(col_len(GOOD_MODEL), 2)}"
          f"\n -> the good matrix has three equal column lengths (uniform 1.4); "
          f"the bad one stretches model +x by {col_len(wrong_model)[0] / 1.4:.1f}x "
          f"and squashes +y to {col_len(wrong_model)[1] / 1.4:.1f}x: an extra "
          f"non-uniform scale sits in the chain (and S/T are written on the wrong "
          f"side of R).")
    return dict(bad=bad, fixed=good, ref=good, bad_depth=db, fixed_depth=dg,
                evidence=ev, pitfall="12.1 matrix multiply order")


BUGS = {"inside-out": bug_inside_out, "zfight": bug_zfight, "stretched": bug_stretched}


def _fmt_db(p):
    return "inf" if np.isinf(p) else f"{p:.1f}"


def run(name, mesh, tex):
    r = BUGS[name](mesh, tex)
    os.makedirs("out", exist_ok=True)
    for k in ("bad", "fixed", "ref"):
        write_ppm(f"out/bug_{name}_{k}.ppm", r[k])
    for k in ("bad", "fixed"):
        write_ppm(f"out/bug_{name}_{k}_depth.ppm", raster3d.depth_image(r[f"{k}_depth"]))
    print(f"\n== bug: {name} ==   (pitfall {r['pitfall']})")
    print("evidence:", r["evidence"])
    sb, sf = ssim(r["bad"], r["ref"]), ssim(r["fixed"], r["ref"])
    pb, pf = psnr(r["bad"], r["ref"]), psnr(r["fixed"], r["ref"])
    print(f"vs reference   BEFORE fix: SSIM {sb:.3f}  PSNR {_fmt_db(pb)} dB   "
          f"AFTER fix: SSIM {sf:.3f}  PSNR {_fmt_db(pf)} dB")
    print(f"SSIM(bad, fixed) = {ssim(r['bad'], r['fixed']):.3f}")
    if "hist" in r:
        print("depth histogram: |code(quad A) - code(quad B)| in a 16-bit buffer, "
              "pixels covered by both quads")
        lines = ["bin," + ",".join(r["hist"])]
        for i, (a, b) in enumerate(HIST_BINS):
            lab = f"{a}" if a == b else (f"{a}-{b}" if b < 10 ** 9 else f">{a - 1}")
            row = [r["hist"][k][0][i] for k in r["hist"]]
            lines.append(lab + "," + ",".join(map(str, row)))
            bars = "   ".join(f"{k}: {n:5d} {'#' * min(40, n // 25)}"
                              for k, n in zip(r["hist"], row))
            print(f"  gap {lab:>8} codes | {bars}")
        tot = {k: v[1] for k, v in r["hist"].items()}
        ties = {k: v[0][0] for k, v in r["hist"].items()}
        print("  ties (gap 0 = fighting):  " + "   ".join(
            f"{k}: {ties[k]}/{tot[k]} = {100 * ties[k] / max(tot[k], 1):.1f}%" for k in tot))
        with open("out/zfight_depth_hist.csv", "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print("wrote out/zfight_depth_hist.csv")
    print(f"wrote out/bug_{name}_{{bad,fixed,ref}}.ppm and *_depth.ppm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bug", choices=list(BUGS))
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    mesh, tex = _load()
    if args.all or not args.bug:
        for k in BUGS:
            run(k, mesh, tex)
    else:
        run(args.bug, mesh, tex)


if __name__ == "__main__":
    main()
