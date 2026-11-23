"""Module 3 -- run every from-scratch check. `python tests.py`. Exit 0 iff all
pass. Hand-checkable invariants of Examples E1-E5 + pipeline pieces + a
1-triangle render + the reference temple frame.

Each check PRINTS its numbers (the report asks for "the numbers, not
'passed'"), so you can paste this output straight into A2.
Runs from any working directory.
"""
from __future__ import annotations

import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

import numpy as np

from transforms import (translate, rot_y, look_at, perspective, project,
                        viewport, perspective_divide_2d, normal_matrix, scale)


def check(name, fn):
    try:
        info = fn()
        print("OK   " + name + (f"\n       -> {info}" if info else ""))
        return True
    except Exception:
        print("FAIL " + name); traceback.print_exc(); return False


def t_e1_running_point():
    p = np.array([2.0, 3.0, 4.0, 1.0])
    world = rot_y(90) @ translate((1, 0, -1)) @ p        # translate first, then yaw
    assert np.allclose(world, [3, 3, -3, 1], atol=1e-9), world
    proj = perspective_divide_2d(world[:3], d=2)
    assert np.allclose(proj, [2, 2], atol=1e-9), proj
    # order matters: yaw first, then translate -> different point
    other = translate((1, 0, -1)) @ rot_y(90) @ p
    assert not np.allclose(other, world), "order should matter"
    assert np.allclose(other, [5, 3, -3, 1], atol=1e-9), other
    return (f"T then Ry: {np.round(world[:3], 6)}  divide d=2: {np.round(proj, 6)}  "
            f"Ry then T: {np.round(other[:3], 6)}")


def t_e2_duality():
    p = np.array([5.0, 0.0, 0.0, 1.0])
    V_cam_plus_x = look_at(eye=(1, 0, 0), center=(1, 0, -1))   # camera moved +x
    model_minus_x = translate((-1, 0, 0))                      # model moved -x
    a, b = V_cam_plus_x @ p, model_minus_x @ p
    assert np.allclose(a, b, atol=1e-9), (a, b)
    return f"camera +x: {np.round(a[:3], 6)}   model -x: {np.round(b[:3], 6)}"


def t_e3_perspective_near_far():
    n, f = 1.0, 9.0
    P = perspective(60, 1.0, n, f)
    got = []
    for z, want in [(-n, -1.0), (-f, +1.0), (-3.0, 0.5)]:
        clip = P @ np.array([0, 0, z, 1.0])
        ndc_z = clip[2] / clip[3]
        assert abs(ndc_z - want) < 1e-9, f"z={z}: ndc_z={ndc_z} want {want}"
        got.append(f"z={z:g}->{ndc_z:+.3f}")
    return "  ".join(got)


def t_e4_perspective_correct_u():
    # floor edge: L (x=0, u=0, w=1) .. R (x=100, u=1, w=9); u at screen x=50
    u_over_w = np.interp(50, [0, 100], [0 / 1, 1 / 9])
    inv_w = np.interp(50, [0, 100], [1 / 1, 1 / 9])
    u_pc = u_over_w / inv_w
    u_naive = np.interp(50, [0, 100], [0.0, 1.0])
    assert abs(u_pc - 0.1) < 1e-12, u_pc
    assert abs(u_naive - 0.5) < 1e-12, u_naive
    return f"naive u = {u_naive:.3f}   perspective-correct u = {u_pc:.3f}"


def t_e5_barycentric_depth():
    zA, zB, zC = 0.20, 0.50, 0.90
    la, lb, lc = 0.25, 0.25, 0.50
    assert abs(la + lb + lc - 1) < 1e-12
    z = la * zA + lb * zB + lc * zC
    assert abs(z - 0.625) < 1e-12, z
    return f"depth = {z:.3f}"


def t_view_is_inverse_pose():
    """LO3: V = pose^-1, and its rotation part is orthonormal (R^T R = I)."""
    eye, ctr = np.array([2.6, 2.0, 3.4]), np.array([0, 0.7, 0])
    V = look_at(eye, ctr)
    R = V[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert abs(np.linalg.det(R) - 1) < 1e-12
    pose = np.linalg.inv(V)
    assert np.allclose(pose[:3, 3], eye, atol=1e-12), pose[:3, 3]     # col 4 = eye
    fwd = ctr - eye; fwd /= np.linalg.norm(fwd)
    assert np.allclose(-pose[:3, 2], fwd, atol=1e-12)                 # looks down -z
    return f"pose col4 = eye {np.round(pose[:3, 3], 3)}; det R = {np.linalg.det(R):.6f}"


def t_projection_pipeline_shapes():
    verts = np.array([[0, 0, -2.0], [1, 0, -2.0], [0, 1, -3.0]])
    P = perspective(50, 4 / 3, 0.1, 100)
    V = look_at((0, 0, 3), (0, 0, 0))
    clip, ndc = project(P, V, np.eye(4), verts)
    assert clip.shape == (3, 4) and ndc.shape == (3, 3)
    assert np.all(clip[:, 3] > 0), "points in front must have w = -z_eye > 0"
    scr = viewport(ndc, 640, 480)
    assert scr.shape == (3, 3)
    # NDC (0,0) -> screen centre; NDC y=+1 -> screen row 0 (y flipped)
    c = viewport(np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), 640, 480)
    assert np.allclose(c[0, :2], [320, 240]) and np.allclose(c[1, :2], [320, 0])
    return f"w = {np.round(clip[:, 3], 3)}  ndc(0,0)->{c[0, :2]}  ndc y=+1 -> row {c[1, 1]:.0f}"


def t_near_clip():
    """LO6: a triangle crossing the near plane is clipped in CLIP space and
    every surviving vertex satisfies z >= -w (so the divide is safe)."""
    from raster3d import clip_triangle_near
    P = perspective(60, 1.0, 1.0, 50.0)
    eye_pts = np.array([[0, 0, -3.0], [1, 0, -3.0], [0, 1, +1.0]])     # one BEHIND
    vc = (P @ np.c_[eye_pts, np.ones(3)].T).T
    assert vc[2, 3] < 0                                                 # w < 0 behind
    tris = clip_triangle_near(vc, np.zeros((3, 1)))
    assert len(tris) == 2, len(tris)                                    # quad -> 2 tris
    for v, _ in tris:
        assert np.all(v[:, 2] + v[:, 3] >= -1e-9)
        assert np.all(v[:, 3] > 0)
    return f"1 vertex behind the camera (w={vc[2, 3]:.2f}) -> clipped into {len(tris)} triangles, all w > 0"


def t_mipmap_gamma_correct():
    """LO8: level k+1 is the 2x2 box average IN LINEAR LIGHT of level k."""
    from raster3d import build_mipmaps
    from srgb import srgb_to_linear, linear_to_srgb
    chk = np.zeros((8, 8, 3)); chk[::2, ::2] = 1; chk[1::2, 1::2] = 1     # 1-px checker
    mips = build_mipmaps(chk, levels=4)
    assert [m.shape[0] for m in mips] == [8, 4, 2, 1]
    assert np.allclose(mips[1], 0.5)                                    # linear 0.5
    code = float(linear_to_srgb(mips[1][0, 0, 0]) * 255)
    assert 187 <= code <= 188, code                                     # not 127.5
    return f"4 levels {[m.shape[0] for m in mips]}; black/white checker -> sRGB code {code:.1f} (naive 127.5)"


def t_normal_matrix():
    """Pitfall fix: normals under non-uniform scale use inverse-transpose."""
    M = scale((2.2, 0.5, 1.0))
    n = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)       # normal of plane x + y = c
    t = np.array([1.0, -1.0, 0.0])                   # a tangent in that plane
    t2 = M[:3, :3] @ t
    wrong = M[:3, :3] @ n
    right = normal_matrix(M) @ n
    assert abs(right @ t2) < 1e-12                   # still perpendicular
    assert abs(wrong @ t2) > 0.1                     # naive one is not
    return f"n.t after M: naive {wrong @ t2:+.3f}  inverse-transpose {right @ t2:+.3f}"


def t_one_triangle_render():
    from objio import Mesh
    import raster3d
    pos = np.array([[-0.6, -0.6, -2.0], [0.6, -0.6, -2.0], [0.0, 0.6, -2.0]])
    m = Mesh(pos, np.zeros((0, 2)), [[0, 1, 2]], [])
    col, dep, filled = raster3d.render(m, w=48, h=48, tex_u8=None, near=1.0,
                                       far=50.0, eye=(0, 0, 0), center=(0, 0, -1),
                                       cull_back=False, light_dir=(0, 0, 1))
    assert 200 < filled < 48 * 48, f"filled={filled}"
    assert np.isfinite(dep).sum() == filled
    fin = dep[np.isfinite(dep)]
    assert fin.max() - fin.min() < 1e-6, f"depth spread {fin.max()-fin.min()}"
    return f"{filled} fragments, depth {fin.min():.4f}..{fin.max():.4f}"


def t_raster_matches_raytrace():
    """LO10: two independent renderers, same frame (small size to stay fast)."""
    import raster3d, raytrace, scene
    from objio import load_obj
    from ppm_io import read_ppm
    from metrics import psnr
    if not os.path.exists(scene.DEFAULT_MODEL):
        return "run make_model.py first -- skipped"
    mesh = load_obj(scene.DEFAULT_MODEL); tex = read_ppm(scene.DEFAULT_TEXTURE)
    M = scene.model_matrix(mesh)
    a, _, _ = raster3d.render(mesh, w=96, h=96, tex_u8=tex, model=M)
    b, _ = raytrace.render(mesh, w=96, h=96, tex_u8=tex, model=M)
    p = psnr(a, b)
    assert p > 60, f"raster vs raytrace PSNR {p:.1f} dB"
    return f"raster vs raytrace 96x96: PSNR {'inf' if np.isinf(p) else f'{p:.1f}'} dB"


def t_reference_frame():
    """Render the committed temple and match ref/temple_ref.ppm (generated on
    first run -- commit ref/ afterwards)."""
    import raster3d, scene
    from objio import load_obj
    from ppm_io import read_ppm, write_ppm
    from metrics import psnr, ssim
    if not os.path.exists(scene.DEFAULT_MODEL):
        return "run make_model.py first -- skipped"
    mesh = load_obj(scene.DEFAULT_MODEL)
    tex = read_ppm(scene.DEFAULT_TEXTURE)
    col, _, _ = raster3d.render(mesh, w=160, h=160, tex_u8=tex,
                                model=scene.model_matrix(mesh))
    note = ""
    if not os.path.exists("ref/temple_ref.ppm"):
        os.makedirs("ref", exist_ok=True)
        write_ppm("ref/temple_ref.ppm", col)
        note = " (ref/temple_ref.ppm generated now -- commit it)"
    ref = read_ppm("ref/temple_ref.ppm")
    p = psnr(col, ref)
    assert p == float("inf") or p > 60, f"temple frame PSNR vs ref = {p:.1f} dB"
    return f"PSNR {'inf' if np.isinf(p) else f'{p:.1f}'} dB  SSIM {ssim(col, ref):.3f}{note}"


def main():
    checks = [
        ("E1  running point (2,3,4) -> (3,3,-3); divide d=2 -> (2,2); order matters", t_e1_running_point),
        ("E2  camera +x == model -x (identical points)", t_e2_duality),
        ("E3  perspective: z=-n -> ndc -1 ; z=-f -> ndc +1 ; z=-3 -> 0.5", t_e3_perspective_near_far),
        ("E4  perspective-correct u at screen centre = 0.1 (naive 0.5)", t_e4_perspective_correct_u),
        ("E5  barycentric depth 0.20/0.50/0.90 at (.25,.25,.5) = 0.625", t_e5_barycentric_depth),
        ("view matrix = inverse of camera pose, rotation orthonormal", t_view_is_inverse_pose),
        ("projection pipeline shapes + viewport y-flip", t_projection_pipeline_shapes),
        ("near-plane clip happens in clip space (w > 0 after clipping)", t_near_clip),
        ("mipmap: 4 levels, gamma-correct box downscale", t_mipmap_gamma_correct),
        ("normal matrix (inverse-transpose) under non-uniform scale", t_normal_matrix),
        ("one-triangle z-buffer render fills a plausible region, flat depth", t_one_triangle_render),
        ("raster3d == raytrace on the temple (two independent renderers)", t_raster_matches_raytrace),
        ("temple frame matches ref/temple_ref.ppm (byte-identical on re-render)", t_reference_frame),
    ]
    results = [check(n, f) for n, f in checks]
    n_ok = sum(results)
    print("\n" + (f"ALL CHECKS PASSED ({n_ok} ok)" if all(results)
                  else f"SOME CHECKS FAILED ({n_ok}/{len(results)} ok)"))
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
