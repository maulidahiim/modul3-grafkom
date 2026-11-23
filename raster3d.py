"""Module 3 -- the from-scratch rasteriser: MVP -> near clip -> w-divide ->
viewport -> barycentric z-buffer fill -> perspective-correct texture
(bilinear, optional trilinear mipmap) -> Lambert (optional shadow map).

NumPy only. This is Module 2's edge-function triangle plus a depth test and
perspective-correct attribute interpolation.

Run:
  python raster3d.py --model data/pusaka_model.obj --w 512 --h 512
  python raster3d.py --model data/mymodel.obj --tex data/mymodel.ppm
  python raster3d.py --model ... --mip 4          # trilinear, 4-level pyramid
  python raster3d.py --model ... --naive-uv       # WRONG on purpose (E4)
  python raster3d.py --model ... --cull --shadow
  python raster3d.py --demo-triangle --w 64 --h 64
  python raster3d.py --demo-overlap  --w 64 --h 64

Texel convention (same as OpenGL GL_LINEAR + GL_REPEAT): texel centres sit
at (i + 0.5) / W, so the sample position is  x = u*W - 0.5  and both
neighbours wrap. Using u*(W-1) instead shifts the texture by up to half a
texel against WebGL.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from transforms import perspective, look_at, ortho
from objio import load_obj
from ppm_io import read_ppm, write_ppm
from srgb import srgb_to_linear, linear_to_srgb
import scene


# ------------------------------------------------------------ texture sampling
def sample_bilinear(tex_lin, u, v):
    """tex_lin: (H,W,3) float in LINEAR light. u,v: arrays (any range; REPEAT
    wrap). Returns (...,3). Blending happens in linear light (Module 1)."""
    H, W = tex_lin.shape[:2]
    x = np.mod(u, 1.0) * W - 0.5
    y = np.mod(v, 1.0) * H - 0.5
    x0f = np.floor(x); y0f = np.floor(y)
    fx = (x - x0f)[..., None]; fy = (y - y0f)[..., None]
    x0 = np.mod(x0f.astype(int), W); x1 = np.mod(x0 + 1, W)
    y0 = np.mod(y0f.astype(int), H); y1 = np.mod(y0 + 1, H)
    top = tex_lin[y0, x0] * (1 - fx) + tex_lin[y0, x1] * fx
    bot = tex_lin[y1, x0] * (1 - fx) + tex_lin[y1, x1] * fx
    return top * (1 - fy) + bot * fy


def build_mipmaps(tex_lin, levels=4):
    """Mip pyramid: each level is a 2x2 BOX average of the previous one, done
    in LINEAR light (gamma-correct downscale, Module 1). levels counts level 0.
    Odd sizes drop their last row/column before averaging."""
    mips = [np.asarray(tex_lin, float)]
    while len(mips) < levels:
        t = mips[-1]
        h, w = t.shape[0] // 2 * 2, t.shape[1] // 2 * 2
        if h < 2 or w < 2:
            break
        t = t[:h, :w]
        mips.append(0.25 * (t[0::2, 0::2] + t[1::2, 0::2] +
                            t[0::2, 1::2] + t[1::2, 1::2]))
    return mips


def sample_trilinear(mips, u, v, lod):
    """Bilinear in the two nearest levels, linear blend by frac(lod)."""
    L = len(mips)
    lod = np.clip(lod, 0.0, L - 1)
    l0 = np.floor(lod).astype(int)
    l1 = np.minimum(l0 + 1, L - 1)
    t = (lod - l0)[..., None]
    out = np.zeros(u.shape + (3,))
    for k in range(L):
        m0 = l0 == k
        if m0.any():
            out[m0] += (1 - t[m0]) * sample_bilinear(mips[k], u[m0], v[m0])
        m1 = (l1 == k)
        if m1.any():
            out[m1] += t[m1] * sample_bilinear(mips[k], u[m1], v[m1])
    return out


# ---------------------------------------------------------------- near clipping
def clip_triangle_near(vc, attrs):
    """Sutherland-Hodgman clip one triangle (clip-space vc: (3,4)) against the
    near plane  z + w >= 0.  attrs: (3, k) per-vertex attributes to interpolate.
    Clipping happens BEFORE the divide, while w still means 'distance in front'.
    Returns a list of (vc', attrs') triangles (0, 1 or 2)."""
    def inside(p):
        return p[2] + p[3] >= 0.0
    poly = [(vc[i], attrs[i]) for i in range(3)]
    out = []
    n = len(poly)
    for i in range(n):
        cur_v, cur_a = poly[i]
        prv_v, prv_a = poly[(i - 1) % n]
        ci, pi = inside(cur_v), inside(prv_v)
        if ci:
            if not pi:
                t = (prv_v[2] + prv_v[3]) / ((prv_v[2] + prv_v[3]) - (cur_v[2] + cur_v[3]))
                out.append((prv_v + t * (cur_v - prv_v), prv_a + t * (cur_a - prv_a)))
            out.append((cur_v, cur_a))
        elif pi:
            t = (prv_v[2] + prv_v[3]) / ((prv_v[2] + prv_v[3]) - (cur_v[2] + cur_v[3]))
            out.append((prv_v + t * (cur_v - prv_v), prv_a + t * (cur_a - prv_a)))
    if len(out) < 3:
        return []
    tris = []
    for k in range(1, len(out) - 1):
        vv = np.stack([out[0][0], out[k][0], out[k + 1][0]])
        aa = np.stack([out[0][1], out[k][1], out[k + 1][1]])
        tris.append((vv, aa))
    return tris


# ------------------------------------------------------------------ shadow map
def light_space(world_pts, light_dir):
    """Orthographic light camera that encloses the whole model (directional
    light). Returns (Vlight, Plight)."""
    L = np.asarray(light_dir, float); L = L / np.linalg.norm(L)
    lo, hi = world_pts.min(0), world_pts.max(0)
    c = (lo + hi) / 2
    r = float(np.linalg.norm(hi - lo)) / 2 + 1e-6
    up = (0, 0, 1) if abs(L[1]) > 0.99 else (0, 1, 0)
    Vl = look_at(c + L * (2 * r + 1), c, up)
    Pl = ortho(-r, r, -r, r, 0.5, 4 * r + 2)
    return Vl, Pl


# --------------------------------------------------------------- the rasteriser
def render(mesh, *, w, h, tex_u8=None, model=None, tri_colors=None,
           eye=scene.CAMERA["eye"], center=scene.CAMERA["center"],
           fovy=scene.CAMERA["fovy"], near=scene.CAMERA["near"],
           far=scene.CAMERA["far"], light_dir=scene.LIGHT["light_dir"],
           ambient=scene.LIGHT["ambient"], cull_back=False, depth_bits=None,
           perspective_correct=True, mip_levels=None, shadow=False,
           shadow_res=512, shadow_bias=0.01, view=None, proj=None,
           depth_mode="float", stats=None):
    """Returns (color uint8 (h,w,3), depth float (h,w) NDC z with +inf = empty,
    filled = number of fragments that PASSED the depth test, overdraw included).

    depth_bits          quantise stored depth (low-precision buffer, z-fight lab)
    depth_mode          "float"  : NDC z in float64 (default)
                        "f32"    : NDC z stored as float32
                        "reverse-z-f32": reverse-Z with infinite far, stored
                                   as float32: d = near / w_eye (1 at near,
                                   -> 0 far), test GREATER. Implemented as
                                   LESS on -d so the buffer code is shared.
    perspective_correct False -> screen-linear UV (the PS1 'swimming' bug, E4)
    mip_levels          None -> plain bilinear; k -> trilinear over k levels
    shadow              one hard shadow from the directional light (shadow map)
    view / proj         override V / P (used for the light's depth pass)
    stats               optional dict, receives counters (culled, tested, ...)
    """
    if model is None:
        model = np.eye(4)
    eye_pos = np.asarray(eye, float)
    P = perspective(fovy, w / h, near, far) if proj is None else proj
    V = look_at(eye, center) if view is None else view
    pos_h = np.c_[mesh.positions, np.ones(len(mesh.positions))]
    world = (model @ pos_h.T).T[:, :3]
    clip = (P @ V @ model @ pos_h.T).T                       # (Vn,4)

    tris_pos = mesh.faces                      # (F,3) position idx
    tris_uv = mesh.triangle_uvs()              # (F,3,2)
    L = np.asarray(light_dir, float); L = L / np.linalg.norm(L)

    tex_lin = mips = None
    if tex_u8 is not None:
        tex_lin = srgb_to_linear(tex_u8.astype(np.float64) / 255.0)
        if mip_levels:
            mips = build_mipmaps(tex_lin, mip_levels)
    TH, TW = (tex_lin.shape[:2] if tex_lin is not None else (1, 1))

    smap = None
    if shadow:
        Vl, Pl = light_space(world, L)
        _, smap, _ = render(mesh, w=shadow_res, h=shadow_res, model=model,
                            view=Vl, proj=Pl, near=near, far=far)
        LPV = Pl @ Vl

    color = np.zeros((h, w, 3), np.float64)
    depth = np.full((h, w), np.inf)
    filled = 0
    n_culled = n_tested = 0

    for fi in range(len(tris_pos)):
        vidx = tris_pos[fi]
        vc = clip[vidx]                        # (3,4)
        # flat normal + Lambert term in world space
        wp = world[vidx]
        nrm = np.cross(wp[1] - wp[0], wp[2] - wp[0])
        nl = np.linalg.norm(nrm)
        if nl < 1e-12:
            continue
        nrm = nrm / nl
        # orient the normal toward the camera (shade the side we actually see)
        if nrm @ (eye_pos - wp[0]) < 0:
            nrm = -nrm
        ndotl = max(0.0, float(nrm @ L))

        uv = tris_uv[fi]                       # (3,2)
        # attrs carried through clipping: [u, v, 1, world xyz]
        attrs = np.c_[uv, np.ones(3), wp]
        for (cvc, cattr) in (clip_triangle_near(vc, attrs) or []):
            wv = cvc[:, 3]
            ndc = cvc[:, :3] / wv[:, None]                     # the w-divide
            sx = (ndc[:, 0] * 0.5 + 0.5) * w                   # viewport
            sy = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * h
            sz = ndc[:, 2]
            inv_w = 1.0 / wv
            uvw = cattr[:, :2] * inv_w[:, None]      # u/w, v/w
            pw = cattr[:, 3:6] * inv_w[:, None]      # world/w (for shadows)

            minx = max(0, int(np.floor(sx.min())))
            maxx = min(w - 1, int(np.ceil(sx.max())))
            miny = max(0, int(np.floor(sy.min())))
            maxy = min(h - 1, int(np.ceil(sy.max())))
            if minx > maxx or miny > maxy:
                continue

            ax, ay = sx[0], sy[0]
            bx, by = sx[1], sy[1]
            cx, cy = sx[2], sy[2]
            # area = edge function of C about edge A->B (2x signed triangle area)
            area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
            if area == 0:
                continue
            if cull_back and area > 0:          # screen y-down: CCW-front -> area<0
                n_culled += 1
                continue

            ys, xs = np.mgrid[miny:maxy + 1, minx:maxx + 1]
            px = xs + 0.5; py = ys + 0.5

            def bary(px, py):
                # edge functions / area -> barycentric weights of vertices 0,1,2
                lA = ((cx - bx) * (py - by) - (cy - by) * (px - bx)) / area
                lB = ((ax - cx) * (py - cy) - (ay - cy) * (px - cx)) / area
                return lA, lB, 1.0 - lA - lB

            l0, l1, l2 = bary(px, py)
            inside = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
            if not inside.any():
                continue

            # depth: NDC z is affine in screen space -> plain barycentric
            frag_z = l0 * sz[0] + l1 * sz[1] + l2 * sz[2]
            inv_w_i = l0 * inv_w[0] + l1 * inv_w[1] + l2 * inv_w[2]
            if depth_bits is not None:
                q = float(2 ** depth_bits)
                frag_z = np.round((frag_z * 0.5 + 0.5) * q) / q * 2.0 - 1.0
            elif depth_mode == "f32":
                frag_z = frag_z.astype(np.float32).astype(np.float64)
            elif depth_mode == "reverse-z-f32":
                # 1/w is affine in screen space, so near/w interpolates exactly
                frag_z = -(near * inv_w_i).astype(np.float32).astype(np.float64)

            reg = depth[miny:maxy + 1, minx:maxx + 1]
            n_tested += int(inside.sum())
            win = inside & (frag_z < reg) & (frag_z <= 1.0)     # z-test + far
            if not win.any():
                continue
            reg[win] = frag_z[win]

            if tex_lin is not None:
                def uv_at(b0, b1, b2):
                    if perspective_correct:              # (u/w)/(1/w)
                        iw = b0 * inv_w[0] + b1 * inv_w[1] + b2 * inv_w[2]
                        uu = (b0 * uvw[0, 0] + b1 * uvw[1, 0] + b2 * uvw[2, 0]) / iw
                        vv = (b0 * uvw[0, 1] + b1 * uvw[1, 1] + b2 * uvw[2, 1]) / iw
                    else:                                # screen-linear (WRONG)
                        uu = b0 * cattr[0, 0] + b1 * cattr[1, 0] + b2 * cattr[2, 0]
                        vv = b0 * cattr[0, 1] + b1 * cattr[1, 1] + b2 * cattr[2, 1]
                    return uu, vv
                u_i, v_i = uv_at(l0, l1, l2)
                if mips is not None:
                    # texel footprint of one pixel: UV at the +x and +y neighbours
                    ux, vx = uv_at(*bary(px + 1, py))
                    uy, vy = uv_at(*bary(px, py + 1))
                    rho = np.maximum(np.hypot((ux - u_i) * TW, (vx - v_i) * TH),
                                     np.hypot((uy - u_i) * TW, (vy - v_i) * TH))
                    lod = np.log2(np.maximum(rho, 1e-8))
                    tex_rgb = sample_trilinear(mips, u_i, v_i, lod)
                else:
                    tex_rgb = sample_bilinear(tex_lin, u_i, v_i)  # linear light
            elif tri_colors is not None:
                tex_rgb = np.ones((*l0.shape, 3)) * srgb_to_linear(
                    np.asarray(tri_colors[fi], float) / 255.0)
            else:
                tex_rgb = np.ones((*l0.shape, 3)) * 0.7

            if smap is not None and ndotl > 0:
                # perspective-correct world position -> light clip -> map lookup
                wpos = np.stack([(l0 * pw[0, k] + l1 * pw[1, k] + l2 * pw[2, k]) / inv_w_i
                                 for k in range(3)], -1)
                lc = (LPV @ np.c_[wpos.reshape(-1, 3), np.ones(wpos[..., 0].size)].T).T
                lx = ((lc[:, 0] * 0.5 + 0.5) * shadow_res).astype(int)
                ly = ((1 - (lc[:, 1] * 0.5 + 0.5)) * shadow_res).astype(int)
                ok = (lx >= 0) & (lx < shadow_res) & (ly >= 0) & (ly < shadow_res)
                occ = np.zeros(len(lc), bool)
                occ[ok] = lc[ok, 2] > smap[ly[ok], lx[ok]] + shadow_bias
                lit_amt = np.where(occ.reshape(l0.shape), 0.0, ndotl)
                shade = (ambient + (1 - ambient) * lit_amt)[..., None]
            else:
                shade = ambient + (1 - ambient) * ndotl

            lit = np.clip(tex_rgb * shade, 0, 1)
            cregion = color[miny:maxy + 1, minx:maxx + 1]
            cregion[win] = linear_to_srgb(lit)[win]
            filled += int(win.sum())

    if stats is not None:
        stats.update(culled_tris=n_culled, coverage_tested=n_tested,
                     depth_passed=filled)
    return (np.clip(np.round(color * 255), 0, 255).astype(np.uint8), depth, filled)


def depth_image(depth):
    """Depth buffer -> grey PPM, normalised to the drawn range (near=white).
    Empty pixels are black."""
    fin = np.isfinite(depth)
    out = np.zeros(depth.shape, np.uint8)
    if fin.any():
        lo, hi = depth[fin].min(), depth[fin].max()
        t = (depth[fin] - lo) / max(hi - lo, 1e-12)
        out[fin] = np.round(255 * (1.0 - 0.85 * t)).astype(np.uint8)
    return np.repeat(out[..., None], 3, axis=2)


def depth_path_for(out):
    d, b = os.path.split(out)
    b = b.replace("raster", "depth") if "raster" in b else \
        os.path.splitext(b)[0] + "_depth.ppm"
    return os.path.join(d, b)


# ------------------------------------------------------------------------ demos
def _demo_triangle(w, h):
    """One textured triangle -- reproduces Fully Worked Example Step 2."""
    from objio import Mesh
    from transforms import project, viewport
    pos = np.array([[-0.8, -0.6, -2.0], [0.9, -0.7, -3.5], [-0.2, 0.9, -2.5]])
    uv = np.array([[0, 0], [1, 0], [0.5, 1.0]])
    m = Mesh(pos, uv, [[0, 1, 2]], [[0, 1, 2]])
    tex = read_ppm(scene.DEFAULT_TEXTURE) if os.path.exists(scene.DEFAULT_TEXTURE) else None
    P = perspective(60, w / h, 1.0, 50.0)
    clip, ndc = project(P, np.eye(4), np.eye(4), pos)
    scr = viewport(ndc, w, h)
    print(f"triangle screen verts: "
          f"({scr[0,0]:.0f},{scr[0,1]:.0f}) ({scr[1,0]:.0f},{scr[1,1]:.0f}) "
          f"({scr[2,0]:.0f},{scr[2,1]:.0f})   uv: (0,0) (1,0) (0.5,1)")
    col, dep, filled = render(m, w=w, h=h, tex_u8=tex, near=1.0, far=50.0,
                              eye=(0, 0, 0), center=(0, 0, -1), cull_back=False,
                              light_dir=(0, 0, 1))
    fin = np.isfinite(dep)
    if fin.any():
        print(f"filled fragments: {filled}   depth range "
              f"[{dep[fin].min():.2f}, {dep[fin].max():.2f}]")
    # perspective-correct vs naive UV at the SCREEN centroid
    invw = 1.0 / clip[:, 3]
    uvw = uv * invw[:, None]
    pc = (uvw.mean(0)) / invw.mean()
    print(f"perspective-correct UV at centroid: ({pc[0]:.3f}, {pc[1]:.3f})   "
          f"(naive linear: ({uv[:,0].mean():.3f}, {uv[:,1].mean():.3f}))")
    os.makedirs("out", exist_ok=True)
    write_ppm("out/tri_raster.ppm", col)
    print("wrote out/tri_raster.ppm  (texture: data/relief.ppm, bilinear)")


def _demo_overlap(w, h):
    from objio import Mesh
    front = [[-0.7, -0.7, -1.5], [0.7, -0.7, -1.5], [0.0, 0.7, -1.5]]
    back = [[-0.2, -0.7, -3.0], [1.2, -0.7, -3.0], [0.5, 0.7, -3.0]]
    kw = dict(w=w, h=h, near=1.0, far=50.0, eye=(0, 0, 0), center=(0, 0, -1),
              cull_back=False, light_dir=(0, 0, 1))
    both = Mesh(np.array(front + back), np.zeros((0, 2)), [[0, 1, 2], [3, 4, 5]], [])
    col, dep, _ = render(both, tri_colors=[(230, 130, 40), (60, 120, 210)], **kw)
    _, depf, _ = render(Mesh(np.array(front), np.zeros((0, 2)), [[0, 1, 2]], []), **kw)
    _, depb, fb = render(Mesh(np.array(back), np.zeros((0, 2)), [[0, 1, 2]], []), **kw)
    zf = depf[np.isfinite(depf)].mean(); zb = depb[np.isfinite(depb)].mean()
    overlap = np.isfinite(depf) & np.isfinite(depb)
    back_won = int(np.sum(overlap & (dep == depb) & (depb != depf)))
    back_only = int(np.sum(np.isfinite(depb) & ~np.isfinite(depf)))
    print(f"front triangle depth ~{(zf + 1) / 2:.2f}, back triangle depth ~{(zb + 1) / 2:.2f}"
          f"  (depth in [0,1]; NDC z {zf:+.2f} / {zb:+.2f})")
    print(f"back triangle alone covers {fb} px; {int(overlap.sum())} of them overlap the front")
    print(f"pixels where BACK won the z-test: {back_won}    (front fully occludes the overlap)")
    print(f"pixels where BACK is visible (no overlap): {back_only}")
    os.makedirs("out", exist_ok=True)
    write_ppm("out/overlap.ppm", col)
    print("wrote out/overlap.ppm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--tex", help="P6 .ppm texture (default: <model>.ppm or data/relief.ppm)")
    ap.add_argument("--w", type=int, default=512)
    ap.add_argument("--h", type=int, default=512)
    ap.add_argument("--out", default="out/temple_raster.ppm")
    ap.add_argument("--mip", type=int, default=0, help="mip levels (0 = off)")
    ap.add_argument("--naive-uv", action="store_true", help="screen-linear UV (bug demo)")
    ap.add_argument("--cull", action="store_true", help="back-face culling (CCW front)")
    ap.add_argument("--shadow", action="store_true", help="hard shadow via shadow map")
    ap.add_argument("--no-fit", action="store_true", help="do not auto-fit the model")
    ap.add_argument("--demo-triangle", action="store_true")
    ap.add_argument("--demo-overlap", action="store_true")
    args = ap.parse_args()

    if args.demo_triangle:
        _demo_triangle(args.w, args.h); return
    if args.demo_overlap:
        _demo_overlap(args.w, args.h); return

    if not args.model:
        raise SystemExit("give --model, or --demo-triangle / --demo-overlap")
    mesh = load_obj(args.model)
    tex_path = scene.texture_for(args.model, args.tex)
    tex = read_ppm(tex_path) if tex_path else None
    model = scene.model_matrix(mesh, fit=not args.no_fit)
    st = {}
    t0 = time.perf_counter()
    col, dep, filled = render(mesh, w=args.w, h=args.h, tex_u8=tex, model=model,
                              mip_levels=args.mip or None, cull_back=args.cull,
                              perspective_correct=not args.naive_uv,
                              shadow=args.shadow, stats=st)
    dt = time.perf_counter() - t0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_ppm(args.out, col)
    dpath = depth_path_for(args.out)
    write_ppm(dpath, depth_image(dep))
    fin = np.isfinite(dep)
    print(f"model: {mesh.n_tris} triangles   framebuffer {args.w}x{args.h}   "
          f"texture: {tex_path}")
    if fin.any():
        print(f"filled fragments: {filled}   visible pixels: {int(fin.sum())}   "
              f"depth(ndc) range [{dep[fin].min():.3f}, {dep[fin].max():.3f}]")
    else:
        print("nothing drawn -- check camera / model matrix / winding")
    if args.cull:
        print(f"back-face culled triangles: {st['culled_tris']}")
    print(f"frame time: {dt*1000:.0f} ms")
    print(f"wrote {args.out} and {dpath}")


if __name__ == "__main__":
    main()
