"""Module 3 -- Lab 3: package the model for the WebGL viewer (and GitHub Pages).

  python build_web.py                                   # data/mymodel.obj if it
                                                        # exists, else the temple
  python build_web.py --model data/mymodel.obj --tex data/mymodel.ppm

Writes web/assets/:
  model.obj      the mesh EXACTLY as the Python renderers see it (winding
                 re-checked by objio.load_obj -> CCW-front), so WebGL can use
                 FrontSide and still match
  texture.<ext>  your texture (.ppm, .png or .jpg)
  scene.json     camera, light and the 4x4 model matrix from scene.py
                 (column-major, the order Three.js Matrix4.fromArray expects)

Then:  npx serve web   ->  open the printed URL, orbit, "Export PPM".
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np

import scene
from objio import load_obj

HERE = os.path.dirname(os.path.abspath(__file__))


def write_obj(path, mesh, title):
    with open(path, "w") as fh:
        fh.write(f"# {title} -- written by build_web.py (CCW-front, v NOT flipped)\n")
        for p in mesh.positions:
            fh.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        for u, v in mesh.uvs:
            fh.write(f"vt {u:.6f} {1.0 - v:.6f}\n")          # undo the load-time flip
        for i, f in enumerate(mesh.faces):
            if mesh.face_uvs is not None:
                t = mesh.face_uvs[i]
                fh.write(f"f {f[0]+1}/{t[0]+1} {f[1]+1}/{t[1]+1} {f[2]+1}/{t[2]+1}\n")
            else:
                fh.write(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")


def main():
    ap = argparse.ArgumentParser()
    mine = os.path.join(HERE, "data", "mymodel.obj")
    ap.add_argument("--model", default=mine if os.path.exists(mine) else scene.DEFAULT_MODEL)
    ap.add_argument("--tex", help=".ppm / .png / .jpg (default: same stem as the model, "
                                  "else data/relief.ppm)")
    ap.add_argument("--size", type=int, default=512, help="canvas / export size (square)")
    ap.add_argument("--out", default=os.path.join(HERE, "web", "assets"))
    a = ap.parse_args()

    if not os.path.exists(a.model):
        raise SystemExit(f"{a.model} not found -- run `python make_model.py` first")
    mesh = load_obj(a.model, verbose=True)
    tex = a.tex
    if not tex:
        stem = os.path.splitext(a.model)[0]
        for ext in (".ppm", ".png", ".jpg", ".jpeg"):
            if os.path.exists(stem + ext):
                tex = stem + ext; break
        else:
            tex = scene.DEFAULT_TEXTURE
    os.makedirs(a.out, exist_ok=True)
    name = os.path.splitext(os.path.basename(a.model))[0]
    write_obj(os.path.join(a.out, "model.obj"), mesh, name)
    tex_ext = os.path.splitext(tex)[1].lower()
    shutil.copyfile(tex, os.path.join(a.out, "texture" + tex_ext))

    M = scene.model_matrix(mesh)
    cfg = dict(
        name=name, model="model.obj", texture="texture" + tex_ext,
        width=a.size, height=a.size,
        camera={k: (list(v) if isinstance(v, tuple) else v) for k, v in scene.CAMERA.items()},
        light=dict(direction=list(scene.LIGHT["light_dir"]), ambient=scene.LIGHT["ambient"]),
        model_matrix_colmajor=[float(x) for x in np.asarray(M).T.reshape(-1)],
        triangles=int(mesh.n_tris),
        conventions="right-handed, camera -z, CCW front, NDC [-1,1], UV origin bottom-left",
    )
    with open(os.path.join(a.out, "scene.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)
    print(f"web assets for '{name}' ({mesh.n_tris} triangles, texture {os.path.basename(tex)})"
          f" -> {os.path.relpath(a.out, HERE)}/  [model.obj, texture{tex_ext}, scene.json]")
    print("next:  npx serve web    (or: python -m http.server -d web 8000)")


if __name__ == "__main__":
    main()
