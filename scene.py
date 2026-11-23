"""Module 3 -- ONE source of truth for the scene: camera, light, model matrix.

raster3d.py, raytrace.py, tests.py, broken_scene.py, bench.py and the WebGL
viewer (via build_web.py -> web/assets/scene.json) all read from here, so the
frames they produce are comparable pixel-for-pixel. Change the camera here and
every renderer follows.

Model matrix chain (column vectors, right to left):
    M = Ry(YAW_DEG) . S(SCALE) . FIT
FIT moves the model so its bounding-box bottom-centre sits at the origin and
scales it uniformly to fit inside FIT_BOX (the fallback temple's own box,
2.0 x 1.52 x 2.0). For the temple FIT is exactly the identity, so the
reference frame does not change; for YOUR model (any units / offset Blender
exported, tall or wide) it lands in the same framed spot.
"""
from __future__ import annotations

import os

import numpy as np

from transforms import rot_y, scale, translate

HERE = os.path.dirname(os.path.abspath(__file__))

CAMERA = dict(eye=(2.6, 2.0, 3.4), center=(0.0, 0.7, 0.0), fovy=45.0,
              near=1.0, far=14.0)
LIGHT = dict(light_dir=(0.4, 0.9, 0.6), ambient=0.15)
YAW_DEG = 35.0
SCALE = 1.4
FIT_BOX = (2.0, 1.52, 2.0)

DEFAULT_MODEL = os.path.join(HERE, "data", "pusaka_model.obj")
DEFAULT_TEXTURE = os.path.join(HERE, "data", "relief.ppm")


def fit_matrix(positions, box=FIT_BOX):
    """T then S: bottom-centre of the bbox -> origin, then one uniform scale
    so the model fits inside `box` (uniform -> no stretching)."""
    p = np.asarray(positions, float)
    lo, hi = p.min(0), p.max(0)
    size = hi - lo
    if not np.any(size > 0):
        return np.eye(4)
    base = np.array([(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2])
    k = float(min(b / s for b, s in zip(box, size) if s > 0))
    return scale((k, k, k)) @ translate(-base)


def model_matrix(mesh=None, fit=True):
    M = rot_y(YAW_DEG) @ scale((SCALE, SCALE, SCALE))
    if fit and mesh is not None:
        M = M @ fit_matrix(mesh.positions)
    return M


def texture_for(model_path, tex_arg=None):
    """--tex wins; else <model>.ppm next to the OBJ; else data/relief.ppm."""
    if tex_arg:
        return tex_arg
    stem = os.path.splitext(model_path)[0] + ".ppm"
    if os.path.exists(stem):
        return stem
    return DEFAULT_TEXTURE if os.path.exists(DEFAULT_TEXTURE) else None
