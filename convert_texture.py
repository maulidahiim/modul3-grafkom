"""Module 3 -- convert your texture (PNG/JPG from Blender, a photo, ...) to the
binary P6 PPM the from-scratch renderers read. Asset preparation only: the
pipeline itself still never touches an imaging library.

  python convert_texture.py my_texture.png data/mymodel.ppm --size 512

--size resizes to a square power of two (mip-friendly). Requires Pillow.
"""
from __future__ import annotations

import argparse

import numpy as np

from ppm_io import write_ppm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--size", type=int, default=0, help="square size, e.g. 256 or 512 (0 = keep)")
    a = ap.parse_args()
    try:
        from PIL import Image
    except ImportError:
        raise SystemExit("Pillow is needed for this helper: pip install pillow")
    im = Image.open(a.src).convert("RGB")
    if a.size:
        if a.size & (a.size - 1):
            raise SystemExit("--size must be a power of two (256, 512, 1024)")
        im = im.resize((a.size, a.size), Image.LANCZOS)
    arr = np.asarray(im, dtype=np.uint8)
    n = write_ppm(a.dst, arr)
    print(f"wrote {a.dst}  ({arr.shape[1]}x{arr.shape[0]}, {n} bytes)")


if __name__ == "__main__":
    main()
