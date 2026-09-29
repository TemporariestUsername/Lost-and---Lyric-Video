"""Render the approval style frames to storyboard/frames/*.png.

Each still is rendered as the last frame of a 3 s sequential warm-up so the
exposure echo carries real history, exactly as in the final render.

    python3 video/style_frames.py [name-filter ...]
"""
import pathlib
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from engine import FPS, ROOT, Timing  # noqa: E402
from compose import Compositor  # noqa: E402

FRAMES = [
    ("A_lost_inside_him", 46.3),
    ("B_stab_the_charts", 125.9),
    ("C_cut_away", 368.3),
]
WARMUP = 3.0


def render_still(T, t, step=1):
    comp = Compositor(T)
    n = int(WARMUP * FPS)
    img = None
    for i in range(0, n + 1, step):
        img = comp.frame(t - (n - i) / FPS)
    return img


def main():
    out = ROOT / "storyboard" / "frames"
    out.mkdir(parents=True, exist_ok=True)
    T = Timing()
    only = sys.argv[1:]
    for name, t in FRAMES:
        if only and not any(o in name for o in only):
            continue
        t0 = time.time()
        img = render_still(T, t, step=2)
        Image.fromarray(np.ascontiguousarray(img[..., :3])).save(out / f"{name}.png")
        print(f"wrote {name}.png t={t} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
