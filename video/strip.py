"""Quick draft strip: render a section at a coarse step (sequentially, so
the exposure echo is roughly right) and tile N evenly spaced frames.

    python3 video/strip.py "Verse A: Through eyes" out.jpg [step]
"""
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from compose import Compositor  # noqa: E402
from engine import Timing  # noqa: E402


def main(name, out, step=0.4, n=12):
    T = Timing()
    sec = next(s for s in T.sections if s["name"] == name)
    c = Compositor(T)
    ts = np.arange(sec["start"] - 2, sec["end"], step)
    want = set(np.linspace(0, len(ts) - 1, n).round().astype(int))
    tiles = []
    for i, t in enumerate(ts):
        img = c.frame(float(t), scene=name)
        if i in want and t >= sec["start"]:
            im = Image.fromarray(np.ascontiguousarray(img[..., :3])).resize((640, 360))
            ImageDraw.Draw(im).text((8, 336), f"{t:6.2f}s", fill=(40, 30, 50),
                                    font=ImageFont.truetype("/usr/share/fonts/truetype/ibm-plex/IBMPlexMono-Regular.ttf", 16))
            tiles.append(np.array(im))
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
    Image.fromarray(grid).save(out, quality=85)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 0.4)
