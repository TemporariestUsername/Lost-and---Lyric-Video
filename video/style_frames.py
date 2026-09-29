"""Render the approval style frames to storyboard/frames/*.png.

    python3 video/style_frames.py
"""
import pathlib
import sys

import skia

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from engine import H, ROOT, W, Timing  # noqa: E402
import scenes  # noqa: E402

FRAMES = [
    ("A_white_coats", 126.0, scenes.scene_white_coats),
    ("B_spoken_help", 189.6, scenes.scene_spoken_help),
    ("C_the_run", 294.7, scenes.scene_run),
]


def main():
    out = ROOT / "storyboard" / "frames"
    out.mkdir(parents=True, exist_ok=True)
    T = Timing()
    only = sys.argv[1:]
    for name, t, fn in FRAMES:
        if only and not any(o in name for o in only):
            continue
        surf = skia.Surface(W, H)
        fn(surf.getCanvas(), t, T)
        img = surf.makeImageSnapshot()
        img.save(str(out / f"{name}.png"), skia.kPNG)
        print("wrote", out / f"{name}.png", f"t={t}")


if __name__ == "__main__":
    main()
