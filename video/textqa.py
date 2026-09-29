"""Fast QA of a section's typography alone (no picture), every frame.

    python3 video/textqa.py "There comes a once"
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import check  # noqa: E402
import compose  # noqa: E402
import kinetic  # noqa: E402
import sections as S  # noqa: E402
from engine import FPS, ROOT, Timing  # noqa: E402


def main(name):
    T = Timing()
    sec = next(s for s in T.sections if s["name"] == name)
    fn, lines = compose.SCENES[name]
    lines = [n for n in lines if n in T.lines]
    k0, k1 = int(np.ceil(sec["start"] * FPS)), int(np.ceil(sec["end"] * FPS))
    fn(k0 / FPS, T, lines)                    # builds the section's Kinetic
    K = S._CACHE.get(("kin", name)) or compose._KIN.get(name)
    if K is None:
        print(name, ": no typography")
        return
    kinetic.TEXT_LOG = []
    img = np.ones((1080, 1920, 3), np.float32) * 0.9
    for k in range(k0, k1):
        K.draw(img, k / FPS)
    qa = dict(frames=[(k, 128.0, 0.0, True) for k in range(k0, k1)], text=kinetic.TEXT_LOG,
              start=k0 / FPS, end=k1 / FPS)
    p = ROOT / "render" / ".textqa.mp4"
    p.with_suffix(".qa.json").write_text(json.dumps(qa))
    check.main(p)


if __name__ == "__main__":
    for n in sys.argv[1:]:
        main(n)
