"""Her visions: real frames rendered from later in the film, cached so a
scene can let the future bleed into the present (she is precognitive).

    python3 video/visions.py          # (re)render the cache
"""
import pathlib
import sys
from functools import lru_cache

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from engine import FPS, H, ROOT, W  # noqa: E402

CACHE = ROOT / "render" / "visions"

# name -> (time in the film, section whose scene renders it)
VISIONS = {
    "corridor": (114.0, "Break I"),
    "coats": (125.9, "White Coats I"),
    "shafts": (189.6, "Spoken I: I can help you find"),
    "bullet": (262.0, "A gun and a bullet"),
    "run": (294.7, "The Run"),
    "cutting": (366.5, "The Cutting"),
}


def render_all():
    from compose import Compositor
    from engine import Timing
    from PIL import Image
    CACHE.mkdir(parents=True, exist_ok=True)
    T = Timing()
    for name, (t, scene) in VISIONS.items():
        c = Compositor(T)
        img = None
        for k in range(45, -1, -3):                # 1.5 s warm-up so the echo is real
            img = c.frame(t - k / FPS, scene=scene)
        Image.fromarray(np.ascontiguousarray(img[..., :3])).save(CACHE / f"{name}.png")
        print("vision", name, t, scene, flush=True)


@lru_cache(maxsize=None)
def load(name):
    img = cv2.cvtColor(cv2.imread(str(CACHE / f"{name}.png")), cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255


def glimpse(img, t, name, t0, strength=0.35, hold=0.3, attack=0.08, decay=0.7,
            zoom=(1.04, 1.1), mask=None):
    """A flash of the future: fast attack, brief hold, slow release, the
    frame slowly pushing in while it is visible."""
    age = t - t0
    if age < 0 or age > attack + hold + decay:
        return img
    if age < attack:
        e = age / attack
    elif age < attack + hold:
        e = 1.0
    else:
        e = 1 - (age - attack - hold) / decay
    e = e * e * (3 - 2 * e)
    v = load(name)
    z = zoom[0] + (zoom[1] - zoom[0]) * min(1.0, age / (attack + hold + decay))
    M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, z)
    v = cv2.warpAffine(v, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    a = strength * e
    if mask is not None:
        a = (a * mask)[..., None]
    return img * (1 - a) + v * a


def approach(img, name, u, strength=0.22):
    """The future coming toward her: a vision growing from the centre."""
    if u <= 0 or u >= 1:
        return img
    v = load(name)
    z = 0.25 + 1.1 * u
    M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, z)
    v = cv2.warpAffine(v, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
    m = cv2.warpAffine(np.ones((H, W), np.float32), M, (W, H), borderValue=0)
    m = cv2.GaussianBlur(m, (0, 0), 30)
    a = (strength * m * np.sin(np.pi * u))[..., None]
    return img * (1 - a) + v * a


if __name__ == "__main__":
    render_all()
