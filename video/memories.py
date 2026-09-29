"""Memory photographs: real contemporary photos (CC0 / public domain, see
assets/photos/CREDITS.md) treated as faded, soft prints that surface on
downbeats and drift through the frame like memories.
"""
import json
import math
import pathlib
from functools import lru_cache

import cv2
import numpy as np

from engine import H, ROOT, W
from fx import F32

PHOTOS = ROOT / "assets" / "photos" / "picks"
PW = 720  # working width of a print; height follows the photo


@lru_cache(maxsize=64)
def print_image(name, seed, border):
    """A faded, soft print of a real photo: float RGB + alpha."""
    r = np.random.default_rng(seed)
    img = cv2.cvtColor(cv2.imread(str(PHOTOS / f"{name}.jpg")), cv2.COLOR_BGR2RGB)
    ph = int(round(img.shape[0] * PW / img.shape[1]))
    img = cv2.resize(img, (PW, ph), interpolation=cv2.INTER_AREA).astype(F32) / 255
    img = cv2.GaussianBlur(img, (0, 0), 1.6)                              # soft focus
    L = img.mean(-1, keepdims=True)
    img = L + (img - L) * 0.75                                            # faded colour
    img = 0.07 + 0.9 * img                                                # lifted blacks
    tint = np.array(r.choice([[1.04, 0.99, 0.93], [0.97, 1.0, 1.04], [1.03, 0.97, 0.98]]), F32)
    img = img * tint
    yy, xx = np.mgrid[0:ph, 0:PW].astype(F32)
    v = 1 - 0.32 * (((xx / PW - 0.5) ** 2 + (yy / ph - 0.5) ** 2) * 2.2)
    img = img * v[..., None]                                              # lens falloff
    leak = np.exp(-((xx - PW * r.uniform(0.7, 1.1)) / (PW * 0.25)) ** 2)[..., None]
    img = img + np.array([0.30, 0.14, 0.05], F32) * leak * r.uniform(0.1, 0.5)
    alpha = np.ones((ph, PW), F32)
    if border:
        b = 16
        out = np.empty_like(img)
        out[:] = np.array([0.92, 0.90, 0.87], F32)
        out[b:-b, b:-b] = cv2.resize(img, (PW - 2 * b, ph - 2 * b))
        img = out
    else:
        m = np.zeros((ph, PW), F32)
        m[28:-28, 28:-28] = 1
        alpha = cv2.GaussianBlur(m, (0, 0), 20)                            # dissolving edges
    return np.clip(img, 0, 1.2), alpha


@lru_cache(maxsize=None)
def is_dark(name):
    img = cv2.imread(str(PHOTOS / f"{name}.jpg"), cv2.IMREAD_GRAYSCALE)
    return float(img.mean()) < 90


# ---------------------------------------------------------------- drift
class Memory:
    """One print's life: surfaces, drifts, turns slightly, sinks away."""

    def __init__(self, scene, t0, seed, life=10.0, zone=None, keep_left=None):
        r = np.random.default_rng(seed)
        self.scene, self.t0, self.life, self.seed = scene, t0, life, seed
        self.border = r.random() < 0.4
        self.depth = r.uniform(0, 1)                         # 0 far, 1 near
        self.w = 820 + 560 * self.depth                      # nearer = bigger
        zx0, zx1, zy0, zy1 = zone or (0.2, 0.9, 0.15, 0.75)
        self.x0 = r.uniform(zx0, zx1) * W
        self.y0 = r.uniform(zy0, zy1) * H
        sp = 8 + 18 * self.depth                             # parallax: nearer moves faster
        ang = r.uniform(-0.6, 0.6)                           # drift away from the lyric column
        self.vx, self.vy = math.cos(ang) * sp, math.sin(ang) * sp * 0.5 - 3
        self.keep_left = keep_left
        if keep_left is not None:                            # left edge never crosses it
            self.x0 = max(self.x0, keep_left + self.w / 2)
        self.rot0, self.omega = r.uniform(-7, 7), r.uniform(-0.6, 0.6)
        self.alpha = 0.62 + 0.28 * (1 - self.depth)
        self.blur = 1 + 5 * self.depth ** 2                  # near prints out of focus

    def envelope(self, t):
        u = (t - self.t0) / self.life
        if u <= 0 or u >= 1:
            return 0.0
        return min(1.0, u / 0.18) ** 1.5 * min(1.0, (1 - u) / 0.35) ** 1.5

    def draw(self, img, t, warm=0.0, offset=(0.0, 0.0)):
        e = self.envelope(t)
        if e <= 0.002:
            return img
        pic, a = print_image(self.scene, self.seed, self.border)
        PH = pic.shape[0]
        dt = t - self.t0
        sc = self.w / PW * (1 + 0.012 * dt)                   # slowly nearing
        cx = self.x0 + self.vx * dt + 10 * math.sin(dt * 0.4 + self.seed) + offset[0] * (0.6 + self.depth)
        cy = self.y0 + self.vy * dt + 6 * math.sin(dt * 0.33 + self.seed * 2) + offset[1] * (0.6 + self.depth)
        rot = self.rot0 + self.omega * dt
        M = cv2.getRotationMatrix2D((PW / 2, PH / 2), rot, sc)
        M[0, 2] += cx - PW / 2
        M[1, 2] += cy - PH / 2
        # warp only into the bounding box the print can occupy
        half = 0.5 * math.hypot(PW, PH) * sc + self.blur * 3
        x0, x1 = max(0, int(cx - half)), min(W, int(cx + half))
        y0, y1 = max(0, int(cy - half)), min(H, int(cy + half))
        if x0 >= x1 or y0 >= y1:
            return img
        M[0, 2] -= x0
        M[1, 2] -= y0
        bw, bh = x1 - x0, y1 - y0
        p = cv2.warpAffine(pic, M, (bw, bh), flags=cv2.INTER_LINEAR, borderValue=0)
        m = cv2.warpAffine(a, M, (bw, bh), flags=cv2.INTER_LINEAR, borderValue=0)
        if self.blur > 1.5:
            p = cv2.GaussianBlur(p, (0, 0), self.blur)
            m = cv2.GaussianBlur(m, (0, 0), self.blur)
        if warm > 0:
            p = p * (1 - 0.35 * warm) + np.array([1.0, 0.72, 0.45], F32) * p.mean(-1, keepdims=True) * 0.35 * warm * 1.3
        k = (m * e * self.alpha)[..., None]
        out = img.copy()
        region = out[y0:y1, x0:x1]
        # a faint soft shadow so pale prints still separate from the haze
        sh = cv2.GaussianBlur(m, (0, 0), 18)[..., None] * e * 0.12
        region = region * (1 - np.roll(np.roll(sh, 10, 0), 6, 1))
        out[y0:y1, x0:x1] = region * (1 - k) + p * k
        return out


def schedule(T, scenes, start, end, every=2, life=11.0, seed=0, zone=None, lead=1.5,
             keep_left=None):
    """A print surfaces on every `every`-th downbeat from `lead` s before the
    section. Dark prints keep to the right so the lyric stays readable."""
    bars = [b for b in T.bars if start - lead - life <= b < end]
    mems = []
    for i, b in enumerate(bars[::every]):
        sc = scenes[i % len(scenes)]
        z = zone or ((0.58, 0.92, 0.25, 0.75) if is_dark(sc) else (0.3, 0.9, 0.2, 0.78))
        mems.append(Memory(sc, b, seed * 1000 + i, life=life, zone=z, keep_left=keep_left))
    return mems


def drift(t, amp=(34, 22), rot=0.8, seed=0):
    """A slow, uneven wander (sum of incommensurate sines)."""
    f = (0.047, 0.071, 0.113)
    dx = amp[0] * (0.6 * math.sin(t * f[0] * 6.283 + seed) + 0.3 * math.sin(t * f[1] * 6.283 + 1.3 + seed)
                   + 0.1 * math.sin(t * f[2] * 6.283 + 2.1))
    dy = amp[1] * (0.6 * math.sin(t * f[1] * 6.283 + 0.7 + seed) + 0.4 * math.sin(t * f[2] * 6.283 + 2.9))
    dr = rot * math.sin(t * 0.031 * 6.283 + seed)
    return dx, dy, dr
