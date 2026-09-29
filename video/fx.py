"""Photographic raster pipeline (numpy + OpenCV, float32 linear RGB).

The look is a long exposure of a memory: fog, bloom, film halation, lifted
milky blacks, grain, dust, and an exposure echo that keeps earlier moments
ghosted into the present frame ("agains").
"""
import math
from collections import deque
from functools import lru_cache

import cv2
import numpy as np
import skia

from engine import H, W, FPS

F32 = np.float32


def hexc(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)], F32)


# ---------------------------------------------------------------- basics
def blur(img, sigma):
    """Large-sigma Gaussian via downsample -> blur -> upsample."""
    if sigma <= 0.5:
        return img
    s = max(1, int(sigma / 3))
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // s, h // s), interpolation=cv2.INTER_AREA) if s > 1 else img
    b = cv2.GaussianBlur(small, (0, 0), sigma / s)
    return cv2.resize(b, (w, h), interpolation=cv2.INTER_LINEAR) if s > 1 else b


def lum(img):
    return img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722


def shift(img, dx, dy, rot=0.0, scale=1.0):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rot, scale)
    M[0, 2] += dx
    M[1, 2] += dy
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def skia_alpha(draw, w=W, h=H):
    """Run a skia draw callback on a transparent surface -> float alpha."""
    surf = skia.Surface(w, h)
    c = surf.getCanvas()
    c.clear(skia.ColorTRANSPARENT)
    draw(c)
    return surf.makeImageSnapshot().toarray()[..., 3].astype(F32) / 255.0


def _bbox(alpha, eps=1e-3):
    """Row/column slice covering where alpha is non-negligible (or None)."""
    rows = np.flatnonzero(alpha.max(axis=1) > eps)
    if not len(rows):
        return None
    cols = np.flatnonzero(alpha[rows[0]:rows[-1] + 1].max(axis=0) > eps)
    return slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)


def over(base, color, alpha):
    """Composite a flat colour through an alpha mask (only where it covers)."""
    bb = _bbox(alpha)
    if bb is None:
        return base
    out = base.copy()
    ys, xs = bb
    a = alpha[ys, xs][..., None]
    out[ys, xs] = base[ys, xs] + (color - base[ys, xs]) * a
    return out


def add(base, color, alpha):
    bb = _bbox(alpha)
    if bb is None:
        return base
    out = base.copy()
    ys, xs = bb
    out[ys, xs] = base[ys, xs] + color * alpha[ys, xs][..., None]
    return out


# ---------------------------------------------------------------- fields
@lru_cache(maxsize=64)
def _noise_slice(seed, k, h=300, w=560):
    r = np.random.default_rng(seed * 100003 + k)
    acc = np.zeros((h, w), F32)
    amp, tot = 1.0, 0.0
    for gh, gw in ((3, 5), (6, 10), (12, 20), (24, 40), (48, 80)):
        g = r.random((gh, gw)).astype(F32)
        acc += amp * cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
        tot += amp
        amp *= 0.55
    acc /= tot
    return (acc - acc.mean()) / (acc.std() + 1e-6)


def fog(t, seed=1, period=4.0, drift=(1.0, 0.3)):
    """Slowly evolving fBm fog in roughly [-2, 2], full resolution."""
    k = int(t // period)
    f = (t % period) / period
    f = f * f * (3 - 2 * f)
    a, b = _noise_slice(seed, k), _noise_slice(seed, k + 1)
    field = (a * (1 - f) + b * f) / math.sqrt(f * f + (1 - f) ** 2)
    ox = 40 + 30 * math.sin(t * 0.021 * drift[0] + seed)
    oy = 20 + 12 * math.sin(t * 0.017 * drift[1] + seed * 2)
    crop = cv2.getRectSubPix(field, (480, 270), (240 + ox, 135 + oy))
    return cv2.resize(crop, (W, H), interpolation=cv2.INTER_CUBIC)


@lru_cache(maxsize=4)
def _yy_xx():
    yy, xx = np.mgrid[0:H, 0:W].astype(F32)
    return yy, xx


def radial(cx, cy, r, power=2.0):
    """Radial falloff, computed only inside its radius."""
    out = np.zeros((H, W), F32)
    y0, y1 = max(0, int(cy - r)), min(H, int(cy + r) + 1)
    x0, x1 = max(0, int(cx - r)), min(W, int(cx + r) + 1)
    if y0 >= y1 or x0 >= x1:
        return out
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(F32)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r
    out[y0:y1, x0:x1] = np.clip(1 - d, 0, 1) ** power
    return out


@lru_cache(maxsize=32)
def vgrad(y0, y1):
    yy, _ = _yy_xx()
    return np.clip((yy - y0) / (y1 - y0), 0, 1)


@lru_cache(maxsize=32)
def hgrad(x0, x1):
    _, xx = _yy_xx()
    return np.clip((xx - x0) / (x1 - x0), 0, 1)


# ---------------------------------------------------------------- light
def hotspot(img, t, x, y, r, color="#FF9A4A", strength=1.0):
    """A warm source just behind something: soft core plus wide spill."""
    if strength <= 0.001:
        return img
    m = radial(x, y, r, 2.4) * 1.6 + radial(x, y, r * 3, 2.0) * 0.5
    m *= 0.9 + 0.1 * math.sin(t * 2.3)
    return img + hexc(color) * (m * strength)[..., None]


def light_leak(img, t, side="right", color="#FF9A4A", strength=1.0, seed=3):
    """Warm leak bleeding in from outside the frame, breathing slowly."""
    if strength <= 0.001:
        return img
    g = fog(t * 0.7, seed=seed, period=5.0)
    breathe = 0.85 + 0.15 * math.sin(t * 1.3 + seed)
    if side == "right":
        m = hgrad(W * 0.35, W * 1.05) ** 2.2
    elif side == "left":
        m = (1 - hgrad(-W * 0.05, W * 0.65)) ** 2.2
    else:
        m = (1 - vgrad(-H * 0.1, H * 0.8)) ** 2
    m = m * np.clip(0.75 + 0.25 * g, 0, 1.5) * breathe * strength
    # anamorphic streak through the brightest band
    streak = np.exp(-((_yy_xx()[0] - H * 0.42) / (H * 0.05)) ** 2) * m * 0.35
    return img + hexc(color) * (m + streak)[..., None]


def bloom(img, amount=0.6, thresh=0.7):
    hi = np.clip(img - thresh, 0, None)
    b = blur(hi, 10) * 0.5 + blur(hi, 32) * 0.35 + blur(hi, 90) * 0.25
    return img + b * amount


def halation(img, amount=0.5, thresh=0.8, tint="#FF5A2A"):
    """Film halation: bright edges bleed a red-orange fringe."""
    hi = np.clip(lum(img) - thresh, 0, None)
    h = blur(hi, 14) - blur(hi, 3) * 0.6
    return img + hexc(tint) * np.clip(h, 0, None)[..., None] * amount


def diffusion(img, amount=0.25):
    """Pro-Mist style: soften by mixing in a wide blur (lighten only)."""
    b = blur(img, 22)
    return img * (1 - amount) + np.maximum(img, b) * amount


def tone(img, exposure=1.0, lift=0.06, shadow="#6B5F80", sat=0.85, white=1.0):
    x = np.clip(img * exposure, 0, None)
    y = 1 - np.exp(-x * 1.35)
    y = y / (1 - math.exp(-1.35 * white)) if white else y
    L = lum(y)[..., None]
    y = L + (y - L) * sat
    sh = hexc(shadow)
    y = lift * sh + (1 - lift) * y + lift * (1 - y) * 0  # milky blacks
    return np.clip(y, 0, 1)


def grain(img, t, amount=0.05, seed=9):
    r = np.random.default_rng(int(t * FPS) + seed * 1_000_003)
    n = r.standard_normal((H // 2, W // 2)).astype(F32)
    n = cv2.resize(n, (W, H), interpolation=cv2.INTER_LINEAR)
    L = lum(img)
    wgt = 0.35 + 4 * L * (1 - L)
    return np.clip(img + (n * amount * wgt)[..., None], 0, 1)


@lru_cache(maxsize=8)
def _vig_mask(strength):
    m = 1 - radial(W / 2, H / 2, W * 0.78, power=1.0)
    return (np.clip(m * 1.4, 0, 1) ** 1.6 * strength)[..., None]


def vignette(img, strength=0.35, color="#3A3246"):
    m = _vig_mask(strength)
    return img * (1 - m) + hexc(color) * m * 0.5


def chroma_edges(img, px=3):
    """Lens colour fringing growing toward the frame edges."""
    r = shift(img[..., 0:1].copy(), 0, 0, scale=1 + px / W)
    b = shift(img[..., 2:3].copy(), 0, 0, scale=1 - px / W)
    out = img.copy()
    out[..., 0] = r if r.ndim == 2 else r[..., 0]
    out[..., 2] = b if b.ndim == 2 else b[..., 0]
    return out


# ---------------------------------------------------------------- particles
def dust(img, t, T, seed=4, n=140, color="#FFF6EC", strength=1.0):
    """Slow motes at three depths; near ones are soft discs (bokeh)."""
    r = np.random.default_rng(seed)
    x0, y0 = r.uniform(0, W, n), r.uniform(0, H, n)
    depth = r.uniform(0, 1, n)
    vx, vy = r.normal(0, 6, n), r.uniform(-10, -2, n)
    ph = r.uniform(0, 6.28, n)
    kick = T.pulse_env(t, 5.0)

    def draw(c):
        for i in range(n):
            x = (x0[i] + vx[i] * t + 20 * math.sin(t * 0.3 + ph[i])) % W
            y = (y0[i] + vy[i] * t) % H
            d = depth[i]
            rad = 1.2 + 9 * d ** 3
            a = (0.35 + 0.65 * (1 - d)) * (0.55 + 0.45 * math.sin(t * 0.8 + ph[i]) ** 2)
            a *= 0.8 + 0.4 * kick
            p = skia.Paint(AntiAlias=True, Color4f=skia.Color4f(1, 1, 1, min(1, a)))
            if d > 0.6:
                p.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, rad * 0.6))
            c.drawCircle(x, y, rad, p)
    m = skia_alpha(draw)
    return img + hexc(color) * (m * 0.55 * strength)[..., None]


# ---------------------------------------------------------------- echo
class Exposure:
    """Multiple-exposure memory. Keeps a short trail plus sparse snapshots
    so earlier moments stay ghosted, displaced, in the current frame.

    Rendering must run sequentially; for a single still, call warm() first.
    """

    def __init__(self, every=3, keep=60):
        self.trail = None
        self.snaps = deque(maxlen=keep)  # (t, half-res float16 frame)
        self.every = every
        self.i = 0
        self.last_t = None

    def apply(self, img, t, trail=0.55, ghosts=((1.2, -150, 0, -5, 0.28),
                                                 (2.4, 150, -10, 4, 0.18))):
        if self.last_t is not None and abs(t - self.last_t) > 0.5:
            self.trail, self.snaps = None, deque(maxlen=self.snaps.maxlen)  # a cut
        self.last_t = t
        if self.trail is None:
            self.trail = img.copy()
        self.trail = self.trail * trail + img * (1 - trail)
        out = np.maximum(img, self.trail * 0.97)
        for dt, dx, dy, rot, a in ghosts:
            snap = self._at(t - dt)
            if snap is not None and a > 0:
                g = cv2.resize(snap.astype(F32), (W, H), interpolation=cv2.INTER_LINEAR)
                g = shift(g, dx, dy, rot)
                out = out + np.clip(g - out * 0.6, 0, None) * a  # soft "screen"
        if self.i % self.every == 0:
            half = cv2.resize(img, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
            self.snaps.append((t, half.astype(np.float16)))
        self.i += 1
        return out

    def _at(self, t):
        best = None
        for ts, s in self.snaps:
            if ts <= t:
                best = s
            else:
                break
        return best
