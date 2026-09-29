"""Shared rendering engine: timing lookups, palette, fonts, drawing helpers.

Everything is drawn with skia-python onto a 1920x1080 surface; frames are
pulled as RGBA arrays and piped to ffmpeg by render.py.
"""
import json
import math
import pathlib
from functools import lru_cache

import numpy as np
import skia

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis" / "out"
W, H = 1920, 1080
FPS = 30


# ---------------------------------------------------------------- palette
def rgb(h, a=1.0):
    h = h.lstrip("#")
    return skia.Color4f(int(h[0:2], 16) / 255, int(h[2:4], 16) / 255,
                        int(h[4:6], 16) / 255, a)


PAL = {
    # Institute: overlit paper, graph grid, ink, red pen
    "paper": "#EEF0EC", "grid": "#A9BFC2", "ink": "#1C2327",
    "ink_soft": "#5E6B70", "red": "#D7263D",
    # Memory / him: sodium night
    "night": "#0C111E", "night2": "#171D33", "sodium": "#F2A541",
    "rose": "#E8695A", "neon": "#6FE3D6",
    # Her: pale violet light
    "her": "#D9D2FF", "her_core": "#FFFFFF",
}


def col(name, a=1.0):
    return rgb(PAL[name], a)


# ---------------------------------------------------------------- fonts
FONT_FILES = {
    "her": "/usr/share/fonts/opentype/ebgaramond/EBGaramond12-Italic.otf",
    "her_roman": "/usr/share/fonts/opentype/ebgaramond/EBGaramond12-Regular.otf",
    "coats": "/usr/share/fonts/truetype/ibm-plex/IBMPlexMono-Regular.ttf",
    "coats_bold": "/usr/share/fonts/truetype/ibm-plex/IBMPlexMono-Bold.ttf",
    "coats_light": "/usr/share/fonts/truetype/ibm-plex/IBMPlexMono-Light.ttf",
    "him": "/usr/share/fonts/opentype/inter/Inter-Light.otf",
    "him_med": "/usr/share/fonts/opentype/inter/Inter-Medium.otf",
}


@lru_cache(None)
def typeface(key):
    return skia.Typeface.MakeFromFile(FONT_FILES[key])


def font(key, size):
    f = skia.Font(typeface(key), size)
    f.setSubpixel(True)
    f.setEdging(skia.Font.Edging.kSubpixelAntiAlias)
    return f


# ---------------------------------------------------------------- timing
class Timing:
    def __init__(self):
        a = json.loads((OUT / "audio.json").read_text())
        g = json.loads((OUT / "grid.json").read_text())
        self.lines = {L["line"]: L for L in
                      json.loads((OUT / "lyrics_timed.json").read_text())["lines"]}
        self.duration = a["duration"]
        self.pulses = np.array(g["pulses"])
        self.bars = np.array(g["downbeats"])
        self.eighths = np.array(a["beats"])
        self.sections = g["sections"]
        self.curve_t = np.array(a["curve_times"])
        self.loud = np.array(a["loudness_db_smooth"])

    def since(self, grid, t):
        """(index, seconds since) of the last grid event <= t."""
        i = int(np.searchsorted(grid, t, side="right")) - 1
        return i, (t - grid[i]) if i >= 0 else 1e9

    def pulse_env(self, t, decay=6.0):
        """1 on each kick pulse, decaying exponentially."""
        _, dt = self.since(self.pulses, t)
        return math.exp(-decay * dt)

    def bar_env(self, t, decay=3.0):
        _, dt = self.since(self.bars, t)
        return math.exp(-decay * dt)

    def loudness(self, t):
        """Smoothed loudness mapped to 0..1 over the song's range."""
        v = float(np.interp(t, self.curve_t, self.loud))
        return float(np.clip((v + 32) / 18, 0, 1))

    def section_at(self, t):
        for s in self.sections:
            if s["start"] <= t < s["end"]:
                return s
        return self.sections[-1]


# ---------------------------------------------------------------- easing
def clamp01(x):
    return max(0.0, min(1.0, x))


def smooth(x):
    x = clamp01(x)
    return x * x * (3 - 2 * x)


def ease_out(x, p=3):
    return 1 - (1 - clamp01(x)) ** p


def ramp(t, t0, t1):
    return clamp01((t - t0) / (t1 - t0)) if t1 > t0 else float(t >= t0)


def rng(seed):
    return np.random.default_rng(seed)


# ---------------------------------------------------------------- paints
def fill(c, blur=0.0, blend=None):
    p = skia.Paint(Color4f=c, AntiAlias=True)
    if blur:
        p.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, blur))
    if blend:
        p.setBlendMode(blend)
    return p


def stroke(c, w=2.0, blur=0.0, trim=None, cap=skia.Paint.kRound_Cap, dash=None,
           blend=None):
    p = skia.Paint(Color4f=c, AntiAlias=True, Style=skia.Paint.kStroke_Style,
                   StrokeWidth=w, StrokeCap=cap, StrokeJoin=skia.Paint.kRound_Join)
    if blur:
        p.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, blur))
    if trim is not None:
        a, b = trim
        if b <= a:
            p.setAlphaf(0)
        elif a > 0 or b < 1:  # Make() returns None for the full range
            p.setPathEffect(skia.TrimPathEffect.Make(a, b))
    if dash:
        p.setPathEffect(skia.DashPathEffect.Make(dash, 0))
    if blend:
        p.setBlendMode(blend)
    return p


def glow_path(canvas, path, c, w, glow=18, strength=0.6, trim=None):
    """Bloomed line: wide blurred under-stroke plus a crisp core."""
    g = skia.Color4f(c.fR, c.fG, c.fB, c.fA * strength)
    canvas.drawPath(path, stroke(g, w * 3, blur=glow, trim=trim,
                                 blend=skia.BlendMode.kPlus))
    canvas.drawPath(path, stroke(c, w, trim=trim))


def poly(points, closed=False):
    p = skia.Path()
    p.moveTo(*points[0])
    for q in points[1:]:
        p.lineTo(*q)
    if closed:
        p.close()
    return p


def smooth_path(points, closed=False):
    """Catmull-Rom through points as cubic Beziers."""
    pts = [tuple(map(float, q)) for q in points]
    if closed:
        pts = [pts[-1]] + pts + pts[:2]
    else:
        pts = [pts[0]] + pts + [pts[-1]]
    p = skia.Path()
    p.moveTo(*pts[1])
    for i in range(1, len(pts) - 2):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[i + 1], pts[i + 2]
        c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
        p.cubicTo(*c1, *c2, *p2)
    if closed:
        p.close()
    return p


# ---------------------------------------------------------------- texture
@lru_cache(None)
def grain_images(n=6, seed=7):
    r = rng(seed)
    imgs = []
    for _ in range(n):
        g = r.normal(128, 38, (H // 2, W // 2)).clip(0, 255).astype(np.uint8)
        rgba = np.dstack([g, g, g, np.full_like(g, 255)])
        imgs.append(skia.Image.fromarray(rgba, colorType=skia.kRGBA_8888_ColorType))
    return imgs


def grain(canvas, t, amount=0.07):
    img = grain_images()[int(t * FPS) % 6]
    p = skia.Paint(Alphaf=amount, BlendMode=skia.BlendMode.kOverlay)
    canvas.drawImageRect(img, skia.Rect(0, 0, W, H),
                         skia.SamplingOptions(skia.FilterMode.kLinear), p)


def vignette(canvas, strength=0.55, c="#000000"):
    shader = skia.GradientShader.MakeRadial(
        (W / 2, H / 2), W * 0.75,
        [rgb(c, 0).toColor(), rgb(c, 0).toColor(), rgb(c, strength).toColor()],
        [0.0, 0.55, 1.0])
    canvas.drawRect(skia.Rect(0, 0, W, H), skia.Paint(Shader=shader))


def graph_paper(canvas, c, minor=24, major=5, alpha=0.35, off=(0, 0)):
    ox, oy = off
    for i, x in enumerate(np.arange(-minor + ox % minor, W + minor, minor)):
        a = alpha if (i % major == 0) else alpha * 0.4
        canvas.drawLine(x, 0, x, H, stroke(rgb(c, a), 1.0 if a == alpha else 0.6))
    for j, y in enumerate(np.arange(-minor + oy % minor, H + minor, minor)):
        a = alpha if (j % major == 0) else alpha * 0.4
        canvas.drawLine(0, y, W, y, stroke(rgb(c, a), 1.0 if a == alpha else 0.6))


def chromatic(surface, shift):
    """Cheap RGB split on the finished frame (psychic moments)."""
    if shift < 0.5:
        return surface
    arr = surface.makeImageSnapshot().toarray()
    s = int(round(shift))
    out = arr.copy()
    out[:, s:, 0] = arr[:, :-s, 0]
    out[:, :-s, 2] = arr[:, s:, 2]
    img = skia.Image.fromarray(out, colorType=skia.kRGBA_8888_ColorType)
    surface.getCanvas().drawImage(img, 0, 0)
    return surface


# ---------------------------------------------------------------- type
def text_width(f, s):
    return f.measureText(s)


def draw_text(canvas, s, x, y, f, c, blur=0.0, align="left", blend=None):
    w = f.measureText(s)
    if align == "center":
        x -= w / 2
    elif align == "right":
        x -= w
    canvas.drawString(s, x, y, f, fill(c, blur=blur, blend=blend))
    return w
