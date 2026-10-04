"""Scenes for the "long exposure of a memory" look.

Nothing is drawn literally. She is a soft, multiply-exposed presence; he
is never drawn at all, only warm light leaking in; the institute is
overexposure, flicker, scratches and film burns; time is tally marks and
the exposure echo. The one hard-focus object in the film is the bullet.
"""
import math
from functools import lru_cache

import cv2
import numpy as np
import skia

import fx
from engine import H, W, clamp01, display_text, ease_out, font, ramp, smooth, smooth_path
from fx import F32, hexc

# ---------------------------------------------------------------- palette
C = {
    "haze": hexc("#E6E1EE"), "haze_hi": hexc("#F7F3F6"), "haze_lo": hexc("#B9B1C9"),
    "plum": hexc("#2B2233"), "plum_ink": hexc("#3A3048"), "plum_deep": hexc("#221A2C"), "night": hexc("#140F19"),
    "amber": hexc("#FF9A4A"), "amber_hot": hexc("#FFC58A"), "rose": hexc("#F2765E"),
    "graphite": hexc("#4A4652"), "clinic": hexc("#DDE3E6"), "body": hexc("#F3ECEE"),
    "shade": hexc("#9D92AE"),
    "void": hexc("#F8F5F8"), "void_dark": hexc("#07050A"),
    "her_edge": hexc("#7D7294"), "her_glow": hexc("#FFFFFF") * 0.06,
    "her_edge_dark": hexc("#C9BEE8"), "her_glow_dark": hexc("#B9A8F0") * 0.5,
    "hair": hexc("#2E2536"),
}


# ================================================================ her
def figure_masks(cx, floor, s, sway=0.0):
    """(body, hair, face) alpha masks for a girl sitting cross-legged,
    hunched, hair falling forward. Kept soft and generic on purpose."""
    def P(x, y):
        return cx + x * s, floor + y * s

    white = skia.Paint(AntiAlias=True, Color4f=skia.Color4f(1, 1, 1, 1))
    hx = sway  # head sway

    def body(c):
        legs = skia.Path()                                                   # crossed legs
        legs.moveTo(*P(-0.43, -0.05))
        legs.cubicTo(*P(-0.45, -0.17), *P(-0.36, -0.22), *P(-0.26, -0.20))
        legs.cubicTo(*P(-0.14, -0.19), *P(-0.06, -0.24), *P(0.04, -0.22))
        legs.cubicTo(*P(0.16, -0.20), *P(0.30, -0.17), *P(0.37, -0.13))
        legs.cubicTo(*P(0.44, -0.09), *P(0.42, -0.02), *P(0.30, -0.01))
        legs.cubicTo(*P(0.10, 0.02), *P(-0.30, 0.02), *P(-0.43, -0.05))
        c.drawPath(legs, white)
        torso = skia.Path()                                                  # oversized shirt
        torso.moveTo(*P(-0.05, -0.70))
        torso.cubicTo(*P(-0.16, -0.70), *P(-0.21, -0.66), *P(-0.23, -0.55))
        torso.cubicTo(*P(-0.25, -0.42), *P(-0.21, -0.30), *P(-0.18, -0.18))
        torso.lineTo(*P(0.18, -0.18))
        torso.cubicTo(*P(0.21, -0.30), *P(0.25, -0.42), *P(0.23, -0.55))
        torso.cubicTo(*P(0.21, -0.66), *P(0.16, -0.70), *P(0.05, -0.70))
        torso.close()
        c.drawPath(torso, white)
        c.drawRect(skia.Rect(*P(-0.035 + hx * 0.5, -0.78), *P(0.035 + hx * 0.5, -0.66)), white)
        c.drawOval(skia.Rect(*P(-0.085 + hx, -1.0), *P(0.085 + hx, -0.76)), white)
        arm = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=0.075 * s,
                         StrokeCap=skia.Paint.kRound_Cap, Color4f=skia.Color4f(1, 1, 1, 1))
        for sx in (-1, 1):                                                   # arms to lap
            a = skia.Path()
            a.moveTo(*P(sx * 0.19, -0.62))
            a.cubicTo(*P(sx * 0.27, -0.48), *P(sx * 0.25, -0.32), *P(sx * 0.07, -0.25))
            c.drawPath(a, arm)

    def hair(c):
        h = skia.Path()                                                      # falls past shoulders
        h.moveTo(*P(hx, -1.02))
        h.cubicTo(*P(0.12 + hx, -1.02), *P(0.14, -0.86), *P(0.13, -0.74))
        h.cubicTo(*P(0.14, -0.62), *P(0.17, -0.52), *P(0.15, -0.44))
        h.lineTo(*P(0.11, -0.47)); h.lineTo(*P(0.09, -0.42)); h.lineTo(*P(0.065, -0.56))
        h.cubicTo(*P(0.07, -0.70), *P(0.075 + hx, -0.80), *P(0.06 + hx, -0.86))
        h.lineTo(*P(-0.06 + hx, -0.86))
        h.cubicTo(*P(-0.075 + hx, -0.80), *P(-0.07, -0.70), *P(-0.065, -0.56))
        h.lineTo(*P(-0.09, -0.40)); h.lineTo(*P(-0.12, -0.46)); h.lineTo(*P(-0.16, -0.43))
        h.cubicTo(*P(-0.17, -0.55), *P(-0.14, -0.64), *P(-0.13, -0.74))
        h.cubicTo(*P(-0.14, -0.86), *P(-0.12 + hx, -1.02), *P(hx, -1.02))
        c.drawPath(h, white)
        r = np.random.default_rng(12)
        st = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=1.3,
                        Color4f=skia.Color4f(1, 1, 1, 0.7))
        for _ in range(70):                                                  # loose strands
            sx = r.choice([-1, 1])
            x0 = r.uniform(0.07, 0.13) * sx + hx
            y0 = r.uniform(-0.90, -0.78)
            pts = [P(x0, y0)]
            for k in range(1, 5):
                pts.append(P(x0 + sx * r.uniform(0.0, 0.025) * k,
                             y0 + k * r.uniform(0.08, 0.11)))
            c.drawPath(smooth_path(pts), st)

    def face(c):
        c.drawOval(skia.Rect(*P(-0.058 + hx, -0.94), *P(0.058 + hx, -0.77)), white)
        for x0, y0, rot in ((-0.075, -0.25, 20), (0.05, -0.23, -15)):          # open hands
            c.save()
            c.translate(*P(x0, y0))
            c.rotate(rot)
            c.drawOval(skia.Rect(-0.045 * s, -0.022 * s, 0.045 * s, 0.022 * s), white)
            c.restore()
    return fx.skia_alpha(body), fx.skia_alpha(hair), fx.skia_alpha(face)


SS = 2  # hair supersampling factor


def _soft(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def _lock_paths(cx, top, s, t, seed, n_locks, steps):
    """Simulate lock centrelines with soft forces: gravity, a skull the hair
    slides over, a face gap it parts around, and a slow sway."""
    r = np.random.default_rng(seed)
    hcx, hcy = cx, top + 0.12 * s
    rx, ry = 0.090 * s, 0.117 * s
    sway = math.sin(t * 0.45 + seed) * 0.010 * s
    locks = []
    for i in range(n_locks):
        th = math.radians(-172 + 164 * (i + r.uniform(0.2, 0.8)) / n_locks)
        side = -1 if math.degrees(th) < -90 else 1
        x, y = hcx + math.cos(th) * rx, hcy + math.sin(th) * ry
        vx, vy = side * 0.5 + math.cos(th) * 0.3, 0.45
        L = r.uniform(0.58, 1.02) * s
        hang = r.uniform(0.11, 0.26) * s
        ph, curl = r.uniform(0, 6.28), r.normal(0, 1.3)
        step = L / steps
        pts = [(x, y)]
        for k in range(steps):
            u = k / steps
            vy += 0.16
            vx *= 0.88
            vx += math.sin(u * 6 + ph) * 0.05 * curl * (0.3 + u) + sway / s * 3 * u
            ex, ey = (x - hcx) / rx, (y - hcy) / ry
            e = math.hypot(ex, ey) + 1e-6
            push = _soft((1.10 - e) / 0.12)              # soft skull
            vx += ex / e * push * 0.9
            vy += ey / e * push * 0.9 * (ey < 0)
            dxf = abs(x - hcx)
            if y > hcy - 0.03 * s and y < hcy + 0.34 * s:  # part around the face
                vx += side * _soft((0.085 * s - dxf) / (0.03 * s)) * 0.6
            if y > hcy + 0.30 * s:                        # settle over shoulders
                vx += (hcx + side * hang - x) / s * 0.9
            sp = math.hypot(vx, vy) + 1e-6
            x, y = x + vx / sp * step, y + vy / sp * step
            pts.append((x, y))
        pts = np.array(pts)
        # a lock-level wave that grows toward the tips
        uu = np.linspace(0, 1, len(pts))
        amp = r.uniform(0.008, 0.028) * s
        fr = r.uniform(1.6, 3.2)
        pts[:, 0] += amp * np.sin(uu * fr * 6.283 + ph + t * 0.35) * uu ** 1.1
        locks.append((pts, side, th))
    return locks


def hair_curtain(cx, top, s, t, seed=0, n=4000, part=1.0, n_locks=38, steps=48):
    """Long hair: thousands of fine, tapering strands clumped into locks,
    drawn supersampled over the head region. Returns (density, sheen)."""
    r = np.random.default_rng(seed + 1000)
    locks = _lock_paths(cx, top, s, t, seed, n_locks, steps)
    x0, y0 = int(cx - 0.55 * s), int(top - 0.06 * s)
    x1, y1 = int(cx + 0.55 * s), int(top + 1.12 * s)
    bw, bh = x1 - x0, y1 - y0
    surf = skia.Surface(bw * SS, bh * SS)
    c = surf.getCanvas()
    c.clear(skia.ColorTRANSPARENT)
    c.scale(SS, SS)
    c.translate(-x0, -y0)
    u = np.linspace(0, 1, steps + 1)[:, None]
    chunks = ((0, 14, 1.0, 1.0), (14, 28, 0.95, 0.9), (28, 40, 0.7, 0.75), (40, steps + 1, 0.3, 0.55))
    paint = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style,
                       StrokeCap=skia.Paint.kRound_Cap, StrokeJoin=skia.Paint.kRound_Join)
    for i in range(n):
        base, side, th = locks[i % len(locks)]
        # tangent spread at the root, clumping toward the tips, a few flyaways
        tang = np.array([-math.sin(th), math.cos(th)])
        root = tang * r.normal(0, 0.030 * s)
        fly = r.random() < 0.05
        tip = r.normal(0, 0.006 * s, 2) * (6.0 if fly else 1.0)
        pts = base + root * (1 - u) ** 1.2 + tip * u ** 1.3
        pts = pts + np.column_stack([np.sin(u[:, 0] * r.uniform(4, 9) + r.uniform(0, 6.28)),
                                     np.zeros(len(u))]) * r.uniform(0, 0.004 * s) * u
        cut = int(len(pts) * r.uniform(0.75, 1.0))           # uneven ends
        w = r.uniform(0.35, 0.9)
        a = r.uniform(0.05, 0.12)
        for i0, i1, am, wm in chunks:
            i1 = min(i1, cut)
            if i1 - i0 < 2:
                continue
            path = skia.Path()
            path.moveTo(*pts[i0])
            for q in range(i0 + 1, i1):
                mx, my = (pts[q - 1] + pts[q]) / 2
                path.quadTo(*pts[q - 1], mx, my)
            path.lineTo(*pts[i1 - 1])
            paint.setStrokeWidth(w * wm)
            paint.setColor4f(skia.Color4f(1, 1, 1, a * am))
            c.drawPath(path, paint)
    arr = surf.makeImageSnapshot().toarray()[..., 3].astype(F32) / 255
    small = cv2.resize(arr, (bw, bh), interpolation=cv2.INTER_AREA)
    m = np.zeros((H, W), F32)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    m[sy0:sy1, sx0:sx1] = small[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]
    m = 1 - np.exp(-m * 3.4)                                   # optical density
    # anisotropic sheen: a soft band where hair curves over the crown
    hcy = top + 0.12 * s
    yy, xx = fx._yy_xx()
    d = np.sqrt(((xx - cx) / (0.10 * s)) ** 2 + ((yy - (hcy - 0.01 * s)) / (0.125 * s)) ** 2)
    band = np.exp(-((d - 0.78) / 0.16) ** 2) * (yy < hcy + 0.05 * s)
    return m, m * band


@lru_cache(maxsize=8)
def face_hint(cx, top, s):
    """Pale face with the faintest shadows for eyes and mouth: a suggestion."""
    hcy = top + 0.12 * s
    face = fx.radial(cx, hcy + 0.05 * s, 0.075 * s, 1.5)
    eyes = (fx.radial(cx - 0.03 * s, hcy + 0.03 * s, 0.022 * s, 1.5)
            + fx.radial(cx + 0.03 * s, hcy + 0.03 * s, 0.022 * s, 1.5))
    mouth = fx.radial(cx, hcy + 0.115 * s, 0.018 * s, 1.5)
    return fx.blur(face, 4), fx.blur(eyes, 3), fx.blur(mouth, 3)


@lru_cache(maxsize=24)
def _hair_cached(cx, top, s, tq, seed, n, n_locks, steps):
    return hair_curtain(cx, top, s, tq, seed=seed, n=n, n_locks=n_locks, steps=steps)


def hair_at(cx, top, s, t, seed=0, n=4000, n_locks=38, steps=48, step=0.2):
    """Hair moves slowly: build it every `step` seconds and crossfade
    between neighbouring states, which reads as smooth, soft motion."""
    k = math.floor(t / step)
    f = (t - k * step) / step
    args = (float(cx), float(top), float(s))
    a0 = _hair_cached(*args, round(k * step, 3), seed, n, n_locks, steps)
    a1 = _hair_cached(*args, round((k + 1) * step, 3), seed, n, n_locks, steps)
    return a0[0] * (1 - f) + a1[0] * f, a0[1] * (1 - f) + a1[1] * f


@lru_cache(maxsize=8)
def body_density(cx, floor, s):
    """Shoulders and folded legs as a heavily blurred density, never a shape."""
    b, _h, _f = figure_masks(cx, floor, s)
    return fx.blur(b, 0.05 * s)


def her_presence(img, t, T, cx=1150, floor=930, s=640, ghosts=True, rim=1.0,
                 fade=1.0, body_col=None, hair_col=None, rim_side=1, seed=0,
                 light=None, face=True, behind=None):
    """Her: hair curtain over a blurred body density, displaced ghost
    exposures, and warm light (him) scattering through the hair near
    `light` (x, y)."""
    body_col = C["shade"] if body_col is None else body_col
    hair_col = C["hair"] if hair_col is None else hair_col
    top = floor - 1.02 * s
    brk = np.clip(0.8 + 0.3 * fx.fog(t, seed=8, period=6.0), 0, 1)
    dens = body_density(cx, floor, s)
    if ghosts:
        kick = T.pulse_env(t, 3.0)
        vt = vtop(floor, s)
        for k, (dx, rot, a) in enumerate(((-215, -10, 0.30), (210, 9, 0.24))):
            wob = math.sin(t * 0.5 + dx) * 18
            hk, _ = hair_at(cx, top, s, t, seed=seed + 11 + k, n=900, n_locks=40, steps=32, step=0.4)
            g = fx.blur(fx.shift(hk, dx + wob, -12, rot), 6) * vt
            gd = fx.blur(fx.shift(dens, dx + wob, -12, rot), 10) * vt
            aa = a * (0.8 + 0.4 * kick) * fade
            img = fx.over(img, body_col, gd * aa * 0.5)
            img = fx.over(img, hair_col, g * aa)
    if behind is not None:
        img = behind(img)
    img = fx.over(img, body_col, dens * 0.7 * brk * fade)
    if face:
        fc, ey, mo = face_hint(cx, top, s)
        img = fx.over(img, C["body"], fc * 0.55 * fade)
    hk, sheen = hair_at(cx, top, s, t, seed=seed)
    img = fx.over(img, hair_col, fx.blur(hk, 5) * 0.45 * brk * fade)      # volume
    img = fx.over(img, hair_col, hk * 0.8 * brk * fade)                   # strands
    img = fx.over(img, C["haze_lo"], fx.blur(sheen, 6) * 0.16 * fade)   # sheen
    if rim > 0:
        lx, ly = light if light else (cx + 0.2 * s * rim_side, top + 0.3 * s)
        near = fx.radial(lx, ly, 0.38 * s, 1.5)
        lit = hk * (1 - hk) * 4 * near          # light passes through the thin parts
        glow = fx.blur(lit, 1.2) * 0.9 + fx.blur(lit, 7) * 0.25
        img = img + C["amber"] * (glow * rim * fade)[..., None]
    return img


def vtop(floor, s):
    """Ghost exposures fade out below the shoulders."""
    return 1 - fx.vgrad(floor - 0.62 * s, floor - 0.28 * s)


# ================================================================ her (absence)
# Alternative treatment, kept for single moments: her as an absence.
def girl_silhouette(cx, floor, s, facing=-1, breath=0.0, bow=0.0):
    """Filled silhouette of a girl sitting on the floor in profile, knees
    drawn up, arms around her shins, head bowed, hair down her back.
    facing=-1 faces left. Returns a soft 0..1 mask."""
    f = facing

    def P(x, y):
        return cx - f * x * s * -1 if False else cx + f * -x * s, floor + y * s

    def draw(c):
        white = skia.Paint(AntiAlias=True, Color4f=skia.Color4f(1, 1, 1, 1))

        def limb(pts, w):
            p = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=w * s,
                           StrokeCap=skia.Paint.kRound_Cap, StrokeJoin=skia.Paint.kRound_Join,
                           Color4f=skia.Color4f(1, 1, 1, 1))
            c.drawPath(smooth_path([P(*q) for q in pts]), p)

        def blob(x, y, rx, ry, rot=0.0):
            c.save()
            c.translate(*P(x, y))
            c.rotate(rot * -f)
            c.drawOval(skia.Rect(-rx * s, -ry * s, rx * s, ry * s), white)
            c.restore()

        by = breath * 0.01
        blob(0.11, -0.075, 0.15, 0.075)                                 # seat
        blob(0.12, -0.34 + by, 0.13, 0.24, -28)                         # back, hunched
        blob(0.03, -0.55 + by, 0.10, 0.075, -35)                        # shoulders
        limb([(0.16, -0.08), (0.00, -0.30), (-0.19, -0.47)], 0.14)       # thigh
        limb([(-0.19, -0.46), (-0.25, -0.26), (-0.29, -0.05)], 0.10)    # shin
        blob(-0.32, -0.035, 0.08, 0.035)                                # foot
        limb([(0.04, -0.55 + by), (-0.06, -0.42), (-0.21, -0.37)], 0.066)  # arm round the shins
        hx, hy = -0.10 - bow * 0.02, -0.63 + bow * 0.02 + by            # head, resting on the knees
        blob(hx, hy, 0.085, 0.10, 35 + bow * 10)
        hair = skia.Path()                                              # hair down the back
        pts = [(hx - 0.02, hy - 0.10), (hx + 0.08, hy - 0.11), (hx + 0.16, hy - 0.04),
               (0.20, -0.46), (0.27, -0.30), (0.30, -0.19), (0.26, -0.23),
               (0.22, -0.30), (0.16, -0.42), (hx + 0.10, hy + 0.06)]
        c.drawPath(smooth_path([P(*q) for q in pts], closed=True), white)
        limb([(hx - 0.07, hy - 0.02), (hx - 0.11, hy + 0.07), (hx - 0.12, hy + 0.14)], 0.03)  # a fall over the face
    return fx.blur(fx.skia_alpha(draw), 1.5)


def her_absence(img, t, T, cx=1150, floor=960, s=640, facing=-1, dark=False,
                fill=0.0, fill_col=None, edge=1.0, ghosts=True, fade=1.0, shed=1.0,
                seed=0, react=True):
    """Her as an emptiness: a hollow in the fog with a soft, broken outline
    that breathes and sheds motes. `fill` pours warm light into the hollow."""
    fill_col = C["amber"] if fill_col is None else fill_col
    kick = T.pulse_env(t, 3.0) if react else 0.0
    m = girl_silhouette(cx, floor, s, facing, breath=math.sin(t * 0.9), bow=math.sin(t * 0.21))
    brk = np.clip(0.45 + 0.75 * fx.fog(t * 0.6, seed=seed + 50, period=5.0), 0, 1)

    # displaced copies of the outline: earlier selves, turned away
    if ghosts:
        for dx, dy, rot, a in ((-150, -6, -4, 0.35), (140, 4, 3, 0.25)):
            wob = math.sin(t * 0.4 + dx) * 14
            g = fx.shift(m, dx + wob, dy, rot)
            ge = np.clip(fx.blur(g, 1.5) - fx.blur(g, 4.5), 0, None) * 3.0
            col = C["her_edge_dark"] if dark else C["her_edge"]
            img = fx.over(img, col, np.clip(ge * a * (0.7 + 0.4 * kick), 0, 1) * brk * fade)

    # parts of her are simply not there
    gone = np.clip(0.25 + 0.95 * fx.fog(t * 0.35, seed=seed + 80, period=7.0), 0, 1)
    # the hollow: fog thins, what is behind bends slightly, like glass
    inner = fx.blur(m, 16) * m * (0.4 + 0.6 * gone)
    fg = fx.fog(t * 0.5, seed=seed + 60, period=6.0)
    dx = cv2.Sobel(fg, cv2.CV_32F, 1, 0, ksize=5) * 0.9
    dy = cv2.Sobel(fg, cv2.CV_32F, 0, 1, ksize=5) * 0.9
    yy, xx = fx._yy_xx()
    bent = cv2.remap(img, xx + dx * inner, yy + dy * inner, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)
    img = img * (1 - inner[..., None]) + bent * inner[..., None]
    void = C["void_dark"] if dark else C["void"]
    img = fx.over(img, void, inner * (0.28 if dark else 0.16) * fade)
    if fill > 0:  # his warmth pours in as mist, not as a solid
        breathe = 0.85 + 0.15 * math.sin(t * 1.1)
        mist = np.clip(0.5 + 0.6 * fx.fog(t * 0.8, seed=seed + 90, period=4.0), 0, 1.2)
        img = img + fill_col * (fx.blur(inner, 10) * mist * fill * 0.22 * breathe * fade)[..., None]

    # the outline: a thin broken line inside a wide soft halo
    thin = np.clip(fx.blur(m, 2.5) - fx.blur(m, 8), 0, None) * 2.6 * gone
    halo = np.clip(fx.blur(m, 12) - fx.blur(m, 34), 0, None) * 2.0 * gone
    line_col = C["her_edge_dark"] if dark else C["her_edge"]
    glow_col = C["her_glow_dark"] if dark else C["her_glow"]
    img = fx.over(img, line_col, np.clip(thin * brk * edge * (0.45 if dark else 0.85), 0, 1) * fade)
    img = img + glow_col * (halo * (0.6 + 0.6 * kick) * edge * fade)[..., None]

    # shedding: motes lifting off the contour
    if shed > 0:
        r = np.random.default_rng(seed + 70)
        ys, xs = np.nonzero(thin[::6, ::6] > 0.25)
        if len(xs):
            k = min(90, len(xs))
            pick = r.choice(len(xs), k, replace=False)
            born = r.uniform(0, 6, k)

            def motes(c):
                for i, j in enumerate(pick):
                    age = (t + born[i]) % 6.0
                    x = xs[j] * 6 + r.normal(0, 1) + age * 9 * (1 if i % 2 else -1) * 0.4
                    y = ys[j] * 6 - age * 16
                    a = (1 - age / 6.0) * min(1, age * 2) * 0.8
                    c.drawCircle(x, y, 1.2 + age * 0.35, skia.Paint(
                        AntiAlias=True, Color4f=skia.Color4f(1, 1, 1, a)))
            mm = fx.blur(fx.skia_alpha(motes), 0.8)
            img = img + glow_col * (mm * 1.2 * shed * fade)[..., None] if dark else \
                fx.over(img, line_col, mm * 0.6 * shed * fade)
    return img


# ================================================================ room
_QUILT = {}


def _quilt_mask(corner_x, floor_y, draw):
    key = (corner_x, floor_y)
    if key not in _QUILT:
        _QUILT[key] = fx.blur(fx.skia_alpha(draw), 7)
    return _QUILT[key]


def padded_room(t, base=None, corner_x=1330, floor_y=760, seam=0.06):
    base = C["haze"] if base is None else base
    img = np.empty((H, W, 3), F32)
    img[:] = base
    right = fx.hgrad(corner_x - 40, corner_x + 40)
    img *= (1 - 0.07 * right)[..., None]
    floor = fx.vgrad(floor_y - 30, floor_y + 120)
    img = img * (1 - floor[..., None] * 0.0) + (C["haze_hi"] - img) * (floor * 0.5)[..., None]

    def quilt(c):
        p = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=3,
                       Color4f=skia.Color4f(1, 1, 1, 1))
        for x in range(-40, corner_x, 230):
            c.drawLine(x, 0, x + 20, floor_y, p)
        for x in range(corner_x + 190, W + 200, 260):
            c.drawLine(x, 0, x - 30, floor_y + 10, p)
        for y in range(80, floor_y, 210):
            c.drawLine(0, y, corner_x, y + 10, p)
            c.drawLine(corner_x, y + 10, W, y - 20, p)
        c.drawLine(corner_x, 0, corner_x, floor_y, p)
        c.drawLine(0, floor_y, W, floor_y + 20, p)
    q = _quilt_mask(corner_x, floor_y, quilt)
    img = img * (1 - (q * seam * 1.6)[..., None])
    fg = fx.fog(t, seed=2)
    img = img + (fg * 0.035)[..., None]
    return img


def tally(img, t, T, x0=64, y0=84, color=None, alpha=0.55, blur=1.3, count=None,
          cols=5, dx=86, dy=112, fade=0.5):
    """One scratched mark per elapsed bar, grouped in fives: time made visible.
    Upper left, behind the lyric column and faded (`fade`) so it never
    competes with the words. The newest mark scratches itself in slowly
    over half a bar rather than appearing on the downbeat."""
    n = count if count is not None else T.since(T.bars, t)[0] + 1
    if n <= 0:
        return img
    color = C["graphite"] if color is None else color
    prog = 1.0
    if count is None and 0 <= n - 1 < len(T.bars):
        b0 = T.bars[n - 1]
        b1 = T.bars[n] if n < len(T.bars) else b0 + 3.4
        prog = clamp01((t - b0) / (0.5 * (b1 - b0)))
        prog = prog * prog * (3 - 2 * prog)

    def draw(c):
        r = np.random.default_rng(77)
        p = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=3.2,
                       StrokeCap=skia.Paint.kRound_Cap, Color4f=skia.Color4f(1, 1, 1, 1))
        for i in range(n):
            g, k = divmod(i, 5)
            gx = x0 + (g % cols) * dx + r.normal(0, 3)
            gy = y0 + (g // cols) * dy + r.normal(0, 3)
            if k < 4:
                xa = gx + k * 14
                p0 = (xa + r.normal(0, 2), gy + r.normal(0, 3))
                p1 = (xa + r.normal(0, 3), gy + 62 + r.normal(0, 4))
            else:
                p0 = (gx - 8, gy + 44 + r.normal(0, 3))
                p1 = (gx + 58, gy + 14 + r.normal(0, 3))
            u = prog if i == n - 1 else 1.0
            if u > 0.01:
                c.drawLine(p0[0], p0[1], p0[0] + (p1[0] - p0[0]) * u, p0[1] + (p1[1] - p0[1]) * u, p)
    m = fx.blur(fx.skia_alpha(draw), blur + 0.8)
    return fx.over(img, color, m * alpha * fade)


# ================================================================ type
TEXT_LOG = None  # set to a list to record (t, word, onset, x0, y0, x1, y1) for QA


def word_list(T, lines):
    out = []
    for n in lines:
        for w in T.lines[n]["words"]:
            d = dict(w)
            d["disp"] = display_text(w["text"])
            d["line_end"] = T.lines[n]["end"]
            out.append(d)
    return out


def line_rows(T, n, size, max_w):
    """How many rows her_words will wrap line n into."""
    f = font("her", size)
    sp = size * 0.26
    rows, cw = 1, 0.0
    for w in T.lines[n]["words"]:
        if w["backing"]:
            continue
        ww = f.measureText(display_text(w["text"]))
        if cw and cw + sp + ww > max_w:
            rows, cw = rows + 1, ww
        else:
            cw += (sp if cw else 0) + ww
    return rows


def stacked_lines(t, T, lines, anchor, size, max_w, lead=1.18, gap=0.55, window=2.6):
    """Place the visible lines so the newest sits on `anchor` (its last row)
    and older lines ease upward out of its way as it arrives.
    Returns [(line, first_row_baseline_y)]."""
    vis = [n for n in lines if T.lines[n]["start"] - 0.35 <= t < T.lines[n]["end"] + window][-2:]
    out = []
    for i, n in enumerate(vis):
        rows = line_rows(T, n, size, max_w)
        y = anchor - (rows - 1) * size * lead
        for m in vis[i + 1:]:
            e = ease_out(ramp(t, T.lines[m]["start"] - 0.35, T.lines[m]["start"] + 0.15), 2)
            y -= e * (line_rows(T, m, size, max_w) * size * lead + gap * size)
        out.append((n, y))
    return out


def her_words(img, t, T, lines, x, y, size=78, color=None, hot=False, align="left",
              ghost_dx=46, max_w=900, hold=1.3, release=1.1):
    """Her voice: focus-pull in, ghost exposures drifting out, dissolve up."""
    color = C["plum_ink"] if color is None else color
    f = font("her", size)
    words = [w for w in word_list(T, lines) if not w["backing"]]
    back = [w for w in word_list(T, lines) if w["backing"]]
    # wrap
    rows, cur, cw = [], [], 0.0
    sp = size * 0.26
    for w in words:
        ww = f.measureText(w["disp"])
        if cur and cw + sp + ww > max_w:
            rows.append((cur, cw)); cur, cw = [], 0.0
        cw += (sp if cur else 0) + ww
        cur.append((w, ww))
    if cur:
        rows.append((cur, cw))

    layers = {}  # sigma -> list of draw ops, merged per blur level

    def put(sig, op):
        layers.setdefault(round(sig * 2) / 2, []).append(op)

    yy = y
    for row, rw in rows:
        xx = x - (rw / 2 if align == "center" else 0)
        for w, ww in row:
            a_in = ease_out(ramp(t, w["start"] - 0.08, w["start"] + 0.45), 2)
            out = smooth(ramp(t, w["line_end"] + hold, w["line_end"] + hold + release))
            a = a_in * (1 - out)
            if a > 0.004:
                sig = (1 - a_in) * 10 + out * 14
                dy = -out * 36
                put(sig, (w["disp"], xx, yy + dy, a))
                if TEXT_LOG is not None and a > 0.5:
                    TEXT_LOG.append((t, w["disp"], w["start"], xx, yy + dy - size * 0.8,
                                     xx + ww, yy + dy + size * 0.25))
                for k, sgn in ((1, 1), (2, -1)):                            # ghosts
                    age = t - w["start"] - 0.12 * k
                    if age > 0:
                        drift = ghost_dx * k * sgn * ease_out(clamp01(age / 2.5))
                        ga = a * (0.34 / k) * (1 - 0.5 * clamp01(age / 4))
                        put(sig + 4 + 2 * k, (w["disp"], xx + drift, yy + dy - 6 * k, ga))
            xx += ww + sp
        yy += size * 1.18
    # backing echoes, small, trailing below
    fb = font("her", size * 0.56)
    by = yy + size * 0.1
    for w in back:
        a = ease_out(ramp(t, w["start"], w["start"] + 0.5))
        a *= 1 - smooth(ramp(t, w["line_end"] + hold, w["line_end"] + hold + release))
        if a > 0.004:
            put(2.5, (w["disp"], x + (rows[-1][1] if rows and align == "left" else 0) - 10, by,
                      a * 0.5, fb))
            put(8, (w["disp"], x + (rows[-1][1] if rows and align == "left" else 0) + 24, by + 10,
                    a * 0.22, fb))
            by += size * 0.6

    for sig, ops in layers.items():
        def draw(c, ops=ops):
            for op in ops:
                s, px, py, a = op[:4]
                ff = op[4] if len(op) > 4 else f
                if align == "right":
                    px -= ff.measureText(s)
                c.drawString(s, px, py, ff, skia.Paint(AntiAlias=True,
                                                     Color4f=skia.Color4f(1, 1, 1, min(1, a))))
        m = fx.skia_alpha(draw)
        if sig > 0.5:
            m = cv2.GaussianBlur(m, (0, 0), sig)
        img = fx.add(img, color, m * 1.4) if hot else fx.over(img, color, m)
    return img


def etched(img, t, T, lines, x, y, size=40, color=None, seed=5, big=None):
    """White-coat chant scratched into the image, letter by letter."""
    color = C["graphite"] if color is None else color
    f = font("coats", size)
    fbig = font("coats", size * 2.1)
    adv = f.measureText("M")
    words = [dict(w, disp=display_text(w["text"]).upper()) for w in word_list(T, lines)]

    def draw(c):
        r = np.random.default_rng(seed)
        cx, cy = x, y
        for w in words:
            is_big = big and w["text"].strip(".,…").upper() in big
            ff = fbig if is_big else f
            a_adv = ff.measureText("M")
            s = w["disp"]
            dur = max(0.15, w["end"] - w["start"])
            n = int(math.ceil(len(s) * clamp01((t - w["start"]) / dur)))
            if cx + len(s) * a_adv > W - 160:
                cx, cy = x, cy + size * 1.5
            for i, ch in enumerate(s):
                jx, jy, rot = r.normal(0, 1.4), r.normal(0, 1.8), r.normal(0, 2.5)
                if i < n:
                    c.save()
                    c.translate(cx + i * a_adv + jx, cy + jy)
                    c.rotate(rot)
                    c.drawString(ch, 0, 0, ff, skia.Paint(AntiAlias=True,
                                                          Color4f=skia.Color4f(1, 1, 1, 0.95)))
                    c.restore()
            cx += (len(s) + 1) * a_adv
    m = fx.skia_alpha(draw)
    erode = np.clip(0.55 + 0.7 * fx.fog(t * 0.2, seed=seed + 20, period=8.0), 0, 1)
    m = fx.blur(m, 0.9) * erode
    return fx.over(img, color, m * 0.9)


def him_light(img, t, T, lines, cx, y, size=30, tracking=0.3):
    """His words arrive as light through a crack: warm, steady, tracked caps."""
    f = font("him", size)
    words = [dict(w, disp=display_text(w["text"]).upper()) for w in word_list(T, lines)]
    total = sum(f.measureText(w["disp"]) + len(w["disp"]) * size * tracking + size * 0.8
                for w in words)

    def draw(c):
        x = cx - total / 2
        for w in words:
            a = ramp(t, w["start"] - 0.03, w["start"] + 0.25)
            flick = 1 if t - w["start"] > 0.25 else 0.6 + 0.4 * (int(t * 60) % 2)
            for ch in w["disp"]:
                if a > 0:
                    c.drawString(ch, x, y, f, skia.Paint(AntiAlias=True,
                                                         Color4f=skia.Color4f(1, 1, 1, a * flick)))
                x += f.measureText(ch) + size * tracking
            x += size * 0.8
    m = fx.skia_alpha(draw)
    img = fx.add(img, C["amber"], fx.blur(m, 10) * 0.9)
    return fx.add(img, C["amber_hot"], m * 1.6)


# ================================================================ events
def film_burn(img, t, t0, cx, cy, seed, cold=False):
    """A stab: the celluloid blisters and burns through. Blown core with
    yellow texture, a molten orange edge, and a brown scorch that spreads.
    `cold`: smaller, and white light burning through with a grey scorch
    (warmth belongs to him)."""
    age = t - t0
    if age < 0 or age > 1.6:
        return img
    r = (18 + 80 * ease_out(clamp01(age / 0.6), 2)) if cold else 25 + 140 * ease_out(clamp01(age / 0.7), 2)
    yy, xx = fx._yy_xx()
    n1 = fx.fog(seed * 7.0, seed=seed + 40, period=50.0)
    n2 = fx.fog(seed * 7.0 + 3, seed=seed + 41, period=50.0)
    if cold:                         # noise centred on the burn, so every stab is the same size
        iy, ix = int(np.clip(cy, 0, H - 1)), int(np.clip(cx, 0, W - 1))
        n1, n2 = n1 - n1[iy, ix], n2 - n2[iy, ix]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r + 0.30 * n1 + 0.10 * n2
    fade = 1 - smooth(clamp01((age - 0.7) / 0.9))
    core = np.clip((0.82 - d) / 0.12, 0, 1)
    edge = np.clip(1 - np.abs(d - 0.92) / 0.10, 0, 1)
    scorch = np.clip(1 - np.abs(d - 1.08) / 0.16, 0, 1)
    if cold:
        img = img * (1 - (scorch * 0.45 * fade)[..., None]) + hexc("#3E444C") * (scorch * 0.2 * fade)[..., None]
        img = img * (1 - (edge * 0.5 * fade)[..., None]) + hexc("#E4ECF4") * (edge * 0.9 * fade)[..., None]
        tex = 0.92 + 0.08 * n2
        hot = np.stack([1.12 * tex, 1.14 * tex, 1.18 * tex], -1)
        return img * (1 - (core * fade)[..., None]) + hot * (core * fade)[..., None]
    img = img * (1 - (scorch * 0.75 * fade)[..., None]) + hexc("#5A2A0E") * (scorch * 0.35 * fade)[..., None]
    img = img * (1 - (edge * 0.6 * fade)[..., None]) + hexc("#E0621C") * (edge * 1.3 * fade)[..., None]
    tex = 0.85 + 0.15 * n2
    hot = np.stack([1.25 * tex, 1.12 * tex, 0.85 * tex], -1)
    return img * (1 - (core * fade)[..., None]) + hot * (core * fade)[..., None]


def pencil_underline(img, t, t0, x0, x1, y, seed, t_end=None, color=None, size=1.0):
    """'Stab stab the charts': a pencil slash struck beneath a word. One fast,
    straight stroke rising a little to the right, pressed hardest where it
    starts and thinning out; it stays like an annotation and fades after
    `t_end`."""
    age = t - t0
    if age < 0 or (t_end is not None and t > t_end + 0.4):
        return img
    r = np.random.default_rng(seed)
    u = clamp01(age / 0.08)                              # the strike: fast, no easing
    fade = 1.0 if t_end is None else 1 - smooth(clamp01((t - t_end) / 0.4))
    xa, xb = x0 - r.uniform(6, 16) * size, x1 + r.uniform(14, 30) * size
    rise = (xb - xa) * math.tan(math.radians(r.uniform(2.5, 4.5)))
    ya = y + 0.6 * rise + r.uniform(0, 4) * size       # centred under the word
    yb = ya - rise
    xe, ye = xa + (xb - xa) * u, ya + (yb - ya) * u
    L = math.hypot(xb - xa, yb - ya)
    nx, ny = -(yb - ya) / L, (xb - xa) / L

    def draw(c):
        p = skia.Paint(AntiAlias=True, Color4f=skia.Color4f(1, 1, 1, 1))
        w0, w1 = 3.2 * size, 3.2 * size * (1 - 0.7 * u)  # thins toward the end
        path = skia.Path()
        path.moveTo(xa + nx * w0, ya + ny * w0)
        path.lineTo(xe + nx * w1, ye + ny * w1)
        path.lineTo(xe - nx * w1 * 0.3, ye - ny * w1 * 0.3)
        path.lineTo(xa - nx * w0, ya - ny * w0)
        path.close()
        c.drawPath(path, p)
        if u > 0.3:
            c.drawCircle(xa, ya, w0, p)                  # where the lead bit in
    m = fx.skia_alpha(draw)
    grain = 0.8 + 0.2 * np.clip(fx.fog(seed * 3.1, seed=seed + 70, period=5.0) / 2.5, -1, 1)
    color = hexc("#1C1F24") if color is None else color
    img = fx.over(img, color, fx.blur(m, 3.0) * 0.12 * fade)          # graphite smudge
    return fx.over(img, color, m * grain * 0.9 * fade)


def scratches(img, t, density, seed=6):
    """Emulsion scratches: jittering vertical hairlines plus a few scrawls."""
    if density <= 0.01:
        return img
    fi = int(t * 30)
    r = np.random.default_rng(seed * 7919 + fi // 2)

    def draw(c):
        p = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=1.3,
                       Color4f=skia.Color4f(1, 1, 1, 1))
        for _ in range(int(3 + 10 * density)):
            x = r.uniform(0, W)
            c.drawLine(x, r.uniform(-200, 300), x + r.normal(0, 6), r.uniform(600, H + 200), p)
        p.setStrokeWidth(2.0)
        for _ in range(int(2 * density)):
            pts, x, y = [], r.uniform(200, W - 200), r.uniform(150, H - 150)
            for _k in range(14):
                x += r.normal(0, 40); y += r.normal(0, 22)
                pts.append((x, y))
            c.drawPath(smooth_path(pts), p)
    m = fx.skia_alpha(draw)
    return fx.over(img, C["graphite"], m * 0.55)


def beams(img, t, T, cut_times, src=(1780, -160), targets=(600, 820, 1040), colors=None,
          open_times=None, strength=1.0):
    """His three virtues as shafts of warm light. Each opens (pours down from
    its source) at its open time and is shuttered at its cut."""
    colors = colors or [C["amber"], C["rose"], C["amber_hot"]]
    open_times = open_times or [None] * len(targets)
    g = np.clip(0.55 + 0.45 * fx.fog(t, seed=31, period=5.0), 0.1, 1.2)
    for k, (tx, ct, col, ot) in enumerate(zip(targets, cut_times, colors, open_times)):
        pour = 1.0 if ot is None else ease_out(clamp01((t - ot) / 0.9), 2)
        if pour <= 0.001:
            continue
        age = t - ct if ct is not None else -1
        # blade closes over 0.5 s from the source end, then the light gutters out
        close = ease_out(clamp01(age / 0.5)) if age >= 0 else 0.0
        gutter = (1 - smooth(clamp01((age - 0.4) / 1.2))) if age >= 0 else 1.0
        flick = 1.0 if age < 0.4 else (0.5 + 0.5 * math.sin(t * 90 + k)) ** 2
        if gutter <= 0.001:
            continue

        def draw(c, tx=tx):
            p = skia.Path()
            p.moveTo(src[0] - 18, src[1]); p.lineTo(src[0] + 18, src[1])
            p.lineTo(tx + 70, H + 60); p.lineTo(tx - 70, H + 60); p.close()
            c.drawPath(p, skia.Paint(AntiAlias=True, Color4f=skia.Color4f(1, 1, 1, 1)))
        m = fx.blur(fx.skia_alpha(draw), 9)
        along = fx.vgrad(src[1], H)
        if close > 0:  # dark blade: everything past the cut point goes
            m = m * (1 - np.clip((close * 1.15 - along) / 0.08, 0, 1))
        if pour < 1:   # light pours down from the source
            m = m * np.clip((pour * 1.15 - along) / 0.1, 0, 1)
        m = m * (0.55 + 0.45 * along) * g
        img = img + col * (m * 0.9 * gutter * flick * strength)[..., None]
    return img


def bullet(img_u8, x, y, scale=1.0, rot=-8.0, alpha=1.0):
    """The only hard-focus object in the film. Drawn after all softening."""
    if alpha <= 0.004:
        return img_u8
    s = skia.Surface(W, H)
    c = s.getCanvas()
    c.clear(skia.ColorTRANSPARENT)
    c.translate(x, y)
    c.rotate(rot)
    c.scale(scale, scale)
    body = skia.RRect.MakeRectXY(skia.Rect(-26, -7, 18, 7), 2, 2)
    grad = skia.GradientShader.MakeLinear([(0, -7), (0, 7)],
                                          [0xFF8A8F99, 0xFFE8ECF2, 0xFF3A3D44], [0, 0.35, 1])
    c.drawRRect(body, skia.Paint(AntiAlias=True, Shader=grad))
    tip = skia.Path()
    tip.moveTo(18, -7); tip.cubicTo(30, -6, 36, -2, 38, 0); tip.cubicTo(36, 2, 30, 6, 18, 7)
    tip.close()
    g2 = skia.GradientShader.MakeLinear([(0, -7), (0, 7)],
                                        [0xFF9A7A58, 0xFFF3DDBE, 0xFF4A3524], [0, 0.35, 1])
    c.drawPath(tip, skia.Paint(AntiAlias=True, Shader=g2))
    c.drawCircle(4, -3, 1.6, skia.Paint(AntiAlias=True, Color=0xFFFFFFFF))
    c.drawRect(skia.Rect(-2, -7, 0, 7), skia.Paint(Color=0x55000000))
    arr = s.makeImageSnapshot().toarray().astype(F32) / 255   # premultiplied
    a = arr[..., 3:4]
    base = img_u8.astype(F32) / 255
    shadow = fx.shift(fx.blur(arr[..., 3], 5), 3, 9) * 0.35 * alpha
    base = base * (1 - shadow[..., None])
    out = base * (1 - a * alpha) + arr[..., :3] * alpha
    return np.clip(out * 255, 0, 255).astype(np.uint8)
