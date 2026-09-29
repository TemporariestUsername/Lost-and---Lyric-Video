"""Scenes for the "long exposure of a memory" look.

Nothing is drawn literally. She is a soft, multiply-exposed presence; he
is never drawn at all, only warm light leaking in; the institute is
overexposure, flicker, scratches and film burns; time is tally marks and
the exposure echo. The one hard-focus object in the film is the bullet.
"""
import math

import cv2
import numpy as np
import skia

import fx
from engine import H, W, clamp01, ease_out, font, ramp, smooth, smooth_path
from fx import F32, hexc

# ---------------------------------------------------------------- palette
C = {
    "haze": hexc("#E6E1EE"), "haze_hi": hexc("#F7F3F6"), "haze_lo": hexc("#B9B1C9"),
    "plum": hexc("#2B2233"), "plum_ink": hexc("#3A3048"), "night": hexc("#140F19"),
    "amber": hexc("#FF9A4A"), "amber_hot": hexc("#FFC58A"), "rose": hexc("#F2765E"),
    "graphite": hexc("#4A4652"), "clinic": hexc("#DDE3E6"), "body": hexc("#F3ECEE"),
    "shade": hexc("#9D92AE"),
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


def hair_curtain(cx, top, s, t, seed=0, n=700, part=1.0):
    """Long hair as procedural strands grouped in locks. Strands start over
    the scalp, slide round the skull, keep clear of the face, then fall.
    Returns a density mask in 0..1."""
    r = np.random.default_rng(seed)
    hcx, hcy = cx, top + 0.12 * s
    rx, ry = 0.088 * s, 0.115 * s
    surf = skia.Surface(W, H)
    c = surf.getCanvas()
    c.clear(skia.ColorTRANSPARENT)
    sway = math.sin(t * 0.45 + seed) * 0.010 * s
    locks = [dict(th=r.uniform(-175, -5), curl=r.normal(0, 0.0016) * s, ph=r.uniform(0, 6.28),
                  L=r.uniform(0.55, 0.95) * s, hang=r.uniform(0.10, 0.30)) for _ in range(46)]
    for i in range(n):
        lk = locks[i % len(locks)]
        th = math.radians(lk["th"] + r.normal(0, 7))
        side = -1 if math.degrees(th) < -90 else 1
        rad = r.uniform(0.85, 1.0)
        x, y = hcx + math.cos(th) * rx * rad, hcy + math.sin(th) * ry * rad
        vx, vy = side * 0.55, 0.55
        L = lk["L"] * r.uniform(0.85, 1.08)
        steps = 40
        step = L / steps
        pts = [(x, y)]
        for k in range(steps):
            vy += 0.20
            vx *= 0.86
            vx += math.sin(k * 0.45 + lk["ph"]) * 0.07 + lk["curl"] / s * 20
            if y > hcy + 0.35 * s:            # below the shoulders, hang straighter
                vx += (hcx + side * lk["hang"] * s - x) / s * 0.35
            vx += sway / s * 4 * (k / steps)
            sp = math.hypot(vx, vy) + 1e-6
            x, y = x + vx / sp * step, y + vy / sp * step
            # slide around the skull (push out of the head ellipse)
            ex, ey = (x - hcx) / rx, (y - hcy) / ry
            e = math.hypot(ex, ey)
            if e < 1.02:
                x, y = hcx + ex / (e + 1e-6) * rx * 1.02, hcy + ey / (e + 1e-6) * ry * 1.02
            # keep the face clear: a soft column below the brow
            if hcy - 0.02 * s < y < hcy + 0.30 * s and abs(x - hcx) < 0.07 * s * part:
                x = hcx + side * 0.07 * s * part
            pts.append((x, y))
        w = r.uniform(0.7, 2.2)
        a = r.uniform(0.05, 0.18)
        c.drawPath(smooth_path(pts), skia.Paint(
            AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=w,
            StrokeCap=skia.Paint.kRound_Cap, Color4f=skia.Color4f(1, 1, 1, a)))
    m = surf.makeImageSnapshot().toarray()[..., 3].astype(F32) / 255
    return np.clip(m * 2.0, 0, 1)


def face_hint(cx, top, s):
    """Pale face with the faintest shadows for eyes and mouth: a suggestion."""
    hcy = top + 0.12 * s
    face = fx.radial(cx, hcy + 0.05 * s, 0.075 * s, 1.5)
    eyes = (fx.radial(cx - 0.03 * s, hcy + 0.03 * s, 0.022 * s, 1.5)
            + fx.radial(cx + 0.03 * s, hcy + 0.03 * s, 0.022 * s, 1.5))
    mouth = fx.radial(cx, hcy + 0.115 * s, 0.018 * s, 1.5)
    return fx.blur(face, 4), fx.blur(eyes, 3), fx.blur(mouth, 3)


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
            hk = hair_curtain(cx, top, s, t, seed=seed + 11 + k, n=320)
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
    hk = hair_curtain(cx, top, s, t, seed=seed)
    img = fx.over(img, hair_col, fx.blur(hk, 1.0) * 0.9 * brk * fade)
    if rim > 0:
        lx, ly = light if light else (cx + 0.2 * s * rim_side, top + 0.3 * s)
        near = fx.radial(lx, ly, 0.45 * s, 1.3)
        lit = hk * near
        glow = fx.blur(lit, 2.0) * 0.9 + fx.blur(lit, 14) * 0.5
        img = img + C["amber"] * (glow * rim * fade)[..., None]
    return img


def vtop(floor, s):
    """Ghost exposures fade out below the shoulders."""
    return 1 - fx.vgrad(floor - 0.62 * s, floor - 0.28 * s)


# ================================================================ room
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
    q = fx.blur(fx.skia_alpha(quilt), 7)
    img = img * (1 - (q * seam * 1.6)[..., None])
    fg = fx.fog(t, seed=2)
    img = img + (fg * 0.035)[..., None]
    return img


def tally(img, t, T, x0=1440, y0=190, color=None, alpha=0.55, blur=1.3, count=None,
          cols=5, dx=92, dy=118):
    """One scratched mark per elapsed bar, grouped in fives: time made visible."""
    n = count if count is not None else T.since(T.bars, t)[0] + 1
    if n <= 0:
        return img
    color = C["graphite"] if color is None else color

    def draw(c):
        r = np.random.default_rng(77)
        p = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=3.2,
                       StrokeCap=skia.Paint.kRound_Cap, Color4f=skia.Color4f(1, 1, 1, 1))
        for i in range(n):
            g, k = divmod(i, 5)
            gx = x0 + (g % cols) * dx + r.normal(0, 3)
            gy = y0 + (g // cols) * dy + r.normal(0, 3)
            if k < 4:
                x = gx + k * 14
                c.drawLine(x + r.normal(0, 2), gy + r.normal(0, 3),
                           x + r.normal(0, 3), gy + 62 + r.normal(0, 4), p)
            else:
                c.drawLine(gx - 8, gy + 44 + r.normal(0, 3), gx + 58, gy + 14 + r.normal(0, 3), p)
    m = fx.blur(fx.skia_alpha(draw), blur)
    return fx.over(img, color, m * alpha)


# ================================================================ type
def word_list(T, lines):
    out = []
    for n in lines:
        for w in T.lines[n]["words"]:
            d = dict(w)
            d["disp"] = w["text"].replace('"', "")
            d["line_end"] = T.lines[n]["end"]
            out.append(d)
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
    words = [dict(w, disp=w["text"].upper().replace('"', "")) for w in word_list(T, lines)]

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
    words = [dict(w, disp=w["text"].upper().replace('"', "")) for w in word_list(T, lines)]
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
def film_burn(img, t, t0, cx, cy, seed):
    """A stab: the celluloid blisters and burns through. Blown core with
    yellow texture, a molten orange edge, and a brown scorch that spreads."""
    age = t - t0
    if age < 0 or age > 1.6:
        return img
    r = 25 + 140 * ease_out(clamp01(age / 0.7), 2)
    yy, xx = fx._yy_xx()
    n1 = fx.fog(seed * 7.0, seed=seed + 40, period=50.0)
    n2 = fx.fog(seed * 7.0 + 3, seed=seed + 41, period=50.0)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r + 0.30 * n1 + 0.10 * n2
    fade = 1 - smooth(clamp01((age - 0.7) / 0.9))
    core = np.clip((0.82 - d) / 0.12, 0, 1)
    edge = np.clip(1 - np.abs(d - 0.92) / 0.10, 0, 1)
    scorch = np.clip(1 - np.abs(d - 1.08) / 0.16, 0, 1)
    img = img * (1 - (scorch * 0.75 * fade)[..., None]) + hexc("#5A2A0E") * (scorch * 0.35 * fade)[..., None]
    img = img * (1 - (edge * 0.6 * fade)[..., None]) + hexc("#E0621C") * (edge * 1.3 * fade)[..., None]
    tex = 0.85 + 0.15 * n2
    hot = np.stack([1.25 * tex, 1.12 * tex, 0.85 * tex], -1)
    return img * (1 - (core * fade)[..., None]) + hot * (core * fade)[..., None]


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


def beams(img, t, T, cut_times, src=(1780, -160), targets=(600, 820, 1040), colors=None):
    """His three virtues as shafts of warm light. Each is shuttered at its cut."""
    colors = colors or [C["amber"], C["rose"], C["amber_hot"]]
    g = np.clip(0.55 + 0.45 * fx.fog(t, seed=31, period=5.0), 0.1, 1.2)
    for k, (tx, ct, col) in enumerate(zip(targets, cut_times, colors)):
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
        m = m * (0.55 + 0.45 * along) * g
        img = img + col * (m * 0.9 * gutter * flick)[..., None]
    return img


def bullet(img_u8, x, y, scale=1.0):
    """The only hard-focus object in the film. Drawn after all softening."""
    s = skia.Surface(W, H)
    c = s.getCanvas()
    c.clear(skia.ColorTRANSPARENT)
    c.translate(x, y)
    c.rotate(-8)
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
    shadow = fx.shift(fx.blur(arr[..., 3], 5), 3, 9) * 0.35
    base = base * (1 - shadow[..., None])
    out = base * (1 - a) + arr[..., :3]
    return np.clip(out * 255, 0, 255).astype(np.uint8)
