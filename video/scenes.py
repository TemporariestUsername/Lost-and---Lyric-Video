"""Scene drawing functions. Each takes (canvas, t, T) and draws a full frame.

All characters and marks are original designs: the girl is a stipple /
point-cloud figure, the Finder is a single warm contour line, the white
coats are faceless clinical silhouettes, and the institute mark is a plain
circle-and-crosshair drawn here.
"""
import math

import numpy as np
import skia

from engine import (H, PAL, W, clamp01, col, draw_text, ease_out, fill,
                    font, glow_path, grain, graph_paper, poly, ramp, rgb,
                    rng, smooth, smooth_path, stroke, vignette)
from typo import coats_voice, her_voice, him_voice


# ============================================================ shared pieces
def institute_mark(canvas, x, y, r, c, w=2.0):
    canvas.drawCircle(x, y, r, stroke(c, w))
    canvas.drawCircle(x, y, r * 0.28, fill(c))
    canvas.drawLine(x - r * 1.35, y, x - r * 0.5, y, stroke(c, w))
    canvas.drawLine(x + r * 0.5, y, x + r * 1.35, y, stroke(c, w))
    canvas.drawLine(x, y - r * 1.35, x, y - r * 0.5, stroke(c, w))


# Stylised profile facing right, unit = head height; origin at crown.
PROFILE = [(-0.30, 1.30), (-0.36, 0.95), (-0.47, 0.55), (-0.40, 0.18),
           (-0.18, 0.00), (0.08, -0.01), (0.27, 0.12), (0.34, 0.33),
           (0.345, 0.43), (0.36, 0.50), (0.46, 0.63), (0.40, 0.675),
           (0.385, 0.72), (0.395, 0.765), (0.37, 0.795), (0.385, 0.835),
           (0.345, 0.88), (0.35, 0.95), (0.27, 1.02), (0.15, 1.06),
           (0.13, 1.18), (0.16, 1.34), (0.40, 1.46), (0.85, 1.60)]
HAIR = [(-0.18, 0.00), (-0.48, 0.20), (-0.60, 0.62), (-0.56, 1.05),
        (-0.46, 1.40), (-0.62, 1.62)]


def profile_path(cx, cy, scale, mirror=False, extra=None):
    s = -1 if mirror else 1
    pts = [(cx + s * x * scale, cy + y * scale) for x, y in (extra or PROFILE)]
    return smooth_path(pts)


def sample_path(path, spacing):
    """Evenly spaced points along a path."""
    out = []
    pm = skia.PathMeasure(path, False)
    while True:
        L = pm.getLength()
        d = 0.0
        while d <= L:
            pos, _tan = pm.getPosTan(d)
            out.append((pos.x(), pos.y()))
            d += spacing
        if not pm.nextContour():
            break
    return np.array(out)


def stipple(canvas, pts, t, seed, c, r=2.2, jitter=3.0, shimmer=0.4,
            glow=0.0):
    g = rng(seed)
    off = g.normal(0, jitter, pts.shape)
    ph = g.uniform(0, 6.28, len(pts))
    rad = g.uniform(0.5, 1.2, len(pts)) * r
    for (x, y), (ox, oy), p, rr in zip(pts, off, ph, rad):
        a = c.fA * (1 - shimmer + shimmer * (0.5 + 0.5 * math.sin(t * 2.1 + p)))
        cc = skia.Color4f(c.fR, c.fG, c.fB, a)
        if glow:
            canvas.drawCircle(x + ox, y + oy, rr * 3,
                              fill(skia.Color4f(c.fR, c.fG, c.fB, a * glow),
                                   blur=rr * 2, blend=skia.BlendMode.kPlus))
        canvas.drawCircle(x + ox, y + oy, rr, fill(cc))


def scribble_path(seed, x0, y0, w, h, n, t_frac):
    """A pen scribble: fast looping random walk confined to a box."""
    g = rng(seed)
    k = max(2, int(n * clamp01(t_frac)))
    pts, x, y = [], x0 + w * 0.1, y0 + h * 0.5
    vx, vy = w * 0.05, 0.0
    for i in range(k):
        ang = g.normal(0, 1.1)
        vx, vy = (vx * math.cos(ang) - vy * math.sin(ang),
                  vx * math.sin(ang) + vy * math.cos(ang))
        vx += (x0 + w * (0.15 + 0.7 * i / n) - x) * 0.08
        vy += (y0 + h * 0.5 - y) * 0.10
        sp = math.hypot(vx, vy) + 1e-6
        vx, vy = vx / sp * w * 0.045, vy / sp * h * 0.22
        x, y = x + vx, y + vy
        pts.append((x, y))
    return smooth_path(pts) if len(pts) > 2 else None


def stab_hole(canvas, x, y, age, seed, ink, red):
    """Pen stab: puncture with torn fibres and a red ink ring."""
    g = rng(seed)
    pop = ease_out(clamp01(age / 0.12))
    r = 9 * pop
    canvas.drawCircle(x, y, r * 2.4, fill(skia.Color4f(red.fR, red.fG, red.fB, 0.35),
                                          blur=4))
    canvas.drawCircle(x, y, r * 1.35, stroke(red, 2.2))
    canvas.drawCircle(x, y, r, fill(rgb("#0B0E10")))
    for a in g.uniform(0, 6.28, 9):
        L = g.uniform(8, 22) * pop
        canvas.drawLine(x + math.cos(a) * r, y + math.sin(a) * r,
                        x + math.cos(a) * (r + L), y + math.sin(a) * (r + L),
                        stroke(ink, 1.2))


def white_coat(canvas, x, y, s, t, T, tilt_seed, ink, paper, red):
    """Faceless clinician: blank oval head, coat with lapels and pens."""
    tilt = math.sin(t * 0.9 + tilt_seed) * 2.5 + T.pulse_env(t, 8) * 3.5
    canvas.save()
    canvas.translate(x, y)
    canvas.rotate(tilt * 0.3)
    head = skia.Rect.MakeXYWH(-0.21 * s, -1.62 * s, 0.42 * s, 0.54 * s)
    coat = poly([(-0.18 * s, -1.02 * s), (0.18 * s, -1.02 * s),
                 (0.58 * s, -0.86 * s), (0.66 * s, 0.4 * s),
                 (-0.66 * s, 0.4 * s), (-0.58 * s, -0.86 * s)], closed=True)
    canvas.drawPath(coat, fill(paper))
    canvas.drawPath(coat, stroke(ink, 2.4))
    canvas.drawLine(0, -1.02 * s, 0, 0.4 * s, stroke(ink, 1.6))
    canvas.drawPath(poly([(-0.18 * s, -1.02 * s), (-0.02 * s, -0.55 * s),
                          (-0.30 * s, -0.78 * s)]), stroke(ink, 1.6))
    canvas.drawPath(poly([(0.18 * s, -1.02 * s), (0.02 * s, -0.55 * s),
                          (0.30 * s, -0.78 * s)]), stroke(ink, 1.6))
    for i, c in enumerate((ink, ink, red)):
        px = 0.27 * s + i * 0.06 * s
        canvas.drawLine(px, -0.70 * s, px, -0.60 * s, stroke(c, 3.0))
    pocket = skia.Rect.MakeXYWH(0.21 * s, -0.60 * s, 0.26 * s, 0.16 * s)
    canvas.drawRect(pocket, fill(paper))
    canvas.drawRect(pocket, stroke(ink, 1.6))
    canvas.save()
    canvas.rotate(tilt)
    canvas.drawOval(head, fill(paper))
    canvas.drawOval(head, stroke(ink, 2.4))
    canvas.drawLine(-0.12 * s, -1.33 * s, 0.12 * s, -1.33 * s, stroke(ink, 1.3))
    canvas.restore()
    canvas.restore()


# ============================================================ White Coats
def scene_white_coats(canvas, t, T, lines=(26, 35)):
    ink, red, paper = col("ink"), col("red"), col("paper")
    canvas.clear(paper)
    graph_paper(canvas, PAL["grid"], minor=24, alpha=0.30, off=(t * 6, t * 2))

    # Header strip
    canvas.drawLine(120, 96, W - 120, 96, stroke(ink, 1.5))
    institute_mark(canvas, 150, 62, 14, ink, 1.8)
    f = font("coats", 20)
    draw_text(canvas, "OBSERVATION RECORD", 190, 70, font("coats_bold", 20), ink)
    draw_text(canvas, "SUBJECT: “VERY LOST”", 560, 70, f, col("ink_soft"))
    draw_text(canvas, "WARD: NICE WHITE ROOM", 900, 70, f, col("ink_soft"))
    bar_i, _ = T.since(T.bars, t)
    draw_text(canvas, f"OBS. {bar_i:03d} / WITH TIME", W - 120, 70, f,
              col("ink_soft"), align="right")

    # Clipboard chart, slightly askew
    canvas.save()
    canvas.translate(560, 500)
    canvas.rotate(-2.2 + T.pulse_env(t, 10) * 0.25)
    board = skia.Rect.MakeXYWH(-380, -330, 760, 700)
    canvas.drawRect(board.makeOffset(10, 12), fill(rgb("#000000", 0.10), blur=14))
    canvas.drawRect(board, fill(rgb("#FBFBF8")))
    canvas.drawRect(board, stroke(ink, 2.2))
    canvas.drawRoundRect(skia.Rect.MakeXYWH(-90, -352, 180, 44), 8, 8, fill(ink))
    fs = font("coats", 18)
    # ID photo with stipple girl
    ph = skia.Rect.MakeXYWH(-340, -280, 200, 240)
    canvas.drawRect(ph, stroke(ink, 1.6))
    canvas.save()
    canvas.clipRect(ph)
    pp = sample_path(profile_path(-205, -250, 150, mirror=True,
                                  extra=None), 5.5)
    hp = sample_path(profile_path(-205, -250, 150, mirror=True, extra=HAIR), 5.5)
    stipple(canvas, np.vstack([pp, hp]), t, 3, col("ink", 0.85), r=1.8,
            jitter=1.6, shimmer=0.15)
    canvas.restore()
    for i, lab in enumerate(["NAME: —", "AGE: —", "STATUS: VERY LOST",
                             "FOUND: YES", "HOW LOST: ?"]):
        draw_text(canvas, lab, -110, -250 + i * 44, fs, ink)
        canvas.drawLine(-110, -242 + i * 44, 330, -242 + i * 44,
                        stroke(col("ink_soft", 0.5), 1))
    # Vitals trace
    gx, gy, gw, gh = -340, 20, 680, 170
    canvas.drawRect(skia.Rect.MakeXYWH(gx, gy, gw, gh), stroke(col("ink_soft"), 1.2))
    for k in range(1, 6):
        canvas.drawLine(gx + gw * k / 6, gy, gx + gw * k / 6, gy + gh,
                        stroke(col("grid", 0.8), 0.8))
    g = rng(11)
    ys = np.cumsum(g.normal(0, 9, 90))
    ys = gy + gh / 2 + (ys - ys.mean()) * 0.8
    trace = poly([(gx + gw * i / 89, float(np.clip(y, gy + 8, gy + gh - 8)))
                  for i, y in enumerate(ys)])
    canvas.drawPath(trace, stroke(ink, 1.8))
    draw_text(canvas, "VITALS / LOSTNESS", gx, gy - 10, fs, col("ink_soft"))
    # Scribbles accumulate on each "scribble" word across the section
    scrib_words = [w for n in range(lines[0], lines[1] + 1) if n in T.lines
                   for w in T.lines[n]["words"]
                   if w["text"].lower().startswith("scribble")]
    for k, w in enumerate(scrib_words):
        frac = (t - w["start"]) / max(0.3, w["end"] - w["start"])
        if frac > 0:
            bx = gx + 30 + (k % 3) * 210
            by = gy + 10 + (k // 3 % 2) * 60
            p = scribble_path(100 + k, bx, by, 240, 120, 70, frac)
            if p:
                canvas.drawPath(p, stroke(col("ink", 0.9), 2.4))
    # Stab holes: one per "stab" word, plus a hot flash
    stab_words = [w for n in range(lines[0], lines[1] + 1) if n in T.lines
                  for w in T.lines[n]["words"] if w["text"].lower().startswith("stab")]
    for k, w in enumerate(stab_words):
        age = t - w["start"]
        if age >= 0:
            g2 = rng(500 + k)
            stab_hole(canvas, g2.uniform(-300, 300), g2.uniform(220, 330),
                      age, 700 + k, ink, red)
    canvas.restore()

    # Three white coats on the right
    for i, x in enumerate((1250, 1490, 1730)):
        white_coat(canvas, x, 700 + (i % 2) * 14, 205, t, T, i * 1.7, ink,
                   col("paper"), red)

    # Notes field with the typed chant
    fx, fy = 120, 880
    canvas.drawRect(skia.Rect.MakeXYWH(fx, fy, W - 240, 150), fill(rgb("#FBFBF8")))
    canvas.drawRect(skia.Rect.MakeXYWH(fx, fy, W - 240, 150), stroke(ink, 1.6))
    draw_text(canvas, "NOTES:", fx + 20, fy + 34, font("coats_bold", 18), col("ink_soft"))
    cur = [n for n in range(lines[0], lines[1] + 1)
           if n in T.lines and T.lines[n]["start"] <= t + 0.05]
    if cur:
        n = cur[-1]
        coats_voice(canvas, t, T, [n], fx + 130, fy + 95, size=46, max_w=W - 420)

    # Stab flash: brief red frame edge on stab onsets
    flash = max((math.exp(-14 * (t - w["start"])) for w in stab_words
                 if t >= w["start"]), default=0)
    if flash > 0.01:
        canvas.drawRect(skia.Rect(0, 0, W, H), stroke(rgb(PAL["red"], 0.8 * flash), 26))
    vignette(canvas, 0.18, "#3A4A50")
    grain(canvas, t, 0.10)


# ============================================================ Spoken I
def scene_spoken_help(canvas, t, T):
    canvas.clear(col("night"))
    bg = skia.GradientShader.MakeRadial(
        (W / 2, 470), 900, [rgb("#1C2440").toColor(), rgb(PAL["night"]).toColor()],
        [0, 1])
    canvas.drawRect(skia.Rect(0, 0, W, H), skia.Paint(Shader=bg))

    breathe = math.sin(t * 0.8) * 6
    # Him: one warm contour line, facing right
    HX, HY, HS = 560, 150, 400
    PX, PY, PS = 1360, 172, 370
    him = profile_path(HX + breathe, HY, HS)
    sod = col("sodium")
    glow_path(canvas, him, sod, 2.6, glow=16, strength=0.55)
    # Her: stipple profile facing left, with hair
    hp = profile_path(PX - breathe, PY, PS, mirror=True)
    hh = profile_path(PX - breathe, PY, PS, mirror=True, extra=HAIR)
    pts = np.vstack([sample_path(hp, 7), sample_path(hh, 7)])
    stipple(canvas, pts, t, 21, col("her", 0.95), r=2.3, jitter=2.4, glow=0.6)
    g = rng(5)
    cloud = np.column_stack([g.normal(PX + 100, 60, 260), g.normal(PY + 200, 110, 260)])
    stipple(canvas, cloud, t, 22, col("her", 0.25), r=1.4, jitter=0, shimmer=0.8)

    # Three threads, one per virtue, drawn on as each word is spoken
    virtues = [("PROTECTING", "protecting,", "sodium"),
               ("TRUSTING", "trusting", "rose"),
               ("BELIEVING", "believing", "her")]
    words = {w["text"]: w for w in T.lines[44]["words"]}
    x0, x1 = HX + 0.40 * HS, PX - 0.40 * PS
    for k, (label, key, cname) in enumerate(virtues):
        w = words[key]
        prog = ease_out(ramp(t, w["start"], w["start"] + 0.9), 2)
        if prog <= 0:
            continue
        pts = []
        for i in range(60):
            u = i / 59
            x = x0 + (x1 - x0) * u
            y = 300 + 62 * k - (1 - (2 * u - 1) ** 2) * 36 \
                + math.sin(u * 9 + t * 1.6 + k * 2.1) * 12 * (1 - abs(2 * u - 1))
            pts.append((x, y))
        p = smooth_path(pts)
        glow_path(canvas, p, col(cname), 1.8, glow=12, strength=0.9, trim=(0, prog))
        la = ramp(t, w["start"] + 0.3, w["start"] + 0.7)
        if la > 0:
            draw_text(canvas, label, (x0 + x1) / 2, 300 + 62 * k - 36 - 16,
                      font("him_med", 15), col(cname, 0.9 * la), align="center")

    fade = skia.GradientShader.MakeLinear(
        [(0, 700), (0, 860)], [rgb(PAL["night"], 0).toColor(),
                               rgb(PAL["night"], 1).toColor()], [0, 1])
    canvas.drawRect(skia.Rect(0, 700, W, H), skia.Paint(Shader=fade))
    him_voice(canvas, t, T, [44], W / 2, 925, size=32)
    vignette(canvas, 0.6)
    grain(canvas, t, 0.09)


# ============================================================ The Run
def scene_run(canvas, t, T, lines=(71,)):
    canvas.clear(col("night"))
    vx, vy = W / 2, 455
    sky = skia.GradientShader.MakeLinear(
        [(0, 0), (0, H)], [rgb("#0A0E1A").toColor(), rgb("#1A1426").toColor(),
                           rgb("#07090F").toColor()], [0, 0.45, 1])
    canvas.drawRect(skia.Rect(0, 0, W, H), skia.Paint(Shader=sky))

    # Tunnel rings rushing toward camera; brightness kicks on each pulse.
    speed = 2.6
    kick = T.pulse_env(t, 7)
    N = 22
    for i in range(N, 0, -1):
        z = ((i - t * speed) % N) + 0.35
        s = 1.0 / z
        hw, hh = 1500 * s, 900 * s
        if hw < 6:
            continue
        a = clamp01(1.2 - z / N * 1.4) * (0.55 + 0.45 * kick) * smooth((z - 0.5) / 0.9)
        rect = skia.Rect(vx - hw, vy - hh * 0.9, vx + hw, vy + hh * 0.55)
        canvas.drawRoundRect(rect, hw * 0.35, hw * 0.35,
                             stroke(col("sodium", a * 0.55), max(1.0, 7 * s)))
        # sodium lamps at the ring's upper shoulders
        for sx in (-1, 1):
            lx, ly = vx + sx * hw * 0.72, vy - hh * 0.82
            canvas.drawCircle(lx, ly, max(3, 40 * s),
                              fill(col("sodium", a), blur=max(4, 50 * s),
                                   blend=skia.BlendMode.kPlus))
            canvas.drawCircle(lx, ly, max(0.8, 5 * s), fill(rgb("#FFE9C2", a)))
    # Road: edge lines and moving centre dashes
    for sx in (-1, 1):
        canvas.drawLine(vx + sx * 30, vy + 8, vx + sx * 1100, H,
                        stroke(col("sodium", 0.45), 3))
    for i in range(14):
        z = ((i - t * speed * 1.5) % 14) + 0.6
        s0, s1 = 1 / z, 1 / (z + 0.35)
        y0, y1 = vy + 8 + 520 * s0, vy + 8 + 520 * s1
        if y0 > vy + 10:
            canvas.drawLine(vx, y0, vx, y1, stroke(rgb("#FFE9C2", clamp01(s0 * 2)),
                                                  max(1, 10 * s0)))

    band = skia.GradientShader.MakeLinear(
        [(0, 760), (0, 900)], [rgb("#07090F", 0).toColor(),
                               rgb("#07090F", 0.85).toColor()], [0, 1])
    canvas.drawRect(skia.Rect(0, 760, W, H), skia.Paint(Shader=band))

    # Motel sign flickers on with the word
    words = {w["text"].strip(".,").lower(): w for w in T.lines[71]["words"]}
    mw = words.get("motel")
    if mw and t >= mw["start"] - 0.1:
        age = t - mw["start"]
        flick = 1.0 if age > 0.35 else (0.3 + 0.7 * (int(age * 40) % 2))
        sx, sy = 1590, 150
        box = skia.Rect.MakeXYWH(sx - 70, sy - 20, 140, 440)
        canvas.drawRoundRect(box, 18, 18, fill(rgb("#0B0A12", 0.92)))
        canvas.drawRoundRect(box, 18, 18, stroke(col("rose", 0.6 * flick), 3))
        fsgn = font("coats_bold", 64)
        for i, ch in enumerate("MOTEL"):
            yy = sy + 60 + i * 80
            draw_text(canvas, ch, sx, yy, fsgn, col("rose", 0.9 * flick), blur=16,
                      align="center", blend=skia.BlendMode.kPlus)
            draw_text(canvas, ch, sx, yy, fsgn, rgb("#FFD9D2", flick), align="center")
        arrow = poly([(sx - 110, sy + 470), (sx + 40, sy + 470), (sx + 40, sy + 440),
                      (sx + 100, sy + 490), (sx + 40, sy + 540), (sx + 40, sy + 510),
                      (sx - 110, sy + 510)], closed=True)
        glow_path(canvas, arrow, col("neon", flick), 2.5, glow=14, strength=0.8)

    # Lyric in her voice, warm
    her_voice(canvas, t, T, list(lines), W / 2, 930, size=92,
              c=rgb("#FFE7C4"), glow=0.55)
    vignette(canvas, 0.65)
    grain(canvas, t, 0.08)
