"""Kinetic lyric typography: three voices, all driven by word timings.

- her_voice   EB Garamond Italic; words bloom from blur on their sung onset,
              drift up and dissolve after the line. Backing words (the
              parenthesised echoes) are smaller, dimmer, with ghost copies.
- coats_voice IBM Plex Mono caps; typed character-by-character across each
              word's duration, with a block cursor that blinks on the pulse.
- him_voice   Inter Light caps, wide tracking, no motion: each word simply
              appears on time, like a dry close-mic subtitle.
"""
import math

import skia

from engine import (clamp01, draw_text, ease_out, fill, font, ramp, rgb,
                    smooth)


def _clean(s):
    return s.replace('"', "").replace("“", "").replace("”", "")


def _layout(words, f, max_w, space, gap_extra=0.0):
    """Greedy wrap -> list of lines, each a list of (word, width)."""
    lines, cur, cur_w = [], [], 0.0
    for w in words:
        ww = f.measureText(w["disp"]) + gap_extra * len(w["disp"])
        add = ww + (space if cur else 0)
        if cur and cur_w + add > max_w:
            lines.append((cur, cur_w))
            cur, cur_w = [], 0.0
            add = ww
        cur.append((w, ww))
        cur_w += add
    if cur:
        lines.append((cur, cur_w))
    return lines


def _words(T, line_nos, upper=False):
    out = []
    for n in line_nos:
        for w in T.lines[n]["words"]:
            d = dict(w)
            d["disp"] = _clean(w["text"]).upper() if upper else _clean(w["text"])
            d["line_end"] = T.lines[n]["end"]
            out.append(d)
    return out


def her_voice(canvas, t, T, line_nos, cx, y, size=88, c=None, glow=0.5,
              max_w=1500, leading=1.2, hold=1.4, release=0.9):
    c = c or rgb("#FFF4E6")
    f = font("her", size)
    fb = font("her", size * 0.62)
    words = _words(T, line_nos)
    lead = [w for w in words if not w["backing"]]
    back = [w for w in words if w["backing"]]
    lines = _layout(lead, f, max_w, size * 0.28)
    yy = y - (len(lines) - 1) * size * leading
    for ln, lw in lines:
        x = cx - lw / 2
        for w, ww in ln:
            a_in = ease_out(ramp(t, w["start"] - 0.10, w["start"] + 0.30))
            out = smooth(ramp(t, w["line_end"] + hold, w["line_end"] + hold + release))
            a = a_in * (1 - out)
            if a > 0.003:
                blur = (1 - a_in) * 16 + out * 22
                dy = (1 - a_in) * 18 - out * 40
                cc = skia.Color4f(c.fR, c.fG, c.fB, c.fA * a)
                if glow:
                    gc = skia.Color4f(c.fR, c.fG, c.fB, c.fA * a * glow)
                    draw_text(canvas, w["disp"], x, yy + dy, f, gc, blur=14,
                              blend=skia.BlendMode.kPlus)
                draw_text(canvas, w["disp"], x, yy + dy, f, cc, blur=blur * 0.35)
            x += ww + size * 0.28
        yy += size * leading
    # Backing echoes: right-aligned under the lead, three staggered ghosts.
    if back:
        bx = cx + lines[-1][1] / 2 if lines else cx
        by = yy - size * leading + size * 0.75
        for w in back:
            for k in range(3):
                a = ease_out(ramp(t, w["start"] + 0.12 * k, w["start"] + 0.12 * k + 0.4))
                out = smooth(ramp(t, w["line_end"] + hold, w["line_end"] + hold + release))
                a *= (1 - out) * (0.55, 0.25, 0.12)[k]
                if a > 0.003:
                    cc = skia.Color4f(c.fR, c.fG, c.fB, c.fA * a)
                    draw_text(canvas, w["disp"], bx + 14 * k, by + 8 * k, fb, cc,
                              blur=2.5 * k, align="right")
            by += size * 0.62


def coats_voice(canvas, t, T, line_nos, x, y, size=44, c=None, hot=None,
                hot_words=("stab",), cursor=True, max_w=1500):
    """Typed monospace. Returns the (x, y) of the cursor."""
    c = c or rgb("#1C2327")
    hot = hot or rgb("#D7263D")
    f = font("coats", size)
    fbold = font("coats_bold", size)
    words = _words(T, line_nos, upper=True)
    adv = f.measureText("M")
    col_max = int(max_w // adv)
    cx, cy, col_i = x, y, 0
    cur_pos = (x, y)
    for w in words:
        s = w["disp"]
        if col_i and col_i + 1 + len(s) > col_max:
            cx, cy, col_i = x, cy + size * 1.35, 0
        elif col_i:
            cx += adv
            col_i += 1
        dur = max(0.12, w["end"] - w["start"])
        n = int(math.ceil(len(s) * clamp01((t - w["start"]) / dur)))
        is_hot = any(s.lower().startswith(h) for h in hot_words)
        for i, ch in enumerate(s[:n]):
            draw_text(canvas, ch, cx + i * adv, cy, fbold if is_hot else f,
                      hot if is_hot else c)
        if n:
            cur_pos = (cx + n * adv, cy)
        cx += len(s) * adv
        col_i += len(s)
    if cursor:
        on = (int(t * 70.2 / 60 * 2) % 2) == 0
        if on:
            canvas.drawRect(skia.Rect.MakeXYWH(cur_pos[0] + 4, cur_pos[1] - size * 0.78,
                                               adv * 0.62, size * 0.92), fill(c))
    return cur_pos


def him_voice(canvas, t, T, line_nos, cx, y, size=34, c=None, tracking=0.22,
              max_w=1300, leading=1.9):
    c = c or rgb("#EDE6DA")
    f = font("him", size)
    words = _words(T, line_nos, upper=True)
    space = size * 0.7
    lines = _layout(words, f, max_w, space, gap_extra=size * tracking)
    yy = y - (len(lines) - 1) * size * leading / 2
    for ln, lw in lines:
        x = cx - lw / 2
        for w, ww in ln:
            a = ramp(t, w["start"] - 0.02, w["start"] + 0.08)
            if a > 0:
                cc = skia.Color4f(c.fR, c.fG, c.fB, c.fA * a)
                xx = x
                for ch in w["disp"]:
                    xx += draw_text(canvas, ch, xx, yy, f, cc) + size * tracking
            x += ww + space
        yy += size * leading
