"""Kinetic typography: every sung line is a small designed shot.

Each line gets a layout (stack / hero / depth / track), one hero word set
large in roman against italic support words, per-letter staggered entrances
inside each word's sung duration, a slow camera push through the type, a
nudge on every kick, and an exit where the line is blown past the camera
as the next one arrives. Ghost exposures trail every word (the film's look).
"""
import math

import cv2
import numpy as np
import skia

import fx
from engine import H, W, clamp01, ease_out, font, ramp, smooth

TEXT_LOG = None  # list -> (t, word, onset, x0, y0, x1, y1) for QA

HERO_WORDS = ["lost", "found", "forever", "never", "once", "him", "her", "time", "find",
              "eyes", "years", "rooms", "self", "voice", "hand", "girl", "again", "charts",
              "white", "gun", "bullet", "ran", "cut", "anything", "world", "lifetime",
              "yes", "no", "sweet", "good", "leave", "learn", "swore", "would"]
STYLES = ["stack", "hero", "depth", "track"]

BOX = (150, 170, 1000, 900)  # x0, y0, x1, y1: where a composition may live
TRACK_MAX = 0.14             # widest letter-spacing a 'track' line opens to (em)


def _clean(s):
    return s.replace('"', "").replace("“", "").replace("”", "")


def _norm(s):
    return "".join(ch for ch in s.lower() if ch.isalpha())


# ---------------------------------------------------------------- layout
def _choose(T, n, style=None, hero=None):
    words = [dict(w, disp=_clean(w["text"])) for w in T.lines[n]["words"] if not w["backing"]]
    if hero is None:
        ranked = [i for k in HERO_WORDS for i, w in enumerate(words) if _norm(w["disp"]) == k]
        hero = ranked[0] if ranked else max(range(len(words)), key=lambda i: len(words[i]["disp"]))
    style = style or STYLES[(n * 3) % len(STYLES)]
    return words, hero, style


def _fit(txt, key, size, max_w, tracking=0.0):
    f = font(key, size)
    w = f.measureText(txt) + tracking * size * max(0, len(txt) - 1)
    if w > max_w:
        size *= max_w / w
    return size


def layout(T, n, style=None, hero=None, seed=0):
    """Place the words of line n. Returns a list of placed words:
    dict(word, x, y, size, key, z, tracking, role)."""
    words, hi, style = _choose(T, n, style, hero)
    r = np.random.default_rng(n * 131 + seed)
    x0, y0, x1, y1 = BOX
    maxw = x1 - x0
    out = []
    small, big = 70, 190

    if style == "stack":
        # rows of words; the hero gets a row to itself, huge, in roman
        rows, cur = [], []
        for i, w in enumerate(words):
            if i == hi:
                if cur:
                    rows.append(cur)
                rows.append([i])
                cur = []
            else:
                cur.append(i)
                f = font("her", small)
                if f.measureText(" ".join(words[j]["disp"] for j in cur)) > maxw * 0.9 or len(cur) == 3:
                    rows.append(cur)
                    cur = []
        if cur:
            rows.append(cur)
        heights = []
        for row in rows:
            if row == [hi]:
                heights.append(_fit(words[hi]["disp"], "her_roman", big, maxw) * 0.92)
            else:
                heights.append(small * 1.12)
        y = (y0 + y1) / 2 - sum(heights) / 2
        indent = r.uniform(0, 60)
        for row, h in zip(rows, heights):
            y += h
            x = x0 + indent * (0 if row == [hi] else 1)
            for i in row:
                if i == hi:
                    sz = _fit(words[i]["disp"], "her_roman", big, maxw)
                    out.append(dict(word=words[i], x=x, y=y, size=sz, key="her_roman", z=1.0,
                                    tracking=0.0, role="hero"))
                else:
                    f = font("her", small)
                    out.append(dict(word=words[i], x=x, y=y, size=small, key="her", z=1.0,
                                    tracking=0.0, role="word"))
                    x += f.measureText(words[i]["disp"]) + small * 0.28

    elif style == "hero":
        # hero word huge and faint behind; the whole line in italic across it
        hs = _fit(words[hi]["disp"].lower(), "her_roman", 300, maxw * 1.05, tracking=0.08)
        cy = (y0 + y1) / 2
        out.append(dict(word=words[hi], x=x0 - 10, y=cy + hs * 0.32, size=hs, key="her_roman",
                        z=0.8, tracking=0.08, role="ghost_hero"))
        f = font("her", small)
        line_w = sum(f.measureText(w["disp"]) for w in words) + small * 0.28 * (len(words) - 1)
        sz = small if line_w <= maxw else small * maxw / line_w
        f = font("her", sz)
        x = x0 + 20
        for i, w in enumerate(words):
            out.append(dict(word=w, x=x, y=cy + sz * 0.3, size=sz, key="her", z=1.1,
                            tracking=0.0, role="word"))
            x += f.measureText(w["disp"]) + sz * 0.28

    elif style == "depth":
        # words scattered through depth; the hero sits nearest and largest
        f = font("her", small)
        x = x0
        y = (y0 + y1) / 2
        for i, w in enumerate(words):
            is_h = i == hi
            z = 1.25 if is_h else r.uniform(0.75, 1.05)
            key = "her_roman" if is_h else "her"
            sz = (small * 1.9 if is_h else small) * z
            ww = font(key, sz).measureText(w["disp"])
            if x + ww > x1 and x > x0:
                x = x0 + r.uniform(0, 80)
                y += small * 1.6
            out.append(dict(word=w, x=x, y=y + r.uniform(-50, 50), size=sz, key=key, z=z,
                            tracking=0.0, role="hero" if is_h else "word"))
            x += ww + small * 0.34
        ys = [o["y"] for o in out]
        shift = (y0 + y1) / 2 - (min(ys) + max(ys)) / 2
        for o in out:
            o["y"] += shift

    else:  # track: one or two rows whose letter-spacing opens up over the line
        f = font("her", small)
        rows, cur, cw = [], [], 0.0
        for i, w in enumerate(words):
            ww = f.measureText(w["disp"]) + TRACK_MAX * small * len(w["disp"])
            if cur and cw + ww > maxw * 0.95:
                rows.append(cur)
                cur, cw = [], 0.0
            cur.append(i)
            cw += ww + small * 0.4
        rows.append(cur)
        y = (y0 + y1) / 2 - (len(rows) - 1) * small * 0.7
        for row in rows:
            x = x0
            for i in row:
                key = "her_roman" if i == hi else "her"
                sz = small * (1.35 if i == hi else 1.0)
                out.append(dict(word=words[i], x=x, y=y, size=sz, key=key, z=1.0,
                                tracking=0.04, role="hero" if i == hi else "word"))
                x += font(key, sz).measureText(words[i]["disp"]) + TRACK_MAX * sz * len(words[i]["disp"]) + small * 0.34
            y += small * 1.4

    # backing vocals: small echoes trailing the last word
    back = [dict(w, disp=_clean(w["text"])) for w in T.lines[n]["words"] if w["backing"]]
    if back:
        last = max(out, key=lambda o: (o["y"], o["x"]))
        bx, by = x0 + 40, last["y"] + small * 1.1
        for w in back:
            out.append(dict(word=w, x=bx, y=by, size=small * 0.6, key="her", z=0.9,
                            tracking=0.05, role="backing"))
            bx += font("her", small * 0.6).measureText(w["disp"]) + small * 0.3
    return out, style


# ---------------------------------------------------------------- motion
def _lifetime(T, n, lines):
    """(appear, exit_start) for line n: it leaves as the next line arrives."""
    start = T.lines[n]["start"] - 0.15
    idx = lines.index(n)
    nxt = T.lines[lines[idx + 1]]["start"] if idx + 1 < len(lines) else None
    end = T.lines[n]["end"] + 2.2
    if nxt is not None:                              # already leaving as the next arrives,
        end = min(end, max(nxt - 0.35, T.lines[n]["end"] - 0.1))   # but not before it's sung
    return start, end


def _char_x(f, txt, tracking, size):
    xs, acc = [], 0.0
    for i in range(len(txt)):
        xs.append(f.measureText(txt[:i]) + acc)
        acc += tracking * size
    return xs


class Kinetic:
    def __init__(self, T, lines, overrides=None, color=None, glow_col=None, seed=0):
        self.T, self.lines = T, [n for n in lines if n in T.lines]
        self.color = np.array([0.09, 0.07, 0.12], np.float32) if color is None else color
        self.glow = np.array([0.97, 0.95, 0.98], np.float32) if glow_col is None else glow_col
        self.plan = {}
        for n in self.lines:
            ov = (overrides or {}).get(n, {})
            placed, style = layout(T, n, ov.get("style"), ov.get("hero"), seed)
            for o in placed:                        # alternate lines sit a little high / low
                o["y"] += 55 if n % 2 else -55
            self.plan[n] = dict(placed=placed, style=style, life=_lifetime(T, n, self.lines))

    def draw(self, img, t, cam=(0.0, 0.0), kick=0.0):
        layers = {}   # blur sigma -> [ops]
        halo = []     # readability halo ops (drawn blurred, in haze colour)

        def put(sig, op):
            layers.setdefault(round(min(sig, 24) / 2) * 2, []).append(op)

        for n in self.lines:
            p = self.plan[n]
            a0, e0 = p["life"]
            if t < a0 - 0.1 or t > e0 + 0.6:
                continue
            ex = smooth(ramp(t, e0, e0 + 0.55))             # exit progress
            age = t - a0
            # camera: slow push through the composition, kick nudge, exit fly-past
            cx, cy = BOX[0] + 120, (BOX[1] + BOX[3]) / 2  # push anchored near the left margin
            push = 1.0 + 0.02 * min(age, 8) + 0.012 * kick
            fly = 1.0 + 0.45 * ex ** 1.4
            for o in p["placed"]:
                w = o["word"]
                txt = w["disp"] if o["role"] != "ghost_hero" else w["disp"].lower()
                f = font(o["key"], o["size"])
                dur = max(0.18, w["end"] - w["start"])
                ws = w["start"] - (0.25 if o["role"] == "ghost_hero" else 0.0)
                if p["style"] == "track" and o["role"] != "backing":
                    track = o["tracking"] + (TRACK_MAX - o["tracking"]) * ease_out(clamp01(age / 5), 2)
                elif o["role"] == "ghost_hero":
                    track = o["tracking"] * (1 + 1.5 * clamp01(age / 6))
                else:
                    track = o["tracking"]
                xs = _char_x(f, txt, track, o["size"])
                stagger = min(0.05, dur * 0.6 / max(1, len(txt)))
                z = o["z"]
                drift_x = (6 + 10 * (z - 1)) * age * (1 if p["style"] == "depth" else 0.4)
                base_a = {"ghost_hero": 0.2, "backing": 0.6}.get(o["role"], 1.0)
                word_alpha = 0.0
                for i, ch in enumerate(txt):
                    cs = ws + i * stagger
                    u = ease_out(ramp(t, cs - 0.05, cs + 0.32), 3)
                    if u <= 0.001:
                        continue
                    a = u * (1 - ex) * base_a
                    if i == 0:
                        word_alpha = u * (1 - ex)
                    # entrance: rise and focus; exit: scatter outward from centre
                    px = o["x"] + xs[i] + drift_x
                    py = o["y"] + (1 - u) * o["size"] * 0.35
                    sx = cx + (px - cx) * push * fly * (1 + 0.1 * (z - 1) * age / 6)
                    sy = cy + (py - cy) * push * fly
                    sx += cam[0] * z + (sx - cx) * 0.25 * ex
                    sy += cam[1] * z - 40 * ex
                    sc = push * fly
                    sig = (1 - u) * 9 + ex * 16 + (4 if o["role"] == "ghost_hero" else 0)
                    if a > 0.004:
                        put(sig, (ch, sx, sy, f, sc, a))
                        if o["role"] in ("word", "hero") and ex < 0.5:
                            halo.append((ch, sx, sy, f, sc, a * 0.9))
                        # ghost exposures drifting off each settled letter
                        g_age = t - cs
                        if o["role"] in ("word", "hero") and g_age > 0.1 and ex < 0.05:
                            for k, sgn in ((1, 1), (2, -1)):
                                d = 38 * k * sgn * ease_out(clamp01(g_age / 3))
                                ga = a * (0.26 / k) * (1 - 0.5 * clamp01(g_age / 5))
                                put(sig + 5 + 3 * k, (ch, sx + d * sc, sy - 5 * k, f, sc, ga))
                if TEXT_LOG is not None and o["role"] in ("word", "hero") and word_alpha > 0.5 \
                        and ex < 0.05:
                    x0 = cx + (o["x"] + drift_x - cx) * push + cam[0] * z
                    y0 = cy + (o["y"] - cy) * push + cam[1] * z
                    wpx = (f.measureText(txt) + track * o["size"] * len(txt)) * push
                    TEXT_LOG.append((t, w["disp"], w["start"], x0, y0 - o["size"] * 0.8 * push,
                                     x0 + wpx, y0 + o["size"] * 0.25 * push))

        def mask(ops):
            def draw(c):
                for ch, x, y, f, sc, a in ops:
                    c.save()
                    c.translate(x, y)
                    c.scale(sc, sc)
                    c.drawString(ch, 0, 0, f, skia.Paint(AntiAlias=True,
                                                         Color4f=skia.Color4f(1, 1, 1, min(1, a))))
                    c.restore()
            return fx.skia_alpha(draw)

        if halo:                                            # lift the haze behind the words
            hm = cv2.GaussianBlur(mask(halo), (0, 0), 16)
            img = fx.over(img, self.glow, np.clip(hm * 1.6, 0, 0.55))
        for sig in sorted(layers, reverse=True):
            m = mask(layers[sig])
            if sig > 0.5:
                m = cv2.GaussianBlur(m, (0, 0), sig)
            img = fx.over(img, self.color, m)
        return img
