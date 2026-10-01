"""Kinetic typography: every sung line is a small designed shot.

Each line gets a layout (stack / hero / depth / track), one hero word set
large against its support words, per-letter entrances timed inside each
word's sung duration, a slow camera push through the type, a nudge on every
kick, and an exit as the next line arrives.

Three voices share the engine but move differently:
  her    EB Garamond italic + roman hero; letters rise out of blur, shed
         ghost exposures, and the line is blown past the camera on exit.
  coats  IBM Plex Mono caps + bold hero; letters are typed on with jitter
         and fluorescent flicker, and blink out letter by letter.
  him    Inter Light caps, wide tracking, warm light; letters simply appear
         and the line stays still, then dims. Nothing about him performs.
"""
import math

import cv2
import numpy as np
import skia

import fx
from engine import H, W, clamp01, display_text, ease_out, font, ramp, smooth

TEXT_LOG = None  # list -> (t, word, onset, x0, y0, x1, y1) for QA

HERO_WORDS = ["lost", "found", "forever", "never", "once", "him", "her", "time", "find",
              "eyes", "years", "rooms", "self", "voice", "hand", "girl", "again", "charts",
              "white", "gun", "bullet", "ran", "cut", "anything", "world", "lifetime",
              "yes", "no", "sweet", "good", "leave", "learn", "swore", "would", "how",
              "stab", "worth", "man", "end", "bend", "protecting", "trusting", "believing",
              "arms", "legs", "eyes", "ears", "alone", "tunnel", "motel", "diners", "alleys"]
STYLES = ["stack", "hero", "depth", "track"]

BOX = (150, 170, 1000, 900)  # x0, y0, x1, y1: where a composition may live
TRACK_MAX = 0.14             # widest letter-spacing a 'track' line opens to (em)
TITLE_AT = (150, 830, 210)   # where 'lost and' is set when a word completes the title

# voice -> (support font, hero font, caps, support size, hero size)
VOICES = {
    "her": ("her", "her_roman", False, 70, 190),
    "coats": ("coats", "coats_bold", True, 56, 150),
    "him": ("him", "him_med", True, 40, 40),
}


def _clean(s):
    return s.replace('"', "").replace("“", "").replace("”", "")


def _norm(s):
    return "".join(ch for ch in s.lower() if ch.isalpha())


# ---------------------------------------------------------------- layout
def _choose(T, n, style=None, hero=None, caps=False):
    words = [dict(w, disp=display_text(w["text"]).upper() if caps else display_text(w["text"]))
             for w in T.lines[n]["words"] if not w["backing"]]
    if isinstance(hero, (list, tuple)):               # two heroes: a juxtaposed pair
        hero = [h for h in hero if 0 <= h < len(words)]
        if len(hero) == 2:
            return words, hero, "pair"
        hero = hero[0] if hero else None
    if hero is not None and not 0 <= hero < len(words):
        hero = None                                   # bad index: choose automatically
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


def layout(T, n, style=None, hero=None, seed=0, voice="her"):
    """Place the words of line n. Returns (placed, style); each placed word
    is dict(word, x, y, size, key, z, tracking, role)."""
    ki, kr, caps, small, big = VOICES[voice]
    if voice == "him":
        style = "spoken"
    words, hi, style = _choose(T, n, style, hero, caps)
    r = np.random.default_rng(n * 131 + seed)
    x0, y0, x1, y1 = BOX
    maxw = x1 - x0
    out = []

    if style == "spoken":
        # his words: one or two centred rows low in frame, wide tracking, no hero
        f = font(ki, small)
        tr = 0.26
        widths = [f.measureText(w["disp"]) + tr * small * len(w["disp"]) for w in words]
        rows, cur, cw = [], [], 0.0
        for i, ww in enumerate(widths):
            if cur and cw + ww > 1300:
                rows.append(cur)
                cur, cw = [], 0.0
            cur.append(i)
            cw += ww + small * 0.9
        rows.append(cur)
        y = 830 - (len(rows) - 1) * small * 1.9
        for row in rows:
            rw = sum(widths[i] for i in row) + small * 0.9 * (len(row) - 1)
            x = W / 2 - rw / 2
            for i in row:
                out.append(dict(word=words[i], x=x, y=y, size=small, key=ki, z=1.0,
                                tracking=tr, role="word"))
                x += widths[i] + small * 0.9
            y += small * 1.9
        return out, style

    if style == "pair":
        # two heroes set against each other on a diagonal (e.g. ever / once):
        # small words before, hero A, small words between, hero B offset right
        ha, hb = hi
        f = font(ki, small)
        sa = _fit(words[ha]["disp"], kr, big * 0.95, maxw * 0.55)
        sb = _fit(words[hb]["disp"], kr, big * 0.95, maxw * 0.55)
        groups = [(list(range(0, ha)), "word"), ([ha], "A"), (list(range(ha + 1, hb)), "word"),
                  ([hb], "B"), (list(range(hb + 1, len(words))), "word")]
        heights = {"A": sa * 0.95, "B": sb * 0.95, "word": small * 1.15}
        # small rows that follow a hero get room for its descenders (g, y, p)
        desc = {1: sa * 0.24, 3: sb * 0.24}
        steps = []
        for gi, (g, k) in enumerate(groups):
            if g:
                extra = desc.get(gi - 1, 0) if k == "word" and groups[gi - 1][0] else 0
                steps.append(heights[k] + extra)
        total = sum(steps)
        y = (y0 + y1) / 2 - total / 2
        a_w = font(kr, sa).measureText(words[ha]["disp"])
        si = 0
        for g, k in groups:
            if not g:
                continue
            y += steps[si]
            si += 1
            if k == "A":
                out.append(dict(word=words[ha], x=x0, y=y, size=sa, key=kr, z=1.0,
                                tracking=0.0, role="hero", stretch=True))
            elif k == "B":
                out.append(dict(word=words[hb], x=x0 + a_w * 0.85, y=y, size=sb, key=kr, z=1.0,
                                tracking=0.0, role="hero", pop=True))
            else:
                x = x0 + 30 + (a_w * 0.35 if g[0] > ha else 0)
                for i in g:
                    out.append(dict(word=words[i], x=x, y=y, size=small, key=ki, z=1.0,
                                    tracking=0.0, role="word"))
                    x += f.measureText(words[i]["disp"]) + small * 0.28

    elif style == "stack":
        # rows of words; the hero gets a row to itself, huge
        rows, cur = [], []
        for i, w in enumerate(words):
            if i == hi:
                if cur:
                    rows.append(cur)
                rows.append([i])
                cur = []
            else:
                cur.append(i)
                f = font(ki, small)
                if f.measureText(" ".join(words[j]["disp"] for j in cur)) > maxw * 0.9 or len(cur) == 3:
                    rows.append(cur)
                    cur = []
        if cur:
            rows.append(cur)
        heights = []
        for row in rows:
            if row == [hi]:
                heights.append(_fit(words[hi]["disp"], kr, big, maxw) * 0.92)
            else:
                heights.append(small * 1.12)
        y = (y0 + y1) / 2 - sum(heights) / 2
        indent = r.uniform(0, 60)
        for row, h in zip(rows, heights):
            y += h
            x = x0 + indent * (0 if row == [hi] else 1)
            for i in row:
                if i == hi:
                    sz = _fit(words[i]["disp"], kr, big, maxw)
                    out.append(dict(word=words[i], x=x, y=y, size=sz, key=kr, z=1.0,
                                    tracking=0.0, role="hero"))
                else:
                    f = font(ki, small)
                    out.append(dict(word=words[i], x=x, y=y, size=small, key=ki, z=1.0,
                                    tracking=0.0, role="word"))
                    x += f.measureText(words[i]["disp"]) + small * 0.28

    elif style == "hero":
        # hero word huge and faint behind; the whole line across it
        htxt = words[hi]["disp"] if caps else words[hi]["disp"].lower()
        hs = _fit(htxt, kr, 300, maxw * 1.05, tracking=0.08)
        cy = (y0 + y1) / 2
        out.append(dict(word=words[hi], x=x0 - 10, y=cy + hs * 0.32, size=hs, key=kr,
                        z=0.8, tracking=0.08, role="ghost_hero"))
        f = font(ki, small)
        line_w = sum(f.measureText(w["disp"]) for w in words) + small * 0.28 * (len(words) - 1)
        sz = small if line_w <= maxw else small * maxw / line_w
        f = font(ki, sz)
        x = x0 + 20
        for i, w in enumerate(words):
            out.append(dict(word=w, x=x, y=cy + sz * 0.3, size=sz, key=ki, z=1.1,
                            tracking=0.0, role="word"))
            x += f.measureText(w["disp"]) + sz * 0.28

    elif style == "depth":
        # words scattered through depth; the hero sits nearest and largest
        x = x0
        y = (y0 + y1) / 2
        for i, w in enumerate(words):
            is_h = i == hi
            z = 1.25 if is_h else r.uniform(0.75, 1.05)
            key = kr if is_h else ki
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

    else:  # track: one or more rows whose letter-spacing opens up over the line
        f = font(ki, small)
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
                key = kr if i == hi else ki
                sz = small * (1.35 if i == hi else 1.0)
                out.append(dict(word=words[i], x=x, y=y, size=sz, key=key, z=1.0,
                                tracking=0.04, role="hero" if i == hi else "word"))
                x += font(key, sz).measureText(words[i]["disp"]) + TRACK_MAX * sz * len(words[i]["disp"]) + small * 0.34
            y += small * 1.4

    # backing vocals: small echoes under the composition
    back = [dict(w, disp=display_text(w["text"]).upper() if caps else display_text(w["text"]))
            for w in T.lines[n]["words"] if w["backing"]]
    if back:
        last = max(out, key=lambda o: (o["y"], o["x"]))
        bx, by = x0 + 40, last["y"] + small * 1.1
        for w in back:
            out.append(dict(word=w, x=bx, y=by, size=small * 0.6, key=ki, z=0.9,
                            tracking=0.05, role="backing"))
            bx += font(ki, small * 0.6).measureText(w["disp"]) + small * 0.3
    return out, style


# ---------------------------------------------------------------- motion
def _lifetime(T, n, lines, hold=2.2):
    """(appear, exit_start) for line n: it leaves as the next line arrives."""
    start = T.lines[n]["start"] - 0.15
    idx = lines.index(n)
    nxt = T.lines[lines[idx + 1]]["start"] if idx + 1 < len(lines) else None
    end = T.lines[n]["end"] + hold
    last_seen = max(w["start"] for w in T.lines[n]["words"] if not w["backing"]) + 0.25
    if nxt is not None:                              # already leaving as the next arrives,
        end = min(end, max(nxt - 0.35, T.lines[n]["end"] - 0.1, last_seen))  # but not before
    return start, end                                # every word has been sung and seen


def _char_x(f, txt, tracking, size):
    xs, acc = [], 0.0
    for i in range(len(txt)):
        xs.append(f.measureText(txt[:i]) + acc)
        acc += tracking * size
    return xs


def _hash01(*k):
    h = 2166136261
    for v in k:
        h = ((h ^ (int(v) & 0xFFFFFFFF)) * 16777619) & 0xFFFFFFFF
    return h / 0xFFFFFFFF


INK = np.array([0.09, 0.07, 0.12], np.float32)
GRAPHITE = np.array([0.09, 0.1, 0.12], np.float32)
AMBER = np.array([1.0, 0.6, 0.29], np.float32)
AMBER_HOT = np.array([1.0, 0.8, 0.56], np.float32)
COLD = np.array([0.78, 0.86, 1.0], np.float32)


class Kinetic:
    """Draws the lines of one section.

    shots:   {line: dict(style=..., hero=..., voice=...)} per-line design
    voice:   default voice for lines without one
    color:   ink colour for 'her'/'coats' voices; `light=True` makes 'her'
             lines glow (additive, for dark scenes)
    hold:    how long the last line lingers after it is sung
    """

    def __init__(self, T, lines, overrides=None, color=None, glow_col=None, seed=0,
                 voice="her", light=False, hold=2.2, box=None):
        self.T, self.lines = T, [n for n in lines if n in T.lines]
        self.color = INK if color is None else np.asarray(color, np.float32)
        self.glow = np.array([0.97, 0.95, 0.98], np.float32) if glow_col is None else glow_col
        self.light = light
        self.plan = {}
        for n in self.lines:
            ov = (overrides or {}).get(n, {})
            v = ov.get("voice", voice)
            placed, style = layout(T, n, ov.get("style"), ov.get("hero"), seed, v)
            if v != "him":
                for o in placed:                    # alternate lines sit a little high / low
                    o["y"] += (55 if n % 2 else -55) + ov.get("dy", 0)
            if ov.get("backing_at"):                 # echoes placed on their own
                bx, by, bs = ov["backing_at"]
                for o in placed:
                    if o["role"] == "backing":
                        o.update(x=bx, y=by, size=bs, key=VOICES[v][0])
                        bx += font(o["key"], bs).measureText(o["word"]["disp"]) + bs * 0.3
            if v == "coats":                        # no punch, no stretch: clinical
                for o in placed:
                    o["pop"] = o["stretch"] = False
            if ov.get("shock"):                      # words slammed in whole, big, out of line;
                sh = ov["shock"]                     # the rest in two orderly rows around them
                lw = T.lines[n]["words"]
                keys = {(lw[k]["start"], lw[k]["text"]): j for j, k in enumerate(sh["words"])}
                ki_, kr_, _c, small_, _b = VOICES[v]
                y1 = (BOX[1] + BOX[3]) / 2 - 190 + (55 if n % 2 else -55)
                rows = ([], [])
                for o in placed:
                    if o["role"] not in ("word", "hero"):
                        continue
                    j = keys.get((o["word"]["start"], o["word"]["text"]))
                    if j is not None:
                        ax, ay = sh["at"][j]
                        o.update(x=BOX[0] + ax, y=y1 + ay, size=small_ * sh.get("size", 2.2),
                                 key=kr_, role="hero", tracking=0.0, shock=True)
                    else:
                        before = o["word"]["start"] < lw[sh["words"][0]]["start"]
                        rows[0 if before else 1].append(o)
                for row, y in zip(rows, (y1, y1 + sh.get("tail", 360))):
                    x = BOX[0]
                    for o in sorted(row, key=lambda q: q["word"]["start"]):
                        o.update(x=x, y=y, size=small_, key=ki_, role="word", tracking=0.0)
                        x += font(ki_, small_).measureText(o["word"]["disp"]) + small_ * 0.28
            if ov.get("set"):                        # phrases set row by row, by hand:
                lw = T.lines[n]["words"]             # [(word indices, scale, dx, flags), ...]
                style = "set"                        # (no style's own spacing or ghost word)
                placed = [o for o in placed if o["role"] != "ghost_hero"]
                ki_, kr_, caps_, small_, _b = VOICES[v]
                byk = {}
                for o in placed:
                    if o["role"] in ("word", "hero"):
                        for k, wk in enumerate(lw):
                            if wk["start"] == o["word"]["start"] and wk["text"] == o["word"]["text"]:
                                byk[k] = o
                rows = []
                for idxs, scale, dx, *fl in ov["set"]:
                    fl = dict(fl[0]) if fl else {}
                    wv = fl.pop("voice", v)
                    size = (VOICES[wv][3] if wv != v else small_) * scale
                    key = VOICES[wv][1] if scale > 1.3 else VOICES[wv][0]
                    rows.append((idxs, size, dx, key, wv, fl.pop("gap", 0.0), fl))
                total = sum(size * 1.05 + gap for _i, size, _d, _k, _v, gap, _f in rows)
                y = (BOX[1] + BOX[3]) / 2 - total / 2 + (55 if n % 2 else -55) + ov.get("dy", 0)
                for idxs, size, dx, key, wv, gap, fl in rows:
                    y += size * 1.05 + gap
                    x = BOX[0] + dx
                    for k in idxs:
                        if k not in byk:
                            continue
                        o = byk[k]
                        if wv != v:
                            o["voice"] = wv
                            o["word"]["disp"] = display_text(o["word"]["text"])
                            if VOICES[wv][2]:
                                o["word"]["disp"] = o["word"]["disp"].upper()
                        upd = dict(x=x, y=y, size=size, key=key, z=1.0, tracking=0.0,
                                   role="hero" if size > small_ * 1.3 else "word", pop=False,
                                   stretch=False)
                        upd.update(fl)
                        o.update(upd)
                        if o.get("bare"):                    # no trailing ellipsis
                            o["word"]["disp"] = o["word"]["disp"].rstrip(".\u2026")
                        x += font(key, size).measureText(o["word"]["disp"]) + size * 0.28
            if ov.get("rows_at"):                    # rows centred at fixed points:
                lw = T.lines[n]["words"]             # [(word indices, cx, y, scale, flags)]
                ki_, kr_, caps_, small_, _b = VOICES[v]
                byk = {}
                for o in placed:
                    if o["role"] in ("word", "hero"):
                        for k, wk in enumerate(lw):
                            if wk["start"] == o["word"]["start"] and wk["text"] == o["word"]["text"]:
                                byk[k] = o
                for idxs, rcx, ry, scale, *fl in ov["rows_at"]:
                    fl = dict(fl[0]) if fl else {}
                    wv = fl.pop("voice", v)
                    size = (VOICES[wv][3] if wv != v else small_) * scale
                    key = fl.pop("key", VOICES[wv][0])
                    if wv != v:                          # this row speaks in another voice
                        for k in idxs:
                            if k in byk:
                                byk[k]["voice"] = wv
                                byk[k]["word"]["disp"] = display_text(byk[k]["word"]["text"])
                                if VOICES[wv][2]:
                                    byk[k]["word"]["disp"] = byk[k]["word"]["disp"].upper()
                    tr = fl.pop("tracking", byk[idxs[0]]["tracking"] if idxs and idxs[0] in byk else 0.0)
                    f = font(key, size)
                    widths = [f.measureText(byk[k]["word"]["disp"]) + tr * size * len(byk[k]["word"]["disp"])
                              for k in idxs if k in byk]
                    gapw = size * (0.9 if v == "him" else 0.34)
                    x = rcx if fl.pop("left", False) else rcx - (sum(widths) + gapw * (len(widths) - 1)) / 2
                    for k, wd in zip([k for k in idxs if k in byk], widths):
                        upd = dict(x=x, y=ry, size=size, key=key, tracking=tr, z=1.0,
                                   role="hero" if scale > 1.3 else "word")
                        upd.update(fl)
                        byk[k].update(upd)
                        if byk[k].get("bare"):               # no trailing ellipsis
                            byk[k]["word"]["disp"] = byk[k]["word"]["disp"].rstrip(".\u2026")
                        x += wd + gapw
            if ov.get("overstrike"):                 # the repeat is typed over the first:
                ws = [o for o in placed if o["role"] in ("word", "hero")]   # True = second half
                pairs = ov["overstrike"]             # over first; or [(i, j), ...] word j over i
                if pairs is True:
                    half = len(ws) // 2
                    pairs = list(zip(ws[:half], ws[half:]))
                else:
                    lw = T.lines[n]["words"]
                    at = {(o["word"]["start"], o["word"]["text"]): o for o in ws}
                    pairs = [(at[(lw[i]["start"], lw[i]["text"])], at[(lw[j]["start"], lw[j]["text"])])
                             for i, j in pairs]
                for a_, b_ in pairs:
                    b_.update(x=a_["x"] + 3, y=a_["y"] - 2, size=a_["size"], key=a_["key"],
                              tracking=a_["tracking"], role=a_["role"])
            if ov.get("diffuse"):                    # words that let go after they are sung:
                dfo = ov["diffuse"]                  # keep=[i], after=i, backing=True, delay, dur
                lw = T.lines[n]["words"]
                keep = {(lw[k]["start"], lw[k]["text"]) for k in dfo.get("keep", ())}
                base_t = lw[dfo["after"]]["start"] if "after" in dfo else None
                for o in placed:
                    if o["role"] not in ("word", "hero", "backing"):
                        continue
                    if dfo.get("backing") and o["role"] != "backing":
                        continue
                    if (o["word"]["start"], o["word"]["text"]) in keep:
                        continue
                    t0 = (base_t if base_t is not None else o["word"]["start"]) + dfo.get("delay", 0.5)
                    o["diffuse"] = (max(t0, o["word"]["start"] + 0.5), dfo.get("dur", 2.5))
            if ov.get("nopop"):
                for o in placed:
                    o["pop"] = False
            if ov.get("swap"):                       # 'lost and found, found and lost': the
                sw = ov["swap"]                      # first words slide into the second's
                lw = T.lines[n]["words"]             # places as the second are sung
                style = "set"
                placed = [o for o in placed if o["role"] != "ghost_hero"]
                for o in placed:
                    if sw.get("lower"):
                        o["word"]["disp"] = o["word"]["disp"].lower()
                ki_, kr_, caps_, small_, _b = VOICES[v]
                key, size = sw.get("key", kr_), sw["size"]
                fsw = font(key, size)
                byk = {}
                for o in placed:
                    for k, wk in enumerate(lw):
                        if wk["start"] == o["word"]["start"] and wk["text"] == o["word"]["text"]:
                            byk[k] = o

                def lay(idxs):
                    xs_, x_ = {}, sw["at"][0]
                    for k in idxs:
                        xs_[k] = x_
                        x_ += fsw.measureText(byk[k]["word"]["disp"]) + size * 0.28
                    return xs_
                xa, xb = lay(sw["first"]), lay(sw["second"])
                for k, x_ in list(xa.items()) + list(xb.items()):
                    byk[k].update(x=x_, y=sw["at"][1], size=size, key=key, role="hero", z=1.0,
                                  tracking=0.0, pop=False, stretch=False)
                t_sw = lw[sw["second"][0]]["start"] - 0.25
                for kf, kt in sw["pairs"]:
                    byk[kf]["slide"] = (xb[kt] - xa[kf], t_sw, sw.get("dur", 1.0))
            if ov.get("stretch_both"):               # both heroes open slowly; nothing punches
                for o in placed:
                    if o["role"] == "hero":
                        o["stretch"], o["pop"] = True, False
            if ov.get("title_word") is not None:     # this word completes the title: it is set
                tw = T.lines[n]["words"][ov["title_word"]]   # in the space after 'lost and'
                tx, ty, ts = TITLE_AT
                ft = font("her_roman", ts)
                gap = tx + sum(ft.measureText(ch) + 0.01 * ts for ch in "lost and") + ft.measureText(" ")
                tail = {(T.lines[n]["words"][k]["start"], T.lines[n]["words"][k]["text"])
                        for k in ov.get("title_tail", ())}       # words that follow it on the line
                rest, tails = [], []
                for o in placed:
                    if o["word"]["start"] == tw["start"] and o["word"]["text"] == tw["text"]:
                        o.update(x=gap, y=ty, size=ts, key="her_roman", role="hero", title=True,
                                 pop=False, stretch=False, z=1.0, tracking=0.0)
                        tx_end = gap + font("her_roman", ts).measureText(o["word"]["disp"])
                    elif (o["word"]["start"], o["word"]["text"]) in tail:
                        tails.append(o)
                    elif o["role"] in ("word", "hero", "backing"):
                        rest.append(o)
                xt = tx_end + ts * 0.22
                for o in sorted(tails, key=lambda q: q["word"]["start"]):
                    sz = ts * 0.42
                    o.update(x=xt, y=ty, size=sz, role="word", title=True, z=1.0, tracking=0.0,
                             pop=False, stretch=False)
                    xt += font(o["key"], sz).measureText(o["word"]["disp"]) + sz * 0.28
                if rest:                             # the rest of the line sits above it
                    low = max(o["y"] + o["size"] * 0.3 for o in rest)
                    lift = min(0.0, ty - ts * 0.8 - 50 - low)
                    for o in rest:
                        o["y"] += lift
            for o in placed:
                if o["role"] == "hero" and ov.get("ghost_second") and o.get("pop"):
                    o["pop"], o["ghost"] = False, True
            if ov.get("split_second"):               # the 2nd echo peels off the 1st
                back = [o for o in placed if o["role"] == "backing"]
                if len(back) >= 2:
                    back[1].update(x=back[0]["x"], y=back[0]["y"], size=back[0]["size"],
                                   key=back[0]["key"], split=True)
            plan = dict(placed=placed, style=style, voice=v, opts=ov,
                        life=_lifetime(T, n, self.lines, hold))
            if ov.get("exit_at") is not None:        # held (or cut) to a set moment
                plan["life"] = (plan["life"][0], ov["exit_at"])
            if ov.get("title_word") is not None:
                plan["title_t"] = T.lines[n]["words"][ov["title_word"]]["start"]
            if ov.get("rewind"):
                order = sorted(((o["word"]["start"], pi, i) for pi, o in enumerate(placed)
                                if o["role"] in ("word", "hero") for i in range(len(o["word"]["disp"]))))
                stop = max(k for k, (_, pi, _i) in enumerate(order)
                           if placed[pi]["role"] == "hero") + 1
                plan["rw_order"] = [(pi, i) for _, pi, i in order]
                plan["rw_keep"] = stop                # never un-type the hero or before it
            if ov.get("split_second"):               # ...and is thrown out on the next
                idx = self.lines.index(n)            # line's first beat
                nxt = T.lines[self.lines[idx + 1]]["start"] if idx + 1 < len(self.lines) else \
                    T.lines[n]["end"] + 1.0
                plan["split_fling"] = float(min((q for q in T.pulses if q >= nxt), default=nxt))
            self.plan[n] = plan
        self._separate()

    @staticmethod
    def _yspan(p):
        ws = [o for o in p["placed"] if o["role"] in ("word", "hero", "backing")]
        if p.get("title_t") is not None:
            ws = [o for o in ws if o.get("title")]
        return (min(o["y"] - o["size"] * 0.85 for o in ws), max(o["y"] + o["size"] * 0.3 for o in ws))

    def _separate(self, gap=36, top=150, bottom=960):
        """If a line is still on screen when the next arrives, move the
        newcomer vertically clear of it (whichever way needs less travel)."""
        for a_n, b_n in zip(self.lines, self.lines[1:]):
            a, b = self.plan[a_n], self.plan[b_n]
            # only when the outgoing line is still fully readable as the newcomer's
            # first word appears (b's life starts 0.15 s before its first word)
            # (typed coats lines are readable the instant they land, so for them
            # any overlap with the outgoing line's life counts)
            lead = 0.0 if b["voice"] == "coats" else 0.15
            tail = 0.75 if a["voice"] == "her" else 0.0      # her lines unwrite slowly
            if a.get("opts", {}).get("exit") == "dissolve":
                tail = 2.0
            if "him" in (a["voice"], b["voice"]) or a["life"][1] + tail <= b["life"][0] + lead:
                continue
            if b.get("opts", {}).get("no_separate"):    # placed by hand around the other
                continue
            if b.get("title_t") is not None:   # a title line keeps its place: the other goes sooner
                a["fast"] = True
                continue
            a0, a1 = self._yspan(a)
            b0, b1 = self._yspan(b)
            if b1 <= a0 - gap or b0 >= a1 + gap:
                continue
            down = a1 + gap - b0
            up = a0 - gap - b1
            options = [s for s in (down, up) if top <= b0 + s and b1 + s <= bottom]
            if not options:            # no room to move it: the outgoing line leaves sooner
                a["fast"] = True
                continue
            shift = min(options, key=abs)
            for o in b["placed"]:
                o["y"] += shift

    def active(self, t):
        return [n for n in self.lines
                if self.plan[n]["life"][0] - 0.1 <= t <= self.plan[n]["life"][1] + 0.9]

    def draw(self, img, t, cam=(0.0, 0.0), kick=0.0, react=1.0):
        layers = {}   # (kind, sigma) -> [ops]; kind: ink | coat | warm | lit
        halo = []

        def put(kind, sig, op):
            layers.setdefault((kind, round(min(sig, 24) / 2) * 2), []).append(op)

        for n in self.lines:
            p = self.plan[n]
            v = p["voice"]
            a0, e0 = p["life"]
            opts = p.get("opts", {})
            fk = opts.get("foreknow", 0.0)
            dissolve = opts.get("exit") == "dissolve"
            linger = 3.2 if opts.get("split_second") else 1.8 if dissolve else 1.2 if v == "her" else 0.9
            if t < a0 - 0.1 - fk or t > e0 + linger:
                continue
            ex_len = {"her": 0.55, "coats": 0.28, "him": 0.9}[v]   # coats: gone before the next types
            ex = smooth(ramp(t, e0, e0 + ex_len))
            lead_end = max(w["end"] for w in self.T.lines[n]["words"] if not w["backing"])
            hidden = set()
            rw = opts.get("rewind")
            if rw and t >= rw["start"]:               # un-type backwards; each snap restores
                seg = max(s_ for s_ in [rw["start"]] + list(rw.get("snaps", ())) if s_ <= t)
                k = int(rw.get("rate", 30) * (t - seg))
                order = p["rw_order"]
                k = min(k, len(order) - p["rw_keep"])
                hidden = set(order[len(order) - k:]) if k > 0 else set()
            age = t - a0
            cx, cy = BOX[0] + 120, (BOX[1] + BOX[3]) / 2
            if v == "her":                          # drifts; never comes toward the viewer
                push = 1.0 + 0.004 * min(age, 8) if p.get("title_t") is None else 1.0
                fly = 1.0
            elif v == "coats":
                push = 1.0                              # typed and still: they never move
                fly = 1.0
            else:                                   # his words are light: they never move
                cx, cy = W / 2, 830
                push, fly = 1.0, 1.0
            push_line = push
            cur = None                               # the coats' cursor: after the last typed letter
            for pi, o in enumerate(p["placed"]):
                w = o["word"]
                v = o.get("voice", p["voice"])         # a word may speak in another voice
                push = 1.0 if v == "coats" else push_line
                kind = o.get("kind") or {"her": "lit" if self.light else "ink",
                                         "coats": "cold" if self.light else "coat", "him": "warm"}[v]
                txt = w["disp"] if (o["role"] != "ghost_hero" or v != "her") else w["disp"].lower()
                f = font(o["key"], o["size"])
                dur = max(0.18, w["end"] - w["start"])
                ws = w["start"] - (0.25 if o["role"] == "ghost_hero" else 0.0)
                if p["style"] == "track" and o["role"] != "backing":
                    track = o["tracking"] + (TRACK_MAX - o["tracking"]) * ease_out(clamp01(age / 5), 2)
                elif o.get("stretch"):                  # 'ever': spacing keeps opening
                    track = o["tracking"] + 0.2 * ease_out(clamp01((t - w["start"]) / 4.5), 2)
                elif o["role"] == "ghost_hero":
                    track = o["tracking"] * (1 + 1.5 * clamp01(age / 6))
                else:
                    track = o["tracking"]
                xs = _char_x(f, txt, track, o["size"])
                stagger = min(0.05 if v != "coats" else 0.06, dur * 0.7 / max(1, len(txt)))
                z = o["z"]
                drift_x = (6 + 10 * (z - 1)) * age * (1 if p["style"] == "depth" else 0.4)
                if v != "her":
                    drift_x = 0.0
                base_a = {"ghost_hero": 0.2 if v == "her" else 0.12, "backing": 0.6}.get(o["role"], 1.0)
                if o["role"] == "backing" and opts.get("backing_alpha"):
                    base_a = opts["backing_alpha"]
                if o.get("ghost"):                      # a word that seeps back rather than lands
                    base_a = 0.42
                word_alpha = 0.0
                for i, ch in enumerate(txt):
                    cs = ws if o.get("shock") else ws + i * stagger   # a shock lands whole
                    if v == "her":
                        u = ease_out(ramp(t, cs - 0.05, cs + 0.32), 3)
                    elif v == "coats":
                        u = 1.0 if t >= cs else 0.0            # typed on
                    else:
                        u = ease_out(ramp(t, cs - 0.03, cs + 0.14), 2)
                    if u <= 0.001:
                        if fk and t >= a0 - fk and v == "her":   # she already knows the words
                            pa = 0.26 * smooth(ramp(t, a0 - fk, a0 - fk + 0.6)) * \
                                (0.55 if o["role"] == "backing" else 1.0)
                            psx = cx + (o["x"] + xs[i] - cx) * push + cam[0] * z
                            psy = cy + (o["y"] - cy) * push + cam[1] * z
                            put(kind, 7, (ch, psx, psy, f, push, pa))
                        continue
                    if (pi, i) in hidden:
                        continue
                    if v == "coats":                            # blink out letter by letter
                        gone = t >= e0 + ex_len * _hash01(n, i, len(txt), 7)
                        a = 0.0 if gone else base_a
                        fl = 0.82 + 0.18 * _hash01(n, i, int(t * 30))
                        if o.get("shock") and t < ws + 0.3:     # crisp and solid as it hits
                            fl = 1.0
                        a *= fl
                    elif o.get("split"):                        # outlasts the line, then is
                        fl = p["split_fling"]                   # unwritten slowly, last of all
                        gt = fl + 1.1 * _hash01(n, pi, i, 17)
                        a = u * base_a * (1 - smooth(ramp(t, gt, gt + 0.9)))
                        ex = 0.0
                    elif dissolve:                              # unwritten, letter by letter
                        if o["role"] == "backing":             # the echoes outlast her words
                            dis = smooth(ramp(t, e0 + 0.2, e0 + 0.9))
                        else:
                            gt = lead_end + 0.35 + 1.3 * _hash01(n, pi, i, 11)
                            dis = smooth(ramp(t, gt, gt + 0.5))
                        a = u * (1 - dis) * base_a
                        ex = 0.0
                    elif v == "her":                            # her lines leave the same way,
                        if p.get("title_t") is not None and not o.get("title"):
                            gt = p["title_t"] + 0.15 + 0.35 * _hash01(n, pi, i, 13)   # step aside
                            dis = smooth(ramp(t, gt, gt + 0.4))                        # for the title
                        elif p.get("fast"):                     # just quicker
                            gt = e0 - 0.3 + 0.2 * _hash01(n, pi, i, 13)
                            dis = smooth(ramp(t, gt, gt + 0.3))
                        else:
                            gt = e0 + 0.5 * _hash01(n, pi, i, 13)
                            dis = smooth(ramp(t, gt, gt + 0.45))
                        a = u * (1 - dis) * base_a
                        ex = 0.0
                    else:
                        a = u * (1 - ex) * base_a
                    if i == 0:
                        word_alpha = a / max(base_a, 1e-3) if v in ("coats", "her") else u * (1 - ex)
                    px = o["x"] + xs[i] + drift_x
                    if o.get("slide"):                      # sliding into its twin's place,
                        sdx, st0, sdur = o["slide"]          # fading as it arrives
                        mv = smooth(clamp01((t - st0) / sdur))
                        px += sdx * mv
                        a *= (1 - 0.5 * math.sin(math.pi * mv)) * \
                            (1 - smooth(clamp01((t - st0 - sdur * 0.55) / (sdur * 0.5))))
                    if o.get("split"):                      # peels away and drifts, slowly
                        g = t - w["start"]
                        px += 95 * ease_out(clamp01(g / 3.5), 2)
                        py_split = -30 * ease_out(clamp01(g / 3.5), 2)
                    else:
                        py_split = 0.0
                    py = o["y"] + ((1 - u) * o["size"] * 0.35 if v == "her" else 0.0) + py_split
                    if v == "coats":                            # hand-set, slightly off
                        px += (_hash01(n, i, 1) - 0.5) * 3.0
                        py += (_hash01(n, i, 2) - 0.5) * 4.0
                    sx = cx + (px - cx) * push * fly * (1 + 0.1 * (z - 1) * age / 6)
                    sy = cy + (py - cy) * push * fly
                    sx += cam[0] * z + (sx - cx) * 0.25 * ex * (v == "her")
                    sy += cam[1] * z - 40 * ex * (v == "her")
                    sc = push * fly
                    if o.get("pop") and t >= w["start"]:    # 'once': lands with a punch
                        pop = 1 + 0.12 * math.exp(-7 * (t - w["start"]))
                        sx += xs[i] * push * (pop - 1)
                        sc *= pop
                    if o.get("rise") and t >= w["start"]:     # reaching: lifts slowly as it opens
                        sy -= 30 * ease_out(clamp01((t - w["start"]) / 5.0), 2)
                    if o.get("diffuse"):                     # lets go: blurs, lifts and fades
                        d0, dd = o["diffuse"]
                        a *= 1 - smooth(ramp(t, d0, d0 + dd))
                    if v == "her":
                        gone = 1 - a / max(base_a * u, 1e-3)      # how far unwritten
                        sig = (1 - u) * 9 + 16 * gone + (4 if o["role"] == "ghost_hero" else 0)
                        sy -= 34 * gone
                        if o.get("ghost"):
                            sig += 5
                    elif v == "coats":
                        sig = 0.8 + (3 if o["role"] == "ghost_hero" else 0)
                    else:
                        sig = (1 - u) * 4 + ex * 6
                    if a > 0.004:
                        put(kind, sig, (ch, sx, sy, f, sc, a))
                        if opts.get("cursor") and v == "coats" and (cur is None or cs >= cur[0]):
                            cur = (cs, sx + f.measureText(ch) * sc + 0.08 * o["size"], sy, f, sc, a, kind)
                        if kind == "coat" and self.light:        # coats' ink in the dark: a cold
                            put("coldhalo", 24, (ch, sx, sy, f, sc, a))   # patch of their light behind
                        if o["role"] in ("word", "hero") and ex < 0.5 and v == "her":
                            halo.append((ch, sx, sy, f, sc, a * 0.9))
                        g_age = t - cs
                        if v == "her" and o["role"] in ("word", "hero") and g_age > 0.1 and ex < 0.05:
                            for k, sgn in ((1, 1), (2, -1)):
                                d = 38 * k * sgn * ease_out(clamp01(g_age / 3))
                                ga = a * (0.26 / k) * (1 - 0.5 * clamp01(g_age / 5))
                                put(kind, sig + 5 + 3 * k, (ch, sx + d * sc, sy - 5 * k, f, sc, ga))
                        if o.get("rise") and self.light:         # and is warmed from within
                            put("warm", 12, (ch, sx, sy, f, sc, a * 0.55))
                        if v == "him":                          # his light: a soft bloom
                            put("warm", 10, (ch, sx, sy, f, sc, a * 0.9))
                if TEXT_LOG is not None and o["role"] in ("word", "hero") and word_alpha > 0.5 \
                        and ex < 0.05:
                    x0 = cx + (o["x"] + drift_x - cx) * push + cam[0] * z
                    y0 = cy + (o["y"] - cy) * push + cam[1] * z
                    wpx = (f.measureText(txt) + track * o["size"] * len(txt)) * push
                    TEXT_LOG.append((t, _clean(w["text"]), w["start"], x0, y0 - o["size"] * 0.8 * push,
                                     x0 + wpx, y0 + o["size"] * 0.25 * push))
            later = [self.T.lines[m]["start"] for m in self.lines if m > n]
            if cur is not None and t < e0 and not (later and t >= min(later)):   # one cursor: the
                c_t, ux, uy, uf, usc, ua, uk = cur           # line being typed; solid while
                idle = t - c_t                              # typing, then it blinks, waiting
                if idle < 0.3 or int((idle - 0.3) / 0.42) % 2 == 1:
                    put(uk, 1.2, ("_", ux, uy, uf, usc, ua * 0.9))

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
        for (kind, sig) in sorted(layers, key=lambda k: -k[1]):
            m = mask(layers[(kind, sig)])
            if sig > 0.5:
                m = cv2.GaussianBlur(m, (0, 0), sig)
            if kind == "ink":
                img = fx.over(img, self.color, m)
            elif kind == "coat":
                img = fx.over(img, GRAPHITE, m * 0.92)
            elif kind == "cold":                            # the coats' type as cold light
                img = fx.add(img, COLD, m * 1.1)
            elif kind == "coldhalo":
                img = fx.add(img, COLD, np.clip(m * 0.9, 0, 0.32))
            elif kind == "lit":
                img = fx.add(img, np.array([1.0, 0.93, 0.86], np.float32), m * 1.25)
            else:                                           # warm
                img = fx.add(img, AMBER if sig >= 8 else AMBER_HOT, m * (0.8 if sig >= 8 else 1.5))
        return img
