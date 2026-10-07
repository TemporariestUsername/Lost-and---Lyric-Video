"""Every section of the film as a world + photos + per-line shots + events.

Worlds share one grammar (haze, drifting memory photographs, kinetic type,
the exposure echo) and differ in light:
  her        lavender haze, her room; ink type
  night      plum dark with milky fog; his warmth; type as light
  institute  cold overexposure, fluorescent flicker, scratches, burns; typed caps
  run        sodium night, fast photos, streaks; type as light
  bleach     fog thinning to white (intro, "But...", outro)
"""
import math

import cv2
import numpy as np
import skia

import fx
import kinetic
import look
import memories as mem
from engine import H, W, clamp01, display_text, ease_out, font, ramp, smooth
from look import C, hexc

# ---------------------------------------------------------------- helpers
_CACHE = {}


def sec_of(T, name):
    return next(x for x in T.sections if x["name"] == name)


def word_at(T, line, prefix, nth=0):
    ws = [w for w in T.lines[line]["words"] if w["text"].lower().strip('"').startswith(prefix)]
    return ws[nth]["start"] if len(ws) > nth else None


def words_like(T, lines, prefix):
    return [w["start"] for n in lines if n in T.lines for w in T.lines[n]["words"]
            if w["text"].lower().strip('"').startswith(prefix)]


def cached(key, fn):
    if key not in _CACHE:
        _CACHE[key] = fn()
    return _CACHE[key]


def flashes(t, times, decay=9.0, attack=0.07):
    """Soft flashes: a short attack (no single-frame jump), then decay."""
    return max((min(1.0, (t - s) / attack) * math.exp(-decay * max(0.0, t - s - attack))
                for s in times if t >= s), default=0.0)


# ---------------------------------------------------------------- worlds
TALLY_INK, TALLY_PALE = hexc("#2E2838"), hexc("#D8CFE4")
TALLY_CONTRAST = (0.1, 0.3)       # on dark, on light (the bright grades compress the light)


def tally_layer(img, t, T):
    """Her count: one mark scratched in slowly on every bar, the same faint
    presence everywhere (dark ink on light, pale on dark, at a constant
    contrast against what is behind it). It goes only while they run; on
    'But...' the bars the run didn't count scratch back in, one per eighth
    note; its last mark lands on the last downbeat."""
    run0 = sec_of(T, "The Run")["start"]
    t_but = T.lines[78]["start"]
    a = 1.0
    count, last_prog = None, None
    if run0 <= t < t_but:                                # the one time it isn't counted
        a = 1 - smooth(ramp(t, run0, run0 + 0.6))
    elif t >= t_but:
        n_now = T.since(T.bars, t)[0] + 1
        n_run = T.since(T.bars, run0)[0] + 1
        k = int((t - t_but) / (60 / 70.2 / 2)) + 1       # one per eighth note
        if n_run + k < n_now:
            count = n_run + k
        a = smooth(ramp(t, t_but, t_but + 0.9))
    last = T.bars[-1]
    if t >= last - 1.4:                                  # the last mark, on the last downbeat
        count = T.since(T.bars, last - 1.5)[0] + 2
        last_prog = smooth(ramp(t, last - 1.4, last - 0.05))
    if a <= 0.004:
        return img
    bg = float(img[80:760:16, 40:540:16].mean())
    w = smooth(clamp01((bg - 0.3) / 0.3))                # pale on the dark, ink on the light
    col = TALLY_PALE * (1 - w) + TALLY_INK * w
    target = TALLY_CONTRAST[0] + (TALLY_CONTRAST[1] - TALLY_CONTRAST[0]) * w
    alpha = min(0.95, target / max(1e-3, abs(float(col.mean()) - bg)))
    return look.tally(img, t, T, color=col, alpha=alpha * a, blur=1.6, fade=1.0, count=count,
                      last_prog=last_prog)


def base_her(t, T, dark=0.0, tally=True):
    """`tally`: draw her count (at time `tally` if it is a number: the scene's
    own clock may be frozen or running backwards; her count never is)."""
    img = look.padded_room(t, base=C["haze"] * (0.95 - 0.35 * dark))
    return _with_tally(img, t, T, tally)


def _with_tally(img, t, T, tally):
    if tally is False:
        return img
    return tally_layer(img, t if tally is True else tally, T)


def base_night(t, T, tally=True, lift=1.0):
    img = np.empty((H, W, 3), np.float32)
    img[:] = hexc("#2A2233") * lift
    img = img + (fx.fog(t, seed=12) * 0.035)[..., None]
    return _with_tally(img, t, T, tally)


def base_institute(t, T, tally=True):
    img = look.padded_room(t, base=C["clinic"], seam=0.03)
    return _with_tally(img, t, T, tally)


def fluorescent(img, t, amount=1.0):
    fi = int(t * 30)
    fl = 1 + 0.05 * amount * math.sin(t * 47) + (-0.12 * amount if (fi * 7919) % 23 == 0 else 0)
    band = np.exp(-(((fx._yy_xx()[0] - (t * 380) % (H + 400) + 200) / 120) ** 2)) * 0.06 * amount
    return img * fl * (1 - band[..., None])


def streaks(img, amount):
    """Horizontal long-exposure smear of the bright parts (the run)."""
    if amount <= 0.01:
        return img
    hi = np.clip(img - 0.55, 0, None)
    k = np.ones((1, 61), np.float32) / 61
    sm = cv2.filter2D(cv2.resize(hi, (W // 2, H // 2)), -1, k)
    return img + cv2.resize(sm, (W, H)) * amount * 1.6


POST = {
    "her": dict(exposure=0.9, lift=0.08, sat=0.85, bloom=0.5, hal=0.55, thresh=1.0,
                diffusion=0.14, grain=0.045, trail=0.62,
                ghosts=((1.2, -160, 0, -5, 0.16), (2.4, 150, -10, 4, 0.10))),
    "night": dict(exposure=1.05, lift=0.07, sat=0.9, bloom=0.8, hal=0.8, thresh=0.62,
                  diffusion=0.2, grain=0.055, trail=0.6, shadow="#3A2F45",
                  ghosts=((1.2, -140, 0, -4, 0.045), (2.4, 130, -8, 3, 0.025))),   # light type doubles easily
    "institute": dict(exposure=1.02, lift=0.04, sat=0.5, bloom=0.7, hal=0.7, thresh=1.1,
                      diffusion=0.12, grain=0.06, trail=0.5, shadow="#7A8290",
                      ghosts=((0.8, -60, 0, 0, 0.14), (1.6, 60, 0, 0, 0.08))),
    "run": dict(exposure=1.1, lift=0.06, sat=1.0, bloom=0.95, hal=0.9, thresh=0.55,
                diffusion=0.18, grain=0.06, trail=0.55, shadow="#2E2436",
                ghosts=((0.6, -110, 0, 0, 0.05), (1.2, -220, 0, 0, 0.025))),   # trails, kept legible
    "bleach": dict(exposure=1.0, lift=0.1, sat=0.7, bloom=0.4, hal=0.4, thresh=1.05,
                   diffusion=0.2, grain=0.045, trail=0.66,
                   ghosts=((1.4, -120, 0, -3, 0.12), (2.8, 110, 0, 3, 0.08))),
}


def post(world, **kw):
    p = dict(POST[world])
    p.update(kw)
    return p


def kin(T, name, lines, shots, **kw):
    return cached(("kin", name), lambda: kinetic.Kinetic(T, lines, overrides=shots, **kw))


def mems(T, name, scenes, **kw):
    s = sec_of(T, name)
    return cached(("mem", name), lambda: mem.schedule(T, scenes, s["start"], s["end"], **kw))


def draw_mems(img, t, ms, warm=0.0, offset=(0.0, 0.0), freeze=None):
    tt = t if freeze is None else min(t, freeze)
    for m in ms:
        img = m.draw(img, tt, warm=warm, offset=offset)
    return img


# ---------------------------------------------------------------- title
def title(img, t, t_in, t_out, x=150, y=600, size=210, color=None, dark=False, still=False, alpha=1.0):
    """'lost and' surfacing letter by letter, with the empty space after it.
    `still`: it neither drifts nor opens (the outro: it stays where 'found' landed)."""
    color = C["plum_deep"] if color is None else color
    f = font("her_roman", size)
    txt = "lost and"
    a_out = 1 - smooth(ramp(t, t_out, t_out + 1.6))
    if t < t_in or a_out <= 0:
        return img
    layers = {}
    age = max(0.0, t - t_in)
    track = size * (0.01 + (0.0 if still else 0.07 * ease_out(clamp01(age / 12), 2)))   # spacing opens
    xx = x + (0.0 if still else 5.0 * age)                           # and the word drifts
    y = y - (0.0 if still else 2.0 * age)
    for i, ch in enumerate(txt):
        cs = t_in + i * 0.32
        u = ease_out(ramp(t, cs, cs + 1.4), 2)
        if u > 0.002:
            sig = round((1 - u) * 14 + (1 - a_out) * 10)
            layers.setdefault(sig, []).append((ch, xx, y + (1 - u) * 30, u * a_out))
        xx += f.measureText(ch) + track
    for sig, ops in layers.items():
        def draw(c, ops=ops):
            for ch, px, py, a in ops:
                c.drawString(ch, px, py, f, skia.Paint(AntiAlias=True,
                                                       Color4f=skia.Color4f(1, 1, 1, a)))
        m = fx.skia_alpha(draw)
        if sig > 0.5:
            m = cv2.GaussianBlur(m, (0, 0), sig)
        m = m * alpha
        img = fx.add(img, np.array([1, 0.95, 0.9], np.float32), m * 1.2) if dark else fx.over(img, color, m)
    return img


# ================================================================ sections
def intro_mems(T):
    """The intro's prints. Verse A carries them on across the cut (the same
    room, one take), so they are shared."""
    return mems(T, "Intro", ["lake_overcast", "curtain_bedroom"], every=2, life=10.0, seed=11,
                keep_left=900)


def intro(t, T, lines):
    s = sec_of(T, "Intro")
    dx, dy, dr = mem.drift(t)
    fade_in = smooth(ramp(t, 0, 4))
    img = base_her(t, T, tally=False)
    img = img * fade_in + (1 - fade_in) * np.float32(0.97)
    img = tally_layer(img, t, T)                                      # from the first downbeat
    img = fx.shift(img, dx * 0.5, dy * 0.5, dr * 0.5, 1.02)          # Verse A's camera
    img = draw_mems(img, t, intro_mems(T), offset=(dx, dy))
    img = title(img, t, T.bars[1], s["end"] - 2.0)
    # the look settles into Verse A's before the cut, so nothing jumps across it
    return img, _mix_post(post("bleach", exposure=0.95), post("her"),
                          smooth(ramp(t, s["end"] - 3.0, s["end"])))


ONCE_SHOTS = {11: dict(style="hero", hero=4), 12: dict(style="stack", hero=2),
              13: dict(style="hero", hero=3), 14: dict(style="stack", hero=2),
              15: dict(style="stack", hero=0), 16: dict(style="track", hero=2),
              17: dict(style="stack", hero=3), 18: dict(style="hero", hero=0)}


def there_comes_a_once(t, T, lines):
    name = "There comes a once"
    s = sec_of(T, name)
    u = clamp01((t - s["start"]) / (s["end"] - s["start"]))
    dx, dy, dr = mem.drift(t, seed=2)
    onces = words_like(T, lines, "once")
    fl = flashes(t, onces, 7.0)
    # the 'no' timeline drifts left, the 'yes' timeline drifts right
    side = -1 if t < T.lines[13]["start"] else 1
    img = base_her(t, T, dark=0.25 * smooth(u))
    img = fx.shift(img, dx * 0.5 + side * 40 * u, dy * 0.5, dr * 0.5, 1.03)
    img = draw_mems(img, t, mems(T, name, ["dusk_drive", "coast_grey", "fog_park", "lake_overcast",
                                           "streets_night", "fog_lamps"],
                                 every=2, life=10.0, seed=13, keep_left=900), offset=(dx, dy))
    # 'lost to the world': focus pulls until nothing is sharp
    pull = smooth(ramp(t, T.lines[17]["start"], T.lines[18]["start"] + 1))
    if pull > 0.01:
        img = img * (1 - pull * 0.6) + fx.blur(img, 10) * pull * 0.6
    img = kin(T, name, lines, ONCE_SHOTS).draw(img, t, cam=(dx * 0.35, dy * 0.35))   # no beat pulse
    img = img * (1 - 0.4 * fl) + np.float32(0.93) * 0.4 * fl   # each 'once' flashes toward white
    return img, post("her")


REFRAIN_SHOTS = {20: dict(style="stack", hero=1), 21: dict(style="stack", hero=4),
                 22: dict(style="stack", hero=2),
                 23: dict(hero=[0, 1]),                  # forever / again
                 24: dict(hero=[0, 1]),                  # never / again
                 107: dict(style="stack", hero=1), 108: dict(style="stack", hero=5),
                 109: dict(style="stack", hero=2),
                 110: dict(hero=[0, 1]),
                 111: dict(hero=[0, 1])}


def refrain(name, react):
    def scene(t, T, lines):
        s = sec_of(T, name)
        dx, dy, dr = mem.drift(t, seed=5)
        kick = T.pulse_env(t, 6.0) if react else 0.0
        img = base_her(t, T)
        img = fx.shift(img, dx * 0.5, dy * 0.5, dr * 0.5, 1.03 + 0.012 * kick)
        photos = ["curtain_window", "rain_window", "doorway_figure", "car_window_night"]
        img = draw_mems(img, t, mems(T, name, photos, every=2, life=9.0, seed=17 if react else 19,
                                     keep_left=900), offset=(dx, dy))
        # 'throw away everything he found': warmth pushed out a pulse at a time
        warm = 0.35 * (1 - smooth(ramp(t, s["start"], s["end"] - 4)))
        img = fx.light_leak(img, t, "right", strength=0.05 + warm * 0.3)
        img = kin(T, name, lines, REFRAIN_SHOTS, hold=3.2).draw(
            img, t, cam=(dx * 0.35, dy * 0.35), kick=kick, react=1.0 if react else 0.0)
        # 'Never again...' held, then the frame bleaches
        last = T.lines[lines[-1]]
        white = smooth(ramp(t, last["end"] + 0.8, s["end"]))
        img = img * (1 - white) + np.float32(1.0) * white
        g = post("her", exposure=0.9 + 0.3 * white)
        if not react:   # she can't hear: more of her agains pile up instead
            g["ghosts"] = ((1.0, -180, 0, -5, 0.24), (2.0, 170, -10, 4, 0.18), (3.2, -60, 12, 2, 0.12))
        return img, g
    return scene


# ================================================================ Break I: it arrives
# What she foresaw arrives. The prophecy's corridor, seeping in at the end
# of the refrain, becomes the room. Break I shares White Coats I's camera and
# memory stream, so the prints drift on across the cut; only the light
# differs. The music drives it (measured on the no-vocals stem):
#   106.4-113.0  steady groove: cold light, the hum and the scratches creep in
#   113.0-115.1  the top end fades ~30 dB (a closing filter): the picture
#                loses its detail the same way, going soft, dim and quiet
#   115.4-116.5  the highs slam back in a stutter (115.42 tick, 115.84 hit,
#                116.06, 116.53 downbeat): the tubes strike, then hold
COATS_I_PHOTOS = ["hospital_corridor", "cinderblock_plate", "stairwell_spiral"]   # no legible signs
CLOSE = (113.0, 115.1)
STRIKE = ((115.42, 0.22), (115.84, 0.75), (116.06, 0.9), (116.53, 1.0))


def _tubes(t):
    """Fluorescent tubes striking: each hit jumps the light up over two
    frames, then it sags back toward the level of the last hit before."""
    lvl, prev = 0.0, 0.0
    for t0, v in STRIKE:
        if t < t0:
            break
        rise = clamp01((t - t0) / 0.07)
        settle = prev + (v - prev) * 0.55 if v < 1.0 else 1.0
        peak = prev + (v - prev) * rise
        lvl = peak if rise < 1 else settle + (v - settle) * math.exp(-9 * (t - t0 - 0.07))
        prev = settle
    return lvl


def _mix_post(a, b, u):
    out = {}
    for k in set(a) | set(b):
        x, y = a.get(k, b.get(k)), b.get(k, a.get(k))
        if k == "shadow":
            c = fx.hexc(x) * (1 - u) + fx.hexc(y) * u
            out[k] = "#" + "".join(f"{int(round(v * 255)):02X}" for v in c)
        elif k == "ghosts":
            out[k] = tuple(tuple(p * (1 - u) + q * u for p, q in zip(g, h)) for g, h in zip(x, y))
        elif isinstance(x, (int, float)):
            out[k] = x * (1 - u) + y * u
        else:
            out[k] = y if u >= 0.5 else x
    return out


def break_i(t, T, lines):
    s = sec_of(T, "Break I")
    t0 = s["start"]
    build = smooth(ramp(t, t0, CLOSE[0]))                     # the institute creeping in
    close = smooth(ramp(t, *CLOSE)) * (1 - smooth(ramp(t, STRIKE[1][0], STRIKE[1][0] + 0.12)))
    tubes = _tubes(t)

    img = base_institute(t, T)
    dx, dy, dr = mem.drift(t, seed=61, amp=(18, 10))           # White Coats I's camera
    img = fx.shift(img, dx * 0.4, dy * 0.4, 0, 1.02)
    img = draw_mems(img, t, mems(T, "White Coats I", COATS_I_PHOTOS, every=2, life=9.0,
                                 seed=61, keep_left=1000), offset=(dx, dy))
    img = look.scratches(img, t, (0.4 + 1.6 * build) * (1 - 0.8 * close))    # ends at White Coats I's 2.0
    img = fluorescent(img, t, max((0.35 + 0.65 * build) * (1 - close), tubes))
    img = fx.vignette(img, 0.55 * (0.5 + 0.5 * build), "#5E6A78")
    if close > 0.01:                                           # the picture loses its highs
        img = fx.blur(img, 1 + 11 * close)
        img = img * (1 - 0.12 * close) + fx.hexc("#8E8698") * 0.12 * close
    g = post("institute", exposure=0.96 + 0.06 * build - 0.1 * close + 0.06 * tubes,   # -> 1.08
             sat=0.5 + 0.1 * close, diffusion=0.12 + 0.1 * close)

    # the refrain lets go: its last frame and 'Never again...' give way
    u = smooth(ramp(t, t0, t0 + 1.6))
    if u < 1:
        ref, rp = refrain_i(t, T, range(20, 25))
        img = ref * (1 - u) + img * u
        g = _mix_post(rp, g, u)
    return img, g


# The coats never move: typed on, still, blinked out. Against her drift.
COATS_SHOTS = {
    26: dict(style="track"),
    27: dict(hero=[1, 7]),                                   # FOUND / LOST: the title's axis
    28: dict(style="stack", hero=5,                          # STAB STAB slammed in, out of line
             shock=dict(words=[2, 3], at=[(40, 150), (430, 255)], size=2.5)),
    29: dict(hero=[2, 5]),                                   # SAFE / NICE: the euphemisms
    30: dict(style="stack", hero=3,
             shock=dict(words=[2, 3], at=[(480, 150), (60, 255)], size=2.5)),
    31: dict(style="stack", hero=4),                         # how LOST
    32: dict(style="track", hero=1, overstrike=True),                 # with time, typed over itself
    33: dict(style="stack", hero=5),                         # ANYTHING
    34: dict(style="track", hero=1, overstrike=True),
    35: dict(style="stack", hero=8),                         # ...hard to FIND
    # White Coats II: escalation, from 'too lost' to 'not worth the time'
    47: dict(set=[([0], 2.4, 0), ([1, 2], 1.3, 10)], overstrike=[(1, 3), (2, 4)]),   # BUT; with time x2
    48: dict(style="track"),
    49: dict(style="stack", hero=5),                         # hmm... too LOST
    50: dict(style="stack", hero=5,
             shock=dict(words=[2, 3], at=[(470, 150), (90, 255)], size=2.5)),
    51: dict(style="stack", hero=4),                         # too HARD to find
    52: dict(style="stack", hero=5,
             shock=dict(words=[2, 3], at=[(60, 150), (440, 255)], size=2.5)),
    53: dict(style="track", hero=1, overstrike=True),
    54: dict(hero=[1, 4]),                                   # FOUND / WORTH: the verdict
    55: dict(style="track", hero=1, overstrike=True),
    56: dict(style="stack", hero=7),                         # ...hard to FIND
    # White Coats III: they turn on him; her 'something hard to find' in their mouths
    83: dict(style="track"),                                             # and the coats said
    84: dict(hero=[2, 5]),                                               # this MAN / too LOST
    85: dict(style="stack", hero=5,
             shock=dict(words=[2, 3], at=[(480, 150), (60, 255)], size=2.5)),
    86: dict(dy=130, exit_at=344.4,
             set=[([0, 1, 2], 1.0, 0), ([3, 4], 2.2, 0, {"key": "coats_bold"}),   # CUT OUT
                  ([5], 1.0, 0),                                                   # his
                  ([6, 7, 8, 9], 1.6, 40, {"voice": "her"})],                      # something hard
             suture=dict(words=[6, 9], mode="operate")),                           # to find: marked
                                                                                   # round and sewn
    87: dict(style="stack", hero=5,
             shock=dict(words=[2, 3], at=[(60, 150), (440, 255)], size=2.5)),
    88: dict(style="track", hero=1, overstrike=True, ellipsis=True),
    89: dict(style="stack", hero=3),                                     # how LOST this girl
    90: dict(style="track", hero=1, overstrike=True, ellipsis=True),
    91: dict(set=[([0, 1, 2, 3, 4], 1.0, 0), ([5, 6, 7], 1.0, 0),
                  ([8], 3.0, 0, {"key": "coats_bold", "kind": "coatdark"})]),     # ANYTHING
    92: dict(set=[([0], 2.4, 0, {"key": "coats_bold"}),                            # EXCEPT...
                  ([1, 2, 3, 4], 1.6, 60, {"voice": "her"})]),                     # something hard to
                                                                                   # find, untouched
}


def _word_box(K, w):
    """Where a placed word sits (the coats' type never moves):
    (x0, top, x1, baseline, size, line's exit time)."""
    for p in K.plan.values():
        for o in p["placed"]:
            if o["word"]["start"] == w["start"] and o["word"]["text"] == w["text"]:
                f = font(o["key"], o["size"])
                return (o["x"], o["y"] - o["size"] * 0.72, o["x"] + f.measureText(o["word"]["disp"]),
                        o["y"], o["size"], p["life"][1])
    return None


def phrase_surgery(img, t, n, mode, t0, t1, b0, b1, cam):
    """Her 'something hard to find' in the coats' mouths, given the surgery
    his virtues get later (the same incision, the same red thread), but with
    nothing under it yet: the blade goes round the phrase while it is sung and
    it is sewn shut on 'find'."""
    e0 = b0[5]
    if t < t0 or t > e0 + 0.7:
        return img
    x0, x1, base, size = b0[0], b1[2], b0[3], b0[4]
    cx, cy = (x0 + x1) / 2 + cam[0], base - 0.3 * size + cam[1]
    wd, sc = x1 - x0, size / VIRTUE_SIZE * 0.85
    inc = cached(("plens", n, round(cx), round(cy)), lambda: _lens(cx, cy, wd / 2 + 40, 0.5 * size, 700 + n))
    sut = cached(("pthread", n, round(cx), round(cy)), lambda: _thread(cx, cy, wd, 710 + n, sc))
    fade = 1 - smooth(ramp(t, e0, e0 + 0.6))         # leaves with the words
    if mode == "operate":
        frac = ease_out(clamp01((t - t0 - 0.1) / max(0.4, t1 - t0 - 0.2)), 1.5)   # round, as sung
        img = trace(img, inc, frac, 0.9 * fade, ground="light")
        front = sut["x0"] + (sut["x1"] - sut["x0"]) * ease_out(clamp01((t - t1) / 0.6), 1.5)
        return sew(img, sut, front, fade, ground="light") if t >= t1 else img
    return img


def coats(name, photos, seed, hot=0.0):
    """`hot`: White Coats II and on run colder, brighter, more scratched."""
    def scene(t, T, lines):
        K = kin(T, name, lines, COATS_SHOTS, voice="coats", hold=0.2)   # last line blinks out at the cut
        img = base_institute(t, T)
        dx, dy, dr = mem.drift(t, seed=seed, amp=(18, 10))
        img = fx.shift(img, dx * 0.4, dy * 0.4, 0, 1.02)
        img = draw_mems(img, t, mems(T, name, photos, every=2, life=9.0, seed=seed, keep_left=1000),
                        offset=(dx, dy))
        scribs = words_like(T, lines, "scribble")
        sc = max((1 - clamp01((t - s) / 1.8)) for s in scribs if t >= s) \
            if any(t >= s for s in scribs) else 0.0
        img = look.scratches(img, t, 2.0 + 1.2 * hot + 9 * sc)
        stabs = [w for n in lines for w in T.lines[n]["words"]
                 if w["text"].lower().startswith("stab")]
        for k, w in enumerate(stabs):                   # struck underneath, hard, in pencil
            b = _word_box(K, w)
            if b:
                x0, top, x1, base, size, e0 = b
                img = look.pencil_underline(img, t, w["start"], x0 + dx * 0.2, x1 + dx * 0.2,
                                            base + 0.16 * size + 9 + dy * 0.2, 50 + k, t_end=e0,
                                            size=0.9 + 0.8 * (size / 150))
        img = fluorescent(img, t, 1.0 + 0.3 * hot)
        ops = []                                        # 'cut out his something hard to find':
        for n in lines:                                 # the cut his virtues get, planned on her phrase
            su = COATS_SHOTS.get(n, {}).get("suture")
            if su:
                lw = T.lines[n]["words"]
                b0, b1 = _word_box(K, lw[su["words"][0]]), _word_box(K, lw[su["words"][1]])
                if b0 and b1:
                    ops.append((n, su["mode"], lw[su["words"][0]]["start"], lw[su["words"][1]]["start"],
                                b0, b1))
        # 'not worth the time': once the verdict has been read, the exposure clips
        # to white for a beat on its last word
        clip = 0.0
        for st in [T.lines[n]["words"][-1]["start"] for n in lines if "worth" in T.lines[n]["text"]]:
            if t >= st:                                 # a quick swell (no single-frame jump), then decay
                clip = max(clip, smooth(min(1.0, (t - st) / 0.3)) * math.exp(-3.0 * max(0.0, t - st - 0.3)))
        img = fx.vignette(img, 0.55, "#5E6A78")
        img = K.draw(img, t, cam=(dx * 0.2, dy * 0.2))
        for n, mode, t0, t1, b0, b1 in ops:
            img = phrase_surgery(img, t, n, mode, t0, t1, b0, b1, (dx * 0.2, dy * 0.2))
        # each stab startles: the frame jolts and the exposure hits down for an instant
        jolt, jx, jy = 0.0, 0.0, 0.0
        for k, w in enumerate(stabs):
            if 0 <= t - w["start"] < 0.35:
                e = math.exp(-(t - w["start"]) / 0.06)
                a_ = 2.4 + 1.7 * k
                jolt, jx, jy = max(jolt, e), 11 * e * math.cos(a_), 7 * e * math.sin(a_)
        if jolt > 0.01:
            img = fx.shift(img, jx, jy)
        img = img + np.float32(0.36) * clip                # near-white, not blown
        # (the echo trail is cut on a stab so the word hits at full strength on its first frame)
        return img, post("institute", exposure=1.08 + 0.06 * hot + 0.16 * clip - 0.08 * jolt,
                         trail=POST["institute"]["trail"] * (1 - jolt))
    return scene


# ================================================================ They brought a man
# The title's missing word arrives. 'lost and ___' has waited since the
# intro; here 'found' is sung four times. Each second 'found...' is written
# into the empty space after a faint 'lost and', so the title reads whole for
# a moment, and again for her (recurrence). 'in her' / 'in him' echo from
# the same place: the mirror.
MAN_SHOTS = {
    37: dict(style="stack", hero=3, dy=-230),                          # MAN (clear of the coats' last line)
    38: dict(set=[([0, 1, 2], 1.0, 0), ([3, 4], 2.1, 40),              # COULD FIND against
                  ([5], 1.0, 90), ([6, 7, 8], 2.1, 140, {"ghost": True})]),   # HARD TO FIND, seeping
    39: dict(set=[([0, 1, 2, 3], 1.0, 0), ([4, 5], 2.4, 180, {"gap": 40}),   # in the end... HE FOUND
                  ([6, 7], 1.0, 200)], title_word=8),                  # apart; 'found...' completes
    40: dict(style="stack", hero=1, backing_at=(780, 330, 64),         # VERY; (in her) diffuses
             diffuse=dict(backing=True, delay=0.7, dur=2.4)),
    41: dict(set=[([0, 1, 2, 3], 1.0, 0), ([4, 5], 2.4, 180, {"gap": 40}),
                  ([6, 7], 1.0, 200)], title_word=8),                  # ...SHE FOUND
    42: dict(set=[([0], 1.0, 0), ([1], 2.6, 20, {"rise": True, "stretch": True}),   # FOREVER reaches;
                  ([2], 1.0, 90), ([3], 2.3, 170, {"voice": "coats", "bare": True, "kind": "coat"})],
             backing_at=(780, 330, 64),                                # EVER traps: the coats' grey
             diffuse=dict(keep=[3], after=3, delay=0.3, dur=2.6)),     # ink; all else lets go
}


def title_ghost(img, a):
    """'lost and' as light, where a word is about to complete it."""
    if a <= 0.004:
        return img
    tx, ty, ts = kinetic.TITLE_AT

    def mask():
        f = font("her_roman", ts)

        def draw(c):
            x = tx
            for ch in "lost and":
                c.drawString(ch, x, ty, f, skia.Paint(AntiAlias=True, Color4f=skia.Color4f(1, 1, 1, 1)))
                x += f.measureText(ch) + 0.01 * ts
        m = fx.skia_alpha(draw)                          # an echo, not a word: unreadable
        return cv2.GaussianBlur(m, (0, 0), 22), cv2.GaussianBlur(m, (0, 0), 44)   # on first viewing
    m, glow = cached(("title_ghost",), mask)
    warm = np.array([1.0, 0.93, 0.86], np.float32)
    img = fx.add(img, kinetic.AMBER, glow * a * 0.5)
    return fx.add(img, warm, m * a * 1.15)


def brought_a_man(t, T, lines):
    name = "They brought a man"
    s = sec_of(T, name)
    dx, dy, dr = mem.drift(t, seed=7)
    img = base_night(t, T)
    img = fx.shift(img, dx * 0.5, dy * 0.5, dr * 0.5, 1.03)
    bars = T.since(T.bars, t)[0] - T.since(T.bars, s["start"])[0]
    reach = clamp01(bars / 8)                            # his warmth searches a step per bar
    very = word_at(T, 40, "very") or s["start"]
    flare = math.exp(-2.0 * (t - very)) if t >= very else 0.0
    img = draw_mems(img, t, mems(T, name, ["doorway_figure", "dark_street", "fog_lamps",
                                           "parking_lamp", "car_window_night", "rain_glass_lights"],
                                 every=2, life=10.0, seed=29, keep_left=900),
                    warm=0.3 + 0.5 * reach, offset=(dx, dy))
    img = fx.light_leak(img, t, "right", strength=0.12 + 0.5 * reach)
    img = fx.hotspot(img, t, 1500 - 400 * reach, 520, 260, strength=0.15 + 0.5 * flare)
    K = kin(T, name, lines, MAN_SHOTS, light=True)
    for n in lines:                                      # 'lost and' surfaces for each 'found...'
        tt = K.plan[n].get("title_t")
        if tt is not None:
            e0 = K.plan[n]["life"][1]
            img = title_ghost(img, 0.26 * smooth(ramp(t, tt - 0.5, tt + 0.2)) *
                              (1 - smooth(ramp(t, e0 + 0.1, e0 + 0.9))))
    img = K.draw(img, t, cam=(dx * 0.35, dy * 0.35))
    g = post("night")
    # they bring him in out of the white: the coats' room drains into his night
    u = smooth(ramp(t, s["start"], s["start"] + 1.8))
    if u < 1:
        ref, rp = coats("White Coats I", COATS_I_PHOTOS, 61)(t, T, range(26, 36))
        img = ref * (1 - u) + img * u
        g = _mix_post(rp, g, u)
    return img, g


VIRTUE_WORDS = ("protecting", "trusting", "believing")


def virtue_opens(T):
    return [word_at(T, 44, v) for v in VIRTUE_WORDS]


# ================================================================ Spoken I / II
# His words are light and never move. The virtues he offers each sit in
# their own shaft, lit as it pours down: the same shafts are cut in The
# Cutting. (Beam k runs from SRC down to (target k, below the frame).)
BEAM_SRC, BEAM_TARGETS = (1780, -160), (690, 860, 1030)


def beam_x(k, y):
    """Where beam k crosses height y."""
    tx = BEAM_TARGETS[k]
    return BEAM_SRC[0] + (tx - BEAM_SRC[0]) * (y - BEAM_SRC[1]) / (H + 60 - BEAM_SRC[1])


SPOKEN_SHOTS = {
    44: dict(voice="him", rows_at=[([0, 1, 2, 3, 4], beam_x(0, 400), 400, 1.0),            # I can help you find
                                   ([5], beam_x(0, 530), 530, 2.0, {"tracking": 0.16}),     # PROTECTING
                                   ([6], beam_x(1, 680), 680, 2.0, {"tracking": 0.16}),     # TRUSTING
                                   ([7], beam_x(2, 760), 760, 1.0),                          # and
                                   ([8], beam_x(2, 860), 860, 2.0, {"tracking": 0.16})]),   # BELIEVING
    45: dict(voice="him", rows_at=[([0, 1, 2], 520, 300, 1.0),                               # even if they're
                                   ([3, 4, 5, 6], 520, 360, 1.0)]),                          # very hard to find
}


SPOKEN_II_SHOTS = {                    # his answer to 'not worth the time'
    66: dict(voice="him", rows_at=[([0, 1, 2, 3], beam_x(0, 400), 400, 1.2)]),               # I won't lose you
    67: dict(voice="him", rows_at=[([0], beam_x(1, 600), 600, 2.2, {"tracking": 0.16}),       # TOGETHER
                                   ([1, 2, 3, 4], beam_x(2, 730), 730, 1.0),                  # we'll find you the
                                   ([5], 150, 850, 2.0, {"tracking": 0.16, "left": True})]),  # TIME, on the
                                                                                               # left, clear of the light
}


def spoken(name, reopen=None, shots=None, after=None):
    """`after`: the section before, whose last words finish leaving; the bullet
    stays in the room once it has appeared."""
    def scene(t, T, lines):
        img = base_night(t, T, lift=0.7)
        opens = virtue_opens(T) if reopen is None else [reopen(T, i) for i in range(3)]
        img = look.beams(img, t, T, [None] * 3, src=BEAM_SRC, open_times=opens,
                         targets=BEAM_TARGETS, strength=0.85)
        if after:
            prev_name, prev_lines, prev_shots = after
            K0 = kin(T, prev_name, prev_lines, prev_shots, light=True)
            for n in prev_lines:
                tt = K0.plan[n].get("title_t")
                if tt is not None:
                    e0 = K0.plan[n]["life"][1]
                    img = title_ghost(img, 0.26 * (1 - smooth(ramp(t, e0 + 0.1, e0 + 0.9))))
            img = K0.draw(img, t)
        sh = shots or {n: dict(voice="him") for n in lines}
        img = kin(T, name, lines, sh, voice="him", hold=1.6).draw(img, t)
        p = post("night", exposure=0.95, ghosts=())   # still words: no ghost copies
        return img, p
    return scene


# ================================================================ Break II
# His three shafts stay after his words and breathe. Dust drifts down inside
# them; when she answers with a wordless held note (vocal stem: 204.35-208.35)
# it turns gold and lifts. One flicker of the future (the shafts being cut)
# just before the coats come back on 'But'.
HUM = (204.35, 208.35)


def hum_env(t):
    """Her wordless held note, as an envelope."""
    return smooth(ramp(t, HUM[0], HUM[0] + 1.2)) * (1 - smooth(ramp(t, HUM[1] - 0.3, HUM[1] + 0.3)))


def mote_travel(t, t0=190.0):
    """How far the dust has fallen by t: it falls at speed 1, slows on her
    note to a hover and then a gentle rise (speed 1 - 1.3 * hum), integrated
    so the change is a glide, never a jump."""
    if t <= t0:
        return t - t0
    ts = np.linspace(t0, t, max(2, int((t - t0) * 30)))
    v = np.array([1 - 1.3 * hum_env(x) for x in ts])
    return float(np.sum((v[1:] + v[:-1]) * 0.5 * np.diff(ts)))


def shaft_motes(img, t, strength, gold, seed=91, n=70):
    """Dust inside the three shafts: her, made visible only by his light."""
    if strength <= 0.01:
        return img
    r = np.random.default_rng(seed)
    span = H + 60 - BEAM_SRC[1]
    travel = mote_travel(t)

    def draw(c):
        for k in range(3):
            s0, lat = r.uniform(0, 1, n), r.normal(0, 0.4, n)
            sp, ph, rad = r.uniform(0.008, 0.022, n), r.uniform(0, 6.28, n), r.uniform(1.6, 4.2, n)
            for i in range(n):
                u = (s0[i] + sp[i] * travel) % 1.0
                y = BEAM_SRC[1] + u * span
                x = beam_x(k, y) + lat[i] * 62 * (0.3 + 0.7 * u) + 9 * math.sin(t * 0.4 + ph[i])
                a = (0.45 + 0.55 * math.sin(t * 0.9 + ph[i]) ** 2) * min(1.0, u * 4)
                c.drawCircle(x, y, rad[i], skia.Paint(AntiAlias=True, Color4f=skia.Color4f(1, 1, 1, a)))
    m = fx.blur(fx.skia_alpha(draw), 1.3)
    col = fx.hexc("#E9E3DC") * (1 - gold) + fx.hexc("#FFC266") * gold
    return img + col * (m * 1.25 * strength)[..., None]


def break_ii(t, T, lines):
    import visions as V
    s = sec_of(T, "Break II")
    hum = hum_env(t)
    breath = 1 + 0.12 * math.sin(2 * math.pi * (t - s["start"]) / 6.8)   # slow, not on the beat
    img = base_night(t, T, lift=0.7 + 0.15 * hum)
    img = look.beams(img, t, T, [None] * 3, src=BEAM_SRC, targets=BEAM_TARGETS,
                     strength=0.85 * breath * (1 + 0.25 * hum))
    gold = clamp01(0.15 + 0.5 * smooth(ramp(t, s["start"], HUM[0])) + 0.5 * hum)
    img = shaft_motes(img, t, smooth(ramp(t, s["start"] - 0.5, s["start"] + 2.0)) * (1 + 0.6 * hum),
                      gold)
    img = V.glimpse(img, t, "cutting", 207.63, strength=0.3, hold=0.1, decay=0.55)   # the future, once
    # his last words finish leaving
    img = kin(T, "Spoken I: I can help you find", range(44, 46), SPOKEN_SHOTS, voice="him",
              hold=1.6).draw(img, t)
    return img, post("night", exposure=0.95, ghosts=())


# ================================================================ A gun and a bullet
# His finding becomes a hunt. The coats' clock follows him into the night;
# their objects (GUN, BULLET) and their order are set in their cold type;
# the bullet appears as the film's first sharp thing. 'He found her...' is
# the title's word again, now in dread: 'found' lands in the title's space
# with 'her...' after it, and his light stops short.
GUN_SHOTS = {                                      # the hunt as their report, typed
    58: dict(voice="coats", style="track", hero=2, overstrike=True),   # with time x2
    59: dict(voice="coats", exit_at=260.8,
             set=[([0, 1, 2], 1.0, 0), ([3, 4, 5, 6], 1.0, 0), ([7], 1.9, 0),
                  ([8, 9], 1.0, 0), ([10], 1.9, 0)]),                                # GUN / BULLET
    60: dict(voice="coats", no_separate=True,
             rows_at=[([0, 1], 150, 215, 1.0, {"left": True}),                     # and said
                      ([2, 3], 180, 470, 1.3, {"left": True}),                     # GO AND FIND
                      ([4], 180, 600, 2.2, {"left": True, "key": "coats_bold"}),  # THAT GIRL
                      ([5, 6], 180, 680, 1.3, {"left": True})]),
    61: dict(voice="coats", style="stack", hero=2),                  # and he DID
    62: dict(voice="coats", set=[([0, 1], 1.6, 80, {"ghost": True})]),   # he did...
    63: dict(voice="coats", style="stack", hero=1),                  # he FOUND her
    64: dict(set=[([0], 1.0, 40)], title_word=1, title_tail=[2]),                # her voice again:
}                                                                                 # lost and found her...


def falling_bullet(t, T, end, line=59, x0=W / 2):
    """From 'bullet' on, the bullet tumbles slowly down the frame at `x0`, in
    from above the top edge and out below the bottom by `end`."""
    t0 = word_at(T, line, "bullet")
    if t0 is None or t < t0 or t > end:
        return None
    u = (t - t0) / (end - t0)
    y = -70 + (H + 140) * u
    x = x0 + 10 * math.sin((t - t0) * 0.7)
    rot = -8 + 62 * (t - t0)                      # a slow turn, end over end
    return (x, y, 1.9, rot)


def bullet_through(t, t0, t_pass, y_pass, x0):
    """The same tumbling fall, at a steady pace from above the frame at `t0`
    so that it is at height `y_pass` at `t_pass`, and on out below the frame."""
    if t < t0:
        return None
    v = (y_pass + 70) / (t_pass - t0)
    y = -70 + v * (t - t0)
    if y > H + 70:
        return None
    return (x0 + 10 * math.sin((t - t0) * 0.7), y, 1.9, -8 + 62 * (t - t0))


def gun_and_bullet(t, T, lines):
    name = "A gun and a bullet"
    s = sec_of(T, name)
    dx, dy, dr = mem.drift(t, seed=9)
    img = base_night(t, T, lift=1.25 - 0.4 * smooth(ramp(t, 257.0, 271.3)))   # darker as he closes in
    img = fx.shift(img, dx * 0.5, dy * 0.5, dr * 0.5, 1.02)
    img = draw_mems(img, t, mems(T, name, ["dark_street", "parking_rain", "streets_night",
                                           "fog_park"], every=2, life=10.0, seed=41,
                                 keep_left=900), offset=(dx, dy), warm=0.2)
    # 'he found her' twice: his light closes on her; the second time it stops short
    for k, st in enumerate(words_like(T, [63, 64], "found")):
        if t >= st:
            u = ease_out(clamp01((t - st) / 1.2))
            x = 1600 - 500 * u * (1 if k == 0 else 0.6)
            img = fx.hotspot(img, t, x, 560, 220, strength=0.45 * (1 - 0.4 * k) * (1 - clamp01((t - st - 2.5) / 2)))
    K = kin(T, name, lines, GUN_SHOTS, light=True)
    for n in lines:                                      # the same unreadable 'lost and'
        tt = K.plan[n].get("title_t")
        if tt is not None:
            e0 = K.plan[n]["life"][1]
            img = title_ghost(img, 0.26 * smooth(ramp(t, tt - 0.5, tt + 0.2)) *
                              (1 - smooth(ramp(t, e0 + 0.1, e0 + 0.9))))
    img = K.draw(img, t, cam=(dx * 0.35, dy * 0.35))
    p = post("night")
    fb = falling_bullet(t, T, T.lines[63]["start"] - 0.3)   # gone just before 'He found her'
    if fb:
        p["bullet"] = fb
    # out of the coats' white into the night
    u = smooth(ramp(t, s["start"], s["start"] + 1.8))
    if u < 1:
        ref, rp = coats("White Coats II", ["stairwell_cage", "cinderblock_hole", "hospital_corridor"],
                        67, hot=1.0)(t, T, range(47, 57))
        img = ref * (1 - u) + img * u
        rp = dict(rp)
        p = dict(_mix_post(rp, p, u), **({"bullet": p["bullet"]} if "bullet" in p else {}))
    return img, p


# ================================================================ The Run
# The one time they are free: warm and fast, but nothing pulses and nothing
# comes at the viewer. Speed is sideways: streaks, rushing photos, and the
# exposure echo trailing her words behind them. 'Lost' turns good, and the
# title is sung outright: 'lost and found' glows in plainly in the title's
# place; 'lost' and 'found' fade out around 'and', which stays, and glow back
# in each other's places: 'found and lost'. Here they run, so 'ran' leans:
# lost, freely.
RUN_SHOTS = {
    69: dict(rows_at=[([0, 1], 150, 300, 1.0, {"left": True}),                        # and they
                      ([2], 190, 520, 1.8, {"left": True, "key": "her_roman", "bare": True}),   # ran
                      ([3], 640, 330, 1.25, {"left": True, "key": "her_roman", "bare": True}),  #  ran
                      ([4], 430, 790, 2.2, {"left": True, "key": "her_roman", "bare": True}),   # ran
                      ([5], 700, 610, 1.5, {"left": True, "key": "her_roman", "bare": True})]), # ran: scattered
    70: dict(hero=[0, 2], stretch_both=True),                                       # FOREVER / EVER
    71: dict(hero=[2, 4], nopop=True),                                              # TUNNEL / MOTEL
    72: dict(hero=[0, 3], nopop=True),                                              # LOST / LOST
    73: dict(style="stack", hero=3),                                                # each OTHER
    74: dict(style="stack", hero=1, backing_at=(760, 300, 64)),                     # FOUND (and found...)
    75: dict(swap=dict(first=[0, 1, 2], second=[3, 4, 5],                              # lost and found ->
                       at=(150, 830), size=190, fade=0.45, lower=True)),                # found and lost
    76: dict(set=[([0], 2.4, 0, {"stretch": True}), ([1, 2], 1.6, 160),
                  ([3, 4], 1.6, 330)]),                                             # forever, stepping on
}


def the_run(t, T, lines):
    name = "The Run"
    dx, dy, dr = mem.drift(t, seed=13, amp=(60, 26))
    img = base_night(t, T, lift=0.9)                            # (time isn't counted here: it goes)
    img = img + hexc("#3A2210") * 0.3
    img = fx.shift(img, dx, dy, dr, 1.04)                       # no beat zoom
    photos = ["tunnel_lights", "light_trails", "no_vacancy", "highway_trails", "open_sign",
              "gas_station", "tunnel_dark", "parking_rain", "streets_night", "dusk_drive"]
    img = draw_mems(img, t, mems(T, name, photos, every=1, life=6.5, seed=43, keep_left=900),
                    offset=(dx * 1.5, dy), warm=0.35)
    img = streaks(img, 0.8 + 0.15 * math.sin(t * 0.9))         # long exposure, not on the beat
    img = fx.light_leak(img, t, "right", color="#FFB35A", strength=0.32)
    img = kin(T, name, lines, RUN_SHOTS, light=True).draw(img, t, cam=(dx * 0.4, dy * 0.4))
    return img, post("run")


# ================================================================ But...
# The one moment her prophecy comes true. The band never stops, so neither
# does the run: time comes back instead (the uncounted bars of the run
# scratch back in at once), the warm lights go out one by one on the sparse
# hits of 'forever always comes to an end', and in the dark the coats arrive
# as cold light swinging round a bend, swelling with the band into their
# white so White Coats III begins inside it.
BUT_I_SHOTS = {78: dict(set=[([0], 2.6, 430)], dy=-120)}               # BUT... in her light, on
BUT_SHOTS = {                                                            # the 'But' of 'But forever'
    80: dict(dy=265, exit_at=325.9, word_flags={0: {"hidden": True}},    # lower left: FOREVER...
             set=[([1], 2.3, 30, {"stretch": True}),                    # always comes to an END
                  ([2, 3, 4, 5], 1.0, 60),
                  ([6], 2.3, 60, {"voice": "coats", "key": "coats_bold"})],
             diffuse=dict(keep=[6], after=6, delay=0.35, dur=2.0)),     # all but END lets go
    81: dict(voice="coats", dy=-250, no_separate=True,                   # upper right, all theirs:
             set=[([0, 1], 1.0, 1010, {"kind": "coatdark"}),            # WHEN THE
                  ([2], 2.3, 1010, {"key": "coats_bold", "kind": "coatdark", "bare": True}),   # COATS
                  ([3, 4, 5], 1.0, 1040, {"kind": "coatdark"}),          # COME ROUND THE
                  ([6], 2.2, 1040, {"key": "coats_bold", "kind": "coatdark"})]),   # BEND, dark in
}                                                                        # their gray
RUN_PHOTOS = ["tunnel_lights", "light_trails", "no_vacancy", "highway_trails", "open_sign",
              "gas_station", "tunnel_dark", "parking_rain", "streets_night", "dusk_drive"]


def _lights_out_hits(T):
    """The sparse hits under 'forever always comes to an end' (no-vocals stem):
    one warm light goes out on each."""
    def find():
        import librosa
        from engine import ROOT
        y, sr = librosa.load(str(ROOT / "render" / "stems" / "htdemucs" / "lost_and" / "no_vocals.wav"),
                             sr=22050, offset=320.0, duration=4.4)
        env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=256)
        on = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=256, units="time", delta=0.25)
        strong = sorted(((float(env[int(x * sr / 256)]), float(x) + 320.0) for x in on), reverse=True)
        picked = []
        for _, x in strong:
            if all(abs(x - q) > 0.55 for q in picked):
                picked.append(x)
            if len(picked) == 5:
                break
        return sorted(picked)
    try:
        return cached(("but_hits",), find)
    except Exception:
        return [320.5, 321.5, 322.5, 323.0, 324.0]


def but(t, T, lines):
    name = "But..."
    s = sec_of(T, name)
    hits = _lights_out_hits(T)
    # warmth: 1 through 'But...', then one step down per hit (each light gutters out)
    warmth = 1.0
    for k, h in enumerate(hits):
        step = 1.0 / len(hits)
        warmth -= step * smooth(clamp01((t - h) / 0.35))
    when_t = T.lines[81]["start"]
    bend = word_at(T, 81, "bend")
    flood = smooth(ramp(t, bend, s["end"] - 0.1))                       # their white, with the band

    # the run, still moving (as The Run's scene), its warmth draining
    dx, dy, dr = mem.drift(t, seed=13, amp=(60, 26))
    dark = np.array(hexc("#14161C"), np.float32)
    img = base_night(t, T, tally=False, lift=0.9 * (0.4 + 0.6 * warmth))
    img = img + hexc("#3A2210") * 0.3 * warmth
    img = fx.shift(img, dx, dy, dr, 1.04)
    bg = img
    lit = draw_mems(img, t, mems(T, "The Run", RUN_PHOTOS, every=1, life=6.5, seed=43, keep_left=900),
                    offset=(dx * 1.5, dy), warm=0.35)
    img = bg + (lit - bg) * warmth                                       # the lights go out
    img = streaks(img, (0.8 + 0.15 * math.sin(t * 0.9)) * warmth)
    img = fx.light_leak(img, t, "right", color="#FFB35A", strength=0.32 * warmth)
    img = img * (0.55 + 0.45 * warmth) + dark * (1 - warmth) * 0.45    # cold dark left behind

    # the coats round the bend: their gray spreads in from the right, ahead of
    # their words, so the words arrive dark inside it
    if t >= when_t - 2.0:
        u = smooth(ramp(t, when_t - 2.0, bend + 0.4))
        yy, xx = fx._yy_xx()
        cx = W + 450 - 1000 * u
        cy = H * 0.42
        r = 520 + 620 * u
        spread = np.exp(-(((xx - cx) / r) ** 2 + ((yy - cy) / (r * 0.8)) ** 2)) * (0.45 + 0.45 * u)
        gray = np.array(hexc("#9AA0A8"), np.float32)
        img = img * (1 - spread[..., None]) + gray * spread[..., None]
    import visions as V
    if bend is not None:                                                 # the vision she had, fulfilled
        img = V.glimpse(img, t, "coats", bend, strength=0.32, hold=0.12, decay=0.6)

    # her words (light); END and COATS in their cold type
    # the held scoop before it is the run stopping; the word comes on the second 'But'
    late = word_at(T, 80, "but") - T.lines[78]["start"]
    img = kin(T, "But...#light", [78], BUT_I_SHOTS, light=True, hold=1.6).draw(img, t - late)
    img = kin(T, name, [n for n in lines if n != 78], BUT_SHOTS, light=True, hold=3.0).draw(img, t)

    # the flood of their white, so White Coats III begins inside it
    if flood > 0:
        white = np.array([0.86, 0.89, 0.93], np.float32)
        img = img * (1 - flood) + white * flood

    # time comes back: the run's uncounted bars scratch in, one per eighth note
    img = tally_layer(img, t, T)
    p = post("run")
    p = _mix_post(p, post("night"), 1 - warmth)
    p = _mix_post(p, post("institute", exposure=1.02), flood)
    p["ghosts"] = ()
    return img, p


# ================================================================ The Cutting
# What they did to each of them, in her words. His three shafts come back
# out of the coats' white with the virtues he offered still lit inside them,
# where he set them in Spoken I. Each 'cut' is surgery: a scalpel traces an
# excision round his word (the frame startles, as on the stabs), his light
# wells out of the incision, redder, and drips; when she sings the word the
# wound is sewn shut over it in thick red thread, a strikeout and sutures at
# once, and his light chokes under it. The incisions are gone before 'But left
# him', then the sewn words; he is left the GUN and the BULLET in the coats'
# type, side by side where his virtues were. The bullet has been falling since
# the cutting began, as it fell when they gave it to him, and passes between
# them as 'bullet' is sung. What they take
# from her is cut out of her lines, leaving the gaps: her ____ to hug, her
# ____ to run. Taking her eyes takes the
# focus; taking her ears stills the world (the drift, the dust, the grain).
# What they leave her is the years: her tally, the one sharp thing she still
# sees, and 'with time... with time...' is the coats' phrase in their type,
# now her condition. Her agains start to double, as in the refrain to come.
# 'Of him...' in his light on his side, 'and her...' on hers.
CUT_OPEN = (362.3, 362.7, 363.1)       # his shafts pour back in the coats' white
CUT_LEAD = (361.9, 363.6)              # White Coats III's white drains into his night
VIRTUE_AT = [(beam_x(0, 530), 530), (beam_x(1, 680), 680), (beam_x(2, 860), 860)]   # as Spoken I
VIRTUE_SIZE, VIRTUE_TRACK = 80, 0.16
DROP_X = 1160                          # the bullet falls here, between GUN and BULLET
GUN_AT, BULLET_AT = (DROP_X - 280, VIRTUE_AT[1][1]), (DROP_X + 380, VIRTUE_AT[1][1])
AGAIN_GHOSTS = ((1.0, -180, 0, -5, 0.08), (2.0, 170, -10, 4, 0.05), (3.2, -60, 12, 2, 0.03))


def _after(txt, x=150, size=70, key="her"):
    """x just past `txt` in her type (for a word in another voice on the same row)."""
    return x + font(key, size).measureText(txt) + 0.6 * size


def _they_cut(y):
    return dict(word_flags={4: {"hidden": True}},          # the virtue: his word, in his shaft
                rows_at=[([0, 1, 2, 3], 150, y, 1.0, {"left": True})])


_COATS_WORD = {"left": True, "voice": "coats", "key": "coats_bold", "bare": True}
_TAKEN = {"taken": 0.12}
CUT_SHOTS = {
    94: _they_cut(650), 95: _they_cut(740), 96: _they_cut(650),   # each clear of the last leaving
    97: dict(rows_at=[([0, 1, 2, 3], 150, 830, 1.0, {"left": True}),                 # but left him the
                      ([4], GUN_AT[0], GUN_AT[1], 1.9, dict(_COATS_WORD, left=False)),   # GUN
                      ([5, 6], 150, 960, 1.0, {"left": True}),                       # and the
                      ([7], BULLET_AT[0], BULLET_AT[1], 1.9, dict(_COATS_WORD, left=False))]),
    # (GUN and BULLET side by side where his virtues were, the bullet falling between)
    # her body, cut out of her lines; the lines stay, with their gaps
    99: dict(exit_at=384.95, word_flags={3: _TAKEN},
             rows_at=[([0, 1, 2, 3, 4, 5], 150, 680, 0.9, {"left": True})]),        # they took her arms
    100: dict(exit_at=384.95, word_flags={3: _TAKEN},
              rows_at=[([0, 1, 2, 3, 4, 5], 150, 780, 0.9, {"left": True})]),       # ...legs
    101: dict(exit_at=384.95, word_flags={1: _TAKEN, 5: _TAKEN},
              rows_at=[([0, 1, 2, 3], 150, 880, 0.9, {"left": True}),               # her eyes to see
                       ([4, 5, 6, 7], 150, 980, 0.9, {"left": True})]),             # and ears to hear
    # what they leave her, beside her count
    102: dict(backing_at=(1090, 520, 58, 0.5),
              rows_at=[([0, 1, 2, 3], 560, 330, 1.0, {"left": True}),                # but left her the
                       ([4], 560, 520, 2.6, {"left": True, "key": "her_roman"})]),   # YEARS
    103: dict(overstrike=[(1, 3), (2, 4)],
              rows_at=[([0], 150, 800, 2.2, {"left": True, "key": "her_roman"}),     # ALONE
                       ([1, 2], 150, 930, 1.0, {"left": True, "voice": "coats"}),    # with time... in
                       ([3, 4], 150, 930, 1.0, {"left": True, "voice": "coats"})]),  # their type, overstruck
    104: dict(rows_at=[([0, 1, 2], 560, 330, 1.0, {"left": True}),                   # and all her
                       ([3, 4, 5, 6, 7], 560, 450, 1.0, {"left": True})]),           # agains and agains...
    105: dict(rows_at=[([0, 1], 1500, 840, 1.6, {"voice": "him", "tracking": VIRTUE_TRACK}),
                       ([2, 3], 150, 840, 1.0, {"left": True})]),                    # OF HIM... / and her...
}
for _n, _o in CUT_SHOTS.items():                         # every line here is placed by hand
    CUT_SHOTS[_n] = dict(_o, style="stack", no_separate=True, no_ghost=True)


def _shaft(k):
    """Beam k's soft wedge of light (static), brighter toward the floor."""
    def make():
        tx = BEAM_TARGETS[k]

        def draw(c):
            p = skia.Path()
            p.moveTo(BEAM_SRC[0] - 18, BEAM_SRC[1]); p.lineTo(BEAM_SRC[0] + 18, BEAM_SRC[1])
            p.lineTo(tx + 70, H + 60); p.lineTo(tx - 70, H + 60); p.close()
            c.drawPath(p, skia.Paint(AntiAlias=True, Color4f=skia.Color4f(1, 1, 1, 1)))
        return fx.blur(fx.skia_alpha(draw), 9) * (0.55 + 0.45 * fx.vgrad(BEAM_SRC[1], H))
    return cached(("shaft", k), make)


def _along(y):
    return (y - BEAM_SRC[1]) / (H - BEAM_SRC[1])


BLEED = np.array([1.0, 0.34, 0.14], np.float32)      # his light, opened: deeper, redder
SUTURE = hexc("#7A0E0A")                               # the thread, red as the wound
SUTURE_GLOW = np.array([0.85, 0.08, 0.05], np.float32)


def _virtue_mask(k, txt):
    def make():
        f = font("him", VIRTUE_SIZE)
        xs = kinetic._char_x(f, txt, VIRTUE_TRACK, VIRTUE_SIZE)
        wd = f.measureText(txt) + VIRTUE_TRACK * VIRTUE_SIZE * len(txt)
        x0, y = VIRTUE_AT[k][0] - wd / 2, VIRTUE_AT[k][1]

        def draw(c):
            for ch, x in zip(txt, xs):
                c.drawString(ch, x0 + x, y, f, skia.Paint(AntiAlias=True, Color4f=skia.Color4f(1, 1, 1, 1)))
        m = fx.skia_alpha(draw)
        return m, cv2.GaussianBlur(m, (0, 0), 10), cv2.GaussianBlur(m, (0, 0), 1.6), wd
    return cached(("virtue", k), make)


def _lens(cx, cy, a, b, seed):
    """A surgeon's incision round a word: a tight lens, pointed at both ends,
    traced by hand (slightly unsteady). Starts at the left point, runs along
    the top, back along the bottom. Returns its path, box and the few places
    a bead of blood gathers on the lower lip."""
    rng = np.random.default_rng(seed)
    ph = rng.uniform(0, 6.28, 4)

    def wob(s_):
        return 1.2 * math.sin(9 * s_ + ph[0]) + 0.8 * math.sin(23 * s_ + ph[1])
    ss = np.linspace(-1, 1, 90)
    top = [(cx + a * q, cy - b * (1 - q * q) ** 0.85 - wob(q)) for q in ss]
    bot = [(cx + a * q, cy + b * (1 - q * q) ** 0.85 + wob(q + 3)) for q in ss[::-1]]
    path = skia.Path()
    path.moveTo(*top[0])
    for pt in top[1:] + bot[1:]:
        path.lineTo(*pt)
    path.close()
    beads = []
    for _ in range(5):                                   # a few short beads, which stop
        q = float(rng.uniform(-0.6, 0.6))
        beads.append(dict(x=cx + a * q, y=cy + b * (1 - q * q) ** 0.85 + wob(q + 3),
                          delay=float(rng.uniform(0.15, 0.6)), len=float(rng.uniform(5, 24)),
                          tau=float(rng.uniform(0.4, 0.9)), w=float(rng.uniform(2.0, 3.4))))
    return dict(path=path, rect=skia.Rect(cx - a, cy - b, cx + a, cy + b), c=(cx, cy), drips=beads)


def _incision(k, wd):
    """The incision round his virtue k (`wd` wide), hugging the word."""
    def make():
        cx, y = VIRTUE_AT[k]
        return _lens(cx, y - 0.37 * VIRTUE_SIZE, wd / 2 + 44, 0.52 * VIRTUE_SIZE, 300 + k)
    return cached(("incision", k), make)


def _stroke_mask(path, width):
    return fx.skia_alpha(lambda c: c.drawPath(path, skia.Paint(
        AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=width,
        StrokeCap=skia.Paint.kRound_Cap, StrokeJoin=skia.Paint.kRound_Join,
        Color4f=skia.Color4f(1, 1, 1, 1))))


def cut_shafts(img, t, cuts, sung, strength=0.95):
    """His shafts. Once cut, the light flickers; when his word is sewn shut,
    the light below it drains away and the light above withdraws into its
    source."""
    g = np.clip(0.7 + 0.4 * fx.fog(t, seed=31, period=5.0), 0.1, 1.2)
    along = fx.vgrad(BEAM_SRC[1], H)
    cols = [C["amber"], C["rose"], C["amber_hot"]]
    for k in range(3):
        pour = ease_out(clamp01((t - CUT_OPEN[k]) / 0.9), 2)
        if pour <= 0.001:
            continue
        m = _shaft(k)
        if pour < 1:                                       # pours down from the source
            m = m * np.clip((pour * 1.15 - along) / 0.1, 0, 1)
        if cuts[k] is not None and t >= cuts[k]:
            m = m * (0.8 + 0.2 * math.sin(t * 29 + 2 * k) ** 2)
            tau = t - sung[k]
            if tau >= 0:
                if tau > 1.3:
                    continue
                inc = _incision(k, _virtue_mask(k, VIRTUE_WORDS[k].upper())[3])
                top, bot = _along(inc["rect"].top()), _along(inc["rect"].bottom())
                rec = top * smooth(clamp01(tau / 1.3))
                fall = 0.9 * (tau / 0.7) ** 2
                upper = np.clip((top - rec - along) / 0.04, 0, 1)
                lower = np.clip((along - bot - fall) / 0.04, 0, 1) * (1 - smooth(clamp01(tau / 0.7)))
                m = m * (upper + lower)
        img = img + cols[k] * (m * 0.9 * g * strength)[..., None]
    return img


def _thread(cx, cy, wd, seed, scale=1.0):
    """The strike through a word and the stitches across it: a slightly
    unsteady line at mid-height, and short vertical stitches at uneven
    spacing, heights and tilts. Each stitch: (x along the strike, line, width)."""
    rng = np.random.default_rng(seed)
    x0, x1 = cx - wd / 2 - 34 * scale, cx + wd / 2 + 34 * scale
    xs = np.linspace(x0, x1, 60)
    strike = [(float(x), cy + 1.8 * scale * math.sin(0.021 * x + seed) + float(rng.normal(0, 0.5)))
              for x in xs]
    stitches, x = [], x0 + float(rng.uniform(8, 20)) * scale
    while x < x1 - 8 * scale:
        h = float(rng.uniform(34, 60)) * scale
        lean = math.radians(float(rng.uniform(-9, 9)))
        yc = cy + float(rng.normal(0, 3.5)) * scale
        dx, dy = math.sin(lean) * h / 2, math.cos(lean) * h / 2
        stitches.append((x, (x - dx, yc - dy, x + dx, yc + dy), float(rng.uniform(5.0, 7.5)) * scale ** 0.5))
        x += float(rng.uniform(26, 50)) * scale
    return dict(strike=strike, stitches=stitches, x0=x0, x1=x1, scale=scale)


def _sutures(k, wd):
    def make():
        cx, y = VIRTUE_AT[k]
        return _thread(cx, y - 0.37 * VIRTUE_SIZE, wd, 500 + k)
    return cached(("sutures", k), make)


def sew(img, sut, front, alpha, ground="dark"):
    """Draw the red thread up to x = `front` (each stitch pulled through as the
    thread passes). It has mass: a shadow under it, a dull glow, a wet edge.
    `ground`='light' on the coats' white, where the glow would vanish."""
    if alpha <= 0.002:
        return img
    sc = sut["scale"]

    def draw(c):
        pts = [p_ for p_ in sut["strike"] if p_[0] <= front]
        if len(pts) >= 2:
            pth = skia.Path()
            pth.moveTo(*pts[0])
            for p_ in pts[1:]:
                pth.lineTo(*p_)
            c.drawPath(pth, skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style,
                                       StrokeWidth=6.5 * sc ** 0.5, StrokeCap=skia.Paint.kRound_Cap,
                                       Color4f=skia.Color4f(1, 1, 1, 1)))
        for x, (xa, ya, xb, yb), sw in sut["stitches"]:
            if x <= front:
                u = clamp01((front - x) / 26)                # each stitch pulled through
                c.drawLine(xa, ya, xa + (xb - xa) * u, ya + (yb - ya) * u, skia.Paint(
                    AntiAlias=True, StrokeWidth=sw, StrokeCap=skia.Paint.kRound_Cap,
                    Color4f=skia.Color4f(1, 1, 1, 1)))
    sm = fx.skia_alpha(draw)
    shade = 0.6 if ground == "dark" else 0.3
    img = fx.over(img, np.float32(0.02), np.roll(cv2.GaussianBlur(sm, (0, 0), 3), 4, 0) * shade * alpha)
    if ground == "dark":
        img = fx.add(img, SUTURE_GLOW, cv2.GaussianBlur(sm, (0, 0), 8) * 0.4 * alpha)
    img = fx.over(img, SUTURE, np.clip(cv2.GaussianBlur(sm, (0, 0), 0.8) * 1.1, 0, 1) * alpha)
    img = fx.add(img, SUTURE_GLOW, cv2.GaussianBlur(np.roll(sm, -1, 0) * 0.3, (0, 0), 1.0) * alpha)
    return img


def trace(img, inc, frac, alpha, ground="dark", tip=True):
    """The blade's line round the word, `frac` of the way round."""
    if frac <= 0 or alpha <= 0.002:
        return img
    L = skia.PathMeasure(inc["path"], False).getLength()
    seg = skia.Path()
    skia.PathMeasure(inc["path"], False).getSegment(0, L * min(frac, 1.0), seg, True)
    cut = _stroke_mask(seg, 2.0)
    if ground == "dark":
        img = fx.add(img, kinetic.COLD, cut * alpha)
    else:                                               # a fine red line on their white
        img = fx.over(img, SUTURE, np.clip(cv2.GaussianBlur(cut, (0, 0), 0.6) * 0.85, 0, 1) * alpha)
    if tip and frac < 1:                                # the blade's point
        pos = skia.PathMeasure(inc["path"], False).getPosTan(L * frac)[0]
        dot = fx.skia_alpha(lambda c: c.drawCircle(pos.x(), pos.y(), 4.0, skia.Paint(
            AntiAlias=True, Color4f=skia.Color4f(1, 1, 1, 1))))
        if ground == "dark":
            img = fx.add(img, kinetic.COLD, cv2.GaussianBlur(dot, (0, 0), 3) * 2.0)
        else:
            img = fx.over(img, kinetic.COAT_DARK, np.clip(cv2.GaussianBlur(dot, (0, 0), 1.5) * 1.4, 0, 1))
    return img


def virtue_words(img, t, T, sung_words, cuts, scar_end, oval_end):
    """His virtues, lit in their shafts, cut and sewn shut. On 'cut' a scalpel
    traces a tight incision round the word and a thin line of his light wells
    along it, redder, with a few beads on the lower lip. The word trembles
    while it is worked on. When she sings
    it, the wound is sewn shut over it in thick red thread: a strike runs
    through the word and the stitches cross it as the thread passes (a
    strikeout, and sutures); his light chokes down under them and the welling
    stops. The incisions are gone by `oval_end`; the sewn words and their
    thread go from `scar_end`, before the coats' GUN and BULLET."""
    hv = 1 - smooth(ramp(t, scar_end, scar_end + 0.9))
    ov = 1 - smooth(ramp(t, oval_end - 0.6, oval_end))
    for k, w in enumerate(sung_words):
        lit = 0.9 * smooth(ramp(t, CUT_OPEN[k] + 0.35, CUT_OPEN[k] + 1.2))
        if lit <= 0.001:
            continue
        ct, ts = cuts[k], w["start"]
        m, bloom, soft, wd = _virtue_mask(k, display_text(w["text"]).upper())
        if ct is None or t < ct:
            img = fx.add(img, kinetic.AMBER, bloom * 0.72 * lit)
            img = fx.add(img, kinetic.AMBER_HOT, m * 1.5 * lit)
            continue
        inc, sut = _incision(k, wd), _sutures(k, wd)
        age, tau = t - ct, t - ts
        sew_dur = 0.7
        sewn = smooth(ramp(tau, 0.0, sew_dur + 0.3))             # how closed the wound is
        frac = ease_out(clamp01(age / 0.28), 2)                  # the blade goes round in ~8 frames
        well = smooth(ramp(age, 0.05, 0.6)) * (0.82 + 0.18 * math.sin(t * 17 + k) * math.sin(t * 7.3))
        well *= (1 - 0.8 * sewn) * ov
        # the incision: a fine cold line, and only a thin welling along it
        img = trace(img, inc, frac, (1.3 * math.exp(-age / 0.3) + 0.1 * sewn) * ov)
        if well > 0.004:
            L = skia.PathMeasure(inc["path"], False).getLength()
            seg = skia.Path()
            skia.PathMeasure(inc["path"], False).getSegment(0, L * frac, seg, True)
            cut = _stroke_mask(seg, 2.0)
            img = fx.add(img, BLEED, cv2.GaussianBlur(cut, (0, 0), 2.5) * 0.9 * well)
        # a few beads gather on the lower lip; they stop once it is closed
        run_t = min(age, ts - ct + sew_dur)
        da = ov * (1 - 0.45 * sewn)

        def beads(c):
            for d in inc["drips"]:
                u = run_t - d["delay"]
                if u <= 0:
                    continue
                ln = d["len"] * (1 - math.exp(-u / d["tau"]))
                c.drawLine(d["x"], d["y"] - 1, d["x"], d["y"] + ln, skia.Paint(
                    AntiAlias=True, StrokeWidth=d["w"], StrokeCap=skia.Paint.kRound_Cap,
                    Color4f=skia.Color4f(1, 1, 1, 0.9)))
                c.drawCircle(d["x"], d["y"] + ln, d["w"] * 0.9, skia.Paint(
                    AntiAlias=True, Color4f=skia.Color4f(1, 1, 1, 1)))
        if da > 0.004:
            dm = fx.skia_alpha(beads)
            img = fx.add(img, BLEED, cv2.GaussianBlur(dm, (0, 0), 1.0) * 0.85 * da)
        # the word: trembling while it is worked on, then choked down under the stitches
        a = lit * (0.85 + 0.15 * math.sin(t * 31 + 3 * k) ** 2) * (1 - 0.72 * sewn) * (hv if tau >= 0 else 1)
        wm, wb = m, bloom
        if tau < sew_dur:
            jx = int(round((kinetic._hash01(k, int(t * 30), 1) - 0.5) * 3.5))
            jy = int(round((kinetic._hash01(k, int(t * 30), 2) - 0.5) * 3.0))
            wm, wb = np.roll(m, (jy, jx), (0, 1)), np.roll(bloom, (jy, jx), (0, 1))
        img = fx.add(img, kinetic.AMBER, wb * 0.72 * a)
        img = fx.add(img, kinetic.AMBER_HOT, wm * 1.5 * a)
        # sewn shut: the strike runs through it, the stitches cross it behind the thread
        if tau >= 0:
            front = sut["x0"] + (sut["x1"] - sut["x0"]) * ease_out(clamp01(tau / sew_dur), 1.5)
            img = sew(img, sut, front, hv)
    return img


def cutting(t, T, lines):
    name = "The Cutting"
    s = sec_of(T, name)
    cuts = [word_at(T, n, "cut") for n in (94, 95, 96)]
    virtues = [T.lines[n]["words"][4] for n in (94, 95, 96)]
    sung = [w["start"] for w in virtues]
    eyes_w = T.lines[101]["words"][1]
    ears_w = T.lines[101]["words"][5]
    eyes = eyes_w["end"] + 0.12                          # as each is taken out of her line
    ears = ears_w["end"] + 0.12
    still = min(t, ears)                                 # deaf: the world stops moving

    img = base_night(still, T, tally=False, lift=0.7)
    img = cut_shafts(img, t, cuts, sung)
    # once the bullet is gone, what she sees of them drifts in the dark: these are
    # what her eyes and ears are taken from
    seen = cached(("mem", name), lambda: mem.schedule(
        T, ["doorway_figure", "car_window_night", "fog_lamps", "rain_window", "curtain_window"],
        word_at(T, 97, "bullet") + 1.5, s["end"], every=1, life=9.0, seed=47, keep_left=900))
    seen = [m for m in seen if m.t0 >= word_at(T, 97, "bullet") + 1.5]   # after the bullet
    img = img + (draw_mems(img, t, seen, warm=0.15, freeze=ears) - img) * 0.55   # faint
    if t >= eyes:                                        # her eyes: the focus goes, for good
        k = smooth(clamp01((t - eyes) / 1.5))
        img = img * (1 - 0.7 * k) + fx.blur(img, 9) * 0.7 * k
    # what they leave her: the years, still sharp; brighter on each 'years'
    img = tally_layer(img, t, T)
    left = T.lines[97]["start"]                          # 'But left him...': the incisions are
    img = virtue_words(img, t, T, virtues, cuts, left, left - 0.1)   # gone, then what they sewed
    img = kin(T, name, lines, CUT_SHOTS, light=True, hold=1.2).draw(img, t)
    # each 'cut' startles, like the stabs, and so does each pull: a jolt, the
    # exposure knocked down for an instant
    jolt = 0.0
    for k, (c, amp) in enumerate([(c, 1.0) for c in cuts] + [(x, 0.7) for x in sung]):
        if c is not None and 0 <= t - c < 0.35:
            jolt = amp * math.exp(-(t - c) / 0.06)
            img = fx.shift(img, 9 * jolt * math.cos(1.1 + 2.1 * k), 7 * jolt * math.sin(1.1 + 2.1 * k))
    p = post("night", exposure=0.95 - 0.1 * jolt)
    # the bullet falls again, as it did when they gave it him: from while his
    # virtues are being cut out, slowly, so that it passes between GUN and
    # BULLET as 'bullet' is sung
    fb = bullet_through(t, word_at(T, 94, "protecting"), word_at(T, 97, "bullet") + 0.2,
                        GUN_AT[1] - 0.35 * 1.9 * 56, DROP_X)
    if fb:
        p["bullet"] = fb
    agains = word_at(T, 104, "agains")                   # still words don't double; her agains do
    g = smooth(ramp(t, agains, agains + 1.5)) if agains else 0.0
    p["ghosts"] = tuple((dt, dx, dy, r, a * g) for dt, dx, dy, r, a in AGAIN_GHOSTS)
    p["grain"] = POST["night"]["grain"] * (1 - 0.6 * smooth(ramp(t, ears, ears + 0.6)))
    return img, p


def lead_into(a, b, b_lines, span):
    """Scene `a`, crossfading over `span` into scene `b` (which is already
    running) before b's section begins."""
    def scene(t, T, lines):
        if t < span[0]:
            return a(t, T, lines)
        img_b, pb = b(t, T, b_lines)
        u = smooth(ramp(t, *span))
        if u >= 1:
            return img_b, pb
        img_a, pa = a(t, T, lines)
        p = _mix_post(pa, pb, u)
        if "bullet" in pb:
            p["bullet"] = pb["bullet"]
        return img_a * (1 - u) + img_b * u, p
    return scene


# ================================================================ Refrain II: the same words, after
# Refrain I was prophecy; this is the same refrain sung after it has all
# happened. Same room, same layout per line, so it is recognised; but she is
# deaf now, so nothing answers the beat (no smear, no rewind, nothing flung)
# and the room stays still. Each line is written over a faint exposure of how
# it looked the first time (FIND lands on the old FOUND: 'everything he found'
# is now 'everything he could find'). Nothing is fully unwritten any more:
# every line leaves its trace, so her agains pile up. The visions that flashed
# past in Refrain I as the future come back as settled exposures, because they
# have happened. 'Never again...' is held and the frame bleaches to white: the
# story starts over in the Verse A reprise.
_ODD = lambda n: -110 if n % 2 else 110                   # sit where the first time sat
REFRAIN_II_SHOTS = {
    107: dict(style="stack", hero=1, foreknow=1.0, exit="dissolve", dy=_ODD(107),   # stay lost
              backing_at=(560, 330, 64), backing_alpha=0.8),
    108: dict(style="stack", hero=5, foreknow=1.0, dy=_ODD(108)),                     # ...could FIND
    109: dict(style="stack", hero=2, foreknow=1.0, dy=_ODD(109),                      # NEVER, no rewind
              backing_at=(620, 300, 92), backing_alpha=0.85),
    110: dict(hero=[0, 1], foreknow=1.0, dy=_ODD(110)),                               # forever / again
    111: dict(hero=[0, 1], ghost_second=True, foreknow=1.0, dy=_ODD(111)),            # never / again...
}
REFRAIN_PAIRS = {107: (20, 88.9), 108: (21, 92.3), 109: (22, 95.9), 110: (23, 104.4), 111: (24, 106.2)}
SETTLED = {107: 400.9, 108: 405.4, 109: 410.2, 110: 415.4}          # each line as it stood


def _line_mask(name, n, shots, at, T):
    """One line's ink as it stood at time `at`, as a mask (cached)."""
    def make():
        K = kinetic.Kinetic(T, [n], overrides=shots, hold=3.0)
        blank = np.ones((H, W, 3), np.float32)
        return np.clip(1 - K.draw(blank, at).mean(-1), 0, 1)
    return cached(("linemask", name, n, at), make)


def refrain_ii(t, T, lines):
    name = "Refrain II: Stay lost now girl"
    s = sec_of(T, name)
    still = s["start"]                                         # deaf: the room doesn't move
    img = base_her(still, T, tally=t)                          # the years, still counted
    # the photographs: she throws away everything he could find, slowly, with no
    # beat to throw them on; they fade over 'Throw away...'
    photos = ["curtain_window", "rain_window", "doorway_figure", "car_window_night"]
    ms = mems(T, name, photos, every=2, life=12.0, seed=19, keep_left=900)
    let_go = smooth(ramp(t, T.lines[108]["start"], T.lines[108]["end"] + 1.0))
    bg = img
    img = draw_mems(img, still + 4.0, ms)
    img = bg + (img - bg) * (1 - 0.85 * let_go)
    # what she foresaw has happened, and comes back exactly as she foresaw it:
    # the same flashes as Refrain I, an inescapable memory (foresight and
    # memory are the same thing for her)
    agains = [w["start"] for w in T.lines[109]["words"] if w["backing"]]
    img = flashes_of(img, t, T, (107, 108, 110), agains[:2], T.lines[111]["start"])
    # her agains: every line written so far leaves its trace, and each is
    # written over how it looked the first time
    for n2, (n1, at1) in REFRAIN_PAIRS.items():
        L2 = T.lines[n2]
        on = smooth(ramp(t, L2["start"] - 1.0, L2["start"] + 0.4))
        if on <= 0.004:
            continue
        m1 = cv2.GaussianBlur(_line_mask("Refrain I", n1, REFRAIN_I_SHOTS, at1, T), (0, 0), 2.0)
        first = 0.2 * on * (1 - 0.5 * smooth(ramp(t, L2["end"] + 0.5, L2["end"] + 2.5)))
        img = fx.over(img, kinetic.INK, m1 * first)
        if n2 in SETTLED and t > SETTLED[n2]:
            m2 = cv2.GaussianBlur(_line_mask(name, n2, REFRAIN_II_SHOTS, SETTLED[n2], T), (0, 0), 1.6)
            k = list(SETTLED).index(n2)
            m2 = np.roll(m2, (6 * (k + 1), 9 * (k + 1)), (0, 1))
            img = fx.over(img, kinetic.INK, m2 * 0.16 * smooth(ramp(t, SETTLED[n2], SETTLED[n2] + 1.2)))
    img = kin(T, name, lines, REFRAIN_II_SHOTS, hold=3.2).draw(img, t, kick=0.0, react=0.0)
    # out of the Cutting's night into her room
    u = smooth(ramp(t, s["start"] - 0.1, s["start"] + 1.4))
    g = post("her", exposure=0.92, sat=0.88)
    g["ghosts"] = ((1.0, -180, 0, -5, 0.2), (2.0, 170, -10, 4, 0.14), (3.2, -60, 12, 2, 0.1))
    if u < 1:
        ref, rp = cutting(t, T, range(94, 106))
        img = ref * (1 - u) + img * u
        g = _mix_post(rp, g, u)
    # 'Never again...' held; then everything, the agains included, goes to white
    note = T.lines[111]["end"]
    white = smooth(ramp(t, note - 1.1, note))                  # as the note ends, and held
    img = img * (1 - white) + np.float32(1.0) * white
    g["exposure"] = g["exposure"] + 0.25 * white
    return img, g


# ================================================================ Verse A reprise: again
# The story starts over (her eternal recurrence). It rhymes Verse A shot for
# shot: the same room, the same layout per line, the same memories surfacing;
# and, as in Refrain II, each line is written over a faint exposure of how it
# looked the first time. Years have accumulated: overexposed, many more of her
# ghosts, the tally across the wall. His warmth still creeps in on 'a voice
# came to her' and 'lost inside him', fainter: memory or return, left open.
# Nothing answers the beat. It comes up out of Refrain II's white.
def reprise(t, T, lines):
    import compose
    s = sec_of(T, "Verse A reprise")
    shots = dict(compose.VERSE_A_SHOTS)

    def first_time(img, tt):
        for n in range(1, 10):
            n2 = n + 112
            if n2 not in T.lines:
                continue
            L2 = T.lines[n2]
            on = smooth(ramp(tt, L2["start"] - 0.8, L2["start"] + 0.4)) * \
                (1 - smooth(ramp(tt, L2["end"] + 0.6, L2["end"] + 2.2)))
            if on <= 0.004:
                continue
            m = _line_mask("Verse A", n, shots, T.lines[n]["end"] - 0.1, T)
            img = fx.over(img, kinetic.INK, cv2.GaussianBlur(m, (0, 0), 2.0) * 0.2 * on)
        return img
    img, p = compose.verse_a_like(t, T, lines, name="Verse A reprise", offset=112, seed=5,
                                  years=True, under=first_time)
    white = 1 - smooth(ramp(t, s["start"], s["start"] + 1.3))   # out of Refrain II's white
    if white > 0:
        img = img * (1 - white) + np.float32(1.0) * white
        p = dict(p, exposure=p["exposure"] + 0.25 * white)
    return img, p


# ================================================================ Outro: the loop closes
# Her last sound is a wordless held note (vocal stem: 458.8-474.5). In Break II
# her hum was the gold dust lit inside his shafts; his light is gone, so the
# dust drifts through her room on its own and warms only while she sings. The
# fog thins toward white and the Intro's first memories go. The title, all film
# long only an unreadable echo where 'found...' landed, surfaces there plainly
# but faint, with the empty space after it. Near the end his shafts return as
# their absence: in this pale room light can't glow, so three faint cool
# shadows at their exact angles, where his light used to fall, held to the
# last frame. The last tally mark lands on the last downbeat.
OUTRO_NOTE = (458.8, 474.5)


def room_motes(img, t, strength, warmth, warm_mask=None, seed=93, n=90):
    """Dust drifting down through her room (steady, no rise), dark specks on the
    pale haze; while she sings they warm, only inside `warm_mask`."""
    if strength <= 0.01:
        return img
    r = np.random.default_rng(seed)
    x0, y0 = r.uniform(0, W, n), r.uniform(0, H, n)
    sp, ph, rad = r.uniform(9, 24, n), r.uniform(0, 6.28, n), r.uniform(3.0, 7.5, n)

    def draw(c):
        for i in range(n):
            y = (y0[i] + sp[i] * (t - 450.0)) % (H + 40) - 20
            x = x0[i] + 14 * math.sin(t * 0.35 + ph[i])
            a = 0.45 + 0.55 * math.sin(t * 0.8 + ph[i]) ** 2
            c.drawCircle(x, y, rad[i], skia.Paint(AntiAlias=True, Color4f=skia.Color4f(1, 1, 1, a)))
    m = fx.blur(fx.skia_alpha(draw), 1.8)
    w = warmth * (warm_mask if warm_mask is not None else 1.0)
    grey, amber = hexc("#6B5844"), hexc("#C8862E")
    w3 = np.asarray(w, np.float32)[..., None] if np.ndim(w) else np.float32(w)
    col = grey * (1 - 0.5 * w3) + amber * 0.5 * w3
    a = np.clip(m * 0.8 * strength, 0, 1)[..., None]
    img = img * (1 - a) + col * a
    return fx.add(img, hexc("#FFC266"), fx.blur(m, 6) * 0.65 * strength * w)


def outro(t, T, lines):
    s = sec_of(T, "Outro")
    thin = smooth(ramp(t, s["start"] + 2.0, s["end"] - 1.0))          # the fog thins toward white
    dx, dy, dr = mem.drift(t, seed=3)
    img = base_her(t, T, tally=False)
    img = fx.shift(img, dx * 0.4, dy * 0.4, dr * 0.4, 1.02)
    bg = img
    img = draw_mems(img, t, mems(T, "Outro", ["curtain_bedroom", "lake_overcast"], every=2, life=12.0,
                                 seed=53, keep_left=900), offset=(dx, dy))
    img = bg + (img - bg) * (1 - smooth(ramp(t, 472.0, 486.0)))     # the first memories go
    # his shafts, as the shadows of where his light fell
    bands = np.clip(_shaft(0) + _shaft(1) + _shaft(2), 0, 1)
    white = 0.86 * thin
    img = img * (1 - white) + np.float32(1.0) * white
    shade = smooth(ramp(t, 486.0, 490.5))                       # near the end, and held, over the white
    if shade > 0.004:
        img = img * (1 - 0.2 * shade * bands)[..., None] + hexc("#8C96A6") * (0.08 * shade * bands)[..., None]
    # her note: the dust warms only where his light used to be
    note = smooth(ramp(t, OUTRO_NOTE[0] - 0.3, OUTRO_NOTE[0] + 0.9)) * \
        (1 - smooth(ramp(t, OUTRO_NOTE[1] - 0.6, OUTRO_NOTE[1] + 0.8)))
    dust = (0.2 + 0.8 * note) * (1 - 0.8 * smooth(ramp(t, OUTRO_NOTE[1], 486.0)))
    img = room_motes(img, t, dust, note, warm_mask=bands)
    # 'lost and', where 'found' landed, a little faint, and nothing after it
    tx, ty, ts = kinetic.TITLE_AT
    img = title(img, t, 476.4, s["end"] + 20.0, x=tx, y=ty, size=ts, still=True, alpha=0.7)
    # the count, kept to the end; its last mark lands on the last downbeat
    img = tally_layer(img, t, T)
    p = post("bleach", exposure=0.95 + 0.05 * thin)
    # out of the reprise
    u = smooth(ramp(t, s["start"], s["start"] + 1.6))
    if u < 1:
        ref, rp = reprise(t, T, range(113, 122))
        img = ref * (1 - u) + img * u
        p = _mix_post(rp, p, u)
    return img, p


# ================================================================ Refrain I: prophecy
# She is precognitive. The refrain comes before the story happens: she has
# seen the future (every branch loses him) and tells herself to stay
# unfindable, speaking of it in the past tense because to her it is done.
# Erasure against recurrence: her words try to unmake it; the echoes and
# the future keep writing it back.
REFRAIN_I_SHOTS = {
    20: dict(style="stack", hero=1, foreknow=1.0, exit="dissolve",          # stay lost: unwritten
             backing_at=(560, 330, 64), backing_alpha=0.8, split_second=True),
    21: dict(style="stack", hero=4),                                           # throw away
    22: dict(style="stack", hero=2, foreknow=0.5,                              # the failed rewind
             rewind=dict(start=96.0, snaps=(96.4, 97.3), rate=32),     # the backing agains, on the kicks
             backing_at=(620, 300, 92), backing_alpha=0.85),
    23: dict(hero=[0, 1]),                                                     # forever / again
    24: dict(hero=[0, 1], ghost_second=True),                                  # never / again...
}


def flashes_of(img, t, T, ln, snaps, stop):
    """The refrain's flashes: the corridor as 'Stay lost' begins, the visions
    thrown out among the photos on each pulse of 'Throw away...', one on each
    'again' (`snaps`), and the 'Forever again' tunnel up to `stop`. In Refrain
    I they are her foresight; in Refrain II the same flashes are memory, which
    she can't escape. `ln`: the (stay, throw, forever) lines."""
    import visions as V
    stay, throw, forever = (T.lines[n] for n in ln)
    pulses = [p for p in T.pulses if throw["start"] <= p < throw["end"]]
    img = V.glimpse(img, t, "corridor", stay["start"] + 0.2, strength=0.16, hold=1.6, decay=1.5)
    for k, p0 in enumerate(pulses):                            # thrown out among the photos
        img = V.glimpse(img, t, ["coats", "bullet", "run", "shafts"][k % 4], p0, strength=0.42,
                        hold=0.12, decay=0.45)
    for k, s0 in enumerate(snaps):                             # each 'again' flashes it back
        img = V.glimpse(img, t, ["cutting", "bullet"][k % 2], s0, strength=0.5, hold=0.1, decay=0.5)
    # 'Forever again': a feedback tunnel, the past receding as the future approaches
    t0, t1 = forever["start"], stop
    if t0 <= t < t1:
        bars = [b for b in T.bars if t0 - 0.5 <= b < t1]
        bi = max([i for i, b in enumerate(bars) if b <= t], default=0)
        b0 = bars[bi] if bars else t0
        b1 = bars[bi + 1] if bi + 1 < len(bars) else t1
        img = V.approach(img, ["shafts", "run", "cutting", "bullet"][bi % 4],
                         clamp01((t - b0) / max(0.1, b1 - b0)), strength=0.24)
    return img


def refrain_i(t, T, lines):
    name = "Refrain I: Stay lost now girl"
    s = sec_of(T, name)
    import visions as V
    L = {n: T.lines[n] for n in lines}
    kick = 0.0                                                 # her words don't pulse on the beat
    pulses = [p for p in T.pulses if L[21]["start"] <= p < L[21]["end"]]
    stop = L[24]["start"]                                      # the hard stop

    # the failed rewind runs the whole picture backwards, snapping forward on each 'again'
    rw = REFRAIN_I_SHOTS[22]["rewind"]
    bg_t = t
    if rw["start"] <= t < L[23]["start"]:
        seg = max(x for x in [rw["start"]] + list(rw["snaps"]) if x <= t)
        bg_t = seg - (t - seg) * 1.6
    frozen = t >= stop
    tt = stop if frozen else bg_t

    dx, dy, dr = mem.drift(tt, seed=5)
    img = base_her(tt, T, tally=t)
    img = fx.shift(img, dx * 0.5, dy * 0.5, dr * 0.5, 1.03)                  # no beat zoom
    warm = 0.35 * (1 - smooth(ramp(t, L[21]["start"], L[21]["end"])))  # his warmth thrown out
    photos = ["curtain_window", "rain_window", "doorway_figure", "car_window_night"]
    ms = mems(T, name, photos, every=2, life=9.0, seed=17, keep_left=900)
    for k, m in enumerate(ms):
        # 'throw away everything he found': each memory is flung out on a pulse
        fling = 0.0
        if pulses and m.t0 < pulses[-1]:
            p0 = pulses[k % len(pulses)]
            fling = 2600 * clamp01((t - p0) / 0.45) ** 2
        img = m.draw(img, tt, warm=warm, offset=(dx + fling, dy - 0.25 * fling))
    img = fx.light_leak(img, t, "right", strength=0.05 + warm * 0.3)

    img = flashes_of(img, t, T, (20, 21, 23), rw["snaps"], stop)
    t0, t1 = L[23]["start"], stop                              # 'Forever again'
    img = kin(T, name, lines, REFRAIN_I_SHOTS).draw(img, t, cam=(dx * 0.35, dy * 0.35),
                                                    kick=0.0 if frozen else kick)
    if t0 <= t < t1:
        # a slow, continuous recession: copies of the line glide back into the
        # distance at an even pace (no beat steps), each smaller, fainter, softer.
        # Redrawn from the type at earlier moments, so every render chunk agrees.
        K = kin(T, name, lines, REFRAIN_I_SHOTS)
        dt, cxy = 1.5, (560, 520)
        phase = ((t - t0) / dt) % 1.0
        ramp_in = smooth(ramp(t, t0, t0 + 1.0))
        for k in range(5):
            depth = k + phase
            tk = max(t0 + 0.02, t - depth * dt)
            blank = np.ones((H, W, 3), np.float32)
            m = np.clip(1 - K.draw(blank, tk).mean(-1), 0, 1)
            M = cv2.getRotationMatrix2D(cxy, 1.6 * depth, 0.82 ** depth)
            m = cv2.warpAffine(m, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
            m = cv2.GaussianBlur(m, (0, 0), 1.0 + 1.5 * depth)
            wgt = 0.5 * 0.72 ** depth * smooth(clamp01(depth / 0.8)) * ramp_in
            img = img * (1 - wgt * m[..., None]) + kinetic.INK * wgt * m[..., None]
    # 'Never again...': clean and cold; then the future seeps in from the edges
    g = post("her", exposure=0.92, sat=0.85)
    if frozen:
        seep = smooth(ramp(t, L[24]["words"][1]["start"] + 0.2, s["end"] + 0.3))
        yy, xx = fx._yy_xx()
        edge = np.clip((np.maximum(np.abs(xx - W / 2) / (W / 2), np.abs(yy - H / 2) / (H / 2))
                        - (1 - seep)) / 0.35, 0, 1)
        v = V.load("corridor")
        img = img * (1 - 0.55 * edge[..., None]) + v * 0.55 * edge[..., None]
        g.update(sat=0.45, exposure=0.95)
    return img, g
