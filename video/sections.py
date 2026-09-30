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
from engine import H, W, clamp01, ease_out, font, ramp, smooth
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
def base_her(t, T, dark=0.0, tally=True):
    img = look.padded_room(t, base=C["haze"] * (0.95 - 0.35 * dark))
    if tally:
        img = look.tally(img, t, T, alpha=0.55 * (1 - 0.4 * dark))
    return img


def base_night(t, T, tally=True, lift=1.0):
    img = np.empty((H, W, 3), np.float32)
    img[:] = hexc("#2A2233") * lift
    img = img + (fx.fog(t, seed=12) * 0.035)[..., None]
    if tally:
        img = look.tally(img, t, T, x0=1440, y0=190, color=hexc("#7A6D86"), alpha=0.35, blur=2.0)
    return img


def base_institute(t, T):
    img = look.padded_room(t, base=C["clinic"], seam=0.03)
    return look.tally(img, t, T, alpha=0.35, blur=2.5)


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
                  ghosts=((1.2, -140, 0, -4, 0.18), (2.4, 130, -8, 3, 0.11))),
    "institute": dict(exposure=1.02, lift=0.04, sat=0.5, bloom=0.7, hal=0.7, thresh=1.1,
                      diffusion=0.12, grain=0.06, trail=0.5, shadow="#7A8290",
                      ghosts=((0.8, -60, 0, 0, 0.14), (1.6, 60, 0, 0, 0.08))),
    "run": dict(exposure=1.1, lift=0.06, sat=1.0, bloom=0.95, hal=0.9, thresh=0.55,
                diffusion=0.18, grain=0.06, trail=0.55, shadow="#2E2436",
                ghosts=((0.6, -110, 0, 0, 0.10), (1.2, -220, 0, 0, 0.05))),
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
def title(img, t, t_in, t_out, x=150, y=600, size=210, color=None, dark=False):
    """'lost and' surfacing letter by letter, with the empty space after it."""
    color = C["plum_deep"] if color is None else color
    f = font("her_roman", size)
    txt = "lost and"
    a_out = 1 - smooth(ramp(t, t_out, t_out + 1.6))
    if t < t_in or a_out <= 0:
        return img
    layers = {}
    age = max(0.0, t - t_in)
    track = size * (0.01 + 0.07 * ease_out(clamp01(age / 12), 2))   # spacing slowly opens
    xx = x + 5.0 * age                                               # and the word drifts
    y = y - 2.0 * age
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
        img = fx.add(img, np.array([1, 0.95, 0.9], np.float32), m * 1.2) if dark else fx.over(img, color, m)
    return img


# ================================================================ sections
def intro(t, T, lines):
    s = sec_of(T, "Intro")
    dx, dy, dr = mem.drift(t)
    fade_in = smooth(ramp(t, 0, 4))
    img = base_her(t, T, tally=False)
    img = img * fade_in + (1 - fade_in) * np.float32(0.97)
    if T.bars[0] <= t:
        img = look.tally(img, t, T, alpha=0.5)
    img = fx.shift(img, dx * 0.4, dy * 0.4, dr * 0.4, 1.02)
    img = draw_mems(img, t, mems(T, "Intro", ["lake_overcast", "curtain_bedroom"], every=2,
                                 life=10.0, seed=11, keep_left=900), offset=(dx, dy))
    img = title(img, t, T.bars[1], s["end"] - 2.0)
    return img, post("bleach", exposure=0.95)


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
    img = kin(T, name, lines, ONCE_SHOTS).draw(img, t, cam=(dx * 0.35, dy * 0.35),
                                              kick=T.pulse_env(t, 7.0))
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


def break_institute(t, T, lines):
    s = sec_of(T, "Break I")
    u = smooth(ramp(t, s["start"], s["end"]))
    a = base_her(t, T)
    b = base_institute(t, T)
    img = a * (1 - u) + b * u
    img = draw_mems(img, t, mems(T, "Break I", ["hospital_corridor", "facade_windows"], every=1, life=8.0,
                                 seed=23, keep_left=700))
    img = look.scratches(img, t, 4 * u)
    img = fluorescent(img, t, u)
    return img, post("institute", exposure=0.9 + 0.18 * u)


COATS_SHOTS = {
    26: dict(style="track"), 27: dict(style="stack", hero=7), 28: dict(style="depth", hero=5),
    29: dict(style="stack", hero=6), 30: dict(style="hero", hero=5), 31: dict(style="stack", hero=3),
    32: dict(style="track", hero=1), 33: dict(style="stack", hero=5), 34: dict(style="track", hero=1),
    35: dict(style="stack", hero=6),
    47: dict(style="track", hero=2), 48: dict(style="track"), 49: dict(style="stack", hero=5),
    50: dict(style="depth", hero=5), 51: dict(style="stack", hero=4), 52: dict(style="hero", hero=5),
    53: dict(style="track", hero=1), 54: dict(style="stack", hero=4), 55: dict(style="track", hero=1),
    56: dict(style="stack", hero=5),
    83: dict(style="track"), 84: dict(style="stack", hero=5), 85: dict(style="depth", hero=5),
    86: dict(style="stack", hero=3), 87: dict(style="hero", hero=5), 88: dict(style="track", hero=1),
    89: dict(style="stack", hero=3), 90: dict(style="track", hero=1), 91: dict(style="stack", hero=8),
    92: dict(style="hero", hero=0),
}


def coats(name, photos, seed):
    def scene(t, T, lines):
        img = base_institute(t, T)
        dx, dy, dr = mem.drift(t, seed=seed, amp=(18, 10))
        img = fx.shift(img, dx * 0.4, dy * 0.4, 0, 1.02)
        img = draw_mems(img, t, mems(T, name, photos, every=2, life=9.0, seed=seed, keep_left=1000),
                        offset=(dx, dy))
        scribs = words_like(T, lines, "scribble")
        sc = max((1 - clamp01((t - s) / 1.8)) for s in scribs if t >= s) \
            if any(t >= s for s in scribs) else 0.0
        img = look.scratches(img, t, 2.0 + 9 * sc)
        for k, st in enumerate(words_like(T, lines, "stab")):
            r = np.random.default_rng(900 + k + seed * 31)
            img = look.film_burn(img, t, st, r.uniform(1150, 1700), r.uniform(250, 850), 50 + k)
        img = fluorescent(img, t)
        # 'not worth the time': the exposure clips to white for a beat
        clip = 0.0
        for st in words_like(T, lines, "worth"):
            clip = max(clip, math.exp(-3.5 * (t - st)) if t >= st else 0.0)
        img = fx.vignette(img, 0.55, "#5E6A78")
        img = kin(T, name, lines, COATS_SHOTS, voice="coats").draw(
            img, t, cam=(dx * 0.2, dy * 0.2), kick=T.pulse_env(t, 9.0))
        img = img + np.float32(0.8) * clip
        return img, post("institute", exposure=1.08 + 0.4 * clip)
    return scene


MAN_SHOTS = {37: dict(style="stack", hero=3), 38: dict(style="depth", hero=4),
             39: dict(style="track", hero=5), 40: dict(style="stack", hero=1),
             41: dict(style="track", hero=5), 42: dict(style="hero", hero=1)}


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
    img = kin(T, name, lines, MAN_SHOTS, light=True).draw(
        img, t, cam=(dx * 0.35, dy * 0.35), kick=T.pulse_env(t, 7.0))
    return img, post("night")


VIRTUE_WORDS = ("protecting", "trusting", "believing")


def virtue_opens(T):
    return [word_at(T, 44, v) for v in VIRTUE_WORDS]


def spoken(name, reopen=None):
    def scene(t, T, lines):
        s = sec_of(T, name)
        img = base_night(t, T, lift=0.7)
        opens = virtue_opens(T) if reopen is None else [reopen(T, i) for i in range(3)]
        img = look.beams(img, t, T, [None] * 3, open_times=opens, targets=(690, 860, 1030),
                         strength=0.85)
        img = draw_mems(img, t, mems(T, name, ["doorway_figure", "rain_glass_lights"], every=2,
                                     life=10.0, seed=31, keep_left=1000), warm=0.6)
        img = kin(T, name, lines, {n: dict(voice="him") for n in lines}, voice="him",
                  hold=1.6).draw(img, t)
        return img, post("night", exposure=0.95)
    return scene


def break_threads(t, T, lines):
    img = base_night(t, T, lift=0.8)
    breathe = 0.75 + 0.25 * T.pulse_env(t, 3.0)
    img = look.beams(img, t, T, [None] * 3, targets=(690, 860, 1030), strength=breathe)
    img = draw_mems(img, t, mems(T, "Break II", ["rain_glass_lights", "fog_lamps"], every=2,
                                 life=10.0, seed=37, keep_left=900), warm=0.6)
    return img, post("night")


GUN_SHOTS = {58: dict(style="track", hero=2), 59: dict(style="track", hero=10),
             60: dict(style="stack", hero=4, voice="coats"), 61: dict(style="hero", hero=2),
             62: dict(style="stack", hero=1), 63: dict(style="stack", hero=1),
             64: dict(style="depth", hero=1)}


def gun_and_bullet(t, T, lines):
    name = "A gun and a bullet"
    dx, dy, dr = mem.drift(t, seed=9)
    img = base_night(t, T, lift=1.25)
    img = fx.shift(img, dx * 0.5, dy * 0.5, dr * 0.5, 1.02)
    img = draw_mems(img, t, mems(T, name, ["dark_street", "parking_rain", "streets_night",
                                           "fog_park"], every=2, life=10.0, seed=41,
                                 keep_left=900), offset=(dx, dy), warm=0.2)
    # 'he found her' twice: his light closes on her, the second time it stops short
    for k, st in enumerate(words_like(T, [63, 64], "found")):
        if t >= st:
            u = ease_out(clamp01((t - st) / 1.2))
            x = 1600 - 500 * u * (1 if k == 0 else 0.6)
            img = fx.hotspot(img, t, x, 560, 220, strength=0.45 * (1 - 0.4 * k) * (1 - clamp01((t - st - 2.5) / 2)))
    img = kin(T, name, lines, GUN_SHOTS, light=True).draw(img, t, cam=(dx * 0.35, dy * 0.35),
                                                        kick=T.pulse_env(t, 7.0))
    p = post("night")
    bt = word_at(T, 59, "bullet")
    if bt is not None and t >= bt:
        p["bullet"] = (1470, 925, 1.9)
    return img, p


RUN_SHOTS = {69: dict(style="track", hero=3), 70: dict(style="hero", hero=0),
             71: dict(style="depth", hero=2), 72: dict(style="stack", hero=2),
             73: dict(style="stack", hero=3), 74: dict(style="hero", hero=1),
             75: dict(style="depth", hero=0), 76: dict(style="track", hero=0)}


def the_run(t, T, lines):
    name = "The Run"
    dx, dy, dr = mem.drift(t, seed=13, amp=(60, 26))
    kick = T.pulse_env(t, 6.0)
    img = base_night(t, T, tally=False, lift=0.9)               # time isn't counted here
    img = img + hexc("#3A2210") * 0.3
    img = fx.shift(img, dx, dy, dr, 1.04 + 0.015 * kick)
    photos = ["tunnel_lights", "light_trails", "no_vacancy", "highway_trails", "open_sign",
              "gas_station", "tunnel_dark", "parking_rain", "streets_night", "dusk_drive"]
    img = draw_mems(img, t, mems(T, name, photos, every=1, life=6.5, seed=43, keep_left=900),
                    offset=(dx * 1.5, dy), warm=0.35)
    img = streaks(img, 0.5 + 0.8 * kick)
    img = fx.light_leak(img, t, "right", color="#FFB35A", strength=0.25 + 0.2 * kick)
    img = kin(T, name, lines, RUN_SHOTS, light=True).draw(img, t, cam=(dx * 0.4, dy * 0.4),
                                                        kick=kick)
    return img, post("run")


BUT_SHOTS = {78: dict(style="stack", hero=0), 80: dict(style="track", hero=1),
             81: dict(style="stack", hero=6)}


def but(t, T, lines):
    name = "But..."
    s = sec_of(T, name)
    t_but = T.lines[78]["start"]
    bar = 60 / 70.2 * 4
    frozen = min(t, t_but)                                     # everything stops on 'But'
    drain = smooth(ramp(t, t_but, t_but + bar))
    img = base_night(frozen, T, tally=False, lift=0.9) + hexc("#3A2210") * 0.3
    img = draw_mems(img, t, mems(T, "The Run", ["tunnel_lights", "light_trails", "no_vacancy",
                                                  "highway_trails", "open_sign", "gas_station",
                                                  "tunnel_dark", "parking_rain", "streets_night",
                                                  "dusk_drive"], every=1, life=6.5, seed=43,
                                 keep_left=900), freeze=t_but, warm=0.35 * (1 - drain))
    L = img.mean(-1, keepdims=True)
    img = img * (1 - drain) + (L * 0.4 + 0.62) * drain          # colour drains, bleaches
    # 'round the bend': one hard cold sweep of light
    bend = word_at(T, 81, "bend")
    if bend is not None and t >= bend - 0.3:
        u = clamp01((t - bend + 0.3) / 1.4)
        xx = fx._yy_xx()[1]
        sweep = np.exp(-((xx - (-400 + 2800 * u)) / 260) ** 2) * (1 - u) * 1.4
        img = img + np.array([0.85, 0.92, 1.0], np.float32) * sweep[..., None]
    img = kin(T, name, lines, BUT_SHOTS, hold=1.5).draw(img, t, kick=0.0, react=0.0)
    return img, post("bleach", exposure=0.95 + 0.1 * drain)


CUT_SHOTS = {94: dict(style="stack", hero=4), 95: dict(style="stack", hero=4),
             96: dict(style="stack", hero=4), 97: dict(style="track", hero=7),
             99: dict(style="depth", hero=3), 100: dict(style="depth", hero=3),
             101: dict(style="track", hero=1), 102: dict(style="hero", hero=4),
             103: dict(style="track", hero=0), 104: dict(style="depth", hero=3),
             105: dict(style="hero", hero=1)}


def cutting(t, T, lines):
    name = "The Cutting"
    cuts = [word_at(T, n, "cut") for n in (94, 95, 96)]
    live = sum(1 - smooth(clamp01((t - ct - 0.4) / 1.2)) if t >= ct else 1.0 for ct in cuts) / 3
    eyes = word_at(T, 101, "eyes")
    ears = word_at(T, 101, "ears")
    deaf = ears is not None and t >= ears
    img = base_night(t, T, lift=0.7)
    img = look.beams(img, t, T, cuts, targets=(690, 860, 1030))
    img = draw_mems(img, t, mems(T, name, ["doorway_figure", "car_window_night", "fog_lamps",
                                           "curtain_window", "rain_window", "lake_overcast"],
                                 every=2, life=10.0, seed=47, keep_left=1000), warm=0.5 * live)
    img = fx.hotspot(img, t, 880, 640, 330, strength=0.3 * live)
    # 'her eyes to see': focus goes and doesn't come back
    if eyes is not None and t >= eyes:
        k = smooth(clamp01((t - eyes) / 1.5))
        img = img * (1 - 0.7 * k) + fx.blur(img, 9) * 0.7 * k
    img = kin(T, name, lines, CUT_SHOTS, light=True).draw(
        img, t, kick=0.0 if deaf else T.pulse_env(t, 7.0), react=0.0 if deaf else 1.0)
    p = post("night", exposure=0.95)
    p["bullet"] = (1470, 925, 1.9)
    return img, p


def reprise(t, T, lines):
    """Verse A again, seen through years of accumulated exposure."""
    import compose
    img, p = compose.verse_a_like(t, T, lines, name="Verse A reprise", offset=112, seed=5,
                                  years=True)
    return img, p


def outro(t, T, lines):
    s = sec_of(T, "Outro")
    u = smooth(ramp(t, s["start"], s["end"]))
    dx, dy, dr = mem.drift(t, seed=3)
    img = base_her(t, T)
    img = fx.shift(img, dx * 0.4, dy * 0.4, dr * 0.4, 1.02)
    img = draw_mems(img, t, mems(T, "Outro", ["curtain_bedroom", "lake_overcast", "doorway_figure"],
                                 every=3, life=12.0, seed=53, keep_left=900), offset=(dx, dy))
    img = title(img, t, s["start"] + 6.0, s["end"] - 6.0)
    white = smooth(ramp(t, s["end"] - 7.0, s["end"] - 1.0))
    img = img * (1 - white) + np.float32(1.0) * white
    p = post("bleach", exposure=0.95 + 0.1 * u)
    if t < s["end"] - 3.0:
        p["bullet"] = (1470, 925, 1.9)
    return img, p


# ================================================================ Refrain I: prophecy
# She is precognitive. The refrain comes before the story happens: she has
# seen the future (every branch loses him) and tells herself to stay
# unfindable, speaking of it in the past tense because to her it is done.
# Erasure against recurrence: her words try to unmake it; the echoes and
# the future keep writing it back.
REFRAIN_I_SHOTS = {
    20: dict(style="stack", hero=1, foreknow=1.0, exit="dissolve",          # stay lost: unwritten
             backing_at=(560, 330, 64), backing_alpha=0.8),
    21: dict(style="stack", hero=4),                                           # throw away
    22: dict(style="stack", hero=2, foreknow=0.5,                              # the failed rewind
             rewind=dict(start=96.0, snaps=(96.23, 96.79), rate=32),
             backing_at=(620, 300, 92), backing_alpha=0.85),
    23: dict(hero=[0, 1]),                                                     # forever / again
    24: dict(hero=[0, 1], ghost_second=True),                                  # never / again...
}
_FEEDBACK = {}


def refrain_i(t, T, lines):
    name = "Refrain I: Stay lost now girl"
    s = sec_of(T, name)
    import visions as V
    L = {n: T.lines[n] for n in lines}
    kick = T.pulse_env(t, 6.0)
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
    img = base_her(tt, T)
    img = fx.shift(img, dx * 0.5, dy * 0.5, dr * 0.5, 1.03 + 0.02 * kick * (not frozen))
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

    # glimpses of the future
    img = V.glimpse(img, t, "corridor", L[20]["start"] + 0.2, strength=0.16, hold=1.6, decay=1.5)
    for k, p0 in enumerate(pulses):                            # thrown out among the photos
        img = V.glimpse(img, t, ["coats", "bullet", "run", "shafts"][k % 4], p0, strength=0.42,
                        hold=0.12, decay=0.45)
    for k, s0 in enumerate(rw["snaps"]):                       # each 'again' flashes it back
        img = V.glimpse(img, t, ["cutting", "bullet"][k], s0, strength=0.5, hold=0.1, decay=0.5)

    # 'Forever again': a feedback tunnel, the past receding as the future approaches
    t0, t1 = L[23]["start"], stop
    if t0 <= t < t1:
        bars = [b for b in T.bars if t0 - 0.5 <= b < t1]
        bi = max([i for i, b in enumerate(bars) if b <= t], default=0)
        b0 = bars[bi] if bars else t0
        b1 = bars[bi + 1] if bi + 1 < len(bars) else t1
        img = V.approach(img, ["shafts", "run", "cutting", "bullet"][bi % 4],
                         clamp01((t - b0) / max(0.1, b1 - b0)), strength=0.24)
    img = kin(T, name, lines, REFRAIN_I_SHOTS).draw(img, t, cam=(dx * 0.35, dy * 0.35),
                                                    kick=0.0 if frozen else kick)
    if t0 <= t < t1:
        fb = _FEEDBACK.get(name)
        if fb is not None:
            sc = 0.93 - 0.025 * kick
            rot = 1.2 if int((t - t0) / 1.7) % 2 == 0 else -1.2
            M = cv2.getRotationMatrix2D((W / 2, H / 2 - 40), rot, sc)
            warped = cv2.warpAffine(fb, M, (W, H), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)
            ramp_in = smooth(ramp(t, t0, t0 + 0.8))
            img = img - 0.62 * ramp_in * np.clip(img - warped, 0, None)   # darker copies recede
        _FEEDBACK[name] = img.copy()
    elif t >= t1:
        _FEEDBACK.pop(name, None)

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
