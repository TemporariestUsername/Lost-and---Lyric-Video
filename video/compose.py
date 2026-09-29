"""Frame compositor: scene (linear RGB) -> exposure echo -> optics -> film.

Frames must be rendered in order through one Compositor, because the
exposure echo carries earlier moments forward. A still needs warm-up.
"""
import math

import numpy as np

import fx
import look
import memories as mem
import kinetic
from engine import H, W, clamp01, ramp, smooth
from look import C


# ---------------------------------------------------------------- scenes
_MEM = {}
_KIN = {}

# per-line shot design for Verse A: layout style and hero word index
VERSE_A_SHOTS = {
    1: dict(style="track", hero=1),     # eyes
    2: dict(style="hero", hero=1),      # years
    3: dict(style="stack", hero=2),     # rooms
    4: dict(style="depth", hero=2),     # self
    5: dict(style="hero", hero=2),      # voice
    6: dict(hero=[3, 7]),               # ever / once: eternity against a single moment
    7: dict(style="stack", hero=2),     # lost
    8: dict(style="depth", hero=0),     # lost (and lost)
    9: dict(style="track", hero=4),     # never
}


def kinetic_for(T, name, lines, shots=None):
    if name not in _KIN:
        _KIN[name] = kinetic.Kinetic(T, lines, overrides=shots)
    return _KIN[name]


def memories_for(T, name, scenes, **kw):
    if name not in _MEM:
        sec = next(x for x in T.sections if x["name"] == name)
        _MEM[name] = mem.schedule(T, scenes, sec["start"], sec["end"], **kw)
    return _MEM[name]


def verse_a_like(t, T, lines, name="Verse A: Through eyes", offset=0, seed=3, years=False):
    """Her room, but she is never shown: memories surface and drift through
    the haze, and his warmth creeps in until it colours them. The reprise
    (years=True) replays it overexposed, with many more of her agains."""
    sec = next(x for x in T.sections if x["name"] == name)
    voice, flood = T.lines[5 + offset]["start"], T.lines[7 + offset]["start"]
    warm = 0.25 * smooth(ramp(t, voice, voice + 6)) + 0.9 * smooth(ramp(t, flood, flood + 2.6))
    if years:
        warm *= 0.55                                                   # fainter now
    dx, dy, dr = mem.drift(t)
    u = clamp01((t - sec["start"]) / (sec["end"] - sec["start"]))
    zoom = 1.02 + 0.04 * smooth(u) + 0.04 * smooth(ramp(t, flood, flood + 6))

    img = look.padded_room(t, base=C["haze"] * 0.95)
    img = look.tally(img, t, T)
    img = fx.shift(img, dx * 0.5, dy * 0.5, dr * 0.5, zoom)            # far wall, slow
    # before -> the room -> his voice arriving -> lost inside him
    scenes = ["curtain_bedroom", "lake_overcast", "curtain_window", "rain_window",
              "fog_lamps", "doorway_figure", "car_window_night", "rain_glass_lights",
              "dusk_drive"]
    for m in memories_for(T, name, scenes, every=2, life=11.0, seed=seed, keep_left=900):
        img = m.draw(img, t, warm=warm, offset=(dx, dy))
    img = fx.light_leak(img, t, "right", strength=0.05 + warm * 0.3)
    shots = {n + offset: v for n, v in VERSE_A_SHOTS.items()}
    img = kinetic_for(T, name, lines, shots).draw(
        img, t, cam=(dx * 0.35, dy * 0.35), kick=T.pulse_env(t, 7.0))
    p = dict(exposure=0.9, lift=0.08, sat=0.85, bloom=0.5, hal=0.55, thresh=1.0,
             diffusion=0.14, grain=0.045, trail=0.62,
             ghosts=((1.2, -160, 0, -5, 0.16), (2.4, 150, -10, 4, 0.10)))
    if years:
        p.update(exposure=1.02, lift=0.12, sat=0.7, trail=0.7,
                 ghosts=((1.0, -180, 0, -5, 0.22), (2.0, 170, -10, 4, 0.16),
                         (3.4, -70, 14, 2, 0.12), (4.6, 90, -16, -3, 0.08)))
    return img, p


def verse_a(t, T, lines):
    return verse_a_like(t, T, lines)


def white_coats(t, T, lines):
    """Institute: overexposed, cold, flickering. Scribbles are emulsion
    scratches, stabs are film burns, the chant is etched in the fog."""
    img = look.padded_room(t, base=C["clinic"], seam=0.03)
    img = look.tally(img, t, T, alpha=0.35, blur=2.5)
    img = look.her_presence(img, t, T, cx=1010, floor=860, s=380, ghosts=True, rim=0.0,
                            fade=0.6, hair_col=C["plum_ink"], body_col=C["haze_lo"])
    ws = [w for n in lines for w in T.lines[n]["words"]]
    scrib = max((1 - clamp01((t - w["end"]) / 1.5)) for w in ws
                if w["text"].lower().startswith("scribble") and t >= w["start"]) \
        if any(w["text"].lower().startswith("scribble") and t >= w["start"] for w in ws) else 0
    img = look.scratches(img, t, 2.0 + 8 * scrib)
    cur = [n for n in lines if T.lines[n]["start"] - 0.2 <= t < T.lines[n]["end"] + 0.8]
    if cur:
        img = look.etched(img, t, T, cur[-1:], 170, 250, size=50, big={"HOW", "ANYTHING"})
    stabs = [w for w in ws if w["text"].lower().startswith("stab")]
    for k, w in enumerate(stabs):
        r = np.random.default_rng(900 + k)
        img = look.film_burn(img, t, w["start"], r.uniform(500, 1500), r.uniform(300, 800), 50 + k)
    # fluorescent flicker: irregular dips and a rolling band
    fi = int(t * 30)
    fl = 1 + 0.05 * math.sin(t * 47) + (-0.12 if (fi * 7919) % 23 == 0 else 0)
    band = np.exp(-(((fx._yy_xx()[0] - (t * 380) % (H + 400) + 200) / 120) ** 2)) * 0.06
    img = img * fl * (1 - band[..., None])
    img = fx.vignette(img, 0.55, "#5E6A78")
    return img, dict(exposure=1.08, lift=0.04, sat=0.5, bloom=0.7, hal=0.7, thresh=1.1,
                     diffusion=0.3, grain=0.06, trail=0.5, shadow="#7A8290",
                     ghosts=((0.8, -60, 0, 0, 0.18), (1.6, 60, 0, 0, 0.1)))


def cutting(t, T, lines):
    """Dark. His three virtues are shafts of warm light; each is shuttered
    on its 'cut'. The bullet is the only thing in focus."""
    img = np.empty((H, W, 3), np.float32)
    img[:] = C["night"]
    img = img + (fx.fog(t, seed=12) * 0.02)[..., None]
    cuts = []
    for n in (94, 95, 96):
        cw = next(w for w in T.lines[n]["words"] if w["text"].lower() == "cut")
        cuts.append(cw["start"])
    img = look.beams(img, t, T, cuts, targets=(690, 860, 1030))
    # the warmth behind her drains with each cut
    live = sum(1 - smooth(clamp01((t - ct - 0.4) / 1.2)) if t >= ct else 1.0 for ct in cuts) / 3
    back = lambda im: fx.hotspot(im, t, 880, 990 - 0.62 * 560, 330, strength=0.30 * live)
    img = look.her_presence(img, t, T, cx=820, floor=990, s=560, ghosts=True, rim=0.5 * live,
                            body_col=look.hexc("#1E1726"), hair_col=look.hexc("#0B080F"),
                            light=(1000, 420), face=False, behind=back)
    img = look.tally(img, t, T, x0=1420, y0=150, color=look.hexc("#6E5F78"), alpha=0.35,
                     blur=2.0)
    cur = [n for n in lines if T.lines[n]["start"] - 0.3 <= t < T.lines[n]["end"] + 2.0]
    for n in cur[-1:]:
        img = look.her_words(img, t, T, [n], 1180, 560, size=72,
                             color=look.hexc("#FFE9D6"), hot=True, max_w=640)
    return img, dict(exposure=1.0, lift=0.05, sat=0.9, bloom=0.9, hal=0.8, thresh=0.6,
                     diffusion=0.25, grain=0.055, trail=0.6, shadow="#2A1F33",
                     bullet=(1500, 930), ghosts=((1.2, -120, 0, -3, 0.2), (2.4, 120, 0, 3, 0.12)))


import sections as S  # noqa: E402

SCENES = {
    "Intro": (S.intro, range(0)),
    "Verse A: Through eyes": (verse_a, range(1, 10)),
    "There comes a once": (S.there_comes_a_once, range(11, 19)),
    "Refrain I: Stay lost now girl": (S.refrain("Refrain I: Stay lost now girl", True), range(20, 25)),
    "Break I": (S.break_institute, range(0)),
    "White Coats I": (S.coats("White Coats I", ["hospital_corridor", "cinderblock_plate", "stairwell_spiral", "corridor"], 61), range(26, 36)),
    "They brought a man": (S.brought_a_man, range(37, 43)),
    "Spoken I: I can help you find": (S.spoken("Spoken I: I can help you find"), range(44, 46)),
    "Break II": (S.break_threads, range(0)),
    "White Coats II": (S.coats("White Coats II", ["stairwell_cage", "cinderblock_hole", "facade_windows", "hospital_corridor"], 67), range(47, 57)),
    "A gun and a bullet": (S.gun_and_bullet, range(58, 65)),
    "Spoken II: I won't lose you": (S.spoken("Spoken II: I won't lose you",
                                             reopen=lambda T, i: T.lines[66]["start"] + 0.9 * i),
                                    range(66, 68)),
    "The Run": (S.the_run, range(69, 77)),
    "But...": (S.but, range(78, 82)),
    "White Coats III": (S.coats("White Coats III", ["cinderblock_hole", "stairwell_spiral", "corridor", "stairwell_cage"], 71), range(83, 93)),
    "The Cutting": (S.cutting, range(94, 106)),
    "Refrain II: Stay lost now girl": (S.refrain("Refrain II: Stay lost now girl", False),
                                       range(107, 112)),
    "Verse A reprise": (S.reprise, range(113, 122)),
    "Outro": (S.outro, range(0)),
}


# ---------------------------------------------------------------- post
class Compositor:
    def __init__(self, T):
        self.T = T
        self.exp = fx.Exposure()

    def frame(self, t, scene=None):
        """Render time t. `scene` forces a section's scene (used for the
        warm-up frames before a section whose neighbour isn't built yet)."""
        name = scene or self.T.section_at(t)["name"]
        fn, lines = SCENES[name]
        img, p = fn(t, self.T, [n for n in lines if n in self.T.lines])
        img = fx.dust(img, t, self.T, strength=0.8 if p["exposure"] < 1.3 else 0.5)
        img = self.exp.apply(img, t, trail=p["trail"], ghosts=p["ghosts"])
        img = fx.bloom(img, p["bloom"], p["thresh"])
        img = fx.halation(img, p["hal"], p["thresh"] + 0.1)
        img = fx.diffusion(img, p["diffusion"])
        img = fx.vignette(img, 0.3)
        img = fx.tone(img, p["exposure"], p["lift"], p.get("shadow", "#6B5F80"), p["sat"])
        img = fx.chroma_edges(img, 3)
        img = fx.grain(img, t, p["grain"])
        out = (img * 255).astype(np.uint8)
        if "bullet" in p:
            out = look.bullet(out, *p["bullet"])
        return out
