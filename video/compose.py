"""Frame compositor: scene (linear RGB) -> exposure echo -> optics -> film.

Frames must be rendered in order through one Compositor, because the
exposure echo carries earlier moments forward. A still needs warm-up.
"""
import math

import numpy as np

import fx
import look
from engine import H, W, clamp01, ramp, smooth
from look import C


# ---------------------------------------------------------------- scenes
def verse_a(t, T, lines):
    """Her room: lavender haze; she sits in three exposures; he is the warm
    light that floods in as she gets lost inside him."""
    img = look.padded_room(t)
    img = look.tally(img, t, T)
    # warmth grows from "a voice came to her" and floods on "lost inside him"
    voice = T.lines[5]["start"]
    flood = T.lines[7]["start"]
    warm = 0.25 * smooth(ramp(t, voice, voice + 6)) + 0.9 * smooth(ramp(t, flood, flood + 2.6))
    # him: warmth just behind her shoulder (her hair occludes it), and a
    # leak from beyond the frame
    hs = lambda im: fx.hotspot(im, t, 1180 + 150, 960 - 0.58 * 650, 340,
                               strength=0.04 + 0.15 * warm)
    img = look.her_presence(img, t, T, cx=1180, floor=960, s=650, rim=0.3 + 0.9 * warm,
                            light=(1180 + 170, 960 - 0.60 * 650), behind=hs)
    img = fx.light_leak(img, t, "right", strength=0.05 + warm * 0.25)
    cur = [n for n in lines if T.lines[n]["start"] - 0.3 <= t < T.lines[n]["end"] + 2.6]
    for n in cur[-2:]:
        img = look.her_words(img, t, T, [n], 150, 470 + 150 * (n % 2), size=80,
                             color=C["plum_ink"], max_w=820)
    return img, dict(exposure=0.95, lift=0.10, sat=0.8, bloom=0.55, hal=0.55, thresh=1.0,
                     diffusion=0.35, grain=0.045, trail=0.62,
                     ghosts=((1.2, -160, 0, -5, 0.22), (2.4, 150, -10, 4, 0.14)))


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


SCENES = {
    "Verse A: Through eyes": (verse_a, range(1, 10)),
    "White Coats I": (white_coats, range(26, 36)),
    "The Cutting": (cutting, range(94, 106)),
}


# ---------------------------------------------------------------- post
class Compositor:
    def __init__(self, T):
        self.T = T
        self.exp = fx.Exposure()

    def frame(self, t):
        sec = self.T.section_at(t)
        fn, lines = SCENES[sec["name"]]
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
