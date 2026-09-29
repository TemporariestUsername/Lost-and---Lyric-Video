"""Musical grid + song structure from audio.json and lyrics_timed.json.

librosa tracks this song at ~140 BPM, but that grid is eighth-note
resolution: the kick lands on every other tracked beat (a ~70 BPM pulse),
and each sung line is one 4-pulse bar (~3.4 s). The tracker also slips by
an eighth somewhere mid-song, so pulse parity is chosen locally from kick
energy, and bar phase is chosen per section from where sung lines begin.

    python3 analysis/grid.py     -> analysis/out/grid.json
"""
import json
import pathlib

import librosa
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis" / "out"

# Structure: (name, first lyric line, last lyric line) or instrumental spans
# given by the gap between neighbouring sections. Names drive the storyboard.
SECTIONS = [
    ("Intro", None, None),
    ("Verse A: Through eyes", 1, 9),
    ("There comes a once", 11, 18),
    ("Refrain I: Stay lost now girl", 20, 24),
    ("Break I", None, None),
    ("White Coats I", 26, 35),
    ("They brought a man", 37, 42),
    ("Spoken I: I can help you find", 44, 45),
    ("Break II", None, None),
    ("White Coats II", 47, 56),
    ("A gun and a bullet", 58, 64),
    ("Spoken II: I won't lose you", 66, 67),
    ("The Run", 69, 76),
    ("But...", 78, 81),
    ("White Coats III", 83, 92),
    ("The Cutting", 94, 105),
    ("Refrain II: Stay lost now girl", 107, 111),
    ("Verse A reprise", 113, 121),
    ("Outro", None, None),
]


def main():
    a = json.loads((OUT / "audio.json").read_text())
    lines = json.loads((OUT / "lyrics_timed.json").read_text())["lines"]
    by_line = {L["line"]: L for L in lines}
    beats = np.array(a["beats"])

    y, sr = librosa.load(ROOT / a["file"], sr=22050)
    P = librosa.effects.percussive(y, margin=2.0)
    S = np.abs(librosa.stft(P, hop_length=256))
    f = librosa.fft_frequencies(sr=sr)
    e = S[f < 120].sum(0)
    kick = np.maximum(0, np.diff(e, prepend=e[0]))
    fr = librosa.time_to_frames(beats, sr=sr, hop_length=256)
    k = np.array([kick[max(0, i - 2):i + 3].max() for i in fr])

    # Local parity: which alternate eighth carries the kick, +-12 beats.
    parity = np.zeros(len(beats), dtype=int)
    for i in range(len(beats)):
        lo, hi = max(0, i - 12), min(len(beats), i + 13)
        idx = np.arange(lo, hi)
        parity[i] = int(k[idx[idx % 2 == 1]].sum() > k[idx[idx % 2 == 0]].sum())
    pulses = [float(beats[i]) for i in range(len(beats)) if i % 2 == parity[i]]
    pulses = np.array(pulses)

    # Section time spans. Lyric sections run from first line start to the
    # next section's start; instrumental ones fill the gaps.
    spans = []
    for name, l0, l1 in SECTIONS:
        spans.append([name, l0, l1,
                      by_line[l0]["start"] if l0 else None,
                      by_line[l1]["end"] if l1 else None])
    for i, s in enumerate(spans):
        if s[3] is None:
            s[3] = 0.0 if i == 0 else spans[i - 1][4]
            nxt = next((t[3] for t in spans[i + 1:] if t[3] is not None),
                       a["duration"])
            s[4] = nxt if i < len(spans) - 1 else a["duration"]
    for i in range(len(spans) - 1):  # lyric sections end where next begins
        spans[i][4] = spans[i + 1][3]
    spans[-1][4] = a["duration"]

    # Bar phase per section: the 4-pulse phase that most line starts sit on.
    downbeats = []
    sections = []
    for name, l0, l1, t0, t1 in spans:
        pidx = np.where((pulses >= t0 - 0.05) & (pulses < t1 - 0.05))[0]
        if not len(pidx):
            continue
        phase = 0
        if l0:
            starts = [by_line[n]["start"] for n in range(l0, l1 + 1) if n in by_line]
            near = [int(np.argmin(np.abs(pulses - s))) for s in starts]
            # a sung pickup lands just before the bar; weight exact hits more
            votes = np.zeros(4)
            for s, n in zip(starts, near):
                votes[n % 4] += 1.0 / (1 + 4 * abs(pulses[n] - s))
            phase = int(np.argmax(votes))
        else:
            phase = int(pidx[0] % 4)  # instrumental: start bars at section start
        bars = [float(pulses[i]) for i in pidx if i % 4 == phase]
        downbeats += bars
        sections.append({"name": name, "lines": [l0, l1] if l0 else None,
                         "start": round(t0, 3), "end": round(t1, 3),
                         "bars": len(bars), "first_downbeat": bars[0] if bars else None})

    downbeats = sorted(set(round(d, 4) for d in downbeats))
    bar_of = lambda t: int(np.searchsorted(downbeats, t + 0.05, side="right"))
    for s in sections:
        s["bar_start"] = bar_of(s["start"])
    for L in lines:
        L["bar"] = bar_of(L["start"])

    ipi = np.diff(pulses)
    grid = {
        "pulse_bpm_median": round(60 / float(np.median(ipi)), 2),
        "eighth_bpm_median": round(120 / float(np.median(ipi)), 2),
        "pulses": [round(float(p), 4) for p in pulses],
        "downbeats": downbeats,
        "sections": sections,
        "line_bars": {L["line"]: L["bar"] for L in lines},
    }
    (OUT / "grid.json").write_text(json.dumps(grid, indent=1))
    print(f"pulse {grid['pulse_bpm_median']} BPM (eighths {grid['eighth_bpm_median']}), "
          f"{len(pulses)} pulses, {len(downbeats)} bars")
    for s in sections:
        m, sec = divmod(s["start"], 60)
        print(f"bar {s['bar_start']:3d}  {int(m)}:{sec:05.2f}  {s['end'] - s['start']:6.1f}s "
              f"{s['bars']:3d} bars  {s['name']}")


if __name__ == "__main__":
    main()
