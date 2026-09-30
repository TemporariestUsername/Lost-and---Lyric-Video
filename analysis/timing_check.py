"""Check every word's timing against the separated vocal stem.

Whisper gives back-to-back word edges (each word ends where the next begins),
so silences get swallowed into the preceding word. This finds words whose
start is off by >= THRESH seconds against actual vocal activity. Report
only; nothing is changed.

    python3 -m demucs --two-stems=vocals -n htdemucs -o render/stems assets/lost_and.mp3
    python3 analysis/timing_check.py

Writes analysis/out/timing_check.md.
"""
import json
import pathlib

import librosa
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
STEM = ROOT / "render" / "stems" / "htdemucs" / "lost_and" / "vocals.wav"
OUT = ROOT / "analysis" / "out"
SR, HOP = 16000, 320          # 20 ms frames
THRESH = 0.3                  # seconds off before a word is reported
GAP = 0.2                     # silence that counts as a real gap


def activity():
    y, _ = librosa.load(STEM, sr=SR, mono=True)
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP)[0]
    db = librosa.amplitude_to_db(rms, ref=1.0)
    thr = np.percentile(db, 95) - 28
    act = db > thr
    # close tiny holes, drop tiny islands
    n_fill, n_isl = int(0.08 * SR / HOP), int(0.06 * SR / HOP)
    runs, i = [], 0
    while i < len(act):
        j = i
        while j < len(act) and act[j] == act[i]:
            j += 1
        runs.append([bool(act[i]), i, j])
        i = j
    for r in runs:
        if not r[0] and r[2] - r[1] <= n_fill:
            r[0] = True
    act = np.zeros_like(act)
    for v, a, b in runs:
        act[a:b] = v
    segs, i = [], 0
    while i < len(act):
        if act[i]:
            j = i
            while j < len(act) and act[j]:
                j += 1
            if j - i > n_isl:
                segs.append((i * HOP / SR, j * HOP / SR))
            i = j
        else:
            i += 1
    return segs


def active_between(segs, a, b):
    return [(max(s, a), min(e, b)) for s, e in segs if e > a and s < b]


def main():
    segs = activity()
    lines = json.loads((OUT / "lyrics_timed.json").read_text())["lines"]
    pins = json.loads((ROOT / "analysis" / "overrides.json").read_text())
    pinned_lines = set(pins.get("lines", {}))
    words = []
    for L in lines:
        for k, w in enumerate(L["words"]):
            words.append(dict(w, line=L["line"], idx=k,
                              pinned=str(L["line"]) in pinned_lines or
                              str(k) in pins.get("words", {}).get(str(L["line"]), {})))
    words.sort(key=lambda w: w["start"])
    rows = []
    for i, w in enumerate(words):
        s, e = w["start"], w["end"]
        prev = words[i - 1] if i else None
        # A) starts late: inside the previous word's span there is a silence of
        #    >= GAP, then the voice starts, well before this word's timestamp
        if prev is not None and s - prev["start"] > THRESH + GAP:
            win = active_between(segs, prev["start"], s)
            for (a0, a1), (b0, b1) in zip(win, win[1:]):
                if b0 - a1 >= GAP and s - b0 >= THRESH:
                    rows.append(dict(kind="starts late", w=w, now=s, sug=round(b0, 2),
                                     delta=round(b0 - s, 2), note=f"silence {b0 - a1:.2f}s before"))
                    break
        # B) starts early: silence at the timestamp, voice arrives later
        here = active_between(segs, s - 0.05, s + THRESH)
        if not here:
            later = [a for a, b in active_between(segs, s, e + 0.4) if a - s >= THRESH]
            if later:
                rows.append(dict(kind="starts early", w=w, now=s, sug=round(later[0], 2),
                                 delta=round(later[0] - s, 2), note="silent at timestamp"))
        # C) held too long (info): long silent tail inside the word
        inside = active_between(segs, s, e)
        if inside and e - inside[-1][1] >= 0.6:
            rows.append(dict(kind="silent tail (info)", w=w, now=e, sug=round(inside[-1][1], 2),
                             delta=round(inside[-1][1] - e, 2), note="end could be earlier"))

    rows.sort(key=lambda r: (r["kind"].endswith("(info)"), r["w"]["start"]))
    md = ["# Word timing check against the vocal stem", "",
          f"Vocal stem: Demucs htdemucs. Threshold {THRESH}s. Report only; nothing changed.", "",
          "| time | line | word | issue | now | suggested | shift | note |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        w = r["w"]
        m, sec = divmod(w["start"], 60)
        md.append(f"| {int(m)}:{sec:05.2f} | {w['line']} | {w['text']}{' (backing)' if w['backing'] else ''}"
                  f"{' [pinned]' if w['pinned'] else ''} | {r['kind']} | {r['now']:.2f} | {r['sug']:.2f} "
                  f"| {r['delta']:+.2f} | {r['note']} |")
    (OUT / "timing_check.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    act = sum(b - a for a, b in segs)
    print(f"\n{len(segs)} vocal segments, {act:.0f}s active; {len(rows)} findings "
          f"({sum(not r['kind'].endswith('(info)') for r in rows)} start issues)")


if __name__ == "__main__":
    main()
