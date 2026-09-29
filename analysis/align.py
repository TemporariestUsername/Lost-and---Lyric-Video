"""Align canonical lyrics (assets/lyrics.txt) to Whisper word timings.

Global Needleman-Wunsch over normalised tokens with fuzzy word similarity,
so repeated refrains ("scribble scribble, stab stab the charts") land on the
right occurrence. Canonical words with no Whisper match are interpolated
inside their line; lines with too little support are reported as gaps so
they can be re-transcribed as clips (see transcribe.py --clip).

    python3 analysis/align.py whisper_full.json [whisper_<tag>.json ...]

Later files override the words of earlier ones inside their clip window.
Writes analysis/out/lyrics_timed.json and prints a gap report.
"""
import difflib
import json
import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
LYRICS = ROOT / "assets" / "lyrics.txt"
OUT = ROOT / "analysis" / "out"

NUM = {"3": "through", "1": "once", "2": "two"}  # Whisper homophone slips


def norm(w):
    w = w.lower().replace("’", "'")
    w = re.sub(r"[^a-z0-9']", "", w)
    return NUM.get(w, w)


def canonical_tokens():
    toks = []
    lines = LYRICS.read_text().splitlines()
    stanza = 0
    for li, line in enumerate(lines):
        if not line.strip():
            stanza += 1
            continue
        depth = 0
        for raw in re.findall(r"\(|\)|[^\s()]+", line):
            if raw == "(":
                depth += 1
                continue
            if raw == ")":
                depth -= 1
                continue
            n = norm(raw)
            if n:
                toks.append({"line": li + 1, "stanza": stanza, "text": raw,
                             "n": n, "backing": depth > 0})
    return toks, lines


def load_whisper(files):
    words = []
    for f in files:
        d = json.loads((OUT / f).read_text())
        if d.get("clip"):
            a, b = d["clip"]
            words = [w for w in words if not (a <= w["start"] < b)]
        words += d["words"]
    words.sort(key=lambda w: w["start"])
    out = []
    for w in words:
        # Whisper sometimes glues hyphenated/compound words; split them.
        parts = [p for p in re.split(r"[\s\-]+", w["word"]) if norm(p)]
        span = (w["end"] - w["start"]) / max(1, len(parts))
        for k, p in enumerate(parts):
            out.append({"start": w["start"] + k * span,
                        "end": w["start"] + (k + 1) * span,
                        "n": norm(p), "p": w["p"]})
    return out


def sim(a, b):
    if a == b:
        return 1.0
    r = difflib.SequenceMatcher(None, a, b).ratio()
    return r if r >= 0.6 else 0.0


def nw_align(C, W, gap=-0.4):
    n, m = len(C), len(W)
    S = np.zeros((n + 1, m + 1))
    S[:, 0] = gap * np.arange(n + 1)
    S[0, :] = gap * np.arange(m + 1)
    P = np.zeros((n + 1, m + 1), dtype=np.int8)  # 0 diag, 1 up, 2 left
    for i in range(1, n + 1):
        ci = C[i - 1]["n"]
        for j in range(1, m + 1):
            s = sim(ci, W[j - 1]["n"])
            d = S[i - 1, j - 1] + (2 * s - 1 if s else -1)
            u = S[i - 1, j] + gap
            l = S[i, j - 1] + gap
            if d >= u and d >= l:
                S[i, j], P[i, j] = d, 0
            elif u >= l:
                S[i, j], P[i, j] = u, 1
            else:
                S[i, j], P[i, j] = l, 2
    pairs, i, j = {}, n, m
    while i > 0 and j > 0:
        if P[i, j] == 0:
            if sim(C[i - 1]["n"], W[j - 1]["n"]):
                pairs[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif P[i, j] == 1:
            i -= 1
        else:
            j -= 1
    return pairs


def main():
    files = sys.argv[1:] or ["whisper_full.json"]
    C, lines = canonical_tokens()
    W = load_whisper(files)
    pairs = nw_align(C, W)

    for ci, t in enumerate(C):
        if ci in pairs:
            w = W[pairs[ci]]
            t.update(start=round(w["start"], 3), end=round(w["end"], 3),
                     matched=True, p=w["p"])
        else:
            t.update(start=None, end=None, matched=False)

    # Interpolate unmatched words between nearest matched neighbours.
    anchors = [i for i, t in enumerate(C) if t["matched"]]
    for i, t in enumerate(C):
        if t["matched"]:
            continue
        prev = max((a for a in anchors if a < i), default=None)
        nxt = min((a for a in anchors if a > i), default=None)
        t0 = C[prev]["end"] if prev is not None else 0.0
        t1 = C[nxt]["start"] if nxt is not None else t0 + 2.0
        lo = prev if prev is not None else -1
        hi = nxt if nxt is not None else len(C)
        frac = (i - lo) / (hi - lo)
        dur = (t1 - t0) / (hi - lo)
        t["start"] = round(t0 + frac * (t1 - t0) - dur, 3)
        t["end"] = round(t["start"] + dur, 3)

    # Manual line pins (analysis/overrides.json) for unresolvable spots.
    pins = json.loads((ROOT / "analysis" / "overrides.json").read_text())
    for ln, (a, b) in pins["lines"].items():
        ws = [t for t in C if t["line"] == int(ln)]
        weights = np.array([len(t["n"]) + 2 for t in ws], dtype=float)
        edges = a + (b - a) * np.concatenate([[0], np.cumsum(weights)]) / weights.sum()
        for t, s0, s1 in zip(ws, edges[:-1], edges[1:]):
            t.update(start=round(float(s0), 3), end=round(float(s1), 3),
                     matched=True, pinned=True)

    # Per-line summary + gap report (backing vocals excluded from support).
    out_lines, gaps = [], []
    for ln in sorted({t["line"] for t in C}):
        ws = [t for t in C if t["line"] == ln]
        lead = [t for t in ws if not t["backing"]] or ws
        support = sum(t["matched"] for t in lead) / len(lead)
        out_lines.append({
            "line": ln, "text": lines[ln - 1].strip(),
            "stanza": ws[0]["stanza"],
            "start": min(t["start"] for t in ws),
            "end": max(t["end"] for t in ws),
            "support": round(support, 2),
            "pinned": any(t.get("pinned") for t in ws),
            "words": [{k: t[k] for k in ("text", "start", "end", "matched",
                                         "backing")} for t in ws],
        })
        if support < 0.5:
            gaps.append(out_lines[-1])

    (OUT / "lyrics_timed.json").write_text(json.dumps(
        {"sources": files, "lines": out_lines}, indent=1))
    matched = sum(t["matched"] for t in C)
    print(f"matched {matched}/{len(C)} canonical words "
          f"({len(W)} whisper words) from {files}")
    for L in out_lines:
        flag = "  <-- GAP" if L["support"] < 0.5 else ""
        print(f"{L['line']:3d} {L['start']:7.2f}-{L['end']:7.2f} "
              f"{L['support']:.2f} {L['text'][:60]}{flag}")
    # Merge adjacent gap lines into clip suggestions for targeted re-runs.
    groups = []
    for g in gaps:
        if groups and g["line"] - groups[-1][-1]["line"] <= 2:
            groups[-1].append(g)
        else:
            groups.append([g])
    for grp in groups:
        a, b = grp[0]["line"], grp[-1]["line"]
        prev = [L for L in out_lines if L["line"] < a and L["support"] >= 0.5]
        nxt = [L for L in out_lines if L["line"] > b and L["support"] >= 0.5]
        t0 = prev[-1]["end"] - 0.5 if prev else 0.0
        t1 = nxt[0]["start"] + 0.5 if nxt else t0 + 30
        print(f"RERUN --clip {t0:.1f} {t1:.1f} --prompt-lines {a} {b}")


if __name__ == "__main__":
    main()
