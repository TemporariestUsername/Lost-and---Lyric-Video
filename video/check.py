"""QA for a rendered clip, from its .qa.json log.

Checks: every sung word appears and reaches full opacity promptly after
its onset; words stay inside title-safe; words on screen together from
different lines never collide; frames are finite, not blown or crushed,
and have no sudden unexplained flashes.

    python3 video/check.py render/verse_a.mp4
"""
import json
import pathlib
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from engine import H, W, Timing  # noqa: E402

SAFE = (W * 0.05, H * 0.05, W * 0.95, H * 0.95)
LATE = 0.35


def main(video):
    video = pathlib.Path(video)
    qa = json.loads(video.with_suffix(".qa.json").read_text())
    T = Timing()
    start, end = qa["start"], qa["end"]
    problems = []

    expected = [(w["text"].replace('"', ""), w["start"], L["line"]) for L in T.lines.values()
                for w in L["words"] if not w["backing"] and start <= w["start"] < end - 0.3]
    first_seen = {}
    by_t = defaultdict(list)
    for t, word, onset, x0, y0, x1, y1 in qa["text"]:
        key = (word, round(onset, 3))
        first_seen.setdefault(key, t)
        by_t[round(t, 4)].append((word, onset, x0, y0, x1, y1))
        if x0 < SAFE[0] or y0 < SAFE[1] or x1 > SAFE[2] or y1 > SAFE[3]:
            problems.append(f"outside title-safe at {t:.2f}s: '{word}' ({x0:.0f},{y0:.0f})-({x1:.0f},{y1:.0f})")
    for word, onset, line in expected:
        seen = first_seen.get((word, round(onset, 3)))
        if seen is None:
            problems.append(f"missing word: L{line} '{word}' @ {onset:.2f}s")
        elif seen - onset > LATE:
            problems.append(f"late word: L{line} '{word}' onset {onset:.2f}s, readable at {seen:.2f}s")

    line_of = {(w["text"].replace('"', ""), round(w["start"], 3)): L["line"]
               for L in T.lines.values() for w in L["words"]}
    collided = set()
    for t, items in by_t.items():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                la, lb = line_of.get((a[0], round(a[1], 3))), line_of.get((b[0], round(b[1], 3)))
                if la == lb:
                    continue
                if a[2] < b[4] - 2 and b[2] < a[4] - 2 and a[3] < b[5] - 2 and b[3] < a[5] - 2:
                    key = (la, lb)
                    if key not in collided:
                        collided.add(key)
                        problems.append(f"text collision at {t:.2f}s: L{la} '{a[0]}' / L{lb} '{b[0]}'")

    frames = qa["frames"]
    bad = [k for k, m, j, ok in frames if not ok]
    blown = [k for k, m, j, ok in frames if m > 248 or m < 6]
    jumps = [(k, j) for k, m, j, ok in frames if j > 20]
    if bad:
        problems.append(f"non-finite frames: {bad[:5]}")
    if blown:
        problems.append(f"blown/crushed frames: {len(blown)} (first {blown[:3]})")
    for k, j in jumps[:10]:
        problems.append(f"brightness jump of {j:.0f} levels at frame {k} ({k / 30:.2f}s)")

    means = [m for _, m, _, _ in frames]
    print(f"{video.name}: {len(frames)} frames, mean luminance {min(means):.0f}-{max(means):.0f}; "
          f"{len(expected)} sung words, {len(first_seen)} placed")
    print("\n".join(problems) if problems else "no problems found")
    return problems


if __name__ == "__main__":
    main(sys.argv[1])
