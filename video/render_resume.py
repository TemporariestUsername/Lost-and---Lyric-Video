"""Finish an interrupted full render: re-render only the missing ranges, in
small resumable chunks (a chunk with its QA log is done), then join them with
the parts that did finish and mux the audio.

    python3 video/render_resume.py render/.0.0-493.6_parts 0 493.62 render/lost_and_full.mp4
"""
import json
import math
import multiprocessing as mp
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from engine import FPS  # noqa: E402
import render as R  # noqa: E402

CHUNK = 600


def main(parts, start, end, out, jobs=4):
    P = pathlib.Path(parts)
    k0, k1 = math.ceil(start * FPS), math.ceil(end * FPS)
    n = k1 - k0
    bounds = [k0 + round(n * i / jobs) for i in range(jobs + 1)]
    plan = []                                   # (k0, k1, mp4, qa) in film order
    for i in range(jobs):
        whole, qa = P / f"part{i:02d}.mp4", P / f"qa{i:02d}.json"
        if whole.exists() and qa.exists():
            plan.append((bounds[i], bounds[i + 1], whole, qa))
            continue
        for a in range(bounds[i], bounds[i + 1], CHUNK):
            b = min(a + CHUNK, bounds[i + 1])
            plan.append((a, b, P / f"c{a:05d}.mp4", P / f"c{a:05d}.json"))
    todo = [(a, b, str(m), None, str(q)) for a, b, m, q in plan if not (m.exists() and q.exists())]
    print(f"{len(plan)} pieces, {len(todo)} to render", flush=True)
    with mp.get_context("fork").Pool(jobs) as pool:
        for p in pool.imap_unordered(R.worker, todo):
            print(f"  done {pathlib.Path(p).name}", flush=True)
    lst = P / "list.txt"
    lst.write_text("".join(f"file '{m.name}'\n" for _, _, m, _ in plan))
    video = P / "video.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", str(lst), "-c", "copy", str(video)], check=True)
    for _, _, m, _ in plan:
        m.unlink()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
                    "-ss", f"{k0 / FPS:.6f}", "-t", f"{n / FPS:.6f}", "-i", str(R.AUDIO),
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "320k",
                    "-movflags", "+faststart", "-shortest", str(out)], check=True)
    video.unlink()
    qa = {"frames": [], "text": []}
    for _, _, _, q in plan:
        d = json.loads(q.read_text())
        qa["frames"] += d["frames"]
        qa["text"] += d["text"]
    qa.update(start=k0 / FPS, end=k1 / FPS, section=None)
    pathlib.Path(out).with_suffix(".qa.json").write_text(json.dumps(qa))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4])
