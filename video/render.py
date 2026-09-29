"""Render a time range to MP4 with the original audio.

    python3 video/render.py --section "Verse A: Through eyes" --jobs 4
    python3 video/render.py --start 16.9 --end 55.2 --out render/clip.mp4

Frames sit on the global 30 fps grid (frame k is at t = k / 30), so clips
line up exactly with the final film. The range is split into chunks that
render in parallel; each chunk first renders a warm-up so the exposure echo
is continuous across the joins. Each worker also writes a QA log.
"""
import argparse
import json
import math
import multiprocessing as mp
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from engine import FPS, H, ROOT, W, Timing  # noqa: E402

AUDIO = ROOT / "assets" / "lost_and.mp3"
WARMUP = 3.0


def worker(args):
    k0, k1, path, scene, qa_path = args
    import cv2
    cv2.setNumThreads(1)
    import numpy as np
    import look
    import kinetic
    from compose import Compositor
    T = Timing()
    comp = Compositor(T)
    for k in range(k0 - int(WARMUP * FPS), k0, 2):          # warm-up, discarded
        comp.frame(k / FPS, scene=scene)
    look.TEXT_LOG = []
    kinetic.TEXT_LOG = look.TEXT_LOG
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium",
         "-crf", "17", "-pix_fmt", "yuv420p", "-r", str(FPS), str(path)],
        stdin=subprocess.PIPE)
    stats = []
    prev = None
    for k in range(k0, k1):
        img = np.ascontiguousarray(comp.frame(k / FPS, scene=scene)[..., :3])
        ff.stdin.write(img.tobytes())
        m = float(img.mean())
        jump = abs(m - prev) if prev is not None else 0.0
        stats.append((k, m, jump, bool(np.isfinite(img).all())))
        prev = m
    ff.stdin.close()
    ff.wait()
    pathlib.Path(qa_path).write_text(json.dumps({"frames": stats, "text": look.TEXT_LOG}))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section")
    ap.add_argument("--start", type=float)
    ap.add_argument("--end", type=float)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out")
    a = ap.parse_args()
    T = Timing()
    scene = None
    if a.section:
        sec = next(s for s in T.sections if s["name"] == a.section)
        a.start, a.end, scene = sec["start"], sec["end"], a.section
    slug = (a.section or f"{a.start:.1f}-{a.end:.1f}").split(":")[0].lower().replace(" ", "_")
    out = pathlib.Path(a.out or ROOT / "render" / f"{slug}.mp4")
    tmp = out.parent / f".{slug}_parts"
    tmp.mkdir(parents=True, exist_ok=True)
    k0, k1 = math.ceil(a.start * FPS), math.ceil(a.end * FPS)
    n = k1 - k0
    bounds = [k0 + round(n * i / a.jobs) for i in range(a.jobs + 1)]
    jobs = [(bounds[i], bounds[i + 1], tmp / f"part{i:02d}.mp4", scene, tmp / f"qa{i:02d}.json")
            for i in range(a.jobs) if bounds[i + 1] > bounds[i]]
    print(f"rendering {n} frames ({k0}..{k1 - 1}, {a.start:.3f}-{a.end:.3f}s) in {len(jobs)} jobs",
          flush=True)
    t0 = time.time()
    with mp.get_context("fork").Pool(len(jobs)) as pool:
        for p in pool.imap_unordered(worker, jobs):
            print(f"  done {p.name} ({time.time() - t0:.0f}s)", flush=True)
    lst = tmp / "list.txt"
    lst.write_text("".join(f"file '{j[2].name}'\n" for j in jobs))
    video = tmp / "video.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", str(lst), "-c", "copy", str(video)], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
                    "-ss", f"{k0 / FPS:.6f}", "-t", f"{n / FPS:.6f}", "-i", str(AUDIO),
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "320k",
                    "-movflags", "+faststart", "-shortest", str(out)], check=True)
    qa = {"frames": [], "text": []}
    for j in jobs:
        d = json.loads(pathlib.Path(j[4]).read_text())
        qa["frames"] += d["frames"]
        qa["text"] += d["text"]
    qa.update(start=k0 / FPS, end=k1 / FPS, section=a.section)
    (out.with_suffix(".qa.json")).write_text(json.dumps(qa))
    print(f"wrote {out} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
