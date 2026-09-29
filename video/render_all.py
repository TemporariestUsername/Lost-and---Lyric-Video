"""Render sections one after another, each followed by QA and a contact sheet.

    python3 video/render_all.py                  # every section, in film order
    python3 video/render_all.py "The Run" But... # just these (name prefixes)

Writes render/<slug>.mp4, render/<slug>.qa.json, render/<slug>_contact.jpg
and appends a line per section to render/summary.txt.
"""
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from engine import ROOT, Timing  # noqa: E402

PY = sys.executable
V = ROOT / "video"


def slug(name):
    return name.split(":")[0].lower().replace(" ", "_").replace(".", "")


def main(only):
    T = Timing()
    names = [s["name"] for s in T.sections]
    if only:
        names = [n for n in names if any(n.startswith(o) for o in only)]
    out = ROOT / "render"
    out.mkdir(exist_ok=True)
    summary = out / "summary.txt"
    for name in names:
        s = slug(name)
        mp4 = out / f"{s}.mp4"
        t0 = time.time()
        r = subprocess.run([PY, str(V / "render.py"), "--section", name, "--jobs", "4",
                            "--out", str(mp4)], capture_output=True, text=True)
        if r.returncode != 0:
            msg = f"{name}: RENDER FAILED\n{r.stderr[-2000:]}"
        else:
            qa = subprocess.run([PY, str(V / "check.py"), str(mp4)], capture_output=True, text=True)
            subprocess.run([PY, str(V / "contact.py"), str(mp4)], capture_output=True, text=True)
            lines = [l for l in qa.stdout.splitlines() if l.strip()]
            msg = f"{name}: {time.time() - t0:.0f}s | " + " | ".join(lines[:6])
        print(msg, flush=True)
        with summary.open("a") as f:
            f.write(msg + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
