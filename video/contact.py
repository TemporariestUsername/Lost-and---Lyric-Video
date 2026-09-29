"""Contact sheet: two thumbnails per bar, stamped with timecode, bar and lyric.

    python3 video/contact.py render/verse_a.mp4
"""
import json
import pathlib
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from engine import FPS, Timing  # noqa: E402

TW, TH, COLS = 480, 270, 4
FONT = "/usr/share/fonts/truetype/ibm-plex/IBMPlexMono-Regular.ttf"


def tc(t):
    m, s = divmod(t, 60)
    return f"{int(m)}:{s:05.2f}"


def main(video):
    video = pathlib.Path(video)
    qa = json.loads(video.with_suffix(".qa.json").read_text())
    T = Timing()
    start, end = qa["start"], qa["end"]
    bars = [b for b in T.bars if start <= b < end]
    times = sorted(set([start] + [b + d for b in bars for d in (0.05, (T.bars[list(T.bars).index(b) + 1] - b) / 2 if list(T.bars).index(b) + 1 < len(T.bars) else 1.7)]))
    times = [t for t in times if t < end - 1 / FPS]
    ks = [round((t - start) * FPS) for t in times]
    sel = "+".join(f"eq(n\\,{k})" for k in ks)
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(video), "-vf",
                          f"select='{sel}',scale={TW}:{TH}", "-vsync", "0", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, TH, TW, 3)
    rows = (len(frames) + COLS - 1) // COLS
    lab = 44
    sheet = Image.new("RGB", (COLS * TW + (COLS + 1) * 8, rows * (TH + lab) + (rows + 1) * 8), (20, 18, 24))
    d = ImageDraw.Draw(sheet)
    f = ImageFont.truetype(FONT, 14)
    for i, (fr, t) in enumerate(zip(frames, times)):
        x = 8 + (i % COLS) * (TW + 8)
        y = 8 + (i // COLS) * (TH + lab + 8)
        sheet.paste(Image.fromarray(fr), (x, y))
        bar = int(np.searchsorted(T.bars, t + 1e-3, side="right")) - 1
        line = next((L for L in T.lines.values() if L["start"] - 0.2 <= t <= L["end"] + 0.2), None)
        d.text((x, y + TH + 4), f"{tc(t)}  bar {bar}", fill=(220, 214, 230), font=f)
        if line:
            d.text((x, y + TH + 22), f"L{line['line']}: {line['text'][:44]}", fill=(160, 150, 175), font=f)
    out = video.with_name(video.stem + "_contact.jpg")
    sheet.save(out, quality=88)
    print("wrote", out, f"({len(frames)} frames)")


if __name__ == "__main__":
    main(sys.argv[1])
