"""Contact sheet of photo candidates (or picks), grouped by theme.

    python3 assets/photos/sheet.py            # candidates -> candidates_sheet_*.jpg
"""
import json
import pathlib
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = pathlib.Path(__file__).resolve().parent
FONT = "/usr/share/fonts/truetype/ibm-plex/IBMPlexMono-Regular.ttf"
TW, TH, COLS, PER = 300, 200, 6, 36


def main(out_dir):
    idx = json.loads((HERE / "candidates.json").read_text())
    items = sorted(idx.values(), key=lambda d: (d["theme"], d["file"]))
    f = ImageFont.truetype(FONT, 13)
    out_dir = pathlib.Path(out_dir)
    for page in range(0, len(items), PER):
        chunk = items[page:page + PER]
        rows = (len(chunk) + COLS - 1) // COLS
        sheet = Image.new("RGB", (COLS * (TW + 6) + 6, rows * (TH + 26) + 6), (18, 16, 22))
        d = ImageDraw.Draw(sheet)
        for i, it in enumerate(chunk):
            im = Image.open(HERE / "candidates" / it["file"]).convert("RGB")
            im.thumbnail((TW, TH))
            x = 6 + (i % COLS) * (TW + 6)
            y = 6 + (i // COLS) * (TH + 26)
            sheet.paste(im, (x + (TW - im.width) // 2, y + (TH - im.height) // 2))
            d.text((x, y + TH + 4), it["file"].replace(".jpg", ""), fill=(220, 214, 230), font=f)
        p = out_dir / f"candidates_sheet_{page // PER + 1}.jpg"
        sheet.save(p, quality=85)
        print("wrote", p)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else HERE)
