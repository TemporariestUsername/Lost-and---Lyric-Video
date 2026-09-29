"""Build storyboard/index.html from the analysis data and spec.py.

    python3 storyboard/build.py
"""
import html
import json
import pathlib
import sys

import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "analysis" / "out"
sys.path.insert(0, str(HERE))
from spec import RENDERS, SECTIONS, STYLE_FRAMES, WORLDS  # noqa: E402


def tc(t):
    m, s = divmod(t, 60)
    return f"{int(m)}:{s:04.1f}"


def main():
    a = json.loads((OUT / "audio.json").read_text())
    g = json.loads((OUT / "grid.json").read_text())
    lines = json.loads((OUT / "lyrics_timed.json").read_text())["lines"]
    pins = json.loads((ROOT / "analysis" / "overrides.json").read_text())["lines"]
    by_line = {L["line"]: L for L in lines}

    t = np.array(a["curve_times"])
    loud = np.array(a["loudness_db_smooth"])
    t2 = np.arange(0, a["duration"], 0.5)
    loud2 = np.interp(t2, t, loud)

    secs = []
    for s in g["sections"]:
        sp = SECTIONS[s["name"]]
        m = (t >= s["start"]) & (t < s["end"])
        l0, l1 = s["lines"] or (None, None)
        first = by_line[l0]["text"] if l0 else ""
        secs.append(dict(
            name=s["name"], start=s["start"], end=s["end"], bar=s["bar_start"],
            bars=s["bars"], world=sp["world"], loud=round(float(loud[m].mean()), 1),
            first=first, lines=[l0, l1] if l0 else None,
            visual=sp["visual"], type=sp["type"], sync=sp["sync"]))
    for i, s in enumerate(secs):
        s["bar_end"] = secs[i + 1]["bar"] - 1 if i + 1 < len(secs) else len(g["downbeats"]) - 1

    frame_meta = []
    for fname, sec, lyric, tt in STYLE_FRAMES:
        src = HERE / "frames" / f"{fname}.png"
        dst = HERE / "frames" / f"{fname}.jpg"
        Image.open(src).convert("RGB").save(dst, quality=88, optimize=True)
        bar = int(np.searchsorted(g["downbeats"], tt, side="right"))
        frame_meta.append(dict(file=f"frames/{fname}.jpg", section=sec, lyric=lyric,
                               t=tt, bar=bar, key=fname[0]))

    data = dict(
        duration=a["duration"], loud_t=0.5,
        loud=[round(float(v), 1) for v in loud2],
        bars=g["downbeats"], sections=[{k: s[k] for k in ("name", "start", "end", "world", "bar")}
                                      for s in secs],
        lines=[dict(n=L["line"], s=L["start"], e=L["end"], x=L["text"],
                    pin=L.get("pinned", False)) for L in lines],
        frames=[{k: f[k] for k in ("key", "t", "section")} for f in frame_meta],
    )

    facts = [
        ("Length", tc(a["duration"]).rsplit(".", 1)[0] + f" ({a['duration']:.1f} s)"),
        ("Pulse", f"{g['pulse_bpm_median']:.1f} BPM kick"),
        ("Grid", f"{g['eighth_bpm_median']:.0f} BPM eighths, drifting 136→142"),
        ("Bars", f"{len(g['downbeats'])} × ~3.4 s"),
        ("Lyrics", "616 / 630 words timed"),
        ("Output", "1920×1080 · 30 fps"),
    ]

    def esc(s):
        return html.escape(s, quote=True)

    rows = []
    for s in secs:
        wname, wdesc = WORLDS[s["world"]]
        frame = next((f for f in frame_meta if f["section"] == s["name"]), None)
        lyr = ""
        if s["lines"]:
            lyr = f'<p class="lyr">{esc(s["first"])}</p>'
        fr = (f'<a class="fref" href="#frame-{frame["key"]}">Style frame {frame["key"]}</a>'
              if frame else "")
        rows.append(f"""
<article class="sec" data-world="{s['world']}" id="sec-{s['bar']}">
  <div class="when">
    <span class="bars">bars {s['bar']}–{s['bar_end']}</span>
    <span class="tc">{tc(s['start'])} → {tc(s['end'])}</span>
    <span class="meta">{s['end'] - s['start']:.1f} s · {s['loud']:.0f} dB</span>
  </div>
  <div class="body">
    <header><span class="chip" data-world="{s['world']}">{esc(wname)}</span>
      <h3>{esc(s['name'])}</h3>{fr}</header>
    {lyr}
    <dl>
      <dt>Picture</dt><dd>{esc(s['visual'])}</dd>
      <dt>Type</dt><dd>{esc(s['type'])}</dd>
      <dt>Sync</dt><dd>{esc(s['sync'])}</dd>
    </dl>
  </div>
</article>""")

    frames_html = "".join(f"""
<figure id="frame-{f['key']}">
  <img src="{f['file']}" alt="Style frame {f['key']}: {esc(f['section'])}, {esc(f['lyric'])}" width="1920" height="1080" loading="lazy">
  <figcaption><span class="k">{f['key']}</span> <b>{esc(f['section'])}</b>
    <span class="tc">{tc(f['t'])} · bar {f['bar']}</span>
    <span class="lyr">{esc(f['lyric'])}</span></figcaption>
</figure>""" for f in frame_meta)

    renders_html = ""
    if (HERE / "renders" / "film" / "index.m3u8").exists():
        renders_html += f"""
<figure class="render" id="render-film">
  <video id="film" controls playsinline preload="none" poster="renders/film_poster.jpg" width="1280" height="720"
         data-src="renders/film/index.m3u8"></video>
  <figcaption><b>The full film</b>
    <span class="tc">0:00 → {tc(a['duration'])} · all 19 sections</span>
    <span class="note">Streamed at 720p for the page. The 1080p master was rendered in one continuous pass
      and verified with ffprobe. <a href="renders/film_contact.jpg">Contact sheet</a></span></figcaption>
</figure>"""
    for slug, name, note in RENDERS:
        sec = next(x for x in secs if x["name"] == name)
        renders_html += f"""
<figure class="render" id="render-{slug}">
  <video controls playsinline preload="metadata" poster="renders/{slug}_poster.jpg" width="1280" height="720">
    <source src="renders/{slug}.mp4" type="video/mp4">
  </video>
  <figcaption><b>{esc(name)}</b>
    <span class="tc">{tc(sec['start'])} \u2192 {tc(sec['end'])} \u00b7 bars {sec['bar']}\u2013{sec['bar_end']}</span>
    <span class="note">{esc(note)} <a href="renders/{slug}_contact.jpg">Contact sheet</a></span></figcaption>
</figure>"""

    facts_html = "".join(f"<div><dt>{esc(k)}</dt><dd>{esc(v)}</dd></div>" for k, v in facts)
    worlds_html = "".join(
        f'<li><span class="chip" data-world="{k}">{esc(n)}</span> {esc(d)}</li>'
        for k, (n, d) in WORLDS.items())
    pin_list = ", ".join(f"line {k}" for k in pins)

    page = (HERE / "template.html").read_text()
    page = (page.replace("{{FACTS}}", facts_html)
                .replace("{{WORLDS}}", worlds_html)
                .replace("{{FRAMES}}", frames_html)
                .replace("{{RENDERS}}", renders_html)
                .replace("{{SECTIONS}}", "".join(rows))
                .replace("{{PINS}}", esc(pin_list))
                .replace("{{DATA}}", json.dumps(data, separators=(",", ":"))))
    (HERE / "index.html").write_text(page)
    print("wrote", HERE / "index.html", f"{len(page) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
