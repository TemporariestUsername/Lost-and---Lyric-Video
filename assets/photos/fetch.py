"""Find public-domain / CC0 photographs for the memory layer via Openverse.

Only licenses that allow unrestricted use are requested (cc0, pdm). Every
download is recorded in candidates.json with title, creator, license and
source page, so final picks can be credited in CREDITS.md.

    python3 assets/photos/fetch.py            # all themes
    python3 assets/photos/fetch.py motel road # just these themes
"""
import io
import re
import json
import pathlib
import sys
import time
import urllib.parse
import urllib.request

from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
CAND = HERE / "candidates"
API = "https://api.openverse.org/v1/images/"
UA = "lost-and-lyric-video/1.0 (personal music video; public-domain photo search)"

THEMES = {
    # her childhood / before
    "window": ["window light empty room", "curtain window sunlight", "lace curtain window"],
    "sea": ["beach shore overcast", "seaside horizon fog", "empty beach"],
    "bedroom": ["empty bedroom morning light", "unmade bed sheets"],
    "house": ["suburban house 1970s", "old house exterior"],
    # the institute
    "corridor": ["hospital corridor", "empty hallway fluorescent", "institution hallway"],
    "clinic": ["hospital room empty", "laboratory vintage"],
    # him
    "doorway": ["doorway light silhouette", "open door light dark room"],
    # the run
    "road": ["night highway headlights", "road at night car lights", "empty road night"],
    "tunnel": ["tunnel lights night", "road tunnel"],
    "motel": ["motel sign night", "motel neon", "vintage motel"],
    "diner": ["diner night", "diner interior", "neon diner"],
    "alley": ["alley night rain", "back alley night"],
    "rain": ["rain on window night", "raindrops glass city lights"],
    # contemporary additions (nothing retro)
    "hospital": ["hospital bed empty", "hospital room window", "fluorescent light ceiling"],
    "room": ["empty room window light", "sheer curtain sunlight", "bedroom window morning light"],
    "institute": ["hospital hallway", "fluorescent lights ceiling", "empty waiting room chairs",
                  "office corridor empty", "laboratory glassware", "hospital window blinds",
                  "clean room laboratory", "stairwell concrete"],
    "night": ["parking lot night", "gas station night", "convenience store night",
              "bus window night", "car window night rain", "streetlight fog"],
}

# Contemporary only: skip anything that announces itself as old.
RETRO = re.compile(r"vintage|retro|postcard|antique|circa|\b19[0-7]\d|\b1[0-8]\d\d|1950s|1960s|1970s|"
                   r"old school|historic|sepia", re.I)


def get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def search(q, n=10):
    params = {"q": q, "license": "cc0,pdm", "category": "photograph", "page_size": n,
              "mature": "false"}
    return json.loads(get(API + "?" + urllib.parse.urlencode(params)))["results"]


def main(themes):
    CAND.mkdir(parents=True, exist_ok=True)
    idx_path = HERE / "candidates.json"
    idx = json.loads(idx_path.read_text()) if idx_path.exists() else {}
    for theme in themes:
        for q in THEMES[theme]:
            try:
                results = search(q)
            except Exception as e:  # noqa: BLE001
                print(f"  search failed for '{q}': {e}")
                continue
            for res in results:
                rid = res["id"]
                if rid in idx:
                    continue
                text = " ".join([res.get("title") or ""] + [t.get("name", "") for t in res.get("tags") or []])
                if RETRO.search(text):
                    continue
                w, h = res.get("width") or 0, res.get("height") or 0
                if w and h and w < h:            # landscape prints only
                    continue
                img = None
                for url in (res.get("url"), res.get("thumbnail")):
                    if not url:
                        continue
                    try:
                        data = get(url)
                        img = Image.open(io.BytesIO(data)).convert("RGB")
                        break
                    except Exception:  # noqa: BLE001
                        img = None
                if img is None or img.width < 400:
                    continue
                img.thumbnail((1400, 1400))
                name = f"{theme}_{rid[:8]}.jpg"
                img.save(CAND / name, quality=90)
                idx[rid] = dict(file=name, theme=theme, query=q, title=res.get("title"),
                                creator=res.get("creator"), license=res.get("license"),
                                license_version=res.get("license_version"),
                                license_url=res.get("license_url"),
                                source=res.get("foreign_landing_url"),
                                provider=res.get("provider"), size=[img.width, img.height])
                print(f"  {name}  {res.get('license')}  {(res.get('title') or '')[:50]}")
            idx_path.write_text(json.dumps(idx, indent=1))
            time.sleep(1.0)
    print(f"{len(idx)} candidates in {idx_path}")


if __name__ == "__main__":
    main(sys.argv[1:] or list(THEMES))
