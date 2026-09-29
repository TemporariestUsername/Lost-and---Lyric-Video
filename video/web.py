"""Web delivery: the master as a 720p HLS stream for the storyboard page.

Artifact pages cap each file at 15 MB, so the film is cut into ~6 s fMP4
segments (a few MB each) behind one playlist; the page plays them with
hls.js (or natively in Safari).

    python3 video/web.py render/lost_and_full.mp4
"""
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "storyboard" / "renders" / "film"


def main(master):
    master = pathlib.Path(master)
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(master),
        "-vf", "scale=1280:720:flags=lanczos,hqdn3d=1.2:1.2:3:3",
        "-c:v", "libx264", "-profile:v", "main", "-level", "3.1", "-preset", "slow",
        "-b:v", "2800k", "-maxrate", "3600k", "-bufsize", "7200k",
        "-g", "180", "-keyint_min", "180", "-sc_threshold", "0", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-ac", "2",
        "-f", "hls", "-hls_time", "6", "-hls_playlist_type", "vod",
        "-hls_segment_type", "fmp4", "-hls_fmp4_init_filename", "init.mp4",
        "-hls_segment_filename", str(OUT / "seg_%03d.m4s"),
        str(OUT / "index.m3u8")], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "46.3", "-i", str(master),
                    "-frames:v", "1", "-vf", "scale=1280:720", "-q:v", "3",
                    str(OUT.parent / "film_poster.jpg")], check=True)
    files = sorted(OUT.iterdir())
    big = max(f.stat().st_size for f in files)
    total = sum(f.stat().st_size for f in files)
    print(f"{len(files)} files, {total / 1e6:.1f} MB total, largest {big / 1e6:.1f} MB")


if __name__ == "__main__":
    main(sys.argv[1])
