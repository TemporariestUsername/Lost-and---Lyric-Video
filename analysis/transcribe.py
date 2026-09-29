"""Word-level timing pass with faster-whisper.

Whisper is used only for *timing*; the canonical text is assets/lyrics.txt.

    python3 analysis/transcribe.py                      # full track
    python3 analysis/transcribe.py --clip 120 150 \
        --prompt-lines 26 35 --tag white_coats          # targeted re-run

A clip re-run passes the matching lyric lines as `initial_prompt`, which
steers Whisper into transcribing a section it skipped (spoken word, chant).
"""
import argparse
import json
import pathlib

import librosa
from faster_whisper import WhisperModel

ROOT = pathlib.Path(__file__).resolve().parent.parent
AUDIO = ROOT / "assets" / "lost_and.mp3"
LYRICS = ROOT / "assets" / "lyrics.txt"
OUT = ROOT / "analysis" / "out"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="medium.en")
    ap.add_argument("--clip", nargs=2, type=float, metavar=("START", "END"))
    ap.add_argument("--prompt-lines", nargs=2, type=int, metavar=("FIRST", "LAST"),
                    help="1-based inclusive line range of lyrics.txt")
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    model = WhisperModel(args.model, device="cpu", compute_type="int8")

    offset = 0.0
    audio = str(AUDIO)
    if args.clip:
        offset = args.clip[0]
        audio, _ = librosa.load(AUDIO, sr=16000, mono=True,
                                offset=args.clip[0],
                                duration=args.clip[1] - args.clip[0])

    prompt = None
    if args.prompt_lines:
        lines = LYRICS.read_text().splitlines()
        a, b = args.prompt_lines
        prompt = " ".join(l.strip() for l in lines[a - 1:b] if l.strip())

    segments, info = model.transcribe(
        audio,
        language="en",
        word_timestamps=True,
        vad_filter=False,          # VAD drops whispered/sung phrases over synths
        condition_on_previous_text=bool(prompt),
        initial_prompt=prompt,
        beam_size=5,
    )
    words, segs = [], []
    for s in segments:
        segs.append({"start": s.start + offset, "end": s.end + offset,
                     "text": s.text.strip()})
        for w in s.words or []:
            words.append({"start": round(w.start + offset, 3),
                          "end": round(w.end + offset, 3),
                          "word": w.word.strip(),
                          "p": round(w.probability, 3)})
        print(f"[{s.start + offset:7.2f} -> {s.end + offset:7.2f}] {s.text.strip()}",
              flush=True)

    out = OUT / f"whisper_{args.tag}.json"
    out.write_text(json.dumps({"model": args.model, "clip": args.clip,
                               "prompt": prompt, "segments": segs,
                               "words": words}, indent=1))
    print(f"wrote {out} ({len(words)} words)")


if __name__ == "__main__":
    main()
