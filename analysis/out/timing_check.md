# Word timing check against the vocal stem

Vocal stem: Demucs htdemucs. Threshold 0.3s. Report only; nothing changed.

| time | line | word | issue | now | suggested | shift | note |
|---|---|---|---|---|---|---|---|
| 0:43.70 | 7 | She | starts early | 43.70 | 44.02 | +0.32 | silent at timestamp |
| 0:55.20 | 11 | But | starts early | 55.20 | 56.02 | +0.82 | silent at timestamp |
| 1:23.09 | 18 | ever. | starts late | 83.09 | 82.58 | -0.51 | silence 0.92s before |
| 1:37.75 | 23 | Forever | starts late | 97.75 | 97.30 | -0.45 | silence 0.20s before |
| 1:42.49 | 23 | forever | starts early | 102.49 | 103.12 | +0.63 | silent at timestamp |
| 1:56.47 | 26 | The | starts early | 116.47 | 117.14 | +0.67 | silent at timestamp |
| 3:03.43 | 44 | "I | starts early | 183.43 | 184.02 | +0.59 | silent at timestamp |
| 3:07.09 | 44 | trusting | starts early | 187.09 | 187.46 | +0.37 | silent at timestamp |
| 3:10.63 | 45 | Even | starts early | 190.63 | 191.08 | +0.45 | silent at timestamp |
| 3:28.42 | 47 | But | starts late | 208.42 | 204.44 | -3.98 | silence 8.18s before |
| 3:44.09 | 51 | This | starts late | 224.09 | 223.66 | -0.43 | silence 1.14s before |
| 3:51.05 | 52 | Scribble | starts late | 231.05 | 229.54 | -1.51 | silence 0.40s before |
| 3:53.65 | 52 | stab | starts late | 233.65 | 233.34 | -0.31 | silence 0.20s before |
| 4:12.73 | 58 | And | starts early | 252.73 | 253.44 | +0.71 | silent at timestamp |
| 4:14.81 | 58 | and | starts early | 254.81 | 255.12 | +0.31 | silent at timestamp |
| 4:26.73 | 62 | he | starts early | 266.73 | 267.12 | +0.39 | silent at timestamp |
| 4:30.31 | 64 | he | starts early | 270.31 | 271.36 | +1.05 | silent at timestamp |
| 4:33.81 | 66 | "I | starts early | 273.81 | 274.54 | +0.73 | silent at timestamp |
| 4:35.99 | 67 | Together, | starts early | 275.99 | 277.08 | +1.09 | silent at timestamp |
| 5:18.00 | 78 | But… [pinned] | starts late | 318.00 | 317.66 | -0.34 | silence 4.28s before |
| 5:32.45 | 83 | And | starts early | 332.45 | 332.88 | +0.43 | silent at timestamp |
| 5:59.15 | 92 | something | starts late | 359.15 | 358.70 | -0.45 | silence 0.60s before |
| 6:03.99 | 94 | They | starts late | 363.99 | 362.56 | -1.43 | silence 0.52s before |
| 6:03.99 | 94 | They | starts early | 363.99 | 364.64 | +0.65 | silent at timestamp |
| 6:36.23 | 105 | her... | starts late | 396.23 | 395.90 | -0.33 | silence 0.36s before |
| 6:59.07 | 113 | eyes | starts late | 419.07 | 418.70 | -0.37 | silence 0.94s before |
| 7:28.05 | 121 | Like | starts early | 448.05 | 448.38 | +0.33 | silent at timestamp |

## Reviewed by hand (sections built so far), with per-0.1 s level and pitch

- **line 11 "But"** 55.20 → **56.0**. There's dead silence from 54.6 to 55.9, so the text currently shows 0.8 s before she sings it. "There comes a" follows quickly, and "once" is held on D#4 from about 56.5 to 57.4. The echo "once..." is interpolated with zero length at 58.68. The only candidate is breathy, unpitched activity between 57.5 and 58.6 (uncertain).
- **line 18 "ever."** 83.09 → **82.6**. "and" is held on F#4 until about 81.6, then there's silence from 81.7 to 82.5 and "ever" enters on a scoop at 82.6. The text is currently 0.5 s late.
- **line 22 backing "again, again"** 96.23 / 96.79 → **96.4 / 97.3**. Each lands on a kick pulse (96.50, 97.37). The rewind snaps and the future flickers are keyed to these, so the second one currently fires 0.5 s early, during the first "again".
- **line 23 first "Forever"**: flagged, but a false positive. The 97.3 onset is the second backing "again", and the lead "Forever" starts about 97.85 (it's at 97.75 now, which is fine).
- **line 23 second "forever"** 102.49 → **103.1**. The track drops to silence from 100.9 to 103.0, so the text currently shows 0.6 s early, in the silence. "again" is at 103.5 (C#4), which mirrors 98.3 in the first half.
- **line 7 "She"** 43.70 → 43.9. The "sh" starts at 43.9. That's only 0.2 s, so leave it.
- **line 26 "The"** (White Coats I) 116.47 → **117.1**. It sits in silence until 117.1. The Break I / White Coats I boundary is taken from this timestamp.

Later rows (from line 44 on) aren't reviewed yet. Most "starts early" rows are the first word after a pause: Whisper stretches it back into the silence. They'll be checked by ear as each section comes up.
