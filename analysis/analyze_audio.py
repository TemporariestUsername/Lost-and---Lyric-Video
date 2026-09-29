"""Musical analysis: tempo, beats, downbeats, loudness, sections.

Writes analysis/out/audio.json. Run from repo root:
    python3 analysis/analyze_audio.py
"""
import json
import pathlib

import librosa
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
AUDIO = ROOT / "assets" / "lost_and.mp3"
OUT = ROOT / "analysis" / "out"
SR = 22050
HOP = 512


def estimate_downbeats(y, sr, beat_frames, beats_per_bar=4):
    """Pick the bar phase whose beats carry the most low-end onset energy.

    librosa has no downbeat tracker; for four-on-the-floor-ish synthpop the
    kick-heavy beat of each bar is a reliable proxy.
    """
    y_low = librosa.effects.preemphasis(y, coef=-0.97)  # tilt toward lows
    S = np.abs(librosa.stft(y_low, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=sr)
    low = S[freqs < 150].sum(axis=0)
    low_onset = np.maximum(0, np.diff(low, prepend=low[0]))
    per_beat = low_onset[np.clip(beat_frames, 0, len(low_onset) - 1)]
    scores = [per_beat[p::beats_per_bar].mean() for p in range(beats_per_bar)]
    phase = int(np.argmax(scores))
    return beat_frames[phase::beats_per_bar], phase, [float(s) for s in scores]


def smooth_beats(beat_times, half_window=8):
    """Remove hop-size quantisation jitter with a sliding local linear fit.

    The track drifts ~136 -> 142 BPM, so a single global grid doesn't fit;
    a 17-beat local fit follows the drift while giving sub-frame accuracy.
    """
    b = np.asarray(beat_times)
    idx = np.arange(len(b))
    out = np.empty_like(b)
    for i in idx:
        lo, hi = max(0, i - half_window), min(len(b), i + half_window + 1)
        k, c = np.polyfit(idx[lo:hi], b[lo:hi], 1)
        out[i] = k * i + c
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    y, sr = librosa.load(AUDIO, sr=SR, mono=True)
    duration = len(y) / sr

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=HOP, trim=False
    )
    tempo = float(np.atleast_1d(tempo)[0])
    beat_times = smooth_beats(
        librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP))

    # Local tempo curve, to spot tempo changes / rubato sections.
    dtempo = librosa.feature.tempo(
        onset_envelope=onset_env, sr=sr, hop_length=HOP, aggregate=None
    )

    down_frames, phase, phase_scores = estimate_downbeats(y, sr, beat_frames)
    down_times = beat_times[phase::4]

    # Loudness: RMS in dBFS at ~10 Hz, plus a smoothed 1 s envelope.
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=1.0)
    t_rms = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=HOP)
    step = int(round(sr / HOP / 10))
    win = int(round(sr / HOP))
    smooth = np.convolve(rms_db, np.ones(win) / win, mode="same")

    # Onsets (for hit-synced typography) and per-beat strength.
    onsets = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=HOP, units="time"
    )
    beat_strength = onset_env[np.clip(beat_frames, 0, len(onset_env) - 1)]
    beat_strength = beat_strength / (beat_strength.max() + 1e-9)

    # Section boundaries: agglomerative clustering on beat-synced chroma+MFCC.
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=HOP, n_mfcc=13)
    feat = np.vstack([librosa.util.normalize(chroma, axis=1),
                      librosa.util.normalize(mfcc, axis=1)])
    feat_sync = librosa.util.sync(feat, beat_frames, aggregate=np.median)
    n_sections = max(8, int(duration // 25))
    bounds = librosa.segment.agglomerative(feat_sync, n_sections)
    bound_frames = librosa.util.fix_frames(
        np.asarray(beat_frames)[np.clip(bounds, 0, len(beat_frames) - 1)],
        x_min=0,
    )
    bound_times = librosa.frames_to_time(bound_frames, sr=sr, hop_length=HOP)

    # Spectral centroid (brightness) at 10 Hz, useful for colour grading.
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP)[0]

    data = {
        "file": str(AUDIO.relative_to(ROOT)),
        "duration": duration,
        "tempo_bpm": tempo,
        "downbeat_phase": phase,
        "downbeat_phase_scores": phase_scores,
        "beats": [round(float(t), 4) for t in beat_times],
        "beat_strength": [round(float(s), 4) for s in beat_strength],
        "downbeats": [round(float(t), 4) for t in down_times],
        "onsets": [round(float(t), 4) for t in onsets],
        "sections": [round(float(t), 3) for t in sorted(set(bound_times))],
        "curves_hz": 10,
        "loudness_db": [round(float(v), 2) for v in rms_db[::step]],
        "loudness_db_smooth": [round(float(v), 2) for v in smooth[::step]],
        "centroid_hz": [round(float(v), 1) for v in cent[::step]],
        "tempo_curve": [round(float(v), 2) for v in dtempo[::step]],
        "curve_times": [round(float(v), 2) for v in t_rms[::step]],
    }
    (OUT / "audio.json").write_text(json.dumps(data))
    print(f"duration {duration:.2f}s  tempo {tempo:.2f} BPM  "
          f"{len(beat_times)} beats  {len(down_times)} bars  "
          f"downbeat phase {phase} {phase_scores}")
    print("sections:", [f"{t:.1f}" for t in data["sections"]])


if __name__ == "__main__":
    main()
