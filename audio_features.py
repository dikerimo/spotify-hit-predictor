from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(max(low, min(high, value)))


def extract_cover_features(audio_path: str | Path) -> dict[str, float]:
    path = Path(audio_path)
    y, sr = librosa.load(path, sr=22050, mono=True)

    if y.size == 0:
        raise ValueError(f"No audio samples found in {path}")

    duration_seconds = librosa.get_duration(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    if beat_times.size > 1:
        beat_gaps = np.diff(beat_times)
        beat_regularity = 1.0 / (1.0 + float(np.std(beat_gaps)))
    else:
        beat_regularity = 0.0

    harmonic, percussive = librosa.effects.hpss(y)
    harmonic_energy = float(np.mean(np.abs(harmonic)))
    percussive_energy = float(np.mean(np.abs(percussive)))
    total_hp = harmonic_energy + percussive_energy + 1e-8
    percussive_ratio = percussive_energy / total_hp
    harmonic_ratio = harmonic_energy / total_hp

    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))
    zcr_mean = float(np.mean(zcr))
    centroid_mean = float(np.mean(centroid))
    bandwidth_mean = float(np.mean(bandwidth))
    rolloff_mean = float(np.mean(rolloff))
    flatness_mean = float(np.mean(flatness))
    chroma_strength = float(np.mean(np.max(chroma, axis=0)))

    loudness_db = float(20 * np.log10(rms_mean + 1e-8))

    energy = _clamp(rms_mean * 6.0)
    danceability = _clamp(
        0.45 * _clamp((float(tempo) - 70.0) / 90.0)
        + 0.35 * _clamp(beat_regularity)
        + 0.20 * _clamp(percussive_ratio)
    )
    valence = _clamp(
        0.45 * _clamp((centroid_mean - 800.0) / 2200.0)
        + 0.35 * chroma_strength
        + 0.20 * _clamp((float(tempo) - 60.0) / 120.0)
    )
    acousticness = _clamp(
        0.50 * harmonic_ratio
        + 0.30 * _clamp(1.0 - percussive_ratio)
        + 0.20 * _clamp(1.0 - flatness_mean * 8.0)
    )
    instrumentalness = _clamp(0.60 * harmonic_ratio + 0.40 * _clamp(1.0 - zcr_mean * 10.0))
    speechiness = _clamp(0.55 * _clamp(zcr_mean * 12.0) + 0.45 * _clamp(flatness_mean * 10.0))
    liveness = _clamp(0.60 * rms_std * 8.0 + 0.40 * _clamp(rolloff_mean / 8000.0))

    return {
        "danceability": danceability,
        "energy": energy,
        "valence": valence,
        "acousticness": acousticness,
        "instrumentalness": instrumentalness,
        "speechiness": speechiness,
        "liveness": liveness,
        "tempo": float(tempo),
        "loudness": loudness_db,
        "duration_ms": duration_seconds * 1000.0,
        "analysis_rms_mean": rms_mean,
        "analysis_rms_std": rms_std,
        "analysis_zcr_mean": zcr_mean,
        "analysis_centroid_mean": centroid_mean,
        "analysis_bandwidth_mean": bandwidth_mean,
        "analysis_rolloff_mean": rolloff_mean,
        "analysis_flatness_mean": flatness_mean,
        "analysis_percussive_ratio": percussive_ratio,
        "analysis_harmonic_ratio": harmonic_ratio,
        "analysis_chroma_strength": chroma_strength,
        "analysis_beat_regularity": beat_regularity,
    }
