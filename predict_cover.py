from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from audio_features import extract_cover_features


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict hit-likeness for a local audio file.")
    parser.add_argument("--audio", required=True, help="Path to the cover audio file.")
    parser.add_argument("--model", default="hit_model.joblib", help="Path to the trained model artifact.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path. Defaults to predictions/<audio-name>.json",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    audio_path = Path(args.audio)
    artifact = joblib.load(args.model)
    feature_names = artifact["feature_names"]
    model = artifact["model"]
    popularity_threshold = artifact["threshold"]

    extracted = extract_cover_features(audio_path)
    row = pd.DataFrame([{name: extracted.get(name) for name in feature_names}])

    probability = float(model.predict_proba(row)[0, 1])
    predicted_label = int(probability >= 0.5)

    result = {
        "audio_file": str(audio_path),
        "predicted_hit_probability": probability,
        "predicted_label": predicted_label,
        "training_popularity_threshold": popularity_threshold,
        "model_features": {name: extracted.get(name) for name in feature_names},
        "analysis_features": {key: value for key, value in extracted.items() if key not in feature_names},
    }

    output_path = Path(args.output) if args.output else Path("predictions") / f"{audio_path.stem}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"Saved prediction to {output_path}")


if __name__ == "__main__":
    main()
