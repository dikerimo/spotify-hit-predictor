from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


MODEL_FEATURES = [
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "instrumentalness",
    "speechiness",
    "liveness",
    "tempo",
    "loudness",
    "duration_ms",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a hit-likeness model from a Spotify-style CSV.")
    parser.add_argument("--data", required=True, help="Path to the Kaggle CSV file.")
    parser.add_argument("--output", default="hit_model.joblib", help="Where to save the trained model.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=70.0,
        help="Popularity threshold used to label tracks as hits.",
    )
    return parser


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [column.strip().lower().replace(" ", "_") for column in df.columns]
    return df


def resolve_target(df: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.Series]:
    if "popularity" not in df.columns:
        raise ValueError("Expected a 'popularity' column in the training dataset.")

    y = (pd.to_numeric(df["popularity"], errors="coerce") >= threshold).astype("Int64")
    valid_mask = y.notna()
    return df.loc[valid_mask].copy(), y.loc[valid_mask].astype(int)


def main() -> None:
    args = build_parser().parse_args()

    csv_path = Path(args.data)
    output_path = Path(args.output)

    df = pd.read_csv(csv_path)
    df = normalize_columns(df)

    missing_columns = [column for column in MODEL_FEATURES if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required feature columns: {missing_columns}")

    df, y = resolve_target(df, args.threshold)
    X = df[MODEL_FEATURES].apply(pd.to_numeric, errors="coerce")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=6,
                    min_samples_split=6,
                    max_features="sqrt",
                    random_state=42,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    auc = roc_auc_score(y_test, probabilities)
    report = classification_report(y_test, predictions, output_dict=True)

    artifact = {
        "model": pipeline,
        "feature_names": MODEL_FEATURES,
        "threshold": args.threshold,
        "metrics": {
            "roc_auc": auc,
            "classification_report": report,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)

    metrics_path = output_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(artifact["metrics"], indent=2), encoding="utf-8")

    print(f"Saved model to {output_path}")
    print(f"Saved metrics to {metrics_path}")
    print(json.dumps(artifact["metrics"], indent=2))


if __name__ == "__main__":
    main()
