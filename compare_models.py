from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


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
    parser = argparse.ArgumentParser(description="Compare multiple hit-prediction models on one dataset.")
    parser.add_argument("--data", required=True, help="Path to the CSV file.")
    parser.add_argument("--threshold", type=float, default=65.0, help="Popularity threshold for hits.")
    parser.add_argument(
        "--output",
        default="models/model_comparison.json",
        help="Where to save the comparison report.",
    )
    return parser


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [column.strip().lower().replace(" ", "_") for column in df.columns]
    return df


def resolve_target(df: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.Series]:
    if "popularity" not in df.columns:
        raise ValueError("Expected a 'popularity' column in the dataset.")

    popularity = pd.to_numeric(df["popularity"], errors="coerce")
    valid_mask = popularity.notna()
    filtered = df.loc[valid_mask].copy()
    y = (popularity.loc[valid_mask] >= threshold).astype(int)
    return filtered, y


def build_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=6,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=10,
                        min_samples_leaf=4,
                        random_state=42,
                        class_weight="balanced_subsample",
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=300,
                        max_depth=8,
                        min_samples_leaf=20,
                        learning_rate=0.05,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def evaluate_model(name: str, pipeline: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> dict:
    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    report = classification_report(y_test, predictions, output_dict=True)

    return {
        "roc_auc": roc_auc_score(y_test, probabilities),
        "hit_precision": report["1"]["precision"],
        "hit_recall": report["1"]["recall"],
        "hit_f1": report["1"]["f1-score"],
        "hit_support": report["1"]["support"],
        "accuracy": report["accuracy"],
        "classification_report": report,
    }


def main() -> None:
    args = build_parser().parse_args()

    df = pd.read_csv(args.data)
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

    results = {
        "threshold": args.threshold,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "models": {},
    }

    for name, pipeline in build_models().items():
        print(f"Training {name}...")
        results["models"][name] = evaluate_model(name, pipeline, X_train, X_test, y_train, y_test)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    summary = []
    for name, metrics in results["models"].items():
        summary.append(
            {
                "model": name,
                "roc_auc": round(metrics["roc_auc"], 4),
                "hit_f1": round(metrics["hit_f1"], 4),
                "hit_precision": round(metrics["hit_precision"], 4),
                "hit_recall": round(metrics["hit_recall"], 4),
                "accuracy": round(metrics["accuracy"], 4),
            }
        )

    print(json.dumps(summary, indent=2))
    print(f"Saved comparison to {output_path}")


if __name__ == "__main__":
    main()
