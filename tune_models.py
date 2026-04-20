from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
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
    parser = argparse.ArgumentParser(description="Tune Random Forest and XGBoost for hit prediction.")
    parser.add_argument("--data", required=True, help="Path to the CSV file.")
    parser.add_argument("--threshold", type=float, default=65.0, help="Popularity threshold for hits.")
    parser.add_argument(
        "--mode",
        choices=["quick", "rf_focus"],
        default="rf_focus",
        help="quick = small RF + XGBoost sweep, rf_focus = deeper Random Forest-only tuning around the best region.",
    )
    parser.add_argument(
        "--output",
        default="models/tuning_results.json",
        help="Where to save the tuning results.",
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


def evaluate_pipeline(pipeline: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> dict:
    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    report = classification_report(y_test, predictions, output_dict=True)

    return {
        "roc_auc": roc_auc_score(y_test, probabilities),
        "hit_precision": report["1"]["precision"],
        "hit_recall": report["1"]["recall"],
        "hit_f1": report["1"]["f1-score"],
        "accuracy": report["accuracy"],
    }


def random_forest_candidates() -> list[tuple[str, dict]]:
    candidates = [
        {
            "n_estimators": 300,
            "max_depth": 10,
            "min_samples_leaf": 4,
            "min_samples_split": 2,
        },
        {
            "n_estimators": 400,
            "max_depth": 16,
            "min_samples_leaf": 2,
            "min_samples_split": 2,
        },
        {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_leaf": 8,
            "min_samples_split": 8,
        },
    ]

    labeled = []
    for params in candidates:
        label = "rf_" + "_".join(f"{key}={value}" for key, value in params.items())
        labeled.append((label, params))
    return labeled


def focused_random_forest_candidates() -> list[tuple[str, dict]]:
    candidates = [
        {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_leaf": 6,
            "min_samples_split": 6,
            "max_features": "sqrt",
        },
        {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_leaf": 8,
            "min_samples_split": 8,
            "max_features": "sqrt",
        },
        {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_leaf": 8,
            "min_samples_split": 8,
            "max_features": "sqrt",
        },
        {
            "n_estimators": 700,
            "max_depth": None,
            "min_samples_leaf": 8,
            "min_samples_split": 8,
            "max_features": "sqrt",
        },
        {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_leaf": 10,
            "min_samples_split": 10,
            "max_features": "sqrt",
        },
        {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_leaf": 12,
            "min_samples_split": 12,
            "max_features": "sqrt",
        },
        {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_leaf": 8,
            "min_samples_split": 8,
            "max_features": "log2",
        },
        {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_leaf": 8,
            "min_samples_split": 10,
            "max_features": "sqrt",
        },
    ]

    labeled = []
    for params in candidates:
        label = "rf_" + "_".join(f"{key}={value}" for key, value in params.items())
        labeled.append((label, params))
    return labeled


def xgboost_candidates() -> list[tuple[str, dict]]:
    candidates = [
        {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        {
            "n_estimators": 400,
            "max_depth": 4,
            "learning_rate": 0.03,
            "subsample": 1.0,
            "colsample_bytree": 0.8,
        },
        {
            "n_estimators": 200,
            "max_depth": 8,
            "learning_rate": 0.08,
            "subsample": 0.8,
            "colsample_bytree": 1.0,
        },
    ]

    labeled = []
    for params in candidates:
        label = "xgb_" + "_".join(f"{key}={value}" for key, value in params.items())
        labeled.append((label, params))
    return labeled


def build_random_forest_pipeline(params: dict) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    random_state=42,
                    class_weight="balanced_subsample",
                    **params,
                ),
            ),
        ]
    )


def build_xgboost_pipeline(params: dict) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                    **params,
                ),
            ),
        ]
    )


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
        "mode": args.mode,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "random_forest": [],
        "xgboost": [],
    }

    rf_candidates = focused_random_forest_candidates() if args.mode == "rf_focus" else random_forest_candidates()

    for label, params in rf_candidates:
        print(f"Training {label}")
        pipeline = build_random_forest_pipeline(params)
        metrics = evaluate_pipeline(pipeline, X_train, X_test, y_train, y_test)
        results["random_forest"].append({"label": label, "params": params, "metrics": metrics})

    if args.mode == "quick":
        for label, params in xgboost_candidates():
            print(f"Training {label}")
            pipeline = build_xgboost_pipeline(params)
            metrics = evaluate_pipeline(pipeline, X_train, X_test, y_train, y_test)
            results["xgboost"].append({"label": label, "params": params, "metrics": metrics})

    results["random_forest"].sort(key=lambda item: item["metrics"]["hit_f1"], reverse=True)
    results["xgboost"].sort(key=lambda item: item["metrics"]["hit_f1"], reverse=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    summary = {
        "best_random_forest": results["random_forest"][:3],
        "best_xgboost": results["xgboost"][:3],
    }
    print(json.dumps(summary, indent=2))
    print(f"Saved tuning results to {output_path}")


if __name__ == "__main__":
    main()
