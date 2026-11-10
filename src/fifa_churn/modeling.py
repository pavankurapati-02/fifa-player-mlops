import json
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = Path(
    os.environ.get("FIFA_DATA_PATH", ROOT / "data" / "raw" / "fifa_players.csv")
)
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_PATH = Path(
    os.environ.get(
        "MODEL_PATH", ROOT / "models" / "churn_proxy_model.joblib"
    )
)
METRICS_PATH = ROOT / "reports" / "model_metrics.csv"
TARGET_DEFINITION_PATH = ROOT / "reports" / "target_definition.json"

TARGET_COLUMN = "churn_proxy"
AGE_THRESHOLD = 32
TARGET_DEFINITION = (
    "Synthetic educational proxy: 1 when age >= 32, otherwise 0. "
    "This is not observed player churn, a transfer label, or a validated risk estimate."
)
NUMERIC_FEATURES = [
    "age",
    "overall",
    "potential",
    "height_cm",
    "weight_kg",
    "weak_foot",
    "skill_moves",
    "international_reputation",
]
CATEGORICAL_FEATURES = [
    "player_positions",
    "preferred_foot",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REQUIRED_COLUMNS = ["age", "overall", "potential"]


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"FIFA dataset not found: {path}")
    frame = pd.read_csv(path, low_memory=False)
    validate_raw_data(frame)
    return frame


def validate_raw_data(frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValueError("FIFA dataset is empty.")
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"FIFA dataset is missing required columns: {missing}")
    if frame["age"].isna().any():
        raise ValueError("FIFA dataset contains missing ages; proxy target cannot be assigned.")
    if not pd.api.types.is_numeric_dtype(frame["age"]):
        raise ValueError("FIFA dataset column 'age' must be numeric.")


def build_training_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    validate_raw_data(frame)
    missing_features = sorted(set(FEATURE_COLUMNS) - set(frame.columns))
    if missing_features:
        raise ValueError(f"FIFA dataset is missing model features: {missing_features}")
    features = frame.loc[:, FEATURE_COLUMNS].copy()
    labels = (frame["age"] >= AGE_THRESHOLD).astype("int8").rename(TARGET_COLUMN)
    if labels.nunique() != 2:
        raise ValueError(
            "The age-based proxy must contain both classes; "
            f"the dataset needs ages on both sides of {AGE_THRESHOLD}."
        )
    return features, labels


def build_model() -> Pipeline:
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline(
        steps=[("preprocessing", preprocessing), ("classifier", classifier)]
    )


def calculate_metrics(
    model: Pipeline, features: pd.DataFrame, labels: pd.Series
) -> dict[str, float]:
    predictions = model.predict(features)
    positive_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(features)[:, positive_index]
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
    }


def save_target_definition(row_count: int, proxy_count: int) -> None:
    TARGET_DEFINITION_PATH.parent.mkdir(parents=True, exist_ok=True)
    TARGET_DEFINITION_PATH.write_text(
        json.dumps(
            {
                "target": TARGET_COLUMN,
                "definition": TARGET_DEFINITION,
                "age_threshold": AGE_THRESHOLD,
                "rows": row_count,
                "proxy_positive_rows": proxy_count,
                "is_observed_churn": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def predict_probabilities(model: Any, features: pd.DataFrame) -> list[float]:
    positive_index = list(model.classes_).index(1)
    return model.predict_proba(features)[:, positive_index].astype(float).tolist()


def save_model(model: Pipeline, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
