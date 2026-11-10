from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from fifa_churn.modeling import (
    FEATURE_COLUMNS,
    MODEL_PATH,
    RAW_DATA_PATH,
    predict_probabilities,
    validate_raw_data,
)

PSI_ALERT_THRESHOLD = 0.2
PREDICTION_RATE_ALERT_THRESHOLD = 0.1


def _population_stability_index(
    reference: pd.Series, current: pd.Series
) -> float:
    reference_values = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    current_values = pd.to_numeric(current, errors="coerce").dropna().to_numpy()
    if not len(reference_values) or not len(current_values):
        return 0.0

    edges = np.unique(
        np.quantile(reference_values, np.linspace(0.0, 1.0, 11))
    )
    if len(edges) < 2:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    expected = np.histogram(reference_values, bins=edges)[0].astype(float)
    observed = np.histogram(current_values, bins=edges)[0].astype(float)
    expected /= expected.sum()
    observed /= observed.sum()
    epsilon = 1e-6
    expected = np.clip(expected, epsilon, None)
    observed = np.clip(observed, epsilon, None)
    return float(np.sum((observed - expected) * np.log(observed / expected)))


def build_monitor_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    model_path: Path = MODEL_PATH,
) -> dict[str, Any]:
    validate_raw_data(reference)
    validate_raw_data(current)
    missing_features = sorted(
        set(FEATURE_COLUMNS) - set(reference.columns)
        | (set(FEATURE_COLUMNS) - set(current.columns))
    )
    if missing_features:
        raise ValueError(f"Monitoring data is missing model features: {missing_features}")
    reference_features = reference.loc[:, FEATURE_COLUMNS].copy()
    current_features = current.loc[:, FEATURE_COLUMNS].copy()
    feature_psi = {
        feature: _population_stability_index(
            reference_features[feature], current_features[feature]
        )
        for feature in FEATURE_COLUMNS
        if feature in reference_features
        and pd.api.types.is_numeric_dtype(reference_features[feature])
    }

    model = joblib.load(model_path)
    reference_probabilities = predict_probabilities(model, reference_features)
    current_probabilities = predict_probabilities(model, current_features)
    reference_high_risk_rate = float(
        (np.asarray(reference_probabilities) >= 0.5).mean()
    )
    current_high_risk_rate = float(
        (np.asarray(current_probabilities) >= 0.5).mean()
    )
    prediction_rate_change = abs(
        current_high_risk_rate - reference_high_risk_rate
    )
    drift_features = [
        name
        for name, value in feature_psi.items()
        if value >= PSI_ALERT_THRESHOLD
    ]

    return {
        "reference_rows": len(reference_features),
        "current_rows": len(current_features),
        "feature_psi": feature_psi,
        "psi_alert_threshold": PSI_ALERT_THRESHOLD,
        "drift_features": drift_features,
        "reference_high_risk_rate": reference_high_risk_rate,
        "current_high_risk_rate": current_high_risk_rate,
        "prediction_rate_change": prediction_rate_change,
        "prediction_rate_alert_threshold": PREDICTION_RATE_ALERT_THRESHOLD,
        "drift_detected": bool(
            drift_features
            or prediction_rate_change >= PREDICTION_RATE_ALERT_THRESHOLD
        ),
        "warning": "Monitoring uses a synthetic age-based target, not observed churn.",
    }


def monitor(current_path: Path | None = None) -> dict[str, Any]:
    reference = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    current_file = current_path or RAW_DATA_PATH
    current = pd.read_csv(current_file, low_memory=False)
    return build_monitor_report(reference, current)
