
import joblib
import pandas as pd

from fifa_churn.modeling import (
    METRICS_PATH,
    MODEL_PATH,
    PROCESSED_DIR,
    calculate_metrics,
)


def evaluate_model() -> dict[str, float]:
    model = joblib.load(MODEL_PATH)
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze("columns")
    metrics = calculate_metrics(model, X_test, y_test)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"metric": key, "value": value} for key, value in metrics.items()]
    ).to_csv(METRICS_PATH, index=False)

    print(f"[SUCCESS] Evaluation saved to {METRICS_PATH}")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    return metrics


if __name__ == "__main__":
    evaluate_model()