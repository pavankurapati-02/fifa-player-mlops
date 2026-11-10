
import pandas as pd

from fifa_churn.modeling import (
    MODEL_PATH,
    PROCESSED_DIR,
    build_model,
    calculate_metrics,
    save_model,
)


def train_model() -> None:
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze("columns")

    model = build_model()
    model.fit(X_train, y_train)
    save_model(model)

    metrics = calculate_metrics(model, X_test, y_test)
    print(f"[SUCCESS] Model saved to {MODEL_PATH}")
    print("[INFO] Held-out test metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    train_model()