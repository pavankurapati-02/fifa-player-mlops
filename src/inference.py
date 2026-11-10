import joblib
import pandas as pd

from fifa_churn.modeling import (
    MODEL_PATH,
    PROCESSED_DIR,
    predict_probabilities,
)


def run_inference() -> None:
    model = joblib.load(MODEL_PATH)
    features = pd.read_csv(PROCESSED_DIR / "X_test.csv").head(5)
    probabilities = predict_probabilities(model, features)
    for row_number, probability in enumerate(probabilities, start=1):
        print(
            f"Player {row_number}: proxy={int(probability >= 0.5)}, "
            f"risk_probability={probability:.4f}"
        )
    print("[WARNING] Predictions are for a synthetic proxy, not real churn.")


if __name__ == "__main__":
    run_inference()
