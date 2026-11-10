import pandas as pd

from fifa_churn.modeling import METRICS_PATH

MINIMUM_RECALL = 0.60


def enforce_quality_gate() -> None:
    metrics = pd.read_csv(METRICS_PATH).set_index("metric")["value"]
    recall = float(metrics["recall"])
    print(f"[INFO] Test recall: {recall:.4f}; minimum: {MINIMUM_RECALL:.2f}")
    if recall < MINIMUM_RECALL:
        raise RuntimeError(
            "Model failed the recall quality gate. "
            "The synthetic proxy should not be treated as real churn performance."
        )
    print("[SUCCESS] Model passed the demonstration quality gate.")


if __name__ == "__main__":
    enforce_quality_gate()
