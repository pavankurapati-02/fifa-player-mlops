
from fifa_churn.modeling import (
    AGE_THRESHOLD,
    RAW_DATA_PATH,
    build_training_data,
    load_raw_data,
)


def validate_data() -> None:
    frame = load_raw_data()
    _, labels = build_training_data(frame)
    print("[INFO] FIFA source data validated.")
    print(f"[INFO] File: {RAW_DATA_PATH}")
    print(f"[INFO] Shape: {frame.shape}; duplicate rows: {frame.duplicated().sum()}")
    print(
        f"[INFO] Synthetic proxy positives (age >= {AGE_THRESHOLD}): "
        f"{int(labels.sum())}/{len(labels)}"
    )
    print("[WARNING] This dataset has no observed player-churn labels.")


if __name__ == "__main__":
    validate_data()