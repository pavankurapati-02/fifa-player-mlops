
from sklearn.model_selection import train_test_split

from fifa_churn.modeling import (
    AGE_THRESHOLD,
    PROCESSED_DIR,
    TARGET_COLUMN,
    build_training_data,
    load_raw_data,
    save_target_definition,
)


def preprocess() -> None:
    frame = load_raw_data().drop_duplicates().reset_index(drop=True)
    features, labels = build_training_data(frame)
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_frame(TARGET_COLUMN).to_csv(
        PROCESSED_DIR / "y_train.csv", index=False
    )
    y_test.to_frame(TARGET_COLUMN).to_csv(
        PROCESSED_DIR / "y_test.csv", index=False
    )
    save_target_definition(len(frame), int(labels.sum()))

    print(f"Deduplicated FIFA rows: {len(frame)}")
    print(f"Proxy target rule: age >= {AGE_THRESHOLD}")
    print(f"Proxy-positive rows: {int(labels.sum())} / {len(labels)}")
    print(f"Training features: {X_train.shape}; test features: {X_test.shape}")
    print(f"Processed data saved to {PROCESSED_DIR}")


if __name__ == "__main__":
    preprocess()