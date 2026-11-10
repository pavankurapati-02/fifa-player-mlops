
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

def validate_outputs():
    print("[INFO] =========================================")
    print("[INFO] Starting Output Data Validation")
    print("[INFO] =========================================")

    required_files = [
        "X_train.csv",
        "X_test.csv",
        "y_train.csv",
        "y_test.csv"
    ]

    datasets = {}

    for filename in required_files:
        file_path = PROCESSED_DIR / filename

        if not file_path.exists():
            print(f"[ERROR] Missing output file: {filename}")
            sys.exit(1)

        try:
            df = pd.read_csv(file_path)
            datasets[filename] = df
            print(f"[SUCCESS] Loaded {filename}: {df.shape}")

        except Exception as e:
            print(f"[ERROR] Failed to read {filename}: {e}")
            sys.exit(1)

    # Check for empty datasets
    for filename, df in datasets.items():
        if df.empty:
            print(f"[ERROR] {filename} is empty!")
            sys.exit(1)

    # Check training and testing feature dimensions
    X_train = datasets["X_train.csv"]
    X_test = datasets["X_test.csv"]

    if X_train.shape[1] != X_test.shape[1]:
        print("[ERROR] Training and testing feature dimensions do not match!")
        sys.exit(1)

    # Check matching feature names
    if list(X_train.columns) != list(X_test.columns):
        print("[ERROR] Training and testing feature columns do not match!")
        sys.exit(1)

    # Check matching feature and target row counts
    if len(X_train) != len(datasets["y_train.csv"]):
        print("[ERROR] Training features and target row counts do not match!")
        sys.exit(1)

    if len(X_test) != len(datasets["y_test.csv"]):
        print("[ERROR] Testing features and target row counts do not match!")
        sys.exit(1)

    print("\n[INFO] OUTPUT VALIDATION SUMMARY")
    print("=" * 40)
    print(f"Training features: {X_train.shape}")
    print(f"Testing features: {X_test.shape}")
    print(f"Training targets: {datasets['y_train.csv'].shape}")
    print(f"Testing targets: {datasets['y_test.csv'].shape}")

    print("\n[SUCCESS] All output validation checks passed!")

if __name__ == "__main__":
    validate_outputs()