
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def execute_preprocessing():
    print("[INFO] =========================================")
    print("[INFO] Starting Production Preprocessing Pipeline")
    print("[INFO] =========================================")

    script = ROOT / "src" / "preprocess.py"

    print("\n[INFO] ---> Executing src/preprocess.py...")

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT
    )

    if result.returncode != 0:
        print("[ERROR] Preprocessing pipeline failed.")
        sys.exit(result.returncode)

    print("\n[INFO] Checking processed output files...")

    output_files = [
        "X_train.csv",
        "X_test.csv",
        "y_train.csv",
        "y_test.csv"
    ]

    processed_dir = ROOT / "data" / "processed"

    for filename in output_files:
        file_path = processed_dir / filename

        if not file_path.exists():
            print(f"[ERROR] Missing output file: {filename}")
            sys.exit(1)

        print(f"[SUCCESS] Verified: {filename}")

    print("\n[SUCCESS] Production preprocessing completed!")


if __name__ == "__main__":
    execute_preprocessing()