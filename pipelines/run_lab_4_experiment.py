
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def execute_pipeline():
    print("[INFO] =========================================")
    print("[INFO] Starting Lab 4: MLflow Experiment Tracking")
    print("[INFO] =========================================")

    scripts = [
        "src/preprocess.py",
        "src/track_experiment.py"
    ]

    for script in scripts:
        print(f"\n[INFO] ---> Executing {script}...")

        result = subprocess.run(
            [sys.executable, str(ROOT / script)],
            cwd=ROOT
        )

        if result.returncode != 0:
            print(f"[ERROR] Pipeline halted: {script} failed.")
            sys.exit(result.returncode)

    print("\n[SUCCESS] Lab 4 Pipeline fully executed!")

if __name__ == "__main__":
    execute_pipeline()