
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def execute_pipeline():
    print("[INFO] =========================================")
    print("[INFO] Starting Lab 5: Production Data Pipeline")
    print("[INFO] =========================================")

    scripts = [
        "src/validate_data.py",
        "src/preprocess_pipeline.py",
        "src/validate_outputs.py"
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

    print("\n[SUCCESS] Lab 5 Production Pipeline fully executed!")

if __name__ == "__main__":
    execute_pipeline()