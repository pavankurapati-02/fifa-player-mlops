import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = [
    ("Raw data validation", "src/validate_data.py"),
    ("Preprocessing and split", "src/preprocess.py"),
    ("Processed output validation", "src/validate_outputs.py"),
    ("Training and test evaluation", "src/train_model.py"),
    ("Evaluation report", "src/evaluate.py"),
    ("Experiment tracking", "src/track_experiment.py"),
    ("Quality gate", "src/quality_gate.py"),
    ("Model registry", "src/train_registry.py"),
    ("Model lifecycle selection", "src/automate_lifecycle.py"),
    ("Batch inference", "src/inference.py"),
    ("Feature and prediction monitoring", "src/monitor.py"),
]


def main() -> None:
    for name, script in STAGES:
        print(f"\n===== {name} =====", flush=True)
        subprocess.run(
            [sys.executable, str(ROOT / script)],
            cwd=ROOT,
            check=True,
        )
    print("\n[SUCCESS] FIFA MLOps stages completed through Lab 13.")
    print("[WARNING] Churn metrics describe a synthetic age-based proxy only.")


if __name__ == "__main__":
    main()
