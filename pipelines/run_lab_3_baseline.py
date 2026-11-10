
import subprocess
import sys
from pathlib import Path

# Get the project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Define the scripts to execute
scripts = [
    ("Data Preprocessing", "src/preprocess.py"),
    ("Model Training", "src/train_model.py"),
    ("Model Evaluation", "src/evaluate.py")
]

print("=" * 50)
print("FIFA PLAYER BASELINE ML PIPELINE")
print("=" * 50)

# Execute each step in sequence
for step_name, script_path in scripts:

    print(f"\nStarting: {step_name}")
    print("-" * 50)

    result = subprocess.run(
        [sys.executable, str(BASE_DIR / script_path)],
        cwd=BASE_DIR
    )

    # Stop the pipeline if any step fails
    if result.returncode != 0:
        print(f"\nERROR: {step_name} failed!")
        sys.exit(result.returncode)

    print(f"\nSUCCESS: {step_name} completed!")

print("\n" + "=" * 50)
print("BASELINE PIPELINE EXECUTED SUCCESSFULLY!")
print("=" * 50)