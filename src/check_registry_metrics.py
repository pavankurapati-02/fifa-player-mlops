from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

from fifa_churn.experiment import REGISTERED_MODEL_NAME, TRACKING_URI

ROOT = Path(__file__).resolve().parents[1]

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

versions = client.search_model_versions(
    f"name='{REGISTERED_MODEL_NAME}'"
)

for version in versions:
    print("\n------------------------------")
    print(f"Model Version: {version.version}")
    print(f"Run ID: {version.run_id}")

    run = client.get_run(version.run_id)

    print("\nStored Metrics:")
    print(run.data.metrics)

    print("\nStored Parameters:")
    print(run.data.params)