import mlflow
from mlflow.tracking import MlflowClient

from fifa_churn.experiment import REGISTERED_MODEL_NAME, TRACKING_URI


def automate_lifecycle() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    versions = client.search_model_versions(
        f"name='{REGISTERED_MODEL_NAME}'"
    )
    candidates = []
    for version in versions:
        run = client.get_run(version.run_id)
        recall = run.data.metrics.get("recall")
        if recall is not None:
            candidates.append((float(recall), version.version))
    if not candidates:
        raise RuntimeError(
            f"No registered {REGISTERED_MODEL_NAME} versions have recall metrics."
        )

    recall, version = max(candidates)
    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME, "champion", version
    )
    print(
        f"[SUCCESS] Champion set to version {version} "
        f"(synthetic-proxy recall={recall:.4f})."
    )


if __name__ == "__main__":
    automate_lifecycle()
