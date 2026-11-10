import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from fifa_churn.modeling import (
    METRICS_PATH,
    MODEL_PATH,
    ROOT,
    TARGET_DEFINITION,
)

TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
)
EXPERIMENT_NAME = "FIFA_Player_Churn_Proxy"
REGISTERED_MODEL_NAME = "FIFA_Player_Churn_Proxy_Model"


def track_experiment(register: bool = False) -> None:
    model = joblib.load(MODEL_PATH)
    metrics_frame = pd.read_csv(METRICS_PATH)
    metrics = dict(zip(metrics_frame["metric"], metrics_frame["value"]))
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(
        run_name="random_forest_churn_proxy"
    ) as run:
        mlflow.log_params(
            {
                "model": "RandomForestClassifier",
                "age_threshold": 32,
                "random_state": 42,
                "synthetic_target": True,
            }
        )
        mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
        mlflow.set_tag("target_definition", TARGET_DEFINITION)
        mlflow.log_artifact(str(METRICS_PATH))
        mlflow.sklearn.log_model(
            sk_model=model,
            name="churn_proxy_model",
            registered_model_name=(
                REGISTERED_MODEL_NAME if register else None
            ),
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.tree._tree.Tree",
            ],
        )
        print(f"[SUCCESS] MLflow run recorded: {run.info.run_id}")
        if register:
            print(f"[SUCCESS] Registered model: {REGISTERED_MODEL_NAME}")


if __name__ == "__main__":
    track_experiment()
