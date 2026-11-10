import pandas as pd
from fastapi.testclient import TestClient

from fifa_churn.api import app
from fifa_churn.modeling import (
    AGE_THRESHOLD,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    build_training_data,
)
from fifa_churn.monitoring import build_monitor_report

client = TestClient(app)


def test_proxy_target_is_explicit_age_rule() -> None:
    frame = pd.DataFrame(
        {
            "age": [AGE_THRESHOLD - 1, AGE_THRESHOLD],
            "overall": [75, 75],
            "potential": [80, 80],
            "height_cm": [175, 175],
            "weight_kg": [70, 70],
            "weak_foot": [3, 3],
            "skill_moves": [3, 3],
            "international_reputation": [1, 1],
            "player_positions": ["CM", "CM"],
            "preferred_foot": ["Right", "Right"],
        }
    )
    _, labels = build_training_data(frame)
    assert labels.name == TARGET_COLUMN
    assert labels.tolist() == [0, 1]


def test_api_health_and_single_prediction() -> None:
    assert client.get("/health").status_code == 200
    response = client.post(
        "/predict",
        json={
            "age": 30,
            "overall": 75,
            "potential": 80,
            "height_cm": 175,
            "weight_kg": 70,
            "weak_foot": 3,
            "skill_moves": 3,
            "international_reputation": 1,
            "player_positions": "CM",
            "preferred_foot": "Right",
        },
    )
    assert response.status_code == 200
    result = response.json()
    assert result["churn_proxy"] in (0, 1)
    assert 0.0 <= result["risk_probability"] <= 1.0
    assert "synthetic" in result["warning"]


def test_batch_api_and_input_validation() -> None:
    response = client.post(
        "/predict/batch",
        json={
            "players": [
                {"age": 24, "overall": 70, "potential": 78},
                {"age": 35, "overall": 80, "potential": 80},
            ]
        },
    )
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2
    assert client.post(
        "/predict", json={"age": 30, "overall": 101, "potential": 80}
    ).status_code == 422


def test_same_snapshot_has_no_monitoring_drift() -> None:
    frame = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    report = build_monitor_report(frame, frame)
    assert report["drift_detected"] is False
    assert report["reference_rows"] == report["current_rows"]


def test_shifted_age_distribution_triggers_monitoring_alert() -> None:
    frame = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    shifted = frame.copy()
    shifted["age"] = 42
    report = build_monitor_report(frame, shifted)
    assert report["feature_psi"]["age"] >= report["psi_alert_threshold"]
    assert report["drift_detected"] is True
