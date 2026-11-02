import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from anomaly.api import app
from anomaly.data import SENSORS
from anomaly.detectors import DETECTORS, run_detector
from anomaly.features import rolling_features, sliding_windows
from anomaly.metrics import changepoint_metrics, event_metrics, point_metrics


def synthetic(n=700, shift_at=500, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 0.1, (n, len(SENSORS)))
    base[:, 1] = 0.8 * base[:, 0] + rng.normal(0, 0.02, n)  # deux capteurs corrélés
    base[shift_at:, 0] += 2.0  # panne : un capteur décroche
    df = pd.DataFrame(base, columns=SENSORS)
    df["anomaly"] = 0.0
    df.loc[shift_at:, "anomaly"] = 1.0
    return df


def test_point_metrics_known_values():
    m = point_metrics([1, 1, 0, 0], [1, 0, 1, 0])
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["fpr"] == 0.5


def test_event_metrics():
    y = [0, 1, 1, 0, 0, 1, 0]
    p = [0, 0, 1, 0, 1, 0, 0]
    m = event_metrics(y, p)
    assert m["events"] == 2 and m["events_detected"] == 1 and m["false_alarm_runs"] == 1


def test_changepoint_margin():
    m = changepoint_metrics([100], [110, 400], margin=15)
    assert m["cp_recall"] == 1.0 and m["cp_precision"] == 0.5


def test_features_shapes():
    df = synthetic()
    assert rolling_features(df).shape == (len(df), 3 * len(SENSORS))
    assert sliding_windows(df[SENSORS].values, 10).shape == (len(df), 10 * len(SENSORS))


@pytest.mark.parametrize("name", list(DETECTORS))
def test_detector_ranks_fault_higher(name):
    df = synthetic()
    scores, thr = run_detector(name, df, baseline=400, quantile=0.99, epochs=15)
    assert scores[520:].mean() > scores[:400].mean()


def test_autoencoder_is_deterministic():
    df = synthetic()
    a, _ = run_detector("autoencoder", df, baseline=400, epochs=5)
    b, _ = run_detector("autoencoder", df, baseline=400, epochs=5)
    assert np.allclose(a, b)


def test_api_detect_and_errors():
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}
    df = synthetic()
    rows = df[SENSORS].to_dict("records")
    r = client.post("/detect", json={"model": "isolation_forest", "rows": rows, "baseline": 400})
    assert r.status_code == 200
    body = r.json()
    assert len(body["scores"]) == len(rows) and body["n_alarms"] == sum(body["alarms"])
    assert client.post("/detect", json={"model": "nope", "rows": rows}).status_code == 422
    assert client.post("/detect", json={"model": "ecod", "rows": rows[:100], "baseline": 400}).status_code == 422
