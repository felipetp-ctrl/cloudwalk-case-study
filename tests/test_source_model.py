import pandas as pd
import numpy as np
import pytest


def _make_source_df():
    """Simulated request data with DDoS-like and benign IPs."""
    rows = []
    # Benign IPs: diverse paths, mostly GET/POST
    for ip_idx in range(20):
        ip = f"10.0.0.{ip_idx}"
        paths = ["/api/v1/users", "/api/v1/transactions", "/api/v1/cards", "/health"]
        for day_offset in range(4):
            date = f"2025-01-{6 + day_offset:02d}"
            for i in range(8):
                rows.append(
                    {
                        "request_id": f"b_{ip_idx}_{day_offset}_{i}",
                        "timestamp": f"{date} {10 + i}:00:00+00:00",
                        "source_ip": ip,
                        "method": np.random.choice(["GET", "POST"], p=[0.7, 0.3]),
                        "path": np.random.choice(paths),
                        "status_code": 200,
                        "user_agent": "Mozilla/5.0 Chrome/131",
                        "is_malicious": False,
                        "attack_class": None,
                        "ua_entropy": 4.2,
                        "ua_is_bot_library": False,
                        "ua_is_browser": True,
                        "header_count": 5,
                        "has_cookie": True,
                        "has_accept_language": True,
                        "path_entropy": 3.1,
                        "is_sensitive_endpoint": "auth" in paths[i % len(paths)]
                        or "login" in paths[i % len(paths)],
                    }
                )
    # DDoS IPs: single path, varied methods, day 2-3 only
    for ip_idx in range(5):
        ip = f"192.168.1.{ip_idx}"
        for day_offset in [2, 3]:
            date = f"2025-01-{6 + day_offset:02d}"
            for i in range(6):
                rows.append(
                    {
                        "request_id": f"d_{ip_idx}_{day_offset}_{i}",
                        "timestamp": f"{date} {10 + i}:00:00+00:00",
                        "source_ip": ip,
                        "method": np.random.choice(
                            ["GET", "POST", "PUT", "HEAD", "DELETE"]
                        ),
                        "path": "/api/v1/transactions",
                        "status_code": 200,
                        "user_agent": "Mozilla/5.0 Chrome/131",
                        "is_malicious": True,
                        "attack_class": "ddos_l7",
                        "ua_entropy": 4.2,
                        "ua_is_bot_library": False,
                        "ua_is_browser": True,
                        "header_count": 3,
                        "has_cookie": False,
                        "has_accept_language": False,
                        "path_entropy": 2.8,
                        "is_sensitive_endpoint": False,
                    }
                )
    return pd.DataFrame(rows)


def test_compute_source_features_shape():
    from src.source_model import compute_source_features

    df = _make_source_df()
    agg = compute_source_features(df, min_requests=5)
    assert len(agg) > 0
    assert "path_concentration" in agg.columns
    assert "method_per_path" in agg.columns
    assert "total_requests" in agg.columns
    assert "day" in agg.columns


def test_compute_source_features_min_requests_filter():
    from src.source_model import compute_source_features

    df = _make_source_df()
    agg_5 = compute_source_features(df, min_requests=5)
    agg_20 = compute_source_features(df, min_requests=20)
    assert len(agg_5) >= len(agg_20)
    assert (agg_5["total_requests"] >= 5).all()
    assert (agg_20["total_requests"] >= 20).all()


def test_compute_source_features_ddos_pattern():
    from src.source_model import compute_source_features

    df = _make_source_df()
    agg = compute_source_features(df, min_requests=5)
    ddos = agg[agg["attack_class"] == "ddos_l7"]
    benign = agg[~agg["is_malicious"]]
    assert (ddos["unique_paths"] == 1).all()
    assert ddos["path_concentration"].mean() > benign["path_concentration"].mean()


def test_train_source_model():
    from src.source_model import (
        compute_source_features,
        train_source_model,
        SOURCE_FEATURE_COLS,
    )

    df = _make_source_df()
    agg = compute_source_features(df, min_requests=5)
    X = agg[SOURCE_FEATURE_COLS].astype(float)
    y = agg["is_malicious"].astype(int)
    model = train_source_model(X, y)
    assert hasattr(model, "predict_proba")
    probs = model.predict_proba(X)
    assert probs.shape == (len(X), 2)


def test_evaluate_source_model():
    from src.source_model import (
        compute_source_features,
        train_source_model,
        evaluate_source_model,
        SOURCE_FEATURE_COLS,
    )

    df = _make_source_df()
    agg = compute_source_features(df, min_requests=5)
    X = agg[SOURCE_FEATURE_COLS].astype(float)
    y = agg["is_malicious"].astype(int)
    model = train_source_model(X, y)
    metrics = evaluate_source_model(model, X, y)
    assert "precision" in metrics
    assert "recall" in metrics
    assert "pr_auc" in metrics
    assert 0 <= metrics["pr_auc"] <= 1


def test_propagate_source_scores():
    from src.source_model import propagate_source_scores

    requests = pd.DataFrame(
        {
            "request_id": ["r1", "r2", "r3"],
            "timestamp": [
                "2025-01-08 10:00:00+00:00",
                "2025-01-08 11:00:00+00:00",
                "2025-01-09 10:00:00+00:00",
            ],
            "source_ip": ["10.0.0.1", "10.0.0.1", "10.0.0.2"],
        }
    )
    scores = pd.DataFrame(
        {
            "source_ip": ["10.0.0.1", "10.0.0.2"],
            "day": [
                pd.Timestamp("2025-01-08").date(),
                pd.Timestamp("2025-01-09").date(),
            ],
            "source_score": [0.9, 0.1],
        }
    )
    result = propagate_source_scores(requests, scores)
    assert len(result) == 3
    assert result[0] == pytest.approx(0.9)
    assert result[1] == pytest.approx(0.9)
    assert result[2] == pytest.approx(0.1)


def test_propagate_missing_scores_default_zero():
    from src.source_model import propagate_source_scores

    requests = pd.DataFrame(
        {
            "request_id": ["r1"],
            "timestamp": ["2025-01-08 10:00:00+00:00"],
            "source_ip": ["10.0.0.99"],
        }
    )
    scores = pd.DataFrame(
        {
            "source_ip": ["10.0.0.1"],
            "day": [pd.Timestamp("2025-01-08").date()],
            "source_score": [0.9],
        }
    )
    result = propagate_source_scores(requests, scores)
    assert result[0] == pytest.approx(0.0)


def test_ensemble_scores():
    from src.source_model import ensemble_scores

    req = np.array([0.8, 0.1, 0.3])
    src = np.array([0.2, 0.9, 0.3])
    result = ensemble_scores(req, src)
    np.testing.assert_array_almost_equal(result, [0.8, 0.9, 0.3])


def test_full_pipeline_on_real_data():
    from src.source_model import (
        compute_source_features,
        train_source_model,
        evaluate_source_model,
        propagate_source_scores,
        SOURCE_FEATURE_COLS,
    )
    from src.label_joining import join_labels
    from src.features import compute_per_request_features

    requests = pd.read_csv("data/http_requests.csv", parse_dates=["timestamp"])
    headers = pd.read_csv("data/request_headers.csv")
    labels = pd.read_csv(
        "data/incident_labels.csv",
        parse_dates=["active_from", "active_until", "labeled_at"],
    )

    df = join_labels(requests, labels)
    df = compute_per_request_features(df, headers)

    agg = compute_source_features(df, min_requests=5)
    cutoff = pd.Timestamp("2025-01-10").date()
    train_agg = agg[agg["day"] < cutoff]
    test_agg = agg[agg["day"] >= cutoff]

    X_train = train_agg[SOURCE_FEATURE_COLS].astype(float)
    y_train = train_agg["is_malicious"].astype(int)
    X_test = test_agg[SOURCE_FEATURE_COLS].astype(float)
    y_test = test_agg["is_malicious"].astype(int)

    model = train_source_model(X_train, y_train)
    metrics = evaluate_source_model(model, X_test, y_test)

    assert metrics["pr_auc"] > 0.5
    assert metrics["recall"] > 0.5

    test_agg = test_agg.copy()
    test_agg["source_score"] = model.predict_proba(X_test)[:, 1]

    test_requests = df[pd.to_datetime(df["timestamp"], utc=True).dt.date >= cutoff]
    source_scores = propagate_source_scores(test_requests, test_agg)
    assert len(source_scores) == len(test_requests)
