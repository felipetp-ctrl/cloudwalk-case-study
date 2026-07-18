import pandas as pd
from src.features import (
    shannon_entropy,
    compute_frequency_encoding,
    compute_per_request_features,
    compute_session_features,
)


def test_shannon_entropy_uniform():
    assert abs(shannon_entropy("aabb") - 1.0) < 0.01


def test_shannon_entropy_single_char():
    assert shannon_entropy("aaaa") == 0.0


def test_shannon_entropy_empty():
    assert shannon_entropy("") == 0.0


def test_frequency_encoding():
    s = pd.Series(["a", "a", "b", "a", "c", "b"])
    result = compute_frequency_encoding(s)
    assert abs(result.iloc[0] - 3 / 6) < 0.001
    assert abs(result.iloc[2] - 2 / 6) < 0.001
    assert abs(result.iloc[4] - 1 / 6) < 0.001


def test_per_request_features_columns():
    requests = pd.DataFrame(
        {
            "request_id": ["r1"],
            "timestamp": pd.to_datetime(["2025-01-07T04:00:00+00:00"]),
            "source_ip": ["10.0.0.1"],
            "method": ["POST"],
            "path": ["/api/v1/auth/login"],
            "status_code": [200],
            "response_time_ms": [12],
            "body_size_bytes": [340],
            "user_agent": ["Mozilla/5.0 (Windows NT 10.0) Chrome/131.0.0.0"],
            "tls_fingerprint": ["ja3_abc"],
            "country": ["BR"],
            "asn": ["AS16509"],
        }
    )
    headers = pd.DataFrame(
        {
            "request_id": ["r1", "r1", "r1"],
            "header_name": ["Accept-Language", "Cookie", "Content-Type"],
            "header_value": ["pt-BR", "session=abc", "application/json"],
        }
    )
    result = compute_per_request_features(requests, headers)
    expected_cols = [
        "method_freq",
        "path_depth",
        "path_length",
        "path_entropy",
        "path_has_params",
        "status_code_group",
        "response_time_ms",
        "body_size_bytes",
        "ua_length",
        "ua_entropy",
        "ua_is_browser",
        "ua_is_bot_library",
        "country_freq",
        "asn_freq",
        "tls_fingerprint_freq",
        "hour_of_day",
        "is_sensitive_endpoint",
        "header_count",
        "has_accept_language",
        "has_referer",
        "has_cookie",
        "has_authorization",
    ]
    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"


def test_per_request_features_values():
    requests = pd.DataFrame(
        {
            "request_id": ["r1"],
            "timestamp": pd.to_datetime(["2025-01-07T14:00:00+00:00"]),
            "source_ip": ["10.0.0.1"],
            "method": ["POST"],
            "path": ["/api/v1/auth/login"],
            "status_code": [403],
            "response_time_ms": [12],
            "body_size_bytes": [340],
            "user_agent": ["python-requests/2.28"],
            "tls_fingerprint": ["ja3_abc"],
            "country": ["BR"],
            "asn": ["AS16509"],
        }
    )
    headers = pd.DataFrame(
        {
            "request_id": ["r1"],
            "header_name": ["Content-Type"],
            "header_value": ["application/json"],
        }
    )
    result = compute_per_request_features(requests, headers)
    row = result.iloc[0]
    assert row["path_depth"] == 4
    assert bool(row["is_sensitive_endpoint"])
    assert bool(row["ua_is_bot_library"])
    assert not bool(row["ua_is_browser"])
    assert row["status_code_group"] == 4
    assert row["hour_of_day"] == 14
    assert row["header_count"] == 1
    assert not bool(row["has_accept_language"])
    assert not bool(row["has_cookie"])
    assert not bool(row["path_has_params"])


def _make_session_data():
    """5 requests from same IP in a 2-minute span, all within one 5min floor window."""
    base_time = pd.Timestamp("2025-01-07T10:00:00", tz="UTC")
    return pd.DataFrame(
        {
            "request_id": [f"r{i}" for i in range(5)],
            "timestamp": [base_time + pd.Timedelta(seconds=i * 15) for i in range(5)],
            "source_ip": ["10.0.0.1"] * 5,
            "tls_fingerprint": ["ja3_abc"] * 5,
            "path": ["/api/v1/auth/login"] * 3 + ["/api/v1/users/me"] * 2,
            "status_code": [200, 401, 403, 200, 200],
            "response_time_ms": [10, 12, 11, 50, 48],
            "body_size_bytes": [100, 100, 100, 200, 200],
            "method": ["POST", "POST", "POST", "GET", "GET"],
            "is_sensitive_endpoint": [True, True, True, False, False],
        }
    )


def test_session_features_columns_exist():
    df = _make_session_data()
    result = compute_session_features(df)
    for prefix in ["ip_1m", "ip_5m", "ip_30m", "tls_1m", "tls_5m", "tls_30m"]:
        assert f"{prefix}_request_count" in result.columns
        assert f"{prefix}_error_rate" in result.columns
        assert f"{prefix}_unique_paths" in result.columns


def test_session_features_causal_first_request():
    """First request in a window should have all-zero session features (no history)."""
    df = _make_session_data()
    result = compute_session_features(df)
    row = result.iloc[0]
    assert row["ip_5m_request_count"] == 0
    assert row["ip_5m_unique_paths"] == 0
    assert row["ip_5m_error_rate"] == 0.0
    assert row["ip_5m_method_diversity"] == 0
    assert row["ip_5m_sensitive_endpoint_ratio"] == 0.0
    assert row["ip_5m_avg_response_time"] == 0.0


def test_session_features_causal_accumulation():
    """Later requests should see only past data, never the current or future rows."""
    df = _make_session_data()
    result = compute_session_features(df)

    # Second request (index 1) sees only request 0
    row1 = result.iloc[1]
    assert row1["ip_5m_request_count"] == 1
    assert row1["ip_5m_unique_paths"] == 1  # only path from request 0
    assert row1["ip_5m_error_rate"] == 0.0  # request 0 had status 200
    assert row1["ip_5m_method_diversity"] == 1

    # Third request (index 2) sees requests 0 and 1
    row2 = result.iloc[2]
    assert row2["ip_5m_request_count"] == 2
    assert row2["ip_5m_unique_paths"] == 1  # both prior requests hit same path
    assert abs(row2["ip_5m_error_rate"] - 0.5) < 0.01  # request 1 had 401

    # Fifth request (index 4) sees requests 0-3
    row4 = result.iloc[4]
    assert row4["ip_5m_request_count"] == 4
    assert row4["ip_5m_unique_paths"] == 2  # /auth/login + /users/me
    assert row4["ip_5m_method_diversity"] == 2  # POST + GET


def test_session_feature_count():
    df = _make_session_data()
    result = compute_session_features(df)
    session_cols = [
        c
        for c in result.columns
        if c.startswith(("ip_", "tls_")) and c != "tls_fingerprint"
    ]
    assert len(session_cols) == 78
