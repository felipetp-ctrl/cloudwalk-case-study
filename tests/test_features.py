import pandas as pd
import numpy as np
from src.features import shannon_entropy, compute_frequency_encoding, compute_per_request_features


def test_shannon_entropy_uniform():
    assert abs(shannon_entropy("aabb") - 1.0) < 0.01


def test_shannon_entropy_single_char():
    assert shannon_entropy("aaaa") == 0.0


def test_shannon_entropy_empty():
    assert shannon_entropy("") == 0.0


def test_frequency_encoding():
    s = pd.Series(['a', 'a', 'b', 'a', 'c', 'b'])
    result = compute_frequency_encoding(s)
    assert abs(result.iloc[0] - 3 / 6) < 0.001
    assert abs(result.iloc[2] - 2 / 6) < 0.001
    assert abs(result.iloc[4] - 1 / 6) < 0.001


def test_per_request_features_columns():
    requests = pd.DataFrame({
        'request_id': ['r1'],
        'timestamp': pd.to_datetime(['2025-01-07T04:00:00+00:00']),
        'source_ip': ['10.0.0.1'],
        'method': ['POST'],
        'path': ['/api/v1/auth/login'],
        'status_code': [200],
        'response_time_ms': [12],
        'body_size_bytes': [340],
        'user_agent': ['Mozilla/5.0 (Windows NT 10.0) Chrome/131.0.0.0'],
        'tls_fingerprint': ['ja3_abc'],
        'country': ['BR'],
        'asn': ['AS16509'],
    })
    headers = pd.DataFrame({
        'request_id': ['r1', 'r1', 'r1'],
        'header_name': ['Accept-Language', 'Cookie', 'Content-Type'],
        'header_value': ['pt-BR', 'session=abc', 'application/json'],
    })
    result = compute_per_request_features(requests, headers)
    expected_cols = [
        'method_freq', 'path_depth', 'path_length', 'path_entropy', 'path_has_params',
        'status_code_group', 'response_time_ms', 'body_size_bytes',
        'ua_length', 'ua_entropy', 'ua_is_browser', 'ua_is_bot_library',
        'country_freq', 'asn_freq', 'tls_fingerprint_freq',
        'hour_of_day', 'is_sensitive_endpoint',
        'header_count', 'has_accept_language', 'has_referer', 'has_cookie', 'has_authorization',
    ]
    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"


def test_per_request_features_values():
    requests = pd.DataFrame({
        'request_id': ['r1'],
        'timestamp': pd.to_datetime(['2025-01-07T14:00:00+00:00']),
        'source_ip': ['10.0.0.1'],
        'method': ['POST'],
        'path': ['/api/v1/auth/login'],
        'status_code': [403],
        'response_time_ms': [12],
        'body_size_bytes': [340],
        'user_agent': ['python-requests/2.28'],
        'tls_fingerprint': ['ja3_abc'],
        'country': ['BR'],
        'asn': ['AS16509'],
    })
    headers = pd.DataFrame({
        'request_id': ['r1'],
        'header_name': ['Content-Type'],
        'header_value': ['application/json'],
    })
    result = compute_per_request_features(requests, headers)
    row = result.iloc[0]
    assert row['path_depth'] == 4
    assert row['is_sensitive_endpoint'] == True
    assert row['ua_is_bot_library'] == True
    assert row['ua_is_browser'] == False
    assert row['status_code_group'] == 4
    assert row['hour_of_day'] == 14
    assert row['header_count'] == 1
    assert row['has_accept_language'] == False
    assert row['has_cookie'] == False
    assert row['path_has_params'] == False
