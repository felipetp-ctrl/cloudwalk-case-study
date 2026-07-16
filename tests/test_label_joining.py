import pandas as pd
import numpy as np
from src.label_joining import join_labels


def test_ip_exact_match_within_window():
    requests = pd.DataFrame({
        'request_id': ['r1', 'r2'],
        'timestamp': pd.to_datetime(['2025-01-07T04:00:00+00:00', '2025-01-07T08:00:00+00:00']),
        'source_ip': ['185.220.101.40', '185.220.101.40'],
        'tls_fingerprint': ['ja3_aaa', 'ja3_aaa'],
        'method': ['GET', 'GET'],
        'path': ['/api/v1/auth/login', '/api/v1/auth/login'],
        'status_code': [200, 200],
        'response_time_ms': [10, 10],
        'body_size_bytes': [100, 100],
        'user_agent': ['Mozilla/5.0', 'Mozilla/5.0'],
        'country': ['BR', 'BR'],
        'asn': ['AS8167', 'AS8167'],
    })
    labels = pd.DataFrame({
        'incident_id': ['INC-001'],
        'source_identifier': ['185.220.101.40'],
        'identifier_type': ['ip'],
        'attack_class': ['credential_stuffing'],
        'confidence': ['high'],
        'labeled_at': pd.to_datetime(['2025-01-09']),
        'active_from': pd.to_datetime(['2025-01-07T03:00:00+00:00']),
        'active_until': pd.to_datetime(['2025-01-07T06:00:00+00:00']),
    })
    result = join_labels(requests, labels)
    assert result.loc[result.request_id == 'r1', 'is_malicious'].iloc[0] == True
    assert result.loc[result.request_id == 'r2', 'is_malicious'].iloc[0] == False


def test_cidr_range_match():
    requests = pd.DataFrame({
        'request_id': ['r1'],
        'timestamp': pd.to_datetime(['2025-01-07T04:00:00+00:00']),
        'source_ip': ['192.168.1.5'],
        'tls_fingerprint': ['ja3_bbb'],
        'method': ['GET'],
        'path': ['/'],
        'status_code': [200],
        'response_time_ms': [10],
        'body_size_bytes': [100],
        'user_agent': ['Mozilla/5.0'],
        'country': ['BR'],
        'asn': ['AS8167'],
    })
    labels = pd.DataFrame({
        'incident_id': ['INC-002'],
        'source_identifier': ['192.168.1.0/24'],
        'identifier_type': ['ip_range'],
        'attack_class': ['scanner'],
        'confidence': ['medium'],
        'labeled_at': pd.to_datetime(['2025-01-09']),
        'active_from': pd.to_datetime(['2025-01-07T03:00:00+00:00']),
        'active_until': pd.to_datetime(['2025-01-07T06:00:00+00:00']),
    })
    result = join_labels(requests, labels)
    assert result.loc[result.request_id == 'r1', 'is_malicious'].iloc[0] == True
    assert result.loc[result.request_id == 'r1', 'attack_class'].iloc[0] == 'scanner'


def test_tls_fingerprint_match():
    requests = pd.DataFrame({
        'request_id': ['r1'],
        'timestamp': pd.to_datetime(['2025-01-07T04:00:00+00:00']),
        'source_ip': ['10.0.0.1'],
        'tls_fingerprint': ['ja3_evil'],
        'method': ['GET'],
        'path': ['/'],
        'status_code': [200],
        'response_time_ms': [10],
        'body_size_bytes': [100],
        'user_agent': ['Mozilla/5.0'],
        'country': ['BR'],
        'asn': ['AS8167'],
    })
    labels = pd.DataFrame({
        'incident_id': ['INC-003'],
        'source_identifier': ['ja3_evil'],
        'identifier_type': ['tls_fingerprint'],
        'attack_class': ['scanner'],
        'confidence': ['high'],
        'labeled_at': pd.to_datetime(['2025-01-09']),
        'active_from': pd.to_datetime(['2025-01-07T03:00:00+00:00']),
        'active_until': pd.to_datetime(['2025-01-07T06:00:00+00:00']),
    })
    result = join_labels(requests, labels)
    assert result.loc[0, 'is_malicious'] == True


def test_no_labels_all_benign():
    requests = pd.DataFrame({
        'request_id': ['r1'],
        'timestamp': pd.to_datetime(['2025-01-07T04:00:00+00:00']),
        'source_ip': ['10.0.0.1'],
        'tls_fingerprint': ['ja3_clean'],
        'method': ['GET'],
        'path': ['/'],
        'status_code': [200],
        'response_time_ms': [10],
        'body_size_bytes': [100],
        'user_agent': ['Mozilla/5.0'],
        'country': ['BR'],
        'asn': ['AS8167'],
    })
    labels = pd.DataFrame(columns=['incident_id', 'source_identifier', 'identifier_type',
                                    'attack_class', 'confidence', 'labeled_at',
                                    'active_from', 'active_until'])
    result = join_labels(requests, labels)
    assert result.loc[0, 'is_malicious'] == False
