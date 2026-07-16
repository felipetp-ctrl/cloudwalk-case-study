import pandas as pd
import numpy as np
from src.pipeline import build_training_dataset, compute_sample_weights


def test_sample_weights():
    df = pd.DataFrame({
        'is_malicious': [True, True, True, False],
        'confidence': ['high', 'medium', 'low', np.nan],
    })
    result = compute_sample_weights(df)
    assert result.iloc[0] == 1.0
    assert result.iloc[1] == 0.6
    assert result.iloc[2] == 0.3
    assert result.iloc[3] == 1.0


def test_build_training_dataset_runs_on_real_data():
    result = build_training_dataset(
        'http_requests.csv',
        'request_headers.csv',
        'incident_labels.csv',
    )
    session_cols = [c for c in result.columns if c.startswith(('ip_', 'tls_')) and c != 'tls_fingerprint' and c != 'tls_fingerprint_freq']
    assert len(session_cols) == 78
    assert 'method_freq' in result.columns
    assert 'is_malicious' in result.columns
    assert 'sample_weight' in result.columns
    assert len(result) == 50000
    assert result['is_malicious'].sum() > 0
