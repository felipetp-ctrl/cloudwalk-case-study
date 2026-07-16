import re
import math
from collections import Counter

import pandas as pd
import numpy as np


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    length = len(s)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def compute_frequency_encoding(series: pd.Series) -> pd.Series:
    freq = series.value_counts(normalize=True)
    return series.map(freq).astype(float)


_SENSITIVE_PATTERN = re.compile(r'/(?:auth|login|payment|tokenize)', re.IGNORECASE)
_BROWSER_PATTERN = re.compile(r'Mozilla|Chrome|Safari|Firefox|Edg/', re.IGNORECASE)
_BOT_PATTERN = re.compile(r'python-requests|curl|wget|Go-http-client|axios|httpx|scrapy|bot|spider', re.IGNORECASE)


def compute_per_request_features(requests: pd.DataFrame, headers: pd.DataFrame) -> pd.DataFrame:
    df = requests.copy()

    df['path_depth'] = df['path'].str.strip('/').str.split('/').str.len()
    df['path_length'] = df['path'].str.len()
    df['path_entropy'] = df['path'].apply(shannon_entropy)
    df['path_has_params'] = df['path'].str.contains(r'[?=]', regex=True)

    df['status_code_group'] = df['status_code'] // 100

    df['ua_length'] = df['user_agent'].fillna('').str.len()
    df['ua_entropy'] = df['user_agent'].fillna('').apply(shannon_entropy)
    df['ua_is_browser'] = df['user_agent'].fillna('').str.contains(_BROWSER_PATTERN).astype(bool)
    df['ua_is_bot_library'] = df['user_agent'].fillna('').str.contains(_BOT_PATTERN).astype(bool)

    df['method_freq'] = compute_frequency_encoding(df['method'])
    df['country_freq'] = compute_frequency_encoding(df['country'])
    df['asn_freq'] = compute_frequency_encoding(df['asn'])
    df['tls_fingerprint_freq'] = compute_frequency_encoding(df['tls_fingerprint'])

    df['hour_of_day'] = pd.to_datetime(df['timestamp'], utc=True).dt.hour

    df['is_sensitive_endpoint'] = df['path'].str.contains(_SENSITIVE_PATTERN).astype(bool)

    header_agg = headers.groupby('request_id').agg(
        header_count=('header_name', 'count'),
        has_accept_language=('header_name', lambda x: 'Accept-Language' in x.values),
        has_referer=('header_name', lambda x: any(h.lower() == 'referer' for h in x.values)),
        has_cookie=('header_name', lambda x: 'Cookie' in x.values),
        has_authorization=('header_name', lambda x: 'Authorization' in x.values),
    ).reset_index()

    df = df.merge(header_agg, on='request_id', how='left')
    df['header_count'] = df['header_count'].fillna(0).astype(int)
    for col in ['has_accept_language', 'has_referer', 'has_cookie', 'has_authorization']:
        df[col] = df[col].fillna(False).astype(bool)

    return df


_WINDOWS = {'1m': 60, '5m': 300, '30m': 1800}
_GRANULARITIES = {'ip': 'source_ip', 'tls': 'tls_fingerprint'}


def _series_entropy(series: pd.Series) -> float:
    counts = series.value_counts()
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    return -float((probs * np.log2(probs)).sum())


def compute_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 78 session-level features using fixed-window group-by."""
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['_ts_epoch'] = df['timestamp'].astype(np.int64) // 10**9

    for gran_prefix, gran_col in _GRANULARITIES.items():
        for win_prefix, win_seconds in _WINDOWS.items():
            col_prefix = f'{gran_prefix}_{win_prefix}'
            window_key = df[gran_col].astype(str) + '_' + (df['_ts_epoch'] // win_seconds).astype(str)
            df['_wk'] = window_key

            g = df.groupby('_wk')

            df[f'{col_prefix}_request_count'] = g['request_id'].transform('count')
            df[f'{col_prefix}_unique_paths'] = g['path'].transform('nunique')
            df[f'{col_prefix}_error_rate'] = g['status_code'].transform(lambda x: (x >= 400).sum() / len(x))
            df[f'{col_prefix}_unique_status_codes'] = g['status_code'].transform('nunique')
            df[f'{col_prefix}_avg_response_time'] = g['response_time_ms'].transform('mean')
            df[f'{col_prefix}_std_response_time'] = g['response_time_ms'].transform(lambda x: x.std(ddof=0) if len(x) > 1 else 0.0)
            df[f'{col_prefix}_avg_body_size'] = g['body_size_bytes'].transform('mean')
            df[f'{col_prefix}_std_body_size'] = g['body_size_bytes'].transform(lambda x: x.std(ddof=0) if len(x) > 1 else 0.0)
            df[f'{col_prefix}_method_diversity'] = g['method'].transform('nunique')
            df[f'{col_prefix}_sensitive_endpoint_ratio'] = g['is_sensitive_endpoint'].transform('mean')

            path_entropy = g['path'].apply(_series_entropy)
            df[f'{col_prefix}_path_entropy'] = df['_wk'].map(path_entropy)

            def _timing_stats(ts_group):
                ts = ts_group.sort_values()
                diffs = ts.diff().dt.total_seconds().dropna()
                if len(diffs) == 0:
                    return pd.Series({'_mean': 0.0, '_std': 0.0})
                return pd.Series({
                    '_mean': diffs.mean(),
                    '_std': diffs.std(ddof=0) if len(diffs) > 1 else 0.0,
                })

            timing = g['timestamp'].apply(_timing_stats).unstack()
            df[f'{col_prefix}_inter_request_time_mean'] = df['_wk'].map(timing['_mean'])
            df[f'{col_prefix}_inter_request_time_std'] = df['_wk'].map(timing['_std'])

    df = df.drop(columns=['_wk', '_ts_epoch'])
    return df
