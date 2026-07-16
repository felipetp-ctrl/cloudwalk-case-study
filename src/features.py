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
