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


_SENSITIVE_PATTERN = re.compile(r"/(?:auth|login|payment|tokenize)", re.IGNORECASE)
_BROWSER_PATTERN = re.compile(r"Mozilla|Chrome|Safari|Firefox|Edg/", re.IGNORECASE)
_BOT_PATTERN = re.compile(
    r"python-requests|curl|wget|Go-http-client|axios|httpx|scrapy|bot|spider",
    re.IGNORECASE,
)


def compute_per_request_features(
    requests: pd.DataFrame, headers: pd.DataFrame
) -> pd.DataFrame:
    df = requests.copy()

    df["path_depth"] = df["path"].str.strip("/").str.split("/").str.len()
    df["path_length"] = df["path"].str.len()
    df["path_entropy"] = df["path"].apply(shannon_entropy)
    df["path_has_params"] = df["path"].str.contains(r"[?=]", regex=True)

    df["status_code_group"] = df["status_code"] // 100

    df["ua_length"] = df["user_agent"].fillna("").str.len()
    df["ua_entropy"] = df["user_agent"].fillna("").apply(shannon_entropy)
    df["ua_is_browser"] = (
        df["user_agent"].fillna("").str.contains(_BROWSER_PATTERN).astype(bool)
    )
    df["ua_is_bot_library"] = (
        df["user_agent"].fillna("").str.contains(_BOT_PATTERN).astype(bool)
    )

    df["method_freq"] = compute_frequency_encoding(df["method"])
    df["country_freq"] = compute_frequency_encoding(df["country"])
    df["asn_freq"] = compute_frequency_encoding(df["asn"])
    df["tls_fingerprint_freq"] = compute_frequency_encoding(df["tls_fingerprint"])

    df["hour_of_day"] = pd.to_datetime(df["timestamp"], utc=True).dt.hour

    df["is_sensitive_endpoint"] = (
        df["path"].str.contains(_SENSITIVE_PATTERN).astype(bool)
    )

    header_agg = (
        headers.groupby("request_id")
        .agg(
            header_count=("header_name", "count"),
            has_accept_language=(
                "header_name",
                lambda x: "Accept-Language" in x.values,
            ),
            has_referer=(
                "header_name",
                lambda x: any(h.lower() == "referer" for h in x.values),
            ),
            has_cookie=("header_name", lambda x: "Cookie" in x.values),
            has_authorization=("header_name", lambda x: "Authorization" in x.values),
        )
        .reset_index()
    )

    df = df.merge(header_agg, on="request_id", how="left")
    df["header_count"] = df["header_count"].fillna(0).astype(int)
    for col in [
        "has_accept_language",
        "has_referer",
        "has_cookie",
        "has_authorization",
    ]:
        df[col] = df[col].fillna(False).astype(bool)

    return df


_WINDOWS = {"1m": 60, "5m": 300, "30m": 1800}
_GRANULARITIES = {"ip": "source_ip", "tls": "tls_fingerprint"}


def _causal_nunique(values) -> np.ndarray:
    """Count of unique values seen strictly before each position."""
    seen = set()
    result = np.empty(len(values), dtype=np.float64)
    for i, v in enumerate(values):
        result[i] = len(seen)
        seen.add(v)
    return result


def _causal_entropy(values) -> np.ndarray:
    """Shannon entropy of value distribution seen strictly before each position."""
    counts: dict = {}
    total = 0
    result = np.empty(len(values), dtype=np.float64)
    for i, v in enumerate(values):
        if total == 0:
            result[i] = 0.0
        else:
            result[i] = -sum(
                (c / total) * math.log2(c / total) for c in counts.values()
            )
        counts[v] = counts.get(v, 0) + 1
        total += 1
    return result


def compute_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 78 session-level features using causal windowing (past-only).

    For each request at time t, features aggregate only prior requests from
    the same source within the same fixed time window. The current request
    is never included in its own session aggregates, preventing data leakage
    in offline training while matching the real-time edge computation model.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["_ts_epoch"] = df["timestamp"].astype(np.int64) // 10**9
    df["_is_error"] = (df["status_code"] >= 400).astype(float)

    if "is_sensitive_endpoint" in df.columns:
        df["_is_sensitive"] = df["is_sensitive_endpoint"].astype(float)
    else:
        df["_is_sensitive"] = df["path"].str.contains(_SENSITIVE_PATTERN).astype(float)

    for gran_prefix, gran_col in _GRANULARITIES.items():
        for win_prefix, win_seconds in _WINDOWS.items():
            col_prefix = f"{gran_prefix}_{win_prefix}"
            df["_wk"] = (
                df[gran_col].astype(str)
                + "_"
                + (df["_ts_epoch"] // win_seconds).astype(str)
            )
            g = df.groupby("_wk", sort=False)

            df[f"{col_prefix}_request_count"] = g.cumcount()

            df[f"{col_prefix}_unique_paths"] = g["path"].transform(
                lambda x: pd.Series(_causal_nunique(x.values), index=x.index)
            )
            df[f"{col_prefix}_path_entropy"] = g["path"].transform(
                lambda x: pd.Series(_causal_entropy(x.values), index=x.index)
            )

            df[f"{col_prefix}_error_rate"] = g["_is_error"].transform(
                lambda x: x.expanding().mean().shift(1).fillna(0)
            )
            df[f"{col_prefix}_unique_status_codes"] = g["status_code"].transform(
                lambda x: pd.Series(_causal_nunique(x.values), index=x.index)
            )

            df[f"{col_prefix}_avg_response_time"] = g["response_time_ms"].transform(
                lambda x: x.expanding().mean().shift(1).fillna(0)
            )
            df[f"{col_prefix}_std_response_time"] = g["response_time_ms"].transform(
                lambda x: x.expanding().std(ddof=0).shift(1).fillna(0)
            )

            df[f"{col_prefix}_avg_body_size"] = g["body_size_bytes"].transform(
                lambda x: x.expanding().mean().shift(1).fillna(0)
            )
            df[f"{col_prefix}_std_body_size"] = g["body_size_bytes"].transform(
                lambda x: x.expanding().std(ddof=0).shift(1).fillna(0)
            )

            df[f"{col_prefix}_method_diversity"] = g["method"].transform(
                lambda x: pd.Series(_causal_nunique(x.values), index=x.index)
            )
            df[f"{col_prefix}_sensitive_endpoint_ratio"] = g["_is_sensitive"].transform(
                lambda x: x.expanding().mean().shift(1).fillna(0)
            )

            diffs = g["timestamp"].transform(lambda x: x.diff().dt.total_seconds())
            df["_diff"] = diffs
            g_diff = df.groupby("_wk", sort=False)["_diff"]
            df[f"{col_prefix}_inter_request_time_mean"] = g_diff.transform(
                lambda x: x.expanding().mean().fillna(0)
            )
            df[f"{col_prefix}_inter_request_time_std"] = g_diff.transform(
                lambda x: x.expanding().std(ddof=0).fillna(0)
            )

    df = df.drop(
        columns=["_wk", "_ts_epoch", "_is_error", "_is_sensitive", "_diff"],
        errors="ignore",
    )
    return df
