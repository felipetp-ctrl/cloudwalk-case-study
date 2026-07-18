import pandas as pd

from src.label_joining import join_labels
from src.features import compute_per_request_features, compute_session_features

CONFIDENCE_WEIGHTS = {"high": 1.0, "medium": 0.6, "low": 0.3}


def compute_sample_weights(df: pd.DataFrame) -> pd.Series:
    weights = df["confidence"].map(CONFIDENCE_WEIGHTS)
    return weights.fillna(1.0)


def build_training_dataset(
    requests_path: str,
    headers_path: str,
    labels_path: str,
) -> pd.DataFrame:
    requests = pd.read_csv(requests_path, parse_dates=["timestamp"])
    headers = pd.read_csv(headers_path)
    labels = pd.read_csv(
        labels_path, parse_dates=["active_from", "active_until", "labeled_at"]
    )

    df = join_labels(requests, labels)
    df = compute_per_request_features(df, headers)
    df = compute_session_features(df)
    df["sample_weight"] = compute_sample_weights(df)

    return df
