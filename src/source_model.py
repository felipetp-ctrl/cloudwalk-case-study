import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)

SOURCE_FEATURE_COLS = [
    "total_requests",
    "unique_paths",
    "unique_methods",
    "path_concentration",
    "method_per_path",
    "sensitive_ratio",
    "ua_entropy_mean",
    "ua_is_bot",
    "ua_is_browser",
    "header_count_mean",
    "has_cookie_ratio",
    "has_accept_language_ratio",
    "path_entropy_mean",
]

DEFAULT_MIN_REQUESTS = 2


def compute_source_features(
    df: pd.DataFrame,
    min_requests: int = DEFAULT_MIN_REQUESTS,
) -> pd.DataFrame:
    """Aggregate per-request data into per-source behavioral profiles.

    Uses IP-day granularity for training because incident labels are
    available at daily resolution. In production, the same 13 features
    are computed over sliding windows (1min, 5min, 30min) shared with
    the Tier 1 session state — the features are proportions and ratios
    that are invariant to window size.

    Only IPs with at least `min_requests` are included — sources with
    fewer requests fall back to the request-level model. Default is 2:
    DDoS IPs send a median of 4 requests/IP-day, so higher thresholds
    miss most of the attack surface.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["_day"] = df["timestamp"].dt.date

    if "is_sensitive_endpoint" not in df.columns:
        from src.features import _SENSITIVE_PATTERN

        df["is_sensitive_endpoint"] = (
            df["path"].str.contains(_SENSITIVE_PATTERN).astype(int)
        )

    agg = (
        df.groupby(["source_ip", "_day"])
        .agg(
            total_requests=("request_id", "count"),
            unique_paths=("path", "nunique"),
            unique_methods=("method", "nunique"),
            sensitive_ratio=("is_sensitive_endpoint", "mean"),
            ua_entropy_mean=("ua_entropy", "mean"),
            ua_is_bot=("ua_is_bot_library", "mean"),
            ua_is_browser=("ua_is_browser", "mean"),
            header_count_mean=("header_count", "mean"),
            has_cookie_ratio=("has_cookie", "mean"),
            has_accept_language_ratio=("has_accept_language", "mean"),
            path_entropy_mean=("path_entropy", "mean"),
            is_malicious=("is_malicious", "max"),
            attack_class=(
                "attack_class",
                lambda x: x.dropna().mode().iloc[0] if x.dropna().any() else np.nan,
            ),
        )
        .reset_index()
    )

    agg["path_concentration"] = 1.0 / agg["unique_paths"].clip(lower=1)
    agg["method_per_path"] = agg["unique_methods"] / agg["unique_paths"].clip(lower=1)

    agg = agg.rename(columns={"_day": "day"})
    agg = agg[agg["total_requests"] >= min_requests].reset_index(drop=True)

    return agg


def train_source_model(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict | None = None,
) -> object:
    import lightgbm as lgb

    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    default_params = {
        "n_estimators": 300,
        "num_leaves": 31,
        "max_depth": 6,
        "learning_rate": 0.05,
        "scale_pos_weight": n_neg / n_pos if n_pos > 0 else 1.0,
        "random_state": 42,
        "verbose": -1,
        "metric": "average_precision",
    }
    if params:
        default_params.update(params)

    model = lgb.LGBMClassifier(**default_params)
    model.fit(X, y)
    return model


def evaluate_source_model(model: object, X: pd.DataFrame, y: pd.Series) -> dict:
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    return {
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "pr_auc": average_precision_score(y, y_prob),
        "roc_auc": roc_auc_score(y, y_prob),
        "y_prob": y_prob,
    }


def propagate_source_scores(
    requests_df: pd.DataFrame,
    source_scores: pd.DataFrame,
) -> np.ndarray:
    """Map source-level scores back to individual requests.

    Uses IP-day joins for offline evaluation (matching the training
    granularity). In production, scores are applied in real time as
    the source model evaluates IPs across sliding windows.

    Returns an array of scores aligned with requests_df. Requests from
    IPs without a source score (below min_requests threshold) get 0.0.
    """
    df = requests_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["_day"] = df["timestamp"].dt.date

    merged = df.merge(
        source_scores[["source_ip", "day", "source_score"]],
        left_on=["source_ip", "_day"],
        right_on=["source_ip", "day"],
        how="left",
    )
    return merged["source_score"].fillna(0.0).values


def ensemble_scores(
    request_probs: np.ndarray,
    source_probs: np.ndarray,
) -> np.ndarray:
    """Combine request-level and source-level scores (max rule)."""
    return np.maximum(
        np.asarray(request_probs),
        np.asarray(source_probs),
    )
