import pandas as pd
import numpy as np

META_COLUMNS = ['request_id', 'timestamp', 'is_malicious', 'attack_class', 'sample_weight']


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLUMNS]


def temporal_train_test_split(
    df: pd.DataFrame, test_date: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ts = pd.to_datetime(df['timestamp'], utc=True)
    cutoff = pd.Timestamp(test_date, tz='UTC')
    train = df[ts < cutoff].copy()
    test = df[ts >= cutoff].copy()
    return train, test


def make_time_series_cv_splits(
    df: pd.DataFrame, min_train_days: int = 1
) -> list[tuple[np.ndarray, np.ndarray]]:
    ts = pd.to_datetime(df['timestamp'], utc=True)
    dates = sorted(ts.dt.date.unique())
    splits = []
    for i in range(min_train_days, len(dates)):
        train_mask = ts.dt.date.isin(dates[:i])
        val_mask = ts.dt.date == dates[i]
        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
        if len(val_idx) == 0:
            continue
        splits.append((train_idx, val_idx))
    return splits
