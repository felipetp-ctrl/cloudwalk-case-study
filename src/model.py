import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score,
    confusion_matrix,
)

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


def _count_class_ratio(y: pd.Series) -> float:
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    return n_neg / n_pos if n_pos > 0 else 1.0


def train_model(
    model_name: str,
    params: dict,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weights: pd.Series | None = None,
) -> object:
    if model_name == 'lr':
        model = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            **params,
        )
    elif model_name == 'rf':
        model = RandomForestClassifier(
            class_weight='balanced',
            random_state=42,
            **params,
        )
    elif model_name == 'xgb':
        import xgboost as xgb
        default_params = {
            'scale_pos_weight': _count_class_ratio(y),
            'random_state': 42,
            'eval_metric': 'aucpr',
        }
        default_params.update(params)
        model = xgb.XGBClassifier(**default_params)
    elif model_name == 'lgbm':
        import lightgbm as lgb
        default_params = {
            'scale_pos_weight': _count_class_ratio(y),
            'random_state': 42,
            'metric': 'average_precision',
        }
        default_params.update(params)
        model = lgb.LGBMClassifier(**default_params)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    fit_params = {}
    if sample_weights is not None:
        fit_params['sample_weight'] = sample_weights.values
    model.fit(X, y, **fit_params)
    return model


def evaluate_model(
    model: object, X: pd.DataFrame, y: pd.Series
) -> dict:
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    return {
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        'pr_auc': average_precision_score(y, y_prob),
        'roc_auc': roc_auc_score(y, y_prob),
        'y_prob': y_prob,
    }


def evaluate_per_attack_type(
    model: object,
    X: pd.DataFrame,
    y: pd.Series,
    attack_classes: pd.Series,
) -> dict:
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    results = {}
    for cls in attack_classes.dropna().unique():
        mask = attack_classes == cls
        if mask.sum() == 0:
            continue
        results[cls] = recall_score(y[mask], y_pred[mask], zero_division=0)
    return results
