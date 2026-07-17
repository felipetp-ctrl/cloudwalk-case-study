import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score,
    confusion_matrix,
)

META_COLUMNS = [
    'request_id', 'timestamp', 'is_malicious', 'attack_class', 'sample_weight', 'day',
    'source_ip', 'method', 'path', 'status_code', 'user_agent', 'tls_fingerprint',
    'country', 'asn', 'confidence', 'incident_id', 'source_identifier',
    'response_time_ms', 'body_size_bytes', 'status_code_group',
]

EXCLUDE_SUFFIXES = ('_freq',)

RESPONSE_SESSION_SUFFIXES = (
    '_error_rate', '_unique_status_codes',
    '_avg_response_time', '_std_response_time',
    '_avg_body_size', '_std_body_size',
)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c not in META_COLUMNS
        and not c.endswith(EXCLUDE_SUFFIXES)
        and not any(c.endswith(s) for s in RESPONSE_SESSION_SUFFIXES)
    ]


def temporal_train_test_split(
    df: pd.DataFrame, test_date: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ts = pd.to_datetime(df['timestamp'], utc=True)
    cutoff = pd.Timestamp(test_date, tz='UTC')
    train = df[ts < cutoff].copy().reset_index(drop=True)
    test = df[ts >= cutoff].copy().reset_index(drop=True)
    return train, test


def stratified_train_test_split(
    df: pd.DataFrame, test_size: float = 0.3, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split
    label = df['attack_class'].fillna('benign')
    train, test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=label,
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


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


def make_stratified_cv_splits(
    y: pd.Series, attack_classes: pd.Series, n_splits: int = 5, random_state: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import StratifiedKFold
    label = attack_classes.fillna('benign')
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(skf.split(y, label))


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
        lr_params = {
            'solver': 'saga',
            'l1_ratio': 0.5,
        }
        lr_params.update(params)
        model = LogisticRegression(
            class_weight='balanced',
            max_iter=2000,
            penalty='elasticnet',
            **lr_params,
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


import optuna


def _suggest_params(trial: optuna.Trial, model_name: str) -> dict:
    if model_name == 'lr':
        l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
        return {
            'C': trial.suggest_float('C', 1e-4, 1e2, log=True),
            'l1_ratio': l1_ratio,
            'solver': 'saga',
        }
    elif model_name == 'rf':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 50),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        }
    elif model_name == 'xgb':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 800),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 50, 150),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
            'verbosity': 0,
            'eval_metric': 'aucpr',
        }
    elif model_name == 'lgbm':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 800),
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 50, 150),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
            'verbose': -1,
            'metric': 'average_precision',
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")


def create_objective(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weights: pd.Series,
    cv_splits: list,
) -> callable:
    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, model_name)
        scores = []
        for train_idx, val_idx in cv_splits:
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            w_train = sample_weights.iloc[train_idx]
            if y_val.sum() == 0 or y_train.nunique() < 2:
                continue
            model = train_model(model_name, params, X_train, y_train, w_train)
            metrics = evaluate_model(model, X_val, y_val)
            scores.append(metrics['pr_auc'])
        return np.mean(scores) if scores else 0.0
    return objective


def tune_model(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weights: pd.Series,
    cv_splits: list,
    n_trials: int = 50,
) -> tuple[dict, optuna.study.Study]:
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(),
    )
    objective = create_objective(model_name, X, y, sample_weights, cv_splits)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params, study


def get_feature_importance(
    model: object, model_name: str, feature_names: list[str]
) -> pd.Series:
    if model_name == 'lr':
        imp = np.abs(model.coef_[0])
    elif model_name in ('rf', 'xgb', 'lgbm'):
        imp = model.feature_importances_
    else:
        raise ValueError(f"Unknown model: {model_name}")
    series = pd.Series(imp, index=feature_names)
    return series.sort_values(ascending=False)


def prune_features(
    importance: pd.Series, threshold: float = 0.95
) -> list[str]:
    total = importance.sum()
    if total == 0:
        return list(importance.index)
    normalized = importance / total
    cumsum = normalized.cumsum()
    mask = cumsum.shift(1, fill_value=0.0) < threshold
    return list(importance.index[mask])


def find_cost_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fp_cost: float,
    fn_cost: float,
    base_rate: float | None = None,
) -> tuple[float, float]:
    if base_rate is None:
        base_rate = y_true.mean()
    thresholds = np.linspace(0.01, 0.99, 500)
    best_threshold = 0.5
    best_cost = float('inf')
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        cost = fpr * (1 - base_rate) * fp_cost + fnr * base_rate * fn_cost
        if cost < best_cost:
            best_cost = cost
            best_threshold = t
    return best_threshold, best_cost


def evaluate_at_threshold(
    model: object, X: pd.DataFrame, y: pd.Series, threshold: float
) -> dict:
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    return {
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        'threshold': threshold,
    }
