import pandas as pd
import numpy as np


def _make_sample_df():
    """7 days of data: 10 rows per day, day 2 and 4 have 1 malicious each."""
    rows = []
    for day in range(7):
        date = f"2025-01-{6 + day:02d}"
        for i in range(10):
            rows.append({
                'request_id': f'd{day}_r{i}',
                'timestamp': pd.Timestamp(f'{date} {i:02d}:00:00', tz='UTC'),
                'is_malicious': (day in [1, 3] and i == 0),
                'attack_class': 'scanner' if (day in [1, 3] and i == 0) else None,
                'sample_weight': 1.0,
                'feat_a': np.random.rand(),
                'feat_b': np.random.rand(),
            })
    return pd.DataFrame(rows)


def test_get_feature_columns():
    from src.model import get_feature_columns
    df = _make_sample_df()
    cols = get_feature_columns(df)
    assert 'feat_a' in cols
    assert 'feat_b' in cols
    assert 'request_id' not in cols
    assert 'timestamp' not in cols
    assert 'is_malicious' not in cols
    assert 'attack_class' not in cols
    assert 'sample_weight' not in cols
    assert len(cols) == 2


def test_temporal_train_test_split():
    from src.model import temporal_train_test_split
    df = _make_sample_df()
    train, test = temporal_train_test_split(df, test_date='2025-01-12')
    assert len(train) == 60
    assert len(test) == 10
    assert train['timestamp'].max() < test['timestamp'].min()


def test_time_series_cv_splits():
    from src.model import make_time_series_cv_splits
    df = _make_sample_df()
    splits = make_time_series_cv_splits(df, min_train_days=1)
    assert len(splits) >= 2
    for train_idx, val_idx in splits:
        train_max_ts = df.iloc[train_idx]['timestamp'].max()
        val_min_ts = df.iloc[val_idx]['timestamp'].min()
        assert train_max_ts < val_min_ts


def test_time_series_cv_expanding_window():
    from src.model import make_time_series_cv_splits
    df = _make_sample_df()
    splits = make_time_series_cv_splits(df, min_train_days=1)
    train_sizes = [len(train_idx) for train_idx, _ in splits]
    assert train_sizes == sorted(train_sizes)


# --- Task 3: Training and evaluation ---

from sklearn.datasets import make_classification


def _make_classification_data():
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5,
        weights=[0.9, 0.1], random_state=42,
    )
    X = pd.DataFrame(X, columns=[f'f{i}' for i in range(10)])
    y = pd.Series(y, name='is_malicious')
    weights = pd.Series(np.ones(len(y)))
    return X, y, weights


def test_train_model_lr():
    from src.model import train_model
    X, y, w = _make_classification_data()
    model = train_model('lr', {}, X, y, w)
    assert hasattr(model, 'predict_proba')


def test_train_model_rf():
    from src.model import train_model
    X, y, w = _make_classification_data()
    model = train_model('rf', {'n_estimators': 10}, X, y, w)
    assert hasattr(model, 'predict_proba')


def test_train_model_xgb():
    from src.model import train_model
    X, y, w = _make_classification_data()
    model = train_model('xgb', {'n_estimators': 10, 'verbosity': 0}, X, y, w)
    assert hasattr(model, 'predict_proba')


def test_train_model_lgbm():
    from src.model import train_model
    X, y, w = _make_classification_data()
    model = train_model('lgbm', {'n_estimators': 10, 'verbose': -1}, X, y, w)
    assert hasattr(model, 'predict_proba')


def test_evaluate_model():
    from src.model import train_model, evaluate_model
    X, y, w = _make_classification_data()
    model = train_model('lr', {}, X, y, w)
    metrics = evaluate_model(model, X, y)
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert 'fpr' in metrics
    assert 'pr_auc' in metrics
    assert 'roc_auc' in metrics
    assert 'y_prob' in metrics
    assert 0 <= metrics['pr_auc'] <= 1
    assert len(metrics['y_prob']) == len(y)


def test_evaluate_per_attack_type():
    from src.model import train_model, evaluate_per_attack_type
    X, y, w = _make_classification_data()
    attack_classes = pd.Series([None] * len(y))
    attack_classes[y == 1] = 'scanner'
    model = train_model('lr', {}, X, y, w)
    result = evaluate_per_attack_type(model, X, y, attack_classes)
    assert 'scanner' in result
    assert 0 <= result['scanner'] <= 1
