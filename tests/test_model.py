import pandas as pd
import numpy as np
from sklearn.datasets import make_classification


def _make_sample_df():
    """7 days of data: 10 rows per day, day 2 and 4 have 1 malicious each."""
    rows = []
    for day in range(7):
        date = f"2025-01-{6 + day:02d}"
        for i in range(10):
            rows.append(
                {
                    "request_id": f"d{day}_r{i}",
                    "timestamp": pd.Timestamp(f"{date} {i:02d}:00:00", tz="UTC"),
                    "is_malicious": (day in [1, 3] and i == 0),
                    "attack_class": "scanner" if (day in [1, 3] and i == 0) else None,
                    "sample_weight": 1.0,
                    "feat_a": np.random.rand(),
                    "feat_b": np.random.rand(),
                }
            )
    return pd.DataFrame(rows)


def test_get_feature_columns():
    from src.model import get_feature_columns

    df = _make_sample_df()
    cols = get_feature_columns(df)
    assert "feat_a" in cols
    assert "feat_b" in cols
    assert "request_id" not in cols
    assert "timestamp" not in cols
    assert "is_malicious" not in cols
    assert "attack_class" not in cols
    assert "sample_weight" not in cols
    assert "feat_a" in cols and "feat_b" in cols


def test_temporal_train_test_split():
    from src.model import temporal_train_test_split

    df = _make_sample_df()
    train, test = temporal_train_test_split(df, test_date="2025-01-12")
    assert len(train) == 60
    assert len(test) == 10
    assert train["timestamp"].max() < test["timestamp"].min()


def test_stratified_train_test_split():
    from src.model import stratified_train_test_split

    df = _make_sample_df()
    train, test = stratified_train_test_split(df, test_size=0.3)
    assert len(train) + len(test) == len(df)
    assert train["is_malicious"].sum() > 0
    assert test["is_malicious"].sum() > 0


def test_time_series_cv_splits():
    from src.model import make_time_series_cv_splits

    df = _make_sample_df()
    splits = make_time_series_cv_splits(df, min_train_days=1)
    assert len(splits) >= 2
    for train_idx, val_idx in splits:
        train_max_ts = df.iloc[train_idx]["timestamp"].max()
        val_min_ts = df.iloc[val_idx]["timestamp"].min()
        assert train_max_ts < val_min_ts


def test_time_series_cv_expanding_window():
    from src.model import make_time_series_cv_splits

    df = _make_sample_df()
    splits = make_time_series_cv_splits(df, min_train_days=1)
    train_sizes = [len(train_idx) for train_idx, _ in splits]
    assert train_sizes == sorted(train_sizes)


def test_stratified_cv_splits():
    from src.model import make_stratified_cv_splits

    X, y, _ = _make_classification_data()
    attack_classes = pd.Series([None] * len(y))
    attack_classes[y == 1] = "scanner"
    splits = make_stratified_cv_splits(y, attack_classes, n_splits=3)
    assert len(splits) == 3
    for tr_idx, val_idx in splits:
        assert len(set(tr_idx) & set(val_idx)) == 0
        assert y.iloc[val_idx].sum() > 0


# --- Task 3: Training and evaluation ---


def _make_classification_data():
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        weights=[0.9, 0.1],
        random_state=42,
    )
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
    y = pd.Series(y, name="is_malicious")
    weights = pd.Series(np.ones(len(y)))
    return X, y, weights


def test_train_model_lr():
    from src.model import train_model

    X, y, w = _make_classification_data()
    model = train_model("lr", {}, X, y, w)
    assert hasattr(model, "predict_proba")


def test_train_model_rf():
    from src.model import train_model

    X, y, w = _make_classification_data()
    model = train_model("rf", {"n_estimators": 10}, X, y, w)
    assert hasattr(model, "predict_proba")


def test_train_model_xgb():
    from src.model import train_model

    X, y, w = _make_classification_data()
    model = train_model("xgb", {"n_estimators": 10, "verbosity": 0}, X, y, w)
    assert hasattr(model, "predict_proba")


def test_train_model_lgbm():
    from src.model import train_model

    X, y, w = _make_classification_data()
    model = train_model("lgbm", {"n_estimators": 10, "verbose": -1}, X, y, w)
    assert hasattr(model, "predict_proba")


def test_evaluate_model():
    from src.model import train_model, evaluate_model

    X, y, w = _make_classification_data()
    model = train_model("lr", {}, X, y, w)
    metrics = evaluate_model(model, X, y)
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "fpr" in metrics
    assert "pr_auc" in metrics
    assert "roc_auc" in metrics
    assert "y_prob" in metrics
    assert 0 <= metrics["pr_auc"] <= 1
    assert len(metrics["y_prob"]) == len(y)


def test_evaluate_per_attack_type():
    from src.model import train_model, evaluate_per_attack_type

    X, y, w = _make_classification_data()
    attack_classes = pd.Series([None] * len(y))
    attack_classes[y == 1] = "scanner"
    model = train_model("lr", {}, X, y, w)
    result = evaluate_per_attack_type(model, X, y, attack_classes)
    assert "scanner" in result
    assert 0 <= result["scanner"] <= 1


# --- Task 4: Optuna tuning ---


def test_create_objective():
    from src.model import create_objective, make_time_series_cv_splits

    X, y, w = _make_classification_data()
    timestamps = pd.Series(
        [pd.Timestamp("2025-01-06", tz="UTC")] * 150
        + [pd.Timestamp("2025-01-07", tz="UTC")] * 50
    )
    df_temp = pd.DataFrame({"timestamp": timestamps, "is_malicious": y})
    splits = make_time_series_cv_splits(df_temp, min_train_days=1)
    objective = create_objective("lr", X, y, w, splits)
    assert callable(objective)


def test_tune_model_runs():
    from src.model import tune_model, make_time_series_cv_splits

    X, y, w = _make_classification_data()
    timestamps = pd.Series(
        [pd.Timestamp("2025-01-06", tz="UTC")] * 100
        + [pd.Timestamp("2025-01-07", tz="UTC")] * 50
        + [pd.Timestamp("2025-01-08", tz="UTC")] * 50
    )
    df_temp = pd.DataFrame({"timestamp": timestamps, "is_malicious": y})
    splits = make_time_series_cv_splits(df_temp, min_train_days=1)
    best_params, study = tune_model("lr", X, y, w, splits, n_trials=3)
    assert isinstance(best_params, dict)
    assert study.best_value > 0


# --- Task 5: Feature importance and pruning ---


def test_get_feature_importance_lr():
    from src.model import train_model, get_feature_importance

    X, y, w = _make_classification_data()
    model = train_model("lr", {}, X, y, w)
    imp = get_feature_importance(model, "lr", list(X.columns))
    assert len(imp) == X.shape[1]
    assert imp.iloc[0] >= imp.iloc[-1]


def test_get_feature_importance_lgbm():
    from src.model import train_model, get_feature_importance

    X, y, w = _make_classification_data()
    model = train_model("lgbm", {"n_estimators": 10, "verbose": -1}, X, y, w)
    imp = get_feature_importance(model, "lgbm", list(X.columns))
    assert len(imp) == X.shape[1]
    assert imp.iloc[0] >= imp.iloc[-1]


def test_prune_features():
    from src.model import prune_features

    imp = pd.Series([50, 30, 10, 5, 3, 2], index=[f"f{i}" for i in range(6)])
    kept = prune_features(imp, threshold=0.90)
    assert "f0" in kept
    assert "f1" in kept
    assert "f2" in kept
    assert len(kept) <= 4


def test_prune_features_keeps_minimum():
    from src.model import prune_features

    imp = pd.Series([99, 1], index=["dominant", "weak"])
    kept = prune_features(imp, threshold=0.95)
    assert "dominant" in kept


# --- Task 6: Cost-optimal threshold ---


def test_cost_optimal_threshold():
    from src.model import find_cost_optimal_threshold

    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    y_prob = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9])
    threshold, cost = find_cost_optimal_threshold(
        y_true, y_prob, fp_cost=2.50, fn_cost=0.10
    )
    assert 0 < threshold < 1
    assert cost >= 0


def test_cost_optimal_threshold_high_fp_cost_raises_threshold():
    from src.model import find_cost_optimal_threshold

    np.random.seed(42)
    y_true = np.array([0] * 100 + [1] * 10)
    y_prob = np.concatenate(
        [
            np.linspace(0.0, 0.6, 100),
            np.linspace(0.4, 1.0, 10),
        ]
    )
    t_high_fp, _ = find_cost_optimal_threshold(
        y_true, y_prob, fp_cost=25.0, fn_cost=0.10
    )
    t_low_fp, _ = find_cost_optimal_threshold(
        y_true, y_prob, fp_cost=0.10, fn_cost=25.0
    )
    assert t_high_fp > t_low_fp


def test_evaluate_at_threshold():
    from src.model import train_model, evaluate_at_threshold

    X, y, w = _make_classification_data()
    model = train_model("lr", {}, X, y, w)
    metrics_low = evaluate_at_threshold(model, X, y, threshold=0.1)
    metrics_high = evaluate_at_threshold(model, X, y, threshold=0.9)
    assert metrics_low["recall"] >= metrics_high["recall"]
