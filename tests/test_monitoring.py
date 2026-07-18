import numpy as np
import pandas as pd
from src.monitoring import (
    detect_data_drift,
    _compute_psi,
    monitor_predictions,
    track_performance,
    generate_monitoring_report,
)


def _make_feature_df(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    return pd.DataFrame(
        {
            "feat_a": rng.normal(0, 1, n),
            "feat_b": rng.uniform(0, 1, n),
            "feat_c": rng.exponential(1, n),
        }
    )


class TestDetectDataDrift:
    def test_no_drift_same_distribution(self):
        ref = _make_feature_df(500, seed=1)
        cur = _make_feature_df(500, seed=2)
        result = detect_data_drift(ref, cur)
        assert len(result) == 3
        assert result["drifted"].sum() == 0

    def test_detects_mean_shift(self):
        ref = _make_feature_df(500, seed=1)
        cur = ref.copy()
        cur["feat_a"] = cur["feat_a"] + 5
        result = detect_data_drift(ref, cur)
        drifted = result[result["drifted"]]
        assert "feat_a" in drifted["feature"].values

    def test_detects_variance_change(self):
        ref = _make_feature_df(500, seed=1)
        cur = ref.copy()
        cur["feat_b"] = cur["feat_b"] * 10
        result = detect_data_drift(ref, cur)
        drifted = result[result["drifted"]]
        assert "feat_b" in drifted["feature"].values

    def test_custom_threshold(self):
        ref = _make_feature_df(500, seed=1)
        cur = ref.copy()
        cur["feat_c"] = cur["feat_c"] + 0.5
        strict = detect_data_drift(ref, cur, p_value_threshold=0.001)
        lenient = detect_data_drift(ref, cur, p_value_threshold=0.5)
        assert lenient["drifted"].sum() >= strict["drifted"].sum()


class TestComputePSI:
    def test_identical_distributions(self):
        scores = np.random.RandomState(42).uniform(0, 1, 1000)
        psi = _compute_psi(scores, scores)
        assert psi < 0.01

    def test_shifted_distribution(self):
        ref = np.random.RandomState(42).uniform(0, 0.5, 1000)
        cur = np.random.RandomState(42).uniform(0.5, 1.0, 1000)
        psi = _compute_psi(ref, cur)
        assert psi > 0.25


class TestMonitorPredictions:
    def test_stable_predictions(self):
        rng = np.random.RandomState(42)
        ref = rng.uniform(0, 0.3, 1000)
        cur = rng.uniform(0, 0.3, 1000)
        result = monitor_predictions(ref, cur)
        assert result["psi"] < 0.1
        assert abs(result["alert_rate_reference"] - result["alert_rate_current"]) < 0.05

    def test_detects_score_inflation(self):
        rng = np.random.RandomState(42)
        ref = rng.uniform(0, 0.3, 1000)
        cur = rng.uniform(0.4, 0.9, 1000)
        result = monitor_predictions(ref, cur)
        assert result["psi"] > 0.25
        assert result["alert_rate_current"] > result["alert_rate_reference"]

    def test_report_keys(self):
        ref = np.array([0.1, 0.2, 0.3])
        cur = np.array([0.4, 0.5, 0.6])
        result = monitor_predictions(ref, cur)
        expected_keys = {
            "psi",
            "mean_score_reference",
            "mean_score_current",
            "std_score_reference",
            "std_score_current",
            "alert_rate_reference",
            "alert_rate_current",
            "ks_statistic",
            "ks_p_value",
        }
        assert set(result.keys()) == expected_keys


class TestTrackPerformance:
    def _make_data(self, n=200, seed=42):
        rng = np.random.RandomState(seed)
        y_true = rng.choice([0, 1], size=n, p=[0.9, 0.1])
        y_prob = np.where(
            y_true == 1, rng.uniform(0.6, 1.0, n), rng.uniform(0.0, 0.4, n)
        )
        return y_true, y_prob

    def test_no_degradation(self):
        y_true, y_prob = self._make_data()
        baseline = {"precision": 0.5, "recall": 0.5, "fpr": 0.05}
        result = track_performance(
            y_true, y_prob, window_size=50, baseline_metrics=baseline
        )
        assert len(result) > 0
        assert result["degraded"].sum() == 0

    def test_detects_degradation(self):
        y_true, y_prob = self._make_data(200)
        y_prob[100:] = np.random.RandomState(99).uniform(0, 1, 100)
        baseline = {"precision": 0.95, "recall": 0.95, "fpr": 0.0}
        result = track_performance(
            y_true, y_prob, window_size=50, baseline_metrics=baseline
        )
        assert result["degraded"].any()

    def test_window_boundaries(self):
        y_true, y_prob = self._make_data(100)
        baseline = {"precision": 0.0, "recall": 0.0, "fpr": 1.0}
        result = track_performance(
            y_true, y_prob, window_size=25, baseline_metrics=baseline
        )
        assert all(result["window_end"] <= 100)


class TestGenerateMonitoringReport:
    def test_healthy_report(self):
        drift = pd.DataFrame(
            {
                "feature": ["a", "b", "c"],
                "ks_statistic": [0.05, 0.03, 0.04],
                "p_value": [0.6, 0.8, 0.7],
                "drifted": [False, False, False],
            }
        )
        pred = {"psi": 0.02, "alert_rate_reference": 0.01, "alert_rate_current": 0.012}
        perf = pd.DataFrame(
            {
                "window_start": [0, 50],
                "window_end": [50, 100],
                "precision": [0.95, 0.93],
                "recall": [0.90, 0.88],
                "fpr": [0.01, 0.02],
                "degraded": [False, False],
            }
        )
        report = generate_monitoring_report(drift, pred, perf)
        assert report["needs_retraining"] is False
        assert report["prediction_stability"]["status"] == "stable"
        assert report["data_drift"]["features_drifted"] == 0

    def test_unhealthy_triggers_retraining(self):
        drift = pd.DataFrame(
            {
                "feature": ["a", "b", "c"],
                "ks_statistic": [0.8, 0.7, 0.6],
                "p_value": [0.001, 0.002, 0.003],
                "drifted": [True, True, True],
            }
        )
        pred = {"psi": 0.35, "alert_rate_reference": 0.01, "alert_rate_current": 0.15}
        perf = pd.DataFrame(
            {
                "window_start": [0],
                "window_end": [50],
                "precision": [0.5],
                "recall": [0.3],
                "fpr": [0.1],
                "degraded": [True],
            }
        )
        report = generate_monitoring_report(drift, pred, perf)
        assert report["needs_retraining"] is True
        assert report["prediction_stability"]["status"] == "significant_shift"
        assert report["data_drift"]["drift_rate"] == 1.0
