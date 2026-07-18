import numpy as np
import pandas as pd
from scipy import stats


def detect_data_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    p_value_threshold: float = 0.05,
) -> pd.DataFrame:
    """Kolmogorov-Smirnov test per feature between reference and current data."""
    results = []
    for col in reference.columns:
        ref_vals = reference[col].dropna().values.astype(float)
        cur_vals = current[col].dropna().values.astype(float)
        if len(ref_vals) == 0 or len(cur_vals) == 0:
            continue
        statistic, p_value = stats.ks_2samp(ref_vals, cur_vals)
        results.append(
            {
                "feature": col,
                "ks_statistic": statistic,
                "p_value": p_value,
                "drifted": p_value < p_value_threshold,
            }
        )
    return pd.DataFrame(results)


def _compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two score distributions."""
    breakpoints = np.linspace(0, 1, bins + 1)
    ref_counts = np.histogram(reference, bins=breakpoints)[0].astype(float)
    cur_counts = np.histogram(current, bins=breakpoints)[0].astype(float)

    ref_pct = (ref_counts + 1e-6) / ref_counts.sum()
    cur_pct = (cur_counts + 1e-6) / cur_counts.sum()

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def monitor_predictions(
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
    alert_threshold: float = 0.5,
) -> dict:
    """Analyze prediction score distribution shift between reference and current."""
    ref = np.asarray(reference_scores)
    cur = np.asarray(current_scores)
    return {
        "psi": _compute_psi(ref, cur),
        "mean_score_reference": float(ref.mean()),
        "mean_score_current": float(cur.mean()),
        "std_score_reference": float(ref.std()),
        "std_score_current": float(cur.std()),
        "alert_rate_reference": float((ref >= alert_threshold).mean()),
        "alert_rate_current": float((cur >= alert_threshold).mean()),
        "ks_statistic": float(stats.ks_2samp(ref, cur).statistic),
        "ks_p_value": float(stats.ks_2samp(ref, cur).pvalue),
    }


def track_performance(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    window_size: int,
    baseline_metrics: dict,
    threshold: float = 0.5,
    degradation_pct: float = 0.10,
) -> pd.DataFrame:
    """Simulate sliding-window performance tracking and detect degradation.

    Splits y_true/y_prob into non-overlapping windows, computes precision/recall/FPR
    per window, and flags windows where metrics drop below (1 - degradation_pct) * baseline.
    """
    from sklearn.metrics import precision_score, recall_score, confusion_matrix

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    records = []

    for start in range(0, n, window_size):
        end = min(start + window_size, n)
        yt = y_true[start:end]
        yp = (y_prob[start:end] >= threshold).astype(int)

        if len(np.unique(yt)) < 2:
            continue

        tn, fp, fn, tp = confusion_matrix(yt, yp).ravel()
        prec = precision_score(yt, yp, zero_division=0)
        rec = recall_score(yt, yp, zero_division=0)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        degraded = (
            prec < baseline_metrics["precision"] * (1 - degradation_pct)
            or rec < baseline_metrics["recall"] * (1 - degradation_pct)
            or fpr > baseline_metrics["fpr"] * (1 + degradation_pct) + 0.01
        )

        records.append(
            {
                "window_start": start,
                "window_end": end,
                "precision": prec,
                "recall": rec,
                "fpr": fpr,
                "degraded": degraded,
            }
        )

    return pd.DataFrame(records)


def generate_monitoring_report(
    drift_results: pd.DataFrame,
    prediction_report: dict,
    performance_windows: pd.DataFrame,
) -> dict:
    """Consolidate monitoring signals into a single report."""
    n_drifted = int(drift_results["drifted"].sum()) if len(drift_results) > 0 else 0
    n_features = len(drift_results)
    top_drifted = (
        drift_results[drift_results["drifted"]]
        .sort_values("ks_statistic", ascending=False)
        .head(5)["feature"]
        .tolist()
    )

    n_degraded = (
        int(performance_windows["degraded"].sum())
        if len(performance_windows) > 0
        else 0
    )
    n_windows = len(performance_windows)

    psi = prediction_report.get("psi", 0.0)
    psi_status = (
        "stable"
        if psi < 0.1
        else ("moderate_shift" if psi < 0.25 else "significant_shift")
    )

    needs_action = (
        n_drifted > n_features * 0.3 or psi >= 0.25 or n_degraded > n_windows * 0.2
    )

    return {
        "data_drift": {
            "features_drifted": n_drifted,
            "total_features": n_features,
            "drift_rate": round(n_drifted / n_features, 3) if n_features > 0 else 0.0,
            "top_drifted_features": top_drifted,
        },
        "prediction_stability": {
            "psi": round(psi, 4),
            "status": psi_status,
            "alert_rate_shift": round(
                prediction_report.get("alert_rate_current", 0)
                - prediction_report.get("alert_rate_reference", 0),
                4,
            ),
        },
        "performance": {
            "windows_degraded": n_degraded,
            "total_windows": n_windows,
            "degradation_rate": round(n_degraded / n_windows, 3)
            if n_windows > 0
            else 0.0,
        },
        "needs_retraining": needs_action,
    }
