import os
import tempfile

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification


def _make_lgbm_model():
    from src.model import train_model

    X, y = make_classification(
        n_samples=200,
        n_features=8,
        n_informative=5,
        weights=[0.9, 0.1],
        random_state=42,
    )
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    y = pd.Series(y)
    w = pd.Series(np.ones(len(y)))
    model = train_model(
        "lgbm",
        {"n_estimators": 10, "verbose": -1, "metric": "average_precision"},
        X,
        y,
        w,
    )
    return model, X, list(X.columns)


def test_export_to_onnx_creates_file():
    from src.export import export_to_onnx

    model, X, feature_names = _make_lgbm_model()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.onnx")
        result = export_to_onnx(model, feature_names, path)
        assert result == path
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


def test_validate_onnx_export_matches():
    from src.export import export_to_onnx, validate_onnx_export

    model, X, feature_names = _make_lgbm_model()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.onnx")
        export_to_onnx(model, feature_names, path)
        result = validate_onnx_export(model, path, X)
        assert result["match"] is True
        assert result["max_abs_error"] < 1e-6
        assert "mean_abs_error" in result
