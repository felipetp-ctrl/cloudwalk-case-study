import numpy as np
import pandas as pd
import onnxruntime as ort
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType


def export_to_onnx(model, feature_names: list[str], output_path: str) -> str:
    initial_type = [("features", FloatTensorType([None, len(feature_names)]))]
    onnx_model = onnxmltools.convert_lightgbm(
        model,
        initial_types=initial_type,
        zipmap=False,
    )
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    return output_path


def validate_onnx_export(model, onnx_path: str, X_sample: pd.DataFrame) -> dict:
    original_probs = model.predict_proba(X_sample)[:, 1]

    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    X_float = X_sample.values.astype(np.float32)
    onnx_output = session.run(None, {input_name: X_float})
    onnx_probs = onnx_output[1][:, 1]

    abs_errors = np.abs(original_probs - onnx_probs)
    max_err = float(abs_errors.max())
    return {
        "max_abs_error": max_err,
        "mean_abs_error": float(abs_errors.mean()),
        "match": max_err < 1e-6,
    }
