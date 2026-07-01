from __future__ import annotations

import logging

from config import settings

logger = logging.getLogger(__name__)

_det_model = None
_ocr_model = None


def _load_local(path):
    from ultralytics import YOLO
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    logger.info(f"Loading model from disk: {path}")
    return YOLO(str(path))


def _load_from_mlflow(model_name: str):
    """Download the raw .pt artifact registered under the Production alias
    and load it with YOLO — the training DAG logs weights as a raw artifact,
    not as an mlflow.pytorch flavor, so we mirror that here."""
    import mlflow
    from ultralytics import YOLO

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.MlflowClient()

    mv = client.get_model_version_by_alias(model_name, "Production")
    run_id = mv.run_id
    logger.info(f"Loading from MLflow registry: {model_name} v{mv.version} (run {run_id})")

    local_path = client.download_artifacts(run_id, "model")
    import os
    pt_files = [f for f in os.listdir(local_path) if f.endswith(".pt")]
    if not pt_files:
        raise FileNotFoundError(f"No .pt artifact found under run {run_id}/model")

    weights_path = os.path.join(local_path, pt_files[0])
    return YOLO(weights_path)


def get_detection_model():
    global _det_model
    if _det_model is None:
        if settings.use_mlflow_registry:
            _det_model = _load_from_mlflow(settings.mlflow_detection_model_name)
        else:
            _det_model = _load_local(settings.detection_model)
    return _det_model


def get_ocr_model():
    global _ocr_model
    if _ocr_model is None:
        try:
            if settings.use_mlflow_registry:
                _ocr_model = _load_from_mlflow(settings.mlflow_ocr_model_name)
            else:
                _ocr_model = _load_local(settings.ocr_model)
        except FileNotFoundError:
            logger.warning("OCR model not found — digit recognition will be unavailable.")
            _ocr_model = None
    return _ocr_model


def reload_models() -> dict[str, str]:
    global _det_model, _ocr_model
    _det_model = None
    _ocr_model = None
    get_detection_model()
    get_ocr_model()
    return {
        "detection": "loaded" if _det_model else "failed",
        "ocr": "loaded" if _ocr_model else "failed",
    }
