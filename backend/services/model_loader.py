from __future__ import annotations
import logging
from config import settings

logger = logging.getLogger(__name__)

_det_model = None
_ocr_model = None


def _load_local(path):
    from ultralytics import YOLO
    if not path.exists():
        raise FileNotFoundError("Model not found: " + str(path))
    logger.info("Loading model from disk: " + str(path))
    return YOLO(str(path))


def _load_from_mlflow(model_name: str):
    import os
    import mlflow
    from mlflow.artifacts import download_artifacts
    from ultralytics import YOLO

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.MlflowClient()

    mv = client.get_model_version_by_alias(model_name, "Production")
    run_id = mv.run_id
    logger.info("Loading from MLflow registry: " + model_name + " v" + str(mv.version) + " (run " + run_id + ")")

    local_path = download_artifacts(run_id=run_id, artifact_path="model")

    pt_files = [f for f in os.listdir(local_path) if f.endswith(".pt")]
    if not pt_files:
        raise FileNotFoundError("No .pt artifact found under run " + run_id + "/model")

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
        except Exception as exc:
            logger.warning("OCR model not available (" + str(exc) + ") - digit recognition will be unavailable.")
            _ocr_model = None
    return _ocr_model


def reload_models():
    global _det_model, _ocr_model
    _det_model = None
    _ocr_model = None
    get_detection_model()
    get_ocr_model()
    return {
        "detection": "loaded" if _det_model else "failed",
        "ocr": "loaded" if _ocr_model else "failed",
    }
