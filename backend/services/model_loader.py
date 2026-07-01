"""
Model loading with two backends:
  - Local disk  (default, USE_MLFLOW_REGISTRY=false) → backend/models/
  - MLflow registry Production alias (USE_MLFLOW_REGISTRY=true)

Models are loaded once at startup and cached as module-level singletons.
"""
from __future__ import annotations

import logging

from config import settings

logger = logging.getLogger(__name__)

_det_model = None
_ocr_model = None


# ── Loaders ──────────────────────────────────────────────────────────────────

def _load_local(path):
    from ultralytics import YOLO
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    logger.info(f"Loading model from disk: {path}")
    return YOLO(str(path))


def _load_from_mlflow(model_name: str):
    """Load a model from the MLflow registry Production alias."""
    import mlflow.pytorch
    uri = f"models:/{model_name}/Production"
    logger.info(f"Loading from MLflow registry: {uri}")
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    return mlflow.pytorch.load_model(uri)


# ── Public API ────────────────────────────────────────────────────────────────

def get_detection_model():
    """Return the cached detection model, loading it on first call."""
    global _det_model
    if _det_model is None:
        if settings.use_mlflow_registry:
            _det_model = _load_from_mlflow(settings.mlflow_detection_model_name)
        else:
            _det_model = _load_local(settings.detection_model)
    return _det_model


def get_ocr_model():
    """Return the cached OCR model, loading it on first call."""
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
    """Force-reload both models (called after a training run completes)."""
    global _det_model, _ocr_model
    _det_model = None
    _ocr_model = None
    get_detection_model()
    get_ocr_model()
    return {
        "detection": "loaded" if _det_model else "failed",
        "ocr": "loaded" if _ocr_model else "failed",
    }
