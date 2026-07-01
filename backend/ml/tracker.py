"""
MLflow inference tracking.

When USE_MLFLOW_TRACKING=false (default) every call is a no-op.
Set USE_MLFLOW_TRACKING=true and point MLFLOW_TRACKING_URI to your server
to start recording inference metrics automatically.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator

from config import settings

logger = logging.getLogger(__name__)


class _NullRun:
    """Drop-in replacement when MLflow is disabled."""
    def log_metric(self, key: str, value: float) -> None: ...
    def log_param(self, key: str, value: str) -> None: ...


class _MlflowRun:
    def __init__(self, run) -> None:
        self._run = run

    def log_metric(self, key: str, value: float) -> None:
        import mlflow
        mlflow.log_metric(key, value)

    def log_param(self, key: str, value: str) -> None:
        import mlflow
        mlflow.log_param(key, value)


@contextmanager
def inference_run(filename: str = "") -> Generator[_NullRun | _MlflowRun, None, None]:
    """
    Context manager that wraps a single inference call in an MLflow run.

    Usage:
        with tracker.inference_run(filename="img.jpg") as run:
            run.log_metric("value_nm", 72.3)
            run.log_param("filter_used", "Grayscale")
    """
    if not settings.use_mlflow_tracking:
        yield _NullRun()
        return

    try:
        import mlflow
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment("lcd-ocr-inference")
        with mlflow.start_run(run_name=f"infer/{filename}"):
            yield _MlflowRun(mlflow.active_run())
    except Exception as exc:
        logger.warning(f"MLflow tracking unavailable: {exc}")
        yield _NullRun()
