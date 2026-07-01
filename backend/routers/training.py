"""
Training management endpoints.

Today these are stubs that return 501.
When Airflow and MLflow are integrated:
  - POST /training/trigger  → calls Airflow REST API to start lcd_ocr_training DAG
  - GET  /training/runs     → lists experiments from MLflow tracking server
  - POST /training/reload   → hot-reloads models from MLflow registry after a run

Wiring checklist:
  1. Set AIRFLOW_API_URL / AIRFLOW_USERNAME / AIRFLOW_PASSWORD in .env
  2. Set MLFLOW_TRACKING_URI and USE_MLFLOW_REGISTRY=true in .env
  3. Uncomment the implementation blocks below
"""
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/training", tags=["training"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class TriggerRequest(BaseModel):
    dag_run_id: str | None = None   # optional custom run ID
    conf: dict = {}                 # extra config forwarded to the DAG


class TriggerResponse(BaseModel):
    dag_run_id: str
    state: str


class RunSummary(BaseModel):
    run_id: str
    experiment: str
    status: str
    metrics: dict


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/trigger", response_model=TriggerResponse)
async def trigger_training(body: TriggerRequest = TriggerRequest()) -> TriggerResponse:
    """
    Trigger the Airflow training DAG (lcd_ocr_training).

    Runs the full pipeline:
      augment data → train detection model → train OCR model → promote to Production
    """
    # ── Airflow integration (uncomment when ready) ────────────────────────────
    # import httpx
    # url = f"{settings.airflow_api_url}/dags/{settings.airflow_dag_id}/dagRuns"
    # payload = {"conf": body.conf}
    # if body.dag_run_id:
    #     payload["dag_run_id"] = body.dag_run_id
    # async with httpx.AsyncClient() as client:
    #     resp = await client.post(
    #         url, json=payload,
    #         auth=(settings.airflow_username, settings.airflow_password),
    #         timeout=10,
    #     )
    #     resp.raise_for_status()
    #     data = resp.json()
    # return TriggerResponse(dag_run_id=data["dag_run_id"], state=data["state"])
    # ─────────────────────────────────────────────────────────────────────────

    raise HTTPException(
        status_code=501,
        detail=(
            "Airflow not configured. "
            "Set AIRFLOW_API_URL in .env and uncomment the integration block "
            "in routers/training.py."
        ),
    )


@router.get("/runs", response_model=list[RunSummary])
async def list_runs(experiment: str = "lcd-ocr-detection", limit: int = 10):
    """List recent training runs from MLflow."""
    # ── MLflow integration (uncomment when ready) ─────────────────────────────
    # import mlflow
    # mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    # client = mlflow.MlflowClient()
    # exp = client.get_experiment_by_name(experiment)
    # if not exp:
    #     return []
    # runs = client.search_runs(
    #     experiment_ids=[exp.experiment_id],
    #     order_by=["start_time DESC"],
    #     max_results=limit,
    # )
    # return [
    #     RunSummary(
    #         run_id=r.info.run_id,
    #         experiment=experiment,
    #         status=r.info.status,
    #         metrics=r.data.metrics,
    #     )
    #     for r in runs
    # ]
    # ─────────────────────────────────────────────────────────────────────────

    raise HTTPException(
        status_code=501,
        detail="MLflow not configured. Set MLFLOW_TRACKING_URI in .env.",
    )


@router.post("/reload")
async def reload_models():
    """
    Hot-reload models from MLflow registry after a training run completes.
    Call this from your Airflow DAG's final task (or a webhook).
    """
    # ── Uncomment when USE_MLFLOW_REGISTRY=true ───────────────────────────────
    # from services.model_loader import reload_models as _reload
    # result = _reload()
    # return {"status": "reloaded", "models": result}
    # ─────────────────────────────────────────────────────────────────────────

    raise HTTPException(
        status_code=501,
        detail="Model hot-reload requires USE_MLFLOW_REGISTRY=true.",
    )


@router.get("/dag-status/{dag_run_id}")
async def dag_status(dag_run_id: str):
    """Check the status of an Airflow DAG run."""
    # ── Airflow integration (uncomment when ready) ────────────────────────────
    # import httpx
    # url = f"{settings.airflow_api_url}/dags/{settings.airflow_dag_id}/dagRuns/{dag_run_id}"
    # async with httpx.AsyncClient() as client:
    #     resp = await client.get(
    #         url, auth=(settings.airflow_username, settings.airflow_password)
    #     )
    #     resp.raise_for_status()
    #     return resp.json()
    # ─────────────────────────────────────────────────────────────────────────

    raise HTTPException(status_code=501, detail="Airflow not configured.")
