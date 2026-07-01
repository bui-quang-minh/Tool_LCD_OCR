"""
FastAPI backend for LCD OCR Inspector.

Startup order:
  1. CORS middleware attached
  2. Prometheus instrumentator attached (/metrics)
  3. Models loaded into memory (det + ocr)
  4. Routers mounted

Run:
  uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator

from config import settings
from routers import predict, training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Custom business metric: readings by verdict (OK / NG / unknown) ──────────
READINGS = Counter(
    "lcd_readings_total", "Total OCR readings by verdict", ["verdict"]
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup: warm up models so the first request isn't slow ──────────────
    logger.info("Loading models...")
    try:
        from services.model_loader import get_detection_model, get_ocr_model
        get_detection_model()
        get_ocr_model()
        logger.info("Models ready.")
    except Exception as exc:
        logger.error(f"Model load failed at startup: {exc}")
    yield
    # ── Shutdown (nothing to clean up for now) ────────────────────────────────


app = FastAPI(
    title="LCD OCR Inspector API",
    description="YOLO OBB detection + OCR pipeline for torque wrench readings",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Prometheus: standard HTTP metrics + /metrics endpoint ─────────────────────
Instrumentator().instrument(app).expose(app)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(predict.router)
app.include_router(training.router)


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["meta"])
async def health():
    from services.model_loader import _det_model, _ocr_model
    return {
        "status": "ok",
        "models": {
            "detection": "loaded" if _det_model is not None else "not loaded",
            "ocr": "loaded" if _ocr_model is not None else "not loaded",
        },
        "mlflow_registry": settings.use_mlflow_registry,
        "mlflow_tracking": settings.use_mlflow_tracking,
    }
