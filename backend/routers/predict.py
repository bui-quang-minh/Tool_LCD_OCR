import logging

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from ml import tracker
from services.model_loader import get_detection_model, get_ocr_model
from services.pipeline import run_full_pipeline

logger = logging.getLogger(__name__)
router = APIRouter(tags=["inference"])


class PredictResponse(BaseModel):
    reading: str
    verdict: str               # "OK" | "NG" | "unknown"
    value_nm: float | None
    n_tries: int
    filter_used: str
    annotated_image: str       # data:image/jpeg;base64,...
    lcd_crop: str | None


@router.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    """
    Receive an image, run the LCD OCR pipeline, return the result.

    - Detects LCD display via YOLO OBB
    - Crops and rotates the LCD region
    - Runs OCR with up to 7 preprocessing filters
    - Validates reading against 60–80 Nm range
    """
    # ── Decode image ──────────────────────────────────────────────────────────
    raw = await file.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=422, detail="Could not decode image.")

    # ── Load models (cached after first call) ─────────────────────────────────
    try:
        det_model = get_detection_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    ocr_model = get_ocr_model()  # may be None — pipeline degrades gracefully

    # ── Run pipeline ──────────────────────────────────────────────────────────
    filename = file.filename or "upload"
    with tracker.inference_run(filename) as run:
        result = run_full_pipeline(det_model, ocr_model, bgr)
        run.log_param("filter_used", result.filter_used)
        run.log_param("verdict", result.verdict)
        if result.value_nm is not None:
            run.log_metric("value_nm", result.value_nm)
        run.log_metric("n_tries", result.n_tries)

    logger.info(
        "predict file=%s reading=%s verdict=%s tries=%d filter=%s",
        filename, result.reading, result.verdict, result.n_tries, result.filter_used,
    )

    return PredictResponse(
        reading=result.reading,
        verdict=result.verdict,
        value_nm=result.value_nm,
        n_tries=result.n_tries,
        filter_used=result.filter_used,
        annotated_image=result.annotated_image,
        lcd_crop=result.lcd_crop,
    )
