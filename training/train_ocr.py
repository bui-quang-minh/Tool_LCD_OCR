"""
Fine-tune the LCD OCR digit model (YOLO OBB) on a small dataset,
log the run to MLflow, and register a new model version if it
meets a minimum quality bar.
Designed to run on CPU in a reasonable time: few epochs, small image size.

Mirrors train_detection.py exactly; only the target (digits vs lcd/torque),
default paths, and env var names differ.
"""
import os
import shutil
import sys
import time
import mlflow
from ultralytics import YOLO

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.environ.get("MLFLOW_OCR_MODEL_NAME", "lcd-ocr-digits")
BASE_MODEL = os.environ.get("BASE_OCR_MODEL", "/opt/mlops/lcd_dataset/base_ocr.pt")
DATA_YAML = "/opt/mlops/lcd_dataset/ocr.yaml"
EPOCHS = int(os.environ.get("TRAIN_EPOCHS", "3"))
IMGSZ = int(os.environ.get("TRAIN_IMGSZ", "320"))
BATCH = int(os.environ.get("TRAIN_BATCH", "4"))
MIN_MAP50 = float(os.environ.get("MIN_MAP50", "0.3"))  # quality gate


def main():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("lcd-ocr-digits-training")

    print(f"Loading base model: {BASE_MODEL}")
    if not os.path.exists(BASE_MODEL):
        print(f"ERROR: base model not found at {BASE_MODEL}. "
              f"Run: docker cp lcd_be:/app/models/ocr.pt {BASE_MODEL}")
        sys.exit(1)
    model = YOLO(BASE_MODEL)

    with mlflow.start_run(run_name=f"train-ocr-{int(time.time())}") as run:
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("imgsz", IMGSZ)
        mlflow.log_param("batch", BATCH)
        mlflow.log_param("base_model", BASE_MODEL)

        t0 = time.time()
        results = model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            device="cpu",
            workers=0,
            project="/tmp/lcd_runs",
            name="ocr_finetune",
            exist_ok=True,
            verbose=True,
        )
        train_seconds = time.time() - t0
        mlflow.log_metric("train_seconds", train_seconds)

        # ── Validate ────────────────────────────────────────────────────────────
        metrics = model.val(data=DATA_YAML, imgsz=IMGSZ, device="cpu")
        map50 = float(metrics.box.map50)
        map50_95 = float(metrics.box.map)
        mlflow.log_metric("map50", map50)
        mlflow.log_metric("map50_95", map50_95)
        print(f"map50={map50:.4f}  map50_95={map50_95:.4f}  train_seconds={train_seconds:.1f}")

        # ── Save best weights ───────────────────────────────────────────────────
        best_weights = "/tmp/lcd_runs/ocr_finetune/weights/best.pt"
        if not os.path.exists(best_weights):
            print("best.pt not found — training may have failed.")
            sys.exit(1)
        mlflow.log_artifact(best_weights, artifact_path="model")

        # ── Quality gate + register ─────────────────────────────────────────────
        if map50 >= MIN_MAP50:
            print(f"map50 {map50:.4f} >= threshold {MIN_MAP50} — registering model.")
            model_uri = f"runs:/{run.info.run_id}/model"
            mv = mlflow.register_model(model_uri, MODEL_NAME)
            client = mlflow.MlflowClient()
            client.set_registered_model_alias(MODEL_NAME, "Production", mv.version)
            mlflow.log_param("registered_version", mv.version)
            mlflow.log_param("promoted_to_production", True)
            print(f"Registered as {MODEL_NAME} v{mv.version}, alias=Production")
        else:
            print(f"map50 {map50:.4f} < threshold {MIN_MAP50} — NOT registering.")
            mlflow.log_param("promoted_to_production", False)


if __name__ == "__main__":
    main()
