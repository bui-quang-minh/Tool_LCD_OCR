from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import field_validator


class Settings(BaseSettings):
    # ── Model paths ───────────────────────────────────────────────────────────
    models_dir: Path = Path(__file__).parent / "models"
    detection_model: Path = Path(__file__).parent / "models" / "detection.pt"
    ocr_model: Path = Path(__file__).parent / "models" / "ocr.pt"

    # ── API ───────────────────────────────────────────────────────────────────
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
    ]

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_detection_model_name: str = "lcd-ocr-detection"
    mlflow_ocr_model_name: str = "lcd-ocr-digits"
    # When True: load models from MLflow Production alias instead of local disk
    use_mlflow_registry: bool = False
    # When True: log each inference call as an MLflow run
    use_mlflow_tracking: bool = False

    # ── Airflow ───────────────────────────────────────────────────────────────
    airflow_api_url: str = "http://localhost:8080/api/v1"
    airflow_dag_id: str = "lcd_ocr_training"
    airflow_username: str = "airflow"
    airflow_password: str = "airflow"

    @field_validator("models_dir", "detection_model", "ocr_model", mode="before")
    @classmethod
    def resolve_path(cls, v: str | Path) -> Path:
        return Path(v).resolve()

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
