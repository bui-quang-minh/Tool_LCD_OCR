from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import subprocess
import sys

default_args = {
    'owner': 'khoant16',
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}


def run_training(**context):
    result = subprocess.run(
        [sys.executable, "/opt/mlops/lcd_training/train_detection.py"],
        capture_output=True, text=True, env={
            "MLFLOW_TRACKING_URI": "http://mlflow:5000",
            "MLFLOW_DETECTION_MODEL_NAME": "lcd-ocr-detection",
            "TRAIN_EPOCHS": "3",
            "TRAIN_IMGSZ": "320",
            "TRAIN_BATCH": "4",
            "MIN_MAP50": "0.3",
            "PATH": "/usr/bin:/bin",
        },
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError("Training script failed with code " + str(result.returncode))


with DAG(
    dag_id='lcd_ocr_training',
    default_args=default_args,
    description='Fine-tune + validate + register LCD detection model (lcd/torque regions only)',
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['mlops', 'lcd-ocr', 'training', 'detection'],
) as dag:

    train_task = PythonOperator(
        task_id='train_and_register',
        python_callable=run_training,
    )
