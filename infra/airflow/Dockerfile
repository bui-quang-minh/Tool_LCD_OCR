FROM apache/airflow:2.8.1
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*
USER airflow
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    torch torchvision \
    pymysql qdrant-client mlflow ultralytics opencv-python-headless
