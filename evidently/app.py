from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

app = FastAPI(title="LCD OCR Drift Monitoring")

LOG_FILE = Path("/app_logs/predictions.jsonl")
REPORTS_DIR = Path("/app/reports")
REPORTS_DIR.mkdir(exist_ok=True)


def load_predictions():
    if not LOG_FILE.exists():
        return pd.DataFrame()
    rows = []
    with open(LOG_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["verdict_ok"] = (df["verdict"] == "OK").astype(int)
    df["value_nm"] = df["value_nm"].fillna(-1)
    return df


@app.get("/")
def root():
    n = 0
    if LOG_FILE.exists():
        with open(LOG_FILE) as f:
            n = sum(1 for _ in f)
    return {"service": "LCD OCR Drift Monitoring", "status": "ok", "n_predictions_logged": n}


@app.post("/run-report")
def run_report(split_ratio: float = 0.5):
    from evidently import Report
    from evidently.presets import DataDriftPreset

    df = load_predictions()
    if len(df) < 10:
        return {"error": "Not enough predictions logged yet (need >= 10, have " + str(len(df)) + "). Run more /predict calls first."}

    split_idx = int(len(df) * split_ratio)
    reference = df.iloc[:split_idx][["value_nm", "n_tries", "verdict_ok"]]
    current = df.iloc[split_idx:][["value_nm", "n_tries", "verdict_ok"]]

    report = Report(metrics=[DataDriftPreset()])
    my_eval = report.run(reference_data=reference, current_data=current)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / ("report_" + timestamp + ".html")
    my_eval.save_html(str(report_path))

    result = my_eval.dict()
    drift_result = result.get("metrics", [{}])[0].get("result", {})
    drift_detected = drift_result.get("dataset_drift", False)
    n_drifted = drift_result.get("number_of_drifted_columns", 0)
    n_total = drift_result.get("number_of_columns", 3)

    log = {
        "drift_detected": drift_detected,
        "drifted_columns": n_drifted,
        "total_columns": n_total,
        "n_reference": len(reference),
        "n_current": len(current),
        "timestamp": timestamp,
    }
    with open(REPORTS_DIR / "drift_log.jsonl", "a") as f:
        f.write(json.dumps(log) + "\n")

    return {**log, "report_file": report_path.name, "message": "DRIFT DETECTED" if drift_detected else "No drift"}


@app.get("/reports")
def list_reports():
    files = sorted(REPORTS_DIR.glob("*.html"), reverse=True)
    return {"count": len(files), "reports": [f.name for f in files]}


@app.get("/reports/{report_name}")
def view_report(report_name: str):
    report_path = REPORTS_DIR / report_name
    if not report_path.exists():
        return {"error": "Report not found"}
    return HTMLResponse(content=report_path.read_text())


@app.get("/drift/status")
def drift_status():
    log_file = REPORTS_DIR / "drift_log.jsonl"
    if not log_file.exists():
        return {"message": "No reports run yet"}
    with open(log_file) as f:
        lines = f.readlines()
    return {"history": [json.loads(l) for l in lines[-10:]]}
