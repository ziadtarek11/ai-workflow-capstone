"""CSV logging for training and prediction. `test=True` writes to logs/test/ so unit
tests can never touch the production logs."""
import csv
import os
import time
import uuid
from datetime import date

BASE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE, "logs")

TRAIN_COLS = ["unique_id", "timestamp", "country", "best_model", "holdout_rmse",
              "baseline_rmse", "runtime", "model_version", "model_note"]
PREDICT_COLS = ["unique_id", "timestamp", "country", "target_date", "y_pred",
                "outlier", "runtime", "model_version"]


def log_dir(test=False):
    d = os.path.join(LOG_DIR, "test") if test else LOG_DIR
    os.makedirs(d, exist_ok=True)
    return d


def log_path(kind, test=False):
    today = date.today()
    name = f"{kind}-{'test' if test else f'{today.year}-{today.month:02d}'}.log"
    return os.path.join(log_dir(test), name)


def _write(path, cols, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh, delimiter=",")
        if new:
            w.writerow(cols)
        w.writerow(row)


def update_train_log(country, best_model, holdout_rmse, baseline_rmse, runtime,
                     model_version, model_note="", test=False):
    _write(log_path("train", test), TRAIN_COLS,
           [uuid.uuid4(), time.time(), country, best_model, round(float(holdout_rmse), 4),
            round(float(baseline_rmse), 4), round(float(runtime), 4), model_version, model_note])


def update_predict_log(country, target_date, y_pred, outlier, runtime, model_version, test=False):
    _write(log_path("predict", test), PREDICT_COLS,
           [uuid.uuid4(), time.time(), country, target_date, round(float(y_pred), 4),
            int(bool(outlier)), round(float(runtime), 6), model_version])


def list_logs(test=False):
    return sorted(f for f in os.listdir(log_dir(test)) if f.endswith(".log"))
