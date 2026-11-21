"""Performance monitoring: service health (latency), model health (live RMSE vs the
RMSE at training time), and input drift (share of out-of-range feature vectors)."""
import glob
import os

import numpy as np
import pandas as pd

import cslib
import logger
import model

DEGRADE_RATIO = 1.5      # live RMSE > 1.5x training RMSE  => degraded
OUTLIER_LIMIT = 0.10     # >10% outlier inputs             => drift warning
MIN_SAMPLES = 5          # need >=5 scored predictions before flagging a country


def _read_predict_logs(test):
    files = glob.glob(os.path.join(logger.log_dir(test), "predict-*.log"))
    if not files:
        return pd.DataFrame(columns=logger.PREDICT_COLS)
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def performance_report(test=False):
    logs = _read_predict_logs(test)
    if logs.empty:
        return {"n_predictions": 0, "status": "no predictions logged yet", "retrain_recommended": False}
    try:
        models = model.model_load(test)
    except FileNotFoundError:
        models = {}

    per_country = []
    for c, g in logs.groupby("country"):
        art = models.get(c)
        errs = []
        if art is not None:
            ts, last = art["ts"], art["ts"].index.max()
            for d, yp in zip(pd.to_datetime(g["target_date"]), g["y_pred"]):
                if d + pd.Timedelta(days=cslib.HORIZON - 1) <= last:      # actuals are known
                    actual = ts.loc[d: d + pd.Timedelta(days=cslib.HORIZON - 1), "revenue"].sum()
                    errs.append((yp - actual) ** 2)
        live = float(np.sqrt(np.mean(errs))) if errs else None
        ref = art["holdout_rmse"] if art else None
        ratio = (live / ref) if (live is not None and ref) else None
        per_country.append({"country": c, "n_predictions": int(len(g)),
                            "n_with_actuals": len(errs), "live_rmse": live,
                            "training_rmse": ref, "ratio": ratio,
                            "degraded": bool(ratio is not None and len(errs) >= MIN_SAMPLES and ratio > DEGRADE_RATIO)})
    outlier_rate = float(logs["outlier"].mean()) if len(logs) >= MIN_SAMPLES else 0.0
    degraded = [p["country"] for p in per_country if p["degraded"]]
    return {
        "n_predictions": int(len(logs)),
        "mean_runtime_seconds": float(logs["runtime"].mean()),
        "p95_runtime_seconds": float(logs["runtime"].quantile(0.95)),
        "outlier_rate": outlier_rate,
        "countries_degraded": degraded,
        "per_country": per_country,
        "retrain_recommended": bool(degraded or outlier_rate > OUTLIER_LIMIT),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(performance_report(), indent=2, default=float))
