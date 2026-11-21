"""Train / compare / persist / serve the 30-day revenue models."""
import os
import shutil
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import cslib
import logger

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "models")
FIG_DIR = os.path.join(BASE, "figures")
DEFAULT_DATA_DIR = os.path.join(BASE, "data", "cs-train")
MODEL_VERSION = "1.0"
MODEL_NOTE = "ridge / random forest / gradient boosting, best chosen on time-based holdout"
HOLDOUT_DAYS = 90


def model_dir(test=False):
    d = os.path.join(MODEL_DIR, "test") if test else MODEL_DIR
    os.makedirs(d, exist_ok=True)
    return d


def clean_test_artifacts():
    """Remove ONLY test artifacts (models/test, logs/test)."""
    for d in (os.path.join(MODEL_DIR, "test"), os.path.join(logger.LOG_DIR, "test")):
        shutil.rmtree(d, ignore_errors=True)


def _candidates(test):
    n = 10 if test else 150
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "random_forest": RandomForestRegressor(n_estimators=n, max_depth=6, random_state=42, n_jobs=1),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=n, max_depth=3, random_state=42),
    }


def baseline_predict(X):
    """Naive run-rate baseline: last 28 days of revenue scaled to 30 days."""
    return X["past_28"].to_numpy() * cslib.HORIZON / 28.0


class RunRate:
    """The baseline wrapped as an estimator so it competes with (and can be served instead
    of) the ML models: a model that cannot beat the baseline on the holdout never ships."""
    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return baseline_predict(X)


def _rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def _train_one(country, ts, test):
    X, y = cslib.build_xy(ts)
    hold = min(HOLDOUT_DAYS, max(len(X) // 4, 10))
    cut = X.index.max() - pd.Timedelta(days=hold - 1)
    tr = X.index < cut - pd.Timedelta(days=cslib.HORIZON)   # gap => no target overlap
    te = X.index >= cut
    if tr.sum() < 20:
        raise ValueError(f"not enough history to train '{country}'")

    rows, preds = [], {}
    base = baseline_predict(X[te])
    rows.append({"country": country, "model": "baseline_run_rate", "rmse": _rmse(y[te], base)})
    preds["baseline"] = base
    fitted_best, best_name, best_rmse = None, None, np.inf
    for name, est in _candidates(test).items():
        est.fit(X[tr], y[tr])
        p = est.predict(X[te])
        r = _rmse(y[te], p)
        rows.append({"country": country, "model": name, "rmse": r})
        if r < best_rmse:
            best_name, best_rmse, preds["model"] = name, r, p
    fell_back = best_rmse >= rows[0]["rmse"]
    if fell_back:                                             # champion/challenger rule
        best_name, best_rmse, preds["model"] = "baseline_run_rate", rows[0]["rmse"], base
        final = RunRate()
    else:
        final = _candidates(test)[best_name]
    final.fit(X, y)                                           # refit on everything
    art = {
        "model": final, "model_name": best_name, "version": MODEL_VERSION,
        "holdout_rmse": best_rmse, "baseline_rmse": rows[0]["rmse"], "fell_back_to_baseline": fell_back,
        "feat_mean": X.mean(), "feat_std": X.std().replace(0, 1.0),
        "ts": ts, "trained_on": str(X.index.max().date()),
        "holdout": pd.DataFrame({"actual": y[te].to_numpy(), "baseline": preds["baseline"],
                                 "model": preds["model"]}, index=X.index[te]),
    }
    return art, rows


def plot_baseline_comparison(results, arts, path):
    """Left: RMSE per country baseline vs best model. Right: holdout forecast vs actual."""
    base = results[results["model"] == "baseline_run_rate"].set_index("country")["rmse"]
    order = [c for c in base.index]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    x = np.arange(len(order)); w = 0.4
    ax[0].bar(x - w / 2, [base[c] for c in order], w, label="Baseline (run-rate)", color="#9aa5b1")
    ax[0].bar(x + w / 2, [arts[c]["holdout_rmse"] for c in order], w,
              label="Served model (best candidate, else baseline)", color="#2b6cb0")
    ax[0].set_xticks(x); ax[0].set_xticklabels(order, rotation=60, ha="right")
    ax[0].set_ylabel("Holdout RMSE (30-day revenue)"); ax[0].set_yscale("log")
    ax[0].set_title("Served model vs baseline error by country (lower is better)"); ax[0].legend()
    h = arts["all"]["holdout"]
    ax[1].plot(h.index, h["actual"], label="Actual", color="black", lw=2)
    ax[1].plot(h.index, h["baseline"], label="Baseline", color="#9aa5b1", ls="--")
    ax[1].plot(h.index, h["model"], label=f"Model ({arts['all']['model_name']})", color="#2b6cb0")
    ax[1].set_title("All countries: next-30-day revenue on the holdout window")
    ax[1].tick_params(axis="x", rotation=30); ax[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def model_train(data_dir=None, test=False, n_countries=10):
    """Train every country model, compare candidates vs baseline, persist + log.
    test=True: smaller models, only 'all' + top country, writes to models/test & logs/test."""
    t0 = time.time()
    df = cslib.fetch_data(data_dir or DEFAULT_DATA_DIR)
    countries = ["all"] + cslib.top_countries(df, 1 if test else n_countries)
    out, all_rows, arts = model_dir(test), [], {}
    for c in countries:
        t1 = time.time()
        art, rows = _train_one(c, cslib.convert_to_ts(df, c), test)
        joblib.dump(art, os.path.join(out, f"model-{c}.joblib"))
        all_rows += rows; arts[c] = art
        logger.update_train_log(c, art["model_name"], art["holdout_rmse"], art["baseline_rmse"],
                                time.time() - t1, MODEL_VERSION, MODEL_NOTE, test=test)
    results = pd.DataFrame(all_rows)
    results.to_csv(os.path.join(out, "model_comparison.csv"), index=False)
    if not test:
        os.makedirs(FIG_DIR, exist_ok=True)
        plot_baseline_comparison(results, arts, os.path.join(FIG_DIR, "model_vs_baseline.png"))
    return {"countries": countries, "runtime_seconds": round(time.time() - t0, 2),
            "model_version": MODEL_VERSION, "comparison": results.round(2).to_dict("records")}


def model_load(test=False):
    d = model_dir(test)
    files = [f for f in os.listdir(d) if f.startswith("model-") and f.endswith(".joblib")]
    if not files:
        raise FileNotFoundError("no trained models found - train first (POST /train)")
    return {f[len("model-"):-len(".joblib")]: joblib.load(os.path.join(d, f)) for f in files}


def model_predict(country, year, month, day, models=None, test=False):
    """Forecast revenue for the 30 days starting on year-month-day."""
    t0 = time.time()
    models = models or model_load(test)
    key = cslib.country_key(country)
    if key not in models:
        raise KeyError(f"country '{country}' not available; choose from {sorted(models)}")
    try:
        target = pd.Timestamp(year=int(year), month=int(month), day=int(day))
    except (ValueError, TypeError) as e:
        raise ValueError(f"invalid date: {e}")
    art = models[key]
    X = cslib.feature_row(art["ts"], target)
    z = ((X.iloc[0] - art["feat_mean"]) / art["feat_std"]).abs()
    outlier = bool(z.max() > 4)
    y_pred = max(float(art["model"].predict(X)[0]), 0.0)
    runtime = time.time() - t0
    logger.update_predict_log(key, str(target.date()), y_pred, outlier, runtime,
                              art["version"], test=test)
    return {"country": key, "date": str(target.date()), "forecast_30d_revenue": round(y_pred, 2),
            "model": art["model_name"], "model_version": art["version"],
            "input_outlier": outlier, "runtime_seconds": round(runtime, 5)}
