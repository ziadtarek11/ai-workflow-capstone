"""Data ingestion + feature engineering for the revenue-forecasting project.

`fetch_data` / `ingest` are importable functions AND `ingest.py` wraps them as a
CLI script so the whole step can be automated (cron, Airflow, CI, Docker build).
"""
import glob
import os
import re

import numpy as np
import pandas as pd

WINDOWS = [7, 14, 28, 56, 84]   # look-back windows (days) used as features
HORIZON = 30                    # we forecast revenue for the next 30 days
FEATURES = [f"past_{w}" for w in WINDOWS] + ["invoices_28", "views_28"]

# column-name variants found in the raw invoice files
_RENAME = {
    "streamid": "stream_id", "stream_id": "stream_id",
    "timesviewed": "times_viewed", "times_viewed": "times_viewed",
    "total_price": "price", "price": "price",
    "invoice_id": "invoice", "invoice": "invoice",
    "customer_id": "customer_id", "country": "country",
    "day": "day", "month": "month", "year": "year",
}


def country_key(name):
    """'United Kingdom' -> 'united_kingdom' (API accepts any casing/spacing)."""
    return re.sub(r"\s+", "_", str(name).strip().lower())


def fetch_data(data_dir):
    """Load every monthly JSON invoice file in `data_dir` into ONE clean DataFrame."""
    files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if not files:
        raise FileNotFoundError(f"no .json invoice files found in '{data_dir}'")
    frames = []
    for f in files:
        d = pd.read_json(f)
        d.columns = [_RENAME.get(c.lower().strip(), c.lower().strip()) for c in d.columns]
        frames.append(d)
    df = pd.concat(frames, ignore_index=True, sort=False)

    needed = ["country", "invoice", "year", "month", "day", "price", "stream_id", "times_viewed"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"input data is missing columns: {missing}")

    df["invoice"] = df["invoice"].astype(str).str.replace(r"\D+", "", regex=True)  # drop 'C' etc.
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["times_viewed"] = pd.to_numeric(df["times_viewed"], errors="coerce")
    df["date"] = pd.to_datetime(df[["year", "month", "day"]], errors="coerce")
    df = df.dropna(subset=["date", "price", "country"]).copy()
    df["times_viewed"] = df["times_viewed"].fillna(0)
    df["country_key"] = df["country"].map(country_key)
    return df.sort_values("date").reset_index(drop=True)


def top_countries(df, n=10):
    return df.groupby("country_key")["price"].sum().nlargest(n).index.tolist()


def convert_to_ts(df, country=None):
    """Daily time series (revenue, invoices, streams, views) for one country or 'all'."""
    d = df if country in (None, "all") else df[df["country_key"] == country_key(country)]
    if d.empty:
        raise ValueError(f"no data for country '{country}'")
    daily = d.groupby("date").agg(
        revenue=("price", "sum"),
        invoices=("invoice", "nunique"),
        streams=("stream_id", "nunique"),
        views=("times_viewed", "sum"),
    )
    full = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    return daily.reindex(full, fill_value=0).rename_axis("date")


def engineer_features(ts):
    """Features at date d use ONLY data strictly before d. Target = revenue over d..d+29."""
    rev = ts["revenue"].shift(1)
    X = pd.DataFrame(index=ts.index)
    for w in WINDOWS:
        X[f"past_{w}"] = rev.rolling(w).sum()
    X["invoices_28"] = ts["invoices"].shift(1).rolling(28).sum()
    X["views_28"] = ts["views"].shift(1).rolling(28).sum()
    y = ts["revenue"][::-1].rolling(HORIZON).sum()[::-1]
    return X, y


def build_xy(ts):
    X, y = engineer_features(ts)
    ok = X.notna().all(axis=1) & y.notna()
    return X[ok], y[ok]


def feature_row(ts, target_date):
    """Feature vector for a forecast starting on `target_date`."""
    target_date = pd.Timestamp(target_date)
    first_ok = ts.index.min() + pd.Timedelta(days=max(WINDOWS))
    last_ok = ts.index.max() + pd.Timedelta(days=1)
    if target_date < first_ok or target_date > last_ok:
        raise ValueError(
            f"date must be between {first_ok.date()} and {last_ok.date()} "
            "(need history before the date and no gaps)")
    ext = ts.reindex(pd.date_range(ts.index.min(), max(ts.index.max(), target_date), freq="D"),
                     fill_value=0)
    X, _ = engineer_features(ext)
    return X.loc[[target_date]]


def ingest(data_dir, out_dir=None, n_countries=10):
    """Automation entry point: load -> clean -> aggregate -> (optionally) write CSVs."""
    df = fetch_data(data_dir)
    series = {"all": convert_to_ts(df)}
    for c in top_countries(df, n_countries):
        series[c] = convert_to_ts(df, c)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        for k, ts in series.items():
            ts.to_csv(os.path.join(out_dir, f"ts-{k}.csv"))
    return series
