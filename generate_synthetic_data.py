"""Generates synthetic streaming-invoice JSON files in the same schema as the course data
(country, customer_id, day, invoice, month, year, price, stream_id, times_viewed).
Replace data/cs-train and data/cs-production with the real files when you have them."""
import json
import os
import sys

import numpy as np
import pandas as pd

COUNTRIES = {"United Kingdom": 22, "EIRE": 3.0, "Germany": 3.5, "France": 3.0, "Norway": 1.6,
             "Spain": 1.6, "Hong Kong": 1.0, "Portugal": 1.2, "Singapore": 1.0,
             "Netherlands": 2.0, "Belgium": 1.0, "Australia": 0.8}
WEEKDAY = [1.0, 1.1, 1.1, 1.05, 1.0, 0.6, 0.7]
ORIGIN = pd.Timestamp("2017-11-01")


def generate(out_dir, start="2017-11-01", end="2019-05-31", scale=1.0, seed=42):
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    days = pd.date_range(start, end, freq="D")
    rows, inv = [], 400000
    for d in days:
        t = (d - ORIGIN).days
        season = 1 + 0.25 * np.sin(2 * np.pi * (d.dayofyear - 60) / 365)   # annual cycle
        for country, base in COUNTRIES.items():
            lam = base * scale * (1 + 0.0009 * t) * WEEKDAY[d.dayofweek] * season
            for _ in range(rng.poisson(lam)):
                inv += 1
                cust = int(rng.integers(12000, 18500))
                for _ in range(1 + rng.poisson(1.0)):
                    rows.append({
                        "country": country, "customer_id": cust, "day": d.day,
                        "invoice": f"C{inv}" if rng.random() < 0.01 else str(inv),
                        "month": d.month, "year": d.year,
                        "price": round(float(rng.lognormal(2.2, 0.6)), 2),
                        "stream_id": int(rng.integers(20000, 20400)),
                        "times_viewed": 1 + int(rng.poisson(3)),
                    })
    df = pd.DataFrame(rows)
    df["ym"] = list(zip(df["year"], df["month"]))
    for i, ((y, m), g) in enumerate(df.groupby("ym")):
        g = g.drop(columns="ym")
        if i % 5 == 0:   # mimic the inconsistent column names in the raw feed
            g = g.rename(columns={"stream_id": "StreamID", "times_viewed": "TimesViewed",
                                  "price": "total_price"})
        with open(os.path.join(out_dir, f"invoices-{y}-{m:02d}.json"), "w") as fh:
            json.dump(g.to_dict("records"), fh)
    return len(df)


if __name__ == "__main__":
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    n1 = generate(os.path.join(root, "cs-train"), "2017-11-01", "2019-05-31")
    n2 = generate(os.path.join(root, "cs-production"), "2019-06-01", "2019-07-31", seed=7)
    print(f"wrote {n1} training rows and {n2} production rows to {root}")
