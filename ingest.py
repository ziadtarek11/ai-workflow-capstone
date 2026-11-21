"""Automation-friendly ingestion: python ingest.py --data-dir data/cs-train --out data/ingested"""
import argparse
import cslib

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/cs-train")
    ap.add_argument("--out", default="data/ingested")
    a = ap.parse_args()
    series = cslib.ingest(a.data_dir, a.out)
    for k, ts in series.items():
        print(f"{k:16s} {len(ts):4d} days  {ts.index.min().date()} -> {ts.index.max().date()}  revenue={ts['revenue'].sum():,.0f}")
