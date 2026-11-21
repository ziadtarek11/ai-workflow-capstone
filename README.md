# 30-Day Revenue Forecasting — AI Workflow Capstone

Forecasts the **next 30 days of revenue** for a streaming business, per country or for all
countries combined, and serves it through a containerised Flask API with logging,
monitoring and unit tests.

> **Data note:** the course's invoice JSON was not available when this was built, so
> `generate_synthetic_data.py` produces data in the *same schema* (including the messy
> column-name variants). To use real data, put the files in `data/cs-train/` and
> `data/cs-production/` — nothing else changes. Numbers below are on synthetic data.

## Quick start
```bash
pip install -r requirements.txt
python generate_synthetic_data.py      # skip if you have the real data
python ingest.py                       # data ingestion (automatable)
python eda.py                          # EDA figures -> figures/
python train.py                        # compare models, save best per country
python run-tests.py                    # ALL unit tests, one command
python app.py                          # API on :8080
```
Docker: `docker build -t revenue-api . && docker run -p 8080:8080 revenue-api`
(the build ingests, runs EDA, trains and runs the full test suite; it fails if any test fails).

```bash
curl -X POST localhost:8080/predict -H 'Content-Type: application/json' \
     -d '{"country":"all","year":2019,"month":5,"day":1}'      # all countries combined
curl -X POST localhost:8080/predict -H 'Content-Type: application/json' \
     -d '{"country":"germany","year":2019,"month":5,"day":1}'  # one country
curl localhost:8080/monitor      # performance report
curl localhost:8080/logs         # list logs
```

## Rubric → evidence
| Question | Where |
|---|---|
| Unit tests for the API | `unittests/ApiTests.py` (8 tests: all/single country, GET+POST, 400/404 errors, logs, monitor) |
| Unit tests for the model | `unittests/ModelTests.py` (9 tests incl. leakage check and "never worse than baseline") |
| Unit tests for the logging | `unittests/LoggerTests.py` (5 tests); plus `MonitorTests.py` (3) |
| One script runs all tests, all pass | `python run-tests.py` → 25 tests, exit code 0 on success |
| Mechanism to monitor performance | `monitor.py` + `GET /monitor`: latency (mean/p95), live RMSE vs training RMSE per country, input-outlier rate, `retrain_recommended` flag (needs ≥5 scored predictions to avoid noisy alarms) |
| Test isolation from production | every function takes `test=True` → writes to `models/test/` and `logs/test/`; tests use a temp-dir dataset; a test asserts production logs are untouched; teardown deletes only test dirs |
| API: single country and all | `POST/GET /predict` with `country` = `all` or any country name (any casing) |
| Data ingestion as function/script | `cslib.fetch_data`, `cslib.ingest` and CLI `ingest.py` |
| Multiple models compared | baseline, Ridge, Random Forest, Gradient Boosting → `models/model_comparison.csv` |
| EDA with visualizations | `eda.py` → 5 figures + `eda_summary.txt` in `figures/` |
| Containerised, working Docker image | `Dockerfile` |
| Visual comparison to baseline | `figures/model_vs_baseline.png` |

## Design decisions
* **Target:** revenue over days *d … d+29*. **Features:** trailing revenue (7/14/28/56/84 d),
  invoices and views (28 d), all computed from data strictly *before* d (tested for leakage).
* **Evaluation:** chronological holdout (last 90 days) with a 30-day gap to the training set
  so overlapping targets don't leak. Final model is refit on all data.
* **Baseline:** trailing-28-day run-rate × 30/28.
* **Champion/challenger:** if no ML model beats the baseline on the holdout, the baseline is
  served. This happened for EIRE, Germany and Netherlands; ML won in 8 of 11 series
  (e.g. all-countries RMSE 2,663 vs 4,529 baseline, −41%).
* **Limitations:** the holdout is used for both model selection and reporting (slightly
  optimistic); no hyper-parameter search; synthetic data has a smooth trend and may be
  easier than real data; dates beyond last-data-day + 1 are rejected rather than extrapolated.

## Layout
`app.py` API · `model.py` train/compare/predict · `cslib.py` ingestion + features ·
`logger.py` logging · `monitor.py` monitoring · `eda.py` · `train.py` · `ingest.py` ·
`run-tests.py` · `unittests/` · `Dockerfile`
