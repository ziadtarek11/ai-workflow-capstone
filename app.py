"""Flask API.  POST /predict {"country": "all"|"germany"|..., "year":2019,"month":5,"day":20}"""
import os

from flask import Flask, jsonify, request, send_from_directory

import logger
import model
import monitor

app = Flask(__name__)
_cache = {}


def _models(test):
    if test not in _cache:
        _cache[test] = model.model_load(test)
    return _cache[test]


def _params():
    p = request.get_json(silent=True) or {}
    return {**request.args.to_dict(), **p}


@app.route("/")
def index():
    return jsonify({
        "service": "30-day revenue forecast", "model_version": model.MODEL_VERSION,
        "endpoints": {"POST /train": "retrain all models", "POST|GET /predict":
                      "country, year, month, day", "GET /monitor": "performance report",
                      "GET /logs": "list logs", "GET /logs/<file>": "download a log",
                      "GET /health": "liveness"}})


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/train", methods=["POST"])
def train():
    p = _params()
    test = str(p.get("test", "false")).lower() in ("true", "1")
    try:
        summary = model.model_train(data_dir=p.get("data_dir") if test else None, test=test)
    except (FileNotFoundError, ValueError) as e:
        return jsonify({"error": str(e)}), 400
    _cache.pop(test, None)
    return jsonify(summary)


@app.route("/predict", methods=["POST", "GET"])
def predict():
    p = _params()
    test = str(p.get("test", "false")).lower() in ("true", "1")
    missing = [k for k in ("country", "year", "month", "day") if k not in p]
    if missing:
        return jsonify({"error": f"missing parameters: {missing}"}), 400
    try:
        return jsonify(model.model_predict(p["country"], p["year"], p["month"], p["day"],
                                           models=_models(test), test=test))
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except KeyError as e:
        return jsonify({"error": e.args[0]}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/monitor")
def monitor_view():
    test = request.args.get("test", "false").lower() in ("true", "1")
    return jsonify(monitor.performance_report(test=test))


@app.route("/logs")
def logs():
    test = request.args.get("test", "false").lower() in ("true", "1")
    return jsonify(logger.list_logs(test))


@app.route("/logs/<path:filename>")
def get_log(filename):
    test = request.args.get("test", "false").lower() in ("true", "1")
    if filename not in logger.list_logs(test):           # whitelist => no path traversal
        return jsonify({"error": "log not found"}), 404
    return send_from_directory(logger.log_dir(test), filename, mimetype="text/plain")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
