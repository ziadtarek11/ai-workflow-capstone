"""CLI: python train.py [--data-dir DIR] [--test]"""
import argparse
import json
import model

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--test", action="store_true", help="fast run, writes to models/test + logs/test")
    a = ap.parse_args()
    s = model.model_train(data_dir=a.data_dir, test=a.test)
    print(json.dumps({k: v for k, v in s.items() if k != "comparison"}, indent=2))
