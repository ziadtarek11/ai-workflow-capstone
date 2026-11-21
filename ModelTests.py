import os
import shutil
import unittest

import pandas as pd

import cslib
import model
from unittests.fixtures import make_test_data


class ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = make_test_data()
        cls.summary = model.model_train(data_dir=cls.data_dir, test=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.data_dir, ignore_errors=True)
        model.clean_test_artifacts()

    def test_01_train_creates_test_models_only(self):
        files = os.listdir(model.model_dir(test=True))
        self.assertIn("model-all.joblib", files)
        self.assertGreaterEqual(len(self.summary["countries"]), 2)

    def test_02_multiple_models_compared_against_baseline(self):
        names = {r["model"] for r in self.summary["comparison"]}
        self.assertTrue({"baseline_run_rate", "ridge", "random_forest", "gradient_boosting"} <= names)

    def test_03_load_returns_artifacts(self):
        m = model.model_load(test=True)
        self.assertIn("all", m)
        for key in ("model", "holdout_rmse", "baseline_rmse", "ts", "version"):
            self.assertIn(key, m["all"])

    def test_04_predict_all_countries(self):
        r = model.model_predict("all", 2018, 9, 15, test=True)
        self.assertIsInstance(r["forecast_30d_revenue"], float)
        self.assertGreaterEqual(r["forecast_30d_revenue"], 0)
        self.assertEqual(r["country"], "all")

    def test_05_predict_single_country_any_casing(self):
        c = [k for k in self.summary["countries"] if k != "all"][0]
        r = model.model_predict(c.replace("_", " ").title(), 2018, 9, 15, test=True)
        self.assertEqual(r["country"], c)

    def test_06_unknown_country_raises(self):
        with self.assertRaises(KeyError):
            model.model_predict("atlantis", 2018, 9, 15, test=True)

    def test_07_bad_or_out_of_range_dates_raise(self):
        with self.assertRaises(ValueError):
            model.model_predict("all", 2018, 2, 31, test=True)
        with self.assertRaises(ValueError):
            model.model_predict("all", 2030, 1, 1, test=True)
        with self.assertRaises(ValueError):
            model.model_predict("all", 2018, 1, 5, test=True)   # not enough history

    def test_08_features_use_only_past_data(self):
        ts = model.model_load(test=True)["all"]["ts"]
        d = pd.Timestamp("2018-09-01")
        a = cslib.feature_row(ts, d)
        tampered = ts.copy(); tampered.loc[d:, :] = 10 ** 9      # change today & the future
        b = cslib.feature_row(tampered, d)
        pd.testing.assert_frame_equal(a, b)                      # no leakage

    def test_09_served_model_never_worse_than_baseline_on_holdout(self):
        for c, m in model.model_load(test=True).items():
            self.assertLessEqual(m["holdout_rmse"], m["baseline_rmse"] + 1e-9, c)


if __name__ == "__main__":
    unittest.main()
