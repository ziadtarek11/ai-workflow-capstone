import shutil
import unittest

import logger
import model
import monitor
from unittests.fixtures import make_test_data


class MonitorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = make_test_data()
        model.clean_test_artifacts()
        model.model_train(data_dir=cls.data_dir, test=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.data_dir, ignore_errors=True)
        model.clean_test_artifacts()

    def setUp(self):
        shutil.rmtree(logger.log_dir(test=True), ignore_errors=True)   # each test starts clean

    def test_01_report_structure_after_predictions(self):
        for d in range(1, 8):
            model.model_predict("all", 2018, 9, d, test=True)
        rep = monitor.performance_report(test=True)
        self.assertEqual(rep["n_predictions"], 7)
        for k in ("mean_runtime_seconds", "p95_runtime_seconds", "outlier_rate", "per_country"):
            self.assertIn(k, rep)

    def test_02_single_prediction_does_not_trigger_retrain(self):
        logger.update_predict_log("all", "2018-09-01", 1.0, True, 0.01, "1.0", test=True)
        rep = monitor.performance_report(test=True)
        self.assertFalse(rep["retrain_recommended"])

    def test_03_degradation_is_flagged_with_enough_bad_predictions(self):
        # inject clearly wrong forecasts (1.0) for dates whose actuals are known
        for d in range(1, 8):
            logger.update_predict_log("all", f"2018-08-{d:02d}", 1.0, False, 0.01, "1.0", test=True)
        rep = monitor.performance_report(test=True)
        self.assertIn("all", rep["countries_degraded"])
        self.assertTrue(rep["retrain_recommended"])


if __name__ == "__main__":
    unittest.main()
