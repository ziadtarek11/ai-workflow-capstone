import os
import unittest

import pandas as pd

import logger
import model


class LoggerTest(unittest.TestCase):
    def setUp(self):
        model.clean_test_artifacts()
        self.prod_before = {f: os.path.getmtime(os.path.join(logger.LOG_DIR, f))
                            for f in os.listdir(logger.LOG_DIR) if f.endswith(".log")}

    def tearDown(self):
        model.clean_test_artifacts()

    def test_01_train_log_written_with_all_columns(self):
        logger.update_train_log("all", "ridge", 10.5, 12.0, 1.2, "0.1", "note", test=True)
        df = pd.read_csv(logger.log_path("train", test=True))
        self.assertEqual(list(df.columns), logger.TRAIN_COLS)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.loc[0, "best_model"], "ridge")

    def test_02_predict_log_appends_rows(self):
        for i in range(3):
            logger.update_predict_log("all", "2018-09-01", 1000.0 + i, False, 0.01, "0.1", test=True)
        df = pd.read_csv(logger.log_path("predict", test=True))
        self.assertEqual(len(df), 3)
        self.assertEqual(list(df.columns), logger.PREDICT_COLS)
        self.assertEqual(df["outlier"].sum(), 0)

    def test_03_test_logs_go_to_test_folder(self):
        logger.update_predict_log("all", "2018-09-01", 1.0, False, 0.01, "0.1", test=True)
        self.assertIn(os.path.join("logs", "test"), logger.log_path("predict", test=True))
        self.assertNotEqual(os.path.dirname(logger.log_path("predict", test=True)),
                            os.path.dirname(logger.log_path("predict", test=False)))

    def test_04_production_logs_untouched_by_test_logging(self):
        logger.update_train_log("all", "ridge", 1, 2, 0.1, "0.1", test=True)
        logger.update_predict_log("all", "2018-09-01", 1.0, False, 0.01, "0.1", test=True)
        after = {f: os.path.getmtime(os.path.join(logger.LOG_DIR, f))
                 for f in os.listdir(logger.LOG_DIR) if f.endswith(".log")}
        self.assertEqual(self.prod_before, after)

    def test_05_cleanup_removes_only_test_artifacts(self):
        logger.update_train_log("all", "ridge", 1, 2, 0.1, "0.1", test=True)
        model.clean_test_artifacts()
        self.assertFalse(os.path.exists(os.path.join(logger.LOG_DIR, "test")))
        self.assertTrue(os.path.isdir(logger.LOG_DIR))


if __name__ == "__main__":
    unittest.main()
