import shutil
import unittest

import app as app_module
import model
from unittests.fixtures import make_test_data


class ApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = make_test_data()
        cls.client = app_module.app.test_client()
        r = cls.client.post("/train", json={"test": True, "data_dir": cls.data_dir})
        assert r.status_code == 200, r.get_json()
        cls.countries = r.get_json()["countries"]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.data_dir, ignore_errors=True)
        model.clean_test_artifacts()

    def _predict(self, **kw):
        return self.client.post("/predict", json={"test": True, **kw})

    def test_01_index_and_health(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/health").get_json(), {"status": "ok"})

    def test_02_predict_all_countries_combined(self):
        r = self._predict(country="all", year=2018, month=9, day=15)
        self.assertEqual(r.status_code, 200)
        self.assertGreater(r.get_json()["forecast_30d_revenue"], 0)

    def test_03_predict_specific_country(self):
        c = [x for x in self.countries if x != "all"][0]
        r = self._predict(country=c, year=2018, month=9, day=15)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["country"], c)

    def test_04_get_with_query_string(self):
        r = self.client.get("/predict?country=all&year=2018&month=9&day=15&test=true")
        self.assertEqual(r.status_code, 200)

    def test_05_missing_params_400(self):
        self.assertEqual(self._predict(country="all").status_code, 400)

    def test_06_unknown_country_404(self):
        self.assertEqual(self._predict(country="atlantis", year=2018, month=9, day=15).status_code, 404)

    def test_07_bad_date_400(self):
        self.assertEqual(self._predict(country="all", year=2018, month=13, day=1).status_code, 400)
        self.assertEqual(self._predict(country="all", year=2035, month=1, day=1).status_code, 400)

    def test_08_logs_and_monitor_endpoints(self):
        self._predict(country="all", year=2018, month=9, day=1)
        names = self.client.get("/logs?test=true").get_json()
        self.assertTrue(any(n.startswith("predict") for n in names))
        self.assertEqual(self.client.get("/logs/../app.py?test=true").status_code, 404)
        rep = self.client.get("/monitor?test=true").get_json()
        self.assertGreaterEqual(rep["n_predictions"], 1)
        self.assertIn("retrain_recommended", rep)


if __name__ == "__main__":
    unittest.main()
