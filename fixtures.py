"""Small synthetic dataset in a TEMP dir so tests never read or write production data."""
import tempfile
import generate_synthetic_data as gen


def make_test_data():
    d = tempfile.mkdtemp(prefix="cs-test-")
    gen.generate(d, start="2018-01-01", end="2018-10-31", scale=0.3, seed=1)
    return d
