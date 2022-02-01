import unittest
import tempfile

from plumber_analysis import resource_measurements

class TestResourceMeasurements(unittest.TestCase):
    def test_filesystem_measurement(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_path = tmp_dir
            ret = resource_measurements.benchmark_filesystem(test_path,
                                                             parse_results=False)
            self.assertGreater(len(ret), 0)
            ret = resource_measurements.benchmark_filesystem(test_path, True)
            self.assertGreater(len(ret), 0)


if __name__ == "__main__":
    unittest.main()
