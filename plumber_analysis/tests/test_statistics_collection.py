import unittest

from plumber_analysis import statistics_collection
import time

class TestStatisticsCollection(unittest.TestCase):
    def test_run(self):
        monitor = statistics_collection.ApplicationMonitoringThreadManager()
        profile_interval = 0.1
        monitor.start_thread(sleep_time=profile_interval)
        time.sleep(2)
        monitor.stop_thread()
        samples = monitor.thread_samples()
        self.assertGreater(len(samples), 0)

class TestContextManagerStatisticsCollection(unittest.TestCase):
    def test_run(self):
        constructor_params = {"sleep_time": 0.1}
        class CallbackConstructor():
            samples = None

            def callback_fn(self, new_samples):
                self.samples = new_samples

        c = CallbackConstructor()
        callback = lambda x: c.callback_fn(x)
        def get_monitor():
            return (statistics_collection
                    .CallbackApplicationMonitoringThreadManager(
                        callback, constructor_params))
        with get_monitor() as m:
            time.sleep(2)
        samples = c.samples
        self.assertGreater(len(samples), 0)

if __name__ == "__main__":
    unittest.main()
