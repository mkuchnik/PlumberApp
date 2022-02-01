import unittest

from plumber_analysis import gen_util
import time

import tensorflow as tf

class TestGenUtil(unittest.TestCase):
    def test_benchmark_dataset(self):
        ds = tf.data.Dataset.from_tensor_slices(
            tf.random.uniform([4, 10], minval=1, maxval=10, dtype=tf.int32))
        ds = ds.map(lambda x: x**2)
        summary = gen_util.benchmark_dataset(ds, print_perf=False,
                                             profile_interval=1)
        self.assertTrue(isinstance(summary, dict))

    def test_benchmark_dataset_autotune(self):
        ds = tf.data.Dataset.from_tensor_slices(
            tf.random.uniform([4, 10], minval=1, maxval=10, dtype=tf.int32))
        ds = ds.map(lambda x: x**2)
        summary = gen_util.benchmark_dataset(ds, print_perf=False,
                                             time_limit_s=gen_util.AUTOTUNE)
        self.assertTrue(isinstance(summary, dict))

    def test_benchmark_dataset_autotune_extra(self):
        ds = tf.data.Dataset.from_tensor_slices(
            tf.random.uniform([4, 10], minval=1, maxval=10, dtype=tf.int32))
        ds = ds.map(lambda x: x**2)
        summary = gen_util.benchmark_dataset(ds, print_perf=False,
                                             time_limit_s=gen_util.AUTOTUNE,
                                             min_time_limit_s=1,
                                             max_time_limit_s=1.1)
        self.assertTrue(isinstance(summary, dict))

if __name__ == "__main__":
    unittest.main()
