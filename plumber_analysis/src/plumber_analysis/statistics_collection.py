"""Collect resource usage while a process runs."""

import collections
import psutil
import inspect
import types
import threading
import time

from typing import List, Callable

Sample = collections.namedtuple("Sample",
                                ["timestep",
                                 "cpu_frequency",
                                 "cpu_temperature",
                                 "cpu_percent",
                                 "io_usage",
                                 "memory_usage"])
Samples = List[Sample]

# Loosely following
# https://stackoverflow.com/questions/33365664/how-to-implement-a-daemon-stoppable-polling-thread-in-python

def flatten_sample(sample: Sample) -> dict:
    assert isinstance(sample, Sample), \
        "Expected Sample to flatten, got {}".format(type(sample))
    flat_sample = dict()
    flat_sample["timestep"] = sample.timestep
    flat_sample["cpu_frequency"] = sample.cpu_frequency.current
    temp = sample.cpu_temperature
    for k, v in temp.items():
        for i, reading in enumerate(v):
            flat_sample["cpu_temperature_{}_{}".format(k, i)] = \
                reading.current
    io_usage = sample.io_usage
    for k, v in io_usage._asdict().items():
        flat_sample["io_usage_{}".format(k)] = v
    memory_usage = sample.memory_usage
    for k, v in memory_usage._asdict().items():
        flat_sample["memory_usage_{}".format(k)] = v
    flat_sample["cpu_percent"] = sample.cpu_percent
    return flat_sample

def subtract_tuple(tup1: tuple, tup2: tuple) -> tuple:
    if isinstance(tup1, type(tup2)):
        vals = map(lambda x1, x2, key: (key, x1 - x2), tup1, tup2, tup1._fields)
        vals = dict(vals)
        out_type = type(tup1)
        return out_type(**vals)
    else:
        vals = map(lambda x1, x2: x1 - x2, tup1, tup2)
        out_tupe = tuple
        return out_type(vals)


class ApplicationMonitoringThread(threading.Thread):
    def __init__(self, sleep_time: float):
        super(ApplicationMonitoringThread, self).__init__()
        # sleep_time in seconds
        if not isinstance(sleep_time, (int, float)):
            raise ValueError("Sleep_time is not float: {}, {}".format(
                sleep_time, type(sleep_time)))
        self.data = [] # Collects samples
        self.stop_event = threading.Event()
        self.sleep_time = sleep_time
        self.proc = psutil.Process() # Current pid
        self.proc.cpu_percent() # Init
        self.init_io_counters = self.proc.io_counters()

    def stop(self) -> None:
        self.stop_event.set()

    def is_stopped(self) -> bool:
        return self.stop_event.isSet()

    def run(self) -> None:
        while not self.is_stopped():
            self.sample_and_store()
            self.stop_event.wait(self.sleep_time)

    def sample_and_store(self) -> None:
        sample = self._sample()
        self.data.append(sample)

    def samples(self) -> Samples:
        return self.data

    def _sample(self) -> Sample:
        t = time.time()
        freq = psutil.cpu_freq()
        temp = psutil.sensors_temperatures()
        with self.proc.oneshot():
            cpu_percent = self.proc.cpu_percent()
            io_counters = self.proc.io_counters()
            mem = self.proc.memory_full_info()
        io_counter_diff = subtract_tuple(io_counters, self.init_io_counters)
        sample = Sample(timestep=t,
                        cpu_frequency=freq,
                        cpu_temperature=temp,
                        cpu_percent=cpu_percent,
                        io_usage=io_counter_diff,
                        memory_usage=mem)
        return sample


class ApplicationMonitoringThreadManager(object):
    def __init__(self):
        self.thread = None

    def start_thread(self, *args, **kwargs):
        if not self.thread or not self.thread.is_alive():
            self.thread = ApplicationMonitoringThread(*args, **kwargs)
            self.thread.start()
        else:
            raise RuntimeError("Thread is already alive!")

    def stop_thread(self, *args, **kwargs):
        if self.thread and self.thread.is_alive():
            self.thread.stop()
        else:
            raise RuntimeError("Thread is not alive!")

    def thread_samples(self) -> Samples:
        if not self.thread:
            return []
        samples = self.thread.samples()
        assert isinstance(samples, list), \
            "Expected to return Samples, but got {}".format(type(samples))
        return samples

class CallbackApplicationMonitoringThreadManager(ApplicationMonitoringThreadManager):
    """For context managers"""
    def __init__(self, callback: Callable, constructor_params: dict):
        super().__init__()
        if (not isinstance(callback, types.FunctionType)
                or len(inspect.signature(callback).parameters) != 1):
            raise ValueError("Expected callback with 1 parameter for samples")
        self.callback = callback
        self.constructor_params = constructor_params

    def __enter__(self):
        self.start_thread(**self.constructor_params)

    def __exit__(self, *exc_info):
        self.stop_thread()
        samples = self.thread_samples()
        self.callback(samples)
