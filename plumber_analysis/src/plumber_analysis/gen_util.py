"""
Utilities to benchmark and count elements from tf.data dataset.
"""

import time

import numpy as np
import tensorflow as tf

import psutil
import subprocess
import os
import threading
import logging
import inspect

from plumber_analysis import statistics_collection

AUTOTUNE = -1

def is_dataset_closure(dataset) -> bool:
    """Checks if a 0-arg closure of a Dataset"""
    if callable(dataset):
        sig = inspect.signature(dataset)
        if len(sig.parameters):
            raise ValueError("Only 0-argument closures are allowed. Found:\n{}".format(
                sig.parameters))
        return True
    elif isinstance(dataset, tf.data.Dataset):
        return False
    else:
        raise ValueError("Unknown type: {}".format(type(dataset)))


def num_closure_args(closure) -> int:
    """Checks if a 0-arg closure of a Dataset"""
    if not callable(closure):
        raise ValueError("Closure must be a closure: {}".format(closure))
    sig = inspect.signature(closure)
    return len(sig.parameters)


def default_element_count_f(data) -> int:
    """Assumes first dimension is batch size
    NOTE(mkuchnik): This is wrong possibly when images are transposed
    For example, if batch is transposed to last dimension, this will take an
    image width or height, which will be e.g., 224 rather than batch dimensions
    of 128 or 256.
    """
    if isinstance(data, tuple):
        x = data[-1] # Take last element, unlikely to be transposed
        return len(x)
    else:
        return len(data)


def scalar_element_count_f(data) -> int:
    return 1


def add_analysis_to_dataset_options(options, hard_fail: bool=False,
                                    stats_filename="stats.pb",
                                    dump_period_s=None):
    # To enable backward compat
    try:
        options.experimental_optimization.autotune_stats_filename = \
            stats_filename
    except AttributeError as ex:
        if hard_fail:
            raise ex
        else:
            logging.warning("Failed to add analysis stats_filename!"
                            "\n{}".format(ex))
        return
    if dump_period_s is not None:
        options.experimental_optimization.autotune_stats_dump_period = \
            int(round(dump_period_s * 1000))
    return options

def benchmark_and_profile_dataset_fn(*args, **kwargs):
    # Note: this is VERY memory heavy and can cause OOM killer.
    options = tf.profiler.experimental.ProfilerOptions()
    with tf.profiler.experimental.Profile('logdir', options=options):
        return benchmark_dataset_fn(*args, **kwargs)

class AutotuneState(object):
    def __init__(self, error_threshold=None):
        if error_threshold is None:
            error_threshold = 0.01
        self.error_threshold = error_threshold
        self.current_rate_offset = None
        self.past_rate_offset = None
    def current_error(self) -> float:
        if self.past_rate_offset is None:
            return self.error_threshold + 1.
        error = abs((self.past_rate_offset / self.current_rate_offset) - 1)
        return error
    def error_converged(self) -> bool:
        error = self.current_error()
        return error < self.error_threshold
    def record_rate(self, rate: float):
        self.past_rate_offset = self.current_rate_offset
        self.current_rate_offset = rate
    def should_terminate(self) -> bool:
        return self.past_rate_offset and self.error_converged()


def benchmark_dataset_fn(dataset_fn, element_count_f=None, warmup:bool=False,
                         time_limit_s: int=None,
                         print_perf: bool=True, profile_interval: int=10,
                         skip_first_n: int=None, lean_bench: bool=True,
                         return_monitoring_data: bool=False,
                         compute_time_s: int=None,
                         num_compute_accelerators: int=None,
                         min_time_limit_s: int=None,
                         max_time_limit_s: int=None):
    """Useful for graph-mode. Will dispatch to a primitive benchmark """
    if tf.executing_eagerly():
        dataset = dataset_fn()
        kwargs = {"dataset": dataset,
                  "element_count_f": element_count_f,
                  "warmup": warmup,
                  "time_limit_s": time_limit_s,
                  "print_perf": print_perf,
                  "profile_interval": profile_interval,
                  "skip_first_n": skip_first_n,
                  "lean_bench": lean_bench,
                  "return_monitoring_data": return_monitoring_data,
                  "compute_time_s": compute_time_s,
                  "num_compute_accelerators": num_compute_accelerators,
                  "min_time_limit_s": min_time_limit_s,
                  "max_time_limit_s": max_time_limit_s,
                }
        return _benchmark_dataset(**kwargs)
    else:
        _ds = dataset_fn()
        logging.warning("Using limited capability graph-mode benchmark with "
                        "{}".format(_ds))
        if min_time_limit_s is None:
            min_time_limit_s = 12
        global_minibatch_rate = _benchmark_dataset_graphmode(
            dataset_fn=dataset_fn, time_limit_s=time_limit_s,
            min_time_limit_s=min_time_limit_s,
        )
        global_element_rate = None
        data_load_times = None
        logging.info("mean minibatch rate: {} minibatch/sec".format(
            global_minibatch_rate))
        summary = {
            "global_element_rate": global_element_rate,
            "global_minibatch_rate": global_minibatch_rate,
            "data_load_times": data_load_times,
        }
        return summary

def benchmark_dataset(*args, **kwargs):
    return _benchmark_dataset(*args, **kwargs)

def benchmark_summary_to_monitoring_df(samples: statistics_collection.Samples):
    """Input is summary's monitoring (summary is a dict with monitoring being
    inner list of samples). Outputs a dataframe.

    Pass for input: samples = summary["Monitoring"]
    """
    import pandas as pd
    assert samples is not None
    assert isinstance(samples, list), \
        "Expected list, got {}".format(type(samples))
    samples = list(
        map(lambda x: statistics_collection.flatten_sample(x), samples))
    df = pd.DataFrame(samples)
    if "timestep" in df:
        df["timestep"] = pd.to_datetime(df["timestep"])
    return df

def start_profile_server():
    # Note: Do not use this with profile dataset. They clash.
    tf.profiler.experimental.server.start(6009)

def benchmark_and_profile_dataset(*args, **kwargs):
    # Note: this is VERY memory heavy and can cause OOM killer.
    options = tf.profiler.experimental.ProfilerOptions()
    with tf.profiler.experimental.Profile('logdir', options=options):
        return _benchmark_dataset(*args, **kwargs)


def _benchmark_dataset_graphmode(
        dataset_fn, time_limit_s, min_time_limit_s):
    """Severly limited version of benchmark_dataset used for running under
    graph-mode."""
    minibatches_produced = 0
    if not is_dataset_closure(dataset_fn):
        raise ValueError("Not a valid closure. Recieved "
                         "{}".format(type(dataset_fn)))
    graph = tf.compat.v1.Graph()
    auto_state = AutotuneState()
    with graph.as_default():
        dataset = dataset_fn()
        sess = tf.compat.v1.Session(graph=graph)
        with sess.as_default() as sess:
            iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
            next_element = iterator.get_next()
            start_time = tf.timestamp("start_time")
            end_time = tf.timestamp("end_time")

            sess.run(tf.compat.v1.global_variables_initializer())

            # Warmup
            try:
                data = sess.run(next_element)
            except tf.errors.OutOfRangeError as ex:
                logging.error("Exception on warmup: {}".format(ex))
                raise ex

            # Start timers
            _start_time = sess.run(start_time)
            def elapsed_time_fn():
                elapsed_time = end_time - _start_time
                return elapsed_time
            elapsed_time_tensor = elapsed_time_fn()

            while True:
                try:
                    data = sess.run(next_element)
                    minibatches_produced += 1
                    elapsed_time = sess.run(elapsed_time_tensor)
                    if minibatches_produced % 20 == 0:
                        logging.info("Elapsed_time: {}".format(elapsed_time))
                    if time_limit_s >= 0:
                        if elapsed_time > time_limit_s:
                            raise StopIteration("Out of time")
                    else:
                        if minibatches_produced % 10 == 0:
                            global_minibatch_rate_tensor = (
                                tf.cast(minibatches_produced, tf.float64) / elapsed_time)
                            global_minibatch_rate = sess.run(global_minibatch_rate_tensor)
                            auto_state.record_rate(global_minibatch_rate)
                            if elapsed_time >= min_time_limit_s and auto_state.should_terminate():
                                logging.info("AUTOTUNE converged (error {:.2}, time {:.2}, iterations {}):"
                                         " Breaking benchmark".format(
                                    auto_state.current_error(), elapsed_time, minibatches_produced))
                except tf.errors.OutOfRangeError as ex:
                    logging.info("Exception: {}".format(ex))
                    break
                except StopIteration as ex:
                    logging.info("Exception: {}".format(ex))
                    break
            global_minibatch_rate_tensor = (
                tf.cast(minibatches_produced, tf.float64) / elapsed_time)
            global_minibatch_rate = sess.run(global_minibatch_rate_tensor)
    return global_minibatch_rate

def _benchmark_dataset(dataset, element_count_f=None, warmup:bool=False,
                       time_limit_s: int=None,
                       print_perf: bool=True, profile_interval: int=10,
                       skip_first_n: int=None, lean_bench: bool=True,
                       return_monitoring_data: bool=False,
                       compute_time_s: int=None,
                       num_compute_accelerators: int=None,
                       min_time_limit_s: int=None,
                       max_time_limit_s: int=None):
    """Benchmarks the dataset for time_limit_s seconds.
    If time_limit_s == AUTOTUNE, then utilize sliding window to get converged rate."""
    if profile_interval:
        monitor = statistics_collection.ApplicationMonitoringThreadManager()
        monitor.start_thread(sleep_time=profile_interval)
    if min_time_limit_s is None:
        min_time_limit_s = 12
    if num_compute_accelerators is None:
        num_compute_accelerators = 1
    assert isinstance(min_time_limit_s, (int, float))
    assert num_compute_accelerators > 0
    if warmup:
        for data in dataset:
            pass
    a = tf.constant([1])
    b = a * a
    if element_count_f is None:
        element_spec = dataset.element_spec
        if isinstance(element_spec, dict):
            output_shapes = [e.shape for e in element_spec.values()]
        elif isinstance(element_spec, tuple):
            first_element_spec = element_spec[0]
            if isinstance(first_element_spec, dict):
                output_shapes = [e.shape for e in first_element_spec.values()]
            else:
                try:
                    output_shapes = [e.shape for e in first_element_spec]
                except:
                    output_shapes = [first_element_spec.shape]
        else:
            try:
                output_shapes = [e.shape for e in element_spec]
            except TypeError:
                # Not iterable
                output_shapes = [element_spec.shape]
        if len(output_shapes) == 1 and not len(output_shapes[0]):
            element_count_f = scalar_element_count_f
        else:
            element_count_f = default_element_count_f
    data_load_times = []
    start_time = time.time()
    start_process_time = time.process_time()
    global_start_time = start_time
    global_start_process_time = start_process_time
    elements_produced = 0
    minibatches_produced = 0
    total_perfs = []
    def init():
        data_load_times = []
        start_time = time.time()
        start_process_time = time.process_time()
        global_start_time = start_time
        global_start_process_time = start_process_time
        elements_produced = 0
        minibatches_produced = 0
        total_perfs = []
    init()

    def reporting_function_autotune(auto_state, is_running):
        wait_time = 10
        while True:
           error = auto_state.current_error()
           past = auto_state.past_rate_offset
           current = auto_state.current_rate_offset
           elapsed_time = time.time() - global_start_time
           logging.info("Current autotune error: {} ({}, {}, {}s)".format(
               error, past, current, elapsed_time))
           if max_time_limit_s is not None:
               remaining_time = max_time_limit_s - elapsed_time
               if remaining_time < 0:
                   is_running.set()
               remaining_time = max(remaining_time, 0)
           else:
               remaining_time = wait_time
           not_running = is_running.wait(min(remaining_time, wait_time))
           if not_running:
               logging.info("Reporting thread exiting")
               break

    def reporting_function_default(is_running):
        if time_limit_s is None:
            return
        wait_time = 10
        while True:
           elapsed_time = time.time() - global_start_time
           remaining_time = time_limit_s - elapsed_time
           logging.info("Current time left: {}s ({}s)".format(
               remaining_time, elapsed_time))
           if remaining_time < 0:
               is_running.set()
           remaining_time = max(remaining_time, 0)
           not_running = is_running.wait(min(remaining_time, wait_time))
           if not_running:
               logging.info("Reporting thread exiting")
               break

    auto_state = AutotuneState()
    is_running = threading.Event()
    if time_limit_s == AUTOTUNE:
        reporting_function = reporting_function_autotune
        reporting_args = (auto_state, is_running)
    else:
        reporting_function = reporting_function_default
        reporting_args = (is_running,)

    reporting_thread = threading.Thread(target=reporting_function,
                                        name="benchmark_reporting_thread",
                                        args=reporting_args)
    reporting_thread.start()

    # TODO(mkuchnik): Move to thread-based implementation to avoid high
    # iterations counts.
    rate_record_frequency = 20

    if lean_bench:
        ds = iter(dataset)
        try:
            while True:
                with tf.profiler.experimental.Trace('train',
                                                    step_num=minibatches_produced,
                                                    _r=1):
                    data = next(ds)
                elements_produced += element_count_f(data)
                minibatches_produced += 1
                start_time = time.time()
                elapsed_time = start_time - global_start_time
                if time_limit_s != AUTOTUNE:
                    if time_limit_s and elapsed_time > time_limit_s:
                        logging.info("TimeElapsed: Breaking benchmark")
                        break
                else:
                    if (minibatches_produced % rate_record_frequency) == 0:
                        auto_state.record_rate(minibatches_produced / elapsed_time)
                        if elapsed_time >= min_time_limit_s and auto_state.should_terminate():
                            logging.info("AUTOTUNE converged (error {:.2}, time {:.2}, iterations {}):"
                                     " Breaking benchmark".format(
                                auto_state.current_error(), elapsed_time, minibatches_produced))
                            break
                        elif not is_running:
                            logging.info("AUTOTUNE failed converged (error {:.2}, time {:.2}, iterations {}):"
                                     " Breaking benchmark".format(
                                auto_state.current_error(), elapsed_time, minibatches_produced))
                            break
                if (compute_time_s
                        and (minibatches_produced % num_compute_accelerators) == 0):
                    time.sleep(compute_time_s)
        except StopIteration:
            logging.info("StopIteration: Breaking benchmark")
        global_end_time = time.time()
        global_end_process_time = time.process_time()
        del ds
    else:
        ds = iter(dataset)
        i = 0
        try:
            if skip_first_n:
                for i in range(skip_first_n):
                    _ = next(ds)
                init()
            i = 0
            while True:
                with tf.profiler.experimental.Trace('train',
                                                    step_num=minibatches_produced,
                                                    _r=1):
                    data = next(ds)
                i += 1
                end_time = time.time()
                end_process_time = time.process_time()
                elements_produced += element_count_f(data)
                minibatches_produced += 1
                data_load_time = end_time - start_time
                data_load_times.append(data_load_time)
                start_time = time.time()
                start_process_time = time.process_time()
                elapsed_time = start_time - global_start_time
                global_end_time = start_time
                global_end_process_time = start_process_time
                if time_limit_s != AUTOTUNE:
                    if time_limit_s and elapsed_time > time_limit_s:
                        logging.info("TimeElapsed: Breaking benchmark")
                        break
                else:
                    if (minibatches_produced % rate_record_frequency) == 0:
                        auto_state.record_rate(minibatches_produced / elapsed_time)
                        if elapsed_time >= min_time_limit_s and auto_state.should_terminate():
                            logging.info("AUTOTUNE converged (error {:.2}, time {:.2}, iterations {}):"
                                     " Breaking benchmark".format(
                                auto_state.current_error(), elapsed_time, minibatches_produced))
                            break
                        elif not is_running:
                            logging.info("AUTOTUNE failed converged (error {:.2}, time {:.2}, iterations {}):"
                                     " Breaking benchmark".format(
                                auto_state.current_error(), elapsed_time, minibatches_produced))
                            break
                if (compute_time_s
                        and (minibatches_produced % num_compute_accelerators) == 0):
                    time.sleep(compute_time_s)
        except StopIteration:
            global_end_time = start_time
            global_end_process_time = start_process_time
            logging.info("StopIteration: Breaking benchmark")
        del ds

    is_running.set()
    if time_limit_s == AUTOTUNE:
        reporting_thread.join(10)
        if reporting_thread.is_alive():
            raise RuntimeError("Failed to kill reporting thread")
    total_time = global_end_time - global_start_time
    logging.info("total_time: {}s".format(total_time))
    total_process_time = global_end_process_time - global_start_process_time
    logging.info("total_process_time: {}s (normalized {}s)".format(
        total_process_time, total_process_time / total_time))
    global_element_rate = elements_produced / total_time
    global_minibatch_rate = minibatches_produced / total_time
    if data_load_times:
        mean_data_load_time = np.mean(data_load_times)
        var_data_load_time = np.var(data_load_times)
        logging.info("mean data_load_time: {} sec (var: {})".format(
            mean_data_load_time, var_data_load_time))
    logging.info("mean minibatch rate: {} minibatch/sec".format(global_minibatch_rate))
    logging.info("mean element rate: {} elements/sec".format(global_element_rate))
    summary = {
        "global_element_rate": global_element_rate,
        "global_minibatch_rate": global_minibatch_rate,
        "data_load_times": data_load_times,
    }
    a = tf.constant([1])
    b = a * a
    if profile_interval:
        monitor.stop_thread()
        samples = monitor.thread_samples()
        assert isinstance(samples, list)
        samples_df = benchmark_summary_to_monitoring_df(samples)
        total_cpu_perfs = list(map(lambda x: x.cpu_frequency, samples))
        summary["Perf_CPU_util"] = float(np.mean(total_cpu_perfs))
        total_mem_perfs = list(map(lambda x: x.memory_usage.rss, samples))
        summary["Perf_memory_usage"] = total_mem_perfs
    else:
        summary["Perf_CPU_util"] = None
        summary["Perf_memory_usage"] = None
    if return_monitoring_data:
        return summary, samples_df
    else:
        return summary

def _get_threads_cpu_percent(p, interval=0.1):
   """ Gets the CPU percentage of each thread in process p
   From https://stackoverflow.com/questions/26238184/cpu-usage-per-thread
   """
   total_percent = p.cpu_percent(interval)
   total_time = sum(p.cpu_times())
   return [total_percent * ((t.system_time + t.user_time)/total_time)
           for t in p.threads()]

def get_threads_cpu_percent(interval=0.1):
   proc = psutil.Process() # Current pid
   total_time = _get_threads_cpu_percent(proc, interval)
   return total_time

def get_size_of_dataset(dataset):
    """Size of dataset when dataset is string"""
    size = 0
    for i, x in enumerate(dataset):
        size += int(tf.strings.length(x).numpy())
        if i % 100000:
            logging.info("Size is {} GB".format(size / 1e9))
    logging.info("Size is {}".format(size))
    return size

def get_size_of_dataset_2(dataset):
    """Size of dataset when dataset is fp16 and int32"""
    size_x = 0
    size_y = 0
    def size():
        return 2 * size_x + 4 * size_y
    for i, x in enumerate(dataset):
        xx = x[0].numpy().flatten()
        yy = x[1].numpy().flatten()
        size_x += len(xx)
        size_y += len(yy)
        if i % 100000:
            logging.info("Size is {} GB".format(size() / 1e9))
    logging.info("Size is x={}, y={}, {}".format(size_x, size_y, size()))
    return size_x, size_y, size()

def drop_caches():
    logging.info("Trying to drop system caches")
    command = ["sync"]
    ret = subprocess.check_output(command, encoding="UTF-8")
    command = "sudo /sbin/sysctl vm.drop_caches=3"
    os.system(command)
    #ret = subprocess.check_output(command, encoding="UTF-8")
    #return ret

