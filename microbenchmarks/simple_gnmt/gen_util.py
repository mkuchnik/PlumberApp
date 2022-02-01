import time

import numpy as np
import tensorflow as tf

import psutil

def default_element_count_f(data):
    if isinstance(data, tuple):
        x = data[0] # Take first element
        return len(x)
    else:
        return len(data)

def scalar_element_count_f(data):
    return 1

def benchmark_dataset(*args, **kwargs):
    return _benchmark_dataset(*args, **kwargs)

def benchmark_and_profile_dataset(*args, **kwargs):
    with tf.profiler.experimental.Profile('logdir'):
        return _benchmark_dataset(*args, **kwargs)

def _benchmark_dataset(dataset, element_count_f=None, warmup=False,
                       time_limit_s=None,
                       print_perf=True, profile_interval=10,
                       skip_first_n=None):
    if warmup:
        for data in dataset:
            pass
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
            output_shapes = [e.shape for e in element_spec]
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
    proc = psutil.Process() # Current pid
    proc.cpu_percent() # Init
    def init():
        data_load_times = []
        start_time = time.time()
        start_process_time = time.process_time()
        global_start_time = start_time
        global_start_process_time = start_process_time
        elements_produced = 0
        minibatches_produced = 0
        total_perfs = []
        proc.cpu_percent() # Init
    for i, data in enumerate(dataset):
        if skip_first_n and i < skip_first_n:
            if i == skip_first_n - 1:
                init()
            else:
                continue
        end_time = time.time()
        end_process_time = time.process_time()
        elements_produced += element_count_f(data)
        minibatches_produced += 1
        data_load_time = end_time - start_time
        data_load_times.append(data_load_time)
        start_time = time.time()
        start_process_time = time.process_time()
        elapsed_time = start_time - global_start_time
        if profile_interval and i % profile_interval == 0:
            total_cpu_percent = proc.cpu_percent()
            total_perfs.append(total_cpu_percent)
            if print_perf:
                print("Perf: {}".format(total_cpu_percent))
            #perc = get_threads_cpu_percent(interval=0.1)
            #total = np.sum(perc)
            #total_perfs.append(total)
        global_end_time = start_time
        global_end_process_time = start_process_time
        if time_limit_s and elapsed_time > time_limit_s:
            break
    total_time = global_end_time - global_start_time
    total_process_time = global_end_process_time - global_start_process_time
    print("total_process_time: {}s (normalized {}s)".format(total_process_time,
                                                            total_process_time/total_time))
    global_element_rate = elements_produced / total_time
    global_minibatch_rate = minibatches_produced / total_time
    mean_data_load_time = np.mean(data_load_times)
    var_data_load_time = np.var(data_load_times)
    print("mean data_load_time: {} sec (var: {})".format(mean_data_load_time,
                                                         var_data_load_time))
    print("mean minibatch rate: {} minibatch/sec".format(global_minibatch_rate))
    print("mean element rate: {} elements/sec".format(global_element_rate))
    summary = {
        "global_element_rate": global_element_rate,
        "global_minibatch_rate": global_minibatch_rate,
    }
    if profile_interval:
        summary["Perf_CPU_util"] = float(np.mean(total_perfs))
    return summary

def _get_threads_cpu_percent(p, interval=0.1):
   """ Gets the CPU percentage of each thread in process p
   From https://stackoverflow.com/questions/26238184/cpu-usage-per-thread
   """
   total_percent = p.cpu_percent(interval)
   total_time = sum(p.cpu_times())
   return [total_percent * ((t.system_time + t.user_time)/total_time) for t in p.threads()]

def get_threads_cpu_percent(interval=0.1):
   proc = psutil.Process() # Current pid
   total_time = _get_threads_cpu_percent(proc, interval)
   return total_time