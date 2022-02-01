"""High-level pipeline optimizer.

The goal here is to populate resource and pipeline metadata
and create optimizer instances acting on that metadata.

Don't use this API directly. Use annotations.
"""

import multiprocessing
import functools
import logging
import os
import itertools
import distutils.util
import pickle
import pprint
import time

import networkx as nx
import numpy as np
import pandas as pd
import tensorflow as tf

from plumber_analysis import pipeline_optimizer, gen_util, extensions, bandwidth_utilities
import plumber_analysis.machine_info

PARAMS_FILENAME = "params.p"

def create_optimizer():
    """Creates an optimizer using machine info and a stats filename.

    NOTE: these values are currently hackily autofilled using heuristics
    and OS queries. They also rely on convention. For example, we assume
    the filename is "stats.pb".
    """
    # NOTE(mkuchnik): The metadata can actually be populated automatically from
    # See https://cloud.google.com/compute/docs/metadata/overview
    # and https://stackoverflow.com/questions/31688646/get-the-name-or-id-of-the-current-google-compute-instance
    # Instances can also be procured via
    # https://cloud.google.com/compute/docs/tutorials/python-guide

    filename = "stats.pb"  # TODO(mkuchnik): Don't hardcode

    # Note(mkuchnik): we can instantiate a machine_info here like this:
    # machine_info = {'HOSTNAME': 'Localhost', 'CORES': 96, 'MEMORY': int(299e9),
    #                 'FILES': [{'PATH': '/mnt/datadisks/data/',
    #                                     'BANDWIDTH': 75e6}]}

    # TODO(mkuchnik): Add space to Paths to use as a cache
    _machine_dict = plumber_analysis.machine_info.generate_localhost_machine_dict()
    current_dir = os.getcwd()
    _machine_dict["FILES"] = []
    _machine_dict["FILES"].append({"PATH": current_dir, "BANDWIDTH": int(10e6), "AVAILABLE_SPACE": int(1e9)})
    logging.info("Detected machine info: {}".format(_machine_dict))
    plumber = tf.data.experimental.analysis.PlumberPerformanceModel(filename)
    optimizer = pipeline_optimizer.DataPipelineOptimizer(plumber,
                                                         calibrate_system=False,
                                                         machine_info=_machine_dict,
                                                         step_size=None)
    # NOTE(mkuchnik): We can use other optimizers, like cost:
    # optimizer = pipeline_optimizer.CostBasedDataPipelineOptimizer(
    #     plumber, calibrate_system=False, step_size=None, min_rate=30)
    return optimizer

def step_par_0():
    """Apply first parallelism optimizations."""
    logging.info("par 0")
    optimizer = create_optimizer()
    optimizer.apply_parallelism()
    return optimizer

def step_cache_0():
    """Apply Cache optimizations."""
    logging.info("cache 0")
    optimizer = create_optimizer()
    logging.info(optimizer.get_cache_summary())
    optimizer.apply_parallelism()
    optimizer.apply_cache()
    return optimizer

def step_par_1():
    """Apply second parallelism optimizations."""
    logging.info("par 1")
    optimizer = create_optimizer()
    optimizer.apply_parallelism()
    optimizer.apply_cache()
    optimizer.update_plumber()
    optimizer.apply_parallelism()
    return optimizer

def step_par_2(apply_cache=True, num_steps=None, load_parameter_cache=False,
               is_fast=False, remove_caches=True):
    """Apply general optimizations."""
    logging.info("Step par 2:\n{}".format(locals()))
    logging.info("par 2")
    times_and_names = []
    def record_event(name):
        # Key, value pairs
        times_and_names.append((name, time.perf_counter()))
    record_event("start_time")
    def report_events_dict():
        start_time_kv = times_and_names[0]
        name, value = start_time_kv
        assert name == "start_time"
        start_time = value
        report_dict = dict()
        for name, value in times_and_names: 
            report_dict[name] = value - start_time
        return report_dict

    logging.info("Creating optimizer")
    optimizer = create_optimizer()
    logging.info("Creation finished for optimizer")
    record_event("optimizer_end_time") # 1s
    logging.info("IS_FAST={}".format(is_fast))
    if load_parameter_cache:
        try:
            params_filename = PARAMS_FILENAME
            with open(params_filename, "rb") as params_f:
                params = pickle.load(params_f)
                experiment_params = params["experiment_params"]
                optimizer.set_experiment_params(experiment_params)
            logging.info("Successfully loaded params '{}': {}".format(
                params_filename,
                experiment_params))
        except FileNotFoundError as ex:
            logging.info("{}".format(ex))
            if not is_fast:
                logging.info("Calibrating Source Parallelism")
                optimizer = calibrate_source_parallelisms(optimizer,
                        override_presets=True, sweep_range=None)
    else:
        if not is_fast:
            logging.info("Calibrating Source Parallelism")
            optimizer = calibrate_source_parallelisms(optimizer, override_presets=True, sweep_range=None)
    record_event("optimizer_calibrate_time") # 70s
    G = optimizer.networkx()
    nx.drawing.nx_pydot.write_dot(G, "networkx_init.dot")

    logging.info("Starting Optimization Passes")
    if remove_caches:
        logging.info("Removing caches")
        optimizer.apply_extension(extensions.RemoveCaches())
        record_event("optimizer_pass_remove_cache") # 126s
    logging.info("Applying first paralellism pass")
    optimizer.apply_parallelism()
    record_event("optimizer_pass_parallelism_0") # 126s
    if apply_cache:
        # TODO(mkuchnik): By avoiding this, disk bottlenecks can (and will) appear
        # TODO(mkuchnik): Control take repeat
        add_take_repeat = True
        logging.info("Applying cache")
        optimizer.apply_cache(add_take_repeat=add_take_repeat)
        record_event("optimizer_pass_apply_cache_0") # 120s
        with pd.option_context('display.max_rows', None,
                               'display.max_columns', None):
            logging.info("cache summary:\n{}".format(optimizer.get_cache_summary()))
    if num_steps is None or num_steps >= 2:
        # NOTE(mkuchnik): is_fast can be used here, but makes num_steps useless
        logging.info("Updating Plumber")
        optimizer.update_plumber()
        record_event("optimizer_update_plumber") # 210s
        logging.info("Applying second parallelism pass")
        optimizer.apply_parallelism()
        record_event("optimizer_pass_parallelism_1") # 210s
    logging.info("Optimizer time breakdown:\n{}".format(
        pprint.pformat(report_events_dict())))
    G = optimizer.networkx()
    nx.drawing.nx_pydot.write_dot(G, "networkx_final.dot")
    return optimizer

def sweep():
    sweep_range = [step_par_0, step_cache_0, step_par_1]
    #sweep_range = [step_cache_0]
    for f in sweep_range:
        optimizer = f()
        dataset = optimizer.instantiate_pipeline()
        gen_util.benchmark_dataset(dataset, time_limit_s=62)

def apply_tracing(dataset):
    options = tf.data.Options()
    gen_util.add_analysis_to_dataset_options(options)
    dataset = dataset.with_options(options)
    return dataset

def apply_default_options(dataset, override_presets, apply_tracing=True):
    options = tf.data.Options()
    if apply_tracing:
        gen_util.add_analysis_to_dataset_options(options)
    # TODO(mkuchnik): Move to seperate optimization pass
    # it seems that these defaults are safe no matter what
    if override_presets:
        options.experimental_deterministic = False
        options.experimental_threading.max_intra_op_parallelism = 1
        # TODO(mkuchnik): Consider setting this conservatively e.g., if not necessary for rate
        # TODO(mkuchnik): Do this in a subsequent optimization pass, by taking the min rate
        # and extrapolating from LP rate
        options.experimental_threading.private_threadpool_size = multiprocessing.cpu_count()
    dataset = dataset.with_options(options)
    return dataset

def plumber_find_best_pipeline():
    env_key = "PLUMBER_FIND_BEST_PIPELINE"
    return find_bool_env_key(env_key, False)

def plumber_fake_pipeline():
    env_key = "PLUMBER_FAKE_PIPELINE"
    return find_bool_env_key(env_key, False)

def plumber_optimize_pipeline():
    env_key = "PLUMBER_OPTIMIZE_PIPELINE"
    return find_bool_env_key(env_key, False)

def find_bool_env_key(env_key: str, default_value: bool) -> bool:
    env_ret = os.environ.get(env_key)
    if env_ret is not None:
        env_ret_bool = bool(distutils.util.strtobool(env_ret))
        logging.info("{} is {}".format(
            env_key, env_ret_bool))
    else:
        env_ret_bool = default_value
    return env_ret_bool

def plumber_fast_optimize():
    env_key = "PLUMBER_FAST_OPTIMIZE"
    return find_bool_env_key(env_key, False)

def plumber_remove_caches():
    env_key = "PLUMBER_REMOVE_CACHES"
    return find_bool_env_key(env_key, True)

def plumber_no_optimize():
    env_key = "PLUMBER_NO_OPTIMIZE"
    return find_bool_env_key(env_key, False)

def get_optimized_pipeline(dataset, override_presets=True, return_test_dataset=False, return_rate=False, use_parameter_cache=False, return_optimizer=False):
    """Given a dataset, automatically applies plumber optimizations to it"""
    # TODO(mkuchnik): Seperate optimization pass
    # TODO(mkuchnik): Factor out environment getting code
    no_opt = plumber_no_optimize()
    if no_opt:
        logging.info("SKIPPING OPTIMIZATION")
        return dataset

    env_override_presets = os.environ.get("PLUMBER_OVERRIDE_PRESETS")
    if env_override_presets is not None and env_override_presets != override_presets:
        old_override_presets = bool(override_presets)
        override_presets = bool(distutils.util.strtobool(env_override_presets))
        logging.info("PLUMBER_OVERRIDE_PRESETS is {}->{} (was {})".format(
            env_override_presets, override_presets, old_override_presets))
    apply_caching = True

    env_apply_caching = os.environ.get("PLUMBER_APPLY_CACHING")
    if env_apply_caching is not None and env_apply_caching != apply_caching:
        old_apply_caching = bool(apply_caching)
        apply_caching = bool(distutils.util.strtobool(env_apply_caching))
        logging.info("PLUMBER_APPLY_CACHING is {}->{} (was {})".format(
            env_apply_caching, apply_caching, old_apply_caching))

    num_steps = None
    env_num_steps = os.environ.get("PLUMBER_NUM_OPTIMIZATION_STEPS")
    if env_num_steps is not None and env_num_steps != num_steps:
        old_num_steps = num_steps
        num_steps = int(env_num_steps)
        logging.info("PLUMBER_NUM_OPTIMIZATION_STEPS is {}->{} (was {})".format(
            env_num_steps, num_steps, old_num_steps))

    is_fast = plumber_fast_optimize()
    remove_caches = plumber_remove_caches()

    logging.info("Running preliminary benchmark")
    dataset = apply_default_options(dataset, override_presets)
    ret = gen_util.benchmark_dataset(dataset, time_limit_s=12)
    logging.info("End preliminary benchmark")
    logging.info("benchmark {}".format(ret))

    optimizer = step_par_2(
        apply_caching, num_steps, load_parameter_cache=use_parameter_cache,
        is_fast=is_fast, remove_caches=remove_caches)
    new_dataset = optimizer.instantiate_pipeline()
    new_dataset = apply_default_options(new_dataset, override_presets)

    ret = [new_dataset]
    if return_test_dataset:
        test_dataset = optimizer.instantiate_test_pipeline()
        test_dataset = apply_default_options(test_dataset, override_presets)
        ret.append(test_dataset)
    experiment_params = optimizer.experiment_params()
    performance_params = optimizer.get_performance_parameters()
    all_params = {
            "experiment_params": experiment_params,
            "performance_params": performance_params,
            }
    if use_parameter_cache:
        params_filename = PARAMS_FILENAME
        with open(params_filename, "wb") as param_f:
            pickle.dump(all_params, param_f)
    logging.info("Experimental params:\n{}".format(experiment_params))
    logging.info("Plumber found parameters:\n{}".format(performance_params))
    if return_rate:
        rate = optimizer.current_plumber_emperical_rate()
        ret.append(rate)
    if return_optimizer:
        ret.append(optimizer)
    if len(ret) == 1:
        ret = ret[0]
    else:
        ret = tuple(ret)
    return ret

def get_fake_pipeline(dataset, override_presets=True):
    dataset = apply_default_options(dataset, override_presets)
    ret = gen_util.benchmark_dataset(dataset, time_limit_s=12)
    logging.info("benchmark {}".format(ret))
    optimizer = create_optimizer()
    dataset = optimizer.fake_dataset()
    return dataset

def get_source_pipeline(dataset, override_presets=True):
    dataset = apply_default_options(dataset, override_presets)
    ret = gen_util.benchmark_dataset(dataset, time_limit_s=12)
    logging.info("benchmark {}".format(ret))
    optimizer = create_optimizer()
    dataset = optimizer.source_dataset()
    return dataset

def get_sweep_source_pipeline_parallelisms(optimizer, sweep_range=None, override_presets=True, max_log_parallelism=None):
    """Returns a sweep of datasets (tuples of parallelism and dataset)
    to be used for source benchmarking"""
    if max_log_parallelism is None:
        # 32
        max_log_parallelism = 5

    if sweep_range is None:
        sweep_range = [2**i  for i in range(max_log_parallelism+1)]

    datasets = []
    for i in sweep_range:
        curr_parallelism = optimizer.get_parallelism()
        logging.info("Found parallelism: {}".format(curr_parallelism))
        source_nodes = optimizer.source_nodes()
        num_source_nodes = 0
        for k in curr_parallelism:
            if k in source_nodes:
                curr_parallelism[k] = i
                num_source_nodes += 1
        if not num_source_nodes:
            logging.info("Didn't find any source nodes.")
            raise RuntimeError("Didn't find any source nodes.")
        assert num_source_nodes == 1, "Expected 1 source node"
        _optimizer = optimizer.fork()
        _optimizer.set_parallelism(curr_parallelism)
        dataset = _optimizer.source_dataset()
        dataset = apply_default_options(dataset, override_presets)
        datasets.append(dataset)
    return zip(sweep_range, datasets), source_nodes[0]

def _benchmark_source_parallelisms(optimizer, override_presets=True, sweep_range=None):
    source_sweep, source_node = get_sweep_source_pipeline_parallelisms(
            optimizer, sweep_range=sweep_range, override_presets=override_presets)

    rets = []
    for p, d in source_sweep:
        dd = d.prefetch(10)
        ret = gen_util.benchmark_dataset(d, time_limit_s=gen_util.AUTOTUNE)
        rets.append((p, ret))

    return rets, source_node

def benchmark_source_parallelisms(dataset, override_presets=True, sweep_range=None):
    """Sweeps a range of parallelism and report benchmark rates"""
    dataset = apply_default_options(dataset, override_presets, apply_tracing=False)
    ret = gen_util.benchmark_dataset(dataset, time_limit_s=12)
    logging.info("benchmark {}".format(ret))
    optimizer = create_optimizer()
    rets, _ = _benchmark_source_parallelisms(optimizer, override_presets, sweep_range)
    return rets

def calibrate_source_parallelisms(optimizer, override_presets=True, sweep_range=None, add_zeros=False):
    """Does a sweep over source node parallelisms and adds this metadata to the optimizer"""
    try:
        rets, source_node = _benchmark_source_parallelisms(optimizer, override_presets, sweep_range)
    except RuntimeError as ex:
        logging.error(ex)
        return optimizer
    xy = list(map(lambda x: (x[0], x[1]["global_minibatch_rate"]), rets))
    if add_zeros:
        # NOTE(mkuchnik): Adding this makes the solution degenerate, probably
        xy.insert(0, (0, 0.))
    x, y = zip(*xy)
    logging.info("Source data collected (x, y):\n{}".format(pprint.pformat(xy)))
    regressor, params = bandwidth_utilities.find_best_piecewise_linear_fit(x, y)
    # NOTE(mkuchnik): To test quickly, pass the following:
    # params = {'m1': 2051.0214506645075, 'b1': 6396.417319298994, 'm2': -34.03928392060331,
    #         'b2': 10503.24875052317,
    #          'x_thresh': 2,
    #          'source_node': 'ParallelInterleaveDatasetV4/_10'}
    params["source_node"] = source_node
    print("Params for source regression are: {}".format(params))
    optimizer.set_bandwidth_parallelism_equations(params)
    return optimizer

def _all_equal(iterable):
    g = itertools.groupby(iterable)
    return next(g, True) and not next(g, False)

def _assert_element_spec_equality(datasets):
    """All datasets must have the same signature"""
    element_specs = [d.element_spec for d in datasets]
    if not _all_equal(element_specs):
        raise ValueError("Element spec mismatch:\n{}".format(element_specs))


def get_best_optimized_pipeline(datasets, override_presets=True, double_test=False):
    """Given a list of datasets, automatically applies plumber optimizations to it and return the fastest"""
    if len(datasets) == 1:
        return datasets[0]
    elif len(datasets) < 1:
        raise ValueError("Datasets must be a list of at least length 1")
    _assert_element_spec_equality(datasets)

    rates = []
    estimated_rates = []
    opt_datasets = []
    for p in datasets:
        opt_p, opt_p_test, rate, optimizer = get_optimized_pipeline(p, override_presets, return_test_dataset=True, return_rate=True, return_optimizer=True)
        estimated_rate = optimizer.estimated_rate
        estimated_rates.append(estimated_rate)
        opt_datasets.append(opt_p)
        if double_test:
            ret = gen_util.benchmark_dataset(opt_p_test, time_limit_s=gen_util.AUTOTUNE)
            rates.append(ret["global_minibatch_rate"])
        else:
            rates.append(rate)
    rates = np.array(rates)
    best_i = np.argmax(rates)
    logging.info("The best pipeline is index={} from rates: {}. Estimated: {}".format(
        best_i, rates, estimated_rates))
    ds = opt_datasets[best_i]
    assert ds is not None
    return ds


def optimize_default():
    optimizer = create_optimizer()
    #optimizer.roofline("roofline.pdf", ylim="all")
    #print(optimizer.roofline("roofline.pdf"))
    logging.info(optimizer.roofline())
    #print(optimizer.all_N_stats())
    #optimizer.disable_inter_op_parallelism()
    optimizer.apply_optimizations(benchmark_time_s=22, inner_benchmarking=False,
                                  num_optimization_passes=3,
                                  rebench=True)
    dataset = optimizer.instantiate_pipeline()
    #print(optimizer.roofline("roofline_opt.pdf"))
    gen_util.benchmark_dataset(dataset, time_limit_s=62)
    #
    #options = tf.data.Options()
    #gen_util.add_analysis_to_dataset_options(options)
    #dataset = dataset.with_options(options)
    #print("Benchmarking")
    #print("*" * 80)
    #gen_util.drop_caches()
    #gen_util.benchmark_dataset(dataset, time_limit_s=22)

def main():
    #optimize_default()
    sweep()

if __name__ == "__main__":
    main()

