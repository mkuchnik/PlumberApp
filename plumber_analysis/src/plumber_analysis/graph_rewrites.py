"""
The main implementation of incremental graph_rewrites.
Simulates a person incrementally acting on Plumber's recommendation of the bottleneck.
Also contains random selection baselines.
"""

from absl import app
from absl import flags

import argparse
import copy
import shutil
import pandas as pd
import pathlib
import pprint
import random
import networkx as nx
import numpy as np
import os
import math
from matplotlib import cm
import matplotlib.pyplot as plt
import functools
import logging

import tensorflow as tf
import graphsurgeon

from plumber_analysis import convex_solver, gen_util, graphdef_util
from plumber_analysis import benchmark_util, util, resource_measurements

# TODO(mkuchnik): Remove this dependency
from plumber_analysis import pipeline_optimizer

BATCH_NODE_OPS = set(["BatchDataset", "BatchDatasetV2"])

STRATEGIES = [None, "random", "random_valid"]
SELECTION_MODES = [None, "p_busy"]
DEFAULT_STATS_FILENAME = "stats.pb"
FRACTION_CACHEABLE_MEMORY = 0.9

def apply_default_flags():
    flags.DEFINE_bool('rebench_baseline',
                      default=False,
                      help=('Run benchmarking on pipeline again for baseline.'))
    flags.DEFINE_bool('sweep_nodes',
                      default=False,
                      help=('Run benchmarking on individual nodes.'))
    flags.DEFINE_bool('skip_baseline',
                      default=False,
                      help=('Don\'t evalute first run.'))
    flags.DEFINE_bool('skip_LP_baseline',
                      default=False,
                      help=('Don\'t evalute LP baseline.'))
    flags.DEFINE_bool('skip_system_benchmark',
                      default=True,
                      help=('Don\'t evalute system performance.'))
    flags.DEFINE_bool('skip_rewrites',
                      default=False,
                      help=('Don\'t do rewrites.'))
    flags.DEFINE_integer('num_deviations',
                         default=1,
                         help=('The number of deviations (non-recommendations) to '
                               'run per step. Set to 1 for none.'))
    flags.DEFINE_integer('num_steps',
                         default=55,
                         help=('The number of steps (max) to take.'))
    flags.DEFINE_integer('time_limit_s',
                         default=42,
                         help=('The number of seconds (max) to run.'))
    flags.DEFINE_string('strategy',
                        default=None,
                        help=('The strategy to run. One of {}'.format(STRATEGIES)))
    flags.DEFINE_string('selection_mode',
                        default=None,
                        help=('The strategy to run. One of '
                              '{}'.format(SELECTION_MODES)))
    flags.DEFINE_integer('up_parallelism',
                         default=1,
                         help=('The amount to increase parallelism by'))
    flags.DEFINE_bool('parallelize_batch',
                      default=False,
                      help=('Whether to enable batch parallelization'))

def apply_parser_flags(parser, skip_time_limit_s=False):
    """For argparse"""
    parser.add_argument('--rebench_baseline',
                        action="store_true",
                        help=('Run benchmarking on pipeline again for baseline.'))
    parser.add_argument('--sweep_nodes',
                        action="store_true",
                      help=('Run benchmarking on individual nodes.'))
    parser.add_argument('--skip_baseline',
                        default=False,
                        type=bool,
                        help=('Don\'t evalute first run.'))
    parser.add_argument('--skip_LP_baseline',
                        default=False,
                        type=bool,
                      help=('Don\'t evalute LP baseline.'))
    parser.add_argument('--skip_system_benchmark',
                        default=True,
                        type=bool,
                      help=('Don\'t evalute system performance.'))
    parser.add_argument('--skip_rewrites',
                        default=False,
                        type=bool,
                        help=('Don\'t do rewrites.'))
    parser.add_argument('--num_deviations',
                         default=1,
                        type=int,
                         help=('The number of deviations (non-recommendations) to '
                               'run per step. Set to 1 for none.'))
    parser.add_argument('--num_steps',
                         default=55,
                        type=int,
                         help=('The number of steps (max) to take.'))
    if not skip_time_limit_s:
        parser.add_argument('--time_limit_s',
                             default=42,
                             type=int,
                             help=('The number of seconds (max) to run.'))
    parser.add_argument('--strategy',
                        default=None,
                        type=str,
                        help=('The strategy to run. One of {}'.format(STRATEGIES)))
    parser.add_argument('--selection_mode',
                        default=None,
                        type=str,
                        help=('The strategy to run. One of '
                              '{}'.format(SELECTION_MODES)))
    parser.add_argument('--up_parallelism',
                        default=1,
                        type=int,
                        help=('The amount to increase parallelism by'))
    parser.add_argument('--parallelize_batch',
                        default=False,
                        type=bool,
                        help=('Whether to enable batch parallelization'))
    return parser

def transform_benchmark_summary_to_df(data):
    """ Convenience method for benchmark_summary dicts or lists to dataframe"""
    if isinstance(data, dict):
        data = [data]
    try:
        df = pd.DataFrame(data=data, index=[0])
    except Exception as ex:
        logging.warning("Exception {}: {}".format(ex, data))
        raise ex
    return df

@functools.lru_cache(maxsize=128)
def _filename_to_file_data(filename: str, skip_benchmark: bool) -> dict:
    # Only run benchmark if previous results don't exist
    if skip_benchmark:
        logging.info("SKIPPING BENCHMARKS")
        bw = 100e6
    else:
        logging.info("BENCHMARKING {}".format(filename))
        ret = resource_measurements.benchmark_filesystem(filename, True)
        logging.info("Measurement {}".format(ret))
        bw = ret["read"]["bw"] * 1e3  # KB/s
    return {"PATH": filename,
            "BANDWIDTH": bw}

def detect_suggestion_cycles(rates, changes):
    """A cycle is defined as 5 identical changes have yielded no improvement"""
    def is_plateau(rates):
        # First rate may be None
        if len(rates) < 5:
            return False
        elif len(rates) >= 5 and rates[-5] is None:
            return False
        else:
            baseline = rates[-5]
            improvements = np.array(rates[-4:]) / baseline
            max_improvement = np.max(improvements)
            avg_improvement = np.mean(improvements)
            return max_improvement < 1.10 or avg_improvement <= 1.01
    def is_identical_changes(changes):
        if len(changes) < 5:
            return False
        else:
            changes_dataset_names = list(
                map(graphdef_util.debug_str_to_dataset_name, changes))
            first_change = changes_dataset_names[-5]
            return all(x == first_change for x in changes_dataset_names)
    assert len(rates) == len(changes)
    cycle_nodes = []
    if is_plateau(rates) and is_identical_changes(changes):
        cycle_node = graphdef_util.debug_str_to_dataset_name(changes[-1])
        logging.info("DETECTED CYCLE on {}".format(cycle_node))
        cycle_nodes.append(cycle_node)
    return cycle_nodes

def try_get_dataset_files(model) -> list:
    graphdef = model.graphdef()
    surgeon = graphsurgeon.StaticGraph(graphdef)
    node = graphdef_util.find_node_by_name(surgeon, "Const/_0")
    value = node.attr["value"].tensor.tensor_content
    return value

def generate_localhost_machine_info(model, recommendation, skip_benchmark):
    num_cores = recommendation._analysis.global_state.machine_info.num_cores
    memory = model.memory_total()
    dataset_files = list(model.dataset_file_sizes().keys())
    are_hash_based = map(lambda x: x.isdigit(), dataset_files)
    is_hash_based = all(are_hash_based)
    common_prefix = util.find_common_prefix(dataset_files)
    logging.info("DATASET PREFIX: {}".format(common_prefix))
    if not len(dataset_files) or not common_prefix or is_hash_based:
        try:
            maybe_files = try_get_dataset_files(model)
            maybe_files = maybe_files.decode("utf-8")
            # Remove header (possibly?)
            maybe_files = maybe_files.encode("ascii", errors="ignore").decode()
            # Heuristically split
            if "/mnt" in maybe_files:
                maybe_files = maybe_files.split("/mnt")
                maybe_files = ["/mnt" + x for x in maybe_files]
            elif "/zpool" in maybe_files:
                maybe_files = maybe_files.split("/zpool")
                maybe_files = ["/zpool" + x for x in maybe_files]
            if len(maybe_files) > 1:
                # Just in case, remove header
                # We just want prefix, so throw away first file
                maybe_files = maybe_files[1:]
                new_common_prefix = util.find_common_prefix(maybe_files)
        except Exception as ex:
            maybe_files = None
            new_common_prefix = None
            logging.warning("Exception {}".format(ex))
        if maybe_files:
            if len(maybe_files) > 5:
                logging.info("Maybe files: {}".format(maybe_files[:5]))
            else:
                logging.info("Maybe files: {}".format(maybe_files))
        if new_common_prefix:
            logging.info("Found common prefix: {}".format(new_common_prefix))
            # Recover
            common_prefix = new_common_prefix
        else:
            logging.info("DID NOT FIND ANY DATASET FILES. Skipping benchmark.")
            logging.info("Dataset files: {}".format(dataset_files))
            if is_hash_based:
                logging.info("HASHING IS PROBABLY USED OR DATASET IS TRUNCATED (e.g., CACHING)")
            skip_benchmark = True
    common_prefix_path = pathlib.Path(common_prefix)
    while not common_prefix_path.is_dir():
        common_prefix_path = common_prefix_path.parent
    files = [str(common_prefix_path)]
    files = list(map(lambda x: _filename_to_file_data(x, skip_benchmark),
                     files))
    machine_dict = {
        "HOSTNAME": "Localhost",
        "CORES": num_cores,
        "MEMORY": memory,
        "FILES": files,
    }
    return machine_dict

def span_context_to_networkx(graphdef, span_context):
    """Joins graph with events"""
    G = graphdef_util.graphdef_to_networkx(graphdef)
    def time_delta(span):
        return span.end_time - span.start_time
    name_average = dict()
    for span in span_context.spans:
        if span.name in name_average:
            name_average[span.name].append(time_delta(span))
        else:
            name_average[span.name] = [time_delta(span)]
    remapper_dict = dict()
    for k in name_average:
        v = name_average[k]
        v_pd = pd.Series(v)
        mean = v_pd.mean()
        std = v_pd.std()
        name_average[k] = (mean, std)
        remapper_dict[k] = "{}\nmean:{}\nstd:{}".format(k, mean, std)

    def is_outlier(span):
        if span.name in name_average:
            mean, std = name_average[span.name]
            if time_delta(span) > (2 * std + mean):
                return True
        return False

    name_counter = dict()
    for span in span_context.spans:
        if span.name in name_counter:
            count = name_counter[span.name]
        else:
            count = 0
            name_counter[span.name] = count
        outlier = is_outlier(span)
        if outlier:
            color = "red"
        else:
            color = "blue"
        if count > 10 and not outlier:
            continue
        name = "{}_span_{}".format(span.name, count)
        mean, std = name_average[span.name]
        attrs = {"mean": mean,
                 "std": std}
        G.add_node(name, color=color, **attrs)
        G.add_edge(name, span.name, label=time_delta(span), color=color)
        name_counter[span.name] += 1

    # Apply relabeling to add means
    #G = nx.relabel_nodes(G, remapper_dict, copy=True)
    #isolated_nodes = nx.isolates(G)
    #G.remove_nodes_from(isolated_nodes)

    return G

def optimize_slowest_node(graphdef, slowest_node, dataset_options,
                          up_parallelism):
    """Dynamically dispatch to optimization routine"""
    if slowest_node.op in BATCH_NODE_OPS and not dataset_options["map_and_batch_fusion"]:
        dataset_options["map_and_batch_fusion"] = True
        debug_string = "{}.map_and_batch_fusion=True".format(slowest_node.name)
    else:
        graphdef, debug_string = graphdef_util.increase_node_parallelism(
            graphdef, slowest_node, up_parallelism)
    return graphdef, dataset_options, debug_string

def ranked_nodes_to_df(ranked_nodes):
    ranked_nodes_cols = ["name",
                         "expected_core_max_rate",
                         "expected_parallel_max_rate",
                         "observed_rate",
                         "p_busy",
                         "scheduling_delay",
                         "element_ratio",
                         "processing_time",
                         "CPU_time",
                         "aggregate_processing_time",
                         "aggregate_CPU_time",
                         "parallelism",
                         "aggregate_elements_produced",
                         "aggregate_udf_processing_time",
                         "aggregate_udf_processing_time_clock",
                         "p_udf",
                         "p_udf_clock",
                         "aggregate_avg_number_active_threads",
                         "aggregate_inter_op_parallelism",
                         "aggregate_wait_time",
                         "aggregate_elements_consumed",
                         "avg_wait_time",
                         "wait_time_diff",
                         "p_wait",
                         "p_wait_blame",
                         "p_scheduling",
                         "num_cores_used",
                         "bandwidth_used",
                         "cardinality",
                         "derived_cardinality",
                         "expected_dataset_size",
                         "expected_num_dataset_files",
                         "dataset_record_ratio",
                         "average_bytes_per_element_produced",
                         "average_bytes_per_element_consumed",
                         "byte_ratio",
                         "parent_name",
                         "max_buffer_size",
                         "max_bytes_per_element",
                         "max_memory_used",
                         "misc_buffer_size",
                         "N_customers",
                         ]

    def p_udf_f(x):
        if x.node.state.aggregate_processing_time:
            return (x.node.state.aggregate_udf_processing_time
                    / x.node.state.aggregate_processing_time)
        else:
            return 0.0
    def p_udf_clock_f(x):
        if x.node.state.aggregate_processing_time_clock:
            return (x.node.state.aggregate_udf_processing_time_clock
                    / x.node.state.aggregate_processing_time_clock)
        else:
            return 0.0
    ranked_nodes_data = [(x.name,
                          x.expected_per_core_max_rate,
                          x.expected_parallel_max_rate(),
                          x.observed_rate,
                          x.p_busy,
                          x.node.state.aggregate_scheduling_delay_time,
                          x.element_ratio,
                          x.node.state.processing_time,
                          x.node.state.processing_time_clock,
                          x.node.state.aggregate_processing_time,
                          x.node.state.aggregate_processing_time_clock,
                          x.parallelism,
                          x.node.state.aggregate_elements_produced,
                          x.node.state.aggregate_udf_processing_time,
                          x.node.state.aggregate_udf_processing_time_clock,
                          p_udf_f(x),
                          p_udf_clock_f(x),
                          x.node.state.aggregate_avg_number_active_threads,
                          x.node.state.aggregate_inter_op_parallelism,
                          x.node.state.aggregate_wait_time,
                          x.node.state.aggregate_elements_consumed,
                          x.wait_time,
                          x.wait_time_diff,
                          x.p_wait,
                          x.p_wait_blame,
                          x.p_scheduling,
                          x.num_cores_used,
                          x.bandwidth_used,
                          x.cardinality,
                          x.derived_cardinality,
                          x.expected_dataset_size,
                          x.expected_num_dataset_files,
                          x.dataset_record_ratio,
                          x.average_bytes_per_element_produced,
                          x.average_bytes_per_element_consumed,
                          x.byte_ratio,
                          x.parent.name if x.parent else "",
                          x.node.state.aggregate_max_buffer_size,
                          x.node.state.aggregate_max_bytes_per_element,
                          x.max_memory_used,
                          x.node.state.aggregate_misc_buffer_size,
                          x.N_customers,
                          )
                         for x in ranked_nodes]
    df = pd.DataFrame(ranked_nodes_data, columns=ranked_nodes_cols)
    # TODO(mkuchnik): use normal elements produced
    df["expected_autotune_latency_s"] = (df["processing_time"] /
                                         df["aggregate_elements_produced"] /
                                         df["parallelism"] *
                                         df["element_ratio"] /
                                         1e9)
    return df

def load_pipeline(filename, dataset_options, skip_benchmark: bool,
                  plot_span_ctxs=False, mode=None):
    plumber = tf.data.experimental.analysis.PlumberPerformanceModel(filename)
    model = plumber.model()
    recommendation = model.recommendation()
    assert isinstance(skip_benchmark, bool)
    runtime_data = get_runtime_data(model, skip_benchmark, mode=mode,
                                    consider_parallelizable_nodes=dataset_options["parallelize_batch"])
    graphdef = model.graphdef()
    if plot_span_ctxs:
        for i, span_context in enumerate(recommendation.span_contexts()):
            span_G = span_context_to_networkx(graphdef, span_context)
            nx.drawing.nx_pydot.write_dot(span_G, "span_{}.dot".format(i))
    surgeon = graphsurgeon.StaticGraph(graphdef)
    element_spec = graphdef_util.element_spec_from_graph(surgeon)
    # TODO(mkuchnik): Clean up control-flow
    try:
        ds = graphdef_util.instantiate_pipeline(graphdef, element_spec, dataset_options)
    except graphdef_util.PlaceholderException as ex:
        logging.error(ex)
        ds = None
    return ds, runtime_data

def get_runtime_data(model, skip_benchmark: bool, mode=None,
                     consider_parallelizable_nodes: bool=False):
    CPU_Util = model.CPU_Util()
    CPU_Util_clock = model.CPU_Util(calculation_mode="CPU_clock")
    process_CPU_Util_clock = model.CPU_Util(calculation_mode="process_CPU_clock")
    Disk_Util = model.Disk_Util()
    Disk_Throughput = model.disk_throughput()
    recommendation = model.recommendation()
    Disk_max_rate_100mb = recommendation.disk_upper_bounds(100e6)
    Disk_bytes_per_root_element = recommendation.disk_bytes_per_root_element()
    max_rate = recommendation.upper_bounds(mode=mode)
    max_rate_p_busy = recommendation.upper_bounds(keep_p_busy=True, mode=mode)
    max_rate_convex_native, convex_theta = recommendation.LP_upper_bounds()
    max_rate_convex_native_naive, _ = recommendation.LP_upper_bounds(naive=True)
    max_rate_convex, _ = convex_solver.LP_upper_bounds_inner(
        model, consider_parallelizable_nodes=False)
    if max_rate_convex_native != max_rate_convex:
        logging.warning("Convex implementations have different rates!"
        "Native {} vs. not {}".format(max_rate_convex_native, max_rate_convex))
    max_rate_convex_existing, convex_theta_existing = \
        convex_solver.LP_upper_bounds_inner(model,
                                            use_existing_usage=True)
    cores_remaining = recommendation.remaining_CPU_cores()
    total_dataset_size = model.dataset_working_set_size()
    total_free_memory = model.memory_free()
    measured_memory_util = model.Memory_Util()
    max_memory_usage = model.max_memory_usage()
    max_memory_util = max_memory_usage / model.memory_total()
    # TODO(mkuchnik): Reconcile these measurements with operator rates
    iter_duration = recommendation.iterator_duration()
    iter_wallclock_duration = recommendation.iterator_wallclock_duration()
    iter_variance = recommendation.iterator_variance()
    iter_autotune_output_time = recommendation.iterator_autotune_output_time()
    # TODO(mkuchnik): Cleanup
    assert isinstance(skip_benchmark, bool)
    machine_dict = generate_localhost_machine_info(
        model, recommendation, skip_benchmark=skip_benchmark)
    estimated_disk_bw = machine_dict["FILES"][0]["BANDWIDTH"]
    Disk_max_rate_estimated = recommendation.disk_upper_bounds(
        estimated_disk_bw)
    runtime_data = {
        "CPU_Util": CPU_Util,
        "CPU_Util_clock": CPU_Util_clock,
        "Process_CPU_Util_clock": process_CPU_Util_clock,
        "Disk_Util": Disk_Util,
        "Disk_Throughput": Disk_Throughput,
        "Disk Bytes Per Minibatch": Disk_bytes_per_root_element,
        "Total Dataset Size": total_dataset_size,
        "Total Free Memory": total_free_memory,
        "Measured_System_Memory_Util": measured_memory_util,
        "Max_Memory_Usage": max_memory_usage,
        "Max_Memory_Util": max_memory_util,
        "Estimated_Disk_Max_Rate_100MB": Disk_max_rate_100mb,
        "Estimated_Disk_Bandwidth": estimated_disk_bw,
        "Estimated_Disk_Max_Rate_estimated": Disk_max_rate_estimated,
        "Estimated_Max_Rate": max_rate,
        "Estimated_Max_Rate_p_busy": max_rate_p_busy,
        "Estimated_Max_Rate_Convex": max_rate_convex,
        "Estimated_Max_Rate_Convex_Existing": max_rate_convex_existing,
        "Estimated_Max_Rate_Convex_Native": max_rate_convex_native,
        "Estimated_Max_Rate_Convex_Native_Naive": max_rate_convex_native_naive,
        "Cores_Remaining": cores_remaining,
        "Iterator_Duration": iter_duration,
        "Iterator_Wallclock_Duration": iter_wallclock_duration,
        "Iterator_Variance": iter_variance,
        "Iterator_Autotune_Output_Time": iter_autotune_output_time,
        "Convex_Theta": convex_theta,
        "Convex_Theta_Existing": convex_theta_existing,
    }
    return runtime_data

def step_pipeline(filename, dataset_options, step_options, strategy=None,
                  mode=None, plot_span_ctxs=False, avoid_unknowns=True,
                  cycle_ignore_list=None, up_parallelism=None):
    if not up_parallelism:
        up_parallelism = 1
    plumber = tf.data.experimental.analysis.PlumberPerformanceModel(filename)
    model = plumber.model()
    runtime_data = get_runtime_data(model, step_options["skip_benchmark"],
                                    mode=mode,
                                    consider_parallelizable_nodes=dataset_options["parallelize_batch"])
    recommendation = model.recommendation()
    num_cores = recommendation._analysis.global_state.machine_info.num_cores
    logging.info("num_cores: {}".format(num_cores))
    ranked_nodes = \
    recommendation.ranked_list_bottleneck_nodes_analysis(mode=mode)
    graphdef = model.graphdef()
    surgeon = graphsurgeon.StaticGraph(graphdef)
    def is_reachable(x) -> bool:
        # TODO(mkuchnik): Since graphsurgeon cannot traverse functions, we must
        # avoid any nodes in functions
        ret = graphdef_util.find_node_by_name(surgeon, x.name,
                                              raise_on_fail=False)
        return ret is not None
    if strategy is None:
        if not avoid_unknowns and not cycle_ignore_list:
            slowest_node = recommendation.bottleneck_node(mode=mode)
        else:
            valid_ranked_nodes = [x for x in ranked_nodes if
                                  x.is_parallel_node()
                                  or (x.op in BATCH_NODE_OPS and
                                      dataset_options["parallelize_batch"] and
                                      is_reachable(x))]
            if cycle_ignore_list:
                valid_ranked_nodes = [x for x in valid_ranked_nodes if x not in
                                      cycle_ignore_list]
            if not len(valid_ranked_nodes):
                def summarize_node(n):
                    return n.to_summary_dict()
                raise ValueError(
                        "Did not fid valid nodes in "
                        "ranked_nodes:\n{}".format(
                            pprint.pformat(list(
                                map(summarize_node, ranked_nodes))))
                            )
            slowest_node = valid_ranked_nodes[0]
    elif strategy == "random":
        # Emulate randomly permuting choice
        slowest_node = random.choice(ranked_nodes)
    elif strategy == "random_valid":
        # Emulate randomly permuting choice
        valid_ranked_nodes = [x for x in ranked_nodes if x.is_parallel_node() or
                              (not dataset_options["map_and_batch_fusion"]
                               and x.op in BATCH_NODE_OPS) or
                              (dataset_options["parallelize_batch"] and
                               x.op in BATCH_NODE_OPS and is_reachable(x))
                              ]
        slowest_node = random.choice(valid_ranked_nodes)
    elif strategy == "random_valid_deviation":
        # Emulate randomly permuting choice with recommendation removed
        _slowest_node = recommendation.bottleneck_node(mode=mode)
        valid_ranked_nodes = [x for x in ranked_nodes if
                              (x.is_parallel_node() or
                                (not dataset_options["map_and_batch_fusion"] and
                                 x.op in BATCH_NODE_OPS) or
                                (dataset_options["parallelize_batch"] and
                                 x.op in BATCH_NODE_OPS and is_reachable(x))) and
                               x.name != _slowest_node.name]
        logging.info("Valid nodes {} without {}: ".format(
            [x.name for x in valid_ranked_nodes],
            _slowest_node.name))
        slowest_node = random.choice(valid_ranked_nodes)
        logging.info("Deviation: {} -> {}".format(_slowest_node.name,
                                           slowest_node.name))
    else:
        raise RuntimeError("Unknown strategy: {}".format(strategy))
    df = ranked_nodes_to_df(ranked_nodes)
    graphdef = model.graphdef()
    if plot_span_ctxs:
        for i, span_context in enumerate(recommendation.span_contexts()):
            span_G = span_context_to_networkx(graphdef, span_context)
            nx.drawing.nx_pydot.write_dot(span_G, "span_{}.dot".format(i))
    # TODO(mkuchnik): Refactor out plotting code
    G = graphdef_util.graphdef_to_networkx(graphdef)
    def extract_important_keys(node):
        n_summary_dict = node.to_summary_dict()
        important_keys = set(["p_wait_blame", "num_cores_used",
                              "expected_parallel_max_rate",
                              "expected_core_max_rate"])
        rets = {k: v for k, v in n_summary_dict.items() if k in
                important_keys}
        rets["expected_parallel_max_rate"] = node.expected_parallel_max_rate()
        return rets

    values_dict = {n.name: extract_important_keys(n)
                   for n in ranked_nodes}
    def values_to_colors(arr_values, custom_colormap=cm.jet):
        # Scale to RGBA
        arr_colors = (cm.ScalarMappable(cmap=custom_colormap)
                      .to_rgba(arr_values, bytes=True))
        return arr_colors
    nx.set_node_attributes(G, values_dict)
    arr_values_dict = {n: d for n, d in
                       G.nodes(data="p_wait_blame", default=0)}
    arr_values = np.array(list(arr_values_dict.values()))
    arr_colors = values_to_colors(arr_values)
    def scale_rgba(x):
        x[3] = 90
        return x
    arr_hex_colors = list(map(lambda x:
                              '#{:02x}{:02x}{:02x}{:02x}'.format(
                                  *scale_rgba(x)),
                              arr_colors))
    hex_arr_values_dict = {n: {"fillcolor": c,
                               "style": "filled"} for n, c
                           in zip(arr_values_dict.keys(),
                                  arr_hex_colors)}
    nx.set_node_attributes(G, hex_arr_values_dict)
    nx.drawing.nx_pydot.write_dot(G, "networkx.dot")
    topo_sort = nx.topological_sort(G)
    topo_sort_dataset = filter(graphdef_util.is_dataset_node, topo_sort)
    remapper = graphdef_util.remap_dataset_names(topo_sort_dataset)
    G_remapped = nx.relabel_nodes(G, remapper)
    nx.drawing.nx_pydot.write_dot(G_remapped, "networkx_remapped.dot")
    random_labels = dict(G_remapped.nodes(data="has_random_seed", default="N/A"))
    for k in random_labels:
        random_labels[k] = "{}: {}".format(k, random_labels[k])
    draw_scale = 5 # Anything bigger than 1 is amplification
    pos = nx.spring_layout(G_remapped, k=draw_scale/math.sqrt(G_remapped.order()))
    nx.draw(G_remapped, pos=pos, labels=random_labels)
    plt.tight_layout()
    plt.savefig("networkx_functions.pdf")
    plt.clf()
    # END plotting code
    def remap_fn(x):
        try:
            return remapper[x]
        except KeyError as ex:
            logging.warning("Did not find {} in remap_fn".format(x))
            return x
    # Logging for per-node dataframe
    df["canonical_name"] = df["name"].map(remap_fn)
    maybe_autotune_latency = 1./df.set_index("canonical_name")["expected_parallel_max_rate"]
    logging.info("Maybe autotune latency (ns):\n{}".format(maybe_autotune_latency *
                                                      1e9))
    autotune_latency = df.set_index("canonical_name")["expected_autotune_latency_s"]
    expected_autotune_time = np.sum(autotune_latency)
    if expected_autotune_time:
        expected_autotune_rate = 1./expected_autotune_time
    else:
        expected_autotune_rate = -1.
    logging.info("Autotune latency (ns):\n{}".format(autotune_latency * 1e9))
    iter_autotune_output_time = recommendation.iterator_autotune_output_time()
    if not iter_autotune_output_time:
        iter_autotune_output_rate = -1.
    else:
        iter_autotune_output_rate = 1./iter_autotune_output_time
    logging.info("Expected autotune time: {}, Measured:"
                 " {}".format(expected_autotune_time,
                              iter_autotune_output_time))
    logging.info("Expected autotune rate: {}, Measured:"
                 " {}".format(expected_autotune_rate,
                              iter_autotune_output_rate))
    # NOTE(mkuchnik): Join with LP recommendation DF
    thetas_dict = extract_theta_from_runtime_data(runtime_data)
    _thetas_dict = round_thetas_dict(thetas_dict["Convex_Theta"])
    _thetas_dict_df = pd.Series(
            data=_thetas_dict,
            name="LP_Thetas_Recommendation").to_frame()
    logging.info(_thetas_dict_df)
    df = df.join(_thetas_dict_df, on="name")
    max_cache_size = FRACTION_CACHEABLE_MEMORY * model.memory_total()
    def is_cacheable_size_recommendation(size):
        # Finds nodes that can be cached
        return (not np.isnan(size)
                and size >= 0.0
                and size <= max_cache_size)
    df["is_cacheable"] = df["expected_dataset_size"].apply(is_cacheable_size_recommendation)

    def find_highest_cacheable(graphdef, candidates, max_memory):
        # Finds the node best fit for a cache
        # TODO(mkuchnik): Consolidate with pipeline_optimizer
        surgeon = graphsurgeon.DynamicGraph(graphdef)
        G = graphdef_util.graphdef_to_networkx(graphdef, keep_const=False)
        topo_sort = nx.topological_sort(G)
        current_candidate = None
        # From source to root names
        for node_name in topo_sort:
            if node_name in candidates:
                cache_size = candidates[node_name]
                is_cacheable = cache_size < max_memory
                logging.info("Cache candidate: {} ({}GB / {}GB)".format(
                    node_name, cache_size / 1e9, max_memory / 1e9))
                if is_cacheable:
                    current_candidate = node_name
        return current_candidate

    # Remove nodes with random UDFs
    random_nodes = pipeline_optimizer.nodes_with_random_udf(graphdef)
    df["is_random_UDF"] = df["name"].apply(lambda name: name in random_nodes)

    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None):
        logging.info("Ranked_nodes:\n{}".format(df))

    current_parallelism = graphdef_util.get_node_parallelism(slowest_node)
    logging.info("Current parallelism for node {} is {}".format(slowest_node.name,
                                                         current_parallelism))
    cache_candidates = (df
            .query("is_cacheable == True")
            [["name", "expected_dataset_size"]]).set_index("name")["expected_dataset_size"]
    logging.info("Random UDF nodes:\n{}".format(random_nodes))
    # name -> cache_size
    cache_candidates = cache_candidates.to_dict()
    cache_candidates = {k: v for k, v in cache_candidates.items() if k not in random_nodes}
    logging.info("Inspecting cache candidates:\n{}".format(cache_candidates))
    # TODO(mkuchnik): Shuffle is ambiguously handled
    cacheable_recommendation = find_highest_cacheable(
            graphdef, cache_candidates, max_cache_size)
    if cacheable_recommendation is None:
        logging.info("Failed to find cacheable node!")
    else:
        logging.info("Cacheable recommendation: {}".format(
            cacheable_recommendation))

    with open("graphdef.txt", "w") as f:
        f.write(str(graphdef))
    graphdef, dataset_options, debug_string = \
        optimize_slowest_node(graphdef, slowest_node, dataset_options,
                              up_parallelism)
    surgeon = graphsurgeon.DynamicGraph(graphdef)
    graphdef = surgeon.as_graph_def()
    element_spec = graphdef_util.element_spec_from_graph(surgeon)
    with open("graphdef2.txt", "w") as f:
        f.write(str(graphdef))
    ds = graphdef_util.instantiate_pipeline(graphdef, element_spec, dataset_options)
    runtime_data.update(dataset_options)
    return ds, debug_string, dataset_options, df, runtime_data

def benchmark_all_nodes_dataset_from_plumber(filename, dataset_options: dict,
                                bench_options: dict,
                                take_amount: int = 500):
    plumber = \
        tf.data.experimental.analysis.PlumberPerformanceModel(filename)
    model = plumber.model()
    graphdef = model.graphdef()
    surgeon = graphsurgeon.DynamicGraph(graphdef)
    all_benchmark_summary = benchmark_util.benchmark_all_nodes_dataset(
        surgeon, dataset_options, bench_options)
    return all_benchmark_summary

def extract_theta_from_runtime_data(runtime_data):
    keys = ["Convex_Theta", "Convex_Theta_Existing"]
    thetas = dict()
    for k in keys:
        try:
            v = runtime_data[k]
        except KeyError as ex:
            logging.error(ex)
            raise KeyError(
                "Failed to find key '{}' of keys '{}' in dict keys '{}'".format(
                keys, k, runtime_data.keys()))
        thetas[k] = v
    return thetas

def round_thetas_dict(thetas_dict):
    """Round to integer parallelism"""
    _thetas_dict = copy.deepcopy(thetas_dict)
    for k in thetas_dict:
        _thetas_dict[k] = max(int(round(_thetas_dict[k])), 1)
    return _thetas_dict

def run_rewriter_runner(num_steps, num_deviations, map_and_batch_fusion,
                        threadpool_size: int,
                        time_limit_s, strategy, rebench_baseline, skip_baseline,
                        skip_LP_baseline: bool,
                        skip_system_benchmark: bool,
                        sweep_nodes: bool,
                        mode,
                        up_parallelism: int,
                        parallelize_batch: bool,
                        skip_rewrites: bool):
    assert num_deviations >= 1, "num_deviations has to be at least 1"
    dataset_options = {
        "stats_filename": "stats_new.pb",
        "map_and_batch_fusion": map_and_batch_fusion,
        "parallelize_batch": parallelize_batch,
        "threadpool_size": threadpool_size,
        "take_amount": None,
    }
    bench_options = {
        "time_limit_s": time_limit_s,
    }
    step_options = {
        "skip_benchmark": skip_system_benchmark,
    }
    if strategy not in STRATEGIES:
        raise ValueError("time_limit_s={} not in {}".format(
            strategy, STRATEGIES))
    ds, runtime_data = load_pipeline(DEFAULT_STATS_FILENAME, dataset_options,
                                     skip_system_benchmark,
                                     mode=mode)
    thetas_dict = extract_theta_from_runtime_data(runtime_data)
    # Round to integer parallelism
    _thetas_dict = round_thetas_dict(thetas_dict["Convex_Theta"])
    logging.info("Runtime_data:\n{}".format(pd.Series(data=runtime_data)))
    logging.info("LP Thetas:\n{}".format(pd.Series(data=_thetas_dict)))
    if not skip_baseline:
        logging.info("bench_options: {}".format(bench_options))
        benchmark_summary = gen_util.benchmark_dataset(ds, **bench_options)
    else:
        benchmark_summary = {"global_minibatch_rate": None}
    if not skip_LP_baseline:
        plumber = tf.data.experimental.analysis.PlumberPerformanceModel(
            DEFAULT_STATS_FILENAME)
        model = plumber.model()
        graphdef = model.graphdef()
        thetas_dict = thetas_dict["Convex_Theta"]
        # Round to integer parallelism
        thetas_dict = round_thetas_dict(thetas_dict)
        LP_graphdef = graphdef_util.apply_thetas_recommendation(
            graphdef, thetas_dict)
        surgeon = graphsurgeon.StaticGraph(LP_graphdef)
        element_spec = graphdef_util.element_spec_from_graph(surgeon)
        LP_ds = graphdef_util.instantiate_pipeline(LP_graphdef,
                                     element_spec,
                                     dataset_options)
        LP_benchmark_summary = gen_util.benchmark_dataset(LP_ds,
                                                          **bench_options)
    else:
        LP_benchmark_summary = {"global_minibatch_rate": None}
    rate = benchmark_summary["global_minibatch_rate"]
    rates = [rate]
    changes = [None]
    # TODO(mkuchnik): Fix flow control
    if ds is None:
        logging.error("Dataset failed to instantiate. Will probably have to exit.")
    del ds
    benchmark_summary["step"] = 0
    benchmark_summary["change"] = None
    benchmark_summary["deviation"] = 0
    benchmark_summary.update(runtime_data)
    graphdef_util.clear_graph()
    # Start with original stats
    if not rebench_baseline:
        shutil.copyfile(DEFAULT_STATS_FILENAME, "stats_new.pb")
    shutil.copyfile("stats_new.pb", "stats_new_0_0.pb")
    if sweep_nodes:
        _bench_options = copy.deepcopy(bench_options)
        _bench_options["profile_interval"] = None
        _dataset_options = copy.deepcopy(dataset_options)
        _dataset_options["stats_filename"] = None
        all_benchmark_summary = benchmark_all_nodes_dataset_from_plumber(
            "stats_new_0_0.pb", _dataset_options, _bench_options)
        all_benchmark_summary_df = [transform_benchmark_summary_to_df(s)
                                    for s in
                                    all_benchmark_summary]
        all_benchmark_summary_df = pd.concat(all_benchmark_summary_df)
        all_benchmark_summary_df.reset_index(inplace=True)
        all_benchmark_summary_df.to_csv("sweep_all_node_benchmark_stats.csv")

    if skip_rewrites:
        curr_dataset_options = copy.deepcopy(dataset_options)
        i = 1
        deviation = 0
        curr_dataset_options["stats_filename"] = \
            "stats_new_{}_{}.pb".format(i, deviation)
        curr_strategy = strategy if not deviation else \
                        "random_valid_deviation"
        cycle_ignore_list = detect_suggestion_cycles(rates, changes)
        # TODO(mkuchnik): Fix flow control
        try:
            ds, changed_node, curr_dataset_options, df, runtime_data = \
                step_pipeline(filename="stats_new_{}_0.pb".format(i - 1),
                              dataset_options=curr_dataset_options,
                              step_options=step_options,
                              strategy=curr_strategy,
                              mode=mode,
                              cycle_ignore_list=cycle_ignore_list,
                              up_parallelism=up_parallelism)
        except graphdef_util.PlaceholderException as ex:
            logging.error(ex)
            logging.error("Failed to instantiate due to placeholders. "
                          "Are you using DT_RESOURCE variables?")
        return

    thetas_df = pd.DataFrame(data=thetas_dict, index=[0])
    thetas_df["step"] = 0
    thetas_df["deviation"] = 0
    thetas_dfs = [thetas_df]
    dfs = []
    benchmark_dfs = [transform_benchmark_summary_to_df(benchmark_summary)]
    del benchmark_summary
    for i in range(1, num_steps + 1):
        # NOTE(mkuchnik): Take is already applied
        _dataset_options = copy.deepcopy(dataset_options)
        _new_dataset_options = None
        for deviation in range(num_deviations):
            curr_dataset_options = copy.deepcopy(_dataset_options)
            curr_dataset_options["stats_filename"] = \
                "stats_new_{}_{}.pb".format(i, deviation)
            curr_strategy = strategy if not deviation else \
                            "random_valid_deviation"
            cycle_ignore_list = detect_suggestion_cycles(rates, changes)
            ds, changed_node, curr_dataset_options, df, runtime_data = \
                step_pipeline(filename="stats_new_{}_0.pb".format(i - 1),
                              dataset_options=curr_dataset_options,
                              step_options=step_options,
                              strategy=curr_strategy,
                              mode=mode,
                              cycle_ignore_list=cycle_ignore_list,
                              up_parallelism=up_parallelism)
            thetas_dict = extract_theta_from_runtime_data(runtime_data)
            thetas_df = pd.DataFrame(data=thetas_dict, index=[0])
            thetas_df["step"] = i
            thetas_df["deviation"] = deviation
            thetas_dfs.append(thetas_df)
            logging.info("Runtime_data\n{}".format(pd.Series(data=runtime_data)))
            df["step"] = i
            df["deviation"] = deviation
            graphdef_util.clear_graph()
            try:
                logging.info("bench_options: {}".format(bench_options))
                benchmark_summary = gen_util.benchmark_dataset(ds,
                                                               **bench_options)
            except TypeError as ex:
                logging.warning("Exception: {}".format(ex))
                break
            rate = benchmark_summary["global_minibatch_rate"]
            del ds
            dfs.append(df)
            if not deviation:
                _new_dataset_options = curr_dataset_options
                rates.append(rate)
                changes.append(changed_node)
            global_df = pd.concat(dfs)
            global_df.to_csv("node_stats.csv")
            global_thetas_df = pd.concat(thetas_dfs).reset_index()
            global_thetas_df.to_csv("thetas.csv")
            benchmark_summary["step"] = i
            benchmark_summary["change"] = changed_node
            benchmark_summary["deviation"] = deviation
            benchmark_summary.update(runtime_data)
            benchmark_df = transform_benchmark_summary_to_df(benchmark_summary)
            benchmark_dfs.append(benchmark_df)
            global_benchmark_df = \
                pd.concat(benchmark_dfs).reset_index(drop=True)
            global_benchmark_df.to_csv("benchmark_stats.csv")
            logging.info("Rates:\n{}".format(pprint.pformat(rates)))
            logging.info("Changes:\n{}".format(pprint.pformat(changes)))
        dataset_options = _new_dataset_options
    logging.info("Rates:\n{}".format(pprint.pformat(rates)))
    logging.info("Changes:\n{}".format(pprint.pformat(changes)))

def default_main(parser=None):
    if isinstance(parser, argparse.ArgumentParser):
        FLAGS = parser.parse_args()
    else:
        FLAGS = flags.FLAGS
    num_steps = FLAGS.num_steps
    num_deviations = FLAGS.num_deviations # set to 1 for normal run
    map_and_batch_fusion = FLAGS.map_and_batch_fusion
    threadpool_size = FLAGS.dataset_threadpool_size
    time_limit_s = FLAGS.time_limit_s
    strategy = FLAGS.strategy
    rebench_baseline = FLAGS.rebench_baseline
    skip_baseline = FLAGS.skip_baseline
    skip_LP_baseline = FLAGS.skip_LP_baseline
    sweep_nodes = FLAGS.sweep_nodes
    mode = None
    skip_system_benchmark = FLAGS.skip_system_benchmark
    up_parallelism = FLAGS.up_parallelism
    parallelize_batch = FLAGS.parallelize_batch
    skip_rewrites = FLAGS.skip_rewrites
    run_rewriter_runner(num_steps, num_deviations, map_and_batch_fusion,
                        threadpool_size,
                        time_limit_s, strategy, rebench_baseline, skip_baseline,
                        skip_LP_baseline,
                        skip_system_benchmark,
                        sweep_nodes, mode, up_parallelism=up_parallelism,
                        parallelize_batch=parallelize_batch,
                        skip_rewrites=skip_rewrites)

def main(_):
    num_steps = 10
    num_deviations = 3
    map_and_batch_fusion = False
    threadpool_size = os.cpu_count()
    time_limit_s = 22
    strategy = None
    rebench_baseline = True
    skip_baseline = False
    skip_LP_baseline = False
    sweep_nodes = False
    mode = None
    skip_system_benchmark = False
    up_parallelism = 1
    parallelize_batch = False
    skip_rewrites = False
    run_rewriter_runner(num_steps, num_deviations, map_and_batch_fusion,
                        threadpool_size,
                        time_limit_s, strategy, rebench_baseline, skip_baseline,
                        skip_LP_baseline, skip_system_benchmark,
                        sweep_nodes, mode, up_parallelism,
                        parallelize_batch=parallelize_batch,
                        skip_rewrites=skip_rewrites)

if __name__ == '__main__':
  app.run(main)
