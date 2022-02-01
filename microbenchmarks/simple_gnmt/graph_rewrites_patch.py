from absl import app

import shutil
import sys
import pprint

import tensorflow as tf
import tensorflow.compat.v1 as tf1
import graphsurgeon

import argparse
import gen_util
import pandas as pd

try:
    import dataloader
except ImportError:
    try:
        import resnet_flags
    except ImportError:
        import dataset_flags

import random
import copy
import networkx as nx
import convex_solver
import numpy as np
import benchmark_mlperf

STRATEGIES = [None, "random", "random_valid"]
DEFAULT_MODE = None
BATCH_NODE_OPS = set(["BatchDataset", "BatchDatasetV2"])
FLAGS = None
_DATASET = None

def add_more_flags(parser):
    parser.add_argument('--rebench_baseline',
                      default=False,
                        type=bool,
                      help=('Run benchmarking on pipeline again for baseline.'))
    parser.add_argument('--sweep_nodes',
                      default=False,
                        type=bool,
                      help=('Run benchmarking on individual nodes.'))
    parser.add_argument('--skip_baseline',
                      default=False,
                        type=bool,
                      help=('Don\'t evalute first run.'))
    parser.add_argument('--num_deviations',
                         default=1,
                        type=int,
                         help=('The number of deviations (non-recommendations) to '
                               'run per step. Set to 1 for none.'))
    parser.add_argument('--num_steps',
                         default=500,
                        type=int,
                         help=('The number of steps (max) to take.'))
    parser.add_argument('--strategy',
                        default=None,
                        type=str,
                        help=('The strategy to run. One of {}'.format(STRATEGIES)))
    return parser

def is_dataset_node(node_name):
    return "Dataset" in node_name or "dataset" in node_name

def get_dataset_node_names(graphdef):
    surgeon = graphsurgeon.StaticGraph(graphdef)
    dataset_nodes = [n.name for n in surgeon if "Dataset" in n.op]
    return dataset_nodes

def get_node_parallelism(node):
    return node.parallelism

def parallelism_parameter_index(surgeon_node):
    if surgeon_node.op == "MapAndBatchDataset":
        return 2
    elif surgeon_node.op == "ParallelMapDatasetV2":
        return -1
    elif surgeon_node.op == "ParallelBatchDataset":
        return -2
    elif surgeon_node.op == "ParallelInterleaveDatasetV4":
        return -1
    else:
        raise RuntimeError("Don't know how to handle"
                           " {}".format(surgeon_node.name))

def cycle_length_parameter_index(surgeon_node):
    if surgeon_node.op == "ParallelInterleaveDatasetV4":
        return -5
    else:
        raise RuntimeError("Don't know how to handle"
                           " {}".format(surgeon_node.name))

def parallelism_parameter_name(surgeon_node):
    idx = parallelism_parameter_index(surgeon_node)
    return surgeon_node.input[idx]

def find_datasets_in_f(graph_def, f_name, datasets=None):
  """Find datasets in a function."""
  # TODO(mkuchnik): Use
  def is_tfdata_node(node):
      return "Dataset" in node.op or "dataset" in node.op

  def find_fs_of_f(f):
    """Find nested function nodes e.g., f1 calls f2."""
    # TODO(mkuchnik): Add support for groupbywindowdataset
    fs_nodes = []
    for node in f.node_def:
      if ("f" in node.attr or "key_func" in node.attr or "reduce_func" in
          node.attr or "window_size_func" in node.attr):
        fs_nodes.append(node)
    return fs_nodes

  datasets = [] if datasets is None else datasets
  for f in graph_def.library.function:
    if f.signature.name == f_name:
      for node in f.node_def:
        if is_tfdata_node(node):
          datasets.append(node)
      child_f_nodes = find_fs_of_f(f)
      for child_node in child_f_nodes:
        child_f_name = child_node.attr["f"].func.name
        find_datasets_in_f(graph_def, child_f_name, datasets)
  return datasets

def find_function_by_name(graph_def, f_name):
    for f in graph_def.library.function:
        if f.signature.name == f_name:
            return f
    raise RuntimeError(
        "Expected to find 1 node for {}, but found {}".format(
        f_name, 0))

def find_functions_of_node(graph_def, surgeon_node):
    fs = []
    if surgeon_node.op == "GroupByWindowDataset":
        for key_str in ["key_func", "reduce_func", "window_size_func"]:
            key_f = surgeon_node.attr[key_str].func
            # TODO(mkuchnik): For some reason, encoding adds "\nF" or "\nA" etc.
            key_f_str = key_f.SerializeToString().decode()
            key_f_str = key_f_str.strip("\n")
            key_f_str = key_f_str[1:]
            f = find_function_by_name(graph_def, key_f_str)
            fs.append(f)
    elif (surgeon_node.op == "ParallelInterleaveDatasetV4" or
          surgeon_node.op == "MapDataset" or
          surgeon_node.op == "ParallelMapDatasetV2"):
        for key_str in ["f"]:
            key_f = surgeon_node.attr[key_str].func
            key_f_str = key_f.SerializeToString().decode()
            key_f_str = key_f_str.strip("\n")
            # TODO(mkuchnik): For some reason, encoding adds "\nF" or "\nA" etc.
            key_f_str = key_f_str[1:]
            f = find_function_by_name(graph_def, key_f_str)
            fs.append(f)
    return fs

def graphdef_to_networkx(graphdef, keep_const=False):
    # NOTE(mkuchnik): Can also use from_pydot
    G = nx.DiGraph()
    surgeon = graphsurgeon.StaticGraph(graphdef)
    retval = surgeon.find_nodes_by_op("_Retval")
    assert len(retval) == 1
    retval = retval[0]

    def descend(node):
        for i_name in node.input:
            i = find_node_by_name(surgeon, i_name)
            if i.op != "Const" or (i.op == "Const" and keep_const):
                G.add_node(i.name)
                G.add_edge(i.name, node.name)
            descend(i)
        fs = find_functions_of_node(graphdef, node)
        for f in fs:
            G.add_node(f.signature.name)
            G.add_edge(f.signature.name, node.name)
            f_datasets = find_datasets_in_f(graphdef, f.signature.name)
            for ff in f_datasets:
                G.add_node(ff.name)
                G.add_edge(ff.name, f.signature.name)


    G.add_node(retval.name)
    descend(retval)
    return G

def find_retvals(graphdef):
    surgeon = graphsurgeon.StaticGraph(graphdef)
    nodes = surgeon.find_nodes_by_op("_Retval")
    return nodes

def find_node_by_name(surgeon, node_name, raise_on_fail=True):
    surgeon_node = surgeon.find_nodes_by_name(node_name)
    if not surgeon_node:
        surgeon_node = [n for n in surgeon if n.name == node_name]
    if len(surgeon_node) != 1:
        if raise_on_fail:
            raise RuntimeError(
                "Expected to find 1 node for {}, but found {}".format(
                node_name, len(surgeon_node)))
        else:
            return None
    surgeon_node = surgeon_node[0]
    return surgeon_node

def fork_node(surgeon, surgeon_node):
    new_node = copy.deepcopy(surgeon_node)
    prefix = "Added_{}/".format(new_node.op)
    new_name =  generate_new_name(surgeon, prefix)
    new_node.name = new_name
    surgeon.append(new_node)
    new_node = find_node_by_name(surgeon, new_node.name)
    return new_node

def increase_node_parallelism(graphdef, node, up_parallelism=1):
    surgeon = graphsurgeon.DynamicGraph(graphdef)
    surgeon_node = find_node_by_name(surgeon, node.name)
    parallelism = get_node_parallelism(node) + up_parallelism
    try:
        parallelism_node = parallelism_parameter_name(surgeon_node)
    except RuntimeError as ex:
        print(ex)
        print("IGNORING")
        graph_def = surgeon.as_graph_def()
        return graph_def, None
    parallelism_surgeon_node = [k for k in surgeon if
                                k.name == parallelism_node]
    assert len(parallelism_surgeon_node) == 1, \
        "Expected to find 1 node for {}, but found {}".format(
            parallelism_node, len(parallelism_surgeon_node))
    parallelism_surgeon_node = parallelism_surgeon_node[0]
    new_parallelism_surgeon_node = fork_node(surgeon, parallelism_surgeon_node)
    i = parallelism_parameter_index(surgeon_node)
    node_input = surgeon_node.input[i]
    assert(node_input == parallelism_node)

    surgeon_node.input[i] = new_parallelism_surgeon_node.name
    parallelism_tensor = new_parallelism_surgeon_node.attr["value"].tensor
    if surgeon_node.op == "ParallelInterleaveDatasetV4":
        # Adjust cycle length to match parallelism
        i = cycle_length_parameter_index(surgeon_node)
        node_input = surgeon_node.input[i]
        cycle_surgeon_node = [k for k in surgeon if
                              k.name == node_input]
        assert len(cycle_surgeon_node) == 1
        cycle_surgeon_node = cycle_surgeon_node[0]
        new_cycle_surgeon_node = fork_node(surgeon,
                                           cycle_surgeon_node)
        surgeon_node.input[i] = new_cycle_surgeon_node.name
        cycle_tensor = new_cycle_surgeon_node.attr["value"].tensor
    else:
        cycle_tensor = None
    parallelism_tensor.int64_val[:] = [parallelism]
    if cycle_tensor:
        cycle_tensor.int64_val[:] = [parallelism]

    debug_string = "{}.parallelism={}".format(node.name,
                                              parallelism)
    graph_def = surgeon.as_graph_def()
    return graph_def, debug_string

def find_placeholders(graphdef):
    surgeon = graphsurgeon.StaticGraph(graphdef)
    return surgeon.find_nodes_by_op("Placeholder")

def apply_pipeline_options(dataset, map_and_batch_fusion, stats_filename):
    options = tf.data.Options()
    options.experimental_deterministic = False
    options.experimental_threading.max_intra_op_parallelism = 1
    options.experimental_threading.private_threadpool_size = \
        FLAGS.dataset_threadpool_size
    options.experimental_optimization.map_and_batch_fusion = \
        map_and_batch_fusion
    if stats_filename:
        options.experimental_optimization.autotune_stats_filename = \
            stats_filename
    dataset = dataset.with_options(options)
    return dataset

def remove_extra_datasets(graphdef):
    """Removes nodes such as `ModelDataset` which are appended to the dataset"""
    NODES_TO_REMOVE = ["ModelDataset",
                       "MaxIntraOpParallelismDataset",
                       "PrivateThreadPoolDataset",
                       ]
    surgeon = graphsurgeon.DynamicGraph(graphdef)
    surgeon = remove_op_datasets(surgeon, NODES_TO_REMOVE)
    return surgeon.as_graph_def()

def remove_op_datasets(surgeon, nodes_to_remove):
    for n in nodes_to_remove:
        nodes = surgeon.find_nodes_by_op(n)
        surgeon.forward_inputs(nodes)
        nodes = surgeon.find_nodes_by_op(n)
        assert not nodes, "{} still found".format(n)
    return surgeon

def remove_name_datasets(surgeon, nodes_to_remove, forward=False):
    for n in nodes_to_remove:
        nodes = find_node_by_name(surgeon, n)
        if forward:
            surgeon.forward_inputs(nodes)
        else:
            surgeon.remove(nodes)
        nodes = find_node_by_name(surgeon, n, raise_on_fail=False)
        assert not nodes, "{} still found".format(n)
    return surgeon


def patch_retval(retval):
    """Retval nodes have only one input, but deleting nodes may have caused them
    to have more than one."""
    patched_inputs = [i for i in retval.input if "Dataset" in i]
    retval.input[:] = patched_inputs

def remap_dataset_names(topo_dataset_names):
    remapper = dict()
    remapper_counter = dict()
    for n in topo_dataset_names:
        if "/" in n:
            base = n.split("/")[0]
        else:
            base = n
        if base in remapper_counter:
            remapper_counter[base] += 1
        else:
            remapper_counter[base] = 0
        new_name = "{}_{}".format(base, remapper_counter[base])
        remapper[n] = new_name
    return remapper

def patch_resource_datasets(graphdef):
    """ Resources are not serializable. We thus must remove vocab file and add
    it back in."""
    surgeon = graphsurgeon.DynamicGraph(graphdef)
    def find_lookup_nodes(surgeon):
        const_nodes = [n for n in surgeon if "Const"
                         in n.op
                         ]
        resource_nodes = [n for n in const_nodes if "DT_RESOURCE" in
                          str(n.attr["dtype"])]
        vocab_lookup_nodes = [n for n in resource_nodes if "lookup" in str(n)]
        return vocab_lookup_nodes
    for i in _DATASET:
        break
    _orig_graphdef = _DATASET._as_serialized_graph()
    _orig_graphdef = bytes(_orig_graphdef.numpy())
    orig_graphdef = tf1.GraphDef()
    orig_graphdef.ParseFromString(_orig_graphdef)
    orig_surgeon = graphsurgeon.StaticGraph(orig_graphdef)
    vocab_lookup_nodes = find_lookup_nodes(surgeon)
    orig_vocab_lookup_nodes = find_lookup_nodes(orig_surgeon)
    global FLAGS
    if not len(vocab_lookup_nodes):
        return graphdef
    elif len(vocab_lookup_nodes) > 1:
        raise RuntimeError("Too many resources")
    else:
        surgeon_node = vocab_lookup_nodes[0]
        new_node = copy.deepcopy(orig_vocab_lookup_nodes[0])
        print("Replacing {} with {}".format(surgeon_node.name, new_node.name))
        new_name = surgeon_node.name
        new_node.name = new_name
        surgeon = remove_name_datasets(surgeon, [surgeon_node.name])
        surgeon.append(new_node)
        graphdef = surgeon.as_graph_def()
    return graphdef

def instantiate_pipeline(graphdef, element_spec, dataset_options):
    #global _DATASET
    #dataset = reinstantiate_pipeline(_DATASET, dataset_options)
    #graph_def = dataset._as_serialized_graph()
    #graph_def = bytes(graph_def.numpy())
    #graphdef = tf1.GraphDef()
    #graphdef.ParseFromString(graph_def)
    #element_spec = dataset.element_spec
    # TODO(mkuchnik): Stats filename is stripped
    placeholders = find_placeholders(graphdef)
    assert not placeholders, \
        "No placeholders can exist in graph but found {}".format(placeholders)
    dataset_nodes = get_dataset_node_names(graphdef)
    print("Found dataset nodes: {}".format(dataset_nodes))
    retvals = find_retvals(graphdef)
    assert len(retvals) == 1
    graphdef = remove_extra_datasets(graphdef)
    graphdef = patch_resource_datasets(graphdef)
    retvals = find_retvals(graphdef)
    assert len(retvals) == 1
    patch_retval(retvals[0])
    print("Retval input: {}".format(retvals[0].input))
    with open("graphdef_rewritten.txt", "w") as f:
        f.write(str(graphdef))
    graph_def = tf.constant(graphdef.SerializeToString())
    ds = tf.data.experimental.analysis.ResumeDataset(graph_def, element_spec)
    if dataset_options["take_amount"]:
        ds = ds.repeat()
        ds = ds.take(dataset_options["take_amount"])
    ds = apply_pipeline_options(
        ds,
        map_and_batch_fusion=dataset_options["map_and_batch_fusion"],
        stats_filename=dataset_options["stats_filename"])
    return ds

def reinstantiate_pipeline(dataset, dataset_options):
    """Use python side graphdef for instantiation"""
    graph_def = dataset._as_serialized_graph()
    graph_def = bytes(graph_def.numpy())
    graphdef = tf1.GraphDef()
    graphdef.ParseFromString(graph_def)
    element_spec = dataset.element_spec
    with open("graphdef_reinstantiated.txt", "w") as f:
        f.write(str(graphdef))
    dataset = instantiate_pipeline(graphdef, element_spec, dataset_options)
    return dataset

def clear_graph():
    tf.compat.v1.reset_default_graph()
    tf.keras.backend.clear_session()

def span_context_to_networkx(graphdef, span_context):
    """Joins graph with events"""
    G = graphdef_to_networkx(graphdef)
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

def get_runtime_data(model, mode=DEFAULT_MODE):
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
        model)
    if max_rate_convex_native != max_rate_convex:
        print("Convex implementations have different rates!")
    max_rate_convex_existing, convex_theta_existing = \
        convex_solver.LP_upper_bounds_inner(model,
                                            use_existing_usage=True)
    cores_remaining = recommendation.remaining_CPU_cores()
    total_dataset_size = model.dataset_working_set_size()
    total_free_memory = model.memory_free()
    iter_duration = recommendation.iterator_duration()
    iter_variance = recommendation.iterator_variance()
    runtime_data = {
        "CPU_Util": CPU_Util,
        "CPU_Util_clock": CPU_Util_clock,
        "Process_CPU_Util_clock": process_CPU_Util_clock,
        "Disk_Util": Disk_Util,
        "Disk_Throughput": Disk_Throughput,
        "Disk Bytes Per Minibatch": Disk_bytes_per_root_element,
        "Total Dataset Size": total_dataset_size,
        "Total Free Memory": total_free_memory,
        "Estimated_Disk_Max_Rate_100MB": Disk_max_rate_100mb,
        "Estimated_Max_Rate": max_rate,
        "Estimated_Max_Rate_p_busy": max_rate_p_busy,
        "Estimated_Max_Rate_Convex": max_rate_convex,
        "Estimated_Max_Rate_Convex_Existing": max_rate_convex_existing,
        "Estimated_Max_Rate_Convex_Native": max_rate_convex_native,
        "Estimated_Max_Rate_Convex_Native_Naive": max_rate_convex_native_naive,
        "Cores_Remaining": cores_remaining,
        "Iterator_Duration": iter_duration,
        "Iterator_Variance": iter_variance,
        "Convex_Theta": convex_theta,
        "Convex_Theta_Existing": convex_theta_existing,
    }
    return runtime_data

def output_shape_types_to_element_spec(output_shapes, output_types):
    assert len(output_shapes.shape) == len(output_types.type), \
           "output shape is len={} but type is={}".format(
               len(output_shapes.shape), len(output_types.type))
    element_spec = [tf.TensorSpec(shape=s, dtype=t) for s, t in
                    zip(output_shapes.shape, output_types.type)]
    return tuple(element_spec)

def element_spec_from_graph(surgeon):
    """Infer element_spec"""
    retval = surgeon.find_nodes_by_op("_Retval")
    assert len(retval) == 1
    retval = retval[0]
    terminal_node = retval.input[0]
    terminal_node = find_node_by_name(surgeon, terminal_node)
    output_shapes = terminal_node.attr["output_shapes"].list
    output_types = terminal_node.attr["output_types"].list
    element_spec = output_shape_types_to_element_spec(output_shapes,
                                                      output_types)
    return element_spec


def load_pipeline(filename, dataset_options, plot_span_ctxs=False):
    plumber = tf.data.experimental.analysis.PlumberPerformanceModel(filename)
    model = plumber.model()
    recommendation = model.recommendation()
    runtime_data = get_runtime_data(model)
    graphdef = model.graphdef()
    if plot_span_ctxs:
        for i, span_context in enumerate(recommendation.span_contexts()):
            span_G = span_context_to_networkx(graphdef, span_context)
            nx.drawing.nx_pydot.write_dot(span_G, "span_{}.dot".format(i))
    surgeon = graphsurgeon.StaticGraph(graphdef)
    element_spec = element_spec_from_graph(surgeon)
    global _DATASET
    ds = reinstantiate_pipeline(_DATASET.repeat(), dataset_options)
    #ds = instantiate_pipeline(graphdef, element_spec, dataset_options)
    return ds, runtime_data

def optimize_slowest_node(graphdef, slowest_node, dataset_options):
    """Dynamically dispatch to optimization routine"""
    if slowest_node.op in BATCH_NODE_OPS:
        dataset_options["map_and_batch_fusion"] = True
        debug_string = "{}.map_and_batch_fusion=True".format(slowest_node.name)
    else:
        graphdef, debug_string = increase_node_parallelism(graphdef,
                                                           slowest_node)
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
                         "p_scheduling",
                         "num_cores_used",
                         "cardinality",
                         "expected_dataset_size",
                         "dataset_record_ratio",
                         "average_bytes_per_element_produced",
                         "average_bytes_per_element_consumed",
                         "parent_name",
                         ]

    def p_udf_f(x):
        return (x.node.state.aggregate_udf_processing_time
                / max(x.node.state.aggregate_processing_time, 1))
    def p_udf_clock_f(x):
        return (x.node.state.aggregate_udf_processing_time_clock
                / max(x.node.state.aggregate_processing_time_clock, 1))
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
                          x.p_scheduling,
                          x.num_cores_used,
                          x.cardinality,
                          x.expected_dataset_size,
                          x.dataset_record_ratio,
                          x.average_bytes_per_element_produced,
                          x.average_bytes_per_element_consumed,
                          x.parent.name if x.parent else "",
                          )
                         for x in ranked_nodes]
    df = pd.DataFrame(ranked_nodes_data, columns=ranked_nodes_cols)
    return df

def step_pipeline(filename, dataset_options, strategy=None,
                  mode=DEFAULT_MODE, plot_span_ctxs=False):
    plumber = tf.data.experimental.analysis.PlumberPerformanceModel(filename)
    model = plumber.model()
    runtime_data = get_runtime_data(model)
    recommendation = model.recommendation()
    num_cores = recommendation._analysis.global_state.machine_info.num_cores
    print("num_cores: {}".format(num_cores))
    ranked_nodes = \
    recommendation.ranked_list_bottleneck_nodes_analysis(mode=mode)
    if strategy is None:
        slowest_node = recommendation.bottleneck_node(mode=mode)
    elif strategy == "random":
        # Emulate randomly permuting choice
        slowest_node = random.choice(ranked_nodes)
    elif strategy == "random_valid":
        # Emulate randomly permuting choice
        valid_ranked_nodes = [x for x in ranked_nodes if x.is_parallel_node() or
                              (not dataset_options["map_and_batch_fusion"] and x.op
                               in BATCH_NODE_OPS)]
        slowest_node = random.choice(valid_ranked_nodes)
    elif strategy == "random_valid_deviation":
        # Emulate randomly permuting choice with recommendation removed
        _slowest_node = recommendation.bottleneck_node(mode=mode)
        valid_ranked_nodes = [x for x in ranked_nodes if
                              (x.is_parallel_node() or
                               (not dataset_options["map_and_batch_fusion"] and
                                x.op in BATCH_NODE_OPS)) and
                               x.name != _slowest_node.name]
        print("Valid nodes {} without {}: ".format(
            [x.name for x in valid_ranked_nodes],
            _slowest_node.name))
        slowest_node = random.choice(valid_ranked_nodes)
        print("Deviation: {} -> {}".format(_slowest_node.name,
                                           slowest_node.name))
    else:
        raise RuntimeError("Unknown strategy: {}".format(strategy))
    df = ranked_nodes_to_df(ranked_nodes)
    graphdef = model.graphdef()
    if plot_span_ctxs:
        for i, span_context in enumerate(recommendation.span_contexts()):
            span_G = span_context_to_networkx(graphdef, span_context)
            nx.drawing.nx_pydot.write_dot(span_G, "span_{}.dot".format(i))
    G = graphdef_to_networkx(graphdef)
    nx.drawing.nx_pydot.write_dot(G, "networkx.dot")
    topo_sort = nx.topological_sort(G)
    topo_sort_dataset = filter(is_dataset_node, topo_sort)
    remapper = remap_dataset_names(topo_sort_dataset)
    G_remapped = nx.relabel_nodes(G, remapper)
    nx.drawing.nx_pydot.write_dot(G_remapped, "networkx_remapped.dot")
    try:
        df["canonical_name"] = df["name"].map(lambda x: remapper[x])
    except KeyError as ex:
        print(ex)
        df["canonical_name"] = df["name"]
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None):
        print("Ranked_nodes:\n{}".format(df))
    current_parallelism = get_node_parallelism(slowest_node)
    print("Current parallelism for node {} is {}".format(slowest_node.name,
                                                         current_parallelism))
    with open("graphdef.txt", "w") as f:
        f.write(str(graphdef))
    graphdef, dataset_options, debug_string = \
        optimize_slowest_node(graphdef, slowest_node, dataset_options)
    surgeon = graphsurgeon.DynamicGraph(graphdef)
    graphdef = surgeon.as_graph_def()
    element_spec = element_spec_from_graph(surgeon)
    with open("graphdef2.txt", "w") as f:
        f.write(str(graphdef))
    ds = instantiate_pipeline(graphdef, element_spec, dataset_options)
    runtime_data.update(dataset_options)
    return ds, debug_string, dataset_options, df, runtime_data

def generate_new_name(surgeon, prefix: str) -> str:
    """Generates a name without collisions using the prefix."""
    i = 0
    new_name = "{}_{}".format(prefix, i)
    surgeon_node = find_node_by_name(surgeon, new_name, raise_on_fail=False)
    while surgeon_node:
        i += 1
        new_name = "{}_{}".format(prefix, i)
        surgeon_node = find_node_by_name(surgeon, new_name, raise_on_fail=False)
    return new_name

def add_const_node(surgeon, dtype: str, value):
    """ Add a constant node to graph_def.

    For example:
    name: "Const/_7"
    op: "Const"
    attr {
      key: "dtype"
      value {
        type: DT_INT64
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT64
          tensor_shape {
          }
          int64_val: 3
        }
      }
    }
    """
    if dtype == "DT_INT64":
        new_name = generate_new_name(surgeon, "Added_Const/")
        # TODO(mkuchnik): Consider tf.make_tensor_proto or using tf.Constant
        surgeon_node = graphsurgeon.create_node(new_name,
                                                op="Const",
                                                dtype=tf.int64,
                                                )
        surgeon_node.attr["value"].tensor.dtype = \
            surgeon_node.attr["dtype"].type
        surgeon_node.attr["value"].tensor.int64_val[:] = [value]
        surgeon_node.attr["value"].tensor.tensor_shape.CopyFrom(
            tf.TensorShape([]).as_proto())
    else:
        raise RuntimeError("Dtype {} is unsupported".format(dtype))
    surgeon.append(surgeon_node)
    return surgeon_node

def add_take_node(surgeon, input_ds_name, const_node_name, output_shapes,
                  output_types):
    """
    node {
      name: "TakeDataset/_30"
      op: "TakeDataset"
      input: "PrefetchDataset/_28"
      input: "Const/_29"
      attr {
        key: "output_shapes"
        value {
          list {
            shape {
              dim {
                size: 19267584
              }
            }
            shape {
              dim {
                size: 128
              }
            }
          }
        }
      }
      attr {
        key: "output_types"
        value {
          list {
            type: DT_BFLOAT16
            type: DT_INT32
          }
        }
      }
    }
    """
    new_name = generate_new_name(surgeon, "Added_TakeDataset/")
    new_node = graphsurgeon.create_node(new_name,
                                        op="TakeDataset",
                                        )
    new_node.input[:] = [input_ds_name, const_node_name]
    new_node.attr["output_shapes"].CopyFrom(output_shapes)
    new_node.attr["output_types"].CopyFrom(output_types)
    surgeon.append(new_node)
    return new_node


def add_repeat_node(surgeon, input_ds_name, const_node_name, output_shapes,
                    output_types):
    """
    node {
      name: "RepeatDataset/_24"
      op: "RepeatDataset"
      input: "CacheDatasetV2/_22"
      input: "Const/_23"
      attr {
        key: "output_shapes"
        value {
          list {
            shape {
            }
          }
        }
      }
      attr {
        key: "output_types"
        value {
          list {
            type: DT_STRING
          }
        }
      }
    }
    """
    new_name = generate_new_name(surgeon, "Added_RepeatDataset/")
    new_node = graphsurgeon.create_node(new_name,
                                        op="RepeatDataset",
                                        )
    new_node.input[:] = [input_ds_name, const_node_name]
    new_node.attr["output_shapes"].CopyFrom(output_shapes)
    new_node.attr["output_types"].CopyFrom(output_types)
    surgeon.append(new_node)
    return new_node

def copy_node_into_graph(surgeon, surgeon_node):
    # Find unique name
    new_node = copy.deepcopy(surgeon_node)
    new_name = generate_new_name(surgeon, "Added_{}/".format(new_node.op))
    new_node.name = new_name
    surgeon.append(new_node)
    return new_node

def find_unreferenced_nodes(surgeon, root_node):
    all_unreferenced_nodes = set(surgeon.node_map.keys())
    all_unreferenced_nodes.discard(root_node.name)

    def descend(node):
        for i_name in node.input:
            i = find_node_by_name(surgeon, i_name)
            all_unreferenced_nodes.discard(i_name)
            descend(i)

    descend(root_node)
    return all_unreferenced_nodes

def add_take_and_cache_node_after_node(surgeon, surgeon_node,
                                       take_amount: int = 500):
    """Adds a take, cache, and repeat dataset node after the given node.
    There is likely one consumer as dataset is DAG"""
    take_amount_node = add_const_node(surgeon, "DT_INT64", take_amount)
    output_shapes = surgeon_node.attr["output_shapes"]
    output_types = surgeon_node.attr["output_types"]
    take_node = add_take_node(surgeon, surgeon_node.name,
                              take_amount_node.name, output_shapes,
                              output_types)
    # NOTE(mkuchnik): We don't know hashcode of memory resource, so we copy it
    ds = tf.data.Dataset.from_tensor_slices([1])
    ds = ds.cache()
    graphdef = ds._as_serialized_graph()
    graph_def = bytes(graphdef.numpy())
    graphdef = tf1.GraphDef()
    graphdef.ParseFromString(graph_def)
    temp_surgeon = graphsurgeon.StaticGraph(graphdef)
    cache_node = temp_surgeon.find_nodes_by_op("CacheDatasetV2")
    assert len(cache_node) == 1
    cache_node = cache_node[0]
    assert len(cache_node.input) == 3
    file_const_node_name = cache_node.input[1]
    file_const_node = find_node_by_name(temp_surgeon, file_const_node_name)
    mem_const_node_name = cache_node.input[2]
    mem_const_node = find_node_by_name(temp_surgeon, mem_const_node_name)
    file_const_node = copy_node_into_graph(surgeon, file_const_node)
    mem_const_node = copy_node_into_graph(surgeon, mem_const_node)
    cache_node.input[0] = take_node.name
    cache_node.input[1] = file_const_node.name
    cache_node.input[2] = mem_const_node.name
    cache_node.attr["output_shapes"].CopyFrom(output_shapes)
    cache_node.attr["output_types"].CopyFrom(output_types)
    cache_node = copy_node_into_graph(surgeon, cache_node)
    repeat_amount_node = add_const_node(surgeon, "DT_INT64", -1)
    repeat_node = add_repeat_node(surgeon, cache_node.name,
                                  repeat_amount_node.name, output_shapes,
                                  output_types)
    return surgeon

def add_retval_after_node(surgeon, surgeon_node):
    # Truncate dataset
    retval = surgeon.find_nodes_by_op("_Retval")
    assert len(retval) == 1
    retval = retval[0]
    retval.input[0] = surgeon_node.name
    # Remove dangling nodes
    unreferenced_nodes = find_unreferenced_nodes(surgeon, retval)
    print("Found unreferenced nodes: {}".format(unreferenced_nodes))
    surgeon = remove_name_datasets(surgeon, unreferenced_nodes)
    return surgeon

def create_benchmark_node_dataset(surgeon, node_name: str, take_amount: int):
    """Creates a dataset to test the
    maximum throughput of a node by inserting caches and truncating
    the dataset at that node."""
    surgeon_node = find_node_by_name(surgeon, node_name)
    num_input_dataset = sum([1 for i in surgeon_node.input if "Dataset" in i])
    assert num_input_dataset == 1
    node_input = find_node_by_name(surgeon, surgeon_node.input[0])
    surgeon = add_take_and_cache_node_after_node(surgeon,
                                            node_input,
                                            take_amount)
    surgeon = add_retval_after_node(surgeon, surgeon_node)
    return surgeon

def maybe_inject_cache_optimization(surgeon, model):
    # TODO(mkuchnik): Check if remainining memory is greater than 0
    total_dataset_size = model.dataset_working_set_size()
    total_free_memory = model.memory_free()
    percentage_data_cacheable = total_free_memory / total_dataset_size
    print("Percentage of data cacheable: {}".format(percentage_data_cacheable))


def benchmark_node_dataset(surgeon, node_name: str, dataset_options: dict,
                           bench_options: dict,
                           take_amount: int = 500,
                           prefetch_amount: int = 300):
    surgeon = copy.deepcopy(surgeon)
    clear_graph()
    surgeon = create_benchmark_node_dataset(surgeon, node_name, take_amount)
    graphdef = surgeon.as_graph_def()
    element_spec = element_spec_from_graph(surgeon)
    ds = instantiate_pipeline(graphdef, element_spec, dataset_options)
    if prefetch_amount:
        ds = ds.prefetch(prefetch_amount)
    benchmark_summary = gen_util.benchmark_dataset(ds, **bench_options)
    return benchmark_summary

def benchmark_all_nodes_dataset(surgeon, dataset_options: dict,
                                bench_options: dict,
                                take_amount: int = 500,
                                parallelism_grid = None,
                                record_model = True):
    """Sweeps through nodes in the dataset and runs a benchmark over their
    parallelism parameter."""
    G = graphdef_to_networkx(surgeon.as_graph_def())
    topo_sort = nx.topological_sort(G)
    topo_sort_dataset = filter(is_dataset_node, topo_sort)
    remapper = remap_dataset_names(topo_sort_dataset)
    if parallelism_grid is None:
        parallelism_grid = range(1, 20)
    bench_options["skip_first_n"] = 10
    all_benchmark_summary = []
    IGNORE_LIST_OPS = ["TensorSliceDataset", "ShardDataset",
                       "MaxIntraOpParallelismDataset",
                       "PrivateThreadPoolDataset",
                       "ModelDataset",
                       ]
    dataset_nodes = [n for n in surgeon if "Dataset"
                     in n.op and n.op not
                     in IGNORE_LIST_OPS]
    # Shuffle for less bias
    random.shuffle(dataset_nodes)
    for node in dataset_nodes:
        _surgeon = copy.deepcopy(surgeon)
        # TODO(mkuchnik): Remove
        if "Parallel" not in node.name:
            continue
        print("Benchmarking {}".format(node.name))
        if record_model:
            filename = "stats_node.pb"
            dataset_options["stats_filename"] = filename
        benchmark_summary = benchmark_node_dataset(
            _surgeon, node.name, dataset_options, bench_options, take_amount)
        # TODO(mkuchnik): End Remove
        benchmark_summary["name"] = node.name
        benchmark_summary["canonical_name"] = remapper[node.name]
        all_benchmark_summary.append(benchmark_summary)
        if record_model:
            def try_record_model():
                try:
                    plumber = tf.data.experimental.analysis.PlumberPerformanceModel(
                        filename)
                    model = plumber.model()
                    runtime_data = get_runtime_data(model)
                    print(runtime_data)
                    recommendation = model.recommendation()
                    ranked_nodes = recommendation.ranked_list_bottleneck_nodes_analysis()
                    df = ranked_nodes_to_df(ranked_nodes)
                    print(df)
                except:
                    pass
            try_record_model()
        # TODO(mkuchnik): Remove surgeon_node
        surgeon_node = find_node_by_name(_surgeon, node.name)
        try:
            parallelism_node = parallelism_parameter_name(surgeon_node)
        except RuntimeError as ex:
            print(ex)
            benchmark_summary["parallelism"] = None
            continue
        parallelism_surgeon_node = [k for k in _surgeon if
                                    k.name == parallelism_node]
        assert len(parallelism_surgeon_node) == 1, \
            "Expected to find 1 node for {}, but found {}".format(
                parallelism_node, len(parallelism_surgeon_node))
        parallelism_surgeon_node = parallelism_surgeon_node[0]
        parallelism_tensor = parallelism_surgeon_node.attr["value"].tensor
        assert len(parallelism_tensor.int64_val) == 1
        parallelism_param = parallelism_tensor.int64_val[0]
        benchmark_summary["parallelism"] = int(parallelism_param)
        new_parallelism_surgeon_node = fork_node(_surgeon,
                                                 parallelism_surgeon_node)
        i = parallelism_parameter_index(surgeon_node)
        node_input = surgeon_node.input[i]
        assert(node_input == parallelism_surgeon_node.name)
        surgeon_node.input[i] = new_parallelism_surgeon_node.name
        parallelism_tensor = new_parallelism_surgeon_node.attr["value"].tensor
        if surgeon_node.op == "ParallelInterleaveDatasetV4":
            # Adjust cycle length to match parallelism
            i = cycle_length_parameter_index(surgeon_node)
            node_input = surgeon_node.input[i]
            cycle_surgeon_node = [k for k in _surgeon if
                                  k.name == node_input]
            assert len(cycle_surgeon_node) == 1
            cycle_surgeon_node = cycle_surgeon_node[0]
            new_cycle_surgeon_node = fork_node(_surgeon,
                                               cycle_surgeon_node)
            surgeon_node.input[i] = new_cycle_surgeon_node.name
            cycle_tensor = new_cycle_surgeon_node.attr["value"].tensor
        else:
            cycle_tensor = None
        for p in parallelism_grid:
            if p != parallelism_tensor.int64_val[0]:
                print("Benchmarking {} with parallelism={}".format(
                    node.name, p))
                parallelism_tensor.int64_val[:] = [p]
                if cycle_tensor:
                    cycle_tensor.int64_val[:] = [p]
                benchmark_summary = benchmark_node_dataset(
                    _surgeon, node.name, dataset_options, bench_options,
                    take_amount)
                benchmark_summary["name"] = node.name
                benchmark_summary["canonical_name"] = remapper[node.name]
                benchmark_summary["parallelism"] = int(p)
                all_benchmark_summary.append(benchmark_summary)
                if record_model:
                    try_record_model()
    return all_benchmark_summary

def benchmark_all_nodes_dataset_from_plumber(filename, dataset_options: dict,
                                bench_options: dict,
                                take_amount: int = 500):
    plumber = \
        tf.data.experimental.analysis.PlumberPerformanceModel(filename)
    model = plumber.model()
    graphdef = model.graphdef()
    surgeon = graphsurgeon.DynamicGraph(graphdef)
    all_benchmark_summary = benchmark_all_nodes_dataset(
        surgeon, dataset_options, bench_options)
    return all_benchmark_summary

def extract_theta_from_runtime_data(runtime_data):
    keys = ["Convex_Theta", "Convex_Theta_Existing"]
    thetas = dict()
    for k in keys:
        v = runtime_data[k]
        thetas[k] = v
        del runtime_data[k]
    return thetas

def main(_):
    nmt_parser = argparse.ArgumentParser()
    dataset_flags.add_arguments(nmt_parser)
    add_more_flags(nmt_parser)
    global FLAGS
    FLAGS = nmt_parser.parse_args()
    global _DATASET
    _DATASET = benchmark_mlperf.get_TFRecord_dataset(FLAGS)
    num_steps = FLAGS.num_steps
    num_deviations = FLAGS.num_deviations # set to 1 for normal run
    assert num_deviations >= 1, "num_deviations has to be at least 1"
    dataset_options = {
        "stats_filename": "stats_new.pb",
        "map_and_batch_fusion": False,
        "take_amount": None,
    }
    bench_options = {
        "time_limit_s": FLAGS.time_limit_s,
    }
    if FLAGS.strategy not in STRATEGIES:
        raise ValueError("time_limit_s={} not in {}".format(
            FLAGS.strategy, STRATEGIES))
    strategy = FLAGS.strategy
    ds, runtime_data = load_pipeline("stats.pb", dataset_options)
    for i in ds:
        print(ds)
        break
    thetas_dict = extract_theta_from_runtime_data(runtime_data)
    print("Runtime_data\n{}".format(pd.Series(data=runtime_data)))
    if not FLAGS.skip_baseline:
        benchmark_summary = gen_util.benchmark_dataset(ds, **bench_options)
    else:
        benchmark_summary = {"global_minibatch_rate": None}
    rate = benchmark_summary["global_minibatch_rate"]
    rates = [rate]
    changes = [None]
    ds_original = ds
    #ds = None
    #del ds
    benchmark_summary["step"] = 0
    benchmark_summary["change"] = None
    benchmark_summary["deviation"] = 0
    benchmark_summary.update(runtime_data)
    #clear_graph()
    # Start with original stats
    if not FLAGS.rebench_baseline:
        shutil.copyfile("stats.pb", "stats_new.pb")
    shutil.copyfile("stats_new.pb", "stats_new_0_0.pb")
    if FLAGS.sweep_nodes:
        _bench_options = copy.deepcopy(bench_options)
        _bench_options["profile_interval"] = None
        _dataset_options = copy.deepcopy(dataset_options)
        _dataset_options["stats_filename"] = None
        all_benchmark_summary = benchmark_all_nodes_dataset_from_plumber(
            "stats_new_0_0.pb", _dataset_options, _bench_options)
        def transform_to_df(data):
            df = pd.DataFrame(data=data, index=[0])
            return df
        all_benchmark_summary_df = [transform_to_df(s) for s in
                                    all_benchmark_summary]
        all_benchmark_summary_df = pd.concat(all_benchmark_summary_df)
        all_benchmark_summary_df.reset_index(inplace=True)
        all_benchmark_summary_df.to_csv("sweep_all_node_benchmark_stats.csv")

    thetas_df = pd.DataFrame(data=thetas_dict, index=[0])
    thetas_df["step"] = 0
    thetas_df["deviation"] = 0
    thetas_dfs = [thetas_df]
    dfs = []
    benchmark_dfs = [pd.DataFrame(benchmark_summary, index=[0])]
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
            ds, changed_node, curr_dataset_options, df, runtime_data = \
                step_pipeline("stats_new_{}_0.pb".format(i - 1),
                              curr_dataset_options,
                              curr_strategy)
            thetas_dict = extract_theta_from_runtime_data(runtime_data)
            thetas_df = pd.DataFrame(data=thetas_dict, index=[0])
            thetas_df["step"] = i
            thetas_df["deviation"] = deviation
            thetas_dfs.append(thetas_df)
            print("Runtime_data\n{}".format(pd.Series(data=runtime_data)))
            df["step"] = i
            df["deviation"] = deviation
            #clear_graph()
            try:
                benchmark_summary = gen_util.benchmark_dataset(ds,
                                                               **bench_options)
            except TypeError as ex:
                print(ex)
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
            benchmark_df = pd.DataFrame(benchmark_summary, index=[0])
            benchmark_dfs.append(benchmark_df)
            global_benchmark_df = \
                pd.concat(benchmark_dfs).reset_index(drop=True)
            global_benchmark_df.to_csv("benchmark_stats.csv")
            print("Rates:\n{}".format(pprint.pformat(rates)))
            print("Changes:\n{}".format(pprint.pformat(changes)))
        dataset_options = _new_dataset_options
    print("Rates:\n{}".format(pprint.pformat(rates)))
    print("Changes:\n{}".format(pprint.pformat(changes)))


if __name__ == '__main__':
  main(None)