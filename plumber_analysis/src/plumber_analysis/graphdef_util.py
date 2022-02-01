import tensorflow as tf
import tensorflow.compat.v1 as tf1
import networkx as nx
import graphsurgeon
import copy
from collections import deque
import pprint

import logging

SUPPORTED_PARALLELISM_OPS = ["MapAndBatchDataset",
                             "ParallelMapDatasetV2",
                             "ParallelInterleaveDatasetV4",
                             "ParallelBatchDataset",
                             ]

SUPPORTED_PARALLELIZABLE_OPS = ["BatchDatasetV2"]

CACHE_OPS = ["CacheDataset", "CacheDatasetV2"]

# TODO(mkuchnik): Add more
SOURCE_OPS = ["TFRecordDataset", "ParallelInterleaveDatasetV4"]

class PlaceholderException(ValueError):
    """For when placeholders exists in the TF graph"""
    pass

def is_dataset_node(node_name):
    # TODO(mkuchnik): Add private_threadpool_dataset, etc.
    return ("Dataset" in node_name
            or "dataset" in node_name)

def get_dataset_node_names(graphdef):
    surgeon = graphsurgeon.StaticGraph(graphdef)
    dataset_nodes = [n.name for n in surgeon if "Dataset" in n.op]
    return dataset_nodes

def get_node_parallelism(node):
    return node.parallelism

def parallelism_parameter_index(surgeon_node):
    # NOTE(mkuchnik): ParallelBatchDataset has options (batch_size,
    # num_parallel_calls, and drop_remander).
    if surgeon_node.op == "MapAndBatchDataset":
        return 2
    elif surgeon_node.op == "ParallelMapDatasetV2":
        return -1
    elif surgeon_node.op == "ParallelInterleaveDatasetV4":
        return -1
    elif surgeon_node.op == "ParallelBatchDataset":
        return -2
    else:
        raise RuntimeError("Don't know how to handle"
                           " {}".format(surgeon_node.name))

def cycle_length_parameter_index(surgeon_node):
    if surgeon_node.op == "ParallelInterleaveDatasetV4":
        return -5
    else:
        raise RuntimeError("Don't know how to handle"
                           " {}".format(surgeon_node.name))

def debug_str_to_dataset_name(debug_str):
    if debug_str is None:
        return None
    else:
        assert isinstance(debug_str, str), \
            "Expected string but got {}".format(type(debug_str))
        dataset_name_and_setting = debug_str.split(".")
        assert len(dataset_name_and_setting) == 2, \
            "Failed to split {}".format(debug_str)
        return dataset_name_and_setting[0]

def parallelism_parameter_name(surgeon_node):
    idx = parallelism_parameter_index(surgeon_node)
    return surgeon_node.input[idx]


def parallelize_op(graphdef, node_name: str):
    surgeon = graphsurgeon.DynamicGraph(graphdef)
    surgeon_node = find_node_by_name(surgeon, node_name)
    new_node = _parallelize_op(surgeon, surgeon_node)
    graph_def = surgeon.as_graph_def()
    return graph_def, new_node.name


def _parallelize_op(surgeon, surgeon_node):
    if surgeon_node.op == "BatchDatasetV2":
        new_op = "ParallelBatchDataset"
        new_node = surgeon_node
        new_node.op = new_op
        # TODO(mkuchnik): Find parent and change their name
        # Until then, cannot change name
        # new_name = generate_new_name(surgeon, "Added_{}/".format(new_node.op))
        parallelism_amount = 1
        parallelism_amount_node = add_const_node(surgeon, "DT_INT64",
                                                 parallelism_amount)
        parallelism_node_name = parallelism_amount_node.name
        #new_node.name = new_name
        inputs = list(new_node.input)
        inputs.insert(-1, parallelism_node_name) # Second to last
        logging.debug("Batch dataset now has inputs: {}".format(inputs))
        new_node.input[:] = inputs
        return new_node
    return None
    #elif surgeon.op == "MapDataset":
    #    new_op = "ParallelMapDatasetV2"

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
          surgeon_node.op == "ParallelMapDatasetV2" or
          surgeon_node.op == "MapAndBatchDataset"):
        for key_str in ["f"]:
            key_f = surgeon_node.attr[key_str].func
            key_f_str = key_f.SerializeToString().decode()
            key_f_str = key_f_str.strip("\n")
            # TODO(mkuchnik): For some reason, encoding adds "\nF" or "\nA" etc.
            key_f_str = key_f_str[1:]
            f = find_function_by_name(graph_def, key_f_str)
            fs.append(f)
    elif (surgeon_node.op == "FilterDataset"):
        for key_str in ["predicate"]:
            key_f = surgeon_node.attr[key_str].func
            key_f_str = key_f.SerializeToString().decode()
            key_f_str = key_f_str.strip("\n")
            # TODO(mkuchnik): For some reason, encoding adds "\nF" or "\nA" etc.
            key_f_str = key_f_str[1:]
            f = find_function_by_name(graph_def, key_f_str)
            fs.append(f)
    return fs

def find_functions_of_function(graphdef, function):
    functions = []

    def get_funcs(node_def):
        node_funcs = []
        for attr_name in node_def.attr:
            attr = node_def.attr[attr_name]
            str_attr = str(attr)
            func = attr.func
            name = func.name
            if name:
                # will be empty string if not valid
                logging.debug("Found function: '{}'".format(name))
                node_funcs.append(name)
            elif "func: {" in str_attr:
                logging.warning("{} CONTAINS FUNCTIONS BUT NOT FOUND:\n{}".format(
                    function.name, str_attr))
        return node_funcs

    for node_def in function.node_def:
        funcs = get_funcs(node_def)
        if funcs:
            logging.debug("node_def: {} has {}".format(node_def.name, funcs))
        functions.extend(funcs)
    return functions

def find_seeds_of_function(graphdef, function):
    """Just collect seed attr names"""
    all_seeds = []

    def get_seeds(node_def):
        node_seeds = []
        for attr_name in node_def.attr:
            if "seed" in attr_name:
                node_seeds.append(attr_name)
        return node_seeds

    for node_def in function.node_def:
        seeds = get_seeds(node_def)
        all_seeds.extend(seeds)

    return all_seeds


def find_retvals(graphdef):
    surgeon = graphsurgeon.StaticGraph(graphdef)
    nodes = surgeon.find_nodes_by_op("_Retval")
    return nodes

def find_node_by_name(surgeon, node_name, raise_on_fail=True):
    surgeon_node = surgeon.find_nodes_by_name(node_name)
    if not surgeon_node:
        # TODO(mkuchnik): need to descend into nested nodes.
        # TODO(mkuchnik): Currently, cannot find nested nodes (e.g., TFRecord).
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

def find_placeholders(graphdef):
    surgeon = graphsurgeon.StaticGraph(graphdef)
    return surgeon.find_nodes_by_op("Placeholder")


def fork_node(surgeon, surgeon_node):
    new_node = copy.deepcopy(surgeon_node)
    prefix = "Added_{}/".format(new_node.op)
    new_name =  generate_new_name(surgeon, prefix)
    new_node.name = new_name
    surgeon.append(new_node)
    new_node = find_node_by_name(surgeon, new_node.name)
    return new_node

def remove_op_datasets(surgeon, nodes_to_remove, forward=True, return_num_removed=False):
    num_removed = 0
    for n in nodes_to_remove:
        nodes = surgeon.find_nodes_by_op(n)
        # NOTE(mkuchnik): Forwarding can push too
        # many inputs to next node
        if forward:
            surgeon.forward_inputs(nodes)
        else:
            surgeon.remove(nodes)
        num_removed += len(nodes)
        nodes = surgeon.find_nodes_by_op(n)
        assert not nodes, "{} still found".format(n)
    if return_num_removed:
        assert num_removed >= 0
        return surgeon, num_removed
    else:
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

def graphdef_to_networkx(graphdef, keep_const=False, find_inner_functions=True):
    # NOTE(mkuchnik): Can also use from_pydot
    G = nx.DiGraph()
    surgeon = graphsurgeon.StaticGraph(graphdef)
    retval = surgeon.find_nodes_by_op("_Retval")
    assert len(retval) == 1
    retval = retval[0]
    visited_nodes = set([])
    node_attrs = dict()

    def descend(node):
        visited_nodes.add(node.name)
        for i_name in node.input:
            i = find_node_by_name(surgeon, i_name)
            if i.op != "Const" or (i.op == "Const" and keep_const):
                G.add_node(i.name)
                node_attr = {
                        "_name": i.name,
                        "op": i.op
                        }
                node_attrs[i.name] = node_attr
                G.add_edge(i.name, node.name)
            if i.name not in visited_nodes:
                descend(i)
        fs = find_functions_of_node(graphdef, node)

        def descend_inner_function(graphdef, f_name):
            f_node = find_function_by_name(graphdef, f_name)
            inner_fs = find_functions_of_function(graphdef, f_node)
            seeds = find_seeds_of_function(graphdef, f_node)
            node_attr = {
                    "_name": f_name,
                    "op": "function",
                    "has_random_seed": len(seeds) > 0,
                    }
            node_attrs[f_name] = node_attr
            for inner_f in inner_fs:
                G.add_node(inner_f)
                G.add_edge(inner_f, f_name)
                descend_inner_function(graphdef, inner_f)

        # Functions
        # TODO(mkuchnik): make primary key the scoped name to avoid collisions
        for f in fs:
            G.add_node(f.signature.name)
            G.add_edge(f.signature.name, node.name)

            seeds = find_seeds_of_function(graphdef, f)
            node_attr = {
                    "_name": f.signature.name,
                    "op": "function",
                    "has_random_seed": len(seeds) > 0,
                    }
            node_attrs[f.signature.name] = node_attr

            f_datasets = find_datasets_in_f(graphdef, f.signature.name)

            for f_ds in f_datasets:
                # Add dataset functions
                G.add_node(f_ds.name)
                node_attr = {
                        "_name": f_ds.name,
                        "op": "function_dataset", # TODO(mkuchnik): Fix
                        }
                node_attrs[f_ds.name] = node_attr
                G.add_edge(f_ds.name, f.signature.name)

            if find_inner_functions:
                # Add inner functions
                inner_fs = find_functions_of_function(graphdef, f)
                for inner_function_name in inner_fs:
                    G.add_node(inner_function_name)
                    G.add_edge(inner_function_name, f.signature.name)
                    descend_inner_function(graphdef, inner_function_name)

    G.add_node(retval.name)
    node_attr = {
            "_name": retval.name,
            "op": "retval",
            }
    node_attrs[retval.name] = node_attr
    descend(retval)

    logging.debug("Node attrs:\n{}".format(pprint.pformat(node_attrs)))
    nx.set_node_attributes(G, node_attrs)

    if find_inner_functions:
        random_seeds = list(
                map(lambda x: x[0],
                filter(lambda x: x[1], G.nodes(data="has_random_seed", default=None)))
                )
        logging.debug("Processing seeds from {}".format(random_seeds))
        to_process = deque(random_seeds)
        random_functions = set()

        def process_node(node_name):
            node_attr = G.nodes[node_name]
            logging.debug("Visiting {}: {}".format(node_name, node_attr))
            if "op" in node_attr and node_attr["op"] == "function":
                random_functions.add(node_name)
            for s in G.successors(node_name):
                process_node(s)

        while to_process:
            curr_node_name = to_process.popleft()
            process_node(curr_node_name)

        logging.debug("Random functions are: {}".format(random_functions))
        for node_name in G.nodes():
            if node_name in node_attrs:
                if node_attrs[node_name]["op"] == "function":
                    node_attrs[node_name]["is_random"] = node_name in random_functions
        nx.set_node_attributes(G, node_attrs)
        logging.debug("Updated node attrs:\n{}".format(pprint.pformat(node_attrs)))

    return G

def clear_graph():
    tf.compat.v1.reset_default_graph()
    tf.keras.backend.clear_session()


def instantiate_pipeline(graphdef, element_spec, dataset_options=None):
    # TODO(mkuchnik): Stats filename is stripped
    placeholders = find_placeholders(graphdef)
    if placeholders:
        raise PlaceholderException("No placeholders can exist in graph but found {}".format(placeholders))
    dataset_nodes = get_dataset_node_names(graphdef)
    logging.debug("Found dataset nodes: {}".format(dataset_nodes))
    retvals = find_retvals(graphdef)
    assert len(retvals) == 1
    graphdef = remove_extra_datasets(graphdef)
    retvals = find_retvals(graphdef)
    assert len(retvals) == 1
    patch_retval(retvals[0])
    logging.debug("Retval input: {}".format(retvals[0].input))
    with open("graphdef_rewritten.txt", "w") as f:
        f.write(str(graphdef))
    graph_def = tf.constant(graphdef.SerializeToString())
    ds = tf.data.experimental.analysis.ResumeDataset(graph_def, element_spec)
    if dataset_options:
        if dataset_options["take_amount"]:
            ds = ds.repeat()
            ds = ds.take(dataset_options["take_amount"])
        ds = apply_pipeline_options(
            ds,
            map_and_batch_fusion=dataset_options["map_and_batch_fusion"],
            threadpool_size=dataset_options["threadpool_size"],
            stats_filename=dataset_options["stats_filename"])
    return ds

def reinstantiate_pipeline(dataset, dataset_options):
    """Use python side graphdef for instantiation"""
    graph_def = dataset._as_serialized_graph()
    graph_def = bytes(graph_def.numpy())
    graphdef = tf1.GraphDef()
    graphdef.ParseFromString(graph_def)
    dataset = instantiate_pipeline(graphdef, element_spec, dataset_options)
    return dataset

def find_unreferenced_nodes(surgeon, root_node):
    all_unreferenced_nodes = set(surgeon.node_map.keys())
    all_unreferenced_nodes.discard(root_node.name)
    visited_nodes = set([])

    def descend(node):
        visited_nodes.add(node.name)
        for i_name in node.input:
            i = find_node_by_name(surgeon, i_name)
            all_unreferenced_nodes.discard(i_name)
            if i.name not in visited_nodes:
                descend(i)

    descend(root_node)
    return all_unreferenced_nodes


def copy_node_into_graph(surgeon, surgeon_node):
    # Find unique name
    new_node = copy.deepcopy(surgeon_node)
    new_name = generate_new_name(surgeon, "Added_{}/".format(new_node.op))
    new_node.name = new_name
    surgeon.append(new_node)
    return new_node

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

def add_prefetch_node(surgeon, input_ds_name, const_node_name, output_shapes,
                      output_types):
    new_name = generate_new_name(surgeon, "Added_PrefetchDataset/")
    new_node = graphsurgeon.create_node(new_name,
                                        op="PrefetchDataset",
                                        )
    new_node.input[:] = [input_ds_name, const_node_name]
    new_node.attr["buffer_size_min"].i = 1
    new_node.attr["legacy_autotune"].b = True
    new_node.attr["output_shapes"].CopyFrom(output_shapes)
    new_node.attr["output_types"].CopyFrom(output_types)
    new_node.attr["slack_period"].i = 0
    surgeon.append(new_node)
    return new_node

def add_cache_after_node(surgeon, surgeon_node, output_shapes, output_types):
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
    cache_node.input[0] = surgeon_node.name
    cache_node.input[1] = file_const_node.name
    cache_node.input[2] = mem_const_node.name
    cache_node.attr["output_shapes"].CopyFrom(output_shapes)
    cache_node.attr["output_types"].CopyFrom(output_types)
    cache_node = copy_node_into_graph(surgeon, cache_node)
    return cache_node

def add_prefetch_after_node(surgeon, surgeon_node, prefetch_amount: int):
    prefetch_amount_node = add_const_node(surgeon, "DT_INT64", prefetch_amount)
    output_shapes = surgeon_node.attr["output_shapes"]
    output_types = surgeon_node.attr["output_types"]
    prefetch_node = add_prefetch_node(surgeon, surgeon_node.name,
                                      prefetch_amount_node.name, output_shapes,
                                      output_types)
    return prefetch_node


def add_take_and_cache_node_after_node(surgeon, surgeon_node,
                                       take_amount: int = 500):
    """Adds a take, cache, and repeat dataset node after the given node.
    There is likely one consumer as dataset is DAG"""
    take_amount_node = add_const_node(surgeon, "DT_INT64", take_amount)
    output_shapes = surgeon_node.attr["output_shapes"]
    output_types = surgeon_node.attr["output_types"]
    take_node = add_take_node(surgeon, surgeon_node.name,
                              take_amount_node.name,
                              output_shapes,
                              output_types)
    cache_node = add_cache_after_node(surgeon, take_node.name,
                                      output_shapes,
                                      output_types)
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
    logging.info("Found unreferenced nodes: {}".format(unreferenced_nodes))
    surgeon = remove_name_datasets(surgeon, unreferenced_nodes)
    return surgeon

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

def remove_extra_datasets(graphdef):
    """Removes nodes such as `ModelDataset` which are appended to the dataset"""
    NODES_TO_REMOVE = ["ModelDataset",
                       "MaxIntraOpParallelismDataset",
                       "PrivateThreadPoolDataset",
                       ]
    surgeon = graphsurgeon.DynamicGraph(graphdef)
    surgeon = remove_op_datasets(surgeon, NODES_TO_REMOVE)
    return surgeon.as_graph_def()


def find_caching_datasets(graphdef):
    surgeon = graphsurgeon.DynamicGraph(graphdef)
    nodes = surgeon.find_nodes_by_op(CACHE_OPS)
    return nodes

def find_source_datasets(graphdef):
    surgeon = graphsurgeon.DynamicGraph(graphdef)
    source_nodes = surgeon.find_nodes_by_op(SOURCE_OPS)
    return source_nodes

def remove_caching_datasets(graphdef, return_num_removed=False):
    """Removes caches e.g., in case they cause OOM for the current machine"""
    surgeon = graphsurgeon.DynamicGraph(graphdef)
    nodes = surgeon.find_nodes_by_op(CACHE_OPS)
    G = graphdef_to_networkx(graphdef, keep_const=False)
    for n in nodes:
        cache_name = n.name
        input_dataset = n.input[0]
        # Inputs
        preds = list(G.predecessors(cache_name))
        # Outputs
        succs = list(G.successors(cache_name))
        logging.info("Found predecessors/successors of cache to be removed {}: {} {}".format(
            cache_name, preds, succs))
        assert len(preds) == 1, "Cache node expects one dataset input"
        assert len(succs) == 1, "Cache node expects one dataset ouput"
        cache_input_name = preds[0]
        cache_output_name = succs[0]
        surgeon_output_node = find_node_by_name(surgeon, cache_output_name)
        output_node_inputs = list(surgeon_output_node.input)
        num_replaced = 0
        for i, node_input in enumerate(output_node_inputs):
            if node_input == cache_name:
                output_node_inputs[i] = cache_input_name
                num_replaced += 1
        assert num_replaced == 1, \
                "Expected to replace one cache node, but replaced {}: {}".format(
                        num_replaced, list(surgeon_output_node.input))
        surgeon_output_node.input[:] = output_node_inputs

    surgeon, num_removed = remove_op_datasets(
            surgeon, CACHE_OPS, forward=False, return_num_removed=True)
    if return_num_removed:
        assert num_removed >= 0
        return surgeon.as_graph_def(), num_removed
    else:
        return surgeon.as_graph_def()

def remove_non_source_datasets(graphdef):
    """Finds the source nodes, and removes nodes not related to source.

    Currently only defined for 1 source. If multiple sources, will either have to return
    subgraphs of each or zip them.
    """
    # TODO(mkuchnik): This implementation assumes TFRecords with Interleave and one soure
    surgeon = graphsurgeon.DynamicGraph(graphdef)
    source_nodes = surgeon.find_nodes_by_op(SOURCE_OPS)
    if len(source_nodes) > 1:
        raise NotImplementedError("Only programmed for one source. Found {}".format(
            len(source_nodes)))
    logging.info("Found source nodes: {}".format(
        list(map(lambda x: x.name, source_nodes))))
    G = graphdef_to_networkx(graphdef, keep_const=False)
    if len(source_nodes) < 1:
        raise NotImplementedError("Can't deal with less than 1 source node: {}".format(
            len(source_nodes)))
    # A source is either a source or a source followed by an interleave
    for n in source_nodes:
        source_name = n.name
        input_dataset = n.input[0]
        # Inputs
        preds = list(G.predecessors(source_name))
        # Outputs
        succs = list(G.successors(source_name))
        logging.info("Found predecessors/successors of source {}: {} {}".format(
            source_name, preds, succs))
        final_node = n

    nodes_to_remove = set()
    nodes_to_visit = deque([final_node.name])
    while nodes_to_visit:
        curr_node = nodes_to_visit.popleft()
        nodes_to_remove.add(curr_node)
        succs = list(G.successors(curr_node))
        nodes_to_visit.extend(succs)
    nodes_to_remove.remove(final_node.name)
    nodes_to_remove.remove("dataset")
    logging.info("Removing all nodes after {}:\n{}".format(final_node.name, pprint.pformat(nodes_to_remove)))

    # Remove everything but the source and the final retval
    surgeon = remove_name_datasets(surgeon, nodes_to_remove, forward=True)
    surgeon = add_retval_after_node(surgeon, final_node)

    return surgeon.as_graph_def()

def patch_retval(retval):
    """Retval nodes have only one input, but deleting nodes may have caused them
    to have more than one."""
    patched_inputs = [i for i in retval.input if "Dataset" in i]
    retval.input[:] = patched_inputs

def apply_pipeline_options(dataset, map_and_batch_fusion: bool,
                           threadpool_size: int, stats_filename: str):
    # TODO(mkuchnik): Deprecate this perhaps, since options are now serialized?
    options = tf.data.Options()
    options.experimental_deterministic = False
    options.experimental_threading.max_intra_op_parallelism = 1
    options.experimental_threading.private_threadpool_size = \
        threadpool_size
    options.experimental_optimization.map_and_batch_fusion = \
        bool(map_and_batch_fusion)
    if stats_filename:
        options.experimental_optimization.autotune_stats_filename = \
            stats_filename
    dataset = dataset.with_options(options)
    return dataset

def set_node_parallelism(graphdef, node_name: str, parallelism: int):
    surgeon = graphsurgeon.DynamicGraph(graphdef)
    surgeon_node = find_node_by_name(surgeon, node_name)
    try:
        parallelism_node = parallelism_parameter_name(surgeon_node)
    except RuntimeError as ex:
        if parallelism != 1 or "Parallel" in node_name:
            # 1 parallelism is a no-op usually
            logging.info("{}. IGNORING".format(str(ex)))
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
    debug_string = "{}.parallelism={}".format(node_name,
                                              parallelism)
    graph_def = surgeon.as_graph_def()
    return graph_def, debug_string

def increase_node_parallelism(graphdef, node, up_parallelism=1):
    node_name = node.name
    if node.op in SUPPORTED_PARALLELIZABLE_OPS:
        graphdef, node_name = parallelize_op(graphdef, node.name)
    parallelism = get_node_parallelism(node) + up_parallelism
    return set_node_parallelism(graphdef, node_name, parallelism)

def apply_thetas_recommendation(graphdef, thetas, debug=True):
    """ For LP autotuning """
    if debug:
        logging.info("*" * 80)
        logging.info("LP autotuning")
    for node_name, parallelism in thetas.items():
        try:
            graphdef, debug_str = set_node_parallelism(graphdef, node_name,
                                                       parallelism)
            if debug:
                logging.info(debug_str)
        except RuntimeError as ex:
            continue
    if debug:
        logging.info("*" * 80)
    return graphdef
