"""Low-level pipeline optimizer.

Don't use this API directly. Use annotations.
"""

import math
import time
import pathlib
import pprint
import logging
import copy
import pickle

import tensorflow as tf

import graphsurgeon
import networkx as nx
import pandas as pd
import psutil
from plumber_analysis import (convex_solver, gen_util, graphdef_util,
                              graph_rewrites, plotting_util, machine_info,
                              extensions, high_level_analysis, bandwidth_utilities)

# TODO(mkuchnik): Move common constants to different file
FRACTION_CACHEABLE_MEMORY = 0.9

def _instantiate_pipeline(graphdef, element_spec):
    placeholders = graphdef_util.find_placeholders(graphdef)
    assert not placeholders, \
        "No placeholders can exist in graph but found {}".format(placeholders)
    dataset_nodes = graphdef_util.get_dataset_node_names(graphdef)
    retvals = graphdef_util.find_retvals(graphdef)
    assert len(retvals) == 1
    graphdef = graphdef_util.remove_extra_datasets(graphdef)
    retvals = graphdef_util.find_retvals(graphdef)
    assert len(retvals) == 1
    graphdef_util.patch_retval(retvals[0])
    graph_def = tf.constant(graphdef.SerializeToString())
    ds = tf.data.experimental.analysis.ResumeDataset(graph_def, element_spec)
    return ds

def find_parent_of_node(surgeon, node_name):
    retvals = surgeon.find_nodes_by_op("_Retval")
    assert len(retvals) == 1
    retval = retvals[0]
    def descend(node):
        for i_name in node.input:
            if i_name == node_name:
                return node
            i = graphdef_util.find_node_by_name(surgeon, i_name)
            ret = descend(i)
            if ret:
                return ret
        return None
    parent_node = descend(retval)
    return parent_node

def get_element_spec(graphdef):
    surgeon = graphsurgeon.StaticGraph(graphdef)
    element_spec = graphdef_util.element_spec_from_graph(surgeon)
    return element_spec

def instantiate_pipeline(graphdef):
    if not tf.executing_eagerly():
        # TODO(mkuchnik): Remove
        NotImplementedError("Instantiating pipeline in graph mode is not "
                            "supported. Try enabling eager execution.")
        # NOTE(mkuchnik): We disable here because graph-mode doesn't know how to
        # instantiate the ResumeDataset.
        # TODO(mkuchnik): Try implementing with
        # tf.graph_util.import_graph_def(graphdef) or similar API
    element_spec = get_element_spec(graphdef)
    ds = _instantiate_pipeline(graphdef, element_spec)
    return ds

def set_graphdef_parameters_from_dict(graphdef, thetas_dict: dict):
    for k in thetas_dict:
        # NOTE(mkuchnik): We use ceiling because generally overallocation is
        # fine
        thetas_dict[k] = max(int(math.ceil(thetas_dict[k])), 1)
    LP_graphdef = graphdef_util.apply_thetas_recommendation(
        graphdef, thetas_dict, debug=False)
    old_spec = get_element_spec(graphdef)
    new_spec = get_element_spec(LP_graphdef)
    assert old_spec == new_spec, \
            "Element spec has changed from {} to {}".format(
                    old_spec, new_spec)
    return LP_graphdef

def get_random_functions(graph_def):
    random_functions = set()

    G = graphdef_util.graphdef_to_networkx(
            graph_def, keep_const=False, find_inner_functions=True)

    def is_random_node(node) -> bool:
        n = G.nodes[node]
        return "is_random" in n and n["is_random"]

    def descend(f):
        f_name = f.signature.name
        if is_random_node(f_name):
            random_functions.add(f_name)

    for f in graph_def.library.function:
        descend(f)

    return random_functions

def remove_all_nodes_but_source(graphdef):
    """Returns a pipeline that only has the source nodes attached to it.
    For example, if there is an interleave of TFRecords followed by a mapper,
    we throw away the mapper. See implementation for limitations."""
    # TODO(mkuchnik): Handle more general cases
    return graphdef_util.remove_non_source_datasets(graphdef)

def nodes_with_random_udf(graphdef, return_cause=False):
    # NOTE(mkuchnik): Can also use from_pydot
    random_nodes = set()
    surgeon = graphsurgeon.StaticGraph(graphdef)
    retval = surgeon.find_nodes_by_op("_Retval")
    assert len(retval) == 1
    retval = retval[0]
    visited_nodes = set()
    random_functions = get_random_functions(graphdef)
    causes = dict()

    def descend(node):
        visited_nodes.add(node.name)
        for i_name in node.input:
            i = graphdef_util.find_node_by_name(surgeon, i_name)
            if i.name not in visited_nodes:
                descend(i)
        fs = graphdef_util.find_functions_of_node(graphdef, node)
        for f in fs:
            f_name = f.signature.name
            if f_name in random_functions:
                random_nodes.add(node.name)
                if return_cause:
                    cause = {"random_function": f_name}
                    causes[node.name] = cause

    descend(retval)
    if return_cause:
        return random_nodes, causes
    else:
        return random_nodes


def patch_caches_with_take_repeat(graphdef, take_amount: int=None):
    """Insert take and repeats before a cache to potentially remove transient behavior.
    Useful for benchmarking"""
    if take_amount is None:
        take_amount = 500
    surgeon = graphsurgeon.DynamicGraph(graphdef)
    cache_nodes = graphdef_util.find_caching_datasets(graphdef)
    num_patches = 0
    for cache_node in cache_nodes:
        logging.info("Patching in take before cache {}".format(
            cache_node.name))
        output_shapes = cache_node.attr["output_shapes"]
        output_types = cache_node.attr["output_types"]
        assert len(cache_node.input[:]) == 3, \
                "Expected 3 input into {}, found {}\n{}".format(
                        cache_node.name, len(cache_node.input[:]),
                        pprint.pformat(cache_node.input[:]))
        cache_input_node_name = cache_node.input[0]
        cache_input_node = graphdef_util.find_node_by_name(surgeon, cache_input_node_name)
        assert cache_input_node_name == cache_input_node.name
        if cache_input_node.op in graphdef_util.CACHE_OPS:
            take_size = get_take_size(surgeon, cache_input_node)
            logging.info("Already found take before cache {} with size {}".format(
                cache_input_node.name,
                take_size))
            # TODO(mkuchnik): More robust check
            continue
        parent_node = find_parent_of_node(surgeon, cache_node.name)
        take_amount_node = graphdef_util.add_const_node(surgeon, "DT_INT64", take_amount)
        take_node = graphdef_util.add_take_node(surgeon, cache_input_node_name,
                                  take_amount_node.name,
                                  output_shapes,
                                  output_types)
        swap_dataset_into_parent(cache_node, take_node)
        repeat_amount_node = graphdef_util.add_const_node(surgeon, "DT_INT64", -1)
        repeat_node = graphdef_util.add_repeat_node(surgeon, cache_node.name,
                                      repeat_amount_node.name, output_shapes,
                                      output_types)
        swap_dataset_into_parent(parent_node, repeat_node)
        num_patches += 1
    new_graphdef = surgeon.as_graph_def()
    return new_graphdef, num_patches


def insert_cache_highest(model, graphdef, recommendation, candidates: dict,
                         max_memory: float, add_take_repeat: bool=False,
                         take_amount: int=None):
    """Insert cache using candidates and max_memory as constraints."""
    # TODO(mkuchnik): Filter out bad trades of memory vs. compute (e.g.,
    # RepeatDataset)
    # TODO(mkuchnik): Shuffle is ambiguously handled
    if take_amount is None:
        take_amount = 500
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
    if current_candidate is None:
        logging.info("Didn't find cache node")
        return graphdef
    surgeon_node = graphdef_util.find_node_by_name(surgeon, current_candidate)
    cache_size = candidates[current_candidate]
    logging.info("Adding cache after: {} ({}GB)".format(surgeon_node.name,
                                                 cache_size / 1e9))
    output_shapes = surgeon_node.attr["output_shapes"]
    output_types = surgeon_node.attr["output_types"]
    parent_node = find_parent_of_node(surgeon, surgeon_node.name)

    if add_take_repeat:
        take_amount_node = graphdef_util.add_const_node(surgeon, "DT_INT64", take_amount)
        take_node = graphdef_util.add_take_node(surgeon, surgeon_node.name,
                                  take_amount_node.name,
                                  output_shapes,
                                  output_types)
        cache_node = graphdef_util.add_cache_after_node(surgeon, take_node,
                                          output_shapes,
                                          output_types)
        repeat_amount_node = graphdef_util.add_const_node(surgeon, "DT_INT64", -1)
        repeat_node = graphdef_util.add_repeat_node(surgeon, cache_node.name,
                                      repeat_amount_node.name, output_shapes,
                                      output_types)
        last_node = repeat_node
    else:
        last_node = graphdef_util.add_cache_after_node(surgeon,
                                                       surgeon_node,
                                                       output_shapes,
                                                       output_types)
    swap_dataset_into_parent(parent_node, last_node)
    new_graphdef = surgeon.as_graph_def()
    return new_graphdef

def swap_dataset_inputs(surgeon_node, input_node_name_from, input_node_name_to):
    found = False
    for i, name in enumerate(surgeon_node.input):
        if name == input_node_name_from:
            found = True
            break
    assert found
    surgeon_node.input[i] = input_node_name_to
    return surgeon_node

def set_prefetch_buffer_size(surgeon, surgeon_node, buffer_size):
    assert surgeon_node.op == "PrefetchDataset"
    i_name = surgeon_node.input[1]
    buffer_node = graphdef_util.find_node_by_name(surgeon, i_name)
    new_buffer_node = graphdef_util.fork_node(surgeon, buffer_node)
    buffer_tensor = new_buffer_node.attr["value"].tensor
    buffer_tensor.int64_val[:] = [buffer_size]
    swap_dataset_inputs(surgeon_node, buffer_node.name, new_buffer_node.name)

def get_prefetch_buffer_size(surgeon, surgeon_node):
    assert surgeon_node.op == "PrefetchDataset"
    i_name = surgeon_node.input[1]
    buffer_node = graphdef_util.find_node_by_name(surgeon, i_name)
    buffer_tensor = buffer_node.attr["value"].tensor
    val_buf = buffer_tensor.int64_val[:]
    return val_buf[0]

def get_take_size(surgeon, surgeon_node):
    assert surgeon_node.op == "TakeDataset"
    i_name = surgeon_node.input[1]
    buffer_node = graphdef_util.find_node_by_name(surgeon, i_name)
    buffer_tensor = buffer_node.attr["value"].tensor
    val_buf = buffer_tensor.int64_val[:]
    return val_buf[0]

def get_node_parallelism(surgeon, surgeon_node) -> int:
    parallelism_node = graphdef_util.parallelism_parameter_name(surgeon_node)
    parallelism_surgeon_node = [k for k in surgeon if
                                k.name == parallelism_node]
    assert len(parallelism_surgeon_node) == 1, \
        "Expected to find 1 node for {}, but found {}".format(
            parallelism_node, len(parallelism_surgeon_node))
    parallelism_surgeon_node = parallelism_surgeon_node[0]
    i = graphdef_util.parallelism_parameter_index(surgeon_node)
    node_input = surgeon_node.input[i]
    assert(node_input == parallelism_node)

    surgeon_node.input[i] = parallelism_surgeon_node.name
    parallelism_tensor = parallelism_surgeon_node.attr["value"].tensor
    parallelism_val = parallelism_tensor.int64_val[0]
    return parallelism_val

def get_all_node_parallelisms(graphdef) -> dict:
    surgeon = graphsurgeon.DynamicGraph(graphdef)
    dataset_nodes = graphdef_util.get_dataset_node_names(graphdef)
    node_parallelisms = dict()
    for node_name in dataset_nodes:
        node = graphdef_util.find_node_by_name(surgeon, node_name)
        if node.op in graphdef_util.SUPPORTED_PARALLELISM_OPS:
            parallelism = get_node_parallelism(surgeon, node)
        else:
            parallelism = 1
        node_parallelisms[node_name] = parallelism
    return node_parallelisms

def recommendation_to_nodes(recommendation):
    """Gets names of nodes that recommendation is tracking (loosely)."""
    recommendation_nodes = recommendation.ranked_list_bottleneck_nodes_analysis()
    recommendation_nodes = filter(lambda x: x.num_cores_used > 0.,
                                  recommendation_nodes)
    recommendation_nodes = map(lambda x: x.name, recommendation_nodes)
    recommendation_nodes = set(recommendation_nodes)
    return recommendation_nodes


def find_first_real_node_fn(surgeon, recommendation):
    recommendation_nodes = recommendation_to_nodes(recommendation)
    IGNORE_LIST = set(["ModelDataset",
                       "MaxIntraOpParallelismDataset",
                       "PrivateThreadPoolDataset",
                       ])
    def is_real_node(node) -> bool:
        return node.name in recommendation_nodes and node.op not in IGNORE_LIST
    def find_first_real_node():
        _lineage = []
        retvals = surgeon.find_nodes_by_op("_Retval")
        assert len(retvals) == 1
        retval = retvals[0]
        nodes = [retval]
        _lineage.append(retval.name)
        while nodes:
            node = nodes.pop()
            if is_real_node(node):
                break
            _lineage.append(list(node.input))
            for i_name in node.input:
                i = graphdef_util.find_node_by_name(surgeon, i_name)
                nodes.append(i)
        logging.info("First real node: {}.\nLineage: {}\nRecommendation nodes: {}".format(
            node.name, _lineage, recommendation_nodes))
        return node
    return find_first_real_node

def swap_dataset_into_parent(parent_node, surgeon_node):
    num_input_dataset = sum([1 for i in parent_node.input if "Dataset" in i])
    assert num_input_dataset == 1, \
            "Expected to find 1 input dataset for {}, found {}".format(
                    parent_node.name, num_input_dataset)
    assert "Dataset" in parent_node.input[0], \
            "Expected to find a dataset, found {}".format(parent_node.input[0])
    parent_node.input[0] = surgeon_node.name

def insert_prefetch_highest(model, graphdef, recommendation, prefetch_amount, return_is_diff=False):
    surgeon = graphsurgeon.DynamicGraph(graphdef)
    find_first_real_node = find_first_real_node_fn(surgeon, recommendation)
    surgeon_node = find_first_real_node()
    logging.info("Adding prefetch after: {}".format(surgeon_node.name))
    is_diff = False
    if surgeon_node.op == "PrefetchDataset":
        existing_size = get_prefetch_buffer_size(surgeon, surgeon_node)
        logging.info("Found existing prefetch dataset with size"
              " {}".format(existing_size))
        set_prefetch_buffer_size(surgeon, surgeon_node, prefetch_amount)
    else:
        parent_node = find_parent_of_node(surgeon, surgeon_node.name)
        logging.info("Expected parent_node is: {}".format(parent_node.name))
        prefetch_node = graphdef_util.add_prefetch_after_node(surgeon,
                                                              surgeon_node,
                                                              prefetch_amount)
        swap_dataset_into_parent(parent_node, prefetch_node)
        is_diff = True
        #surgeon = graphdef_util.add_retval_after_node(surgeon, surgeon_node)
    new_graphdef = surgeon.as_graph_def()
    if return_is_diff:
        return new_graphdef, is_diff
    else:
        return new_graphdef

def remove_prefetch(graphdef):
    surgeon = graphsurgeon.DynamicGraph(graphdef)
    #surgeon = graphdef_util.remove_op_datasets(surgeon, ["PrefetchDataset"])
    nodes = surgeon.find_nodes_by_op(["PrefetchDataset"])
    for n in nodes:
        n.input[:] = n.input[:][:1]
    surgeon.forward_inputs(nodes)
    new_graphdef = surgeon.as_graph_def()
    return new_graphdef

def remove_unreferenced_nodes(graphdef):
    surgeon = graphsurgeon.DynamicGraph(graphdef)
    retvals = surgeon.find_nodes_by_op("_Retval")
    assert len(retvals) == 1
    retval = retvals[0]
    unreferenced_nodes = graphdef_util.find_unreferenced_nodes(surgeon, retval)
    surgeon = graphdef_util.remove_name_datasets(surgeon, unreferenced_nodes)
    graphdef_util.patch_retval(retval)
    new_graphdef = surgeon.as_graph_def()
    return new_graphdef

def visualize_graphdef(graphdef):
    G = graphdef_util.graphdef_to_networkx(graphdef, keep_const=False)
    topo_sort = nx.topological_sort(G)
    for node in topo_sort:
        logging.info(node)

def get_topo_remapper(graphdef, numeric=False):
    G = graphdef_util.graphdef_to_networkx(graphdef, keep_const=False)
    topo_sort = nx.topological_sort(G)
    topo_sort_dataset = filter(graphdef_util.is_dataset_node, topo_sort)
    if numeric:
        return dict([(k, i) for i, k in enumerate(topo_sort_dataset)])
    else:
        remapper = graphdef_util.remap_dataset_names(topo_sort_dataset)
        return remapper

def get_safe_topo_remapper_fn(graphdef, numeric=False):
    remapper = get_topo_remapper(graphdef, numeric)
    def lookup(x):
        if x in remapper:
            return remapper[x]
        else:
            if numeric:
                return -1
            else:
                return x
    return lookup

def disable_inter_op_parallelism(graphdef):
    surgeon = graphsurgeon.DynamicGraph(graphdef)
    nodes = surgeon.find_nodes_by_op("ParallelMapDatasetV2")
    for node in nodes:
        attr = node.attr["use_inter_op_parallelism"]
        attr.b = False
    new_graphdef = surgeon.as_graph_def()
    return new_graphdef

def get_performance_parameters(graphdef) -> dict:
    surgeon = graphsurgeon.StaticGraph(graphdef)
    parallel_ops = graphdef_util.SUPPORTED_PARALLELISM_OPS
    nodes = surgeon.find_nodes_by_op(parallel_ops)
    parameters = dict()
    for node in nodes:
        parameters[node.name] = get_node_parallelism(surgeon, node)
    nodes = surgeon.find_nodes_by_op("PrefetchDataset")
    for node in nodes:
        parameters[node.name] = get_prefetch_buffer_size(surgeon, node)
    return parameters

def get_cache_nodes(graphdef) -> list:
    surgeon = graphsurgeon.StaticGraph(graphdef)
    cache_ops = ["CacheDatasetV2"]
    nodes = surgeon.find_nodes_by_op(cache_ops)
    node_names = list(map(lambda x: x.name, nodes))
    return node_names

def check_graphdef_compatibility(graphdef1, graphdef2):
    """Checks if two graphdefs have same name structure"""
    G1 = graphdef_util.graphdef_to_networkx(graphdef1, keep_const=False)
    G2 = graphdef_util.graphdef_to_networkx(graphdef2, keep_const=False)
    def node_match_fn(n1, n2):
        # Compare internal names
        return n1["_name"] == n2["_name"]
    edge_match_fn = None
    is_match = nx.algorithms.isomorphism.is_isomorphic(
            G1, G2, node_match_fn, edge_match_fn)
    if is_match:
        return is_match, None
    node_diff_removed = set(G1.nodes) - set(G2.nodes)
    node_diff_added = set(G2.nodes) - set(G1.nodes)
    # TODO(mkuchnik): Add edge diff
    diff = {
            "node_diff_remove": node_diff_removed,
            "node_diff_added": node_diff_added,
            }
    return is_match, diff

def check_graphdef_recommendation_compatibility(graphdef, recommendation):
    """Diff relative to recommendation"""
    recommendation_nodes_set = recommendation_to_nodes(
            recommendation)
    graphdef_nodes_set = set(graphdef_util.get_dataset_node_names(graphdef))
    nodes_not_in_graphdef = recommendation_nodes_set - graphdef_nodes_set
    nodes_not_in_recommendation = graphdef_nodes_set - recommendation_nodes_set
    diff = {
            "node_diff_added": nodes_not_in_recommendation,
            "node_diff_remove": nodes_not_in_graphdef,
            }
    is_match = not len(nodes_not_in_graphdef) and not len(nodes_not_in_recommendation)
    return is_match, diff

def check_graphdef_equality(graphdef1, graphdef2) -> bool:
    return graphdef1.SerializeToString() == graphdef2.SerializeToString()

class GraphdefMismatchError(ValueError):
    """Represents incompatible graphdefs"""
    pass

class DataPipelineOptimizer(object):
    def __init__(self, plumber, calibrate_system=None, step_size=None,
                 machine_info=None):
        """
        Creates a GraphOptimizer using a Plumber object.
        Current abstraction is mostly immutable with respect to plumber and with checks
        to ensure the mutable graphdef doesn't get too out of sync with plumber.
        TODO(mkuchnik): Make immutable and functional
        """
        self.plumber = plumber
        self.step_size = step_size
        self.machine_info = machine_info
        self.core_multiplier = None
        if calibrate_system:
            self._calibrate_system()
            logging.info("machine_info", self.machine_info)
        self.prefetch_amount_added = 0
        self.disable_optimizations = True
        self.cache_enabled = False
        self.prefetching_optimization_enabled = True
        self.parallelism_optimization_enabled = True
        self.disk_optimization_enabled = True
        self.cache_optimization_enabled = True
        self.materialized_view_cache_optimization_enabled = True
        self._throw_if_element_spec_changed()

    def fork(self):
        """Returns a copy of this optimizer, useful for reverting optimizations."""
        return copy.deepcopy(self)

    @property
    def plumber(self):
        return self._plumber

    @plumber.setter
    def plumber(self, plumber):
        logging.info("Setting plumber")
        self._plumber = plumber
        self._model = self._plumber.model()
        self._recommendation = self._model.recommendation()
        if self.mutable_graphdef:
            is_compatible, diff = check_graphdef_compatibility(
                    self.mutable_graphdef, self._model.graphdef())
            if not is_compatible:
                logging.warning("Discarding mutable graphdef. Changes discarded:\n{}".format(
                    pprint.pformat(diff)))
            self._invalidate_mutable_graphdef_state()
        if self.original_element_spec is None:
            self.original_element_spec = self.element_spec()
        self._throw_if_element_spec_changed()

    @property
    def model(self):
        # Immutable
        return copy.deepcopy(self._model)

    @property
    def recommendation(self):
        # Immutable
        return copy.deepcopy(self._recommendation)

    @property
    def graphdef(self):
        # Immutable
        return copy.deepcopy(self.model.graphdef())

    @property
    def mutable_graphdef(self):
        try:
            mutable_graphdef = self._mutable_graphdef
            if mutable_graphdef is None:
                # TODO(mkuchnik): can just delete attribute
                self._mutable_graphdef = self.graphdef
                mutable_graphdef = self._mutable_graphdef
        except AttributeError:
            self._mutable_graphdef = self.graphdef
            mutable_graphdef = self._mutable_graphdef
        return mutable_graphdef

    @mutable_graphdef.setter
    def mutable_graphdef(self, graphdef):
        spec = get_element_spec(graphdef)
        # spec is initially none until cache is updated
        if self.original_element_spec and spec != self.original_element_spec:
            raise ValueError("Cannot change graphdef spec from {} to {}".format(
                self.original_element_spec, spec))
        if self.recommendation:
            is_compatible, diff = check_graphdef_recommendation_compatibility(graphdef, self.recommendation)
            if not is_compatible:
                logging.warning(
                        "Found nodes in graphdef that are not in recommendation. Diff:\n{}".format(
                            pprint.pformat(diff)))
        is_compatible, diff = check_graphdef_compatibility(self.last_safe_graphdef, graphdef)
        if not is_compatible:
            raise GraphdefMismatchError("GraphDef is significantly different. Diff:\n{}".format(
                pprint.pformat(diff)))
        self._mutable_graphdef = graphdef

    def unsafe_mutable_graphdef_update(self, graphdef):
        logging.info("Unsafe graphdef update!")
        spec = get_element_spec(graphdef)
        # spec is initially none until cache is updated
        if self.original_element_spec and spec != self.original_element_spec:
            raise ValueError("Cannot change graphdef spec from {} to {}".format(
                self.original_element_spec, spec))
        if self.recommendation:
            is_compatible, diff = check_graphdef_recommendation_compatibility(graphdef, self.recommendation)
            if not is_compatible:
                logging.warning(
                        "Found nodes in graphdef that are not in recommendation. Diff:\n{}".format(
                            pprint.pformat(diff)))
        is_compatible, diff = check_graphdef_compatibility(self.last_safe_graphdef, graphdef)
        if not is_compatible:
            logging.warning("GraphDef is significantly different. Diff:\n{}".format(
                pprint.pformat(diff)))
        if not is_compatible or not check_graphdef_equality(graphdef, self.mutable_graphdef):
            logging.info("Detected actual unsafe update")
            # UNSAFE EXCEPTION. Safety is now relative to this graphdef
            self._last_safe_graphdef = copy.deepcopy(graphdef)
        else:
            # TODO(mkuchnik): graph equality is weird, investigate to be sure no false positives
            logging.info("Unsafe update is No-Op")
        self._mutable_graphdef = graphdef

    def _invalidate_mutable_graphdef_state(self):
        """Call whenever graphdef is no longer in sync with model"""
        try:
            del self._mutable_graphdef
        except AttributeError:
            pass
        try:
            del self._last_safe_graphdef
        except AttributeError:
            pass

    @property
    def last_safe_graphdef(self):
        """Since we enable unsafe updates, we break safety. Make safety then relative"""
        try:
            return copy.deepcopy(self._last_safe_graphdef)
        except AttributeError:
            return self.graphdef

    @property
    def original_element_spec(self):
        try:
            return self._original_element_spec
        except AttributeError:
            return None

    @original_element_spec.setter
    def original_element_spec(self, original_element_spec):
        if self.original_element_spec is None:
            self._original_element_spec = original_element_spec
        else:
            raise ValueError("Cannot re-set original element spec")

    def _throw_if_element_spec_changed(self):
        spec = self.element_spec()
        if spec != self.original_element_spec:
            raise RuntimeError("Element spec has changed from {} to {}".format(
                self.original_element_spec, spec))

    def networkx(self):
        return high_level_analysis.HighLevelPlumberModel(self.plumber).networkx()

    def current_plumber_emperical_rate(self) -> float:
        rate = self.recommendation.actual_rate()
        return rate

    def apply_parallelism(self):
        """parallelism 0 is just increasing the parallelism of the node
        disk bottlenecks may be hit"""
        ret = self._apply_prefetch_optimizations()
        self._apply_parallelism_optimizations()
        ret = self._apply_disk_optimizations()

    def update_plumber(self, benchmark_time_s=None):
        self._throw_if_element_spec_changed()
        self._update_plumber(benchmark_time_s)
        self._throw_if_element_spec_changed()

    def apply_cache(self, add_take_repeat: bool=True):
        """cache 0 is just increasing the cache of the node
        disk bottlenecks may be hit"""
        self._apply_cache_optimizations(add_take_repeat=add_take_repeat)

    def instantiate_pipeline(self):
        """
        Returns a potentially optimized variant of the pipeline
        """
        return instantiate_pipeline(self.mutable_graphdef)

    def instantiate_test_pipeline(self, return_num_patches=False):
        """
        Returns a potentially optimized variant of the pipeline
        with transient behavior removed. Do not use in production.
        """
        graphdef, num_patches = patch_caches_with_take_repeat(self.mutable_graphdef)
        if return_num_patches:
            return instantiate_pipeline(graphdef), num_patches
        else:
            return instantiate_pipeline(graphdef)

    def source_dataset(self):
        """
        Instantiates a source-only variant of the dataset.
        Useful for checking dataset throughput
        """
        graphdef = remove_all_nodes_but_source(self.mutable_graphdef)
        return instantiate_pipeline(graphdef)

    def element_spec(self):
        """
        Returns the element spec of the graphdef
        """
        return get_element_spec(self.mutable_graphdef)

    def fake_dataset(self, apply_repeat=True):
        """
        Returns a cheap to evaluate fake dataset
        """
        elem_spec = self.element_spec()
        features = []
        logging.info("element_spec: {}".format(elem_spec))
        for e in elem_spec:
            shape = e.shape
            dtype = e.dtype
            feature = tf.zeros(shape=shape, dtype=dtype)
            feature_ds = tf.data.Dataset.from_tensors(feature)
            features.append(feature_ds)
        features = tuple(features)
        ds = tf.data.Dataset.zip(features)
        if apply_repeat:
            ds = ds.repeat()
        return ds

    def roofline(self, filename=None, **kwargs) -> dict:
        Rmin = self.recommendation.min_latency()
        nodes = self.recommendation.ranked_list_bottleneck_nodes_analysis()
        cpu_bottleneck_node = nodes[0]
        bottleneck_rate = cpu_bottleneck_node.expected_parallel_max_rate()
        if bottleneck_rate == 0:
            logging.warning("Bottleneck rate is {}".format(bottleneck_rate))
            bottleneck_rate = 1e-6
        VbSb = 1. / bottleneck_rate
        topo_remapper_fn = get_safe_topo_remapper_fn(self.mutable_graphdef)
        X_cpu_bounds, thetas = self.recommendation.LP_upper_bounds()
        if self.machine_info:
            # TODO(mkuchnik): check if dataset is cached
            estimated_disk_bw = self.machine_info["FILES"][0]["BANDWIDTH"]
            X_disk_bounds = self.recommendation.disk_upper_bounds(
                estimated_disk_bw)
        else:
            X_disk_bounds = None
        def par_name(node):
            name = node.name
            parallelism = node.parallelism
            try:
                theta = thetas[name]
            except KeyError:
                theta = "?"
            name = topo_remapper_fn(name)
            return "{}={}({:.1f})".format(name, parallelism, theta)
        # Rate, name=parallelism(theta)
        nodes_rates = map(lambda x:
                          (x.expected_parallel_max_rate(), par_name(x)), nodes)
        N_star = Rmin / VbSb
        W = self.model.total_CPU_time()
        T = self.model.total_wallclock_time()
        N = W / T
        internal_N_stats = self.all_N_stats()
        N_internal = sum(internal_N_stats.values())
        C = self.recommendation.actual_rate() * T
        if C == 0:
            logging.warning("C={}".format(C))
            C = 1e-6
        R = W / C
        curr_bounds = min(X_cpu_bounds, N / R)
        if self.machine_info:
            estimated_disk_bw = self.machine_info["FILES"][0]["BANDWIDTH"]
            X_disk_bounds = self.recommendation.disk_upper_bounds(
                estimated_disk_bw)
        else:
            X_disk_bounds = None
        stats = {
            "min_latency": Rmin,
            "VbSb": VbSb,
            "X_cpu_bounds": X_cpu_bounds,
            "X_disk_bounds": X_disk_bounds,
            "N_star": N_star,
            "N": N,
            "N_internal": N_internal,
            "R": R,
            "CPU_current_bounds": curr_bounds,
            "disk_bounds": X_disk_bounds,
        }
        if filename:
            plotting_util.generate_roofline(filename, N, R, X_cpu_bounds,
                                            N_star, nodes_rates=nodes_rates,
                                            X_disk_bounds=X_disk_bounds,
                                            **kwargs)
        return stats

    def all_N_stats(self) -> dict:
        T = self.model.total_wallclock_time()
        nodes = self.recommendation.ranked_list_bottleneck_nodes_analysis()
        stats = dict()
        for x in nodes:
            stats[x.name] = x.N_customers
        return stats

    def disable_inter_op_parallelism(self):
        new_graphdef = self.mutable_graphdef
        new_graphdef = disable_inter_op_parallelism(new_graphdef)
        self.mutable_graphdef = new_graphdef

    def enable_optimizations(self):
        self.disable_optimizations = True

    def get_performance_parameters(self) -> dict:
        params = get_performance_parameters(self.mutable_graphdef)
        return params

    def _update_plumber(self, benchmark_time_s, patch_caches=True):
        """
        mutable_graphdef -> plumber

        This function is DANGEROUS. state is maintained in graphdef, which
        can be discarded if plumber file overwrites the graphdef.
        """
        if benchmark_time_s is None:
            benchmark_time_s = 22

        def remove_file_if_exists(filename):
            if filename:
                f = pathlib.Path(filename)
                f.unlink()

        graphdef_util.clear_graph()
        stats_filename = "_optimizer_stats.pb"
        if patch_caches:
            ds, num_patches = self.instantiate_test_pipeline(return_num_patches=True)
            if num_patches:
                stats_filename = None
        else:
            ds = self.instantiate_pipeline()
        def run_test(ds, stats_filename):
            options = tf.data.Options()
            if stats_filename:
                gen_util.add_analysis_to_dataset_options(options,
                                                         hard_fail=True,
                                                         stats_filename=stats_filename)
            if self.disable_optimizations:
                options.experimental_optimization.autotune = False
                options.experimental_optimization.map_parallelization = False
                options.experimental_optimization.map_and_batch_fusion = False
            ds = ds.with_options(options)
            gen_util.drop_caches()
            summary = gen_util.benchmark_dataset(ds, time_limit_s=benchmark_time_s)
            return summary
        summary = run_test(ds, stats_filename)
        rate = summary["global_minibatch_rate"]
        if patch_caches and num_patches:
            del ds
            remove_file_if_exists(stats_filename)
            graphdef_util.clear_graph()
            stats_filename = "_optimizer_stats.pb"
            # NOTE(mkuchnik): To get a proper analysis file,
            # we just run the unpatched model
            ds = self.instantiate_pipeline()
            summary2 = run_test(ds, stats_filename)
            rate2 = summary2["global_minibatch_rate"]
            logging.info("Re-ran pipeline without patch: {} vs {}".format(
                rate, rate2))
        plumber = tf.data.experimental.analysis.PlumberPerformanceModel(
            stats_filename)
        self.plumber = plumber
        remove_file_if_exists(stats_filename)
        graphdef_util.clear_graph()

    def _print_graphdef(self):
        # visualize_graphdef(self.graphdef)
        summary_df = self.get_cache_summary()
        with pd.option_context('display.max_rows', None,
                               'display.max_columns', None):
            logging.info("Cache DF:\n".format(summary_df))

    def _calibrate_system(self):
        self.machine_info = graph_rewrites.generate_localhost_machine_info(
            self.model, self.recommendation, False)

    def _clean_up_graphdef(self, remove_fake_nodes=False):
        new_graphdef = self.mutable_graphdef
        if remove_fake_nodes:
            new_graphdef = graphdef_util.remove_extra_datasets(new_graphdef)
        new_graphdef = remove_unreferenced_nodes(new_graphdef)
        self.mutable_graphdef = new_graphdef

    def _dump_graphdef(self, filename, graphdef=None):
        with open(filename, "w") as f:
            if graphdef is None:
                graphdef = self.mutable_graphdef
            f.write(str(graphdef))

    def _remove_prefetching(self):
        new_graphdef = remove_prefetch(self.mutable_graphdef)
        self.mutable_graphdef = new_graphdef

    def _reset_parallelism(self):
        nodes = self.recommendation.ranked_list_bottleneck_nodes_analysis()
        node_names = map(lambda node: node.name, nodes)
        thetas_dict = {k: 1 for k in node_names}
        LP_graphdef = set_graphdef_parameters_from_dict(self.mutable_graphdef,
                                                        thetas_dict)
        self.mutable_graphdef = LP_graphdef

    def set_bandwidth_parallelism_equations(self, params):
        logging.info("Setting bandwidth params to {}".format(params))
        assert "m1" in params
        assert "m2" in params
        assert "b1" in params
        assert "b2" in params
        assert "x_thresh" in params
        assert "source_node" in params
        self.bandwidth_params = dict(params)

    def experiment_params(self):
        try:
            bandwidth_params = self.bandwidth_params
        except AttributeError:
            bandwidth_params = None
        params = {
                "bandwidth_params": bandwidth_params,
                }
        return params

    def apply_experiment_params(self, params):
        bandwidth_params = params["bandwidth_params"]
        self.bandwidth_params = bandwidth_params

    def get_parallelism(self):
        # TODO(mkuchnik): Overlaps with get_performance_parameters
        return get_all_node_parallelisms(self.mutable_graphdef)

    def set_parallelism(self, parallelism_dict):
        """Takes a dict of node: parallelism pairs and sets them."""
        # NOTE(mkuchnik): Not recorded
        logging.info("Setting parallelism to:\n{}".format(
            pprint.pformat(parallelism_dict)))
        self._throw_if_element_spec_changed()
        new_graphdef = set_graphdef_parameters_from_dict(
                self.mutable_graphdef, parallelism_dict)
        self.mutable_graphdef = new_graphdef

    def source_nodes(self):
        """Returns names of source nodes"""
        nodes = graphdef_util.find_source_datasets(self.mutable_graphdef)
        return list(map(lambda x: x.name, nodes))

    def _get_bandwidth_params(self):
        try:
            bandwidth_params = self.bandwidth_params
        except AttributeError:
            logging.info("Didn't find bandwidth params")
            bandwidth_params = None
        if bandwidth_params:
            # TODO(mkuchnik): Hack to update source node
            source_nodes = self.source_nodes()
            assert len(source_nodes) == 1
            bandwidth_params["source_node"] = source_nodes[0]
        return bandwidth_params

    def _apply_parallelism_optimizations(self):
        # TODO(mkuchnik): Add interleave logic.
        bandwidth_params = self._get_bandwidth_params()
        if (self.step_size is not None
                or self.core_multiplier is not None
                or bandwidth_params is not None):
            if bandwidth_params:
                estimated_rate, thetas_dict = convex_solver.LP_upper_bounds(
                    self.plumber, max_change=self.step_size,
                    core_multiplier=self.core_multiplier,
                    bandwidth_params=None)
                logging.info("Without BW: rate {} thetas {}".format(estimated_rate, thetas_dict))
            estimated_rate, thetas_dict = convex_solver.LP_upper_bounds(
                self.plumber, max_change=self.step_size,
                core_multiplier=self.core_multiplier,
                bandwidth_params=bandwidth_params)
            if bandwidth_params:
                logging.info("With BW: rate {} thetas {}".format(estimated_rate, thetas_dict))
                bw_rate_predictor = bandwidth_utilities.piecewise_linear_predictor_from_params(
                        bandwidth_params)
                source_node = bandwidth_params["source_node"]
                if source_node in thetas_dict:
                    bw_theta = thetas_dict[source_node]
                    bw_rate_hat = bw_rate_predictor(bw_theta)
                    logging.info("Predicting max BW rate of {} elements/second".format(bw_rate_hat))
                else:
                    # TODO(mkuchnik): Probably caching, but add more checks
                    logging.info("Did not find source {}, so assuming caching is used.".format(
                        source_node))
        else:
            estimated_rate, thetas_dict = self.recommendation.LP_upper_bounds()
        logging.info("thetas: {}".format(pprint.pformat(thetas_dict)))
        LP_graphdef = set_graphdef_parameters_from_dict(self.mutable_graphdef,
                                                        thetas_dict)
        self.mutable_graphdef = LP_graphdef

    @property
    def estimated_rate(self):
        # TODO(mkuchnik): Code is identical to above, but without logging
        # TODO(mkuchnik): Add interleave logic.
        bandwidth_params = self._get_bandwidth_params()
        if (self.step_size is not None
                or self.core_multiplier is not None
                or bandwidth_params is not None):
            if bandwidth_params:
                estimated_rate, thetas_dict = convex_solver.LP_upper_bounds(
                    self.plumber, max_change=self.step_size,
                    core_multiplier=self.core_multiplier,
                    bandwidth_params=None)
            estimated_rate, thetas_dict = convex_solver.LP_upper_bounds(
                self.plumber, max_change=self.step_size,
                core_multiplier=self.core_multiplier,
                bandwidth_params=bandwidth_params)
            if bandwidth_params:
                bw_rate_predictor = bandwidth_utilities.piecewise_linear_predictor_from_params(
                        bandwidth_params)
                source_node = bandwidth_params["source_node"]
                if source_node in thetas_dict:
                    bw_theta = thetas_dict[source_node]
                    bw_rate_hat = bw_rate_predictor(bw_theta)
        else:
            estimated_rate, thetas_dict = self.recommendation.LP_upper_bounds()
        LP_graphdef = set_graphdef_parameters_from_dict(self.mutable_graphdef,
                                                        thetas_dict)
        return estimated_rate

    def _apply_disk_optimizations(self):
        cache_nodes = get_cache_nodes(self.mutable_graphdef)
        self.cache_enabled = bool(cache_nodes)
        if self.machine_info and not self.cache_enabled:
            # TODO(mkuchnik): check if dataset is cached
            estimated_disk_bw = self.machine_info["FILES"][0]["BANDWIDTH"]
            X_disk_bounds = self.recommendation.disk_upper_bounds(
                estimated_disk_bw)
            Disk_Throughput = self.model.disk_throughput()
            Disk_bytes_per_root_element = \
                self.recommendation.disk_bytes_per_root_element()
            if Disk_bytes_per_root_element is None:
                logging.info("Didn't find any source byte I/O. Skipping disk optimization.")
                return False
            disk_util = Disk_Throughput / estimated_disk_bw
            current_rate = self.estimated_rate
            disk_bw_requirements = Disk_bytes_per_root_element * current_rate
            if not disk_bw_requirements:
                logging.info("No disk used")
                return
            required_util = Disk_Throughput / disk_bw_requirements
            logging.debug("Disk throughput: {}".format(Disk_Throughput))
            logging.debug("Disk bounds: {}".format(estimated_disk_bw))
            logging.debug("Disk usage: {}".format(disk_util))
            logging.debug("Disk bytes per root_element: {}".format(Disk_bytes_per_root_element))
            logging.debug("Disk bw requirements: {}".format(disk_bw_requirements))
            logging.debug("Disk required util: {}".format(required_util))
            # Don't do the impossible: stop if hit bw wall
            required_util = Disk_Throughput / min(disk_bw_requirements,
                                                  estimated_disk_bw)
            if disk_bw_requirements > estimated_disk_bw:
                self.disk_bound = X_disk_bounds
            if required_util < 1.0:
                logging.info("NEED TO INCREASE DISK. Utilization {}".format(required_util))
                nodes = self.recommendation.ranked_list_bottleneck_nodes_analysis()
                disk_nodes = [n for n in nodes if n.is_disk_node()]
                assert len(disk_nodes) == 1, "Only 1 disk node supported"
                interleave_node = disk_nodes[0].parent
                assert interleave_node.is_interleave_node()
                surgeon = graphsurgeon.StaticGraph(self.mutable_graphdef)
                surgeon_node = graphdef_util.find_node_by_name(
                    surgeon, interleave_node.name)
                interleave_parallelism = get_node_parallelism(surgeon,
                                                              surgeon_node)
                new_parallelism = interleave_parallelism / required_util
                new_parallelism = math.ceil(new_parallelism)
                logging.info("Found interleave parallelism:"
                      " {}->{}".format(interleave_parallelism, new_parallelism))
                thetas_dict = {interleave_node.name: new_parallelism}
                graphdef = set_graphdef_parameters_from_dict(self.mutable_graphdef,
                                                             thetas_dict)
                self.mutable_graphdef = graphdef
                return True
        return False

    def _apply_prefetch_optimizations(self, aggressive=False):
        # TODO(mkuchnik): Add trace of execution and optimize against it
        roof_stats = self.roofline()
        N_star = roof_stats["N_star"]
        N = roof_stats["N"]
        N_gap = max(N_star - N, 0)
        logging.info("N gap: {} - {} => {}".format(N_star, N, N_gap))
        if aggressive:
            prefetch_amount = self.prefetch_amount_added + math.ceil(N_gap)
        else:
            # No reason to keep adding, probably impossible to close gap
            prefetch_amount = math.ceil(N_gap)
        logging.info("Adding prefetching of {}".format(prefetch_amount))
        if prefetch_amount:
            new_graphdef, is_diff = insert_prefetch_highest(self.model,
                                                   self.mutable_graphdef,
                                                   self.recommendation,
                                                   prefetch_amount,
                                                   return_is_diff=True)
            if is_diff:
                # UNSAFE
                self.unsafe_mutable_graphdef_update(new_graphdef)
            else:
                self.mutable_graphdef = new_graphdef
        self.prefetch_amount_added = prefetch_amount
        return bool(prefetch_amount)

    def get_cache_summary(self):
        random_nodes = self._get_random_nodes()
        node_cols = ["name", "canonical_name", "topo_idx",
                     "size_GB", "cardinality",
                     "expected_core_max_rate",
                     "expected_parallel_max_rate",
                     "observed_rate",
                     "p_busy",
                     "bytes_per_record",
                     "random",
                     #"elements_produced",
                     #"record_ratio"
                     ]
        node_data = []
        remapper_fn = get_safe_topo_remapper_fn(self.mutable_graphdef)
        num_remapper_fn = get_safe_topo_remapper_fn(self.mutable_graphdef, numeric=True)
        for node in self.recommendation.ranked_list_bottleneck_nodes_analysis():
            size = node.expected_dataset_size
            if size and size > 0 and node.name != "dataset":
                size /= 1e9
            node_data.append((node.name, remapper_fn(node.name),
                              num_remapper_fn(node.name),
                              size, node.derived_cardinality,
                              node.expected_per_core_max_rate,
                              node.expected_parallel_max_rate(),
                              node.observed_rate,
                              node.p_busy,
                              node.average_bytes_per_element_produced,
                              node.name in random_nodes,
                              #node.node.state.aggregate_elements_produced,
                              #node.dataset_record_ratio,
                              ))
        summary_df = pd.DataFrame(data=node_data, columns=node_cols)
        summary_df = summary_df.sort_values(by="topo_idx", axis=0)
        summary_df = summary_df.set_index("topo_idx")
        #summary_df["inferred_bytes_per_record"] = (
        #    summary_df["size_GB"] * 1e9 / summary_df["cardinality"])
        return summary_df

    def _topo_sort_nodes(self):
        G = graphdef_util.graphdef_to_networkx(self.mutable_graphdef,
                                               keep_const=False)
        topo_sort = nx.topological_sort(G)
        return topo_sort

    def _get_random_nodes(self, **kwargs):
        random_nodes = nodes_with_random_udf(self.mutable_graphdef, **kwargs)
        return random_nodes

    def _get_cache_candidates(self, filter_random_nodes: bool=True) -> dict:
        candidates = dict()
        FILTER_OP_LIST = set(["_Retval", "ModelDataset",
                              "MaxIntraOpParallelismDataset",
                              "PrivateThreadPoolDataset"])
        analysis_nodes = self.recommendation.ranked_list_bottleneck_nodes_analysis(
                extended=False)

        logging.info("Analysis nodes: {}".format(list(map(lambda x: x.name, analysis_nodes))))
        for node in analysis_nodes:
            size = node.expected_dataset_size
            logging.debug("Cache evaluation for {} has size {}".format(node.name, size))
            if self.materialized_view_cache_optimization_enabled:
                if size and size > 0 and node.op not in FILTER_OP_LIST:
                    candidates[node.name] = size
            else:
                # We only consider interleave nodes
                if size and size > 0 and node.op not in FILTER_OP_LIST and node.is_interleave_node():
                    candidates[node.name] = size
        logging.info("Current size candidates: {}".format(candidates))
        if filter_random_nodes:
            random_nodes, causes = self._get_random_nodes(return_cause=True)
            valid_candidates = set()
            topo_sort = list(self._topo_sort_nodes())
            # NOTE(mkuchnik): A better implementation would use a relation
            # is_parent to invalidate nodes
            logging.info("Traversing nodes: {}".format(topo_sort))
            for node_name in topo_sort:
                if node_name in random_nodes:
                    cause = causes[node_name]["random_function"]
                    logging.info("INVALIDATING random node: {} because of function {}".format(node_name, cause))
                    break
                if node_name in candidates:
                    valid_candidates.add(node_name)
                else:
                    logging.info("Not node in candidates: {}".format(node_name))
            new_candidates = {k: v for k, v in candidates.items() if k in
                              valid_candidates}
            candidates = new_candidates
        logging.info("Current post-pruning candidates: {}".format(candidates))
        return candidates

    def input_hash(self) -> str:
        """Returns a hash value that indicates the input state into Plumber.
        For example, this covers the graph structure of the pipeline.
        The usefulness is to check if the inputs are the same, to allow
        caching, for example."""
        G = self.networkx()
        curr_hash = nx.weisfeiler_lehman_graph_hash(G, node_attr="op")
        return curr_hash

    def apply_extension(self, extension: extensions.Extension):
        graph_object = extensions.GraphObject(plumber=self.plumber,
                graphdef=self.mutable_graphdef)
        new_graph_object, is_same = extension.apply_transaction_if_possible(
                graph_object)
        plumber = new_graph_object.plumber
        graphdef = new_graph_object.graphdef()
        # If we get a plumber change, discard everything.
        # Else, just update graphdef
        if plumber != self.plumber:
            assert not is_same
            logging.info("Extension has changed plumber.")
            # We didn't touch plumber e.g., remove caches
            # TODO(mkuchnik): Consider just forking
            self.plumber = plumber
        elif not is_same:
            logging.info("Extension has changed graphdef.")
            # UNSAFE
            # TODO(mkuchnik): Make this more restricted by having add/remove guarantees
            self.unsafe_mutable_graphdef_update(graphdef)

    def _apply_cache_optimizations(self, add_take_repeat=False, use_machine_info=True):
        """This operation is UNSAFE and adds nodes."""
        if add_take_repeat:
            # TODO(mkuchnik): Fix potentially
            logging.warning("WARNING: add_take_repeat is being added to cache!")
        if self.machine_info and use_machine_info:
            total_free_memory = self.machine_info["MEMORY"]
            logging.info("Using machine info for memory: {}".format(total_free_memory))
        else:
            # NOTE(mkuchnik): We use current memory as it is accurate
            memory_stats = psutil.virtual_memory()
            logging.info("Using current memory stats: {}".format(memory_stats))
            total_free_memory = memory_stats.available
        total_dataset_size_seen = self.model.dataset_working_set_size()
        try:
            total_dataset_size_estimated = \
                self.recommendation.projected_dataset_working_set_size()
        except RuntimeError as ex:
            logging.warning(ex)
            total_dataset_size_estimated = None
        total_dataset_size = total_dataset_size_seen
        if total_dataset_size_estimated and total_dataset_size_seen < total_dataset_size_estimated:
            logging.info("Saw a subset of total files. "
                  "{}/{}GB=({:.2%})".format(
                      total_dataset_size_seen / 1e9,
                      total_dataset_size_estimated / 1e9,
                      total_dataset_size_seen/total_dataset_size_estimated))
            total_dataset_size = total_dataset_size_estimated
        elif total_dataset_size_estimated and total_dataset_size_seen > total_dataset_size_estimated:
            logging.warning("WARNING Saw a superset of total files. "
                  "{}/{}GB=({:.2%})".format(
                      total_dataset_size_seen / 1e9,
                      total_dataset_size_estimated / 1e9,
                      total_dataset_size_seen/total_dataset_size_estimated))
            logging.info(self.model.dataset_file_sizes())
        cache_nodes = get_cache_nodes(self.mutable_graphdef)
        if (total_dataset_size_estimated
                and total_dataset_size and not cache_nodes):
            # No cache nodes
            candidates = self._get_cache_candidates()
            logging.info("Current cache candidates: {}".format(candidates))
            memory_fraction = total_dataset_size / total_free_memory
            if total_free_memory > total_dataset_size:
                logging.info("Caching at source is possible ({:1}GB/{:1}GB={:1%} memory)".format(
                    total_dataset_size / 1e9,
                    total_free_memory / 1e9,
                    memory_fraction))
            else:
                logging.info("Caching at source is not possible ({:1}GB/{:1}GB={:1%} memory)".format(
                    total_dataset_size / 1e9,
                    total_free_memory / 1e9,
                    memory_fraction))
            max_memory = total_free_memory * FRACTION_CACHEABLE_MEMORY
            new_graphdef = insert_cache_highest(self.model,
                                                self.mutable_graphdef,
                                                self.recommendation,
                                                candidates,
                                                max_memory,
                                                add_take_repeat=add_take_repeat)
            # NOTE(mkuchnik): UNSAFE
            self.unsafe_mutable_graphdef_update(new_graphdef)
            cache_nodes = get_cache_nodes(self.mutable_graphdef)
            if cache_nodes:
                self.cache_enabled = True
        elif cache_nodes:
            # Cache nodes
            self.cache_enabled = True
            # NOTE(mkuchnik): Check if cache already exists
            logging.info("CACHE DETECTED... SKIPPING")
        elif (total_dataset_size and not cache_nodes):
            # Estimate not available, but dataset is known
            logging.info("Defaulting to total dataset size: "
                  "{}".format(total_dataset_size/1e9))
            candidates = self._get_cache_candidates()
            logging.info("Current cache candidates: {}".format(candidates))
            memory_fraction = total_dataset_size / total_free_memory
            if total_free_memory > total_dataset_size:
                logging.info("Caching at source is possible ({:1}GB/{:1}GB={:1%} memory)".format(
                    total_dataset_size / 1e9,
                    total_free_memory / 1e9,
                    memory_fraction))
            else:
                logging.info("Caching at source is not possible ({:1}GB/{:1}GB={:1%} memory)".format(
                    total_dataset_size / 1e9,
                    total_free_memory / 1e9,
                    memory_fraction))
            max_memory = total_free_memory * 0.9
            new_graphdef = insert_cache_highest(self.model,
                                                self.mutable_graphdef,
                                                self.recommendation,
                                                candidates,
                                                max_memory)
            # NOTE(mkuchnik): UNSAFE
            self.unsafe_mutable_graphdef_update(new_graphdef)
            cache_nodes = get_cache_nodes(self.mutable_graphdef)
            if cache_nodes:
                self.cache_enabled = True
        else:
            logging.info("No dataset size detected ({}GB). Can't"
            " cache properly. Is the dataset truncated?".format(
                total_dataset_size / 1e9))

class CostBasedDataPipelineOptimizer(DataPipelineOptimizer):
    def __init__(self, plumber, min_rate, calibrate_system=None,
                 step_size=None):
        """
        Cost-based optimizer

        If min_rate is specified, the optimization is performed such that the
        rate is satisfied, but the cost is minimized.
        """
        self.plumber = plumber
        self.step_size = step_size
        self.instance_info = machine_info.GCPN1OnDemand()
        self.disk_info = machine_info.GCPLocalSSD()
        self.core_multiplier = None
        self.machine_info = None  # NOTE(mkuchnik): No reason to use currently
        if calibrate_system:
            self._calibrate_system()
            logging.info("machine_info", self.machine_info)
        self.min_rate = min_rate
        self.prefetch_amount_added = 0
        self.rates = []

    def query_configuration(self, min_rate=None, resource_costs=None):
        """Cost in $ per hour.

        min_rate: The minimum rate in minibatches/second
        resource_costs: CPU (vCPU/hour), memory (GB/hour), and disk costs
        (100MBps/hour)
        """
        if min_rate is None:
            min_rate = self.min_rate
        if resource_costs is None:
            resource_costs = {"CPU": self.instance_info.price_per_vCPU_hour(),
                             "memory": self.instance_info.price_per_GB_hour(),
                             "disk": self.disk_info.price_per_MBps_hour() * 100,
                              }
        cache_candidates = self._get_cache_candidates()
        logging.info("Current cache candidates: {}".format(candidates))
        for k, v in cache_candidates.items():
            cache_candidates[k] = v / 1e9 # To GB
        X_disk_bounds = self.recommendation.disk_upper_bounds(100e6)
        disk_usage = {"100MB": X_disk_bounds}
        # NOTE(mkuchnik): Assume mostly linear
        topo_sort = self._topo_sort_nodes()
        cost, thetas_dict, cache_dict = convex_solver.LP_resource_lower_bounds(
            self.plumber,
            min_rate,
            disk_usage=disk_usage,
            resource_costs=resource_costs,
            cache_candidates=cache_candidates,
            topological_sort=topo_sort,
        )
        return cost, thetas_dict, cache_dict


    def _apply_parallelism_optimizations(self):
        # TODO(mkuchnik): Add interleave logic.
        cost, thetas_dict, cache_dict = self.query_configuration()
        logging.info("min_rate: {}".format(self.min_rate))
        logging.info("total_cost=${}/hour".format(cost))
        logging.info("thetas: {}".format(thetas_dict))
        LP_graphdef = set_graphdef_parameters_from_dict(self.graphdef,
                                                        thetas_dict)
        self.graphdef = LP_graphdef
