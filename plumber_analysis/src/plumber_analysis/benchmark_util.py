"""
Utilities to brute force benchmark performance of parameters of tf.data Datasets.
"""

import copy
import networkx as nx
import random

from plumber_analysis import gen_util, graphdef_util

def create_benchmark_node_dataset(surgeon, node_name: str, take_amount: int):
    """Creates a dataset to test the
    maximum throughput of a node by inserting caches and truncating
    the dataset at that node."""
    surgeon_node = graphdef_util.find_node_by_name(surgeon, node_name)
    num_input_dataset = sum([1 for i in surgeon_node.input if "Dataset" in i])
    assert num_input_dataset == 1
    node_input = graphdef_util.find_node_by_name(surgeon, surgeon_node.input[0])
    surgeon = graphdef_util.add_take_and_cache_node_after_node(surgeon,
                                            node_input,
                                            take_amount)
    surgeon = graphdef_util.add_retval_after_node(surgeon, surgeon_node)
    return surgeon

def benchmark_node_dataset(surgeon, node_name: str, dataset_options: dict,
                           bench_options: dict,
                           take_amount: int = 500,
                           prefetch_amount: int = 300):
    surgeon = copy.deepcopy(surgeon)
    graphdef_util.clear_graph()
    surgeon = create_benchmark_node_dataset(surgeon, node_name, take_amount)
    graphdef = surgeon.as_graph_def()
    element_spec = graphdef_util.element_spec_from_graph(surgeon)
    ds = graphdef_util.instantiate_pipeline(graphdef, element_spec,
                                            dataset_options)
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
    G = graphdef_util.graphdef_to_networkx(surgeon.as_graph_def())
    topo_sort = nx.topological_sort(G)
    topo_sort_dataset = filter(graphdef_util.is_dataset_node, topo_sort)
    remapper = graphdef_util.remap_dataset_names(topo_sort_dataset)
    if parallelism_grid is None:
        parallelism_grid = [2**i for i in range(1, 4)]
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
        # TODO(mkuchnik): Start Remove
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
        surgeon_node = graphdef_util.find_node_by_name(_surgeon, node.name)
        try:
            parallelism_node = graphdef_util.parallelism_parameter_name(surgeon_node)
        except RuntimeError as ex:
            print(ex)
            benchmark_summary["parallelism"] = None
            continue
        # TODO(mkuchnik): need to descend into nested nodes.
        parallelism_surgeon_node = graphdef_util.find_node_by_name(
            _surgeon, parallelism_node)
        parallelism_tensor = parallelism_surgeon_node.attr["value"].tensor
        assert len(parallelism_tensor.int64_val) == 1
        parallelism_param = parallelism_tensor.int64_val[0]
        benchmark_summary["parallelism"] = int(parallelism_param)
        new_parallelism_surgeon_node = graphdef_util.fork_node(
            _surgeon, parallelism_surgeon_node)
        i = graphdef_util.parallelism_parameter_index(surgeon_node)
        node_input = surgeon_node.input[i]
        assert(node_input == parallelism_surgeon_node.name)
        surgeon_node.input[i] = new_parallelism_surgeon_node.name
        parallelism_tensor = new_parallelism_surgeon_node.attr["value"].tensor
        if surgeon_node.op == "ParallelInterleaveDatasetV4":
            # Adjust cycle length to match parallelism
            i = graphdef_util.cycle_length_parameter_index(surgeon_node)
            node_input = surgeon_node.input[i]
            cycle_surgeon_node = [k for k in _surgeon if
                                  k.name == node_input]
            assert len(cycle_surgeon_node) == 1
            cycle_surgeon_node = cycle_surgeon_node[0]
            new_cycle_surgeon_node = graphdef_util.fork_node(
                _surgeon, cycle_surgeon_node)
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

