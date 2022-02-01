"""
Linear programming formulations to optimize Plumber rates and costs.
"""

import time
import logging

import numpy as np
import cvxpy as cp
import tensorflow as tf

def LP_upper_bounds(plumber, **kwargs):
    """Calculates the upper bounds of plumber using a linear program
    The optimal throughput can be calculated according to the program:

    maximize T(theta) s.t., sum_i theta_i <= 1.0
    T(theta) = min[f_1(theta_1), ..., f_N(theta_N)]
    """
    model = plumber.model()
    return LP_upper_bounds_inner(model, **kwargs)

def LP_resource_lower_bounds(plumber, min_rate, **kwargs):
    """Calculates the resource required of plumber using a linear program
    to hit min_rate
    """
    model = plumber.model()
    return LP_resource_bounds_inner(model, min_rate=min_rate, **kwargs)

def LP_upper_bounds_inner(model, use_existing_usage=False,
                          debug=False, max_change=None, core_multiplier=None,
                          consider_parallelizable_nodes=False, bandwidth_params=None):
    recommendation = model.recommendation()
    ranked_nodes = \
    recommendation.ranked_list_bottleneck_nodes_analysis()
    num_cores = recommendation._analysis.global_state.machine_info.num_cores
    if core_multiplier:
        num_cores *= core_multiplier
    CPU_Util_clock = model.CPU_Util(calculation_mode="CPU_clock")
    names = [n.name for n in ranked_nodes]
    rates = [n.expected_per_core_max_rate for n in ranked_nodes]
    if consider_parallelizable_nodes:
        is_parallel = [n.is_parallel_node() or n.is_parallelizable_node()
                       for n in ranked_nodes]
    else:
        is_parallel = [n.is_parallel_node() for n in ranked_nodes]
    parallelism = [n.parallelism for n in ranked_nodes]
    num_cores_used = [n.num_cores_used for n in ranked_nodes]
    if debug:
        print("rates:\n{}".format(rates))
        print("is_parallel:\n{}".format(is_parallel))
        print("parallelism:\n{}".format(parallelism))
    N = len(rates)
    if N <= 0:
        print("Number of parameters <= 0: {}".format(N))
    _theta_min = np.zeros(N)
    if use_existing_usage:
        for i in range(len(num_cores_used)):
            c = num_cores_used[i]
            if not is_parallel[i] and c > 1.:
                print("WARNING: cores used greater than 1 for sequential node: "
                      "{}".format(names[i]))
                c = min(c, 1)
            _theta_min[i] = c
        modeling_cores_used = np.sum(_theta_min)
        modeling_bias = CPU_Util_clock * num_cores - modeling_cores_used
        if debug:
            print("modeling_bias: {}".format(modeling_bias))
            if modeling_bias < 0.:
                print("WARNING: modeling bias < 0")
                modeling_bias = 0.
    else:
        modeling_bias = 0.
    num_cores_avail = num_cores - modeling_bias
    if num_cores_avail <= 0:
        print("num_cores_avail <= 0: {}".format(num_cores_avail))
        num_cores_avail = 1e-10
    total_theta_min = np.sum(_theta_min)
    if np.sum(_theta_min) > num_cores_avail:
        print("The number of cores used {} is greater than available "
              "{}.".format(total_theta_min,  num_cores_avail))
        theta_min_ratio = total_theta_min / num_cores_avail
        _theta_min /= (theta_min_ratio + 1e-3)
    theta_min = cp.Parameter(pos=True, shape=(N,), value=_theta_min,
                             name='theta_min')
    _theta_max = num_cores * np.ones(N)
    for i, is_par in enumerate(is_parallel):
        if not is_par:
            _theta_max[i] = 1
    theta_max = cp.Parameter(pos=True, shape=(N,), value=_theta_max,
                             name='theta_max')
    theta = cp.Variable(N)
    expression = cp.multiply(rates, theta)
    constraints = [
        cp.sum(theta) <= num_cores_avail,
        theta <= theta_max,
        theta >= theta_min,
    ]
    if max_change:
        total_cores_used = np.sum(num_cores_used)
        core_upper_bound = total_cores_used + max_change
        constraint = cp.sum(theta) <= core_upper_bound
        print("keepig theta under {}".format(core_upper_bound))
        constraints.append(constraint)
    if bandwidth_params:
        source_node = bandwidth_params["source_node"]
        logging.info("Starting bw optimization with {}".format(source_node))
        bandwidth_node_i = [i for i, name in enumerate(names) if name == source_node]
        if len(bandwidth_node_i) != 1:
            logging.error("Expected 1 bw node matching {}, found {}.\n{}".format(
                        source_node, len(bandwidth_node_i), names))
            # Give up, probably cached
            # TODO(mkuchnik): Check for caching
            joined_expression = expression
        else:
            bandwidth_node_i = bandwidth_node_i[0]
            # This is likely 128
            bandwidth_ratio = ranked_nodes[bandwidth_node_i].element_ratio
            logging.info("BW element ratio: {}".format(bandwidth_ratio))
            # These are in elements/second, convert to batches/second
            m1 = bandwidth_params["m1"] / bandwidth_ratio
            b1 = bandwidth_params["b1"] / bandwidth_ratio
            m2 = bandwidth_params["m2"] / bandwidth_ratio
            b2 = bandwidth_params["b2"] / bandwidth_ratio
            bandwidth_theta = theta[bandwidth_node_i]
            # TODO(mkuchnik): Look at resource cost implementation below
            bw_bound = m1 * bandwidth_theta + b1
            upper_bw_bound = m2 * bandwidth_theta + b2
            joined_expression = cp.hstack([expression, bw_bound, upper_bw_bound])
    else:
        joined_expression = expression

    objective_fn = cp.min(expression)
    problem = cp.Problem(cp.Maximize(objective_fn), constraints)
    t1 = time.time()
    problem.solve()
    t2 = time.time()
    if debug:
        print("Solve took {} seconds".format(t2 - t1))
        print("theta=\n{}".format(theta.value))
    if problem.status in ["infeasible", "unbounded"]:
        raise RuntimeError("LP problem is "
                           "{}\n{}\ntheta_min:{} (sum: {})\ntheta_max:{} "
                           "(sum: {})\nnum_cores:{}"
                           .format(problem.status, problem, _theta_min,
                                   np.sum(_theta_min),
                                   _theta_max, np.sum(_theta_max), num_cores))
    max_throughput = problem.value
    theta_hat = theta.value
    num_cores_allocated = theta_hat
    params_dict = {name: theta_i for name, theta_i in zip(names, theta_hat)}
    if debug:
        print("num_cores=\n{}".format(num_cores_allocated))
        print("num_cores=\n{}".format(np.round(num_cores_allocated)))
        print("throughput={}".format(max_throughput))
        print(params_dict)
    return max_throughput, params_dict

def LP_resource_bounds_inner(model, min_rate, use_existing_usage=False,
                             debug=False,
                             cache_candidates=None,
                             disk_usage=None,
                             resource_costs=None,
                             topological_sort=None):
    print("cache_candidates", cache_candidates)
    print("resource_costs", resource_costs)
    print("disk_usage", disk_usage)
    if resource_costs is None:
        resource_costs = {
            "CPU": 1.0,
            "memory": 1.0,
            "disk": 1.0, # currently for 100 MBps
        }
    disk_rate_100mb = disk_usage["100MB"]
    recommendation = model.recommendation()
    ranked_nodes = recommendation.ranked_list_bottleneck_nodes_analysis()
    ranked_nodes_lookup = dict()
    for x in ranked_nodes:
        ranked_nodes_lookup[x.name] = x
    topo_ranked_nodes = [ranked_nodes_lookup[x] for x in topological_sort
                         if x in ranked_nodes_lookup]
    assert len(topo_ranked_nodes) == len(ranked_nodes)
    ranked_nodes = topo_ranked_nodes
    num_cores = recommendation._analysis.global_state.machine_info.num_cores
    CPU_Util_clock = model.CPU_Util(calculation_mode="CPU_clock")
    names = [n.name for n in ranked_nodes]
    print("ranked_nodes", names)
    rates = [n.expected_per_core_max_rate for n in ranked_nodes]
    is_parallel = [n.is_parallel_node() for n in ranked_nodes]
    parallelism = [n.parallelism for n in ranked_nodes]
    num_cores_used = [n.num_cores_used for n in ranked_nodes]
    if debug:
        print("rates:\n{}".format(rates))
        print("is_parallel:\n{}".format(is_parallel))
        print("parallelism:\n{}".format(parallelism))
    N = len(rates)
    if N <= 0:
        print("Number of parameters <= 0: {}".format(N))
    _theta_min = np.zeros(N)
    if use_existing_usage:
        for i in range(len(num_cores_used)):
            c = num_cores_used[i]
            if not is_parallel[i] and c > 1.:
                print("WARNING: cores used greater than 1 for sequential node: "
                      "{}".format(names[i]))
                c = min(c, 1)
            _theta_min[i] = c
        modeling_cores_used = np.sum(_theta_min)
        modeling_bias = CPU_Util_clock * num_cores - modeling_cores_used
        if debug:
            print("modeling_bias: {}".format(modeling_bias))
            if modeling_bias < 0.:
                print("WARNING: modeling bias < 0")
                modeling_bias = 0.

    else:
        modeling_bias = 0.
    total_theta_min = np.sum(_theta_min)
    theta_min = cp.Parameter(pos=True, shape=(N,), value=_theta_min,
                             name='theta_min')
    _theta_max = num_cores * np.ones(N)
    for i, is_par in enumerate(is_parallel):
        if not is_par:
            _theta_max[i] = 1
    theta_max = cp.Parameter(pos=True, shape=(N,), value=_theta_max,
                             name='theta_max')
    theta_cpu = cp.Variable(N)
    theta_disk_bw = cp.Variable(1)
    num_candidates = len(cache_candidates)
    decision_cache = cp.Variable(num_candidates, boolean=True)
    cache_mask_scale = 10**6
    cache_masks = []
    def find_index_of_candidate(c: str):
        return names.index(c)
    cpu_expression = cp.multiply(rates, theta_cpu)
    disk_expression = cp.multiply(disk_rate_100mb, theta_disk_bw)
    # Disk -> TFRecord -> Interleave -> other ops -> Retval
    joined_expression = cp.hstack([disk_expression, cpu_expression])
    cache_costs = []
    decision_inverse_index = []
    for i, c in enumerate(cache_candidates):
        decision_inverse_index.append(c)
        idx = find_index_of_candidate(c)
        _cache_mask = np.zeros(N + 1)
        _cache_mask[:idx+2] = cache_mask_scale
        cache_mask = cp.Parameter(pos=True, shape=(len(_cache_mask),),
                                  value=_cache_mask,
                                  name="cache_mask_{}".format(c))
        cache_expression = decision_cache[i] * cache_mask
        joined_expression += cache_expression
        cost = cache_candidates[c] * decision_cache[i]
        cache_costs.append(cost)
    #agg_expression = cp.hstack([disk_expression, cpu_expression])
    agg_rate = cp.min(joined_expression)
    #agg_rate = cp.min(agg_expression)
    constraints = [
        theta_cpu <= theta_max,
        theta_cpu >= theta_min,
        agg_rate >= min_rate,
        theta_disk_bw >= 0,
        cp.sum(decision_cache) <= 1,
    ]
    cpu_cost = cp.sum(theta_cpu * resource_costs["CPU"])
    disk_cost = theta_disk_bw * resource_costs["disk"]
    memory_cost = cp.sum(cache_costs) * resource_costs["memory"]
    objective_fn = cpu_cost + disk_cost + memory_cost
    problem = cp.Problem(cp.Minimize(objective_fn), constraints)
    t1 = time.time()
    problem.solve(verbose=False)
    t2 = time.time()
    if debug:
        print("Solve took {} seconds".format(t2 - t1))
        print("theta=\n{}".format(theta_cpu.value))
    if problem.status in ["infeasible", "unbounded"]:
        raise RuntimeError("LP problem is "
                           "{}\n{}\ntheta_min:{} (sum: {})\ntheta_max:{} "
                           "(sum: {})\nnum_cores:{}"
                           .format(problem.status, problem, _theta_min,
                                   np.sum(_theta_min),
                                   _theta_max, np.sum(_theta_max), num_cores))
    total_cost = problem.value
    theta_hat = theta_cpu.value
    num_cores_allocated = theta_hat
    num_disk_bw = theta_disk_bw.value
    print("BW required: {}".format(num_disk_bw * 100e6))
    for i, x in enumerate(decision_cache.value):
        if x:
            cache_point = decision_inverse_index[i]
            print("Caching at {} ({}GB)".format(cache_point,
                                                cache_candidates[cache_point]))
    params_dict = {name: theta_i for name, theta_i in zip(names, theta_hat)}
    if debug:
        print("num_cores=\n{}".format(num_cores_allocated))
        print("num_cores=\n{}".format(np.round(num_cores_allocated)))
        print(params_dict)
    if not cache_candidates:
        return total_cost, params_dict
    else:
        cache_dict = dict()
        cache_dict["disk_bw"] = num_disk_bw * 100e6
        return total_cost, params_dict, cache_dict

if __name__ == "__main__":
    filename = "stats.pb"
    plumber = tf.data.experimental.analysis.PlumberPerformanceModel(filename)
    LP_upper_bounds(plumber)
    LP_upper_bounds(plumber, use_existing_usage=True)
