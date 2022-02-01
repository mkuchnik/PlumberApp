"""
A front-end for plumber. Various annotations are stored here, which allow for Plumber
to work (more or less) as a single-line of code.

It is preferrable to set configuration at this stage, as it is expected to be the caller
into Plumber functionality and has the most up-to-date view into configurations.
"""
import functools
import logging
import time

import tensorflow as tf
from plumber_analysis import pipeline_optimizer_wrapper, config

# TODO(mkuchnik): Disable plumber logging
# https://stackoverflow.com/questions/35325042/python-logging-disable-logging-from-imported-modules

# TODO(mkuchnik): Add decorators to
# 1) check that triaining is enabled and 2) check that element_spec is identical
def condition(pre_condition=None, post_condition=None):
    """
    A pre or pos-condition
    https://stackoverflow.com/questions/12151182/python-precondition-postcondition-for-member-function-how
    """
    def decorator(func):
        @functools.wraps(func) # presever name, docstring, etc
        def wrapper(*args, **kwargs): #NOTE: no self
            if pre_condition is not None:
               assert pre_condition(*args, **kwargs)
            retval = func(*args, **kwargs) # call original function or method
            if post_condition is not None:
               assert post_condition(retval)
            return retval
        return wrapper
    return decorator

def pre_condition(check):
    return condition(pre_condition=check)

def post_condition(check):
    return condition(post_condition=check)

def _check_environment_for_errors():
    if not tf.executing_eagerly():
        logging.error("Eager execution not enabled. TF1 API support isn't well-tested.")

def _maybe_patch_element_spec(new_dataset, dataset):
    """Checks that new_dataset follows the element_spec. If not, attempts to fix it."""
    if isinstance(dataset, dict) and isinstance(new_dataset, dict):
        logging.warning("Don't know how to patch element spec of dicts {} with {}".format(
            new_dataset, dataset))
        return new_dataset
    elif isinstance(dataset, dict) or isinstance(new_dataset, dict):
        raise ValueError("Recieved one dict and one not.")
    curr_element_spec = dataset.element_spec
    new_dataset_element_spec = new_dataset.element_spec
    # NOTE(mkuchnik): The element spec we get is different from the one that
    # is provided by the python runtime due to dict unpacking
    if curr_element_spec != new_dataset_element_spec:
        logging.info("Element spec has changed from:\n{}\nto:\n{}".format(
                    curr_element_spec, new_dataset_element_spec))
        logging.debug("dataset: {}".format(type(dataset)))
        logging.debug("new_dataset: {}".format(type(new_dataset)))
        logging.debug("dataset: {}".format(repr(dataset)))
        logging.debug("new_dataset: {}".format(repr(new_dataset)))
        dataset_type = type(dataset)

        restructured_dataset_type = tf.data.experimental.analysis.RestructuredDataset
        restructured_dataset = restructured_dataset_type(new_dataset, curr_element_spec)
        try:
            wrapped_dataset = dataset_type(restructured_dataset)
        except TypeError as ex:
            logging.error("Cannot wrap {} over {}: {}".format(
                dataset_type, type(restructured_dataset_type), ex))
            wrapped_dataset = restructured_dataset
        new_dataset = wrapped_dataset
        new_dataset_element_spec = new_dataset.element_spec
        if curr_element_spec != new_dataset_element_spec:
            logging.warning("dataset: {}".format(repr(dataset)))
            logging.warning("new_dataset: {}".format(repr(new_dataset)))
            raise RuntimeError("Element spec has changed from:\n{}\nto:\n{}".format(
                               curr_element_spec, new_dataset_element_spec))
    return new_dataset

def optimize_pipeline(kwargs_precondition_f=None, use_parameter_cache=False):
    """
    Wraps and optimizes a loader function. If kwargs_precondition_f is True, runs optimizer.

    Usage (from SSD pipeline):

    @plumber_analysis.annotations.optimize_pipeline(kwargs_precondition_f=lambda x: bool(x["is_training"]))
    def ssd_input_pipeline(params, file_pattern, is_training=False,
                           use_fake_data=False,
                           transpose_input=False, distributed_eval=False,
                           count=-1, host_batch_size=-1):
        # ... build closure for dataset with dataset_fn
        return dataset_fn
    """
    def inner_decorator(loader_fn):
        # TODO(mkuchnik): Wrap with LRU cache to avoid re-computing if the arguments are the same
        @functools.wraps(loader_fn)
        def wrapper(*args, **kwargs):
            _check_environment_for_errors()
            start_time = time.perf_counter()
            dataset = loader_fn(*args, **kwargs)
            logging.info("OPTIMIZE PIPELINE ENTER WITH DATASET: {} ({})".format(
                dataset, id(dataset)))
            if pipeline_optimizer_wrapper.plumber_fake_pipeline():
                new_dataset = pipeline_optimizer_wrapper.get_fake_pipeline(dataset)
                logging.warning("USING FAKE DATASET")
            elif kwargs_precondition_f:
                if kwargs_precondition_f(kwargs):
                    new_dataset = pipeline_optimizer_wrapper.get_optimized_pipeline(
                        dataset, override_presets=True,
                        return_test_dataset=False, return_rate=False,
                        use_parameter_cache=use_parameter_cache)
                else:
                    logging.info("Precondition failed! Passing through dataset.")
                    new_dataset = dataset
            else:
                new_dataset = pipeline_optimizer_wrapper.get_optimized_pipeline(
                        dataset, override_presets=True,
                    return_test_dataset=False, return_rate=False,
                    use_parameter_cache=use_parameter_cache)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            logging.info("OPTIMIZE PIPELINE END: {} seconds".format(elapsed_time))

            new_dataset = _maybe_patch_element_spec(new_dataset, dataset)
            logging.info("OPTIMIZE PIPELINE EXIT WITH DATASET: {} ({})".format(
                new_dataset, id(new_dataset)))

            return new_dataset
        return wrapper
    return inner_decorator

def maybe_optimize_pipeline(kwargs_precondition_f=None, use_parameter_cache=False):
    """
    Wraps and optimizes a loader function. If kwargs_precondition_f is True, runs optimizer.
    Doesn't optimize the pipeline unless instructed by environment variables.
    Use like optimize_pipeline, but allows for optimization control through environment rather than kwargs.

    Usage (from SSD pipeline):

    @plumber_analysis.annotations.maybe_optimize_pipeline(kwargs_precondition_f=lambda x: bool(x["is_training"]))
    def ssd_input_pipeline(params, file_pattern, is_training=False,
                           use_fake_data=False,
                           transpose_input=False, distributed_eval=False,
                           count=-1, host_batch_size=-1):
        # ... build closure for dataset with dataset_fn
        return dataset_fn
    """
    def inner_decorator(loader_fn):
        @functools.wraps(loader_fn)
        def wrapper(*args, **kwargs):
            _check_environment_for_errors()
            start_time = time.perf_counter()
            dataset = loader_fn(*args, **kwargs)
            logging.info("OPTIMIZE PIPELINE ENTER WITH DATASET: {} ({})".format(
                dataset, id(dataset)))
            if pipeline_optimizer_wrapper.plumber_fake_pipeline():
                new_dataset = pipeline_optimizer_wrapper.get_fake_pipeline(dataset)
                logging.warning("USING FAKE DATASET")
            elif pipeline_optimizer_wrapper.plumber_optimize_pipeline():
                if kwargs_precondition_f:
                    if kwargs_precondition_f(kwargs):
                        new_dataset = pipeline_optimizer_wrapper.get_optimized_pipeline(
                            dataset, override_presets=True,
                            return_test_dataset=False, return_rate=False,
                            use_parameter_cache=use_parameter_cache)
                    else:
                        logging.info("Precondition failed! Passing through dataset.")
                        new_dataset = dataset
                else:
                    new_dataset = pipeline_optimizer_wrapper.get_optimized_pipeline(
                        dataset, override_presets=True,
                        return_test_dataset=False, return_rate=False,
                        use_parameter_cache=use_parameter_cache)
            else:
                logging.info("Optimization not enabled! Passing through dataset.")
                new_dataset = dataset
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            logging.info("OPTIMIZE PIPELINE END: {} seconds".format(elapsed_time))

            new_dataset = _maybe_patch_element_spec(new_dataset, dataset)
            logging.info("OPTIMIZE PIPELINE EXIT WITH DATASET: {} ({})".format(
                new_dataset, id(new_dataset)))

            return new_dataset
        return wrapper
    return inner_decorator


def expand_grid_combinations(arg_grid_dict):
    """All possible value combinations with keys"""
    # https://stackoverflow.com/questions/38721847/how-to-generate-all-combination-from-values-in-dict-of-lists-in-python
    import itertools
    keys, values = zip(*arg_grid_dict.items())
    values = list(map(lambda v: [v] if not isinstance(v, list) else v, values))
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return permutations_dicts


def maybe_find_best_pipeline(kwargs_precondition_f=None, optimization_arg_grid=None):
    """
    Wraps and optimizes a loader function. If kwargs_precondition_f is True, runs optimizer.
    Finds the best configuration across a grid of configurations.
    Doesn't optimize the pipeline unless instructed by environment variables

    NOTE: this currently uses kwargs only.

    Usage (from ResNet) to optimize on train only and a flag, FLAGS.optimize_plumber_pipeline. The grid is over
    caching being on or off:

    @plumber_analysis.annotations.maybe_find_best_pipeline(
            kwargs_precondition_f=lambda x: bool(x["train"]) and FLAGS.optimize_plumber_pipeline,
            optimization_arg_grid = {"cache_uncompressed": [False, True]},
            )
    def _load_split(batch_size, train, dtype, image_format, space_to_depth,
                   cache_uncompressed, image_size=IMAGE_SIZE, reshape_to_r1=False,
                   shuffle_size=16384):
        # ... build closure for dataset with dataset_fn
        return dataset_fn
    """
    def inner_decorator(loader_fn):
        @functools.wraps(loader_fn)
        def wrapper(*args, **kwargs):
            _check_environment_for_errors()
            start_time = time.perf_counter()
            dataset = loader_fn(*args, **kwargs)
            logging.info("OPTIMIZE PIPELINE ENTER WITH DATASET: {} ({})".format(
                dataset, id(dataset)))
            if pipeline_optimizer_wrapper.plumber_fake_pipeline():
                new_dataset = pipeline_optimizer_wrapper.get_fake_pipeline(dataset)
                logging.warning("USING FAKE DATASET")
            elif pipeline_optimizer_wrapper.plumber_find_best_pipeline():
                if kwargs_precondition_f:
                    if kwargs_precondition_f(kwargs):
                        arg_grid = dict(kwargs)
                        for k, v in optimization_arg_grid.items():
                            assert k in arg_grid
                            arg_grid[k] = v
                        arg_grid = expand_grid_combinations(arg_grid)
                        for args in arg_grid:
                            logging.info("args {}".format(args))
                        datasets = [loader_fn(**args) for args in arg_grid]
                        logging.info("Finding optimized pipeline over {} pipelines".format(len(datasets)))
                        new_dataset = pipeline_optimizer_wrapper.get_best_optimized_pipeline(datasets)
                    else:
                        logging.info("Precondition failed! Passing through dataset.")
                        new_dataset = dataset
                else:
                    arg_grid = dict(kwargs)
                    for k, v in optimization_arg_grid.items():
                        assert k in arg_grid
                        arg_grid[k] = v
                    arg_grid = expand_grid_combinations(arg_grid)
                    for args in arg_grid:
                        logging.info("args {}".format(args))
                    datasets = [loader_fn(**args) for args in arg_grid]
                    logging.info("Finding optimized pipeline over {} pipelines".format(len(datasets)))
                    new_dataset = pipeline_optimizer_wrapper.get_best_optimized_pipeline(datasets)
            else:
                logging.info("Optimization not enabled! Passing through dataset.")
                new_dataset = dataset
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            logging.info("OPTIMIZE PIPELINE END: {} seconds".format(elapsed_time))

            new_dataset = _maybe_patch_element_spec(new_dataset, dataset)
            logging.info("OPTIMIZE PIPELINE EXIT WITH DATASET: {} ({})".format(
                new_dataset, id(new_dataset)))

            return new_dataset
        return wrapper
    return inner_decorator

def trace_pipeline(kwargs_precondition_f=None):
    """
    Just tracing. Adds a stats.pb Plumber dump to the pipeline.

    Usage (from SSD pipeline):

    @plumber_analysis.annotations.trace_pipeline(kwargs_precondition_f=lambda x: bool(x["is_training"]))
    def ssd_input_pipeline(params, file_pattern, is_training=False,
                           use_fake_data=False,
                           transpose_input=False, distributed_eval=False,
                           count=-1, host_batch_size=-1):
        # ... build closure for dataset with dataset_fn
        return dataset_fn
    """
    def inner_decorator(loader_fn):
        @functools.wraps(loader_fn)
        def wrapper(*args, **kwargs):
            _check_environment_for_errors()
            start_time = time.perf_counter()
            dataset = loader_fn(*args, **kwargs)
            logging.info("TRACING PIPELINE ENTER WITH DATASET: {} ({})".format(
                dataset, id(dataset)))
            if kwargs_precondition_f:
                if kwargs_precondition_f(kwargs):
                    new_dataset = pipeline_optimizer_wrapper.apply_tracing(
                            dataset)
                else:
                    logging.info("Precondition failed! Passing through dataset.")
                    new_dataset = dataset
            else:
                new_dataset = pipeline_optimizer_wrapper.apply_tracing(
                        dataset)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            logging.info("TRACING PIPELINE END: {} seconds".format(elapsed_time))

            return new_dataset
        return wrapper
    return inner_decorator
