import math

import tensorflow as tf
try:
    import tensorflow_probability as tfp
except ImportError as ex:
    tfp = None

@tf.function
def apply_function_to_args(f, *args):
    return [f(x) for x in args]

def paper_echoing(dataset, echo_factor: int):
    """As described in paper"""
    e = echo_factor
    echo_fn = lambda tt: tf.data.Dataset.from_tensors(tt).repeat(e)
    dataset = dataset.flat_map(lambda *t: echo_fn(t))
    return dataset

def paper_echoing_continuous(dataset, echo_factor: float):
    """Smoothed out echoing by using statistical mean"""
    e_integral = math.floor(echo_factor)
    e_diff = echo_factor - e_integral
    if e_diff > 0:
        noise = tfp.distributions.Bernoulli(
                    probs=e_diff, dtype=tf.int64, validate_args=False,
                    name="echo_bernoulli_noise"
        )
        e = e_integral + noise.sample()
    else:
        e = e_integral
    echo_fn = lambda tt: tf.data.Dataset.from_tensors(tt).repeat(e)
    dataset = dataset.flat_map(lambda *t: echo_fn(t))
    return dataset

def paper_echoing_prefetch(dataset, echo_factor: int):
    """Add some prefetching to paper dataset. May not do anything"""
    e = echo_factor
    echo_fn = lambda tt: tf.data.Dataset.from_tensors(tt).repeat(e)
    dataset = dataset.flat_map(lambda *t: echo_fn(t)).prefetch(e)
    return dataset

def unbatch_echoing(dataset, echo_factor: int, parallelism: int=None):
    """Using unbatch mechanism"""
    e = echo_factor
    echo_fn = lambda tt: tf.repeat(tt, e)
    dataset = (dataset
            .map(lambda *t: apply_function_to_args(echo_fn, *t),
                 num_parallel_calls=parallelism)
            .unbatch().prefetch(e))
    return dataset

def apply_dataset_echoing(dataset, echo_factor: int, shuffle_buffer: int=0):
    """Applies dataset echoing with echo_factor echoes"""
    e = echo_factor
    #parallelism = 64
    #dataset = unbatch_echoing(dataset, echo_factor, parallelism)
    dataset = paper_echoing(dataset, echo_factor)
    #dataset = paper_echoing_continuous(dataset, echo_factor)
    if shuffle_buffer:
        dataset = dataset.shuffle(shuffle_buffer)
    return dataset
