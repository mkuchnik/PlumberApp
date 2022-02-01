from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'benchmark_num_elements', default=1000,
    help=('The number of elements to use for the benchmark'))

flags.DEFINE_bool(
    'use_cache', default=False, help=('Enable cache for training input.'))

flags.DEFINE_integer(
    'dataset_threadpool_size', default=48,
    help=('The size of the private datapool size in dataset.'))

flags.DEFINE_bool(
    'map_and_batch_fusion', default=None,
    help=('tf.data options'))
