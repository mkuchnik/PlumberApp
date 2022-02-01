from absl import app
from absl import flags

import tensorflow as tf
import dataloader
from plumber_analysis import gen_util

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'time_limit_s', default=None,
    help=('Number of seconds to run for'))
flags.DEFINE_bool(
    'profile', default=False,
    help=('Whether to profile'))


def get_TFRecord_dataset():
    dataset = dataloader.get_dataset(FLAGS.data_dir)
    return dataset

def main(_):
    dataset = get_TFRecord_dataset()
    dataset = dataset.take(FLAGS.benchmark_num_elements)
    options = tf.data.Options()
    options.experimental_deterministic = False
    options.experimental_threading.max_intra_op_parallelism = 1
    options.experimental_threading.private_threadpool_size = FLAGS.dataset_threadpool_size
    options.experimental_optimization.map_and_batch_fusion = FLAGS.map_and_batch_fusion
    gen_util.add_analysis_to_dataset_options(options)
#    options.experimental_optimization.autotune_stats_dump_period = 1000
    #try:
    #    options.experimental_optimization.autotune_span_collection_interval = 50
    #except Exception as ex:
    #    print(ex)
    dataset = dataset.with_options(options)
    if FLAGS.profile:
        summary = gen_util.benchmark_and_profile_dataset(
            dataset, time_limit_s=FLAGS.time_limit_s)
    else:
        # TODO(mkuchnik): Adjust profile interval?
        summary = gen_util.benchmark_dataset(
            dataset, time_limit_s=FLAGS.time_limit_s)

if __name__ == '__main__':
  app.run(main)
