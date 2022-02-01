from absl import app
from absl import flags

import ssd_train
import tensorflow as tf
import dataset_flags

from plumber_analysis import gen_util

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'time_limit_s', default=None,
    help=('Number of seconds to run for'))

def get_dataset(*args, **kwargs):
    dataset = ssd_train.get_dataset(*args, **kwargs)
    return dataset

def main(_):
    dataset, params = get_dataset(return_params=True)
    if FLAGS.benchmark_num_elements:
        dataset = dataset.take(FLAGS.benchmark_num_elements)
    #dataset = dataset.prefetch(50)
    options = tf.data.Options()
    options.experimental_deterministic = False
    options.experimental_threading.max_intra_op_parallelism = 1
    options.experimental_threading.private_threadpool_size = \
        FLAGS.dataset_threadpool_size
    if FLAGS.map_and_batch_fusion is not None:
        options.experimental_optimization.map_and_batch_fusion = \
            FLAGS.map_and_batch_fusion
    gen_util.add_analysis_to_dataset_options(options)
    #try:
    #    options.experimental_optimization.autotune_span_collection_interval = 50
    #except Exception as ex:
    #    print(ex)
    dataset = dataset.with_options(options)
    def element_count_f(x) -> int:
        return params["host_batch_size"]
    if FLAGS.profile and False: # TODO(mkuchnik): Flag collision
        tf.print("Profiling")
        summary = gen_util.benchmark_and_profile_dataset(
            dataset, time_limit_s=FLAGS.time_limit_s,
            element_count_f=element_count_f)
        tf.print(summary)
    else:
        tf.print("Not Profiling")
        summary, df = gen_util.benchmark_dataset(
            dataset, time_limit_s=FLAGS.time_limit_s, profile_interval=10,
            return_monitoring_data=True)
        tf.print(summary)
        tf.print(df)
        df.to_csv("monitoring.csv", index=False)

if __name__ == '__main__':
  app.run(main)
