from absl import app
from absl import flags

import jax
import tensorflow.compat.v2 as tf
import mlperf_input_pipeline as input_pipeline
import new_flags

from plumber_analysis import gen_util

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'time_limit_s', default=None,
    help=('Number of seconds to run for'))

flags.DEFINE_bool(
    'profile', default=False,
    help=('Whether to profile'))


def get_TFRecord_dataset():
    batch_size = 128
    train = True
    space_to_depth = False
    input_dtype = tf.bfloat16
    image_format = "HWCN"
    dataset = input_pipeline.load_split(
            batch_size,
            dtype=input_dtype,
            train=train,
            image_format=image_format,
            space_to_depth=space_to_depth,
            cache_uncompressed=FLAGS.force_cache_uncompressed)

    return dataset

def main(argv):
    del argv
    tf.enable_v2_behavior()
    dataset = get_TFRecord_dataset()
    #dataset = dataset.take(FLAGS.benchmark_num_elements)
    options = tf.data.Options()
    #options.experimental_deterministic = False
    #options.experimental_threading.max_intra_op_parallelism = 1
    #options.threading.private_threadpool_size = 90
    #options.experimental_optimization.map_and_batch_fusion = True
    gen_util.add_analysis_to_dataset_options(options)
    #try:
    #    options.experimental_optimization.autotune_span_collection_interval = \
    #        100
    #except Exception as ex:
    #    print(ex)
    dataset = dataset.with_options(options)
    tf.print("Profiling {}".format(FLAGS.profile))
    if FLAGS.profile:
        tf.print("Profiling")
        summary = gen_util.benchmark_and_profile_dataset(
            dataset, time_limit_s=FLAGS.time_limit_s)
    else:
        tf.print("Not Profiling")
        summary, df = gen_util.benchmark_dataset(
            dataset, time_limit_s=FLAGS.time_limit_s, profile_interval=10,
            return_monitoring_data=True)
        df.to_csv("monitoring.csv", index=False)


if __name__ == '__main__':
  app.run(main)
