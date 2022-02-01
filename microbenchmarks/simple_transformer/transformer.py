# Lint as: python3
"""Input pipeline for the MLPerf WMT de-en dataset.

This version of the WMT17 dataset uses a shared 32K subword tokenization
and training data packed into 256-length sequences with additional
segmentation masks and sub-sequence position data.  The evaluation
data is from WMT14 and not packed, and has a max length of 97.

Generation script instructions can be found at:
https://github.com/mlperf/training_results_v0.6/tree/master/Google/benchmarks/transformer/implementations/tpu-v3-32-transformer/dataset_preproc

We load the WMT train and eval datasets in their entirety into host memory.
"""

import tensorflow as tf
from concurrent.futures import thread
import functools
import gc
import math
import os
import time
from absl import app
from absl import flags
from absl import logging
import numpy as np

# MLPerf Dataset Constants.
# Packed WMT17 training data.
MAX_TRAIN_LEN = 256  # multiple sequences are packed into this length.
N_TRAIN = 566340  # number of (packed) training tfrecords.
TRAIN_KEYS = ['inputs', 'inputs_position', 'inputs_segmentation',
              'targets', 'targets_position', 'targets_segmentation']
# Truncated WMT14 eval data.
MAX_EVAL_LEN = 97  # no eval sequences are longer than this.
N_EVAL = 3003  # number of eval tfrecords.
EVAL_KEYS = ['inputs', 'targets']
# Default data paths.
TRAIN_PATH = None
EVAL_PATH = None
VOCAB_PATH = None

# BEGIN GOOGLE-INTERNAL
# original CNS location of transformer mlperf data:
# ROOT = '/REDACTED/ei-d/home/tpu-perf-team/shibow/'
ROOT = '/readahead/128M/REDACTED/je-d/home/tpu-perf-team/shibow/transformer_new_sharded1024_train80_data/'
# Packed Training Data:
TRAIN_PATH = ROOT + 'translate_ende_wmt32k_packed-train*'
# Evaluation Data -- Not Packed:
EVAL_PATH = ROOT + 'translate_ende_wmt32k-dev*'
VOCAB_PATH = ROOT + 'vocab.translate_ende_wmt32k.32768.subwords'
# END GOOGLE-INTERNAL


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'train_data_path', default=TRAIN_PATH,
    help='Path to packed training dataset tfrecords.')

flags.DEFINE_string(
    'eval_data_path', default=EVAL_PATH,
    help='Path to unpacked evaluation dataset tfrecords.')

flags.DEFINE_string(
    'vocab_path', default=VOCAB_PATH,
    help='Path to shared vocabulary file.')

flags.DEFINE_integer(
    'batch_size', default=None,
    help='Global batch size for training.')
flags.DEFINE_integer(
    'num_partitions', default=1,
    help='Number of SPMD partitions to use.')
flags.DEFINE_bool(
    'use_cache', default=False,
    help='')
flags.DEFINE_integer(
    'read_parallelism', default=10,
    help=(''))
flags.DEFINE_integer(
    'map_1_parallelism', default=32,
    help=(''))
flags.DEFINE_integer(
    'map_2_parallelism', default=32,
    help=(''))
flags.DEFINE_integer(
    'map_3_parallelism', default=32,
    help=(''))



def length_filter(max_len):
  def filter_fn(batch):
    l = tf.maximum(tf.shape(batch['inputs'])[0],
                   tf.shape(batch['targets'])[0])
    return tf.less(l, max_len + 1)
  return filter_fn


def pad_up_to(t, max_in_dims):
  s = tf.shape(t)
  paddings = [[0, m - s[i]] for (i, m) in enumerate(max_in_dims)]
  return tf.pad(t, paddings, 'CONSTANT', constant_values=0)


def get_wmt_dataset(batch_size, train, shuffle_size=16384):
  """Get the train or eval split of WMT as a tf.data.Dataset."""
  keys = TRAIN_KEYS if train else EVAL_KEYS

  def parse_function(example_proto):
    return tf.io.parse_single_example(
        example_proto,
        {k: tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True)
         for k in keys})

  def cast_to_int32(x):
    return {k: tf.dtypes.cast(x[k], tf.int32) for k in keys}

  def pad(x):
    return {k: pad_up_to(x[k], [MAX_TRAIN_LEN if train else MAX_EVAL_LEN,])
            for k in keys}

  file_pattern = os.path.join(
      FLAGS.train_data_path if train else FLAGS.eval_data_path)
  print("Using files: {}".format(file_pattern))
  dataset = tf.data.Dataset.list_files(file_pattern, shuffle=False)
  #dataset = dataset.shard(jax.host_count(), jax.host_id())
  #concurrent_files = min(10, 1024 // jax.host_count())
  #concurrent_files = min(10, 1024 // 1)
  dataset = dataset.interleave(
      tf.data.TFRecordDataset,
      FLAGS.read_parallelism, 1, FLAGS.read_parallelism)
  if FLAGS.use_cache:
      dataset = dataset.take(500).cache().repeat()

  dataset = dataset.map(parse_function,
                        num_parallel_calls=FLAGS.map_1_parallelism)
  dataset = dataset.map(cast_to_int32, num_parallel_calls=FLAGS.map_2_parallelism)
  if train:
    # Filter out rare long, unpacked single-examples.
    dataset = dataset.filter(length_filter(MAX_TRAIN_LEN))
  #outer_parallelism = 4
  #o_dataset = tf.data.Dataset.from_tensor_slices([i for i in
  #                                              range(outer_parallelism)])
  #dataset = o_dataset.interleave(lambda x: dataset,
  #                             outer_parallelism, 1, outer_parallelism)

  dataset = dataset.map(pad, num_parallel_calls=FLAGS.map_3_parallelism)
  if train:
      # TODO(mkuchnik): We move cache up
      if FLAGS.use_cache and False:
        dataset = dataset.cache().shuffle(shuffle_size).repeat()
      else:
        dataset = dataset.shuffle(shuffle_size).repeat()

  dataset = dataset.batch(batch_size, drop_remainder=train)
  if not train:
    dataset = dataset.cache().repeat()
  dataset = dataset.prefetch(1024)

  return dataset

def get_dataset():
   num_partitions = FLAGS.num_partitions
   batch_size = FLAGS.batch_size
   if batch_size is None:
     #batch_size = min(16 * jax.device_count() // num_partitions, 2048)
     batch_size = min(16 * 1 // num_partitions, 2048)

   train_ds = get_wmt_dataset(
       batch_size=batch_size // 1, train=True)
   return train_ds

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  tf.enable_v2_behavior()

  ds = get_dataset()
  for x in ds:
      print(x)


if __name__ == '__main__':
  app.run(main)
