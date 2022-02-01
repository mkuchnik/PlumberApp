# Lint as: python3
"""SSD main training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from concurrent.futures import thread
import math

import time
from absl import app
from absl import flags
from absl import logging

import jax

from jax import config
from jax import lax
from jax import random

from jax.interpreters import sharded_jit
import jax.numpy as jnp
from jax.util import partial
import numpy as onp

import tensorflow.compat.v2 as tf

import input_pipeline
import ssd_constants


flags.DEFINE_string(
    'resnet_checkpoint', None,
    'Location of the ResNet checkpoint to use for model '
    'initialization.')
flags.DEFINE_string('hparams', '',
                    'Comma separated k=v pairs of hyperparameters.')

flags.DEFINE_integer('global_batch_size', 1024, 'training batch size')
flags.DEFINE_integer('eval_batch_size', 1024, 'evaluation batch size')
flags.DEFINE_integer('eval_samples', ssd_constants.EVAL_SAMPLES,
                     'The number of samples for evaluation.')
flags.DEFINE_integer('iterations_per_loop', 1000,
                     'Number of iterations per TPU training loop')
flags.DEFINE_string(
    'training_file_pattern',
    #'/placer/prod/home/tpu-perf-team/mlperf/ssd/train*',
    "/zpool1/Datasets/Official/MLPerf/COCO/train*",
    'Glob for training data files (e.g., COCO train - minival set)')
flags.DEFINE_string('validation_file_pattern',
                    #'/placer/prod/home/tpu-perf-team/mlperf/ssd/coco_val*',
                    "/zpool1/Datasets/Official/MLPerf/COCO/val*",
                    'Glob for evaluation tfrecords (e.g., COCO val2017 set)')

flags.DEFINE_bool(
    'use_fake_data', False,
    'Use fake data to reduce the input preprocessing overhead (for unit tests)')

flags.DEFINE_string(
    'val_json_file',
    '/placer/prod/home/tpu-perf-team/mlperf/ssd/instances_val2017.json',
    'COCO validation JSON containing golden bounding boxes.')

flags.DEFINE_integer('num_examples_per_epoch', 118287,
                     'Number of examples in one epoch')
flags.DEFINE_integer('num_epochs', 78, 'Number of epochs for training')
flags.DEFINE_multi_integer(
    'input_partition_dims',
    default=None,
    help=('Number of partitions on each dimension of the input. Each TPU core'
          ' processes a partition of the input image in parallel using spatial'
          ' partitioning.'))
flags.DEFINE_bool('run_cocoeval', True, 'Whether to run cocoeval')
flags.DEFINE_string(
    'model_dir',
    default=None,
    help=('The directory where the model and summaries are stored.'))
flags.DEFINE_bool('use_bfloat16', True, 'use bfloat16')

flags.DEFINE_integer('lr_warmup_epoch', 5, '')
flags.DEFINE_float(
    'base_learning_rate', default=ssd_constants.BASE_LEARNING_RATE,
    help='Base learning rate.')

flags.DEFINE_float('first_lr_drop_epoch', default=42.6, help='')
flags.DEFINE_float('second_lr_drop_epoch', default=53.3, help='')

flags.DEFINE_bool(
    'precompile', default=True,
    help='Perform all XLA compilation before touching data.')

flags.DEFINE_bool(
    'transpose_input', default=True,
    help='Apply double transpose.')

# Used a different flag name, so we dont trigger the optimization in shared code
flags.DEFINE_bool(
    'conv0_space_to_depth', default=True,
    help='Space to Depth Optimization.')

flags.DEFINE_bool(
    'infeed', default=True,
    help='Stage out training loop to XLA using infeed for data loading.')

flags.DEFINE_bool(
    'profile', default=False,
    help='Enable programmatic profile with xprof.')

flags.DEFINE_bool(
    'detailed_time', default=False,
    help='Shows eval, train and coco_eval times separately(Adds barriers avoid '
    'in default mode)')

flags.DEFINE_bool(
    'enable_wus', default=True,
    help='Whether to enable weight update sharding')

flags.DEFINE_integer(
    'num_partitions', default=1, help=('Number of partitions in SPMD.'))

flags.DEFINE_integer(
    'bn_group_size', default=1, help=('Num. cores for Distributed Batch Norm.'))

flags.DEFINE_integer(
    'repeat_experiment', default=1, help=('Number of runs'))

flags.DEFINE_integer(
    'train_batch_size', default=4, help='Batch size for training.')

#flags.DEFINE_integer(
#    'dataset_threadpool_size', default=48,
#    help=('The size of the private datapool size in dataset.'))

flags.DEFINE_integer('seed', None, 'Random seed')
# Adds jax_log_compiles flag to print compilation logs on the jax side.
config.parse_flags_with_absl()
FLAGS = flags.FLAGS

coco_gt = None


def construct_run_config():
  """Construct the run config parameters.

  Returns:
    A dictionary containing run parameters.
  """

  global_batch_size = FLAGS.global_batch_size
  num_shards = jax.local_device_count() * jax.host_count()
  num_replicas = num_shards // FLAGS.num_partitions
  # Do not transpose input if spatial partitioning is enabled for now.
  transpose_input = False if FLAGS.num_partitions > 1 else FLAGS.transpose_input
  dtype = jnp.bfloat16 if FLAGS.use_bfloat16 else jnp.float32
  return dict(
      base_learning_rate=FLAGS.base_learning_rate,
      batch_size=global_batch_size,  # global batch size
      host_batch_size=global_batch_size // jax.host_count(),  # Used in input_fn
      device_batch_size=global_batch_size // num_replicas,  # Used in model_fn
      eval_batch_size=FLAGS.eval_batch_size,  # global batch size
      host_eval_batch_size=FLAGS.eval_batch_size // jax.host_count(),
      device_eval_batch_size=FLAGS.eval_batch_size // num_replicas,
      conv0_space_to_depth=FLAGS.conv0_space_to_depth,
      dataset_index=jax.host_id(),
      dataset_num_shards=jax.host_count(),
      dbn_tile_col=-1,  # number of cols in each distributed batch norm group.
      dbn_tile_row=-1,  # number of rows in each distributed batch norm group.
      enable_wus=FLAGS.enable_wus,
      eval_every_checkpoint=False,
      eval_samples=FLAGS.eval_samples,
      first_lr_drop_epoch=FLAGS.first_lr_drop_epoch,
      second_lr_drop_epoch=FLAGS.second_lr_drop_epoch,
      iterations_per_loop=FLAGS.iterations_per_loop,
      local_device_count=jax.local_device_count(),
      lr_warmup_epoch=FLAGS.lr_warmup_epoch,
      model_dir=FLAGS.model_dir,
      num_epochs=FLAGS.num_epochs,
      num_examples_per_epoch=FLAGS.num_examples_per_epoch,
      num_shards=num_shards,
      num_replicas=num_replicas,
      local_num_replicas=jax.local_device_count() // FLAGS.num_partitions,
      resnet_checkpoint=FLAGS.resnet_checkpoint,
      steps_per_epoch=int(
          math.ceil(FLAGS.num_examples_per_epoch / global_batch_size)),
      tpu_slice_col=-1,
      tpu_slice_row=-1,
      transpose_input=transpose_input,
      use_bfloat16=FLAGS.use_bfloat16,
      dtype=dtype,
      bn_group_size=FLAGS.bn_group_size,
      use_spatial_partitioning=FLAGS.num_partitions > 1,
      num_partitions=FLAGS.num_partitions,
      val_json_file=FLAGS.val_json_file,
      visualize_dataloader=False,
      eval_steps=int(math.ceil(FLAGS.eval_samples / FLAGS.eval_batch_size)),
      weight_decay=ssd_constants.WEIGHT_DECAY,
  )


def main(argv):
  # BEGIN GOOGLE-INTERNAL
  #xm.setup_work_unit()
  # END GOOGLE-INTERNAL

  del argv

  tf.enable_v2_behavior()
  for _ in range(FLAGS.repeat_experiment):
    run_ssd()


def get_dataset():
  """Runs a single end to end ssd experiment."""
  params = construct_run_config()
  params['dataset_threadpool_size'] = FLAGS.dataset_threadpool_size
  params['host_batch_size'] = FLAGS.train_batch_size

  train_ds = input_pipeline.ssd_input_pipeline(
      params,
      FLAGS.training_file_pattern,
      is_training=True,
      use_fake_data=FLAGS.use_fake_data,
      host_batch_size=params['host_batch_size'],
      transpose_input=params['transpose_input'])
  return train_ds

def run_ssd():
  train_ds = get_dataset()

  for x in train_ds:
      pass

if __name__ == '__main__':
  app.run(main)
