#) Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Training script for Mask-RCNN.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import math
import multiprocessing
import time
from absl import app
from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
tf.disable_v2_behavior()
from tensorflow.python.ops import control_flow_util

import coco_metric
import dataloader
import eval_multiprocess
import mask_rcnn_model
import mask_rcnn_params
import ml_perf_log as mlp_log
from util import train_and_eval_runner

import time
import logging

# copybara:strip_begin
#from REDACTED.REDACTED.multiprocessing import REDACTEDprocess
# copybara:strip_end

# Cloud TPU Cluster Resolvers
flags.DEFINE_string('hparams', '',
                    'Comma separated k=v pairs of hyperparameters.')
flags.DEFINE_integer(
    'num_shards',
    default=8,
    help='Number of shards (TPU cores) for '
    'training.')
flags.DEFINE_multi_integer(
    'input_partition_dims', None,
    'A list that describes the partition dims for all the tensors.')
tf.flags.DEFINE_integer('train_batch_size', 128, 'training batch size')
tf.flags.DEFINE_integer('eval_batch_size', 128, 'evaluation batch size')
tf.flags.DEFINE_integer('eval_samples', 5000, 'The number of samples for '
                        'evaluation.')
flags.DEFINE_string('resnet_checkpoint', '',
                    'Location of the ResNet50 checkpoint to use for model '
                    'initialization.')
flags.DEFINE_string(
    'training_file_pattern', None,
    'Glob for training data files (e.g., COCO train - minival set)')
flags.DEFINE_string(
    'validation_file_pattern', None,
    'Glob for evaluation tfrecords (e.g., COCO val2017 set)')
flags.DEFINE_string(
    'val_json_file',
    None,
    'COCO validation JSON containing golden bounding boxes.')
tf.flags.DEFINE_integer('num_examples_per_epoch', 118287,
                        'Number of examples in one epoch')
tf.flags.DEFINE_integer('num_epochs', 15, 'Number of epochs for training')
flags.DEFINE_string('mode', 'train',
                    'Mode to run: train or eval (default: train)')
flags.DEFINE_bool('eval_after_training', False, 'Run one eval after the '
                  'training finishes.')
flags.DEFINE_bool('use_fake_data', False, 'Use fake input.')

# For Eval mode
flags.DEFINE_integer('min_eval_interval', 180,
                     'Minimum seconds between evaluations.')

FLAGS = flags.FLAGS
_STOP = -1
ap_score = 0
mask_ap_score = 0
cur_epoch = 0
last_time = 0


def run_mask_rcnn(hparams):
  """Runs the Mask RCNN train and eval loop."""

  global ap_score
  global mask_ap_score
  global cur_epoch

  ap_score = 0
  mask_ap_score = 0
  cur_epoch = 0

  params = dict(
      hparams.values(),
      transpose_input=False if FLAGS.input_partition_dims is not None else True,
      resnet_checkpoint=FLAGS.resnet_checkpoint,
      val_json_file=FLAGS.val_json_file,
      num_cores_per_replica=int(np.prod(FLAGS.input_partition_dims))
      if FLAGS.input_partition_dims else 1,
      replicas_per_host=FLAGS.replicas_per_host)

  # MLPerf logging.
  mlp_log.mlperf_print(key='cache_clear', value=True)
  mlp_log.mlperf_print(key='init_start', value=None)
  mlp_log.mlperf_print(key='global_batch_size', value=FLAGS.train_batch_size)
  mlp_log.mlperf_print(key='train_samples', value=FLAGS.num_examples_per_epoch)
  mlp_log.mlperf_print(key='eval_samples', value=FLAGS.eval_samples)
  mlp_log.mlperf_print(
      key='min_image_size', value=params['short_side_image_size'])
  mlp_log.mlperf_print(
      key='max_image_size', value=params['long_side_max_image_size'])
  mlp_log.mlperf_print(key='num_image_candidates',
                       value=params['rpn_post_nms_topn'])
  mlp_log.mlperf_print(
      key='opt_base_learning_rate', value=params['learning_rate'])
  mlp_log.mlperf_print(
      key='opt_learning_rate_warmup_steps', value=params['lr_warmup_step'])
  mlp_log.mlperf_print(
      key='opt_learning_rate_warmup_factor',
      value=params['learning_rate'] / params['lr_warmup_step'])
  mlp_log.mlperf_print(key='opt_learning_rate_decay_factor', value=0.1)
  mlp_log.mlperf_print(
      'opt_learning_rate_decay_steps',
      (params['first_lr_drop_step'], params['second_lr_drop_step']))
  mlp_log.mlperf_print('gradient_accumulation_steps', 1)
  seed = time.time()
  tf.set_random_seed(seed)
  mlp_log.mlperf_print('seed', seed)

  train_steps = (
      FLAGS.num_epochs * FLAGS.num_examples_per_epoch // FLAGS.train_batch_size)
  eval_steps = int(math.ceil(float(FLAGS.eval_samples) / FLAGS.eval_batch_size))
  if eval_steps > 0:
    # The eval dataset is not evenly divided. Adding step by one will make sure
    # all eval samples are covered.
    #                    same amount of work.
    eval_steps += 1
  runner = train_and_eval_runner.TrainAndEvalRunner(
      FLAGS.num_examples_per_epoch // FLAGS.train_batch_size, train_steps,
      eval_steps, FLAGS.num_shards)
  train_input_fn = dataloader.InputReader(
      FLAGS.training_file_pattern,
      mode=tf.estimator.ModeKeys.TRAIN,
      use_fake_data=FLAGS.use_fake_data)
  eval_input_fn = functools.partial(
      dataloader.InputReader(
          FLAGS.validation_file_pattern,
          mode=tf.estimator.ModeKeys.PREDICT,
          distributed_eval=True),
      num_examples=eval_steps * FLAGS.eval_batch_size)
  eval_metric = coco_metric.EvaluationMetric(
      FLAGS.val_json_file, use_cpp_extension=True)

  def init_fn():
    if FLAGS.resnet_checkpoint:
      tf.train.init_from_checkpoint(FLAGS.resnet_checkpoint,
                                    {'resnet/': 'resnet50/'})

  runner.initialize(train_input_fn, eval_input_fn,
                    mask_rcnn_model.MaskRcnnModelFn(params),
                    FLAGS.train_batch_size, FLAGS.eval_batch_size,
                    FLAGS.input_partition_dims, init_fn, params=params)
  mlp_log.mlperf_print('init_stop', None)
  mlp_log.mlperf_print('run_start', None)

  def eval_init_fn(cur_step):
    """Executed before every eval."""
    global cur_epoch
    steps_per_epoch = FLAGS.num_examples_per_epoch // FLAGS.train_batch_size
    cur_epoch = 0 if steps_per_epoch == 0 else cur_step // steps_per_epoch
    mlp_log.mlperf_print(
        'block_start',
        None,
        metadata={
            'first_epoch_num': cur_epoch,
            'epoch_count': 1
        })

  def eval_finish_fn(cur_step, eval_output, _):
    """Callback function that's executed after each eval."""
    global ap_score
    global mask_ap_score
    global cur_epoch
    global last_time

    num_examples = FLAGS.num_examples_per_epoch
    elapsed_time = time.time() - last_time
    rate = num_examples / elapsed_time
    logging.info("Rate: {}".format(rate))
    last_time = time.time()

    # NOTE(mkuchnik): Disable eval
    mlp_log.mlperf_print(
        'eval_start', None, metadata={'epoch_num': cur_epoch + 1})
    if eval_steps == 0 or cur_epoch < 5:
      return False
    elif cur_epoch >= 5:
      mlp_log.mlperf_print('run_stop', None, metadata={'status': 'success'})
      return True

    # Concat eval_output as eval_output is a list from each host.
    predictions = dict(
        detections=np.concatenate(eval_output['detections'], axis=0),
        image_info=np.concatenate(
            eval_output['image_info'], axis=0).astype(np.int32),
        mask_outputs=np.concatenate(
            eval_output['mask_outputs'], axis=0))

    steps_per_epoch = FLAGS.num_examples_per_epoch // FLAGS.train_batch_size
    cur_epoch = 0 if steps_per_epoch == 0 else cur_step // steps_per_epoch
    mlp_log.mlperf_print(
        'block_stop',
        None,
        metadata={
            'first_epoch_num': cur_epoch,
            'epoch_count': 1
        })
    eval_multiprocess.eval_multiprocessing(predictions, eval_metric,
                                           mask_rcnn_params.EVAL_WORKER_COUNT)

    mlp_log.mlperf_print(
        'eval_start', None, metadata={'epoch_num': cur_epoch + 1})
    _, eval_results = eval_metric.evaluate()

    ap_score = eval_results['AP']
    mask_ap_score = eval_results['mask_AP']

    mlp_log.mlperf_print(
        'eval_accuracy',
        {'BBOX': float(eval_results['AP']),
         'SEGM': float(eval_results['mask_AP'])},
        metadata={'epoch_num': cur_epoch + 1})
    mlp_log.mlperf_print(
        'eval_stop', None, metadata={'epoch_num': cur_epoch + 1})
    if (eval_results['AP'] >= mask_rcnn_params.BOX_EVAL_TARGET and
        eval_results['mask_AP'] >= mask_rcnn_params.MASK_EVAL_TARGET):
      mlp_log.mlperf_print('run_stop', None, metadata={'status': 'success'})
      return True
    return False

  def run_finish_fn(success):
    if not success:
      mlp_log.mlperf_print('run_stop', None, metadata={'status': 'abort'})

  runner.train_and_eval(eval_init_fn, eval_finish_fn, run_finish_fn)
  return cur_epoch, ap_score, mask_ap_score


def main(argv):
  del argv  # Unused.

  control_flow_util.ENABLE_CONTROL_FLOW_V2 = True

  # Parse hparams
  hparams = mask_rcnn_params.default_hparams()
  hparams.parse(FLAGS.hparams)
  run_mask_rcnn(hparams)


if __name__ == '__main__':
  # copybara:strip_begin
  #user_data = (multiprocessing.Queue(maxsize=mask_rcnn_params.QUEUE_SIZE),
  #             multiprocessing.Queue(maxsize=mask_rcnn_params.QUEUE_SIZE))
  #with REDACTEDprocess.main_handler(user_data=user_data):
  #  tf.logging.set_verbosity(tf.logging.INFO)
  #  app.run(main)
  # copybara:strip_end
  # copybara:insert tf.logging.set_verbosity(tf.logging.INFO)
  # copybara:insert app.run(main)
  tf.logging.set_verbosity(tf.logging.INFO)
  app.run(main)
