# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Efficient ImageNet input pipeline using tf.data.Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
from absl import flags
import jax
import tensorflow as tf
# NEW
from plumber_analysis import gen_util, config
from plumber_analysis import pipeline_optimizer_wrapper as pipeline_optimizer
import plumber_analysis.annotations
import dataset_echoing
import mixup
import augmentations
import logging

from plumber_analysis import pipeline_optimizer as _pipeline_optimizer

_pipeline_optimizer.DEFAULT_BENCHMARK_TIME = 60 + 2
pipeline_optimizer.BENCHMARK_TIME = 60 + 2

config.enable_compat_logging()

FLAGS = flags.FLAGS
TRAIN_IMAGES = 1281167
EVAL_IMAGES = 50000
IMAGE_SIZE = 224
CROP_PADDING = 32
MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]
NUM_CLASSES = 1000


flags.DEFINE_string(
    'data_dir', default='/readahead/128M/placer/prod/home/distbelief/imagenet-tensorflow/imagenet-2012-tfrecord',
    help='Directory to load data from.')
# NEW
flags.DEFINE_string(
    'val_data_dir', default=None,
    help='Directory to load validation data from.')
flags.DEFINE_integer(
    'input_pipeline_default_parallelism',
    default=64,
    help='The input pipeline default parallelism')
flags.DEFINE_integer(
    'input_pipeline_default_prefetching',
    default=100,
    help='The input pipeline default prefetching')
flags.DEFINE_integer(
    'input_pipeline_threadpool_size',
    default=48,
    help='The input pipeline threadpool size')

def DEFAULT_PARALLELISM():
    return FLAGS.input_pipeline_default_parallelism

def DEFAULT_PREFETCHING():
    return FLAGS.input_pipeline_default_prefetching

def expand_grid_combinations(arg_grid_dict):
    """All possible value combinations with keys"""
    # https://stackoverflow.com/questions/38721847/how-to-generate-all-combination-from-values-in-dict-of-lists-in-python
    import itertools
    keys, values = zip(*arg_grid_dict.items())
    values = list(map(lambda v: [v] if not isinstance(v, list) else v, values))
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return permutations_dicts

def _load_split_debug(batch_size, train, dtype, image_format, space_to_depth,
               cache_uncompressed, image_size=IMAGE_SIZE, reshape_to_r1=False,
               shuffle_size=16384):
    """Returns the input_fn (wrapped with Plumber).

    This is for debugging. Use annotation variant of load_split otherwise.
    """
    ENABLED_BEST_PIPELINE = pipeline_optimizer.plumber_find_best_pipeline()
    ENABLED_FAKE_PIPELINE = False
    ENABLED_SOURCE_PIPELINE = False
    # TODO(mkuchnik): Make both pipelines identical from augmentation point of view
    if ENABLED_BEST_PIPELINE and FLAGS.optimize_plumber_pipeline and train:
        arg_grid = {"batch_size": batch_size,
                    "train": train,
                    "dtype": dtype,
                    "image_format": image_format,
                    "space_to_depth": space_to_depth,
                    "cache_uncompressed": [False, True],
                    "image_size": image_size,
                    "reshape_to_r1": reshape_to_r1,
                    "shuffle_size": shuffle_size}
        arg_grid = expand_grid_combinations(arg_grid)
        for args in arg_grid:
            logging.info("args {}".format(args))
        datasets = [_load_split(**args) for args in arg_grid]
        logging.info("Finding optimized pipeline over {} pipelines".format(len(datasets)))
        dataset = pipeline_optimizer.get_best_optimized_pipeline(datasets)
    else:
        dataset = _load_split(batch_size, train, dtype, image_format, space_to_depth,
                cache_uncompressed, image_size, reshape_to_r1, shuffle_size)
        if FLAGS.optimize_plumber_pipeline and train:
            if ENABLED_FAKE_PIPELINE:
                logging.info("Using fake pipeline")
                new_dataset = pipeline_optimizer.get_fake_pipeline(dataset)
            elif ENABLED_SOURCE_PIPELINE:
                logging.info("Using source pipeline")
                rets = pipeline_optimizer.benchmark_source_parallelisms(dataset)
                import pprint
                print("Source pipeline sweep rets:\n{}".format(pprint.pformat(rets)))
                new_dataset = pipeline_optimizer.get_source_pipeline(dataset)
            else:
                logging.info("Finding optimized pipeline")
                new_dataset = pipeline_optimizer.get_optimized_pipeline(dataset)
            dataset = new_dataset
    return dataset

def load_split(batch_size, train, dtype, image_format, space_to_depth,
               cache_uncompressed, image_size=IMAGE_SIZE, reshape_to_r1=False,
               shuffle_size=16384):
    """Returns the input_fn (wrapped with Plumber)."""
    # NOTE(mkuchnik): Use below for testing, otherwise, use annotations
    # dataset = _load_split_debug(batch_size=batch_size,
    #                             train=train,
    #                             dtype=dtype,
    #                             image_format=image_format,
    #                             space_to_depth=space_to_depth,
    #                             cache_uncompressed=cache_uncompressed,
    #                             image_size=image_size,
    #                             reshape_to_r1=reshape_to_r1,
    #                             shuffle_size=shuffle_size)
    dataset = _annotated_load_split(batch_size=batch_size,
                                    train=train,
                                    dtype=dtype,
                                    image_format=image_format,
                                    space_to_depth=space_to_depth,
                                    cache_uncompressed=cache_uncompressed,
                                    image_size=image_size,
                                    reshape_to_r1=reshape_to_r1,
                                    shuffle_size=shuffle_size)
    return dataset


@plumber_analysis.annotations.maybe_find_best_pipeline(
        kwargs_precondition_f=lambda x: bool(x["train"]) and FLAGS.optimize_plumber_pipeline,
        optimization_arg_grid = {"cache_uncompressed": [False, True]},
        )
def _annotated_load_split(batch_size, train, dtype, image_format, space_to_depth,
               cache_uncompressed, image_size=IMAGE_SIZE, reshape_to_r1=False,
               shuffle_size=16384):
    """Optimize pipeline using Plumber annotations."""
    return _load_split(batch_size=batch_size,
                       train=train,
                       dtype=dtype,
                       image_format=image_format,
                       space_to_depth=space_to_depth,
                       cache_uncompressed=cache_uncompressed,
                       image_size=image_size,
                       reshape_to_r1=reshape_to_r1,
                       shuffle_size=shuffle_size)

def _load_split(batch_size, train, dtype, image_format, space_to_depth,
               cache_uncompressed, image_size=IMAGE_SIZE, reshape_to_r1=False,
               shuffle_size=16384):
  """Returns the input_fn."""

  def dataset_parser(value):
    """Parses an image and its label from a serialized ResNet-50 TFExample."""
    parsed = tf.io.parse_single_example(
        value, {
            'image/encoded': tf.io.FixedLenFeature((), tf.string, ''),
            'image/class/label': tf.io.FixedLenFeature([], tf.int64, 0)
        })
    image_bytes = tf.reshape(parsed['image/encoded'], [])
    label = tf.cast(tf.reshape(parsed['image/class/label'], []), tf.int32) - 1

    def preprocess_fn():
      """Preprocess the image."""
      shape = tf.image.extract_jpeg_shape(image_bytes)
      if train:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
                           dtype=tf.float32,
                           shape=[1, 1, 4])
        bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
            tf.image.extract_jpeg_shape(image_bytes),
            bbox,
            min_object_covered=0.1,
            aspect_ratio_range=(0.75, 1.33),
            area_range=(0.05, 1.0),
            max_attempts=10,
            use_image_if_no_bounding_boxes=True)
        offset_y, offset_x, _ = tf.unstack(bbox_begin)
        target_height, target_width, _ = tf.unstack(bbox_size)
        crop_window = tf.stack(
            [offset_y, offset_x, target_height, target_width])
      else:
        crop_size = tf.cast(
            ((image_size / (image_size + CROP_PADDING)) *
             tf.cast(tf.minimum(shape[0], shape[1]), tf.float32)), tf.int32)
        offset_y, offset_x = [
            ((shape[i] - crop_size) + 1) // 2 for i in range(2)
        ]
        crop_window = tf.stack([offset_y, offset_x, crop_size, crop_size])

      image = tf.image.decode_and_crop_jpeg(
          image_bytes, crop_window, channels=3)
      image = tf.image.resize(
          [image], [image_size, image_size], method='bicubic')[0]
      if train:
        image = tf.image.random_flip_left_right(image)
      image = tf.reshape(image, [image_size, image_size, 3])
      return tf.image.convert_image_dtype(image, dtype)

    empty_example = tf.zeros([image_size, image_size, 3], dtype)
    return tf.cond(label < 0, lambda: empty_example, preprocess_fn), label

  def cached_parser(value):
    """Parses an image and its label from a serialized ResNet-50 TFExample."""
    parsed = tf.io.parse_single_example(
        value, {
            'image/encoded': tf.io.FixedLenFeature((), tf.string, ''),
            'image/class/label': tf.io.FixedLenFeature([], tf.int64, 0)
        })
    image_bytes = tf.reshape(parsed['image/encoded'], [])
    image_bytes = tf.io.decode_jpeg(image_bytes, channels=3)
    label = tf.cast(tf.reshape(parsed['image/class/label'], []), tf.int32) - 1
    return image_bytes, label

  def crop_image(image_bytes, label):
    """Preprocess the image."""
    shape = tf.shape(image_bytes)
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
        shape,
        bbox,
        min_object_covered=0.1,
        aspect_ratio_range=(0.75, 1.33),
        area_range=(0.05, 1.0),
        max_attempts=10,
        use_image_if_no_bounding_boxes=True)
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    image = tf.image.crop_to_bounding_box(image_bytes, offset_y, offset_x,
                                          target_height, target_width)
    image = tf.image.resize(
        [image], [image_size, image_size], method='bicubic')[0]
    image = tf.image.random_flip_left_right(image)
    image = tf.reshape(image, [image_size, image_size, 3])
    return tf.image.convert_image_dtype(image, dtype), label

  def set_shapes(images, labels):
    """Statically set the batch_size dimension."""
    if image_format == 'NHWC':
      shape = [batch_size, None, None, None]
    elif image_format == 'HWCN':
      shape = [None, None, None, batch_size]
    elif image_format == 'HWNC':
      shape = [None, None, batch_size, None]
    else:
      raise ValueError('unknown format: {}'.format(image_format))
    images.set_shape(images.get_shape().merge_with(tf.TensorShape(shape)))
    if reshape_to_r1:
      images = tf.reshape(images, [-1])
    labels.set_shape([batch_size])
    return images, labels

  index = jax.host_id()
  num_hosts = jax.host_count()
  replicas_per_host = jax.local_device_count()
  steps = math.ceil(EVAL_IMAGES / (batch_size * replicas_per_host * num_hosts))
  num_dataset_per_shard = max(1, int(steps * batch_size * replicas_per_host))
  padded_dataset = tf.data.Dataset.from_tensors(
      tf.constant(
          tf.train.Example(
              features=tf.train.Features(
                  feature={
                      'image/class/label':
                          tf.train.Feature(
                              int64_list=tf.train.Int64List(value=[0])),
                      'image/encoded':
                          tf.train.Feature(
                              bytes_list=tf.train.BytesList(
                                  value=[str.encode('')]))
                  })).SerializeToString(),
          dtype=tf.string)).repeat(num_dataset_per_shard)

  if FLAGS.data_dir is None:
    dataset = padded_dataset.repeat().map(dataset_parser, DEFAULT_PARALLELISM())
  else:
    if "*" in FLAGS.data_dir:
        train_file_pattern = FLAGS.data_dir
    else:
        train_file_pattern = os.path.join(FLAGS.data_dir, 'train-*')
    val_data_dir = FLAGS.val_data_dir  if FLAGS.val_data_dir else FLAGS.data_dir
    val_file_pattern = os.path.join(val_data_dir, 'validation-*')
    file_pattern = train_file_pattern if train else val_file_pattern
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=False)
    dataset = dataset.shard(num_hosts, index)
    # concurrent_files = min(10, 1024 // num_hosts)
    concurrent_files = DEFAULT_PARALLELISM()
    dataset = dataset.interleave(
        tf.data.TFRecordDataset, concurrent_files, 1, concurrent_files)

    if train:
      if cache_uncompressed:
        dataset = dataset.map(cached_parser, DEFAULT_PARALLELISM()).cache()
        dataset = dataset.shuffle(
            shuffle_size, reshuffle_each_iteration=True).repeat()
        dataset = dataset.map(crop_image, DEFAULT_PARALLELISM())
      else:
        if FLAGS.cache:
            dataset = dataset.cache()  # cache compressed JPEGs instead
        dataset = dataset.shuffle(
            shuffle_size, reshuffle_each_iteration=True).repeat()
        dataset = dataset.map(dataset_parser, DEFAULT_PARALLELISM())
        if FLAGS.randaugment_num_layers:
            randaug_f = lambda image: augmentations.apply_randaugment_to_image(
                    image,
                    randaug_num_layers=FLAGS.randaugment_num_layers,
                    randaug_magnitude=FLAGS.randaugment_magnitude,
                    dtype=dtype)
            logging.info("Adding randaug {}".format(FLAGS.randaugment_num_layers))
            dataset = dataset.map(
                    lambda x, y:
                    (randaug_f(x), y),
                num_parallel_calls=DEFAULT_PARALLELISM())
    else:
      dataset = dataset.concatenate(padded_dataset).take(num_dataset_per_shard)
      dataset = dataset.map(dataset_parser, DEFAULT_PARALLELISM())
  dataset = dataset.batch(batch_size, True)

  mixup_alpha = FLAGS.mixup_alpha
  if train and mixup_alpha and mixup_alpha > 0:
    tf.print("Applying mixup: {}".format(mixup_alpha))
    dataset = dataset.map(
            lambda x, y:
            mixup.mixup(batch_size, mixup_alpha, x, tf.one_hot(y, NUM_CLASSES)),
        num_parallel_calls=DEFAULT_PARALLELISM())
    def set_shapes_mixup(images, labels):
      """Statically set the batch_size dimension."""
      if image_format == 'NHWC':
        shape = [batch_size, None, None, None]
      elif image_format == 'HWCN':
        shape = [None, None, None, batch_size]
      elif image_format == 'HWNC':
        shape = [None, None, batch_size, None]
      else:
        raise ValueError('unknown format: {}'.format(image_format))
      images.set_shape(images.get_shape().merge_with(tf.TensorShape(shape)))
      if reshape_to_r1:
        images = tf.reshape(images, [-1])
      labels.set_shape(labels.get_shape().merge_with(
          tf.TensorShape([batch_size, None])))
      return images, labels
    set_shapes = set_shapes_mixup

  if space_to_depth:
    dataset = dataset.map(
        lambda images, labels: (tf.nn.space_to_depth(images, 2), labels), DEFAULT_PARALLELISM())
  # Transpose for performance on TPU
  if image_format == 'HWCN':
    transpose_array = [1, 2, 3, 0]
  elif image_format == 'HWNC':
    transpose_array = [1, 2, 0, 3]
  if image_format != 'NCHW':
    dataset = dataset.map(
        lambda imgs, labels: (tf.transpose(imgs, transpose_array), labels), DEFAULT_PARALLELISM())
  dataset = dataset.map(set_shapes, DEFAULT_PARALLELISM())

  if FLAGS.echoing_factor and train:
      logging.info("Echoing with {} and shuffle {}".format(
          FLAGS.echoing_factor, FLAGS.echoing_shuffle_buffer_size))
      dataset = dataset_echoing.apply_dataset_echoing(dataset, FLAGS.echoing_factor)
      dataset = dataset.shuffle(FLAGS.echoing_shuffle_buffer_size)

  if not train:
    dataset = dataset.cache().repeat()
  if DEFAULT_PREFETCHING():
    dataset = dataset.prefetch(DEFAULT_PREFETCHING())

  options = tf.data.Options()
  options.experimental_deterministic = False
  options.experimental_threading.max_intra_op_parallelism = 1
  options.experimental_threading.private_threadpool_size = FLAGS.input_pipeline_threadpool_size
  if train:
    gen_util.add_analysis_to_dataset_options(options)
  dataset = dataset.with_options(options)
  return dataset
