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

import functools
import math
import os
from absl import flags
import tensorflow.compat.v1 as tf
import tensorflow as tf2

FLAGS = flags.FLAGS
CROP_PADDING = 32


def get_input_fn(data_dir,
                 is_training,
                 dtype,
                 image_size,
                 reshape_to_r1,
                 shuffle_size=16384):
  """Returns the input_fn."""

  def dataset_parser(value):
    """Parses an image and its label from a serialized ResNet-50 TFExample."""
    parsed = tf.parse_single_example(
        value, {
            'image/encoded': tf.FixedLenFeature((), tf.string, ''),
            'image/class/label': tf.FixedLenFeature([], tf.int64, -1)
        })
    image_bytes = tf.reshape(parsed['image/encoded'], [])
    label = tf.cast(tf.reshape(parsed['image/class/label'], []), tf.int32) - 1

    def preprocess_fn():
      """Preprocess the image."""
      shape = tf.image.extract_jpeg_shape(image_bytes)
      if is_training:
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
      image = tf.image.resize_bicubic([image], [image_size, image_size])[0]
      if is_training:
        image = tf.image.random_flip_left_right(image)
      image = tf.reshape(image, [image_size, image_size, 3])
      return tf.image.convert_image_dtype(image, dtype)

    empty_example = tf.zeros([image_size, image_size, 3], dtype)
    return tf.cond(label < 0, lambda: empty_example, preprocess_fn), label

  def cached_parser(value):
    """Parses an image and its label from a serialized ResNet-50 TFExample."""
    parsed = tf.parse_single_example(
        value, {
            'image/encoded': tf.FixedLenFeature((), tf.string, ''),
            'image/class/label': tf.FixedLenFeature([], tf.int64, -1)
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
        area_range=(0.08, 1.0),
        max_attempts=10,
        use_image_if_no_bounding_boxes=True)
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    image = tf.image.crop_to_bounding_box(image_bytes, offset_y, offset_x,
                                          target_height, target_width)
    image = tf.image.resize_bicubic([image], [image_size, image_size])[0]
    image = tf.image.random_flip_left_right(image)
    image = tf.reshape(image, [image_size, image_size, 3])
    return tf.image.convert_image_dtype(image, dtype), label

  def set_shapes(batch_size, images, labels):
    """Statically set the batch_size dimension."""
    if FLAGS.train_batch_size // FLAGS.num_replicas > 8:
      shape = [None, None, None, batch_size]
    else:
      shape = [None, None, batch_size, None]
    images.set_shape(images.get_shape().merge_with(tf.TensorShape(shape)))
    if reshape_to_r1:
      images = tf.reshape(images, [-1])
    labels.set_shape([batch_size])
    return images, labels

  def input_fn(params):
    """Input function which provides a single batch for train or eval."""
    batch_size = params['batch_size']
    index = params['dataset_index']
    num_hosts = params['dataset_num_shards']
    num_dataset_per_shard = max(
        1,
        int(
            math.ceil(FLAGS.num_eval_images / FLAGS.eval_batch_size) *
            FLAGS.eval_batch_size / num_hosts))
    padded_dataset = tf.data.Dataset.from_tensors(
        tf.constant(
            tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'image/class/label':
                            tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[-1])),
                        'image/encoded':
                            tf.train.Feature(
                                bytes_list=tf.train.BytesList(
                                    value=[str.encode('')]))
                    })).SerializeToString(),
            dtype=tf.string)).repeat(num_dataset_per_shard)

    if data_dir is None:
      tf.print("Using synthetic data")
      dataset = padded_dataset.repeat(batch_size * 100)
    else:
      file_pattern = os.path.join(data_dir,
                                  'train-*' if is_training else 'validation-*')
      dataset_files = tf2.data.Dataset.list_files(file_pattern, shuffle=False)
      dataset = dataset_files
      dataset = dataset.shard(num_hosts, index)
      dataset = dataset.interleave(tf.data.TFRecordDataset,
                                   FLAGS.read_parallelism, 1,
                                   FLAGS.read_parallelism)

      tf2.print("p_cached", FLAGS.percentage_cached)
      if is_training:
        #dataset = dataset.cache().shuffle(shuffle_size).repeat(100)
        if FLAGS.cache_records:
            if FLAGS.percentage_cached:
                p_cache = FLAGS.percentage_cached
                total_cached = int(len(dataset_files) * p_cache)
                tf2.print("Partially caching: {}/{} files."
                      .format(total_cached, len(dataset_files)))
                curr_dataset = dataset
                partition_size = int(len(dataset_files) / total_cached)
                num_partitions = math.ceil(len(dataset_files) /
                                           partition_size)
                all_datasets = []
                for i in range(num_partitions - 1):
                    files_processed = i * partition_size
                    inner_dataset = curr_dataset.skip(files_processed)
                    disk_dataset = inner_dataset.take(partition_size - 1)
                    memory_dataset = inner_dataset.take(1).cache()
                    combined_dataset = disk_dataset.concatenate(memory_dataset)
                    all_datasets.append(combined_dataset)
                files_processed = (num_partitions - 1) * partition_size
                inner_dataset = curr_dataset.skip(files_processed)
                disk_dataset = inner_dataset
                for ds in all_datasets:
                    disk_dataset = disk_dataset.concatenate(ds)
                dataset = disk_dataset.shuffle(shuffle_size).repeat()
            else:
                dataset = dataset.take(500).cache().shuffle(shuffle_size).repeat()
        else:
            dataset = dataset.shuffle(shuffle_size).repeat()
      else:
        dataset = dataset.concatenate(padded_dataset).take(
            num_dataset_per_shard)

    if FLAGS.filter_num_classes:
          dataset = dataset.map(cached_parser,
                                FLAGS.map_parse_parallelism)
          dataset = dataset.filter(lambda x, y: y < FLAGS.filter_num_classes)
          dataset = dataset.repeat()
          dataset = dataset.map(crop_image,
                                FLAGS.map_crop_parallelism)
          dataset = dataset.batch(batch_size, True)
    else:
        if FLAGS.use_cache and is_training:
          dataset = (dataset.map(cached_parser,
                                 FLAGS.map_parse_parallelism)
                           .repeat()
                           .map(crop_image,
                                FLAGS.map_crop_parallelism)
                           .batch(batch_size, True))
        else:
          dataset = (dataset.map(dataset_parser, FLAGS.map_parse_parallelism)
                           .batch(batch_size, True))

    if FLAGS.use_space_to_depth:
      dataset = dataset.map(
          lambda images, labels: (tf.space_to_depth(images, 2), labels),
          FLAGS.map_image_postprocessing_parallelism)
    # Transpose for performance on TPU
    if FLAGS.train_batch_size // FLAGS.num_replicas > 8:
      transpose_array = [1, 2, 3, 0]
    else:
      transpose_array = [1, 2, 0, 3]
    dataset = dataset.map(
        lambda imgs, labels: (tf.transpose(imgs, transpose_array), labels),
        FLAGS.map_image_transpose_postprocessing_parallelism)
    dataset = dataset.map(functools.partial(set_shapes, batch_size), FLAGS.map_image_postprocessing_parallelism)
    dataset = dataset.prefetch(10)

    options = tf.data.Options()
    options.experimental_deterministic = False
    options.experimental_threading.max_intra_op_parallelism = 1
    options.experimental_threading.private_threadpool_size = FLAGS.dataset_threadpool_size
    options.experimental_optimization.map_and_batch_fusion = \
    FLAGS.map_and_batch_fusion
    dataset = dataset.with_options(options)
    return dataset

  return input_fn
