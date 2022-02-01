from absl import app
from absl import flags

import tensorflow as tf

import pandas as pd
from plumber_analysis import graph_rewrites

import tokenizer


FLAGS = flags.FLAGS
graph_rewrites.apply_default_flags()

flags.DEFINE_bool(
    'map_and_batch_fusion', default=True,
    help=('Enables tf.data options map and batch fusion'))
flags.DEFINE_integer(
    'dataset_threadpool_size', default=48,
    help=('The size of the private datapool size in dataset.'))

def main(_):
    graph_rewrites.default_main(_)

if __name__ == '__main__':
  app.run(main)
