from absl import app
from absl import flags

import tensorflow as tf

import pandas as pd
from plumber_analysis import graph_rewrites
import dataset_flags


FLAGS = flags.FLAGS
graph_rewrites.apply_default_flags()

def main(_):
    graph_rewrites.default_main(_)

if __name__ == '__main__':
  app.run(main)
