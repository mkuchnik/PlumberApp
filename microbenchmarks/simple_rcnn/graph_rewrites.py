from absl import app
from absl import flags

import tensorflow as tf

import pandas as pd
try:
    import dataloader
except ImportError:
    try:
        import resnet_flags
    except ImportError:
        import dataset_flags
from plumber_analysis import graph_rewrites


FLAGS = flags.FLAGS
graph_rewrites.apply_default_flags()

def main(_):
    graph_rewrites.default_main(_)

if __name__ == '__main__':
  app.run(main)