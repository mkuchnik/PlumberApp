from absl import app
from absl import flags

import argparse
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

def main():
    nmt_parser = argparse.ArgumentParser()
    dataset_flags.add_arguments(nmt_parser)
    graph_rewrites.apply_parser_flags(nmt_parser, skip_time_limit_s=True)
    graph_rewrites.default_main(nmt_parser)

if __name__ == '__main__':
    main()