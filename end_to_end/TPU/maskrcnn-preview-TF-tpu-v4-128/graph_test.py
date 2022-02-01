from absl import app
from absl import flags

import tensorflow.compat.v1 as tf
import benchmark_mlperf

def main(_):
    graph = tf.Graph()
    print("graph")
    
    with graph.as_default():
        print("ctx")

    dataset_fn = benchmark_mlperf.get_TFRecord_dataset_fn()
    dataset_fn = benchmark_mlperf.apply_options_fn(dataset_fn)

    for x in dataset_fn():
        print(x)
        break

if __name__ == "__main__":
    app.run(main)
