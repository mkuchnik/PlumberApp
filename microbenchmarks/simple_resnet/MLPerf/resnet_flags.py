from absl import app
from absl import flags

import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS

# Model specific flags
flags.DEFINE_string(
    'data_dir', default=None,
    help=('The directory where the ImageNet input data is stored. Please see'
          ' the README.md for the expected data format.'))

flags.DEFINE_integer(
    'resnet_depth', default=50,
    help=('Depth of ResNet model to use. Must be one of {18, 34, 50, 101, 152,'
          ' 200}. ResNet-18 and 34 use the pre-activation residual blocks'
          ' without bottleneck layers. The other models use pre-activation'
          ' bottleneck layers. Deeper models require more training time and'
          ' more memory and may require reducing --train_batch_size to prevent'
          ' running out of memory.'))

flags.DEFINE_integer(
    'train_steps', default=112590,
    help=('The number of steps to use for training. Default is 112590 steps'
          ' which is approximately 90 epochs at batch size 1024. This flag'
          ' should be adjusted according to the --train_batch_size flag.'))

flags.DEFINE_integer(
    'train_batch_size', default=1024, help='Batch size for training.')

flags.DEFINE_integer(
    'eval_batch_size', default=1024, help='Batch size for evaluation.')

flags.DEFINE_integer(
    'num_train_images', default=1281167, help='Size of training data set.')

flags.DEFINE_integer(
    'num_eval_images', default=50000, help='Size of evaluation data set.')

flags.DEFINE_integer(
    'num_label_classes', default=1000, help='Number of classes, at least 2')

flags.DEFINE_integer(
    'steps_per_eval', default=1251,
    help=('Controls how often evaluation is performed. Since evaluation is'
          ' fairly expensive, it is advised to evaluate as infrequently as'
          ' possible (i.e. up to --train_steps, which evaluates the model only'
          ' after finishing the entire training regime).'))

flags.DEFINE_integer(
    'iterations_per_loop', default=1251,
    help=('Number of steps to run on TPU before outfeeding metrics to the CPU.'
          ' If the number of iterations in the loop would exceed the number of'
          ' train steps, the loop will exit before reaching'
          ' --iterations_per_loop. The larger this value is, the higher the'
          ' utilization on the TPU.'))

flags.DEFINE_integer('num_replicas', default=8, help=('Number of replicas.'))

flags.DEFINE_string(
    'precision', default='bfloat16',
    help=('Precision to use; one of: {bfloat16, float32}'))

flags.DEFINE_float(
    'base_learning_rate', default=0.1,
    help=('Base learning rate when train batch size is 256.'))

flags.DEFINE_float(
    'momentum', default=0.9,
    help=('Momentum parameter used in the MomentumOptimizer.'))

flags.DEFINE_float(
    'weight_decay', default=1e-4,
    help=('Weight decay coefficiant for l2 regularization.'))

flags.DEFINE_float(
    'label_smoothing', default=0.0,
    help=('Label smoothing parameter used in the softmax_cross_entropy'))

flags.DEFINE_bool('enable_lars',
                  default=False,
                  help=('Enable LARS optimizer for large batch training.'))

flags.DEFINE_float('poly_rate', default=0.0,
                   help=('Set LARS/Poly learning rate.'))

flags.DEFINE_float(
    'stop_threshold', default=0.759, help=('Stop threshold for MLPerf.'))

flags.DEFINE_integer('image_size', 224, 'The input image size.')

flags.DEFINE_integer(
    'distributed_group_size',
    default=1,
    help=('When set to > 1, it will enable distributed batch normalization'))

tf.flags.DEFINE_multi_integer(
    'input_partition_dims',
    default=None,
    help=('Number of partitions on each dimension of the input. Each TPU core'
          ' processes a partition of the input image in parallel using spatial'
          ' partitioning.'))

flags.DEFINE_bool(
    'use_space_to_depth',
    default=False,
    help=('Enable space-to-depth optimization for conv-0.'))

flags.DEFINE_bool(
    'use_cache', default=False, help=('Enable cache for training input.'))

flags.DEFINE_bool(
    'cache_records', default=False, help=('Enable cache for data only.'))

flags.DEFINE_float(
    'percentage_cached', default=None, help=('Partial caching percentage'))

flags.DEFINE_integer(
    'dataset_threadpool_size', default=48,
    help=('The size of the private datapool size in dataset.'))

flags.DEFINE_integer(
    'read_parallelism', default=64,
    help=(''))

flags.DEFINE_integer(
    'map_parse_parallelism', default=64,
    help=(''))

flags.DEFINE_integer(
    'map_crop_parallelism', default=64,
    help=(''))

flags.DEFINE_integer(
    'map_image_postprocessing_parallelism', default=64,
    help=(''))

flags.DEFINE_integer(
    'map_image_transpose_postprocessing_parallelism', default=64,
    help=(''))

flags.DEFINE_bool(
    'convert_spectral', default=False,
    help=('Converts progressive images to spectral form'))

flags.DEFINE_bool(
    'use_PCR', default=False,
    help=('Use Progressive Compressed Loader'))

flags.DEFINE_integer(
    'benchmark_num_elements', default=500,
    help=('The number of elements to use for the benchmark'))

flags.DEFINE_integer(
    'shuffle_size', default=16384,
    help=('The shuffle buffer size'))

flags.DEFINE_bool(
    'map_and_batch_fusion', default=True,
    help=('Enables tf.data options map and batch fusion'))

flags.DEFINE_integer(
    'filter_num_classes',
    default=None,
    help=("The number of classes to filter"))

flags.DEFINE_integer(
    'batch_parallelism', default=None,
    help=(''))