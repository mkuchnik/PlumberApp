from absl import app
from absl import flags

import tensorflow as tf
import tensorflow.compat.v1 as tf1
#tf1.disable_eager_execution()
import dataloader
from plumber_analysis import gen_util, pipeline_optimizer_wrapper, pipeline_optimizer
import mask_rcnn_params

pipeline_optimizer.DEFAULT_BENCHMARK_TIME = 62
pipeline_optimizer_wrapper.BENCHMARK_TIME = 62

FLAGS = flags.FLAGS

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
tf1.flags.DEFINE_integer('train_batch_size', 128, 'training batch size')
tf1.flags.DEFINE_integer('eval_batch_size', 128, 'evaluation batch size')
tf1.flags.DEFINE_integer('eval_samples', 5000, 'The number of samples for '
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
tf1.flags.DEFINE_integer('num_examples_per_epoch', 118287,
                        'Number of examples in one epoch')
tf1.flags.DEFINE_integer('num_epochs', 15, 'Number of epochs for training')
flags.DEFINE_string('mode', 'train',
                    'Mode to run: train or eval (default: train)')
flags.DEFINE_bool('eval_after_training', False, 'Run one eval after the '
                  'training finishes.')
flags.DEFINE_bool('use_fake_data', False, 'Use fake input.')

# For Eval mode
flags.DEFINE_integer('min_eval_interval', 180,
                     'Minimum seconds between evaluations.')
flags.DEFINE_string(
    "master",
    default=None,
    help="The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.")

flags.DEFINE_string(
    "gcp_project",
    default=None,
    help="Project name for the Cloud TPU-enabled project. If not specified, we "
    "will attempt to automatically detect the GCE project from metadata.")

flags.DEFINE_string(
    "tpu_zone",
    default=None,
    help="GCE zone where the Cloud TPU is located in. If not specified, we "
    "will attempt to automatically detect the GCE project from metadata.")

flags.DEFINE_integer(
    "replicas_per_host", default=8, help=("Number of replicas per host."))

flags.DEFINE_bool("enable_summary", default=False, help=("Enable summary"))

flags.DEFINE_string(
    "model_dir",
    default=None,
    help=("The directory where the model and summaries are stored."))

flags.DEFINE_bool("save_checkpoint", default=False, help=("Save checkpoint"))

flags.DEFINE_bool(
    "restore_checkpoint", default=False, help=("Restore checkpoint"))

flags.DEFINE_integer(
    "sleep_after_init", default=60, help=("Sleep for N seconds after init."))

flags.DEFINE_bool(
    "enable_mlir_bridge", default=False, help=("Enable TF/XLA MLIR bridge"))

flags.DEFINE_bool(
    "enable_profiling",
    default=False,
    help=("Get xprof traces at"
          "the start and middle of the train loops"))

flags.DEFINE_string(
    "init_dummy_file",
    default=None,
    help="Read a dummy file to initialize datacenter connection.")
flags.DEFINE_integer(
    'time_limit_s', default=None,
    help=('Number of seconds to run for'))
flags.DEFINE_bool(
    'profile', default=False,
    help=('Whether to profile'))

def get_loader_fn():
    training_file_pattern = FLAGS.training_file_pattern 
    use_fake_data = False
    train_input_fn = dataloader.InputReader(
        training_file_pattern,
        mode=tf.estimator.ModeKeys.TRAIN,
        use_fake_data=use_fake_data)
    return train_input_fn

def get_dataset():
    dataset_fn = get_loader_fn()
    hparams = mask_rcnn_params.default_hparams()
    hparams.parse(FLAGS.hparams)
    params = dict(
        hparams.values(),
        transpose_input=False if FLAGS.input_partition_dims is not None else True,
        resnet_checkpoint=FLAGS.resnet_checkpoint,
        val_json_file=FLAGS.val_json_file,
        num_cores_per_replica=int(np.prod(FLAGS.input_partition_dims))
        if FLAGS.input_partition_dims else 1,
        replicas_per_host=FLAGS.replicas_per_host)
    params["batch_size"] = FLAGS.train_batch_size
    dataset = dataset_fn(params)
    return dataset

def get_TFRecord_dataset():
    dataset = get_dataset()
    return dataset

def get_TFRecord_dataset_fn():
    # NOTE(muchnik): Necessary for graph-mode
    return get_dataset

def apply_options_fn(fn):
    def get_options_dataset_fn():
        options = tf.data.Options()
        options.experimental_deterministic = False
        options.experimental_threading.max_intra_op_parallelism = 1
        # TODO(mkuchnik): Don't hardcode
        options.experimental_threading.private_threadpool_size = 96
        options.experimental_optimization.autotune_stats_filename = "stats.pb"
        #options.experimental_optimization.autotune_stats_dump_period = 1000
        dataset = fn()
        dataset = dataset.with_options(options)
        return dataset
    return get_options_dataset_fn

def main(_):
    print("Main enter")
    dataset_fn = get_TFRecord_dataset_fn()
    dataset_fn = apply_options_fn(dataset_fn)
    dataset = dataset_fn()
    batch_size = FLAGS.train_batch_size
    print("Dataset constructed")
    if FLAGS.profile:
        print("Running benchmark and profile")
        #summary = gen_util.benchmark_and_profile_dataset_fn(
        #    dataset_fn, time_limit_s=FLAGS.time_limit_s)
        summary = gen_util.benchmark_and_profile_dataset(
            dataset, time_limit_s=FLAGS.time_limit_s, element_count_f=lambda x: batch_size)
    else:
        print("Running benchmark")
        #summary = gen_util.benchmark_dataset_fn(
        #    dataset_fn, time_limit_s=FLAGS.time_limit_s)
        summary = gen_util.benchmark_dataset(
            dataset, time_limit_s=FLAGS.time_limit_s, element_count_f=lambda x: batch_size)

if __name__ == '__main__':
  app.run(main)
