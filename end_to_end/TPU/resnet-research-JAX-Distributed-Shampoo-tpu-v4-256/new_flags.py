from absl import flags

flags.DEFINE_bool(
    'cache', default=True,
    help='Turns on caching for JPEG')

flags.DEFINE_bool(
    'trace_iter', default=False,
    help='Turns on tracing for iterator')

flags.DEFINE_bool(
    'force_cache_uncompressed', default=False,
    help='Turns on caching for de-compressed JPEG')

flags.DEFINE_float(
    'memory_bloat_percentage', default=None,
    help='The fraction of memory to occupy with bloat'
)

flags.DEFINE_integer(
    'resnet_depth', default=50,
    help='ResNet depth (default: ResNet50).')

flags.DEFINE_bool(
    'no_eval', default=True,
    help='Turns off evaluation')

flags.DEFINE_bool(
    'optimize_plumber_pipeline', default=False,
    help='Enables plumber rewrites')

flags.DEFINE_integer(
    'local_batch_size', default=128,
    help='Local batch size for training.')

flags.DEFINE_integer(
    'echoing_factor', default=None,
    help='Dataset echoing factor.')

flags.DEFINE_integer(
    'echoing_shuffle_buffer_size', default=None,
    help='Dataset echoing shuffle buffer size.')

flags.DEFINE_float(
    'mixup_alpha', default=None,
    help='Coefficient for mixup at train time.')

flags.DEFINE_integer(
    'randaugment_magnitude', default=None,
    help='RandAugment magnitude.')

flags.DEFINE_integer(
    'randaugment_num_layers', default=None,
    help='RandAugment number layers.')
