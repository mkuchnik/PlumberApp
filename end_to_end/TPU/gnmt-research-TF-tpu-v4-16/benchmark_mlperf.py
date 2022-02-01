from absl import app
from absl import flags

import tensorflow.compat.v1 as tf1
import tensorflow as tf
#tf.disable_eager_execution()
from utils import iterator_utils, vocab_utils, misc_utils
import six
import estimator
from compat import hparam as compat_hparam
import argparse
import sys
import dataset_flags

from plumber_analysis import gen_util, annotations

def create_hparams(flags):
  """Create training hparams."""

  if flags.use_preprocessed_data:
    # NOTE(mkuchnik): preprocessed data used different path
    train_prefix = flags.data_dir + "preprocessed"
  else:
    train_prefix = flags.data_dir + flags.train_prefix
  return compat_hparam.HParams(
      # Data
      src=flags.src,
      tgt=flags.tgt,
      train_prefix=train_prefix,
      test_prefix=flags.data_dir + flags.test_prefix,
      vocab_prefix=flags.data_dir + flags.vocab_prefix,
      out_dir=flags.out_dir,

      # Networks
      num_units=flags.num_units,
      num_encoder_layers=(flags.num_encoder_layers or flags.num_layers),
      num_decoder_layers=(flags.num_decoder_layers or flags.num_layers),
      dropout=flags.dropout,
      num_embeddings_partitions=flags.num_embeddings_partitions,

      # Train
      optimizer=flags.optimizer,
      max_train_epochs=flags.max_train_epochs,
      num_examples_per_epoch=flags.num_examples_per_epoch,
      batch_size=flags.batch_size,
      num_train_steps=int(flags.num_examples_per_epoch / flags.batch_size *
                          flags.max_train_epochs),
      init_op=flags.init_op,
      init_weight=flags.init_weight,
      max_gradient_norm=flags.max_gradient_norm,
      learning_rate=flags.learning_rate,
      label_smoothing=flags.label_smoothing,
      warmup_steps=flags.warmup_steps,
      warmup_scheme=flags.warmup_scheme,
      decay_start=flags.decay_start,
      decay_interval=flags.decay_interval,
      decay_steps=flags.decay_steps,
      decay_factor=flags.decay_factor,

      # Data constraints
      num_buckets=flags.num_buckets,
      choose_buckets=flags.choose_buckets,
      src_max_len=flags.src_max_len,
      tgt_max_len=flags.tgt_max_len,
      use_preprocessed_data=flags.use_preprocessed_data,

      # Inference
      src_max_len_infer=flags.src_max_len_infer,
      tgt_max_len_infer=flags.tgt_max_len_infer,
      infer_batch_size=flags.infer_batch_size,
      examples_to_infer=flags.examples_to_infer,
      detokenizer_file=flags.data_dir + flags.detokenizer_file,
      use_REDACTED=flags.use_REDACTED,
      target_bleu=flags.target_bleu,

      # Advanced inference arguments
      infer_mode=flags.infer_mode,
      beam_width=flags.beam_width,
      length_penalty_weight=flags.length_penalty_weight,
      coverage_penalty_weight=flags.coverage_penalty_weight,

      # Vocab
      sos=vocab_utils.SOS,
      eos=vocab_utils.EOS,
      subword_option=flags.subword_option,
      check_special_token=flags.check_special_token,

      # Misc
      forget_bias=flags.forget_bias,
      num_shards=flags.num_shards,
      num_shards_per_host=flags.num_shards_per_host,
      num_gpus=flags.num_gpus,
      num_infeed_workers=flags.num_infeed_workers,
      epoch_step=0,  # record where we were within an epoch.
      steps_per_stats=flags.steps_per_stats,
      random_seed=flags.random_seed,

      # TPU
      use_tpu=flags.use_tpu,
      master=flags.master,
      tpu_name=flags.tpu_name,
      use_dynamic_rnn=flags.use_dynamic_rnn,
      use_synthetic_data=flags.use_synthetic_data,
      mode=flags.mode,
      activation_dtype=flags.activation_dtype,
      tpu_job_name=flags.tpu_job_name)

def _add_argument(hparams, key, value, update=True):
  """Add an argument to hparams; if exists, change the value if update==True."""
  if hasattr(hparams, key):
    if update:
      setattr(hparams, key, value)
  else:
    hparams.add_hparam(key, value)


def extend_hparams(hparams):
  """Add new arguments to hparams."""
  # Sanity checks
  if hparams.subword_option and hparams.subword_option not in ["spm", "bpe"]:
    raise ValueError("subword option must be either spm, or bpe")
  if hparams.infer_mode == "beam_search" and hparams.beam_width <= 0:
    raise ValueError("beam_width must greater than 0 when using beam_search"
                     "decoder.")

  # Different number of encoder / decoder layers
  assert hparams.num_encoder_layers == hparams.num_decoder_layers

  # The first unidirectional layer (after the bi-directional layer) in
  # the GNMT encoder can't have residual connection due to the input is
  # the concatenation of fw_cell and bw_cell's outputs.
  num_encoder_residual_layers = hparams.num_encoder_layers - 2
  num_decoder_residual_layers = num_encoder_residual_layers
  _add_argument(hparams, "num_encoder_residual_layers",
                num_encoder_residual_layers)
  _add_argument(hparams, "num_decoder_residual_layers",
                num_decoder_residual_layers)

  ## Vocab
  # Get vocab file names first
  if hparams.vocab_prefix:
    src_vocab_file = six.ensure_str(
        hparams.vocab_prefix) + "." + six.ensure_str(hparams.src)
    tgt_vocab_file = six.ensure_str(
        hparams.vocab_prefix) + "." + six.ensure_str(hparams.tgt)
    # TODO(mkuchnik): For some reason, '.en' is not appended to file?
    src_vocab_file = six.ensure_str(
        hparams.vocab_prefix)
    tgt_vocab_file = six.ensure_str(
        hparams.vocab_prefix)
  else:
    raise ValueError("hparams.vocab_prefix must be provided.")

  # Source vocab
  src_vocab_size, src_vocab_file = vocab_utils.check_vocab(
      src_vocab_file,
      hparams.out_dir,
      check_special_token=hparams.check_special_token,
      sos=hparams.sos,
      eos=hparams.eos,
      unk=vocab_utils.UNK)

  # Target vocab
  misc_utils.print_out("  using source vocab for target")
  tgt_vocab_file = src_vocab_file
  tgt_vocab_size = src_vocab_size
  _add_argument(hparams, "src_vocab_size", src_vocab_size)
  _add_argument(hparams, "tgt_vocab_size", tgt_vocab_size)
  _add_argument(hparams, "src_vocab_file", src_vocab_file)
  _add_argument(hparams, "tgt_vocab_file", tgt_vocab_file)

  # Num embedding partitions
  _add_argument(
      hparams, "num_enc_emb_partitions", hparams.num_embeddings_partitions)
  _add_argument(
      hparams, "num_dec_emb_partitions", hparams.num_embeddings_partitions)

  # Pretrained Embeddings
  _add_argument(hparams, "src_embed_file", "")
  _add_argument(hparams, "tgt_embed_file", "")

  return hparams


def create_or_load_hparams(default_hparams, hparams_path):
  """Create hparams or load hparams from out_dir."""
  hparams = utils.maybe_parse_standard_hparams(default_hparams, hparams_path)
  hparams = extend_hparams(hparams)
  # Print HParams
  misc_utils.print_hparams(hparams)
  return hparams

def get_vocab_tables(flags):
  src_file = "%s.%s" % (flags.data_dir + flags.train_prefix, flags.src)
  tgt_file = "%s.%s" % (flags.data_dir + flags.train_prefix, flags.tgt)
  vocab_file = flags.data_dir + flags.vocab_prefix
  _, vocab_file = vocab_utils.check_vocab(vocab_file, flags.out_dir)
  out_file = six.ensure_str(flags.out_dir) + "preprocessed_dataset"
  src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(vocab_file)
  return src_vocab_table, tgt_vocab_table


def prepare_dataset(flags):
  """Generate the preprocessed dataset."""
  src_file = "%s.%s" % (flags.data_dir + flags.train_prefix, flags.src)
  tgt_file = "%s.%s" % (flags.data_dir + flags.train_prefix, flags.tgt)
  vocab_file = flags.data_dir + flags.vocab_prefix
  _, vocab_file = vocab_utils.check_vocab(vocab_file, flags.out_dir)
  out_file = six.ensure_str(flags.out_dir) + "preprocessed_dataset"
  src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(vocab_file)
  src_dataset = tf.data.TextLineDataset(src_file)
  tgt_dataset = tf.data.TextLineDataset(tgt_file)
  iterator = iterator_utils.get_iterator(
      src_dataset,
      tgt_dataset,
      src_vocab_table,
      tgt_vocab_table,
      batch_size=1,
      global_batch_size=1,
      sos=vocab_utils.SOS,
      eos=vocab_utils.EOS,
      random_seed=1,
      num_buckets=flags.num_buckets,
      src_max_len=flags.src_max_len,
      tgt_max_len=flags.tgt_max_len,
      filter_oversized_sequences=True,
      return_raw=True)

  iterator = iterator.make_initializable_iterator()

  with tf.Session() as sess:
    sess.run(tf.tables_initializer())
    sess.run(iterator.initializer)
    try:
      i = 0
      while True:
        with open(out_file + "_%d" % i, "wb") as f:
          i += 1
          for _ in range(100):
            for j in sess.run(iterator.get_next()):
              tf.logging.info(j)
              f.write(bytearray(j))
    except tf.errors.OutOfRangeError:
      pass

def create_or_load_hparams(default_hparams, hparams_path):
  """Create hparams or load hparams from out_dir."""
  hparams = misc_utils.maybe_parse_standard_hparams(default_hparams, hparams_path)
  hparams = extend_hparams(hparams)
  # Print HParams
  misc_utils.print_hparams(hparams)
  return hparams

#@annotations.trace_pipeline()
def get_TFRecord_dataset(FLAGS):
    FLAGS.out_dir = ""
    if FLAGS.mode == "preprocess":
        tf.print("Preprocessing")
        prepare_dataset(FLAGS)
        return None
    else:
        default_hparams = create_hparams(FLAGS)
        # Load hparams.
        hparams = create_or_load_hparams(default_hparams, FLAGS.hparams_path)
        mode = tf.estimator.ModeKeys.TRAIN
        hparams.other_args = FLAGS
        train_fn = estimator.make_input_fn(hparams, mode)
        tf.print("train_fn: {}".format(train_fn))
        params = {}
        dataset = train_fn(params)
        return dataset

def main(_):
    nmt_parser = argparse.ArgumentParser()

    dataset_flags.add_arguments(nmt_parser)
    nmt_parser.add_argument(
        '--profile', type=bool, default=False,
        help=('Whether to profile'))

    FLAGS, unparsed = nmt_parser.parse_known_args()
    dataset = get_TFRecord_dataset(FLAGS)
    dataset = dataset.take(FLAGS.benchmark_num_elements)
    options = tf.data.Options()
    options.experimental_deterministic = False
    options.experimental_threading.max_intra_op_parallelism = 1
    options.experimental_threading.private_threadpool_size = FLAGS.dataset_threadpool_size
    options.experimental_optimization.map_and_batch_fusion = FLAGS.map_and_batch_fusion
    gen_util.add_analysis_to_dataset_options(options)
    dataset = dataset.with_options(options)
    if FLAGS.profile:
        summary = gen_util.benchmark_and_profile_dataset(
            dataset, time_limit_s=FLAGS.time_limit_s)
    else:
        summary, df = gen_util.benchmark_dataset(
            dataset, time_limit_s=FLAGS.time_limit_s, profile_interval=10,
            return_monitoring_data=True)
        df.to_csv("monitoring.csv", index=False)

if __name__ == "__main__":
  tf1.logging.set_verbosity(tf1.logging.INFO)
  tf1.app.run(main=main, argv=[sys.argv[0]])
