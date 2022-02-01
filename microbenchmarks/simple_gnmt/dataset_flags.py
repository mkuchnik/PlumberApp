
def add_arguments(parser):
  """Build ArgumentParser."""
  parser.register("type", "bool", lambda v: v.lower() == "true")

  # network
  parser.add_argument(
      "--num_units", type=int, default=1024, help="Network size.")
  parser.add_argument(
      "--num_layers", type=int, default=4, help="Network depth.")
  parser.add_argument("--num_encoder_layers", type=int, default=None,
                      help="Encoder depth, equal to num_layers if None.")
  parser.add_argument("--num_decoder_layers", type=int, default=None,
                      help="Decoder depth, equal to num_layers if None.")
  parser.add_argument("--num_embeddings_partitions", type=int, default=0,
                      help="Number of partitions for embedding vars.")

  # optimizer
  parser.add_argument(
      "--optimizer", type=str, default="adam", help="sgd | adam")
  parser.add_argument(
      "--learning_rate",
      type=float,
      default=0.001,
      help="Learning rate. Adam: 0.001 | 0.0001")
  parser.add_argument(
      "--warmup_steps",
      type=int,
      default=200,
      help="How many steps we inverse-decay learning.")
  parser.add_argument("--warmup_scheme", type=str, default="t2t", help="""\
      How to warmup learning rates. Options include:
        t2t: Tensor2Tensor's way, start with lr 100 times smaller, then
             exponentiate until the specified lr.\
      """)
  parser.add_argument(
      "--decay_start", type=int, default=3000, help="step to start decay")
  parser.add_argument(
      "--decay_interval",
      type=int,
      default=400,
      help="interval steps between 2 decays")
  parser.add_argument(
      "--decay_steps", type=int, default=5, help="number of decays")
  parser.add_argument(
      "--decay_factor", type=float, default=0.66, help="decay rate")

  parser.add_argument(
      "--max_train_epochs", type=int, default=8,
      help="Maximum number of training epochs.")
  parser.add_argument("--num_examples_per_epoch", type=int, default=3442299,
                      help="Number of examples in one epoch")
  parser.add_argument("--label_smoothing", type=float, default=0.1,
                      help=("If nonzero, smooth the labels towards "
                            "1/num_classes."))

  # initializer
  parser.add_argument("--init_op", type=str, default="uniform",
                      help="uniform | glorot_normal | glorot_uniform")
  parser.add_argument("--init_weight", type=float, default=0.1,
                      help=("for uniform init_op, initialize weights "
                            "between [-this, this]."))

  # data
  parser.add_argument(
      "--src", type=str, default="en", help="Source suffix, e.g., en.")
  parser.add_argument(
      "--tgt", type=str, default="de", help="Target suffix, e.g., de.")
  parser.add_argument(
      "--data_dir", type=str, default="", help="Training/eval data directory.")
  parser.add_argument(
      "--local_data_dir",
      type=str,
      default="",
      help="Training/eval data directory.")

  parser.add_argument(
      "--train_prefix",
      type=str,
      default="train.tok.clean.bpe.32000",
      help="Train prefix, expect files with src/tgt suffixes.")
  parser.add_argument(
      "--test_prefix",
      type=str,
      default="newstest2014",
      help="Test prefix, expect files with src/tgt suffixes.")
  parser.add_argument(
      "--use_preprocessed_data",
      type="bool",
      default=True,
      help="Whether to use preprocessed training data.")

  parser.add_argument(
      "--out_dir", type=str, default=None, help="Store log/model files.")

  # Vocab
  parser.add_argument(
      "--vocab_prefix",
      type=str,
      default="vocab.bpe.32000",
      help="""\
      Vocab prefix, expect files with src/tgt suffixes.\
      """)

  parser.add_argument("--check_special_token", type="bool", default=True,
                      help="""\
                      Whether check special sos, eos, unk tokens exist in the
                      vocab files.\
                      """)

  # Sequence lengths
  parser.add_argument(
      "--src_max_len",
      type=int,
      default=48,
      help="Max length of src sequences during training.")
  parser.add_argument(
      "--tgt_max_len",
      type=int,
      default=48,
      help="Max length of tgt sequences during training.")
  parser.add_argument(
      "--src_max_len_infer",
      type=int,
      default=160,
      help="Max length of src sequences during inference.")
  parser.add_argument(
      "--tgt_max_len_infer",
      type=int,
      default=160,
      help="""\
      Max length of tgt sequences during inference.  Also use to restrict the
      maximum decoding length.\
      """)

  # Default settings works well (rarely need to change)
  parser.add_argument("--forget_bias", type=float, default=0.0,
                      help="Forget bias for BasicLSTMCell.")
  parser.add_argument("--dropout", type=float, default=0.2,
                      help="Dropout rate (not keep_prob)")
  parser.add_argument("--max_gradient_norm", type=float, default=5.0,
                      help="Clip gradients to this norm.")
  parser.add_argument("--batch_size", type=int, default=512, help="Batch size.")

  parser.add_argument("--steps_per_stats", type=int, default=5,
                      help=("How many training steps to do per stats logging."
                            "Save checkpoint every 10x steps_per_stats"))
  parser.add_argument(
      "--num_buckets",
      type=int,
      default=5,
      help="Put data into similar-length buckets.")
  parser.add_argument(
      "--choose_buckets",
      type=int,
      default=1,
      help="Choose from this number of length buckets per training step.")

  # SPM
  parser.add_argument("--subword_option", type=str, default="bpe",
                      choices=["", "bpe", "spm"],
                      help="""\
                      Set to bpe or spm to activate subword desegmentation.\
                      """)

  # Misc
  parser.add_argument(
      "--num_shards", type=int,
      default=8, help="Number of shards (TPU cores).")
  parser.add_argument(
      "--num_shards_per_host", type=int,
      default=8, help="Number of shards (TPU cores) per host.")
  parser.add_argument(
      "--num_gpus", type=int, default=4, help="Number of gpus in each worker.")
  parser.add_argument(
      "--num_infeed_workers",
      type=int,
      default=1,
      help="Number of TPU workers used for input generation.")
  parser.add_argument(
      "--num_tpu_workers",
      type=int,
      default=1,
      help="Number of TPU workers; if set, uses the distributed-sync pipeline.")
  parser.add_argument("--hparams_path", type=str, default=None,
                      help=("Path to standard hparams json file that overrides"
                            "hparams values from FLAGS."))
  parser.add_argument(
      "--random_seed",
      type=int,
      default=None,
      help="Random seed (>0, set a specific seed).")

  # Inference
  parser.add_argument("--ckpt", type=str, default="",
                      help="Checkpoint file to load a model for inference.")
  parser.add_argument(
      "--infer_batch_size",
      type=int,
      default=512,
      help="Batch size for inference mode.")
  parser.add_argument(
      "--examples_to_infer",
      type=int,
      default=3003,
      help="Number of examples to infer.")
  parser.add_argument("--detokenizer_file", type=str,
                      default="mosesdecoder/scripts/tokenizer/detokenizer.perl",
                      help=("""Detokenizer script file."""))
  parser.add_argument("--use_REDACTED", type=bool, default=False)
  parser.add_argument(
      "--target_bleu", type=float, default=24.0, help="Target accuracy.")

  # Advanced inference arguments
  parser.add_argument("--infer_mode", type=str, default="beam_search",
                      choices=["greedy", "sample", "beam_search"],
                      help="Which type of decoder to use during inference.")
  parser.add_argument("--beam_width", type=int, default=5,
                      help=("""\
      beam width when using beam search decoder. If 0 (default), use standard
      decoder with greedy helper.\
      """))
  parser.add_argument(
      "--length_penalty_weight",
      type=float,
      default=0.6,
      help="Length penalty for beam search.")
  parser.add_argument(
      "--coverage_penalty_weight",
      type=float,
      default=0.1,
      help="Coverage penalty for beam search.")

  # Job info
  parser.add_argument("--jobid", type=int, default=0,
                      help="Task id of the worker.")

  # TPU
  parser.add_argument("--use_tpu", type=bool, default=True)
  parser.add_argument("--master", type=str, default="",
                      help=("Address of the master. Either --master or "
                            "--tpu_name must be specified."))
  parser.add_argument("--tpu_name", type=str, default=None,
                      help=("Name of the TPU for Cluster Resolvers. Either "
                            "--tpu_name or --master must be specified."))
  parser.add_argument("--use_dynamic_rnn", type=bool, default=False)
  parser.add_argument("--use_synthetic_data", type=bool, default=False)
  parser.add_argument(
      "--mode",
      type=str,
      default="train_and_eval",
      choices=["train", "train_and_eval", "infer", "preprocess"])
  parser.add_argument(
      "--activation_dtype",
      type=str,
      default="bfloat16",
      choices=["float32", "bfloat16"])
  parser.add_argument("--tpu_job_name", type=str, default=None)

  # copybara:strip_begin
  # Vizier
  parser.add_argument("--client_handle", type=str, default="",
                      help=("Client_handle for the tuner."))
  parser.add_argument("--study_name", type=str, default=None,
                      help=("Name of Vizier hparams tuning study."))

  # benchmark flags
  parser.add_argument(
    '--time_limit_s', default=None, type=int,
    help=('Number of seconds to run for'))

  parser.add_argument(
    '--dataset_threadpool_size', default=48, type=int,
    help=('The size of the private datapool size in dataset.'))

  parser.add_argument(
    '--map_and_batch_fusion', default=True, type=bool,
    help=('Enables tf.data options map and batch fusion'))

  parser.add_argument(
    '--benchmark_num_elements', default=500, type=int,
    help=('The number of elements to use for the benchmark'))

  parser.add_argument(
      '--read_parallelism', default=100, type=int,
      help=(''))

  parser.add_argument(
      '--map_0_parallelism', default=64, type=int,
      help=(''))

  parser.add_argument(
      '--map_1_parallelism', default=64, type=int,
      help=(''))

  parser.add_argument(
      '--batch_parallelism', default=None, type=int,
      help=(''))

  parser.add_argument(
    '--use_cache', default=False, type=bool,
      help=('Enable cache for training input.'))

  parser.add_argument(
    '--cache_records', default=False, type=bool,
      help=('Enable cache for data only.'))

  parser.add_argument(
    '--resource_compatibility', default=False, type=bool,
      help=('Enable custom hash_table implementation to avoid resources.'))

  parser.add_argument(
    '--shuffle_size', default=None, type=int,
    help=('The shuffle buffer size'))

  parser.add_argument(
      '--outer_parallelism', default=None, type=int,
      help=(''))
