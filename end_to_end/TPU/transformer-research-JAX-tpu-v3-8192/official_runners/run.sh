#!/bin/bash
#export TF_CPP_MIN_LOG_LEVEL=0
#export TF_CPP_MIN_VLOG_LEVEL=3
train_path="gs://mkuchnik_data_eu_west4/data/wmt32k"
train_data_path="${train_path}/translate_ende_official_tfrecord_packed/translate*-train-*"
eval_data_path="${train_path}/translate_ende_official_tfrecord_packed/translate*-train-*"
#EVAL_PATH = ROOT + 'translate_ende_wmt32k-dev*'
vocab_path="${train_path}/translate_ende/vocab.ende.32768"

use_cache=true
time_limit_s=32
profile=false
curr_dir="$(pwd)"
script_name="$curr_dir/train.py"
benchmark_script_name="$curr_dir/benchmark_mlperf.py"
benchmark_global_opt="--time_limit_s=$time_limit_s"
global_opt="--num_epochs=6 --no_eval=True --compute_train_metrics=False --xprof=False"
precompile=False

experiment_dir="official_experiments/default_model_96_core_96_thread"
experiment_prefix="${experiment_dir}/run_0/transformer_train"

. run_def.sh

function drop_caches {
  sync
  sudo /sbin/sysctl vm.drop_caches=3
}

function benchmark_overhead {
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate py37v2
  step_heuristic
  conda activate py37
  step_heuristic
  #step_heuristic
  #step_heuristic
  #step_heuristic
  #step_heuristic
  #cache_records=false
  #echo "Caching off"
  #conda activate py37v2
  #step_heuristic
  #step_heuristic
  #step_heuristic
  #conda activate py37
  #step_heuristic
  #step_heuristic
  #step_heuristic
}

# Basic setup
function step_0 {
  name=step_0
  experiment_name=${experiment_prefix}_${name}
  mkdir -p ${experiment_name}
  pushd ${experiment_name}
  PLUMBER_NO_OPTIMIZE=True \
	  python3 $script_name \
    	  --train_data_path=${train_data_path} \
	  --eval_data_path=${eval_data_path} \
	  --vocab_path=${vocab_path} \
	  --model_dir="my_model_dir" \
	  --precompile=${precompile} \
      ${global_opt} | tee ${name}_log.txt
  cp stats.pb $name.pb
  popd
}

function step_naive {
  name=step_naive
  experiment_name=${experiment_prefix}_${name}
  mkdir -p ${experiment_name}
  pushd ${experiment_name}
  PLUMBER_NO_OPTIMIZE=True \
	  python3 $script_name \
    	  --train_data_path=${train_data_path} \
	  --eval_data_path=${eval_data_path} \
	  --vocab_path=${vocab_path} \
	  --model_dir="my_model_dir" \
	  --precompile=${precompile} \
	  --input_pipeline_default_parallelism=1 \
	  --input_pipeline_default_prefetching=1 \
	  --num_epochs=6 \
      | tee ${name}_log.txt
  cp stats.pb $name.pb
  popd
}

function step_0_benchmark {
  name=step_0_benchmark
  experiment_name=${experiment_prefix}_${name}
  mkdir -p ${experiment_name}
  pushd ${experiment_name}
  PLUMBER_NO_OPTIMIZE=True \
  python3 $benchmark_script_name \
	  --training_file_pattern=$train_path \
          --read_parallelism=1 \
          --map_parse_parallelism=1 \
          --map_tfrecord_decode_parallelism=1 \
          --map_image_postprocessing_parallelism=1 \
          --map_image_transpose_postprocessing_parallelism=1 \
          --shard_parallelism=1 \
      ${benchmark_global_opt} 2>&1 | tee ${name}_log.txt
  cp stats.pb $name.pb
  popd
}

function step_plumber {
  name=step_plumber
  experiment_name=${experiment_prefix}_${name}
  mkdir -p ${experiment_name}
  pushd ${experiment_name}
  PLUMBER_NO_OPTIMIZE=False \
  PLUMBER_OPTIMIZE_PIPELINE=True \
	  python3 $script_name \
    	  --train_data_path=${train_data_path} \
	  --eval_data_path=${eval_data_path} \
	  --vocab_path=${vocab_path} \
	  --model_dir="my_model_dir" \
	  --precompile=${precompile} \
      ${global_opt} | tee ${name}_log.txt
  cp stats.pb $name.pb
  popd
}

function step_plumber_fake {
  name=step_plumber_fake
  experiment_name=${experiment_prefix}_${name}
  mkdir -p ${experiment_name}
  pushd ${experiment_name}
  PLUMBER_FAKE_PIPELINE=True \
  python3 $script_name \
	  --training_file_pattern=$train_path \
	  --validation_file_pattern=$validation_path \
	  --detailed_time=True \
	  --precompile=${precompile} \
	  --no_eval=True \
          --read_parallelism=1 \
          --map_parse_parallelism=1 \
          --map_tfrecord_decode_parallelism=1 \
          --map_image_postprocessing_parallelism=1 \
          --map_image_transpose_postprocessing_parallelism=1 \
          --shard_parallelism=1 \
      ${global_opt} 2>&1 | tee ${name}_log.txt
  cp stats.pb $name.pb
  popd
}

function step_plumber_benchmark {
  name=step_plumber_benchmark
  experiment_name=${experiment_prefix}_${name}
  mkdir -p ${experiment_name}
  pushd ${experiment_name}
  #PLUMBER_FAKE_PIPELINE=True \
  #PLUMBER_FAST_OPTIMIZE=True \
  python3 $benchmark_script_name \
	  --training_file_pattern=$train_path \
          --read_parallelism=1 \
          --map_parse_parallelism=1 \
          --map_tfrecord_decode_parallelism=1 \
          --map_image_postprocessing_parallelism=1 \
          --map_image_transpose_postprocessing_parallelism=1 \
          --shard_parallelism=1 \
      ${benchmark_global_opt} 2>&1 | tee ${name}_log.txt
  cp stats.pb $name.pb
  popd
}

function step_autotune {
  name=step_autotune
  experiment_name=${experiment_prefix}_${name}
  mkdir -p ${experiment_name}
  pushd ${experiment_name}
  PLUMBER_NO_OPTIMIZE=True \
	  python3 $script_name \
    	  --train_data_path=${train_data_path} \
	  --eval_data_path=${eval_data_path} \
	  --vocab_path=${vocab_path} \
	  --model_dir="my_model_dir" \
	  --precompile=${precompile} \
	  --input_pipeline_default_parallelism=-1 \
	  --input_pipeline_default_prefetching=-1 \
      ${global_opt} 2>&1 | tee ${name}_log.txt
  cp stats.pb $name.pb
  popd
}

function step_autotune_benchmark {
  name=step_autotune_benchmark
  experiment_name=${experiment_prefix}_${name}
  mkdir -p ${experiment_name}
  pushd ${experiment_name}
  PLUMBER_NO_OPTIMIZE=True \
  python3 $benchmark_script_name \
	  --training_file_pattern=$train_path \
	  --validation_file_pattern=$validation_path \
	  --detailed_time=True \
  	  --precompile_eval=False \
	  --no_eval=True \
          --read_parallelism=-1 \
          --map_parse_parallelism=-1 \
          --map_tfrecord_decode_parallelism=-1 \
          --map_image_postprocessing_parallelism=-1 \
          --map_image_transpose_postprocessing_parallelism=-1 \
          --shard_parallelism=-1 \
      ${benchmark_global_opt} 2>&1 | tee ${name}_log.txt
  cp stats.pb $name.pb
  popd
}

function step_heuristic {
  drop_caches
  name=step_heuristic
  experiment_name=${experiment_prefix}_${name}
  mkdir -p ${experiment_name}
  pushd ${experiment_name}
  PLUMBER_NO_OPTIMIZE=True \
	  python3 $script_name \
    	  --train_data_path=${train_data_path} \
	  --eval_data_path=${eval_data_path} \
	  --vocab_path=${vocab_path} \
	  --model_dir="my_model_dir" \
	  --precompile=True \
	  --input_pipeline_default_parallelism=96 \
	  --input_pipeline_default_prefetching=1024 \
      ${global_opt} 2>&1 | tee ${name}_log.txt
  cp stats.pb $name.pb
  popd
}

function step_heuristic_benchmark {
  drop_caches
  name=step_heuristic_benchmark
  experiment_name=${experiment_prefix}_${name}
  mkdir -p ${experiment_name}
  pushd ${experiment_name}
  PLUMBER_NO_OPTIMIZE=True \
  python3 $benchmark_script_name \
	  --training_file_pattern=$train_path \
          --read_parallelism=96 \
          --map_parse_parallelism=96 \
          --map_tfrecord_decode_parallelism=96 \
          --map_image_postprocessing_parallelism=96 \
          --map_image_transpose_postprocessing_parallelism=96 \
          --shard_parallelism=96 \
      ${benchmark_global_opt} 2>&1 | tee ${name}_log.txt
  cp stats.pb $name.pb
  popd
}

list_python_programs
kill_python_programs
kill_python_programs
list_python_programs
step_autotune
list_python_programs
kill_python_programs
kill_python_programs
list_python_programs
step_naive
list_python_programs
kill_python_programs
kill_python_programs
list_python_programs
step_heuristic
list_python_programs
kill_python_programs
kill_python_programs
list_python_programs
step_plumber
