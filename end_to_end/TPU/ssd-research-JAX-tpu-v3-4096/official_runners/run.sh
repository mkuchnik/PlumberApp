#!/bin/bash
#export TF_CPP_MIN_LOG_LEVEL=0
#export TF_CPP_MIN_VLOG_LEVEL=3
train_path="gs://mkuchnik_data_eu_west4/data/ILSVRC12/train_TFRecord"
train_path="gs://mkuchnik_data_eu_west4/data/COCO/train*"
validation_path="gs://mkuchnik_data_eu_west4/data/COCO/val*"
use_cache=true
time_limit_s=62
profile=false
curr_dir="$(pwd)"
script_name="$curr_dir/ssd_train.py"
benchmark_script_name="$curr_dir/benchmark_mlperf.py"
benchmark_global_opt="--time_limit_s=$time_limit_s --dataset_threadpool_size=48"
global_opt="--num_epochs=5"

experiment_dir="official_experiments/default_model_48_core_48_thread"
experiment_prefix="${experiment_dir}/run_0/ssd_train"

. run_def.sh

function drop_caches {
  sync
  sudo /sbin/sysctl vm.drop_caches=3
}

# Basic setup
function step_0 {
  name=step_0
  experiment_name=${experiment_prefix}_${name}
  mkdir -p ${experiment_name}
  pushd ${experiment_name}
  PLUMBER_NO_OPTIMIZE=True \
	  python3 $script_name \
	  --training_file_pattern=$train_path \
	  --validation_file_pattern=$validation_path \
	  --detailed_time=True \
  	  --precompile_eval=True \
	  --no_eval=False \
          --read_parallelism=1 \
          --map_parse_parallelism=1 \
          --map_tfrecord_decode_parallelism=1 \
          --map_image_postprocessing_parallelism=1 \
          --map_image_transpose_postprocessing_parallelism=1 \
          --shard_parallelism=1 \
	  --prefetch_amount=0 \
      ${global_opt} 2>&1 | tee ${name}_log.txt
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
  python3 $script_name \
	  --training_file_pattern=$train_path \
	  --validation_file_pattern=$validation_path \
	  --detailed_time=True \
  	  --precompile_eval=True \
	  --no_eval=False \
          --read_parallelism=1 \
          --map_parse_parallelism=1 \
          --map_tfrecord_decode_parallelism=1 \
          --map_image_postprocessing_parallelism=1 \
          --map_image_transpose_postprocessing_parallelism=1 \
          --shard_parallelism=1 \
	  --prefetch_amount=0 \
      ${global_opt} 2>&1 | tee ${name}_log.txt
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
  	  --precompile_eval=True \
	  --no_eval=False \
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
	  --training_file_pattern=$train_path \
	  --validation_file_pattern=$validation_path \
	  --detailed_time=True \
  	  --precompile_eval=True \
	  --no_eval=False \
          --read_parallelism=-1 \
          --map_parse_parallelism=-1 \
          --map_tfrecord_decode_parallelism=-1 \
          --map_image_postprocessing_parallelism=-1 \
          --map_image_transpose_postprocessing_parallelism=-1 \
          --shard_parallelism=-1 \
	  --prefetch_amount=-1 \
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
  	  --precompile_eval=True \
	  --no_eval=False \
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
	  --training_file_pattern=$train_path \
	  --validation_file_pattern=$validation_path \
	  --detailed_time=True \
  	  --precompile_eval=True \
	  --no_eval=False \
          --read_parallelism=48 \
          --map_parse_parallelism=48 \
          --map_tfrecord_decode_parallelism=48 \
          --map_image_postprocessing_parallelism=48 \
          --map_image_transpose_postprocessing_parallelism=48 \
          --shard_parallelism=48 \
      ${global_opt} 2>&1 | tee ${name}_log.txt
  cp stats.pb $name.pb
}

function step_heuristic_benchmark {
  drop_caches
  name=step_heuristic_benchmark
  PLUMBER_NO_OPTIMIZE=True \
  python3 $benchmark_script_name \
	  --training_file_pattern=$train_path \
          --read_parallelism=48 \
          --map_parse_parallelism=48 \
          --map_tfrecord_decode_parallelism=48 \
          --map_image_postprocessing_parallelism=48 \
          --map_image_transpose_postprocessing_parallelism=48 \
          --shard_parallelism=48 \
      ${benchmark_global_opt} 2>&1 | tee ${name}_log.txt
  cp stats.pb $name.pb
}

list_python_programs
kill_python_programs
kill_python_programs
list_python_programs
step_heuristic
list_python_programs
kill_python_programs
kill_python_programs
list_python_programs
step_autotune
list_python_programs
kill_python_programs
kill_python_programs
list_python_programs
step_plumber
list_python_programs
kill_python_programs
kill_python_programs
list_python_programs
step_0
