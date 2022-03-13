#!/bin/bash
#export TF_CPP_MIN_LOG_LEVEL=0
#export TF_CPP_MIN_VLOG_LEVEL=3
data_dir="gs://mkuchnik_data_eu_west4/data/GNMT_Google/"

time_limit_s=32
profile=false
curr_dir="$(pwd)"
script_name="$curr_dir/nmt.py"
benchmark_script_name="$curr_dir/benchmark_mlperf.py"
benchmark_global_opt="--time_limit_s=$time_limit_s"
model_dir="my_models"
global_opt="--max_train_epochs=5 --mode=train  --num_examples_per_epoch=344576"

experiment_dir="official_experiments/default_model_96_core_96_thread"
experiment_prefix="${experiment_dir}/run_0/gnmt_train"

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
  	  --data_dir=${data_dir} \
	  --out_dir=${model_dir} \
	  --use_preprocessed_data=True \
    	  --input_pipeline_read_parallelism=1 \
	  --input_pipeline_map_0_parallelism=1 \
	  --input_pipeline_map_1_parallelism=1 \
          --input_pipeline_default_prefetching=1 \
      ${global_opt} 2>&1 | tee ${name}_log.txt
  cp stats.pb $name.pb
  popd
}

function step_0_no_prefetch {
  name=step_0_no_prefetch
  experiment_name=${experiment_prefix}_${name}
  mkdir -p ${experiment_name}
  pushd ${experiment_name}
  PLUMBER_NO_OPTIMIZE=True \
	  python3 $script_name \
  	  --data_dir=${data_dir} \
	  --out_dir=${model_dir} \
	  --use_preprocessed_data=True \
    	  --input_pipeline_read_parallelism=1 \
	  --input_pipeline_map_0_parallelism=1 \
	  --input_pipeline_map_1_parallelism=1 \
          --input_pipeline_default_prefetching=0 \
      ${global_opt} | tee ${name}_log.txt
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
  	  --data_dir=${data_dir} \
	  --out_dir=${model_dir} \
	  --use_preprocessed_data=True \
    	  --input_pipeline_read_parallelism=1 \
	  --input_pipeline_map_0_parallelism=1 \
	  --input_pipeline_map_1_parallelism=1 \
          --input_pipeline_default_prefetching=0 \
      ${benchmark_global_opt} 2>&1 | tee ${name}_log.txt
  cp stats.pb $name.pb
  popd
}

function step_plumber {
  name=step_plumber
  experiment_name=${experiment_prefix}_${name}
  mkdir -p ${experiment_name}
  pushd ${experiment_name}
  PLUMBER_NO_OPTIMIZE=True \
  python3 $script_name \
	  --data_dir=${data_dir} \
	  --out_dir=${model_dir} \
	  --use_preprocessed_data=True \
	  --input_pipeline_read_parallelism=47 \
	  --input_pipeline_map_0_parallelism=20 \
	  --input_pipeline_map_1_parallelism=9 \
	  --input_pipeline_default_prefetching=16 \
	  --input_pipeline_cache=True \
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
  	  --precompile_eval=False \
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
          --input_pipeline_default_prefetching=0 \
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
  	  --data_dir=${data_dir} \
	  --out_dir=${model_dir} \
	  --use_preprocessed_data=True \
    	  --input_pipeline_read_parallelism=-1 \
	  --input_pipeline_map_0_parallelism=-1 \
	  --input_pipeline_map_1_parallelism=-1 \
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
  	  --data_dir=${data_dir} \
	  --out_dir=${model_dir} \
	  --use_preprocessed_data=True \
    	  --input_pipeline_read_parallelism=96 \
	  --input_pipeline_map_0_parallelism=96 \
	  --input_pipeline_map_1_parallelism=96 \
          --input_pipeline_default_prefetching=100 \
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


# Get Plumber recommendation with:
# name="params"
# python3 get_params.py 2>&1 | tee ${name}_log.txt
# after tracing pipeline.
# Alternatively, get parameters by running Plumber benchmark

list_python_programs
kill_python_programs
kill_python_programs
list_python_programs
step_0_no_prefetch
list_python_programs
kill_python_programs
kill_python_programs
list_python_programs
step_plumber
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
