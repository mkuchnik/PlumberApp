#!/bin/bash
#export TF_CPP_MIN_LOG_LEVEL=0
#export TF_CPP_MIN_VLOG_LEVEL=0
train_path="gs://mkuchnik_data_eu_west4/data/COCO/train*"
validation_path="gs://mkuchnik_data_eu_west4/data/COCO/val*"
time_limit_s=122 # NOTE(mkuchnik): groupby makes convergence slower
profile=false
curr_dir="$(pwd)"
script_name="$curr_dir/mask_rcnn_main.py"
benchmark_script_name="$curr_dir/graph_test.py"
benchmark_script_name="$curr_dir/benchmark_mlperf.py"
benchmark_global_opt="--time_limit_s=$time_limit_s"
global_opt="--num_epochs=6 --input_pipeline_threadpool_size=48"

model_dir="gs://mkuchnik_data_eu_west4/models"
model_dir="my_model_dir"
log_dir="my_log_dir"

bench_dir="official_experiments/bench_normal_${bench_time_limit_s}_48_core_48_thread"
experiment_dir="official_experiments/linear_model_48_core_48_thread"
experiment_prefix="${experiment_dir}/run_0/rcnn_train"

. run_def.sh

function drop_caches {
  sync
  sudo /sbin/sysctl vm.drop_caches=3
}

autotune_options() {
	input_pipeline_default_parallelism=-1
	input_pipeline_default_prefetching=-1
}

static_options() {
	input_pipeline_default_parallelism=48
	#input_pipeline_default_parallelism=96
	input_pipeline_default_prefetching=100
	#enable_small_dataset
	#force_cache_uncompressed
}

naive_options() {
	input_pipeline_default_parallelism=1
	input_pipeline_default_prefetching=0
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
	  --master="local" \
	  --model_dir=$model_dir \
	  --use_fake_data=False \
	  --log_dir=$log_dir \
	  --train_batch_size=32 \
	  --eval_batch_size=32 \
      ${global_opt} 2>&1 | tee ${name}_log.txt
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
	  --training_file_pattern=$train_path \
	  --validation_file_pattern=$validation_path \
	  --master="local" \
	  --model_dir=$model_dir \
	  --use_fake_data=False \
	  --log_dir=$log_dir \
	  --train_batch_size=32 \
	  --eval_batch_size=32 \
	  --input_pipeline_default_parallelism=1 \
	  --input_pipeline_default_prefetching=0 \
      ${global_opt} 2>&1 | tee ${name}_log.txt
  cp stats.pb $name.pb
  popd
}

function step_plumber {
  name=step_plumber
  experiment_name=${experiment_prefix}_${name}
  mkdir -p ${experiment_name}
  pushd ${experiment_name}
  #PLUMBER_NO_OPTIMIZE=False \
  PLUMBER_OPTIMIZE_PIPELINE=False \
	  python3 $script_name \
	  --training_file_pattern=$train_path \
	  --validation_file_pattern=$validation_path \
	  --master="local" \
	  --model_dir=$model_dir \
	  --use_fake_data=False \
	  --log_dir=$log_dir \
	  --train_batch_size=32 \
	  --eval_batch_size=32 \
	  --input_pipeline_default_parallelism=1 \
	  --input_pipeline_default_prefetching=0 \
      ${global_opt} 2>&1 | tee ${name}_log.txt
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
	  --master="local" \
	  --model_dir=$model_dir \
	  --use_fake_data=False \
	  --log_dir=$log_dir \
	  --train_batch_size=32 \
	  --eval_batch_size=32 \
	  --input_pipeline_default_parallelism=-1 \
	  --input_pipeline_default_prefetching=-1 \
      ${global_opt} 2>&1 | tee ${name}_log.txt
  cp stats.pb $name.pb
  popd
}

function step_heuristic {
  name=step_heuristic
  experiment_name=${experiment_prefix}_${name}
  mkdir -p ${experiment_name}
  pushd ${experiment_name}
  PLUMBER_NO_OPTIMIZE=True \
	  python3 $script_name \
	  --training_file_pattern=$train_path \
	  --validation_file_pattern=$validation_path \
	  --master="local" \
	  --model_dir=$model_dir \
	  --use_fake_data=False \
	  --log_dir=$log_dir \
	  --train_batch_size=32 \
	  --eval_batch_size=32 \
	  --input_pipeline_default_parallelism=48 \
	  --input_pipeline_default_prefetching=100 \
      ${global_opt} 2>&1 | tee ${name}_log.txt
  cp stats.pb $name.pb
  popd
}

function step_expert {
  name=step_heuristic
  experiment_name=${experiment_prefix}_${name}
  mkdir -p ${experiment_name}
  pushd ${experiment_name}
  PLUMBER_NO_OPTIMIZE=True \
	  python3 $script_name \
	  --training_file_pattern=$train_path \
	  --validation_file_pattern=$validation_path \
	  --master="local" \
	  --model_dir=$model_dir \
	  --use_fake_data=False \
	  --log_dir=$log_dir \
	  --train_batch_size=32 \
	  --eval_batch_size=32 \
	  --input_pipeline_default_parallelism=64 \
	  --input_pipeline_default_prefetching=100 \
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
	  --validation_file_pattern=$validation_path \
	  --log_dir=$log_dir \
	  --train_batch_size=32 \
	  --eval_batch_size=32 \
      ${benchmark_global_opt} 2>&1 | tee ${name}_log.txt
  cp stats.pb $name.pb
  popd
}

function step_autotune_benchmark {
  #TODO(mkuchnik): Error piping not working
  name=step_autotune_benchmark
  experiment_name=${experiment_prefix}_${name}
  mkdir -p ${experiment_name}
  pushd ${experiment_name}
  PLUMBER_NO_OPTIMIZE=True \
  python3 $benchmark_script_name \
	  --training_file_pattern=$train_path \
	  --validation_file_pattern=$validation_path \
	  --log_dir=$log_dir \
	  --train_batch_size=32 \
	  --eval_batch_size=32 \
	  --input_pipeline_default_parallelism=-1 \
	  --input_pipeline_default_prefetching=-1 \
	  ${benchmark_global_opt} | tee -a ${name}_log.txt
	  #${benchmark_global_opt} 2>&1 | tee -a ${name}_log.txt
  cp stats.pb $name.pb
  popd
}

function step_plumber_benchmark {
  name=step_plumber_benchmark
  experiment_name=${experiment_prefix}_${name}
  mkdir -p ${experiment_name}
  pushd ${experiment_name}
  PLUMBER_NO_OPTIMIZE=False \
  PLUMBER_OPTIMIZE_PIPELINE=True \
  PLUMBER_FAST_OPTIMIZE=True \
  python3 $benchmark_script_name \
	  --training_file_pattern=$train_path \
	  --validation_file_pattern=$validation_path \
	  --log_dir=$log_dir \
	  --train_batch_size=32 \
	  --eval_batch_size=32 \
	  --input_pipeline_default_parallelism=1 \
	  --input_pipeline_default_prefetching=0 \
      ${benchmark_global_opt} 2>&1 | tee ${name}_log.txt
  cp stats.pb $name.pb
  popd
}

function step_static_benchmark {
  #TODO(mkuchnik): Error piping not working
  name=step_static_benchmark
  experiment_name=${experiment_prefix}_${name}
  mkdir -p ${experiment_name}
  pushd ${experiment_name}
  python3 $benchmark_script_name \
	  --training_file_pattern=$train_path \
	  --validation_file_pattern=$validation_path \
	  --log_dir=$log_dir \
	  --train_batch_size=32 \
	  --eval_batch_size=32 \
	  --input_pipeline_default_parallelism=64 \
	  --input_pipeline_default_prefetching=100 \
	  ${benchmark_global_opt} | tee -a ${name}_log.txt
	  #${benchmark_global_opt} 2>&1 | tee -a ${name}_log.txt
  cp stats.pb $name.pb
  popd
}

function step_naive_benchmark {
  #TODO(mkuchnik): Error piping not working
  name=step_naive_benchmark
  experiment_name=${experiment_prefix}_${name}
  mkdir -p ${experiment_name}
  pushd ${experiment_name}
  python3 $benchmark_script_name \
	  --training_file_pattern=$train_path \
	  --validation_file_pattern=$validation_path \
	  --log_dir=$log_dir \
	  --train_batch_size=32 \
	  --eval_batch_size=32 \
	  --input_pipeline_default_parallelism=1 \
	  --input_pipeline_default_prefetching=1 \
	  ${benchmark_global_opt} | tee -a ${name}_log.txt
	  #${benchmark_global_opt} 2>&1 | tee -a ${name}_log.txt
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
step_autotune
kill_python_programs
step_heuristic
list_python_programs
kill_python_programs
step_naive
kill_python_programs
kill_python_programs
cp dataloader_plumber.py dataloader.py
step_plumber
