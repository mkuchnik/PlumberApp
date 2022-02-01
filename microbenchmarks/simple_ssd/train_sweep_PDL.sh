#!/bin/bash
#export POSIX_THROTTLE_TOKEN_RATE=300000
#export TF_CPP_MIN_LOG_LEVEL=0
#export TF_CPP_MIN_VLOG_LEVEL=3
#export TF_CPP_MIN_VLOG_LEVEL=1
#export TF_CPP_MIN_LOG_LEVEL=0

set -e

test_dir_name="train_sweep_v2_ssd_0_PDL"
mkdir -p $test_dir_name
script_dir=$(pwd)
run_path=$(pwd)/run.sh
benchmark_path=$(pwd)/benchmark_mlperf.py
graph_rewrite_path=$(pwd)/graph_rewrites.py
baseline_stats_path=$(pwd)/$test_dir_name/baseline
plumber_i=0
random_i=0

global_opt="--use_cache=True --benchmark_num_elements=100000 --dataset_threadpool_size=32 --time_limit_s=72 --map_and_batch_fusion=True"

# Naive benchmark
function step_0 {
  name=step_0
  python3 $benchmark_path \
    --read_parallelism=1 \
    --map_parse_parallelism=1 \
    --map_tfrecord_decode_parallelism=1 \
    --map_image_postprocessing_parallelism=1 \
    --map_image_transpose_postprocessing_parallelism=1 \
    --shard_parallelism=1 \
    --training_file_pattern="/mnt/data/BigLearning/mkuchnik/datasets/coco/COCO/train*" \
    ${global_opt} 2>&1 | tee ${name}_log.txt
  cp stats.pb $name.pb
}

function run_baseline() {
  test_name=$test_dir_name/baseline
  mkdir -p $test_name
  pushd $test_name
  step_0
  popd
}

# Naive benchmark
function step_heuristic {
  name=step_heuristic
  python3 $benchmark_path \
    --read_parallelism=32 \
    --map_parse_parallelism=32 \
    --map_tfrecord_decode_parallelism=32 \
    --map_image_postprocessing_parallelism=32 \
    --map_image_transpose_postprocessing_parallelism=32 \
    --shard_parallelism=32 \
    --training_file_pattern="/mnt/data/BigLearning/mkuchnik/datasets/coco/COCO/train*" \
    ${global_opt} 2>&1 | tee ${name}_log.txt
  cp stats.pb $name.pb
}

# Naive benchmark
function step_autotune {
  name=step_autotune
  python3 $benchmark_path \
    --read_parallelism=-1 \
    --map_parse_parallelism=-1 \
    --map_tfrecord_decode_parallelism=-1 \
    --map_image_postprocessing_parallelism=-1 \
    --map_image_transpose_postprocessing_parallelism=-1 \
    --shard_parallelism=-1 \
    --training_file_pattern="/mnt/data/BigLearning/mkuchnik/datasets/coco/COCO/train*" \
    ${global_opt} 2>&1 | tee ${name}_log.txt
  cp stats.pb $name.pb
}

function run_baseline() {
  test_name=$test_dir_name/baseline
  mkdir -p $test_name
  pushd $test_name
  step_0
  popd
}

function run_heuristic() {
  test_name=$test_dir_name/heuristic
  mkdir -p $test_name
  pushd $test_name
  step_heuristic
  popd
}

function run_autotune() {
  test_name=$test_dir_name/autotune
  mkdir -p $test_name
  pushd $test_name
  step_autotune
  popd
}


# Copies the initial file to current directory
function pull_baseline_stats() {
  cp $baseline_stats_path/stats.pb $(pwd)
}

function run_plumber_rewrites() {
  test_name=$test_dir_name/plumber_rewrites_$plumber_i
  name="plumber_rewrite"
  mkdir -p $test_name
  pushd $test_name
  pull_baseline_stats
  python3 $graph_rewrite_path \
    --skip_baseline=False \
    --num_steps=40 \
    --num_deviations=3 \
    ${global_opt} 2>&1 | tee ${name}_log.txt
  popd
}

function run_random_rewrites() {
  test_name=$test_dir_name/random_rewrites_$random_i
  name="random_rewrite"
  mkdir -p $test_name
  pushd $test_name
  pull_baseline_stats
  python3 $graph_rewrite_path \
    --skip_baseline=False \
    --num_steps=40 \
    --strategy=random_valid \
    ${global_opt} 2>&1 | tee ${name}_log.txt
  popd
}

run_heuristic
run_autotune
run_baseline
for plumber_i in 0
do
  run_plumber_rewrites
done
for random_i in 0 1 2
do
  run_random_rewrites
done
for plumber_i in 1 2
do
  run_plumber_rewrites
done
