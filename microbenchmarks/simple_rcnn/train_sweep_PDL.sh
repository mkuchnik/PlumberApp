#!/bin/bash
#export POSIX_THROTTLE_TOKEN_RATE=300000

set -e

data_dir="/zpool1/Datasets/Official/MLPerf/COCO"
data_dir="/mnt/data/BigLearning/mkuchnik/datasets/coco/COCO"
test_dir_name="train_sweep_v2_rcnn_0_PDL"
mkdir -p $test_dir_name
script_dir=$(pwd)
run_path=$(pwd)/run.sh
benchmark_path=$(pwd)/benchmark_mlperf.py
graph_rewrite_path=$(pwd)/graph_rewrites.py
baseline_stats_path=$(pwd)/$test_dir_name/baseline
plumber_i=0
random_i=0
num_steps=30
num_cores=$(nproc --all)

global_opt="--data_dir=$data_dir --benchmark_num_elements=10000 --dataset_threadpool_size=$num_cores --time_limit_s=72 --map_and_batch_fusion=True"

# Naive benchmark
function step_0 {
  name=step_0
  python3 $benchmark_path \
    --cache_records=True \
    --read_parallelism=1 \
    --map_parse_parallelism=1 \
    --map_transform_images_parallelism=1 \
    ${global_opt} 2>&1 | tee ${name}_log.txt
  cp stats.pb $name.pb
}

# Naive benchmark
function step_heuristic {
  name=step_heuristic
  python3 $benchmark_path \
    --cache_records=True \
    --read_parallelism=$num_cores \
    --map_parse_parallelism=$num_cores \
    --map_transform_images_parallelism=$num_cores \
    ${global_opt} 2>&1 | tee ${name}_log.txt
  cp stats.pb $name.pb
}

# Naive benchmark
function step_autotune {
  name=step_autotune
  python3 $benchmark_path \
    --cache_records=True \
    --read_parallelism=-1 \
    --map_parse_parallelism=-1 \
    --map_transform_images_parallelism=-1 \
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
  name="plumber_rewrite"
  test_name=$test_dir_name/plumber_rewrites_$plumber_i
  mkdir -p $test_name
  pushd $test_name
  pull_baseline_stats
  python3 $graph_rewrite_path \
    --skip_baseline=False \
    --num_steps=$num_steps \
    --num_deviations=3 \
    ${global_opt} 2>&1 | tee ${name}_log.txt
  popd
}

function run_random_rewrites() {
  name="random_rewrite"
  test_name=$test_dir_name/random_rewrites_$random_i
  mkdir -p $test_name
  pushd $test_name
  pull_baseline_stats
  python3 $graph_rewrite_path \
    --skip_baseline=False \
    --num_steps=$num_steps \
    --strategy=random_valid \
    ${global_opt} 2>&1 | tee ${name}_log.txt
  popd
}

run_baseline
run_heuristic
run_autotune
for plumber_i in 0 1 2
do
  run_plumber_rewrites
done
for random_i in 0 1 2
do
  run_random_rewrites
done
