#!/bin/bash
#export POSIX_THROTTLE_TOKEN_RATE=300000

set -e

data_dir="/mnt/data/BigLearning/mkuchnik/datasets/coco/COCO"
data_dir="/zpool1/Datasets/Official/MLPerf/COCO"
test_dir_name="train_sweep_0_rcnn_test"
mkdir -p $test_dir_name
script_dir=$(pwd)
run_path=$(pwd)/run.sh
benchmark_path=$(pwd)/benchmark_mlperf.py
graph_rewrite_path=$(pwd)/graph_rewrites.py
baseline_stats_path=$(pwd)/$test_dir_name/baseline
plumber_i=0
random_i=0
num_steps=30
up_parallelism=1

global_opt="--data_dir=$data_dir --benchmark_num_elements=10000 --dataset_threadpool_size=16 --time_limit_s=62 --map_and_batch_fusion=True"

function drop_caches {
  sync
  sudo /sbin/sysctl vm.drop_caches=3
}

# Naive benchmark
function step_0 {
  drop_caches
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
  drop_caches
  name=step_heuristic
  python3 $benchmark_path \
    --cache_records=True \
    --read_parallelism=16 \
    --map_parse_parallelism=16 \
    --map_transform_images_parallelism=16 \
    ${global_opt} 2>&1 | tee ${name}_log.txt
  cp stats.pb $name.pb
}

# Naive benchmark
function step_autotune {
  drop_caches
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
    --up_parallelism=$up_parallelism \
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
    --up_parallelism=$up_parallelism \
    ${global_opt} 2>&1 | tee ${name}_log.txt
  popd
}

run_baseline
run_heuristic
run_autotune
for plumber_i in 0
do
  run_plumber_rewrites
done
for random_i in 0 1 2 3 4
do
  run_random_rewrites
done
for plumber_i in 1 2 3 4
do
  run_plumber_rewrites
done
