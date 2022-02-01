#!/bin/bash
#export POSIX_THROTTLE_TOKEN_RATE=300000

set -e

test_dir_name="train_sweep_v2_transformer_0_PDL"
mkdir -p $test_dir_name
script_dir=$(pwd)
run_path=$(pwd)/run.sh
benchmark_path=$(pwd)/benchmark_mlperf.py
graph_rewrite_path=$(pwd)/graph_rewrites.py
baseline_stats_path=$(pwd)/$test_dir_name/baseline
plumber_i=0
random_i=0

train_data_path="/mnt/data/BigLearning/mkuchnik/datasets/wmt32k/translate_ende_official_tfrecord_packed/translate*-train-*"

global_opt="--benchmark_num_elements=1500000 --dataset_threadpool_size=32 --time_limit_s=62 --map_and_batch_fusion=True"

# Naive benchmark
function step_0 {
  name=step_0
  python3 $benchmark_path \
    --read_parallelism=1 \
    --map_1_parallelism=1 \
    --map_2_parallelism=1 \
    --map_3_parallelism=1 \
    --use_cache=True \
    --train_data_path=${train_data_path} \
    ${global_opt} 2>&1 | tee ${name}_log.txt
  cp stats.pb $name.pb
}

function step_autotune {
  name=step_autotune
  python3 $benchmark_path \
    --read_parallelism=-1 \
    --map_1_parallelism=-1 \
    --map_2_parallelism=-1 \
    --map_3_parallelism=-1 \
    --use_cache=True \
    --train_data_path=${train_data_path} \
    ${global_opt} 2>&1 | tee ${name}_log.txt
  cp stats.pb $name.pb
}

function step_heuristic {
  name=step_heuristic
  python3 $benchmark_path \
    --read_parallelism=32 \
    --map_1_parallelism=32 \
    --map_2_parallelism=32 \
    --map_3_parallelism=32 \
    --use_cache=True \
    --train_data_path=${train_data_path} \
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

function run_autotune() {
  test_name=$test_dir_name/autotune
  mkdir -p $test_name
  pushd $test_name
  step_autotune
  popd
}

function run_heuristic() {
  test_name=$test_dir_name/heuristic
  mkdir -p $test_name
  pushd $test_name
  step_heuristic
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
    --num_steps=15 \
    --num_deviations=1 \
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
    --num_steps=15 \
    --strategy=random_valid \
    ${global_opt} 2>&1 | tee ${name}_log.txt
  popd
}

run_baseline
run_autotune
run_heuristic
for plumber_i in 0 1 2
do
  run_plumber_rewrites
done
for random_i in 0 1 2
do
  run_random_rewrites
done
