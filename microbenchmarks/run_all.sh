#!/bin/bash

function run_resnet() {
  pushd simple_resnet/MLPerf
  bash train_sweep.sh
  popd
}

function run_rcnn() {
  pushd simple_rcnn
  bash train_sweep.sh
  popd
}

function run_ssd() {
  pushd simple_ssd
  bash train_sweep.sh
  popd
}

function run_transformer() {
  pushd simple_transformer
  bash train_sweep.sh
  popd
}

function run_gnmt() {
  pushd simple_gnmt
  bash train_sweep.sh
  popd
}

run_rcnn
run_transformer
run_gnmt
run_resnet
run_ssd