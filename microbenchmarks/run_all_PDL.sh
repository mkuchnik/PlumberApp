#!/bin/bash

function run_resnet() {
  pushd simple_resnet/MLPerf
  bash train_sweep_PDL.sh
  popd
}

function run_rcnn() {
  pushd simple_rcnn
  bash train_sweep_PDL.sh
  popd
}

function run_ssd() {
  pushd simple_ssd
  bash train_sweep_PDL.sh
  popd
}

function run_transformer() {
  pushd simple_transformer
  bash train_sweep_PDL.sh
  popd
}

function run_gnmt() {
  pushd simple_gnmt
  bash train_sweep_PDL.sh
  popd
}

run_resnet
run_rcnn
run_ssd
run_transformer
run_gnmt