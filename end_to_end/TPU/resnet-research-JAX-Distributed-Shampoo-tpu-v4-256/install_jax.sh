#!/bin/bash
pip install "jax[tpu]==0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pushd ../../../plumber_analysis/
bash install.sh
popd
