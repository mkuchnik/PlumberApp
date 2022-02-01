#!/bin/bash
cp $HOME/miniconda3/envs/py37/lib/python3.7/site-packages/libtpu/libtpu.so /tmp
sudo cp /tmp/libtpu.so /usr/lib/
