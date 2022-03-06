#!/bin/bash
# Remove the file, if it exists
sudo rm /tmp/libtpu.so
# Copy the libtpu file to /tmp
cp $HOME/miniconda3/envs/py37/lib/python3.7/site-packages/libtpu/libtpu.so /tmp
# Copy the libtpu file to system path
sudo cp /tmp/libtpu.so /usr/lib/
