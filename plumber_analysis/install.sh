#!/bin/bash

set -e

python3 -m pip install --upgrade build
python3 -m build
python3 -m pip uninstall -y plumber-analysis-mkuchnik || echo "No existing build"
python3 -m pip install nvidia-pyindex
python3 -m pip install graphsurgeon
python3 -m pip install dist/*.whl