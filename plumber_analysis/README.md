# Plumber Common

This package represents common utilities used to interact with Plumber from
Python programs. This package works with the Tensorflow fork to accomplish
rewriting and other potentially sophisticated analysis. The primary reason to
keep it seperate is to decouple statistics from high-level processing, which
typically is dependency heavy.

## Manual Install
From this directory, run:
```bash
python3 -m pip install --upgrade build
python3 -m build
python3 -m pip uninstall -y plumber-analysis-mkuchnik
python3 -m pip install dist/*.whl
```

Alternatively, the commands are in `install.sh`.
Note, there are quite a few dependencies currently. The main dependency is the
Tensorflow fork, which must be installed from source.
