# Plumber
A repository for replicating the experiments found in "Plumber: Diagnosing and Removing Performance Bottlenecks in Machine Learning Data Pipelines" (MLSys '22).

Plumber consists of two things: a Tensorflow installation and a Plumber "app",
which is installed as a front-end to the data generated by Plumber.
You can find the Tensorflow release [here](https://github.com/mkuchnik/PlumberTensorflow.git).
This repository contains the application layer, which you can find in
[`plumber_analysis`](plumber_analysis).
The application layer is installed on top of the Tensorflow release, so you will
want to first install the Tensorflow release and then install
`plumber_analysis`.

## Hardware Requirements
We use the following hardware setups in our experiments.

### Setup A (microbenchmarks)
These experiments were run on a consumer-grade desktop machine.
The data is read off of ZFS.

Node hardware:
CPU: 8-core AMD Ryzen 7 2700X
RAM: 32GiB
Drives: HP EX920 1TB SSD, WDC WD60EFRX-68L HDD (3x in ZFS configuration)


### Setup B (microbenchmarks)
These experiments were run on the Carnegie Mellon University [Parallel Data
Lab](https://www.pdl.cmu.edu/)
[Orca](https://orca.pdl.cmu.edu)
cluster.
Nodes are connected with a Cisco Nexus 3264-Q 64-port QSFP+ 40GbE switch.
Each node usually has a HDD and SSD.
The data is read off a Network File System.

Node hardware:
CPU: 16-core Intel E5-2698Bv3 Xeon 2GHz
RAM: 64GiB
Drives: 400GB Intel P3600 NVMe SSD, 4TB 7200RPM Seagate ST4000NM0023 HDD
NIC: Mellanox MCX314A-BCCT 40GbE


### Setup C (end-to-end)
These are cloud TPU VMs as described
[here](https://cloud.google.com/blog/products/compute/introducing-cloud-tpu-vms).
You can readily access them on Google Cloud.
The data is read from Google Cloud Storage.
According to CLI tools, they have 300GB of memory and 96 Intel Xeon cores along with the TPUv3-8 accelerators.


## Software Requirements
The Tensorflow component of Plumber is just a fork of Plumber.
Building Tensorflow from source is well described via [official
documentation](https://www.tensorflow.org/install/source), but we summarize it
below.
If you want to see how to just run some code, we provide the install [steps](#full-example-on-tpu) on TPU.

It is useful to use [Bazelisk](https://docs.bazel.build/versions/main/install-bazelisk.html) to build Tensorflow (we can rename the file to `bazel`).
The [`refresh_tf_build.sh`](https://github.com/mkuchnik/PlumberTensorflow/blob/main/refresh_tf_build.sh) script (and [TPU variant](https://github.com/mkuchnik/PlumberTensorflow/blob/main/refresh_tf_build_tpu.sh)) assumes it is in the Tensorflow directory,
which is why it calls `./bazel` when building.
Feel free to change this to something else if you placed `bazel` elsewhere.

```bash
wget https://github.com/bazelbuild/bazelisk/releases/download/v1.11.0/bazelisk-linux-amd64
mv bazelisk-linux-amd64 bazel
chmod +x bazel
```

We also recommend using
[Miniconda](https://docs.conda.io/en/latest/miniconda.html) to create a standardized environment.
We use a Python3.7 environment (py37) for both building and installing the
following software.

To install:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
You will have to scroll though and type "yes" at the end. You should leave the
options as the defaults.


After install, you will want to create the environment. To create it:

```bash
conda create -n py37 python=3.7
conda activate py37
```


### Dependencies
Tensorflow has its own stated dependencies, which are quite general.
We recommend the following for maximum compatibility, though we also provided Tensorflow's build from
source recommendation.

#### Recommended Install
Using Miniconda, please install these pip packages inside the `py37` environment.
For **maximum compatibility**, specify the version of numpy explicitly:
```bash
sudo apt install python3-dev python3-pip
pip install pip numpy==1.21.5 wheel
pip install keras_preprocessing --no-deps
```

#### Other Potential Install
The dependencies to build Tensorflow [from source](https://www.tensorflow.org/install/source) are:

```bash
sudo apt install python3-dev python3-pip
pip install -U --user pip numpy wheel
pip install -U --user keras_preprocessing --no-deps
```

##### Numpy compatibility
Note that the numpy version used *WILL CAUSE BINARY INCOMPATIBILITY* if it does
not exist with the exact version when run compared to when built.
This is especially problematic with Python wheels, which may uninstall the
current numpy version after it was used for building.

As shown in `tensorflow/tools/pip_package/setup.py`, a standard value of
`numpy==1.19.2` was used.
We tested with `numpy==1.21.5`, which requires avoiding the dependency
implied by Tensorflow, so we bumped this version to `1.21.5`, though you may
decide to change the version back.
In any case, you should be consistent with this value.
You should either install the declared version for building or change the version to the
version you are using to
ensure the generated wheel does not overwrite your numpy install with different
versions, causing the above binary incompatibility issue.

If you get an error about version mismatch (e.g., `RuntimeError: module compiled
against API version 0xe but this version of numpy is 0xd`), it may be helpful to run:

```bash
pip install numpy --upgrade
```


### Building
We provide convenience scripts for building Tensorflow inside the directory.
If you don't want to use these, skip to the Manual Build section.

#### Scripted Build
To build:

1. Clone the Tensorflow repository with Plumber patch.
2. (For TPU builds) Copy `libtpu.so` into the Tensorflow directory.
3. Run `./configure`, setting the appropriate flags. Using Miniconda requires
   setting the environment before running this part, so the paths are found automatically. The default options are fine, unless you are adding GPU support.
4. Run the build script `refresh_tf_build.sh` (CPU-only) or `refresh_tf_build_tpu.sh` (TPU).
5. Install the `plumber_analysis` application layer by cloning this repository,
   changing directory to [`plumber_analysis`](plumber_analysis), and running the `install.sh` script.

##### Full Example on TPU

###### Initial Setup
Install miniconda (as described above).
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Add a Python3.7 env, `py37`.
```bash
conda create python=3.7 -n py37
```

You need to install `libtpu.so` from a JAX installation.
*DO NOT INSTALL UNDER MINICONDA*.
We are simply trying to get the `libtpu.so` file consistent across the system, which we will use to both
build and run Tensorflow.
```bash
conda deactivate # Get out of conda
conda deactivate # Get out of conda
conda deactivate # Get out of conda
python3 -m pip install "jax[tpu]==0.2.19" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html  # Match libtpu.so to version used
sudo cp $HOME/.local/libtpu.so/libtpu.so /usr/lib/libtpu.so  # Overwrite existing libtpu.so
```

You can then do a Tensorflow build with that `libtpu.so` version, which we'll copy
into the source directory for Tensorflow, below.
When done correctly, that version of `libtpu.so` should return the following
checksum.
```bash
$sha1sum /usr/lib/libtpu.so 
44c88d0d50f2f8ffc9eedc17dde72e60b28f3963  /usr/lib/libtpu.so
```

###### Tensorflow Build and Install
Assuming Miniconda is installed with `py37` environment and `libtpu.so` is the
version you want to use:

```bash
conda activate py37
git clone --recurse-submodules https://github.com/mkuchnik/PlumberTensorflow.git
cd PlumberTensorflow
cp /usr/lib/libtpu.so .  # Copy libtpu version to Tensorflow for building

# Get bazel. We will call it with ./bazel in the TF Build script.
wget https://github.com/bazelbuild/bazelisk/releases/download/v1.11.0/bazelisk-linux-amd64
mv bazelisk-linux-amd64 bazel
chmod +x bazel

# Install Dependencies
sudo apt -y update
sudo apt -y install python3-venv
conda activate py37
# Standard Tensorflow Dependencies
pip install -U pip numpy==1.21.5 wheel
pip install -U keras_preprocessing --no-deps

# Default options are probably fine (make sure you see a path to miniconda)
./configure

# Build
bash refresh_tf_build_tpu.sh

# Exit Directory
cd ..

# Install Plumber Analysis App
git clone https://github.com/mkuchnik/PlumberApp.git
cd PlumberApp/plumber_analysis
bash install.sh
```

To test everything is working fine, start a python shell and import tensorflow.
If `tf.ones(1)` activates over multiple (8) devices, the TPU is working.
If you can import `plumber_analysis`, it is installed.

```python3
import tensorflow as tf
tf.ones(1) # Should show 8 TPU "StreamExecutor" devices initializing in logs
tpu_devices = tf.config.list_physical_devices("TPU")
print("tpu_devices: {}".format(tpu_devices)) # Should be 8 TPUs
print("num TPU: {}".format(len(tpu_devices))) # Should be 8
import plumber_analysis # Should not throw error
```

Similarly, if using JAX for an end-to-end run:
```python3
import jax
jax.device_count() # Should show 8 TPU devices
```

If you see a timeout connecting to TPU, it is likely that some other process is
holding a lock on `libtpu.so`. You can try to find and kill it by killing all
`python` processes (that are not root).

```bash
ps aux | grep python | grep -v "grep python" | awk '{print $2}' | xargs kill -9
```

You can also try to find the process using `libtpu` with:
```bash
sudo lsof -w /dev/accel0
```

As a last resort, the lock is on `/tmp/libtpu_lockfile`, so you can delete that
file at the risk of causing `libtpu` to enter undefined state.

#### Manual Build 
To build:

1. Clone the Tensorflow repository with Plumber patch.
2. Copy `libtpu.so` into the Tensorflow directory.
3. Run `./configure`, setting the appropriate flags. Using Anaconda requires
   setting the environment before running this part, so the paths are found automatically. The default options are fine, unless you are adding GPU support.
4. Run the build. TPU build is shown below, since it's most complicated
   (requiring an extra `--config=tpu` flag). Note,
   for TPU, you need a `libtpu.so` file to build against, which is distributed with JAX TPU Python packages. Otherwise, you can find them pre-installed on TPU machines under `/usr/lib/libtpu.so`. *Place `libtpu.so` in the Tensorflow source directory so it can be found during the build installation.* Set `N_JOBS` to the thread count you want to use for the build.
5. Install the `plumber_analysis` application layer by cloning this repository,
   changing directory to [`plumber_analysis`](plumber_analysis), and running the `install.sh` script.

The Tensorflow build in step 4 will look something like:
```bash
  bazel build \
    --config=opt \
    --config=tpu \
    -c opt \
    -j $N_JOBS \
    --verbose_failures \
    //tensorflow/tools/pip_package:build_pip_package
```

If you get an error about something TPU related, it is almost always because
there is some issue with finding the appropriate function calls in `libtpu.so`.
You may have not placed `libtpu.so` in the Tensorflow directory if this is the
case.

If you don't use TPU, omit that config.
```bash
  bazel build \
    --config=opt \
    -c opt \
    -j $N_JOBS \
    --verbose_failures \
    //tensorflow/tools/pip_package:build_pip_package
```

This will generate a executable that builds the pip package. To generate the
Python wheel:
```bash
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
```

You can then install with:
```bash
pip install /tmp/tensorflow_pkg/tensorflow-${VERSION}-${TAGS}.whl
```

where `VERSION` (the release version e.g., 2.7.0) and `TAGS` (a function of Python version e.g., py37)
are set appropriately for the selected Tensorflow build.
You can just look in the `/tmp` directory to see what was generated.

A typical set of dependencies after installation is given in
`requirements_post_install.txt`.

#### Tests
Note that if you also wish to run unit-tests, you may need to install JAVA.
```bash
sudo apt install openjdk-11-jdk
```

It's also hard to run the tests with the TPU build; CPU-only is better for
testing.


### TPU Libraries Gotchas
TPUs need a driver and library to run, much like GPUs have CUDA.

#### Install
To get the `libtpu.so` file, run the following (shown above in install):
```bash
python3  -m pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

You then have to find the file install location and place it in the Tensorflow
build directory.
You will also want to move it to be the system library (so you don't get version mismatch errors).
```bash
sudo cp $HOME/.local/libtpu.so/libtpu.so /usr/lib/libtpu.so
```

As noted below, we used `jax[tpu]==0.2.19`.
This gives us `libtpu-nightly-0.1.dev20210809`.

#### Running
Only one process can run with `libtpu` at a time.
Therefore, if you encounter a problem with:
`libtpu.so is already is used by anoter process. Not attempting to load libtpu.so in this process.`

(or some variant of a TPU timeout) you will need to find and stop that process.

You can try to find and kill it by killing all
`python` processes (that are not root).

```bash
ps aux | grep python | grep -v "grep python" | awk '{print $2}' | xargs kill -9
```

You can also try to find the process using `libtpu` with:
```bash
sudo lsof -w /dev/accel0
```

As a last resort, the lock is on `/tmp/libtpu_lockfile`, so you can delete that
file at the risk of causing `libtpu` to enter undefined state.

Also, note that the `libtpu.so` writes logs to `/tmp/tpu_logs`, so you should
clear these logs if you see warnings about overwriting these files. The typical
reason for this happening is another user wrote to `/tmp/tpu_logs`, which your
user doesn't have permissions to overwrite. 
```bash
sudo rm -rf /tmp/tpu_logs/`
```

## A Simple Example
In the `notebook` directory, we show how to run Plumber to analyze a simple
pipeline.


## Experiments
Below, we describe the main experiments from the paper.

### Models and Datasets
We use MLPerf along with Official Tensorflow pipelines to evaluate Plumber.
The subset of MLPerf we use are the 5 MLPerfv0.6 models: ResNet50, Mask-RCNN,
SSD, Transformer, and GNMT.
These models use the following dataset: ImageNet, COCO, COCO, WMT17, and WMT16.
Note that due to differences in APIs between older Tensorflow and the current version
(namely, that Tensorflow 1 didn't have eager mode and relied on graph-mode),
along with the fact that some internal code was redacted in MLPerf code,
we used at least MLPerfv0.7 variants and had to make some
minor modifications to the code to get it to run.

### Objectives
Each pipeline uses tf.data. We care about 4 pipelines:
* Naive: Using 0 prefetching and parallelism set to 1.
* AUTOTUNE: Using prefetching and parallelism set to tf.data.AUTOTUNE.
* HEURISTIC: Using prefetching and parallelism set to the number of cores.
* Plumber: Using prefetching and parallelism and caching set to what Plumber
  recommends.

What we care about is the rate (speed/throughput) of the pipeline under these
configurations. You can measure this by either counting the number of samples
processed in a given amount of time or you can look at the rate estimated per-epoch.
We provide bash scripts which go through these permutations for the experiments.

### Microbenchmarks (CPU-only)
The microbenchmarks are CPU-only and are run as fast as they can go in a loop.
The rate measured is therefore the maximum possible rate sustained by that
pipeline under that configuration.
These are run on Setup A and B, and do not need to be build with `libtpu.so` for
TPUs; a CPU-only Tensorflow build suffices.
In fact, if you are using the TPU build, you may hit errors like
`tensorflow.python.framework.errors_impl.UnauthenticatedError: ioctl failed`
caused by JAX/TensorFlow's TPU libraries attempting authentication (which is not
used)---in that case, please block `libtpu` from being loaded by intentionally
breaking out of a running `libtpu` session with CTRL+C (which would create `/tmp/libtpu_lockfile`).
Note that you will still need to install `plumber_analysis` with the `install.sh` install
script.

To run these, we provide high level runner scripts, which you can run from the
`microbenchmark` directory.

```bash
bash run_all.sh
bash run_all_PDL.sh
```

**Dataset Path**: We leave the dataset paths hardcoded. Please adjust these in
the respective microbenchmark's `train_sweep.sh` script.
For example, ResNet has the path set [here](https://github.com/mkuchnik/PlumberApp/blob/fae0701838436e33dfd498758c6da2df0a9f5a73/microbenchmarks/simple_resnet/MLPerf/train_sweep.sh#L16).

#### Plotting
Each of the microbenchmark directories will emit a directory within it after
running the above runner scripts, which will contain the results of each of the experiments.
To plot the results to show throughput, local decision optimality,
and predictions for
throughput, we provide an example script in `microbenchmarks/plot_stats.py`.
You can point this file to the directory created during the microbenchmark to
create the corresponding throughput plots.
For example, for ResNet, run the following from the `microbenchmarks` directory:

```bash
python3 plot_stats.py simple_resnet/MLPerf/train_sweep_0_resnet_test
```

### End-to-End (TPU)
The end-to-end results run with the pipeline on CPU and the machine learning
model on a Tensor Processing Unit (TPU).
We specifically use a TPUv3-8 (Setup C).
The rate measured is therefore the minimum of both the model performance and the
pipeline performance.
Just like for microbenchmarks, you will likely have to adjust the path to the
corresponding dataset in the script that runs the experiment.


#### Install
After installing [Tensorflow](https://github.com/mkuchnik/PlumberTensorflow.git) and the [`plumber_analysis`](plumber_analysis) library, you will need to get JAX and JAX libtpu for most
end-to-end runs. For example to get tpu version 0.2.16, run:
```bash
pip install "jax[tpu]==0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

Note that if this libtpu version is different from the system version, you will
have to copy it into the system to avoid version mismatch.
When using miniconda with a python3.7 environment named py37, you will find it
in `$HOME/miniconda3/envs/py37/lib/python3.7/site-packages/libtpu/libtpu.so`.
That file needs to go to `/usr/lib`.

We used `jax[tpu]==0.2.19` in our experiments, along with `jaxlib==0.1.70`. The
script we use install 0.2.16 first and then these, though you can probably just
install these versions directly.

**Run Order.**
Note that we recommend starting with ResNet first before moving to other
end-to-end runs. It is preferable to do ResNet
before other runs because it ensures that the dependencies are working.

**What is using TPU.**
For some runs, we use Tensorflow to control the model, and therefore the TPU.
In other runs, we use [JAX](https://github.com/google/jax) to do the same.
In all cases, Tensorflow is in control of the data pipeline with tf.data.

**Constraints.**
TF2 with native ops is the main target of Plumber. TF1 workflows currently
cannot be resumed, and therefore require either benchmarking the input pipeline
in a TF2 context (i.e., a microbenchmark)
or tracing the pipeline in a TF1 context and resuming it in a
TF2 context. For the latter, we provide the relevant pipelines (i.e., RCNN/GNMT) with
a `get_params.py`, which can start Plumber optimization after tracing in a TF2 context.
Similarly, pipelines with custom kernels (i.e., Flax WMT) cannot be
easily deserialized in either TF2 or TF1, so we provide a `show_params.sh` to
print out the currently recommended parameters after tracing.
The parameters can then be copied over to the input pipeline.


##### ResNet
This pipeline uses JAX (TPU) + Tensorflow.
You should install the minimal requirements to avoid dependency clobber (e.g.,
tensorflow-gpu being installed):
```bash
bash install_jax.sh  # Install JAX + plumber_analysis
bash reinstall_tpu_lib.sh  # Sync libtpu.so version (note: may need sudo)
pip install -r requirements_very_minimal.txt  # Get some pip packages
```

**Dependency Notes.**
Note that sometimes keras gets installed alongside keras-nightly, causing an
error when the model is run. If this happens, make sure there are not multiple
versions of keras/keras-nightly installed.
Note we provide a `requirements.txt` file with a full dependency list to compare
against with `pip freeze` if dependency issues come up.
`requirements_very_minimal.txt` is recommended to install from (above) because the amount of
dependencies in `requirements.txt` makes it difficult to satisfy them all, and
causes issues with `pip` (e.g., the pip dependency solver becomes slow or
complains about incompatible versions).

Then to run with standard 96 threads/cores for resnet18:
```bash
bash official_runners/resnet18_model_96.sh
```

Then to run with standard 96 threads/cores for the linear model:
```bash
bash official_runners/linear_model_96.sh
```

These runs will output something like the following (Linear Autotune runs first
and is shown):
```bash
I0227 07:24:55.529147 140132377122816 train.py:888] Images per loop: 1282048
I0227 07:24:55.529258 140132377122816 train.py:889] Loop bounds: 5
I0227 07:24:55.529352 140132377122816 train.py:896] Loop 0, rate=0.0, current_loop rate=5072923636.407547
I0227 07:24:55.610690 140132377122816 train.py:898] LR=0.0
I0227 07:27:24.209707 140119247529728 train.py:474] train epoch: 1, loss: 6.8660, accuracy: 0.32
I0227 07:27:31.735340 140132377122816 train.py:896] Loop 1, rate=8207.412015521126, current_loop rate=8211.708982486187
I0227 07:27:31.769288 140132377122816 train.py:898] LR=0.08000000566244125
I0227 07:27:31.775491 140119247529728 train.py:474] eval epoch: 1, loss: 6.7389, accuracy: 0.53
I0227 07:27:31.776827 140119247529728 mllog.py:168] :::MLLOG {"namespace": 0, "time_ms": 1645946851775, "event_type": "POINT_IN_TIME", "key": "eval_accuracy", "value": 0.00528, "metadata": {"file": "/home/mkuchnik_andrew_cmu_edu/PlumberApp/end_to_end/TPU/resnet-research-JAX-Distributed-Shampoo-tpu-v4-256/train.py", "lineno": 478, "epoch_num": 1}}
I0227 07:27:31.777864 140119247529728 mllog.py:168] :::MLLOG {"namespace": 0, "time_ms": 1645946851777, "event_type": "INTERVAL_END", "key": "block_stop", "value": null, "metadata": {"file": "/home/mkuchnik_andrew_cmu_edu/PlumberApp/end_to_end/TPU/resnet-research-JAX-Distributed-Shampoo-tpu-v4-256/train.py", "lineno": 482, "first_epoch_num": 1, "epoch_count": 1}}
I0227 07:27:31.780810 140119247529728 mllog.py:168] :::MLLOG {"namespace": 0, "time_ms": 1645946851780, "event_type": "INTERVAL_START", "key": "block_start", "value": null, "metadata": {"file": "/home/mkuchnik_andrew_cmu_edu/PlumberApp/end_to_end/TPU/resnet-research-JAX-Distributed-Shampoo-tpu-v4-256/train.py", "lineno": 489, "first_epoch_num": 2, "epoch_count": 1}}
I0227 07:29:29.230601 140119247529728 train.py:474] train epoch: 2, loss: 6.6795, accuracy: 0.67
I0227 07:29:29.403882 140132377122816 train.py:896] Loop 2, rate=9362.29673340281, current_loop rate=10898.607160054378
I0227 07:29:31.711492 140132377122816 train.py:898] LR=0.01600000075995922
I0227 07:29:31.717248 140119247529728 train.py:474] eval epoch: 2, loss: 6.6185, accuracy: 0.77
I0227 07:29:31.718048 140119247529728 mllog.py:168] :::MLLOG {"namespace": 0, "time_ms": 1645946971717, "event_type": "POINT_IN_TIME", "key": "eval_accuracy", "value": 0.00772, "metadata": {"file": "/home/mkuchnik_andrew_cmu_edu/PlumberApp/end_to_end/TPU/resnet-research-JAX-Distributed-Shampoo-tpu-v4-256/train.py", "lineno": 478, "epoch_num": 2}}
I0227 07:29:31.718366 140119247529728 mllog.py:168] :::MLLOG {"namespace": 0, "time_ms": 1645946971718, "event_type": "INTERVAL_END", "key": "block_stop", "value": null, "metadata": {"file": "/home/mkuchnik_andrew_cmu_edu/PlumberApp/end_to_end/TPU/resnet-research-JAX-Distributed-Shampoo-tpu-v4-256/train.py", "lineno": 482, "first_epoch_num": 2, "epoch_count": 1}}
I0227 07:29:31.719552 140119247529728 mllog.py:168] :::MLLOG {"namespace": 0, "time_ms": 1645946971719, "event_type": "INTERVAL_START", "key": "block_start", "value": null, "metadata": {"file": "/home/mkuchnik_andrew_cmu_edu/PlumberApp/end_to_end/TPU/resnet-research-JAX-Distributed-Shampoo-tpu-v4-256/train.py", "lineno": 489, "first_epoch_num": 3, "epoch_count": 1}}
I0227 07:31:52.656227 140119247529728 train.py:474] train epoch: 3, loss: 6.6371, accuracy: 0.76
I0227 07:31:52.775688 140132377122816 train.py:896] Loop 3, rate=9217.917191839828, current_loop rate=9088.433776361755
```

Where the "rate" is the throughput for that epoch (around 9-10k images/second).
For a naive run, for example, we'd expect much worse performance.
You can run the experiment in a window using `tmux`
and check the CPU utilization while it is running---you should see many cores
being used for the ResNet runs.


##### RCNN
This pipeline uses Tensorflow (TPU).
Run the dependency install script.
```bash
bash install_deps.sh
```

Then to run with standard 96 threads/cores:
```bash
bash official_runners/run_96.sh
```

To obtain the Plumber parameters, please run the naive pipeline with tracing and then run
`get_params.py` with the emitted `stats.pb` placed in the same directory as you
are running `get_params.py` from.

##### SSD
This pipeline uses JAX (TPU) + Tensorflow.
Run the dependency install script.
```bash
bash install_deps.sh
```

We can disable 48/96 cores with:
```bash
bash cores_sweep_48.sh
```

Then to run with 48 threads/cores:
```bash
bash official_runners/run.sh
```

To turn all cores back online:
```bash
sudo ./enable_cores.sh
```

Then to run with 96 threads/cores:
```bash
bash official_runners/run_96.sh
```

##### Transformer (MLPerf)
This pipeline uses JAX (TPU) + Tensorflow.
To run:
```bash
bash official_runners/run.sh
```


##### Transformer (Flax WMT)
This pipeline uses JAX (TPU) + Tensorflow + Tensorflow-text.
You need to install [Tensorflow-text](https://github.com/tensorflow/text), which likely requires building from source
to get compatibility with the Tensorflow fork.
Since we are building a 2.7.0 Tensorflow fork, checkout Tensorflow-text commit
`02587f8b6c6871079d35d6ebe22c97b2c358cc3b`.
To ensure you don't clobber Tensorflow with standard 2.6.0 installs, modify the
`project_version` and tensorflow version to reflect 2.7.0 in `oss_scripts/pip_package/setup.py`.

After installing this dependency, you can run:
```bash
bash official_runners/run.sh
```

Because this pipeline uses custom operators which cannot be easily
deserialized, we utilize the provided `show_params.sh` script to
print out Plumber's recommendation when optimizing the `stats.pb` file emitted by
tracing the naive pipeline.
Note that this requires running the optimization at least twice to get the final
parameters to account for cache insertion.
Each of the pipelines configurations are provided as separate files which can be
copied over `input_pipeline.py`.

##### GNMT
Run the dependency install script.
```bash
bash install_deps.sh
```

Then to run:
```bash
bash official_runners/run.sh
```

To obtain the Plumber parameters, please run the naive pipeline with tracing and then run
`get_params.py` with the emitted `stats.pb` placed in the same directory as you
are running `get_params.py` from.
