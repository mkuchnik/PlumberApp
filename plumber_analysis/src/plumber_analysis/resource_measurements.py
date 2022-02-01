"""Utilities to benchmark the filesystem."""

import os
import tempfile
import subprocess
import json
import multiprocessing

def is_fio_installed() -> bool:
    try:
        _ = subprocess.check_output(["fio", "--help"])
    except FileNotFoundError:
        return False
    return True

def is_fio_path_exists(fio_file_path) -> bool:
    return os.path.exists(fio_file_path)

def is_fio_test_directory_exists(fio_test_path) -> bool:
    return os.path.exists(fio_test_path)

def run_fio_test(fio_file_path):
    """
    Runs a fio test using passed path and returns results
    """
    if not is_fio_installed():
        raise RuntimeError("fio is not installed or not on path."
                           " Install it with 'apt-get fio' or similar.")
    if not is_fio_path_exists(fio_file_path):
        raise FileNotFoundError("fio input '{}' does not "
                                "exist.".format(fio_file_path))
    command = ["fio",  fio_file_path, "--output-format=json"]
    try:
        ret = subprocess.check_output(command, encoding="UTF-8")
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    return ret

def is_cloud_directory(test_path_str: str) -> bool:
    return test_path_str.startswith("gs://")

def benchmark_filesystem(test_path, add_tmp_dir=False, parse_results=True,
                         runtime: int=60, size_gb: int=10, numjobs: int=8,
                         threads=None, direct: bool=False, early_stop: bool=True):
    """
    Runs a large read benchmark on the filesystem pointed to by 'path'.
    Requires fio to be installed.
    Params:
    add_tmp_dir: whether to create a temporary directory in the path for
    testing.
    """
    if not is_fio_test_directory_exists(test_path):
        raise FileNotFoundError("fio test directory '{}' does not "
                                "exist.".format(test_path))
    if is_cloud_directory(str(test_path)):
        raise NotImplementedError("Cloud storage is not supported "
                                  "for benchmarking: {}".format(test_path))
    runtime = int(runtime)
    numjobs = int(numjobs)
    size_gb = int(size_gb)
    if threads is None:
        threads = multiprocessing.cpu_count()
    # TODO(mkuchnik) Threads not used
    #threads = int(threads)
    direct = int(bool(direct))
    if early_stop:
        early_stop_str = "steadystate=bw:20\nsteadystate_duration=5s"
    def run_test(test_path):
        # NOTE(mkuchnik): direct=1 may be preferable, though it can fail with no
        # support
        fio_file_contents = \
            """
            [global]
            time_based=1
            ioengine=posixaio
            rw=read
            size={size_gb}G
            runtime={runtime}
            directory={test_path}
            numjobs={numjobs}
            group_reporting=1
            direct={direct}
            ramp_time=2s
            verify=0
            bs=1M
            iodepth=64
            {early_stop}

            [trivial-readwrite-{runtime}]
            """.format(runtime=runtime, numjobs=numjobs, test_path=test_path,
                       size_gb=size_gb, threads=threads, direct=direct,
                       early_stop=early_stop_str)
        with tempfile.NamedTemporaryFile("w") as tmp:
            tmp.write(fio_file_contents)
            tmp.flush()
            results = run_fio_test(str(tmp.name))
        return results
    if add_tmp_dir:
        with tempfile.TemporaryDirectory(dir=test_path) as final_test_path:
            results = run_test(final_test_path)
    else:
        final_test_path = test_path
        results = run_test(final_test_path)
    if parse_results:
        results = parse_fio_out(results)
    return results

def parse_fio_out(fio_out: str) -> dict:
    try:
        data = json.loads(fio_out)
    except json.decoder.JSONDecodeError as ex:
        print(fio_out)
        raise ex
    jobs = data["jobs"]
    assert len(jobs) == 1, "Expected 1 job, found {}".format(len(jobs))
    job_data = jobs[0]
    return job_data
