"""
Calculate simulation statistics for a directory.

Only used for back-of-the-envelope evaluation of things like subsampling
pipeline sample sizes in the context of caching.
"""

import pathlib
import numpy as np
import random

def get_dir_file_sizes(directory, glob=None):
    p = pathlib.Path(directory)
    if glob is None:
        files = p.glob("*")
    else:
        files = p.glob(glob)
    sizes = get_file_sizes(files)
    return sizes

def simulate_subsampling(sizes, num_samples, num_trials, sampler=None):
    """Input is list of file sizes"""
    return _simulate_subsampling(sizes, num_samples, num_trials,
                                 sampler)

def _simulate_subsampling(sizes, num_samples, num_trials, sampler=None,
                          reduce_early=True):
    sizes = np.array(sizes)
    total_N = len(sizes)
    total_size = np.sum(sizes)
    def draw_samples_choice(data, num_samples):
        subsample = np.random.choice(data, size=(num_samples,),
                                     replace=False)
        return subsample

    def draw_samples_shuffle(data, num_samples):
        items = list(range(len(data)))
        np.random.shuffle(items)
        subsampled_items = items[:num_samples]
        return data[subsampled_items]

    if sampler is None or sampler == "shuffle":
        draw_samples = draw_samples_choice
    elif sampler == "choice":
        draw_samples = draw_samples_choice
    elif sampler == "shuffle":
        draw_samples = draw_samples_shuffle
    else:
        raise ValueError("Unknown shuffler: {}".format(sampler))

    if reduce_early:
        subsamples = np.zeros((num_trials,), dtype=np.int32)
        for i, trial in enumerate(range(num_trials)):
            subsamples[i] = np.sum(draw_samples(sizes, num_samples=num_samples))
        assert subsamples.shape == (num_trials,)
        sub_size = subsamples
    else:
        subsamples = np.zeros((num_samples, num_trials), dtype=np.int32)
        for i, trial in enumerate(range(num_trials)):
            subsamples[:, i] = draw_samples(sizes, num_samples=num_samples)
        assert subsamples.shape == (num_samples, num_trials)
        sub_size = np.sum(subsamples, axis=0)
    sub_size_float = sub_size.astype(float)
    fraction_seen = num_samples / total_N
    expected_size = sub_size_float / fraction_seen
    error = total_size - expected_size
    return error

def get_file_sizes(filepaths: list):
    sizes = []
    for p in filepaths:
        p = pathlib.Path(p)
        size = p.stat().st_size
        sizes.append(size)
    return sizes

class PipelineStage(object):
    def __init__(self):
        self._observed_samples = []

    def get_samples(self, n):
        samples = self._get_samples(self, n)
        self._record_samples(samples)
        return samples

    def get_observed_samples(self):
        return self._observed_samples

    def _get_samples(self, n):
        raise NotImplementedError("Please use inherited pipeline stage")

    def reset(self):
        raise NotImplementedError("Please use inherited pipeline stage")

    def _record_samples(self, samples):
        self._observed_samples.extend(samples)


class SourcePipelineStage(PipelineStage):
    def __init__(self, samples):
        super().__init__()
        self.samples = samples

    def _get_samples(n):
        samples = random.sample(self.samples)
        samples = list(map(self.func, self.samples))
        return samples

class TransformPipelineStage(PipelineStage):
    def __init__(self, stage, func=None):
        super().__init__()
        self.stage = stage
        self.func = func

    def _get_samples(n):
        samples = self.stage.get_samples(n)
        if self.func:
            samples = list(map(self.func, samples))
        return samples

    def get_observed_samples(self):
        observed = self.stage.get_observed_samples()
        observed.append(self._observed_samples)
        return observed
