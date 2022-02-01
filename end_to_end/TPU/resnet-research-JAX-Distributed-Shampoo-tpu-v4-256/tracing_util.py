import time
import sys
import inspect
import types
import numpy as np
import pandas as pd

def tracefunc(frame, event, arg, indent=[0]):
    if event == "call":
        indent[0] += 2
        print("-" * indent[0] + "> call function", frame.f_code.co_name)
    elif event == "return":
        print("<" + "_" * indent[0], "exit_function", frame.f_code.co_name)
        indent[0] -= 2
    return tracefunc

class CallbackConstructor():
    def __init__(self):
        self.samples = None
    def callback_fn(self, new_samples):
        print("Enter callback")
        self.samples = new_samples

class TracedIterator():
    def __init__(self, iterator, callback_fn=None):
        self.iterator = iterator
        self.start_times = []
        self.stop_times = []
        self.i = 0
        if (not isinstance(callback_fn, types.FunctionType)
                or len(inspect.signature(callback_fn).parameters) != 1):
            raise ValueError("Expected callback with 1 parameter for samples")
        self.callback_fn = callback_fn
        self.reference_time = time.perf_counter()

    def __next__(self):
        self.start_times.append((self.i, time.perf_counter()))
        ret = next(self.iterator)
        self.stop_times.append((self.i, time.perf_counter()))
        self.i += 1
        return ret

    def relative_times(self, times):
        return np.array(times) - self.reference_time

    def dict_stats(self):
        return {"start_times": self.relative_times(self.start_times),
                "stop_times": self.relative_times(self.stop_times)}

    def dump_stats(self):
        if self.callback_fn:
            self.callback_fn(self.dict_stats())

    def __del__(self):
        self.dump_stats()

def tag_iterator(iterator, callback_fn):
    """Tags the producer of the pipeline.
    On iterator destruction, calls the callback to obtain a dict of statistics"""
    it = TracedIterator(iterator, callback_fn)
    return it

def tag_consumer(infeed_or_model):
    """Tags the consumer of the pipeline"""
    return


def stats_to_df(stats):
    t1 = stats["start_times"]
    t2 = stats["stop_times"]
    df1 = pd.DataFrame(data=t1, columns=["id", "start_time"])
    df2 = pd.DataFrame(data=t2, columns=["id", "stop_time"])
    df1 = df1.set_index("id")
    df2 = df2.set_index("id")
    df = pd.merge(df1, df2, left_index=True, right_index=True)
    df["elapsed_time"] = df["stop_time"] - df["start_time"]
    df["start_time_diff"] = df["start_time"].diff(1)
    return df

# NOTE(mkuchnik): This is higher performance than settrace
#sys.setprofile(tracefunc)
