"""
Plotting code for microbenchmarks.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib
import argparse

import numpy as np

import tensorflow as tf

sns.set_style("whitegrid")
sns.set_palette("colorblind")
sns.set_context("paper", font_scale=1.65)

MAX_STEPS = 1000
MAX_TRIALS = 3

parser = argparse.ArgumentParser(
    description="Plotting code for microbenchmarks")
parser.add_argument('root_dir', type=str,
                    help='The directory containing all microbenchmark outputs.')
parser.add_argument('--file_tag', type=str,
                    default="resnet",
                    help='The tag to apply to generated PDF filenames.')
args = parser.parse_args()

root_dir = args.root_dir
root_dir = pathlib.Path(root_dir)
assert root_dir.exists(), "{} does not exist".format(root_dir)

FILE_TAG = args.file_tag

def add_baseline(ax, value: float, name_str: str, color_index=None):
    pos_tup = (1, value+1)
    if color_index is not None:
        color = sns.color_palette("dark").as_hex()[color_index]
    else:
        color = "gray"
    ax.axhline(value, xmin=0, linestyle="--", label=name_str, color=color)
    return ax

def add_baselines(ax):
    add_baseline(ax, AUTOTUNE_BEST, "Autotune", 1)
    add_baseline(ax, HEURISTIC_BEST, "Heuristic", 2)
    plt.legend()
    return ax

def grab_rate_from_logfile(logfile_data):
    """Grabs rate from the log of the form:
    mean minibatch rate: XYZ minibatch/sec
    """
    minibatch_sec_line = logfile_data.split("\n")[-3]
    if "mean minibatch rate:" in minibatch_sec_line:
        words = minibatch_sec_line.split()
        rate_i = words.index("rate:") + 1
        rate = words[rate_i]
        try:
            rate = float(rate)
        except ValueError as ex:
            print("Can't understand: {} -> {}".format(words, rate))
            raise ex
        return rate
    else:
        candidates = []
        for line in logfile_data.split("\n"):
            if "mean minibatch rate:" in line:
                candidates.append(line)
        assert len(candidates) == 1, \
            "Expected 1 candidate, got {}".format(candidates)
        minibatch_sec_line = candidates[0]
        if "]" in minibatch_sec_line:
            # Logging on
            words = minibatch_sec_line.split()
            offset = 0
            for i in range(len(words)):
                if "]" in words[i]:
                    offset = i
            rate = words[-2]
            rate = float(rate)
        else:
            words = minibatch_sec_line.split()
            rate = words[3]
            rate = float(rate)
        return rate

def autoconfig_baselines(set_baselines=True):
    if set_baselines:
        autotune_results = []
        heuristic_results = []
        autotune_log = root_dir / "autotune/step_autotune_log.txt"
        if autotune_log.exists():
            with autotune_log.open() as f:
                autotune_rate = grab_rate_from_logfile(autotune_log.read_text())
                autotune_results.append(autotune_rate)
        for i in range(10):
            autotune_log = root_dir / "autotune_{}/step_autotune_log.txt".format(i)
            if autotune_log.exists():
                with autotune_log.open() as f:
                    autotune_rate = grab_rate_from_logfile(autotune_log.read_text())
                    autotune_results.append(autotune_rate)
        heuristic_log = root_dir / "heuristic/step_heuristic_log.txt"
        if heuristic_log.exists():
            with heuristic_log.open() as f:
                heuristic_rate = grab_rate_from_logfile(heuristic_log.read_text())
                heuristic_results.append(heuristic_rate)
        for i in range(10):
            heuristic_log = root_dir / "heuristic_{}/step_heuristic_log.txt".format(i)
            if heuristic_log.exists():
                with heuristic_log.open() as f:
                    heuristic_rate = grab_rate_from_logfile(heuristic_log.read_text())
                    heuristic_results.append(heuristic_rate)
        global AUTOTUNE_BEST
        global HEURISTIC_BEST
        heuristic_results = np.array(heuristic_results)
        autotune_results = np.array(autotune_results)
        heuristic_rate = np.mean(heuristic_results)
        autotune_rate = np.mean(autotune_results)
        HEURISTIC_BEST = heuristic_rate
        AUTOTUNE_BEST = autotune_rate

def plot_estimated_max_rate(filename_or_df, ax=None, color=None, legend=True,
                            p_busy=False, x_axis_key="step", style="scatter",
                            show_convex=True, show_convex_existing=False,
                            show_naive=True, use_native_convex=True,
                            lineplot_style=None, show_autotune=False):
    """Plots throughput rate estimates (e.g., using LP)"""
    if isinstance(filename_or_df, str):
        filename = filename_or_df
        df = pd.read_csv(filename)
    else:
        df = filename_or_df
    _df = df.query("step < {}".format(MAX_STEPS))
    filtered_df = _df.query("deviation == 0")
    if p_busy:
        rate_key = "Estimated_Max_Rate_p_busy"
    else:
        rate_key = "Estimated_Max_Rate"
    id_vars = [x_axis_key, "step"]
    value_vars=["global_minibatch_rate",
                rate_key,
                ]
    if lineplot_style:
        id_vars.append(lineplot_style)
    if show_convex:
        if use_native_convex:
            convex_key = "Estimated_Max_Rate_Convex_Native"
        else:
            convex_key = "Estimated_Max_Rate_Convex"
        value_vars.append(convex_key)
    if show_convex_existing:
        value_vars.append("Estimated_Max_Rate_Convex_Existing")
    if show_naive:
        value_vars.append("Estimated_Max_Rate_Convex_Native_Naive")
    if show_autotune:
        value_vars.append("iterator_autotune_output_rate")
    id_vars = set(id_vars)
    filtered_df_long = pd.melt(filtered_df,
                               id_vars=id_vars,
                               value_vars=value_vars,
                               value_name="Rate (minibatch/sec)",
                               var_name="Rate Type")
    rename_fn = {
        "global_minibatch_rate": "Observed Rate",
        rate_key: "Estimated Max Rate (Local)",
        convex_key: "Estimated Max Rate (LP)",
        "Estimated_Max_Rate_Convex_Existing": "Estimated Max Rate (LP-Existing)",
        "Estimated_Max_Rate_Convex_Native_Naive": "Estimated Max Rate (LP-Naive)",
        "iterator_autotune_output_rate": "Estimated AUTOTUNE Rate",
    }
    filtered_df_long["Rate Type"] = \
        filtered_df_long["Rate Type"].map(rename_fn)
    if style == "scatter":
        plot_f = sns.scatterplot
    else:
        plot_f = sns.lineplot
    if ax is None:
        fig, ax = plt.subplots()
    if color:
        ax = plot_f(data=filtered_df_long,
                    x=x_axis_key,
                    y="Rate (minibatch/sec)",
                    hue="Rate Type",
                    ax=ax,
                    style=lineplot_style,
                    color=color,
                    )
    else:
        ax = plot_f(data=filtered_df_long,
                    x=x_axis_key,
                    y="Rate (minibatch/sec)",
                    hue="Rate Type",
                    style=lineplot_style,
                    ax=ax,
                    )
    return ax


def _patch_step_0(root_dir, mega_df):
    """Add step_0 stats which aren't included

    This is mostly stats_filename, which can then be used to retroactively load
    all data
    """
    # Stats filename
    first_row = mega_df.loc[0, "stats_filename"]
    assert np.all(np.isnan(first_row)), "{} is not null".format(first_row)
    plumber_rewrites_dir = "baseline"
    dir_name = root_dir / plumber_rewrites_dir
    stats_filename_path = dir_name / "step_0.pb"
    assert stats_filename_path.exists(), \
        "{} does not exist".format(stats_filename_path)
    mega_df.loc[0, "stats_filename"] = str(stats_filename_path.resolve())
    return mega_df

def create_mega_df(root_dir):
    dfs = []
    for i in range(MAX_TRIALS):
        plumber_rewrites_dir = "plumber_rewrites_{}".format(i)
        dir_name = root_dir / plumber_rewrites_dir
        if dir_name.exists():
            filename = dir_name / "benchmark_stats.csv"
            if filename.exists():
                df = pd.read_csv(filename)
                df["filename"] = str(filename)
                df["trial"] = i
                df["test_type"] = "Plumber"
                dfs.append(df)
    for i in range(MAX_TRIALS):
        plumber_rewrites_dir = "random_rewrites_{}".format(i)
        dir_name = root_dir / plumber_rewrites_dir
        if dir_name.exists():
            filename = dir_name / "benchmark_stats.csv"
            if filename.exists():
                df = pd.read_csv(filename)
                df["filename"] = str(filename)
                df["trial"] = i
                df["test_type"] = "Random"
                dfs.append(df)
    mega_df = pd.concat(dfs)
    mega_df = mega_df.reset_index()
    mega_df = _patch_step_0(root_dir, mega_df)
    return mega_df

def load_plumber_model(filepath):
    plumber = tf.data.experimental.analysis.PlumberPerformanceModel(filepath)
    model = plumber.model()
    return model

def load_plumber_from_df(mega_df_row):
    _stats_filename = mega_df_row["stats_filename"]
    if not pathlib.Path(_stats_filename).exists():
        # If the path is not absolute, we will likely not find it and default to
        # this relative path
        _dir_filepath = pathlib.Path(mega_df_row["filename"]).parents[0]
        _stats_filename = _dir_filepath / _stats_filename
    model = load_plumber_model(_stats_filename)
    return model

def populate_current_estimated_throughput(mega_df):
    mega_df["current_estimated_throughput"] = -1
    is_plumber_installed = False
    try:
        _ = tf.data.experimental.analysis.PlumberPerformanceModel
        is_plumber_installed = True
    except Exception:
        print("Plumber is not installed!")
    for i in range(len(mega_df)):
        row = mega_df.loc[i]
        if not pd.isna(row["stats_filename"]) and is_plumber_installed:
            model = load_plumber_from_df(row)
            recommendation = model.recommendation()
            autotune_latency = recommendation.iterator_autotune_output_time()
            estimated, thetas, current = recommendation.LP_upper_bounds(
                return_current_throughput=True, )
            mega_df.loc[i,"current_estimated_throughput"] = current
            mega_df.loc[i,"optimal_estimated_throughput"] = estimated
            mega_df.loc[i,"iterator_autotune_output_time"] = autotune_latency
            mega_df.loc[i,"iterator_autotune_output_rate"] = 1./autotune_latency
            if np.isnan(mega_df.loc[i,"global_minibatch_rate"]):
                print(dir(recommendation))
                mb_rate = recommendation.actual_rate()
                mega_df.loc[i,"global_minibatch_rate"] = mb_rate
        else:
            mega_df.loc[i,"current_estimated_throughput"] = float("NaN")
            mega_df.loc[i,"optimal_estimated_throughput"] = float("NaN")
            mega_df.loc[i,"iterator_autotune_output_time"] = float("NaN")
            mega_df.loc[i,"iterator_autotune_output_rate"] = float("NaN")
    return mega_df

autoconfig_baselines()
mega_df = create_mega_df(root_dir)
mega_df = populate_current_estimated_throughput(mega_df)
mega_df["Strategy"] = mega_df["test_type"]
g = sns.lineplot(data=mega_df.query("deviation == 0 and step < "
                                    "{}".format(MAX_STEPS)),
                 x="step",
                 y="global_minibatch_rate",
                 hue="Strategy")
g = add_baselines(g)
g.set_xlabel("Step")
g.set_ylabel("Throughput (minibatch/s)")
plt.tight_layout()
plt.savefig("benchmark_stats_{}_agg.pdf".format(FILE_TAG))
plt.clf()

mega_df = populate_current_estimated_throughput(mega_df)
if "iterator_autotune_output_rate" in mega_df:
    mega_df = mega_df.query("test_type == 'Plumber'")
    ax = plot_estimated_max_rate(mega_df, legend=False, p_busy=True, style="line",
                                 show_naive=False, show_autotune=True)
    ax.set_xlabel("Step")
    plt.tight_layout()
    plt.savefig("max_rate_{}_p_busy_line_mega_autotune.pdf".format(FILE_TAG))
    plt.clf()
