"""Plotting utilities.

Currently only for roofline plots
"""

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np

LINESTYLE_TUPLE = dict([
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),

     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])

def generate_roofline(filename, N, R, X_cpu_bounds, N_star, nodes_rates=None,
                      ylim=None, X_disk_bounds=None):
    max_N = max(N_star * 2, N + 1)
    N_range = np.linspace(0,  N_star)
    X = N_range / R
    plot_params = {"alpha": 0.6}
    plt.plot(N_range, X, "k")
    N_range = np.linspace(N_star,  max_N)
    X = np.full(N_range.shape, N_star / R)
    plt.plot(N_range, X, "k")
    N_range = np.linspace(0,  max_N)
    # Predicted
    plt.plot(N_range, np.full(N_range.shape, X_cpu_bounds),
             "k",
             linestyle=LINESTYLE_TUPLE["densely dashed"],
             label="LP Compute Bound ({:.1f})".format(X_cpu_bounds), **plot_params)
    if X_disk_bounds:
        plt.plot(N_range, np.full(N_range.shape, X_disk_bounds),
                 "k",
                 linestyle=LINESTYLE_TUPLE["dashdotdotted"],
                 label="Disk Bound ({:.1f})".format(X_disk_bounds),
                 **plot_params)
    bound = min(N/R, X_cpu_bounds)
    plt.scatter(N, bound, marker="*", s=40) # Default size is 20
    plt.vlines(N, 0, bound, linestyles="--", label="N={:.2f}".format(N))
    plt.scatter(N_star, N_star / R, marker="*", c="k", s=40)
    plt.vlines(N_star, 0, N_star / R, color="k",
               linestyles=LINESTYLE_TUPLE["densely dotted"],
               label="N*={:.2f}".format(N_star))
    if ylim is None:
        ylim = 2*X_cpu_bounds
    elif ylim == "all":
        ylim = max([x[0] for x in nodes_rates]) + 1
    if nodes_rates:
        nodes_rates = list(filter(lambda x: x[0] < ylim, nodes_rates))
        n = len(nodes_rates)
        # https://stackoverflow.com/questions/4971269/how-to-pick-a-new-color-for-each-plotted-line-within-a-figure-in-matplotlib
        # Go from red to purple to make first instances 'worse'
        color = iter(cm.rainbow(np.linspace(1, 0, n)))
        for i, (bound, name) in enumerate(nodes_rates):
            c = next(color)
            plt.hlines(bound, 0, max_N, color=c, linestyles="--", label=name, **plot_params)
    plt.ylim(0, ylim)
    plt.ylabel("X (minibatches/second)")
    plt.xlabel("N (jobs in system)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()
