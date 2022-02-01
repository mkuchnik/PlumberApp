"""
Utilities for fitting I/O bandwidth curves.
"""

import numpy as np

def find_best_piecewise_linear_fit(x, y, return_params=True):
    """Does a grid search over the parameters to find best fit.
    Function is assumed to be 2 pieces: one slopped and one flat"""
    errors = []
    sweep_i = list(range(2, len(x)))
    for i in sweep_i:
        s_lin = piecewise_linear_fit(x, y, i)
        y_hat = s_lin(x)
        error = error_fn(y, y_hat)
        errors.append(error)
    best_i = np.argmin(errors)
    return piecewise_linear_fit(x, y, sweep_i[best_i],
                                return_params=return_params)

def solve_least_squares(x, y):
    """mx+b"""
    A = np.vstack([x, np.ones(len(x))]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, b

def piecewise_linear_predictor_from_params(params):
    m1 = params["m1"]
    b1 = params["b1"]
    m2 = params["m2"]
    b2 = params["b2"]
    xi = params["x_thresh"]
    def predictor(xx):
        xx = np.asarray(xx)
        xx = np.atleast_1d(xx)
        m1_mask = xx < xi
        yy = np.zeros(len(xx), dtype=np.float64)
        yy[m1_mask] = m1 * xx[m1_mask] + b1
        yy[~m1_mask] = m2 * xx[~m1_mask] + b2
        return yy
    return predictor

def piecewise_linear_fit(x, y, i, return_params=False):
    """Splits the data at index i and then solves two linear equations"""
    x1 = x[:i]
    y1 = y[:i]
    x2 = x[i:]
    y2 = y[i:]
    m1, b1 = solve_least_squares(x1, y1)
    m2, b2 = solve_least_squares(x2, y2)
    xi = np.max(x1)
    def predictor(xx):
        xx = np.asarray(xx)
        xx = np.atleast_1d(xx)
        m1_mask = xx < xi
        yy = np.zeros(len(xx), dtype=np.float64)
        yy[m1_mask] = m1 * xx[m1_mask] + b1
        yy[~m1_mask] = m2 * xx[~m1_mask] + b2
        return yy
    if return_params:
        params = {"m1": m1,
                  "b1": b1,
                  "m2": m2,
                  "b2": b2,
                  "x_thresh": xi}
        return predictor, params
    else:
        return predictor

def error_fn(y, y_hat):
    """A one-sided error function that penalizes big y values more"""
    y = np.asarray(y)
    y_hat = np.asarray(y_hat)
    weight = 1. / (np.abs(y - np.max(y)) + 1)
    diff = (y - y_hat)
    diff[diff < 0] = 0
    return np.mean(weight * diff ** 2)
