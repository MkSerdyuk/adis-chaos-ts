from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit


@njit(cache=True)
def _lorenz_rhs(state, sigma, beta, rho):
    x, y, z = state
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])


@njit(cache=True)
def lorenz(n, discard=3000, dt=0.1, initial=(1.0, 1.0, 1.0), sigma=10.0, beta=8 / 3, rho=28.0, component=0):
    state = np.array(initial)
    values = np.empty(n)
    for i in range(n + discard):
        if i >= discard:
            values[i - discard] = state[component]
        a = _lorenz_rhs(state, sigma, beta, rho)
        b = _lorenz_rhs(state + dt * a / 2, sigma, beta, rho)
        c = _lorenz_rhs(state + dt * b / 2, sigma, beta, rho)
        d = _lorenz_rhs(state + dt * c, sigma, beta, rho)
        state += dt * (a + 2 * b + 2 * c + d) / 6
    return values


def windows(series, origins, context, horizon):
    origins = np.asarray(origins)
    histories = series[origins[:, None] - np.arange(context - 1, -1, -1)]
    targets = series[origins[:, None] + np.arange(1, horizon + 1)]
    return histories, targets


def test_windows(series, start, context, horizon, stride=1, evaluation="paper"):
    if evaluation == "paper":
        origins = np.arange(start - horizon, len(series) - 1, stride)
    elif evaluation == "common_origins":
        origins = np.arange(start - 1, len(series) - horizon, stride)
    else:
        raise ValueError(evaluation)
    histories = series[origins[:, None] - np.arange(context - 1, -1, -1)]
    indices = origins[:, None] + np.arange(1, horizon + 1)
    oracle_truth = np.full(indices.shape, np.nan)
    available = indices < len(series)
    oracle_truth[available] = series[indices[available]]
    truth = np.where(indices >= start, oracle_truth, np.nan)
    return origins, histories, truth, oracle_truth


def german_load(path):
    return pd.read_csv(Path(path))["load_mw"].to_numpy()
