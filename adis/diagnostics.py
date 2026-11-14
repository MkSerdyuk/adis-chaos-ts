from math import factorial

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import entropy


def entropy_complexity(series, dimension=5, delay=1):
    series = np.asarray(series)
    embedded = series[
        np.arange(len(series) - (dimension - 1) * delay)[:, None] + np.arange(dimension) * delay
    ]
    ranks = np.argsort(embedded, axis=1, kind="stable")
    _, counts = np.unique(ranks, axis=0, return_counts=True)
    states = factorial(dimension)
    probabilities = np.zeros(states)
    probabilities[: len(counts)] = counts / counts.sum()
    uniform = np.full(states, 1 / states)
    h = entropy(probabilities) / np.log(states)
    divergence = entropy((probabilities + uniform) / 2) - (entropy(probabilities) + np.log(states)) / 2
    point_mass = np.zeros(states)
    point_mass[0] = 1
    maximum = entropy((point_mass + uniform) / 2) - np.log(states) / 2
    return float(h), float(h * divergence / maximum)


def eckmann_lyapunov(series, embedding=10, matrix_dimension=4, min_neighbors=12, min_separation=20, dt=1.0):
    if (embedding - 1) % (matrix_dimension - 1):
        raise ValueError("embedding - 1 must be divisible by matrix_dimension - 1")
    step = (embedding - 1) // (matrix_dimension - 1)
    series = np.asarray(series, float)
    count = len(series) - embedding - step + 1
    vectors = series[np.arange(count)[:, None] + np.arange(embedding)]
    tree = cKDTree(vectors)
    q = np.eye(matrix_dimension)
    logs = np.zeros(matrix_dimension)
    for i, vector in enumerate(vectors):
        k = min(count, min_neighbors + 2 * min_separation + 2)
        neighbors = tree.query(vector, k=k, p=np.inf)[1]
        neighbors = neighbors[np.abs(neighbors - i) > min_separation][:min_neighbors]
        if len(neighbors) < min_neighbors:
            raise ValueError("Insufficient temporally separated Lyapunov neighbors")
        a = (
            series[neighbors[:, None] + np.arange(matrix_dimension) * step]
            - series[i + np.arange(matrix_dimension) * step]
        )
        b = series[neighbors + matrix_dimension * step] - series[i + matrix_dimension * step]
        coefficients = np.linalg.lstsq(a, b, rcond=None)[0]
        tangent = np.zeros((matrix_dimension, matrix_dimension))
        tangent[:-1, 1:] = np.eye(matrix_dimension - 1)
        tangent[-1] = coefficients
        q, r = np.linalg.qr(tangent @ q)
        diagonal = np.diag(r)
        q *= np.where(diagonal < 0, -1, 1)
        logs += np.log(np.abs(diagonal))
    return logs / (count * step * dt)


def predictability_horizon(lyapunov=0.92, initial_error=1e-4, maximum_error=0.1, dt=0.1):
    time = np.log(maximum_error / initial_error) / lyapunov
    return {"time": float(time), "steps": float(time / dt)}
