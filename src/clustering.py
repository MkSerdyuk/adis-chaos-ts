import numpy as np
from numba import njit
from scipy.spatial import cKDTree
from scipy.special import gammaln


@njit(cache=True)
def _root(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


@njit(cache=True, nogil=True)
def _wishart(order, indptr, neighbors, density, mu):
    n = len(order)
    labels = np.zeros(n, np.int64)
    parent = np.arange(n + 1)
    completed = np.zeros(n + 1, np.bool_)
    completed[0] = True
    low = np.zeros(n + 1)
    high = np.zeros(n + 1)
    visited = np.zeros(n, np.bool_)
    number = 0
    for point in order:
        connected = set()
        for k in range(indptr[point], indptr[point + 1]):
            other = neighbors[k]
            if visited[other]:
                connected.add(_root(parent, labels[other]))
        visited[point] = True
        if not connected:
            number += 1
            labels[point] = number
            low[number] = high[number] = density[point]
            continue
        roots = sorted(connected)
        all_completed = True
        for c in roots:
            if not completed[c]:
                all_completed = False
        if all_completed:
            continue
        significant = [c for c in roots if c > 0 and high[c] - low[c] >= mu]
        if len(significant) > 1 or roots[0] == 0:
            for c in roots:
                if c == 0:
                    continue
                if high[c] - low[c] >= mu:
                    completed[c] = True
                else:
                    parent[c] = 0
            continue
        target = roots[0]
        for c in roots[1:]:
            parent[c] = target
            low[target] = min(low[target], low[c])
            high[target] = max(high[target], high[c])
        labels[point] = target
        low[target] = min(low[target], density[point])
        high[target] = max(high[target], density[point])
    for i in range(n):
        labels[i] = _root(parent, labels[i])
    return labels


def wishart(x, r=11, mu=0.2):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if not np.isfinite(x).all() or not 1 <= r < len(x) or mu < 0:
        raise ValueError("Wishart requires finite observations, 1 <= r < n and mu >= 0")
    tree = cKDTree(x)
    radii = tree.query(x, k=r + 1)[0][:, -1]
    balls = tree.query_ball_point(x, np.nextafter(radii, np.inf))
    indptr = np.r_[0, np.cumsum([len(a) for a in balls])]
    neighbors = np.concatenate(balls).astype(np.int64)
    dimension = x.shape[1]
    log_volume = dimension / 2 * np.log(np.pi) - gammaln(dimension / 2 + 1)
    log_density = np.log(r / len(x)) - log_volume - dimension * np.log(np.maximum(radii, np.finfo(float).eps))
    density = np.exp(np.minimum(log_density, 600))
    return _wishart(np.argsort(radii, kind="stable"), indptr, neighbors, density, mu)


@njit(cache=True)
def dbscan_1d(x, eps=0.01, min_samples=3):
    n = len(x)
    order = np.argsort(x)
    labels = np.full(n, -1, np.int64)
    left = np.empty(n, np.int64)
    right = np.empty(n, np.int64)
    lo = hi = 0
    for i in range(n):
        while x[order[i]] - x[order[lo]] > eps:
            lo += 1
        while hi < n and x[order[hi]] - x[order[i]] <= eps:
            hi += 1
        left[i], right[i] = lo, hi
    cluster = -1
    last_core = -1
    assigned_until = 0
    for i in range(n):
        if right[i] - left[i] < min_samples:
            continue
        if last_core < 0 or x[order[i]] - x[order[last_core]] > eps:
            cluster += 1
        for j in range(max(left[i], assigned_until), right[i]):
            labels[order[j]] = cluster
        assigned_until = right[i]
        last_core = i
    return labels


def cluster_centers(values, method="dbscan", eps=0.01, min_samples=3, r=11, mu=0.2):
    values = np.asarray(values, float)
    if not len(values):
        return np.empty(0), np.empty(0, int), np.empty(0, int)
    if method == "dbscan":
        labels = dbscan_1d(values, eps, min_samples)
    elif method == "wishart":
        if len(values) <= r:
            return np.empty(0), np.empty(0, int), np.full(len(values), -1)
        labels = wishart(values, r, mu)
        labels = np.where(labels > 0, labels, -1)
    else:
        raise ValueError(method)
    ids = np.unique(labels[labels >= 0])
    sizes = np.array([np.sum(labels == i) for i in ids])
    centers = np.array([values[labels == i].mean() for i in ids])
    return centers, sizes, labels
