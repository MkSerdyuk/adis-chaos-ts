from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise, product
from pathlib import Path

import numpy as np
from numba import njit, prange
from numba.typed import List
from scipy.spatial import cKDTree

from .clustering import wishart


@njit(cache=True, nogil=True)
def _query_one(history, reliability, lags, motifs, roots, left, right, axes, epsilon, decay):
    indices = []
    depths = []
    distances = []
    dimension = lags.shape[1]
    query = np.empty(dimension)
    stack = np.empty(128, np.int64)
    for p in range(len(lags)):
        valid = True
        weight = 0.0
        for j in range(dimension):
            query[j] = history[-lags[p, j]]
            weight += reliability[-lags[p, j]]
            if not np.isfinite(query[j]):
                valid = False
        if not valid or roots[p] < 0:
            continue
        size = 1
        stack[0] = roots[p]
        while size:
            size -= 1
            node = stack[size]
            squared = 0.0
            for j in range(dimension):
                squared += (query[j] - motifs[node, j]) ** 2
            if squared < epsilon**2:
                indices.append(node)
                depths.append(decay * weight / dimension)
                distances.append(1 - np.sqrt(squared) / epsilon)
            axis = axes[node]
            delta = query[axis] - motifs[node, axis]
            if left[node] >= 0 and delta < epsilon:
                stack[size] = left[node]
                size += 1
            if right[node] >= 0 and delta > -epsilon:
                stack[size] = right[node]
                size += 1
    indices_array = np.array(indices, dtype=np.int64)
    order = np.argsort(indices_array)
    return (motifs[indices_array[order], -1], np.array(depths)[order], np.array(distances)[order])


@njit(cache=True, parallel=True, nogil=True)
def _query_batch(histories, reliability, lags, motifs, roots, left, right, axes, epsilon, decay):
    result = List()
    for _ in range(len(histories)):
        result.append((np.empty(0), np.empty(0), np.empty(0)))
    for i in prange(len(histories)):
        result[np.int64(i)] = _query_one(
            histories[i], reliability[i], lags, motifs, roots, left, right, axes, epsilon, decay
        )
    return result


def pattern_offsets(pattern):
    return np.r_[0, np.cumsum(pattern)]


def embed(series, pattern):
    offsets = pattern_offsets(pattern)
    return np.asarray(series)[np.arange(len(series) - offsets[-1])[:, None] + offsets]


class MotifModel:
    def __init__(self, length=4, max_gap=10, r=11, mu=0.2, clustered=True):
        if length < 2 or max_gap < 1:
            raise ValueError("Motifs require length >= 2 and max_gap >= 1")
        self.length, self.max_gap = length, max_gap
        self.r, self.mu, self.clustered = r, mu, clustered
        self.patterns = np.array(list(product(range(1, max_gap + 1), repeat=length - 1)))
        offsets = np.c_[np.zeros(len(self.patterns), int), self.patterns.cumsum(axis=1)]
        self.lags = offsets[:, -1, None] - offsets[:, :-1]
        self.context_size = int(self.lags.max())

    def fit(self, series, workers=4):
        series = np.asarray(series, float)
        if series.ndim != 1 or not np.isfinite(series).all():
            raise ValueError("Expected a finite univariate training series")

        def fit_pattern(pattern):
            vectors = embed(series, pattern)
            if not self.clustered:
                return vectors
            labels = wishart(vectors, self.r, self.mu)
            ids = np.unique(labels[labels > 0])
            return np.array([vectors[labels == i].mean(axis=0) for i in ids]).reshape(-1, self.length)

        # Compile before workers enter the numerical kernel.
        wishart(embed(series[:100], self.patterns[0]), self.r, self.mu)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            groups = list(pool.map(fit_pattern, self.patterns))
        self.sizes = np.array([len(x) for x in groups])
        self.motifs = np.concatenate(groups)
        self.pattern_ids = np.repeat(np.arange(len(groups)), self.sizes)
        self._index()
        return self

    def _index(self):
        if not len(self.motifs):
            raise ValueError("No non-noise motifs found")
        self.tree = cKDTree(np.c_[self.pattern_ids, self.motifs[:, :-1]])
        self.left = np.full(len(self.motifs), -1, int)
        self.right = self.left.copy()
        self.axes = np.zeros(len(self.motifs), int)

        def build(indices, depth=0):
            if not len(indices):
                return -1
            axis = depth % (self.length - 1)
            middle = len(indices) // 2
            indices = indices[np.argpartition(self.motifs[indices, axis], middle)]
            node = indices[middle]
            self.axes[node] = axis
            self.left[node] = build(indices[:middle], depth + 1)
            self.right[node] = build(indices[middle + 1 :], depth + 1)
            return node

        boundaries = np.r_[0, self.sizes.cumsum()]
        self.roots = np.array([build(np.arange(a, b)) for a, b in pairwise(boundaries)])

    def save(self, path):
        np.savez_compressed(
            path,
            motifs=self.motifs,
            sizes=self.sizes,
            settings=[self.length, self.max_gap, self.r, self.mu, self.clustered],
        )

    @classmethod
    def load(cls, path):
        data = np.load(Path(path))
        length, gap, r, mu, clustered = data["settings"]
        model = cls(int(length), int(gap), int(r), mu, bool(clustered))
        model.motifs, model.sizes = data["motifs"], data["sizes"]
        model.pattern_ids = np.repeat(np.arange(len(model.patterns)), model.sizes)
        model._index()
        return model

    def candidates(self, histories, reliability, epsilon, decay=0.99, force=False):
        histories, reliability = np.asarray(histories), np.asarray(reliability)
        if (
            histories.ndim != 2
            or histories.shape != reliability.shape
            or histories.shape[1] < self.context_size
        ):
            raise ValueError("Candidate histories and reliability must share a sufficient 2D context")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if not force:
            return _query_batch(
                histories,
                reliability,
                self.lags,
                self.motifs,
                self.roots,
                self.left,
                self.right,
                self.axes,
                epsilon,
                decay,
            )
        n = len(histories)
        contexts = histories[:, -self.lags]
        valid = np.isfinite(contexts).all(axis=2)
        row, pattern = np.nonzero(valid)
        queries = np.c_[pattern, contexts[row, pattern]]
        if force:
            distance, index = self.tree.query(queries)
            # The pattern coordinate isolates radius searches, not unrestricted nearest searches.
            wrong = self.pattern_ids[index] != pattern
            for j in np.flatnonzero(wrong):
                candidates = np.flatnonzero(self.pattern_ids == pattern[j])
                if not len(candidates):
                    raise ValueError("Forced prediction needs at least one motif per pattern")
                d = np.linalg.norm(self.motifs[candidates, :-1] - queries[j, 1:], axis=1)
                index[j], distance[j] = candidates[np.argmin(d)], d.min()
            balls = [[i] for i in index]
        else:
            balls = self.tree.query_ball_point(queries, epsilon, workers=1, return_sorted=True)
        lengths = np.fromiter(map(len, balls), int, len(balls))
        query_ids = np.repeat(np.arange(len(queries)), lengths)
        indices = np.concatenate(balls).astype(int) if lengths.sum() else np.empty(0, int)
        rows = row[query_ids]
        selected_patterns = pattern[query_ids]
        distance = np.linalg.norm(contexts[rows, selected_patterns] - self.motifs[indices, :-1], axis=1)
        if not force:
            keep = distance < epsilon
            rows, selected_patterns, indices, distance = (
                a[keep] for a in (rows, selected_patterns, indices, distance)
            )
        depth_weight = decay * reliability[rows[:, None], -self.lags[selected_patterns]].mean(axis=1)
        distance_weight = np.maximum(0, (epsilon - distance) / epsilon)
        counts = np.bincount(rows, minlength=n)
        boundaries = np.r_[0, counts.cumsum()]
        values = self.motifs[indices, -1]
        return [(values[a:b], depth_weight[a:b], distance_weight[a:b]) for a, b in pairwise(boundaries)]
