from dataclasses import dataclass, replace

import numpy as np
from numba import njit, prange
from numba.typed import List

from .clustering import cluster_centers, dbscan_1d


@dataclass(frozen=True)
class ForecastConfig:
    aggregation: str = "avg"
    selection: str = "cm"
    epsilon: float = 0.01
    oracle_tolerance: float = 0.05
    decay: float = 0.99
    cluster_eps: float = 0.01
    min_samples: int = 3
    weight_threshold: float = 0.5
    perturbation_variance: float = 1e-5
    entropy_threshold: float = 0.8
    spread_threshold: float = 0.1
    seed: int = 42


DEFAULT_CONFIG = ForecastConfig()


FEATURE_NAMES = [
    "count",
    "multimodality",
    "percentile_range",
    "variance",
    "moment3",
    "moment4",
    "kurtosis",
    "entropy",
    "clusters",
    "largest_fraction",
    "size_difference",
    "spread",
]


def candidate_features(values, config):
    if not len(values):
        return np.zeros(len(FEATURE_NAMES))
    centered = values - values.mean()
    variance = np.mean(centered**2)
    centers, sizes, _ = cluster_centers(values, eps=config.cluster_eps, min_samples=config.min_samples)
    sizes = np.sort(sizes)[::-1]
    first = sizes[0] / len(values) if len(sizes) else 0
    second = sizes[1] / len(values) if len(sizes) > 1 else 0
    histogram = np.histogram(values, bins=np.linspace(0, 1, 21))[0]
    probabilities = histogram[histogram > 0] / len(values)
    entropy = -np.sum(probabilities * np.log(probabilities)) / np.log(20)
    return np.array(
        [
            len(values),
            1 - first,
            np.diff(np.quantile(values, [0.05, 0.95]))[0],
            variance,
            np.mean(centered**3),
            np.mean(centered**4),
            np.mean(centered**4) / variance**2 if variance > 0 else 0,
            entropy,
            len(centers),
            first,
            first - second,
            np.ptp(values),
        ]
    )


def aggregate(candidates, config, rng):
    values, depth, distance = candidates
    if not len(values):
        return np.nan, 0.0
    mode = config.aggregation
    mask = np.ones(len(values), bool)
    weights = np.ones(len(values))
    if mode == "mf":
        bins = np.floor(values / config.cluster_eps).astype(int)
        ids, counts = np.unique(bins, return_counts=True)
        mask = bins == ids[np.argmax(counts)]
    elif mode in {"wavgl", "wavgd", "wavgc"}:
        weights = {"wavgl": depth, "wavgd": distance, "wavgc": depth * distance}[mode]
    elif mode in {"clc", "mfp", "threshold_clc", "fuzzy", "roulette", "wishart"}:
        if mode == "threshold_clc":
            mask = depth > config.weight_threshold
        if not mask.any():
            return np.nan, 0.0
        ids = np.flatnonzero(mask)
        method = "wishart" if mode == "wishart" else "dbscan"
        centers, sizes, labels = cluster_centers(values[mask], method, config.cluster_eps, config.min_samples)
        if not len(centers):
            return np.nan, 0.0
        cluster_ids = np.unique(labels[labels >= 0])
        masses = np.array([depth[ids[labels == c]].sum() for c in cluster_ids])
        if mode == "roulette":
            choice = rng.choice(len(centers), p=masses / masses.sum())
        else:
            choice = np.argmax(masses if mode == "fuzzy" else sizes)
        mask[:] = False
        mask[ids[labels == cluster_ids[choice]]] = True
        if mode == "fuzzy":
            weights = depth
    elif mode != "avg":
        raise ValueError(mode)
    if weights[mask].sum() <= 0:
        return np.nan, 0.0
    value = np.average(values[mask], weights=weights[mask])
    if mode == "mfp":
        value += rng.normal(0, np.sqrt(config.perturbation_variance))
    return value, depth[mask].mean()


@njit(cache=True, parallel=True, nogil=True)
def _aggregate_batch(candidates, mode, eps, min_samples):
    result = np.empty((len(candidates), 2))
    for row in prange(len(candidates)):
        values, depth, distance = candidates[row]
        result[row, 0], result[row, 1] = np.nan, 0.0
        if not len(values):
            continue
        labels = np.zeros(len(values), np.int64)
        chosen = 0
        if mode >= 4:
            labels = dbscan_1d(values, eps, min_samples)
            largest = labels.max()
            if largest < 0:
                continue
            counts = np.zeros(largest + 1, np.int64)
            for label in labels:
                if label >= 0:
                    counts[label] += 1
            chosen = np.argmax(counts)
        total = mass = reliability = 0.0
        count = 0
        for j in range(len(values)):
            if labels[j] != chosen:
                continue
            weight = 1.0
            if mode == 1:
                weight = depth[j]
            elif mode == 2:
                weight = distance[j]
            elif mode == 3:
                weight = depth[j] * distance[j]
            total += weight * values[j]
            mass += weight
            reliability += depth[j]
            count += 1
        if mass > 0:
            result[row, 0], result[row, 1] = total / mass, reliability / count
    return result


def _step(model, histories, reliability, config, rng, features=False):
    candidates = model.candidates(
        histories, reliability, config.epsilon, config.decay, force=config.selection == "fp"
    )
    modes = {"avg": 0, "wavgl": 1, "wavgd": 2, "wavgc": 3, "clc": 4, "mfp": 5}
    if config.aggregation in modes:
        if not isinstance(candidates, List):
            candidates = List(candidates)
        result = _aggregate_batch(
            candidates, modes[config.aggregation], config.cluster_eps, config.min_samples
        )
        if config.aggregation == "mfp":
            valid = np.isfinite(result[:, 0])
            result[valid, 0] += rng.normal(0, np.sqrt(config.perturbation_variance), valid.sum())
    else:
        result = np.array([aggregate(c, config, rng) for c in candidates])
    extra = np.array([candidate_features(c[0], config) for c in candidates]) if features else None
    return result[:, 0], result[:, 1], candidates, extra


def _spread(candidates, config):
    result = np.full(len(candidates), np.nan)
    for i, (values, _, _) in enumerate(candidates):
        count_clusters = config.selection in {"dbscan", "wshrt"}
        if config.selection in {"rgdbscan", "dbscan"}:
            values = cluster_centers(values, "dbscan", config.cluster_eps, config.min_samples)[0]
        elif config.selection in {"rgwshrt", "wshrt"}:
            values = cluster_centers(values, "wishart")[0]
        if count_clusters:
            result[i] = len(values)
        elif len(values):
            result[i] = np.ptp(values)
    return result


def forecast(
    model, contexts, horizon, config=DEFAULT_CONFIG, classifier=None, truth=None, collect_features=False
):
    contexts = np.atleast_2d(np.asarray(contexts, float))
    if contexts.shape[1] < model.context_size or not np.isfinite(contexts).all():
        raise ValueError("A complete observed context of sufficient length is required")
    if horizon < 1 or config.epsilon <= 0:
        raise ValueError("Positive horizon and epsilon required")
    if config.selection == "fp" and (config.aggregation not in {"avg", "wavgl"} or classifier is not None):
        raise ValueError("Forced prediction requires avg or wavgl aggregation without a classifier")
    if config.selection == "ap" and (truth is None or np.shape(truth) != (len(contexts), horizon)):
        raise ValueError("The AP oracle requires a separately supplied truth matrix")
    n, observed = contexts.shape
    histories = np.c_[contexts, np.full((n, horizon + 2), np.nan)]
    reliability = np.c_[np.ones_like(contexts), np.zeros((n, horizon + 2))]
    predictions = np.full((n, horizon), np.nan)
    feature_log = np.zeros((n, horizon, len(FEATURE_NAMES))) if collect_features else None
    rng = np.random.default_rng(config.seed)
    for h in range(horizon):
        stop = observed + h
        need_features = collect_features or classifier is not None or config.selection == "en"
        values, weights, candidates, features = _step(
            model, histories[:, :stop], reliability[:, :stop], config, rng, need_features
        )
        accept = np.isfinite(values)
        if config.selection == "ap":
            accept &= np.abs(values - truth[:, h]) < config.oracle_tolerance
        elif config.selection == "en":
            accept &= features[:, 7] <= config.entropy_threshold
        elif classifier is not None:
            accept &= classifier.predict(features).astype(bool)
        elif config.selection in {"rg", "rgdbscan", "rgwshrt", "dbscan", "wshrt"}:
            tentative = histories[:, : stop + 2].copy()
            tentative_weights = reliability[:, : stop + 2].copy()
            spreads = [_spread(candidates, config)]
            tentative[:, stop] = values
            tentative_weights[:, stop] = weights
            for ahead in range(1, 3):
                v, w, c, _ = _step(
                    model, tentative[:, : stop + ahead], tentative_weights[:, : stop + ahead], config, rng
                )
                spreads.append(_spread(c, config))
                if ahead == 1:
                    tentative[:, stop + ahead] = v
                    tentative_weights[:, stop + ahead] = w
            accept &= ~((spreads[0] < spreads[1]) & (spreads[1] < spreads[2]))
        elif config.selection == "spread":
            accept &= _spread(candidates, config) <= config.spread_threshold
        elif config.selection not in {"cm", "fp"}:
            raise ValueError(f"Unknown selection rule: {config.selection}")
        values[~accept], weights[~accept] = np.nan, 0
        predictions[:, h] = histories[:, stop] = values
        reliability[:, stop] = weights
        if collect_features:
            feature_log[:, h] = features
    return (predictions, feature_log) if collect_features else predictions


def trajectory_forecast(model, contexts, horizon, config, trajectories=16, truth=None):
    if config.aggregation != "avg":
        raise ValueError("Trajectory forecasts require avg endpoints; intermediate steps use mfp")
    contexts = np.atleast_2d(contexts)
    n, observed = contexts.shape
    histories = np.repeat(contexts, trajectories, axis=0)
    histories = np.c_[histories, np.full((n * trajectories, horizon + 2), np.nan)]
    reliability = np.c_[np.ones((n * trajectories, observed)), np.zeros((n * trajectories, horizon + 2))]
    result = np.full((n, horizon), np.nan)
    rng = np.random.default_rng(config.seed)
    inner = replace(config, aggregation="mfp", selection="cm")
    if config.selection == "ap" and (truth is None or np.shape(truth) != (n, horizon)):
        raise ValueError("AP trajectories require ground truth")
    for h in range(horizon):
        stop = observed + h
        values, weights, _, _ = _step(model, histories[:, :stop], reliability[:, :stop], inner, rng)
        endpoints = values.reshape(n, trajectories)
        valid = np.isfinite(endpoints)
        counts = valid.sum(axis=1)
        means = np.divide(np.nansum(endpoints, axis=1), counts, out=np.full(n, np.nan), where=counts > 0)
        accept = counts > 0
        if config.selection == "ap":
            accept &= np.abs(means - truth[:, h]) < config.oracle_tolerance
        elif config.selection in {"rg", "rgdbscan", "rgwshrt", "dbscan", "wshrt", "spread"}:

            def spread(points):
                return _spread([(row[np.isfinite(row)], None, None) for row in points], config)

            first_spread = spread(endpoints)
            if config.selection == "spread":
                accept &= first_spread <= config.spread_threshold
            else:
                tentative = histories[:, : stop + 2].copy()
                tentative_weights = reliability[:, : stop + 2].copy()
                tentative[:, stop], tentative_weights[:, stop] = values, weights
                spreads = [first_spread]
                for ahead in range(1, 3):
                    v, w, _, _ = _step(
                        model, tentative[:, : stop + ahead], tentative_weights[:, : stop + ahead], inner, rng
                    )
                    spreads.append(spread(v.reshape(n, trajectories)))
                    if ahead == 1:
                        tentative[:, stop + ahead], tentative_weights[:, stop + ahead] = v, w
                accept &= ~((spreads[0] < spreads[1]) & (spreads[1] < spreads[2]))
        elif config.selection == "wamdt":
            path_weights = weights.reshape(n, trajectories)
            deviations = np.abs(endpoints - endpoints[:, :1])
            comparable = np.isfinite(deviations)
            mass = np.sum(np.where(comparable, path_weights, 0), axis=1)
            disagreement = np.divide(
                np.nansum(deviations * path_weights, axis=1), mass, out=np.full(n, np.nan), where=mass > 0
            )
            accept &= np.isfinite(endpoints[:, 0]) & (disagreement <= config.spread_threshold)
        elif config.selection != "cm":
            raise ValueError(config.selection)
        result[accept, h] = means[accept]
        values[np.repeat(~accept, trajectories)] = np.nan
        weights[np.repeat(~accept, trajectories)] = 0
        histories[:, stop], reliability[:, stop] = values, weights
    return result


def branching_forecast(model, context, horizon, config, prune_weight=0.0, max_paths=100_000):
    if config.selection != "cm":
        raise ValueError("Branching supports selection='cm'; path pruning uses prune_weight")
    paths = [(np.asarray(context, float), np.ones(len(context)), 1.0)]
    predictions = np.full(horizon, np.nan)
    for h in range(horizon):
        children = []
        for history, reliability, path_weight in paths:
            candidates = model.candidates(
                history[None, :], reliability[None, :], config.epsilon, config.decay
            )[0]
            values, weights, _ = candidates
            centers, _, labels = cluster_centers(
                values, eps=config.cluster_eps, min_samples=config.min_samples
            )
            for center, label in zip(centers, np.unique(labels[labels >= 0]), strict=True):
                weight = weights[labels == label].mean()
                if path_weight * weight >= prune_weight:
                    children.append(
                        (np.r_[history, center], np.r_[reliability, weight], path_weight * weight)
                    )
            if not len(centers) and path_weight >= prune_weight:
                children.append((np.r_[history, np.nan], np.r_[reliability, 0], path_weight))
        if len(children) > max_paths:
            raise RuntimeError("Branching exceeded max_paths; choose an explicit pruning threshold")
        paths = children
        endpoints = np.array([path[0][-1] for path in paths])
        if np.isfinite(endpoints).any():
            predictions[h] = endpoints[np.isfinite(endpoints)].mean()
    return predictions


def reduced_history_forecast(model, history, horizon, config, previous=3, rule="cwat"):
    history = np.asarray(history)
    paths, reliability = [], []
    rng = np.random.default_rng(config.seed)
    inner = replace(config, selection="cm")
    for shift in range(previous + 1):
        context = history[: len(history) - shift][-model.context_size :]
        path, rel = context[None, :], np.ones((1, len(context)))
        for _ in range(shift):
            value, weight, _, _ = _step(model, path, rel, inner, rng)
            path, rel = np.c_[path, value], np.c_[rel, weight]
        paths.append(path[0, -model.context_size :])
        reliability.append(rel[0, -model.context_size :])
    paths, reliability = np.array(paths), np.array(reliability)
    result = np.full(horizon, np.nan)
    geometric = 0.5 ** np.arange(1, previous + 1)
    linear = np.arange(previous, 0, -1) * 2 / (previous * (previous + 1))
    for h in range(horizon):
        values, weights, _, _ = _step(model, paths, reliability, inner, rng)
        if rule == "cwat":
            disagreement = abs(values[0] - np.average(values[1:], weights=geometric))
        elif rule in {"amd", "wamd"}:
            differences = np.abs(values[1:] - values[0])
            disagreement = differences.mean() if rule == "amd" else linear @ differences
        else:
            raise ValueError(rule)
        if disagreement <= config.spread_threshold:
            result[h] = values[0]
        else:
            values[:], weights[:] = np.nan, 0
        paths, reliability = np.c_[paths, values], np.c_[reliability, weights]
    return result
