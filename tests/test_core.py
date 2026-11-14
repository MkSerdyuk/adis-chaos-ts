from dataclasses import replace
from itertools import product

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score

from adis import ForecastConfig, MotifModel, forecast
from adis.clustering import _wishart, dbscan_1d, wishart
from adis.data import lorenz
from adis.diagnostics import entropy_complexity, predictability_horizon
from adis.evaluation import errors, mask_metrics
from adis.motifs import embed
from adis.prediction import (
    _step,
    aggregate,
    branching_forecast,
    reduced_history_forecast,
    trajectory_forecast,
)


def test_paper_embedding_example():
    np.testing.assert_array_equal(embed(np.arange(11), (2, 3, 4)), [[0, 2, 5, 9], [1, 3, 6, 10]])
    model = MotifModel()
    assert len(model.patterns) == 1000
    np.testing.assert_array_equal(model.patterns, list(product(range(1, 11), repeat=3)))


@pytest.mark.parametrize("seed", range(10))
def test_dbscan_matches_sklearn(seed):
    rng = np.random.default_rng(seed)
    x = np.round(rng.normal(size=100), 2)
    order = np.argsort(x)
    expected = np.empty(100, int)
    expected[order] = DBSCAN(eps=0.1, min_samples=4).fit_predict(x[order, None])
    actual = dbscan_1d(x, 0.1, 4)
    np.testing.assert_array_equal(actual < 0, expected < 0)
    assert adjusted_rand_score(actual, expected) == 1


def test_wishart_significant_clusters_complete_at_saddle():
    order = np.arange(7)
    balls = [[], [0], [], [2], [1, 3], [0], [2]]
    boundaries = np.r_[0, np.cumsum(list(map(len, balls)))]
    neighbors = np.concatenate(balls).astype(int)
    labels = _wishart(order, boundaries, neighbors, np.array([10, 8, 10, 8, 5, 4, 4.0]), 1)
    np.testing.assert_array_equal(labels, [1, 1, 2, 2, 0, 0, 0])


def test_wishart_insignificant_cluster_deleted_on_noise_contact():
    balls = [[], [0], [], [2], [1, 3], [], [4, 5]]
    labels = _wishart(
        np.arange(7),
        np.r_[0, np.cumsum(list(map(len, balls)))],
        np.concatenate(balls).astype(int),
        np.array([10, 8, 10, 8, 5, 4, 3.0]),
        1,
    )
    np.testing.assert_array_equal(labels, [1, 1, 2, 2, 0, 0, 0])


def test_wishart_duplicates_are_deterministic():
    x = np.r_[np.zeros(20), np.ones(20)]
    np.testing.assert_array_equal(wishart(x), wishart(x))


def test_candidates_match_exhaustive_search_with_missing_values():
    rng = np.random.default_rng(42)
    series = rng.random(300)
    model = MotifModel(max_gap=2, clustered=False).fit(series)
    histories = rng.random((4, 20))
    histories[1, -1] = np.nan
    reliability = rng.random(histories.shape)
    for epsilon in [0.1, 1.5]:
        actual = model.candidates(histories, reliability, epsilon)
        for row, (values, depth, distance) in enumerate(actual):
            expected_values, expected_depth, expected_distance = [], [], []
            for index, pattern in enumerate(model.pattern_ids):
                context = histories[row, -model.lags[pattern]]
                d = np.linalg.norm(context - model.motifs[index, :-1])
                if d < epsilon:
                    expected_values.append(model.motifs[index, -1])
                    expected_depth.append(0.99 * reliability[row, -model.lags[pattern]].mean())
                    expected_distance.append(1 - d / epsilon)
            np.testing.assert_allclose(values, expected_values)
            np.testing.assert_allclose(depth, expected_depth)
            np.testing.assert_allclose(distance, expected_distance)


def test_weighted_aggregation_equations():
    candidates = (np.array([1.0, 3]), np.array([0.5, 1]), np.array([0.2, 0.8]))
    rng = np.random.default_rng(0)
    for mode, expected in [("avg", 2), ("wavgl", 7 / 3), ("wavgd", 2.6), ("wavgc", 25 / 9)]:
        value, reliability = aggregate(candidates, ForecastConfig(aggregation=mode), rng)
        assert value == pytest.approx(expected)
        assert reliability == 0.75


def test_mfp_variance_is_not_standard_deviation():
    config = ForecastConfig(aggregation="mfp", perturbation_variance=0.04)
    rng = np.random.default_rng(7)
    candidates = (np.ones(4), np.ones(4), np.ones(4))
    samples = np.array([aggregate(candidates, config, rng)[0] for _ in range(3000)])
    assert samples.var() == pytest.approx(0.04, rel=0.08)


class BridgeModel:
    context_size = 2

    def candidates(self, histories, reliability, epsilon, decay=0.99, force=False):
        result = []
        for history in histories:
            values = np.array([0.7]) if len(history) % 2 else np.array([0.9])
            result.append((values, np.array([0.99]), np.ones(1)))
        return result


def test_oracle_rejects_without_replacing_with_truth():
    config = ForecastConfig(selection="ap", oracle_tolerance=0.05)
    prediction = forecast(BridgeModel(), [0.1, 0.2], 3, config, truth=np.array([[0.1, 0.7, 0.9]]))
    assert np.isnan(prediction[0, 0])
    np.testing.assert_allclose(prediction[0, 1:], [0.7, 0.9])
    with pytest.raises(ValueError, match="truth"):
        forecast(BridgeModel(), [0.1, 0.2], 3, config)


def test_gap_patterns_bridge_a_missing_intermediate():
    model = MotifModel(length=2, max_gap=2, clustered=False)
    model.motifs = np.array([[0.2, 0.4], [0.2, 0.6]])
    model.sizes, model.pattern_ids = np.array([1, 1]), np.array([0, 1])
    model._index()
    candidates = model.candidates(np.array([[0.2, np.nan]]), np.array([[1.0, 0]]), 0.01)
    np.testing.assert_allclose(candidates[0][0], [0.6])


def test_metrics_missing_and_oracle_bounds():
    result = errors(np.array([0.2, 0.5]), np.array([np.nan, np.nan]))
    assert result["coverage"] == result["i2"] == 0
    assert np.isnan(result["rmse"])
    result = errors(np.array([0.2, 0.5]), np.array([0.3, np.nan]))
    assert result["i2"] == pytest.approx(result["coverage"] * result["mae"])
    result = mask_metrics(np.array([1.0, np.nan, 2.0]), np.array([1.0, 2.0, 2.0]))
    assert result["predictable_precision"] == 1
    assert result["predictable_recall"] == 2 / 3
    assert result["paper_precision"] == 2 / 3


def test_rk4_matches_high_accuracy_short_trajectory():
    def rhs(t, state):
        x, y, z = state
        return [10 * (y - x), x * (28 - z) - y, x * y - 8 / 3 * z]

    times = np.arange(20) * 0.001
    exact = solve_ivp(rhs, (0, times[-1]), [1, 1, 1], t_eval=times, rtol=1e-12, atol=1e-14).y[0]
    np.testing.assert_allclose(lorenz(20, discard=0, dt=0.001), exact, atol=3e-9)
    assert predictability_horizon()["steps"] == pytest.approx(75.0842965106754)


def test_entropy_complexity_limits():
    assert entropy_complexity(np.ones(100)) == pytest.approx((0, 0))
    h, c = entropy_complexity(np.random.default_rng(0).normal(size=20000), dimension=3)
    assert h > 0.999 and c < 0.001


def test_seeded_trajectories_and_branching_contract():
    model = MotifModel(length=2, max_gap=2, clustered=False).fit(np.tile([0.2, 0.4, 0.6], 50))
    config = ForecastConfig(epsilon=0.5, cluster_eps=0.3)
    first = trajectory_forecast(model, [[0.2, 0.4]], 4, config, trajectories=3)
    second = trajectory_forecast(model, [[0.2, 0.4]], 4, config, trajectories=3)
    np.testing.assert_array_equal(first, second)
    with pytest.raises(ValueError, match="Branching supports"):
        branching_forecast(model, [0.2, 0.4], 3, replace(config, selection="ap"))
    with pytest.raises(ValueError, match="wamdt"):
        reduced_history_forecast(model, np.tile([0.2, 0.4], 5), 2, config, rule="wamdt")


def test_branching_has_explicit_limits_and_weight_pruning():
    model = MotifModel(length=2, max_gap=1, clustered=False).fit(np.tile([0.0, 0.0, 1.0], 50))
    config = ForecastConfig(cluster_eps=0.01)
    with pytest.raises(RuntimeError, match="max_paths"):
        branching_forecast(model, [0.0], 2, config, max_paths=1)
    assert np.isnan(branching_forecast(model, [0.0], 2, config, prune_weight=0.995)).all()


@pytest.mark.parametrize(
    "aggregation",
    ["avg", "wavgl", "wavgd", "wavgc", "clc", "mf", "mfp", "threshold_clc", "fuzzy", "roulette", "wishart"],
)
def test_aggregation_modes_obey_empty_set_and_seed_contract(aggregation):
    candidates = (np.r_[np.full(30, 0.2), np.full(20, 0.8)], np.full(50, 0.99), np.ones(50))
    config = ForecastConfig(aggregation=aggregation)
    first = aggregate(candidates, config, np.random.default_rng(42))
    second = aggregate(candidates, config, np.random.default_rng(42))
    np.testing.assert_array_equal(first, second)
    empty = aggregate((np.empty(0), np.empty(0), np.empty(0)), config, np.random.default_rng(42))
    assert np.isnan(empty[0]) and empty[1] == 0


@pytest.mark.parametrize("mode", ["avg", "wavgl", "wavgd", "wavgc", "clc", "mfp"])
def test_batched_aggregation_matches_individual_reference(mode):
    rng = np.random.default_rng(11)
    candidates = [(rng.random(n), rng.random(n), rng.random(n)) for n in [0, 1, 3, 40, 100]]

    class CandidateModel:
        def candidates(self, *args, **kwargs):
            return candidates

    config = ForecastConfig(aggregation=mode, cluster_eps=0.2)
    reference_rng, batch_rng = np.random.default_rng(42), np.random.default_rng(42)
    expected = np.array([aggregate(c, config, reference_rng) for c in candidates])
    values, reliability, _, _ = _step(CandidateModel(), None, None, config, batch_rng)
    np.testing.assert_allclose(np.c_[values, reliability], expected, rtol=1e-14, atol=1e-15)
    assert reference_rng.normal() == batch_rng.normal()
