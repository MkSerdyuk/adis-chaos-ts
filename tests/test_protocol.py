from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from adis import MotifModel
from adis.baselines import _recursive, seasonal_naive
from adis.data import test_windows as make_test_windows
from adis.evaluation import compare_masks, errors, horizon_metrics
from adis.prediction import (
    ForecastConfig,
    _spread,
    calibrate_trajectory_spread,
    forecast,
    reduced_history_forecast,
    trajectory_forecast,
)


class LastValueRegressor:
    def predict(self, x):
        return x[:, 0] + 1


def test_recursive_baseline_never_uses_future_actual_load():
    series = np.arange(400, dtype=float)
    mutated = series.copy()
    mutated[201:] = -1e20
    dates = pd.date_range("2015-01-01", periods=len(series), freq="h", tz="UTC")
    first = _recursive(LastValueRegressor(), series, np.array([200]), 100, dates)
    second = _recursive(LastValueRegressor(), mutated, np.array([200]), 100, dates)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[0], np.arange(201, 301))


def test_seasonal_naive_wraps_without_using_future():
    values = np.arange(300)
    prediction = seasonal_naive(values, np.array([200]), 100, 24)
    assert prediction.max() <= 200
    np.testing.assert_array_equal(prediction[:, :24], prediction[:, 24:48])


def test_mask_comparison_keeps_selected_denominator():
    truth = np.ones((2, 3))
    a = np.array([[1.0, np.nan, 2], [np.nan, 3, np.nan]])
    b = np.array([[np.nan, 1.0, 2], [2.0, np.nan, 1.0]])
    result = compare_masks(truth, {"a": a, "b": b, "baseline": truth}, {"a": a})
    assert set(result.method) == {"a", "baseline"}
    assert (result["count"] == result.selected_count).all()


class GrowingSpreadModel:
    context_size = 2

    def __init__(self):
        self.seen_missing = False

    def candidates(self, histories, reliability, epsilon, decay=0.99, force=False):
        self.seen_missing |= bool(np.isnan(histories).any())
        spread = 0.01 * (histories.shape[1] - 1)
        return [(np.array([0.5 - spread, 0.5, 0.5 + spread]), np.ones(3), np.ones(3)) for _ in histories]


def test_rapid_growth_looks_ahead_and_rejects_first_point():
    model = GrowingSpreadModel()
    predicted = forecast(model, [0.2, 0.4], 3, ForecastConfig(selection="rg"))
    assert np.isnan(predicted).all()
    assert model.seen_missing


def test_count_growth_differs_from_spread_growth():
    candidates = [(np.array([0.1, 0.1, 0.1, 0.7, 0.7, 0.7]), None, None)]
    config = ForecastConfig(selection="dbscan")
    assert _spread(candidates, config)[0] == 2
    assert _spread(candidates, replace(config, selection="rgdbscan"))[0] == pytest.approx(0.6)


def test_reduced_history_rejection_is_fed_back():
    class ShiftedCenterModel(GrowingSpreadModel):
        def candidates(self, histories, reliability, epsilon, decay=0.99, force=False):
            self.seen_missing |= bool(np.isnan(histories).any())
            return [(np.array([history[0]]), np.ones(1), np.ones(1)) for history in histories]

    model = ShiftedCenterModel()
    config = ForecastConfig(spread_threshold=0)
    reduced_history_forecast(model, np.linspace(0.1, 0.6, 10), 3, config, rule="cwat")
    assert model.seen_missing


class MissingMainModel:
    context_size = 2

    def candidates(self, histories, reliability, epsilon, decay=0.99, force=False):
        return [
            (np.empty(0), np.empty(0), np.empty(0))
            if i % 2 == 0
            else (np.full(4, 0.5), np.ones(4), np.ones(4))
            for i in range(len(histories))
        ]


def test_spread_calibration_is_per_horizon_and_frozen_for_evaluation():
    class DivergingPaths:
        context_size = 2

        def candidates(self, histories, reliability, epsilon, decay=0.99, force=False):
            step = histories.shape[1] - 1
            return [
                (np.full(4, 0.3 + (i % 2) * 0.01 * step), np.ones(4), np.ones(4))
                for i in range(len(histories))
            ]

    model = DivergingPaths()
    config = ForecastConfig(selection="spread", perturbation_variance=0)
    contexts = np.array([[0.1, 0.2], [0.2, 0.3]])
    thresholds, counts = calibrate_trajectory_spread(model, contexts, 3, config, factor=2, trajectories=2)
    np.testing.assert_allclose(thresholds, [0.02, 0.04, 0.06])
    np.testing.assert_array_equal(counts, [2, 2, 2])
    accepted = trajectory_forecast(model, contexts, 3, config, trajectories=2, spread_thresholds=thresholds)
    rejected = trajectory_forecast(
        model, contexts, 3, config, trajectories=2, spread_thresholds=thresholds / 4
    )
    assert np.isfinite(accepted).all() and np.isnan(rejected).all()


def test_spread_calibration_records_available_trajectories_and_rejects_empty_horizons():
    config = ForecastConfig(perturbation_variance=0)
    thresholds, counts = calibrate_trajectory_spread(MissingMainModel(), [[0.1, 0.2]], 2, config, 1, 2)
    np.testing.assert_array_equal(thresholds, [0, 0])
    np.testing.assert_array_equal(counts, [1, 1])
    with pytest.raises(ValueError, match="finite validation spread"):
        calibrate_trajectory_spread(MissingMainModel(), [[0.1, 0.2]], 2, config, 1, 1)


def test_wamdt_requires_a_main_trajectory():
    prediction = trajectory_forecast(
        MissingMainModel(), [[0.2, 0.3]], 2, ForecastConfig(selection="wamdt"), trajectories=2
    )
    assert np.isnan(prediction).all()


@pytest.mark.parametrize("aggregation", ["wavgl", "clc", "mfp"])
def test_trajectory_forecast_rejects_ignored_endpoint_aggregation(aggregation):
    with pytest.raises(ValueError, match="avg endpoints; intermediate steps use mfp"):
        trajectory_forecast(GrowingSpreadModel(), [0.2, 0.4], 3, ForecastConfig(aggregation=aggregation))


@pytest.mark.parametrize("aggregation", ["avg", "wavgl"])
def test_forced_prediction_uses_distant_motifs_without_abstaining(aggregation):
    model = MotifModel(length=2, max_gap=1, clustered=False).fit(np.tile([0.1, 0.2, 0.3], 50))
    config = ForecastConfig(selection="fp", aggregation=aggregation)
    prediction = forecast(model, [5.0], 3, config)
    np.testing.assert_allclose(prediction, [[0.1, 0.2, 0.3]])


@pytest.mark.parametrize("aggregation", ["wavgd", "wavgc", "clc", "mfp"])
def test_forced_prediction_rejects_aggregators_that_can_abstain(aggregation):
    config = ForecastConfig(selection="fp", aggregation=aggregation)
    with pytest.raises(ValueError, match="Forced prediction requires"):
        forecast(GrowingSpreadModel(), [0.2, 0.4], 3, config)


def test_forced_prediction_rejects_predictability_classifier():
    with pytest.raises(ValueError, match="without a classifier"):
        forecast(GrowingSpreadModel(), [0.2, 0.4], 3, ForecastConfig(selection="fp"), classifier=object())


def test_paper_evaluates_every_test_target_at_every_horizon():
    series = np.arange(300.0)
    origins, contexts, truth, oracle = make_test_windows(series, 200, 30, 10)
    assert origins[0] == 190
    np.testing.assert_array_equal(contexts[:, -1], origins)
    np.testing.assert_array_equal(np.isfinite(truth).sum(axis=0), np.full(10, 100))
    for h in range(10):
        np.testing.assert_array_equal(truth[np.isfinite(truth[:, h]), h], series[200:])
    assert np.isnan(truth[0, 0]) and oracle[0, 0] == 191
    assert np.isnan(oracle[-1, -1])
    calibration_end = min(200, origins[0] + 1)
    validation_origins = np.arange(150 - 1, calibration_end - 10, 8)
    assert validation_origins[-1] + 10 <= origins[0]


def test_common_origins_protocol_keeps_original_support():
    origins, contexts, truth, oracle = make_test_windows(
        np.arange(300.0), 200, 30, 10, evaluation="common_origins"
    )
    assert truth.shape == (91, 10)
    np.testing.assert_array_equal(truth, oracle)
    np.testing.assert_array_equal(contexts[:, -1], origins)


def test_metrics_exclude_targets_outside_the_evaluation_set():
    truth = np.array([[np.nan, 2.0], [1.0, 3.0], [2.0, np.nan]])
    prediction = np.array([[100.0, 2.0], [1.0, np.nan], [3.0, 100.0]])
    scores = errors(truth, prediction)
    assert scores["total"] == 4 and scores["count"] == 3
    assert scores["rmse"] == pytest.approx(np.sqrt(1 / 3))
    assert scores["coverage"] == 0.75
    metrics = horizon_metrics(truth, prediction, "test", oracle=truth)
    np.testing.assert_array_equal(metrics.total, [2, 2])
    np.testing.assert_allclose(metrics.predictable_recall, [1, 0.5])
    masks = compare_masks(truth, {"partial": prediction, "full": truth}, {"partial": prediction})
    np.testing.assert_array_equal(masks.selected_count, [3, 3])
    np.testing.assert_array_equal(masks["count"], [3, 3])
    with pytest.raises(ValueError, match="No evaluation targets"):
        errors(np.full(3, np.nan), np.zeros(3))


def test_cwat_uses_the_printed_geometric_sum():
    class ConstantModel(GrowingSpreadModel):
        def candidates(self, histories, reliability, epsilon, decay=0.99, force=False):
            return [(np.array([0.8]), np.ones(1), np.ones(1)) for _ in histories]

    model = ConstantModel()
    config = ForecastConfig(spread_threshold=0.1)
    prediction = reduced_history_forecast(model, np.full(10, 0.8), 1, config, previous=1)
    assert np.isnan(prediction).all()


def test_selector_receives_only_the_ten_paper_features():
    class Classifier:
        def predict(self, features):
            assert features.shape == (1, 10)
            return np.ones(1)

    result = forecast(GrowingSpreadModel(), [0.2, 0.4], 1, classifier=Classifier())
    assert np.isfinite(result).all()
