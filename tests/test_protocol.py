from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from adis import MotifModel
from adis.baselines import _recursive, seasonal_naive
from adis.evaluation import compare_masks
from adis.prediction import ForecastConfig, _spread, forecast, reduced_history_forecast, trajectory_forecast


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


def test_wamdt_requires_a_main_trajectory():
    prediction = trajectory_forecast(
        MissingMainModel(), [[0.2, 0.3]], 2, ForecastConfig(selection="wamdt"), trajectories=2
    )
    assert np.isnan(prediction).all()


@pytest.mark.parametrize("aggregation", ["wavgl", "clc", "mfp"])
def test_trajectory_forecast_rejects_ignored_endpoint_aggregation(aggregation):
    with pytest.raises(ValueError, match="avg endpoints; intermediate steps use mfp"):
        trajectory_forecast(
            GrowingSpreadModel(), [0.2, 0.4], 3, ForecastConfig(aggregation=aggregation)
        )


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
