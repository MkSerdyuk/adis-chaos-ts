import hashlib
import json
from dataclasses import replace
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from adis.motifs import MotifModel
from adis.prediction import ForecastConfig, trajectory_forecast
from adis.trading.backtest import expected_shortfall, run_backtest
from adis.trading.baselines import technical_targets
from adis.trading.data import download_month
from adis.trading.evaluation import collect_model_metadata, paired_bootstrap, yearly_metrics
from adis.trading.forecasts import fit_quarter, predict_quarter, trajectory_predictions


def candles(count=300, start="2020-12-25"):
    index = pd.date_range(start, periods=count, freq="h", tz="UTC")
    close = 100 + np.arange(count) * 0.1 + np.sin(np.arange(count))
    return pd.DataFrame(
        dict(Open=close, High=close + 2, Low=close - 2, Close=close, Volume=1.0, Observed=True), index=index
    )


def test_model_manifest_survives_prediction_cache_without_local_models(tmp_path):
    existing = {"model_id": "cached", "training_sha256": "unchanged", "fit_seconds": 2.0}
    (tmp_path / "model_manifest.json").write_text(json.dumps([existing]))
    assert collect_model_metadata(tmp_path) == [existing]
    folder = tmp_path / "models" / "new"
    folder.mkdir(parents=True)
    new = {"model_id": "new", "training_sha256": "different", "fit_seconds": 3.0}
    (folder / "metadata.json").write_text(json.dumps(new))
    assert collect_model_metadata(tmp_path) == [existing, new]
    (folder / "metadata.json").write_text(json.dumps(dict(existing, training_sha256="corrupted")))
    with pytest.raises(AssertionError):
        collect_model_metadata(tmp_path)


@pytest.mark.parametrize("month,factor", [("2020-01", 1000), ("2025-01", 1000000)])
def test_archive_timestamp_units_and_zero_volume_placeholders(tmp_path, month, factor):
    epoch = int(pd.Timestamp(month + "-01", tz="UTC").timestamp()) * factor
    rows = f"{epoch},100,101,99,100,1,{epoch + 3600 * factor - 1},0,1,0,0,0\n"
    rows += f"{epoch + 3600 * factor},100,100,100,100,0,{epoch},0,0,0,0,0\n"
    name = f"BTCUSDT-1h-{month}.zip"
    with ZipFile(tmp_path / name, "w") as archive:
        archive.writestr(name.replace(".zip", ".csv"), rows)
    digest = hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
    (tmp_path / (name + ".CHECKSUM")).write_text(digest + " " + name)
    frame, metadata = download_month("BTCUSDT", month, tmp_path)
    assert len(frame) == 1
    assert frame.index[0] == pd.Timestamp(month + "-01", tz="UTC")
    assert len(metadata["zero_volume_open_times"]) == 1


@pytest.fixture(scope="module")
def motif_model():
    series = 0.5 + 0.2 * np.sin(np.arange(100) / 7)
    return MotifModel(length=3, max_gap=2, clustered=False).fit(series, workers=1)


def test_per_origin_noise_matches_original_and_does_not_depend_on_future_batch(motif_model):
    config = ForecastConfig(epsilon=0.3, cluster_eps=0.05, min_samples=1, selection="rgdbscan", seed=42)
    contexts = np.array([[0.45, 0.47, 0.49, 0.51], [0.4, 0.5, 0.6, 0.7]])
    alone = trajectory_predictions(motif_model, contexts[:1], 8, config, [42], paths=4)
    original = trajectory_forecast(motif_model, contexts[:1], 8, config, trajectories=4)
    combined = trajectory_predictions(motif_model, contexts, 8, config, [42, 17], paths=4)
    assert np.isfinite(alone).any()
    np.testing.assert_allclose(alone, original, rtol=0, atol=1e-13, equal_nan=True)
    np.testing.assert_array_equal(alone[0], combined[0])
    contexts[1] *= 3
    changed = trajectory_predictions(motif_model, contexts, 8, config, [42, 17], paths=4)
    np.testing.assert_array_equal(alone[0], changed[0])
    without_rejection = trajectory_predictions(
        motif_model, contexts[:1], 8, replace(config, selection="cm"), [42], paths=4
    )
    assert np.isfinite(without_rejection).all()


def test_fit_window_never_uses_future_prices(tmp_path):
    frame = candles()
    quarter = pd.Timestamp("2021-01-01", tz="UTC")
    settings = dict(length=3, max_gap=2, r=5, mu=0.2, training_hours=100)
    original, metadata = fit_quarter(frame, quarter, settings, tmp_path, workers=1)
    changed = frame.copy()
    changed.loc[changed.index >= quarter, "Close"] *= 100
    second, second_metadata = fit_quarter(changed, quarter, settings, tmp_path, workers=1)
    np.testing.assert_array_equal(original.motifs, second.motifs)
    assert metadata["fit_end"] == str(quarter)
    assert metadata["training_sha256"] == second_metadata["training_sha256"]
    training = frame.loc[frame.index < quarter].Close.iloc[-100:]
    assert metadata["minimum"] == training.min()
    assert metadata["scale"] == np.ptp(training)


def test_broker_next_open_delay_costs_and_fractional_accounting():
    frame = candles(12, "2022-01-01")
    targets = pd.Series([0.95, 0], index=frame.index[[2, 6]])
    result = run_backtest(frame, targets, "2022-01-01", "2022-01-01 12:00", costs_bps=15, delay_hours=1)
    trade = result["trades"].iloc[0]
    assert len(result["trades"]) == 1
    assert trade.EntryTime == frame.index[4]
    assert trade.ExitTime == frame.index[8]
    assert trade.EntryPrice == frame.Open.iloc[4]
    assert trade.ExitPrice == frame.Open.iloc[8]
    quantity = np.floor((100000 * 0.95 / (trade.EntryPrice * 1.0015)) / 1e-8) * 1e-8
    assert trade.Size == pytest.approx(quantity, abs=1e-8)
    fees = quantity * (trade.EntryPrice + trade.ExitPrice) * 0.0015
    expected = 100000 + quantity * (trade.ExitPrice - trade.EntryPrice) - fees
    assert trade.Commission == pytest.approx(fees)
    assert result["metrics"]["final_equity"] == pytest.approx(expected)
    free = run_backtest(frame, targets, "2022-01-01", "2022-01-01 12:00", costs_bps=0, delay_hours=1)
    assert free["metrics"]["final_equity"] > expected


def test_no_execution_on_a_missing_candle():
    frame = candles(12, "2022-01-01")
    frame.loc[frame.index[4], "Observed"] = False
    result = run_backtest(
        frame,
        pd.Series([0.95, 0], index=frame.index[[2, 6]]),
        "2022-01-01",
        "2022-01-01 12:00",
        delay_hours=1,
    )
    assert result["trades"].iloc[0].EntryTime == frame.index[5]


def test_indicators_have_prefix_invariance_and_es_has_known_value():
    frame = candles(6000, "2020-01-01")
    prefix = frame.iloc[:5500]
    all_targets = technical_targets(frame)
    prefix_targets = technical_targets(prefix)
    comparable = prefix_targets.index[prefix_targets.index <= prefix.index[-1]]
    pd.testing.assert_frame_equal(all_targets.loc[comparable], prefix_targets.loc[comparable])
    assert expected_shortfall(np.array([-0.20, -0.10] + [0.01] * 38)) == pytest.approx(0.15)


def test_constant_exposure_control_rebalances_only_at_daily_decisions():
    frame = candles(24 * 20, "2022-01-01")
    decisions = frame.index[frame.index.hour == 23]
    targets = pd.Series(0.5, index=decisions)
    fixed = run_backtest(frame, targets, "2022-01-01", "2022-01-21", costs_bps=0)
    rebalanced = run_backtest(frame, targets, "2022-01-01", "2022-01-21", costs_bps=0, rebalance_daily=True)
    active = slice(pd.Timestamp("2022-01-02 02:00", tz="UTC"), pd.Timestamp("2022-01-20 20:00", tz="UTC"))
    assert abs(rebalanced["exposure"].loc[active].mean() - 0.5) < 0.01
    assert fixed["exposure"].loc[active].mean() > 0.53
    ordinary = rebalanced["orders"].iloc[:-1]
    assert (pd.to_datetime(ordinary.decision_bar).dt.hour == 0).all()


def test_paired_bootstrap_and_yearly_opening_equity():
    returns = np.array([-0.1, 0.01, 0.02, -0.02, 0.03] * 12)
    interval = paired_bootstrap(returns, returns, repetitions=30)
    assert interval["difference"] == interval["lower"] == interval["upper"] == 0
    frame = candles(24 * 3, "2022-12-30")
    result = run_backtest(
        frame, pd.Series(0.0, index=frame.index[frame.index.hour == 23]), "2022-12-30", "2023-01-02"
    )
    result["equity"].loc[:] = 200000
    result["equity"].loc["2023-01-01":] = 220000
    years = yearly_metrics(result)
    assert years[1]["total_return"] == pytest.approx(0.1)
    assert years[1]["entries"] == 0


def test_prediction_ledger_cache_and_future_prefix(tmp_path):
    frame = candles()
    start, end = pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2021-01-06", tz="UTC")
    parameters = dict(
        length=3,
        max_gap=2,
        r=5,
        mu=0.2,
        training_hours=100,
        epsilon=0.5,
        cluster_eps=0.1,
        min_samples=1,
        perturbation_variance=1e-5,
        paths=4,
    )
    original = predict_quarter(frame, start, end, parameters, tmp_path, workers=1, mode="cm")
    changed = frame.copy()
    changed.loc["2021-01-04":, "Close"] *= 10
    repeated = predict_quarter(changed, start, end, parameters, tmp_path, workers=1, mode="cm")
    pd.testing.assert_frame_equal(original.loc[:"2021-01-03"], repeated.loc[:"2021-01-03"])
    assert not original.close.equals(repeated.close)
    assert len(list((tmp_path / "predictions").glob("*.csv"))) == 2
