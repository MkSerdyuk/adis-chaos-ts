import numpy as np
import pandas as pd
import pytest

from adis.motifs import MotifModel
from adis.prediction import ForecastConfig
from adis.trading.equity_backtest import audit_result, run_daily
from adis.trading.equity_data import weekly_decisions
from adis.trading.equity_experiment import weighted_bootstrap
from adis.trading.forecasts import trajectory_predictions
from adis.trading.predictive_comparison import portfolio


def market():
    index = pd.bdate_range("2022-01-03", periods=60)
    frame = pd.DataFrame(
        dict(
            Open=100.0, High=101.0, Low=99.0, Close=100.0, Volume=1000.0, Dividend=0.0, FutureSplitFactor=1.0
        ),
        index=index,
    )
    frame.loc[index[10] :, ["Open", "High", "Low", "Close"]] -= 2
    frame.loc[index[10], "Dividend"] = 2.0
    return frame


def test_dividend_holder_ex_date_seller_and_ex_date_buyer():
    frame = market()
    for entry, exit_, entitled in [(4, 15, True), (4, 9, True), (9, 15, False)]:
        target = pd.Series([0.95, 0.0], index=frame.index[[entry, exit_]])
        result = run_daily(frame, target, str(frame.index[2].date()), "2022-03-28", costs_bps=0)
        assert (result["metrics"]["dividends"] > 0) == entitled
        assert result["metrics"]["final_equity"] == pytest.approx(100000)
        audit_result(frame, result, costs_bps=0)
    cash = run_daily(frame, pd.Series(0.0, index=frame.index[[4, 15]]), "2022-01-05", "2022-03-28")
    assert cash["metrics"]["dividends"] == 0


def test_delayed_dividend_is_equity_but_not_buying_power_and_survives_sale():
    frame = market()
    target = pd.Series([0.95, 0.0], index=frame.index[[4, 11]])
    result = run_daily(frame, target, "2022-01-05", "2022-03-28", costs_bps=0, dividend_delay_days=30)
    first = result["dividends"].iloc[0]
    before_payment = result["curve"].loc[
        (result["curve"].index >= frame.index[12]) & (result["curve"].index < first.paid_date)
    ]
    np.testing.assert_allclose(before_payment.equity, 100000)
    np.testing.assert_allclose(before_payment.available_cash + before_payment.receivables, 100000)
    assert (before_payment.available_cash < 100000).all()
    assert first.paid_date >= first.ex_date + pd.Timedelta(days=30)
    audit_result(frame, result, costs_bps=0)
    unsettled = run_daily(frame, target, "2022-01-05", "2022-01-31", costs_bps=0, dividend_delay_days=30)
    assert unsettled["metrics"]["final_receivables"] > 0
    assert unsettled["metrics"]["final_cash"] + unsettled["metrics"]["final_receivables"] == pytest.approx(
        100000
    )


def test_future_split_changes_units_not_dividend_pnl_or_proportional_costs():
    frame = market()
    target = pd.Series([0.95, 0.5, 0.0], index=frame.index[[4, 12, 16]])
    original = run_daily(frame, target, "2022-01-05", "2022-03-28")
    adjusted = frame.copy()
    adjusted[["Open", "High", "Low", "Close", "Dividend"]] /= 4
    adjusted["Volume"] *= 4
    adjusted["FutureSplitFactor"] *= 4
    changed = run_daily(adjusted, target, "2022-01-05", "2022-03-28")
    np.testing.assert_allclose(original["curve"].equity, changed["curve"].equity, rtol=0, atol=1e-5)
    assert original["metrics"]["fees"] == pytest.approx(changed["metrics"]["fees"], abs=1e-7)
    assert original["metrics"]["dividends"] == pytest.approx(changed["metrics"]["dividends"], abs=1e-7)
    audit_result(adjusted, changed)


def test_weekly_decision_uses_exchange_holiday_calendar():
    decisions = pd.DatetimeIndex(weekly_decisions())
    assert pd.Timestamp("2022-04-14") in decisions
    assert pd.Timestamp("2022-04-15") not in decisions
    assert pd.Timestamp("2026-11-27") in decisions


def test_price_forecast_and_target_are_invariant_to_future_split_units():
    series = 100 + 20 * np.sin(np.arange(120) / 7)
    predictions = []
    config = ForecastConfig(epsilon=0.3, cluster_eps=0.05, min_samples=1, selection="rgdbscan")
    for factor in [1, 4]:
        price = series / factor
        minimum, scale = price.min(), np.ptp(price)
        normalized = (price - minimum) / scale
        model = MotifModel(length=3, max_gap=2, r=5, mu=0.2).fit(normalized, workers=1)
        forecast = trajectory_predictions(model, normalized[-4:], 5, config, [42], paths=4)
        predictions.append((forecast * scale + minimum) / price[-1] - 1)
    assert np.isfinite(predictions[0]).any()
    np.testing.assert_allclose(predictions[0], predictions[1], rtol=0, atol=1e-13, equal_nan=True)
    np.testing.assert_array_equal(predictions[0] > 0.003, predictions[1] > 0.003)


def test_bootstrap_balances_asset_classes_and_preserves_calendar_dependence():
    high = np.array([-0.1, -0.04, 0.02, 0.03, 0.01] * 20)
    low = high * 0.2
    method = np.array([high] * 5 + [low, low])
    control = np.array([low] * 5 + [high, low])
    result = weighted_bootstrap(method, control, [1 / 15] * 5 + [1 / 3, 1 / 3], repetitions=100)
    np.testing.assert_allclose([result["difference"], result["lower"], result["upper"]], 0, atol=1e-15)


def test_portfolio_preserves_initial_allocations_without_daily_rebalancing():
    dates = pd.bdate_range("2022-01-03", periods=3)
    first = pd.DataFrame(dict(equity=[100000, 200000, 300000], exposure=1.0), index=dates)
    second = pd.DataFrame(dict(equity=100000, exposure=0.0), index=dates)
    combined = portfolio({"first": first, "second": second}, {"first": 0.5, "second": 0.5})
    np.testing.assert_allclose(combined.equity, [100000, 150000, 200000])
    np.testing.assert_allclose(combined.exposure, [0.5, 2 / 3, 0.75])
