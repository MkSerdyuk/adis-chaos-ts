import hashlib
import json

import numpy as np
import pandas as pd

from .equity_backtest import audit_result
from .equity_data import calendar, load_asset, weekly_decisions
from .predictive_comparison import CANDIDATES, EQUITY_ROOT, PERIODS, ROOT, freeze


def audit():
    state = freeze()
    completed = json.loads((ROOT / "completed.json").read_text())
    assert completed["frozen"] == state
    assert completed["backtests"] == 210 and completed["portfolios"] == 30
    symbols = state["noncrypto"]["symbols"]
    frames = {symbol: load_asset(symbol) for symbol in symbols}
    decisions = pd.DatetimeIndex(weekly_decisions())
    decisions = decisions[(decisions >= "2021-01-01") & (decisions < "2026-11-01")]
    forecast_count = 0
    for symbol, frame in frames.items():
        directory = ROOT / symbol
        forecasts = pd.read_csv(directory / "forecasts.csv")
        assert not forecasts.duplicated(["decision_session", "model"]).any()
        assert set(forecasts.model) == set(CANDIDATES[:4])
        for _, rows in forecasts.groupby("model"):
            assert pd.DatetimeIndex(rows.decision_session).equals(decisions)
        for quarter, rows in forecasts.groupby("quarter"):
            quarter = pd.Timestamp(quarter)
            boundary = frame.index.searchsorted(quarter)
            assert (pd.to_datetime(rows.fit_start) == frame.index[boundary - 2520]).all()
            assert (pd.to_datetime(rows.last_label_session) == frame.index[boundary - 1]).all()
            assert frame.index[boundary - 1] < quarter
            dates = pd.to_datetime(rows.decision_session)
            assert ((dates >= quarter) & (dates < quarter + pd.DateOffset(months=3))).all()
        assert np.isfinite(forecasts.predicted_return).all()
        np.testing.assert_array_equal(forecasts.target, np.where(forecasts.predicted_return > 0.003, 0.95, 0))
        targets = pd.read_csv(directory / "targets.csv", index_col=0, parse_dates=True)
        upstream = pd.read_csv(EQUITY_ROOT / symbol / "targets.csv", index_col=0, parse_dates=True)
        assert targets.index.equals(decisions) and upstream.index.equals(decisions)
        np.testing.assert_array_equal(targets.random_forest, upstream.random_forest5)
        for name in targets:
            if name in CANDIDATES[:4]:
                expected = forecasts.loc[forecasts.model == name, "target"]
            else:
                expected = upstream[name]
            np.testing.assert_allclose(targets[name], expected, rtol=0, atol=0, equal_nan=True)
        forecast_count += len(forecasts)

    checks = json.loads((ROOT / "audit.json").read_text())
    assert len(checks) == 210
    curves = {}
    main_parity = 0
    for key in checks:
        symbol, period, name = key.split("/")
        folder = ROOT / key
        curve = pd.read_csv(folder / "equity.csv.gz", index_col=0, parse_dates=True)
        returns = pd.read_csv(folder / "returns.csv", index_col=0, parse_dates=True).iloc[:, 0]
        metrics = json.loads((folder / "metrics.json").read_text())
        start, end = PERIODS[period]
        expected = calendar().sessions_in_range(start, pd.Timestamp(end) - pd.Timedelta(days=1))
        assert curve.index.equals(expected) and returns.index.equals(expected)
        audit_result(
            frames[symbol],
            dict(
                curve=curve,
                returns=returns,
                metrics=metrics,
                trades=pd.read_csv(folder / "trades.csv.gz"),
                dividends=pd.read_csv(folder / "dividends.csv"),
                orders=pd.read_csv(folder / "orders.csv"),
            ),
        )
        np.testing.assert_allclose(
            returns, curve.equity / curve.equity.shift(1).fillna(100000) - 1, atol=1e-12
        )
        curves[symbol, period, name] = curve
        shared_name = "random_forest5" if name == "random_forest" else name
        if period != "validation" and name not in ["gradient_boosting", "ridge", "svr"]:
            original = pd.read_csv(
                EQUITY_ROOT / symbol / period / shared_name / "equity.csv.gz", index_col=0, parse_dates=True
            )
            pd.testing.assert_frame_equal(curve, original, check_exact=True)
            main_parity += 1

    metrics = pd.read_csv(ROOT / "portfolio_metrics.csv").set_index(["period", "strategy"])
    for (period, name), row in metrics.iterrows():
        folder = ROOT / "portfolios" / period / name
        curve = pd.read_csv(folder / "equity.csv", index_col=0, parse_dates=True)
        capital = pd.Series(0.0, index=curve.index)
        invested = capital.copy()
        for symbol, category in state["noncrypto"]["instruments"].items():
            weight = 1 / 15 if category == "equity" else 1 / 3
            component = curves[symbol, period, name]
            capital += weight * component.equity
            invested += weight * component.equity * component.exposure
        np.testing.assert_allclose(curve.equity, capital, rtol=1e-12, atol=1e-8)
        np.testing.assert_allclose(curve.exposure, invested / capital, rtol=1e-12, atol=1e-12)
        returns = curve.equity / curve.equity.shift(1).fillna(100000) - 1
        saved = pd.read_csv(folder / "returns.csv", index_col=0, parse_dates=True).iloc[:, 0]
        np.testing.assert_allclose(saved, returns, atol=1e-12)
        independent = dict(
            total_return=capital.iloc[-1] / 100000 - 1,
            cagr=(capital.iloc[-1] / 100000) ** (252 / len(capital)) - 1,
            max_drawdown=(1 - capital / capital.cummax().clip(lower=100000)).max(),
            volatility=returns.std(ddof=1) * np.sqrt(252),
            es95=max(0, -np.sort(returns)[: int(np.ceil(0.05 * len(returns)))].mean()),
            worst_day=returns.min(),
            mean_exposure=(invested / capital).mean(),
            final_equity=capital.iloc[-1],
        )
        for metric, value in independent.items():
            np.testing.assert_allclose(row[metric], value, rtol=1e-10, atol=1e-12)

    eligibility = pd.read_csv(ROOT / "eligibility.csv").set_index("strategy")
    validation = metrics.loc["validation"].loc[CANDIDATES]
    np.testing.assert_allclose(eligibility.validation_return, validation.total_return, atol=1e-12)
    np.testing.assert_allclose(eligibility.validation_exposure, validation.mean_exposure, atol=1e-12)
    np.testing.assert_array_equal(
        eligibility.eligible, (validation.total_return > 0) & (validation.mean_exposure >= 0.1)
    )
    eligibility_hash = hashlib.sha256((ROOT / "eligibility.csv").read_bytes()).hexdigest()
    selection = json.loads((ROOT / "eligibility_frozen.json").read_text())
    assert selection["eligibility_sha256"] == completed["eligibility_sha256"] == eligibility_hash
    assert selection["eligible"] == eligibility.index[eligibility.eligible].tolist()
    comparisons = pd.read_csv(ROOT / "comparisons.csv")
    assert len(comparisons) == 36
    assert not comparisons.duplicated(["period", "comparator", "block"]).any()
    for (period, block), rows in comparisons.groupby(["period", "block"]):
        assert period in ["historical", "holdout"] and block in [5, 10, 20]
        assert set(rows.comparator) == set(CANDIDATES)
        for row in rows.itertuples():
            expected = (
                metrics.loc[(period, "partial_prediction"), "es95"]
                - metrics.loc[(period, row.comparator), "es95"]
            )
            np.testing.assert_allclose(row.difference, expected, atol=1e-12)
            np.testing.assert_allclose(row.interval_confidence, 1 - 0.05 / 6, atol=1e-12)
            assert row.lower <= row.upper
    result = dict(
        backtests_verified=len(checks),
        portfolios_verified=len(metrics),
        forecast_rows_verified=forecast_count,
        shared_instrument_curves_exactly_match_main=main_parity,
        split_and_five_session_label_purge_verified=True,
        rf_targets_match_original=True,
        all_candidates_retained=True,
        eligibility_recomputed_from_2021_only=True,
        eligibility_sha256=eligibility_hash,
        portfolio_initial_allocations_and_drift_verified=True,
        saved_portfolio_risk_recomputed_independently=True,
        comparison_differences_verified=len(comparisons),
    )
    (ROOT / "artifact_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    audit()
