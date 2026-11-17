import gzip
import hashlib
import json
import re

import numpy as np
import pandas as pd

from .equity_backtest import audit_result
from .equity_data import ROOT, calendar, load_asset, weekly_decisions
from .equity_experiment import PERIODS, audit_forecasts, freeze


def audit():
    state = freeze()
    completed = json.loads((ROOT / "completed.json").read_text())
    assert completed["frozen"] == state
    assert completed["main_backtests"] == 154
    assert completed["sensitivity_backtests"] == 140
    assert completed["seed_backtests"] == 56
    manifests = json.loads((ROOT / "data_manifest.json").read_text())
    for item in manifests:
        archived = gzip.decompress((ROOT / "snapshots" / f"{item['symbol']}.json.gz").read_bytes())
        assert hashlib.sha256(archived).hexdigest() == item["raw_sha256"]
    frame_by_symbol = {symbol: load_asset(symbol) for symbol in state["symbols"]}
    forecast_rows, checks = 0, {}
    decisions = pd.DatetimeIndex(weekly_decisions())
    decisions = decisions[(decisions >= "2021-01-01") & (decisions < "2026-11-01")]
    for symbol, frame in frame_by_symbol.items():
        directory = ROOT / symbol
        for filename in [
            "predictions.csv",
            "paper_predictions.csv",
            "seed_7_predictions.csv",
            "seed_123_predictions.csv",
        ]:
            ledger = pd.read_csv(directory / filename, index_col=0, parse_dates=True)
            assert ledger.index.equals(decisions)
            forecast_rows += audit_forecasts(frame, ledger, directory)["rows"]
        targets = pd.read_csv(directory / "targets.csv", index_col=0, parse_dates=True)
        expected = targets.partial_prediction.shift(1).rolling(13, min_periods=13).mean()
        np.testing.assert_allclose(targets.causal_control, expected, rtol=0, atol=1e-12, equal_nan=True)
    original = json.loads((ROOT / "audit.json").read_text())["backtests"]
    assert len(original) == 350
    for key in original:
        symbol, period, name = key.split("/")
        frame = frame_by_symbol[symbol]
        folder = ROOT / key
        metrics = json.loads((folder / "metrics.json").read_text())
        curve = pd.read_csv(folder / "equity.csv.gz", index_col=0, parse_dates=True)
        returns = pd.read_csv(folder / "returns.csv", index_col=0, parse_dates=True).iloc[:, 0]
        start, end = PERIODS[period]
        expected = calendar().sessions_in_range(start, pd.Timestamp(end) - pd.Timedelta(days=1))
        assert curve.index.equals(expected) and returns.index.equals(expected)
        costs = re.search(r"_cost(\d+)_", name)
        result = dict(
            metrics=metrics,
            curve=curve,
            returns=returns,
            trades=pd.read_csv(folder / "trades.csv.gz"),
            dividends=pd.read_csv(folder / "dividends.csv"),
            orders=pd.read_csv(folder / "orders.csv"),
        )
        checks[key] = audit_result(frame, result, costs_bps=int(costs[1]) if costs else 15)
        np.testing.assert_allclose(
            returns, curve.equity / curve.equity.shift(1).fillna(100000) - 1, atol=1e-12
        )
        np.testing.assert_allclose(metrics["volatility"], returns.std(ddof=1) * np.sqrt(252), atol=1e-12)
        np.testing.assert_allclose(metrics["mean_exposure"], curve.exposure.mean(), atol=1e-12)
        np.testing.assert_allclose(
            metrics["max_drawdown"],
            (1 - curve.equity / curve.equity.cummax().clip(lower=100000)).max(),
            atol=1e-12,
        )
        dividend_total = float(result["dividends"].amount.sum())
        np.testing.assert_allclose(metrics["dividends"], dividend_total, atol=1e-8)
        if "_dividends30" in name:
            events = result["dividends"]
            ex_date, due_date, paid_date = (
                pd.to_datetime(events[column]) for column in ["ex_date", "due_date", "paid_date"]
            )
            assert (due_date == ex_date + pd.Timedelta(days=30)).all()
            assert (paid_date.dropna() >= due_date.loc[paid_date.notna()]).all()
            for date, row in curve.iterrows():
                outstanding = events.loc[
                    (ex_date <= date) & (paid_date.isna() | (paid_date > date)), "amount"
                ].sum()
                np.testing.assert_allclose(row.receivables, outstanding, atol=1e-8)
                assert row.available_cash <= row.equity - row.receivables + 1e-7
    yearly = pd.read_csv(ROOT / "yearly_metrics.csv")
    metrics = pd.read_csv(ROOT / "metrics.csv").set_index(["symbol", "period", "strategy"])
    for key, rows in yearly.groupby(["symbol", "period", "strategy"]):
        np.testing.assert_allclose(
            np.prod(1 + rows.total_return) - 1, metrics.loc[key, "total_return"], atol=1e-10
        )
    result = dict(
        backtests_verified=len(checks),
        forecast_rows_verified=forecast_rows,
        symbols=state["symbols"],
        snapshot_archives_verified=True,
        complete_session_calendars_verified=True,
        training_boundaries_and_hashes_verified=True,
        lagged_controls_verified=True,
        dividend_entitlement_and_receivables_verified=True,
        independent_saved_pnl_fees_and_risk_verified=True,
        raw_projection_scope="Internal value-equivalence; not validation against independent raw exchange quotes",
        yearly_returns_compound=True,
    )
    (ROOT / "artifact_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    audit()
