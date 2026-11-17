import hashlib
import json
import math

import numpy as np
import pandas as pd

from .data import load_market
from .experiment import ROOT, experiment_identity
from .forecasts import source_hashes


def audit():
    frame = load_market()
    folder = ROOT / "test"
    selected_file = ROOT / "selected.json"
    selected = json.loads(selected_file.read_text())
    frozen = json.loads((folder / "frozen.json").read_text())
    completed = json.loads((folder / "completed.json").read_text())
    assert selected["completed_trials"] == 32
    assert selected["identity"] == experiment_identity() == frozen["identity"]
    assert frozen["selection_sha256"] == hashlib.sha256(selected_file.read_bytes()).hexdigest()
    assert frozen == completed["freeze"]
    models = {row["model_id"]: row for row in json.loads((ROOT / "model_manifest.json").read_text())}
    checked_models = set()
    origin_rows = 0
    for name in [
        "adis_predictions.csv",
        "paper_control_predictions.csv",
        "cm_predictions.csv",
        "forced_predictions.csv",
        "seed_7_predictions.csv",
        "seed_123_predictions.csv",
    ]:
        ledger = pd.read_csv(folder / name, index_col=0, parse_dates=True)
        ledger.index = pd.to_datetime(ledger.index, utc=True)
        assert ledger.index.is_unique and ledger.index.is_monotonic_increasing
        assert len(ledger) == 1461 and (ledger.index.hour == 23).all()
        np.testing.assert_allclose(ledger.close, frame.Close.reindex(ledger.index), rtol=0, atol=1e-9)
        expected_target = np.where(ledger.accepted & (ledger.predicted_return > 0.003), 0.95, 0)
        np.testing.assert_array_equal(ledger.target, expected_target)
        np.testing.assert_array_equal(ledger.accepted, np.isfinite(ledger.prediction))
        for model_id, rows in ledger.groupby("model_id"):
            metadata = models[model_id]
            fit_end = pd.Timestamp(metadata["fit_end"])
            assert fit_end <= rows.index.min() + pd.Timedelta(hours=1)
            assert metadata["source_hashes"] == source_hashes()
            if model_id not in checked_models:
                train = frame.Close.loc[
                    (frame.index >= pd.Timestamp(metadata["fit_start"])) & (frame.index < fit_end)
                ]
                assert len(train) == 10000
                assert (
                    hashlib.sha256(train.to_numpy(dtype="<f8").tobytes()).hexdigest()
                    == metadata["training_sha256"]
                )
                assert train.min() == metadata["minimum"] and np.ptp(train) == metadata["scale"]
                checked_models.add(model_id)
            width = (metadata["length"] - 1) * metadata["max_gap"]
            positions = frame.index.get_indexer(rows.index)
            indices = positions[:, None] - np.arange(width - 1, -1, -1)
            missing = ~frame.Observed.to_numpy()[indices].all(axis=1)
            np.testing.assert_array_equal(rows.missing_context, missing)
            assert (rows.loc[missing, "target"] == 0).all()
        origin_rows += len(ledger)
    metrics = pd.read_csv(folder / "metrics.csv").set_index("strategy")
    hourly_index = frame.loc[(frame.index >= "2022-01-01") & (frame.index < "2026-01-01")].index
    observed_times = frame.index[frame.Observed]
    trades_checked = 0
    for name, row in metrics.iterrows():
        destination = folder / name
        curve = pd.read_csv(destination / "equity.csv.gz", index_col=0, parse_dates=True)
        curve.index = pd.to_datetime(curve.index, utc=True)
        assert curve.index.equals(hourly_index)
        trades = pd.read_csv(destination / "trades.csv")
        orders = pd.read_csv(destination / "orders.csv")
        entry = pd.to_datetime(trades.EntryTime, utc=True)
        exit_ = pd.to_datetime(trades.ExitTime, utc=True)
        assert (trades.Size > 0).all() and (entry <= exit_).all()
        np.testing.assert_allclose(trades.EntryPrice, frame.Open.reindex(entry), rtol=0, atol=1e-8)
        np.testing.assert_allclose(trades.ExitPrice, frame.Open.reindex(exit_), rtol=0, atol=1e-8)
        fees = trades.Size * (trades.EntryPrice + trades.ExitPrice) * 0.0015
        pnl = trades.Size * (trades.ExitPrice - trades.EntryPrice) - fees
        np.testing.assert_allclose(fees, trades.Commission, rtol=1e-9, atol=1e-6)
        np.testing.assert_allclose(pnl, trades.PnL, rtol=1e-8, atol=1e-5)
        np.testing.assert_allclose(100000 + pnl.sum(), curve.equity.iloc[-1], rtol=1e-9, atol=1e-4)
        order_times = pd.to_datetime(orders.decision_bar, utc=True)
        fills = observed_times[observed_times.searchsorted(order_times, side="right")]
        assert set(entry).issubset(set(fills[orders.action == "buy"]))
        assert set(exit_).issubset(set(fills[orders.action.isin(["close", "reduce"])]))
        daily_equity = curve.equity.resample("1D").last().to_numpy()
        returns = daily_equity / np.r_[100000, daily_equity[:-1]] - 1
        tail = math.ceil(0.05 * len(returns))
        es = max(0, -np.sort(returns)[:tail].mean())
        peak = np.maximum.accumulate(np.r_[100000, curve.equity.to_numpy()])[1:]
        np.testing.assert_allclose(row.es95, es, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(row.daily_volatility, np.std(returns, ddof=1) * np.sqrt(365), atol=1e-12)
        np.testing.assert_allclose(row.max_drawdown, np.max(1 - curve.equity.to_numpy() / peak), atol=1e-12)
        np.testing.assert_allclose(row.mean_exposure, curve.exposure.mean(), atol=1e-12)
        trades_checked += len(trades)
    years = pd.read_csv(folder / "yearly_metrics.csv")
    for name, group in years.groupby("strategy"):
        np.testing.assert_allclose(
            np.prod(1 + group.total_return) - 1, metrics.loc[name, "total_return"], atol=1e-10
        )
        np.testing.assert_allclose(group.fees.sum(), metrics.loc[name, "fees"], rtol=1e-8, atol=1e-5)
    result = dict(
        forecast_rows=origin_rows,
        unique_test_models=len(checked_models),
        strategies=len(metrics),
        closed_trade_records=trades_checked,
        frozen_selection_verified=True,
        data_and_code_hashes_verified=True,
        training_boundaries_verified=True,
        no_synthetic_fills=True,
        next_real_open_order_fills_verified=True,
        independent_cash_pnl_verified=True,
        risk_metrics_independently_recomputed=True,
        yearly_returns_compound_to_total=True,
    )
    (folder / "audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    audit()
