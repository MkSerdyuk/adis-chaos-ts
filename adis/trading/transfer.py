import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numba import set_num_threads
from threadpoolctl import threadpool_limits

from .backtest import run_backtest, summarize
from .baselines import technical_targets
from .evaluation import collect_model_metadata, yearly_metrics
from .experiment import experiment_identity
from .forecasts import walk_forward
from .transfer_data import RETIREMENT, load_asset, retirement
from .universe import DATA, PROTOCOL, ROOT

PERIODS = {"historical": ("2022-01-01", "2026-01-01"), "holdout": ("2026-01-01", "2026-11-01")}


def synchronized_bootstrap(method, control, blocks=14, repetitions=10000, seed=20261117):
    method, control = np.asarray(method), np.asarray(control)
    assert method.shape == control.shape and method.ndim == 2
    assert np.isfinite(method).all() and np.isfinite(control).all()
    assets, days = method.shape
    count = int(np.ceil(0.05 * days))

    def es(values):
        return np.maximum(0, -np.partition(values, count - 1, axis=1)[:, :count].mean(axis=1))

    estimate = es(method) - es(control)
    samples = np.empty((repetitions, assets))
    rng = np.random.default_rng(seed)
    for i in range(repetitions):
        starts = rng.integers(0, days, int(np.ceil(days / blocks)))
        indices = ((starts[:, None] + np.arange(blocks)) % days).ravel()[:days]
        samples[i] = es(method[:, indices]) - es(control[:, indices])
    aggregate = np.quantile(samples.mean(axis=1), [0.025, 0.975])
    adjusted = np.quantile(samples, [0.025 / assets, 1 - 0.025 / assets], axis=0)
    return dict(
        mean_difference=float(estimate.mean()),
        lower=float(aggregate[0]),
        upper=float(aggregate[1]),
        asset_differences=estimate.tolist(),
        asset_lower=adjusted[0].tolist(),
        asset_upper=adjusted[1].tolist(),
        asset_interval_confidence=1 - 0.05 / assets,
        block_days=blocks,
        repetitions=repetitions,
        seed=seed,
    )


def retire_targets(targets, notice):
    targets = targets.copy()
    if notice:
        targets.loc[targets.index >= pd.Timestamp(notice["retirement_decision_bar_utc"])] = 0.0
    return targets


def retired_cash(frame, start, end):
    index = frame.loc[(frame.index >= start) & (frame.index < end)].index
    equity = pd.Series(100000.0, index=index)
    exposure = pd.Series(0.0, index=index)
    columns = [
        "Size",
        "EntryPrice",
        "ExitPrice",
        "Commission",
        "PnL",
        "EntryBar",
        "ExitBar",
        "EntryTime",
        "ExitTime",
    ]
    trades = pd.DataFrame(columns=columns)
    return dict(
        equity=equity,
        exposure=exposure,
        trades=trades,
        orders=pd.DataFrame(columns=["decision_bar", "target", "action"]),
        daily_returns=pd.Series(0.0, index=equity.resample("1D").last().index),
        metrics=summarize(
            equity, exposure, trades, 100000, pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
        ),
    )


def audit_execution(frame, result, notice):
    trades = result["trades"].copy()
    numeric = ["Size", "EntryPrice", "ExitPrice", "Commission", "PnL"]
    trades[numeric] = trades[numeric].astype(float)
    entry, exit_ = pd.to_datetime(trades.EntryTime, utc=True), pd.to_datetime(trades.ExitTime, utc=True)
    assert (trades.Size > 0).all() and (entry <= exit_).all()
    assert frame.Observed.reindex(entry).all() and frame.Observed.reindex(exit_).all()
    np.testing.assert_allclose(trades.EntryPrice, frame.Open.reindex(entry), rtol=0, atol=1e-8)
    np.testing.assert_allclose(trades.ExitPrice, frame.Open.reindex(exit_), rtol=0, atol=1e-8)
    fees = trades.Size * (trades.EntryPrice + trades.ExitPrice) * 0.0015
    pnl = trades.Size * (trades.ExitPrice - trades.EntryPrice) - fees
    np.testing.assert_allclose(fees, trades.Commission, rtol=1e-9, atol=1e-6)
    np.testing.assert_allclose(pnl, trades.PnL, rtol=1e-8, atol=1e-5)
    np.testing.assert_allclose(100000 + pnl.sum(), result["equity"].iloc[-1], rtol=1e-9, atol=1e-4)
    orders = result["orders"]
    observed = frame.index[frame.Observed]
    order_times = pd.to_datetime(orders.decision_bar, utc=True)
    fills = observed[observed.searchsorted(order_times, side="right")]
    assert set(entry).issubset(set(fills[orders.action == "buy"]))
    assert set(exit_).issubset(set(fills[orders.action.isin(["close", "reduce"])]))
    if notice:
        cutoff = pd.Timestamp(notice["retirement_decision_bar_utc"]) + pd.Timedelta(hours=1)
        liquidation = observed[observed.searchsorted(cutoff, side="right")]
        assert liquidation < pd.Timestamp(notice["spot_ceases_utc"])
        assert (exit_ <= liquidation).all()
        assert (result["exposure"].loc[result["exposure"].index >= liquidation] == 0).all()
    daily = result["equity"].resample("1D").last().to_numpy()
    returns = daily / np.r_[100000, daily[:-1]] - 1
    np.testing.assert_allclose(returns, result["daily_returns"], atol=1e-12)
    es = max(0, -np.sort(returns)[: int(np.ceil(0.05 * len(returns)))].mean())
    np.testing.assert_allclose(es, result["metrics"]["es95"], atol=1e-12)
    peak = np.maximum.accumulate(np.r_[100000, result["equity"].to_numpy()])[1:]
    np.testing.assert_allclose(
        np.max(1 - result["equity"].to_numpy() / peak), result["metrics"]["max_drawdown"], atol=1e-12
    )
    return dict(
        trades=len(trades),
        cash_pnl_verified=True,
        observed_open_fills_verified=True,
        retirement_verified=bool(notice),
    )


def save_result(folder, result):
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(equity=result["equity"], exposure=result["exposure"])).to_csv(folder / "equity.csv.gz")
    result["trades"].to_csv(folder / "trades.csv.gz", index=False)
    result["orders"].to_csv(folder / "orders.csv", index=False)
    result["daily_returns"].rename("return").to_csv(folder / "daily_returns.csv")
    (folder / "metrics.json").write_text(json.dumps(result["metrics"], indent=2) + "\n")


def freeze_inputs():
    protocol = json.loads(PROTOCOL.read_text())
    universe = json.loads((ROOT / "universe.json").read_text())
    selected = json.loads(Path("results/trading/selected.json").read_text())
    assert selected["identity"] == experiment_identity()
    assert (
        hashlib.sha256(Path("results/trading/selected.json").read_bytes()).hexdigest()
        == protocol["selection_sha256"]
    )
    assert universe["protocol_sha256"] == hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    addendum = json.loads(RETIREMENT.read_text())
    assert addendum["base_protocol_sha256"] == universe["protocol_sha256"]
    for symbol in universe["selected"]:
        notice = retirement(symbol)
        if notice:
            decision_close = pd.Timestamp(notice["retirement_decision_bar_utc"]) + pd.Timedelta(hours=1)
            assert decision_close >= pd.Timestamp(notice["available_after_utc"])
            assert decision_close < pd.Timestamp(notice["spot_ceases_utc"])
    files = [
        PROTOCOL,
        RETIREMENT,
        ROOT / "universe.json",
        Path("research/trading/eos_announcement.txt"),
        Path(__file__),
        Path(__file__).with_name("transfer_data.py"),
        Path(__file__).with_name("evaluation.py"),
        Path(__file__).with_name("data.py"),
    ]
    for symbol in universe["selected"]:
        manifest = json.loads((ROOT / symbol / "data_manifest.json").read_text())
        path = DATA / symbol / "market.csv.gz"
        assert manifest["csv_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        files.extend([ROOT / symbol / "data_manifest.json", path])
    frozen = dict(
        btc_selection_identity=selected["identity"],
        parameters=selected["parameters"],
        symbols=universe["selected"],
        files={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
    )
    path = ROOT / "frozen.json"
    if path.exists():
        assert json.loads(path.read_text()) == frozen, "Transfer inputs changed after freeze"
    else:
        path.write_text(json.dumps(frozen, indent=2) + "\n")
    return frozen


def run_transfer():
    frozen = freeze_inputs()
    records, years, audits, coverage = [], [], {}, []
    returns = {period: {} for period in PERIODS}
    for symbol in frozen["symbols"]:
        frame, notice = load_asset(symbol), retirement(symbol)
        directory = ROOT / symbol
        active_end = (
            pd.Timestamp(notice["retirement_decision_bar_utc"])
            if notice
            else pd.Timestamp("2026-11-01", tz="UTC")
        )
        ledger = walk_forward(
            frame, "2021-01-01", active_end.strftime("%Y-%m-%d %H:%M:%S"), frozen["parameters"], directory
        )
        ledger.to_csv(directory / "active_predictions.csv")
        origins = frame.index[(frame.index >= "2021-01-01") & (frame.index.hour == 23)]
        target = retire_targets(ledger.target.reindex(origins, fill_value=0.0), notice)
        control = retire_targets(target.shift(1).rolling(90, min_periods=90).mean(), notice)
        technical = technical_targets(frame)
        strategies = {name: retire_targets(technical[name], notice) for name in technical}
        strategies.update(partial_prediction=target, causal_exposure_control=control)
        pd.DataFrame(dict(target=target, causal_control=control)).to_csv(directory / "targets.csv")
        models = collect_model_metadata(directory)
        (directory / "model_manifest.json").write_text(json.dumps(models, indent=2) + "\n")
        assert (pd.to_datetime(ledger.fit_end, utc=True) <= ledger.index + pd.Timedelta(hours=1)).all()
        for period, (start, end) in PERIODS.items():
            inactive = bool(notice and active_end < pd.Timestamp(start, tz="UTC"))
            subset = ledger.loc[(ledger.index >= start) & (ledger.index < end)]
            coverage.append(
                dict(
                    symbol=symbol,
                    period=period,
                    retired_slot=inactive,
                    active_origins=len(subset),
                    accepted=int(subset.accepted.sum()),
                )
            )
            for name, targets in strategies.items():
                assert targets.loc[(targets.index >= start) & (targets.index < end)].notna().all()
                result = (
                    retired_cash(frame, start, end)
                    if inactive
                    else run_backtest(
                        frame,
                        targets,
                        start,
                        end,
                        rebalance_daily=name in {"causal_exposure_control", "vol_target_20pct_30daily"},
                    )
                )
                audits[f"{symbol}/{period}/{name}"] = audit_execution(frame, result, notice)
                save_result(directory / period / name, result)
                records.append(
                    dict(
                        symbol=symbol,
                        period=period,
                        strategy=name,
                        retired_slot=inactive,
                        **result["metrics"],
                    )
                )
                years.extend(
                    dict(symbol=symbol, period=period, strategy=name, **row) for row in yearly_metrics(result)
                )
                if name in {"partial_prediction", "causal_exposure_control"}:
                    returns[period][(symbol, name)] = result["daily_returns"]
            print(symbol, period, "backtests and accounting verified", flush=True)
    pd.DataFrame(records).to_csv(ROOT / "metrics.csv", index=False)
    pd.DataFrame(years).to_csv(ROOT / "yearly_metrics.csv", index=False)
    pd.DataFrame(coverage).to_csv(ROOT / "coverage.csv", index=False)
    intervals = []
    for period, values in returns.items():
        calendar = next(iter(values.values())).index
        assert all(series.index.equals(calendar) for series in values.values())
        assert len(calendar) == (1461 if period == "historical" else 304)
        method = np.array([values[symbol, "partial_prediction"] for symbol in frozen["symbols"]])
        control = np.array([values[symbol, "causal_exposure_control"] for symbol in frozen["symbols"]])
        for block in [7, 14, 28]:
            statistics = synchronized_bootstrap(method, control, blocks=block)
            intervals.append(
                dict(
                    period=period,
                    primary=period == "holdout" and block == 14,
                    symbols=frozen["symbols"],
                    **statistics,
                )
            )
    (ROOT / "bootstrap.json").write_text(json.dumps(intervals, indent=2) + "\n")
    (ROOT / "audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    assert freeze_inputs() == frozen
    (ROOT / "completed.json").write_text(
        json.dumps(dict(symbols=frozen["symbols"], backtests=len(records), frozen=frozen), indent=2) + "\n"
    )


if __name__ == "__main__":
    set_num_threads(4)
    with threadpool_limits(limits=1):
        run_transfer()
