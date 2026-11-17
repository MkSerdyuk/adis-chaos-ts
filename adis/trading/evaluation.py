import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numba import set_num_threads
from threadpoolctl import threadpool_limits

from .backtest import expected_shortfall, run_backtest, summarize
from .baselines import forest_targets, technical_targets
from .data import load_market
from .experiment import PAPER, ROOT, experiment_identity
from .forecasts import walk_forward


def paired_bootstrap(first, second, blocks=14, repetitions=2000, seed=20261117):
    if isinstance(first, pd.Series) and isinstance(second, pd.Series):
        assert first.index.equals(second.index)
    first, second = np.asarray(first), np.asarray(second)
    assert first.shape == second.shape and np.isfinite(first).all() and np.isfinite(second).all()
    rng = np.random.default_rng(seed)
    length = len(first)
    differences = np.empty(repetitions)
    for i in range(repetitions):
        starts = rng.integers(0, length, int(np.ceil(length / blocks)))
        indices = ((starts[:, None] + np.arange(blocks)) % length).ravel()[:length]
        differences[i] = expected_shortfall(first[indices]) - expected_shortfall(second[indices])
    return dict(
        difference=expected_shortfall(first) - expected_shortfall(second),
        lower=float(np.quantile(differences, 0.025)),
        upper=float(np.quantile(differences, 0.975)),
        block_days=blocks,
        replicates=repetitions,
        seed=seed,
    )


def yearly_metrics(result, cost_bps=15, initial_cash=100000):
    records = []
    equity, exposure, trades = result["equity"], result["exposure"], result["trades"]
    entry_time = pd.to_datetime(trades.EntryTime, utc=True)
    exit_time = pd.to_datetime(trades.ExitTime, utc=True)
    for year in sorted(set(equity.index.year)):
        start, end = pd.Timestamp(f"{year}-01-01", tz="UTC"), pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
        previous = equity.loc[equity.index < start]
        opening_equity = float(previous.iloc[-1]) if len(previous) else initial_cash
        mask = (equity.index >= start) & (equity.index < end)
        metrics = summarize(equity.loc[mask], exposure.loc[mask], trades.iloc[:0], opening_equity, start, end)
        entry = trades.loc[(entry_time >= start) & (entry_time < end)]
        exit_ = trades.loc[(exit_time >= start) & (exit_time < end)]
        notional = float(
            (entry.Size.abs() * entry.EntryPrice).sum() + (exit_.Size.abs() * exit_.ExitPrice).sum()
        )
        metrics.update(
            year=year,
            entries=int(entry.EntryBar.nunique()),
            fees=notional * cost_bps / 10000,
            turnover_initial_capital=notional / opening_equity,
        )
        records.append(metrics)
    return records


def save_result(name, result, folder):
    destination = folder / name
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "metrics.json").write_text(json.dumps(result["metrics"], indent=2))
    result["trades"].to_csv(destination / "trades.csv", index=False)
    result["orders"].to_csv(destination / "orders.csv", index=False)
    pd.DataFrame(dict(equity=result["equity"], exposure=result["exposure"])).to_csv(
        destination / "equity.csv.gz"
    )
    result["daily_returns"].rename("return").to_csv(destination / "daily_returns.csv")


def forecast_diagnostics(frame, ledger, folder):
    diagnostic = ledger.copy()
    future = frame.Close.reindex(ledger.index + pd.Timedelta(hours=24)).to_numpy()
    diagnostic["realized_24h_return"] = future / ledger.close.to_numpy() - 1
    diagnostic["relative_forecast_error"] = (ledger.prediction.to_numpy() - future) / ledger.close.to_numpy()
    diagnostic["group"] = np.select(
        [ledger.missing_context, ~ledger.accepted, ledger.target > 0],
        ["missing_context", "no_terminal_forecast", "accepted_long"],
        default="accepted_cash",
    )
    diagnostic.to_csv(folder / "forecast_diagnostics.csv")
    rows = []
    for group, values in diagnostic.groupby("group"):
        returns = values.realized_24h_return.dropna()
        errors = values.relative_forecast_error.dropna()
        rows.append(
            dict(
                group=group,
                origins=len(values),
                evaluated=len(returns),
                mean_absolute_underlying_return=float(returns.abs().mean()),
                underlying_es95=expected_shortfall(returns) if len(returns) else None,
                relative_forecast_rmse=float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
            )
        )
    pd.DataFrame(rows).to_csv(folder / "conditional_risk.csv", index=False)
    yearly = ledger.groupby(ledger.index.year).agg(
        origins=("accepted", "size"),
        forecast_coverage=("accepted", "mean"),
        h1_coverage=("accepted_h1", "mean"),
        outside_train_range=("outside_train_range", "mean"),
        missing_context=("missing_context", "mean"),
    )
    yearly.to_csv(folder / "forecast_coverage.csv")


def run_test():
    selected_path = ROOT / "selected.json"
    selected = json.loads(selected_path.read_text())
    assert selected["completed_trials"] == 32
    assert selected["identity"] == experiment_identity(), (
        "Frozen selection no longer matches code/data/protocol"
    )
    frame = load_market()
    folder = ROOT / "test"
    folder.mkdir(parents=True, exist_ok=True)
    freeze = dict(
        selection_sha256=hashlib.sha256(selected_path.read_bytes()).hexdigest(),
        identity=selected["identity"],
        evaluation_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    manifest = folder / "frozen.json"
    if manifest.exists():
        assert json.loads(manifest.read_text()) == freeze
    else:
        manifest.write_text(json.dumps(freeze, indent=2))
    start, end = "2022-01-01", "2026-01-01"
    parameters = selected["parameters"]
    ledger = walk_forward(frame, start, end, parameters)
    ledger.to_csv(folder / "adis_predictions.csv")
    forecast_diagnostics(frame, ledger, folder)
    paper = walk_forward(frame, start, end, PAPER)
    paper.to_csv(folder / "paper_control_predictions.csv")
    no_growth = walk_forward(frame, start, end, parameters, mode="cm")
    no_growth.to_csv(folder / "cm_predictions.csv")
    forced = walk_forward(frame, start, end, parameters, mode="fp")
    forced.to_csv(folder / "forced_predictions.csv")
    validation = pd.read_csv(
        ROOT / "validation" / f"trial_{selected['trial']:03d}" / "predictions.csv",
        index_col=0,
        parse_dates=True,
    )
    combined = pd.concat([validation.target, ledger.target]).sort_index()
    causal_control = combined.shift(1).rolling(90, min_periods=90).mean().loc[ledger.index]
    assert causal_control.notna().all()
    targets = technical_targets(frame)
    forest, fits = forest_targets(frame, start, end)
    (folder / "forest_fits.json").write_text(json.dumps(fits, indent=2))
    strategies = {name: (targets[name], name == "vol_target_20pct_30daily") for name in targets}
    strategies.update(
        adis_tuned=(ledger.target, False),
        adis_paper_control=(paper.target, False),
        adis_no_growth_gate=(no_growth.target, False),
        forced_nearest_avg=(forced.target, False),
        random_forest_24h_return=(forest, False),
        causal_exposure_control=(causal_control, True),
    )
    results, records, yearly = {}, [], []
    for name, (target, rebalance) in strategies.items():
        result = run_backtest(frame, target, start, end, rebalance_daily=rebalance)
        save_result(name, result, folder)
        results[name] = result
        records.append(dict(strategy=name, **result["metrics"]))
        yearly.extend(dict(strategy=name, **row) for row in yearly_metrics(result))
        print(name, result["metrics"], flush=True)
    requested_fraction = results["adis_tuned"]["metrics"]["mean_exposure"]
    used_fraction = min(requested_fraction, 0.95)
    diagnostic_target = pd.Series(used_fraction, index=ledger.index)
    (folder / "retrospective_control.json").write_text(
        json.dumps(
            dict(
                requested_allocation=requested_fraction,
                used_allocation=used_fraction,
                retrospective=True,
                daily_rebalanced=True,
            ),
            indent=2,
        )
    )
    diagnostic = run_backtest(frame, diagnostic_target, start, end, rebalance_daily=True)
    save_result("retrospective_exposure_control", diagnostic, folder)
    results["retrospective_exposure_control"] = diagnostic
    records.append(dict(strategy="retrospective_exposure_control", **diagnostic["metrics"]))
    yearly.extend(
        dict(strategy="retrospective_exposure_control", **row) for row in yearly_metrics(diagnostic)
    )
    pd.DataFrame(records).to_csv(folder / "metrics.csv", index=False)
    pd.DataFrame(yearly).to_csv(folder / "yearly_metrics.csv", index=False)
    pd.DataFrame({name: result["daily_returns"] for name, result in results.items()}).to_csv(
        folder / "daily_returns.csv"
    )
    pd.DataFrame({name: result["equity"] for name, result in results.items()}).to_csv(
        folder / "equities.csv.gz"
    )
    intervals = []
    for name in [
        "causal_exposure_control",
        "retrospective_exposure_control",
        "buy_hold",
        "sma_50_200_daily",
        "random_forest_24h_return",
        "adis_no_growth_gate",
    ]:
        for block in [7, 14, 28] if name == "causal_exposure_control" else [14]:
            interval = paired_bootstrap(
                results["adis_tuned"]["daily_returns"], results[name]["daily_returns"], block
            )
            intervals.append(
                dict(comparator=name, primary=name == "causal_exposure_control" and block == 14, **interval)
            )
    pd.DataFrame(intervals).to_csv(folder / "bootstrap.csv", index=False)
    sensitivity = []
    for cost in [0, 5, 15, 30]:
        for delay in [0, 1, 2]:
            for name in ["adis_tuned", "causal_exposure_control", "buy_hold", "sma_50_200_daily"]:
                target, rebalance = strategies[name]
                result = run_backtest(
                    frame, target, start, end, costs_bps=cost, delay_hours=delay, rebalance_daily=rebalance
                )
                sensitivity.append(
                    dict(strategy=name, costs_bps=cost, delay_hours=delay, **result["metrics"])
                )
    pd.DataFrame(sensitivity).to_csv(folder / "sensitivity.csv", index=False)
    seeds = [dict(seed=42, **results["adis_tuned"]["metrics"])]
    for seed in [7, 123]:
        alternative = walk_forward(frame, start, end, parameters, seed=seed)
        alternative.to_csv(folder / f"seed_{seed}_predictions.csv")
        result = run_backtest(frame, alternative.target, start, end)
        save_result(f"adis_seed_{seed}", result, folder)
        seeds.append(dict(seed=seed, **result["metrics"]))
    pd.DataFrame(seeds).to_csv(folder / "seeds.csv", index=False)
    models = [json.loads(path.read_text()) for path in sorted((ROOT / "models").glob("*/metadata.json"))]
    (ROOT / "model_manifest.json").write_text(json.dumps(models, indent=2))
    (ROOT / "data_manifest.json").write_bytes(Path("data/trading/manifest.json").read_bytes())
    demo = results["adis_tuned"]
    plot = demo["engine"].plot(
        results=demo["stats"],
        filename=str(folder / "adis_backtest.html"),
        open_browser=False,
        resample="1D",
        plot_drawdown=True,
        plot_return=True,
    )
    from bokeh.io import save
    from bokeh.resources import INLINE

    save(plot, filename=str(folder / "adis_backtest.html"), resources=INLINE, title="ADIS BTC/USDT backtest")
    (folder / "completed.json").write_text(
        json.dumps(
            dict(
                strategies=len(results),
                test_start=start,
                test_end_exclusive=end,
                forecast_origins=len(ledger),
                seeds=[42, 7, 123],
                freeze=freeze,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    set_num_threads(4)
    with threadpool_limits(limits=1):
        run_test()
