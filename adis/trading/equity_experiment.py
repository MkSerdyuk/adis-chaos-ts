import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numba import set_num_threads
from threadpoolctl import threadpool_limits

from .equity_backtest import audit_result, risk_metrics, run_daily
from .equity_data import DATA, PROTOCOL, ROOT, calendar, load_asset
from .equity_forecasts import forest_targets, predict_daily, technical_targets
from .experiment import PAPER, experiment_identity

PERIODS = {"historical": ("2022-01-01", "2026-01-01"), "holdout": ("2026-01-01", "2026-11-01")}


def weighted_bootstrap(method, control, weights, block=10, repetitions=10000, seed=20261117):
    method, control, weights = np.asarray(method), np.asarray(control), np.asarray(weights)
    assert method.shape == control.shape and method.ndim == 2
    assert len(weights) == len(method) and np.isclose(weights.sum(), 1)
    assert np.isfinite(method).all() and np.isfinite(control).all()
    assets, days = method.shape
    count = int(np.ceil(0.05 * days))

    def es(values):
        return np.maximum(0, -np.partition(values, count - 1, axis=1)[:, :count].mean(axis=1))

    observed = es(method) - es(control)
    samples = np.empty((repetitions, assets))
    rng = np.random.default_rng(seed)
    for i in range(repetitions):
        starts = rng.integers(0, days, int(np.ceil(days / block)))
        indices = ((starts[:, None] + np.arange(block)) % days).ravel()[:days]
        samples[i] = es(method[:, indices]) - es(control[:, indices])
    lower, upper = np.quantile(samples @ weights, [0.025, 0.975])
    adjusted = np.quantile(samples, [0.025 / assets, 1 - 0.025 / assets], axis=0)
    return dict(
        difference=float(observed @ weights),
        lower=float(lower),
        upper=float(upper),
        asset_differences=observed.tolist(),
        asset_lower=adjusted[0].tolist(),
        asset_upper=adjusted[1].tolist(),
        weights=weights.tolist(),
        sessions=days,
        block=block,
        repetitions=repetitions,
        seed=seed,
    )


def freeze():
    protocol = json.loads(PROTOCOL.read_text())
    selected_path = Path("results/trading/selected.json")
    selected = json.loads(selected_path.read_text())
    assert selected["identity"] == experiment_identity()
    assert hashlib.sha256(selected_path.read_bytes()).hexdigest() == protocol["selection_sha256"]
    files = [
        PROTOCOL,
        selected_path,
        Path("uv.lock"),
        ROOT / "calendar.csv",
        ROOT / "data_manifest.json",
        ROOT / "corporate_action_checks.json",
        Path("research/trading/noncrypto/corporate_action_reference.json"),
    ]
    files.extend(
        Path(__file__).parent / name
        for name in ["equity_experiment.py", "equity_data.py", "equity_forecasts.py", "equity_backtest.py"]
    )
    for manifest in json.loads((ROOT / "data_manifest.json").read_text()):
        for suffix, key in [("json", "raw_sha256"), ("csv.gz", "csv_sha256")]:
            path = DATA / f"{manifest['symbol']}.{suffix}"
            assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest[key]
            files.append(path)
    state = dict(
        symbols=list(protocol["instruments"]),
        instruments=protocol["instruments"],
        parameters={key: value for key, value in selected["parameters"].items() if key != "training_hours"},
        paper_parameters={key: value for key, value in PAPER.items() if key != "training_hours"},
        btc_identity=selected["identity"],
        exchange_calendars=importlib.metadata.version("exchange_calendars"),
        files={
            str(path.resolve().relative_to(Path.cwd().resolve())): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in files
        },
    )
    output = ROOT / "frozen.json"
    if output.exists():
        assert json.loads(output.read_text()) == state, (
            "Noncrypto inputs or numerical code changed after freeze"
        )
    else:
        output.write_text(json.dumps(state, indent=2) + "\n")
    return state


def save_result(folder, result):
    folder.mkdir(parents=True, exist_ok=True)
    result["curve"].to_csv(folder / "equity.csv.gz")
    result["returns"].rename("return").to_csv(folder / "returns.csv")
    result["trades"].to_csv(folder / "trades.csv.gz", index=False)
    result["dividends"].to_csv(folder / "dividends.csv", index=False)
    result["orders"].to_csv(folder / "orders.csv", index=False)
    (folder / "metrics.json").write_text(json.dumps(result["metrics"], indent=2) + "\n")


def audit_forecasts(frame, ledger, directory):
    assert ledger.index.is_unique and ledger.index.is_monotonic_increasing
    np.testing.assert_allclose(ledger.close, frame.Close.reindex(ledger.index), atol=1e-9)
    np.testing.assert_array_equal(ledger.accepted, np.isfinite(ledger.prediction))
    np.testing.assert_array_equal(
        ledger.target, np.where(ledger.accepted & (ledger.predicted_return > 0.003), 0.95, 0)
    )
    for model_id, rows in ledger.groupby("model_id"):
        metadata = json.loads((directory / "metadata" / f"{model_id}.json").read_text())
        training = frame.Close.loc[
            (frame.index >= pd.Timestamp(metadata["fit_start"]))
            & (frame.index < pd.Timestamp(metadata["fit_end_exclusive"]))
        ]
        assert len(training) == 2520
        assert (
            metadata["training_sha256"]
            == hashlib.sha256(training.to_numpy(dtype="<f8").tobytes()).hexdigest()
        )
        assert training.min() == metadata["minimum"] and np.ptp(training) == metadata["scale"]
        assert training.index[-1] < rows.index.min()
    return dict(
        rows=len(ledger), models=ledger.model_id.nunique(), training_boundaries_and_hashes_verified=True
    )


def run():
    state = freeze()
    records, yearly, coverage, sensitivities, seed_metrics = [], [], [], [], []
    audits, forecast_audits = {}, {}
    returns = {period: {} for period in PERIODS}
    for symbol in state["symbols"]:
        directory = ROOT / symbol
        frame = load_asset(symbol)
        ledger = predict_daily(frame, state["parameters"], directory)
        paper = predict_daily(frame, state["paper_parameters"], directory)
        ledger.to_csv(directory / "predictions.csv")
        paper.to_csv(directory / "paper_predictions.csv")
        forecast_audits[f"{symbol}/main"] = audit_forecasts(frame, ledger, directory)
        forecast_audits[f"{symbol}/paper"] = audit_forecasts(frame, paper, directory)
        control = ledger.target.shift(1).rolling(13, min_periods=13).mean()
        forest, fits = forest_targets(frame, ledger.index)
        (directory / "forest_fits.json").write_text(json.dumps(fits, indent=2) + "\n")
        strategies = technical_targets(frame, ledger.index)
        strategies["random_forest5"] = forest
        strategies["partial_prediction"] = ledger.target
        strategies["paper_control"] = paper.target
        strategies["causal_control"] = control
        strategies.to_csv(directory / "targets.csv")
        for period, (start, end) in PERIODS.items():
            subset = ledger.loc[(ledger.index >= start) & (ledger.index < end)]
            coverage.append(
                dict(
                    symbol=symbol,
                    period=period,
                    decisions=len(subset),
                    accepted=int(subset.accepted.sum()),
                    outside_training_range=float(subset.outside_train_range.mean()),
                )
            )
            for name in strategies:
                assert (
                    strategies.loc[(strategies.index >= start) & (strategies.index < end), name].notna().all()
                )
                result = run_daily(
                    frame,
                    strategies[name],
                    start,
                    end,
                    rebalance=name in {"causal_control", "vol_target20pct_30"},
                )
                audits[f"{symbol}/{period}/{name}"] = audit_result(frame, result)
                save_result(directory / period / name, result)
                records.append(dict(symbol=symbol, period=period, strategy=name, **result["metrics"]))
                initial = 100000.0
                for year, curve in result["curve"].groupby(result["curve"].index.year):
                    yearly.append(
                        dict(
                            symbol=symbol,
                            period=period,
                            strategy=name,
                            year=year,
                            **risk_metrics(curve.equity, curve.exposure, initial),
                        )
                    )
                    initial = curve.equity.iloc[-1]
                if name in {"partial_prediction", "causal_control"}:
                    returns[period][symbol, name] = result["returns"]
                if symbol == "SPY" and period == "holdout" and name == "partial_prediction":
                    plot = result["engine"].plot(
                        results=result["stats"],
                        filename=str(ROOT / "stock_backtest.html"),
                        open_browser=False,
                        resample=False,
                        plot_drawdown=True,
                    )
                    from bokeh.io import save
                    from bokeh.resources import INLINE

                    save(
                        plot,
                        filename=str(ROOT / "stock_backtest.html"),
                        resources=INLINE,
                        title="Partial multistep prediction — SPY with dividends",
                    )
            cases = [(0, 0, 0), (5, 0, 0), (30, 0, 0), (15, 1, 0), (15, 0, 30)]
            for cost, delay, dividend_delay in cases:
                for name in ["partial_prediction", "causal_control"]:
                    result = run_daily(
                        frame,
                        strategies[name],
                        start,
                        end,
                        costs_bps=cost,
                        delay_sessions=delay,
                        dividend_delay_days=dividend_delay,
                        rebalance=name == "causal_control",
                    )
                    case = f"{name}_cost{cost}_delay{delay}_dividends{dividend_delay}"
                    audits[f"{symbol}/{period}/{case}"] = audit_result(frame, result, costs_bps=cost)
                    sensitivities.append(
                        dict(
                            symbol=symbol,
                            period=period,
                            strategy=name,
                            costs_bps=cost,
                            execution_delay_sessions=delay,
                            dividend_delay_days=dividend_delay,
                            **result["metrics"],
                        )
                    )
                    save_result(directory / period / case, result)
            print(symbol, period, "main and sensitivity accounting verified", flush=True)
        for seed in [7, 123]:
            alternative = predict_daily(frame, state["parameters"], directory, seed=seed)
            alternative.to_csv(directory / f"seed_{seed}_predictions.csv")
            forecast_audits[f"{symbol}/seed{seed}"] = audit_forecasts(frame, alternative, directory)
            seed_control = alternative.target.shift(1).rolling(13, min_periods=13).mean()
            for period, (start, end) in PERIODS.items():
                for name, target in [
                    ("partial_prediction", alternative.target),
                    ("causal_control", seed_control),
                ]:
                    result = run_daily(frame, target, start, end, rebalance=name == "causal_control")
                    audits[f"{symbol}/{period}/{name}_seed{seed}"] = audit_result(frame, result)
                    save_result(directory / period / f"{name}_seed{seed}", result)
                    seed_metrics.append(
                        dict(symbol=symbol, period=period, strategy=name, seed=seed, **result["metrics"])
                    )
    pd.DataFrame(records).to_csv(ROOT / "metrics.csv", index=False)
    pd.DataFrame(yearly).to_csv(ROOT / "yearly_metrics.csv", index=False)
    pd.DataFrame(coverage).to_csv(ROOT / "coverage.csv", index=False)
    pd.DataFrame(sensitivities).to_csv(ROOT / "sensitivity.csv", index=False)
    pd.DataFrame(seed_metrics).to_csv(ROOT / "seeds.csv", index=False)
    weights = [1 / 15 if state["instruments"][symbol] == "equity" else 1 / 3 for symbol in state["symbols"]]
    intervals = []
    for period, values in returns.items():
        start, end = PERIODS[period]
        expected = calendar().sessions_in_range(start, pd.Timestamp(end) - pd.Timedelta(days=1))
        assert all(series.index.equals(expected) for series in values.values())
        method = np.array([values[symbol, "partial_prediction"] for symbol in state["symbols"]])
        control = np.array([values[symbol, "causal_control"] for symbol in state["symbols"]])
        for block in [5, 10, 20]:
            intervals.append(
                dict(
                    period=period,
                    primary=period == "holdout" and block == 10,
                    symbols=state["symbols"],
                    **weighted_bootstrap(method, control, weights, block=block),
                )
            )
    (ROOT / "bootstrap.json").write_text(json.dumps(intervals, indent=2) + "\n")
    (ROOT / "audit.json").write_text(
        json.dumps(dict(backtests=audits, forecasts=forecast_audits), indent=2) + "\n"
    )
    assert freeze() == state
    (ROOT / "completed.json").write_text(
        json.dumps(
            dict(
                main_backtests=len(records),
                sensitivity_backtests=len(sensitivities),
                seed_backtests=len(seed_metrics),
                frozen=state,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    set_num_threads(4)
    with threadpool_limits(limits=1):
        run()
