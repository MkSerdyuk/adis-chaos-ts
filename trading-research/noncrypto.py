import json

import numpy as np
import pandas as pd
from numba import set_num_threads
from threadpoolctl import threadpool_limits
from tools.data import PAPER, TUNED, load_market, start_run
from tools.equity_backtest import audit_result, risk_metrics, run_daily
from tools.equity_baselines import technical_targets
from tools.inputs import daily_forecast, predictors
from tools.report import record, write_report
from tools.statistics import portfolio, weighted_bootstrap

CONFIG = dict(
    instruments=dict(
        AAPL="equity",
        MSFT="equity",
        XOM="equity",
        SPY="equity",
        QQQ="equity",
        TLT="treasury_bonds",
        GLD="gold",
    ),
    parameters=TUNED.copy(),
    paper_parameters=PAPER.copy(),
    recompute_comparators=False,
    costs_bps=15,
    delay_sessions=0,
    dividend_delay_days=0,
    periods=dict(
        validation=("2021-01-01", "2022-01-01"),
        historical=("2022-01-01", "2026-01-01"),
        holdout=("2026-01-01", "2026-11-01"),
    ),
    sensitivity_cases=[(0, 0, 0), (5, 0, 0), (30, 0, 0), (15, 1, 0), (15, 0, 30)],
    seeds=[7, 123],
)
CANDIDATES = ["random_forest", "gradient_boosting", "ridge", "svr", "sma50_200", "momentum90"]


def run(config=CONFIG):
    output = start_run("noncrypto", config)
    symbols = list(config["instruments"])
    weights = {
        symbol: 1 / 15 if category == "equity" else 1 / 3
        for symbol, category in config["instruments"].items()
    }
    results, audits, years, coverage = [], {}, [], []
    frames, targets_by_symbol, forecasts_by_symbol = {}, {}, {}
    returns = {period: {} for period in config["periods"] if period != "validation"}
    for symbol in symbols:
        frame = load_market(symbol)
        ledger = daily_forecast(frame, symbol, "main", config["parameters"], output)
        paper = daily_forecast(frame, symbol, "paper", config["paper_parameters"], output)
        forecasts = predictors(frame, symbol, ledger.index, config["recompute_comparators"])
        predicted = forecasts.pivot(index="decision_session", columns="model", values="target")
        strategies = technical_targets(frame, ledger.index)
        strategies["random_forest5"] = predicted.random_forest
        strategies["forecaster"] = ledger.target
        strategies["forecaster_paper_geometry"] = paper.target
        strategies["causal_control"] = ledger.target.shift(1).rolling(13, min_periods=13).mean()
        frames[symbol], targets_by_symbol[symbol], forecasts_by_symbol[symbol] = frame, strategies, predicted
        for period, (start, end) in config["periods"].items():
            if period == "validation":
                continue
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
                target = strategies[name]
                assert target.loc[(target.index >= start) & (target.index < end)].notna().all()
                result = run_daily(
                    frame,
                    target,
                    start,
                    end,
                    costs_bps=config["costs_bps"],
                    delay_sessions=config["delay_sessions"],
                    dividend_delay_days=config["dividend_delay_days"],
                    rebalance=name in {"causal_control", "vol_target20pct_30"},
                )
                audits[f"{symbol}/{period}/{name}"] = audit_result(frame, result, config["costs_bps"])
                initial = 100000.0
                for year, curve in result["curve"].groupby(result["curve"].index.year):
                    years.append(
                        dict(
                            symbol=symbol,
                            period=period,
                            strategy=name,
                            year=year,
                            **risk_metrics(curve.equity, curve.exposure, initial),
                        )
                    )
                    initial = curve.equity.iloc[-1]
                saved = record(results, result, scenario="main", symbol=symbol, period=period, strategy=name)
                if name in {"forecaster", "causal_control"}:
                    returns[period][symbol, name] = saved["returns"]
            for cost, delay, dividend_delay in config["sensitivity_cases"]:
                for name in ["forecaster", "causal_control"]:
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
                    audits[f"{symbol}/{period}/{name}/{cost}/{delay}/{dividend_delay}"] = audit_result(
                        frame, result, cost
                    )
                    record(
                        results,
                        result,
                        scenario="sensitivity",
                        symbol=symbol,
                        period=period,
                        strategy=name,
                        costs_bps=cost,
                        execution_delay_sessions=delay,
                        dividend_delay_days=dividend_delay,
                    )
        for seed in config["seeds"]:
            alternative = daily_forecast(
                frame,
                symbol,
                f"seed_{seed}",
                config["parameters"],
                output,
                seed,
            )
            control = alternative.target.shift(1).rolling(13, min_periods=13).mean()
            for period, (start, end) in config["periods"].items():
                if period == "validation":
                    continue
                for name, target in [("forecaster", alternative.target), ("causal_control", control)]:
                    result = run_daily(
                        frame,
                        target,
                        start,
                        end,
                        costs_bps=config["costs_bps"],
                        delay_sessions=config["delay_sessions"],
                        dividend_delay_days=config["dividend_delay_days"],
                        rebalance=name == "causal_control",
                    )
                    audits[f"{symbol}/{period}/{name}/seed{seed}"] = audit_result(
                        frame, result, config["costs_bps"]
                    )
                    record(
                        results,
                        result,
                        scenario="seed",
                        symbol=symbol,
                        period=period,
                        strategy=name,
                        seed=seed,
                    )
    intervals = []
    for period, values in returns.items():
        index = next(iter(values.values())).index
        assert all(series.index.equals(index) for series in values.values())
        method = np.array([values[symbol, "forecaster"] for symbol in symbols])
        control = np.array([values[symbol, "causal_control"] for symbol in symbols])
        for block in [5, 10, 20]:
            intervals.append(
                dict(
                    period=period,
                    primary=period == "holdout" and block == 10,
                    symbols=symbols,
                    **weighted_bootstrap(method, control, list(weights.values()), block=block),
                )
            )
    names = ["forecaster", *CANDIDATES, "buy_hold", "causal_control", "forecaster_paper_geometry"]
    portfolios, eligibility = {}, None
    for period, (start, end) in config["periods"].items():
        if period != "validation":
            assert eligibility is not None
        curves = {name: {} for name in names}
        for symbol in symbols:
            strategies = targets_by_symbol[symbol]
            predicted = forecasts_by_symbol[symbol]
            targets = predicted.copy()
            for name in [
                "forecaster",
                "sma50_200",
                "momentum90",
                "buy_hold",
                "causal_control",
                "forecaster_paper_geometry",
            ]:
                targets[name] = strategies[name]
            for name in names:
                target = targets[name]
                if name == "causal_control" and period == "validation":
                    target = target.fillna(0.0)
                result = run_daily(
                    frames[symbol],
                    target,
                    start,
                    end,
                    costs_bps=config["costs_bps"],
                    delay_sessions=config["delay_sessions"],
                    dividend_delay_days=config["dividend_delay_days"],
                    rebalance=name == "causal_control",
                )
                audits[f"predictive/{symbol}/{period}/{name}"] = audit_result(
                    frames[symbol], result, config["costs_bps"]
                )
                curves[name][symbol] = result["curve"]
                record(results, result, scenario="predictive", symbol=symbol, period=period, strategy=name)
        for name in names:
            combined = portfolio(curves[name], weights)
            result = dict(
                curve=combined,
                metrics=risk_metrics(combined.equity, combined.exposure),
                returns=combined.equity / combined.equity.shift(1).fillna(100000) - 1,
            )
            portfolios[period, name] = result
            record(
                results,
                result,
                scenario="portfolio",
                symbol="Class-balanced portfolio",
                period=period,
                strategy=name,
            )
        if period == "validation":
            eligibility = pd.DataFrame(
                [
                    dict(
                        strategy=name,
                        validation_return=portfolios[period, name]["metrics"]["total_return"],
                        validation_exposure=portfolios[period, name]["metrics"]["mean_exposure"],
                    )
                    for name in CANDIDATES
                ]
            )
            eligibility["eligible"] = (eligibility.validation_return > 0) & (
                eligibility.validation_exposure >= 0.10
            )
            eligibility.to_csv(output / "eligibility.csv", index=False)
    comparisons = []
    for period in config["periods"]:
        if period == "validation":
            continue
        method = portfolios[period, "forecaster"]["returns"]
        assert all(portfolios[period, name]["returns"].index.equals(method.index) for name in CANDIDATES)
        controls = np.array([portfolios[period, name]["returns"] for name in CANDIDATES])
        for block in [5, 10, 20]:
            interval = weighted_bootstrap(
                np.tile(method, (len(CANDIDATES), 1)),
                controls,
                np.full(len(CANDIDATES), 1 / len(CANDIDATES)),
                block=block,
            )
            for i, name in enumerate(CANDIDATES):
                comparisons.append(
                    dict(
                        period=period,
                        comparator=name,
                        block=block,
                        difference=interval["asset_differences"][i],
                        lower=interval["asset_lower"][i],
                        upper=interval["asset_upper"][i],
                        interval_confidence=1 - 0.05 / len(CANDIDATES),
                    )
                )
    (output / "audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    return write_report(
        output,
        "Stocks, bonds and gold research",
        results,
        dict(
            asset_risk_bootstrap=pd.DataFrame(intervals),
            portfolio_comparisons=pd.DataFrame(comparisons),
            eligibility=eligibility,
            yearly_metrics=pd.DataFrame(years),
            coverage=pd.DataFrame(coverage),
        ),
        "Unmodified src.Forecaster with transferred legacy BTC parameters. Daily split-adjusted prices with separate dividends. Quarterly 2520-session training, 5-session horizon, "
        "weekly decisions and next-open execution. 2021 validation; 2022–2025 historical test; January–October "
        "2026 historical holdout. All six predictive comparators are reported; eligibility uses only 2021. "
        "Asset-risk bootstrap measures weighted mean instrument ES95. Portfolio comparisons measure portfolio ES95. "
        "Portfolio initial weights: 1/15 per equity, 1/3 TLT, 1/3 GLD; no rebalancing between instruments.",
    )


if __name__ == "__main__":
    set_num_threads(4)
    with threadpool_limits(limits=1):
        print(run())
