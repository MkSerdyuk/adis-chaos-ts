import pandas as pd
from numba import set_num_threads
from threadpoolctl import threadpool_limits
from tools.backtest import run_backtest
from tools.baselines import forest_targets, technical_targets
from tools.data import PAPER, TUNED, load_market, start_run
from tools.inputs import hourly_forecast
from tools.report import record, write_report
from tools.statistics import paired_bootstrap, yearly_metrics

CONFIG = dict(
    parameters=TUNED.copy(),
    paper_parameters=PAPER.copy(),
    start="2022-01-01",
    end="2026-01-01",
    costs_bps=15,
    delay_hours=1,
    sensitivity_costs=[0, 5, 15, 30],
    sensitivity_delays=[0, 1, 2],
    seeds=[7, 123],
)


def run(config=CONFIG):
    assert pd.Timestamp(config["start"]).is_quarter_start
    output = start_run("btc", config)
    frame = load_market("BTCUSDT")
    ledgers = {}
    for name, parameters, start, end, seed in [
        ("main", config["parameters"], "2022-01-01", config["end"], None),
        ("paper", config["paper_parameters"], "2022-01-01", config["end"], None),
        ("validation", config["parameters"], "2021-01-01", "2022-01-01", None),
        *[
            (f"seed_{seed}", config["parameters"], config["start"], config["end"], seed)
            for seed in config["seeds"]
        ],
    ]:
        ledgers[name] = hourly_forecast(frame, "BTCUSDT", name, parameters, output, start, end, seed)
    target = ledgers["main"].target
    combined = pd.concat([ledgers["validation"].target, target]).sort_index()
    control = combined.shift(1).rolling(90, min_periods=90).mean().loc[target.index]
    assert control.notna().all()
    technical = technical_targets(frame)
    forest, fits = forest_targets(frame, config["start"], config["end"])
    pd.DataFrame(fits).to_csv(output / "forest_fits.csv", index=False)
    forest.rename("target").to_csv(output / "forest_targets.csv")
    strategies = {name: (technical[name], name == "vol_target_20pct_30daily") for name in technical}
    strategies.update(
        forecaster=(target, False),
        forecaster_paper_geometry=(ledgers["paper"].target, False),
        random_forest_24h_return=(forest, False),
        causal_exposure_control=(control, True),
    )
    results, main, years = [], {}, []
    for name, (targets, rebalance) in strategies.items():
        result = run_backtest(
            frame,
            targets,
            config["start"],
            config["end"],
            costs_bps=config["costs_bps"],
            delay_hours=config["delay_hours"],
            rebalance_daily=rebalance,
        )
        years.extend(dict(strategy=name, **row) for row in yearly_metrics(result, config["costs_bps"]))
        main[name] = record(
            results, result, scenario="main", symbol="BTCUSDT", period="historical", strategy=name
        )
    retrospective = pd.Series(min(main["forecaster"]["metrics"]["mean_exposure"], 0.95), index=target.index)
    result = run_backtest(
        frame,
        retrospective,
        config["start"],
        config["end"],
        costs_bps=config["costs_bps"],
        delay_hours=config["delay_hours"],
        rebalance_daily=True,
    )
    years.extend(
        dict(strategy="retrospective_exposure_control", **row)
        for row in yearly_metrics(result, config["costs_bps"])
    )
    main["retrospective_exposure_control"] = record(
        results,
        result,
        scenario="main",
        symbol="BTCUSDT",
        period="historical",
        strategy="retrospective_exposure_control",
    )
    intervals = []
    for name in [
        "causal_exposure_control",
        "retrospective_exposure_control",
        "buy_hold",
        "sma_50_200_daily",
        "random_forest_24h_return",
    ]:
        for block in [7, 14, 28] if name == "causal_exposure_control" else [14]:
            intervals.append(
                dict(
                    comparator=name,
                    primary=name == "causal_exposure_control" and block == 14,
                    **paired_bootstrap(main["forecaster"]["returns"], main[name]["returns"], block),
                )
            )
    for cost in config["sensitivity_costs"]:
        for delay in config["sensitivity_delays"]:
            for name in [
                "forecaster",
                "causal_exposure_control",
                "buy_hold",
                "sma_50_200_daily",
            ]:
                targets, rebalance = strategies[name]
                result = run_backtest(
                    frame,
                    targets,
                    config["start"],
                    config["end"],
                    costs_bps=cost,
                    delay_hours=delay,
                    rebalance_daily=rebalance,
                )
                record(
                    results,
                    result,
                    scenario="sensitivity",
                    symbol="BTCUSDT",
                    period="historical",
                    strategy=name,
                    costs_bps=cost,
                    delay_hours=delay,
                )
    seeds = [dict(seed=config["parameters"]["seed"], **main["forecaster"]["metrics"])]
    for seed in config["seeds"]:
        result = run_backtest(
            frame,
            ledgers[f"seed_{seed}"].target,
            config["start"],
            config["end"],
            costs_bps=config["costs_bps"],
            delay_hours=config["delay_hours"],
        )
        seeds.append(dict(seed=seed, **result["metrics"]))
        record(
            results,
            result,
            scenario="seed",
            symbol="BTCUSDT",
            period="historical",
            strategy="forecaster",
            seed=seed,
        )
    coverage = (
        pd.concat(
            {name: ledger.groupby(ledger.index.year).accepted.mean() for name, ledger in ledgers.items()},
            names=["forecast", "year"],
        )
        .rename("coverage")
        .reset_index()
    )
    return write_report(
        output,
        "Bitcoin research",
        results,
        dict(
            bootstrap=pd.DataFrame(intervals),
            seeds=pd.DataFrame(seeds),
            yearly_metrics=pd.DataFrame(years),
            forecast_coverage=coverage,
        ),
        "Hourly BTC/USDT using src.Forecaster; 2021 validation, 2022–2025 historical test. Quarterly training on 10000 hours; "
        "24-hour horizon. Costs per side and execution delays are in the configuration. "
        "Retrospective exposure control uses test-period exposure and is diagnostic only. "
        "The old BTC-selected parameters are transferred to Forecaster without retuning. Random forest targets are recomputed on every run.",
    )


if __name__ == "__main__":
    set_num_threads(4)
    with threadpool_limits(limits=1):
        print(run())
