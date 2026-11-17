import json

import numpy as np
import pandas as pd
from numba import set_num_threads
from threadpoolctl import threadpool_limits
from tools.backtest import run_backtest
from tools.baselines import technical_targets
from tools.data import DATA, TUNED, load_market, start_run
from tools.inputs import hourly_forecast
from tools.report import record, write_report
from tools.retirement import audit_execution, retire_targets, retired_cash
from tools.statistics import synchronized_bootstrap, yearly_metrics

CONFIG = dict(
    symbols=["ETHUSDT", "BNBUSDT", "XRPUSDT", "EOSUSDT", "LTCUSDT"],
    parameters=TUNED.copy(),
    costs_bps=15,
    delay_hours=1,
    periods=dict(historical=("2022-01-01", "2026-01-01"), holdout=("2026-01-01", "2026-11-01")),
)


def run(config=CONFIG):
    output = start_run("crypto", config)
    retirement = json.loads((DATA / "protocols" / "transfer_retirement.json").read_text())
    results, years, coverage, audits = [], [], [], {}
    returns = {period: {} for period in config["periods"]}
    for symbol in config["symbols"]:
        frame = load_market(symbol)
        notice = retirement.get(symbol)
        end = notice["retirement_decision_bar_utc"] if notice else "2026-11-01"
        end = pd.Timestamp(end).strftime("%Y-%m-%d %H:%M:%S")
        ledger = hourly_forecast(
            frame,
            symbol,
            "main",
            config["parameters"],
            output,
            "2021-01-01",
            end,
        )
        origins = frame.index[(frame.index >= "2021-01-01") & (frame.index.hour == 23)]
        target = retire_targets(ledger.target.reindex(origins, fill_value=0.0), notice)
        control = retire_targets(target.shift(1).rolling(90, min_periods=90).mean(), notice)
        technical = technical_targets(frame)
        strategies = {name: retire_targets(technical[name], notice) for name in technical}
        strategies.update(forecaster=target, causal_exposure_control=control)
        assert (
            pd.to_datetime(ledger.fit_end_exclusive, utc=True) <= ledger.index + pd.Timedelta(hours=1)
        ).all()
        for period, (start, stop) in config["periods"].items():
            inactive = bool(notice and pd.Timestamp(end, tz="UTC") < pd.Timestamp(start, tz="UTC"))
            subset = ledger.loc[(ledger.index >= start) & (ledger.index < stop)]
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
                assert targets.loc[(targets.index >= start) & (targets.index < stop)].notna().all()
                result = (
                    retired_cash(frame, start, stop)
                    if inactive
                    else run_backtest(
                        frame,
                        targets,
                        start,
                        stop,
                        costs_bps=config["costs_bps"],
                        delay_hours=config["delay_hours"],
                        rebalance_daily=name in {"causal_exposure_control", "vol_target_20pct_30daily"},
                    )
                )
                audits[f"{symbol}/{period}/{name}"] = audit_execution(
                    frame, result, notice, config["costs_bps"], config["delay_hours"]
                )
                years.extend(
                    dict(symbol=symbol, period=period, strategy=name, **row)
                    for row in yearly_metrics(result, config["costs_bps"])
                )
                saved = record(
                    results,
                    result,
                    scenario="main",
                    symbol=symbol,
                    period=period,
                    strategy=name,
                    retired_slot=inactive,
                )
                if name in {"forecaster", "causal_exposure_control"}:
                    returns[period][symbol, name] = saved["returns"]
    intervals = []
    for period, values in returns.items():
        calendar = next(iter(values.values())).index
        assert all(series.index.equals(calendar) for series in values.values())
        method = np.array([values[symbol, "forecaster"] for symbol in config["symbols"]])
        control = np.array([values[symbol, "causal_exposure_control"] for symbol in config["symbols"]])
        for block in [7, 14, 28]:
            intervals.append(
                dict(
                    period=period,
                    primary=period == "holdout" and block == 14,
                    symbols=config["symbols"],
                    **synchronized_bootstrap(method, control, blocks=block),
                )
            )
    (output / "audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    return write_report(
        output,
        "Crypto transfer research",
        results,
        dict(
            bootstrap=pd.DataFrame(intervals),
            yearly_metrics=pd.DataFrame(years),
            coverage=pd.DataFrame(coverage),
        ),
        "Fixed five-asset panel selected by 2019 volume. The legacy BTC-selected parameters are transferred to src.Forecaster without retuning. "
        "2022–2025 historical test; January–October 2026 historical holdout. EOS exits under the dated "
        "announcement rule and remains a cash slot. Bootstrap estimates mean individual-asset ES95, not portfolio ES95.",
    )


if __name__ == "__main__":
    set_num_threads(4)
    with threadpool_limits(limits=1):
        print(run())
