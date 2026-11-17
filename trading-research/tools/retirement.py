import numpy as np
import pandas as pd

from .backtest import summarize


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


def audit_execution(frame, result, notice, costs_bps=15, delay_hours=1):
    trades = result["trades"].copy()
    numeric = ["Size", "EntryPrice", "ExitPrice", "Commission", "PnL"]
    trades[numeric] = trades[numeric].astype(float)
    entry, exit_ = pd.to_datetime(trades.EntryTime, utc=True), pd.to_datetime(trades.ExitTime, utc=True)
    assert (trades.Size > 0).all() and (entry <= exit_).all()
    assert frame.Observed.reindex(entry).all() and frame.Observed.reindex(exit_).all()
    np.testing.assert_allclose(trades.EntryPrice, frame.Open.reindex(entry), rtol=0, atol=1e-8)
    np.testing.assert_allclose(trades.ExitPrice, frame.Open.reindex(exit_), rtol=0, atol=1e-8)
    fees = trades.Size * (trades.EntryPrice + trades.ExitPrice) * (costs_bps / 10000)
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
        cutoff = pd.Timestamp(notice["retirement_decision_bar_utc"]) + pd.Timedelta(hours=delay_hours)
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
