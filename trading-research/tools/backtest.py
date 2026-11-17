import numpy as np
import pandas as pd
from backtesting import Strategy
from backtesting.lib import FractionalBacktest


class TargetPosition(Strategy):
    rebalance_daily = False

    def init(self):
        self.last_target = 0.0
        self.exposure = []
        self.orders_log = []

    def next(self):
        price = self.data.Close[-1]
        current = self.position.size * price
        self.exposure.append((self.data.index[-1], current / self.equity))
        target = float(self.data.Target[-1])
        if target == self.last_target and (not (self.rebalance_daily and self.data.Rebalance[-1])):
            return
        assert 0 <= target <= 0.95
        self.last_target = target
        difference = self.equity * target - current
        if target == 0:
            if self.position:
                self.position.close()
                self.orders_log.append((self.data.index[-1], target, "close"))
        elif difference > self.equity * 1e-06:
            available = self.equity - current
            fraction = min(difference / available, 0.999999)
            self.buy(size=fraction)
            self.orders_log.append((self.data.index[-1], target, "buy"))
        elif difference < -self.equity * 1e-06 and self.position:
            self.position.close(portion=min(-difference / current, 1))
            self.orders_log.append((self.data.index[-1], target, "reduce"))


def daily_returns(equity, initial_cash):
    daily = equity.resample("1D").last()
    returns = daily.pct_change()
    returns.iloc[0] = daily.iloc[0] / initial_cash - 1
    return returns


def expected_shortfall(returns, level=0.95):
    returns = np.asarray(returns, float)
    count = max(1, int(np.ceil(round(1 - level, 12) * len(returns))))
    return max(0.0, -float(np.sort(returns)[:count].mean()))


def summarize(equity, exposure, trades, initial_cash, start, end):
    returns = daily_returns(equity, initial_cash)
    peak = equity.cummax().clip(lower=initial_cash)
    years = (end - start).total_seconds() / (365.25 * 86400)
    turnover = float((trades.Size.abs() * (trades.EntryPrice + trades.ExitPrice)).sum() / initial_cash)
    return dict(
        total_return=equity.iloc[-1] / initial_cash - 1,
        cagr=(equity.iloc[-1] / initial_cash) ** (1 / years) - 1,
        max_drawdown=float((1 - equity / peak).max()),
        daily_volatility=float(returns.std(ddof=1) * np.sqrt(365)),
        es95=expected_shortfall(returns),
        worst_day=float(returns.min()),
        downside_deviation=float(np.sqrt(np.mean(np.minimum(returns, 0) ** 2) * 365)),
        mean_exposure=float(exposure.mean()),
        time_in_market=float((exposure > 1e-06).mean()),
        turnover_initial_capital=turnover,
        entries=int(trades.EntryBar.nunique()),
        fees=float(trades.Commission.sum()),
        final_equity=float(equity.iloc[-1]),
    )


def run_backtest(
    frame, targets, start, end, costs_bps=15, delay_hours=1, initial_cash=100000, rebalance_daily=False
):
    start, end = (pd.Timestamp(start), pd.Timestamp(end))
    start = start.tz_localize("UTC") if start.tzinfo is None else start
    end = end.tz_localize("UTC") if end.tzinfo is None else end
    events = targets.loc[(targets.index >= start) & (targets.index < end)].copy().astype(float)
    assert events.index.is_unique and events.between(0, 0.95).all()
    events.index += pd.Timedelta(hours=delay_hours)
    hourly = frame.loc[(frame.index >= start - pd.Timedelta(hours=2)) & (frame.index < end)].copy()
    target_grid = events.reindex(events.index.union(hourly.index)).sort_index().ffill().reindex(hourly.index)
    hourly["Target"] = target_grid.fillna(0)
    event_time = pd.Series(events.index, index=events.index)
    hourly["EventTime"] = (
        event_time.reindex(event_time.index.union(hourly.index)).ffill().reindex(hourly.index)
    )
    hourly.loc[hourly.index < start, "Target"] = 0.0
    observed = hourly.loc[hourly.Observed].copy()
    observed["Rebalance"] = observed.EventTime.notna() & observed.EventTime.ne(observed.EventTime.shift())
    observed = observed.drop(columns="EventTime")
    observed.loc[observed.index[-2] :, "Target"] = 0.0
    engine = FractionalBacktest(
        observed,
        TargetPosition,
        cash=initial_cash,
        commission=costs_bps / 10000,
        spread=0,
        margin=1,
        trade_on_close=False,
        hedging=False,
        exclusive_orders=False,
        finalize_trades=False,
        fractional_unit=1e-08,
    )
    stats = engine.run(rebalance_daily=rebalance_daily)
    strategy = stats["_strategy"]
    assert not strategy.position and (not strategy.orders)
    index = hourly.index[hourly.index >= start]
    equity = stats["_equity_curve"].Equity.reindex(index).ffill()
    exposure = pd.Series(dict(strategy.exposure)).reindex(index).ffill().fillna(0)
    assert equity.notna().all() and (equity > 0).all()
    assert exposure.between(0, 1.000001).all()
    trades = stats["_trades"]
    metrics = summarize(equity, exposure, trades, initial_cash, start, end)
    orders = pd.DataFrame(strategy.orders_log, columns=["decision_bar", "target", "action"])
    return dict(
        metrics=metrics,
        equity=equity,
        exposure=exposure,
        trades=trades,
        orders=orders,
        daily_returns=daily_returns(equity, initial_cash),
        engine=engine,
        stats=stats,
    )
