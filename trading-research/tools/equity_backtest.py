from functools import partial

import numpy as np
import pandas as pd
from backtesting import Strategy
from backtesting.backtesting import _Broker
from backtesting.lib import FractionalBacktest

from .backtest import expected_shortfall

UNIT = 1e-08


class DividendBroker(_Broker):
    def __init__(self, *args, dividend_delay_days=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.dividend_delay_days = dividend_delay_days
        self.dividend_events = []
        self.pending = []

    @property
    def receivables(self):
        return sum((event["amount"] for event in self.pending))

    @property
    def equity(self):
        return super().equity + self.receivables

    @property
    def margin_available(self):
        return max(0.0, super().margin_available - self.receivables)

    def next(self):
        date = self._data.index[-1]
        remaining = []
        for event in self.pending:
            if event["due_date"] <= date:
                self._cash += event["amount"]
                event["paid_date"] = date
            else:
                remaining.append(event)
        self.pending = remaining
        shares = self.position.size * UNIT
        dividend = float(self._data.Dividend[-1])
        if shares and dividend:
            event = dict(
                ex_date=date,
                due_date=date + pd.Timedelta(days=self.dividend_delay_days),
                paid_date=pd.NaT,
                shares=shares,
                dividend_per_share=dividend,
                amount=shares * dividend,
            )
            self.dividend_events.append(event)
            if self.dividend_delay_days:
                self.pending.append(event)
            else:
                self._cash += event["amount"]
                event["paid_date"] = date
        super().next()


class WeeklyPosition(Strategy):
    rebalance = False

    def init(self):
        self.last_target = 0.0
        self.observations = []
        self.order_records = []

    def next(self):
        date = self.data.index[-1]
        value = self.position.size * self.data.Close[-1]
        self.observations.append(
            (date, self.equity, value / self.equity, self._broker.margin_available, self._broker.receivables)
        )
        target = float(self.data.Target[-1])
        if target == self.last_target and (not (self.rebalance and self.data.Event[-1])):
            return
        self.last_target = target
        difference = self.equity * target - value
        if target == 0:
            if self.position:
                self.position.close()
                self.order_records.append((date, target, "close"))
        elif difference > self.equity * 1e-06 and self._broker.margin_available > 0:
            self.buy(size=min(difference / self._broker.margin_available, 0.999999))
            self.order_records.append((date, target, "buy"))
        elif difference < -self.equity * 1e-06 and self.position:
            self.position.close(portion=min(-difference / value, 1))
            self.order_records.append((date, target, "reduce"))


def risk_metrics(equity, exposure, initial_cash=100000):
    returns = equity / equity.shift(1).fillna(initial_cash) - 1
    peak = equity.cummax().clip(lower=initial_cash)
    return dict(
        total_return=float(equity.iloc[-1] / initial_cash - 1),
        cagr=float((equity.iloc[-1] / initial_cash) ** (252 / len(equity)) - 1),
        max_drawdown=float((1 - equity / peak).max()),
        volatility=float(returns.std(ddof=1) * np.sqrt(252)),
        es95=expected_shortfall(returns),
        worst_day=float(returns.min()),
        mean_exposure=float(exposure.mean()),
        final_equity=float(equity.iloc[-1]),
    )


def run_daily(
    frame, targets, start, end, costs_bps=15, delay_sessions=0, dividend_delay_days=0, rebalance=False
):
    index = frame.index[(frame.index >= start) & (frame.index < end)]
    assert len(index) > 2
    first = frame.index.get_loc(index[0])
    market = frame.iloc[first - 2 : frame.index.get_loc(index[-1]) + 1].copy()
    events = targets.loc[(targets.index >= start) & (targets.index < end)].copy()
    assert events.index.is_unique and events.between(0, 0.95).all()
    shifted = frame.index.get_indexer(events.index) + delay_sessions
    valid = shifted < len(frame)
    events = events.iloc[np.flatnonzero(valid)]
    events.index = frame.index[shifted[valid]]
    market["Target"] = (
        events.reindex(events.index.union(market.index)).sort_index().ffill().reindex(market.index).fillna(0)
    )
    market["Event"] = market.index.isin(events.index)
    market.loc[market.index < index[0], "Target"] = 0.0
    market.loc[market.index[-2] :, "Target"] = 0.0
    engine = FractionalBacktest(
        market,
        WeeklyPosition,
        cash=100000,
        commission=costs_bps / 10000,
        margin=1,
        trade_on_close=False,
        finalize_trades=False,
        fractional_unit=UNIT,
    )
    engine._broker = partial(
        DividendBroker, **engine._broker.keywords, dividend_delay_days=dividend_delay_days
    )
    stats = engine.run(rebalance=rebalance)
    strategy = stats["_strategy"]
    assert not strategy.position and (not strategy.orders)
    curve = (
        pd.DataFrame(
            strategy.observations, columns=["session", "equity", "exposure", "available_cash", "receivables"]
        )
        .set_index("session")
        .reindex(index)
    )
    assert np.isfinite(curve.to_numpy()).all()
    assert curve.exposure.between(0, 1.000001).all()
    np.testing.assert_allclose(curve.equity, stats["_equity_curve"].Equity.reindex(index), atol=1e-08)
    trades = stats["_trades"].copy()
    dividends = pd.DataFrame(
        strategy._broker.dividend_events,
        columns=["ex_date", "due_date", "paid_date", "shares", "dividend_per_share", "amount"],
    )
    orders = pd.DataFrame(strategy.order_records, columns=["decision_session", "target", "action"])
    metrics = risk_metrics(curve.equity, curve.exposure)
    metrics.update(
        entries=int(trades.EntryBar.nunique()),
        fees=float(trades.Commission.sum()),
        dividends=float(dividends.amount.sum()),
        final_cash=float(curve.available_cash.iloc[-1]),
        final_receivables=float(curve.receivables.iloc[-1]),
        turnover=float((trades.Size.abs() * (trades.EntryPrice + trades.ExitPrice)).sum() / 100000),
    )
    returns = curve.equity / curve.equity.shift(1).fillna(100000) - 1
    np.testing.assert_allclose(
        100000 + trades.PnL.sum() + dividends.amount.sum(), curve.equity.iloc[-1], rtol=1e-09, atol=1e-05
    )
    return dict(
        metrics=metrics,
        curve=curve,
        returns=returns,
        trades=trades,
        dividends=dividends,
        orders=orders,
        engine=engine,
        stats=stats,
    )


def audit_result(frame, result, costs_bps=15):
    trades = result["trades"]
    entry, exit_ = (pd.to_datetime(trades.EntryTime), pd.to_datetime(trades.ExitTime))
    price_entry, price_exit = (frame.Open.reindex(entry).to_numpy(), frame.Open.reindex(exit_).to_numpy())
    quantity = trades.Size.to_numpy(dtype=float)
    np.testing.assert_allclose(trades.EntryPrice.to_numpy(dtype=float), price_entry, rtol=0, atol=1e-08)
    np.testing.assert_allclose(trades.ExitPrice.to_numpy(dtype=float), price_exit, rtol=0, atol=1e-08)
    fees = quantity * (price_entry + price_exit) * costs_bps / 10000
    np.testing.assert_allclose(fees, trades.Commission.to_numpy(dtype=float), rtol=1e-09, atol=1e-06)
    np.testing.assert_allclose(
        quantity * (price_exit - price_entry) - fees, trades.PnL.to_numpy(dtype=float), rtol=1e-08, atol=1e-05
    )
    orders = result["orders"]
    order_dates = pd.to_datetime(orders.decision_session)
    fills = frame.index[frame.index.get_indexer(order_dates) + 1]
    assert set(entry).issubset(set(fills[orders.action == "buy"]))
    assert set(exit_).issubset(set(fills[orders.action.isin(["close", "reduce"])]))
    for side, dates, prices in [("entry", entry, price_entry), ("exit", exit_, price_exit)]:
        factors = frame.FutureSplitFactor.reindex(dates).to_numpy()
        trades[f"raw_{side}_price"] = prices * factors
        trades[f"raw_{side}_shares"] = quantity / factors
        np.testing.assert_allclose(
            trades[f"raw_{side}_price"] * trades[f"raw_{side}_shares"], quantity * prices, atol=1e-07
        )
    dividends = result["dividends"]
    for row in dividends.itertuples():
        date = pd.Timestamp(row.ex_date)
        shares = quantity[(entry < date) & (exit_ >= date)].sum()
        np.testing.assert_allclose(shares, row.shares, rtol=1e-09, atol=1e-08)
        np.testing.assert_allclose(row.amount, shares * frame.loc[date, "Dividend"], rtol=1e-09, atol=1e-08)
    for date in frame.index[
        (frame.index >= result["curve"].index[0])
        & (frame.index <= result["curve"].index[-1])
        & (frame.Dividend > 0)
    ]:
        shares = quantity[(entry < date) & (exit_ >= date)].sum()
        amount = dividends.loc[pd.to_datetime(dividends.ex_date) == date, "amount"].sum()
        np.testing.assert_allclose(amount, shares * frame.loc[date, "Dividend"], rtol=1e-09, atol=1e-08)
    equity = result["curve"].equity.to_numpy()
    returns = equity / np.r_[100000, equity[:-1]] - 1
    es = max(0, -np.sort(returns)[: int(np.ceil(0.05 * len(returns)))].mean())
    np.testing.assert_allclose(es, result["metrics"]["es95"], atol=1e-12)
    np.testing.assert_allclose(
        100000 + trades.PnL.sum() + dividends.amount.sum(), equity[-1], rtol=1e-09, atol=1e-05
    )
    np.testing.assert_allclose(
        result["metrics"]["final_cash"] + result["metrics"]["final_receivables"], equity[-1], atol=1e-07
    )
    return dict(
        trade_records=len(trades),
        dividend_records=len(dividends),
        fills_fees_pnl_verified=True,
        entitlement_and_cash_verified=True,
        raw_adjusted_value_equivalence_verified=True,
    )
