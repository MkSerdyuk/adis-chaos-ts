import numpy as np
import pandas as pd

from .backtest import expected_shortfall, summarize


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


def portfolio(curves, weights):
    equity = pd.concat({name: curve.equity for name, curve in curves.items()}, axis=1)
    exposure = pd.concat({name: curve.exposure for name, curve in curves.items()}, axis=1)
    assert equity.notna().all().all() and exposure.notna().all().all()
    allocation = pd.Series(weights).reindex(equity.columns)
    assert allocation.notna().all() and np.isclose(allocation.sum(), 1)
    capital = equity @ allocation
    invested_fraction = (equity * exposure) @ allocation / capital
    return pd.DataFrame(dict(equity=capital, exposure=invested_fraction))
