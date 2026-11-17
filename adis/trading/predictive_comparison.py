import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits

from .equity_backtest import audit_result, risk_metrics, run_daily
from .equity_data import ROOT as EQUITY_ROOT
from .equity_data import load_asset
from .equity_experiment import freeze as equity_freeze
from .equity_experiment import save_result, weighted_bootstrap

ROOT = EQUITY_ROOT / "predictive"
PROTOCOL = Path("research/trading/noncrypto/predictive_protocol.json")
PERIODS = {
    "validation": ("2021-01-01", "2022-01-01"),
    "historical": ("2022-01-01", "2026-01-01"),
    "holdout": ("2026-01-01", "2026-11-01"),
}
CANDIDATES = ["random_forest", "gradient_boosting", "ridge", "svr", "sma50_200", "momentum90"]


def forecast_models(frame, decisions, directory):
    output = directory / "forecasts.csv"
    if output.exists():
        return pd.read_csv(output, parse_dates=["decision_session"])
    close = frame.Close
    returns = np.log(close).diff()
    features = pd.DataFrame({f"momentum_{lag}": np.log(close / close.shift(lag)) for lag in [1, 5, 20, 60]})
    for lag in [20, 60]:
        features[f"volatility_{lag}"] = returns.rolling(lag).std()
    features["volume_ratio"] = frame.Volume / frame.Volume.rolling(20).mean()
    labels = close.shift(-5) / close - 1
    models = {
        "random_forest": RandomForestRegressor(
            n_estimators=200, max_depth=6, min_samples_leaf=50, max_features=0.8, random_state=42, n_jobs=4
        ),
        "gradient_boosting": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=1,
            early_stopping=False,
            random_state=42,
        ),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10)),
        "svr": make_pipeline(StandardScaler(), SVR(C=1, epsilon=0.01, gamma="scale")),
    }
    chunks = []
    for quarter in pd.date_range("2021-01-01", "2026-10-01", freq="QS"):
        boundary = frame.index.searchsorted(quarter)
        rows = np.arange(boundary - 2520, boundary - 5)
        assert frame.index[rows[-1] + 5] < quarter
        dates = decisions[(decisions >= quarter) & (decisions < quarter + pd.DateOffset(months=3))]
        assert np.isfinite(features.iloc[rows]).all().all() and np.isfinite(labels.iloc[rows]).all()
        for name, specification in models.items():
            model = clone(specification).fit(features.iloc[rows], labels.iloc[rows])
            prediction = model.predict(features.loc[dates])
            chunks.append(
                pd.DataFrame(
                    dict(
                        decision_session=dates,
                        model=name,
                        predicted_return=prediction,
                        target=np.where(prediction > 0.003, 0.95, 0),
                        fit_start=str(frame.index[rows[0]]),
                        last_label_session=str(frame.index[rows[-1] + 5]),
                        quarter=str(quarter),
                    )
                )
            )
        print(directory.name, quarter.date(), "four fixed predictors fitted", flush=True)
    result = pd.concat(chunks, ignore_index=True)
    result.to_csv(output, index=False)
    return result


def portfolio(curves, weights):
    equity = pd.concat({name: curve.equity for name, curve in curves.items()}, axis=1)
    exposure = pd.concat({name: curve.exposure for name, curve in curves.items()}, axis=1)
    assert equity.notna().all().all() and exposure.notna().all().all()
    allocation = pd.Series(weights).reindex(equity.columns)
    assert allocation.notna().all() and np.isclose(allocation.sum(), 1)
    capital = equity @ allocation
    invested_fraction = (equity * exposure) @ allocation / capital
    return pd.DataFrame(dict(equity=capital, exposure=invested_fraction))


def freeze():
    base = equity_freeze()
    files = [PROTOCOL, Path(__file__), EQUITY_ROOT / "frozen.json"]
    files.extend(EQUITY_ROOT / symbol / "targets.csv" for symbol in base["symbols"])
    state = dict(
        noncrypto=base,
        files={
            str(path.resolve().relative_to(Path.cwd().resolve())): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in files
        },
    )
    output = ROOT / "frozen.json"
    ROOT.mkdir(exist_ok=True)
    if output.exists():
        assert json.loads(output.read_text()) == state
    else:
        output.write_text(json.dumps(state, indent=2) + "\n")
    return state


def run():
    state = freeze()
    names = ["partial_prediction", *CANDIDATES, "buy_hold", "causal_control", "paper_control"]
    curves = {(period, name): {} for period in PERIODS for name in names}
    records, audits = [], {}
    all_targets, frames = {}, {}
    for symbol in state["noncrypto"]["symbols"]:
        directory = ROOT / symbol
        directory.mkdir(exist_ok=True)
        frame = load_asset(symbol)
        saved = pd.read_csv(EQUITY_ROOT / symbol / "targets.csv", index_col=0, parse_dates=True)
        forecasts = forecast_models(frame, saved.index, directory)
        targets = forecasts.pivot(index="decision_session", columns="model", values="target")
        np.testing.assert_array_equal(targets.random_forest, saved.random_forest5)
        for name in [
            "partial_prediction",
            "sma50_200",
            "momentum90",
            "buy_hold",
            "causal_control",
            "paper_control",
        ]:
            targets[name] = saved[name]
        targets.to_csv(directory / "targets.csv")
        all_targets[symbol], frames[symbol] = targets, frame
    weights = {
        symbol: 1 / 15 if category == "equity" else 1 / 3
        for symbol, category in state["noncrypto"]["instruments"].items()
    }
    portfolio_metrics, portfolio_returns = [], {}
    eligibility_hash = None
    for period, (start, end) in PERIODS.items():
        if period != "validation":
            assert hashlib.sha256((ROOT / "eligibility.csv").read_bytes()).hexdigest() == eligibility_hash
        for symbol, targets in all_targets.items():
            frame, directory = frames[symbol], ROOT / symbol
            for name in names:
                target = targets[name]
                if name == "causal_control" and period == "validation":
                    target = target.fillna(0.0)
                result = run_daily(frame, target, start, end, rebalance=name == "causal_control")
                audits[f"{symbol}/{period}/{name}"] = audit_result(frame, result)
                save_result(directory / period / name, result)
                curves[period, name][symbol] = result["curve"]
                records.append(dict(symbol=symbol, period=period, strategy=name, **result["metrics"]))
            print(symbol, period, "predictor backtests verified", flush=True)
        for name in names:
            combined = portfolio(curves[period, name], weights)
            metrics = risk_metrics(combined.equity, combined.exposure)
            destination = ROOT / "portfolios" / period / name
            destination.mkdir(parents=True, exist_ok=True)
            combined.to_csv(destination / "equity.csv")
            returns = combined.equity / combined.equity.shift(1).fillna(100000) - 1
            returns.rename("return").to_csv(destination / "returns.csv")
            portfolio_returns[period, name] = returns
            portfolio_metrics.append(dict(period=period, strategy=name, **metrics))
        if period == "validation":
            validation = pd.DataFrame(portfolio_metrics).set_index("strategy")
            eligibility = pd.DataFrame(
                dict(
                    strategy=CANDIDATES,
                    validation_return=validation.loc[CANDIDATES, "total_return"].to_numpy(),
                    validation_exposure=validation.loc[CANDIDATES, "mean_exposure"].to_numpy(),
                )
            )
            eligibility["eligible"] = (eligibility.validation_return > 0) & (
                eligibility.validation_exposure >= 0.10
            )
            eligibility.to_csv(ROOT / "eligibility.csv", index=False)
            eligibility_hash = hashlib.sha256((ROOT / "eligibility.csv").read_bytes()).hexdigest()
            selection = dict(
                rule="2021 net portfolio return>0 and mean exposure>=10%; all six candidates retained",
                eligibility_sha256=eligibility_hash,
                before_test_backtests=True,
                eligible=eligibility.loc[eligibility.eligible, "strategy"].tolist(),
            )
            selection_path = ROOT / "eligibility_frozen.json"
            if selection_path.exists():
                assert json.loads(selection_path.read_text()) == selection
            else:
                selection_path.write_text(json.dumps(selection, indent=2) + "\n")
            print("Validation profitability flags frozen before test backtests", flush=True)
    metrics = pd.DataFrame(portfolio_metrics)
    metrics.to_csv(ROOT / "portfolio_metrics.csv", index=False)
    pd.DataFrame(records).to_csv(ROOT / "instrument_metrics.csv", index=False)
    intervals = []
    for period in ["historical", "holdout"]:
        method = portfolio_returns[period, "partial_prediction"]
        assert all(portfolio_returns[period, name].index.equals(method.index) for name in CANDIDATES)
        controls = np.array([portfolio_returns[period, name] for name in CANDIDATES])
        for block in [5, 10, 20]:
            interval = weighted_bootstrap(
                np.tile(method, (len(CANDIDATES), 1)),
                controls,
                np.full(len(CANDIDATES), 1 / len(CANDIDATES)),
                block=block,
            )
            for i, name in enumerate(CANDIDATES):
                intervals.append(
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
    pd.DataFrame(intervals).to_csv(ROOT / "comparisons.csv", index=False)
    (ROOT / "audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    assert freeze() == state
    (ROOT / "completed.json").write_text(
        json.dumps(
            dict(
                backtests=len(records),
                portfolios=len(metrics),
                candidates=CANDIDATES,
                eligibility_sha256=eligibility_hash,
                frozen=state,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
