import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from adis.motifs import MotifModel
from adis.prediction import ForecastConfig

from .equity_data import weekly_decisions
from .forecasts import fingerprint, source_hashes, trajectory_predictions


def predict_daily(frame, parameters, directory, seed=42):
    sources = source_hashes()
    sources["equity_forecasts.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    origins = pd.DatetimeIndex(weekly_decisions())
    origins = origins[(origins >= "2021-01-01") & (origins < "2026-11-01")]
    chunks = []
    for quarter in pd.date_range("2021-01-01", "2026-10-01", freq="QS"):
        end = min(quarter + pd.DateOffset(months=3), pd.Timestamp("2026-11-01"))
        dates = origins[(origins >= quarter) & (origins < end)]
        boundary = frame.index.searchsorted(quarter)
        train = frame.Close.iloc[boundary - 2520 : boundary].to_numpy()
        assert len(train) == 2520 and np.isfinite(train).all()
        training_hash = hashlib.sha256(train.astype("<f8").tobytes()).hexdigest()
        settings = {name: parameters[name] for name in ["length", "max_gap", "r", "mu"]}
        model_id = fingerprint(
            dict(quarter=str(quarter), settings=settings, train_hash=training_hash, sources=sources)
        )
        metadata = dict(
            model_id=model_id,
            fit_start=str(frame.index[boundary - 2520]),
            fit_end_exclusive=str(quarter),
            last_training_session=str(frame.index[boundary - 1]),
            training_sessions=2520,
            training_sha256=training_hash,
            minimum=float(train.min()),
            scale=float(np.ptp(train)),
            settings=settings,
            source_hashes=sources,
        )
        assert metadata["scale"] > 0
        positions = frame.index.get_indexer(dates)
        width = (parameters["length"] - 1) * parameters["max_gap"]
        contexts = frame.Close.to_numpy()[positions[:, None] - np.arange(width - 1, -1, -1)]
        key = fingerprint(
            dict(
                model_id=model_id,
                parameters=parameters,
                seed=seed,
                contexts_sha256=hashlib.sha256(contexts.astype("<f8").tobytes()).hexdigest(),
                origins=dates.astype(str).tolist(),
            )
        )
        destination = directory / "predictions" / f"{key}.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata_path = directory / "metadata" / f"{model_id}.json"
        metadata_path.parent.mkdir(exist_ok=True)
        if destination.exists():
            assert json.loads(metadata_path.read_text())["training_sha256"] == training_hash
            chunks.append(pd.read_csv(destination, index_col=0, parse_dates=True))
            continue
        model_file = directory / "models" / f"{model_id}.npz"
        model_file.parent.mkdir(exist_ok=True)
        if model_file.exists():
            stored = json.loads(metadata_path.read_text())
            assert all(stored[key] == value for key, value in metadata.items())
            model = MotifModel.load(model_file)
        else:
            started = time.monotonic()
            model = MotifModel(**settings).fit((train - metadata["minimum"]) / metadata["scale"], workers=4)
            model.save(model_file)
            metadata.update(fit_seconds=time.monotonic() - started, motifs=len(model.motifs))
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        config = ForecastConfig(
            aggregation="avg",
            selection="rgdbscan",
            epsilon=parameters["epsilon"],
            cluster_eps=parameters["cluster_eps"],
            min_samples=parameters["min_samples"],
            perturbation_variance=parameters["perturbation_variance"],
            seed=seed,
        )
        normalized = (contexts - metadata["minimum"]) / metadata["scale"]
        seeds = [np.random.SeedSequence([seed, int(date.timestamp())]) for date in dates]
        predictions = (
            trajectory_predictions(model, normalized, 5, config, seeds, paths=parameters["paths"])
            * metadata["scale"]
            + metadata["minimum"]
        )
        ledger = pd.DataFrame(
            dict(
                close=frame.Close.reindex(dates),
                prediction=predictions[:, -1],
                accepted=np.isfinite(predictions[:, -1]),
                accepted_h1=np.isfinite(predictions[:, 0]),
                outside_train_range=((normalized < 0) | (normalized > 1)).any(axis=1),
                model_id=model_id,
                fit_end_exclusive=str(quarter),
                seed=seed,
            ),
            index=dates,
        )
        ledger["predicted_return"] = ledger.prediction / ledger.close - 1
        ledger["target"] = np.where(ledger.accepted & (ledger.predicted_return > 0.003), 0.95, 0.0)
        ledger.index.name = "decision_session"
        ledger.to_csv(destination)
        chunks.append(ledger)
        print(
            directory.name,
            quarter.date(),
            "L",
            parameters["length"],
            "K",
            parameters["max_gap"],
            "seed",
            seed,
            flush=True,
        )
    return pd.concat(chunks)


def technical_targets(frame, decisions):
    close = frame.Close
    change = close.diff()
    up = change.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    down = -change.clip(upper=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rsi = 100 - 100 / (1 + up / down)
    rsi_state = pd.Series(np.where(rsi < 30, 1.0, np.where(rsi > 50, 0.0, np.nan)), index=frame.index)
    breakout = pd.Series(
        np.where(
            close > frame.High.rolling(20).max().shift(1),
            1.0,
            np.where(close < frame.Low.rolling(10).min().shift(1), 0.0, np.nan),
        ),
        index=frame.index,
    )
    volatility = np.log(close).diff().rolling(30).std() * np.sqrt(252)
    targets = pd.DataFrame(
        dict(
            cash=0.0,
            buy_hold=0.95,
            sma50_200=0.95 * (close.rolling(50).mean() > close.rolling(200).mean()),
            momentum90=0.95 * (close.pct_change(90) > 0),
            donchian20_10=0.95 * breakout.ffill().fillna(0),
            rsi14=0.95 * rsi_state.ffill().fillna(0),
            vol_target20pct_30=(0.20 / volatility).clip(0, 0.95).fillna(0),
        ),
        index=frame.index,
    )
    return targets.reindex(decisions)


def forest_targets(frame, decisions):
    close = frame.Close
    returns = np.log(close).diff()
    features = pd.DataFrame({f"momentum_{lag}": np.log(close / close.shift(lag)) for lag in [1, 5, 20, 60]})
    for lag in [20, 60]:
        features[f"volatility_{lag}"] = returns.rolling(lag).std()
    features["volume_ratio"] = frame.Volume / frame.Volume.rolling(20).mean()
    labels = close.shift(-5) / close - 1
    predictions, fits = [], []
    for quarter in pd.date_range("2021-01-01", "2026-10-01", freq="QS"):
        boundary = frame.index.searchsorted(quarter)
        rows = np.arange(boundary - 2520, boundary - 5)
        assert frame.index[rows[-1] + 5] < quarter
        dates = decisions[(decisions >= quarter) & (decisions < quarter + pd.DateOffset(months=3))]
        model = RandomForestRegressor(
            n_estimators=200, max_depth=6, min_samples_leaf=50, max_features=0.8, random_state=42, n_jobs=4
        )
        model.fit(features.iloc[rows], labels.iloc[rows])
        predictions.append(
            pd.Series(np.where(model.predict(features.loc[dates]) > 0.003, 0.95, 0), index=dates)
        )
        fits.append(
            dict(
                quarter=str(quarter),
                fit_start=str(frame.index[rows[0]]),
                last_label_session=str(frame.index[rows[-1] + 5]),
            )
        )
    return pd.concat(predictions), fits
