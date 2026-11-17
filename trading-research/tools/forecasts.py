import hashlib
import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import Forecaster

from .data import ROOT, sha256, weekly_decisions


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:20]


def implementation():
    files = [Path(__file__), ROOT / "tools/data.py", *sorted((ROOT.parent / "src").glob("*.py"))]
    return dict(
        files={str(path.relative_to(ROOT.parent)): sha256(path) for path in files},
        versions={
            name: importlib.metadata.version(name) for name in ["numpy", "scipy", "numba", "scikit-learn"]
        },
    )


def predict_window(frame, quarter, dates, parameters, training, horizon, hourly):
    boundary = frame.index.searchsorted(quarter)
    assert boundary >= training and len(dates) > 0
    train = frame.Close.iloc[boundary - training : boundary].to_numpy()
    minimum, scale = float(train.min()), float(np.ptp(train))
    assert scale > 0 and np.isfinite(train).all()
    geometry = {name: parameters[name] for name in ["length", "max_gap", "r", "mu", "clustered", "patterns"]}
    model_state = dict(
        geometry=geometry,
        train_sha256=hashlib.sha256(train.astype("<f8").tobytes()).hexdigest(),
        implementation=implementation(),
    )
    model_key = fingerprint(model_state)
    cache = ROOT / "results/cache"
    models = cache / "models"
    models.mkdir(parents=True, exist_ok=True)
    model_path = models / f"{model_key}.joblib"
    if model_path.exists():
        model = joblib.load(model_path)
        model.set_params(**parameters)
    else:
        model = Forecaster(**parameters).fit((train - minimum) / scale, workers=4)
        joblib.dump(model, model_path)
    assert type(model) is Forecaster
    positions = frame.index.get_indexer(dates)
    indices = positions[:, None] - np.arange(model.context_size_ - 1, -1, -1)
    assert (positions >= 0).all() and (indices >= 0).all()
    contexts = frame.Close.to_numpy()[indices]
    valid = frame.Observed.to_numpy()[indices].all(axis=1) if hourly else np.ones(len(dates), dtype=bool)
    state = dict(
        model=model_state,
        parameters=parameters,
        horizon=horizon,
        contexts_sha256=hashlib.sha256(contexts.astype("<f8").tobytes()).hexdigest(),
        observed=valid.tolist(),
        origins=dates.astype(str).tolist(),
    )
    key = fingerprint(state)
    predictions = cache / "predictions"
    predictions.mkdir(parents=True, exist_ok=True)
    path = predictions / f"{key}.csv"
    if path.exists():
        assert json.loads(path.with_suffix(".json").read_text()) == state
        ledger = pd.read_csv(path, index_col=0, parse_dates=True)
        assert ledger.index.equals(dates)
        return ledger
    values = np.full((len(dates), horizon), np.nan)
    if valid.any():
        origin_ids = np.array([int(date.timestamp()) for date in dates[valid]], dtype=np.int64)
        values[valid] = (
            model.predict((contexts[valid] - minimum) / scale, horizon=horizon, origin_ids=origin_ids) * scale
            + minimum
        )
    ledger = pd.DataFrame(
        dict(
            close=frame.Close.iloc[positions].to_numpy(),
            prediction=values[:, -1],
            accepted=np.isfinite(values[:, -1]),
            accepted_h1=np.isfinite(values[:, 0]),
            missing_context=~valid,
            outside_train_range=((contexts < minimum) | (contexts > minimum + scale)).any(axis=1),
            model_id=model_key,
            fit_start=str(frame.index[boundary - training]),
            fit_end_exclusive=str(quarter),
            last_training_observation=str(frame.index[boundary - 1]),
            seed=parameters["seed"],
            predictor="src.Forecaster",
        ),
        index=dates,
    )
    ledger["predicted_return"] = ledger.prediction / ledger.close - 1
    ledger["target"] = np.where(ledger.accepted & (ledger.predicted_return > 0.003), 0.95, 0.0)
    ledger.index.name = "decision_bar" if hourly else "decision_session"
    temporary = path.with_suffix(".tmp")
    ledger.to_csv(temporary)
    path.with_suffix(".json").write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)
    return ledger


def walk_forward(frame, start, end, parameters, training=10000, horizon=24, hourly=True):
    timezone = "UTC" if hourly else None
    start, end = pd.Timestamp(start, tz=timezone), pd.Timestamp(end, tz=timezone)
    origins = frame.index[frame.index.hour == 23] if hourly else pd.DatetimeIndex(weekly_decisions())
    origins = origins[(origins >= start) & (origins < end)]
    quarters = pd.date_range(start, end, freq="QS", inclusive="left")
    assert len(quarters) and quarters[0] == start
    chunks = []
    for quarter in quarters:
        dates = origins[(origins >= quarter) & (origins < quarter + pd.DateOffset(months=3))]
        if len(dates):
            chunks.append(predict_window(frame, quarter, dates, parameters, training, horizon, hourly))
    return pd.concat(chunks)
