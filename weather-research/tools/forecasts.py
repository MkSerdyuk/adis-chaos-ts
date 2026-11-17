import hashlib
import importlib.metadata
import json

import joblib
import numpy as np

from src import Forecaster

from .data import ROOT, sha256


def fingerprint(state):
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()[:20]


def core_identity():
    return dict(
        files={path.name: sha256(path) for path in sorted((ROOT.parent / "src").glob("*.py"))},
        versions={
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scipy", "numba", "scikit-learn"]
        },
    )


def predict(frame, quarter, origins, parameters, training, horizon):
    boundary = frame.index.searchsorted(quarter)
    assert boundary >= training
    raw = frame.ModelTemperature.iloc[boundary - training : boundary].to_numpy()
    minimum, scale = float(raw.min()), float(np.ptp(raw))
    assert scale > 0 and np.isfinite(raw).all()
    train = (raw - minimum) / scale
    geometry = {name: parameters[name] for name in ["length", "max_gap", "r", "mu", "clustered", "patterns"]}
    model_state = dict(
        geometry=geometry,
        normalized_train_sha256=hashlib.sha256(train.astype("<f8").tobytes()).hexdigest(),
        core=core_identity(),
    )
    model_id = fingerprint(model_state)
    directory = ROOT / "results/cache/models"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{model_id}.joblib"
    if path.exists():
        model = joblib.load(path).set_params(**parameters)
    else:
        model = Forecaster(**parameters).fit(train, workers=4)
        temporary = path.with_suffix(".tmp")
        joblib.dump(model, temporary)
        temporary.replace(path)
    assert type(model) is Forecaster
    positions = frame.index.get_indexer(origins)
    assert (positions >= 0).all()
    indices = positions[:, None] - np.arange(model.context_size_ - 1, -1, -1)
    assert (indices >= 0).all()
    contexts = (frame.ModelTemperature.to_numpy()[indices] - minimum) / scale
    valid = frame.Eligible.reindex(origins).to_numpy()
    state = dict(
        model=model_state,
        minimum=minimum,
        scale=scale,
        parameters=model.get_params(),
        horizon=horizon,
        normalized_context_sha256=hashlib.sha256(contexts.astype("<f8").tobytes()).hexdigest(),
        valid=valid.tolist(),
        origins=origins.astype(str).tolist(),
        adapter=sha256(ROOT / "tools/forecasts.py"),
    )
    directory = ROOT / "results/cache/predictions"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{fingerprint(state)}.npz"
    if path.exists():
        assert json.loads(path.with_suffix(".json").read_text()) == state
        values = np.load(path)["values"]
        assert values.shape == (len(origins), horizon)
    else:
        values = np.full((len(origins), horizon), np.nan)
        if valid.any():
            ids = np.array([int(date.timestamp()) for date in origins[valid]], dtype=np.int64)
            values[valid] = model.predict(contexts[valid], horizon=horizon, origin_ids=ids) * scale + minimum
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, values=values)
        path.with_suffix(".json").write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(path)
    fit = dict(
        model_id=model_id,
        fit_start=str(frame.index[boundary - training]),
        fit_end_exclusive=str(quarter),
        last_training_observation=str(frame.index[boundary - 1]),
        training_hours=training,
        imputed_training_hours=int((~frame.Observed.iloc[boundary - training : boundary]).sum()),
        minimum=minimum,
        scale=scale,
        parameters=model.get_params(),
    )
    return values, fit
