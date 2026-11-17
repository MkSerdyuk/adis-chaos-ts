import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from adis.motifs import MotifModel
from adis.prediction import ForecastConfig, _spread, _step, forecast


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    files = [root / name for name in ["motifs.py", "clustering.py", "prediction.py"]] + [Path(__file__)]
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def trajectory_predictions(model, contexts, horizon, config, origin_seeds, paths=16):
    contexts = np.atleast_2d(contexts)
    assert len(contexts) == len(origin_seeds)
    assert np.isfinite(contexts).all()
    n, observed = contexts.shape
    histories = np.c_[np.repeat(contexts, paths, axis=0), np.full((n * paths, horizon + 2), np.nan)]
    reliability = np.c_[np.ones((n * paths, observed)), np.zeros((n * paths, horizon + 2))]
    result = np.full((n, horizon), np.nan)
    generators = [np.random.default_rng(seed) for seed in origin_seeds]
    inner = replace(config, aggregation="mfp", selection="cm", perturbation_variance=0)
    unused = np.random.default_rng(0)

    def step(history, weights):
        values, confidence, _, _ = _step(model, history, weights, inner, unused)
        endpoints = values.reshape(n, paths)
        for row, generator in zip(endpoints, generators, strict=True):
            finite = np.isfinite(row)
            row[finite] += generator.normal(0, np.sqrt(config.perturbation_variance), finite.sum())
        return values, confidence

    def spread(values):
        return _spread([(row[np.isfinite(row)], None, None) for row in values.reshape(n, paths)], config)

    for h in range(horizon):
        stop = observed + h
        values, weights = step(histories[:, :stop], reliability[:, :stop])
        endpoints = values.reshape(n, paths)
        counts = np.isfinite(endpoints).sum(axis=1)
        means = np.divide(np.nansum(endpoints, axis=1), counts, out=np.full(n, np.nan), where=counts > 0)
        accept = counts > 0
        if config.selection == "rgdbscan":
            spreads = [spread(values)]
            tentative = histories[:, : stop + 2].copy()
            tentative_weights = reliability[:, : stop + 2].copy()
            tentative[:, stop], tentative_weights[:, stop] = values, weights
            for ahead in [1, 2]:
                next_values, next_weights = step(
                    tentative[:, : stop + ahead], tentative_weights[:, : stop + ahead]
                )
                spreads.append(spread(next_values))
                if ahead == 1:
                    tentative[:, stop + ahead], tentative_weights[:, stop + ahead] = next_values, next_weights
            accept &= ~((spreads[0] < spreads[1]) & (spreads[1] < spreads[2]))
        else:
            assert config.selection == "cm"
        result[accept, h] = means[accept]
        values[np.repeat(~accept, paths)] = np.nan
        weights[np.repeat(~accept, paths)] = 0
        histories[:, stop], reliability[:, stop] = values, weights
    return result


def fit_quarter(frame, quarter, parameters, directory, workers=4):
    boundary = frame.index.searchsorted(quarter)
    length = parameters.get("training_hours", 10000)
    assert boundary >= length
    training = frame.Close.iloc[boundary - length : boundary].to_numpy()
    settings = {name: parameters[name] for name in ["length", "max_gap", "r", "mu"]}
    sources = source_hashes()
    key = fingerprint(dict(quarter=str(quarter), training_hours=length, **settings, source_hashes=sources))
    folder = directory / "models" / key
    folder.mkdir(parents=True, exist_ok=True)
    minimum, scale = float(training.min()), float(np.ptp(training))
    assert scale > 0
    metadata = dict(
        model_id=key,
        quarter=str(quarter),
        fit_start=str(frame.index[boundary - length]),
        fit_end=str(frame.index[boundary - 1] + pd.Timedelta(hours=1)),
        training_hours=length,
        minimum=minimum,
        scale=scale,
        **settings,
        training_sha256=hashlib.sha256(training.astype("<f8").tobytes()).hexdigest(),
        source_hashes=sources,
    )
    if (folder / "motifs.npz").exists():
        stored = json.loads((folder / "metadata.json").read_text())
        assert all(stored[name] == value for name, value in metadata.items())
        model = MotifModel.load(folder / "motifs.npz")
    else:
        started = time.monotonic()
        model = MotifModel(**settings).fit((training - minimum) / scale, workers=workers)
        model.save(folder / "motifs.npz")
        metadata["fit_seconds"] = time.monotonic() - started
        metadata["motifs"] = len(model.motifs)
        (folder / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return model, metadata


def predict_quarter(frame, quarter, end, parameters, directory, workers=4, mode="rgdbscan", seed=42):
    used = frame.loc[frame.index < end, ["Close", "Observed"]].to_numpy(dtype="<f8")
    data_hash = hashlib.sha256(used.tobytes()).hexdigest()
    key = fingerprint(
        dict(
            parameters=parameters,
            mode=mode,
            seed=seed,
            quarter=str(quarter),
            end=str(end),
            data_sha256=data_hash,
            source_hashes=source_hashes(),
        )
    )
    output = directory / "predictions" / f"{key}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        result = pd.read_csv(output, index_col=0, parse_dates=True)
        result.index = pd.to_datetime(result.index, utc=True)
        return result
    model, metadata = fit_quarter(frame, quarter, parameters, directory, workers)
    origins = frame.index[(frame.index >= quarter) & (frame.index < end) & (frame.index.hour == 23)]
    positions = frame.index.get_indexer(origins)
    history_indices = positions[:, None] - np.arange(model.context_size - 1, -1, -1)
    raw_contexts = frame.Close.to_numpy()[history_indices]
    valid = frame.Observed.to_numpy()[history_indices].all(axis=1)
    contexts = (raw_contexts[valid] - metadata["minimum"]) / metadata["scale"]
    config = ForecastConfig(
        aggregation="avg",
        selection=mode,
        epsilon=parameters["epsilon"],
        cluster_eps=parameters["cluster_eps"],
        min_samples=parameters["min_samples"],
        perturbation_variance=parameters["perturbation_variance"],
        seed=seed,
    )
    prediction = np.full((len(origins), 24), np.nan)
    if len(contexts):
        origin_seeds = [np.random.SeedSequence([seed, int(value.timestamp())]) for value in origins[valid]]
        if mode == "fp":
            values = forecast(model, contexts, 24, config)
        else:
            values = trajectory_predictions(
                model, contexts, 24, config, origin_seeds, paths=parameters["paths"]
            )
        prediction[valid] = values * metadata["scale"] + metadata["minimum"]
    result = pd.DataFrame(
        index=origins,
        data={
            "close": frame.Close.iloc[positions].to_numpy(),
            "prediction": prediction[:, -1],
            "accepted": np.isfinite(prediction[:, -1]),
            "accepted_h1": np.isfinite(prediction[:, 0]),
            "missing_context": ~valid,
            "outside_train_range": (
                (raw_contexts < metadata["minimum"])
                | (raw_contexts > metadata["minimum"] + metadata["scale"])
            ).any(axis=1),
            "model_id": metadata["model_id"],
            "fit_start": metadata["fit_start"],
            "fit_end": metadata["fit_end"],
            "seed": seed,
            "mode": mode,
        },
    )
    assert (pd.to_datetime(result.fit_end, utc=True) <= result.index + pd.Timedelta(hours=1)).all()
    result["predicted_return"] = result.prediction / result.close - 1
    result["target"] = np.where(result.accepted & (result.predicted_return > 0.003), 0.95, 0.0)
    result.index.name = "decision_bar"
    result.to_csv(output)
    print(f"predicted {quarter.date()} {mode} {key} accepted={result.accepted.mean():.3f}", flush=True)
    return result


def walk_forward(
    frame, start, end, parameters, directory=Path("results/trading"), workers=4, mode="rgdbscan", seed=42
):
    start, end = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    quarters = pd.date_range(start, end, freq="QS", inclusive="left")
    assert quarters[0] == start
    return pd.concat(
        [
            predict_quarter(
                frame,
                quarter,
                min(quarter + pd.DateOffset(months=3), end),
                parameters,
                directory,
                workers,
                mode,
                seed,
            )
            for quarter in quarters
        ]
    )
