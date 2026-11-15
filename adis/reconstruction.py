import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from numba import set_num_threads
from threadpoolctl import threadpool_limits

from .baselines import run_baselines
from .data import (
    LORENZ_ARCHIVE_SHA256,
    LORENZ_ARCHIVE_URL,
    OPSD_COLUMN,
    OPSD_URL,
    archived_lorenz,
    german_load,
    lorenz,
    test_windows,
    windows,
)
from .diagnostics import eckmann_lyapunov, entropy_complexity
from .evaluation import compare_masks, errors, horizon_metrics
from .motifs import MotifModel
from .prediction import ForecastConfig, _aggregate_batch, trajectory_forecast

PROFILES = ("lorenz_rk4", "lorenz_archive", "germany", "germany_power_statistics")
BASE = ForecastConfig(selection="rgdbscan")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare(root, profile, workers):
    if not (root / "protocol.json").exists():
        (root / "protocol.json").write_text(Path("references/reconstruction_protocol.json").read_text())
    directory = root / profile
    directory.mkdir(parents=True, exist_ok=True)
    if profile.startswith("lorenz"):
        raw = archived_lorenz(13000) if profile == "lorenz_archive" else lorenz(13000)
        train_end = 10000
        source = (
            {"url": LORENZ_ARCHIVE_URL, "sha256": LORENZ_ARCHIVE_SHA256, "discard": 3000}
            if profile == "lorenz_archive"
            else {"generator": "adis.data.lorenz", "discard": 3000, "dt": 0.1, "component": "x"}
        )
    else:
        url, column, path = OPSD_URL, OPSD_COLUMN, Path("data/opsd_2020.csv")
        if profile == "germany_power_statistics":
            url = "https://data.open-power-system-data.org/time_series/2018-06-30/time_series_60min_singleindex.csv"
            column, path = "DE_load_entsoe_power_statistics", Path("data/opsd_2018.csv")
            if not path.exists():
                urlretrieve(url, path)
            expected = "6f2fb8b24614f45377518fb6528dc1925aff456ba5705b79b1d4a4fb698867e7"
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("OPSD 2018 source checksum differs from the audited release")
        elif not path.exists():
            german_load()
        frame = pd.read_csv(path, usecols=["utc_timestamp", column])
        frame.index = pd.to_datetime(frame.pop("utc_timestamp"), utc=True)
        start, count, train_end = (
            ("2014-12-31", 11008, 8008)
            if profile == "germany_power_statistics"
            else ("2015-01-01", 10984, 7984)
        )
        frame = frame.loc[start:].iloc[:count]
        assert len(frame) == count and frame[column].notna().all()
        assert frame.index.equals(pd.date_range(frame.index[0], periods=len(frame), freq="h"))
        raw = frame[column].to_numpy()
        source = {
            "url": url,
            "column": column,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "start": str(frame.index[0]),
            "end": str(frame.index[-1]),
        }
    minimum, scale = float(raw[:train_end].min()), float(np.ptp(raw[:train_end]))
    series = (raw - minimum) / scale
    test_start, fresh_start = train_end + 1000, train_end + 2000
    calibration_end = test_start - 99
    origins = np.arange(train_end - 1, calibration_end - 100, 8)
    np.savez_compressed(directory / "development.npz", series=series[:calibration_end], origins=origins)
    np.savez_compressed(directory / "evaluation.npz", series=series, raw=raw)
    metadata = {
        "profile": profile,
        "source": source,
        "raw_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
        "train": [0, train_end],
        "calibration": [train_end, calibration_end],
        "validation_embargo": 99,
        "test": [test_start, fresh_start],
        "fresh": [fresh_start, len(raw)],
        "normalization": {"minimum": minimum, "range": scale, "fit": "training only"},
        "exact_paper_data_identified": False,
    }
    write_json(directory / "data.json", metadata)
    train = series[:train_end]
    write_json(
        directory / "diagnostics.json",
        {
            "entropy_complexity": entropy_complexity(train),
            "entropy_complexity_method": "Ordinal Shannon entropy and MPR Jensen-Shannon complexity",
            "entropy_complexity_paper_comparable": False,
            "dimension": 5,
            "delay": 1,
            "lyapunov_spectrum": eckmann_lyapunov(
                train, dt=0.1 if profile.startswith("lorenz") else 1
            ).tolist(),
            "eckmann_settings": {
                "embedding": 10,
                "matrix_dimension": 4,
                "min_neighbors": 12,
                "min_separation": 20,
            },
            "status": "training-only diagnostics; paper estimator settings unknown; not used for model selection",
        },
    )
    model = MotifModel().fit(series[:train_end], workers=workers)
    model.save(directory / "motifs.npz")
    write_json(
        directory / "motifs_manifest.json",
        {
            "training_sha256": hashlib.sha256(series[:train_end].tobytes()).hexdigest(),
            "clustering_source_sha256": hashlib.sha256(Path("adis/clustering.py").read_bytes()).hexdigest(),
            "motifs_source_sha256": hashlib.sha256(Path("adis/motifs.py").read_bytes()).hexdigest(),
            "model_sha256": hashlib.sha256((directory / "motifs.npz").read_bytes()).hexdigest(),
            "settings": {"length": 4, "max_gap": 10, "r": 11, "mu": 0.2},
            "centers": len(model.motifs),
        },
    )
    print(profile, "prepared", len(model.motifs), "motifs", flush=True)


def development(directory):
    split = np.load(directory / "development.npz")
    contexts, truth = windows(split["series"], split["origins"], 30, 100)
    return MotifModel.load(directory / "motifs.npz"), contexts, truth


def screen(directory):
    model, contexts, truth = development(directory)
    rows = []
    for epsilon in [0.01, 0.05]:
        candidates = model.candidates(contexts, np.ones_like(contexts), epsilon)
        for radius, minimum in product([0.001, 0.005, 0.01, 0.03], [1, 3, 5, 10]):
            prediction = _aggregate_batch(candidates, 4, radius, minimum)[:, 0]
            rows.append(
                {"epsilon": epsilon, "cluster_eps": radius, "min_samples": minimum}
                | errors(truth[:, 0], prediction)
            )
    scores = pd.DataFrame(rows)
    scores.to_csv(directory / "screen.csv", index=False)
    spatial = [{"epsilon": 0.01, "cluster_eps": 0.01, "min_samples": 3}]
    for epsilon in [0.01, 0.05]:
        eligible = scores[(scores.epsilon == epsilon) & (scores.coverage >= 0.9)]
        if len(eligible):
            best = eligible.sort_values(["rmse", "cluster_eps", "min_samples"]).iloc[0]
            settings = {
                "epsilon": epsilon,
                "cluster_eps": float(best.cluster_eps),
                "min_samples": int(best.min_samples),
            }
            if settings not in spatial:
                spatial.append(settings)
    trials = [
        asdict(replace(BASE, **s, perturbation_variance=v)) for s, v in product(spatial, [0.0, 1e-5, 1e-4])
    ]
    write_json(directory / "trials.json", trials)
    print(directory.name, "screen", scores.sort_values("rmse").head(4).to_dict("records"), flush=True)


def batched_prediction(model, contexts, config, trajectories=16, truth=None, progress=False):
    prediction = np.full((len(contexts), 100), np.nan)
    for start in range(0, len(contexts), 64):
        prediction[start : start + 64] = trajectory_forecast(
            model,
            contexts[start : start + 64],
            100,
            replace(config, seed=config.seed + start),
            trajectories=trajectories,
            truth=None if truth is None else truth[start : start + 64],
        )
        if progress:
            print(config.selection, min(start + 64, len(contexts)), "/", len(contexts), flush=True)
    return prediction


def score(truth, prediction):
    per_horizon = horizon_metrics(truth, prediction, "trajectory")
    return errors(truth, prediction) | {
        "early_coverage": errors(truth[:, :10], prediction[:, :10])["coverage"],
        "late_coverage": errors(truth[:, 49:], prediction[:, 49:])["coverage"],
        "minimum_late_coverage": float(per_horizon.coverage.iloc[49:].min()),
        "macro_rmse": float(per_horizon.rmse.mean()) if per_horizon.rmse.notna().all() else np.inf,
    }


def ranked(scores, profile):
    scores = scores.copy()
    floor = 0.37 if profile.startswith("germany") else 0.16
    scores["coverage_deficit"] = (
        (0.9 - scores.early_coverage).clip(lower=0)
        + (0.5 - scores.coverage).clip(lower=0)
        + (floor - scores.minimum_late_coverage).clip(lower=0)
    )
    scores["feasible"] = (scores.coverage_deficit == 0) & np.isfinite(scores.macro_rmse)
    return scores.sort_values(["coverage_deficit", "macro_rmse", "trial"])


def search(directory):
    model, contexts, truth = development(directory)
    indices = np.linspace(0, len(contexts) - 1, 24, dtype=int)
    trials = json.loads((directory / "trials.json").read_text())
    rows = []
    for index, settings in enumerate(trials):
        started = time.monotonic()
        prediction = batched_prediction(model, contexts[indices], ForecastConfig(**settings))
        np.save(directory / f"coarse_{index}.npy", prediction)
        row = {"trial": index, "seconds": time.monotonic() - started} | score(truth[indices], prediction)
        rows.append(row)
        pd.DataFrame(rows).to_csv(directory / "coarse.csv", index=False)
        print(directory.name, "coarse", index, row, flush=True)


def validate(directory):
    model, contexts, truth = development(directory)
    trials = json.loads((directory / "trials.json").read_text())
    baseline = trials.index(asdict(BASE))
    shortlist = sorted(
        set([baseline, *ranked(pd.read_csv(directory / "coarse.csv"), directory.name).trial[:2]])
    )
    write_json(directory / "shortlist.json", shortlist)
    rows, horizons = [], []
    for index, seed in product(shortlist, [42, 137]):
        config = replace(ForecastConfig(**trials[index]), seed=seed)
        prediction = batched_prediction(model, contexts, config)
        np.save(directory / f"validation_{index}_{seed}.npy", prediction)
        rows.append({"trial": index, "seed": seed} | score(truth, prediction))
        horizons.append(horizon_metrics(truth, prediction, "trajectory").assign(trial=index, seed=seed))
        pd.DataFrame(rows).to_csv(directory / "validation.csv", index=False)
        pd.concat(horizons).to_csv(directory / "validation_horizons.csv", index=False)
        print(directory.name, "validation", index, seed, rows[-1], flush=True)
    scores = (
        pd.DataFrame(rows)
        .groupby("trial")
        .agg(
            macro_rmse=("macro_rmse", "mean"),
            coverage=("coverage", "min"),
            early_coverage=("early_coverage", "min"),
            minimum_late_coverage=("minimum_late_coverage", "min"),
        )
        .reset_index()
    )
    ranking = ranked(scores, directory.name)
    ranking.to_csv(directory / "ranking.csv", index=False)
    eligible = ranking[ranking.feasible]
    selected = int(eligible.iloc[0].trial) if len(eligible) else baseline
    write_json(
        directory / "selection.json",
        {
            "trial": selected,
            "config": trials[selected],
            "baseline_trial": baseline,
            "eligible": bool(len(eligible)),
            "decision_source": "validation.csv",
            "reason": "minimum mean macro RMSE across both seeds subject to coverage constraints"
            if len(eligible)
            else "no feasible candidate; retain baseline without relaxing constraints",
        },
    )


def freeze(root):
    write_json(
        root / "frozen_selection.json",
        {
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "versions": {
                name: importlib.metadata.version(name)
                for name in ["numpy", "scipy", "numba", "pandas", "scikit-learn"]
            },
            "profiles": {
                profile: {
                    "selection": json.loads((root / profile / "selection.json").read_text()),
                    "selection_sha256": hashlib.sha256(
                        (root / profile / "selection.json").read_bytes()
                    ).hexdigest(),
                    "data_sha256": hashlib.sha256((root / profile / "data.json").read_bytes()).hexdigest(),
                }
                for profile in PROFILES
            },
            "protocol_sha256": hashlib.sha256((root / "protocol.json").read_bytes()).hexdigest(),
            "rule": "No changes to selections after first final test evaluation.",
        },
    )


def check_frozen(directory):
    frozen = json.loads((directory.parent / "frozen_selection.json").read_text())
    for filename, field in [("selection.json", "selection_sha256"), ("data.json", "data_sha256")]:
        if (
            hashlib.sha256((directory / filename).read_bytes()).hexdigest()
            != frozen["profiles"][directory.name][field]
        ):
            raise ValueError(f"Frozen input changed: {directory / filename}")
    if (
        hashlib.sha256((directory.parent / "protocol.json").read_bytes()).hexdigest()
        != frozen["protocol_sha256"]
    ):
        raise ValueError("Frozen search protocol changed")


def evaluate(directory):
    check_frozen(directory)
    selection = json.loads((directory / "selection.json").read_text())
    metadata = json.loads((directory / "data.json").read_text())
    model = MotifModel.load(directory / "motifs.npz")
    series = np.load(directory / "evaluation.npz")["series"]
    configurations = {"baseline": BASE, "selected": ForecastConfig(**selection["config"])}
    summaries, metrics, intersections = [], [], []
    for partition in ["test", "fresh"]:
        start, end = metadata[partition]
        origins, contexts, truth, oracle_truth = test_windows(series[:end], start, 30, 100)
        np.savez_compressed(directory / f"{partition}_targets.npz", origins=origins, truth=truth)
        predictions = {}
        for name, config in configurations.items():
            prediction = batched_prediction(model, contexts, config, progress=True)
            predictions[name] = prediction
            np.save(directory / f"{partition}_{name}.npy", prediction)
            oracle = batched_prediction(
                model, contexts, replace(config, selection="ap"), truth=oracle_truth, progress=True
            )
            np.save(directory / f"{partition}_{name}_ap.npy", oracle)
            row = {"partition": partition, "method": name} | score(truth, prediction)
            row["rmse_original_units"] = row["rmse"] * metadata["normalization"]["range"]
            summaries.append(row)
            metrics.append(horizon_metrics(truth, prediction, name, oracle).assign(partition=partition))
            print(directory.name, partition, name, row, flush=True)
        common = np.isfinite(predictions["baseline"]) & np.isfinite(predictions["selected"])
        for name, prediction in predictions.items():
            intersections.append({"partition": partition, "method": name} | errors(truth, prediction, common))
        pd.DataFrame(summaries).to_csv(directory / "summary.csv", index=False)
        pd.concat(metrics).to_csv(directory / "metrics.csv", index=False)
        pd.DataFrame(intersections).to_csv(directory / "intersection.csv", index=False)


def baselines(directory):
    check_frozen(directory)
    metadata = json.loads((directory / "data.json").read_text())
    series = np.load(directory / "evaluation.npz")["series"]
    validation_origins = np.load(directory / "development.npz")["origins"]
    origins = [np.load(directory / f"{partition}_targets.npz")["origins"] for partition in ["test", "fresh"]]
    dates = None
    if directory.name.startswith("germany"):
        dates = pd.date_range(metadata["source"]["start"], periods=len(series) + 100, freq="h")
    predictions, trials = run_baselines(
        series, metadata["train"][1], validation_origins, np.concatenate(origins), 100, dates
    )
    pd.DataFrame(trials).to_csv(directory / "baseline_validation.csv", index=False)
    summaries, matched = [], []
    offset = 0
    for partition, positions in zip(["test", "fresh"], origins, strict=True):
        truth = np.load(directory / f"{partition}_targets.npz")["truth"]
        part = {
            name: prediction[offset : offset + len(positions)] for name, prediction in predictions.items()
        }
        for name, prediction in part.items():
            np.save(directory / f"{partition}_{name}.npy", prediction)
            row = {"partition": partition, "method": name} | errors(truth, prediction)
            row["rmse_original_units"] = row["rmse"] * metadata["normalization"]["range"]
            summaries.append(row)
        selected = np.load(directory / f"{partition}_selected.npy")
        matched.append(
            compare_masks(truth, part | {"selected": selected}, {"selected": selected}).assign(
                partition=partition
            )
        )
        offset += len(positions)
    pd.DataFrame(summaries).to_csv(directory / "baseline_summary.csv", index=False)
    pd.concat(matched).to_csv(directory / "matched_masks.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase", choices=["prepare", "screen", "search", "validate", "freeze", "evaluate", "baselines"]
    )
    parser.add_argument("--profile", choices=[*PROFILES, "all"], default="all")
    parser.add_argument("--output", default="results/reconstruction")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    write_json(
        root / f"{args.phase}_{args.profile}_source.json",
        {
            "source_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(Path("adis").glob("*.py"))
            },
            "uv_lock_sha256": hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest(),
        },
    )
    set_num_threads(args.workers)
    if args.phase == "freeze":
        freeze(root)
        return
    with threadpool_limits(limits=1):
        for profile in PROFILES if args.profile == "all" else [args.profile]:
            if args.phase == "prepare":
                prepare(root, profile, args.workers)
            else:
                {
                    "screen": screen,
                    "search": search,
                    "validate": validate,
                    "evaluate": evaluate,
                    "baselines": baselines,
                }[args.phase](root / profile)


if __name__ == "__main__":
    main()
