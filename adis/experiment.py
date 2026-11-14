import argparse
import hashlib
import importlib.metadata
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import set_num_threads
from threadpoolctl import threadpool_limits

from . import ForecastConfig, MotifModel, forecast
from .baselines import run_baselines
from .data import german_load, lorenz, windows
from .diagnostics import eckmann_lyapunov, entropy_complexity, predictability_horizon
from .evaluation import compare_masks, errors, horizon_metrics
from .prediction import trajectory_forecast
from .selection import fit_selectors


def _plot(directory, metrics):
    names = [
        "cl_avg_cm",
        "cl_avg_ap",
        "cl_avg_lr",
        "cl_avg_adblr",
        "cl_avg_rgdbscan",
        "trj_trjp_rgdbscan",
        "seasonal_168",
        "ridge",
        "random_forest",
        "svr_recursive",
        "dlinear_ridge",
        "hist_gradient_boosting_recursive",
    ]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    colors = plt.get_cmap("tab20")
    for index, name in enumerate(names):
        sample = metrics[metrics.method == name]
        if len(sample):
            axes[0].plot(sample.horizon, sample.rmse, label=name, color=colors(index))
            axes[1].plot(sample.horizon, sample.coverage, label=name, color=colors(index))
    axes[0].set_ylabel("RMSE, training range units")
    axes[1].set_ylabel("Coverage")
    axes[1].set_xlabel("Forecast horizon (steps)")
    axes[1].set_ylim(0, 1.02)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", fontsize=8, ncol=3)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(directory / "horizons.png", dpi=160)
    fig.savefig(directory / "horizons.pdf")
    plt.close(fig)


def _published_comparison(directory, dataset, metrics):
    published = pd.read_csv("references/published_tables.csv")
    published = published[published.dataset == dataset]
    joined = published.merge(metrics, on=["method", "horizon"], suffixes=("_paper", "_measured"))
    rows = []
    for _, row in joined.iterrows():
        for metric in ["np_percent", "mape", "rmse", "paper_precision", "paper_recall", "paper_f1"]:
            expected, measured = row[f"{metric}_paper"], row[f"{metric}_measured"]
            tolerance = 0.5 if metric == "np_percent" else 0.005
            comparable = pd.notna(expected) and pd.notna(measured)
            status = (
                "comparable_reconstruction"
                if comparable
                else "paper_undefined"
                if pd.isna(expected)
                else "measured_undefined"
            )
            rows.append(
                {
                    "dataset": dataset,
                    "method": row.method,
                    "horizon": row.horizon,
                    "metric": metric,
                    "published": expected,
                    "measured": measured,
                    "absolute_difference": abs(expected - measured) if comparable else np.nan,
                    "rounding_tolerance": tolerance,
                    "within_rounding": abs(expected - measured) <= tolerance if comparable else None,
                    "status": status,
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(directory / "paper_comparison.csv", index=False)
    return result


def run(dataset, args):
    started = time.monotonic()
    directory = Path(args.output) / dataset
    directory.mkdir(parents=True, exist_ok=True)
    timestamps = None
    if dataset == "lorenz":
        raw = lorenz(12000)
        train_end, validation_end = 10000, 11000
    else:
        raw, timestamps = german_load()
        train_end, validation_end = len(raw) - 2000, len(raw) - 1000
    minimum, scale = raw[:train_end].min(), np.ptp(raw[:train_end])
    series = (raw - minimum) / scale
    validation_origins = np.arange(train_end - 1, validation_end - args.horizon, args.validation_stride)
    test_origins = np.arange(validation_end - 1, len(series) - args.horizon, args.test_stride)
    config = ForecastConfig(seed=args.seed)
    metadata = {
        "dataset": dataset,
        "data_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
        "train": [0, train_end],
        "validation": [train_end, validation_end],
        "test": [validation_end, len(series)],
        "intervals": "zero-based, end-exclusive",
        "normalization": {"minimum": minimum, "range": scale, "fit": "training only"},
        "horizon": args.horizon,
        "test_stride": args.test_stride,
        "validation_stride": args.validation_stride,
        "test_origins": len(test_origins),
        "validation_origins": len(validation_origins),
        "forecast_config": asdict(config),
        "motifs": {"length": 4, "max_gap": 10, "patterns": 1000, "r": 11, "mu": 0.2},
        "trajectories": args.trajectories,
        "seed": args.seed,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "scipy", "pandas", "numba", "scikit-learn", "imbalanced-learn"]
        },
        "exact_paper_replication": False,
        "execution_source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path("adis").glob("*.py"))
        },
    }
    if timestamps is not None:
        metadata["dates"] = {
            "train_start": str(timestamps[0]),
            "train_end": str(timestamps[train_end - 1]),
            "validation_start": str(timestamps[train_end]),
            "validation_end": str(timestamps[validation_end - 1]),
            "test_start": str(timestamps[validation_end]),
            "test_end": str(timestamps[-1]),
        }
    data_digest = metadata["data_sha256"]
    cache_manifest = directory / "motifs_manifest.json"
    source_digest = hashlib.sha256(
        b"".join(Path(__file__).with_name(name).read_bytes() for name in ["motifs.py", "clustering.py"])
    ).hexdigest()
    expected_manifest = {
        "data_sha256": data_digest,
        "source_sha256": source_digest,
        "train_end": train_end,
        "settings": metadata["motifs"],
    }
    if args.reuse_motifs and cache_manifest.exists():
        if json.loads(cache_manifest.read_text()) != expected_manifest:
            raise ValueError("Motif cache differs from the data/code/settings; rerun without --reuse-motifs")
        model = MotifModel.load(directory / "motifs.npz")
    else:
        print(f"{dataset}: fitting 1000 complete Wishart motif sets on {train_end} observations", flush=True)
        model = MotifModel().fit(series[:train_end], workers=args.workers)
        model.save(directory / "motifs.npz")
        cache_manifest.write_text(json.dumps(expected_manifest, indent=2) + "\n")
    metadata["motifs"]["total_centers"] = len(model.motifs)
    pd.DataFrame(model.patterns, columns=["gap1", "gap2", "gap3"]).assign(motifs=model.sizes).to_csv(
        directory / "motif_counts.csv", index=False
    )
    contexts, truth = windows(series, test_origins, model.context_size, args.horizon)
    val_contexts, val_truth = windows(series, validation_origins, model.context_size, args.horizon)
    np.savez_compressed(
        directory / "split.npz",
        series=series,
        raw=raw,
        validation_origins=validation_origins,
        test_origins=test_origins,
        targets=test_origins[:, None] + np.arange(1, args.horizon + 1),
        truth=truth,
    )
    print(f"{dataset}: training predictability selectors on validation trajectories", flush=True)
    val_oracle, features = forecast(
        model,
        val_contexts,
        args.horizon,
        replace(config, selection="ap"),
        truth=val_truth,
        collect_features=True,
    )
    np.savez_compressed(
        directory / "selector_training.npz", predictions=val_oracle, features=features, truth=val_truth
    )
    selectors = fit_selectors(features, val_oracle, val_truth, config.oracle_tolerance, args.seed)
    labels = np.abs(val_oracle - val_truth) < config.oracle_tolerance
    nonempty = features[..., 0] > 0
    thresholds = np.linspace(0, 1, 51)
    scores = []
    for threshold in thresholds:
        accepted = nonempty & (features[..., 7] <= threshold)
        tp, fp, fn = np.sum(accepted & labels), np.sum(accepted & ~labels), np.sum(~accepted & labels)
        scores.append(2 * tp / (2 * tp + fp + fn))
    config = replace(config, entropy_threshold=float(thresholds[np.argmax(scores)]))
    metadata["forecast_config"] = asdict(config)
    predictions, timing = {}, {}
    specifications = [("cl", "avg", selection) for selection in ["cm", "ap", "en"]]
    specifications += [("cl", aggregation, "cm") for aggregation in ["wavgd", "wavgl", "wavgc"]]
    specifications += [("cl", "avg", selection) for selection in selectors]
    specifications += [("cl", "avg", selection) for selection in ["rg", "rgdbscan", "rgwshrt"]]
    specifications += [("trj", "avg", selection) for selection in ["cm", "ap"]]
    specifications += [("trj", "trjp", "rgdbscan")]
    for family, aggregation, selection in specifications:
        name = f"{family}_{aggregation}_{selection}"
        current = replace(config, aggregation=aggregation if family == "cl" else "avg", selection=selection)
        start = time.monotonic()
        if family == "cl":
            prediction = forecast(
                model,
                contexts,
                args.horizon,
                current,
                classifier=selectors.get(selection),
                truth=truth if selection == "ap" else None,
            )
        else:
            prediction = trajectory_forecast(
                model,
                contexts,
                args.horizon,
                current,
                trajectories=args.trajectories,
                truth=truth if selection == "ap" else None,
            )
        predictions[name] = prediction
        np.save(directory / f"{name}.npy", prediction)
        timing[name] = time.monotonic() - start
        summary = errors(truth, prediction)
        print(
            f"{dataset}: {name}: coverage={summary['coverage']:.3f}, RMSE={summary['rmse']:.6f}, "
            f"{timing[name]:.1f}s",
            flush=True,
        )
    if not args.skip_baselines:
        baselines, trials = run_baselines(
            series, train_end, validation_origins, test_origins, args.horizon, timestamps, args.seed
        )
        predictions.update(baselines)
        pd.DataFrame(trials).to_csv(directory / "baseline_validation.csv", index=False)
        for name, prediction in baselines.items():
            np.save(directory / f"{name}.npy", prediction)
        selective = {
            name: p
            for name, p in predictions.items()
            if name.startswith(("cl_", "trj_")) and not name.endswith("_ap")
        }
        compare_masks(truth, predictions, selective).to_csv(directory / "matched_masks.csv", index=False)
    metrics = pd.concat(
        [
            horizon_metrics(
                truth,
                prediction,
                name,
                oracle=predictions["trj_avg_ap" if name.startswith("trj_") else "cl_avg_ap"],
            )
            for name, prediction in predictions.items()
        ],
        ignore_index=True,
    )
    metrics["rmse_original_units"] = metrics.rmse * scale
    metrics["mae_original_units"] = metrics.mae * scale
    metrics.to_csv(directory / "metrics.csv", index=False)
    summaries = pd.DataFrame(
        [{"method": name} | errors(truth, prediction) for name, prediction in predictions.items()]
    )
    summaries["rmse_original_units"] = summaries.rmse * scale
    summaries.to_csv(directory / "summary.csv", index=False)
    comparison = _published_comparison(directory, dataset, metrics)
    metadata["table_slots"] = len(comparison)
    metadata["published_numeric_cells"] = int(comparison.published.notna().sum())
    metadata["comparable_cells"] = int((comparison.status == "comparable_reconstruction").sum())
    metadata["within_published_rounding"] = int(comparison.within_rounding.sum())
    metadata["timing_seconds"] = timing
    metadata["total_seconds"] = time.monotonic() - started
    _plot(directory, metrics)
    diagnostics = {
        "entropy_complexity_dimension": 5,
        "entropy_complexity_delay": 1,
        "entropy_complexity": entropy_complexity(series[:train_end]),
        "eckmann_embedding": 10,
        "eckmann_matrix_dimension": 4,
        "eckmann_min_neighbors": 12,
        "eckmann_min_separation": 20,
        "lyapunov_spectrum": eckmann_lyapunov(
            series[:train_end], dt=0.1 if dataset == "lorenz" else 1
        ).tolist(),
        "paper_horizon_arithmetic": predictability_horizon(),
        "status": "estimators use explicit assumptions; paper estimator settings unavailable",
    }
    (directory / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"{dataset}: finished in {metadata['total_seconds']:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["lorenz", "germany", "all"], default="all")
    parser.add_argument("--output", default="results")
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--test-stride", type=int, default=1)
    parser.add_argument("--validation-stride", type=int, default=8)
    parser.add_argument("--trajectories", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse-motifs", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    args = parser.parse_args()
    set_num_threads(args.workers)
    with threadpool_limits(limits=1):
        for dataset in ["lorenz", "germany"] if args.dataset == "all" else [args.dataset]:
            run(dataset, args)


if __name__ == "__main__":
    main()
