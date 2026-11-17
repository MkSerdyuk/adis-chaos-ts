import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numba import set_num_threads
from threadpoolctl import threadpool_limits

from src import Forecaster
from src.data import german_load, lorenz
from src.evaluation import horizon_metrics

ROOT = Path(__file__).resolve().parent
HORIZONS = [1, 10, 50, 100]
THREADS = 4
CONFIG = {
    "Lorenz": {
        "train": 10000,
        "data": {
            "n": 11000,
            "discard": 3000,
            "dt": 0.1,
            "initial": (1.0, 0.0, 0.0),
            "sigma": 10.0,
            "rho": 28.0,
            "beta": 2.6666666666666665,
            "component": 0,
        },
        "paper": {
            "length": 4,
            "max_gap": 10,
            "r": 11,
            "mu": 0.2,
            "clustered": True,
            "epsilon": 0.01,
            "cluster_eps": 0.2046638891241483,
            "min_samples": 1,
            "trajectories": 16,
            "noise_variance": 0.0001,
            "endpoint_eps": 0.013503140378698722,
            "endpoint_min_samples": 1,
            "decay": 0.99,
            "seed": 42,
            "noise_schedule": "fixed",
            "growth_rule": "strict",
        },
        "custom": {
            "length": 4,
            "max_gap": 10,
            "r": 11,
            "mu": 0.2,
            "clustered": True,
            "epsilon": 0.02,
            "cluster_eps": 0.2,
            "min_samples": 1,
            "trajectories": 16,
            "noise_variance": 0.0001,
            "endpoint_eps": 0.013503140378698722,
            "endpoint_min_samples": 1,
            "decay": 0.99,
            "seed": 42,
            "noise_schedule": "fixed",
            "growth_rule": "strict",
        },
    },
    "German electricity load": {
        "train": 9008,
        "data": {"path": "data/germany.csv"},
        "paper": {
            "cluster_eps": 0.1,
            "clustered": True,
            "decay": 0.99,
            "endpoint_eps": 0.013503140378698722,
            "endpoint_min_samples": 1,
            "epsilon": 0.05,
            "growth_rule": "strict",
            "length": 4,
            "max_gap": 10,
            "min_samples": 50,
            "mu": 0.2,
            "noise_law": "normal",
            "noise_schedule": "fixed",
            "noise_variance": 0.0001,
            "patterns": None,
            "r": 11,
            "seed": 42,
            "trajectories": 16,
        },
        "custom": {
            "length": 5,
            "max_gap": 168,
            "r": 11,
            "mu": 0.2,
            "clustered": False,
            "patterns": [[1, 166, 1, 1], [1, 165, 1, 2]],
            "epsilon": 0.04,
            "cluster_eps": 0.1,
            "min_samples": 1,
            "trajectories": 16,
            "noise_variance": 9e-06,
            "endpoint_eps": 0.012,
            "endpoint_min_samples": 1,
            "decay": 0.99,
            "seed": 42,
            "noise_schedule": "fixed",
            "growth_rule": "strict",
        },
    },
}


def evaluate(raw, train_size, parameters, output):
    minimum, scale = raw[:train_size].min(), np.ptp(raw[:train_size])
    series = (raw - minimum) / scale
    model = Forecaster(**parameters).fit(series[:train_size], workers=THREADS)
    origins = np.arange(train_size - 1, len(series) - 1)
    contexts = series[origins[:, None] - np.arange(model.context_size_ - 1, -1, -1)]
    truth = np.r_[series, np.full(99, np.nan)][origins[:, None] + np.arange(1, 101)]
    prediction = model.predict(contexts, horizon=100, origin_ids=origins)
    metrics = horizon_metrics(truth, prediction, "trj_trjp_rgdbscan")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "predictions.npz", origins=origins, truth=truth, prediction=prediction)
    metrics.to_csv(output / "metrics.csv", index=False)
    (output / "config.json").write_text(json.dumps(model.get_params(), indent=2) + "\n")
    metadata = {
        "raw_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
        "train": train_size,
        "test": len(raw) - train_size,
        "normalization": "train_minmax",
        "minimum": float(minimum),
        "scale": float(scale),
        "evaluation": "all available targets from origins within the test interval",
        "metrics": "conditional on finite predictions; unavailable targets excluded",
        "selection": "exploratory reconstruction; test period inspected during prior searches",
        "original_author_data_verified": False,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metrics.set_index("horizon")


def markdown_table(frame):
    rows = [frame.columns.tolist(), ["---"] * len(frame.columns)] + frame.astype(str).values.tolist()
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def write_history():
    configs = []
    scores = []
    for path in sorted((ROOT / "results").glob("*/*/metrics.csv")):
        dataset, variant = path.parent.parent.name, path.parent.name
        parameters = json.loads(path.with_name("config.json").read_text())
        configs.append([dataset, variant, json.dumps(parameters, separators=(",", ":"))])
        for row in pd.read_csv(path).itertuples(index=False):
            if row.horizon in HORIZONS:
                scores.append(
                    [
                        dataset,
                        variant,
                        row.horizon,
                        row.total,
                        row.count,
                        f"{row.rmse:.9f}",
                        f"{row.mape:.9f}",
                        f"{row.coverage:.6f}",
                    ]
                )
    protocol = pd.DataFrame(
        [
            ["Lorenz", "RK4; sigma=10; rho=28; beta=8/3; dt=.1; discard=3000; initial=(1,0,0)", 10000, 1000],
            [
                "German electricity load",
                "OPSD 2018-06-30; DE_load_entsoe_power_statistics; 2014-12-31 to 2016-02-20 UTC",
                9008,
                1000,
            ],
        ],
        columns=["Dataset", "Input", "Train", "Test"],
    )
    definitions = pd.DataFrame(
        [
            ["Normalization", "Train-only min-max; shared by paper and custom runs"],
            ["Targets at horizons 1/10/50/100", "1000/991/951/901; unavailable targets excluded"],
            ["RMSE and MAPE", "Conditional on accepted predictions; normalized units; fractional MAPE"],
            [
                "Published parameters",
                "L=4; Kmax=10; Wishart r=11, mu=.2; decay=.99; stated Lorenz dynamics",
            ],
            [
                "Unreported parameters",
                "Initial condition; normalization; DBSCAN; trajectory count; noise; seed; evaluation origins",
            ],
            [
                "Selection",
                "Exploratory reconstruction and transferred custom configurations; previously inspected test data",
            ],
            ["Author input and exact reproduction", "Not verified"],
            [
                "Published matching radii",
                ".01 in Algorithm 2; .05 in the prose; Germany uses the prose variant",
            ],
        ],
        columns=["Setting", "Value"],
    )
    configurations = pd.DataFrame(configs, columns=["Dataset", "Variant", "Parameters"])
    metrics = pd.DataFrame(
        scores, columns=["Dataset", "Variant", "Horizon", "Targets", "Accepted", "RMSE", "MAPE", "Coverage"]
    )
    profiles = pd.read_json(ROOT / "results/history_profiles.json")
    profile_rows = [
        [record["name"], record["normalization"], json.dumps(record["metadata"], separators=(",", ":"))]
        for record in profiles.to_dict("records")
    ]
    profile_table = pd.DataFrame(
        profile_rows, columns=["Historical profile", "Normalization", "Input parameters"]
    )
    history = pd.read_csv(ROOT / "results/history.csv.gz", keep_default_na=False, dtype=str)
    tables = [protocol, definitions, configurations, metrics, profile_table, history]
    (ROOT / "EXPERIMENTS.md").write_text("\n\n".join(markdown_table(frame) for frame in tables) + "\n")


def main():
    set_num_threads(THREADS)
    paper = pd.read_csv(ROOT / "references/published_tables.csv")
    paper = paper[paper.method == "trj_trjp_rgdbscan"].set_index(["dataset", "horizon"])
    datasets = {
        "Lorenz": lorenz(**CONFIG["Lorenz"]["data"]),
        "German electricity load": german_load(ROOT / CONFIG["German electricity load"]["data"]["path"]),
    }
    frames = []
    reports = []
    measured = {name: {} for name in datasets}
    with threadpool_limits(limits=1):
        for variant in ["paper", "custom"]:
            for name, raw in datasets.items():
                dataset = "lorenz" if name == "Lorenz" else "germany"
                settings = CONFIG[name]
                measured[name][variant] = evaluate(
                    raw, settings["train"], settings[variant], ROOT / "results" / dataset / variant
                )
        for name in datasets:
            dataset = "lorenz" if name == "Lorenz" else "germany"
            metrics = measured[name]
            rows = [
                {
                    "Horizon": h,
                    "Paper RMSE": f"{paper.loc[(dataset, h), 'rmse']:.2f}",
                    "Paper-parameter RMSE": f"{metrics['paper'].loc[h, 'rmse']:.6f}",
                    "Custom-parameter RMSE": f"{metrics['custom'].loc[h, 'rmse']:.6f}",
                }
                for h in HORIZONS
            ]
            frame = pd.DataFrame(rows)
            frames.append(frame.assign(Dataset=name))
            reports.append(f"## {name}\n\n" + markdown_table(frame))
    pd.concat(frames, ignore_index=True).to_csv(ROOT / "results/comparison.csv", index=False)
    (ROOT / "results/config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")
    report = "\n\n".join(reports)
    directories = pd.DataFrame(
        [
            [
                "src/",
                "Forecasting methods; Forecaster(**parameters).fit(train).predict(context, horizon=100)",
            ],
            ["data/", "Hourly German electricity load and source metadata"],
            ["references/", "Original paper and published metrics"],
            ["results/", "Configurations, predictions and metrics"],
            ["trading-research/data/", "Frozen market data, forecasts and original experiment provenance"],
            ["trading-research/tools/", "Forecasting, comparator models, execution and report tools"],
            [
                "trading-research/results/",
                "Separate generated HTML reports and numerical results for each market block",
            ],
        ],
        columns=["Directory", "Contents"],
    )
    commands = pd.DataFrame(
        [
            [label, f"`uv run --extra trading python trading-research/{name}.py`"]
            for name, label in [
                ("btc", "Bitcoin"),
                ("crypto", "Other crypto"),
                ("noncrypto", "Stocks, bonds and gold"),
            ]
        ],
        columns=["Market block", "Run"],
    )
    usage = (
        "Parameters are defined in each script’s `CONFIG`. Trading calls the unchanged "
        "`src.Forecaster.fit/predict`; identical inputs reuse its computed cache. "
        "Each script prints its HTML report path."
    )
    (ROOT / "README.md").write_text(
        report
        + "\n\n"
        + markdown_table(directories)
        + "\n\n"
        + markdown_table(commands)
        + "\n\n"
        + usage
        + "\n"
    )
    write_history()
    print(report)


if __name__ == "__main__":
    main()
