import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
from numba import set_num_threads
from numba.typed import List
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

from . import ForecastConfig, MotifModel, forecast
from .data import windows
from .evaluation import errors
from .motifs import embed
from .prediction import _aggregate_batch, trajectory_forecast
from .selection import fit_selectors


def validation(root):
    rows = []
    for dataset in ["lorenz", "germany"]:
        original = Path("results") / dataset
        split = np.load(original / "split.npz")
        metadata = json.loads((original / "metadata.json").read_text())
        model = MotifModel.load(root.parent / dataset / "motifs.npz")
        contexts, truth = windows(split["series"], split["validation_origins"], model.context_size, 100)
        config = ForecastConfig()
        for epsilon in [0.01, 0.05]:
            candidates = model.candidates(contexts, np.ones_like(contexts), epsilon)
            counts = np.array([len(c[0]) for c in candidates])
            for mode, name in [(0, "avg"), (4, "clc")]:
                prediction = _aggregate_batch(candidates, mode, config.cluster_eps, config.min_samples)[:, 0]
                rows.append(
                    {
                        "dataset": dataset,
                        "variant": "wishart",
                        "epsilon": epsilon,
                        "aggregation": name,
                        "horizon": 1,
                        "median_candidates": float(np.median(counts)),
                    }
                    | errors(truth[:, 0], prediction)
                )
            prediction = trajectory_forecast(
                model, contexts, 100, replace(config, epsilon=epsilon, selection="rgdbscan"), trajectories=16
            )
            np.save(root / f"{dataset}_validation_epsilon_{epsilon}.npy", prediction)
            rows.append(
                {
                    "dataset": dataset,
                    "variant": "wishart",
                    "epsilon": epsilon,
                    "aggregation": "trj_trjp_rgdbscan",
                    "horizon": "1-100",
                }
                | errors(truth, prediction)
            )
            pd.DataFrame(rows).to_csv(root / "validation.csv", index=False)
            print(dataset, epsilon, errors(truth, prediction), flush=True)
        raw_values = [[] for _ in contexts]
        for pattern, lags in zip(model.patterns, model.lags, strict=True):
            vectors = embed(split["series"][: metadata["train"][1]], pattern)
            neighbors = cKDTree(vectors[:, :-1]).query_ball_point(contexts[:, -lags], config.epsilon)
            for values, indices in zip(raw_values, neighbors, strict=True):
                values.extend(vectors[indices, -1])
        candidates = List(
            [
                (np.array(values), np.full(len(values), config.decay), np.ones(len(values)))
                for values in raw_values
            ]
        )
        for mode, name in [(0, "avg"), (4, "clc")]:
            prediction = _aggregate_batch(candidates, mode, config.cluster_eps, config.min_samples)[:, 0]
            rows.append(
                {
                    "dataset": dataset,
                    "variant": "raw_section_3_2",
                    "epsilon": config.epsilon,
                    "aggregation": name,
                    "horizon": 1,
                    "median_candidates": float(np.median([len(values) for values in raw_values])),
                }
                | errors(truth[:, 0], prediction)
            )
        pd.DataFrame(rows).to_csv(root / "validation.csv", index=False)
    (root / "validation_protocol.json").write_text(
        json.dumps(
            {
                "configuration": asdict(config),
                "trajectories": 16,
                "matching_radii": [0.01, 0.05],
                "sources": ["Algorithm 2", "Section 3.4, equation (5)"],
                "raw_variant": "Section 3.2 diagnostic only; Section 5 numerical experiments cluster motifs.",
                "validation": "Original 113 validation origins, no test scores used for this comparison.",
                "selection": "Report both candidates. Change no default unless both RMSE and coverage improve.",
            },
            indent=2,
        )
        + "\n"
    )


def controls(root):
    rows = []
    for dataset in ["lorenz", "germany"]:
        original = Path("results") / dataset
        directory = root / dataset
        directory.mkdir(exist_ok=True)
        split = np.load(original / "split.npz")
        config = ForecastConfig()
        model = MotifModel.load(root.parent / dataset / "motifs.npz")
        contexts, truth = windows(split["series"], split["test_origins"], model.context_size, 100)
        training = np.load(original / "selector_training.npz")
        selectors = fit_selectors(training["features"], training["predictions"], training["truth"])
        specifications = [(f"cl_{mode}_cm", mode, None) for mode in ["wavgd", "wavgc"]]
        specifications += [(f"cl_avg_{name}", "avg", selector) for name, selector in selectors.items()]
        for name, aggregation, selector in specifications:
            prediction = forecast(
                model, contexts, 100, replace(config, aggregation=aggregation), classifier=selector
            )
            np.save(directory / f"{name}.npy", prediction)
            before = np.load(original / f"{name}.npy")
            common = np.isfinite(before) & np.isfinite(prediction)
            for support, mask in [("own", None), ("intersection", common)]:
                for version, values in [("before", before), ("after", prediction)]:
                    rows.append(
                        {
                            "dataset": dataset,
                            "method": name,
                            "version": version,
                            "support": support,
                        }
                        | errors(truth, values, mask)
                    )
            pd.DataFrame(rows).to_csv(root / "controls.csv", index=False)
            print(dataset, name, errors(truth, prediction), flush=True)


def radius_prediction(model, contexts, epsilon):
    def batch(start):
        set_num_threads(2)
        config = ForecastConfig(epsilon=epsilon, selection="rgdbscan", seed=42 + start)
        return start, trajectory_forecast(model, contexts[start : start + 64], 100, config, trajectories=16)

    prediction = np.full((len(contexts), 100), np.nan)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for start, values in pool.map(batch, range(0, len(contexts), 64)):
            prediction[start : start + 64] = values
            print(epsilon, min(start + 64, len(contexts)), "/", len(contexts), flush=True)
    return prediction


def radius_test(root):
    validation_rows = pd.read_csv(root / "validation.csv")
    validation_rows = validation_rows[validation_rows.aggregation == "trj_trjp_rgdbscan"]
    decisions, rows = [], []
    for dataset in ["lorenz", "germany"]:
        scores = validation_rows[validation_rows.dataset == dataset].set_index("epsilon")
        first, second = scores.loc[0.01], scores.loc[0.05]
        selected = bool(second.rmse < first.rmse and second.coverage >= first.coverage)
        decisions.append({"dataset": dataset, "test_epsilon_005": selected})
    (root / "radius_test_protocol.json").write_text(
        json.dumps(
            {
                "decisions": decisions,
                "decision_data": "validation.csv",
                "criterion": "epsilon=0.05 must improve validation RMSE without reducing coverage",
                "support": "All original 901 test origins and 100 horizons; independent of new paper alignment",
                "batch_size": 64,
                "parallel_batches": 2,
                "seed": "42 + first row index in batch, identical for both radii",
                "reason_for_batching": "Bound candidate-array memory for the larger matching radius",
            },
            indent=2,
        )
        + "\n"
    )
    for decision in decisions:
        if not decision["test_epsilon_005"]:
            continue
        dataset = decision["dataset"]
        original = Path("results") / dataset
        split = np.load(original / "split.npz")
        model = MotifModel.load(root.parent / dataset / "motifs.npz")
        contexts, truth = windows(split["series"], split["test_origins"], model.context_size, 100)
        predictions = {}
        for epsilon in [0.01, 0.05]:
            prediction = radius_prediction(model, contexts, epsilon)
            predictions[epsilon] = prediction
            np.save(root / f"{dataset}_test_epsilon_{epsilon}.npy", prediction)
            for h in [1, 10, 50, 100]:
                rows.append(
                    {"dataset": dataset, "epsilon": epsilon, "horizon": str(h)}
                    | errors(truth[:, h - 1], prediction[:, h - 1])
                )
            rows.append(
                {"dataset": dataset, "epsilon": epsilon, "horizon": "1-100"} | errors(truth, prediction)
            )
            pd.DataFrame(rows).to_csv(root / "radius_test.csv", index=False)
        common = np.isfinite(predictions[0.01]) & np.isfinite(predictions[0.05])
        for epsilon, prediction in predictions.items():
            rows.append(
                {"dataset": dataset, "epsilon": epsilon, "horizon": "intersection"}
                | errors(truth, prediction, common)
            )
        pd.DataFrame(rows).to_csv(root / "radius_test.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["validation", "controls", "radius_test"])
    parser.add_argument("--output", default="results/fidelity/audit")
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    set_num_threads(2)
    with threadpool_limits(limits=1):
        {"validation": validation, "controls": controls, "radius_test": radius_test}[args.phase](root)
    (root / f"{args.phase}_source_sha256.json").write_text(
        json.dumps(
            {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(Path("adis").glob("*.py"))
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
