import json
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from multiprocessing import get_context

import numpy as np
import pandas as pd
from numba import set_num_threads
from threadpoolctl import threadpool_limits
from tools import baselines, forecasts
from tools.data import load_station, start_run
from tools.evaluation import bootstrap, metrics
from tools.report import write_report

CONFIG = {
    "stations": {"00433": "Berlin-Tempelhof", "01975": "Hamburg-Fuhlsbuettel", "03379": "Munich-City"},
    "parameters": {
        "length": 3,
        "max_gap": 3,
        "r": 5,
        "mu": 0.02,
        "epsilon": 0.06502211399080364,
        "cluster_eps": 0.008371633741976994,
        "min_samples": 3,
        "noise_variance": 5.8071705698799575e-05,
        "endpoint_eps": 0.008371633741976994,
        "endpoint_min_samples": 3,
        "trajectories": 16,
        "clustered": True,
        "patterns": None,
        "decay": 0.99,
        "seed": 42,
        "noise_schedule": "fixed",
        "noise_law": "normal",
        "growth_rule": "strict",
    },
    "paper_parameters": {
        "length": 4,
        "max_gap": 10,
        "r": 11,
        "mu": 0.2,
        "epsilon": 0.01,
        "cluster_eps": 0.01,
        "min_samples": 3,
        "noise_variance": 1e-05,
        "endpoint_eps": 0.01,
        "endpoint_min_samples": 3,
        "trajectories": 16,
        "clustered": True,
        "patterns": None,
        "decay": 0.99,
        "seed": 42,
        "noise_schedule": "fixed",
        "noise_law": "normal",
        "growth_rule": "strict",
    },
    "training_hours": 10000,
    "station_workers": 3,
    "numerical_threads": 2,
    "horizons": [1, 6, 12, 24],
    "seeds": [7, 123],
    "periods": {
        "validation": ("2021-01-01", "2022-01-01"),
        "historical": ("2022-01-01", "2026-01-01"),
        "holdout": ("2026-01-01", "2026-11-01"),
    },
}


def evaluate_station(station, config):
    set_num_threads(config["numerical_threads"])
    with threadpool_limits(limits=1):
        chunks, fits = [], []
        start = min(period[0] for period in config["periods"].values())
        end = max(period[1] for period in config["periods"].values())
        frame = load_station(station)
        origins = frame.index[(frame.index >= start) & (frame.index < end) & (frame.index.hour == 23)]
        for quarter in pd.date_range(start, end, freq="QS", inclusive="left", tz="UTC"):
            dates = origins[(origins >= quarter) & (origins < quarter + pd.DateOffset(months=3))]
            variants = dict(forecaster=config["parameters"], paper_geometry=config["paper_parameters"])
            variants.update(
                {f"forecaster_seed_{seed}": dict(config["parameters"], seed=seed) for seed in config["seeds"]}
            )
            predictions = {}
            for method, parameters in variants.items():
                values, fit = forecasts.predict(
                    frame, quarter, dates, parameters, config["training_hours"], max(config["horizons"])
                )
                predictions[method] = values[:, np.array(config["horizons"]) - 1]
                fits.append(dict(station=station, method=method, quarter=str(quarter), **fit))
            comparisons, fit = baselines.predict(
                frame, quarter, dates, config["training_hours"], config["horizons"]
            )
            predictions.update(comparisons)
            fits.append(dict(station=station, method="supervised_comparators", quarter=str(quarter), **fit))
            for column, horizon in enumerate(config["horizons"]):
                target = dates + pd.Timedelta(hours=horizon)
                truth = frame.Temperature.reindex(target).to_numpy()
                eligible = frame.Eligible.reindex(dates).to_numpy()
                for method, values in predictions.items():
                    chunks.append(
                        pd.DataFrame(
                            dict(
                                station=station,
                                origin=dates,
                                target_time=target,
                                horizon=horizon,
                                method=method,
                                truth=truth,
                                prediction=values[:, column],
                                eligible=eligible,
                            )
                        )
                    )
        return chunks, fits


def run(config=CONFIG):
    assert min(config["horizons"]) >= 1 and max(config["horizons"]) <= 24
    assert list(config["periods"])[0] == "validation"
    output = start_run(config)
    chunks, fits = [], []
    start = min(period[0] for period in config["periods"].values())
    assert pd.Timestamp(start).is_quarter_start
    with ProcessPoolExecutor(max_workers=config["station_workers"], mp_context=get_context("spawn")) as pool:
        for station_chunks, station_fits in pool.map(evaluate_station, config["stations"], repeat(config)):
            chunks.extend(station_chunks)
            fits.extend(station_fits)
    predictions = pd.concat(chunks, ignore_index=True)
    measured = metrics(predictions, config["periods"])
    intervals = bootstrap(predictions, list(config["stations"]), config["periods"])
    predictions.to_csv(output / "forecasts.csv.gz", index=False, compression=dict(method="gzip", mtime=0))
    measured.to_csv(output / "metrics.csv", index=False)
    intervals.to_csv(output / "bootstrap.csv", index=False)
    (output / "fits.json").write_text(json.dumps(fits, indent=2) + "\n")
    (output / "completed.json").write_text(
        json.dumps(
            dict(
                stations=list(config["stations"]),
                forecast_rows=len(predictions),
                metric_rows=len(measured),
                bootstrap_comparisons=len(intervals),
                origin_station_pairs=len(predictions[["station", "origin"]].drop_duplicates()),
            ),
            indent=2,
        )
        + "\n"
    )
    return write_report(output, predictions, measured, intervals, config)


if __name__ == "__main__":
    set_num_threads(4)
    with threadpool_limits(limits=1):
        print(run())
